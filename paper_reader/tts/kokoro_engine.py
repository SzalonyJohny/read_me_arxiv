"""Kokoro-82M, local and free: one shared KModel on the GPU (or CPU), one KPipeline per accent.

Every returned Chunk is one sentence. The sentences are split here and fed to Kokoro
newline-joined, so each Kokoro segment maps back to exactly one sentence; when Kokoro cuts an
over-long sentence into several pieces, the pieces are merged back into that sentence's chunk.

The sentence splitter also lives here and is reused by the ElevenLabs engine.
"""

from __future__ import annotations

import re
import threading
import warnings
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from .audio_utils import trim_silence
from .base import Chunk, SynthResult, VoiceInfo

if TYPE_CHECKING:
    from kokoro import KModel, KPipeline

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000
DEFAULT_VOICE = "af_heart"

# Kokoro pads every segment with ~0.2-0.35 s of silence before the speech and ~0.3-0.5 s after
# it. Fed sentence by sentence, that adds up to ~0.8 s pauses (Kokoro's own paragraph mode
# pauses ~0.3-0.6 s between sentences), so each sentence keeps only this much (at speed 1.0).
KEEP_LEAD_S = 0.10
KEEP_TRAIL_S = 0.25

# --------------------------------------------------------------------------- sentences

# Lower-case abbreviations (without the final period) that do not end a sentence.
_ABBREVIATIONS = frozenset({
    "al", "approx", "ca", "cf", "ch", "dr", "eq", "eqs", "fig", "figs", "jr", "mr", "mrs", "ms", "prof",
    "ref", "refs", "resp", "sec", "secs", "sr", "st", "tab", "vol", "vs", "viz",
})
_SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*$")
_DOTTED_ACRONYM = re.compile(r"(?:[a-z]\.)+[a-z]")  # e.g, i.e, u.s (final period already removed)


def _ends_sentence(token: str, next_token: str) -> bool:
    end = _SENTENCE_END.search(token)
    if end is None or next_token[0].islower():
        return False
    if end.group().rstrip("\"'”’)]") != ".":
        return True  # "!", "?", "…", "..."
    word = token[: end.start()].lstrip("([\"'“‘").lower()
    return not (word in _ABBREVIATIONS or _DOTTED_ACRONYM.fullmatch(word))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character offsets of each sentence in `text`, whitespace excluded.

    A sentence ends at ., !, ?, or … (optionally followed by closing quotes/brackets) when the
    next word does not start in lower case and the period does not belong to an abbreviation
    ("Fig.", "et al.", "e.g."). Decimals ("3.5") never split: their period is inside a word.
    A line break always ends a sentence.
    """
    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    spans: list[tuple[int, int]] = []
    start = None
    for (tok_start, tok_end), following in pairwise([*tokens, None]):
        if start is None:
            start = tok_start
        if (following is None or "\n" in text[tok_end:following[0]]
                or _ends_sentence(text[tok_start:tok_end], text[following[0]:following[1]])):
            spans.append((start, tok_end))
            start = None
    return spans


def split_sentences(text: str) -> list[str]:
    return [text[start:end] for start, end in sentence_spans(text)]


# --------------------------------------------------------------------------- voices

# Kokoro v1.0 English voices and their overall grades, from the model card's VOICES.md
# (huggingface.co/hexgrad/Kokoro-82M). Prefix: a/b = American/British, f/m = female/male.
_VOICE_GRADES = {
    "af_heart": "A", "af_alloy": "C", "af_aoede": "C+", "af_bella": "A-", "af_jessica": "D",
    "af_kore": "C+", "af_nicole": "B-", "af_nova": "C", "af_river": "D", "af_sarah": "C+",
    "af_sky": "C-",
    "am_adam": "F+", "am_echo": "D", "am_eric": "D", "am_fenrir": "C+", "am_liam": "D",
    "am_michael": "C+", "am_onyx": "D", "am_puck": "C+", "am_santa": "D-",
    "bf_alice": "D", "bf_emma": "B-", "bf_isabella": "C", "bf_lily": "D",
    "bm_daniel": "D", "bm_fable": "C", "bm_george": "C", "bm_lewis": "D+",
}


def _voice_info(voice_id: str, grade: str) -> VoiceInfo:
    american = voice_id[0] == "a"
    gender = "female" if voice_id[1] == "f" else "male"
    name = voice_id[3:].capitalize()
    return VoiceInfo(
        id=voice_id,
        label=f"{name} ({'US' if american else 'UK'}, {gender})",
        language="en-us" if american else "en-gb",
        gender=gender,
        grade=grade,
    )


VOICES = [_voice_info(voice_id, grade) for voice_id, grade in _VOICE_GRADES.items()]


# --------------------------------------------------------------------------- engine


class KokoroEngine:
    name = "kokoro"

    def __init__(self, device: str | None = None) -> None:
        self._device = device  # None: cuda if available, else cpu (decided on first use)
        self._model: KModel | None = None
        self._pipelines: dict[str, KPipeline] = {}
        self._lock = threading.Lock()  # one synthesis at a time; also guards lazy loading

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def available(self) -> tuple[bool, str | None]:
        try:
            import kokoro  # noqa: F401
        except Exception as exc:  # ImportError, or a torch/CUDA install that fails on import
            return False, f"kokoro is not importable: {exc}"
        return True, None

    def voices(self) -> list[VoiceInfo]:
        return list(VOICES)

    def default_voice(self) -> str:
        return DEFAULT_VOICE

    def synthesize(self, text: str, voice: str, speed: float = 1.0, *,
                   previous_text: str | None = None, next_text: str | None = None) -> SynthResult:
        """One chunk per sentence. previous_text/next_text are ignored: Kokoro has no context input."""
        if voice not in _VOICE_GRADES:
            raise ValueError(f"unknown Kokoro voice {voice!r}")
        sentences = split_sentences(text)
        pieces: dict[int, list[np.ndarray]] = {}
        if sentences:
            with self._lock:
                pipeline = self._pipeline(voice[0])
                for result in pipeline("\n".join(sentences), voice=voice, speed=speed, split_pattern=r"\n+"):
                    pieces.setdefault(result.text_index, []).append(result.audio.numpy())

        # Sentences without audio (punctuation only) produce no chunk.
        parts, chunks, cursor = [], [], 0
        for index, sentence in enumerate(sentences):
            if index in pieces:
                part = trim_silence(np.concatenate(pieces[index]), SAMPLE_RATE,
                                    KEEP_LEAD_S / speed, KEEP_TRAIL_S / speed)
                chunks.append(Chunk(sentence, cursor / SAMPLE_RATE, (cursor + len(part)) / SAMPLE_RATE))
                parts.append(part)
                cursor += len(part)
        audio = np.clip(np.concatenate(parts), -1.0, 1.0) if parts else np.zeros(0, dtype=np.float32)
        return SynthResult(audio=audio.astype(np.float32), sample_rate=SAMPLE_RATE, chunks=chunks)

    def _pipeline(self, lang_code: str) -> KPipeline:
        """The pipeline for 'a' (American) or 'b' (British); loads the shared model on first use.
        Call with the lock held."""
        if lang_code not in self._pipelines:
            from kokoro import KModel, KPipeline

            # Kokoro's modules trigger torch deprecation warnings (LSTM dropout, weight_norm, jit)
            # that nothing here can act on.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self._model is None:
                    self._model = KModel(repo_id=REPO_ID).to(self.device).eval()
                self._pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id=REPO_ID, model=self._model)
        return self._pipelines[lang_code]
