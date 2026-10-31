"""Qwen3-TTS engine (local GPU): a voice designed from a text description, then cloned for narration.

Designing a voice from a natural-language description exists only in the 1.7B VoiceDesign model,
so this engine follows the README's "voice design then clone" workflow:

1. once per voice, the 1.7B VoiceDesign model renders a ~10 s reference clip from the description
   (cached under voice_dir, so the voice stays the same across runs and papers);
2. a Base model (0.6B or 1.7B) clones that clip for every sentence, so a whole paper keeps one voice.

model="1.7B-design" skips the cloning and asks VoiceDesign for every sentence. It follows the
description most literally, but each call samples the voice anew, so it can drift between sentences.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .audio_utils import read_wav, silence, trim_silence, write_wav
from .base import Chunk, SynthResult, VoiceInfo
from .kokoro_engine import split_sentences

MODELS = {
    "0.6B": {"repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "label": "Qwen3-TTS 0.6B (designed voice, cloned)"},
    "1.7B": {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "label": "Qwen3-TTS 1.7B (designed voice, cloned)"},
    "1.7B-design": {"repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                    "label": "Qwen3-TTS 1.7B VoiceDesign (description per sentence)"},
}
DEFAULT_MODEL = "0.6B"
DESIGN_REPO = MODELS["1.7B-design"]["repo"]

# Built-in voices (id -> description). Users design more in the voice lab; those live in
# <voice_dir>/voices.json. A reference clip is designed on a voice's first use.
VOICE_PRESETS = {
    "technical_male": "Clear and precise and technical, male middle age man",
}
DEFAULT_VOICE = "technical_male"

LANGUAGE = "English"
REFERENCE_TEXT = (
    "In this section, we describe the experimental setup in detail. The policy was trained for two "
    "thousand iterations, and the results were evaluated on five different terrains."
)
SENTENCE_GAP = 0.18  # seconds between sentences of one unit
# Sentences per generate call. Measured on the RTX 5070 Ti Laptop (12 GB) with real paper sentences:
# 0.6B 1.6x realtime at batch 1, 9.5x at 8, 21x at 32, 22x at 48; 1.7B 1.7x -> 18.7x at 32 (48: no gain).
BATCH_SENTENCES = 32
# The library decodes a whole batch to audio at once, each item with the 11 s reference prepended;
# decoding 4 at a time keeps peak VRAM at ~5 GB (0.6B) / ~7.5 GB (1.7B) instead of running out at 16.
DECODE_CHUNK = 4

_lock = threading.Lock()
_loaded: dict[str, object] = {}  # single slot: repo id -> Qwen3TTSModel (12 GB GPU holds one comfortably)
_registry_lock = threading.Lock()


class VoiceRegistry:
    """Designed voices: the presets plus the user's own (<voice_dir>/voices.json).

    A voice is a description plus a take number: designing the same description again samples a
    different voice, so every take gets its own id and its own reference clip."""

    def __init__(self, voice_dir: Path):
        self.dir = voice_dir
        self.file = voice_dir / "voices.json"

    def _custom(self) -> list[dict]:
        return json.loads(self.file.read_text(encoding="utf-8"))["voices"] if self.file.exists() else []

    def _write(self, voices: list[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"voices": voices}, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.file)

    def all(self) -> list[dict]:
        presets = [{"id": vid, "description": desc, "take": 1, "preset": True, "created": None}
                   for vid, desc in VOICE_PRESETS.items()]
        return presets + [dict(v, preset=False) for v in self._custom()]

    def get(self, voice_id: str) -> dict:
        for voice in self.all():
            if voice["id"] == voice_id:
                return voice
        raise ValueError(f"unknown Qwen3-TTS voice {voice_id!r}")

    def add(self, description: str = "", *, redesign_of: str | None = None) -> dict:
        """A new voice for `description`, or a new take of voice `redesign_of`'s description."""
        description = self.get(redesign_of)["description"] if redesign_of else " ".join(description.split())
        if not description or len(description) > 500:
            raise ValueError("a voice description must be 1-500 characters")
        with _registry_lock:
            take = 1 + max((v["take"] for v in self.all() if v["description"] == description), default=0)
            slug = "_".join(re.findall(r"[a-z]+", description.lower())[:3]) or "voice"
            digest = hashlib.sha1(description.encode()).hexdigest()[:4]
            record = {"id": f"{slug}_{digest}_t{take}", "description": description, "take": take,
                      "created": round(time.time())}
            self._write(self._custom() + [record])
        return dict(record, preset=False)

    def delete(self, voice_id: str) -> None:
        if voice_id in VOICE_PRESETS:
            raise ValueError("built-in voices can't be deleted")
        with _registry_lock:
            custom = self._custom()
            kept = [v for v in custom if v["id"] != voice_id]
            if len(kept) == len(custom):
                raise KeyError(voice_id)
            self._write(kept)
        for clip in self.dir.glob(f"{voice_id}-*.wav"):
            clip.unlink()


def _import_qwen():
    """qwen_tts prints SoX / flash-attn notices on import; SoX is only used by the 25 Hz tokenizer."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import qwen_tts
    return qwen_tts


def _model(repo: str, device: str):
    """Load `repo`, evicting any other Qwen model first. Call with _lock held."""
    if repo in _loaded:
        return _loaded[repo]
    import torch

    for other in list(_loaded):
        del _loaded[other]
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    qwen_tts = _import_qwen()
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = qwen_tts.Qwen3TTSModel.from_pretrained(repo, device_map=device, dtype=dtype, attn_implementation="sdpa")
    _chunk_decoding(model)
    _loaded[repo] = model
    return model


def release_gpu() -> None:
    """Drop the loaded Qwen model (e.g. before local OCR needs the GPU); it reloads on next use."""
    with _lock:
        _loaded.clear()
    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _chunk_decoding(model) -> None:
    """Make the model decode generated codes DECODE_CHUNK items at a time (generation stays batched).
    Wraps the speech tokenizer's public decode(list[dict]) -> (wavs, sample_rate) on this instance."""
    tokenizer = model.model.speech_tokenizer
    decode = tokenizer.decode

    def chunked(encoded):
        if not isinstance(encoded, list) or len(encoded) <= DECODE_CHUNK:
            return decode(encoded)
        wavs, sample_rate = [], None
        for start in range(0, len(encoded), DECODE_CHUNK):
            part, sample_rate = decode(encoded[start:start + DECODE_CHUNK])
            wavs.extend(part)
        return wavs, sample_rate

    tokenizer.decode = chunked


class QwenTTSEngine:
    name = "qwen3"

    def __init__(self, model: str = DEFAULT_MODEL, *, voice_dir: Path | None = None, device: str | None = None):
        if model not in MODELS:
            raise ValueError(f"unknown Qwen3-TTS model {model!r}; choose from {', '.join(MODELS)}")
        self.model = model
        self.voice_dir = voice_dir or Path.home() / ".cache" / "paper_reader" / "qwen_voices"
        self.registry = VoiceRegistry(self.voice_dir)
        self._device = device
        self._prompts: dict[str, object] = {}  # voice -> clone prompt for this Base model

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return self._device

    def available(self) -> tuple[bool, str | None]:
        try:
            _import_qwen()
        except ImportError as exc:
            return False, f"qwen-tts is not installed ({exc})"
        return True, None

    def voices(self) -> list[VoiceInfo]:
        def gender(desc: str) -> str | None:
            words = set(re.findall(r"[a-z]+", desc.lower()))
            return "female" if words & {"female", "woman", "girl"} else "male" if words & {"male", "man", "boy"} else None

        return [VoiceInfo(id=v["id"], label=v["description"], language="en-us", gender=gender(v["description"]))
                for v in self.registry.all()]

    def default_voice(self) -> str:
        return DEFAULT_VOICE

    # ------------------------------------------------------------ the designed voice

    def reference_path(self, voice: str) -> Path:
        """Where the voice's reference clip lives (it may not be designed yet)."""
        key = hashlib.sha256(f"{self._description(voice)}\n{REFERENCE_TEXT}".encode()).hexdigest()[:12]
        return self.voice_dir / f"{voice}-{key}.wav"

    def reference_clip(self, voice: str) -> tuple[Path, str]:
        """(wav path, transcript) of the voice's designed reference clip; designs it on first use."""
        description = self._description(voice)
        path = self.reference_path(voice)
        if not path.exists():
            with _lock:
                design = _model(DESIGN_REPO, self.device)
                wavs, sr = design.generate_voice_design(text=REFERENCE_TEXT, instruct=description, language=LANGUAGE)
            self.voice_dir.mkdir(parents=True, exist_ok=True)
            write_wav(path, np.asarray(wavs[0], dtype=np.float32), sr)
        return path, REFERENCE_TEXT

    def _description(self, voice: str) -> str:
        return self.registry.get(voice)["description"]

    # ------------------------------------------------------------ synthesis

    def synthesize(self, text: str, voice: str, speed: float = 1.0, *,
                   previous_text: str | None = None, next_text: str | None = None) -> SynthResult:
        return self.synthesize_many([text], voice, speed)[0]

    def synthesize_many(self, texts: list[str], voice: str, speed: float = 1.0) -> list[SynthResult]:
        """Voice several units at once. Their sentences are pooled, sorted by length (so a batch
        doesn't wait on one long sentence) and generated BATCH_SENTENCES at a time; each unit is
        then reassembled from its own sentences. ~13x faster than one sentence per call."""
        per_unit = [split_sentences(text) for text in texts]
        keys = sorted(((u, i) for u, sentences in enumerate(per_unit) for i in range(len(sentences))),
                      key=lambda k: len(per_unit[k[0]][k[1]]))
        description = self._description(voice)
        if keys and self.model != "1.7B-design":
            reference = self.reference_clip(voice)  # outside the lock: may load the design model itself

        clips: dict[tuple[int, int], np.ndarray] = {}
        sample_rate = 24_000
        if keys:
            with _lock:
                model = _model(MODELS[self.model]["repo"], self.device)
                for start in range(0, len(keys), BATCH_SENTENCES):
                    batch_keys = keys[start:start + BATCH_SENTENCES]
                    batch = [per_unit[u][i] for u, i in batch_keys]
                    languages = [LANGUAGE] * len(batch)
                    if self.model == "1.7B-design":
                        wavs, sample_rate = model.generate_voice_design(
                            text=batch, instruct=[description] * len(batch), language=languages)
                    else:
                        prompt = self._clone_prompt(model, voice, reference)
                        wavs, sample_rate = model.generate_voice_clone(
                            text=batch, language=languages, voice_clone_prompt=prompt)
                    clips.update(zip(batch_keys, (np.asarray(w, dtype=np.float32) for w in wavs)))

        return [self._assemble(sentences, [clips[(u, i)] for i in range(len(sentences))], sample_rate, speed)
                if sentences else SynthResult(audio=np.zeros(0, dtype=np.float32), sample_rate=sample_rate)
                for u, sentences in enumerate(per_unit)]

    def _clone_prompt(self, model, voice: str, reference: tuple[Path, str]):
        """Built once per voice: it depends only on this engine's (fixed) model weights."""
        if voice not in self._prompts:
            audio, sr = read_wav(reference[0])
            self._prompts[voice] = model.create_voice_clone_prompt(ref_audio=(audio, sr), ref_text=reference[1])
        return self._prompts[voice]

    @staticmethod
    def _assemble(sentences: list[str], pieces: list[np.ndarray], sr: int, speed: float) -> SynthResult:
        """Join sentence clips with short gaps; every Chunk is exactly one sentence."""
        chunks: list[Chunk] = []
        out: list[np.ndarray] = []
        t = 0.0
        for i, (sentence, audio) in enumerate(zip(sentences, pieces, strict=True)):
            audio = trim_silence(audio, sr, keep_lead=0.05, keep_trail=0.12)
            if speed != 1.0:
                import librosa

                audio = librosa.effects.time_stretch(audio, rate=speed).astype(np.float32)
            if i:
                out.append(silence(SENTENCE_GAP / speed, sr))
                t += SENTENCE_GAP / speed
            dur = len(audio) / sr
            chunks.append(Chunk(sentence, round(t, 3), round(t + dur, 3)))
            out.append(audio)
            t += dur
        audio = np.clip(np.concatenate(out), -1.0, 1.0).astype(np.float32)
        return SynthResult(audio=audio, sample_rate=sr, chunks=chunks)
