"""TTS engine interface. The pipeline only talks to engines through this.

An engine turns ONE unit of text (a paragraph or a section title) into mono float32
PCM plus timed chunks (sentence-ish pieces) relative to the start of that audio.
The pipeline concatenates units, inserts pauses, encodes MP3 and writes the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Chunk:
    text: str
    start: float  # seconds, relative to the start of SynthResult.audio
    end: float


@dataclass
class SynthResult:
    audio: np.ndarray  # float32, mono, range [-1, 1]
    sample_rate: int
    chunks: list[Chunk] = field(default_factory=list)  # cover the audio in order; may be a single chunk
    characters_billed: int = 0  # for paid engines
    cost_usd: float = 0.0


@dataclass
class VoiceInfo:
    id: str  # engine voice id, e.g. "af_heart"
    label: str  # "Heart (US, female)"
    language: str = "en-us"
    gender: str | None = None
    grade: str | None = None  # Kokoro's quality grade, if known


class TTSEngine(Protocol):
    name: str  # "kokoro" | "qwen3" | "elevenlabs"

    def available(self) -> tuple[bool, str | None]:
        """(True, None) if usable now; else (False, reason) e.g. missing API key."""
        ...

    def voices(self) -> list[VoiceInfo]: ...

    def default_voice(self) -> str: ...

    def synthesize(self, text: str, voice: str, speed: float = 1.0, *,
                   previous_text: str | None = None, next_text: str | None = None) -> SynthResult:
        """Synthesize one unit. previous/next_text are continuity hints (ElevenLabs uses them)."""
        ...


def get_engine(name: str, **kwargs) -> TTSEngine:
    """Factory used by the pipeline/server. Imports lazily so torch is only loaded when needed."""
    if name == "kokoro":
        from .kokoro_engine import KokoroEngine

        return KokoroEngine(**kwargs)
    if name == "qwen3":
        from .qwen_engine import QwenTTSEngine

        return QwenTTSEngine(**kwargs)
    if name == "elevenlabs":
        from .elevenlabs_engine import ElevenLabsEngine

        return ElevenLabsEngine(**kwargs)
    raise ValueError(f"unknown TTS engine: {name!r}")
