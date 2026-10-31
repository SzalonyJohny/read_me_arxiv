"""ElevenLabs (paid, cloud): POST /v1/text-to-speech/{voice_id}/with-timestamps.

Checked against the ElevenLabs API reference on 2026-10-31: the key goes in the `xi-api-key`
header; the JSON body takes `text`, `model_id`, `previous_text`/`next_text` (request stitching,
not available for eleven_v3) and `voice_settings.speed`; the response is
{audio_base64, alignment, normalized_alignment}, where alignment holds `characters`,
`character_start_times_seconds` and `character_end_times_seconds` for the text as sent.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from itertools import accumulate
from typing import Any

import httpx
import numpy as np

from .audio_utils import read_wav
from .base import Chunk, SynthResult, VoiceInfo
from .kokoro_engine import sentence_spans

API_URL = "https://api.elevenlabs.io"
# WAV carries its own header, so decoding does not depend on assumptions about raw PCM.
# Only the 44.1 kHz PCM/WAV formats need the Pro tier; 24 kHz works on every tier.
OUTPUT_FORMAT = "wav_24000"
SAMPLE_RATE = 24_000
DEFAULT_MODEL = "eleven_flash_v2_5"
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # George, the voice used in the official quickstart
MAX_ATTEMPTS = 4
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)

# Used when paper_reader.pricing is missing. Ids and per-request limits are from the models
# page, prices from elevenlabs.io/pricing/api (USD per 1,000 characters), both on 2026-10-31.
FALLBACK_MODELS: dict[str, dict[str, Any]] = {
    "eleven_flash_v2_5": {"label": "Eleven Flash v2.5", "usd_per_1k_chars": 0.04, "max_chars": 40_000},
    "eleven_v4": {"label": "Eleven v4", "usd_per_1k_chars": 0.08, "max_chars": 10_000},
    "eleven_v3": {"label": "Eleven v3", "usd_per_1k_chars": 0.08, "max_chars": 5_000},
    "eleven_multilingual_v2": {"label": "Eleven Multilingual v2", "usd_per_1k_chars": 0.08, "max_chars": 10_000},
}
NO_REQUEST_STITCHING = frozenset({"eleven_v3"})

# Shown when there is no API key. These are ElevenLabs "Default voices" (ids from the help
# center's default-voices page); ElevenLabs retires them on 2026-12-31.
PREMADE_VOICES = [
    VoiceInfo(id=voice_id, label=f"{name} (premade, {accent}, {gender})",
              language="en-gb" if accent == "British" else "en-us", gender=gender)
    for voice_id, name, accent, gender in (
        ("JBFqnCBsd6RMkjVDRZzb", "George", "British", "male"),
        ("onwK4e9ZLuTAKqWW03F9", "Daniel", "British", "male"),
        ("Xb7hH8MSUJpSbSDYk0k2", "Alice", "British", "female"),
        ("pFZP5JQG7iQjIQuC4Bku", "Lily", "British", "female"),
        ("nPczCjzI2devNBz1zQrb", "Brian", "American", "male"),
        ("iP95p4xoKVk53GoZ742B", "Chris", "American", "male"),
        ("EXAVITQu4vr4xnSDxMaL", "Sarah", "American", "female"),
        ("XrExE9yKIg1WjnnlVkGX", "Matilda", "American", "female"),
    )
]


def model_table() -> dict[str, dict[str, Any]]:
    """model_id -> {label, usd_per_1k_chars, max_chars}, from paper_reader.pricing when present."""
    try:
        from ..pricing import ELEVENLABS_MODELS
    except ImportError:
        return FALLBACK_MODELS
    return ELEVENLABS_MODELS


class ElevenLabsEngine:
    name = "elevenlabs"

    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        models = model_table()
        if model not in models:
            raise ValueError(f"unknown ElevenLabs model {model!r}; known: {', '.join(models)}")
        self.model = model
        self._usd_per_1k_chars = float(models[model]["usd_per_1k_chars"])
        self._max_chars = int(models[model]["max_chars"])
        key = (api_key or os.environ.get("ELEVENLABS_API_KEY") or "").strip()
        # No client without a key; `transport` lets tests plug in an httpx.MockTransport.
        self._client = httpx.Client(base_url=API_URL, headers={"xi-api-key": key}, timeout=_TIMEOUT,
                                    transport=transport) if key else None
        self._voices: list[VoiceInfo] | None = None
        self._lock = threading.Lock()

    def available(self) -> tuple[bool, str | None]:
        if self._client is None:
            return False, "ELEVENLABS_API_KEY not set"
        return True, None

    def voices(self) -> list[VoiceInfo]:
        """The account's voices (GET /v1/voices, fetched once); the premade list without a key."""
        if self._client is None:
            return list(PREMADE_VOICES)
        with self._lock:
            if self._voices is None:
                data = self._request("GET", "/v1/voices").json()
                self._voices = [_voice_info(v) for v in data["voices"]]
        return list(self._voices)

    def default_voice(self) -> str:
        return DEFAULT_VOICE

    def synthesize(self, text: str, voice: str, speed: float = 1.0, *,
                   previous_text: str | None = None, next_text: str | None = None) -> SynthResult:
        text = text.strip()
        if not text:
            return SynthResult(audio=np.zeros(0, dtype=np.float32), sample_rate=SAMPLE_RATE)
        if len(text) > self._max_chars:
            raise ValueError(f"{len(text)} characters exceed the {self._max_chars}-character "
                             f"request limit of {self.model}")

        body: dict[str, Any] = {"text": text, "model_id": self.model}
        if self.model not in NO_REQUEST_STITCHING:
            body.update({k: v for k, v in (("previous_text", previous_text), ("next_text", next_text)) if v})
        if speed != 1.0:
            body["voice_settings"] = {"speed": speed}
        payload = self._request("POST", f"/v1/text-to-speech/{voice}/with-timestamps",
                                params={"output_format": OUTPUT_FORMAT}, json=body).json()

        audio, sr = read_wav(io.BytesIO(base64.b64decode(payload["audio_base64"])))
        chunks = sentence_chunks(text, payload.get("alignment"), len(audio) / sr)
        return SynthResult(audio=audio, sample_rate=sr, chunks=chunks, characters_billed=len(text),
                           cost_usd=len(text) / 1000 * self._usd_per_1k_chars)

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send with retries (exponential backoff, honouring Retry-After) on 429, 5xx and network
        errors; any other error status raises RuntimeError with ElevenLabs' message."""
        if self._client is None:
            raise RuntimeError("ELEVENLABS_API_KEY not set")
        attempt = 1
        while True:
            delay = 2.0 ** (attempt - 1)
            try:
                response = self._client.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(f"ElevenLabs request failed: {exc}") from exc
            else:
                status = response.status_code
                if status < 400:
                    return response
                if attempt == MAX_ATTEMPTS or not (status == 429 or status >= 500):
                    raise RuntimeError(f"ElevenLabs {method} {path} failed with HTTP {status}: "
                                       f"{_error_detail(response)}")
                retry_after = response.headers.get("retry-after", "")
                if retry_after.isdigit():
                    delay = float(retry_after)
            time.sleep(delay)
            attempt += 1


def sentence_chunks(text: str, alignment: dict[str, Any] | None, total: float) -> list[Chunk]:
    """One chunk per sentence: each sentence starts when its first character is spoken and runs
    until the next one starts, so the chunks cover [0, total] in order."""
    spans = sentence_spans(text)
    starts = (alignment or {}).get("character_start_times_seconds") or []
    if not starts:
        return [Chunk(text, 0.0, total)]
    # The alignment echoes the request text character by character; scale defensively anyway.
    scale = len(starts) / len(text)
    cuts = [min(starts[min(int(begin * scale), len(starts) - 1)], total) for begin, _ in spans[1:]]
    bounds = list(accumulate([0.0, *cuts, total], max))
    return [Chunk(text[begin:end], bounds[i], bounds[i + 1]) for i, (begin, end) in enumerate(spans)]


def _voice_info(voice: dict[str, Any]) -> VoiceInfo:
    labels = voice.get("labels") or {}
    details = [d for d in (voice.get("category"), labels.get("accent"), labels.get("gender")) if d]
    name = voice.get("name") or voice["voice_id"]
    return VoiceInfo(id=voice["voice_id"], label=f"{name} ({', '.join(details)})" if details else name,
                     language="en", gender=labels.get("gender"))


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    return str(body.get("detail", body) if isinstance(body, dict) else body)[:500]
