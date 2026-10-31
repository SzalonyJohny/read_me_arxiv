"""TTS layer: sentence splitting, audio helpers, ElevenLabs against a mocked API, Kokoro smoke test."""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
from itertools import pairwise

import httpx
import numpy as np
import pytest
import soundfile as sf

from paper_reader.tts import audio_utils, elevenlabs_engine
from paper_reader.tts.base import get_engine
from paper_reader.tts.elevenlabs_engine import FALLBACK_MODELS, PREMADE_VOICES, ElevenLabsEngine, sentence_chunks
from paper_reader.tts.kokoro_engine import VOICES, KokoroEngine, sentence_spans, split_sentences

# --------------------------------------------------------------------------- sentences


@pytest.mark.parametrize(("text", "expected"), [
    ("We set lambda to 0.5. The loss drops by 3.2 dB.", ["We set lambda to 0.5.", "The loss drops by 3.2 dB."]),
    ("See Fig. 2 and Eq. 4. Smith et al. agree.", ["See Fig. 2 and Eq. 4.", "Smith et al. agree."]),
    ("It works, e.g. on images. The U.S. Army funded it.", ["It works, e.g. on images.", "The U.S. Army funded it."]),
    ("Does it work? Yes! Mostly...", ["Does it work?", "Yes!", "Mostly..."]),
    ('He said "stop." Then it stopped.', ['He said "stop."', "Then it stopped."]),
    ("Results improve (see Figure 2.) Next we ablate.", ["Results improve (see Figure 2.)", "Next we ablate."]),
    ("It is approx. 40 percent. it goes on.", ["It is approx. 40 percent. it goes on."]),
    ("Introduction\nDeep learning works.", ["Introduction", "Deep learning works."]),
    ("No final punctuation", ["No final punctuation"]),
    ("", []),
    (" \n\t ", []),
])
def test_split_sentences(text, expected):
    assert split_sentences(text) == expected


def test_sentence_spans_are_offsets_into_the_text():
    text = "  First one.   Second one!  "
    assert sentence_spans(text) == [(2, 12), (15, 26)]


# --------------------------------------------------------------------------- audio utils


def test_silence_concat_duration():
    gap = audio_utils.silence(0.5, 24_000)
    assert gap.dtype == np.float32 and len(gap) == 12_000 and not gap.any()
    joined = audio_utils.concat([np.ones(10), gap, np.ones((2, 5))])
    assert joined.dtype == np.float32 and joined.shape == (12_020,)
    assert audio_utils.concat([]).shape == (0,)
    assert audio_utils.duration(joined, 1000) == pytest.approx(12.02)


def test_trim_silence_keeps_requested_margins():
    sr = 1000
    audio = np.concatenate([np.zeros(500), np.full(100, 0.5), np.zeros(700)]).astype(np.float32)
    trimmed = audio_utils.trim_silence(audio, sr, keep_lead=0.1, keep_trail=0.2)
    assert len(trimmed) == 100 + 100 + 200
    assert audio_utils.trim_silence(np.zeros(50, np.float32), sr, 0.1, 0.1).shape == (50,)


def test_wav_round_trip_and_downmix(tmp_path):
    audio = np.linspace(-0.5, 0.5, 2400, dtype=np.float32)
    audio_utils.write_wav(tmp_path / "clip.wav", audio, 24_000)
    back, sr = audio_utils.read_wav(tmp_path / "clip.wav")
    assert sr == 24_000 and back.dtype == np.float32
    np.testing.assert_allclose(back, audio, atol=1e-4)

    stereo = io.BytesIO()
    sf.write(stereo, np.stack([np.full(10, 0.2), np.full(10, 0.4)], axis=1), 8000, format="WAV")
    stereo.seek(0)
    mono, _ = audio_utils.read_wav(stereo)
    np.testing.assert_allclose(mono, 0.3, atol=1e-4)


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@needs_ffmpeg
def test_write_mp3(tmp_path):
    path = tmp_path / "tone.mp3"
    tone = 0.3 * np.sin(np.linspace(0, 2 * np.pi * 440, 48_000)).astype(np.float32)
    audio_utils.write_mp3(tone, 24_000, path)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,channels:format=duration",
                            "-of", "json", str(path)], capture_output=True, check=True, text=True)
    info = json.loads(probe.stdout)
    assert info["streams"][0]["codec_name"] == "mp3" and info["streams"][0]["channels"] == 1
    assert float(info["format"]["duration"]) == pytest.approx(2.0, abs=0.1)
    assert [p.name for p in tmp_path.iterdir()] == ["tone.mp3"]  # no temp files left behind


@needs_ffmpeg
def test_write_mp3_reports_ffmpeg_errors(tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        audio_utils.write_mp3(np.zeros(100, np.float32), 24_000, tmp_path / "bad.mp3", bitrate="bogus")
    assert not any(tmp_path.iterdir())


# --------------------------------------------------------------------------- ElevenLabs (mocked)

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
CHAR_SECONDS = 0.05


def _with_timestamps_response(text: str) -> dict:
    """Shaped like the documented 200 response of POST /v1/text-to-speech/{voice_id}/with-timestamps."""
    wav = io.BytesIO()
    sf.write(wav, np.zeros(round(len(text) * CHAR_SECONDS * 24_000) + 2400, np.float32), 24_000, format="WAV",
             subtype="PCM_16")
    alignment = {
        "characters": list(text),
        "character_start_times_seconds": [i * CHAR_SECONDS for i in range(len(text))],
        "character_end_times_seconds": [(i + 1) * CHAR_SECONDS for i in range(len(text))],
    }
    return {"audio_base64": base64.b64encode(wav.getvalue()).decode(), "alignment": alignment,
            "normalized_alignment": alignment}


@pytest.fixture
def fallback_prices(monkeypatch):
    monkeypatch.setattr(elevenlabs_engine, "model_table", lambda: FALLBACK_MODELS)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(elevenlabs_engine.time, "sleep", sleeps.append)
    return sleeps


def _engine(handler, model="eleven_flash_v2_5") -> ElevenLabsEngine:
    return ElevenLabsEngine(model, api_key="test-key", transport=httpx.MockTransport(handler))


def test_elevenlabs_synthesize_builds_sentence_chunks(fallback_prices):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_with_timestamps_response(json.loads(request.content)["text"]))

    text = "The loss drops by 3.2 dB. As Fig. 2 shows, it then levels off."
    result = _engine(handler).synthesize(text, VOICE_ID, previous_text="Before.", next_text="After.")

    (request,) = requests
    assert request.method == "POST"
    assert request.url.path == f"/v1/text-to-speech/{VOICE_ID}/with-timestamps"
    assert request.url.params["output_format"] == "wav_24000"
    assert request.headers["xi-api-key"] == "test-key"
    assert json.loads(request.content) == {"text": text, "model_id": "eleven_flash_v2_5",
                                           "previous_text": "Before.", "next_text": "After."}

    total = len(result.audio) / result.sample_rate
    assert result.sample_rate == 24_000 and result.audio.dtype == np.float32
    assert [c.text for c in result.chunks] == ["The loss drops by 3.2 dB.", "As Fig. 2 shows, it then levels off."]
    second = text.index("As Fig.")
    assert result.chunks[0].start == 0.0
    assert result.chunks[0].end == result.chunks[1].start == pytest.approx(second * CHAR_SECONDS)
    assert result.chunks[1].end == pytest.approx(total)
    assert result.characters_billed == len(text)
    assert result.cost_usd == pytest.approx(len(text) / 1000 * 0.04)


def test_elevenlabs_v3_omits_request_stitching_and_sends_speed(fallback_prices):
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_with_timestamps_response(bodies[-1]["text"]))

    result = _engine(handler, "eleven_v3").synthesize("Hello there.", VOICE_ID, 1.1, previous_text="x", next_text="y")
    assert bodies == [{"text": "Hello there.", "model_id": "eleven_v3", "voice_settings": {"speed": 1.1}}]
    assert result.cost_usd == pytest.approx(12 / 1000 * 0.08)


def test_elevenlabs_retries_429_then_succeeds(fallback_prices, no_sleep):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, json={"detail": "too many requests"})
        if len(calls) == 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json=_with_timestamps_response("Hi."))

    result = _engine(handler).synthesize("Hi.", VOICE_ID)
    assert len(calls) == 3 and no_sleep == [3.0, 2.0]
    assert [c.text for c in result.chunks] == ["Hi."]


def test_elevenlabs_client_errors_are_not_retried(fallback_prices, no_sleep):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": {"status": "invalid_api_key", "message": "Invalid API key"}})

    with pytest.raises(RuntimeError, match="HTTP 401.*Invalid API key"):
        _engine(handler).synthesize("Hi.", VOICE_ID)
    assert no_sleep == []


def test_elevenlabs_voices_are_fetched_once(fallback_prices):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.method == "GET" and request.url.path == "/v1/voices"
        return httpx.Response(200, json={"voices": [
            {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel", "category": "premade",
             "labels": {"accent": "American", "gender": "female"}},
            {"voice_id": "abc", "name": "My clone", "category": "cloned"},
        ]})

    engine = _engine(handler)
    voices = engine.voices()
    assert engine.voices() == voices and len(calls) == 1
    assert [(v.id, v.label, v.gender) for v in voices] == [
        ("21m00Tcm4TlvDq8ikWAM", "Rachel (premade, American, female)", "female"),
        ("abc", "My clone (cloned)", None),
    ]


def test_elevenlabs_without_key(monkeypatch, fallback_prices):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    engine = get_engine("elevenlabs")
    assert engine.available() == (False, "ELEVENLABS_API_KEY not set")
    assert engine.voices() == PREMADE_VOICES
    assert engine.default_voice() in {v.id for v in PREMADE_VOICES}
    with pytest.raises(RuntimeError, match="ELEVENLABS_API_KEY not set"):
        engine.synthesize("Hello.", VOICE_ID)


def test_elevenlabs_rejects_unknown_model_and_long_text(fallback_prices):
    with pytest.raises(ValueError, match="unknown ElevenLabs model"):
        ElevenLabsEngine("eleven_imaginary_v9", api_key="k")
    with pytest.raises(ValueError, match="5000-character request limit"):
        ElevenLabsEngine("eleven_v3", api_key="k").synthesize("a" * 5001, VOICE_ID)


def test_sentence_chunks_without_alignment():
    assert [(c.text, c.start, c.end) for c in sentence_chunks("One. Two.", None, 2.0)] == [("One. Two.", 0.0, 2.0)]


# --------------------------------------------------------------------------- Kokoro


def test_kokoro_voice_catalogue():
    engine = get_engine("kokoro")
    ids = [v.id for v in engine.voices()]
    assert len(ids) == len(set(ids)) == 28 and engine.default_voice() == "af_heart"
    assert all(v.id[:2] in {"af", "am", "bf", "bm"} and v.grade for v in VOICES)
    heart = next(v for v in VOICES if v.id == "af_heart")
    assert (heart.label, heart.language, heart.gender, heart.grade) == ("Heart (US, female)", "en-us", "female", "A")
    with pytest.raises(ValueError, match="unknown Kokoro voice"):
        engine.synthesize("Hello.", "xx_nobody")


def test_kokoro_empty_text_needs_no_model():
    result = KokoroEngine().synthesize("  ", "af_heart")
    assert result.audio.shape == (0,) and result.chunks == [] and result.sample_rate == 24_000


def test_kokoro_gpu_smoke():
    torch = pytest.importorskip("torch")
    pytest.importorskip("kokoro")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    engine = KokoroEngine()
    assert engine.available() == (True, None) and engine.device == "cuda"

    text = "Kokoro runs on the GPU. As Fig. 2 shows, the error is 3.5 percent!"
    result = engine.synthesize(text, "bf_emma")
    total = len(result.audio) / result.sample_rate
    assert result.sample_rate == 24_000 and result.audio.dtype == np.float32 and result.audio.ndim == 1
    assert 2.0 < total < 15.0 and np.abs(result.audio).max() <= 1.0
    assert [c.text for c in result.chunks] == split_sentences(text)
    assert result.chunks[0].start == 0.0 and result.chunks[-1].end == pytest.approx(total)
    assert all(a.end == b.start for a, b in pairwise(result.chunks))
    assert all(c.end > c.start for c in result.chunks)
