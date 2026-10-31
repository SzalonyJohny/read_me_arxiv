"""Qwen3-TTS engine: logic that needs no model weights (the GPU run is exercised by `paper-reader samples`)."""

from __future__ import annotations

import numpy as np
import pytest

from paper_reader.tts.qwen_engine import MODELS, SENTENCE_GAP, VOICE_PRESETS, QwenTTSEngine

SR = 24_000


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_one_chunk_per_sentence_with_gaps():
    sentences = ["First sentence.", "Second one here.", "Third."]
    result = QwenTTSEngine._assemble(sentences, [_tone(1.0), _tone(0.5), _tone(0.8)], SR, 1.0)
    assert [c.text for c in result.chunks] == sentences
    starts = [c.start for c in result.chunks]
    assert starts == sorted(starts) and result.chunks[0].start == 0.0
    assert result.chunks[1].start - result.chunks[0].end == pytest.approx(SENTENCE_GAP, abs=1e-3)
    assert result.chunks[-1].end == pytest.approx(len(result.audio) / SR, abs=1e-3)
    assert result.audio.dtype == np.float32 and np.abs(result.audio).max() <= 1.0


def test_speed_shortens_the_audio():
    normal = QwenTTSEngine._assemble(["A.", "B."], [_tone(1.0), _tone(1.0)], SR, 1.0)
    fast = QwenTTSEngine._assemble(["A.", "B."], [_tone(1.0), _tone(1.0)], SR, 1.25)
    assert len(fast.audio) < 0.85 * len(normal.audio)


def test_presets_models_and_validation(tmp_path):
    assert VOICE_PRESETS["technical_male"] == "Clear and precise and technical, male middle age man"
    assert set(MODELS) == {"0.6B", "1.7B", "1.7B-design"}
    engine = QwenTTSEngine("0.6B", voice_dir=tmp_path)
    assert engine.default_voice() == "technical_male"
    assert [v.gender for v in engine.voices()] == ["male"]
    with pytest.raises(ValueError):
        QwenTTSEngine("3B")
    with pytest.raises(ValueError):
        engine.synthesize("Hello there.", "no_such_voice")
    assert len(engine.synthesize("", "technical_male").audio) == 0


def test_voice_registry_takes_and_delete(tmp_path):
    from paper_reader.tts.qwen_engine import VoiceRegistry

    reg = VoiceRegistry(tmp_path)
    first = reg.add("  Warm,   calm female narrator ")
    assert first["description"] == "Warm, calm female narrator" and first["take"] == 1
    again = reg.add("Warm, calm female narrator")
    redo = reg.add(redesign_of=first["id"])
    assert (again["take"], redo["take"]) == (2, 3) and len({first["id"], again["id"], redo["id"]}) == 3
    preset_take = reg.add(VOICE_PRESETS["technical_male"])
    assert preset_take["take"] == 2  # the preset is take 1 of its description

    (tmp_path / f"{first['id']}-abc.wav").write_bytes(b"x")
    reg.delete(first["id"])
    assert first["id"] not in {v["id"] for v in reg.all()} and not list(tmp_path.glob(f"{first['id']}-*"))
    with pytest.raises(ValueError):
        reg.delete("technical_male")
    with pytest.raises(KeyError):
        reg.delete("no_such_voice")
    with pytest.raises(ValueError):
        reg.add("   ")

    engine = QwenTTSEngine("1.7B", voice_dir=tmp_path)
    voices = {v.id: v for v in engine.voices()}
    assert voices[again["id"]].gender == "female" and voices["technical_male"].gender == "male"


class _FakeQwenModel:
    """Stands in for Qwen3TTSModel: clip length encodes the sentence, so mapping errors show."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def generate_voice_clone(self, text, language, voice_clone_prompt):
        self.batches.append(list(text))
        return [_tone(0.01 * len(t)) for t in text], SR


def test_synthesize_many_pools_sorts_and_maps_back(tmp_path, monkeypatch):
    from paper_reader.tts import qwen_engine

    fake = _FakeQwenModel()
    monkeypatch.setattr(qwen_engine, "_model", lambda repo, device: fake)
    monkeypatch.setattr(qwen_engine, "BATCH_SENTENCES", 4)
    engine = QwenTTSEngine("0.6B", voice_dir=tmp_path, device="cpu")
    monkeypatch.setattr(engine, "reference_clip", lambda voice: (tmp_path / "ref.wav", "ref"))
    monkeypatch.setattr(engine, "_clone_prompt", lambda model, voice, ref: ["prompt"])

    texts = ["A short one. This second sentence is quite a bit longer than the first.",
             "", "Mid length sentence here. Tiny. Another fairly long sentence closes the unit."]
    results = engine.synthesize_many(texts, "technical_male")

    assert len(fake.batches) == 2 and all(len(b) <= 4 for b in fake.batches)  # 5 sentences, batches of 4
    pooled = [s for b in fake.batches for s in b]
    assert pooled == sorted(pooled, key=len)  # one long sentence doesn't hold up a batch of short ones
    assert len(results[1].audio) == 0
    for text, result in zip(texts, results):
        expected = qwen_engine.split_sentences(text)
        assert [c.text for c in result.chunks] == expected
        for chunk in result.chunks:  # each chunk's duration matches its own sentence's clip
            assert chunk.end - chunk.start == pytest.approx(0.01 * len(chunk.text), abs=0.02)


def test_decoding_is_chunked():
    from paper_reader.tts import qwen_engine

    calls = []

    class Tokenizer:
        def decode(self, encoded):
            calls.append(len(encoded))
            return [f"wav{e['audio_codes']}" for e in encoded], SR

    class Model:
        model = type("Inner", (), {"speech_tokenizer": Tokenizer()})()

    model = Model()
    qwen_engine._chunk_decoding(model)
    wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": i} for i in range(10)])
    assert wavs == [f"wav{i}" for i in range(10)] and sr == SR
    assert calls == [4, 4, 2]
