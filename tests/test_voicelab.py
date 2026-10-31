"""Voice lab: lab text, rendering into the per-paper state, selection defaults (fake engines, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from paper_reader import pipeline, voicelab
from paper_reader.models import Block, Paper, Section
from paper_reader.textfilter import apply_filter
from paper_reader.tts.base import Chunk, SynthResult, VoiceInfo


class FakeEngine:
    def __init__(self, name):
        self.name = name

    def available(self):
        return True, None

    def voices(self):
        return [VoiceInfo("v1", "Voice one", gender="male"), VoiceInfo("v2", "Voice two", gender="female")]

    def default_voice(self):
        return "v1"

    def synthesize(self, text, voice, speed=1.0, **_):
        return SynthResult(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000, chunks=[Chunk(text, 0, 0.1)])


@pytest.fixture
def lab(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(pipeline, "SAMPLES_DIR", tmp_path / "samples")
    monkeypatch.setattr(pipeline, "SELECTED_VOICE_FILE", tmp_path / "voices" / "selected.json")
    engines = {}
    monkeypatch.setattr(pipeline, "engine", lambda name, model=None: engines.setdefault(name, FakeEngine(name)))
    abstract = Block(id="b0001", type="paragraph",
                     text="First sentence here. Second sentence follows. Third one is not in the lab.")
    paper = Paper(paper_id="lab.1v1", arxiv_id="lab.1", title="Lab", source_url="x",
                  sections=[Section(id="abstract", title="Abstract", kind="abstract", blocks=[abstract])])
    apply_filter(paper)
    pipeline.save_paper(paper)
    return paper


def test_lab_text_is_the_first_two_sentences(lab):
    assert voicelab.lab_text(lab.paper_id) == "First sentence here. Second sentence follows."


def test_render_kokoro_records_clips_and_resets_when_text_changes(lab):
    assert voicelab.render(lab.paper_id, "kokoro", ["all"]) == {"rendered": 2}
    state = voicelab._load_state(lab.paper_id, voicelab.lab_text(lab.paper_id))
    assert set(state["kokoro"]) == {"v1", "v2"}
    assert (voicelab._lab_dir(lab.paper_id) / state["kokoro"]["v1"]["file"]).exists()
    assert voicelab._load_state(lab.paper_id, "Different text.")["kokoro"] == {}
    with pytest.raises(ValueError):
        voicelab.render(lab.paper_id, "kokoro", ["nope"])
    with pytest.raises(ValueError):
        voicelab.render(lab.paper_id, "elevenlabs", ["v1"])
    with pytest.raises(ValueError):
        voicelab.render(lab.paper_id, "qwen3", ["v1"], models=["9B"])


def test_render_qwen_models_are_stored_per_model(lab):
    voicelab.render(lab.paper_id, "qwen3", ["v2"], models=["1.7B", "0.6B"])
    state = voicelab._load_state(lab.paper_id, voicelab.lab_text(lab.paper_id))
    assert set(state["qwen3"]["v2"]) == {"0.6B", "1.7B"}


def test_selection_fills_missing_voice_and_model(lab):
    # Nothing selected yet: the built-in default (Qwen3 0.6B, technical_male) applies.
    assert pipeline.selected_voice() == pipeline.DEFAULT_VOICE_SELECTION
    assert pipeline.resolve_voice("qwen3", None, None) == ("technical_male", "0.6B")
    pipeline.select_voice("qwen3", "v2", "0.6B")
    assert pipeline.selected_voice() == {"engine": "qwen3", "voice": "v2", "model": "0.6B"}
    assert pipeline.resolve_voice("qwen3", None, None) == ("v2", "0.6B")
    assert pipeline.resolve_voice("qwen3", None, "1.7B") == (None, "1.7B")  # other model: no voice carry-over
    assert pipeline.resolve_voice("kokoro", None, None) == (None, None)  # other engine: untouched
    assert pipeline.resolve_voice("qwen3", "v1", None) == ("v1", "0.6B")  # an explicit voice wins
    with pytest.raises(ValueError):
        pipeline.select_voice("kokoro", "nope")
