"""Pipeline tests with a fake TTS engine: figure cues, price gate, clip cache, manifest timing."""

from __future__ import annotations

import numpy as np
import pytest

from paper_reader import pipeline
from paper_reader.models import Block, Figure, Paper, Section
from paper_reader.tts.base import Chunk, SynthResult, VoiceInfo

SR = 24000


class FakeEngine:
    name = "kokoro"
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def available(self):
        return True, None

    def voices(self):
        return [VoiceInfo("af_heart", "Heart")]

    def default_voice(self):
        return "af_heart"

    def synthesize(self, text, voice, speed=1.0, *, previous_text=None, next_text=None):
        self.calls += 1
        sentences = [s.strip() + "." for s in text.split(".") if s.strip()] or [text]
        chunks, t = [], 0.0
        for s in sentences:
            dur = 0.05 * len(s.split())
            chunks.append(Chunk(s, t, t + dur))
            t += dur
        return SynthResult(audio=np.zeros(int(t * SR), dtype=np.float32), sample_rate=SR, chunks=chunks)


def _paper() -> Paper:
    figs = {
        "S1.F1": Figure(id="S1.F1", kind="figure", number="1", label="Figure 1"),
        "S1.F2": Figure(id="S1.F2", kind="figure", number="2", label="Figure 2"),
        "S1.T1": Figure(id="S1.T1", kind="table", number="1", label="Table 1"),
    }
    blocks = [
        Block(id="b0001", type="paragraph", text="We walk. Figure 2 shows terrain.", fig_refs=["S1.F2"]),
        Block(id="b0002", type="figure", figure_id="S1.F1", speak=False),
        Block(id="b0003", type="paragraph", text="Robots climb stairs. Results are good.", fig_refs=["S1.T1"]),
    ]
    return Paper(paper_id="test.0001v1", arxiv_id="test.0001", title="T", source_url="x",
                 sections=[Section(id="S1", number="1", title="Intro", blocks=blocks)], figures=figs)


@pytest.fixture
def env(tmp_path, monkeypatch):
    from paper_reader.textfilter import apply_filter

    monkeypatch.setattr(pipeline, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache")
    fake = FakeEngine()
    monkeypatch.setattr(pipeline, "engine", lambda name, model=None: fake)
    paper = _paper()
    apply_filter(paper)
    pipeline.save_paper(paper)
    return paper, fake


def test_mentioned_figures():
    index = {("figure", "3"): "F3", ("figure", "4"): "F4", ("table", "2"): "T2"}
    assert pipeline.mentioned_figures("as Figures 3 and 4 show, and Table 2", index) == ["F3", "F4", "T2"]
    assert pipeline.mentioned_figures("no figures here", index) == []


def test_floats_named_in_the_text_cue_at_their_mention_not_on_pass_by():
    index = pipeline._figure_index(_paper())
    # Figures 1 and 2 were placed before this paragraph, which names Figure 2 in its 2nd sentence.
    unit = pipeline._Unit("b9", "block", "x", fig_refs=["S1.F2"], pending_figs=["S1.F1", "S1.F2"])
    result = FakeEngine().synthesize("Robots walk. See Figure 2 now.", "af_heart")
    segs = list(pipeline._unit_segments(unit, result, 0.0, index, set()))
    assert segs[0].figure_ids == ["S1.F1"]  # passed by, never named
    assert segs[1].figure_ids == ["S1.F2"]  # named here


def test_price_gate_refuses_without_acceptance(env):
    with pytest.raises(pipeline.CostNotAccepted):
        pipeline.run_tts("test.0001v1", engine_name="kokoro", accepted_cost_usd=None)


def test_run_tts_manifest_and_figure_cues(env):
    paper, fake = env
    m = pipeline.run_tts(paper.paper_id, engine_name="kokoro", accepted_cost_usd=0.0)

    starts = [s.start for s in m.segments]
    assert starts == sorted(starts) and all(s.end >= s.start for s in m.segments)
    assert m.segments[-1].end <= m.duration + 1e-6
    assert (pipeline.paper_dir(paper.paper_id) / m.audio_file).stat().st_size > 0

    by_unit = {}
    for seg in m.segments:
        by_unit.setdefault(seg.unit_id, []).append(seg)
    # "Figure 2" is spoken in the 2nd sentence of b0001 -> cue on that piece, not before.
    assert by_unit["b0001"][0].figure_ids == []
    assert "S1.F2" in by_unit["b0001"][1].figure_ids
    # The unmentioned float F1 (placed before b0003) and the parser ref T1 cue on b0003's first piece.
    assert by_unit["b0003"][0].figure_ids[:2] == ["S1.F1", "S1.T1"]

    # Second run is served from the clip cache.
    calls = fake.calls
    pipeline.run_tts(paper.paper_id, engine_name="kokoro", accepted_cost_usd=0.0)
    assert fake.calls == calls


class BatchingEngine(FakeEngine):
    """A FakeEngine that also voices many units per call, like Qwen3-TTS."""

    def __init__(self):
        super().__init__()
        self.many_calls: list[int] = []

    def synthesize_many(self, texts, voice, speed=1.0):
        self.many_calls.append(len(texts))
        return [FakeEngine.synthesize(self, t, voice, speed) for t in texts]


def test_batched_engines_pool_units_and_match_the_per_unit_manifest(env, monkeypatch):
    paper, per_unit_engine = env
    sequential = pipeline.run_tts(paper.paper_id, engine_name="kokoro", accepted_cost_usd=0.0)

    batching = BatchingEngine()
    monkeypatch.setattr(pipeline, "engine", lambda name, model=None: batching)
    monkeypatch.setattr(pipeline, "CACHE_DIR", pipeline.CACHE_DIR.parent / "cache2")
    batched = pipeline.run_tts(paper.paper_id, engine_name="kokoro", accepted_cost_usd=0.0)

    assert batching.many_calls == [3]  # section title + 2 paragraphs in one call
    assert [s.model_dump() for s in batched.segments] == [s.model_dump() for s in sequential.segments]
    assert batched.duration == sequential.duration

    pipeline.run_tts(paper.paper_id, engine_name="kokoro", accepted_cost_usd=0.0)
    assert batching.many_calls == [3]  # everything cached now


def test_offline_files_cover_text_figures_and_one_narration(env):
    paper, _ = env
    m = pipeline.run_tts(paper.paper_id, engine_name="kokoro", accepted_cost_usd=0.0)
    listing = pipeline.offline_files(paper.paper_id)
    urls = [f["url"] for f in listing["files"]]
    base = f"/papers/{paper.paper_id}"
    assert urls == [f"{base}/paper.json", f"{base}/audio/{m.run_id}/manifest.json", f"{base}/{m.audio_file}"]
    assert listing["run_id"] == m.run_id and listing["total_bytes"] == sum(f["bytes"] for f in listing["files"]) > 0
    with pytest.raises(FileNotFoundError):
        pipeline.offline_files(paper.paper_id, "no-such-run")
