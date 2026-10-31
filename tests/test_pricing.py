"""Pricing: Haiku 5.5 clean-up cost and the per-engine TTS estimate."""

from __future__ import annotations

import pytest

from paper_reader import pricing
from paper_reader.cleanup import SYSTEM_PROMPT
from paper_reader.models import Block, CleanupStats, Paper, Section
from paper_reader.pricing import ELEVENLABS_MODELS, build_estimate, cleanup_cost, estimate_cleanup_tokens
from paper_reader.textfilter import apply_filter, speech_units

ENGINES = [
    {"name": "kokoro", "available": True, "reason": None, "device": "cuda"},
    {"name": "elevenlabs", "available": False, "reason": "no ELEVENLABS_API_KEY"},
]


def _paper(sections: int = 2) -> Paper:
    paper = Paper(paper_id="t.1v1", arxiv_id="t.1", title="Test paper", source_url="x", sections=[
        Section(id=f"S{i}", number=str(i), title="Method", blocks=[
            Block(id=f"b{i}1", type="paragraph", text="We train the policy with PPO [CITE] for 2B steps. " * 20),
            Block(id=f"b{i}2", type="equation", latex=r"x_{t+1} = f(x_t, u_t)", eq_number=str(i)),
            Block(id=f"b{i}3", type="figure", figure_id="F1", speak=False),
        ]) for i in range(1, sections + 1)
    ])
    apply_filter(paper)
    return paper


@pytest.mark.parametrize(("args", "kwargs", "usd"), [
    ((1_000_000, 0), {}, 0.10),
    ((0, 1_000_000), {}, 0.50),
    ((0, 0, 1_000_000, 0), {}, 0.01),
    ((0, 0, 0, 1_000_000), {}, 0.125),
    ((1_000_000, 1_000_000), {"long_prompt": True}, 3.00),
    ((15_000, 15_000), {}, 0.009),  # plan.md's paper: ~15K in, ~15K out ≈ $0.01
])
def test_cleanup_cost(args, kwargs, usd):
    assert cleanup_cost(*args, **kwargs) == pytest.approx(usd)


def test_estimate_cleanup_tokens_counts_the_cached_system_prompt_once():
    one, two = estimate_cleanup_tokens(_paper(1)), estimate_cleanup_tokens(_paper(2))
    system = len(SYSTEM_PROMPT) / pricing.CHARS_PER_TOKEN
    assert one[0] > system * pricing.CACHE_WRITE_MULTIPLIER
    added_input = two[0] - one[0]  # the second section: a cache read plus its own message
    assert system * pricing.CACHE_READ_MULTIPLIER < added_input < system * 0.5
    assert two[1] == pytest.approx(2 * one[1], rel=0.01)
    assert estimate_cleanup_tokens(Paper(paper_id="e", arxiv_id="e", title="E", source_url="x")) == (0, 0)


def test_build_estimate_before_cleanup():
    paper = _paper()
    estimate = build_estimate(paper, ENGINES)
    units = speech_units(paper)
    characters = sum(len(u.text) for u in units)
    words = sum(len(u.text.split()) for u in units)
    assert (estimate.characters, estimate.words) == (characters, words)
    assert estimate.audio_minutes == pytest.approx(words / pricing.WORDS_PER_MINUTE, abs=0.01)
    assert estimate.cleanup_model == "claude-haiku-5-5" and not estimate.cleanup_cost_is_actual
    assert 0 < estimate.cleanup_cost_usd < 0.01

    kokoro, *eleven = estimate.tts
    assert (kokoro.engine, kokoro.cost_usd, kokoro.available) == ("kokoro", 0.0, True)
    assert kokoro.est_processing_minutes == pytest.approx(estimate.audio_minutes / 100 + 0.15, abs=0.01)
    assert [e.model for e in eleven] == list(ELEVENLABS_MODELS)
    for option in eleven:
        meta = ELEVENLABS_MODELS[option.model]
        assert option.label == meta["label"]
        assert option.cost_usd == pytest.approx(characters / 1000 * meta["usd_per_1k_chars"], abs=1e-4)
        assert (option.available, option.note) == (False, "no ELEVENLABS_API_KEY")


@pytest.mark.parametrize(("stats", "model", "usd", "actual"), [
    (CleanupStats(model="claude-haiku-5-5", input_tokens=9000, output_tokens=8000, cost_usd=0.0049),
     "claude-haiku-5-5", 0.0049, True),
    (CleanupStats(skipped_reason="no ANTHROPIC_API_KEY"), None, 0.0, True),
    (CleanupStats(model="claude-haiku-5-5", input_tokens=500, cost_usd=0.0001, skipped_reason="AuthenticationError"),
     "claude-haiku-5-5", 0.0001, True),
])
def test_build_estimate_after_cleanup(stats, model, usd, actual):
    paper = _paper()
    paper.cleanup = stats
    estimate = build_estimate(paper, ENGINES)
    assert (estimate.cleanup_model, estimate.cleanup_cost_usd, estimate.cleanup_cost_is_actual) == (model, usd, actual)


def test_engines_and_devices():
    paper = _paper()
    estimate = build_estimate(paper, [{"name": "kokoro", "available": False, "reason": "not installed"},
                                      {"name": "unknown", "available": True}])
    [kokoro] = estimate.tts
    assert (kokoro.available, kokoro.note) == (False, "not installed")
    assert kokoro.est_processing_minutes == pytest.approx(estimate.audio_minutes / 3 + 0.15, abs=0.01)  # CPU
    assert build_estimate(paper, []).tts == []


def test_elevenlabs_table():
    assert set(ELEVENLABS_MODELS) == {"eleven_flash_v2_5", "eleven_v4", "eleven_v3", "eleven_multilingual_v2"}
    for meta in ELEVENLABS_MODELS.values():
        assert set(meta) == {"label", "usd_per_1k_chars", "max_chars"}
        assert meta["label"].startswith("ElevenLabs ")
