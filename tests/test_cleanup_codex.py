"""Codex clean-up runner, with `codex exec` replaced by a fake process (no network, no CLI needed)."""

from __future__ import annotations

import json
import subprocess

import pytest

from paper_reader import cleanup_codex, glossary, pipeline
from tests.test_cleanup import GOOD_ANSWERS, _paper


def _fake_codex(answers: dict[str, str], *, fail_sections: tuple[str, ...] = (), names: dict[str, str] | None = None):
    """A stand-in for subprocess.run(["codex", "exec", ...]) that answers from `answers`."""
    def run(cmd, input, capture_output, text, timeout):
        assert cmd[:2] == ["codex", "exec"] and "--output-schema" in cmd and "read-only" in cmd
        if glossary.SYSTEM_PROMPT in input:  # stage 2 (meaning mode): name the symbols
            assert 'model_reasoning_effort="medium"' in cmd
            symbols = [json.loads(line) for line in input.splitlines() if line.startswith('{"id"')]
            with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as f:
                json.dump({"symbols": [{"id": s["id"], "name": (names or {})[s["latex"]]}
                                       for s in symbols if s["latex"] in (names or {})]}, f)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        assert cleanup_codex.RUN_RULES in input and "Return the narration" in input
        if any(f"Section: {name}" in input for name in fail_sections):
            return subprocess.CompletedProcess(cmd, 1, "", "error: model overloaded")
        ids = [b["id"] for b in json.loads(input[input.index("blocks:\n[") + len("blocks:\n"):])]
        out = cmd[cmd.index("-o") + 1]
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"blocks": [{"id": i, "speech": answers[i]} for i in ids]}, f)
        usage = {"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 200, "output_tokens": 50}}
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"type": "turn.started"}) + "\n" + json.dumps(usage), "")
    return run


@pytest.fixture
def codex_ok(monkeypatch):
    monkeypatch.setattr(cleanup_codex, "codex_available", lambda: (True, None))


def test_codex_answers_go_through_the_same_guard(codex_ok, monkeypatch):
    monkeypatch.setattr(cleanup_codex.subprocess, "run", _fake_codex(GOOD_ANSWERS))
    paper = _paper()
    stats = cleanup_codex.cleanup_paper_codex(paper, max_workers=2)
    blocks = {b.id: b for b in paper.iter_blocks()}
    assert stats.model == "codex:gpt-6-luna" and stats.cost_usd == 0.0
    assert (stats.blocks_llm, stats.blocks_rejected) == (4, 1)  # b0006 is a summary: rejected
    assert blocks["b0002"].speech == GOOD_ANSWERS["b0002"] and blocks["b0006"].speech_source == "llm_rejected"
    # two sections, then one repair request for the rejected b0006 (rejected again: same answer)
    assert (stats.input_tokens, stats.output_tokens, stats.cache_read_input_tokens) == (3000, 150, 600)


def test_a_failed_section_keeps_the_filter_text(codex_ok, monkeypatch):
    monkeypatch.setattr(cleanup_codex.subprocess, "run", _fake_codex(GOOD_ANSWERS, fail_sections=("Section 2.",)))
    paper = _paper()
    stats = cleanup_codex.cleanup_paper_codex(paper)
    blocks = {b.id: b for b in paper.iter_blocks()}
    assert blocks["b0001"].speech_source == "llm" and blocks["b0004"].speech_source == "filter"
    assert stats.skipped_reason is None and any("model overloaded" in w for w in paper.warnings)


def test_all_failing_or_missing_cli_skips_cleanly(codex_ok, monkeypatch):
    monkeypatch.setattr(cleanup_codex.subprocess, "run", _fake_codex(GOOD_ANSWERS, fail_sections=("Section",)))
    stats = cleanup_codex.cleanup_paper_codex(_paper())
    assert stats.skipped_reason.startswith("every Codex request failed")

    monkeypatch.setattr(cleanup_codex, "codex_available", lambda: (False, "the codex CLI is not installed"))
    assert cleanup_codex.cleanup_paper_codex(_paper()).skipped_reason == "the codex CLI is not installed"


def test_runner_choice_in_the_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PAPERS_DIR", tmp_path)
    runners = [{"id": "haiku", "label": "H", "available": False, "reason": "no key"},
               {"id": "codex", "label": "C", "available": True, "reason": None}]
    monkeypatch.setattr(pipeline, "cleanup_runners", lambda: runners)
    monkeypatch.setattr(pipeline, "estimate", lambda paper_id, paper=None: None)
    assert pipeline.default_cleanup_runner() == "codex"

    paper = pipeline.run_cleanup(_paper(), "haiku")
    assert paper.cleanup.skipped_reason == "no key"
    assert pipeline.run_cleanup(_paper(), None).cleanup.skipped_reason == "LLM clean-up disabled"
    with pytest.raises(ValueError):
        pipeline.run_cleanup(_paper(), "gpt")


def test_meaning_mode_names_the_symbols_first(codex_ok, monkeypatch):
    """Stage 2 runs once before the narration, and its names travel with the blocks that use them."""
    prompts = []
    fake = _fake_codex({**GOOD_ANSWERS, "b0005": "Equation 4: the loss equals the sum over time steps t from 1 to T "
                                                 "of the squared norm of x hat t minus x t, plus beta times the KL "
                                                 "divergence between q and p."},
                       names={"\\beta": "beta", "x_{t}": "observation"})

    def recording(cmd, input, **kwargs):
        prompts.append(input)
        return fake(cmd, input, **kwargs)

    monkeypatch.setattr(cleanup_codex.subprocess, "run", recording)
    paper = _paper()
    stats = cleanup_codex.cleanup_paper_codex(paper, equations="meaning")
    assert glossary.SYSTEM_PROMPT in prompts[0]  # the glossary request comes first
    assert stats.glossary_names == 1  # "observation" is not a word of the test paper: dropped
    equation_request = next(p for p in prompts if '"b0005"' in p)
    assert '"names": {' in equation_request and '"\\\\beta": "beta"' in equation_request
