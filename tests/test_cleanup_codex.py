"""Codex clean-up runner, with `codex exec` replaced by a fake process (no network, no CLI needed)."""

from __future__ import annotations

import json
import subprocess

import pytest

from paper_reader import cleanup_codex, pipeline
from tests.test_cleanup import GOOD_ANSWERS, _paper


def _fake_codex(answers: dict[str, str], *, fail_sections: tuple[str, ...] = ()):
    """A stand-in for subprocess.run(["codex", "exec", ...]) that answers from `answers`."""
    def run(cmd, input, capture_output, text, timeout):
        assert cmd[:2] == ["codex", "exec"] and "--output-schema" in cmd and "read-only" in cmd
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
