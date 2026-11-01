"""LLM clean-up through the Codex CLI (`codex exec`), by default with OpenAI's gpt-6-luna.

Same requests, system prompt, worked examples and rewrite guard as the Claude Haiku pass in
cleanup.py. Each section request is one non-interactive `codex exec` run, whose final message is
forced into RESPONSE_SCHEMA with --output-schema. The run happens in an empty scratch directory
with a read-only sandbox, so the agent has nothing to touch. Usage goes against the Codex login
(a ChatGPT plan), not per-token API billing, so cost_usd stays 0 and tokens are recorded.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import ValidationError

from . import cleanup
from . import glossary as glossary_stage
from .cleanup import EquationMode, _Answer, system_prompt
from .models import CleanupStats, Paper

CODEX_MODEL = "gpt-6-luna"
CODEX_EFFORT = "low"
GLOSSARY_EFFORT = "medium"  # one or two requests per paper, and every name depends on them
TIMEOUT_SECONDS = 300
RUN_RULES = ("You are running non-interactively as a text transformer. Do not run commands, read files "
             "or use tools. Reply with the JSON object only.")


def codex_available() -> tuple[bool, str | None]:
    """The CLI is installed and has a login (no network call)."""
    if not shutil.which("codex"):
        return False, "the codex CLI is not installed"
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    if not (home / "auth.json").exists():
        return False, "codex is not logged in (run `codex login`)"
    return True, None


def _codex(prompt: str, schema: Path, out: Path, workdir: Path, model: str, effort: str) -> tuple[str, dict]:
    """One `codex exec` run -> (final message, usage)."""
    cmd = ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only",
           "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--output-schema", str(schema),
           "-o", str(out), "--json", "-C", str(workdir), "-"]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    if proc.returncode != 0 or not out.exists():
        detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"codex exited with {proc.returncode}: {detail[:300]}")
    usage: dict = {}
    for line in proc.stdout.splitlines():
        if '"turn.completed"' in line:
            usage = json.loads(line).get("usage") or {}
    return out.read_text(encoding="utf-8"), usage


def _run_request(request: dict, schema: Path, workdir: Path, model: str, effort: str,
                 mode: EquationMode = "symbols", inline: EquationMode | None = None) -> tuple[list[dict], dict]:
    """One narration request -> (answers, usage)."""
    out = workdir.parent / f"answer_{request['request_id']}.json"
    prompt = f"{system_prompt(mode, inline)}\n\n---\n\n{RUN_RULES}\n\n{request['user_message']}"
    text, usage = _codex(prompt, schema, out, workdir, model, effort)
    try:
        answers = _Answer.model_validate_json(text).blocks
    except ValidationError as exc:
        raise RuntimeError(f"the answer is not the expected JSON ({exc.error_count()} errors)") from exc
    return [a.model_dump() for a in answers], usage


def _glossary(paper: Paper, tmp: Path, workdir: Path, model: str, max_workers: int,
              tokens: dict, errors: list[str]) -> dict:
    """Stage 2: the paper-wide name of every symbol, before any narration request."""
    schema = tmp / "glossary_schema.json"
    schema.write_text(json.dumps(glossary_stage.SCHEMA), encoding="utf-8")
    symbols = glossary_stage.inventory(paper)

    def one(item):
        i, message = item
        prompt = f"{glossary_stage.SYSTEM_PROMPT}\n\n---\n\n{RUN_RULES}\n\n{message}"
        try:
            text, usage = _codex(prompt, schema, tmp / f"glossary_{i}.json", workdir, model, GLOSSARY_EFFORT)
            return json.loads(text).get("symbols", []), usage
        except (RuntimeError, subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
            errors.append(f"glossary: {exc}")
            return [], {}

    answers: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for found, usage in pool.map(one, enumerate(glossary_stage.requests(paper, symbols))):
            answers += found
            for key in tokens:
                tokens[key] += int(usage.get(key) or 0)
    return cleanup.accept_glossary(paper, answers, symbols)


def cleanup_paper_codex(paper: Paper, *, model: str = CODEX_MODEL, effort: str = CODEX_EFFORT,
                        max_workers: int = 4,
                        progress_cb: Callable[[float, str], None] | None = None,
                        equations: EquationMode = "symbols", inline: EquationMode | None = None) -> CleanupStats:
    """Run the clean-up through Codex. Mutates blocks and sets paper.cleanup (like cleanup_paper).
    In meaning mode, blocks that named a symbol differently are asked once more with the names."""
    label = f"codex:{model}"
    ok, reason = codex_available()
    if not ok:
        paper.cleanup = CleanupStats(model=label, skipped_reason=reason, equation_mode=equations,
                                     inline_mode=inline or equations)
        return paper.cleanup

    errors: list[str] = []
    tokens = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    with tempfile.TemporaryDirectory(prefix="paper-reader-codex-") as tmp:
        workdir = Path(tmp) / "work"
        workdir.mkdir()
        names = None
        if "meaning" in (equations, inline or equations):
            if progress_cb:
                progress_cb(0.0, "Naming the symbols for the whole paper")
            cleanup.apply_filter(paper)
            names = _glossary(paper, Path(tmp), workdir, model, max_workers, tokens, errors)
        run = cleanup.begin_import(paper, model=label, mode=equations, inline=inline, glossary=names)  # same requests and guard
        schema = Path(tmp) / "schema.json"
        schema.write_text(json.dumps(cleanup.response_schema(run.mode, run.inline)), encoding="utf-8")

        def ask(requests: list, first_id: int, label_prefix: str) -> tuple[list[dict], int]:
            """Run the requests through codex; returns the answers and how many requests failed."""
            dicts = [cleanup.request_dict(r, first_id + i) for i, r in enumerate(requests)]

            def one(request: dict):
                try:
                    return request, _run_request(request, schema, workdir, model, effort, run.mode, run.inline), None
                except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
                    return request, None, f"{request['section']}: {exc}"

            answers, failed = [], 0
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for done, (request, result, error) in enumerate(pool.map(one, dicts), 1):
                    if error:
                        errors.append(error)
                        failed += 1
                    else:
                        answers.extend(result[0])
                        for key in tokens:
                            tokens[key] += int(result[1].get(key) or 0)
                    if progress_cb:
                        progress_cb(done / max(len(dicts), 1), f"{label_prefix} {done}/{len(dicts)}: {request['section']}")
            return answers, failed

        answers, failed = ask(run.requests, 0, "Clean-up")
        if run.requests and failed == len(run.requests):
            paper.cleanup = CleanupStats(model=label, skipped_reason=f"every Codex request failed, e.g. {errors[0]}",
                                         equation_mode=run.mode, inline_mode=run.inline)
            return paper.cleanup
        cleanup.apply_answers(run, run.requests, answers)
        if repairs := cleanup.repair_requests(run):  # inconsistent names and rejected answers, asked once more
            repaired, _ = ask(repairs, len(run.requests), "Repair")
            cleanup.apply_answers(run, repairs, repaired)

    stats = cleanup.finish(run)
    stats.input_tokens, stats.output_tokens = tokens["input_tokens"], tokens["output_tokens"]
    stats.cache_read_input_tokens = tokens["cached_input_tokens"]
    paper.warnings += [f"clean-up: {e}; kept the filter text" for e in errors]
    return stats
