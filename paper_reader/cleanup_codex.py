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

from .cleanup import RESPONSE_SCHEMA, SYSTEM_PROMPT, _Answer, export_requests, import_answers
from .models import CleanupStats, Paper

CODEX_MODEL = "gpt-6-luna"
CODEX_EFFORT = "low"
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


def _run_request(request: dict, schema: Path, workdir: Path, model: str, effort: str) -> tuple[list[dict], dict]:
    """One `codex exec` run -> (answers, usage)."""
    out = workdir.parent / f"answer_{request['request_id']}.json"
    prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{RUN_RULES}\n\n{request['user_message']}"
    cmd = ["codex", "exec", "-m", model, "-c", f'model_reasoning_effort="{effort}"', "-s", "read-only",
           "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--output-schema", str(schema),
           "-o", str(out), "--json", "-C", str(workdir), "-"]
    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    if proc.returncode != 0 or not out.exists():
        detail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        raise RuntimeError(f"codex exited with {proc.returncode}: {detail[:300]}")
    try:
        answers = _Answer.model_validate_json(out.read_text(encoding="utf-8")).blocks
    except ValidationError as exc:
        raise RuntimeError(f"the answer is not the expected JSON ({exc.error_count()} errors)") from exc
    usage: dict = {}
    for line in proc.stdout.splitlines():
        if '"turn.completed"' in line:
            usage = json.loads(line).get("usage") or {}
    return [a.model_dump() for a in answers], usage


def cleanup_paper_codex(paper: Paper, *, model: str = CODEX_MODEL, effort: str = CODEX_EFFORT,
                        max_workers: int = 4,
                        progress_cb: Callable[[float, str], None] | None = None) -> CleanupStats:
    """Run the clean-up through Codex. Mutates blocks and sets paper.cleanup (like cleanup_paper)."""
    label = f"codex:{model}"
    ok, reason = codex_available()
    if not ok:
        paper.cleanup = CleanupStats(model=label, skipped_reason=reason)
        return paper.cleanup

    requests = [r for group in export_requests(paper) for r in group]
    answers: list[dict] = []
    errors: list[str] = []
    tokens = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    with tempfile.TemporaryDirectory(prefix="paper-reader-codex-") as tmp:
        schema = Path(tmp) / "schema.json"
        schema.write_text(json.dumps(RESPONSE_SCHEMA), encoding="utf-8")
        workdir = Path(tmp) / "work"
        workdir.mkdir()

        def run(request: dict):
            try:
                return request, _run_request(request, schema, workdir, model, effort), None
            except (RuntimeError, subprocess.TimeoutExpired, OSError) as exc:
                return request, None, f"{request['section']}: {exc}"

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for done, (request, result, error) in enumerate(pool.map(run, requests), 1):
                if error:
                    errors.append(error)
                else:
                    answers.extend(result[0])
                    for key in tokens:
                        tokens[key] += int(result[1].get(key) or 0)
                if progress_cb:
                    progress_cb(done / max(len(requests), 1), f"Clean-up {done}/{len(requests)}: {request['section']}")

    if requests and len(errors) == len(requests):
        paper.cleanup = CleanupStats(model=label, skipped_reason=f"every Codex request failed, e.g. {errors[0]}")
        return paper.cleanup
    stats = import_answers(paper, answers, model=label)  # same guard as the Haiku pass
    stats.input_tokens, stats.output_tokens = tokens["input_tokens"], tokens["output_tokens"]
    stats.cache_read_input_tokens = tokens["cached_input_tokens"]
    paper.warnings += [f"clean-up: {e}; kept the filter text" for e in errors]
    return stats
