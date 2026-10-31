"""LLM clean-up pass (Claude Haiku 5.5): the optional second pass after the rule filter.

The rule filter (textfilter.py) already produced Block.speech. This pass asks Claude Haiku 5.5
to repair only what rules cannot: grammar broken by removed citations, maths read the way a
researcher says it, and display equations put into words. The system prompt states the job and
shows worked INPUT -> OUTPUT examples; it is byte-stable and cached across the per-section calls.
A rewrite guard keeps the filter text whenever the model changed too much.

    llm_available() -> (ok, reason)      cheap credential check, no network
    cleanup_paper(paper, client=None)    mutates Block.speech / speech_source, returns CleanupStats

One request per section; long sections are split into chunks of blocks. The first request runs
alone so that it writes the prompt cache, the rest run on a small thread pool and read it.
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from pydantic import BaseModel, ValidationError

from . import pricing
from .models import Block, CleanupStats, Paper, Section
from .textfilter import apply_filter, equation_label, equation_speech, filter_text, inline_math, latex_to_speech

log = logging.getLogger(__name__)

MODEL = "claude-haiku-5-5"
# Effort "low": the task is a high-volume, example-driven edit, and the guard catches overreach.
# Keep it constant: changing effort between requests invalidates the prompt cache.
EFFORT = "low"
MAX_OUTPUT_TOKENS = 16_000  # thinking counts toward it; a chunk's answer is ~2-4K tokens
MAX_CHUNK_CHARS = 12_000  # JSON characters of blocks per request
CONTEXT_CHARS = 600  # how much of the neighbouring paragraphs an equation block carries

# ---------------------------------------------------------------- rewrite guard
# Changed words = words deleted from the filter text + words inserted by the model (difflib on
# lower-cased word lists), so replacing one word costs 2. A block is accepted when
#   changed <= REWRITE_BUDGET * prose words + MATH_REWRITE_FACTOR * maths words
#              + CITATION_REPAIR_WORDS * citations + SLACK_WORDS
# Prose: 10% changed words is ~5% of words substituted, the plan's "rewrites above 5-10% outside
# citation spans". Maths: a re-verbalization may replace every word of a formula's reading and
# add a few ("alpha equals 0.5" -> "alpha to 0.5"; "gamma to the t r t" -> "gamma to the t times
# r t"), hence 2.5x. Citations: "earlier work" plus one agreement fix ("who rely" -> "which relies").
REWRITE_BUDGET = 0.10
MATH_REWRITE_FACTOR = 2.5
CITATION_REPAIR_WORDS = 4
SLACK_WORDS = 2
MAX_EQUATION_WORDS = 80
_PROSE_TYPES = ("paragraph", "list_item", "heading_note")
_MARKUP = re.compile(r"[\\$]|\[CITE\]|[{}]")
_FIGURE_MENTION = re.compile(r"\b(Figure|Table)s? (\d+)")

# ---------------------------------------------------------------- the system prompt

_JOB = """\
You prepare the narration of a scientific paper for a text-to-speech voice. The listener hears \
the paper read aloud faithfully and completely, as if a colleague were reading it to them, so \
the narration must follow the paper word for word wherever words can be spoken as they are.

Each request contains one section of the paper (or part of a long section) as a JSON list of \
blocks. Every block has:
- "id": echo it back unchanged.
- "type": paragraph, list_item, heading_note or equation.
- "source": the paper's text. Inline maths is written as $LaTeX$. Numeric citations were \
replaced by the marker [CITE]; textual citations look like "Smith et al. [CITE]".
- "filter_speech": the same text after a rule-based filter. It is usually right already.
Equation blocks also have "latex" (the display equation), "eq_number" (when it is numbered), \
"context" (the paragraph before it) and "context_after" (the paragraph after it).

Your job: return the final narration of every block. Start from filter_speech and fix only what \
the rules got wrong or could not do:
1. Citations. The markers are removed. Where the removal broke a sentence, repair the grammar \
with the smallest change, using "earlier work" (or "earlier studies") for a citation that was \
part of the sentence. "Smith et al." is read "Smith and colleagues". Never invent author names, \
years, titles or reference numbers.
2. Inline maths. Say it the way a researcher reads it aloud ("alpha equals 0.5", "x t", \
"the sum over i of", "pi theta of a given s"). Brackets inside maths are maths, not citations: \
[0, 1] is "the interval from 0 to 1".
3. Display equations. Begin with "Equation N:" when eq_number is given. If the equation is \
short, read it verbatim. If it is long, give ONE plain-English sentence that says what it \
computes, using the names that the context gives the symbols, for example "Equation 4: the \
total loss is the reconstruction loss plus beta times the KL term." Prefer that over a bare \
"Equation N."
4. Abbreviations, numbers and units. "e.g." is "for example", "i.e." is "that is", "w.r.t." is \
"with respect to", "SOTA" is "state of the art", "3.2 ms" is "3.2 milliseconds", "1e-4" is \
"ten to the minus four", "±0.3" is "plus or minus 0.3", "∼40" is "about 40".
5. Figures and tables. Mentions must stay exactly "Figure N" and "Table N" with the number in \
digits, for example "Figure 3 b" or "Table 2": the player uses them to show the figure.

Hard rules:
- Do not summarize, shorten, paraphrase, reorder, explain or add content. Every sentence of the \
source stays, in the same order and with the same words, apart from the repairs above.
- If filter_speech is already right, return it unchanged, character for character.
- Never output LaTeX, dollar signs, backslashes, braces, [CITE] or maths symbols such as ≤ or ∑.
- Return every input id exactly once, in input order, and no other ids.
- Plain text only: no markdown, no notes, no quotes around the speech.

Worked examples follow. Each shows one input block and the output for it."""


@dataclass(frozen=True)
class _Example:
    title: str
    source: str
    speech: str  # the expected output
    eq_number: str | None = None  # equation examples: source is the LaTeX
    context: str = ""
    context_after: str = ""


_EXAMPLES = [
    _Example("A citation at the end of a sentence",
             "Domain randomization is a common way to make sim-to-real transfer robust [CITE].",
             "Domain randomization is a common way to make sim-to-real transfer robust."),
    _Example("A citation that was part of the sentence: fix the grammar it leaves behind",
             "Unlike [CITE], who rely on motion capture, we estimate the state on board.",
             "Unlike earlier work, which relies on motion capture, we estimate the state on board."),
    _Example("A textual citation",
             "Rudin et al. [CITE] showed that massively parallel simulation shortens training to minutes.",
             "Rudin and colleagues showed that massively parallel simulation shortens training to minutes."),
    _Example("Citations as sentence subjects",
             "[CITE] proposed a diffusion policy for visuomotor control. In contrast, [CITE] train a "
             "transformer on teleoperation data.",
             "Earlier work proposed a diffusion policy for visuomotor control. In contrast, earlier "
             "studies train a transformer on teleoperation data."),
    _Example("Citations in parentheses and after a preposition",
             "Prior studies (e.g., [CITE]) rely on a fixed curriculum, as in [CITE].",
             "Prior studies rely on a fixed curriculum, as in earlier work."),
    _Example("Figure, table, equation and section references",
             "As shown in Fig. 3(b) and Tab. II, the gait remains stable (see Sec. 4.2 and Eq. (5)).",
             "As shown in Figure 3 b and Table 2, the gait remains stable (see Section 4.2 and Equation 5)."),
    _Example("Latin abbreviations and SOTA",
             "Our method outperforms SOTA baselines w.r.t. success rate, e.g., on peg insertion, i.e., "
             "the hardest task.",
             "Our method outperforms state of the art baselines with respect to success rate, for "
             "example, on peg insertion, that is, the hardest task."),
    _Example("Numbers and units",
             "Inference takes 3.2 ms per step, the learning rate is 1e-4, and the position error is "
             "±0.3 mm over ∼40 trials.",
             "Inference takes 3.2 milliseconds per step, the learning rate is ten to the minus four, "
             "and the position error is plus or minus 0.3 millimeters over about 40 trials."),
    _Example("Greek letters: say the maths naturally",
             r"We set $\alpha = 0.5$ and $\gamma = 0.99$ in all experiments.",
             "We set alpha to 0.5 and gamma to 0.99 in all experiments."),
    _Example("Brackets inside maths are intervals, not citations",
             r"Actions are normalized to $[-1, 1]$ and rewards are clipped to $[0, 1]$ [CITE].",
             "Actions are normalized to the interval from minus 1 to 1 and rewards are clipped to "
             "the interval from 0 to 1."),
    _Example("Scientific notation",
             r"We use Adam with a learning rate of $3 \times 10^{-4}$ and a weight decay of $10^{-2}$.",
             "We use Adam with a learning rate of 3 times ten to the minus four and a weight decay of "
             "ten to the minus two."),
    _Example("Longer inline maths: keep the reading, make it flow",
             r"The policy $\pi_\theta$ maximizes $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}"
             r"[\sum_{t=0}^{T} \gamma^t r_t]$.",
             "The policy pi theta maximizes J of theta, which equals the expectation over tau drawn "
             "from pi theta of the sum from t equals 0 to T of gamma to the t times r t."),
    _Example("A paragraph that needs no change is returned as it is",
             "The robot walks on stairs, slopes and gravel without any terrain-specific tuning.",
             "The robot walks on stairs, slopes and gravel without any terrain-specific tuning."),
    _Example("A URL and a footnote mark",
             "Code and videos are available at https://lp-acrl.github.io.¹",
             "Code and videos are available online."),
    _Example("A numbered display equation with a sum: one plain-English sentence",
             r"\mathcal{L} = \sum_{t=1}^{T} \| \hat{x}_t - x_t \|^2 + \beta \, D_{\mathrm{KL}}"
             r"\big(q_\phi(z \mid x) \,\|\, p(z)\big)",
             "Equation 4: the loss L is the squared error between x hat t and x t, summed over the "
             "time steps t from 1 to T, plus beta times the KL divergence between q phi of z given x "
             "and p of z.",
             eq_number="4",
             context="We train the world model by minimizing a reconstruction loss and a KL regularizer:",
             context_after=r"where $\beta$ weights the KL term."),
    _Example("A short display equation: read it verbatim",
             r"a_t = \pi_\theta(o_t)",
             "Equation 2: a t equals pi theta of o t.",
             eq_number="2",
             context="At every step the policy maps the observation to an action:"),
]


def _example_block(example: _Example, index: int) -> dict:
    block_id = f"ex{index:02d}"
    if example.eq_number is None:
        return {"id": block_id, "type": "paragraph", "source": example.source,
                "filter_speech": filter_text(example.source)}
    block = Block(id=block_id, type="equation", latex=example.source, eq_number=example.eq_number)
    return {"id": block_id, "type": "equation", "latex": example.source, "eq_number": example.eq_number,
            "context": example.context, "context_after": example.context_after,
            "filter_speech": equation_speech(block)}


def _build_system_prompt() -> str:
    parts = [_JOB]
    for index, example in enumerate(_EXAMPLES, 1):
        block = _example_block(example, index)
        answer = {"id": block["id"], "speech": example.speech}
        parts.append(f"Example {index}. {example.title}.\n"
                     f"Input: {json.dumps(block, ensure_ascii=False)}\n"
                     f"Output: {json.dumps(answer, ensure_ascii=False)}")
    return "\n\n".join(parts) + "\n"


# Built once per process from constants only, so it is byte-identical on every request.
SYSTEM_PROMPT = _build_system_prompt()

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "speech": {"type": "string"}},
                "required": ["id", "speech"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


class _BlockSpeech(BaseModel):
    id: str
    speech: str


class _Answer(BaseModel):
    blocks: list[_BlockSpeech]


# ---------------------------------------------------------------- requests


@dataclass
class CleanupRequest:
    section: Section
    blocks: list[Block]
    user_message: str


def _neighbour_text(blocks: list[Block], start: int, step: int) -> str:
    i = start + step
    while 0 <= i < len(blocks):
        if blocks[i].type in _PROSE_TYPES and blocks[i].text:
            return blocks[i].text[-CONTEXT_CHARS:] if step < 0 else blocks[i].text[:CONTEXT_CHARS]
        i += step
    return ""


def _block_payload(section: Section, index: int) -> dict:
    block = section.blocks[index]
    payload = {"id": block.id, "type": block.type}
    if block.type == "equation":
        payload |= {"latex": block.latex or "", "eq_number": block.eq_number or "",
                    "context": _neighbour_text(section.blocks, index, -1),
                    "context_after": _neighbour_text(section.blocks, index, +1)}
    else:
        payload["source"] = block.text
    payload["filter_speech"] = block.speech or ""
    return payload


def plan_requests(paper: Paper) -> list[CleanupRequest]:
    """One request per section, split into chunks of at most MAX_CHUNK_CHARS of block JSON."""
    requests = []
    for section in paper.sections:
        heading = section.speech_title or section.title
        chunk: list[tuple[Block, dict]] = []
        size = 0
        for index, block in enumerate(section.blocks):
            if not block.speak or block.type == "figure" or not (block.speech or block.type == "equation"):
                continue
            payload = _block_payload(section, index)
            length = len(json.dumps(payload, ensure_ascii=False))
            if chunk and size + length > MAX_CHUNK_CHARS:
                requests.append(_request(paper, section, heading, chunk))
                chunk, size = [], 0
            chunk.append((block, payload))
            size += length
        if chunk:
            requests.append(_request(paper, section, heading, chunk))
    return requests


def _request(paper: Paper, section: Section, heading: str, chunk: list[tuple[Block, dict]]) -> CleanupRequest:
    blocks_json = json.dumps([payload for _, payload in chunk], ensure_ascii=False, indent=1)
    message = (f"Paper: {paper.title}\nSection: {heading}\n\n"
               f"Return the narration of these {len(chunk)} blocks:\n{blocks_json}")
    return CleanupRequest(section, [block for block, _ in chunk], message)


# ---------------------------------------------------------------- guard


def _words(text: str) -> list[str]:
    return re.findall(r"[\w.'’-]+", text.lower().replace(",", " "))


def _figure_mentions(text: str) -> set[tuple[str, str]]:
    return set(_FIGURE_MENTION.findall(text))


def guard_accepts(block: Block, speech: str) -> bool:
    """Whether the model's speech may replace the filter speech of this block."""
    speech = " ".join(speech.split())
    filtered = block.speech if block.speech is not None else filter_text(block.text)
    if not speech or _MARKUP.search(speech) or not _figure_mentions(filtered) <= _figure_mentions(speech):
        return False
    if block.type == "equation":
        return len(speech.split()) <= MAX_EQUATION_WORDS
    before, after = _words(filtered), _words(speech)
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    kept = sum(match.size for match in matcher.get_matching_blocks())
    changed = len(before) + len(after) - 2 * kept
    math_words = sum(len(latex_to_speech(span).split()) for span in inline_math(block.text))
    citations = block.text.count("[CITE]")
    allowance = (REWRITE_BUDGET * max(len(before) - math_words, 0) + MATH_REWRITE_FACTOR * math_words
                 + CITATION_REPAIR_WORDS * citations + SLACK_WORDS)
    return changed <= allowance


# ---------------------------------------------------------------- the run


def llm_available() -> tuple[bool, str | None]:
    """Cheap credential check (no network): ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
    `ant` profile under ~/.config/anthropic (ANTHROPIC_CONFIG_DIR), as the SDK resolves them."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True, None
    config_dir = Path(os.environ.get("ANTHROPIC_CONFIG_DIR") or Path.home() / ".config" / "anthropic")
    if os.environ.get("ANTHROPIC_PROFILE") or any((config_dir / "configs").glob("*.json")):
        return True, None
    return False, "no ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN / ant profile)"


class _SectionFailure(Exception):
    pass


_FATAL = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
          anthropic.CredentialsError)


@dataclass
class _Run:
    paper: Paper
    client: object
    model: str
    total: int
    progress_cb: Callable[[float, str], None] | None
    stats: CleanupStats = field(default_factory=CleanupStats)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stopped: threading.Event = field(default_factory=threading.Event)
    done: int = 0

    def execute(self, request: CleanupRequest) -> None:
        if self.stopped.is_set():
            return
        label = request.section.speech_title or request.section.id
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": request.user_message}],
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
            )
            self._account(response.usage)
            self._apply(request, _parse(response), label)
        except _FATAL as exc:
            self.stopped.set()
            with self.lock:
                self.stats.skipped_reason = f"{type(exc).__name__}: {exc}"
            log.error("clean-up stopped: %s", exc)
        except (anthropic.APIError, _SectionFailure) as exc:
            self._warn(f"{label}: {type(exc).__name__}: {exc}; kept the filter text")
        finally:
            with self.lock:
                self.done += 1
                done = self.done
            if self.progress_cb:
                self.progress_cb(done / self.total, f"Clean-up {done}/{self.total}: {label}")

    def _account(self, usage) -> None:
        input_tokens = usage.input_tokens or 0
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        cost = pricing.cleanup_cost(
            input_tokens, usage.output_tokens or 0, cache_read, cache_write,
            long_prompt=input_tokens + cache_read + cache_write > pricing.HAIKU_LONG_PROMPT_TOKENS,
        )
        with self.lock:
            self.stats.input_tokens += input_tokens
            self.stats.output_tokens += usage.output_tokens or 0
            self.stats.cache_read_input_tokens += cache_read
            self.stats.cache_creation_input_tokens += cache_write
            self.stats.cost_usd += cost

    def _apply(self, request: CleanupRequest, answers: list[_BlockSpeech], label: str) -> None:
        speech_by_id: dict[str, str] = {}
        for answer in answers:
            speech_by_id.setdefault(answer.id, " ".join(answer.speech.split()))
        accepted = rejected = missing = 0
        for block in request.blocks:
            speech = speech_by_id.get(block.id)
            if speech is None:
                missing += 1
            elif guard_accepts(block, speech):
                label_text = equation_label(block.eq_number) if block.type == "equation" else None
                if label_text and label_text not in speech:
                    speech = f"{label_text}: {speech}"
                block.speech, block.speech_source = speech, "llm"
                accepted += 1
            else:
                block.speech_source = "llm_rejected"
                rejected += 1
        with self.lock:
            self.stats.blocks_llm += accepted
            self.stats.blocks_rejected += rejected
        if missing:
            self._warn(f"{label}: {missing} block(s) missing from the answer; kept the filter text")

    def _warn(self, message: str) -> None:
        log.warning("clean-up: %s", message)
        with self.lock:
            self.paper.warnings.append(f"clean-up: {message}")


def _parse(response) -> list[_BlockSpeech]:
    if response.stop_reason == "refusal":
        raise _SectionFailure("the model declined this section (stop_reason refusal)")
    if response.stop_reason == "max_tokens":
        raise _SectionFailure("the answer was cut off (stop_reason max_tokens)")
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise _SectionFailure("the answer has no text block")
    try:
        return _Answer.model_validate_json(text).blocks
    except ValidationError as exc:
        raise _SectionFailure(f"the answer is not the expected JSON ({exc.error_count()} errors)") from exc


# ---------------------------------------------------------------- offline runner (e.g. Claude Code subagents)


def export_requests(paper: Paper, parts: int = 1) -> list[list[dict]]:
    """The exact requests the API pass would send, in `parts` groups balanced by size, so another
    runner (e.g. Claude Code subagents on Haiku) can answer them with SYSTEM_PROMPT as instructions.
    Each request: {request_id, section, block_ids, user_message}."""
    apply_filter(paper)
    requests = [{"request_id": i, "section": r.section.speech_title or r.section.id,
                 "block_ids": [b.id for b in r.blocks], "user_message": r.user_message}
                for i, r in enumerate(plan_requests(paper))]
    groups: list[list[dict]] = [[] for _ in range(max(1, min(parts, len(requests))))]
    sizes = [0] * len(groups)
    for request in sorted(requests, key=lambda r: -len(r["user_message"])):
        smallest = sizes.index(min(sizes))
        groups[smallest].append(request)
        sizes[smallest] += len(request["user_message"])
    return [sorted(group, key=lambda r: r["request_id"]) for group in groups]


def import_answers(paper: Paper, answers: list[dict], *, model: str) -> CleanupStats:
    """Apply externally produced answers ([{"id", "speech"}], any order) through the same guard
    as the API pass. Re-imports are idempotent: speech is reset to the filter text first."""
    apply_filter(paper)
    paper.warnings = [w for w in paper.warnings if not w.startswith("clean-up:")]
    try:
        parsed = [_BlockSpeech.model_validate(a) for a in answers]
    except ValidationError as exc:
        raise ValueError(f"answers are not [{{id, speech}}] ({exc.error_count()} errors)") from exc
    requests = plan_requests(paper)
    run = _Run(paper, None, model, max(len(requests), 1), None, CleanupStats(model=model))
    for request in requests:
        ids = {b.id for b in request.blocks}
        run._apply(request, [a for a in parsed if a.id in ids], request.section.speech_title or request.section.id)
    paper.cleanup = run.stats
    return run.stats


def cleanup_paper(paper: Paper, *, client=None, model: str = MODEL, max_workers: int = 4,
                  progress_cb: Callable[[float, str], None] | None = None) -> CleanupStats:
    """Run the Haiku 5.5 clean-up over a filtered paper. Mutates blocks and sets paper.cleanup.

    client: an anthropic.Anthropic (or a test double); None builds one from the environment, or
    skips the pass with skipped_reason when no credentials are available.
    progress_cb(fraction, message) is called after every request."""
    if client is None:
        ok, reason = llm_available()
        if not ok:
            paper.cleanup = CleanupStats(model=model, skipped_reason=reason)
            return paper.cleanup
        client = anthropic.Anthropic()
    if any(b.speak and b.speech is None for b in paper.iter_blocks()):
        apply_filter(paper)

    requests = plan_requests(paper)
    run = _Run(paper, client, model, max(len(requests), 1), progress_cb, CleanupStats(model=model))
    if requests:
        run.execute(requests[0])  # writes the prompt cache; parallel requests could not read it yet
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(run.execute, requests[1:]))
    run.stats.cost_usd = round(run.stats.cost_usd, 6)
    paper.cleanup = run.stats
    return run.stats
