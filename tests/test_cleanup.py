"""Clean-up pass tests with a fake client that answers with real SDK response objects."""

from __future__ import annotations

import json
import re
import threading

import anthropic
import httpx2
import pytest
from anthropic.types import Message, TextBlock, ThinkingBlock, Usage

from paper_reader import cleanup, textfilter
from paper_reader.cleanup import SYSTEM_PROMPT, cleanup_paper, guard_accepts, llm_available, plan_requests
from paper_reader.definitions import paper_vocabulary
from paper_reader.models import Block, Paper, Section
from paper_reader.textfilter import apply_filter, equation_speech, filter_text

_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def _message(text: str, *, stop_reason: str = "end_turn", input_tokens: int = 400, output_tokens: int = 120,
             cache_read: int = 0, cache_write: int = 0) -> Message:
    return Message(
        id="msg_test", type="message", role="assistant", model="claude-haiku-5-5",
        content=[ThinkingBlock(type="thinking", thinking="", signature="sig"), TextBlock(type="text", text=text)],
        stop_reason=stop_reason, stop_sequence=None,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens,
                    cache_read_input_tokens=cache_read, cache_creation_input_tokens=cache_write),
    )


def _blocks_in(request_kwargs: dict) -> list[dict]:
    content = request_kwargs["messages"][0]["content"]
    return json.loads(content[content.index("\n["):])


class FakeClient:
    """client.messages.create(**kwargs) -> Message; `answer(blocks)` decides the speech per block."""

    def __init__(self, answer=None, *, error: Exception | None = None, raw_text: str | None = None,
                 stop_reason: str = "end_turn"):
        self.answer = answer or (lambda blocks: [{"id": b["id"], "speech": b["filter_speech"]} for b in blocks])
        self.error, self.raw_text, self.stop_reason = error, raw_text, stop_reason
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        self.messages = self

    def create(self, **kwargs) -> Message:
        with self.lock:
            first = not self.calls
            self.calls.append(kwargs)
        error = self.error(kwargs) if callable(self.error) else self.error
        if error:
            raise error
        text = self.raw_text if self.raw_text is not None else json.dumps({"blocks": self.answer(_blocks_in(kwargs))})
        return _message(text, stop_reason=self.stop_reason,
                        cache_write=0 if not first else 3000, cache_read=3000 if not first else 0)


def _paper() -> Paper:
    intro = [
        Block(id="b0001", type="paragraph", text="Unlike [CITE], who rely on motion capture, we estimate the state on board."),
        Block(id="b0002", type="paragraph", text=r"We set $\alpha = 0.5$ and $\gamma = 0.99$ in all experiments, see Fig. 2."),
        Block(id="b0003", type="figure", figure_id="S1.F2", speak=False),
    ]
    method = [
        Block(id="b0004", type="paragraph", text="We train the world model by minimizing a reconstruction loss and a KL term:"),
        Block(id="b0005", type="equation", eq_number="4",
              latex=r"\mathcal{L} = \sum_{t=1}^{T} \| \hat{x}_t - x_t \|^2 + \beta \, D_{\mathrm{KL}}(q \,\|\, p)"),
        Block(id="b0006", type="list_item", text="(1) The policy, trained with PPO for 10M steps in simulation, "
                                                 "transfers zero-shot to the real robot (Table 1)."),
    ]
    paper = Paper(paper_id="t.1v1", arxiv_id="t.1", title="Test paper", source_url="x", sections=[
        Section(id="S1", number="1", title="Introduction", blocks=intro),
        Section(id="S2", number="2", title="Method", blocks=method),
    ])
    apply_filter(paper)
    return paper


GOOD_ANSWERS = {
    "b0001": "Unlike earlier work, which relies on motion capture, we estimate the state on board.",
    "b0002": "We set alpha to 0.5 and gamma to 0.99 in all experiments, see Figure 2.",
    "b0004": "We train the world model by minimizing a reconstruction loss and a KL term:",
    "b0005": "Equation 4: L equals the sum from t equals 1 to T of the squared norm of x hat t minus x t, "
             "plus beta times the KL divergence between q and p.",
    "b0006": "The policy transfers to the real robot.",  # a summary: must be rejected
}


def _good(blocks):
    return [{"id": b["id"], "speech": GOOD_ANSWERS[b["id"]]} for b in blocks]


def test_cleanup_applies_guarded_answers_and_accounts_usage():
    paper = _paper()
    progress = []
    client = FakeClient(_good)
    stats = cleanup_paper(paper, client=client, progress_cb=lambda f, msg: progress.append(f))
    blocks = {b.id: b for b in paper.iter_blocks()}

    for block_id in ("b0001", "b0002", "b0004", "b0005"):
        assert blocks[block_id].speech == GOOD_ANSWERS[block_id]
        assert blocks[block_id].speech_source == "llm"
    assert blocks["b0006"].speech_source == "llm_rejected"
    assert blocks["b0006"].speech == filter_text(blocks["b0006"].text)
    assert blocks["b0003"].speech is None  # figures are never sent
    assert (stats.blocks_llm, stats.blocks_rejected) == (4, 1)

    assert len(client.calls) == 3  # one request per section, then the rejected b0006 asked once more
    repair = _blocks_in(client.calls[2])
    assert [b["id"] for b in repair] == ["b0006"] and "Figure/Table mention was lost" in repair[0]["problem"]
    assert sum("b0006 kept the rule-based text" in w for w in paper.warnings) == 1  # warned once, at the end
    assert (stats.input_tokens, stats.output_tokens) == (1200, 360)
    assert (stats.cache_creation_input_tokens, stats.cache_read_input_tokens) == (3000, 6000)
    expected = (1200 * 0.10 + 3000 * 1.25 * 0.10 + 6000 * 0.1 * 0.10 + 360 * 0.50) / 1e6
    assert stats.cost_usd == pytest.approx(expected, abs=1e-6)
    assert stats.model == "claude-haiku-5-5" and stats.skipped_reason is None
    assert paper.cleanup is stats
    assert progress == [0.5, 1.0, 1.0]


def test_request_shape():
    paper = _paper()
    client = FakeClient()
    cleanup_paper(paper, client=client)
    for call in client.calls:
        assert call["model"] == "claude-haiku-5-5"
        assert call["system"] == [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        assert call["output_config"]["effort"] == "low"
        assert call["output_config"]["format"]["type"] == "json_schema"
        assert call["messages"][-1]["role"] == "user"  # no assistant prefill (a 400 on Haiku 5.5)
        assert not {"temperature", "top_p", "top_k", "thinking"} & call.keys()
    equation = next(b for c in client.calls for b in _blocks_in(c) if b["type"] == "equation")
    assert equation["eq_number"] == "4" and equation["latex"].startswith(r"\mathcal{L}")
    assert equation["context"].startswith("We train the world model")
    assert equation["context_after"].startswith("(1) The policy")
    assert equation["filter_speech"] == "Equation 4."


def test_long_sections_are_chunked(monkeypatch):
    monkeypatch.setattr(cleanup, "MAX_CHUNK_CHARS", 400)
    paper = _paper()
    requests = plan_requests(paper)
    assert [[b.id for b in r.blocks] for r in requests] == [["b0001"], ["b0002"], ["b0004"], ["b0005"], ["b0006"]]
    assert all(r.user_message.startswith("Paper: Test paper\nSection: Section ") for r in requests)

    client = FakeClient()  # echoes filter_speech, which for the long equation b0005 is only "Equation 4."
    stats = cleanup_paper(paper, client=client)
    assert len(client.calls) == 6  # five chunks, then b0005 asked again
    assert [b["id"] for b in _blocks_in(client.calls[0])] == ["b0001"]  # the cache-warming request ran first
    assert sorted(b["id"] for c in client.calls[:5] for b in _blocks_in(c)) == ["b0001", "b0002", "b0004", "b0005", "b0006"]
    repair = _blocks_in(client.calls[5])
    assert [b["id"] for b in repair] == ["b0005"] and "not read in full" in repair[0]["problem"]
    # a bare "Equation 4." was rejected; the repair request carries the full rule reading, which passes
    assert repair[0]["filter_speech"].startswith("Equation 4: L equals the sum from t equals 1 to T")
    assert (stats.blocks_llm, stats.blocks_rejected) == (5, 0)


def test_missing_and_extra_ids():
    paper = _paper()

    def answer(blocks):
        replies = [{"id": b["id"], "speech": GOOD_ANSWERS[b["id"]]} for b in blocks if b["id"] != "b0002"]
        return replies + [{"id": "b9999", "speech": "Invented."}]

    stats = cleanup_paper(paper, client=FakeClient(answer))
    b0002 = next(b for b in paper.iter_blocks() if b.id == "b0002")
    assert b0002.speech_source == "filter"
    assert b0002.speech == filter_text(b0002.text)
    assert (stats.blocks_llm, stats.blocks_rejected) == (3, 1)  # b0006 is the rejected summary
    assert any("1 block(s) missing" in w for w in paper.warnings)
    assert all(b.id != "b9999" for b in paper.iter_blocks())


def test_authentication_error_stops_the_run():
    paper = _paper()
    before = [b.speech for b in paper.iter_blocks()]
    error = anthropic.AuthenticationError("invalid x-api-key", response=httpx2.Response(401, request=_REQUEST), body=None)
    client = FakeClient(error=error)
    stats = cleanup_paper(paper, client=client)
    assert len(client.calls) == 1  # the first request fails alone; the rest never start
    assert stats.skipped_reason.startswith("AuthenticationError")
    assert [b.speech for b in paper.iter_blocks()] == before
    assert all(b.speech_source in ("filter", None) for b in paper.iter_blocks())
    assert stats.cost_usd == 0


def test_other_errors_keep_the_filter_text_for_that_section():
    paper = _paper()

    def error(kwargs):
        if any(b["id"] == "b0004" for b in _blocks_in(kwargs)):
            return anthropic.InternalServerError("overloaded", response=httpx2.Response(500, request=_REQUEST), body=None)
        return None

    stats = cleanup_paper(paper, client=FakeClient(_good, error=error))
    sources = {b.id: b.speech_source for b in paper.iter_blocks()}
    assert sources["b0001"] == sources["b0002"] == "llm"
    assert sources["b0004"] == sources["b0005"] == "filter"
    assert stats.skipped_reason is None
    assert any("InternalServerError" in w for w in paper.warnings)


@pytest.mark.parametrize(("client", "warning"), [
    (FakeClient(raw_text="not json"), "not the expected JSON"),
    (FakeClient(raw_text='{"blocks": [{"id": "b0001"}]}'), "not the expected JSON"),
    (FakeClient(stop_reason="refusal"), "declined"),
    (FakeClient(stop_reason="max_tokens"), "cut off"),
])
def test_unusable_answers_keep_the_filter_text(client, warning):
    paper = _paper()
    stats = cleanup_paper(paper, client=client)
    assert stats.blocks_llm == 0
    assert all(b.speech_source in ("filter", None) for b in paper.iter_blocks())
    assert sum(warning in w for w in paper.warnings) == 2
    assert stats.input_tokens == 800  # billed even when unusable


def test_equation_label_is_kept():
    paper = _paper()
    reading = "L equals the sum from t equals 1 to T of the squared norm of x hat t minus x t, plus beta times the KL divergence between q and p."
    answer = {**GOOD_ANSWERS, "b0005": reading}
    cleanup_paper(paper, client=FakeClient(lambda blocks: [{"id": b["id"], "speech": answer[b["id"]]} for b in blocks]))
    b0005 = next(b for b in paper.iter_blocks() if b.id == "b0005")
    assert b0005.speech == f"Equation 4: {reading}"


def test_skips_cleanly_without_credentials(monkeypatch, tmp_path):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    assert llm_available() == (False, "no ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN / ant profile)")
    paper = _paper()
    stats = cleanup_paper(paper)
    assert stats.skipped_reason.startswith("no ANTHROPIC_API_KEY")
    assert all(b.speech_source in ("filter", None) for b in paper.iter_blocks())

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.json").write_text("{}")
    assert llm_available() == (True, None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm_available() == (True, None)


def _paragraph(text: str) -> Block:
    block = Block(id="b1", type="paragraph", text=text)
    block.speech = filter_text(text)
    return block


@pytest.mark.parametrize(("source", "speech", "accepted"), [
    # legitimate repairs
    ("Unlike [CITE], who rely on motion capture, we estimate the state on board.",
     "Unlike earlier work, which relies on motion capture, we estimate the state on board.", True),
    ("In contrast, [CITE] train one model for all tasks and all robots.",
     "In contrast, earlier work trains one model for all tasks and all robots.", True),
    (r"The policy $\pi_\theta$ maximizes $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_{t=0}^{T} \gamma^t r_t]$.",
     "The policy pi theta maximizes J of theta, the expected sum over t from 0 to T of gamma to the t times r t, "
     "for tau drawn from pi theta.", True),
    (r"We set $\alpha = 0.5$ and $\gamma = 0.99$.", "We set alpha to 0.5 and gamma to 0.99.", True),
    ("The robot walks on stairs and slopes without any tuning.",
     "The robot walks on stairs and slopes without any tuning.", True),
    # rewrites
    ("The policy, trained with PPO for ten million steps in simulation, transfers zero-shot to the real robot.",
     "The policy transfers zero-shot to the real robot.", False),
    ("We evaluate our approach on three challenging manipulation tasks in the real world.",
     "Three challenging real-world manipulation tasks are used to evaluate the approach.", False),
    ("The robot walks on stairs and slopes without any tuning.",
     "The robot walks on stairs and slopes without any tuning, which is remarkable.", False),
    # broken output
    ("Results are in Fig. 3 and Tab. 1.", "Results are in the third figure and Table 1.", False),
    (r"We set $\alpha = 0.5$.", r"We set $\alpha$ to 0.5.", False),
    ("The controller is robust [CITE].", "The controller is robust [CITE].", False),
    ("The controller is robust [CITE].", "", False),
])
def test_guard(source, speech, accepted):
    assert guard_accepts(_paragraph(source), speech) is accepted


def test_guard_for_equations():
    block = Block(id="b1", type="equation", latex=r"\mathcal{L} = a + b", eq_number="4")
    block.speech = equation_speech(block)
    description = "Equation 4: the total loss is the reconstruction loss plus beta times the KL term."
    assert guard_accepts(block, "Equation 4: L equals a plus b.")
    assert not guard_accepts(block, description)  # read by symbols: every symbol must be said
    assert guard_accepts(block, description, mode="meaning")
    assert not guard_accepts(block, "Equation 4: " + "and so on " * 40)
    assert not guard_accepts(block, r"Equation 4: \mathcal{L} = a + b")


SAPG_PPO = (r"\begin{split}L_{\mathrm{on}}(\pi_{\theta})&=\mathop{\mathbb{E}}_{\pi_{\mathrm{old}}}\left[\min(r_{t}"
            r"(\pi_{\theta}),\right.\\ &\left.\mathop{\mathrm{clip}}\left(r_{t}(\pi_{\theta}),1-\epsilon,1+\epsilon\right))"
            r"A_{t}^{\pi_{old}}\right]\end{split}")


def test_symbols_mode_rejects_a_description_of_a_long_equation():
    """SAPG Eq. 2 (cleaned by symbols): a summary of what it computes is not a reading of it."""
    block = Block(id="b2", type="equation", latex=SAPG_PPO, eq_number="2")
    block.speech = equation_speech(block)
    assert block.speech == "Equation 2."  # too long for the rules alone
    summary = ("Equation 2: the PPO objective is the expectation under the old policy of the minimum of the "
               "policy ratio and its clipped value, multiplied by the old policy advantage.")
    reading = ("Equation 2: L on of pi theta equals the expectation over pi old of the minimum of r t of pi theta "
               "and clip of r t of pi theta between 1 minus epsilon and 1 plus epsilon, times A t to the pi old.")
    assert "not read in full" in cleanup.guard_reason(block, summary)
    assert guard_accepts(block, reading)
    assert cleanup.symbol_coverage(textfilter.equation_reading(block), reading) >= 0.9


def test_rejected_equation_falls_back_to_the_full_reading():
    paper = Paper(paper_id="p", arxiv_id="p", title="", source_url="x", sections=[Section(id="S1", title="", blocks=[
        Block(id="b2", type="equation", latex=SAPG_PPO, eq_number="2")])])
    textfilter.apply_filter(paper)
    request = cleanup.plan_requests(paper)[0]
    run = cleanup._new_run(paper, None, "test", [request], None, mode="symbols", inline="symbols")
    run._apply(request, [cleanup._BlockSpeech(id="b2", speech="Equation 2: the PPO objective.")], "S1")
    block = paper.sections[0].blocks[0]
    assert block.speech_source == "llm_rejected"
    assert block.speech.startswith("Equation 2: L on of pi theta equals") and "pi old" in block.speech


@pytest.mark.parametrize(("latex", "spoken"), [
    (r"V_{off,\pi_{i}}^{target}(s)", "V target off pi i of s"),
    (r"A_{t}^{\pi_{old}}", "A t to the pi old"),
    (r"\lambda_{ent}", "lambda ent"),
    (r"L_{on}^{critic}", "L critic on"),
    (r"x_{ij} + w_{kl}", "x i j plus w k l"),  # indices stay letters
    (r"e^{iwt}", "e to the i w t"),  # a product in an exponent, not a label
    (r"r^{c}_{t}", "r t to the c"),
    (r"\pi(\cdot|s)", "pi of dot given s"),  # a placeholder, not a product
    (r"a \cdot b", "a times b"),
])
def test_multi_letter_script_labels_are_words(latex, spoken):
    assert textfilter.latex_to_speech(latex) == spoken


def _vocabulary(*texts: str) -> set[str]:
    """What paper_vocabulary would give for a paper made of these texts."""
    blocks = [Block(id=f"v{i}", type="paragraph", text=t) for i, t in enumerate(texts)]
    return paper_vocabulary(Paper(paper_id="v", arxiv_id="v", title="", source_url="x",
                                  sections=[Section(id="S1", title="", blocks=blocks)]))


READINGS = [(mode, inline) for mode in cleanup.EQUATION_MODES for inline in cleanup.EQUATION_MODES]


@pytest.mark.parametrize(("mode", "inline"), READINGS)
def test_few_shot_examples_pass_the_guard_and_match_the_filter(mode, inline):
    for example in cleanup.examples_for(mode, inline):
        if example.eq_number or example.is_equation:
            block = Block(id="x", type="equation", latex=example.source, eq_number=example.eq_number)
            block.speech = equation_speech(block)
        else:
            block = _paragraph(example.source)
        # meaning mode: every added word must come from the example's own context and definitions
        vocabulary = _vocabulary(example.source, example.context, example.context_after, *example.definitions)
        assert guard_accepts(block, example.speech, mode=mode, inline=inline, vocabulary=vocabulary), example.title
        # the prompt shows the real filter output
        assert json.dumps(block.speech, ensure_ascii=False) in cleanup.system_prompt(mode, inline)


def test_meaning_prompts_cover_each_part_separately():
    both = cleanup.system_prompt("meaning")  # inline follows the equations setting by default
    assert both == cleanup.system_prompt("meaning", "meaning") and both.startswith(cleanup._JOB)
    assert cleanup._MEANING_EQUATIONS in both and cleanup._MEANING_INLINE in both
    assert '"definitions"' in both and "Never guess a meaning" in both
    assert "a t equals pi theta of o t" not in both  # symbol-by-symbol equation examples left out

    equations_only = cleanup.system_prompt("meaning", "symbols")
    assert cleanup._MEANING_EQUATIONS in equations_only and cleanup._SYMBOLS_INLINE in equations_only
    assert "slip penalty" not in equations_only  # the sentence examples belong to inline meaning

    text_only = cleanup.system_prompt("symbols", "meaning")
    assert cleanup._MEANING_INLINE in text_only and cleanup._SYMBOLS_EQUATIONS in text_only
    assert "The slip penalty discourages foot sliding" in text_only  # the symbol after its name is dropped
    assert "a t equals pi theta of o t" in text_only  # equations stay symbol by symbol

    assert cleanup.system_prompt("symbols", "symbols") == cleanup.SYSTEM_PROMPT
    with pytest.raises(ValueError):
        cleanup.system_prompt("letters")


def test_system_prompt_is_stable_and_covers_the_rules():
    assert cleanup._build_system_prompt("symbols") == SYSTEM_PROMPT
    assert 12 <= SYSTEM_PROMPT.count("\nInput: ") <= 17
    assert len(SYSTEM_PROMPT) / 4 > 512  # above Haiku 5.5's minimum cacheable prefix, even at 4 chars/token
    sources = " ".join(e.source for e in cleanup._EXAMPLES)
    for needle in ("[CITE].", "et al. [CITE]", "[CITE] proposed", "Fig. 3(b)", "Tab. II", "Eq. (5)", "Sec. 4.2",
                   "e.g.,", "i.e.,", "w.r.t.", "SOTA", "3.2 ms", "1e-4", "±0.3", "∼40", r"\alpha", "$[0, 1]$",
                   "10^{-4}", r"\sum", "https://"):
        assert needle in sources, needle
    assert any(e.source == e.speech for e in cleanup._EXAMPLES)  # a block that needs no change
    assert not re.search(r"\d{4}-\d{2}-\d{2}|uuid", SYSTEM_PROMPT)


def test_offline_export_and_import_use_the_same_requests_and_guard():
    paper = _paper()
    groups = cleanup.export_requests(paper, parts=2)
    exported = sorted((r for group in groups for r in group), key=lambda r: r["request_id"])
    assert len(groups) == 2
    assert [r["user_message"] for r in exported] == [r.user_message for r in plan_requests(paper)]

    answers = [{"id": i, "speech": GOOD_ANSWERS[i]} for r in exported for i in r["block_ids"]]
    stats = cleanup.import_answers(paper, list(reversed(answers)), model="haiku-subagent")
    blocks = {b.id: b for b in paper.iter_blocks()}
    assert (stats.model, stats.blocks_llm, stats.blocks_rejected) == ("haiku-subagent", 4, 1)
    assert blocks["b0001"].speech_source == "llm" and blocks["b0006"].speech_source == "llm_rejected"

    # Importing again starts from the filter text, so the result is identical (idempotent).
    again = cleanup.import_answers(paper, answers, model="haiku-subagent")
    assert (again.blocks_llm, again.blocks_rejected) == (4, 1)
    assert blocks["b0002"].speech == GOOD_ANSWERS["b0002"]

    with pytest.raises(ValueError):
        cleanup.import_answers(paper, [{"id": "b0001"}], model="x")


def test_a_defining_sentence_must_keep_its_symbol():
    """SAPG: replacing the symbol in the sentence that defines it says the name twice."""
    source = (r"where $\hat{A}^{\pi_{\theta}}(s,a)$ is an advantage function that estimates the contribution of "
              r"the transition to the gradient.")
    block = _paragraph(source)
    restated = ("where the estimated advantage function of the policy, state, and action is an advantage function "
                "that estimates the contribution of the transition to the gradient.")
    kept = ("where A hat to the policy of s and a is an advantage function that estimates the contribution of the "
            "transition to the gradient.")
    reason = cleanup.guard_reason(block, restated, mode="meaning")
    assert reason and '"advantage function" twice' in reason
    assert cleanup.guard_reason(block, kept, mode="meaning") is None
    assert cleanup.guard_reason(block, restated, mode="symbols", inline="symbols") != reason  # meaning mode only
    plural = _paragraph(r"where $\hat{Q}(s,a)$, $\hat{V}(s)$ are estimated $Q$ and value functions.")
    assert cleanup.guard_reason(plural, "where the estimated Q function and the estimated value function are "
                                        "estimated Q and value functions.", mode="meaning")
