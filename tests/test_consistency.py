"""Meaning mode: the same symbol must be read the same way in every equation (check + repair pass)."""

from __future__ import annotations

import pytest

from paper_reader import cleanup
from paper_reader.cleanup import _SymbolName as Sym
from paper_reader.cleanup import check_consistency, normalize_name
from paper_reader.models import Block, Paper, Section
from paper_reader.textfilter import apply_filter
from tests.test_cleanup import FakeClient


def test_normalize_name():
    assert normalize_name("The Imitation-Weight ") == normalize_name("imitation weight") == "imitation weight"
    assert normalize_name("action at time step t") == normalize_name("the action at each step t") == "action"
    assert normalize_name("imitation objective for motion i") == "imitation objective for motion i"


def test_prose_may_keep_the_papers_own_words():
    uses = {"b1": [Sym(latex="s_{t}", name="state of the character")],
            "b2": [Sym(latex="s_{t}", name="state of the character")]}
    speeches = {"b1": "the state of the character at time t equals x.",  # an equation reading
                "b2": "but the observation is augmented with the ball."}  # prose keeps the paper's own wording
    assert check_consistency(uses, speeches, ["b1", "b2"], equations={"b1"}).issues == []
    assert check_consistency(uses, speeches, ["b1", "b2"], equations={"b1", "b2"}).issues  # as an equation it fails


def test_check_flags_a_different_name_and_an_unspoken_name():
    uses = {
        "b1": [Sym(latex=r"\omega^{I}", name="imitation weight"), Sym(latex=r"r^{I}_{t}", name="imitation objective")],
        "b2": [Sym(latex=r"\omega^{I}", name="the imitation weight"), Sym(latex=r"r^{I}_{t+1}", name="imitation objective")],
        "b3": [Sym(latex=r"\omega^{I}", name="imitation coefficient")],
        "b4": [Sym(latex=r"w_{p}", name="pose weight"), Sym(latex=r"w_{v}", name="velocity weight")],
    }
    speeches = {
        "b1": "the reward equals the imitation weight times the imitation objective at time step t.",
        "b2": "we set the imitation weight to 0.7; the imitation objective at the next step is high.",
        "b3": "the imitation coefficient equals 0.7.",
        "b4": "the pose weight and the velocity weight sum to 0.75.",  # w_p and w_v are different symbols
    }
    report = check_consistency(uses, speeches, ["b1", "b2", "b3", "b4"])
    assert report.names[("omega", "I", "")] == "imitation weight"  # 2 of 3 blocks
    assert report.names[("r", "I", "")] == "imitation objective"  # r^I_t and r^I_{t+1} are one symbol
    assert report.issues == [("b3", r"\omega^{I}", 'said "imitation coefficient"', "imitation weight")]

    speeches["b2"] = "we set the imitation factor to 0.7; the imitation objective at the next step is high."
    unspoken = check_consistency(uses, speeches, ["b1", "b2", "b3", "b4"]).issues
    assert ("b2", r"\omega^{I}", "doesn't say the name", "imitation weight") in unspoken  # reported one name, said another


def test_tie_goes_to_the_first_name_in_reading_order():
    uses = {"b2": [Sym(latex=r"\gamma", name="discount rate")], "b1": [Sym(latex=r"\gamma", name="discount factor")]}
    speeches = {"b1": "the discount factor", "b2": "the discount rate"}
    assert check_consistency(uses, speeches, ["b1", "b2"]).names[("gamma", "", "")] == "discount factor"


def _paper() -> Paper:
    defs = Block(id="b0001", type="paragraph", text=r"At every time step $t$ the reward $r_{t}$ combines terms. "
                                                    r"Here, $r^{I}_{t}$ is the imitation objective and $\omega^{I}$ "
                                                    r"is its weight, the imitation weight. The mean reward $\bar{r}$ "
                                                    r"sums over time steps.")
    eq1 = Block(id="b0002", type="equation", eq_number="1", latex=r"r_{t}=\omega^{I}r^{I}_{t}")
    eq2 = Block(id="b0003", type="equation", eq_number="2", latex=r"\bar{r}=\omega^{I}\sum_{t}r^{I}_{t}")
    eq3 = Block(id="b0004", type="equation", eq_number="3", latex=r"\omega^{I}=0.7")
    paper = Paper(paper_id="c.1v1", arxiv_id="c.1", title="Imitation", source_url="x", sections=[
        Section(id="S1", number="1", title="Method", blocks=[defs, eq1, eq2, eq3])])
    apply_filter(paper)
    return paper


def test_haiku_pass_repairs_an_inconsistent_block_once():
    readings = {
        "b0002": ("Equation 1: the reward equals the imitation weight times the imitation objective at time step t.",
                  [{"latex": r"\omega^{I}", "name": "imitation weight"}, {"latex": r"r^{I}_{t}", "name": "imitation objective"}]),
        "b0003": ("Equation 2: the mean reward equals the imitation weight times the sum over time steps of the "
                  "imitation objective.",
                  [{"latex": r"\omega^{I}", "name": "imitation weight"}, {"latex": r"r^{I}_{t}", "name": "imitation objective"}]),
        "b0004": ("Equation 3: the weight of the imitation objective equals 0.7.", [{"latex": r"\omega^{I}", "name": "weight of the imitation objective"}]),
    }
    repaired = "Equation 3: the imitation weight equals 0.7."
    seen_names = []

    def answer(blocks):
        out = []
        for b in blocks:
            if "names" in b:  # the repair request
                seen_names.append(b["names"])
                out.append({"id": b["id"], "speech": repaired, "symbols": [{"latex": r"\omega^{I}", "name": "imitation weight"}]})
            elif b["id"] in readings:
                speech, symbols = readings[b["id"]]
                out.append({"id": b["id"], "speech": speech, "symbols": symbols})
            else:
                out.append({"id": b["id"], "speech": b["filter_speech"], "symbols": []})
        return out

    paper = _paper()
    client = FakeClient(answer)
    stats = cleanup.cleanup_paper(paper, client=client, equations="meaning")

    assert seen_names == [{r"\omega^{I}": "imitation weight", r"r^{I}_{t}": "imitation objective"}] or \
        seen_names == [{r"\omega^{I}": "imitation weight"}]
    assert next(b for b in paper.iter_blocks() if b.id == "b0004").speech == repaired
    assert stats.symbols_repaired == 1 and stats.inconsistent_symbols == []
    assert stats.symbol_names[r"\omega^{I}"] == "imitation weight"
    assert all(c["output_config"]["format"]["schema"] == cleanup.RESPONSE_SCHEMA_MEANING for c in client.calls)


def test_remaining_inconsistency_is_reported():
    """A name made of the paper's own words passes the grounding check, so only consistency catches it."""
    def answer(blocks):  # never fixes block b0004, not even when asked with the names
        return [{"id": b["id"], "speech": "Equation 3: the weight of the imitation objective equals 0.7.",
                 "symbols": [{"latex": r"\omega^{I}", "name": "weight of the imitation objective"}]} if b["id"] == "b0004" else
                {"id": b["id"], "speech": "Equation 1: the reward equals the imitation weight times the imitation "
                                          "objective at time step t." if b["id"] == "b0002" else
                 "Equation 2: the mean reward equals the imitation weight times the sum over time steps of the "
                 "imitation objective." if b["id"] == "b0003" else b["filter_speech"],
                 "symbols": [{"latex": r"\omega^{I}", "name": "imitation weight"}] if b["id"] in ("b0002", "b0003") else []}
                for b in blocks]

    paper = _paper()
    stats = cleanup.cleanup_paper(paper, client=FakeClient(answer), equations="meaning")
    assert stats.inconsistent_symbols and "weight of the imitation objective" in stats.inconsistent_symbols[0]
    assert any("symbol read inconsistently" in w for w in paper.warnings)


def test_says_name_ignores_word_order_but_not_different_words():
    from paper_reader.cleanup import says_name

    assert says_name("policy probability of the action given the state",
                     "the probability of the action given the state under the policy")
    assert says_name("imitation weight", "the weight for imitation times the objective")
    assert not says_name("imitation weight", "the imitation coefficient times the objective")
    assert says_name("end effector weight", "the end effector weights")  # word forms


@pytest.mark.parametrize(("a", "b", "same"), [
    (r"r_{t}(\pi_{\theta})", r"r_k", False),  # SAPG: the probability ratio is not the reward
    (r"V_{\pi_i}(s)", r"V_{\pi_i}(s_{t+1})", True),  # plain variables as arguments don't matter
    (r"s'_t", r"s^{\prime}_{t+1}", True),  # an apostrophe is a prime
    (r"s'_t", r"s_t", False),
])
def test_symbol_keys(a, b, same):
    from paper_reader.definitions import symbol_key
    assert (symbol_key(a) == symbol_key(b)) is same


def test_glossary_names_reach_every_block_and_win_the_consistency_check():
    """Stage 2: one name per symbol for the whole paper, decided before the narration requests."""
    seen = {}

    def answer(blocks):
        for b in blocks:
            seen[b["id"]] = b.get("names")
        return [{"id": b["id"], "speech": "Equation 3: the imitation weight equals 0.7." if b["id"] == "b0004" else b["filter_speech"],
                 "symbols": [{"latex": r"\omega^{I}", "name": "imitation weight"}] if b["id"] == "b0004" else []}
                for b in blocks]

    paper = _paper()
    client = FakeClient(answer, names={r"\omega^{I}": "imitation weight", r"r^{I}_{t}": "imitation objective",
                                       r"\bar{r}": "average reward of the policy"})  # "policy": not in the paper
    stats = cleanup.cleanup_paper(paper, client=client, equations="meaning")
    assert len(client.glossary_calls) == 1
    asked = client.glossary_calls[0]["messages"][0]["content"]
    assert r"\omega^{I}" in asked and "the imitation weight" in asked  # its definition goes along
    assert seen["b0002"] == {r"\omega^{I}": "imitation weight", r"r^{I}_{t}": "imitation objective"}
    assert seen["b0004"] == {r"\omega^{I}": "imitation weight"}
    assert stats.glossary_names == 2  # the ungrounded name was dropped
    assert stats.symbol_names == {r"\omega^{I}": "imitation weight"} and stats.inconsistent_symbols == []
