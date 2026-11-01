"""Reading equations by meaning: symbol atoms, definition lookup, request payloads and the grounding guard."""

from __future__ import annotations

from paper_reader import cleanup
from paper_reader.definitions import DefinitionIndex, math_atoms, paper_vocabulary
from paper_reader.models import Block, Paper, Section
from paper_reader.textfilter import apply_filter


def test_math_atoms_normalize_scripts_and_skip_function_names():
    assert math_atoms(r"r_{t}^{I}") == math_atoms(r"r^{I}_{t}") == {("r", "I", "t")}
    assert math_atoms(r"r_{t}=\omega^{I}r^{I}_{t}+\omega^{G}r^{G}_{t}") == {
        ("r", "", "t"), ("omega", "I", ""), ("r", "I", "t"), ("omega", "G", ""), ("r", "G", "t")}
    assert math_atoms(r"\mathcal{L}_{\mathrm{rec}}+\exp(x)") == {("mathcal:L", "", "rec"), ("x", "", "")}


def _paper() -> Paper:
    background = [
        Block(id="b0001", type="paragraph", text=r"At each step $t$ the agent receives a reward $r_{t}$ and "
                                                 r"$\gamma$ is the discount factor. Training uses many GPUs."),
    ]
    method = [
        Block(id="b0002", type="paragraph", text="The reward combines imitation and task terms:"),
        Block(id="b0003", type="equation", latex=r"r_{t}=\omega^{I}r^{I}_{t}+\omega^{G}r^{G}_{t}.", eq_number="3"),
        Block(id="b0004", type="paragraph", text=r"Here, $r^{I}_{t}$ and $r^{G}_{t}$ represent the imitation and "
                                                 r"task objectives, with $\omega^{I}$ and $\omega^{G}$ being their "
                                                 r"respective weights."),
    ]
    experiments = [
        Block(id="b0005", type="paragraph", text=r"We set $\omega^{I}=0.7$ and $\omega^{G}=0.3$ for all tasks. "
                                                 r"The plot shows $r_{t}$ over time."),
    ]
    paper = Paper(paper_id="d.1v1", arxiv_id="d.1", title="Imitation", source_url="x", sections=[
        Section(id="S1", number="1", title="Background", blocks=background),
        Section(id="S2", number="2", title="Method", blocks=method),
        Section(id="S3", number="3", title="Experiments", blocks=experiments),
    ])
    apply_filter(paper)
    return paper


def test_definitions_come_from_anywhere_in_the_paper_and_skip_the_context():
    paper = _paper()
    index = DefinitionIndex(paper)
    equation = paper.sections[1].blocks[1]
    found = index.for_block(equation)
    assert any("respective weights" in s for s in found)  # the "where" sentence after the equation
    assert any("receives a reward" in s for s in found)  # defined in an earlier section
    assert not any("plot shows" in s for s in found)  # mentions r_t but defines nothing
    assert not any("GPUs" in s for s in found)
    skipped = index.for_block(equation, skip=(paper.sections[1].blocks[2].text,))
    assert not any("respective weights" in s for s in skipped)  # already sent as context_after

    inline = paper.sections[2].blocks[0]  # "We set $\omega^{I}=0.7$ ..." gets the weights' definition
    assert any("respective weights" in s for s in index.for_block(inline))


def test_payloads_carry_definitions_only_in_meaning_mode():
    paper = _paper()

    def equation_payload(mode):
        request = next(r for r in cleanup.plan_requests(paper, mode) if any(b.id == "b0003" for b in r.blocks))
        message = request.user_message
        return message[message.index('"id": "b0003"') - 4:]

    assert '"definitions"' not in equation_payload("symbols")
    meaning = equation_payload("meaning")
    assert '"definitions"' in meaning and "receives a reward" in meaning


def test_vocabulary_and_grounding_guard():
    paper = _paper()
    vocabulary = paper_vocabulary(paper)
    assert {"imitation", "objective", "weight", "omega"} <= vocabulary  # "objectives"/"weights" singularized
    equation = next(b for b in paper.iter_blocks() if b.id == "b0003")
    grounded = ("Equation 3: the reward at step t equals the imitation weight times the imitation objective at "
                "step t, plus the task weight times the task objective at step t.")
    invented = ("Equation 3: the reward at step t equals the friction coefficient times the contact penalty at "
                "step t, plus the task weight times the task objective at step t.")
    assert cleanup.guard_accepts(equation, grounded, mode="meaning", vocabulary=vocabulary)
    assert not cleanup.guard_accepts(equation, invented, mode="meaning", vocabulary=vocabulary)
    # symbols mode: no grounding check, but the equation must be read symbol by symbol
    assert not cleanup.guard_accepts(equation, grounded, mode="symbols")
    assert cleanup.guard_accepts(equation, "Equation 3: r t equals omega I times r t I plus omega G times r t G.",
                                 mode="symbols")


def test_apposition_counts_as_a_definition_and_every_symbol_gets_one():
    """Regression for DeepMimic (1804.02717): "The center-of-mass reward $r^{c}_{t}$ penalizes ..." has no
    cue word, and a 4-term equation used to run out of room before its last symbols were defined."""
    defs = [r"The pose reward $r^{p}_{t}$ matches joint orientations.",
            r"The velocity reward $r^{v}_{t}$ matches joint velocities.",
            r"The end-effector reward $r^{e}_{t}$ matches hand and foot positions.",
            r"Finally, the center-of-mass reward $r^{c}_{t}$ penalizes deviations of the center of mass.",
            r"The weights $w^{p}$, $w^{v}$, $w^{e}$ and $w^{c}$ balance the four terms."]
    noise = [rf"The plot shows $r^{{p}}_{{t}}$ over epoch {i}." for i in range(8)]  # mentions, defines nothing
    blocks = [Block(id=f"b{i:04d}", type="paragraph", text=t) for i, t in enumerate(noise + defs)]
    equation = Block(id="b9999", type="equation", eq_number="5",
                     latex=r"r^{I}_{t}=w^{p}r^{p}_{t}+w^{v}r^{v}_{t}+w^{e}r^{e}_{t}+w^{c}r^{c}_{t}")
    paper = Paper(paper_id="m.1v1", arxiv_id="m.1", title="Mimic", source_url="x",
                  sections=[Section(id="S1", title="Reward", blocks=blocks + [equation])])
    found = DefinitionIndex(paper).for_block(equation)
    for needle in ("pose reward", "velocity reward", "end-effector reward", "center-of-mass reward", "weights"):
        assert any(needle in s for s in found), needle
    assert not any("plot shows" in s for s in found)
