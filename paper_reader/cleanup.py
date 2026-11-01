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
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, ValidationError

from . import pricing
from .definitions import DefinitionIndex, atom_key, math_atoms, paper_vocabulary, symbol_key
from .models import Block, CleanupStats, Paper, Section
from .textfilter import (apply_filter, equation_label, equation_reading, equation_speech, filter_text, inline_math,
                         latex_to_speech)

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

# Equation mode "meaning" reads defined symbols by what they mean ("the imitation weight" instead
# of "omega to the I"): readings get longer, so the maths allowance and the equation length grow.
# In return every word the model adds must occur in the paper (or be ordinary maths language or a
# spoken unit), so a meaning can't be invented. MAX_UNGROUNDED is 0: on DeepMimic a single allowed
# unknown word let "energy weight" through for w^e, which that paper defines as the end-effector weight.
EquationMode = Literal["symbols", "meaning"]
EQUATION_MODES: tuple[EquationMode, ...] = ("symbols", "meaning")
MEANING_MATH_FACTOR = 4.0
MAX_EQUATION_WORDS_MEANING = 140
MAX_UNGROUNDED = 0
DEFINITIONS_FOR_PROSE = {"max_sentences": 3, "max_chars": 600}
_SPOKEN_MATH = set("""
equals equal plus minus times divided over sum product integral expectation expected exponential
logarithm square squared cube cubed root power inverse transpose norm absolute value gradient partial
derivative respect drawn given from with each every all over interval between maximum minimum
argmax argmin softmax function fraction quantity previous next current first last zero infinity
greater less than approximately proportional distributed according vector matrix element entry
divergence variance mean average probability distribution sequence tuple set index which where
superscript subscript hat bar tilde star prime dotted dot cross conjugate magnitude elementwise
capital uppercase lowercase calligraphic bold script blackboard
projected projection onto
hadamard concatenation concatenated composition indicator otherwise case cases condition such that per
hertz kilohertz megahertz gigahertz second seconds milliseconds microseconds minute minutes hour hours
meter meters centimeters millimeters kilometers kilogram kilograms grams newton newtons joules watts
volts degrees radians percent pixels frames bytes kilobytes megabytes gigabytes
""".split())
# Words the prompt's own rules introduce: citation repairs, abbreviation expansions, spoken numbers.
_RULE_WORDS = set("""
earlier work works studies study colleagues example respect state versus approximately respectively
compare online about zero three four five seven eight nine eleven twelve thirteen fourteen fifteen
sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred
thousand million billion point half quarter third fourth fifth tenth hundredth thousandth
equation equations cetera
their them they these those this that then than when what whose whom with within without into onto
upon under between across along through also only just both either neither other others same such
very more most less much many some here there have been being were will would could should
""".split())
# Symbols mode reads display equations in full. A reading must say at least this share of the
# symbols the rule reading says (counted with repeats), so a description of what the equation
# computes ("the PPO objective is the expectation of the clipped ratio...") is rejected. On SAPG,
# full readings cover 0.77-1.0 and such descriptions 0.13-0.72.
MIN_SYMBOL_COVERAGE = 0.75
EQUATION_LENGTH_FACTOR = 1.6  # a full reading may run this much longer than the rule reading
_NOT_SYMBOLS = _SPOKEN_MATH | _RULE_WORDS | set("""
the of to and by for is as in at it its an be min max exp log clip sin cos tan tanh sigmoid relu lim det tr
diag sign arg sub sup if else for all
""".split())
_DIGITS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
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
3. Display equations. Begin with "Equation N:" when eq_number is given, then read the whole \
equation aloud the way a researcher reads it at the board: every symbol, term, operator, sum, \
fraction, exponent and condition, in the same order. A long equation is still read in full, in \
one or two flowing sentences. Never replace the symbols with a description of what the equation \
computes, and never answer with a bare "Equation N." A label of several letters in a subscript \
or superscript is said as a word: V^{target} is "V target", \\pi_{old} is "pi old".
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

# Reading maths by meaning, for display equations and/or the symbols inside sentences (two options).
_MEANING_COMMON = """MATHS BY MEANING. This listener wants {what} read by what the symbols mean, not \
letter by letter. For every symbol the paper defines:
- Blocks with maths may carry "definitions": sentences from elsewhere in the paper that define \
the symbols the block uses. Together with "context", "context_after" and the block itself, they \
are the ONLY source of meanings.
- Name a defined symbol by its meaning ("the reward at time step t", "the imitation weight", "the \
discount factor") instead of its letters ("r t", "omega I", "gamma"). Indices become their \
meaning too when it is defined ("at time step t", "of skill k", "at the previous iteration").
- A symbol the paper does not define, but uses in a standard role of its field, gets the field's \
standard term, provided the paper's own text uses that term. In reinforcement learning: pi is "the \
policy", V "the value function", Q "the Q function", A "the advantage function", gamma "the \
discount factor", s "the state", a "the action", r "the reward", theta "the policy parameters". \
Likewise in other fields: L "the loss", eta "the learning rate", u "the control input", q "the joint \
positions", tau "the joint torques". Only when the block or its context shows the symbol in that role. \
The paper's own definitions come first: a symbol that the definitions, context or block define, in \
words or by a formula, keeps that meaning even where the field usually uses the letter for something \
else (an r_t defined as the ratio of two policies is "the probability ratio", not "the reward").
- Any other undefined symbol keeps its spoken name ("beta", "K", "x"). Never guess a meaning.
- The worked examples use their own symbols. Apart from the field's standard terms, a symbol in a \
request means only what that request's definitions, context and block say: never carry a meaning \
over from an example or another paper.
- One name per symbol for the whole paper: a symbol is read the same way in every equation and \
sentence. When a request gives "names" (symbol -> name), use exactly those names.
- For every block also return "symbols": each symbol you named by meaning, as {"latex": the bare \
symbol, "name": the name you said, without "the" and without its indices}, for example \
{"latex": "r^{I}_{t}", "name": "imitation objective"}. Leave it empty when you named none."""
_MEANING_EQUATIONS = """- Display equations (instead of rule 3): keep the mathematical structure exact. Every \
term, operator, sum, fraction, exponent, condition and constant stays, in the same order, said with \
"equals", "plus", "minus", "times", "divided by", "the sum over ... of", "the expectation over ... \
of", "the exponential of", "to the power". Do not simplify, merge or drop terms, and do not explain \
why the equation holds. "Equation N:" when it is numbered, then one flowing sentence (two at most \
for a long equation)."""
_MEANING_INLINE = """- Symbols inside sentences (instead of rule 2): read a defined symbol by its meaning, inside \
the sentence's own words; the rest of the sentence stays word for word. When the words right \
before the symbol already name it ("the slip penalty $\\kappa_{t}$"), drop the symbol instead of \
saying the name twice. But in the sentence that defines a symbol ("$u_{t}$ denotes the foot speed", \
"with $u_{t}$ being ...", "where $u_{t}$ is ..."), keep the symbol's spoken letters, also for a \
symbol with a standard name in the field: replacing it there would make the definition say "the \
foot speed denotes the foot speed". A bracketed list of \
words ("$e \\in$ [left foot, right foot]") is a set, read "one of left foot, right foot", not an interval."""
_SYMBOLS_EQUATIONS = "- Display equations are read as rule 3 says, symbol by symbol."
_SYMBOLS_INLINE = "- Symbols inside sentences are read as rule 2 says, symbol by symbol."
_MEANING_SCOPE = {("meaning", "meaning"): "equations and the symbols inside sentences",
                  ("meaning", "symbols"): "display equations",
                  ("symbols", "meaning"): "the symbols inside sentences"}


def _meaning_rules(mode: EquationMode, inline: EquationMode) -> str:
    return "\n".join([
        _MEANING_COMMON.replace("{what}", _MEANING_SCOPE[mode, inline]),
        _MEANING_EQUATIONS if mode == "meaning" else _SYMBOLS_EQUATIONS,
        _MEANING_INLINE if inline == "meaning" else _SYMBOLS_INLINE,
    ])


@dataclass(frozen=True)
class _Example:
    title: str
    source: str
    speech: str  # the expected output
    eq_number: str | None = None  # equation examples: source is the LaTeX
    context: str = ""
    context_after: str = ""
    mode: Literal["both", "symbols", "meaning"] = "both"  # which prompt variant shows it
    definitions: tuple[str, ...] = ()
    symbols: tuple[tuple[str, str], ...] = ()  # meaning mode: (LaTeX, name) pairs the output names
    is_equation: bool = False  # an unnumbered display equation (eq_number None) is still an equation


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
    _Example("A long display equation: still read in full, every symbol in order",
             r"\mathcal{L} = \sum_{t=1}^{T} \| \hat{x}_t - x_t \|^2 + \beta \, D_{\mathrm{KL}}"
             r"\big(q_\phi(z \mid x) \,\|\, p(z)\big)",
             "Equation 4: L equals the sum from t equals 1 to T of the squared norm of x hat t minus "
             "x t, plus beta times the KL divergence between q phi of z given x and p of z.",
             eq_number="4",
             context="We train the world model by minimizing a reconstruction loss and a KL regularizer:",
             context_after=r"where $\beta$ weights the KL term.", mode="symbols"),
    _Example("Labels of several letters in sub- and superscripts are words",
             r"V^{target}_{t} = r_{t} + \gamma \, V_{old}(s_{t+1})",
             "Equation 6: V target t equals r t plus gamma times V old of s t plus 1.",
             eq_number="6",
             context="The critic regresses onto a bootstrapped target:", mode="symbols"),
    _Example("A short display equation: read it verbatim",
             r"a_t = \pi_\theta(o_t)",
             "Equation 2: a t equals pi theta of o t.",
             eq_number="2",
             context="At every step the policy maps the observation to an action:", mode="symbols"),
    # ---- equation mode "meaning"
    _Example("Meaning mode: a weighted sum, every symbol named by what it means",
             r"r_{t}=\lambda_{1}r^{\mathrm{trk}}_{t}-\lambda_{2}r^{\mathrm{tq}}_{t}",
             "Equation 3: the reward at time step t equals the tracking weight times the tracking reward at "
             "time step t, minus the torque weight times the torque penalty at time step t.",
             eq_number="3", mode="meaning",
             context="The reward combines target tracking with a torque penalty:",
             context_after=r"Here, $r^{\mathrm{trk}}_{t}$ rewards tracking the target and $r^{\mathrm{tq}}_{t}$ "
                           r"penalizes large joint torques, with $\lambda_{1}$ and $\lambda_{2}$ being their weights.",
             definitions=(r"The reward $r_{t}$ is given to the policy at every time step $t$.",),
             symbols=(("r_{t}", "reward"), (r"\lambda_{1}", "tracking weight"), (r"r^{\mathrm{trk}}_{t}", "tracking reward"),
                      (r"\lambda_{2}", "torque weight"), (r"r^{\mathrm{tq}}_{t}", "torque penalty"))),
    _Example("Meaning mode: an expectation with a sum, structure kept exactly",
             r"J(\theta)=\mathbb{E}_{\tau\sim p_{\theta}(\tau)}\left[\sum_{t=0}^{T}\gamma^{t}r_{t}\right]",
             "Equation 1: the expected return, as a function of the policy parameters, equals the expectation, "
             "over trajectories drawn from the trajectory distribution of the policy, of the sum from time step "
             "0 to the episode horizon T of the discount factor to the power t times the reward at time step t.",
             eq_number="1", mode="meaning",
             context=r"The agent learns policy parameters $\theta$ that maximize its expected return:",
             context_after=r"where $p_{\theta}(\tau)$ is the trajectory distribution induced by the policy and "
                           r"$\gamma$ is the discount factor.",
             definitions=(r"The reward $r_{t}$ is received at time step $t$, and $T$ is the episode horizon.",),
             symbols=((r"J(\theta)", "expected return"), (r"\theta", "policy parameters"),
                      (r"p_{\theta}(\tau)", "trajectory distribution of the policy"), ("T", "episode horizon"),
                      (r"\gamma", "discount factor"), ("r_{t}", "reward"))),
    _Example("Meaning mode: a fraction, both parts said in full",
             r"p_{k}=\frac{\exp(s_{k}/\tau)}{\sum_{j=1}^{K}\exp(s_{j}/\tau)}",
             "Equation 5: the probability of sampling skill k equals the exponential of its learning score "
             "divided by the temperature, divided by the sum over all K skills of the exponential of their "
             "learning score divided by the temperature.",
             eq_number="5", mode="meaning",
             context="Skills are sampled in proportion to how much they are improving:",
             context_after=r"where $s_{k}$ is the learning score of skill $k$ and $\tau$ is a temperature.",
             definitions=(r"We train $K$ skills, and $p_{k}$ is the probability of sampling skill $k$ in the "
                          r"next episode.",),
             symbols=(("p_{k}", "probability of sampling skill"), ("s_{k}", "learning score"), (r"\tau", "temperature"))),
    _Example("Meaning mode: a symbol the paper never defines keeps its name (beta)",
             r"\mathcal{L}=\mathcal{L}_{\mathrm{rec}}+\beta\,D_{\mathrm{KL}}\big(q_{\phi}(z\mid x)\,\|\,p(z)\big)",
             "Equation 2: the loss equals the reconstruction loss plus beta times the KL divergence between "
             "the encoder distribution over the latent code z given the input x and the standard normal prior.",
             eq_number="2", mode="meaning",
             context=r"We train the model by minimizing the loss $\mathcal{L}$:",
             definitions=(r"$\mathcal{L}_{\mathrm{rec}}$ is the reconstruction loss of the decoder.",
                          r"The encoder $q_{\phi}(z\mid x)$ maps an input $x$ to a distribution over the latent "
                          r"code $z$, and $p(z)$ is a standard normal prior."),
             symbols=((r"\mathcal{L}", "loss"), (r"\mathcal{L}_{\mathrm{rec}}", "reconstruction loss"),
                      (r"q_{\phi}(z\mid x)", "encoder distribution over the latent code"), ("p(z)", "standard normal prior"))),
    _Example("Meaning mode: standard symbols of the field get the field's terms, when the paper uses them",
             r"\delta_{t}=r_{t}+\gamma V(s_{t+1})-V(s_{t})",
             "Equation 4: the temporal-difference error at time step t equals the reward at time step t plus the "
             "discount factor times the value function of the next state, minus the value function of the state "
             "at time step t.",
             eq_number="4", mode="meaning",
             context=r"The critic is trained on the temporal-difference error $\delta_{t}$:",
             context_after="The actor then favours actions with a positive error.",
             definitions=("At every time step the agent observes a state, takes an action and receives a reward; "
                          "future rewards are weighted by a discount factor, and the critic learns the value function.",),
             symbols=((r"\delta_{t}", "temporal-difference error"), ("r_{t}", "reward"), (r"\gamma", "discount factor"),
                      ("V", "value function"), ("s_{t}", "state"))),
    _Example("Meaning mode: indices said by their meaning",
             r"\Delta_{k}=\bar{R}_{k}^{(i)}-\bar{R}_{k}^{(i-1)}",
             "Equation 6: the learning progress of skill k equals the average return of skill k at iteration "
             "i minus its average return at the previous iteration.",
             eq_number="6", mode="meaning",
             context="We measure how fast each skill improves:",
             context_after=r"where $\bar{R}_{k}^{(i)}$ is the average return of skill $k$ at iteration $i$.",
             definitions=(r"We call $\Delta_{k}$ the learning progress of skill $k$.",),
             symbols=((r"\Delta_{k}", "learning progress"), (r"\bar{R}_{k}^{(i)}", "average return"))),
    _Example("Symbols in sentences: the words already name the symbol, so the symbol is dropped",
             r"The slip penalty $\kappa_{t}$ discourages foot sliding and is computed from the foot speed $u_{t}$.",
             "The slip penalty discourages foot sliding and is computed from the foot speed.",
             mode="meaning", symbols=((r"\kappa_{t}", "slip penalty"), ("u_{t}", "foot speed"))),
    _Example("Symbols in sentences: a symbol on its own is read by its meaning",
             r"When $\kappa_{t}$ grows, the policy shortens its stride.",
             "When the slip penalty grows, the policy shortens its stride.",
             mode="meaning",
             definitions=(r"The slip penalty $\kappa_{t}$ discourages foot sliding and is computed from the foot "
                          r"speed $u_{t}$.",),
             symbols=((r"\kappa_{t}", "slip penalty"),)),
    _Example("Symbols in sentences: the sentence that defines a symbol keeps its letters",
             r"Here, $u_{t}$ denotes the foot speed of leg $l \in$ [front left, front right], in meters per second.",
             "Here, u t denotes the foot speed of leg l, one of front left and front right, in meters per second.",
             mode="meaning"),
    _Example("Symbols in sentences: the field's standard terms",
             r"The update keeps $\pi_{\theta}$ close to $\pi_{\theta_{old}}$ and is stable for any $\gamma$ below 1.",
             "The update keeps the policy close to the old policy and is stable for any discount factor below 1.",
             mode="meaning",
             definitions=("The old policy collected the data, and future rewards are weighted by a discount factor.",),
             symbols=((r"\pi_{\theta}", "policy"), (r"\pi_{\theta_{old}}", "old policy"), (r"\gamma", "discount factor"))),
    _Example("Symbols in sentences: a defining sentence keeps the letters, also for a standard symbol",
             r"where $\hat{V}(s)$ is a value function that predicts the return from state $s$, and "
             r"$\pi_{\theta}$ is updated to increase it.",
             "where V hat of s is a value function that predicts the return from state s, and the policy is "
             "updated to increase it.",
             mode="meaning",
             definitions=("The policy is a neural network with parameters theta.",),
             symbols=((r"\pi_{\theta}", "policy"),)),
    _Example("Meaning mode: inline maths inside a sentence",
             r"For all tasks we set $\lambda_{1}=1.0$ and $\lambda_{2}=0.05$.",
             "For all tasks we set the tracking weight to 1.0 and the torque weight to 0.05.",
             mode="meaning",
             definitions=(r"Here, $r^{\mathrm{trk}}_{t}$ rewards tracking the target and $r^{\mathrm{tq}}_{t}$ "
                          r"penalizes large joint torques, with $\lambda_{1}$ and $\lambda_{2}$ being their weights.",),
             symbols=((r"\lambda_{1}", "tracking weight"), (r"\lambda_{2}", "torque weight"))),
]


def _example_block(example: _Example, index: int) -> dict:
    block_id = f"ex{index:02d}"
    definitions = {"definitions": list(example.definitions)} if example.definitions else {}
    if example.eq_number is None and not example.is_equation:
        return {"id": block_id, "type": "paragraph", "source": example.source, **definitions,
                "filter_speech": filter_text(example.source)}
    block = Block(id=block_id, type="equation", latex=example.source, eq_number=example.eq_number)
    return {"id": block_id, "type": "equation", "latex": example.source, "eq_number": example.eq_number or "",
            "context": example.context, "context_after": example.context_after, **definitions,
            "filter_speech": equation_speech(block)}


def examples_for(mode: EquationMode, inline: EquationMode | None = None) -> list[_Example]:
    """The worked examples for a reading: equation examples follow `mode`, sentence examples `inline`."""
    inline = inline or mode
    return [e for e in _EXAMPLES
            if e.mode == "both" or e.mode == (mode if e.eq_number or e.is_equation else inline)]


def _build_system_prompt(mode: EquationMode, inline: EquationMode | None = None) -> str:
    inline = inline or mode
    meaning = "meaning" in (mode, inline)
    parts = [_JOB] + ([_meaning_rules(mode, inline)] if meaning else [])
    for index, example in enumerate(examples_for(mode, inline), 1):
        block = _example_block(example, index)
        answer = {"id": block["id"], "speech": example.speech}
        if meaning:
            answer["symbols"] = [{"latex": latex, "name": name} for latex, name in example.symbols]
        parts.append(f"Example {index}. {example.title}.\n"
                     f"Input: {json.dumps(block, ensure_ascii=False)}\n"
                     f"Output: {json.dumps(answer, ensure_ascii=False)}")
    return "\n\n".join(parts) + "\n"


# Built once per process from constants only, so each is byte-identical on every request
# (one prompt cache per reading: equations x symbols inside sentences).
SYSTEM_PROMPTS: dict[tuple[EquationMode, EquationMode], str] = {
    (mode, inline): _build_system_prompt(mode, inline) for mode in EQUATION_MODES for inline in EQUATION_MODES}
SYSTEM_PROMPT = SYSTEM_PROMPTS["symbols", "symbols"]


def system_prompt(mode: EquationMode = "symbols", inline: EquationMode | None = None) -> str:
    """mode: display equations; inline: symbols inside sentences (None = same as mode)."""
    key = (mode, inline or mode)
    if key not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown reading {key!r}; each part is one of {', '.join(EQUATION_MODES)}")
    return SYSTEM_PROMPTS[key]

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
RESPONSE_SCHEMA_MEANING = json.loads(json.dumps(RESPONSE_SCHEMA))
_item = RESPONSE_SCHEMA_MEANING["properties"]["blocks"]["items"]
_item["properties"]["symbols"] = {
    "type": "array",
    "items": {"type": "object", "properties": {"latex": {"type": "string"}, "name": {"type": "string"}},
              "required": ["latex", "name"], "additionalProperties": False},
}
_item["required"] = ["id", "speech", "symbols"]


def response_schema(mode: EquationMode = "symbols", inline: EquationMode | None = None) -> dict:
    return RESPONSE_SCHEMA_MEANING if "meaning" in (mode, inline or mode) else RESPONSE_SCHEMA


class _SymbolName(BaseModel):
    latex: str
    name: str


class _BlockSpeech(BaseModel):
    id: str
    speech: str
    symbols: list[_SymbolName] = []  # meaning mode: the names this block's reading used


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


def _block_payload(section: Section, index: int, definitions: DefinitionIndex | None = None,
                   mode: EquationMode = "meaning", inline: EquationMode | None = None) -> dict:
    """One block as the model sees it. With a DefinitionIndex, blocks whose maths is read by meaning
    (equations when mode is "meaning", sentences when inline is) carry the paper's definitions."""
    inline = inline or mode
    block = section.blocks[index]
    payload = {"id": block.id, "type": block.type}
    if block.type == "equation":
        context = _neighbour_text(section.blocks, index, -1)
        context_after = _neighbour_text(section.blocks, index, +1)
        payload |= {"latex": block.latex or "", "eq_number": block.eq_number or "",
                    "context": context, "context_after": context_after}
        found = definitions.for_block(block, skip=(context, context_after)) if definitions and mode == "meaning" else []
    else:
        payload["source"] = block.text
        by_meaning = definitions and inline == "meaning" and "$" in block.text
        found = definitions.for_block(block, **DEFINITIONS_FOR_PROSE) if by_meaning else []
    if found:
        payload["definitions"] = found
    payload["filter_speech"] = block.speech or ""
    return payload


def plan_requests(paper: Paper, mode: EquationMode = "symbols", inline: EquationMode | None = None) -> list[CleanupRequest]:
    """One request per section, split into chunks of at most MAX_CHUNK_CHARS of block JSON."""
    inline = inline or mode
    definitions = DefinitionIndex(paper) if "meaning" in (mode, inline) else None
    requests = []
    for section in paper.sections:
        heading = section.speech_title or section.title
        chunk: list[tuple[Block, dict]] = []
        size = 0
        for index, block in enumerate(section.blocks):
            if not block.speak or block.type == "figure" or not (block.speech or block.type == "equation"):
                continue
            payload = _block_payload(section, index, definitions, mode, inline)
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


def _word_forms(word: str) -> set[str]:
    """The word and its likely base forms: entries/entry, separated/separate, weighting/weight."""
    forms = {word, word.rstrip("s")}
    for suffix, replacement in (("ies", "y"), ("es", ""), ("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("ly", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            forms.add(word[: -len(suffix)] + replacement)
    return forms


def _ungrounded(speech: str, filtered: str, vocabulary: set[str]) -> list[str]:
    """Content words the model added that the paper never uses: an invented meaning."""
    known = set(_words(filtered)) | vocabulary | _SPOKEN_MATH | _RULE_WORDS
    added = re.findall(r"[a-z]+", speech.lower())
    return [w for w in added if len(w) > 3 and not (_word_forms(w) & known)]


_COPULAS = {"is", "are", "denotes", "denote", "represents", "represent", "being", "means"}
_FUNCTION_WORDS = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "by", "for", "with", "that", "this",
                   "its", "their", "which", "where", "from", "as", "is", "are", "be", "been"}


def _content_bigrams(words: list[str]) -> set[tuple[str, str]]:
    return {(a, b.removesuffix("s")) for a, b in zip(words, words[1:])
            if len(a) > 2 and len(b) > 2 and a not in _FUNCTION_WORDS and b not in _FUNCTION_WORDS}


def _restated_names(text: str) -> set[str]:
    """Phrases a sentence says before "is"/"denotes" and again right after it: "the estimated advantage
    function ... is an advantage function" (a defining sentence whose symbol was replaced by its name)."""
    found = set()
    for sentence in re.split(r"[.;:?!]", text.lower()):
        words = re.findall(r"[a-z0-9]+", sentence)
        for i, word in enumerate(words):
            if word in _COPULAS:
                found |= {" ".join(pair) for pair in _content_bigrams(words[:i]) & _content_bigrams(words[i + 1:i + 8])}
    return found


def _symbol_words(text: str) -> Counter:
    words = (_DIGITS.get(w, w) for w in re.findall(r"[a-z0-9]+", text.lower()))
    return Counter(w for w in words if w not in _NOT_SYMBOLS)


def symbol_coverage(rule_reading: str, speech: str) -> float:
    """Share of the symbols in the rule reading of an equation that the speech says too (with repeats)."""
    expected, said = _symbol_words(rule_reading), _symbol_words(speech)
    total = sum(expected.values())
    return 1.0 if total == 0 else sum(min(n, said[w]) for w, n in expected.items()) / total


def guard_accepts(block: Block, speech: str, *, mode: EquationMode = "symbols", inline: EquationMode | None = None,
                  vocabulary: set[str] | None = None) -> bool:
    """Whether the model's speech may replace the filter speech of this block. Where maths is read by
    meaning, readings may be longer, and with a vocabulary every added word must be grounded in the paper."""
    return guard_reason(block, speech, mode=mode, inline=inline, vocabulary=vocabulary) is None


def guard_reason(block: Block, speech: str, *, mode: EquationMode = "symbols", inline: EquationMode | None = None,
                 vocabulary: set[str] | None = None) -> str | None:
    """None if the speech is acceptable, else why not (shown as a clean-up warning). `mode` is how
    display equations are read, `inline` how symbols inside sentences are (None = same as mode)."""
    by_meaning = (mode if block.type == "equation" else inline or mode) == "meaning"
    speech = " ".join(speech.split())
    filtered = block.speech if block.speech is not None else filter_text(block.text)
    if not speech:
        return "empty answer"
    if _MARKUP.search(speech):
        return "LaTeX or markup left in the answer"
    if not _figure_mentions(filtered) <= _figure_mentions(speech):
        return "a Figure/Table mention was lost"
    if by_meaning and vocabulary is not None:
        unknown = _ungrounded(speech, filtered, vocabulary)
        if len(unknown) > MAX_UNGROUNDED:
            return f"words not found in the paper: {', '.join(unknown[:5])}"
    if by_meaning and block.type != "equation":
        restated = _restated_names(speech) - _restated_names(filtered)
        if restated:
            return (f'says "{sorted(restated)[0]}" twice in the sentence that defines it: keep the symbol\'s '
                    f"letters in a defining sentence")
    if block.type == "equation":
        words = len(speech.split())
        if by_meaning:
            limit = MAX_EQUATION_WORDS_MEANING
        else:
            reference = equation_reading(block)
            coverage = symbol_coverage(reference, speech)
            if coverage < MIN_SYMBOL_COVERAGE:
                return f"equation not read in full (says {coverage:.0%} of its symbols)"
            limit = max(MAX_EQUATION_WORDS, round(EQUATION_LENGTH_FACTOR * len(reference.split())))
        return f"equation reading too long ({words} > {limit} words)" if words > limit else None
    before, after = _words(filtered), _words(speech)
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    kept = sum(match.size for match in matcher.get_matching_blocks())
    changed = len(before) + len(after) - 2 * kept
    math_words = sum(len(latex_to_speech(span).split()) for span in inline_math(block.text))
    citations = block.text.count("[CITE]")
    math_factor = MEANING_MATH_FACTOR if by_meaning else MATH_REWRITE_FACTOR
    allowance = (REWRITE_BUDGET * max(len(before) - math_words, 0) + math_factor * math_words
                 + CITATION_REPAIR_WORDS * citations + SLACK_WORDS)
    return None if changed <= allowance else f"rewrote too much ({changed} words changed, {allowance:.0f} allowed)"


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
    mode: EquationMode = "symbols"  # display equations
    inline: EquationMode = "symbols"  # symbols inside sentences
    vocabulary: set[str] | None = None  # meaning mode: words the paper uses (grounding check)
    requests: list[CleanupRequest] = field(default_factory=list)
    uses: dict[str, list[_SymbolName]] = field(default_factory=dict)  # block id -> names its reading used
    rejected: dict[str, str] = field(default_factory=dict)  # block id -> why the guard rejected its answer
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
                system=[{"type": "text", "text": system_prompt(self.mode, self.inline),
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": request.user_message}],
                output_config={"effort": EFFORT,
                               "format": {"type": "json_schema", "schema": response_schema(self.mode, self.inline)}},
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
                self.progress_cb(min(done / self.total, 1.0), f"Clean-up {done}/{self.total}: {label}")

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
        by_id: dict[str, _BlockSpeech] = {}
        for answer in answers:
            by_id.setdefault(answer.id, answer)
        missing = 0
        for block in request.blocks:
            answer = by_id.get(block.id)
            speech = " ".join(answer.speech.split()) if answer else None
            if speech is None:
                missing += 1
            elif (reason := guard_reason(block, speech, mode=self.mode, inline=self.inline,
                                         vocabulary=self.vocabulary)) is None:
                label_text = equation_label(block.eq_number) if block.type == "equation" else None
                if label_text and label_text not in speech:
                    speech = f"{label_text}: {speech}"
                block.speech, block.speech_source = speech, "llm"
                with self.lock:
                    self.uses[block.id] = answer.symbols
                    self.rejected.pop(block.id, None)
            elif block.speech_source == "llm":
                log.info("clean-up: %s keeps its earlier reading; the repair was rejected: %s", block.id, reason)
            else:
                block.speech_source = "llm_rejected"
                if block.type == "equation" and self.mode == "symbols":
                    # the filter says only "Equation 4." for a long equation: read it in full instead
                    block.speech = equation_reading(block) or block.speech
                with self.lock:
                    self.uses.pop(block.id, None)
                    self.rejected[block.id] = reason
        if missing:
            self._warn(f"{label}: {missing} block(s) missing from the answer; kept the filter text")

    def _warn(self, message: str) -> None:
        log.warning("clean-up: %s", message)
        with self.lock:
            self.paper.warnings.append(f"clean-up: {message}")


# ---------------------------------------------------------------- symbol consistency (meaning mode)

_NAME_PREFIX = re.compile(r"^(?:the|a|an|its|their|our|each)\s+")
# "action at time step t", "reward at each step t plus 1": the time index is not part of the name
_TIME_INDEX = re.compile(r"\s+(?:at|for|in)\s+(?:each\s+|every\s+|the\s+)?(?:time\s+)?(?:step|time)\s+[a-z]"
                         r"(?:\s+(?:plus|minus)\s+\w+)?$")


def normalize_name(name: str) -> str:
    """'The Imitation-Weight at time step t' -> 'imitation weight' (articles, case, hyphens, spacing
    and a trailing time index don't count)."""
    text = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return _TIME_INDEX.sub("", _NAME_PREFIX.sub("", text))


_NAME_FILLER = {"the", "and", "for", "with", "over", "given", "from", "into", "under", "its", "their", "per"}


def says_name(name: str, speech: str) -> bool:
    """Whether a reading says this name: all of its content words occur, in any order and word form.
    So "policy probability of the action given the state" is said by "the probability of the action
    given the state under the policy", and "imitation weight" by "the weight for imitation", while
    "the imitation coefficient" does not say "imitation weight": a different word means a different name."""
    spoken = {form for word in speech.split() for form in _word_forms(word)}
    words = [w for w in name.split() if len(w) > 2 and w not in _NAME_FILLER]
    return all(_word_forms(w) & spoken for w in words) if words else name in speech


@dataclass
class ConsistencyReport:
    names: dict[tuple, str]  # symbol key -> the canonical name (most used, ties: first in reading order)
    shown: dict[tuple, str]  # symbol key -> its LaTeX as first reported, for display and repair requests
    issues: list[tuple[str, str, str, str]]  # (block id, LaTeX, name used / problem, canonical name)

    def describe(self) -> list[str]:
        return [f"{latex}: {block_id} {used} instead of \"{name}\"" for block_id, latex, used, name in self.issues]


def check_consistency(uses: dict[str, list[_SymbolName]], speeches: dict[str, str],
                      order: list[str], equations: set[str] | None = None) -> ConsistencyReport:
    """The same symbol must be read the same way in every block. A block fails when it reported a
    different name for a symbol, or (display equations, whose reading is built from the symbols)
    when its speech doesn't actually say the canonical name. Prose keeps the paper's own words, so
    "the character state $s_t$" read as "the character state" is fine there.
    `order` is the reading order of block ids (it breaks ties and orders the issues);
    `equations` are the ids of display-equation blocks (default: every block is checked as one)."""
    position = {block_id: i for i, block_id in enumerate(order)}
    counts: dict[tuple, Counter] = {}
    first_use: dict[tuple, int] = {}  # (key, name) -> position of its first use
    shown: dict[tuple, tuple[int, str]] = {}  # key -> (position, LaTeX) of its first report
    for block_id, symbols in uses.items():
        at = position.get(block_id, len(position))
        for symbol in symbols:
            key, name = symbol_key(symbol.latex), normalize_name(symbol.name)
            if key is None or not name:
                continue
            counts.setdefault(key, Counter())[name] += 1
            first_use[key, name] = min(first_use.get((key, name), at), at)
            shown[key] = min(shown.get(key, (at, symbol.latex)), (at, symbol.latex))
    # the most used name wins; a tie goes to the name used first in reading order
    canonical = {key: max(names, key=lambda n: (names[n], -first_use[key, n])) for key, names in counts.items()}
    issues = []
    for block_id in sorted(uses, key=lambda b: position.get(b, 1 << 30)):
        spoken = re.sub(r"[^a-z0-9]+", " ", speeches.get(block_id, "").lower())
        for symbol in uses[block_id]:
            key = symbol_key(symbol.latex)
            if key not in canonical:
                continue
            name = canonical[key]
            if normalize_name(symbol.name) != name:
                issues.append((block_id, symbol.latex, f"said \"{normalize_name(symbol.name)}\"", name))
            elif (equations is None or block_id in equations) and not says_name(name, spoken):
                issues.append((block_id, symbol.latex, "doesn't say the name", name))
    return ConsistencyReport(names=canonical, shown={k: latex for k, (_, latex) in shown.items()}, issues=issues)


def consistency(run: "_Run") -> ConsistencyReport:
    blocks = list(run.paper.iter_blocks())
    return check_consistency(run.uses, {b.id: b.speech or "" for b in blocks}, [b.id for b in blocks],
                             equations={b.id for b in blocks if b.type == "equation"})


def repair_requests(run: "_Run") -> list[CleanupRequest]:
    """One more request per section for the blocks to fix: those that named a symbol differently from
    the rest of the paper (meaning mode; asked again with the canonical names) and those whose answer
    the guard rejected (asked again with the reason)."""
    meaning = "meaning" in (run.mode, run.inline)
    report = consistency(run) if meaning else None
    inconsistent = {block_id for block_id, *_ in report.issues} if report else set()
    flagged = inconsistent | set(run.rejected)
    if not flagged:
        return []
    definitions = DefinitionIndex(run.paper) if meaning else None
    requests = []
    for section in run.paper.sections:
        payloads, blocks = [], []
        for index, block in enumerate(section.blocks):
            if block.id not in flagged:
                continue
            payload = _block_payload(section, index, definitions, run.mode, run.inline)
            if report:
                latex = block.latex if block.type == "equation" else " ".join(inline_math(block.text))
                keys = {symbol_key(x.latex) for x in run.uses.get(block.id, [])}
                keys |= {atom_key(a) for a in math_atoms(latex or "")}
                payload["names"] = {report.shown[k]: report.names[k] for k in report.names if k in keys}
            if block.id in run.rejected:
                payload["problem"] = f"your previous answer was rejected: {run.rejected[block.id]}"
            payloads.append(payload)
            blocks.append(block)
        if blocks:
            heading = section.speech_title or section.title
            message = (f"Paper: {run.paper.title}\nSection: {heading}\n\nThese blocks are asked again. Where a "
                       f"block has \"names\", use exactly those names for those symbols (with \"the\" and the "
                       f"indices as usual). Where it has \"problem\", your previous answer was rejected for that "
                       f"reason: fix it, and follow the rules and examples exactly.\n"
                       f"Return the narration of these {len(blocks)} blocks:\n"
                       f"{json.dumps(payloads, ensure_ascii=False, indent=1)}")
            requests.append(CleanupRequest(section, blocks, message))
    with run.lock:
        run.stats.symbols_repaired += len(inconsistent)
    return requests


def finish(run: "_Run") -> CleanupStats:
    """Close a pass: count the blocks, warn about answers still rejected, and in meaning mode record
    the symbol names and any remaining inconsistency."""
    asked = {b.id for request in run.requests for b in request.blocks}
    run.stats.blocks_llm = sum(1 for b in run.paper.iter_blocks() if b.id in asked and b.speech_source == "llm")
    run.stats.blocks_rejected = len(run.rejected)
    for block_id, reason in run.rejected.items():
        run._warn(f"{block_id} kept the rule-based text: {reason}")
    if "meaning" in (run.mode, run.inline):
        report = consistency(run)
        run.stats.symbol_names = {report.shown[k]: name for k, name in report.names.items()}
        run.stats.inconsistent_symbols = report.describe()
        for line in run.stats.inconsistent_symbols:
            run._warn(f"symbol read inconsistently: {line}")
    run.stats.cost_usd = round(run.stats.cost_usd, 6)
    run.paper.cleanup = run.stats
    return run.stats


def begin_import(paper: Paper, *, model: str, mode: EquationMode = "symbols",
                 inline: EquationMode | None = None) -> "_Run":
    """A pass whose answers come from outside (Codex, subagents): filter text reset, requests planned."""
    apply_filter(paper)
    paper.warnings = [w for w in paper.warnings if not w.startswith("clean-up:")]
    requests = plan_requests(paper, mode, inline)
    return _new_run(paper, None, model, requests, None, mode, inline)


def apply_answers(run: "_Run", requests: list[CleanupRequest], answers: list[dict]) -> None:
    """External answers ([{"id", "speech", "symbols"?}], any order) through the same guard."""
    try:
        parsed = [_BlockSpeech.model_validate(a) for a in answers]
    except ValidationError as exc:
        raise ValueError(f"answers are not [{{id, speech}}] ({exc.error_count()} errors)") from exc
    for request in requests:
        ids = {b.id for b in request.blocks}
        run._apply(request, [a for a in parsed if a.id in ids], request.section.speech_title or request.section.id)


def request_dict(request: CleanupRequest, request_id: int) -> dict:
    return {"request_id": request_id, "section": request.section.speech_title or request.section.id,
            "block_ids": [b.id for b in request.blocks], "user_message": request.user_message}


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


def _new_run(paper: Paper, client, model: str, requests: list[CleanupRequest], progress_cb,
             mode: EquationMode, inline: EquationMode | None = None) -> _Run:
    inline = inline or mode
    for value in (mode, inline):
        if value not in EQUATION_MODES:
            raise ValueError(f"unknown reading {value!r}; choose from {', '.join(EQUATION_MODES)}")
    return _Run(paper, client, model, max(len(requests), 1), progress_cb,
                CleanupStats(model=model, equation_mode=mode, inline_mode=inline), mode=mode, inline=inline,
                vocabulary=paper_vocabulary(paper) if "meaning" in (mode, inline) else None, requests=requests)


def export_requests(paper: Paper, parts: int = 1, mode: EquationMode = "symbols",
                    inline: EquationMode | None = None) -> list[list[dict]]:
    """The exact requests the API pass would send, in `parts` groups balanced by size, so another
    runner (e.g. Claude Code subagents on Haiku) can answer them with system_prompt(mode) as
    instructions. Each request: {request_id, section, block_ids, user_message}."""
    apply_filter(paper)
    requests = [request_dict(r, i) for i, r in enumerate(plan_requests(paper, mode, inline))]
    groups: list[list[dict]] = [[] for _ in range(max(1, min(parts, len(requests))))]
    sizes = [0] * len(groups)
    for request in sorted(requests, key=lambda r: -len(r["user_message"])):
        smallest = sizes.index(min(sizes))
        groups[smallest].append(request)
        sizes[smallest] += len(request["user_message"])
    return [sorted(group, key=lambda r: r["request_id"]) for group in groups]


def import_answers(paper: Paper, answers: list[dict], *, model: str, mode: EquationMode = "symbols",
                   inline: EquationMode | None = None) -> CleanupStats:
    """Apply externally produced answers ([{"id", "speech"}], any order) through the same guard
    as the API pass. Re-imports are idempotent: speech is reset to the filter text first."""
    run = begin_import(paper, model=model, mode=mode, inline=inline)
    apply_answers(run, run.requests, answers)
    return finish(run)


def cleanup_paper(paper: Paper, *, client=None, model: str = MODEL, max_workers: int = 4,
                  progress_cb: Callable[[float, str], None] | None = None,
                  equations: EquationMode = "symbols", inline: EquationMode | None = None) -> CleanupStats:
    """Run the Haiku 5.5 clean-up over a filtered paper. Mutates blocks and sets paper.cleanup.
    equations="meaning" reads display equations by what their symbols mean, inline="meaning" the
    symbols inside sentences (None = like equations); see _meaning_rules and definitions.py.

    client: an anthropic.Anthropic (or a test double); None builds one from the environment, or
    skips the pass with skipped_reason when no credentials are available.
    progress_cb(fraction, message) is called after every request."""
    if client is None:
        ok, reason = llm_available()
        if not ok:
            paper.cleanup = CleanupStats(model=model, skipped_reason=reason, equation_mode=equations,
                                         inline_mode=inline or equations)
            return paper.cleanup
        client = anthropic.Anthropic()
    if any(b.speak and b.speech is None for b in paper.iter_blocks()):
        apply_filter(paper)

    requests = plan_requests(paper, equations, inline)
    run = _new_run(paper, client, model, requests, progress_cb, equations, inline)
    if requests:
        run.execute(requests[0])  # writes the prompt cache; parallel requests could not read it yet
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(run.execute, requests[1:]))
    if not run.stopped.is_set() and (repairs := repair_requests(run)):  # inconsistent names, rejected answers
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(run.execute, repairs))
    return finish(run)
