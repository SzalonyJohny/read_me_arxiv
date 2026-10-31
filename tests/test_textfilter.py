"""Table-driven tests for the rule-based speech filter."""

from __future__ import annotations

import random
import re

import pytest

from paper_reader.models import Block, Paper, Section
from paper_reader.textfilter import (
    SpeechUnit,
    apply_filter,
    equation_speech,
    filter_text,
    latex_to_speech,
    section_title_speech,
    speech_units,
)

# Every row of the rules table in plan.md.
PLAN_TABLE = [
    ("The controller is robust [3, 7–9].", "The controller is robust."),
    ("The controller is robust [CITE].", "The controller is robust."),
    ("This effect was first observed as shown in [12].", "This effect was first observed as shown in earlier work."),
    ("Smith et al. [7] showed that gaits emerge.", "Smith and colleagues showed that gaits emerge."),
    ("Smith et al. [CITE] showed that gaits emerge.", "Smith and colleagues showed that gaits emerge."),
    ("See Fig. 3(b), Tab. II, Eq. (4) and Sec. 3.2.", "See Figure 3 b, Table 2, Equation 4 and Section 3.2."),
    ("We tune it, e.g., the gains, i.e., the PD terms, w.r.t. SOTA.",
     "We tune it, for example, the gains, that is, the PD terms, with respect to state of the art."),
    ("It takes 3.2 ms with a rate of 1e-4, an error of ±0.3 over ∼40 runs.",
     "It takes 3.2 milliseconds with a rate of ten to the minus four, an error of plus or minus 0.3 over about 40 runs."),
    (r"We use $\alpha = 0.5$ throughout.", "We use alpha equals 0.5 throughout."),
    ("Code is available at https://github.com/org/repo.¹", "Code is available online."),
    ("The dataset (www.example.org/data) is public.", "The dataset is public."),
]

CITATIONS = [
    ("[CITE] proposed a diffusion policy.", "Earlier work proposed a diffusion policy."),
    ("[CITE] propose a diffusion policy.", "Earlier studies propose a diffusion policy."),
    ("[CITE] also uses a curriculum.", "Earlier work also uses a curriculum."),
    ("[CITE], [CITE] are popular. [CITE] were first.", "Earlier studies are popular. Earlier studies were first."),
    ("In contrast, [CITE] train one model.", "In contrast, earlier studies train one model."),
    ("Such methods, [CITE] and others, work.", "Such methods, and others, work."),
    ("This is robust (see [CITE]).", "This is robust."),
    ("This is robust (see, e.g., [CITE], [CITE]).", "This is robust."),
    ("This is robust (Smith et al. [CITE]; Lee et al. [CITE]).", "This is robust."),
    ("We use PPO ([CITE]) for training.", "We use PPO for training."),
    ("We use proximal policy optimization (PPO [CITE]).", "We use proximal policy optimization (PPO)."),
    ("Several works, e.g., [CITE], use vision.", "Several works use vision."),
    ("The approach of [CITE] fails.", "The approach of earlier work fails."),
    ("Following [CITE], we normalize rewards.", "Following earlier work, we normalize rewards."),
    ("[CITE]'s method fails on stairs.", "Earlier work's method fails on stairs."),
    ("Rudin et al.'s terrain curriculum works.", "Rudin and colleagues' terrain curriculum works."),
    ("This follows Lee et al. [CITE]. We extend it.", "This follows Lee and colleagues. We extend it."),
    ("This was shown in [1] and [2].", "This was shown in earlier work."),
    ("A missing reference [?] here.", "A missing reference here."),
]

REFERENCES = [
    ("In fig. 2 and Figs. 4–6 we compare.", "In Figure 2 and Figures 4 to 6 we compare."),
    ("Figure 3a shows the gait.", "Figure 3 a shows the gait."),
    ("Fig. 3(a)–(c) shows it.", "Figure 3 a to c shows it."),
    ("Fig. 3(a) and (c) show it.", "Figure 3 a and c show it."),
    ("Fig.~5 shows it.", "Figure 5 shows it."),
    ("As shown in table 4, we win.", "As shown in Table 4, we win."),
    ("Eqs. (4) and (5) define it.", "Equations 4 and 5 define it."),
    ("Eqs. (3)–(5) define it.", "Equations 3 to 5 define it."),
    ("Consistent with Sec. IV-C, it works.", "Consistent with Section 4 C, it works."),
    ("Table III lists the rewards.", "Table 3 lists the rewards."),
    ("See Alg. 1 and App. B.", "See Algorithm 1 and Appendix B."),
    ("See §3.2 for details.", "See Section 3.2 for details."),
    ("By Thm. 2 the bound holds.", "By Theorem 2 the bound holds."),
]

NUMBERS_AND_UNITS = [
    ("A latency of 1 ms and 0.5 s.", "A latency of 1 millisecond and 0.5 seconds."),
    ("It runs at 50 Hz and 1 kHz.", "It runs at 50 hertz and 1 kilohertz."),
    ("It needs 24 GB and 30 fps.", "It needs 24 gigabytes and 30 frames per second."),
    ("Speeds of 3.0 m/s and 2 rad/s.", "Speeds of 3.0 meters per second and 2 radians per second."),
    ("A 90° turn at 20 °C.", "A 90 degrees turn at 20 degrees Celsius."),
    ("Success rose to 95% (from 80 %).", "Success rose to 95 percent (from 80 percent)."),
    ("A gain of 3e-5 and 2.5E3.", "A gain of 3 times ten to the minus five and 2.5 times ten to the three."),
    ("Use 3–5 layers, 10-20 epochs, 2020–2023.", "Use 3 to 5 layers, 10 to 20 epochs, 2020 to 2023."),
    ("Inputs are 224×224 images, 3× faster, 10x cheaper.",
     "Inputs are 224 by 224 images, 3 times faster, 10 times cheaper."),
    ("Values range from -1 to −0.5.", "Values range from minus 1 to minus 0.5."),
    ("It uses ≤5 ms and ≥ 90% accuracy.", "It uses at most 5 milliseconds and at least 90 percent accuracy."),
    ("We require x ≤ 5 and > 10 steps.", "We require x less than or equal to 5 and more than 10 steps."),
    ("A model with 7B parameters and 100K steps.", "A model with 7 billion parameters and 100 thousand steps."),
    ("An arm with a 7-DoF design.", "An arm with a 7 degree of freedom design."),
    ("Accuracy ≈ 0.9 and sim → real.", "Accuracy approximately 0.9 and sim to real."),
    ("It runs in 2 h on 8 GPUs and walks 5 m.", "It runs in 2 hours on 8 GPUs and walks 5 meters."),
]

ABBREVIATIONS = [
    ("Speed vs. accuracy vs cost.", "Speed versus accuracy versus cost."),
    ("Stairs, slopes, etc. are hard.", "Stairs, slopes, et cetera are hard."),
    ("Stairs, slopes, etc.", "Stairs, slopes, et cetera."),
    ("This takes approx. 3 days (resp. 4 days).", "This takes approximately 3 days (respectively 4 days)."),
    ("Samples are i.i.d. and w/o noise.", "Samples are IID and without noise."),
    ("Gains are tuned by hand (cf. Figure 2).", "Gains are tuned by hand (compare Figure 2)."),
    ("A SoTA result.", "A state of the art result."),
]

INLINE_MATH = [
    (r"The window $H=50$ steps.", "The window H equals 50 steps."),
    (r"We reach $\sim$40 FPS.", "We reach about 40 frames per second."),
    (r"Speed is $3.0~\mathrm{m/s}$ here.", "Speed is 3.0 meters per second here."),
    (r"A grid of $2.4\,\mathrm{m}$ by $1.0\,\mathrm{m}$.", "A grid of 2.4 meters by 1.0 meters."),
    (r"Heights within $0.02-0.10$m.", "Heights within 0.02 to 0.10 meters."),
    (r"Along the $x$-axis.", "Along the x-axis."),
    (r"Keep $\gamma\in[0,1]$ fixed [CITE].", "Keep gamma in the interval from 0 to 1 fixed."),
    (r"A $3\times$ speed-up.", "A 3 times speed-up."),
    (r"Error $\pm 0.3$ m.", "Error plus or minus 0.3 meters."),
    (r"The 1$^{st}$ stage.", "The 1st stage."),
    (r"It takes 5 $\mu$s.", "It takes 5 microseconds."),
    (r"Below $30\%$.", "Below 30 percent."),
    (r"The policy $\pi_{\theta}(a\mid s)$ acts.", "The policy pi theta of a given s acts."),
]

# Twenty realistic robotics / ML sentences, read aloud.
REALISTIC = [
    ("We train the policy with PPO [CITE] for 2B environment steps on 4096 parallel robots.",
     "We train the policy with PPO for 2 billion environment steps on 4096 parallel robots."),
    (r"The learning rate is set to $3 \times 10^{-4}$ and decayed by $0.99$ every epoch.",
     "The learning rate is set to 3 times ten to the minus four and decayed by 0.99 every epoch."),
    ("Kumar et al. [CITE] distill a teacher into a student that only sees proprioception.",
     "Kumar and colleagues distill a teacher into a student that only sees proprioception."),
    ("As shown in Fig. 4, the success rate drops below 60% on stairs higher than 20 cm.",
     "As shown in Figure 4, the success rate drops below 60 percent on stairs higher than 20 centimeters."),
    ("The control loop runs at 50 Hz, while the low-level PD controller runs at 400 Hz.",
     "The control loop runs at 50 hertz, while the low-level PD controller runs at 400 hertz."),
    (r"The reward is $r_t = -\|\mathbf{v}_t - \mathbf{v}^*\|^2$, i.e., a velocity tracking term.",
     "The reward is r t equals minus the squared norm of v t minus v star, that is, a velocity tracking term."),
    ("Inference on the Jetson Orin takes 4.1 ms per step (Tab. 3).",
     "Inference on the Jetson Orin takes 4.1 milliseconds per step (Table 3)."),
    ("Unlike [CITE], we do not require motion capture during deployment.",
     "Unlike earlier work, we do not require motion capture during deployment."),
    (r"We sample the friction coefficient $\mu \sim \mathcal{U}(0.2, 1.2)$ at every reset.",
     "We sample the friction coefficient mu is distributed as U of 0.2 and 1.2 at every reset."),
    (r"The latent $z \in \mathbb{R}^{32}$ is decoded by an MLP with $[512, 256, 128]$ units.",
     "The latent z in R to the 32 is decoded by an MLP with 512, 256, 128 units."),
    ("Our method improves the success rate by ∼15% over the SOTA baseline [CITE, CITE].",
     "Our method improves the success rate by about 15 percent over the state of the art baseline."),
    (r"We use a discount factor $\gamma = 0.99$ and GAE with $\lambda = 0.95$ [CITE].",
     "We use a discount factor gamma equals 0.99 and GAE with lambda equals 0.95."),
    ("Training takes approx. 8 h on a single RTX 4090 with 24 GB of memory.",
     "Training takes approximately 8 hours on a single RTX 4090 with 24 gigabytes of memory."),
    ("Images are resized to 128×128 and stacked over 3 frames, see Sec. 3.3.",
     "Images are resized to 128 by 128 and stacked over 3 frames, see Section 3.3."),
    (r"The policy outputs joint targets $q^{\mathrm{des}}_t \in \mathbb{R}^{12}$ for the 12 actuators.",
     "The policy outputs joint targets q des t in R to the 12 for the 12 actuators."),
    ("[CITE] first showed that large-scale domain randomization enables sim-to-real transfer.",
     "Earlier work first showed that large-scale domain randomization enables sim-to-real transfer."),
    (r"We optimize $\mathcal{L} = \mathcal{L}_{\text{BC}} + \beta\,\mathcal{L}_{\text{KL}}$ with $\beta = 10^{-3}$.",
     "We optimize L equals L BC plus beta L KL with beta equals ten to the minus three."),
    ("The arm reaches the target within ±2 mm in 95% of the 200 trials (Fig. 6(b)).",
     "The arm reaches the target within plus or minus 2 millimeters in 95 percent of the 200 trials (Figure 6 b)."),
    ("Each episode lasts 20 s, i.e., 1000 control steps at 50 Hz.",
     "Each episode lasts 20 seconds, that is, 1000 control steps at 50 hertz."),
    (r"The gradient $\nabla_\theta J(\theta)$ is estimated from $N = 64$ rollouts, cf. Eq. (3).",
     "The gradient the gradient with respect to theta of J of theta is estimated from N equals 64 rollouts, "
     "compare Equation 3."),
]

PLAIN = [
    "We propose a simple method for learning robust locomotion policies.",
    "The robot walks on stairs, slopes and gravel without any terrain-specific tuning.",
    "Our results show that curriculum design matters more than network size.",
    "However, the student policy fails when the height map is noisy.",
    "In contrast to prior work, we train a single policy for all tasks.",
    "Table 2 summarizes the results, and Figure 3 shows the learning curves.",
    "The agent’s performance improves steadily during the first 1000 iterations.",
    "We release the code, the trained models and the evaluation scripts.",
    "Section 4 describes the hardware, the training setup and the baselines.",
    "A 12-dimensional action vector controls the joints; each entry is a position target.",
    "The teacher has access to privileged information such as friction and terrain heights.",
    "Results are averaged over five random seeds... and the variance is small.",
    "The [MASK] token is replaced by a learned embedding.",
    "We compare LP-ACRL with ALP-GMM, PLR and a hand-crafted curriculum.",
    "This is the reason why the ResNet-50 encoder is frozen during fine-tuning.",
]


@pytest.mark.parametrize(
    ("source", "expected"),
    PLAN_TABLE + CITATIONS + REFERENCES + NUMBERS_AND_UNITS + ABBREVIATIONS + INLINE_MATH + REALISTIC,
)
def test_filter_text(source, expected):
    assert filter_text(source) == expected


@pytest.mark.parametrize("sentence", PLAIN)
def test_plain_sentences_pass_through(sentence):
    assert filter_text(sentence) == sentence


@pytest.mark.parametrize("source", [s for s, _ in PLAN_TABLE + CITATIONS + REALISTIC + INLINE_MATH])
def test_no_markup_survives(source):
    speech = filter_text(source)
    assert not re.search(r"[\\${}]|\[CITE\]|\[\d", speech), speech


def test_figure_mentions_are_canonical():
    speech = filter_text("Fig. 2 and fig. 7(a), Tab. IV and table 5; Figs. 3 and 4.")
    assert re.findall(r"\b(Figure|Table)s? (\d+)", speech) == [
        ("Figure", "2"), ("Figure", "7"), ("Table", "4"), ("Table", "5"), ("Figure", "3")]


LATEX = [
    (r"\alpha = 0.5", "alpha equals 0.5"),
    (r"\varepsilon", "epsilon"),
    (r"\Delta t", "Delta t"),
    (r"x^2 + y^3", "x squared plus y cubed"),
    (r"A^\top B", "A transpose B"),
    (r"W^{T}", "W transpose"),
    (r"\Sigma^{-1}", "Sigma inverse"),
    (r"x^n", "x to the n"),
    (r"e^{-x}", "e to the minus x"),
    (r"x_t", "x t"),
    (r"x_{t+1}", "x sub t plus 1"),
    (r"\theta_{\text{old}}", "theta old"),
    (r"x_{1:T}", "x sub 1 to T"),
    (r"\hat{x}_t", "x hat t"),
    (r"\bar{r}", "r bar"),
    (r"\tilde{\mathbf{v}}", "v tilde"),
    (r"\dot{q}", "q dot"),
    (r"\mathbf{x}^{*}", "x star"),
    (r"f'(x)", "f prime of x"),
    (r"\frac{1}{2}", "one half"),
    (r"\frac{a}{b}", "a over b"),
    (r"\frac{a+b}{c}", "a plus b, all over c"),
    (r"\frac{1}{1+e^{-x}}", "1 over the quantity 1 plus e to the minus x"),
    (r"\frac{\partial L}{\partial \theta}", "the partial derivative of L with respect to theta"),
    (r"\frac{dy}{dx}", "the derivative of y with respect to x"),
    (r"\sqrt{x^2 + y^2}", "the square root of x squared plus y squared"),
    (r"\sqrt[3]{x}", "the cube root of x"),
    (r"\sum_{i=1}^{N} x_i", "the sum from i equals 1 to N of x i"),
    (r"\sum_i w_i", "the sum over i of w i"),
    (r"\prod_{k} p_k", "the product over k of p k"),
    (r"\int_0^1 f(x)\,dx", "the integral from 0 to 1 of f of x d x"),
    (r"\lim_{n \to \infty} a_n", "the limit as n goes to infinity of a n"),
    (r"x \leq y", "x less than or equal to y"),
    (r"x \geq 0", "x greater than or equal to 0"),
    (r"a \approx b", "a is approximately b"),
    (r"x \sim \mathcal{N}(0, I)", "x is distributed as a Gaussian with mean 0 and covariance I"),
    (r"\epsilon \sim \mathcal{N}(0, \sigma^2)", "epsilon is distributed as a Gaussian with mean 0 and variance sigma squared"),
    (r"x \in \mathbb{R}^n", "x in R to the n"),
    (r"a \cdot b", "a times b"),
    (r"a \times b", "a times b"),
    (r"\mathbb{E}[X]", "the expectation of X"),
    (r"\mathbb{E}_{x \sim p}[f(x)]", "the expectation over x drawn from p of f of x"),
    (r"\nabla_\theta J", "the gradient with respect to theta of J"),
    (r"\log p(x \mid z)", "log p of x given z"),
    (r"\exp(-x)", "exp of minus x"),
    (r"\min_x f(x)", "min over x of f of x"),
    (r"\max_{\theta} J(\theta)", "max over theta of J of theta"),
    (r"\arg\max_a Q(s, a)", "arg max over a of Q of s and a"),
    (r"\operatorname*{arg\,max}_{a} Q(s,a)", "arg max over a of Q of s and a"),
    (r"\left( a + b \right)^2", "the quantity a plus b, squared"),
    (r"|x|", "the absolute value of x"),
    (r"|\mathcal{D}|", "the size of D"),
    (r"\|x\|_2", "the L2 norm of x"),
    (r"\lVert x \rVert^2", "the squared norm of x"),
    (r"10^{-4}", "ten to the minus four"),
    (r"1 \times 10^{-4}", "ten to the minus four"),
    (r"2.5 \times 10^{6}", "2.5 times ten to the six"),
    (r"[0, 1]", "the interval from 0 to 1"),
    (r"(0, 1]", "the interval from 0 to 1"),
    (r"x_1, \dots, x_n", "x 1 up to x n"),
    (r"k = 0, 1, 2, \ldots", "k equals 0, 1, 2, and so on"),
    (r"a_1 + \cdots + a_n", "a 1 plus dot dot dot plus a n"),
    (r"D_{\mathrm{KL}}(q \,\|\, p)", "the KL divergence between q and p"),
    (r"f: \mathcal{X} \to \mathcal{Y}", "f maps X to Y"),
    (r"\{\tau_i\}_{i=1}^{N}", "the set of tau i for i equals 1 to N"),
    (r"\binom{n}{k}", "n choose k"),
    (r"\mathcal{O}(n \log n)", "big O of n log n"),
    (r"\sim", "about"),
    (r"> 90\%", "more than 90 percent"),
    (r"\pm 0.3", "plus or minus 0.3"),
    (r"\theta \leftarrow \theta - \eta \nabla_\theta L", "theta gets theta minus eta the gradient with respect to theta of L"),
    (r"\mathrm{softmax}(z)", "softmax of z"),
    (r"softmax(z)", "softmax of z"),
    (r"\langle u, v \rangle", "the inner product of u and v"),
    (r"x := y", "x is defined as y"),
    (r"x^{(i)}", "x i"),
    (r"\text{if } x > 0", "if x greater than 0"),
    (r"f(x) = \begin{cases} x & x > 0 \\ 0 & \text{otherwise} \end{cases}",
     "f of x equals x if x greater than 0, 0 otherwise"),
]


@pytest.mark.parametrize(("latex", "expected"), LATEX)
def test_latex_to_speech(latex, expected):
    assert latex_to_speech(latex) == expected


@pytest.mark.parametrize("latex", [
    r"\mycustommacro{x}_{\foo}", r"\frac{a}{", r"x^{", r"\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}",
    r"}}{{", r"\left\{ \middle| \right.", r"\color{red}{x}", r"\\ & \\",
])
def test_latex_fallback_never_emits_markup(latex):
    speech = latex_to_speech(latex)
    assert not re.search(r"[\\{}$^_&]", speech), speech


def _equation(latex, number=None):
    return Block(id="b1", type="equation", latex=latex, eq_number=number)


@pytest.mark.parametrize(("latex", "number", "expected"), [
    (r"\theta_{k+1}=\mathcal{G}\left(\theta_{k},\mathcal{D}_{k}\right).", "4",
     "Equation 4: theta sub k plus 1 equals G of theta k and D k."),
    (r"a_t = \pi_\theta(o_t),", None, "a t equals pi theta of o t."),
    (r"c_{j+1}(\zeta)=\frac{e^{LP_{c_{j}}(\zeta)/\beta}}{\sum_{\zeta^{\prime}\in\mathcal{T}}e^{LP_{c_{j}}(\zeta^{\prime})/\beta}}.",
     "7", "Equation 7."),
    (r"\begin{aligned} L &= a + b \\ &= c \end{aligned}", "2", "Equation 2."),
    (r"\begin{aligned} L &= a + b \\ &= c \end{aligned}", None, "An equation follows."),
    (r"\mycustom{x} = 1", "3", "Equation 3."),
    (r"x = y", "6, 7", "Equations 6 and 7: x equals y."),
    (r"x = y", "∗", "x equals y."),
    (r"x = y", "(5)", "Equation 5: x equals y."),
])
def test_equation_speech(latex, number, expected):
    assert equation_speech(_equation(latex, number)) == expected


@pytest.mark.parametrize(("section", "expected"), [
    (Section(id="abstract", title="Abstract", kind="abstract"), "Abstract."),
    (Section(id="S3.SS1", number="3.1", title="Policy learning", level=2), "Section 3.1. Policy learning."),
    (Section(id="S1", number="I", title="Introduction"), "Section 1. Introduction."),
    (Section(id="S3.SS1.SSS1", number="III-A1", title="Definition of Curriculum", level=3),
     "Section 3 A1. Definition of Curriculum."),
    (Section(id="A1", number="A", title="Proofs", kind="appendix"), "Appendix A. Proofs."),
    (Section(id="S5", number="5", title=r"What is the Optimal Rank $r$ for LoRA?"),
     "Section 5. What is the Optimal Rank r for LoRA?"),
    (Section(id="S6", title="Related work"), "Related work."),
])
def test_section_title_speech(section, expected):
    assert section_title_speech(section) == expected


def _paper() -> Paper:
    blocks = [
        Block(id="b0001", type="paragraph", text="We walk [CITE]. Fig. 2 shows terrain."),
        Block(id="b0002", type="figure", figure_id="S1.F2", speak=False),
        Block(id="b0003", type="equation", latex=r"x = y", eq_number="1"),
        Block(id="b0004", type="heading_note", text="Implementation details."),
        Block(id="b0005", type="paragraph", text="[CITE]"),
    ]
    return Paper(paper_id="t.1v1", arxiv_id="t.1", title="T", source_url="x",
                 sections=[Section(id="S1", number="1", title="Intro", blocks=blocks),
                           Section(id="S2", number="2", title="Empty")])


def test_apply_filter_and_speech_units():
    paper = _paper()
    apply_filter(paper)
    blocks = {b.id: b for b in paper.iter_blocks()}
    assert blocks["b0001"].speech == "We walk. Figure 2 shows terrain."
    assert blocks["b0001"].speech_source == "filter"
    assert blocks["b0002"].speech is None and blocks["b0002"].speech_source is None
    assert blocks["b0003"].speech == "Equation 1: x equals y."
    assert paper.sections[0].speech_title == "Section 1. Intro."
    assert speech_units(paper) == [
        SpeechUnit("S1", "section_title", "Section 1. Intro."),
        SpeechUnit("b0001", "block", "We walk. Figure 2 shows terrain."),
        SpeechUnit("b0003", "block", "Equation 1: x equals y."),
        SpeechUnit("b0004", "block", "Implementation details."),
        SpeechUnit("S2", "section_title", "Section 2. Empty."),
    ]


def test_speech_units_fall_back_to_the_filter():
    paper = _paper()
    assert [u.unit_id for u in speech_units(paper)] == ["S1", "b0001", "b0003", "b0004", "S2"]
    assert all(b.speech is None for b in paper.iter_blocks())  # not mutated


def test_latex_fuzz_never_crashes_or_emits_markup():
    tokens = [r"\frac", r"\sqrt", "{", "}", "(", ")", "[", "]", "^", "_", "|", r"\|", r"\left", r"\right", ".",
              ",", "x", "2", r"\sum", r"\mathbb", r"\begin{cases}", r"\end{cases}", "&", r"\\", r"\not", r"\hat",
              r"\text", "=", r"\sim", "-", "'", r"\{", r"\}", r"\langle", r"\dots", r"\arg", r"\max", r"\foo",
              r"\underset", r"\binom", ":", r"\middle", "\\", "~"]
    rng = random.Random(7)
    for _ in range(2000):
        latex = " ".join(rng.choice(tokens) for _ in range(rng.randint(1, 12)))
        speech = latex_to_speech(latex)
        assert not re.search(r"[\\{}$^_&]", speech), (latex, speech)
        assert "[CITE]" not in filter_text(f"Text ${latex}$ more [CITE].")
