"""Rule-based speech text: the free, always-on first clean-up pass.

    filter_text(text)              Block.text (plain text, $LaTeX$, [CITE]) -> speakable text
    latex_to_speech(latex)         inline maths -> words ("\\alpha = 0.5" -> "alpha equals 0.5")
    equation_speech(block)         display equation -> "Equation 4: ..." or "Equation 4."
    section_title_speech(section)  -> "Section 3.1. Policy learning."
    apply_filter(paper)            fills Block.speech / Section.speech_title for a whole paper
    speech_units(paper)            ordered list of everything that will be spoken (pricing + TTS)

Figure and table mentions always come out as "Figure N" / "Table N": the pipeline finds them
in the narration with the regex  \\b(Figure|Table)s? (\\d+)  to switch the right-hand pane.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .models import Block, Paper, Section

# ============================================================================ shared tables

_GREEK_NAMES = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
    "sigma tau upsilon phi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega"
).split()
_GREEK = {name: name for name in _GREEK_NAMES} | {
    "varepsilon": "epsilon", "vartheta": "theta", "varpi": "pi", "varrho": "rho",
    "varsigma": "sigma", "varphi": "phi", "varkappa": "kappa", "digamma": "digamma",
}
_GREEK_CHARS = dict(zip(
    "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩϵϑϕϱςϖµ",
    _GREEK_NAMES + ["epsilon", "theta", "phi", "rho", "sigma", "pi", "mu"],
))

_SMALL_NUMBERS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty"
).split()


def _number_words(value: int) -> str:
    """Small integers as words ("minus four"), larger ones as digits."""
    if value < 0:
        return "minus " + _number_words(-value)
    return _SMALL_NUMBERS[value] if value <= 20 else str(value)


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_to_int(numeral: str) -> int:
    total = 0
    for char, following in zip(numeral, numeral[1:] + " "):
        value = _ROMAN[char]
        total += -value if _ROMAN.get(following, 0) > value else value
    return total


# ============================================================================ LaTeX -> words
#
# A small recursive-descent parser builds a tree of the nodes below, and a renderer reads the
# tree aloud.  Anything unknown degrades gracefully (commands and braces are stripped) and
# clears `_Parser.clean`, which equation_speech() uses to decide whether to read an equation.


@dataclass
class _Sym:
    words: str
    kind: str  # var num func text op rel punct bigop dots sep
    prose: str | None = None  # relations without a left operand: "$\sim$40" -> "about 40"
    tag: str | None = None  # bigop kind (sum, opt, lim, expect, prob, indicator, grad) / "gauss"


@dataclass
class _Group:
    open: str  # "{" (invisible), "(", "[", "set", "abs", "norm", "angle", "floor", "ceil"
    body: list
    close: str | None = None


@dataclass
class _Script:
    base: object | None
    sub: list | None = None
    sup: list | None = None


@dataclass
class _Frac:
    num: list
    den: list
    binom: bool = False


@dataclass
class _Root:
    index: list | None
    body: list


@dataclass
class _Accent:
    name: str
    body: list


@dataclass
class _Env:
    name: str
    rows: list  # list of rows, each a list of cells (node lists)


_UNICODE_MATH = {char: rf"\{name} " for char, name in _GREEK_CHARS.items()} | {
    "≤": r"\leq ", "≥": r"\geq ", "≈": r"\approx ", "∈": r"\in ", "∉": r"\notin ", "∼": r"\sim ",
    "→": r"\to ", "←": r"\leftarrow ", "⇒": r"\Rightarrow ", "⇔": r"\Leftrightarrow ", "×": r"\times ",
    "·": r"\cdot ", "⋅": r"\cdot ", "±": r"\pm ", "∓": r"\mp ", "∞": r"\infty ", "∑": r"\sum ",
    "∏": r"\prod ", "∫": r"\int ", "∂": r"\partial ", "∇": r"\nabla ", "≠": r"\neq ", "≡": r"\equiv ",
    "∝": r"\propto ", "⊂": r"\subset ", "⊆": r"\subseteq ", "∪": r"\cup ", "∩": r"\cap ",
    "∀": r"\forall ", "∃": r"\exists ", "¬": r"\neg ", "∧": r"\wedge ", "∨": r"\vee ", "′": "'",
    "″": "''", "‖": r"\| ", "⟨": r"\langle ", "⟩": r"\rangle ", "ℝ": r"\mathbb{R}", "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}", "𝔼": r"\mathbb{E}", "ℓ": r"\ell ", "−": "-", "…": r"\dots ", "⋯": r"\cdots ",
    "∘": r"\circ ", "°": r"^\circ ", "⊤": r"\top ", "∗": "*", "≪": r"\ll ", "≫": r"\gg ",
    "↦": r"\mapsto ", "∅": r"\emptyset ", "⌊": r"\lfloor ", "⌋": r"\rfloor ", "⌈": r"\lceil ",
    "⌉": r"\rceil ", "⊙": r"\odot ", "⊗": r"\otimes ", "⊕": r"\oplus ", "≜": r"\triangleq ",
    "≔": r"\coloneqq ", "≃": r"\simeq ", "≅": r"\cong ", "∣": r"\mid ", "²": "^{2}", "³": "^{3}",
    "\u00a0": " ", "\u2009": " ", "\u202f": " ",
}

_TOKEN_RE = re.compile(r"\\[A-Za-z]+\*?|\\.|\d*\.\d+|\d+|\s+|.", re.S)

# Sizing commands vanish; the delimiter that follows them stays ("." is the invisible one).
_SIZING = {
    r"\left", r"\right", r"\middle", r"\big", r"\Big", r"\bigg", r"\Bigg", r"\bigl", r"\bigr",
    r"\Bigl", r"\Bigr", r"\biggl", r"\biggr", r"\Biggl", r"\Biggr", r"\bigm", r"\Bigm",
}
_TOKEN_ALIASES = {
    r"\lVert": r"\|", r"\rVert": r"\|", r"\Vert": r"\|", r"\lvert": "|", r"\rvert": "|",
    r"\vert": "|", r"\lbrace": r"\{", r"\rbrace": r"\}", r"\lbrack": "[", r"\rbrack": "]",
    r"\colon": ":", r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\cfrac": r"\frac",
    r"\dbinom": r"\binom", r"\tbinom": r"\binom", r"\le": r"\leq", r"\ge": r"\geq",
    r"\leqslant": r"\leq", r"\geqslant": r"\geq", r"\ne": r"\neq", r"\gets": r"\leftarrow",
    r"\rightarrow": r"\to", r"\longrightarrow": r"\to", r"\implies": r"\Rightarrow",
    r"\Longrightarrow": r"\Rightarrow", r"\iff": r"\Leftrightarrow",
    r"\Longleftrightarrow": r"\Leftrightarrow", r"\ldots": r"\dots", r"\cdots": r"\dots",
    r"\dotsc": r"\dots", r"\dotsb": r"\dots", r"\vdots": r"\dots", r"\ddots": r"\dots",
    r"\land": r"\wedge", r"\lor": r"\vee", r"\lnot": r"\neg", r"\varnothing": r"\emptyset",
}
_CLOSERS = {")", "]", "}", r"\}", r"\rangle", r"\rfloor", r"\rceil", r"\end", "&", r"\\"}
_PAIRED = {r"\langle": (r"\rangle", "angle"), r"\lfloor": (r"\rfloor", "floor"), r"\lceil": (r"\rceil", "ceil")}

_IGNORED = {
    r"\,", r"\;", r"\:", r"\!", r"\ ", r"\>", r"\displaystyle", r"\textstyle", r"\scriptstyle",
    r"\scriptscriptstyle", r"\limits", r"\nolimits", r"\nonumber", r"\notag", r"\mathstrut",
    r"\strut", r"\hfill", r"\allowbreak", r"\nobreak", r"\protect", r"\enspace", r"\thinspace",
    r"\medspace", r"\thickspace", r"\negthinspace", r"\cr", r"\hline",
}
_SKIP_ARG = {r"\label", r"\tag", r"\tag*", r"\hspace", r"\hspace*", r"\vspace", r"\phantom",
             r"\hphantom", r"\vphantom", r"\color"}
_TRANSPARENT = {
    r"\mathbf", r"\boldsymbol", r"\bm", r"\mathit", r"\mathsf", r"\mathtt", r"\mathnormal", r"\pmb",
    r"\textbf", r"\textit", r"\emph", r"\underline", r"\boxed", r"\cancel", r"\mathop", r"\mathbfit",
    r"\symbf", r"\symbfit", r"\substack", r"\mathbin", r"\mathrel", r"\mathord", r"\ensuremath",
}
_TEXT = {r"\text", r"\textrm", r"\textnormal", r"\textup", r"\textsf", r"\texttt", r"\mbox", r"\hbox"}
_ROMAN_TEXT = {r"\mathrm", r"\operatorname", r"\operatorname*"}
_LETTER_FONTS = {r"\mathcal", r"\mathscr", r"\mathfrak", r"\EuScript"}
_ACCENTS = {
    r"\hat": "hat", r"\widehat": "hat", r"\bar": "bar", r"\overline": "bar", r"\tilde": "tilde",
    r"\widetilde": "tilde", r"\dot": "dot", r"\ddot": "double dot", r"\check": "check",
    r"\breve": "breve", r"\mathring": "ring", r"\vec": "vector", r"\overrightarrow": "vector",
}

_REL = {  # command -> (reading, reading when nothing stands on its left)
    "=": ("equals", None), "<": ("less than", "less than"), ">": ("greater than", "more than"),
    r"\leq": ("less than or equal to", "at most"), r"\geq": ("greater than or equal to", "at least"),
    r"\neq": ("is not equal to", None), r"\approx": ("is approximately", "approximately"),
    r"\simeq": ("is approximately", "approximately"), r"\sim": ("is distributed as", "about"),
    r"\propto": ("is proportional to", None), r"\equiv": ("is equivalent to", None),
    r"\cong": ("is congruent to", None), r"\ll": ("is much less than", "much less than"),
    r"\gg": ("is much greater than", "much more than"), r"\in": ("in", None),
    r"\notin": ("not in", None), r"\ni": ("contains", None), r"\subset": ("is a subset of", None),
    r"\subseteq": ("is a subset of", None), r"\supset": ("is a superset of", None),
    r"\supseteq": ("is a superset of", None), r"\to": ("to", "to"), r"\mapsto": ("maps to", None),
    r"\Rightarrow": ("implies", None), r"\Leftrightarrow": ("if and only if", None),
    r"\leftarrow": ("gets", None), r"\coloneqq": ("is defined as", None),
    r"\triangleq": ("is defined as", None), r"\doteq": ("is defined as", None),
    r"\defeq": ("is defined as", None), r"\perp": ("is perpendicular to", None),
    r"\parallel": ("is parallel to", None), r"\mid": ("given", None),
}
_OPS = {
    "+": "plus", "-": "minus", "*": "times", "/": "over", r"\pm": "plus or minus",
    r"\mp": "minus or plus", r"\times": "times", r"\cdot": "times", r"\div": "divided by",
    r"\ast": "times", r"\star": "star", r"\circ": "composed with", r"\otimes": "tensor",
    r"\odot": "element-wise times", r"\oplus": "direct sum", r"\cup": "union",
    r"\cap": "intersection", r"\setminus": "minus", r"\wedge": "and", r"\vee": "or",
    r"\bullet": "times",
}
_BIGOPS = {  # command -> (words, tag)
    r"\sum": ("sum", "sum"), r"\prod": ("product", "sum"), r"\int": ("integral", "sum"),
    r"\iint": ("double integral", "sum"), r"\iiint": ("triple integral", "sum"),
    r"\oint": ("contour integral", "sum"), r"\bigcup": ("union", "sum"),
    r"\bigcap": ("intersection", "sum"), r"\bigoplus": ("direct sum", "sum"),
    r"\bigotimes": ("tensor product", "sum"), r"\min": ("min", "opt"), r"\max": ("max", "opt"),
    r"\argmax": ("arg max", "opt"), r"\argmin": ("arg min", "opt"), r"\sup": ("sup", "opt"),
    r"\inf": ("inf", "opt"), r"\limsup": ("lim sup", "opt"), r"\liminf": ("lim inf", "opt"),
    r"\lim": ("limit", "lim"), r"\Pr": ("probability", "prob"), r"\nabla": ("gradient", "grad"),
    r"\E": ("expectation", "expect"),
}
_FUNCS = {
    r"\log": "log", r"\ln": "natural log", r"\lg": "log", r"\exp": "exp", r"\sin": "sine",
    r"\cos": "cosine", r"\tan": "tangent", r"\tanh": "tanh", r"\sinh": "sinh", r"\cosh": "cosh",
    r"\arcsin": "arc sine", r"\arccos": "arc cosine", r"\arctan": "arc tangent",
    r"\det": "determinant", r"\dim": "dimension", r"\ker": "kernel", r"\deg": "degree",
    r"\gcd": "gcd", r"\sgn": "sign", r"\Tr": "trace", r"\tr": "trace", r"\diag": "diag",
}
_SYMBOLS = {
    r"\infty": "infinity", r"\ell": "ell", r"\hbar": "h bar", r"\emptyset": "the empty set",
    r"\partial": "partial", r"\forall": "for all", r"\exists": "there exists",
    r"\nexists": "there is no", r"\neg": "not", r"\top": "transpose", r"\intercal": "transpose",
    r"\bot": "bottom", r"\dagger": "dagger", r"\ddagger": "double dagger", r"\prime": "prime",
    r"\angle": "angle", r"\triangle": "triangle", r"\checkmark": "check", r"\aleph": "aleph",
    r"\Re": "real part", r"\Im": "imaginary part", r"\%": "percent", r"\$": "dollars",
    r"\&": "and", r"\#": "number", r"\R": "R",
}
_NAMED_TEXT = {  # \mathrm / \operatorname / \text contents worth reading differently
    "s.t.": "such that", "w.r.t.": "with respect to", "i.e.": "that is", "e.g.": "for example",
    "tr": "trace", "Tr": "trace", "Var": "variance", "Cov": "covariance", "sgn": "sign",
    "argmax": "arg max", "argmin": "arg min", "arg max": "arg max", "arg min": "arg min",
    "relu": "ReLU", "otherwise": "otherwise", "if": "if",
}
_OPTIMIZER_TEXT = {"arg max", "arg min", "max", "min", "sup", "inf"}
# Lower-case letter runs that are words, not products of variables ("softmax(x)" written in italics).
_MATH_WORDS = {
    "softmax": "softmax", "sigmoid": "sigmoid", "relu": "ReLU", "argmax": "arg max",
    "argmin": "arg min", "max": "max", "min": "min", "log": "log", "exp": "exp", "sin": "sine",
    "cos": "cosine", "tanh": "tanh", "sign": "sign", "clip": "clip", "mean": "mean", "std": "std",
}
_FOOTNOTE_SUP = {"star", "times", "dagger", "double dagger", "prime"}


def _tokenize(latex: str) -> list[str]:
    for char, replacement in _UNICODE_MATH.items():
        latex = latex.replace(char, replacement)
    tokens: list[str] = []
    drop_dot = False
    for tok in _TOKEN_RE.findall(latex):
        if tok.isspace():
            tokens.append(" ")
            continue
        if drop_dot and tok == ".":
            drop_dot = False
            continue
        drop_dot = False
        if tok in _SIZING:
            drop_dot = True
            if tok == r"\middle":
                tokens.append(r"\middle")
            continue
        if tokens and tokens[-1] == r"\middle":
            tokens[-1] = r"\mid" if tok == "|" else tok
            continue
        tokens.append(_TOKEN_ALIASES.get(tok, tok))
    return tokens


class _Parser:
    def __init__(self, latex: str) -> None:
        self.toks = _tokenize(latex)
        self.pos = 0
        self.clean = True

    # ---------------------------------------------------------------- token access

    def _skip_space(self) -> None:
        while self.pos < len(self.toks) and self.toks[self.pos] == " ":
            self.pos += 1

    def peek(self) -> str | None:
        self._skip_space()
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def take(self) -> str | None:
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, tok: str) -> None:
        if self.peek() == tok:
            self.pos += 1
        else:
            self.clean = False

    # ---------------------------------------------------------------- grammar

    def parse(self) -> list:
        nodes = self.seq(frozenset())
        while self.peek() is not None:  # stray closers at the top level
            self.pos += 1
            self.clean = False
            nodes += self.seq(frozenset())
        return nodes

    def seq(self, stop: frozenset) -> list:
        nodes: list = []
        while (tok := self.peek()) is not None and tok not in stop:
            if tok in _CLOSERS:  # closes nothing that is open
                self.pos += 1
                self.clean = self.clean and tok in ("&", r"\\")
                if tok == r"\\":
                    nodes.append(_Sym(",", "punct"))
                continue
            if tok in ("^", "_"):
                base = nodes.pop() if nodes and _takes_scripts(nodes[-1]) else None
                nodes.append(self.scripts(base, stop))
                continue
            if tok == "'":
                count = 0
                while self.peek() == "'":
                    self.pos += 1
                    count += 1
                prime = _Sym({1: "prime", 2: "double prime"}.get(count, "triple prime"), "text")
                base = nodes.pop() if nodes else None
                nodes.append(_attach(base, "sup", [prime]))
                continue
            node = self.atom(stop)
            if node is not None:
                nodes.append(node)
        return nodes

    def scripts(self, base, stop: frozenset):
        while (tok := self.peek()) in ("^", "_"):
            self.pos += 1
            base = _attach(base, "sup" if tok == "^" else "sub", self.arg(stop))
        return base

    def arg(self, stop: frozenset) -> list:
        tok = self.peek()
        if tok is None:
            self.clean = False
            return []
        if tok == "{":
            self.pos += 1
            body = self.seq(stop | {"}"})
            self.expect("}")
            return body
        node = self.atom(stop)
        return [node] if node is not None else []

    def raw_arg(self) -> str:
        """The verbatim text of a {...} argument (spaces kept), for \\text and font commands."""
        if self.peek() != "{":
            tok = self.take() or ""
            return tok.lstrip("\\")
        self.pos += 1
        depth, parts = 1, []
        while self.pos < len(self.toks):
            tok = self.toks[self.pos]
            self.pos += 1
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth -= 1
                if depth == 0:
                    break
            elif tok in (r"\,", r"\;", r"\ ", r"\:", r"\quad", r"\qquad", r"\_", "~"):
                parts.append(" ")
            elif tok.startswith("\\"):
                parts.append(_SYMBOLS.get(tok, "") if len(tok) == 2 else "")
            elif tok != "$":
                parts.append(tok)
        return "".join(parts)

    def atom(self, stop: frozenset):
        tok = self.take()
        if tok is None:
            self.clean = False
            return None
        if tok == "{":
            body = self.seq(stop | {"}"})
            self.expect("}")
            return _Group("{", body)
        if tok in ("(", "["):
            body = self.seq(stop | {")", "]"})
            close = self.take() if self.peek() in (")", "]") else None
            self.clean = self.clean and close is not None
            return _Group(tok, body, close)
        if tok == r"\{":
            body = self.seq(stop | {r"\}"})
            self.expect(r"\}")
            return _Group("set", body)
        if tok == "|" and self.peek() == "|":  # ||x|| written with single bars
            self.pos += 1
            body = self.seq(stop | {"|"})
            self.expect("|")
            self.expect("|")
            return _Group("norm", body)
        if tok in ("|", r"\|"):
            return self.bars(tok, stop)
        if tok in _PAIRED:
            closer, kind = _PAIRED[tok]
            body = self.seq(stop | {closer})
            self.expect(closer)
            return _Group(kind, body)
        if tok[0].isdigit() or (tok[0] == "." and len(tok) > 1):
            return _Sym(tok, "num")
        if tok.isalpha() and len(tok) == 1:
            return self.letters(tok)
        if tok.startswith("\\"):
            return self.command(tok, stop)
        if tok == ":" and self.peek() == "=":
            self.pos += 1
            return _Sym("is defined as", "rel")
        if tok in _REL:
            return _rel(tok)
        if tok in _OPS:
            return _Sym(_OPS[tok], "op")
        if tok in (",", ";", ":", "."):
            return _Sym(tok, "punct")
        if tok == "!":
            return _Sym("factorial", "text")
        if tok in ("~", "$", '"', "?"):
            return None
        self.clean = False
        return None

    def bars(self, bar: str, stop: frozenset):
        """|x| is an absolute value and ||x|| a norm, but p(x | z) reads "given"."""
        start = self.pos
        body = self.seq(stop | {bar})
        if self.peek() == bar and body:
            self.pos += 1
            return _Group("abs" if bar == "|" else "norm", body)
        self.pos = start
        return _Sym("given", "rel") if bar == "|" else _Sym("parallel", "sep")

    def letters(self, first: str):
        run = [first]
        i = self.pos
        while i < len(self.toks) and len(self.toks[i]) == 1 and self.toks[i].isalpha():
            run.append(self.toks[i])
            i += 1
        word = "".join(run)
        if len(word) >= 2 and word.isupper():  # an acronym such as LP, KL, MSE
            self.pos = i
            return _Sym(word, "func")
        for length in range(len(word), 2, -1):
            if word[:length] in _MATH_WORDS:
                self.pos += length - 1
                return _Sym(_MATH_WORDS[word[:length]], "func")
        return _Sym(first, "var")

    def command(self, tok: str, stop: frozenset):  # a flat dispatch table
        if tok in _GREEK_COMMANDS:
            return _Sym(_GREEK_COMMANDS[tok], "var")
        if tok in _IGNORED:
            return None
        if tok in (r"\quad", r"\qquad", r"\\"):
            return _Sym(",", "punct")
        if tok in _SKIP_ARG:
            self.raw_arg()
            return None
        if tok == r"\textcolor":
            self.raw_arg()
            return _Group("{", self.arg(stop))
        if tok in _TRANSPARENT:
            return _Group("{", self.arg(stop))
        if tok in _TEXT or tok in _ROMAN_TEXT:
            return _text_sym(self.raw_arg(), roman=tok in _ROMAN_TEXT)
        if tok == r"\mathbb":
            return _blackboard(self.raw_arg().strip())
        if tok in _LETTER_FONTS:
            letter = self.raw_arg().strip()
            return {"N": _Sym("N", "var", tag="gauss"), "O": _Sym("big O", "func")}.get(
                letter, _Sym(letter, "var"))
        if tok == r"\frac":
            return _Frac(self.arg(stop), self.arg(stop))
        if tok == r"\binom":
            return _Frac(self.arg(stop), self.arg(stop), binom=True)
        if tok == r"\sqrt":
            index = None
            if self.peek() == "[":
                self.pos += 1
                index = self.seq(stop | {"]"})
                self.expect("]")
            return _Root(index, self.arg(stop))
        if tok in _ACCENTS:
            return _Accent(_ACCENTS[tok], self.arg(stop))
        if tok in (r"\overset", r"\stackrel"):
            self.arg(stop)
            return _Group("{", self.arg(stop))
        if tok == r"\underset":
            below = self.arg(stop)
            return _Script(_Group("{", self.arg(stop)), sub=below)
        if tok in (r"\overbrace", r"\underbrace"):
            body = self.arg(stop)
            while self.peek() in ("^", "_"):  # the brace's label is not part of the formula
                self.pos += 1
                self.arg(stop)
            return _Group("{", body)
        if tok == r"\begin":
            return self.environment(stop)
        if tok == r"\not":
            negated = self.atom(stop)
            if isinstance(negated, _Sym) and negated.kind == "rel":
                return _Sym("not " + negated.words.removeprefix("is "), "rel")
            return negated
        if tok == r"\arg" and self.peek() in (r"\max", r"\min"):
            return _Sym("arg " + self.take()[1:], "bigop", tag="opt")
        if tok in _REL:
            return _rel(tok)
        if tok in _OPS:
            return _Sym(_OPS[tok], "op")
        if tok in _BIGOPS:
            words, tag = _BIGOPS[tok]
            return _Sym(words, "bigop", tag=tag)
        if tok in _FUNCS:
            return _Sym(_FUNCS[tok], "func")
        if tok in _SYMBOLS:
            return _Sym(_SYMBOLS[tok], "var")
        if tok == r"\dots":
            return _Sym("…", "dots")
        if tok == r"\_":
            return None
        # Unknown command (often a paper's own macro): read its name, keep its arguments.
        self.clean = False
        name = tok.lstrip("\\").rstrip("*")
        if not name.isalpha():
            return None
        return _Sym(name if len(name) >= 3 else " ".join(name), "var")

    def environment(self, stop: frozenset):
        name = self.raw_arg().strip().rstrip("*")
        if name in ("array", "tabular"):
            self.raw_arg()  # column spec
        if "matrix" in name:
            self.clean = False  # read as "a matrix"; too long to say element by element
        rows, cells = [], []
        cell_stop = stop | {"&", r"\\", r"\end"}
        while True:
            cells.append(self.seq(cell_stop))
            tok = self.take()
            if tok == "&":
                continue
            rows.append(cells)
            cells = []
            if tok == r"\\":
                continue
            if tok == r"\end":
                self.raw_arg()
            else:
                self.clean = False
            break
        return _Env(name, [row for row in rows if any(row)])


_GREEK_COMMANDS = {"\\" + name: words for name, words in _GREEK.items()}


def _rel(tok: str) -> _Sym:
    words, prose = _REL[tok]
    return _Sym(words, "rel", prose=prose)


def _text_sym(raw: str, *, roman: bool) -> _Sym:
    text = re.sub(r"\s+", " ", raw).strip()
    text = _NAMED_TEXT.get(text, text)
    if roman and text in _OPTIMIZER_TEXT:
        return _Sym(text, "bigop", tag="opt")
    if len(text) == 1:
        return _Sym(text, "var")
    return _Sym(text, "func" if roman else "text")


def _blackboard(letter: str) -> _Sym:
    if letter == "E":
        return _Sym("expectation", "bigop", tag="expect")
    if letter == "P":
        return _Sym("probability", "bigop", tag="prob")
    if letter in ("1", "I"):
        return _Sym("indicator", "bigop", tag="indicator")
    return _Sym(letter, "var")


def _attach(base, slot: str, nodes: list) -> _Script:
    if not isinstance(base, _Script) or getattr(base, slot) is not None:
        base = _Script(base)
    setattr(base, slot, nodes)
    return base


def _takes_scripts(node) -> bool:
    return not (isinstance(node, _Sym) and node.kind in ("op", "rel", "punct", "sep"))


# ---------------------------------------------------------------- rendering


def _is_sym(node, kind: str, words: str | None = None) -> bool:
    return isinstance(node, _Sym) and node.kind == kind and (words is None or node.words == words)


def _unwrap(nodes: list) -> list:
    """Drop invisible {...} groups around a single node: {{2}} -> 2."""
    while len(nodes) == 1 and isinstance(nodes[0], _Group) and nodes[0].open == "{":
        nodes = nodes[0].body
    return nodes


def _has_operator(nodes: list) -> bool:
    return any(isinstance(n, _Sym) and n.kind in ("op", "rel") for n in _unwrap(nodes))


def _callable(node) -> bool:
    if isinstance(node, _Sym):
        return node.kind in ("var", "func") and node.words not in ("infinity", "partial")
    if isinstance(node, _Script):
        return node.base is not None and not _is_sym(node.base, "bigop") and _callable(node.base)
    if isinstance(node, _Accent):
        return True
    if isinstance(node, _Group) and node.open == "{":
        return len(node.body) == 1 and _callable(node.body[0])
    return False


def _split_args(nodes: list) -> list[list]:
    args, current = [], []
    for node in nodes:
        if (isinstance(node, _Sym) and node.kind == "punct" and node.words in (",", ";")) or \
                _is_sym(node, "sep"):
            args.append(current)
            current = []
        else:
            current.append(node)
    args.append(current)
    return [a for a in args if a]


def _join_and(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _is_interval(group: _Group) -> bool:
    return (group.open == "[" or group.close == "]") and len(_split_args(group.body)) == 2 and \
        sum(_is_sym(n, "punct", ",") for n in group.body) == 1


def _render(nodes: list, ctx: str = "") -> str:
    out: list[str] = []
    prev = ""  # "", operand, op, rel, punct
    i = 0
    while i < len(nodes):
        node = nodes[i]
        nxt = nodes[i + 1] if i + 1 < len(nodes) else None
        if _is_sym(node, "num", "1") and _is_sym(nxt, "op", "times") and i + 2 < len(nodes) \
                and _power_of_ten(nodes[i + 2]) is not None:
            i += 2  # "1 \times 10^{-4}" reads "ten to the minus four"
            continue
        base = node.base if isinstance(node, _Script) else node
        if _is_sym(base, "bigop"):
            phrase, used = _render_bigop(nodes, i, ctx)
            out.append(phrase)
            i += used
            prev = "operand"
            continue
        if _callable(node) and isinstance(nxt, _Group) and nxt.open == "(" and not _is_interval(nxt):
            out.append(_render_call(node, nxt, ctx))
            i += 2
            prev = "operand"
            continue
        if isinstance(node, _Sym) and node.kind == "rel":
            leading = prev in ("", "op", "rel", "punct")
            out.append(node.prose if leading and node.prose else _relation_words(node, ctx))
            prev = "rel"
        elif isinstance(node, _Sym) and node.kind == "op":
            out.append(node.words)
            prev = "op"
        elif isinstance(node, _Sym) and node.kind == "punct":
            out.append(_punct_words(node, ctx, nodes[i + 1:]))
            prev = "punct"
        elif isinstance(node, _Group) and node.open == "(" and len(nodes) > 1 and \
                _has_operator(node.body) and not _is_interval(node):
            inner = _render(node.body, ctx)
            out.append(f"{'times ' if prev == 'operand' else ''}the quantity {inner},")
            prev = "operand"
        else:
            out.append(_render_node(node, ctx))
            prev = "operand"
        i += 1
    return _speak_dots(" ".join(w for w in out if w))


def _speak_dots(text: str) -> str:
    """'x 1, …, x n' -> 'x 1 up to x n'; a trailing '…' -> 'and so on' (per sequence, so that
    the dots inside (c_0, ..., c_j, ...) are read before the surrounding commas join in)."""
    text = re.sub(r"\s*,\s*…\s*,\s*", " up to ", text)
    return re.sub(r"\s*,?\s*…$", ", and so on", text)


def _relation_words(node: _Sym, ctx: str) -> str:
    if ctx == "lim" and node.words == "to":
        return "goes to"
    if ctx == "expect" and node.words == "is distributed as":
        return "drawn from"
    return node.words


def _punct_words(node: _Sym, ctx: str, rest: list) -> str:
    if node.words == ":":
        if ctx in ("sub", "sup"):
            return "to"
        if ctx == "set":
            return "such that"
        return "maps" if any(_is_sym(n, "rel", "to") for n in rest) else ","
    return "" if node.words == "." else node.words


def _render_node(node, ctx: str) -> str:
    if isinstance(node, _Sym):
        return node.words
    if isinstance(node, _Script):
        return _render_script(node, ctx)
    if isinstance(node, _Frac):
        return _render_frac(node, ctx)
    if isinstance(node, _Root):
        body = _render(node.body, ctx)
        index = _render(node.index or [], ctx)
        if index in ("", "2"):
            return f"the square root of {body}"
        return f"the cube root of {body}" if index == "3" else f"the {index}-th root of {body}"
    if isinstance(node, _Accent):
        body = _render(node.body, ctx)
        return f"vector {body}" if node.name == "vector" else f"{body} {node.name}"
    if isinstance(node, _Env):
        return _render_env(node, ctx)
    return _render_group(node, ctx)


def _render_group(group: _Group, ctx: str) -> str:
    if group.open in ("(", "[") and _is_interval(group):
        low, high = (_render(part, ctx) for part in _split_args(group.body))
        return f"the interval from {low} to {high}"
    if group.open in ("{", "(", "["):
        return _render(group.body, ctx)
    if group.open == "set":
        return f"the set of {_render(group.body, 'set')}" if group.body else "the empty set"
    if group.open == "abs":
        body = _unwrap(group.body)
        inner = _render(body, ctx)
        single_capital = len(body) == 1 and _is_sym(body[0], "var") and body[0].words.isupper()
        return f"the size of {inner}" if single_capital else f"the absolute value of {inner}"
    if group.open == "angle":
        return f"the inner product of {_join_and([_render(a, ctx) for a in _split_args(group.body)])}"
    if group.open == "norm":
        return f"the norm of {_render(group.body, ctx)}"
    return f"the {'ceiling' if group.open == 'ceil' else 'floor'} of {_render(group.body, ctx)}"


def _render_call(func, args_group: _Group, ctx: str) -> str:
    name = _render([func], ctx)
    args = [_render(arg, ctx) for arg in _split_args(args_group.body)]
    if re.sub(r"\W", "", name.lower()) in ("kl", "dkl") and len(args) == 2:
        return f"the KL divergence between {args[0]} and {args[1]}"
    if isinstance(func, _Sym) and func.tag == "gauss" and len(args) == 2:
        spread = "variance" if args[1].endswith("squared") else "covariance"
        return f"a Gaussian with mean {args[0]} and {spread} {args[1]}"
    return f"{name} of {_join_and(args)}"


def _render_bigop(nodes: list, i: int, ctx: str) -> tuple[str, int]:
    node = nodes[i]
    sym, sub, sup = (node.base, node.sub, node.sup) if isinstance(node, _Script) else (node, None, None)
    following = nodes[i + 1] if i + 1 < len(nodes) else None
    rest = following is not None and not (isinstance(following, _Sym) and following.kind in ("rel", "punct"))
    of = " of" if rest else ""
    if sym.tag == "sum":
        if sub and sup:
            limits = f" from {_render(sub, 'sub')} to {_render(sup, 'sup')}"
        else:
            limits = f" over {_render(sub, 'sub')}" if sub else ""
        return f"the {sym.words}{limits}{of}", 1
    if sym.tag == "opt":
        return f"{sym.words}{' over ' + _render(sub, 'sub') if sub else ''}{of}", 1
    if sym.tag == "lim":
        return f"the limit{' as ' + _render(sub, 'lim') if sub else ''}{of}", 1
    if sym.tag == "grad":
        return f"the gradient{' with respect to ' + _render(sub, 'sub') if sub else ''}{of}", 1
    # expectation, probability, indicator: E_{x ~ p}[f(x)]
    over = f" over {_render(sub, 'expect')}" if sub else ""
    if isinstance(following, _Group) and following.open in ("(", "[", "{", "set"):
        return f"the {sym.words}{over} of {_render(following.body, ctx)}", 2
    return f"the {sym.words}{over}{of}", 1


def _power_of_ten(node) -> int | None:
    if isinstance(node, _Script) and _is_sym(node.base, "num", "10") and node.sup and not node.sub:
        text = _render(node.sup).replace("minus ", "-")
        if re.fullmatch(r"-?\d+", text):
            return int(text)
    return None


def _render_script(node: _Script, ctx: str) -> str:
    base = node.base
    if base is None:  # "$^{1}$" footnote marks, "m$^2$", "$^\circ$"
        if node.sup:
            reading = _sup_words(node.sup)
            return reading if reading in ("squared", "cubed", "inverse", "degrees") else ""
        return _render(node.sub or [], "sub")
    exponent = _power_of_ten(node)
    if exponent is not None:
        return f"ten to the {_number_words(exponent)}"
    if isinstance(base, _Group) and base.open == "norm":
        kind = _render(node.sub or [], "sub")
        squared = "squared " if node.sup and _sup_words(node.sup) == "squared" else ""
        label = {"": "", "2": "L2 ", "1": "L1 ", "F": "Frobenius ", "infinity": "infinity "}.get(kind, f"{kind} ")
        return f"the {squared}{label}norm of {_render(base.body, ctx)}"
    if isinstance(base, _Group) and base.open == "set" and node.sub:
        words = f"{_render_group(base, ctx)} for {_render(node.sub, 'sub')}"
        return words + (f" to {_render(node.sup, 'sup')}" if node.sup else "")
    if isinstance(base, _Group) and base.open in ("(", "[") and node.sup and _has_operator(base.body):
        parts = [f"the quantity {_render(base.body, ctx)},"]
    else:
        parts = [_render([base], ctx)]
    label = _unwrap(node.sup or [])
    if len(label) == 1 and isinstance(label[0], _Sym) and label[0].kind in ("text", "func") \
            and label[0].words not in _FOOTNOTE_SUP:
        parts.append(label[0].words)  # q^{des}_t: a name, not a power -> "q des t"
        node = _Script(node.base, node.sub)
    if node.sub:
        sub = _render(node.sub, "sub")
        complex_sub = _has_operator(node.sub) or any(_is_sym(n, "punct", ":") for n in node.sub)
        parts.append(f"sub {sub}" if complex_sub else sub.replace(",", ""))
    if node.sup:
        parts.append(_sup_words(node.sup))
    return " ".join(p for p in parts if p)


def _sup_words(sup: list) -> str:
    nodes = _unwrap(sup)
    if len(nodes) == 1:
        only = nodes[0]
        if isinstance(only, _Sym):
            if only.kind == "num" and only.words in ("2", "3"):
                return "squared" if only.words == "2" else "cubed"
            if only.words in ("T", "transpose"):
                return "transpose"
            if only.words in _FOOTNOTE_SUP:
                return "star" if only.words == "times" else only.words
            if only.words == "composed with":
                return "degrees"
        if isinstance(only, _Group) and only.open == "(":  # x^{(i)}: the i-th sample
            return _render(only.body, "sup")
    if len(nodes) == 2 and _is_sym(nodes[0], "op", "minus") and _is_sym(nodes[1], "num", "1"):
        return "inverse"
    return f"to the {_render(nodes, 'sup')}"


_FRACTION_WORDS = {("1", "2"): "one half", ("1", "3"): "one third", ("1", "4"): "one quarter"}


def _render_frac(node: _Frac, ctx: str) -> str:
    num, den = _render(node.num, ctx), _render(node.den, ctx)
    if node.binom:
        return f"{num} choose {den}"
    for marker, word in (("partial", "partial derivative"), ("d", "derivative")):
        top, bottom = _unwrap(node.num), _unwrap(node.den)
        if top and bottom and _is_sym(top[0], "var", marker) and _is_sym(bottom[0], "var", marker):
            what, wrt = _render(top[1:], ctx), _render(bottom[1:], ctx)
            return f"the {word} of {what} with respect to {wrt}" if what else \
                f"the {word} with respect to {wrt} of"
    if (num, den) in _FRACTION_WORDS:
        return _FRACTION_WORDS[(num, den)]
    num_complex, den_complex = _has_operator(node.num), _has_operator(node.den)
    if den_complex:
        return f"{num}{', all' if num_complex else ''} over the quantity {den}"
    return f"{num}, all over {den}" if num_complex else f"{num} over {den}"


def _render_env(env: _Env, ctx: str) -> str:
    if env.name == "cases":
        rows = []
        for cells in env.rows:
            value = _render(cells[0], ctx)
            condition = " ".join(_render(c, ctx) for c in cells[1:]).strip()
            if condition and not re.match(r"(if|otherwise|else|for|when|where)\b", condition):
                condition = "if " + condition
            rows.append(f"{value} {condition}".strip())
        return ", ".join(rows)
    if "matrix" in env.name:
        return "a matrix"
    return ", ".join(_render([n for cell in cells for n in cell], ctx) for cells in env.rows)


def _tidy_math(text: str) -> str:
    text = text.replace("…", "dot dot dot")
    text = re.sub(r"[\\{}$^_&|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,;.])", r"\1", text)
    text = re.sub(r"([,;])(?:\s*[,;])+", r"\1", text)
    text = re.sub(r"^[\s,;]+|[\s,;]+$", "", text)
    return re.sub(r"\s+of$", "", text)


def _convert(latex: str) -> tuple[str, bool]:
    """(spoken text, clean) — clean is False when anything had to be guessed or dropped."""
    parser = _Parser(latex.strip())
    nodes = parser.parse()
    if len(nodes) == 3 and _is_sym(nodes[0], "num") and _is_sym(nodes[1], "op", "minus") \
            and _is_sym(nodes[2], "num"):
        text = f"{nodes[0].words} to {nodes[2].words}"  # "$0.02-0.10$ m" is a range
    else:
        text = _render(nodes)
    return _speak_units(_tidy_math(text)), parser.clean


def latex_to_speech(latex: str) -> str:
    """Inline LaTeX as words. Best effort: never returns backslashes or braces."""
    return _convert(latex)[0]


# ============================================================================ plain text rules

_MATH_SPAN = re.compile(r"\$([^$]+)\$")
_URL = re.compile(
    r"(?:https?://|www\.)[^\s<>()\[\]]*[^\s<>()\[\].,;:!?'\"]"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|org|net|io|ai|edu|dev|gov|co|me|info)\b(?:/[^\s<>()\[\]]*[^\s<>()\[\].,;:!?'\"])?"
)
_URL_MARK = "\x00URL\x00"


def _remove_urls(text: str) -> str:
    text = _URL.sub(_URL_MARK, text)
    text = re.sub(rf"\b(available|released|provided|hosted|found|accessible|published)\s+"
                  rf"(?:at|on|from|via|in)\s+{_URL_MARK}", r"\1 online", text)
    text = re.sub(rf"\s*\(\s*(?:see\s+)?{_URL_MARK}\s*\)", "", text)
    text = re.sub(rf"\b(?:at|on|from|via)\s+{_URL_MARK}", "online", text)
    return re.sub(rf"\s*:?\s*{_URL_MARK}", "", text)


# ---------------------------------------------------------------- citations

_NUMERIC_CITE = re.compile(
    r"\[\s*[1-9]\d*(?:\s*[-–—]\s*[1-9]\d*)?(?:\s*[,;]\s*[1-9]\d*(?:\s*[-–—]\s*[1-9]\d*)?)*\s*\]|\[\?\]")
_CITE_RUN = re.compile(r"\[CITE\](?:\s*(?:[,;]|,?\s*and)?\s*\[CITE\])+")
_CITE_FILLER = re.compile(
    r"\b(?:see(?:\s+also)?|cf\.|e\.g\.|i\.e\.|for\s+(?:example|instance)|and\s+references\s+therein"
    r"|among\s+others|and|or|for\s+(?:a\s+|an\s+)?(?:recent\s+)?(?:review|survey|overview|details))(?!\w)")
_ET_AL_AUTHOR = re.compile(r"\b[A-Z][\w'’-]*(?:\s+(?:and|&)\s+[A-Z][\w'’-]*)?\s+et\s+al\.")
_ADVERBS = {"also", "both", "further", "recently", "first", "instead", "similarly", "then", "previously",
            "successfully", "directly", "explicitly", "originally", "already", "independently"}
_VERBS = set("""
propose show use introduce present demonstrate develop employ study train learn apply leverage
extend report suggest argue find achieve adopt consider combine formulate address tackle focus rely
treat design investigate explore observe note compare evaluate utilize exploit incorporate optimize
require provide build make take obtain outperform attempt aim seek describe define model predict
generate estimate represent enable improve integrate frame cast view rely deploy collect transfer
""".split())
_IRREGULAR_PAST = {"showed", "shown", "found", "made", "took", "gave", "built", "brought", "wrote",
                   "began", "led", "set", "put", "sought", "taught", "drew", "chose", "had", "did"}
_SINGULAR = {"is", "was", "has", "does", "can", "could", "may", "might", "will", "would", "should"}
_PLURAL = {"are", "were", "have", "do"}


def _subject_verb(words: list[str]) -> str | None:
    """The verb after a citation used as a sentence subject, skipping one adverb."""
    if words and words[0] in _ADVERBS:
        words = words[1:]
    return words[0] if words else None


def _verb_like(word: str | None) -> bool:
    if not word:
        return False
    stem = word[:-2] if word.endswith("es") else word[:-1] if word.endswith("s") else word
    return (word in _SINGULAR or word in _PLURAL or word in _IRREGULAR_PAST or word in _VERBS
            or stem in _VERBS or (word.endswith("ed") and len(word) > 4) or stem.rstrip("e") + "e" in _VERBS)


def _subject_noun(verb: str | None) -> str:
    """'Earlier work proposed/proposes' but 'Earlier studies propose/are'."""
    if verb in _PLURAL or (verb in _VERBS and not verb.endswith("ed")):
        return "earlier studies"
    return "earlier work"


def _cite_as_subject(match: re.Match, *, need_verb: bool) -> str:
    """'[CITE] proposed' -> 'Earlier work proposed' (group 2 is the look-ahead at the verb)."""
    lead = match.group(1)
    verb = _subject_verb(match.group(2).split())
    if need_verb and not _verb_like(verb):
        return match.group(0)
    noun = _subject_noun(verb)
    if not lead.strip() or lead.rstrip()[-1:] in ".!?":
        noun = noun.capitalize()
    return f"{lead}{noun} "


def _drop_citation_parentheses(match: re.Match) -> str:
    inner = match.group(1)
    if "[CITE]" not in inner:
        return match.group(0)
    rest = _ET_AL_AUTHOR.sub(" ", _CITE_FILLER.sub(" ", inner.replace("[CITE]", " ")))
    return "" if re.fullmatch(r"[\s,;.]*", rest) else match.group(0)


_CITE_PREPOSITIONS = (
    "in|by|of|from|see|following|follow|follows|like|unlike|than|to|with|on|as|per|after|before|via"
    "|using|use|uses|used|adopt|adopts|adopting|extend|extends|extending|versus|beyond|against|over"
)


def _repair_citations(text: str) -> str:
    text = _NUMERIC_CITE.sub("[CITE]", text)
    text = re.sub(r"\[CITE(?:\s*[,;]\s*CITE)+\]", "[CITE]", text)
    text = _CITE_RUN.sub("[CITE]", text)
    text = re.sub(r"\s*\(([^()]*)\)", _drop_citation_parentheses, text)
    text = re.sub(r"(?:,\s*)?\b(?:e\.g\.|cf\.)\s*,?\s*\[CITE\]\s*,?", "", text)
    # textual citations: "Smith et al. [CITE] showed" -> "Smith and colleagues showed"
    text = re.sub(r"\bet\s+al\.?\s*\[CITE\]", "et al.", text)
    text = re.sub(r"\bet\s+al\.?['’]s\b", "and colleagues'", text)
    text = re.sub(r"\bet\s+al\.(?=\s*$|\s+[A-Z])", "and colleagues.", text)
    text = re.sub(r"\bet\s+al\b\.?", "and colleagues", text)
    # the citation is the subject: "[CITE] proposed" -> "Earlier work proposed"
    text = re.sub(r"(^|[.!?:;]\s+)\[CITE\]\s+(?=([a-z]+(?:\s+[a-z]+)?))",
                  lambda m: _cite_as_subject(m, need_verb=False), text)
    text = re.sub(r"(,\s+|\b(?:while|whereas|but|although|though|since|because|that|where|when)\s+)"
                  r"\[CITE\]\s+(?=([a-z]+(?:\s+[a-z]+)?))",
                  lambda m: _cite_as_subject(m, need_verb=True), text)
    # the citation is an object: "as shown in [CITE]" -> "as shown in earlier work"
    text = re.sub(rf"\b({_CITE_PREPOSITIONS})\s+\[CITE\]", r"\1 earlier work", text, flags=re.I)
    text = re.sub(r"(^|[.!?]\s+)\[CITE\]['’]s\b", r"\1Earlier work's", text)
    text = re.sub(r"\[CITE\]['’]s\b", "earlier work's", text)
    return re.sub(r"\s*\[CITE\]", "", text)


# ---------------------------------------------------------------- references and abbreviations

_REF_WORDS = {  # abbreviation (lower case, no dot) -> (singular, plural)
    "fig": ("Figure", "Figures"), "tab": ("Table", "Tables"), "eq": ("Equation", "Equations"),
    "eqn": ("Equation", "Equations"), "sec": ("Section", "Sections"), "sect": ("Section", "Sections"),
    "alg": ("Algorithm", "Algorithms"), "app": ("Appendix", "Appendices"),
    "appx": ("Appendix", "Appendices"), "thm": ("Theorem", "Theorems"), "lem": ("Lemma", "Lemmas"),
    "prop": ("Proposition", "Propositions"), "def": ("Definition", "Definitions"),
    "cor": ("Corollary", "Corollaries"), "ch": ("Chapter", "Chapters"), "chap": ("Chapter", "Chapters"),
}
_REF_KEYS = "|".join(sorted(_REF_WORDS, key=len, reverse=True))
_REF_ABBREV = re.compile(  # "fig. 3", "Eqs. (4)", "Fig. S2"; "Sec. IV-C" and "App. B" need a capital
    rf"\b((?i:{_REF_KEYS}))((?i:s)?)\.\s*(?=\(?\d|S\d)"
    rf"|\b((?:{_REF_KEYS.title()}))(s?)\.\s*(?=[IVX]+\b|[A-Z]\d*\b)")
_REF_NOUNS = r"Figures?|Tables?|Equations?|Sections?|Algorithms?|Theorems?|Lemmas?"  # appendices use letters


def _expand_reference(match: re.Match) -> str:
    abbreviation, plural_s = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
    singular, plural = _REF_WORDS[abbreviation.lower()]
    return (plural if plural_s else singular) + " "


def _ref_list(match: re.Match) -> str:
    items = re.sub(r"\s*[–—-]\s*", " to ", match.group(2).replace("(", "").replace(")", ""))
    return f"{match.group(1)} {items}"


def _expand_references(text: str) -> str:
    text = _REF_ABBREV.sub(_expand_reference, text)
    text = re.sub(r"§§\s*", "Sections ", text)
    text = re.sub(r"§\s*(?=\d)", "Section ", text)
    text = re.sub(r"\b(figure|table|equation|section|algorithm|appendix)(s?)\s+(?=\(?\d)",
                  lambda m: f"{m.group(1).capitalize()}{m.group(2)} ", text)
    # Equation (4), Equations (4)-(6), Equations (4) and (5)
    text = re.sub(r"\b(Equations?)\s+(\(?\d+[a-z]?\)?(?:\s*(?:,\s*and|,|and|or|to|[–—-])\s*\(?\d+[a-z]?\)?)*)",
                  _ref_list, text)
    # Table II, Section IV-C
    text = re.sub(rf"\b({_REF_NOUNS})\s+([IVX]+)\b(?:-([A-Z])(\d*)\b)?",
                  lambda m: " ".join(p for p in (m.group(1), str(_roman_to_int(m.group(2))),
                                                 m.group(3), m.group(4)) if p), text)
    # sub-figures: Figure 3(b) -> Figure 3 b, Figure 3(a, b) -> Figure 3 a and b
    figure = r"\b(?:Figure|Table)s? \d+"
    text = re.sub(rf"({figure})\s*\(([a-z])\s*,\s*([a-z])\)", r"\1 \2 and \3", text)
    text = re.sub(rf"({figure})\s*\(([a-z])\)", r"\1 \2", text)
    text = re.sub(rf"({figure})([a-z])\b", r"\1 \2", text)
    text = re.sub(rf"({figure} [a-z])\s*[–—-]\s*\(?([a-z])\)?(?!\w)", r"\1 to \2", text)
    return re.sub(rf"({figure} [a-z](?: to [a-z])?\s*(?:,|,?\s*and|or)\s*)\(([a-z])\)", r"\1\2", text)


_ABBREVIATIONS = [
    (r"\be\.\s?g\.(,?)", lambda m: "for example" + m.group(1)),
    (r"\bi\.\s?e\.,?", "that is,"),
    (r"\bw\.\s?r\.\s?t\.?|\bwrt\b", "with respect to"),
    (r"\bcf\.", "compare"),
    (r"\bv\.?s\.?(?=\s)", "versus"),
    (r"\betc\.(?=\s*$|\s+[A-Z])", "et cetera."),
    (r"\betc\.", "et cetera"),
    (r"\bapprox\.", "approximately"),
    (r"\bresp\.", "respectively"),
    (r"\bSo[Tt]A\b|\bSOTA\b", "state of the art"),
    (r"\ba\.k\.a\.?", "also known as"),
    (r"\bi\.i\.d\.?", "IID"),
    (r"\bs\.t\.", "such that"),
    (r"\bviz\.", "namely"),
    (r"\bw/o\b", "without"),
    (r"\bw/(?=\s)", "with"),
    (r"\band/or\b", "and or"),
    (r"\b(\d+)[- ]?Do[Ff]\b", r"\1 degree of freedom"),
    (r"\bDo[Ff]s?\b|\bDOFs?\b", "degrees of freedom"),
]
_ABBREVIATIONS_RE = [(re.compile(pattern), repl) for pattern, repl in _ABBREVIATIONS]


# ---------------------------------------------------------------- numbers, units, symbols

_UNITS = {
    "ms": "millisecond", "µs": "microsecond", "μs": "microsecond", "ns": "nanosecond",
    "sec": "second", "secs": "second", "min": "minute", "mins": "minute", "hr": "hour", "hrs": "hour",
    "Hz": "hertz", "kHz": "kilohertz", "MHz": "megahertz", "GHz": "gigahertz",
    "KB": "kilobyte", "MB": "megabyte", "GB": "gigabyte", "TB": "terabyte", "GiB": "gibibyte",
    "mm": "millimeter", "cm": "centimeter", "km": "kilometer", "µm": "micrometer", "μm": "micrometer",
    "nm": "nanometer", "kg": "kilogram", "Nm": "newton meter", "kW": "kilowatt", "dB": "decibel",
    "px": "pixel", "rad": "radian", "deg": "degree", "fps": "frame per second", "FPS": "frame per second",
    "m/s": "meter per second", "m/s²": "meter per second squared", "m/s^2": "meter per second squared",
    "km/h": "kilometer per hour", "rad/s": "radian per second", "deg/s": "degree per second",
    "°/s": "degree per second",
}
_SPACED_UNITS = {"s": "second", "m": "meter", "h": "hour"}  # only after a space: "1990s" stays
_UNIT_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)*)\s?(" + "|".join(re.escape(u) for u in sorted(_UNITS, key=len, reverse=True))
    + r")(?![\w/²^])")
_SPACED_UNIT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?) ([smh])(?![\w'’/^²-])")
_MULTIPLIERS = {"k": "thousand", "K": "thousand", "M": "million", "B": "billion", "T": "trillion"}


def _plural(unit: str, number: str) -> str:
    if number == "1":
        return unit
    head, sep, tail = unit.partition(" per ")
    head = head if head.endswith(("hertz", "s")) else head + "s"
    return head + sep + tail


def _speak_units(text: str) -> str:
    text = _UNIT_RE.sub(lambda m: f"{m.group(1)} {_plural(_UNITS[m.group(2)], m.group(1))}", text)
    text = _SPACED_UNIT_RE.sub(lambda m: f"{m.group(1)} {_plural(_SPACED_UNITS[m.group(2)], m.group(1))}", text)
    text = re.sub(r"(\d)\s?°\s?C\b", r"\1 degrees Celsius", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?°", lambda m: f"{m.group(1)} degree{'' if m.group(1) == '1' else 's'}", text)
    return re.sub(r"(\d)\s?%", r"\1 percent", text)


def _scientific(match: re.Match) -> str:
    mantissa, sign, exponent = match.groups()
    power = f"ten to the {_number_words(-int(exponent) if sign in ('-', '−') else int(exponent))}"
    return power if mantissa == "1" else f"{mantissa} times {power}"


_COMPARISONS = {"≤": ("less than or equal to", "at most"), "≥": ("greater than or equal to", "at least"),
                "<": ("less than", "less than"), ">": ("greater than", "more than")}


def _comparison(match: re.Match) -> str:
    """'x ≤ 5' reads as a relation, but 'with ≤5 ms' reads 'at most 5 ms'."""
    relation, prose = _COMPARISONS[match.group(1)]
    operand = re.search(r"(?:^|[\s(])(?:[A-Za-z]|\d+(?:\.\d+)?)\s*$", match.string[:match.start()])
    return f" {relation if operand else prose} "


def _speak_numbers(text: str) -> str:
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)[eE]([-+−]?)(\d+)(?!\w|\.\d)", _scientific, text)
    text = re.sub(r"\b10\^\{?([-−]?\d+)\}?", lambda m: f"ten to the {_number_words(int(m.group(1).replace('−', '-')))}", text)
    # ranges: "3–5" and "3-5" (but not dates or model names such as ResNet-50)
    text = re.sub(r"(?<![\w.,\-])(\d+(?:\.\d+)?)\s*[–—]\s*(\d+(?:\.\d+)?)(?!\d|\.\d|[–-]\d)", r"\1 to \2", text)
    text = re.sub(r"(?<![\w.,\-])(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)(?!\d|\.\d|-\d)", r"\1 to \2", text)
    text = re.sub(r"(^|[\s(\[,;:/])[-−](?=\d)", r"\1minus ", text)
    # multiplication: "224×224", "3× faster", "10x faster"
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)(?!\w|\.\d)", r"\1 by \2", text)
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+)?)\s?[×x](?!\w)", r"\1 times", text)
    text = re.sub(r"\s*×\s*", " times ", text)
    text = re.sub(r"\s*±\s*", " plus or minus ", text)
    text = re.sub(r"(^|[\s(\[,;:])[~∼]\s?(?=\d)", r"\1about ", text)
    text = re.sub(r"(^|[\s(\[,;:])≈\s?(?=\d)", r"\1approximately ", text)
    text = re.sub(r"\s*([≤≥<>])\s*(?=\d)", _comparison, text)
    text = re.sub(r"\s*≈\s*", " approximately ", text)
    for symbol, (relation, _) in _COMPARISONS.items():
        text = re.sub(rf"\s+{symbol}\s+", f" {relation} ", text)
    text = re.sub(r"\s*≠\s*", " not equal to ", text)
    text = re.sub(r"\s*(?:→|⟶|->)\s*", " to ", text)
    text = re.sub(r"\s*⇒\s*", " implies ", text)
    text = re.sub(r"\s*[↔⟷]\s*", " and ", text)
    text = re.sub(r"(?<![\w.\-])(\d+(?:\.\d+)?)([kKMBT])(?!\w)", lambda m: f"{m.group(1)} {_MULTIPLIERS[m.group(2)]}", text)
    return _speak_units(text)


def _speak_symbols(text: str) -> str:
    text = re.sub(r"\bC\+\+", "C plus plus", text)
    text = re.sub(r"(\d)\+(?=\s|$|[,.;)])", r"\1 plus", text)
    text = re.sub(r"(^|[\s(])\+(?=\d)", r"\1plus ", text)
    text = re.sub(r"\s\+\s", " plus ", text)
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"#\s?(?=\d)", "number ", text)
    text = re.sub(r"#\s?(?=[A-Za-z])", "number of ", text)
    text = re.sub(r"\b([A-Za-z]+)/(?:s|sec)\b", r"\1 per second", text)
    text = re.sub(r"\b([A-Za-z]+)/(step|epoch|iteration|episode|frame|GPU|day|hour|image|sample)\b", r"\1 per \2", text)
    text = re.sub(r"\b([A-Za-z]{2,})/([A-Za-z]{2,})\b", r"\1 or \2", text)
    text = re.sub(r"(?<=\w)\s?=\s?(?=\w)", " equals ", text)
    text = re.sub(f"[{''.join(_GREEK_CHARS)}]", lambda m: f" {_GREEK_CHARS[m.group(0)]} ", text)
    text = re.sub(r"[¶‖•※↑↓\\{}^_|$~<>]", " ", text)
    return text.replace("§", " ")


def _tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s*[,;:]?\s*\)|\[\s*\]", "", text)
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r",\s*(?=[,.;:!?)])", "", text)
    text = re.sub(r"([;:])\s*(?=[.;:])", "", text)
    text = re.sub(r"(?<!\.)\.\s+\.(?!\.)|(?<!\.)\.\.(?!\.)", ".", text)
    text = re.sub(r"\(\s*,\s*", "(", text)
    text = re.sub(r"^[\s,;:.]+", "", text)
    return text.strip()


def _speak_inline_math(text: str) -> str:
    text = re.sub(r"(\d)\s*\$\^\{?\s*(?:\\(?:text|mathrm|textrm)\{)?(st|nd|rd|th)\}?\s*\}?\$", r"\1\2", text)
    text = re.sub(r"\$\\(?:mu|upmu)\$\s?([sm])\b", r"µ\1", text)

    def speak(match: re.Match) -> str:
        speech = latex_to_speech(match.group(1))
        before = text[match.start() - 1] if match.start() > 0 else " "
        after = text[match.end()] if match.end() < len(text) else " "
        return f"{' ' if before.isalnum() else ''}{speech}{' ' if after.isalnum() else ''}"

    return _MATH_SPAN.sub(speak, text)


def filter_text(text: str) -> str:
    """Block source text -> speakable text (see the module docstring and plan.md's rules table)."""
    text = re.sub(r"[\u00a0\u2009\u202f\u2007\u2002\u2003]", " ", text)
    text = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]", "", text)
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("ﬀ", "ff")
    text = re.sub(r"(?<=[A-Za-z.])~(?=\S)", " ", text)  # LaTeX non-breaking space: "Fig.~3"
    text = _speak_inline_math(text)
    text = re.sub(r"\b(m|cm|mm|km|s)([²³])", lambda m: f"{m.group(1)} {'squared' if m.group(2) == '²' else 'cubed'}", text)
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹*†‡]+", "", text)  # footnote marks, before URLs swallow them
    text = _remove_urls(text)
    text = _repair_citations(text)
    text = _expand_references(text)
    for pattern, repl in _ABBREVIATIONS_RE:
        text = pattern.sub(repl, text)
    text = _speak_numbers(text)
    text = _speak_symbols(text)
    return _tidy(text)


# ============================================================================ blocks, titles, papers

EQUATION_MAX_WORDS = 25  # longer readings are left to the LLM pass ("Equation 4." otherwise)
EQUATION_MAX_LATEX = 120


def equation_speech(block: Block) -> str:
    """Rules-only reading of a display equation: 'Equation 4: x equals ...' when it is short and
    fully understood, else 'Equation 4.' (numbered) or 'An equation follows.' (unnumbered)."""
    latex = re.sub(r"(?:\\[,;:!]|\\quad|\s|[.,;])+$", "", block.latex or "")
    reading, clean = _convert(latex) if latex else ("", False)
    short = (clean and reading and len(reading.split()) <= EQUATION_MAX_WORDS
             and len(latex) <= EQUATION_MAX_LATEX and r"\\" not in latex and r"\begin" not in latex)
    label = equation_label(block.eq_number)
    if label:
        return f"{label}: {reading}." if short else f"{label}."
    return f"{reading}." if short else "An equation follows."


def equation_label(eq_number: str | None) -> str | None:
    """'4' -> 'Equation 4', '6, 7' -> 'Equations 6 and 7'; '∗' and '' are unnumbered."""
    numbers = [n for n in re.split(r"[,;\s]+", (eq_number or "").strip("() ")) if re.search(r"\w", n)]
    if not numbers:
        return None
    return f"Equation {numbers[0]}" if len(numbers) == 1 else f"Equations {_join_and(numbers)}"


def _spoken_section_number(number: str) -> str:
    """IEEE-style 'III-A1' -> '3 A1'; '3.1' and 'A.2' stay as they are."""
    head, _, rest = number.partition("-")
    if re.fullmatch(r"[IVXL]+", head):
        head = str(_roman_to_int(head))
    return f"{head} {rest}" if rest else head


def section_title_speech(section: Section) -> str:
    """'Section 3.1. Policy learning.', 'Appendix A. Proofs.' or 'Abstract.'"""
    title = filter_text(section.title).rstrip(" .:")
    if section.kind == "abstract":
        return "Abstract." if title.lower() in ("", "abstract") else f"Abstract. {title}."
    end = "" if title.endswith(("?", "!")) else "."
    if not section.number:
        return f"{title}{end}" if title else ""
    label = "Appendix" if section.kind == "appendix" else "Section"
    heading = f"{label} {_spoken_section_number(section.number)}."
    return f"{heading} {title}{end}" if title else heading


def block_filter_speech(block: Block) -> str:
    return equation_speech(block) if block.type == "equation" else filter_text(block.text)


def apply_filter(paper: Paper) -> None:
    """Fill speech (source 'filter') for every speakable block and speech_title for every section."""
    for section in paper.sections:
        section.speech_title = section_title_speech(section)
        for block in section.blocks:
            if block.speak:
                block.speech = block_filter_speech(block)
                block.speech_source = "filter"


@dataclass
class SpeechUnit:
    unit_id: str  # Section.id for titles, Block.id for blocks
    kind: Literal["section_title", "block"]
    text: str


def speech_units(paper: Paper) -> list[SpeechUnit]:
    """Everything that will be spoken, in reading order; falls back to the filter for anything
    that has no speech yet."""
    units: list[SpeechUnit] = []
    for section in paper.sections:
        title = section.speech_title if section.speech_title is not None else section_title_speech(section)
        if title.strip():
            units.append(SpeechUnit(section.id, "section_title", title.strip()))
        for block in section.blocks:
            if not block.speak:
                continue
            speech = block.speech if block.speech is not None else block_filter_speech(block)
            if speech.strip():
                units.append(SpeechUnit(block.id, "block", speech.strip()))
    return units


def inline_math(text: str) -> list[str]:
    """The $...$ spans of a block's source text (used by the clean-up guard)."""
    return _MATH_SPAN.findall(text)

