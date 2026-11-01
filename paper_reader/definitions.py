"""Where a paper defines its symbols, for reading equations by meaning.

"r_t = ω^I r^I_t + ω^G r^G_t" can only be read as "the reward at timestep t equals the imitation
weight times …" if the reader knows what the symbols mean, and papers define them all over the
place: in the "where …" sentence after an equation, in an earlier section, in a figure caption.
This module finds those sentences deterministically, so the LLM gets the paper's own definitions
next to each equation instead of guessing:

    index = DefinitionIndex(paper)
    index.for_block(block, skip=(context, context_after))  -> ["Here, $r^{I}_{t}$ and … weights.", …]
    paper_vocabulary(paper)                                -> words the paper uses (grounding check)

Symbols are compared as atoms (base, superscript, subscript): r_{t}^{I} and r^{I}_{t} are the same,
r^{I} matches r^{I}_{t} partly, and a bare single Latin letter is too generic to match on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import Block, Paper

Atom = tuple[str, str, str]  # (base, superscript, subscript), normalized

_GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi "
          "omicron pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega "
          "Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega").split()
_DECORATED = r"\\(?:mathcal|mathbf|boldsymbol|bm|mathbb|mathsf|mathscr|hat|bar|tilde|vec|dot|ddot|widehat|widetilde|overline)"
_BASE = re.compile(
    rf"(?P<deco>{_DECORATED})\s*\{{(?P<deco_arg>[^{{}}]*)\}}"
    rf"|\\(?P<greek>{'|'.join(_GREEK)})(?![A-Za-z])"
    r"|(?<![\\A-Za-z])(?P<latin>[A-Za-z])(?![A-Za-z])"
)
_SCRIPT = re.compile(r"\s*([_^])\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\\[A-Za-z]+|.)")
_MATH = re.compile(r"\$(.+?)\$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\x00])")  # \x00 = a sentence starting with maths
# Words that introduce or state a definition ("where", "denotes", "is the", "their weights", ...).
_CUE = re.compile(
    r"\b(?:where|wherein|here|let|denot\w*|represent\w*|refer\w*\s+to|stands?\s+for|correspond\w*\s+to|"
    r"defin\w*|called|we\s+call|is\s+(?:the|a|an|our|its)|are\s+(?:the|their|our|its)|being|"
    r"weight\w*|number\s+of|set\s+of|parameter\w*|coefficient\w*|factor|term\w*|given\s+by|"
    r"computed\s+as|measured\s+by|indicat\w*|describ\w*)\b", re.IGNORECASE)
# Apposition, the most common way papers define a symbol: "the center-of-mass reward $r^{c}_{t}$",
# "the discount factor $\gamma$", "a policy $\pi_\theta$" (a determiner, up to four words, the maths).
_APPOSITION = re.compile(r"\b(?:the|a|an|our|its|their|each|every)\s+(?:[\w-]+\s+){0,4}\$", re.IGNORECASE)
# ... but only when a noun precedes the maths: "the plot shows $r_t$" and "the sum of $x$" define nothing.
_NOT_A_NOUN = set("""shows show plots gives give uses use has have is are was were be been reaches reach achieves
achieve maximizes maximize minimizes minimize optimizes optimize increases decreases sets computes compute trains
train keeps keep varies vary yields yield returns obtains obtain of to for by with than and or over at in on from
as into under the a an our its their each every""".split())
_WORD = re.compile(r"[a-z]+")

MAX_DEFINITIONS = 6  # equations get more: one per symbol (+2), so every symbol can be covered
MAX_DEFINITION_CHARS = 1200
MIN_SCORE = 2  # e.g. one exact distinctive symbol, or two partial matches; common symbols count less


def _script_text(raw: str) -> str:
    text = raw[1:-1] if raw.startswith("{") else raw
    text = re.sub(r"\\(?:mathrm|text|operatorname|mathit)\s*\{([^{}]*)\}", r"\1", text)
    return re.sub(r"[\s{}\\,]", "", text)


def math_atoms(latex: str) -> set[Atom]:
    """The symbols used in a LaTeX string, as normalized (base, sup, sub) atoms."""
    return set(math_atom_list(latex))


_INDEX_LETTERS = {"t", "i", "j", "k", "l", "m", "n", "h"}


def atom_key(atom: Atom) -> Atom:
    """What identifies a symbol across equations: a pure index subscript is dropped (r^{I}_{t} and
    r^{I}_{t+1} are the same symbol), a naming one is kept (w_{p} and w_{v} are different)."""
    base, sup, sub = atom
    letters = _index_letters(sub)
    return base, sup, "" if not letters or letters in _INDEX_LETTERS else sub


def symbol_key(latex: str) -> Atom | None:
    """The key of the first symbol in a LaTeX string (a reported symbol like "r^{I}_{t}")."""
    atoms = math_atom_list(latex)
    return atom_key(atoms[0]) if atoms else None


def math_atom_list(latex: str) -> list[Atom]:
    """The symbols of a LaTeX string in order of appearance (duplicates kept)."""
    atoms: list[Atom] = []
    pos = 0
    while match := _BASE.search(latex, pos):
        if match["deco"]:
            base = f"{match['deco'][1:]}:{_script_text('{' + match['deco_arg'] + '}')}"
        else:
            base = match["greek"] or match["latin"]
        sup = sub = ""
        pos = match.end()
        while script := _SCRIPT.match(latex, pos):
            if script.group(1) == "^":
                sup = sup or _script_text(script.group(2))
            else:
                sub = sub or _script_text(script.group(2))
            pos = script.end()  # continue after the scripts: their letters are indices, not symbols
        atoms.append((base, sup, sub))
    return atoms


def _is_apposition(sentence: str) -> bool:
    """A noun phrase directly naming a symbol: "the center-of-mass reward $r^{c}_{t}$"."""
    return any(match.group(0).split()[-2].lower() not in _NOT_A_NOUN for match in _APPOSITION.finditer(sentence))


def _index_letters(script: str) -> str:
    """'j+1' and 'j' share the index letter j; 't-1' and 't' share t."""
    return "".join(ch for ch in script if ch.isalpha())


def _match(a: Atom, b: Atom) -> int:
    """3 = same symbol, 2 = same symbol with a different index, 1 = same distinctive base only."""
    if a[0] != b[0]:
        return 0
    if a == b:
        return 3
    sup_same, sub_same = a[1] == b[1], _index_letters(a[2]) == _index_letters(b[2])
    if sup_same and (sub_same or not a[2] or not b[2]):
        return 2
    if sub_same and (not a[1] or not b[1]):
        return 2
    return 1 if len(a[0]) > 1 else 0  # Greek letters / decorated symbols are distinctive; "r" is not


@dataclass
class _Sentence:
    order: int  # position in the paper (block order), for "nearby" ranking and output order
    text: str  # with $LaTeX$, exactly as the block text has it
    atoms: set[Atom]


def _sentences(text: str) -> list[str]:
    """Split prose into sentences without splitting inside $…$ maths."""
    spans: list[str] = []

    def mask(match: re.Match) -> str:
        spans.append(match.group(0))
        return f"\x00{len(spans) - 1}\x00"

    masked = _MATH.sub(mask, text)
    return [re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], part)
            for part in _SENTENCE_END.split(masked)]


class DefinitionIndex:
    """All definition-like sentences of a paper, with the symbols they mention."""

    def __init__(self, paper: Paper):
        self._order: dict[str, int] = {}
        self._sentences: list[_Sentence] = []
        order = 0
        for section in paper.sections:
            for block in section.blocks:
                self._order[block.id] = order
                if block.type in ("paragraph", "list_item", "heading_note"):
                    self._add(order, block.text)
                elif block.type == "figure" and block.figure_id in paper.figures:
                    self._add(order, paper.figures[block.figure_id].caption_text)
                order += 1
        self._counts: dict[Atom, int] = {}
        for sentence in self._sentences:
            for atom in sentence.atoms:
                self._counts[atom] = self._counts.get(atom, 0) + 1

    def _rarity(self, atom: Atom) -> float:
        """1 for a symbol few sentences mention, down to 0.25 for one that is everywhere (s_t, a_t):
        distinctive symbols should decide which definitions an equation gets."""
        count = self._counts.get(atom, 0)
        return 1.0 if count <= 3 else max(0.25, 3 / count)

    def _add(self, order: int, text: str) -> None:
        for sentence in _sentences(text):
            atoms = {a for span in _MATH.findall(sentence) for a in math_atoms(span)}
            if atoms and (_CUE.search(_MATH.sub(" ", sentence)) or _is_apposition(sentence)):
                self._sentences.append(_Sentence(order, sentence.strip(), atoms))

    def for_block(self, block: Block, *, skip: tuple[str, ...] = (), max_sentences: int = MAX_DEFINITIONS,
                  max_chars: int = MAX_DEFINITION_CHARS) -> list[str]:
        """The paper's sentences that define the symbols this block uses, best first by symbol
        match and then by distance, returned in paper order. Sentences contained in `skip`
        (e.g. the context paragraphs the request already carries) and in the block itself are left out."""
        latex = block.latex if block.type == "equation" else " ".join(_MATH.findall(block.text))
        wanted = math_atoms(latex or "")
        if not wanted:
            return []
        here = self._order.get(block.id, 0)
        candidates = [c for c in self._sentences
                      if c.text not in block.text and not any(c.text in s for s in skip if s)]

        def score(sentence: _Sentence) -> float:
            return sum(max((_match(w, a) for a in sentence.atoms), default=0) * self._rarity(w) for w in wanted)

        def rank(sentence: _Sentence, atom: Atom | None = None) -> tuple:
            own = max((_match(atom, a) for a in sentence.atoms), default=0) if atom else 0
            return -own, -score(sentence), abs(sentence.order - here)

        # Coverage first: every symbol gets the sentence that defines it best (exactly, else partly),
        # rarest symbols first; then the remaining room goes to the best overall matches.
        picks: list[_Sentence] = []
        for atom in sorted(wanted, key=lambda a: self._counts.get(a, 0)):
            defining = [c for c in candidates if max((_match(atom, a) for a in c.atoms), default=0) >= 2]
            if defining:
                picks.append(min(defining, key=lambda c: rank(c, atom)))
        picks += sorted((c for c in candidates if score(c) >= MIN_SCORE), key=rank)
        max_sentences = max(max_sentences, len(wanted) + 2) if block.type == "equation" else max_sentences
        max_chars = max(max_chars, 300 * max_sentences) if block.type == "equation" else max_chars
        chosen, size = [], 0
        for sentence in picks:
            if sentence in chosen:
                continue
            if len(chosen) >= max_sentences or size + len(sentence.text) > max_chars:
                break
            chosen.append(sentence)
            size += len(sentence.text)
        return [s.text for s in sorted(chosen, key=lambda s: s.order)]


def paper_vocabulary(paper: Paper) -> set[str]:
    """Every word the paper uses outside maths (plus simple singular forms), for the grounding check."""
    texts = [paper.title] + [s.title for s in paper.sections]
    texts += [b.text for b in paper.iter_blocks()] + [f.caption_text for f in paper.figures.values()]
    words: set[str] = set()
    for text in texts:
        for word in _WORD.findall(_MATH.sub(" ", text or "").lower()):
            words.add(word)
            if word.endswith("ies"):
                words.add(word[:-3] + "y")
            elif word.endswith("s"):
                words.add(word[:-1])
    return words | {name.lower() for name in _GREEK}
