"""Reading maths by meaning, stage 2: one paper-wide name for every symbol, decided before narration.

Narrating section by section, each request used to name its symbols on its own: one called r_t(pi)
"the probability ratio", another "the reward", a third kept the letters. Here, before any narration:

1. (rules) inventory(paper): every symbol as written, how often it is used, the paper's sentences
   that define it and a couple of places where it is used;
2. (LLM, one request per GROUP_SIZE symbols) the glossary prompt names each symbol the way a
   researcher of the field says it, the paper's own definitions first;
3. checked like any answer: a name must be short, plain words, and grounded in the paper's vocabulary.

The narration requests then carry, per block, the names of its symbols ("names"), and the
consistency check treats them as the canonical ones.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .definitions import DefinitionIndex, math_atom_list, symbol_key, symbol_spans
from .models import Paper

GROUP_SIZE = 60  # symbols per request
MAX_EXAMPLES = 2
EXAMPLE_CHARS = 220
MAX_NAME_WORDS = 7
_MATH = re.compile(r"\$(.+?)\$")
_SKIP_BASES = {"mathbb:E", "mathbb:P", "mathbb:1", "mathbb:I", "mathbb:R", "mathbb:N", "mathbb:Z"}  # operators, number sets
_INDEX_LETTERS = set("ijklmnth")

SYSTEM_PROMPT = """\
You name the symbols of a scientific paper for a listener. The paper's narration will read its maths \
by meaning: instead of the letters ("pi i", "V target", "r t") the listener hears what a researcher of \
the field would say ("policy i", "the target value", "the probability ratio at time step t").

For every symbol in the request, give that name:
1. The paper's own definitions (the "definitions" sentences, in words or by a formula) come first. A \
symbol the paper defines keeps that meaning even where the field usually uses the letter for \
something else: an r_t defined as a ratio of two policies is "probability ratio", not "reward".
2. A symbol the paper uses in a standard role of its field without defining it gets the field's \
standard term, judged from the examples: in reinforcement learning pi is "policy", V "value \
function", Q "Q function", A "advantage", gamma "discount factor", s "state", a "action", r \
"reward", theta "policy parameters"; elsewhere L "loss", eta "learning rate", x "input", y "label".
3. Decorations and labels are part of the name: a hat is an estimate ("estimated advantage"), \
V^{target} is "target value", pi_{old} is "old policy", L^{critic}_{on} is "on-policy critic loss".
4. The name does not include an index that the narration says anyway: pi_{i} is "policy" (read \
"policy i"), s_{t} is "state" (read "the state at time step t"). It does include what makes the \
symbol different from its relatives: pi_{old} "old policy", V_{pi_i} "value function".
5. When neither the paper nor the field gives a meaning, the name is the symbol's spoken letters \
("beta", "M", "x"). Never guess.
6. A name is a short noun phrase (one to six words) in the paper's own words, lower case except \
proper names, without "the", without LaTeX or symbols.

Return {"symbols": [{"id", "name"}]} with every id of the request exactly once."""

SCHEMA = {
    "type": "object",
    "properties": {
        "symbols": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}},
                "required": ["id", "name"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["symbols"],
    "additionalProperties": False,
}


def glossary_key(written: str) -> tuple | None:
    """symbol_key, with a power taken off: gamma^{t} and gamma^{2} are gamma (V^{target} keeps its label)."""
    key = symbol_key(written)
    if key and re.fullmatch(r"[ijklmnth]|\d+|-1", key[1]):
        key = (key[0], "", key[2], *key[3:])
    return key


@dataclass
class SymbolInfo:
    id: str
    key: tuple
    latex: str  # as first written
    count: int = 0
    first: int = 0  # position of the first use (block order)
    examples: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)


def _clip(text: str, latex: str) -> str:
    """The part of a sentence or equation around the symbol, at most EXAMPLE_CHARS long."""
    text = " ".join(text.split())
    if len(text) <= EXAMPLE_CHARS:
        return text
    at = max(text.find(latex), 0)
    start = max(0, min(at - EXAMPLE_CHARS // 3, len(text) - EXAMPLE_CHARS))
    return ("…" if start else "") + text[start:start + EXAMPLE_CHARS] + "…"


def inventory(paper: Paper) -> list[SymbolInfo]:
    """Every symbol of the paper's prose and equations (not index letters, operators or number sets),
    in order of first use, with how often it is used, where it is defined and two places it is used."""
    definitions = DefinitionIndex(paper)
    found: dict[tuple, SymbolInfo] = {}
    for block in paper.iter_blocks():
        if not block.speak:
            continue
        if block.type == "equation":
            places = [(block.latex or "", f"${block.latex}$")]
        elif block.text:
            places = [(span, sentence) for sentence in re.split(r"(?<=[.!?])\s+", block.text)
                      for span in _MATH.findall(sentence)]
        else:
            continue
        for latex, context in places:
            for written in symbol_spans(latex):
                key = glossary_key(written)
                atoms = math_atom_list(written)
                if key is None or not atoms or atoms[0][0] in _SKIP_BASES:
                    continue
                if len(key) == 3 and key[0] in _INDEX_LETTERS and not key[1] and not key[2]:
                    continue  # i, j, k, t, n: indices, said as letters
                info = found.get(key)
                if info is None:
                    info = found[key] = SymbolInfo(id="", key=key, latex=written, first=definitions.position(block.id))
                    info.definitions = definitions.for_symbol(atoms[0], near=info.first)
                info.count += 1
                example = _clip(context, written)
                if len(info.examples) < MAX_EXAMPLES and example not in info.examples \
                        and example not in " ".join(info.definitions):
                    info.examples.append(example)
    symbols = sorted(found.values(), key=lambda s: s.first)
    for i, symbol in enumerate(symbols, 1):
        symbol.id = f"s{i}"
    return symbols


def requests(paper: Paper, symbols: list[SymbolInfo]) -> list[str]:
    """User messages, GROUP_SIZE symbols each, with the paper's title and the start of its abstract."""
    abstract = next((b.text for s in paper.sections if s.kind == "abstract" for b in s.blocks if b.text), "")
    head = f"Paper: {paper.title}\nAbstract: {' '.join(abstract.split())[:700]}\n\n"
    messages = []
    for start in range(0, len(symbols), GROUP_SIZE):
        group = symbols[start:start + GROUP_SIZE]
        lines = [json.dumps({"id": s.id, "latex": s.latex, "uses": s.count, "definitions": s.definitions,
                             "examples": s.examples}, ensure_ascii=False) for s in group]
        messages.append(head + f"Name these {len(group)} symbols:\n" + "\n".join(lines))
    return messages


def accept(symbols: list[SymbolInfo], answers: list[dict], vocabulary: set[str] | None,
           ungrounded) -> tuple[dict[tuple, tuple[str, str]], list[str]]:
    """Checked names: key -> (LaTeX as written, name). `ungrounded(name)` lists words the paper never
    uses; such names are dropped (the narration then names the symbol from its definitions as before)."""
    by_id = {s.id: s for s in symbols}
    names: dict[tuple, tuple[str, str]] = {}
    dropped = []
    for answer in answers:
        symbol = by_id.get(str(answer.get("id")))
        name = " ".join(str(answer.get("name", "")).split()).strip(" .").removeprefix("the ")
        if symbol is None or not name:
            continue
        if re.search(r"[\\$^_{}=]", name) or len(name.split()) > MAX_NAME_WORDS:
            dropped.append(f"{symbol.latex}: {name!r}")
            continue
        if vocabulary is not None and (unknown := ungrounded(name)):
            dropped.append(f"{symbol.latex}: {name!r} ({', '.join(unknown)} not in the paper)")
            continue
        names[symbol.key] = (symbol.latex, name)
    return names, dropped


def names_for(latex: str, glossary: dict[tuple, tuple[str, str]]) -> dict[str, str]:
    """The glossary names of the symbols in a piece of LaTeX (for one block's "names")."""
    out = {}
    for written in symbol_spans(latex):
        entry = glossary.get(glossary_key(written))
        if entry:
            out[entry[0]] = entry[1]
    return out
