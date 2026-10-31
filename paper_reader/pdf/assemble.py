"""Elements (PDF text layer or OCR) -> Paper, with the arXiv HTML path's conventions (see models.py).

Shared by both extractors, so it relies only on the Element contract (elements.py):

* Sections are flattened in reading order. A heading's number is split from its title ("3.1",
  "II-A"); its level comes from Element.level or from the numbering. IEEE numbering is composed
  the way LaTeXML writes it: "II" + "A." -> "II-A", "III-A" + "1)" -> "III-A1". Headings of level 4
  (run-in paragraph titles) become heading_note blocks.
* The abstract becomes Section(id="abstract", kind="abstract"), from an "Abstract" heading or an
  inline "Abstract—…" paragraph. "Index Terms—"/"Keywords:" paragraphs are dropped.
* References, footnotes, page headers/footers and back matter (acknowledgements, funding, …,
  as in parse_html) are always dropped; the sub-sections of dropped back matter too.
* The appendix starts at an "Appendix…"/"Supplementary…" heading or at the first section after
  the references; it is skipped unless include_appendix, else kind="appendix".
* Content before the first section is kept only if it is a float or a real paragraph, as in
  parse_html, and moved into the next section.
* Captions are paired with the nearest figure/table element on the same page (preferring the
  matching kind) and keyed "F3"/"T1"/"ALG1"; floats without a caption get a page-based id.
* Inline text: "$LaTeX$" (and "\\(…\\)") is maths, a literal dollar is "\\$". Numeric citations
  ("[12]", "[3, 7–9]"; never "[0, 1]", numbers beyond the reference list, or a bracket after
  "∈"/"="), author-year citations ("(Smith et al., 2020)", "Smith et al. (2020)") and figure
  mentions ("Fig. 3", "Table II") are marked up LaTeXML-style and rendered by parse_html's
  whitelist, so Block.html/text/fig_refs follow exactly the arXiv path's conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from typing import Literal

from ..models import Block, Figure, Paper, Section, SectionKind
from ..parse_html import (
    ABSTRACT_TITLE,
    APPENDIX_TITLE,
    BACKMATTER_TITLE,
    BIBLIOGRAPHY_TITLE,
    author_names,
    clean_latex,
    inline_fragment,
    norm_text,
    sanitize_table_html,
)
from .elements import Element

_MIN_LOOSE_PARAGRAPH_CHARS = 200  # as parse_html: shorter loose text before the first section is front matter
_SKIPPED = {"title", "authors", "caption", "reference", "footnote", "page_header", "page_footer"}

_HEADING_NUMBER = re.compile(
    r"^(?:(?P<appendix>(?i:appendix|appendices))(?:\s+(?P<anum>[A-Z](?:\.\d+)*))?\s*[.:—–-]?\s*"
    r"|(?P<num>\d+(?:\.\d+)*\.?|[IVXLC]+(?:-[A-Z]\d*)?\.|[IVXLC]+-[A-Z]\d*|[A-Z](?:\.\d+)+\.?|[A-Z]\.|\d+\)|[a-z]\)"
    r"|[A-Z](?=\s+[A-Z]))\s+)(?P<title>.*)$"
)
_ROMAN = re.compile(r"^[IVXLC]+$")
_INLINE_ABSTRACT = re.compile(r"^(?:abstract|ABSTRACT|Abstract)\s*(?:[—–:.\-]\s*|\s+(?=[A-Z]))")
_KEYWORDS = re.compile(r"^(?:index\s+terms|keywords|key\s+words|ccs\s+concepts)\s*[—–:.\-]", re.I)
_BULLET = re.compile(r"^[•∙●◦○▪■□‣⁃◆◇►▸➢✓✔*–—-]\s*")
_CAPTION_LABEL = re.compile(
    r"^(?P<word>Fig(?:ure)?|FIG(?:URE)?|Table|TABLE|Tab|Algorithm|ALGORITHM)\.?\s?"
    r"(?P<num>[A-Z]?\d{1,3}(?:\.\d{1,2})?|[IVXLC]{1,6})\s*[.:|—–-]?\s*"
)

# Inline maths: $$…$$ (display inside text), $…$ and \(…\); "\$" is a literal dollar.
_MATH = re.compile(r"(?<!\\)\$\$(?P<display>.+?)(?<!\\)\$\$|(?<!\\)\$(?P<inline>[^$]+?)(?<!\\)\$|\\\((?P<paren>.+?)\\\)")
_YEAR = r"(?:19|20)\d{2}[a-z]?"
_NAME = r"(?:(?:van|von|de|der|den|da|di|del|du|la|le)\s+)*[A-Z][\w'’\-]+"
_INLINE_MARKS = re.compile(
    rf"(?P<textual>{_NAME}(?:(?:,\s*{_NAME})*,?\s+(?:and|&)\s+{_NAME}|\s+et\s+al\.?)?\s*\(\s*{_YEAR}(?:\s*[,;]\s*{_YEAR})*\s*\))"
    rf"|(?P<paren>\([^()]*?{_YEAR}(?:\s*;[^();]*?{_YEAR})*\))"
    r"|(?P<numeric>\[\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?(?:\s*[,;]\s*\d{1,4}(?:\s*[-–—]\s*\d{1,4})?)*\s*\])"
    r"|(?P<figref>\b(?:Fig(?:ure)?s?|FIG(?:URE)?S?|Tab(?:le)?s?|TABLE|Alg(?:orithm)?s?)\.?\s?(?P<refnum>[A-Z]?\d{1,3}\b|[IVXLC]{1,6}\b))"
)
_CITATION_PART = re.compile(rf"^(?:(?:e\.g\.|see(?:\s+also)?|cf\.),?\s+)?(?P<names>[^\d()]{{2,120}}?),?\s+{_YEAR}(?:\s*,\s*{_YEAR})*$")
_NOT_AUTHOR = {
    "The", "This", "That", "These", "Those", "In", "On", "At", "As", "For", "From", "Since", "Until", "See",
    "We", "Our", "It", "Its", "And", "Or", "But", "Of", "To", "By", "With", "Section", "Table", "Figure", "Fig",
    "Eq", "Equation", "Appendix", "Algorithm", "Year", "Version", "Release", "Published",
}
_AUTHOR_MARKS = re.compile(r"\$[^$]*\$|\^\{[^}]*\}|\^\S+")  # "Li $^{1,2}$", "Li^1": affiliation markers
_SMALL_WORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "via", "with"}


# --------------------------------------------------------------------------- numbering


def _roman_value(roman: str) -> int:
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    for char, nxt in zip(roman, roman[1:] + " "):
        value = values[char]
        total += -value if values.get(nxt, 0) > value else value
    return total


@dataclass
class _Heading:
    number: str | None
    title: str
    level: int


def _parse_heading(text: str, level: int | None, letters_allowed: bool, top: str | None) -> _Heading:
    """Split "3.1 Policy" / "II-A Foo" / "1) Bar" / "Appendix B: Baz"; infer the level if not given.

    `top` is the current top-level number ("" if unnumbered, None before the first section): a Roman
    numeral is a new section, not an IEEE subsection letter such as "C.", when its title is in
    capitals, when it opens the paper, or when it follows `top`.
    """
    text = norm_text(text)
    m = _HEADING_NUMBER.match(text)
    if m is None or (m["num"] and re.fullmatch(r"[A-Z]", m["num"]) and not letters_allowed) or not m["title"] and not m["appendix"]:
        return _Heading(None, _title_case(text.rstrip(".:")), level or 1)
    raw_title = m["title"].rstrip(".:").strip()
    title = _title_case(raw_title)
    if m["appendix"]:
        return _Heading(m["anum"], title or "Appendix", level or 1)
    number = m["num"].rstrip(".")
    if level is None:
        if number.endswith(")"):
            level = 3 if number[0].isdigit() else 4
        elif _ROMAN.match(number) and (
            _caps_title(raw_title) or top is None
            or bool(_ROMAN.match(top)) and _roman_value(number) == _roman_value(top) + 1
        ):
            level = 1
        elif "-" in number:
            level = 2 + bool(re.search(r"\d$", number))
        elif re.fullmatch(r"[A-Z]", number) and m["num"].endswith("."):
            level = 2
        else:
            level = number.count(".") + 1
    return _Heading(number.rstrip(")"), title, level)


def _caps_title(title: str) -> bool:
    letters = [c for c in title if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)


def _title_case(title: str) -> str:
    """Small caps come out of PDFs as capitals: "RELATED WORK" -> "Related Work"."""
    if not _caps_title(title):
        return title
    words = re.split(r"(\s+)", title.lower())
    out = []
    for i, word in enumerate(words):
        if i and word in _SMALL_WORDS and not "".join(out).rstrip().endswith(":"):
            out.append(word)
        else:
            out.append(re.sub(r"[^\W\d_]+", lambda m: m.group().capitalize(), word, count=1))
    return "".join(out)


class _Numbering:
    """Composes IEEE-style numbers from the enclosing headings ("II" + "A" -> "II-A")."""

    def __init__(self) -> None:
        self.by_level: dict[int, str | None] = {}

    def top(self) -> str | None:
        """The current top-level number: "" under an unnumbered section, None before the first one."""
        return (self.by_level[1] or "") if 1 in self.by_level else None

    def compose(self, heading: _Heading) -> str | None:
        number, level = heading.number, heading.level
        parent = self.by_level.get(level - 1)
        if number and parent:
            if re.fullmatch(r"[A-Z]", number) and _ROMAN.match(parent):
                number = f"{parent}-{number}"
            elif re.fullmatch(r"\d+", number) and re.fullmatch(r"(?:[IVXLC]+-)?[A-Z]", parent):
                number = f"{parent}{number}"
        self.by_level = {lv: n for lv, n in self.by_level.items() if lv < level}
        self.by_level[level] = number
        return number


# --------------------------------------------------------------------------- inline text


class _Inline:
    """Plain extractor text -> (sanitized html, speech text, figure refs) via parse_html's whitelist."""

    def __init__(self, figures: dict[tuple[str, str], str], max_reference: int, warnings: list[str]):
        self.figures = figures  # (kind, number) -> figure id
        self.max_reference = max_reference
        self.warnings = warnings

    def __call__(self, text: str) -> tuple[str, str, list[str]]:
        parts: list[str] = []
        pos = 0
        for m in _MATH.finditer(text):
            parts.append(self._marks(text[pos : m.start()]))
            latex = m["display"] or m["inline"] or m["paren"]
            display = ' display="block"' if m["display"] else ""
            parts.append(f'<math alttext="{escape(latex)}"{display}></math>')
            pos = m.end()
        parts.append(self._marks(text[pos:]))
        return inline_fragment("".join(parts), figure_ids=set(self.figures.values()), warnings=self.warnings)

    def _marks(self, text: str) -> str:
        """Escape plain text, marking up citations and figure mentions."""
        text = text.replace(r"\$", "$")
        out: list[str] = []
        pos = 0
        for m in _INLINE_MARKS.finditer(text):
            markup = self._markup(m, text)
            if markup is None:
                continue
            out.append(escape(text[pos : m.start()], quote=False))
            out.append(markup)
            pos = m.end()
        out.append(escape(text[pos:], quote=False))
        return "".join(out)

    def _markup(self, m: re.Match, text: str) -> str | None:
        visible = escape(m.group(), quote=False)
        if m["textual"]:
            if m["textual"].split()[0].rstrip(",") in _NOT_AUTHOR:
                return None
            return f'<cite class="ltx_citemacro_citet">{visible}</cite>'
        if m["paren"]:
            parts = m["paren"][1:-1].split(";")
            if not all((p := _CITATION_PART.match(part.strip())) and re.search(r"[A-Z][a-z]", p["names"]) for part in parts):
                return None
            return f'<cite class="ltx_citemacro_citep">{visible}</cite>'
        if m["numeric"]:
            numbers = [int(n) for n in re.findall(r"\d+", m["numeric"])]
            before = text[: m.start()].rstrip()[-1:]
            if 0 in numbers or (self.max_reference and max(numbers) > self.max_reference) or before in "∈=<>≤≥⊂⊆×":
                return None
            return f"<cite>{visible}</cite>"
        word = m["figref"].lower()
        kind = "table" if word.startswith("tab") else "algorithm" if word.startswith("alg") else "figure"
        fig_id = self.figures.get((kind, m["refnum"]))
        return f'<a href="#{escape(fig_id)}">{visible}</a>' if fig_id else None


# --------------------------------------------------------------------------- floats


def _caption_parts(text: str) -> tuple[str, str | None, str, str]:
    """(kind, number, label, caption text without the label) of "Fig. 3. Foo" / "TABLE II Bar"."""
    m = _CAPTION_LABEL.match(norm_text(text))
    if m is None:
        return "figure", None, "Figure", norm_text(text)
    word = m["word"].lower()
    kind = "table" if word.startswith("tab") else "algorithm" if word.startswith("alg") else "figure"
    label = f"{kind.capitalize()} {m['num']}"
    return kind, m["num"], label, norm_text(text[m.end() :] if text.startswith(m.group()) else norm_text(text)[m.end() :])


def _pair_captions(elements: list[Element]) -> dict[int, int]:
    """float element index -> caption element index: same page, matching kind first, then nearest."""
    floats = [i for i, e in enumerate(elements) if e.type in ("figure", "table")]
    pairs: dict[int, int] = {}
    for c, caption in enumerate(elements):
        if caption.type != "caption":
            continue
        kind = _caption_parts(caption.text)[0]
        wanted = "table" if kind == "table" else "figure"
        candidates = [f for f in floats if f not in pairs and elements[f].page == caption.page]
        if candidates:
            best = min(candidates, key=lambda f: (elements[f].type != wanted, _distance(elements[f], caption, f, c), abs(f - c)))
            pairs[best] = c
    return pairs


def _distance(a: Element, b: Element, ia: int, ib: int) -> float:
    if a.bbox and b.bbox:
        ax0, ay0, ax1, ay1 = a.bbox
        bx0, by0, bx1, by1 = b.bbox
        vertical = max(0.0, by0 - ay1, ay0 - by1)
        horizontal = max(0.0, bx0 - ax1, ax0 - bx1)
        return vertical + 2 * horizontal
    return abs(ia - ib) * 100.0


@dataclass
class _Floats:
    figures: dict[str, Figure] = field(default_factory=dict)
    by_element: dict[int, str] = field(default_factory=dict)  # float element index -> figure id
    lookup: dict[tuple[str, str], str] = field(default_factory=dict)  # (kind, number) -> figure id
    captions: dict[str, str] = field(default_factory=dict)  # figure id -> raw caption text


def _collect_floats(elements: list[Element], warnings: list[str]) -> _Floats:
    out = _Floats()
    pairs = _pair_captions(elements)
    uncaptioned: dict[int, int] = {}
    for i, element in enumerate(elements):
        if element.type not in ("figure", "table"):
            continue
        if i in pairs:
            kind, number, label, caption = _caption_parts(elements[pairs[i]].text)
            arabic = str(_roman_value(number)) if number and _ROMAN.match(number) else number
            fig_id = {"figure": "F", "table": "T", "algorithm": "ALG"}[kind] + re.sub(r"\W", "-", arabic or "")
        else:
            kind, number, caption = element.type, None, ""
            label, arabic = kind.capitalize(), None
            uncaptioned[element.page] = uncaptioned.get(element.page, 0) + 1
            fig_id = f"{'F' if kind == 'figure' else 'T'}-p{element.page + 1}-{uncaptioned[element.page]}"
        base, n = fig_id, 1
        while fig_id in out.figures:
            n += 1
            fig_id = f"{base}-{n}"
        table_html = sanitize_table_html(element.html) if element.html else None
        out.figures[fig_id] = Figure(
            id=fig_id, kind="table" if kind == "table" else "figure", number=number, label=label,
            images=[element.image] if element.image else [], table_html=table_html,
        )
        out.by_element[i] = fig_id
        out.captions[fig_id] = caption
        for num in {number, arabic} - {None}:
            out.lookup.setdefault((kind, num), fig_id)
    orphans = len([e for e in elements if e.type == "caption"]) - len(pairs)
    if orphans:
        warnings.append(f"Skipped {orphans} caption(s) without a figure or table next to them")
    return out


# --------------------------------------------------------------------------- sections


@dataclass
class _State:
    include_appendix: bool
    warnings: list[str]
    sections: list[Section] = field(default_factory=list)
    current: Section | None = None  # None: content is dropped (or not yet in a section)
    started: bool = False  # a section (or the abstract) has been opened
    in_appendix: bool = False
    after_bibliography: bool = False
    homeless: list[tuple[str, Section | None]] = field(default_factory=list)  # (float, last body section)
    dropped_level: int | None = None  # level of a dropped back-matter section
    pending: list[Block] = field(default_factory=list)  # content before the first section
    numbering: _Numbering = field(default_factory=_Numbering)

    def open(self, title: str, *, number: str | None = None, level: int = 1, kind: SectionKind = "body") -> None:
        sec_id = "abstract" if kind == "abstract" else f"S{sum(s.kind != 'abstract' for s in self.sections) + 1}"
        self.current = Section(id=sec_id, number=number, title=title, level=level, kind=kind)
        if self.pending and kind != "abstract":
            self.current.blocks.extend(self._front_matter_filtered())
        self.sections.append(self.current)
        self.started = True

    def add(self, block: Block) -> None:
        if self.current is not None:
            self.current.blocks.append(block)
        elif not self.started:
            self.pending.append(block)
        elif block.figure_id and not self.in_appendix:  # a body float LaTeX placed after e.g. the acknowledgements
            self.homeless.append((block.figure_id, self.sections[-1] if self.sections else None))

    def has_abstract(self) -> bool:
        return any(s.kind == "abstract" for s in self.sections)

    def _front_matter_filtered(self) -> list[Block]:
        kept, dropped = [], []
        for block in self.pending:
            if block.type == "figure" or len(block.text) >= _MIN_LOOSE_PARAGRAPH_CHARS:
                kept.append(block)
            else:
                dropped.append(block.text[:40])
        if dropped:
            self.warnings.append(f"Dropped front-matter fragment(s) outside any section: {dropped}")
        self.pending = []
        return kept


def _heading(state: _State, element: Element) -> Block | None:
    """Open/close sections for a heading; returns a heading_note block for level-4 headings."""
    letters_allowed = state.in_appendix or state.after_bibliography
    heading = _parse_heading(element.text, element.level, letters_allowed, state.numbering.top())
    title = heading.title
    if heading.level >= 4:
        return Block(id="", type="heading_note", text=norm_text(element.text).rstrip(".:"))
    if BIBLIOGRAPHY_TITLE.match(title):
        state.after_bibliography = True
        state.current, state.started = None, True
        return None
    if ABSTRACT_TITLE.match(title) and not state.in_appendix and not state.has_abstract():
        state.dropped_level = None
        state.open("Abstract", kind="abstract")
        return None
    appendix_title = bool(APPENDIX_TITLE.match(title) or APPENDIX_TITLE.match(norm_text(element.text)))
    if not state.in_appendix and (appendix_title or state.after_bibliography):
        state.in_appendix = True
        state.dropped_level = None
        if not appendix_title:
            state.warnings.append(f"Treating sections from {title!r} on (after the references) as appendix")
    number = state.numbering.compose(heading)
    if state.dropped_level is not None and heading.level > state.dropped_level:
        state.current = None
        return None
    state.dropped_level = None
    if BACKMATTER_TITLE.match(title):
        state.dropped_level = heading.level
        state.current, state.started = None, True
        return None
    if state.in_appendix and not state.include_appendix:
        state.current, state.started = None, True
        return None
    state.open(title, number=number, level=min(heading.level, 3), kind="appendix" if state.in_appendix else "body")
    return None


def assemble(
    elements: list[Element],
    *,
    paper_id: str,
    source: Literal["pdf-text", "pdf-ocr"],
    source_url: str,
    include_appendix: bool = False,
    arxiv_id: str | None = None,
    title: str | None = None,
) -> Paper:
    """Build a Paper from extracted elements (reading order). See the module docstring for the rules."""
    warnings: list[str] = []
    floats = _collect_floats(elements, warnings)
    state = _State(include_appendix, warnings)
    for i, element in enumerate(elements):
        _place(state, element, floats.by_element.get(i))
    if state.pending:
        warnings.append("Content outside any section was collected into a 'Main text' section")
        state.started = False
        state.open("Main text")
    sections = [s for s in state.sections if s.blocks or s.kind != "abstract"]
    if not any(s.kind != "abstract" for s in sections):
        warnings.append("No body sections found")

    # Text is rendered once the kept floats are known: mentions of dropped ones are not links.
    placed = {b.figure_id for s in sections for b in s.blocks if b.figure_id}
    placed.update(fig_id for fig_id, _ in state.homeless)
    lookup = {key: fig_id for key, fig_id in floats.lookup.items() if fig_id in placed}
    rich = _Inline(lookup, sum(e.type == "reference" for e in elements), warnings)
    for section in sections:
        for block in section.blocks:
            _render(block, rich)
    _rehome(sections, state.homeless)
    for fig_id, caption in floats.captions.items():
        figure = floats.figures[fig_id]
        figure.caption_html, figure.caption_text, _ = rich(caption)

    n = 0
    figures: dict[str, Figure] = {}
    for section in sections:
        for block in section.blocks:
            n += 1
            block.id = f"b{n:04d}"
            if block.figure_id:
                figures[block.figure_id] = floats.figures[block.figure_id]
                figures[block.figure_id].section_id = section.id

    authors: list[str] = []
    for element in elements:
        if element.type == "authors":
            line = _AUTHOR_MARKS.sub(" ", element.text.replace(r"\$", ""))
            authors.extend(name for name in author_names(line) if name not in authors)
    title_element = next((e for e in elements if e.type == "title"), None)
    paper_title = _title_case(norm_text(title or (title_element.text.replace(r"\$", "$") if title_element else "")))
    if not paper_title:
        warnings.append("No document title found")
    m = re.fullmatch(r"(?P<id>.+?)(?P<version>v\d+)?", arxiv_id or "")
    return Paper(
        paper_id=paper_id,
        arxiv_id=m["id"] if m else "",
        version=m["version"] if m else None,
        title=paper_title or "Untitled",
        authors=authors,
        source_url=source_url,
        source=source,
        include_appendix=include_appendix,
        sections=sections,
        figures=figures,
        warnings=warnings,
    )


def _place(state: _State, element: Element, figure_id: str | None) -> None:
    """Route one element into the section structure; text blocks keep their raw text until _render."""
    kind = element.type
    if kind in _SKIPPED:
        return
    if kind == "heading":
        note = _heading(state, element)
        if note is not None:
            state.add(note)
    elif kind in ("figure", "table"):
        state.add(Block(id="", type="figure", figure_id=figure_id, speak=False))
    elif kind == "equation":
        state.add(_equation(element))
    else:
        text = norm_text(element.text)
        if not state.has_abstract() and state.current is None and (m := _INLINE_ABSTRACT.match(text)):
            state.open("Abstract", kind="abstract")
            text = text[m.end() :]
        if kind == "list_item":
            text = _BULLET.sub("", text)
        if text and not _KEYWORDS.match(text):
            state.add(Block(id="", type="list_item" if kind == "list_item" else "paragraph", text=text))


def _render(block: Block, rich: _Inline) -> None:
    if block.type not in ("paragraph", "list_item", "heading_note"):
        return
    html, text, refs = rich(block.text)
    if block.type == "heading_note":
        html, text = f"<strong>{html}</strong>", text if text[-1:] in ".:!?" else f"{text}."
    block.html, block.text, block.fig_refs = html, text, refs


def _rehome(sections: list[Section], homeless: list[tuple[str, Section | None]]) -> None:
    """Body floats found inside dropped back matter go to the end of the section that first mentions
    them, else of the last body section before the back matter."""
    for fig_id, before in homeless:
        home = next((s for s in sections if any(fig_id in b.fig_refs for b in s.blocks)), before)
        if home is not None and home in sections:
            home.blocks.append(Block(id="", type="figure", figure_id=fig_id, speak=False))


def _equation(element: Element) -> Block:
    number = (element.eq_number or "").strip("() ") or None
    if element.latex:
        latex = re.sub(r"^\s*(?:\$\$|\\\[|\$)|(?:\$\$|\\\]|\$)\s*$", "", element.latex)
        if tag := re.search(r"\\tag\*?\{([^}]*)\}", latex):
            number = number or tag[1]
            latex = latex.replace(tag.group(), "")
        latex = clean_latex(latex)
        return Block(id="", type="equation", html=f'<div class="equation">\\[ {escape(latex, quote=False)} \\]</div>',
                     text=f"$${latex}$$", latex=latex, eq_number=number)
    raw = norm_text(element.text.replace(r"\$", "$"))
    return Block(id="", type="equation", html=f'<div class="equation">{escape(raw, quote=False)}</div>',
                 text=raw, eq_number=number)
