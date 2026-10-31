"""PDF text layer -> Elements: the fast path for born-digital PDFs (see elements.py for the contract).

How a PDF is read
-----------------
1. Every page's spans (PyMuPDF "dict", ligatures expanded), images and vector drawings are read.
2. Noise is dropped: running headers/footers (the same text in the top/bottom band of several
   pages), page numbers, rotated text (the arXiv side stamp, rotated axis labels) and the
   line-number margins of review copies.
3. Columns: a vertical gutter free of body text splits a page into a left and a right column;
   lines crossing it are full width. The document's gutter is used for every page that is not
   clearly single-column.
4. Rows: spans are clustered by baseline within a column, so sub- and superscripts join their line.
5. Floats: a caption ("Fig. 3.", "Figure 3:", "TABLE I", "Algorithm 1:") grows a region over the
   graphics and non-body text next to it (above for figures, below for tables), stopping at body
   text. The region is rendered at 2x to figures/ and its rows leave the text flow. Large
   graphics without a caption become uncaptioned figures.
6. Reading order: full-width items split a page into bands; each band is read left column, then
   right column, top to bottom.
7. Blocks: rows become headings (size/weight + numbering patterns), list items, display equations
   (centred maths, or an "(n)" at the right margin), footnotes and paragraphs (lines joined and
   de-hyphenated against the document's own vocabulary). Rows above the abstract on page 1 are
   the title and author lines. Paragraphs that clearly continue across a column, page or float
   are joined; entries after a "References" heading become reference elements.

Text uses the contract's conventions: inline maths cannot be recovered as LaTeX, so a literal "$"
is escaped as "\\$" and display equations keep their raw text with ``latex=None``.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

from ..parse_html import ABSTRACT_TITLE, BIBLIOGRAPHY_TITLE, norm_text
from .elements import Element

ProgressCallback = Callable[[float, str], None]

_TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_LIGATURES & ~pymupdf.TEXT_PRESERVE_IMAGES
_RENDER_ZOOM = 2.0  # figure crops are rendered at 2x (144 dpi)
_FIGURE_DIR = "figures"

# Fonts: TeX maths fonts (cmmi/cmsy/cmex...), STIX/Cambria/MathJax maths, symbol fonts.
MATH_FONT = re.compile(
    r"^(?:cm(?:mi|sy|ex|bsy|mib)|msam|msbm|eu[frs][mb]|rsfs|stmary|wasy|esint|(?:tx|px)(?:sy|ex|mi))"
    r"|math|stix|symbol|mt(?:mi|sy|ex)\d|euler",
    re.I,
)
_BOLD_FONT = re.compile(r"bold|black|heavy|semibold|demi|medi|cmbx|cmb\d", re.I)
_ITALIC_FONT = re.compile(r"ital|oblique|cmti|cmsl|-(?:it|bi|boldit)$", re.I)
_BIG_OPERATOR_FONT = re.compile(r"^cmex|largesymbols", re.I)

_DROP_CHARS = re.compile("[\x00-\x08\x0b-\x1f\x7f�-]")
_SPACING_ACCENTS = {"¨": "̈", "´": "́", "ˆ": "̂", "˜": "̃", "¸": "̧",
                    "ˇ": "̌", "˘": "̆", "¯": "̄", "˙": "̇", "˚": "̊"}
_ACCENT_BEFORE_LETTER = re.compile(rf"([{''.join(_SPACING_ACCENTS)}])([A-Za-z])")

ARXIV_STAMP = re.compile(r"arXiv:\s?\S+\s+\[[\w.\-]+\]\s+\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4}")
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?(?:\d{1,4}|[ivxlc]{1,6})(?:\s*(?:/|of)\s*\d{1,4})?$", re.I)
_CAPTION = re.compile(
    r"^(?P<label>(?:Fig(?:ure)?|FIG(?:URE)?|Table|TABLE|Tab|Algorithm|ALGORITHM)\.?\s?"
    r"(?:[A-Z]?\d{1,3}(?:\.\d{1,2})?|[IVXLC]{1,6}))(?:\s*[.:|—–-]|$)"
)
_EQ_NUMBER = re.compile(r"^\((\d{1,3}(?:\.\d{1,3})?[a-z]?|[A-Z]\.?\d{1,3}(?:\.\d{1,3})?)\)$")
_NUMBERED_HEADING = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>\S.*)$")
_ROMAN_HEADING = re.compile(r"^(?P<num>[IVX]{1,5})\.\s+(?P<title>\S.*)$")
_LETTER_HEADING = re.compile(r"^(?P<num>[A-Z](?:\.\d{1,2}){0,3})(?P<dot>\.?)\s+(?P<title>\S.*)$")
_APPENDIX_HEADING = re.compile(r"^(?:appendix|appendices|supplementary material)\b", re.I)
_ABSTRACT_START = re.compile(r"^abstract\b", re.I)
_RUN_IN_NUMBER = re.compile(r"^(?:(?P<sub>\d{1,2})|(?P<par>[a-z]))\)\s")
_LIST_MARKER = re.compile(
    r"^(?:[•∙●◦○▪■□‣⁃◆◇►▸➢✓✔]|\(\d{1,2}\)|\d{1,2}[.)]|\([a-z]\)|[a-z]\)|\([ivx]{1,4}\)|[ivx]{1,4}\))(?:\s|$)"
)
_DASH_ITEM = re.compile(r"^[–—-]\s")
_FOOTNOTE_MARK = re.compile(r"^[*†‡§¶∗⋆#\d]")
_REFERENCE_MARK = re.compile(r"^(?:\[\d{1,4}\]|\d{1,4}\.\s)")
_SENTENCE_END = re.compile(r"[.!?:]['’”)\]]*$")
_HYPHEN_PREFIXES = {"non", "multi", "self", "semi", "pseudo", "quasi", "cross"}
_WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


def clean_text(text: str) -> str:
    """Drop control/private-use glyphs and compose TeX's spacing accents ("W¨unsch" -> "Wünsch")."""
    text = _DROP_CHARS.sub("", text)
    return _ACCENT_BEFORE_LETTER.sub(
        lambda m: unicodedata.normalize("NFC", m[2] + _SPACING_ACCENTS[m[1]]), text
    )


def is_math_char(char: str) -> bool:
    code = ord(char)
    return (
        0x0370 <= code <= 0x03FF  # Greek
        or 0x2190 <= code <= 0x22FF  # arrows, mathematical operators
        or 0x27C0 <= code <= 0x27EF
        or 0x2980 <= code <= 0x2AFF
        or 0x1D400 <= code <= 0x1D7FF  # mathematical alphanumerics
    )


# --------------------------------------------------------------------------- page content


@dataclass(eq=False)
class _Span:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    baseline: float
    bold: bool
    italic: bool
    math: bool
    big: bool  # a big operator/delimiter glyph: its baseline says little about its line


def _span(raw: dict) -> _Span | None:
    text = _DROP_CHARS.sub("", raw["text"])
    if not text.strip():
        return None
    font = raw["font"].split("+")[-1]
    flags = raw["flags"]
    math = bool(MATH_FONT.search(font))
    x0, y0, x1, y1 = raw["bbox"]
    return _Span(
        text=text, x0=x0, y0=y0, x1=x1, y1=y1, size=round(raw["size"], 2), baseline=raw["origin"][1],
        bold=bool(flags & 16 or _BOLD_FONT.search(font)),
        italic=bool(flags & 2 or _ITALIC_FONT.search(font)),
        math=math, big=bool(_BIG_OPERATOR_FONT.search(font)),
    )


@dataclass(eq=False)
class _Line:
    """A PyMuPDF line: the unit for noise removal and column assignment."""

    spans: list[_Span]

    @property
    def x0(self) -> float:
        return min(s.x0 for s in self.spans)

    @property
    def x1(self) -> float:
        return max(s.x1 for s in self.spans)

    @property
    def y0(self) -> float:
        return min(s.y0 for s in self.spans)

    @property
    def y1(self) -> float:
        return max(s.y1 for s in self.spans)

    @property
    def text(self) -> str:
        return norm_text(" ".join(s.text for s in self.spans))

    @property
    def size(self) -> float:
        return max(s.size for s in self.spans)

    @property
    def baseline(self) -> float:
        return max(self.spans, key=lambda s: s.size).baseline


@dataclass(eq=False)
class _Page:
    index: int
    rect: pymupdf.Rect
    lines: list[_Line]
    graphics: list[pymupdf.Rect]  # images and clusters of vector drawings
    rules: list[pymupdf.Rect]  # thin horizontal lines (table rules, footnote rules)
    rotated: list[pymupdf.Rect]  # rotated text (axis titles; the arXiv stamp)


def _read_page(page: pymupdf.Page) -> _Page:
    lines: list[_Line] = []
    rotated: list[pymupdf.Rect] = []
    for block in page.get_text("dict", flags=_TEXT_FLAGS)["blocks"]:
        for line in block.get("lines", ()):
            dx, dy = line["dir"]
            if dx <= 0 or abs(dy) > 0.02:  # not part of the text flow
                rotated.append(pymupdf.Rect(line["bbox"]))
                continue
            spans = [s for s in map(_span, line["spans"]) if s is not None]
            if spans:
                lines.append(_Line(spans))
    area = page.rect.get_area()
    drawings = page.get_drawings()
    graphics = []
    for rect in [pymupdf.Rect(i["bbox"]) for i in page.get_image_info()] + page.cluster_drawings(drawings=drawings):
        rect &= page.rect
        if rect.width >= 8 and rect.height >= 8 and rect.get_area() < 0.7 * area:  # not a scan/background
            graphics.append(rect)
    rules = [d["rect"] for d in drawings if d["rect"].height <= 2 and d["rect"].width >= 15]
    return _Page(page.number, page.rect, lines, graphics, rules, rotated)


def _drop_noise(pages: list[_Page]) -> None:
    """Remove running headers/footers, page numbers, arXiv stamps and line-number margins (not inside figures)."""
    def band_key(page: _Page, line: _Line) -> str | None:
        height = page.rect.height
        if line.y1 < 0.1 * height:
            band = "top"
        elif line.y0 > 0.9 * height:
            band = "bottom"
        else:
            return None
        return band + ":" + re.sub(r"\d+", "#", line.text.lower())

    counts = Counter(k for page in pages for k in {band_key(page, line) for line in page.lines} if k)
    repeated = max(2, round(0.25 * len(pages)))
    for page in pages:
        numbers = _line_numbers(page.lines)
        kept = []
        for line in page.lines:
            key = band_key(page, line)
            text = line.text
            in_figure = any(_inside(pymupdf.Rect(line.x0, line.y0, line.x1, line.y1), g) for g in page.graphics)
            if line in numbers or ARXIV_STAMP.match(text):
                continue
            repeated_words = counts[key] >= repeated and re.search(r"[A-Za-z]{3}", text)  # not "(2)", "(7)"
            if key and not in_figure and (repeated_words or _PAGE_NUMBER.match(text)):
                continue
            kept.append(line)
        page.lines = kept


def _line_numbers(lines: list[_Line]) -> set[_Line]:
    """Review-copy line numbers: >= 8 bare numbers, aligned on one x, increasing down the page."""
    columns: dict[int, list[_Line]] = {}
    for line in lines:
        if re.fullmatch(r"\d{1,4}", line.text):
            columns.setdefault(round(line.x1 / 4), []).append(line)
    found: set[_Line] = set()
    for column in columns.values():
        column.sort(key=lambda ln: ln.y0)
        values = [int(ln.text) for ln in column]
        if len(column) >= 8 and all(a < b for a, b in zip(values, values[1:])):
            found.update(column)
    return found


# --------------------------------------------------------------------------- rows


@dataclass(eq=False)
class _Row:
    """One visual line inside one column: spans sharing a baseline, sub/superscripts included."""

    spans: list[_Span]
    page: int
    region: str  # "L"/"R" (columns), "F" (full width on a two-column page), "S" (single column)
    baseline: float
    key_size: float
    text: str = ""
    size: float = 0.0  # dominant (character-weighted) font size
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    bold: bool = False
    italic: bool = False
    caps: bool = False
    math_frac: float = 0.0
    cells: list[tuple[float, float, str]] = field(default_factory=list)  # text runs split at wide gaps
    role: str = "flow"  # flow / front / float / caption / footnote
    body: bool = False  # looks like running body text (see _PageLayout._mark_body)
    eq_number: str | None = None
    display_math: bool = False
    seg_start: bool = False  # first row of a column segment in reading order

    def finish(self) -> None:
        self.spans.sort(key=lambda s: s.x0)
        self.x0 = min(s.x0 for s in self.spans)
        self.x1 = max(s.x1 for s in self.spans)
        self.y0 = min(s.y0 for s in self.spans)
        self.y1 = max(s.y1 for s in self.spans)
        sizes: Counter[float] = Counter()
        letters = bold = italic_letters = text_letters = upper = math_chars = chars = 0
        for s in self.spans:
            n = len(s.text.strip())
            sizes[round(s.size, 1)] += n
            alpha = sum(c.isalpha() for c in s.text)
            letters += alpha
            bold += alpha if s.bold else 0
            if not s.math:
                text_letters += alpha
                italic_letters += alpha if s.italic else 0
            upper += sum(c.isupper() for c in s.text)
            chars += n
            math_chars += n if s.math else sum(map(is_math_char, s.text))
        self.size = sizes.most_common(1)[0][0]
        self.bold = letters > 0 and bold >= 0.8 * letters
        self.italic = text_letters > 0 and italic_letters >= 0.8 * text_letters
        self.caps = letters >= 3 and upper >= 0.9 * letters
        self.math_frac = math_chars / max(chars, 1)
        self.cells = _cells(self.spans)
        self.text = _join_spans(self.spans)

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.x0, self.y0, self.x1, self.y1)


def _join_spans(spans: list[_Span]) -> str:
    out: list[str] = []
    prev: _Span | None = None
    for span in spans:
        if prev is not None and span.x0 - prev.x1 > 0.15 * min(span.size, prev.size):
            out.append(" ")
        out.append(span.text)
        prev = span
    return norm_text(clean_text("".join(out)))


def _cells(spans: list[_Span]) -> list[tuple[float, float, str]]:
    cells: list[list[_Span]] = []
    for span in spans:
        if cells and span.x0 - cells[-1][-1].x1 <= max(2 * span.size, 15):
            cells[-1].append(span)
        else:
            cells.append([span])
    return [(c[0].x0, max(s.x1 for s in c), _join_spans(c)) for c in cells]


def _make_row(spans: list[_Span], page: int, region: str) -> _Row:
    main = max((s for s in spans if not s.big), key=lambda s: s.size, default=spans[0])
    row = _Row(spans, page, region, main.baseline, main.size)
    row.finish()
    return row


def _cluster_rows(spans: list[_Span]) -> list[list[_Span]]:
    """Group one column's spans by baseline; sub/superscripts and big operators join the line they overlap."""
    rows: list[tuple[list[_Span], float, float, list[float]]] = []  # spans, baseline, size, [y0, y1]
    for span in sorted(spans, key=lambda s: (s.big, -s.size, s.baseline)):
        best, best_distance = None, 0.0
        height = max(span.y1 - span.y0, 0.1)
        for row in rows:
            _, baseline, size, (y0, y1) = row
            distance = abs(baseline - span.baseline)
            overlap = min(span.y1, y1) - max(span.y0, y0)
            small = span.size < 0.85 * size
            joins = (not span.big and distance <= (0.6 if small else 0.5) * size) or (
                span.size <= size and overlap >= (0.3 if span.big else 0.4 if small else 0.6) * height
            )
            if joins and (best is None or distance < best_distance):
                best, best_distance = row, distance
        if best is None:
            rows.append(([span], span.baseline, span.size, [span.y0, span.y1]))
        else:
            best[0].append(span)
            best[3][0], best[3][1] = min(best[3][0], span.y0), max(best[3][1], span.y1)
    return [row[0] for row in rows]


# --------------------------------------------------------------------------- document statistics


def _mode(values: list[float]) -> float | None:
    return Counter(round(v) for v in values).most_common(1)[0][0] if values else None


def _gutter(lines: list[_Line], width: float, body: float) -> tuple[float, float] | None:
    """(gutter x, share of body lines crossing it) if the page shows two columns of body text."""
    body_lines = [ln for ln in lines if ln.size >= body - 1.5]
    if len(body_lines) < 6:
        return None
    coverage = np.zeros(int(width) + 2)
    for line in body_lines:
        coverage[max(int(line.x0), 0) : int(line.x1) + 1] += 1
    lo, hi = int(0.35 * width), int(0.65 * width)
    x = lo + int(np.argmin(coverage[lo:hi]))
    if coverage[x] > 0.12 * len(body_lines):
        return None
    left, right = x, x
    while left > lo and coverage[left - 1] <= coverage[x]:
        left -= 1
    while right < hi and coverage[right + 1] <= coverage[x]:
        right += 1
    gutter = (left + right) / 2
    sides = Counter("L" if ln.x1 <= gutter else "R" if ln.x0 >= gutter else "F" for ln in body_lines)
    if sides["L"] < 0.2 * len(body_lines) or sides["R"] < 0.2 * len(body_lines):
        return None
    return gutter, sides["F"] / len(body_lines)


def _crossing_share(lines: list[_Line], gutter: float, width: float, body: float) -> float:
    body_lines = [ln for ln in lines if ln.size >= body - 1.5]
    reach = 0.15 * width
    crossing = sum(ln.x0 < gutter - reach and ln.x1 > gutter + reach for ln in body_lines)
    return crossing / max(len(body_lines), 1)


@dataclass
class _Doc:
    body: float  # body font size
    gutter: float | None  # the document's column gutter (None: single column)
    columns: dict[str, tuple[float, float]] = field(default_factory=dict)  # region -> typical (x0, x1)
    justified: bool = True
    words: set[str] = field(default_factory=set)
    hyphenated: set[str] = field(default_factory=set)


def _body_size(pages: list[_Page]) -> float:
    sizes: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            for span in line.spans:
                sizes[round(span.size * 2) / 2] += len(span.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _document(pages: list[_Page]) -> _Doc:
    body = _body_size(pages)
    gutters = []
    text_pages = 0
    for page in pages:
        if sum(ln.size >= body - 1.5 for ln in page.lines) >= 6:
            text_pages += 1
            if (own := _gutter(page.lines, page.rect.width, body)) is not None:
                gutters.append(own[0])
    gutter = statistics.median(gutters) if gutters and len(gutters) >= 0.4 * text_pages else None
    return _Doc(body=body, gutter=gutter)


def _measure_columns(doc: _Doc, layouts: list[_PageLayout]) -> None:
    """Typical column bounds per region over the document, and whether the text is justified."""
    per_region: dict[str, list[_Row]] = {}
    for layout in layouts:
        for row in layout.rows:
            if abs(row.size - doc.body) <= 1:
                per_region.setdefault(row.region, []).append(row)
    for region, rows in per_region.items():
        if region != "F" and len(rows) >= 4:
            doc.columns[region] = (_mode([r.x0 for r in rows]), _mode([r.x1 for r in rows]))
    flush = total = 0
    for region, rows in per_region.items():
        if region in doc.columns:
            x1 = doc.columns[region][1]
            total += len(rows)
            flush += sum(abs(r.x1 - x1) <= 2 for r in rows)
    doc.justified = total == 0 or flush >= 0.5 * total


# --------------------------------------------------------------------------- page layout


@dataclass(eq=False)
class _Float:
    kind: str  # "figure" | "table"
    page: int
    region: str
    rect: pymupdf.Rect | None  # None: a caption whose figure/table was not found
    caption: list[_Row] = field(default_factory=list)
    caption_above: bool = False
    image: str = ""

    @property
    def y0(self) -> float:
        return min([r.y0 for r in self.caption] + ([self.rect.y0] if self.rect else []))

    @property
    def y1(self) -> float:
        return max([r.y1 for r in self.caption] + ([self.rect.y1] if self.rect else []))


_Item = _Row | _Float


class _PageLayout:
    """Columns, rows, floats, footnotes and reading order of one page."""

    def __init__(self, page: _Page, doc: _Doc):
        self.page = page
        self.doc = doc
        width = page.rect.width
        self.gutter: float | None = None
        if doc.gutter is not None and _crossing_share(page.lines, doc.gutter, width, doc.body) < 0.3:
            self.gutter = doc.gutter
        elif doc.gutter is None and (own := _gutter(page.lines, width, doc.body)) and own[1] < 0.1:
            self.gutter = own[0]
        self.rows = self._rows()
        self._bounds = self._own_bounds()
        self.floats: list[_Float] = []

    # -- columns and rows

    def _rows(self) -> list[_Row]:
        by_region: dict[str, list[_Span]] = {}
        for spans in self._visual_lines():
            by_region.setdefault(self._region(spans), []).extend(spans)
        rows = [
            _make_row(group, self.page.index, region)
            for region, spans in by_region.items()
            for group in _cluster_rows(spans)
        ]
        return sorted(rows, key=lambda r: (r.baseline, r.x0))

    def _visual_lines(self) -> list[list[_Span]]:
        """PyMuPDF lines on one baseline a few points apart ("Fig. 3." + its text) are one line."""
        g = self.gutter
        lines = sorted(self.page.lines, key=lambda ln: ln.baseline)
        out: list[list[_Span]] = []
        bucket: list[_Line] = []
        for line in lines + [None]:
            if line is not None and bucket and abs(line.baseline - bucket[0].baseline) <= 1:
                bucket.append(line)
                continue
            merged: list[list[_Span]] = []
            for item in sorted(bucket, key=lambda ln: ln.x0):
                prev = merged[-1] if merged else None
                end = max(s.x1 for s in prev) if prev else 0.0
                if prev and 0 <= item.x0 - end <= 12 and not (g is not None and end <= g <= item.x0):
                    prev.extend(item.spans)
                else:
                    merged.append(list(item.spans))
            out.extend(merged)
            bucket = [line] if line is not None else []
        return out

    def _region(self, spans: list[_Span]) -> str:
        g = self.gutter
        if g is None:
            return "S"
        x0, x1 = min(s.x0 for s in spans), max(s.x1 for s in spans)
        if min(g - x0, x1 - g) >= 0.1 * self.page.rect.width:
            return "F"
        left = sum(max(0.0, min(s.x1, g) - s.x0) for s in spans)
        right = sum(max(0.0, s.x1 - max(s.x0, g)) for s in spans)
        return "L" if left >= right else "R"

    def bounds(self, region: str) -> tuple[float, float]:
        if region == "F":
            return self.bounds("L")[0], self.bounds("R")[1]
        return self._bounds.get(region) or self.doc.columns.get(region) or (self.page.rect.x0 + 50, self.page.rect.x1 - 50)

    def _own_bounds(self) -> dict[str, tuple[float, float]]:
        """Column bounds from this page's body-size rows (bounds() falls back to the document's)."""
        bounds = {}
        for region in "LRS":
            rows = [r for r in self.rows if r.region == region and abs(r.size - self.doc.body) <= 1]
            if len(rows) >= 4:
                bounds[region] = (_mode([r.x0 for r in rows]), _mode([r.x1 for r in rows]))
        return bounds

    def _mark_body(self) -> None:
        """Body rows: body-size, column-wide, no table-like gaps; a paragraph's short last line too."""
        for region in {r.region for r in self.rows}:
            x0, x1 = self.bounds(region)
            prev: _Row | None = None
            for row in sorted((r for r in self.rows if r.region == region), key=lambda r: r.baseline):
                sized = abs(row.size - self.doc.body) <= 1.2
                row.body = sized and row.width >= 0.6 * (x1 - x0) and len(row.cells) == 1
                if (
                    not row.body and sized and prev is not None and prev.body
                    and 0 < row.baseline - prev.baseline <= 1.6 * row.size
                    and x0 - 2 <= row.x0 <= prev.x0 + 2
                ):
                    row.body = True
                prev = row

    def _mark_equations(self) -> None:
        for row in self.rows:
            x0, x1 = self.bounds(row.region)
            last = row.cells[-1]
            number = _EQ_NUMBER.match(last[2])
            if number and last[1] >= x1 - 0.6 * row.size and (len(row.cells) > 1 or row.x0 > x0 + 0.5 * (x1 - x0)):
                row.eq_number = number[1]
            indented = row.x0 >= x0 + 1.5 * row.size and (row.eq_number or row.x1 <= x1 - 1.5 * row.size)
            row.display_math = indented and row.math_frac >= 0.3 and not row.caps

    # -- floats

    def find_floats(self, figure_dir: Path, render: Callable[[pymupdf.Rect, Path], None]) -> None:
        """Pair captions with float regions, render the crops and take their rows out of the text flow."""
        self._mark_body()
        self._mark_equations()
        captions = [r for r in self._sorted_rows() if self._caption_start(r)]
        claimed: list[pymupdf.Rect] = []
        for first in captions:
            if first.role != "flow":
                continue
            rows = self._caption_rows(first)
            label = _CAPTION.match(first.text)["label"]
            kind = "table" if label.lower().startswith("tab") else "figure"
            float_ = self._grow(rows, kind, claimed) or _Float(kind, self.page.index, first.region, None, rows)
            self.floats.append(float_)
            if float_.rect is not None:
                claimed.append(float_.rect)
            for row in rows:
                row.role = "caption"
        for rect in self.page.graphics:  # large uncaptioned graphics
            if rect.width >= 60 and rect.height >= 40 and rect.get_area() >= 0.03 * self.page.rect.get_area():
                if not any(rect.intersects(c) for c in claimed) and self._body_rows_in(rect) < 3:
                    self.floats.append(_Float("figure", self.page.index, self._region_of(rect), rect))
                    claimed.append(rect)
        for row in self.rows:
            if row.role == "flow" and any(_inside(row.rect, rect) for rect in claimed):
                row.role = "float"
        located = [f for f in self.floats if f.rect is not None]
        for n, float_ in enumerate(sorted(located, key=lambda f: (f.y0, f.rect.x0)), 1):
            name = f"p{self.page.index + 1}_{n}.png"
            render(float_.rect, figure_dir / name)
            float_.image = f"{_FIGURE_DIR}/{name}"

    def _sorted_rows(self, region: str | None = None) -> list[_Row]:
        rows = [r for r in self.rows if region is None or r.region == region]
        return sorted(rows, key=lambda r: (r.baseline, r.x0))

    def _row_above(self, row: _Row) -> _Row | None:
        above = [
            r for r in self.rows
            if r.region == row.region and r.baseline < row.baseline - 0.5 * row.size
            and min(r.x1, row.x1) - max(r.x0, row.x0) > 0
        ]
        return max(above, key=lambda r: r.baseline, default=None)

    def _caption_start(self, row: _Row) -> bool:
        if not _CAPTION.match(row.text):
            return False
        prev = self._row_above(row)
        continues = (
            prev is not None and prev.body and abs(prev.size - row.size) <= 0.5
            and row.baseline - prev.baseline <= 1.6 * row.size and not _SENTENCE_END.search(prev.text)
        )
        return not continues

    def _caption_rows(self, first: _Row) -> list[_Row]:
        """The caption's first row and the rows below it (any region: a wide caption's short last line is in a column)."""
        rows = [first]
        for row in self._sorted_rows():
            last = rows[-1]
            overlap = min(row.x1, last.x1) - max(row.x0, last.x0)
            if row.baseline <= last.baseline or row.role != "flow" or overlap < 0.5 * min(row.width, last.width):
                continue
            if (
                row.baseline - last.baseline > 1.6 * max(row.key_size, last.key_size)
                or abs(row.key_size - first.key_size) > 0.6
                or self._caption_start(row)
            ):
                break
            rows.append(row)
        return rows

    def _span_for(self, caption: list[_Row]) -> tuple[float, float]:
        """Horizontal extent a caption's float may occupy: its column, or its own half for side-by-side floats."""
        x0, x1 = self.bounds(caption[0].region)
        cx0, cx1 = min(r.x0 for r in caption), max(r.x1 for r in caption)
        middle = (x0 + x1) / 2
        if cx1 - cx0 < 0.6 * (x1 - x0) and abs((cx0 + cx1) / 2 - middle) > 0.15 * (x1 - x0):
            return (x0, middle) if (cx0 + cx1) / 2 < middle else (middle, x1)
        return x0, x1

    def _grow(self, caption: list[_Row], kind: str, claimed: list[pymupdf.Rect]) -> _Float | None:
        top, bottom = caption[0].y0, caption[-1].y1
        span = self._span_for(caption)
        body = self.doc.body
        items: list[tuple[pymupdf.Rect, _Row | None]] = [
            (g, None) for g in self.page.graphics if not any(g.intersects(c) for c in claimed)
        ] + [(r.rect, r) for r in self.rows if r.role == "flow" and r not in caption]
        items = [(rect, row) for rect, row in items if span[0] - 2 <= (rect.x0 + rect.x1) / 2 <= span[1] + 2]
        for above in ((True, False) if kind == "figure" else (False, True)):
            region: pymupdf.Rect | None = None
            near = sorted(
                (it for it in items if (it[0].y1 <= top + 2 if above else it[0].y0 >= bottom - 2)),
                key=lambda it: -it[0].y1 if above else it[0].y0,
            )
            for rect, row in near:
                edge = (region.y0 if region else top) if above else (region.y1 if region else bottom)
                gap = edge - rect.y1 if above else rect.y0 - edge
                if gap > (3 if region is None else 2.5) * body:
                    break
                if row is not None and (row.body or self._caption_start(row)):
                    break
                region = rect if region is None else region | rect
            if region is not None and region.height >= 0.8 * body:
                region = self._with_rules(region, span)
                for rect in self.page.rotated:  # axis titles next to a plot
                    if rect.intersects(pymupdf.Rect(region.x0 - 12, region.y0, region.x1 + 12, region.y1)):
                        region |= rect
                region = pymupdf.Rect(region.x0 - 2, region.y0 - 2, region.x1 + 2, region.y1 + 2) & self.page.rect
                return _Float(kind, self.page.index, caption[0].region, region, caption, caption_above=not above)
        return None

    def _with_rules(self, region: pymupdf.Rect, span: tuple[float, float]) -> pymupdf.Rect:
        """Add table rules lying just above/below or inside the region."""
        for rule in self.page.rules:
            if rule.x0 >= span[0] - 4 and rule.x1 <= span[1] + 4 and region.y0 - 6 <= rule.y0 <= region.y1 + 6:
                region = region | rule
        return region

    def _body_rows_in(self, rect: pymupdf.Rect) -> int:
        return sum(r.body and _inside(r.rect, rect) for r in self.rows)

    def _region_of(self, rect: pymupdf.Rect) -> str:
        if self.gutter is None:
            return "S"
        if rect.x1 <= self.gutter + 4:
            return "L"
        return "R" if rect.x0 >= self.gutter - 4 else "F"

    # -- footnotes and front matter

    def mark_footnotes(self) -> None:
        """Small-font rows at the bottom of a column under body text, starting with a mark or a rule."""
        limit = self.doc.body - 0.8
        for region in {r.region for r in self.rows}:
            tail: list[_Row] = []
            above_body = False
            for row in reversed(self._sorted_rows(region)):
                if row.role != "flow":
                    continue
                if row.size > limit or row.math_frac >= 0.3 or row.eq_number:
                    above_body = row.size > limit
                    break
                tail.insert(0, row)
            start = next((i for i, row in enumerate(tail) if self._footnote_start(row)), None)
            if above_body and start is not None:
                for row in tail[start:]:
                    row.role = "footnote"

    def _footnote_start(self, row: _Row) -> bool:
        first = row.spans[0]
        raised = first.size < row.size - 0.5 and first.baseline < row.baseline - 0.1 * row.size
        ruled = any(
            0 <= row.y0 - rule.y1 <= 8 and rule.x0 <= row.x0 + 5 and rule.width < 0.6 * self.page.rect.width
            for rule in self.page.rules
        )
        return raised or ruled or bool(_FOOTNOTE_MARK.match(row.text))

    def front_rows(self) -> list[_Row]:
        """Page 1: the rows above the abstract (or the first heading): title and author lines."""
        flow = [r for r in self.rows if r.role == "flow"]
        marks = [r for r in flow if _ABSTRACT_START.match(r.text) or self._looks_like_first_heading(r)]
        if not marks:
            return []
        cut = min(r.y0 for r in marks)
        front = sorted((r for r in flow if r.y1 <= cut + 1), key=lambda r: (round(r.y0), r.x0))
        for row in front:
            row.role = "front"
        return front

    def _looks_like_first_heading(self, row: _Row) -> bool:
        text = row.text
        styled = row.bold or row.caps or row.size >= self.doc.body + 0.8
        return styled and bool(re.match(r"^(?:1\.?|I\.)\s+(?:introduction|INTRODUCTION)\b|^introduction$", text, re.I))

    # -- reading order

    def ordered(self) -> list[_Item]:
        items: list[_Item] = [r for r in self.rows if r.role in ("flow", "footnote")] + list(self.floats)
        full = sorted((it for it in items if it.region == "F"), key=lambda it: it.y0)

        def key(item: _Item) -> tuple[int, int, float]:
            if item.region == "F":
                return full.index(item), 2, item.y0
            middle = (item.y0 + item.y1) / 2
            band = sum((f.y0 + f.y1) / 2 < middle for f in full)
            return band, 0 if item.region in "LS" else 1, item.y0

        ordered = sorted(items, key=key)
        previous = None
        for item in ordered:
            segment = key(item)[:2]
            if isinstance(item, _Row):
                item.seg_start = segment != previous
            previous = segment
        return ordered


def _inside(inner: pymupdf.Rect, outer: pymupdf.Rect) -> bool:
    area = inner.get_area()
    return area > 0 and (inner & outer).get_area() >= 0.7 * area


# --------------------------------------------------------------------------- blocks


@dataclass(eq=False)
class _Block:
    type: str  # text / heading / list_item / equation / footnote
    rows: list[_Row]
    level: int | None = None
    text: str = ""
    eq_number: str | None = None


class _BlockBuilder:
    """Groups rows in reading order into paragraphs, headings, list items, equations and footnotes."""

    def __init__(self, doc: _Doc, bounds: dict[tuple[int, str], tuple[float, float]]):
        self.doc = doc
        self.bounds = bounds  # (page, region) -> column (x0, x1)

    def col(self, row: _Row) -> tuple[float, float]:
        return self.bounds[(row.page, row.region)]

    def build(self, stream: list[_Item]) -> list[_Block | _Float]:
        out: list[_Block | _Float] = []
        current: _Block | None = None
        for i, item in enumerate(stream):
            if isinstance(item, _Float):
                current = None
                out.append(item)
                continue
            row = item
            if current is not None and current.type == "heading" and self._heading_continues(current, row):
                current.rows.append(row)  # a heading wrapped over two lines
                continue
            nxt = next((it for it in stream[i + 1 : i + 2] if isinstance(it, _Row)), None)
            kind = self._start_kind(row, current, nxt)
            if current is not None and kind is None and self._continues(current, row):
                current.rows.append(row)
                continue
            centred = kind is None and self._centred(row)
            if current is not None and current.type == "equation" and (kind == "equation" or centred) and self._same_equation(current, row):
                current.rows.append(row)
                continue
            current = _Block(kind or "text", [row])
            if kind == "heading":
                current.level = self._heading_level(row)
            out.append(current)
        for block in out:
            if isinstance(block, _Block):
                self._finish(block)
        return [b for b in out if not isinstance(b, _Block) or b.text]

    # -- classification

    def _start_kind(self, row: _Row, current: _Block | None, nxt: _Row | None) -> str | None:
        if row.role == "footnote":
            if current is not None and current.type == "footnote" and not _FOOTNOTE_MARK.match(row.text):
                return None
            return "footnote"
        if row.eq_number or row.display_math:
            return "equation"
        if self._is_heading(row, current, nxt):
            return "heading"
        x0 = self.col(row)[0]
        run_in = _RUN_IN_NUMBER.match(row.text) and row.spans[0].italic and not row.spans[0].math
        if not run_in and (_LIST_MARKER.match(row.text) or (_DASH_ITEM.match(row.text) and row.x0 >= x0 + 0.5 * row.size)):
            return "list_item"
        return None

    def _block_start(self, row: _Row, current: _Block | None) -> bool:
        return current is None or row.seg_start or not self._continues(current, row)

    def _is_heading(self, row: _Row, current: _Block | None, nxt: _Row | None) -> bool:
        body = self.doc.body
        text = row.text
        if len(text) > 160 or row.key_size < body - 1.5 or not re.search(r"[A-Za-z]{2}", text):
            return False
        if not self._block_start(row, current):
            return False
        larger = row.key_size >= body + 0.8
        emphasised = larger or row.bold
        if m := _NUMBERED_HEADING.match(text):
            title = m["title"]
            return (emphasised or _caps(title)) and title[0].isupper() and not title.endswith((".", ",", ";"))
        if m := _ROMAN_HEADING.match(text):
            return emphasised or _caps(m["title"])
        if m := _LETTER_HEADING.match(text):
            title = m["title"]
            if title[0].isupper() and not title.endswith((".", ",", ";")) and len(title) <= 120:
                if m["dot"] and (row.italic or row.bold):
                    return True  # IEEE subsection "A. Policy learning"
                if not m["dot"] and emphasised:
                    return True  # appendix "A Proofs" / "A.1 Details"
        if _APPENDIX_HEADING.match(text) and (emphasised or row.caps) and len(text.split()) <= 10:
            return True
        words = len(text.split())
        if words > 12 or text.endswith((".", ",", ";", ":")) or not text[0].isupper():
            return False
        x0, x1 = self.col(row)
        if larger:
            return True
        if row.caps and words <= 6:
            return row.bold or abs((row.x0 + row.x1) / 2 - (x0 + x1) / 2) < 0.1 * (x1 - x0)
        return row.bold and row.width < 0.7 * (x1 - x0) and (nxt is None or not nxt.bold)

    def _heading_continues(self, heading: _Block, row: _Row) -> bool:
        last = heading.rows[-1]
        return (
            not row.seg_start and len(heading.rows) < 3 and row.region == last.region and abs(row.size - last.size) <= 0.3
            and row.bold == last.bold and row.italic == last.italic
            and 0 < row.baseline - last.baseline <= 1.5 * row.size
            and not _NUMBERED_HEADING.match(row.text) and not row.text.endswith(".")
        )

    def _heading_level(self, row: _Row) -> int | None:
        """Level from the numbering; None for unnumbered headings (set by style in _unnumbered_levels)."""
        text = row.text
        if m := _NUMBERED_HEADING.match(text):
            return m["num"].count(".") + 1
        if _ROMAN_HEADING.match(text) or _APPENDIX_HEADING.match(text):
            return 1
        if m := _LETTER_HEADING.match(text):
            return 2 if m["dot"] else m["num"].count(".") + 1
        return None

    # -- grouping

    def _continues(self, block: _Block, row: _Row) -> bool:
        if block.type in ("heading", "equation") or row.seg_start:
            return False
        last = block.rows[-1]
        if block.type == "footnote":
            return row.role == "footnote"
        if row.role != "flow" or last.page != row.page or last.region != row.region:
            return False
        pitch = row.baseline - last.baseline
        size = max(row.size, last.size)
        if pitch <= 0 or pitch > 3 * size:
            return False
        clearly_continues = not _SENTENCE_END.search(last.text) and row.text[:1].islower()
        if (pitch > 1.6 * size or abs(row.size - last.size) > 0.6) and not clearly_continues:
            return False
        x0, x1 = self.col(row)
        if block.type == "list_item":  # hanging indent: continuation lines start right of the marker
            hang = block.rows[1].x0 if len(block.rows) > 1 else block.rows[0].x0 + row.size
            return row.x0 >= hang - 2 or clearly_continues
        left = min(r.x0 for r in block.rows)
        if row.x0 > left + 0.6 * row.size and row.x0 <= x0 + 4 * row.size and len(block.rows) > 1:
            return False  # first-line indent of the next paragraph
        if len(block.rows) == 1 and row.x0 > last.x0 + 0.6 * row.size:
            return False
        short = last.x1 < x1 - max(2 * size, 0.1 * (x1 - x0))
        return not (self.doc.justified and short and _SENTENCE_END.search(last.text))

    def _centred(self, row: _Row) -> bool:
        x0, x1 = self.col(row)
        return row.x0 >= x0 + 1.5 * row.size and row.x1 <= x1 - 1.5 * row.size

    def _same_equation(self, block: _Block, row: _Row) -> bool:
        last = block.rows[-1]
        if last.page != row.page or last.region != row.region or row.seg_start:
            return False
        numbered = [r for r in block.rows if r.eq_number]
        if numbered:  # an equation ends at its number; parts touching it (a denominator) still belong
            return not row.eq_number and row.y0 < numbered[-1].y1 + 0.3 * row.size
        return row.y0 - max(r.y1 for r in block.rows) <= 1.2 * row.size

    # -- text

    def _finish(self, block: _Block) -> None:
        if block.type == "equation":
            numbers = [r.eq_number for r in block.rows if r.eq_number]
            block.eq_number = ", ".join(numbers) or None
            parts = []
            for row in block.rows:
                cells = row.cells[:-1] if row.eq_number else row.cells
                parts.extend(c[2] for c in cells)
            block.text = norm_text(" ".join(parts))
            return
        block.text = self.join_rows(block.rows)

    def join_rows(self, rows: list[_Row]) -> str:
        return self.join_texts([row.text for row in rows])

    def join_texts(self, lines: list[str]) -> str:
        text = ""
        for line in lines:
            text = self.join(text, line)
        return text

    def join(self, left: str, right: str) -> str:
        """Join two lines, removing a line-break hyphen unless the hyphen is real ("well-defined")."""
        if not left:
            return right
        if m := re.search(r"([A-Za-z]+)-$", left):
            tail = re.match(r"[a-z]+", right)
            if tail is None:
                return left + right  # "LP-" + "ACRL", "2-" + "3"
            return left + right if self._keep_hyphen(m[1], tail[0]) else left[:-1] + right
        if left.endswith(("—", "–")):
            return left + right
        return f"{left} {right}"

    def _keep_hyphen(self, head: str, tail: str) -> bool:
        head, tail = head.lower(), tail.lower()
        if f"{head}-{tail}" in self.doc.hyphenated:
            return True
        if head + tail in self.doc.words:
            return False
        if head in _HYPHEN_PREFIXES:
            return True
        return len(head) >= 3 and len(tail) >= 3 and head in self.doc.words and tail in self.doc.words


def _caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 3 and sum(c.isupper() for c in letters) >= 0.9 * len(letters)


def _vocabulary(pages: list[_PageLayout]) -> tuple[set[str], set[str]]:
    """Words and hyphenated compounds seen inside lines (never at a line break)."""
    words: set[str] = set()
    hyphenated: set[str] = set()
    for layout in pages:
        for row in layout.rows:
            tokens = _WORD.findall(row.text)[1:]  # the first word may continue a line-break hyphen
            if row.text.endswith("-"):
                tokens = tokens[:-1]
            for token in tokens:
                token = token.lower()
                if "-" in token:
                    hyphenated.add(token)
                    words.update(token.split("-"))
                else:
                    words.add(token)
    return words, hyphenated


def _unnumbered_levels(blocks: list[_Block]) -> None:
    """Unnumbered headings take the level of the numbered headings styled most like them, else rank by size."""
    def style(block: _Block) -> tuple[float, bool, bool, bool]:
        row = block.rows[0]
        return round(row.key_size * 2) / 2, row.bold, row.italic, row.caps

    headings = [b for b in blocks if b.type == "heading"]
    known: dict[tuple[float, bool, bool, bool], int] = {}
    for block in headings:
        if block.level is not None:
            known.setdefault(style(block), block.level)
    sizes = sorted({style(b)[0] for b in headings if b.level is None}, reverse=True)
    for block in headings:
        if block.level is not None:
            continue
        own = style(block)
        if ABSTRACT_TITLE.match(block.text) or BIBLIOGRAPHY_TITLE.match(block.text) or _APPENDIX_HEADING.match(block.text):
            block.level = 1
        elif known:
            nearest = min(known.items(), key=lambda kv: (kv[0][1:] != own[1:], abs(kv[0][0] - own[0]), kv[1]))
            block.level = nearest[1]
        else:
            block.level = min(sizes.index(own[0]) + 1, 3)


# --------------------------------------------------------------------------- elements


def _split_run_in(block: _Block, builder: _BlockBuilder) -> list[Element] | None:
    """'1) Definition of X: text' (IEEE) or a bold 'Training details. text' -> heading + paragraph."""
    spans = [s for row in block.rows for s in row.spans]
    prefix: list[_Span] = []
    for span in spans:
        if not (span.italic or span.bold) or span.math:
            break
        prefix.append(span)
        if re.search(r"[.:]\s*$", span.text):
            break
    rows = [[s for s in row.spans if s in prefix] for row in block.rows]
    head = builder.join_texts([_join_spans(r) for r in rows if r])
    if not prefix or len(prefix) == len(spans) or len(head) > 120 or not re.search(r"[.:]$", head):
        return None
    numbered = _RUN_IN_NUMBER.match(head)
    if not numbered and not all(s.bold for s in prefix):
        return None
    rest = _strip_prefix(block.text, head)
    if not rest or rest == block.text:
        return None
    page = block.rows[0].page
    level = 3 if numbered and numbered["sub"] else 4
    return [
        _element("heading", page, block.rows[:1], head.rstrip(".: "), level=level),
        _element("text", page, block.rows, rest),
    ]


def _paragraph_title(block: _Block) -> bool:
    """A bold paragraph title on its own line ("Implementation details.")."""
    row = block.rows[0]
    return len(block.rows) == 1 and row.bold and len(block.text.split()) <= 10 and block.text.endswith((".", ":"))


def _strip_prefix(text: str, head: str) -> str:
    """Remove a run-in title (whitespace may differ between the spans and the joined text)."""
    m = re.match(r"\s*".join(re.escape(c) for c in head.replace(" ", "")), text)
    return text[m.end():].strip() if m else text


def _element(kind: str, page: int, rows: list[_Row], text: str, **extra) -> Element:
    rect = pymupdf.Rect(rows[0].rect)
    for row in rows[1:]:
        rect |= row.rect
    return Element(
        type=kind, page=page, text=text.replace("$", r"\$"), bbox=tuple(round(v, 1) for v in rect),
        font_size=rows[0].size, **extra,
    )


def _float_elements(float_: _Float, builder: _BlockBuilder) -> list[Element]:
    out = []
    if float_.rect is not None:
        bbox = tuple(round(v, 1) for v in float_.rect)
        out.append(Element(type=float_.kind, page=float_.page, bbox=bbox, image=float_.image))
    if float_.caption:
        text = builder.join_rows(float_.caption)
        out.insert(0 if float_.caption_above else len(out), _element("caption", float_.page, float_.caption, text))
    return out


def _front_elements(rows: list[_Row], body: float) -> list[Element]:
    """Title = the largest rows on top; author lines = the following rows in the first author line's style."""
    if not rows:
        return []
    biggest = max(r.size for r in rows)
    start = next(i for i, r in enumerate(rows) if r.size >= biggest - 0.5)
    end = start
    while end + 1 < len(rows) and rows[end + 1].size >= biggest - 0.5:
        end += 1
    title_rows = rows[start : end + 1]
    out = []
    if biggest >= body + 1:
        out.append(_element("title", 0, title_rows, " ".join(r.text for r in title_rows)))
    rest = rows[end + 1 :]
    if rest:
        style = (round(rest[0].size), rest[0].bold)
        for row in rest:
            if (round(row.size), row.bold) == style:
                names = [s for s in row.spans if s.size >= 0.8 * row.size]
                text = ", ".join(c[2] for c in _cells(names)) if names else row.text
                out.append(_element("authors", 0, [row], text))
    return out


def _reference_elements(rows: list[_Row], builder: _BlockBuilder) -> list[Element]:
    """Split bibliography rows into entries: "[n]"/"n." markers, else hanging indents or gaps."""
    numbered = sum(bool(_REFERENCE_MARK.match(r.text)) for r in rows) >= 2
    hanging = any(r.x0 > builder.col(r)[0] + 0.5 * r.size for r in rows)
    entries: list[list[_Row]] = []
    for row in rows:
        x0 = builder.col(row)[0]
        prev = entries[-1][-1] if entries else None
        if numbered:
            new = bool(_REFERENCE_MARK.match(row.text))
        else:
            gap = prev is not None and row.page == prev.page and row.region == prev.region and (
                row.baseline - prev.baseline > 1.35 * row.size
            )
            new = (row.x0 <= x0 + 0.5 * row.size) if hanging else (gap or row.seg_start)
        if new or not entries:
            entries.append([row])
        else:
            entries[-1].append(row)
    return [_element("reference", e[0].page, e, builder.join_rows(e)) for e in entries]


def _elements(items: list[_Block | _Float], builder: _BlockBuilder) -> list[Element]:
    out: list[Element] = []
    in_references = False
    reference_rows: list[_Row] = []
    for item in items:
        if isinstance(item, _Float):
            out.extend(_float_elements(item, builder))
            continue
        block = item
        if block.type == "heading":
            if reference_rows:
                out.extend(_reference_elements(reference_rows, builder))
                reference_rows = []
            in_references = bool(BIBLIOGRAPHY_TITLE.match(block.text))
            out.append(_element("heading", block.rows[0].page, block.rows, block.text, level=block.level))
        elif block.type == "footnote":
            out.append(_element("footnote", block.rows[0].page, block.rows, block.text))
        elif in_references:
            reference_rows.extend(block.rows)
        elif block.type == "equation":
            out.append(_element("equation", block.rows[0].page, block.rows, block.text, eq_number=block.eq_number))
        elif block.type == "text" and _paragraph_title(block):
            out.append(_element("heading", block.rows[0].page, block.rows, block.text.rstrip(".: "), level=4))
        elif block.type == "text" and (split := _split_run_in(block, builder)):
            out.extend(split)
        else:
            out.append(_element(block.type, block.rows[0].page, block.rows, block.text))
    if reference_rows:
        out.extend(_reference_elements(reference_rows, builder))
    return out


def _merge_continued(items: list[_Block | _Float], builder: _BlockBuilder) -> list[_Block | _Float]:
    """Join a paragraph with its continuation after a column/page break or a float in between."""
    out: list[_Block | _Float] = []
    open_block: _Block | None = None  # the last paragraph, if only floats/footnotes followed it
    interrupted = False
    for item in items:
        if isinstance(item, _Float) or item.type == "footnote":
            interrupted = interrupted or isinstance(item, _Float)
            out.append(item)
            continue
        if (
            open_block is not None and item.type == "text" and (item.rows[0].seg_start or interrupted)
            and not _SENTENCE_END.search(open_block.text)
            and (item.text[:1].islower() or open_block.text.endswith(("-", ",")))
        ):
            open_block.text = builder.join(open_block.text, item.text)
            open_block.rows.extend(item.rows)
            continue
        out.append(item)
        open_block = item if item.type in ("text", "list_item") else None
        interrupted = False
    return out


def _render(page: pymupdf.Page) -> Callable[[pymupdf.Rect, Path], None]:
    def render(rect: pymupdf.Rect, path: Path) -> None:
        page.get_pixmap(matrix=pymupdf.Matrix(_RENDER_ZOOM, _RENDER_ZOOM), clip=rect, alpha=False).save(path)

    return render


def extract_text_layer(pdf_path: Path, paper_dir: Path, progress_cb: ProgressCallback | None = None) -> list[Element]:
    """Elements in reading order from a born-digital PDF; figure/table crops go to paper_dir/figures/."""
    figure_dir = Path(paper_dir) / _FIGURE_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as pdf:
        pages = [_read_page(page) for page in pdf]
        _drop_noise(pages)
        doc = _document(pages)
        layouts = [_PageLayout(page, doc) for page in pages]
        _measure_columns(doc, layouts)
        words, hyphenated = _vocabulary(layouts)
        doc.words, doc.hyphenated = words, hyphenated

        stream: list[_Item] = []
        front: list[_Row] = []
        bounds: dict[tuple[int, str], tuple[float, float]] = {}
        for layout, page in zip(layouts, pdf):
            layout.find_floats(figure_dir, _render(page))
            layout.mark_footnotes()
            if layout.page.index == 0:
                front = layout.front_rows()
            stream.extend(layout.ordered())
            bounds.update({(layout.page.index, r): layout.bounds(r) for r in "LRFS"})
            if progress_cb:
                progress_cb((layout.page.index + 1) / len(layouts), f"Page {layout.page.index + 1}/{len(layouts)}")

    builder = _BlockBuilder(doc, bounds)
    items = _merge_continued(builder.build(stream), builder)
    _unnumbered_levels([b for b in items if isinstance(b, _Block)])
    return _front_elements(front, doc.body) + _elements(items, builder)
