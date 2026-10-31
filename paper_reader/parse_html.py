"""arXiv HTML (LaTeXML) -> Paper (see models.py for the data contract).

How the document is mapped
--------------------------
* Sections (section/subsection/subsubsection, chapters, appendices) are FLATTENED in
  reading order; each Section holds only its *direct* content. ``\\paragraph``,
  theorem and proof titles are not sections: they become ``heading_note`` blocks.
* The abstract becomes Section(id="abstract", kind="abstract").
* The bibliography and acknowledgement-like back matter are always dropped (a dropped
  section's sub-sections are dropped too, unless they start the appendix).
* The appendix starts at the first section.ltx_appendix, the first section whose id
  starts with "A<digit>", the first section titled "Appendix…"/"Supplementary…", or
  the first section after the bibliography. Everything after that point is appendix
  and is skipped unless ``include_appendix``.
* Article-level content outside any section (e.g. a teaser figure) is moved to the
  start of the next section.

Sanitizing
----------
No source markup is copied through. All HTML is re-emitted by ``_Inline`` (and the
table/equation builders) from a whitelist of tags and attributes, with every piece of
text escaped, so the output is safe by construction.
"""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field
from html import escape
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag

from .models import Block, Figure, Paper, Section, SectionKind

_SECTION_LEVELS = {
    "ltx_part": 1,
    "ltx_chapter": 1,
    "ltx_section": 1,
    "ltx_appendix": 1,
    "ltx_subsection": 2,
    "ltx_subsubsection": 3,
}
_FLOAT_KINDS = {"ltx_figure": "figure", "ltx_table": "table"}
_EQUATION_TABLE_CLASSES = {"ltx_equation", "ltx_equationgroup", "ltx_eqn_table"}
_EQUATION_SKIP_CELLS = {"ltx_eqn_eqno", "ltx_eqn_center_padleft", "ltx_eqn_center_padright"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

# Removed from the article before anything else is looked at.
_NOISE_SELECTOR = ", ".join(
    [
        "script", "style", "noscript", "template", "button", "input", "select", "textarea",
        "form", "iframe", "nav", "header", "footer",
        ".ltx_note",  # footnotes / endnotes and their marks
        ".ltx_pagination", ".ltx_rdf", ".ltx_TOC", ".ltx_dates", ".ltx_keywords",
        ".ltx_classification", ".ltx_acknowledgements", ".ltx_page_logo",
        ".ltx_phantom", ".sr-only", ".visually-hidden", "[aria-hidden=true]",
    ]
)
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)

_APPENDIX_ID = re.compile(r"^A\d")
APPENDIX_TITLE = re.compile(r"^(appendix|appendices|supplementary|supplemental)\b", re.I)
BIBLIOGRAPHY_TITLE = re.compile(r"^(references|bibliography)$", re.I)
BACKMATTER_TITLE = re.compile(
    r"^(acknowledge?ments?|funding|author contributions?|competing interests?"
    r"|conflicts? of interests?|declarations? of (competing )?interests?|(statements? and )?declarations"
    r"|(data|code|software|data and code) availability)\b",
    re.I,
)
ABSTRACT_TITLE = re.compile(r"^abstract\.?$", re.I)
# Loose text before the first section shorter than this is front matter (affiliations, "and", ...).
_MIN_LOOSE_PARAGRAPH_CHARS = 200

# Inline tags kept as-is (renamed where LaTeXML uses a near-synonym).
_INLINE_TAGS = {
    "em": "em", "strong": "strong", "b": "b", "i": "i", "u": "u", "sub": "sub",
    "sup": "sup", "code": "code", "small": "small", "tt": "code", "kbd": "code",
    "var": "i",
}
# LaTeXML font classes on <span> -> tag (None = keep as a span with that class).
_FONT_CLASSES = {
    "ltx_font_bold": "b",
    "ltx_font_italic": "i",
    "ltx_font_slanted": "i",
    "ltx_font_typewriter": "code",
    "ltx_framed_underline": "u",
    "ltx_font_smallcaps": None,
}
_DROPPED_INLINE = {
    "img", "object", "svg", "video", "audio", "canvas", "iframe", "figure", "script",
    "style", "button", "input", "select", "textarea", "form", "noscript", "template",
}
_IMAGE_TAGS = {"img", "object", "svg"}
_LOOSE_IMAGE_WARNING = "Dropped image(s) outside figure/table floats"
_BLOCKISH_INLINE = {"p", "div", "li", "dt", "dd", "ul", "ol", "dl", "blockquote"}
# Tags that may appear loose in a content container and are gathered into a paragraph.
_PHRASING = set(_INLINE_TAGS) | {"span", "a", "math", "cite", "br", "abbr", "q", "mark"}

_TEXT_TRANSLATION = str.maketrans(
    {"\u00ad": None, "\u200b": None, "\u200c": None, "\u200d": None, "\ufeff": None, "\u2011": "-"}
)
_LATEX_EDGE_SPACE = r"(?:\s|~|(?<!\\)\\[,;:! ]|\\q?quad(?![a-zA-Z]))+"
_YEAR = re.compile(r"\b(?:1[89]|20)\d{2}[a-z]?\b")
_CITE_MARKS = re.compile(r"[\[(][^\])]*[\])]")
_TEXTUAL_CITE_MACROS = {
    "ltx_citemacro_citet", "ltx_citemacro_Citet", "ltx_citemacro_citeauthor",
    "ltx_citemacro_Citeauthor", "ltx_citemacro_citealt", "ltx_citemacro_Citealt",
}
_PARENTHETICAL_CITE_MACROS = {
    "ltx_citemacro_citep", "ltx_citemacro_Citep", "ltx_citemacro_citealp",
    "ltx_citemacro_Citealp", "ltx_citemacro_citeyear", "ltx_citemacro_citeyearpar",
}
_NOT_A_NAME = re.compile(
    r"\b(universit\w*|institut\w*|department|dept|laborator\w*|lab|school|college|"
    r"cent(er|re)|inc|ltd|llc|gmbh|corp\w*|academy|faculty|hospital|research|equal contribution)\b",
    re.I,
)


# --------------------------------------------------------------------------- helpers


def _classes(tag: Tag) -> set[str]:
    return set(tag.get("class") or [])


def norm_text(s: str) -> str:
    """Single-line, whitespace-normalized plain text."""
    return re.sub(r"\s+", " ", s.translate(_TEXT_TRANSLATION)).strip()


def _norm_html(s: str) -> str:
    return re.sub(r"[ \t\r\n]+", " ", s.translate(_TEXT_TRANSLATION)).strip()


def clean_latex(latex: str) -> str:
    """One-line LaTeX without edge spacing (``~``, ``\\,``, ``\\quad``…) or a leading ``\\displaystyle``."""
    latex = re.sub(rf"^{_LATEX_EDGE_SPACE}|{_LATEX_EDGE_SPACE}$", "", latex)
    return norm_text(re.sub(r"^\\(?:displaystyle|textstyle)(?![a-zA-Z])", "", latex))


def _fix_appendix_dash(text: str) -> str:
    """IEEE-style appendices have an empty section counter, rendered "-A", "-A1", "§-A"."""
    return re.sub(r"^(§?)[-\u2212\u2013](?=[A-Z])", r"\1", text)


def _clean_number(tag_text: str) -> str | None:
    """Section/float number from a LaTeXML tag: "III-A1 " -> "III-A1", "Appendix B" -> "B"."""
    text = norm_text(tag_text).rstrip(":.").strip()
    text = re.sub(r"^(appendix|section|§)\s*", "", text, flags=re.I)
    return _fix_appendix_dash(text) or None


def _heading(section: Tag) -> Tag | None:
    for child in section.find_all(_HEADINGS, recursive=False):
        if "ltx_title" in _classes(child):
            return child
    return None


def _is_flat_section(tag: Tag) -> bool:
    return tag.name == "section" and bool(_classes(tag) & _SECTION_LEVELS.keys())


def _section_level(tag: Tag) -> int:
    return min(_SECTION_LEVELS[c] for c in _classes(tag) & _SECTION_LEVELS.keys())


def _float_kind(tag: Tag) -> str | None:
    for cls in _classes(tag):
        if cls in _FLOAT_KINDS:
            return _FLOAT_KINDS[cls]
    return None


def _is_tabular(tag: Tag) -> bool:
    return tag.name == "table" and "ltx_tabular" in _classes(tag)


def _float_media(fig: Tag) -> tuple[list[Tag], list[Tag]]:
    """(images/SVG objects/inline SVG drawings, outermost tabulars) of a float, caption excluded."""
    images: list[Tag] = []
    tables: list[Tag] = []
    for el in fig.find_all(lambda t: t.name in _IMAGE_TAGS or _is_tabular(t)):
        if el.find_parent("figcaption") or el.find_parent("svg"):
            continue
        if el.name != "table":
            images.append(el)
        elif el.find_parent("table") is None:
            tables.append(el)
    return images, tables


def _outermost_floats(node: Tag) -> list[Tag]:
    """<figure>s in a content node (or the node itself) that are not nested in another figure."""
    if node.name == "figure":
        return [node]
    return [f for f in node.find_all("figure") if f.find_parent("figure") is None]


def _citation_names(visible: str) -> str:
    """Author names of a textual citation: "Smith et al. (2020a, b)" -> "Smith et al."."""
    stripped = _YEAR.sub(" ", re.sub(r"\[[^\]]*\]", " ", visible))
    parts = (p.strip(" ()[]") for p in re.split(r"[;,()\[\]]", stripped))
    return ", ".join(p for p in parts if len(re.findall(r"[^\W\d_]", p)) >= 2)


# --------------------------------------------------------------------------- inline


@dataclass
class _Rich:
    html: str
    text: str
    refs: list[str]


class _Inline:
    """Walks inline content, emitting sanitized display HTML and speech text in parallel."""

    def __init__(self, conv: _Converter, skip: tuple[Tag, ...] = ()):
        self.conv = conv
        self.skip = skip
        self.html: list[str] = []
        self.text: list[str] = []
        self.refs: list[str] = []

    def run(self, *nodes: Tag | NavigableString) -> _Rich:
        for node in nodes:
            self._node(node)
        return _Rich(_norm_html("".join(self.html)), norm_text("".join(self.text)), self.refs)

    def _emit(self, html: str, text: str) -> None:
        self.html.append(html)
        self.text.append(text)

    def _children(self, tag: Tag) -> None:
        for child in tag.children:
            self._node(child)

    def _node(self, node) -> None:
        if isinstance(node, NavigableString):
            if type(node) is NavigableString:  # skips comments, CDATA, doctypes
                self._emit(escape(str(node), quote=False), str(node))
            return
        if not isinstance(node, Tag) or any(node is s for s in self.skip):
            return
        name, cls = node.name, _classes(node)
        if name == "math":
            self._math(node)
        elif name == "cite":
            self._cite(node)
        elif name == "a":
            self._link(node)
        elif name == "br":
            self._emit("<br>", " ")
        elif name in _DROPPED_INLINE:
            if name in _IMAGE_TAGS and node.find_parent("figure") is None:
                self.conv.warn(_LOOSE_IMAGE_WARNING)
        elif name == "table":
            self._flat_table(node)
        elif name in _BLOCKISH_INLINE or "ltx_p" in cls:
            self._separator()
            self._children(node)
        elif "ltx_ref_tag" in cls and not node.find(True):
            text = _fix_appendix_dash(norm_text(node.get_text()))
            self._emit(escape(text, quote=False), text)
        elif name in _INLINE_TAGS:
            self._wrap(node, [_INLINE_TAGS[name]])
        elif name == "span" and cls & _FONT_CLASSES.keys():
            self._wrap(node, [_FONT_CLASSES[c] or f'span class="{c}"' for c in sorted(cls & _FONT_CLASSES.keys())])
        else:
            self._children(node)

    def _wrap(self, node: Tag, open_tags: list[str]) -> None:
        for t in open_tags:
            self.html.append(f"<{t}>")
        self._children(node)
        for t in reversed(open_tags):
            self.html.append(f"</{t.split()[0]}>")

    def _separator(self) -> None:
        if "".join(self.html).strip():
            self._emit("<br>", " ")

    def _flat_table(self, table: Tag) -> None:
        """A tabular nested in inline content (e.g. a multi-line table cell): rows -> lines."""
        for row in table.find_all("tr"):
            if row.find_parent("table") is not table:
                continue
            self._separator()
            for cell in row.find_all(["td", "th"], recursive=False):
                self._children(cell)
                self._emit(" ", " ")

    def _math(self, math: Tag) -> None:
        latex = math.get("alttext")
        if latex is None:
            annotation = math.find("annotation", encoding="application/x-tex")
            latex = annotation.get_text() if annotation else None
        if latex is None:
            fallback = norm_text(math.get_text())
            self.conv.warn(f"Math without LaTeX source rendered as text: {fallback[:60]!r}")
            self._emit(escape(fallback, quote=False), fallback)
            return
        latex = clean_latex(latex)
        if not latex:
            return
        if math.get("display") == "block":
            delims, text = (r"\[", r"\]"), f"$${latex}$$"
        else:
            delims, text = (r"\(", r"\)"), f"${latex}$"
        self._emit(
            f'<span class="math" data-latex="{escape(latex)}">'
            f"{delims[0]}{escape(latex, quote=False)}{delims[1]}</span>",
            f" {text} " if text.startswith("$$") else text,
        )

    def _cite(self, cite: Tag) -> None:
        visible = norm_text(cite.get_text())
        if not visible:
            return
        cls = _classes(cite)
        textual = bool(cls & _TEXTUAL_CITE_MACROS) or (
            not cls & _PARENTHETICAL_CITE_MACROS and visible[0] not in "([" and bool(_citation_names(visible))
        )
        names = _citation_names(visible) if textual else ""
        if not names:
            self._emit(f'<span class="cite">{escape(visible, quote=False)}</span>', "[CITE]")
            return
        # "Smith et al. (2020)": the names stay visible, only "(2020)" / "[7]" is a hideable cite.
        marks = _CITE_MARKS if _CITE_MARKS.search(visible) else _YEAR
        html = marks.sub(lambda m: f'<span class="cite">{m.group()}</span>', escape(visible, quote=False))
        self._emit(html, f"{names} [CITE]")

    def _link(self, a: Tag) -> None:
        href = a.get("href") or ""
        target = href[1:] if href.startswith("#") else ""
        if not target or target.startswith("bib."):
            self._children(a)  # external links and stray bibliography links: text only
            return
        start = len(self.html)
        self._children(a)
        if not "".join(self.html[start:]).strip():
            return
        figure_id = self.conv.figure_alias.get(target)
        if figure_id:
            open_tag = f'<a class="ref" data-ref="{escape(figure_id)}" href="#{escape(figure_id)}">'
            if figure_id not in self.refs:
                self.refs.append(figure_id)
        else:
            open_tag = f'<a href="#{escape(target)}">'
        self.html.insert(start, open_tag)
        self.html.append("</a>")


# --------------------------------------------------------------------------- blocks


@dataclass
class _Unit:
    """Direct content of one output Section, collected before conversion."""

    kind: SectionKind
    el: Tag | None = None  # the <section>; None for the abstract and loose article content
    nodes: list[Tag] = field(default_factory=list)
    id: str = ""
    number: str | None = None
    title: str = ""
    level: int = 1


class _Converter:
    """Turns planned section content into Blocks and Figures."""

    def __init__(self, page_url: str, warnings: list[str]):
        # arXiv serves pages as documents ("/html/<id>v1"); their relative asset paths
        # ("<id>v1/x1.png") break if resolved against a trailing-slash URL.
        self.base_url = page_url.rstrip("/")
        self.warnings = warnings
        # included float id (or sub-figure id) -> outer float id; set once sections are planned
        self.figure_alias: dict[str, str] = {}
        self.figures: dict[str, Figure] = {}

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def inline(self, *nodes, skip: tuple[Tag, ...] = ()) -> _Rich:
        return _Inline(self, skip).run(*nodes)

    # -- content walk

    def blocks(self, nodes: list[Tag]) -> list[Block]:
        out: list[Block] = []
        self._container(nodes, out, "paragraph")
        return out

    def _container(self, children, out: list[Block], para_type: str) -> None:
        """Convert a sequence of sibling nodes; loose inline runs become one paragraph."""
        run: list = []
        for child in list(children) + [None]:
            phrasing = child is not None and (
                isinstance(child, NavigableString)
                or (child.name in _PHRASING and "ltx_p" not in _classes(child))
            )
            if phrasing:
                run.append(child)
                continue
            if run:
                self._paragraph(run, out, para_type)
                run = []
            if isinstance(child, Tag):
                self._content(child, out, para_type)

    def _content(self, node: Tag, out: list[Block], para_type: str) -> None:
        name, cls = node.name, _classes(node)
        if name == "figure":
            self._float(node, out)
        elif name == "table" and cls & _EQUATION_TABLE_CLASSES:
            self._equation(node, out)
        elif name in _HEADINGS:
            self._heading_note(node, out)
        elif name == "p" or "ltx_p" in cls:
            self._paragraph([node], out, para_type)
        elif name in ("ul", "ol", "dl"):
            self._list(node, out)
        elif name == "table":
            self._loose_table(node, out)
        elif name == "pre" or cls & {"ltx_listing", "ltx_verbatim"}:
            self.warn(f"Skipped a code listing ({node.get('id', 'no id')})")
        elif name in _DROPPED_INLINE:
            if name in _IMAGE_TAGS:
                self.warn(_LOOSE_IMAGE_WARNING)
        else:  # div.ltx_para, section.ltx_paragraph, theorem/proof, quote, inline-block, ...
            heading = _heading(node)
            if heading is not None and BACKMATTER_TITLE.match(norm_text(heading.get_text())):
                return  # e.g. \paragraph{Acknowledgements}
            self._container(node.children, out, para_type)

    def _paragraph(self, nodes: list, out: list[Block], block_type: str) -> None:
        rich = self.inline(*nodes)
        if rich.text:
            out.append(Block(id="", type=block_type, html=rich.html, text=rich.text, fig_refs=rich.refs))

    def _heading_note(self, heading: Tag, out: list[Block]) -> None:
        rich = self.inline(heading)
        if not rich.text:
            return
        text = rich.text if rich.text[-1] in ".:!?" else rich.text + "."
        out.append(Block(id="", type="heading_note", html=f"<strong>{rich.html}</strong>", text=text))

    def _list(self, lst: Tag, out: list[Block]) -> None:
        label: Tag | None = None  # a <dt> waiting for its <dd>
        for item in lst.find_all(["li", "dt", "dd"], recursive=False):
            if item.name == "dt":
                label = item
                continue
            if item.name == "li":
                label = item.find(class_="ltx_tag", recursive=False)
            start = len(out)
            self._container((c for c in item.children if c is not label), out, "list_item")
            prefix = self.inline(label) if label is not None else None
            label = None
            if not prefix or not re.search(r"\w", prefix.text):  # bullets are not spoken
                continue
            first = next((b for b in out[start:] if b.type == "list_item"), None)
            if first is not None:
                first.html = f"{prefix.html} {first.html}"
                first.text = f"{prefix.text} {first.text}"

    # -- equations

    def _equation(self, table: Tag, out: list[Block]) -> None:
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            cells = [
                self._equation_cell(td)
                for td in tr.find_all("td", recursive=False)
                if not _classes(td) & _EQUATION_SKIP_CELLS
            ]
            if any(cells):
                rows.append(cells)
        if not rows:
            self.warn(f"Display equation without LaTeX skipped ({table.get('id', 'no id')})")
            return
        if len(rows) == 1:
            latex = " ".join(c for c in rows[0] if c)
        else:
            env = "aligned" if max(len(r) for r in rows) > 1 else "gathered"
            body = r" \\ ".join(" & ".join(r) for r in rows)
            latex = norm_text(rf"\begin{{{env}}}{body}\end{{{env}}}")
        numbers: list[str] = []
        for tag in table.select(".ltx_eqn_eqno .ltx_tag"):
            number = norm_text(tag.get_text()).strip("()")
            if number and number not in numbers:
                numbers.append(number)
        out.append(
            Block(
                id="",
                type="equation",
                html=f'<div class="equation">\\[ {escape(latex, quote=False)} \\]</div>',
                text=f"$${latex}$$",
                latex=latex,
                eq_number=", ".join(numbers) or None,
            )
        )

    @staticmethod
    def _equation_cell(td: Tag) -> str:
        parts = []
        for node in td.descendants:
            if isinstance(node, Tag) and node.name == "math":
                parts.append(clean_latex(node.get("alttext") or ""))
            elif type(node) is NavigableString and node.find_parent("math") is None:
                text = re.sub(r"[{}\\$%&#_^~]", "", norm_text(str(node)))
                if text:
                    parts.append(rf"\text{{{text}}}")
        return " ".join(p for p in parts if p)

    # -- figures and tables

    def _float(self, fig: Tag, out: list[Block]) -> None:
        kind = _float_kind(fig)
        fig_id = fig.get("id")
        if kind is None or not fig_id or fig_id not in self.figure_alias:
            caption = fig.find("figcaption")
            what = self.inline(caption).text[:50] if caption else fig_id or "untitled float"
            self.warn(f"Skipped a float without image or table content (e.g. an algorithm): {what}")
            return
        self.figures[fig_id] = self._figure(fig, kind, fig_id)
        out.append(Block(id="", type="figure", figure_id=fig_id, speak=False))

    def _figure(self, fig: Tag, kind: str, fig_id: str) -> Figure:
        captions = fig.find_all("figcaption")
        own = [c for c in captions if c.find_parent("figure") is fig]
        caption = own[0] if own else None
        tag = caption.find(class_="ltx_tag") if caption else None
        tag_text = _clean_number(tag.get_text()) if tag else None
        number = tag_text.split()[-1] if tag_text else None
        label = f"{kind.capitalize()} {number}" if number else kind.capitalize()

        main = self.inline(caption, skip=(tag,) if tag else ()) if caption else _Rich("", "", [])
        html, text = main.html, main.text
        subs = [self.inline(c) for c in captions if c is not caption]
        subs = [s for s in subs if s.text]
        if subs:
            html = " <br>".join(filter(None, [html] + [s.html for s in subs]))
            text = " ".join(filter(None, [text] + [s.text for s in subs]))

        image_els, table_els = _float_media(fig)
        images: list[str] = []
        for el in image_els:
            src = el.get("src") if el.name == "img" else el.get("data") if el.name == "object" else None
            if el.name == "svg":
                self.warn(f"{label} ({fig_id}): inline SVG drawing (TikZ/pgfplots) not supported; caption only")
            elif src and (el.name == "img" or (el.get("type") or "image/").startswith("image/")):
                url = urljoin(self.base_url, src)
                if url not in images:
                    images.append(url)
        if kind == "figure" and images:
            table_els = []  # a tabular inside a figure with images only lays out the panels
        table_html = "".join(self.table_html(t) for t in table_els) or None
        return Figure(
            id=fig_id, kind=kind, number=number, label=label, caption_html=html,
            caption_text=text, images=images, table_html=table_html,
        )

    def _loose_table(self, table: Tag, out: list[Block]) -> None:
        """A tabular outside any float becomes an unnumbered table (rule-only spacers are dropped)."""
        if not norm_text(table.get_text()):
            return
        table_id = table.get("id") or f"table{len(self.figures) + 1}"
        self.figures[table_id] = Figure(id=table_id, kind="table", label="Table", table_html=self.table_html(table))
        out.append(Block(id="", type="figure", figure_id=table_id, speak=False))

    def table_html(self, table: Tag) -> str:
        parts = ["<table>"]
        for child in table.find_all(["thead", "tbody", "tfoot", "tr"], recursive=False):
            if child.name == "tr":
                parts.append(self._row_html(child))
                continue
            group = "tbody" if child.name == "tfoot" else child.name
            parts.append(f"<{group}>")
            parts.extend(self._row_html(tr) for tr in child.find_all("tr", recursive=False))
            parts.append(f"</{group}>")
        parts.append("</table>")
        return "".join(parts)

    def _row_html(self, tr: Tag) -> str:
        cells = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            attrs = "".join(
                f' {a}="{cell[a]}"' for a in ("colspan", "rowspan") if str(cell.get(a, "")).isdigit()
            )
            cells.append(f"<{cell.name}{attrs}>{self.inline(*cell.children).html}</{cell.name}>")
        return f"<tr>{''.join(cells)}</tr>"


# --------------------------------------------------------------------------- document


def _remove_noise(article: Tag, warnings: list[str]) -> None:
    errors = []
    for err in article.select(".ltx_ERROR"):
        errors.append(norm_text(err.get_text()))
        err.decompose()
    if errors:
        shown = ", ".join(dict.fromkeys(e for e in errors if e))
        warnings.append(f"Dropped {len(errors)} unconverted LaTeX fragment(s): {shown[:200]}")
    for tag in article.select(_NOISE_SELECTOR):
        tag.decompose()
    for tag in article.find_all(style=_HIDDEN_STYLE):
        tag.decompose()


def _title(soup: BeautifulSoup, article: Tag, conv: _Converter) -> str:
    h1 = article.find(class_="ltx_title_document")
    if h1 is not None:
        title = conv.inline(h1).text
        h1.decompose()
        if title:
            return title
    conv.warn("No document title found")
    return norm_text(soup.title.get_text()) if soup.title else "Untitled"


def author_names(line: str) -> list[str]:
    """Clean author names in one line of an author block (affiliations, emails and marks removed)."""
    names: list[str] = []
    for part in re.split(r",|;|&|\band\b|\u2003|\s{3,}", line):
        # drop "(ORCID)", "[1]", emails, digits and footnote symbols
        junk = r"\([^)]*\)|\[[^\]]*\]|\S+@\S+|[\d*†‡§¶‖∗⋆✉#]"
        name = norm_text(re.sub(junk, " ", part)).strip(" ,.;:-–")
        if (
            re.search(r"[^\W\d_]{2}", name)
            and len(name.split()) <= 6
            and not _NOT_A_NAME.search(name)
            and name not in names
        ):
            names.append(name)
    return names


def _authors(article: Tag) -> list[str]:
    """Best-effort clean author names (affiliations, emails and marks removed)."""
    names: list[str] = []
    for person in article.select(".ltx_personname"):
        for junk in person.select(
            "sup, .ltx_contact, .ltx_role_email, .ltx_role_affiliation, .ltx_role_address, a[href^='mailto:']"
        ):
            junk.decompose()
        first_line = []
        for node in person.descendants:
            if isinstance(node, Tag) and node.name == "br":
                break
            if type(node) is NavigableString:
                first_line.append(str(node))
        names.extend(name for name in author_names("".join(first_line)) if name not in names)
    for authors in article.select(".ltx_authors"):
        authors.decompose()
    return names


class _Planner:
    """Decides which sections are kept, their kind, and which content belongs to each."""

    def __init__(self, conv: _Converter, include_appendix: bool):
        self.conv = conv
        self.include_appendix = include_appendix
        self.units: list[_Unit] = []
        self.in_appendix = False
        self.after_bibliography = False

    def plan(self, article: Tag) -> list[_Unit]:
        self._visit(article, None, dropped_kind=None)
        return self._merge_loose()

    def _visit(self, container: Tag, unit: _Unit | None, dropped_kind: SectionKind | None) -> None:
        """Walk a container's children. `dropped_kind`: kind of a dropped ancestor section."""
        heading = _heading(container) if unit is not None else None
        loose: _Unit | None = None  # current run of article-level content outside sections
        for child in container.find_all(True, recursive=False):
            cls = _classes(child)
            if "ltx_bibliography" in cls:
                self.after_bibliography = True
            elif _is_flat_section(child):
                loose = None
                self._section(child, dropped_kind)
            elif "ltx_abstract" in cls and unit is None:
                loose = None
                self._abstract(child)
            elif unit is not None:
                if child is not heading:
                    unit.nodes.append(child)
            elif not (self.in_appendix and not self.include_appendix):
                if loose is None:
                    loose = _Unit(kind="appendix" if self.in_appendix else "body")
                    self.units.append(loose)
                loose.nodes.append(child)

    def _section(self, sec: Tag, parent_dropped_kind: SectionKind | None) -> None:
        heading = _heading(sec)
        tag = heading.find(class_="ltx_tag") if heading else None
        tag_text = norm_text(tag.get_text()) if tag else ""
        title = self.conv.inline(heading, skip=(tag,) if tag else ()).text if heading else ""
        sec_id = sec.get("id") or ""
        if BIBLIOGRAPHY_TITLE.match(title):
            self.after_bibliography = True
            return
        if ABSTRACT_TITLE.match(title) and not self.in_appendix and not self._has_abstract():
            unit = _Unit(kind="abstract", id="abstract", title="Abstract")  # \section*{Abstract}
            self.units.append(unit)
            self._visit(sec, unit, None)
            return
        explicit = "ltx_appendix" in _classes(sec) or bool(_APPENDIX_ID.match(sec_id))
        if not self.in_appendix and (
            explicit or APPENDIX_TITLE.match(title) or APPENDIX_TITLE.match(tag_text) or self.after_bibliography
        ):
            self.in_appendix = True
            if not explicit and self.after_bibliography:
                self.conv.warn(f"Treating sections from {title!r} on (after the references) as appendix")
        if self.in_appendix and not self.include_appendix:
            return
        kind: SectionKind = "appendix" if self.in_appendix else "body"
        # Sub-sections of dropped back matter are dropped too, unless the appendix started
        # in between (the test paper nests its appendix inside "Acknowledgment").
        dropped = parent_dropped_kind == kind or bool(BACKMATTER_TITLE.match(title))
        unit = _Unit(
            kind=kind,
            el=sec,
            id=sec_id,
            number=_clean_number(tag_text) if tag else None,
            title=title,
            level=_section_level(sec),
        )
        if not dropped:
            self.units.append(unit)
        self._visit(sec, unit, kind if dropped else None)

    def _has_abstract(self) -> bool:
        return any(u.kind == "abstract" for u in self.units)

    def _abstract(self, abstract: Tag) -> None:
        if self._has_abstract():
            self.conv.warn("More than one abstract found; only the first is used")
            return
        unit = _Unit(kind="abstract", id="abstract", title="Abstract")
        unit.nodes = [c for c in abstract.find_all(True, recursive=False) if c.name not in _HEADINGS]
        self.units.append(unit)

    def _merge_loose(self) -> list[_Unit]:
        """Move article-level content into the next section (or a synthetic one at the end)."""
        merged: list[_Unit] = []
        pending: list[Tag] = []
        for unit in self.units:
            if unit.el is None and unit.kind != "abstract":
                pending.extend(unit.nodes)
            elif pending and unit.el is not None:
                unit.nodes[:0] = self._drop_front_matter(pending)
                pending = []
                merged.append(unit)
            else:
                merged.append(unit)
        if pending:
            self.conv.warn("Content outside any section was collected into a 'Main text' section")
            kind: SectionKind = "appendix" if self.in_appendix else "body"
            merged.append(_Unit(kind=kind, id="main", title="Main text", nodes=pending))
        return merged


    def _drop_front_matter(self, nodes: list[Tag]) -> list[Tag]:
        """Keep floats (teasers) and real paragraphs; drop short stray front-matter fragments."""
        kept, dropped = [], []
        for node in nodes:
            text = norm_text(node.get_text())
            if _outermost_floats(node) or len(text) >= _MIN_LOOSE_PARAGRAPH_CHARS:
                kept.append(node)
            elif text:
                dropped.append(text[:40])
        if dropped:
            self.conv.warn(f"Dropped front-matter fragment(s) outside any section: {dropped}")
        return kept


def _figure_alias(units: list[_Unit]) -> dict[str, str]:
    """Map every included float id (and its sub-figure ids) to the outer float id."""
    alias: dict[str, str] = {}
    for unit in units:
        for node in unit.nodes:
            for fig in _outermost_floats(node):
                fig_id = fig.get("id")
                if not fig_id or _float_kind(fig) is None or not any(_float_media(fig)):
                    continue
                alias[fig_id] = fig_id
                for sub in fig.find_all("figure", id=True):
                    alias.setdefault(sub["id"], fig_id)
    return alias


def inline_fragment(
    fragment: str, *, figure_ids: Collection[str] = (), warnings: list[str] | None = None
) -> tuple[str, str, list[str]]:
    """Run LaTeXML-style inline markup through the parser's whitelist: (html, text, figure refs).

    For front ends that mark up plain text themselves (the PDF assembler): ``<math alttext=…>``,
    ``<cite class="ltx_citemacro_citep|citet">`` and ``<a href="#FIGURE_ID">`` come out exactly
    as the arXiv path writes them (see models.py).
    """
    conv = _Converter("", warnings if warnings is not None else [])
    conv.figure_alias = {fig_id: fig_id for fig_id in figure_ids}
    rich = conv.inline(*BeautifulSoup(f"<div>{fragment}</div>", "lxml").div.contents)
    return rich.html, rich.text, rich.refs


def sanitize_table_html(html: str) -> str | None:
    """Re-emit the first <table> of an HTML fragment through the whitelist (None if there is none)."""
    table = BeautifulSoup(html, "lxml").find("table")
    return _Converter("", []).table_html(table) if table is not None else None


def parse_latexml(
    html: str,
    *,
    page_url: str,
    arxiv_id: str,
    version: str | None,
    source_url: str,
    include_appendix: bool = False,
) -> Paper:
    """Parse an arXiv LaTeXML HTML page into a Paper. Figure.images are absolute URLs."""
    soup = BeautifulSoup(html, "lxml")
    article = soup.find("article", class_="ltx_document") or soup.find(class_="ltx_document")
    if article is None:
        raise ValueError("Not a LaTeXML document: no element with class 'ltx_document' found")
    warnings: list[str] = []
    _remove_noise(article, warnings)

    conv = _Converter(page_url, warnings)
    title = _title(soup, article, conv)
    authors = _authors(article)

    units = _Planner(conv, include_appendix).plan(article)
    conv.figure_alias = _figure_alias(units)
    sections = [
        Section(
            id=u.id or f"sec{i}", number=u.number, title=u.title, level=u.level, kind=u.kind,
            blocks=conv.blocks(u.nodes),
        )
        for i, u in enumerate(units, 1)
    ]
    if not any(s.kind != "abstract" for s in sections):
        warnings.append("No body sections found")

    n = 0
    for section in sections:
        for block in section.blocks:
            n += 1
            block.id = f"b{n:04d}"
            if block.figure_id:
                conv.figures[block.figure_id].section_id = section.id

    return Paper(
        paper_id=f"{arxiv_id}{version or ''}".replace("/", "_"),
        arxiv_id=arxiv_id,
        version=version,
        title=title,
        authors=authors,
        source_url=source_url,
        html_url=page_url,
        source="arxiv-html",
        include_appendix=include_appendix,
        sections=sections,
        figures=conv.figures,
        warnings=warnings,
    )
