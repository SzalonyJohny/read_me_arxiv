"""Contract shared by the PDF extractors, the assembler and the upload step.

    PDF --extract_text_layer() or extract_ocr()--> list[Element] (reading order)
        --assemble()--> Paper (same model as the arXiv HTML path, so filter/clean-up/TTS/reader just work)

Extractors write figure/table crops into <paper_dir>/figures/ and reference them by relative path.
Coordinates are PDF points on the page (x0, y0, x1, y1), origin top-left, as PyMuPDF reports them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ElementType = Literal[
    "title",        # the paper title (first page)
    "authors",      # author/affiliation lines under the title
    "heading",      # section heading; `text` keeps its numbering, e.g. "3.1 Policy learning"
    "text",         # a paragraph (already de-hyphenated and joined across lines/columns/pages)
    "list_item",
    "equation",     # display equation: `latex` when known (OCR), else the raw text in `text`
    "figure",       # a figure image: `image` = crop path
    "table",        # `html` when the extractor recognised the table, else `image`
    "caption",      # "Figure 3: ..." / "Table 1: ..." (the assembler pairs it with its float)
    "footnote",
    "reference",    # one bibliography entry (the assembler drops them)
    "page_header",  # running header / footer / page number / arXiv side stamp (dropped)
    "page_footer",
]
ExtractMethod = Literal["arxiv", "text", "ocr"]


class Element(BaseModel):
    type: ElementType
    page: int  # 0-based page index
    text: str = ""  # plain text; inline maths as $LaTeX$ when the extractor recovers it (OCR)
    latex: str | None = None  # display equations
    eq_number: str | None = None  # "(4)" -> "4" when the extractor separates it
    level: int | None = None  # headings: 1 section, 2 subsection, 3 subsubsection; None = infer from numbering
    bbox: tuple[float, float, float, float] | None = None
    image: str | None = None  # path relative to the paper dir, e.g. "figures/p3_1.png"
    html: str | None = None  # tables recognised as markup (sanitized by the assembler)
    font_size: float | None = None  # text layer only, helps heading detection


class MethodEstimate(BaseModel):
    method: ExtractMethod
    label: str  # e.g. "Fast: PDF text layer"
    seconds: float  # extraction time estimate (not including the LLM clean-up)
    available: bool = True
    recommended: bool = False
    note: str | None = None  # why it is (not) recommended / unavailable


class PdfAnalysis(BaseModel):
    """What the upload step learns from a PDF in well under a second, plus the time estimates."""

    upload_id: str
    filename: str
    pages: int
    size_bytes: int
    title_guess: str | None = None
    text_pages: int  # pages with a usable text layer
    scanned_pages: int  # pages that are (nearly) only images -> need OCR
    math_hint: int = 0  # rough count of maths glyphs/fonts in the text layer (equation-heavy papers want OCR)
    arxiv_id: str | None = None  # e.g. "2601.17428v1" from the side stamp or metadata
    cleanup_seconds: float = 0.0  # estimate for the default LLM clean-up runner
    estimates: list[MethodEstimate] = Field(default_factory=list)
