"""What an uploaded PDF is, in well under a second: pages, text vs scanned pages, maths, title, arXiv id.

The estimates are left empty here; ingest.py fills them in.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import pymupdf

from ..arxiv import parse_arxiv_url
from ..parse_html import norm_text
from .elements import PdfAnalysis
from .text_layer import ARXIV_STAMP, MATH_FONT, clean_text, is_math_char

# Seconds per page of extract_text_layer(), figure crops included. Measured (best of 3) on a 16-thread
# desktop over five arXiv papers of 8-25 pages (2601.17428v1, 2507.10543, 1512.03385, 1412.6980,
# 2006.11239): 0.014-0.048 s per page, 0.031 on average; this is the slowest paper, rounded up.
TEXT_SECONDS_PER_PAGE = 0.05

_MIN_TEXT_CHARS = 50  # a page with fewer characters has no usable text layer
_SCAN_IMAGE_SHARE = 0.5  # ... and if images cover this much of it, it is a scanned page
_ARXIV_REF = re.compile(
    r"arXiv:\s?(?P<a>[\w.\-/]+)|arxiv\.org/(?:abs|pdf)/(?P<b>[\w.\-/]+?)(?:\.pdf)?(?=[\s\"'<]|$)"
    r"|10\.48550/arXiv\.(?P<c>[\w.\-/]+)",
    re.I,
)
_BAD_TITLE = re.compile(r"^(?:untitled|title|paper|document|main|template|microsoft word\b)|\.(?:docx?|tex|pdf|dvi)$", re.I)


def analyze_pdf(pdf_path: Path, *, upload_id: str, filename: str) -> PdfAnalysis:
    """Read the PDF's text layer once (no rendering) and summarise it for the upload step."""
    pdf_path = Path(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        text_pages = scanned_pages = math_hint = 0
        for page in doc:
            chars, math = _text_stats(page)
            math_hint += math
            if chars >= _MIN_TEXT_CHARS:
                text_pages += 1
            elif _image_share(page) >= _SCAN_IMAGE_SHARE:
                scanned_pages += 1
        first = doc[0] if doc.page_count else None
        return PdfAnalysis(
            upload_id=upload_id,
            filename=filename,
            pages=doc.page_count,
            size_bytes=pdf_path.stat().st_size,
            title_guess=_title(doc, first),
            text_pages=text_pages,
            scanned_pages=scanned_pages,
            math_hint=math_hint,
            arxiv_id=_arxiv_id(doc, first),
        )


def _text_stats(page: pymupdf.Page) -> tuple[int, int]:
    """(visible characters, maths characters): glyphs in maths fonts or in Unicode maths ranges."""
    chars = math = 0
    for block in page.get_text("dict", flags=0)["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                text = span["text"].replace(" ", "")
                chars += len(text)
                if MATH_FONT.search(span["font"].split("+")[-1]):
                    math += len(text)
                else:
                    math += sum(map(is_math_char, text))
    return chars, math


def _image_share(page: pymupdf.Page) -> float:
    area = page.rect.get_area() or 1.0
    covered = sum((pymupdf.Rect(i["bbox"]) & page.rect).get_area() for i in page.get_image_info())
    return min(covered / area, 1.0)


def _title(doc: pymupdf.Document, first: pymupdf.Page | None) -> str | None:
    """The largest text on the top half of page 1; the metadata title when it is sane and agrees."""
    meta = norm_text((doc.metadata or {}).get("title") or "")
    meta = meta if 8 <= len(meta) <= 300 and len(meta.split()) >= 2 and not _BAD_TITLE.search(meta) else ""
    printed = _largest_text(first) if first is not None else None
    if printed and meta and _similar(printed, meta):
        return meta  # same title, usually with nicer casing than small caps
    return printed or meta or None


def _largest_text(page: pymupdf.Page) -> str | None:
    lines = []
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]:
        for line in block.get("lines", ()):
            if line["dir"] != (1.0, 0.0) or line["bbox"][1] > 0.5 * page.rect.height:
                continue
            text = norm_text(clean_text("".join(s["text"] for s in line["spans"])))
            size = max(s["size"] for s in line["spans"])
            if len(re.findall(r"[^\W\d_]", text)) >= 8 and len(text.split()) >= 2:
                lines.append((size, line["bbox"][1], text))
    if not lines:
        return None
    biggest = max(size for size, _, _ in lines)
    title = [text for size, _, text in sorted(lines, key=lambda t: t[1]) if size >= biggest - 0.5]
    return norm_text(" ".join(title))[:300]


def _similar(a: str, b: str) -> bool:
    a, b = (re.sub(r"\W+", "", s).lower() for s in (a, b))
    return a in b or b in a or difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def _arxiv_id(doc: pymupdf.Document, first: pymupdf.Page | None) -> str | None:
    """From the side stamp on page 1, then the metadata (XMP identifier, DOI 10.48550/arXiv.*)."""
    sources = []
    if first is not None:
        text = first.get_text()
        stamp = ARXIV_STAMP.search(text)
        sources += [stamp.group(0)] if stamp else []
    sources += [str(v) for v in (doc.metadata or {}).values() if v]
    sources.append(doc.get_xml_metadata() or "")
    for source in sources:
        for match in _ARXIV_REF.finditer(source):
            candidate = (match["a"] or match["b"] or match["c"]).rstrip(".,;)")
            try:
                arxiv_id, version = parse_arxiv_url(candidate)
            except ValueError:
                continue
            return arxiv_id + (version or "")
    return None
