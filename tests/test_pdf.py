"""Offline tests for the PDF path: analyze, text-layer extraction and the shared assembler.

The fixture PDF is arXiv 2601.17428v1, the same paper as tests/fixtures/2601.17428v1.html; its
arXiv HTML parse is the reference structure (21 sections, 45 paragraphs, 8 equations, 9 figures).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf
import pytest

from paper_reader.models import Paper
from paper_reader.pdf.analyze import analyze_pdf
from paper_reader.pdf.assemble import assemble
from paper_reader.pdf.elements import Element
from paper_reader.pdf.ocr import parse_pages
from paper_reader.pdf.text_layer import extract_text_layer

FIXTURES = Path(__file__).parent / "fixtures"
PDF = FIXTURES / "2601.17428v1.pdf"  # not in git; see tests/conftest.py
RAW = FIXTURES / "ocr_raw_2601.17428v1.json"
pytestmark = pytest.mark.skipif(not PDF.exists(), reason="arXiv test PDF not downloaded (offline?)")


@pytest.fixture(scope="module")
def paper_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("paper")


@pytest.fixture(scope="module")
def elements(paper_dir) -> list[Element]:
    return extract_text_layer(PDF, paper_dir)


def _assemble(elements: list[Element], **kwargs) -> Paper:
    kwargs.setdefault("source", "pdf-text")
    return assemble(elements, paper_id="pdf-test", source_url="upload:test.pdf", **kwargs)


@pytest.fixture(scope="module")
def paper(elements) -> Paper:
    return _assemble(elements, arxiv_id="2601.17428v1")


@pytest.fixture(scope="module")
def paper_appendix(elements) -> Paper:
    return _assemble(elements, include_appendix=True)


def block_with(paper: Paper, needle: str):
    return next(b for b in paper.iter_blocks() if needle in b.text)


def all_text(paper: Paper) -> str:
    return " ".join([b.text for b in paper.iter_blocks()] + [s.title for s in paper.sections])


# --------------------------------------------------------------------------- analyze


def test_analyze_fixture():
    analysis = analyze_pdf(PDF, upload_id="u1", filename="paper.pdf")
    assert analysis.arxiv_id == "2601.17428v1"
    assert (analysis.pages, analysis.text_pages, analysis.scanned_pages) == (8, 8, 0)
    assert analysis.title_guess == "Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning"
    assert analysis.math_hint > 100
    assert analysis.size_bytes == PDF.stat().st_size
    assert analysis.estimates == []


def test_analyze_scanned_pdf(tmp_path):
    scan = pymupdf.open()
    with pymupdf.open(PDF) as source:
        pixmap = source[0].get_pixmap(dpi=50)
        page = scan.new_page(width=source[0].rect.width, height=source[0].rect.height)
        page.insert_image(page.rect, pixmap=pixmap)
    scan.save(tmp_path / "scan.pdf")
    analysis = analyze_pdf(tmp_path / "scan.pdf", upload_id="u2", filename="scan.pdf")
    assert (analysis.pages, analysis.text_pages, analysis.scanned_pages) == (1, 0, 1)
    assert analysis.arxiv_id is None and analysis.title_guess is None


# --------------------------------------------------------------------------- extraction


def test_running_headers_page_numbers_and_stamp_dropped(tmp_path, elements):
    assert not any("arXiv:2601.17428" in e.text for e in elements)  # the rotated side stamp
    doc = pymupdf.open()
    for n in range(4):
        page = doc.new_page(width=612, height=792)
        page.insert_text((72, 40), "Preprint. Under review at Some Conference 2026", fontsize=9)
        page.insert_text((300, 760), str(n + 1), fontsize=9)
        for i in range(30):
            page.insert_text((72, 100 + 12 * i), f"Line {i} on page {n} of the body text, long enough to be prose.", fontsize=10)
    doc.save(tmp_path / "headers.pdf")
    texts = [e.text for e in extract_text_layer(tmp_path / "headers.pdf", tmp_path)]
    assert texts and not any("Preprint" in t for t in texts)
    assert not any(re.fullmatch(r"\d", t) for t in texts)
    assert "Line 0 on page 1" in " ".join(texts)


def test_title_and_authors(paper):
    assert paper.title == "Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning"
    assert paper.authors == ["Ziming Li", "Chenhao Li", "Marco Hutter"]
    assert (paper.arxiv_id, paper.version, paper.source) == ("2601.17428", "v1", "pdf-text")


def test_dehyphenation_keeps_real_hyphens(paper):
    abstract = paper.sections[0]
    assert abstract.kind == "abstract" and abstract.title == "Abstract"
    text = abstract.blocks[0].text
    assert text.startswith("Curriculum learning has demonstrated substantial effectiveness")
    assert "adaptively adjusts" in text and "wide-ranging task spaces" in text
    assert "well-defined difficulty structure" in text
    assert "- " not in text


def test_reading_order_across_columns_pages_and_floats(paper):
    # left column -> (Fig. 1 at the top of the right column) -> right column
    block_with(paper, "suited to the agent’s current competence, removing the need for prior knowledge")
    # end of page 4's right column -> two figures on page 5 -> page 5's left column
    block_with(paper, "similarly indicated in the previous experiment, where the discriminability")


def test_headings_and_levels(elements, paper):
    headings = {(e.text, e.level) for e in elements if e.type == "heading"}
    assert {("I. INTRODUCTION", 1), ("A. Curriculum Reinforcement Learning in Legged Robotics", 2),
            ("1) Definition of Curriculum", 3)} <= headings
    numbers = [s.number for s in paper.sections]
    assert numbers == [
        None, "I", "II", "II-A", "II-B", "III", "III-A", "III-A1", "III-A2", "III-A3", "III-A4", "III-B",
        "III-B1", "III-B2", "IV", "IV-A", "IV-B", "IV-C", "IV-D", "IV-E", "V",
    ]
    by_number = {s.number: s for s in paper.sections}
    assert by_number["II"].title == "Related Work" and by_number["II"].level == 1
    assert by_number["III-A1"].title == "Definition of Curriculum" and by_number["III-A1"].level == 3
    assert by_number["III-B2"].title == "Learning Progress-based Automatic Curriculum Reinforcement Learning (LP-ACRL)"
    assert block_with(paper, "The task space is denoted by").text.startswith("The task space is denoted by")


def test_structure_matches_the_html_parse(paper):
    counts = {kind: sum(b.type == kind for b in paper.iter_blocks()) for kind in ("paragraph", "equation", "list_item")}
    assert len(paper.sections) == 21
    assert counts == {"paragraph": 45, "equation": 8, "list_item": 7}
    assert sorted(paper.figures) == [f"F{n}" for n in range(1, 10)]


def test_captions_paired_with_crops(paper, paper_dir):
    for n, fig in enumerate(sorted(paper.figures.values(), key=lambda f: int(f.number)), 1):
        assert (fig.kind, fig.number, fig.label) == ("figure", str(n), f"Figure {n}")
        assert len(fig.images) == 1 and (paper_dir / fig.images[0]).read_bytes().startswith(b"\x89PNG")
        assert not fig.caption_text.startswith("Fig")
    assert paper.figures["F3"].caption_text.startswith("Episodic Percentage Tracking Error with Stability Penalty")
    assert paper.figures["F3"].caption_text.endswith("even in the most challenging task instance.")
    assert paper.figures["F1"].section_id == "S1"
    assert paper.figures["F8"].section_id is not None  # never mentioned, floated after the acknowledgement
    block = block_with(paper, "achieves the best overall performance in terms of both EPTE-SP")
    assert block.fig_refs == ["F3", "F4", "F5"]
    assert '<a class="ref" data-ref="F3" href="#F3">Fig. 3</a>' in block.html


def test_equations_keep_raw_text(paper):
    equations = [b for b in paper.iter_blocks() if b.type == "equation"]
    assert [b.eq_number for b in equations] == ["1", "2", "3", "4", "5", "6", "7", "8"]  # (2), (7): page bottom
    first = equations[0]
    assert first.latex is None and first.text.startswith("C = (c0, c1")
    assert first.html.startswith('<div class="equation">') and r"\[" not in first.html


def test_references_footnotes_and_back_matter_dropped(elements, paper):
    assert sum(e.type == "reference" for e in elements) == 22
    assert any(e.type == "footnote" and "Robotic Systems Lab" in e.text for e in elements)
    text = all_text(paper)
    for gone in ("Hoeller", "Robotic Systems Lab", "Swiss National Science Foundation", "References", "Acknowledgment"):
        assert gone not in text


def test_appendix_excluded_by_default_included_on_request(paper, paper_appendix):
    assert all(s.kind != "appendix" for s in paper.sections)
    assert "T1" not in paper.figures
    assert '<a class="ref"' not in block_with(paper, "provided in Appendix A").html
    appendix = [s for s in paper_appendix.sections if s.kind == "appendix"]
    assert [s.number for s in appendix] == [None, "A", "A1", "A2", "A3", "B", "B1", "B2", "B3", "B4", "C"]
    assert appendix[0].title == "Appendix" and appendix[1].title == "Locomotion Training Framework"
    assert paper_appendix.figures["T2"].label == "Table II"
    assert paper_appendix.figures["T2"].kind == "table" and paper_appendix.figures["T2"].images
    assert "T2" in block_with(paper_appendix, "summarizes the variables used in the reward functions").fig_refs


def test_citations_in_extracted_text(paper):
    block = block_with(paper, "Mainstream CRL applications")
    assert "applications [CITE], [CITE], [CITE], [CITE], [CITE] predominantly" in block.text
    assert '<span class="cite">[1]</span>' in block.html
    assert "[0, 4.0] m/s" in block_with(paper, "absolute commanded linear velocity").text


# --------------------------------------------------------------------------- assembler (OCR-like input)


def _el(type_: str, text: str = "", page: int = 0, **kwargs) -> Element:
    return Element(type=type_, page=page, text=text, **kwargs)


OCR_LIKE = [
    _el("title", "A Tiny Paper on $\\alpha$-Things"),
    _el("authors", "Jane Doe$^{1,*}$, John Smith$^{2}$, Ann Lee^3†"),
    _el("authors", "$^{1}$University of Somewhere, $^{2}$Example Research Lab"),
    _el("heading", "Abstract"),
    _el("text", "We study $\\alpha$ with a loss in $[0, 1]$ and cite [1, 2]."),
    _el("heading", "1 Introduction"),
    _el("text", "As shown in Figure 1 and Table 1, prior work (Smith et al., 2020; Doe, 2019) and Lee et al. (2021) "
                "fails. Values lie in [0, 1], the bound x ∈ [1, 2] holds, see [2, 1] and [1–2]. It costs \\$5."),
    _el("figure", image="figures/p1_1.png", bbox=(50, 100, 300, 300)),
    _el("caption", "Figure 1: A plot of $\\beta$ over time.", bbox=(50, 305, 300, 320)),
    _el("equation", latex="\\[ \\mathcal{L} = a + b \\tag{1} \\]"),
    _el("heading", "2 Method", page=1),
    _el("heading", "2.1 Details", page=1),
    _el("list_item", "• first point", page=1),
    _el("caption", "Table 1: Results.", page=1, bbox=(50, 80, 300, 95)),
    _el("table", html='<table onclick="x()"><tr><td>a<script>bad()</script></td><td colspan="2">b</td></tr></table>',
        page=1, bbox=(50, 100, 300, 200)),
    _el("heading", "Acknowledgements", page=1),
    _el("text", "We thank everybody.", page=1),
    _el("heading", "References", page=1),
    _el("reference", "[1] A. Author. A paper.", page=1),
    _el("reference", "[2] B. Author. Another paper.", page=1),
    _el("heading", "A Proofs", page=2),
    _el("text", "The proof is trivial.", page=2),
]


def test_assemble_ocr_like_elements():
    paper = _assemble(OCR_LIKE, source="pdf-ocr")
    assert paper.title == "A Tiny Paper on $\\alpha$-Things"
    assert paper.authors == ["Jane Doe", "John Smith", "Ann Lee"]
    assert paper.arxiv_id == "" and paper.version is None
    assert [(s.id, s.number, s.title, s.level, s.kind) for s in paper.sections] == [
        ("abstract", None, "Abstract", 1, "abstract"),
        ("S1", "1", "Introduction", 1, "body"),
        ("S2", "2", "Method", 1, "body"),
        ("S3", "2.1", "Details", 2, "body"),
    ]
    abstract = paper.sections[0].blocks[0]
    assert abstract.text == "We study $\\alpha$ with a loss in $[0, 1]$ and cite [CITE]."
    assert '<span class="math" data-latex="\\alpha">\\(\\alpha\\)</span>' in abstract.html

    intro = paper.sections[1].blocks
    assert intro[0].text == (
        "As shown in Figure 1 and Table 1, prior work [CITE] and Lee et al. [CITE] fails. Values lie in [0, 1], "
        "the bound x ∈ [1, 2] holds, see [CITE] and [CITE]. It costs $5."
    )
    assert intro[0].fig_refs == ["F1", "T1"]
    assert 'Lee et al. <span class="cite">(2021)</span>' in intro[0].html
    assert intro[1].type == "figure" and intro[1].figure_id == "F1"
    equation = intro[2]
    assert (equation.latex, equation.eq_number, equation.text) == ("\\mathcal{L} = a + b", "1", "$$\\mathcal{L} = a + b$$")
    assert equation.html == '<div class="equation">\\[ \\mathcal{L} = a + b \\]</div>'

    figure = paper.figures["F1"]
    assert (figure.label, figure.images, figure.section_id) == ("Figure 1", ["figures/p1_1.png"], "S1")
    assert figure.caption_text == "A plot of $\\beta$ over time."
    table = paper.figures["T1"]
    assert table.kind == "table" and table.caption_text == "Results."
    assert table.table_html == '<table><tr><td>a</td><td colspan="2">b</td></tr></table>'
    assert paper.sections[3].blocks[0].type == "list_item" and paper.sections[3].blocks[0].text == "first point"
    assert [b.id for b in paper.iter_blocks()] == [f"b{n:04d}" for n in range(1, 7)]
    assert "thank" not in all_text(paper) and "proof" not in all_text(paper)

    with_appendix = _assemble(OCR_LIKE, source="pdf-ocr", include_appendix=True)
    last = with_appendix.sections[-1]
    assert (last.number, last.title, last.kind) == ("A", "Proofs", "appendix")
    assert any("after the references" in w for w in with_appendix.warnings)


def test_assemble_sanitizes_html():
    paper = _assemble([_el("heading", "1 Intro"), _el("text", '<script>alert(1)</script> <img src=x onerror="y()"> & "q"')])
    block = paper.sections[0].blocks[0]
    assert block.html == '&lt;script&gt;alert(1)&lt;/script&gt; &lt;img src=x onerror="y()"&gt; &amp; "q"'
    assert block.text == '<script>alert(1)</script> <img src=x onerror="y()"> & "q"'


def test_numeric_citation_ranges_and_reference_bounds():
    paper = _assemble([_el("heading", "1 Intro"), _el("text", "Robust [3, 7–9] and fast [12–14] on [0, 1].")])
    block = paper.sections[0].blocks[0]
    assert block.text == "Robust [CITE] and fast [CITE] on [0, 1]."
    assert '<span class="cite">[3, 7–9]</span>' in block.html
    paper = _assemble([_el("heading", "1 Intro"), _el("text", "See [2] and the set [5]."), _el("reference", "[1] x"),
                       _el("reference", "[2] y")])
    assert paper.sections[0].blocks[0].text == "See [CITE] and the set [5]."


def test_ieee_roman_sections_without_levels():
    headings = ["I. INTRODUCTION", "A. Background", "IV. EXPERIMENTS", "C. Setup", "E. Scaled Locomotion",
                "V. CONCLUSION", "VI. Outlook"]
    elements = [e for h in headings for e in (_el("heading", h), _el("text", f"Body of {h}."))]
    paper = _assemble(elements, source="pdf-ocr")
    assert [(s.number, s.level, s.title) for s in paper.sections] == [
        ("I", 1, "Introduction"), ("I-A", 2, "Background"), ("IV", 1, "Experiments"), ("IV-C", 2, "Setup"),
        ("IV-E", 2, "Scaled Locomotion"), ("V", 1, "Conclusion"), ("VI", 1, "Outlook"),
    ]


def test_author_affiliation_markers_are_stripped():
    paper = _assemble([_el("authors", "Ziming Li $^{1}$, Chenhao Li $^{12}$, Marco Hutter $^{1,*}$"),
                       _el("authors", "Ann Lee^{2}†, Bob Ray‡3"), _el("heading", "1 Intro")])
    assert paper.authors == ["Ziming Li", "Chenhao Li", "Marco Hutter", "Ann Lee", "Bob Ray"]


@pytest.mark.skipif(not RAW.exists(), reason="local OCR fixture missing; see tests/conftest.py")
def test_ocr_fixture_assembles():
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    paper = _assemble(parse_pages(raw["pages"]), source="pdf-ocr", include_appendix=True)
    assert paper.authors == ["Ziming Li", "Chenhao Li", "Marco Hutter"]
    numbers = [s.number for s in paper.sections]
    assert numbers[:3] == [None, "I", "II"] and "IV-E" in numbers and "V" in numbers and "IV-V" not in numbers
    assert [s.number for s in paper.sections if s.kind == "appendix"] == [None, "A", "B", "C"]
