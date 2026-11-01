"""Offline tests for arxiv.py and parse_html.py (fixture: arXiv 2601.17428v1)."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from paper_reader import arxiv
from paper_reader.arxiv import ArxivHTMLUnavailable, download_figures, fetch_html, parse_arxiv_url
from paper_reader.models import Figure, Paper
from paper_reader.parse_html import parse_latexml

FIXTURE = Path(__file__).parent / "fixtures" / "2601.17428v1.html"  # not in git; see tests/conftest.py
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="arXiv test HTML not downloaded (offline?)")
PAGE_URL = "https://arxiv.org/html/2601.17428"  # unversioned, as arXiv serves it


def parse(html: str | None = None, **kwargs) -> Paper:
    return parse_latexml(
        html if html is not None else FIXTURE.read_text(encoding="utf-8"),
        page_url=kwargs.pop("page_url", PAGE_URL),
        arxiv_id="2601.17428",
        version="v1",
        source_url="https://arxiv.org/pdf/2601.17428",
        **kwargs,
    )


@pytest.fixture(scope="module")
def paper() -> Paper:
    return parse()


@pytest.fixture(scope="module")
def paper_appendix() -> Paper:
    return parse(include_appendix=True)


def block_with(paper: Paper, needle: str):
    return next(b for b in paper.iter_blocks() if needle in b.text)


# --------------------------------------------------------------------------- URLs


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://arxiv.org/abs/2601.17428", ("2601.17428", None)),
        ("https://arxiv.org/pdf/2601.17428", ("2601.17428", None)),
        ("https://arxiv.org/pdf/2601.17428v2", ("2601.17428", "v2")),
        ("https://arxiv.org/pdf/2601.17428.pdf", ("2601.17428", None)),
        ("https://arxiv.org/pdf/2601.17428v3.pdf", ("2601.17428", "v3")),
        ("https://arxiv.org/html/2601.17428v1", ("2601.17428", "v1")),
        ("https://arxiv.org/html/2601.17428v1/#S3", ("2601.17428", "v1")),
        ("http://export.arxiv.org/abs/2601.17428", ("2601.17428", None)),
        ("https://www.arxiv.org/abs/2601.17428?context=cs", ("2601.17428", None)),
        ("arxiv.org/abs/2601.17428", ("2601.17428", None)),
        ("arXiv:2601.17428", ("2601.17428", None)),
        ("arxiv:2601.17428v4", ("2601.17428", "v4")),
        (" 2601.17428v2 ", ("2601.17428", "v2")),
        ("0704.0001", ("0704.0001", None)),
        ("hep-th/9901001", ("hep-th/9901001", None)),
        ("math.GT/0309136v2", ("math.GT/0309136", "v2")),
        ("https://arxiv.org/abs/hep-th/9901001v3", ("hep-th/9901001", "v3")),
    ],
)
def test_parse_arxiv_url(url, expected):
    assert parse_arxiv_url(url) == expected


@pytest.mark.parametrize(
    "url",
    ["", "hello", "https://example.com/abs/2601.17428", "https://arxiv.org/list/cs.RO/new", "2601.1742"],
)
def test_parse_arxiv_url_rejects_non_arxiv(url):
    with pytest.raises(ValueError, match="Not an arXiv link"):
        parse_arxiv_url(url)


# --------------------------------------------------------------------------- document structure


def test_title_and_authors(paper):
    assert paper.title == "Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning"
    assert paper.authors == ["Ziming Li", "Chenhao Li", "Marco Hutter"]
    assert (paper.paper_id, paper.arxiv_id, paper.version) == ("2601.17428v1", "2601.17428", "v1")
    assert paper.warnings == []


def test_sections_flattened_in_reading_order(paper):
    ids = [s.id for s in paper.sections]
    assert ids == [
        "abstract", "S1", "S2", "S2.SS1", "S2.SS2", "S3", "S3.SS1", "S3.SS1.SSS1", "S3.SS1.SSS2",
        "S3.SS1.SSS3", "S3.SS1.SSS4", "S3.SS2", "S3.SS2.SSS1", "S3.SS2.SSS2", "S4", "S4.SS1",
        "S4.SS2", "S4.SS3", "S4.SS4", "S4.SS5", "S5",
    ]
    by_id = {s.id: s for s in paper.sections}
    assert by_id["abstract"].kind == "abstract" and by_id["abstract"].title == "Abstract"
    assert (by_id["S1"].number, by_id["S1"].title, by_id["S1"].level) == ("I", "Introduction", 1)
    assert (by_id["S2.SS1"].number, by_id["S2.SS1"].level) == ("II-A", 2)
    assert by_id["S2.SS1"].title == "Curriculum Reinforcement Learning in Legged Robotics"
    sss = by_id["S3.SS1.SSS2"]
    assert (sss.number, sss.title, sss.level) == ("III-A2", "Markov Decision Process", 3)
    assert by_id["S2"].blocks == []  # only direct content; its text lives in the subsections
    assert all(s.kind == "body" for s in paper.sections[1:])


def test_appendix_excluded_by_default(paper):
    assert not any(s.kind == "appendix" or s.id.startswith("A") for s in paper.sections)
    assert not any(fid.startswith("A") for fid in paper.figures)
    assert not any(ref.startswith("A") for b in paper.iter_blocks() for ref in b.fig_refs)
    # links to the appendix stay plain links, with the IEEE "-A" counter cleaned up
    block = block_with(paper, "is provided in Appendix A.")
    assert '<a href="#A0.SS1">A</a>' in block.html


def test_appendix_included_on_request(paper_appendix):
    appendix = [s for s in paper_appendix.sections if s.kind == "appendix"]
    assert [s.id for s in appendix][:3] == ["A0.SS1", "A0.SS1.SSS1", "A0.SS1.SSS2"]
    assert (appendix[0].number, appendix[0].title, appendix[0].level) == ("A", "Locomotion Training Framework", 2)
    assert {"A0.T1", "A0.T2", "A0.T3", "A0.SS3.tab1"} <= paper_appendix.figures.keys()
    assert paper_appendix.include_appendix


def test_bibliography_and_acknowledgements_dropped(paper, paper_appendix):
    for p in (paper, paper_appendix):
        assert not any(s.id in ("Sx1", "bib") for s in p.sections)
        titles = {s.title.lower() for s in p.sections}
        assert not titles & {"references", "acknowledgment", "acknowledgements"}
        text = " ".join(b.text for b in p.iter_blocks())
        assert "Swiss National Science Foundation" not in text  # acknowledgement
        assert "ANYmal parkour" not in text  # first bibliography entry


def test_block_ids_sequential_and_unique(paper, paper_appendix):
    for p in (paper, paper_appendix):
        ids = [b.id for b in p.iter_blocks()]
        assert ids == [f"b{i:04d}" for i in range(1, len(ids) + 1)]


def test_block_counts(paper, paper_appendix):
    def counts(p):
        types = [b.type for b in p.iter_blocks()]
        return {t: types.count(t) for t in sorted(set(types))}

    assert counts(paper) == {"equation": 8, "figure": 9, "list_item": 7, "paragraph": 45}
    assert counts(paper_appendix) == {"equation": 13, "figure": 13, "list_item": 7, "paragraph": 57}


# --------------------------------------------------------------------------- blocks


def test_paragraph_citations(paper):
    block = block_with(paper, "Mainstream CRL applications")
    assert "Mainstream CRL applications [CITE] predominantly rely" in block.text
    # each citation keeps the bibliography entries it points to (the reader shows their titles)
    assert '<span class="cite" data-refs="bib.bib2 bib.bib7 bib.bib8 bib.bib12 bib.bib16">[1, 2, 3, 4, 5]</span>' in block.html
    prose = [re.sub(r"\$[^$]*\$", "", b.text) for b in paper.iter_blocks()]  # maths may hold "[0,1]"
    assert not any(re.search(r"\[\d", t) for t in prose)


def test_inline_math(paper):
    block = block_with(paper, "is the discount factor")
    assert r"and $\gamma\in[0,1]$ is the discount factor." in block.text
    assert r'<span class="math" data-latex="\gamma\in[0,1]">\(\gamma\in[0,1]\)</span>' in block.html


def test_equations(paper):
    equations = [b for b in paper.iter_blocks() if b.type == "equation"]
    assert [e.eq_number for e in equations] == [str(n) for n in range(1, 9)]
    eq4 = equations[3]
    assert eq4.latex == r"\theta_{k+1}=\mathcal{G}\left(\theta_{k},\mathcal{D}_{k}\right)."
    assert eq4.html == rf'<div class="equation">\[ {eq4.latex} \]</div>'
    assert eq4.text == f"$${eq4.latex}$$"
    assert all(e.latex and "displaystyle" not in e.latex for e in equations)


def test_list_items(paper):
    items = [b for b in paper.iter_blocks() if b.type == "list_item"]
    assert items[0].text.startswith("We propose LP-ACRL")  # bullets are not spoken
    assert items[3].text.startswith("(1) Minimizing the size")  # enumerate labels are kept
    assert items[5].html.startswith("(1) <b>Categorical dimensions</b>:")


def test_figures(paper):
    assert list(paper.figures) == [f"S{s}.F{n}" for s, n in [(1, 1), (3, 2)] + [(4, n) for n in range(3, 10)]]
    fig = paper.figures["S4.F4"]
    assert (fig.kind, fig.number, fig.label, fig.section_id) == ("figure", "4", "Figure 4", "S4.SS3")
    assert fig.images == [
        "https://arxiv.org/html/2601.17428v1/eval_rewards_main.svg",
        "https://arxiv.org/html/2601.17428v1/eval_rewards_legend.svg",
    ]
    assert fig.caption_text.startswith("Evolution of Episodic Reward")
    assert "<b>LP-ACRL</b>" in fig.caption_html and "Fig." not in fig.caption_html
    blocks = [b for b in paper.iter_blocks() if b.type == "figure"]
    assert [b.figure_id for b in blocks] == list(paper.figures)
    assert all(not b.speak for b in blocks)


def test_figure_images_resolve_against_versioned_or_slashed_urls():
    for page_url in ("https://arxiv.org/html/2601.17428v1", "https://arxiv.org/html/2601.17428v1/"):
        images = parse(page_url=page_url).figures["S1.F1"].images
        assert images == ["https://arxiv.org/html/2601.17428v1/real_deploy.png"]


def test_tables(paper_appendix):
    table = paper_appendix.figures["A0.T3"]
    assert (table.kind, table.number, table.label) == ("table", "III", "Table III")
    assert table.caption_text == "Reward function definitions and weights"
    assert table.images == []
    assert table.table_html.startswith("<table><thead><tr><th><b>Term</b></th>")
    assert r'<span class="math" data-latex="1\,dt">' in table.table_html
    untitled = paper_appendix.figures["A0.SS3.tab1"]
    assert (untitled.label, untitled.number, untitled.caption_text) == ("Table", None, "")


def test_fig_refs(paper, paper_appendix):
    block = block_with(paper, "The complete workflow of LP-ACRL is shown in Figure 2.")
    assert block.fig_refs == ["S3.F2"]
    assert '<a class="ref" data-ref="S3.F2" href="#S3.F2">2</a>' in block.html
    block = block_with(paper, "both EPTE-SP (Fig. 3)")
    assert block.fig_refs == ["S4.F3", "S4.F4", "S4.F5"]  # first-mention order, no duplicates
    block = block_with(paper_appendix, "Table II summarizes")
    assert block.fig_refs == ["A0.T2", "A0.T3"]
    for p in (paper, paper_appendix):
        assert all(ref in p.figures for b in p.iter_blocks() for ref in b.fig_refs)


# --------------------------------------------------------------------------- sanitizing

_ALLOWED_TAGS = {"p", "span", "a", "em", "strong", "b", "i", "u", "sub", "sup", "code", "br", "small",
                 "table", "thead", "tbody", "tr", "td", "th", "ul", "ol", "li", "div"}
_ALLOWED_ATTRS = {"class", "data-latex", "data-ref", "data-refs", "href", "colspan", "rowspan"}


class _Whitelist(HTMLParser):
    def __init__(self):
        super().__init__()
        self.bad: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in _ALLOWED_TAGS:
            self.bad.append(tag)
        for name, value in attrs:
            if name not in _ALLOWED_ATTRS or (name == "href" and not value.startswith("#")):
                self.bad.append(f"{tag}[{name}={value}]")


def _all_html(p: Paper) -> list[str]:
    figures = [h for f in p.figures.values() for h in (f.caption_html, f.table_html or "")]
    return [b.html for b in p.iter_blocks()] + figures


def test_html_is_sanitized(paper, paper_appendix):
    for p in (paper, paper_appendix):
        for html in _all_html(p):
            assert not re.search(r"<script|<style|\bon\w+=|style=|javascript:", html, re.I), html
            checker = _Whitelist()
            checker.feed(html)
            assert checker.bad == [], html


def test_text_is_single_line_plain(paper):
    for block in paper.iter_blocks():
        assert "\n" not in block.text and "  " not in block.text and block.text == block.text.strip()
        assert "<span" not in block.text and "&amp;" not in block.text


SYNTHETIC = """<html><body><header>arXiv chrome</header><article class="ltx_document">
<h1 class="ltx_title ltx_title_document">A <math alttext="\\alpha" display="inline">α</math>-Test<span class="ltx_note ltx_role_thanks">thanks</span></h1>
<div class="ltx_authors"><span class="ltx_personname">Ada Lovelace<sup>1</sup>, Alan Turing<br>1 University of Somewhere</span></div>
<figure id="F0" class="ltx_figure"><img src="teaser.png"><figcaption><span class="ltx_tag">Figure 0: </span>Teaser.</figcaption></figure>
<section id="S1" class="ltx_section"><h2 class="ltx_title"><span class="ltx_tag">1 </span>Intro</h2>
<div class="ltx_para"><p class="ltx_p">As <cite class="ltx_cite ltx_citemacro_citet">Smith et al. (<a href="#bib.bib1" class="ltx_ref">2020a</a>)</cite> show
<cite class="ltx_cite ltx_citemacro_citep">(<a href="#bib.bib2" class="ltx_ref">Doe, 2019</a>)</cite>,
see <a href="https://evil.example/x" onclick="steal()">this</a><span class="ltx_note ltx_role_footnote"><sup>1</sup>a footnote</span>
<span class="ltx_ERROR undefined">\\badmacro</span><script>alert(1)</script><span style="display:none">hidden</span>
<span class="ltx_text ltx_font_bold" onmouseover="x()">bold</span> &lt;tag&gt; and Fig.&nbsp;<a href="#F0" class="ltx_ref"><span class="ltx_text ltx_ref_tag">0</span></a>.</p>
<table class="ltx_equationgroup ltx_eqn_align ltx_eqn_table" id="S1.EGx1"><tbody id="S1.E1"><tr class="ltx_equation ltx_eqn_row">
<td class="ltx_eqn_cell ltx_eqn_center_padleft"></td>
<td class="ltx_eqn_cell"><math alttext="\\displaystyle a" display="inline">a</math></td>
<td class="ltx_eqn_cell"><math alttext="\\displaystyle=b+c" display="inline">=b+c</math></td>
<td class="ltx_eqn_cell ltx_eqn_center_padright"></td>
<td class="ltx_eqn_cell ltx_eqn_eqno"><span class="ltx_tag ltx_tag_equation">(1)</span></td></tr></tbody>
<tbody id="S1.E2"><tr class="ltx_equation ltx_eqn_row">
<td class="ltx_eqn_cell ltx_eqn_center_padleft"></td><td class="ltx_eqn_cell"></td>
<td class="ltx_eqn_cell"><math alttext="\\displaystyle=d&lt;e" display="inline">=d</math></td>
<td class="ltx_eqn_cell ltx_eqn_center_padright"></td>
<td class="ltx_eqn_cell ltx_eqn_eqno"><span class="ltx_tag ltx_tag_equation">(2)</span></td></tr></tbody></table>
</div>
<section id="S1.SS0.SSS0.Px1" class="ltx_paragraph"><h5 class="ltx_title">Acknowledgements</h5><div class="ltx_para"><p class="ltx_p">Thanks, funders.</p></div></section>
</section>
<section id="bib" class="ltx_bibliography"><h2 class="ltx_title">References</h2><p>Smith 2020</p></section>
<section id="S2" class="ltx_section"><h2 class="ltx_title"><span class="ltx_tag">A </span>Extra proofs</h2>
<div class="ltx_para"><p class="ltx_p">Supplementary text.</p></div></section>
</article><footer>arXiv footer</footer></body></html>"""


def test_synthetic_document():
    p = parse(SYNTHETIC)
    assert p.title == r"A $\alpha$-Test"
    assert p.authors == ["Ada Lovelace", "Alan Turing"]
    # the teaser figure outside any section moves into the first section; after-bibliography = appendix
    assert [s.id for s in p.sections] == ["S1"]
    blocks = p.sections[0].blocks
    assert [b.type for b in blocks] == ["figure", "paragraph", "equation"]
    para = blocks[1]
    assert para.text == "As Smith et al. [CITE] show [CITE], see this bold <tag> and Fig. 0."
    assert 'Smith et al. <span class="cite" data-refs="bib.bib1">(2020a)</span>' in para.html
    assert '<span class="cite" data-refs="bib.bib2">(Doe, 2019)</span>' in para.html
    assert "<b>bold</b> &lt;tag&gt;" in para.html
    assert para.fig_refs == ["F0"]
    for bad in ("evil", "onclick", "script", "alert", "hidden", "footnote", "badmacro", "onmouseover"):
        assert bad not in para.html
    eq = blocks[2]
    assert eq.latex == r"\begin{aligned}a & =b+c \\ & =d<e\end{aligned}"
    assert eq.eq_number == "1, 2"
    assert r"=d&lt;e" in eq.html
    assert "Thanks, funders" not in " ".join(b.text for b in p.iter_blocks())
    assert any("badmacro" in w for w in p.warnings)
    assert any("after the references" in w for w in p.warnings)
    assert parse(SYNTHETIC, include_appendix=True).sections[-1].kind == "appendix"


def test_rejects_non_latexml_html():
    with pytest.raises(ValueError, match="LaTeXML"):
        parse("<html><body><p>No paper here</p></body></html>")


# --------------------------------------------------------------------------- network (mocked)


@pytest.fixture
def mock_http(monkeypatch):
    """Route arxiv.py's httpx.Client through a MockTransport serving `routes` (else 404)."""
    mock = SimpleNamespace(routes={}, requested=[])

    def handler(request: httpx.Request) -> httpx.Response:
        mock.requested.append(str(request.url))
        assert "paper-reader" in request.headers["user-agent"]
        return mock.routes.get(str(request.url), httpx.Response(404, text="not found"))

    real_client = httpx.Client
    monkeypatch.setattr(arxiv.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    return mock


def test_fetch_html_detects_version_and_unavailable(mock_http):
    mock_http.routes["https://arxiv.org/html/2601.17428"] = httpx.Response(200, text=FIXTURE.read_text(encoding="utf-8"))
    html, final_url, version = fetch_html("2601.17428", None)
    assert (final_url, version) == ("https://arxiv.org/html/2601.17428v1", "v1")
    assert "ltx_document" in html
    with pytest.raises(ArxivHTMLUnavailable, match="PDF fallback is not in v1"):
        fetch_html("2601.17428", "v9")
    mock_http.routes["https://arxiv.org/html/2601.99999"] = httpx.Response(200, text="<html>Conversion failed</html>")
    with pytest.raises(ArxivHTMLUnavailable):
        fetch_html("2601.99999", None)


def test_download_figures(mock_http, tmp_path):
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 20
    base = "https://arxiv.org/html/2601.17428v1/"
    mock_http.routes[base + "a/plot.png"] = httpx.Response(200, content=png, headers={"content-type": "image/png"})
    mock_http.routes[base + "b/plot.png"] = httpx.Response(200, content=png, headers={"content-type": "image/png"})
    mock_http.routes[base + "x1.svg"] = httpx.Response(200, text="<svg/>", headers={"content-type": "image/svg+xml"})
    paper = parse()
    paper.figures = {
        "F1": Figure(id="F1", kind="figure", label="Figure 1", images=[base + "a/plot.png", base + "missing.png"]),
        "F2": Figure(id="F2", kind="figure", label="Figure 2", images=[base + "b/plot.png", base + "a/plot.png"]),
        "F3": Figure(id="F3", kind="figure", label="Figure 3", images=[base + "x1.svg"]),
    }
    paper.warnings = []
    download_figures(paper, tmp_path)
    assert paper.figures["F1"].images == ["figures/plot.png"]
    assert paper.figures["F2"].images == ["figures/plot-2.png", "figures/plot.png"]
    assert paper.figures["F3"].images == ["figures/x1.svg"]
    assert (tmp_path / "figures" / "plot-2.png").read_bytes() == png
    assert len(paper.warnings) == 1 and "missing.png" in paper.warnings[0]

    # a second run reuses the files and leaves already-relative paths alone
    mock_http.requested.clear()
    paper.figures["F4"] = Figure(id="F4", kind="figure", label="Figure 4", images=[base + "x1.svg"])
    download_figures(paper, tmp_path)
    assert mock_http.requested == []
    assert paper.figures["F4"].images == ["figures/x1.svg"]


def test_references_are_kept_for_citation_titles(paper):
    """The bibliography is not read aloud, but the reader shows a citation's title from it."""
    assert paper.references, "no bibliography entries parsed"
    block = block_with(paper, "Mainstream CRL applications")
    ids = re.search(r'data-refs="([^"]+)"', block.html)[1].split()
    assert all(i in paper.references for i in ids)
    first = paper.references[ids[0]]
    assert first.label == "1" and first.title and first.title in first.text
    assert not any("[" in r.label for r in paper.references.values())  # "[14]" is stored as "14"
