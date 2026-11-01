"""Citation titles: papers ingested before the bibliography was kept get it added afterwards,
without touching their clean-up or narration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from paper_reader import pipeline
from paper_reader.parse_html import parse_latexml

FIXTURE = Path(__file__).parent / "fixtures" / "2601.17428v1.html"  # not in git; see tests/conftest.py
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="arXiv test HTML not downloaded (offline?)")


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "PAPERS_DIR", tmp_path / "papers")
    return tmp_path


def _old_paper():
    """The paper as it was saved before references: no bibliography, plain citation spans, a clean-up."""
    paper = parse_latexml(FIXTURE.read_text(encoding="utf-8"), page_url="https://arxiv.org/html/2601.17428v1",
                          arxiv_id="2601.17428", version="v1", source_url="https://arxiv.org/abs/2601.17428")
    paper.references = {}
    for block in paper.iter_blocks():
        block.html = re.sub(r' data-refs="[^"]*"', "", block.html or "")
        block.speech, block.speech_source = f"spoken {block.id}", "llm"
    return paper


def test_add_references_links_citations_and_keeps_the_cleanup(data):
    old = _old_paper()
    old.sections[0].blocks[0].text += " (edited since)"  # a block whose text changed is left alone
    pipeline.save_paper(old)

    result = pipeline.add_references(old.paper_id, html=FIXTURE.read_text(encoding="utf-8"))
    paper = pipeline.load_paper(old.paper_id)

    assert result["references"] == len(paper.references) > 10
    assert result["linked"] > 0
    cited = [b for b in paper.iter_blocks() if 'class="cite"' in (b.html or "")]
    assert cited and all("data-refs=" in b.html for b in cited if b.id != old.sections[0].blocks[0].id)
    assert all(b.speech == f"spoken {b.id}" and b.speech_source == "llm" for b in paper.iter_blocks())
    for ids in re.findall(r'data-refs="([^"]+)"', " ".join(b.html for b in cited)):
        assert all(i in paper.references for i in ids.split())


def test_pdf_papers_have_no_bibliography_to_add(data):
    old = _old_paper()
    old.source = "pdf-text"
    pipeline.save_paper(old)
    with pytest.raises(ValueError, match="PDF"):
        pipeline.add_references(old.paper_id, html="")
