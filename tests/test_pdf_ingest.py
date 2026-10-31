"""PDF upload step: storage, validation, estimates and the recommended method (no GPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

from paper_reader import pipeline
from paper_reader.pdf import ingest, ocr
from paper_reader.pdf.elements import PdfAnalysis

FIXTURE = Path(__file__).parent / "fixtures" / "2601.17428v1.pdf"  # not in git; see tests/conftest.py


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "default_cleanup_runner", lambda: "codex")
    monkeypatch.setattr(ocr, "ocr_available", lambda: (True, None))
    return tmp_path


def _analysis(**kw) -> PdfAnalysis:
    base = dict(upload_id="0" * 16, filename="x.pdf", pages=10, size_bytes=1, text_pages=10, scanned_pages=0)
    return PdfAnalysis(**(base | kw))


@pytest.mark.skipif(not FIXTURE.exists(), reason="arXiv test PDF not downloaded (offline?)")
def test_upload_is_stored_deduplicated_and_analyzed(data_root):
    data = FIXTURE.read_bytes()
    first = ingest.save_upload(data, "../../evil/2601.17428v1.pdf")
    again = ingest.save_upload(data, "copy.pdf")
    assert first.upload_id == again.upload_id and first.filename == "2601.17428v1.pdf"
    assert (data_root / "uploads" / first.upload_id / "original.pdf").read_bytes() == data
    assert ingest.load_upload(first.upload_id).pages == 8
    assert first.cleanup_seconds == pytest.approx(8 * ingest.CLEANUP_SECONDS_PER_PAGE["codex"])
    assert [e.method for e in first.estimates if e.recommended] == ["arxiv"]


def test_rejects_non_pdfs_and_bad_ids(data_root):
    with pytest.raises(ValueError):
        ingest.save_upload(b"hello", "x.pdf")
    with pytest.raises(ValueError):
        ingest.load_upload("../etc/passwd")
    with pytest.raises(FileNotFoundError):
        ingest.load_upload("f" * 16)


@pytest.mark.parametrize(("kw", "best"), [
    ({}, "text"),  # born-digital prose: the text layer is enough
    ({"math_hint": 1000}, "ocr"),  # equation-heavy: LaTeX matters
    ({"scanned_pages": 6, "text_pages": 4}, "ocr"),
    ({"text_pages": 0, "scanned_pages": 10}, "ocr"),
    ({"arxiv_id": "2601.17428v1", "math_hint": 1000}, "arxiv"),
])
def test_recommended_method(data_root, kw, best):
    options = ingest.estimates(_analysis(**kw))
    assert [o.method for o in options if o.recommended] == [best]
    text = next(o for o in options if o.method == "text")
    assert text.available == bool(_analysis(**kw).text_pages)


def test_ocr_unavailable_falls_back_to_text(data_root, monkeypatch):
    monkeypatch.setattr(ocr, "ocr_available", lambda: (False, "no GPU"))
    options = {o.method: o for o in ingest.estimates(_analysis(math_hint=1000))}
    assert not options["ocr"].available and options["ocr"].note == "no GPU"
    assert options["text"].recommended
