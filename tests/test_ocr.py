"""DeepSeek-OCR 2 extraction: the raw-output parser on a saved model output, crops, and the worker
plumbing with a fake worker. No GPU needed; the real-model smoke test is opt-in (PAPER_READER_OCR_SMOKE=1)."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from paper_reader.pdf import ocr
from paper_reader.pdf.ocr import equation_parts, inline_math, parse_page, parse_pages, split_blocks, write_crops

FIXTURES = Path(__file__).parent / "fixtures"
PDF = FIXTURES / "2601.17428v1.pdf"
RAW = FIXTURES / "ocr_raw_2601.17428v1.json"  # worker output for the whole test PDF (gundam mode)
pytestmark = pytest.mark.skipif(not (PDF.exists() and RAW.exists()),
                                reason="local OCR fixture / arXiv test PDF missing; see tests/conftest.py")


@pytest.fixture(scope="module")
def raw_pages() -> list[dict]:
    return json.loads(RAW.read_text(encoding="utf-8"))["pages"]


@pytest.fixture(scope="module")
def elements(raw_pages):
    return parse_pages(raw_pages)


# ------------------------------------------------------------------ pieces


def test_split_blocks_reads_labels_boxes_and_multi_box_blocks():
    raw = ("<|ref|>sub_title<|/ref|><|det|>[[220, 479, 355, 492]]<|/det|>\n## I. INTRODUCTION\n\n"
           "<|ref|>text<|/ref|><|det|>[[92, 865, 489, 926], [517, 66, 914, 127]]<|/det|>\nSpans two columns.\n\n"
           "<|ref|>image<|/ref|><|det|>[[510, 198, 909, 402]]<|/det|>\n\n")
    blocks = split_blocks(raw)
    assert [b[0] for b in blocks] == ["sub_title", "text", "image"]
    assert blocks[0][2] == "## I. INTRODUCTION"
    assert blocks[1][1] == [[92, 865, 489, 926], [517, 66, 914, 127]]
    assert blocks[2][2] == ""
    assert split_blocks("no tags at all") == [("text", [], "no tags at all")]
    assert split_blocks("<|ref|>text<|/ref|><|det|>[[1, 2]]bad<|/det|>\nx")[0][1] == []  # malformed box


def test_equation_number_is_split_off():
    assert equation_parts(r" \[ \theta_{k+1}=\mathcal{G}\left(\theta_{k},\mathcal{D}_{k}\right). \quad (4) \] ") == (
        r"\theta_{k+1}=\mathcal{G}\left(\theta_{k},\mathcal{D}_{k}\right).", "4")
    assert equation_parts(r"\[ a = b \tag{12} \]") == ("a = b", "12")
    assert equation_parts(r"\[ x = f(2) \]") == ("x = f(2)", None)  # an argument, not an equation number
    assert equation_parts(r"\[ \mathcal{C}=(c_{0}),\quad c_{j}\in\Delta, \quad (1) \]") == (
        r"\mathcal{C}=(c_{0}),\quad c_{j}\in\Delta,", "1")


def test_inline_math_and_spacing():
    assert inline_math(r"applications  \( [1] \) ,  \( [2] \)  predominantly") == "applications [1], [2] predominantly"
    assert inline_math(r"each  \( c_{j} \)  represents") == "each $c_{j}$ represents"
    assert inline_math(r"falls ( \( k_{f} = K \)  if no fall)") == "falls ($k_{f} = K$ if no fall)"
    assert inline_math(r"Ziming Li \( ^{1} \) , Chenhao") == "Ziming Li $^{1}$, Chenhao"


def test_parse_page_types_boxes_and_merges():
    raw = textwrap.dedent(r"""
        <|ref|>title<|/ref|><|det|>[[130, 88, 864, 136]]<|/det|>
        # A Paper Title

        <|ref|>text<|/ref|><|det|>[[100, 66, 490, 156]]<|/det|>
        • First bullet.
        • Second bullet.

        <|ref|>text<|/ref|><|det|>[[84, 565, 480, 580]]<|/det|>
        C. Terrain Parameters for Scaled Locomotion

        <|ref|>image<|/ref|><|det|>[[85, 285, 290, 375]]<|/det|>


        <|ref|>image<|/ref|><|det|>[[293, 384, 480, 467]]<|/det|>


        <|ref|>figure_title<|/ref|><|det|>[[82, 691, 490, 784]]<|/det|>
        Fig. 4. Two panels of one figure.

        <|ref|>figure_title<|/ref|><|det|>[[682, 275, 740, 286]]<|/det|>
        TABLE I

        <|ref|>figure_title<|/ref|><|det|>[[533, 290, 890, 302]]<|/det|>
        OBSERVATION SPACE DIMENSIONS.

        <|ref|>table<|/ref|><|det|>[[600, 313, 819, 430]]<|/det|>
        <table><tr><td>Joint</td><td>\( q_{j} \)</td></tr></table>

        <|ref|>equation<|/ref|><|det|>[[576, 816, 912, 832]]<|/det|>
         \[ E = mc^{2}. \quad (3) \]

        <|ref|>text<|/ref|><|det|>[[480, 960, 520, 975]]<|/det|>
        7
    """)
    els = parse_page(raw, 0, 612.0, 792.0)
    assert [e.type for e in els] == ["title", "list_item", "list_item", "heading", "figure", "caption", "caption",
                                     "table", "equation", "page_footer"]
    title, first = els[0], els[1]
    assert title.text == "A Paper Title"
    assert title.bbox == pytest.approx((130 / 999 * 612, 88 / 999 * 792, 864 / 999 * 612, 136 / 999 * 792), abs=0.01)
    assert first.text == "First bullet."
    figure = els[4]  # the two sub-panels became one figure box
    assert figure.bbox == pytest.approx((85 / 999 * 612, 285 / 999 * 792, 480 / 999 * 612, 467 / 999 * 792), abs=0.01)
    assert els[6].text == "TABLE I: OBSERVATION SPACE DIMENSIONS."
    assert els[7].html == "<table><tr><td>Joint</td><td>$q_{j}$</td></tr></table>"
    assert (els[8].latex, els[8].eq_number) == ("E = mc^{2}.", "3")
    assert parse_page(raw, 2, 612.0, 792.0)[0].type == "heading"  # 'title' blocks after page 1 are headings


def test_consecutive_repeats_are_dropped():
    raw = "".join(f"<|ref|>text<|/ref|><|det|>[[82, {y}, 491, {y + 10}]]<|/det|>\nThe same line again.\n\n"
                  for y in (100, 120, 140))
    assert len(parse_page(raw, 0, 612, 792)) == 1


# ------------------------------------------------------------------ the whole test paper


def test_fixture_elements_by_type(elements):
    counts = Counter(e.type for e in elements)
    assert counts["title"] == 1 and counts["authors"] == 1
    assert counts["equation"] == 13
    assert counts["reference"] == 22
    assert counts["figure"] == 9  # Fig. 1-9; Fig. 4's eight sub-plots are one crop
    assert counts["table"] == 4 and all(e.html and e.html.startswith("<table>") for e in elements if e.type == "table")
    assert counts["heading"] >= 18 and counts["text"] >= 50
    assert counts["page_header"] == counts["page_footer"] == 0  # this paper has none; the arXiv stamp isn't emitted
    pages = {e.page for e in elements}
    assert pages == set(range(8))


def test_fixture_reading_order_and_content(elements):
    assert elements[0].type == "title"
    assert elements[0].text == "Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning"
    assert elements[1].text.startswith("Ziming Li")
    headings = [e.text for e in elements if e.type == "heading"]
    assert headings[:3] == ["I. INTRODUCTION", "II. RELATED WORK", "A. Curriculum Reinforcement Learning in Legged Robotics"]
    assert "C. Terrain Parameters for Scaled Locomotion Experiment" in headings  # labelled 'text' by the model
    equations = [e for e in elements if e.type == "equation"]
    assert [e.eq_number for e in equations] == [str(n) for n in range(1, 14)]
    assert equations[6].latex.startswith(r"c_{j+1}(\zeta)=\frac{e^{L P_{c_{j}}(\zeta)/\beta}}")
    captions = [e.text for e in elements if e.type == "caption"]
    assert [c.split(".")[0] for c in captions if c.startswith("Fig.")] == [f"Fig" for _ in range(9)]
    assert "TABLE I: OBSERVATION SPACE DIMENSIONS FOR THE LOCOMOTION TASK." in captions
    refs = [e.text for e in elements if e.type == "reference"]
    assert refs[0].startswith("[1] D. Hoeller") and refs[-1].startswith("[22] J. Schulman")
    for e in elements:
        assert e.bbox is not None and 0 <= e.bbox[0] < e.bbox[2] <= 612.5 and 0 <= e.bbox[1] < e.bbox[3] <= 792.5


def test_fixture_paragraphs_joined_across_columns_and_pages(elements):
    texts = [e.text for e in elements if e.type == "text"]
    for prev, el in zip(elements, elements[1:]):  # lower-case starts remain only after an equation ("where ...")
        if el.type == "text" and el.text[:1].islower():
            assert prev.type == "equation", el.text
    joined = next(t for t in texts if t.startswith("However, in real-world scenarios"))
    assert "suited to the agent's current competence, removing the need" in joined  # column 1 -> column 2, past Fig. 1
    page_break = next(e for e in elements if e.type == "text" and e.text.startswith("Consistent with Sec. IV-C"))
    assert page_break.page == 3 and "experiment, where the discriminability" in page_break.text  # page 4 -> 5


def test_crops_written_from_the_pdf(tmp_path, elements):
    els = [e.model_copy() for e in elements if e.type in ("figure", "table")]
    write_crops(PDF, tmp_path, els)
    assert all(e.image and (tmp_path / e.image).is_file() for e in els)
    assert els[0].image == "figures/ocr_p1_1.png"
    from PIL import Image

    with Image.open(tmp_path / els[0].image) as img:  # Fig. 1: ~245 x 163 pt at 200 dpi
        assert 650 < img.width < 720 and 430 < img.height < 490


# ------------------------------------------------------------------ worker plumbing (fake worker)


FAKE_WORKER = textwrap.dedent("""
    import argparse, json
    p = argparse.ArgumentParser()
    for a in ("--pdf", "--out", "--mode", "--batch", "--pages"):
        p.add_argument(a)
    args = p.parse_args()
    raw = open(%r, encoding="utf-8").read()
    pages = json.loads(raw)["pages"][:2]
    print(json.dumps({"event": "start", "pages": 2}), flush=True)
    print("noise that is not JSON", flush=True)
    print(json.dumps({"event": "loaded", "seconds": 1.0}), flush=True)
    for i, page in enumerate(pages):
        print(json.dumps({"event": "page", "page": page["page"], "done": i + 1, "pages": 2}), flush=True)
    json.dump({"model": "fake", "pages": pages}, open(args.out, "w"))
    print(json.dumps({"event": "done", "result": args.out}), flush=True)
""")


def test_extract_ocr_with_a_fake_worker(tmp_path, monkeypatch):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE_WORKER % str(RAW), encoding="utf-8")
    monkeypatch.setattr(ocr, "WORKER_PYTHON", Path(sys.executable))
    monkeypatch.setattr(ocr, "WORKER_SCRIPT", script)
    monkeypatch.setattr(ocr, "ocr_available", lambda: (True, None))
    calls = []
    paper_dir = tmp_path / "paper"
    els = ocr.extract_ocr(PDF, paper_dir, lambda f, m: calls.append((f, m)))
    assert [m for _, m in calls] == ["Loading OCR model", "OCR page 0/2", "OCR page 1/2", "OCR page 2/2",
                                     "OCR done (2 pages)"]
    fractions = [f for f, _ in calls]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert {e.page for e in els} == {0, 1} and els[0].type == "title"
    assert (paper_dir / "ocr_raw.json").is_file()
    assert (paper_dir / els[[e.type for e in els].index("figure")].image).is_file()


def test_worker_error_is_raised(tmp_path, monkeypatch):
    script = tmp_path / "bad_worker.py"
    script.write_text('import json, sys\nprint(json.dumps({"event": "error", "message": "CUDA out of memory"}))\n'
                      'sys.exit(1)\n', encoding="utf-8")
    monkeypatch.setattr(ocr, "WORKER_PYTHON", Path(sys.executable))
    monkeypatch.setattr(ocr, "WORKER_SCRIPT", script)
    monkeypatch.setattr(ocr, "ocr_available", lambda: (True, None))
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        ocr.extract_ocr(PDF, tmp_path / "paper")


def test_ocr_available_reports_missing_pieces(tmp_path, monkeypatch):
    ok, reason = ocr.ocr_available()
    assert isinstance(ok, bool) and (reason is None) == ok
    monkeypatch.setattr(ocr, "WORKER_PYTHON", tmp_path / "missing" / "python")
    ok, reason = ocr.ocr_available()
    assert not ok and "uv sync" in reason
    monkeypatch.setattr(ocr, "WORKER_SCRIPT", tmp_path / "missing.py")
    assert ocr.ocr_available()[0] is False


def test_constants():
    assert ocr.MODEL_ID == "unsloth/DeepSeek-OCR-2"
    assert ocr.OCR_SECONDS_PER_PAGE > 0 and ocr.OCR_LOAD_SECONDS > 0


# ------------------------------------------------------------------ real model (opt-in, GPU)


@pytest.mark.skipif(os.environ.get("PAPER_READER_OCR_SMOKE") != "1" or not ocr.ocr_available()[0],
                    reason="GPU smoke test: set PAPER_READER_OCR_SMOKE=1 (needs the OCR worker env, weights and CUDA)")
def test_gpu_smoke_first_page(tmp_path):
    result = ocr.run_worker(PDF, tmp_path / "raw.json", pages=[0], batch=1)
    els = parse_pages(result["pages"])
    assert els[0].type == "title" and "Rough Terrain Locomotion" in els[0].text
    assert any(e.type == "figure" for e in els) and any(e.type == "heading" for e in els)
