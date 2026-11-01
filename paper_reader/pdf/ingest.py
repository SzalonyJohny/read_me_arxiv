"""PDF uploads: store, analyze + estimate (instant), then ingest with the chosen method.

    data/uploads/<upload_id>/original.pdf + analysis.json      (upload_id = sha256 of the bytes)
    data/papers/pdf-<upload_id[:12]>/                           (same layout as arXiv papers, + source.pdf)

Methods: "arxiv" (the PDF is an arXiv paper -> the HTML path), "text" (PDF text layer, seconds),
"ocr" (local DeepSeek-OCR 2, equations as LaTeX, works on scans).
"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path

from .. import pipeline
from ..models import Paper
from .elements import ExtractMethod, MethodEstimate, PdfAnalysis

MAX_PDF_BYTES = 100 * 1024 * 1024
ARXIV_SECONDS = 8.0  # fetch the HTML + figures (measured ~5-8 s for 2601.17428)
CLEANUP_SECONDS_PER_PAGE = {"codex": 2.2, "haiku": 2.0}  # codex measured: 28 s for the 13-page test paper
MATH_HEAVY_PER_PAGE = 40  # math_hint per page above which equations matter enough to prefer OCR
_UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{16}$")


def upload_dir(upload_id: str) -> Path:
    if not _UPLOAD_ID_RE.match(upload_id):
        raise ValueError(f"invalid upload id: {upload_id!r}")
    return pipeline.DATA_ROOT / "uploads" / upload_id


def save_upload(data: bytes, filename: str) -> PdfAnalysis:
    """Store the PDF (deduplicated by content) and analyze it."""
    from .analyze import analyze_pdf

    if not data.startswith(b"%PDF-"):
        raise ValueError("this file is not a PDF")
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"the PDF is larger than {MAX_PDF_BYTES // 2**20} MB")
    upload_id = hashlib.sha256(data).hexdigest()[:16]
    folder = upload_dir(upload_id)
    folder.mkdir(parents=True, exist_ok=True)
    pdf = folder / "original.pdf"
    if not pdf.exists():
        pdf.write_bytes(data)
    name = Path(filename or "upload.pdf").name[:200]
    analysis = analyze_pdf(pdf, upload_id=upload_id, filename=name)
    analysis.cleanup_seconds = cleanup_seconds(analysis.pages, pipeline.default_cleanup_runner())
    analysis.estimates = estimates(analysis)
    (folder / "analysis.json").write_text(analysis.model_dump_json(indent=1), encoding="utf-8")
    return analysis


def load_upload(upload_id: str) -> PdfAnalysis:
    path = upload_dir(upload_id) / "analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"no upload {upload_id!r}")
    return PdfAnalysis.model_validate_json(path.read_text(encoding="utf-8"))


def cleanup_seconds(pages: int, runner: str | None) -> float:
    return round(pages * CLEANUP_SECONDS_PER_PAGE.get(runner or "", 0.0), 1)


def estimates(analysis: PdfAnalysis) -> list[MethodEstimate]:
    """Extraction time per method, with the recommended one marked."""
    from .analyze import TEXT_SECONDS_PER_PAGE
    from .ocr import OCR_LOAD_SECONDS, OCR_SECONDS_PER_PAGE, ocr_available

    pages = max(analysis.pages, 1)
    mostly_scanned = analysis.scanned_pages > 0.3 * pages
    math_heavy = analysis.math_hint > MATH_HEAVY_PER_PAGE * pages
    ocr_ok, ocr_reason = ocr_available()

    options: list[MethodEstimate] = []
    if analysis.arxiv_id:
        options.append(MethodEstimate(
            method="arxiv", label=f"arXiv HTML (arXiv:{analysis.arxiv_id})", seconds=ARXIV_SECONDS,
            note="Exact text, LaTeX maths and figures from arXiv; needs arXiv's HTML version of the paper."))
    options.append(MethodEstimate(
        method="text", label="Fast: PDF text layer", seconds=round(1 + pages * TEXT_SECONDS_PER_PAGE, 1),
        available=analysis.text_pages > 0,
        note=("This PDF has no text layer (scanned)." if analysis.text_pages == 0 else
              "Exact words; equations stay as plain text." + (" Scanned pages will be missing." if
                                                              analysis.scanned_pages else ""))))
    options.append(MethodEstimate(
        method="ocr", label="Accurate: DeepSeek-OCR 2 (local GPU)",
        seconds=round(OCR_LOAD_SECONDS + pages * OCR_SECONDS_PER_PAGE, 1), available=ocr_ok,
        note=ocr_reason if not ocr_ok else "Equations as LaTeX, figures located on the page, works on scans."))

    if analysis.arxiv_id:
        best = "arxiv"
    elif ocr_ok and (mostly_scanned or math_heavy or analysis.text_pages == 0):
        best = "ocr"
    else:
        best = "text" if analysis.text_pages else "ocr"
    for option in options:
        option.recommended = option.method == best and option.available
    return options


def _noop(_fraction: float, _message: str) -> None:
    pass


def ingest_pdf(upload_id: str, method: ExtractMethod, *, include_appendix: bool = False,
               llm: str | None = None, equations: str = "meaning", inline: str = "meaning",
               progress: pipeline.Progress = _noop) -> Paper:
    """Extract (text layer / OCR) or hand over to the arXiv path, then run the usual clean-up."""
    analysis = load_upload(upload_id)
    if method == "arxiv":
        if not analysis.arxiv_id:
            raise ValueError("no arXiv id was found in this PDF")
        return pipeline.ingest(f"https://arxiv.org/abs/{analysis.arxiv_id}", include_appendix=include_appendix,
                               llm=llm, equations=equations, inline=inline, progress=progress)

    from .assemble import assemble

    pdf = upload_dir(upload_id) / "original.pdf"
    paper_id = f"pdf-{upload_id[:12]}"
    folder = pipeline.paper_dir(paper_id)
    shutil.rmtree(folder / "figures", ignore_errors=True)  # crops from an earlier method would linger
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf, folder / "source.pdf")

    extract_until = 0.7 if method == "ocr" else 0.3
    extract_progress = pipeline.scaled_progress(progress, 0.0, extract_until)
    if method == "ocr":
        from ..tts.qwen_engine import release_gpu
        from .ocr import extract_ocr

        release_gpu()  # the OCR model needs the GPU memory a cached TTS voice may hold
        elements = extract_ocr(pdf, folder, progress_cb=extract_progress)
        source = "pdf-ocr"
    elif method == "text":
        from .text_layer import extract_text_layer

        elements = extract_text_layer(pdf, folder, progress_cb=extract_progress)
        source = "pdf-text"
    else:
        raise ValueError(f"unknown method {method!r}")

    progress(extract_until, "Assembling sections")
    paper = assemble(elements, paper_id=paper_id, source=source, source_url=f"upload:{analysis.filename}",
                     include_appendix=include_appendix, arxiv_id=analysis.arxiv_id)
    if not paper.title.strip() and analysis.title_guess:
        paper.title = analysis.title_guess
    pipeline.run_cleanup(paper, llm, equations=equations, inline=inline,
                         progress=pipeline.scaled_progress(progress, extract_until + 0.02, 0.99))
    progress(1.0, "Done")
    return paper
