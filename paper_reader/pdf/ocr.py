"""Local OCR extraction with DeepSeek-OCR 2: PDF -> list[Element] in reading order.

The model runs in its own uv project (ocr_worker/, own .venv) as a subprocess:
- its remote code needs torchvision/easydict/matplotlib, which the main env must not get (TTS pins);
- the GPU memory (~8.5 GB peak for a batch of 8 pages) is freed when the worker exits, so the
  server's TTS models can have the 12 GB GPU back.

The worker returns the model's raw grounded markdown per page ("<|ref|>label<|/ref|><|det|>[[x0, y0, x1,
y1]]<|/det|>" + content, boxes normalised to 0-999 of the page image). parse_pages() turns that into
Elements; it is a pure function, tested on a saved raw output (tests/fixtures/ocr_raw_2601.17428v1.json).

Set up once:  cd ocr_worker && uv sync   (plus the 6.8 GB weights in the HF cache, see ocr_available()).
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .elements import Element

MODEL_ID = "unsloth/DeepSeek-OCR-2"  # = deepseek-ai/DeepSeek-OCR-2's weights (same sha256); code patched for new transformers
MODE = "gundam"  # the model card's default: up to 6 tiles of 768 px + one 1024 px global view per page
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."  # set in the worker; kept here for reference

# Measured on this machine (RTX 5070 Ti Laptop 12 GB, nothing else on the GPU), timed from the parent
# process around extract_ocr()/run_worker() on tests/fixtures/2601.17428v1.pdf (8 pages, 2 columns):
# - load = subprocess start -> the worker's "loaded" event (python + torch import, page render, weights to
#   GPU): 7.0 s with the 6.8 GB weights evicted from the OS page cache (posix_fadvise DONTNEED), 4.7 s warm.
# - per page = "loaded" -> last page done, / 8: 88.8 s / 8 = 11.1 s (an earlier identical run: 69 s / 8 =
#   8.6 s; decoding is CPU-bound by PyTorch's per-op overhead, so CPU clocks matter). All 8 pages ran as one
#   batch (12.2k output tokens, ~140-180 tok/s); longer papers run in batches of BATCH_PAGES.
# A lone page doesn't benefit from batching: 1 page took 22.6 s after loading.
OCR_LOAD_SECONDS = 7.0
OCR_SECONDS_PER_PAGE = 11.0

WORKER_DIR = Path(__file__).resolve().parents[2] / "ocr_worker"
WORKER_PYTHON = WORKER_DIR / ".venv" / "bin" / "python"
WORKER_SCRIPT = WORKER_DIR / "worker.py"
WEIGHTS_FILE = "model-00001-of-000001.safetensors"
BATCH_PAGES = 8  # pages generated together (the worker halves it on CUDA out-of-memory)
CROP_DPI = 200  # figure/table crops are re-rendered from the PDF at this resolution (vector-sharp)
CROP_PAD = 4.0  # points around the model's box (its boxes are tight)
BOX_SCALE = 999  # the model's box coordinates are 0..999 of the page image's width/height

ProgressCb = Callable[[float, str], None]


# ------------------------------------------------------------------ availability


def _weights_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    found = try_to_load_from_cache(MODEL_ID, WEIGHTS_FILE)
    return isinstance(found, str) and os.path.exists(found)


def _has_nvidia_gpu() -> bool:
    gpus = Path("/proc/driver/nvidia/gpus")
    return (gpus.is_dir() and any(gpus.iterdir())) or shutil.which("nvidia-smi") is not None


def ocr_available() -> tuple[bool, str | None]:
    """Cheap check (no model load, no torch import): worker env, worker script, cached weights, a GPU."""
    if not WORKER_SCRIPT.exists():
        return False, f"OCR worker script missing ({WORKER_SCRIPT})"
    if not WORKER_PYTHON.exists():
        return False, f"OCR worker env not set up: run `uv sync` in {WORKER_DIR}"
    if not _weights_cached():
        return False, (f"{MODEL_ID} weights (6.8 GB) not in the Hugging Face cache: run "
                       f"`{WORKER_PYTHON} -c \"from huggingface_hub import snapshot_download; "
                       f"snapshot_download('{MODEL_ID}')\"`")
    if not _has_nvidia_gpu():
        return False, "no NVIDIA GPU found (DeepSeek-OCR 2 needs CUDA)"
    return True, None


# ------------------------------------------------------------------ running the worker


def run_worker(pdf_path: Path, out_json: Path, progress_cb: ProgressCb | None = None, *, mode: str = MODE,
               batch: int = BATCH_PAGES, pages: list[int] | None = None, log_path: Path | None = None) -> dict:
    """Run ocr_worker/worker.py on the PDF; returns its result dict (raw text per page + timings)."""
    ok, reason = ocr_available()
    if not ok:
        raise RuntimeError(f"OCR unavailable: {reason}")
    report = progress_cb or (lambda fraction, message: None)
    cmd = [str(WORKER_PYTHON), "-I", str(WORKER_SCRIPT), "--pdf", str(Path(pdf_path).resolve()),
           "--out", str(Path(out_json).resolve()), "--mode", mode, "--batch", str(batch)]
    if pages is not None:
        cmd += ["--pages", ",".join(str(p) for p in pages)]
    # Offline: ocr_available() checked the weights are cached; don't let a load wait on the network.
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    log_path = log_path or Path(out_json).with_suffix(".log")
    started = time.monotonic()
    count = 0
    timings: dict[str, float] = {}
    error = None
    report(0.0, "Loading OCR model")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log, text=True, env=env, cwd=WORKER_DIR)
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = event.get("event")
                if kind == "start":
                    count = int(event["pages"])
                elif kind == "loaded":
                    timings["load"] = time.monotonic() - started
                    report(_fraction(0, count), f"OCR page 0/{count}")
                elif kind == "page":
                    report(_fraction(int(event["done"]), count), f"OCR page {event['done']}/{count}")
                elif kind == "error":
                    error = event.get("message", "unknown error")
                elif kind == "done":
                    timings["total"] = time.monotonic() - started
            code = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    if error or code != 0 or not Path(out_json).exists():
        tail = log_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
        raise RuntimeError(f"OCR worker failed ({error or f'exit code {code}'}); log {log_path}: " + " | ".join(tail))
    result = json.loads(Path(out_json).read_text(encoding="utf-8"))
    result["wall_seconds"] = {k: round(v, 2) for k, v in timings.items()}
    return result


def _fraction(done: int, pages: int) -> float:
    """Progress 0..0.98 weighted by the measured load and per-page times (the parse/crops take the rest)."""
    total = OCR_LOAD_SECONDS + max(pages, 1) * OCR_SECONDS_PER_PAGE
    return round(0.98 * (OCR_LOAD_SECONDS + done * OCR_SECONDS_PER_PAGE) / total, 4)


def extract_ocr(pdf_path: Path, paper_dir: Path, progress_cb: ProgressCb | None = None) -> list[Element]:
    """OCR the PDF with DeepSeek-OCR 2 (subprocess) -> Elements in reading order.

    Figures (and tables) are cropped from the PDF page at CROP_DPI into paper_dir/figures/ and referenced
    as Element.image = "figures/<name>.png". The raw model output stays in paper_dir/ocr_raw.json (and the
    worker's log in ocr_raw.log), so parse_pages() can be re-run without the GPU."""
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    result = run_worker(Path(pdf_path), paper_dir / "ocr_raw.json", progress_cb)
    elements = parse_pages(result["pages"])
    write_crops(Path(pdf_path), paper_dir, elements)
    if progress_cb:
        progress_cb(1.0, f"OCR done ({len(result['pages'])} pages)")
    return elements


def write_crops(pdf_path: Path, paper_dir: Path, elements: list[Element]) -> None:
    """Render each figure/table box from the PDF (sharper than the OCR raster) and set Element.image."""
    import pymupdf

    figures = paper_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    counts: dict[int, int] = {}
    with pymupdf.open(pdf_path) as doc:
        for el in elements:
            if el.type not in ("figure", "table") or el.bbox is None:
                continue
            page = doc[el.page]
            x0, y0, x1, y1 = el.bbox
            clip = pymupdf.Rect(x0 - CROP_PAD, y0 - CROP_PAD, x1 + CROP_PAD, y1 + CROP_PAD) & page.rect
            if clip.is_empty or clip.width < 4 or clip.height < 4:
                continue
            counts[el.page] = counts.get(el.page, 0) + 1
            name = f"ocr_p{el.page + 1}_{counts[el.page]}.png"
            page.get_pixmap(clip=clip, dpi=CROP_DPI, alpha=False).save(figures / name)
            el.image = f"figures/{name}"


# ------------------------------------------------------------------ raw output -> Elements (pure)

_BLOCK = re.compile(r"<\|ref\|>(?P<label>.*?)<\|/ref\|><\|det\|>(?P<boxes>.*?)<\|/det\|>", re.S)
_INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.S)
_CITATION = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")
_BULLET = re.compile(r"^\s*(?:[•◦▪‣●○■□∙·]|[-*–](?=\s))\s*")
_EQ_NUMBER = re.compile(r"\s*(?:\\q?quad|\\hfill|~)(?:\s|\\q?quad|\\hfill|~)*\(\s*([0-9]+[a-z]?|[A-Z]\.?[0-9]+)\s*\)\s*$")
_EQ_TAG = re.compile(r"\\tag\*?\{([^{}]*)\}")
_SECTION_NUMBER = re.compile(r"^(?:[IVXLC]+\.|[A-Z]\.|\d+(?:\.\d+)*\.?)\s+\S")
_TABLE_LABEL = re.compile(r"^(?:TABLE|Table)\s+[IVXLC\d]+[.:]?$")
_REFERENCES = re.compile(r"^(?:[IVXLC\d.]+\s+)?(?:references|bibliography|literature cited)$", re.I)
_TERMINAL = re.compile(r"[.!?:;][\"'”’)\]]*$")
_ARXIV_STAMP = re.compile(r"^arXiv:\d{4}\.\d{4,5}")

LABELS = {
    "title": "title", "sub_title": "heading", "section_header": "heading",
    "text": "text", "equation": "equation", "image": "figure", "figure": "figure", "table": "table",
    "figure_title": "caption", "image_caption": "caption", "table_caption": "caption", "figure_caption": "caption",
    "caption": "caption", "footnote": "footnote", "table_footnote": "footnote",
    "header": "page_header", "page_header": "page_header", "footer": "page_footer", "page_footer": "page_footer",
    "page_number": "page_footer",
}
FLOATS = {"figure", "table", "caption", "page_header", "page_footer", "footnote"}


def split_blocks(raw: str) -> list[tuple[str, list[list[float]], str]]:
    """Raw model output -> [(label, boxes 0..999, content)] in output order."""
    blocks = []
    matches = list(_BLOCK.finditer(raw))
    lead = raw[:matches[0].start()] if matches else raw
    if lead.strip():
        blocks.append(("text", [], lead.strip()))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        try:
            boxes = ast.literal_eval(m.group("boxes").strip())
            boxes = [list(map(float, b)) for b in boxes if len(b) == 4]
        except (ValueError, SyntaxError, TypeError):
            boxes = []
        blocks.append((m.group("label").strip(), boxes, raw[m.end():end].strip()))
    return blocks


def inline_math(text: str) -> str:
    """\\( x \\) -> $x$ (a citation the model typeset as maths, \\( [3] \\), becomes plain [3]); tidy spacing."""
    def repl(m: re.Match) -> str:
        body = " ".join(m.group(1).split())
        return body if _CITATION.fullmatch(body) else f"${body}$"

    text = _INLINE_MATH.sub(repl, text)
    parts = re.split(r"(\$[^$]+\$)", text)
    for i in range(0, len(parts), 2):  # outside maths only
        part = re.sub(r"[ \t]+", " ", parts[i])
        part = re.sub(r" ([,.;:!?)\]])", r"\1", part)
        parts[i] = re.sub(r"([(\[]) ", r"\1", part)
    return "".join(parts).strip()


def equation_parts(content: str) -> tuple[str, str | None]:
    """'\\[ a = b \\quad (3) \\]' -> ('a = b', '3')."""
    latex = content.strip()
    for left, right in (("\\[", "\\]"), ("$$", "$$"), ("\\(", "\\)")):
        if latex.startswith(left) and latex.endswith(right) and len(latex) >= len(left) + len(right):
            latex = latex[len(left):len(latex) - len(right)].strip()
            break
    number = None
    tag = _EQ_TAG.search(latex)
    if tag:
        number = tag.group(1).strip().strip("()")
        latex = _EQ_TAG.sub("", latex).strip()
    else:
        m = _EQ_NUMBER.search(latex)
        if m and m.start() > 0:
            number = m.group(1)
            latex = latex[:m.start()].rstrip()
    latex = re.sub(r"(?:\\q?quad|\\,|\\;|~|\s)+$", "", latex) if number else latex
    return latex, number


def _to_points(box: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (round(x0 / BOX_SCALE * width, 2), round(y0 / BOX_SCALE * height, 2),
            round(x1 / BOX_SCALE * width, 2), round(y1 / BOX_SCALE * height, 2))


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _text_type(text: str, box: list[float] | None) -> str:
    """Classify a plain 'text' block: running header/footer, heading the model missed, or paragraph."""
    if _ARXIV_STAMP.match(text) or (box and box[2] < 70 and box[3] - box[1] > 300):  # arXiv side stamp
        return "page_header"
    if box and len(text) < 120:
        if box[3] <= 45:
            return "page_header"
        if box[1] >= 955 or (box[1] >= 920 and re.fullmatch(r"\d{1,4}", text)):
            return "page_footer"
    if "\n" not in text and len(text) <= 90 and _SECTION_NUMBER.match(text) and not _TERMINAL.search(text):
        return "heading"
    return "text"


def parse_page(raw: str, page: int, width: float, height: float) -> list[Element]:
    """One page of raw DeepSeek-OCR 2 output -> Elements (bbox in PDF points, origin top-left)."""
    out: list[Element] = []
    for label, boxes, content in split_blocks(raw):
        kind = LABELS.get(label, "text")
        points = [_to_points(b, width, height) for b in boxes]
        bbox = points[0] if points else None
        content = re.sub(r"</?center>", "", content).strip()
        if kind == "figure":
            if out and out[-1].type == "figure" and out[-1].page == page and out[-1].bbox and points:
                out[-1].bbox = _union([out[-1].bbox, *points])  # sub-panels of one figure -> one crop
            else:
                out.append(Element(type="figure", page=page, bbox=_union(points) if points else None))
            continue
        if kind == "table":
            html = inline_math(content) if "<table" in content else None
            text = "" if html else inline_math(content)
            out.append(Element(type="table", page=page, bbox=_union(points) if points else None, html=html, text=text))
            continue
        if not content:
            continue
        if kind == "equation":
            latex, number = equation_parts(content)
            if latex:
                out.append(Element(type="equation", page=page, latex=latex, eq_number=number, bbox=bbox))
            continue
        if kind in ("title", "heading"):
            text = inline_math(re.sub(r"^#+\s*", "", " ".join(content.split())))
            out.append(Element(type="title" if kind == "title" and page == 0 else "heading", page=page,
                               text=text, bbox=bbox))
            continue
        if kind == "caption":
            text = inline_math(" ".join(content.split()))
            prev = out[-1] if out else None
            if prev is not None and prev.type == "caption" and _TABLE_LABEL.match(prev.text):
                prev.text = f"{prev.text.rstrip('.:')}: {text}"  # "TABLE I" + "OBSERVATION SPACE ..." -> one caption
            else:
                out.append(Element(type="caption", page=page, text=text, bbox=bbox))
            continue
        for line in (ln.strip() for ln in content.split("\n")):  # list-like blocks put one item per line
            if not line:
                continue
            line_kind = kind if kind != "text" else _text_type(line, boxes[0] if boxes else None)
            if line_kind == "text" and _BULLET.match(line):
                line_kind, line = "list_item", _BULLET.sub("", line)
            if line_kind == "heading":
                line = re.sub(r"^#+\s*", "", line)
            out.append(Element(type=line_kind, page=page, text=inline_math(line), bbox=bbox))
    # A loop that slipped past the n-gram guard: drop exact consecutive repeats.
    return [el for i, el in enumerate(out)
            if not (i and el.type in ("text", "list_item") and el.text and el.text == out[i - 1].text)]


def parse_pages(pages: list[dict]) -> list[Element]:
    """The worker's pages ({"page", "width", "height", "raw"}) -> Elements for the whole document:
    per-page parse, then authors under the title, bibliography entries, and paragraphs that continue
    across a column/page break (previous one ends mid-sentence, next starts lower-case)."""
    elements: list[Element] = []
    for p in sorted(pages, key=lambda p: p["page"]):
        elements.extend(parse_page(p.get("raw", ""), int(p["page"]), float(p["width"]), float(p["height"])))
    _mark_authors(elements)
    _mark_references(elements)
    return join_split_paragraphs(elements)


def _mark_authors(elements: list[Element]) -> None:
    """Text between the title and the abstract/first heading on page 1 is the author block."""
    title = next((i for i, el in enumerate(elements) if el.type == "title"), None)
    if title is None:
        return
    for el in elements[title + 1:]:
        if el.page != elements[title].page or el.type != "text" or len(el.text) > 400:
            break
        if re.match(r"^\W*abstract\b", el.text, re.I):
            break
        el.type = "authors"


def _mark_references(elements: list[Element]) -> None:
    in_refs = False
    for el in elements:
        if el.type in ("heading", "title"):
            in_refs = bool(_REFERENCES.match(el.text.strip()))
        elif in_refs and el.type in ("text", "list_item"):
            el.type = "reference"


def join_split_paragraphs(elements: list[Element]) -> list[Element]:
    """Join a paragraph the layout split (end of a column/page -> next column/page, maybe past floats)."""
    out: list[Element] = []
    last_text: Element | None = None
    for el in elements:
        if el.type == "text" and last_text is not None and el.text[:1].islower() and not _TERMINAL.search(last_text.text):
            if last_text.text.endswith("-") and last_text.text[-2:-1].isalpha():
                last_text.text = last_text.text[:-1] + el.text
            else:
                last_text.text = f"{last_text.text} {el.text}"
            continue
        out.append(el)
        if el.type == "text":
            last_text = el
        elif el.type not in FLOATS:
            last_text = None
    return out
