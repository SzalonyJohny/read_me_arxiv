"""FastAPI app: the HTTP API from plan.md, the static web UI and the data files."""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import pipeline, voicelab
from .pdf import ingest as pdf_ingest
from .pdf.elements import PdfAnalysis

WEB_DIR = pipeline.PROJECT_ROOT / "web"


# ---------------------------------------------------------------- jobs


class JobManager:
    """In-memory background jobs. GPU work (TTS, samples) is serialized on one worker."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._pools = {"io": ThreadPoolExecutor(2, "ingest"), "gpu": ThreadPoolExecutor(1, "tts")}

    def submit(self, kind: str, pool: str, fn: Callable[..., Any], **kwargs) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {"job_id": job_id, "kind": kind, "status": "queued", "progress": 0.0,
                                  "message": "Queued", "result": None, "error": None,
                                  "created": time.time()}
        self._pools[pool].submit(self._run, job_id, fn, kwargs)
        return job_id

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            self._jobs[job_id].update(fields)

    def _run(self, job_id: str, fn: Callable[..., Any], kwargs: dict) -> None:
        self._update(job_id, status="running", message="Starting")
        progress = lambda f, msg: self._update(job_id, progress=round(f, 4), message=msg)  # noqa: E731
        try:
            result = fn(progress=progress, **kwargs)
            self._update(job_id, status="done", progress=1.0, message="Done", result=result)
        except Exception as exc:  # reported to the UI, full trace to the console
            traceback.print_exc()
            self._update(job_id, status="error", message="Failed", error=f"{type(exc).__name__}: {exc}")

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])


jobs = JobManager()
app = FastAPI(title="Paper Reader")


# ---------------------------------------------------------------- request bodies


class IngestRequest(BaseModel):
    url: str
    include_appendix: bool = False
    llm: Literal["haiku", "codex", "none"] | None = None  # None: the first available runner


class SamplesRequest(BaseModel):
    engine: str = "kokoro"
    voices: list[str] = Field(min_length=1, max_length=12)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    paper_id: str | None = None
    text: str | None = Field(None, max_length=2000)


class PdfIngestRequest(BaseModel):
    method: Literal["arxiv", "text", "ocr"]
    include_appendix: bool = False
    llm: Literal["haiku", "codex", "none"] | None = None


class CleanupRequest(BaseModel):
    llm: Literal["haiku", "codex", "none"] | None = None


class LabRenderRequest(BaseModel):
    engine: Literal["kokoro", "qwen3"]
    voices: list[str] = Field(min_length=1, max_length=64)
    models: list[str] | None = None


class LabDesignRequest(BaseModel):
    description: str = Field("", max_length=500)
    models: list[str] | None = None
    redesign_of: str | None = None


class SelectVoiceRequest(BaseModel):
    engine: str
    voice: str
    model: str | None = None


class TTSRequest(BaseModel):
    engine: str | None = None  # None: the selected voice's engine
    voice: str | None = None
    speed: float = Field(1.0, ge=0.5, le=2.0)
    model: str | None = None
    accepted_cost_usd: float = Field(ge=0)


# ---------------------------------------------------------------- helpers


def _runner(llm: str | None) -> str | None:
    """Request value -> pipeline runner: None picks the first available one, "none" disables."""
    return None if llm == "none" else (llm or pipeline.default_cleanup_runner())


def _paper_or_404(paper_id: str):
    try:
        return pipeline.load_paper(paper_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------- API


@app.get("/api/config")
def config() -> dict:
    return {"cleanup_runners": pipeline.cleanup_runners(), "default_cleanup": pipeline.default_cleanup_runner()}


@app.get("/api/papers")
def papers() -> list[dict]:
    return pipeline.list_papers()


@app.post("/api/papers")
def create_paper(req: IngestRequest) -> dict:
    from .arxiv import parse_arxiv_url

    try:
        arxiv_id, _version = parse_arxiv_url(req.url)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    def work(progress):
        paper = pipeline.ingest(req.url, include_appendix=req.include_appendix,
                                llm=_runner(req.llm), progress=progress)
        return {"paper_id": paper.paper_id, "title": paper.title,
                "cleanup": paper.cleanup.model_dump() if paper.cleanup else None}

    return {"job_id": jobs.submit("ingest", "io", work), "arxiv_id": arxiv_id, "paper_id": None}


@app.post("/api/pdf")
def upload_pdf(file: UploadFile) -> PdfAnalysis:
    """Store + analyze a PDF (well under a second) and estimate each extraction method."""
    data = file.file.read(pdf_ingest.MAX_PDF_BYTES + 1)
    if len(data) > pdf_ingest.MAX_PDF_BYTES:
        raise HTTPException(413, f"the PDF is larger than {pdf_ingest.MAX_PDF_BYTES // 2**20} MB")
    try:
        return pdf_ingest.save_upload(data, file.filename or "upload.pdf")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/pdf/{upload_id}/ingest")
def ingest_pdf(upload_id: str, req: PdfIngestRequest) -> dict:
    try:
        pdf_ingest.load_upload(upload_id)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, str(exc)) from exc

    def work(progress):
        paper = pdf_ingest.ingest_pdf(upload_id, req.method, include_appendix=req.include_appendix,
                                      llm=_runner(req.llm), progress=progress)
        return {"paper_id": paper.paper_id, "title": paper.title,
                "cleanup": paper.cleanup.model_dump() if paper.cleanup else None}

    # OCR runs on the GPU worker so it never overlaps with TTS or the voice lab.
    return {"job_id": jobs.submit(f"pdf-{req.method}", "gpu" if req.method == "ocr" else "io", work)}


@app.post("/api/papers/{paper_id}/cleanup")
def recleanup_paper(paper_id: str, req: CleanupRequest) -> dict:
    """Re-run the clean-up of an already ingested paper (e.g. with another LLM runner)."""
    paper = _paper_or_404(paper_id)

    def work(progress):
        cleaned = pipeline.run_cleanup(paper, _runner(req.llm), progress=progress)
        return {"paper_id": paper_id, "cleanup": cleaned.cleanup.model_dump() if cleaned.cleanup else None}

    return {"job_id": jobs.submit("cleanup", "io", work)}


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: str):
    return _paper_or_404(paper_id)


@app.get("/api/papers/{paper_id}/estimate")
def get_estimate(paper_id: str):
    paper = _paper_or_404(paper_id)
    return pipeline.estimate(paper_id, paper=paper)


@app.get("/api/papers/{paper_id}/audio")
def get_audio_runs(paper_id: str):
    _paper_or_404(paper_id)
    return pipeline.list_audio_runs(paper_id)


@app.post("/api/papers/{paper_id}/tts")
def start_tts(paper_id: str, req: TTSRequest) -> dict:
    paper = _paper_or_404(paper_id)
    req.engine = req.engine or pipeline.selected_voice()["engine"]
    # Check the price gate synchronously so the UI gets a 409 instead of a failed job
    # (run_tts re-checks it before voicing anything).
    try:
        est_cost = pipeline.engine_cost(pipeline.estimate(paper_id, paper=paper), req.engine, req.model)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if est_cost > req.accepted_cost_usd + pipeline.COST_TOLERANCE_USD:
        raise HTTPException(409, {"message": "estimate is above the accepted price",
                                  "estimated_cost_usd": est_cost, "accepted_cost_usd": req.accepted_cost_usd})

    def work(progress):
        manifest = pipeline.run_tts(paper_id, engine_name=req.engine, voice=req.voice, speed=req.speed,
                                    model=req.model, accepted_cost_usd=req.accepted_cost_usd,
                                    progress=progress)
        return {"run_id": manifest.run_id, "manifest_url": f"/papers/{paper_id}/audio/{manifest.run_id}/manifest.json",
                "duration": manifest.duration, "cost_usd": manifest.cost_usd}

    return {"job_id": jobs.submit("tts", "gpu", work)}


@app.get("/api/papers/{paper_id}/offline")
def offline_files(paper_id: str, run: str | None = None) -> dict:
    """The files to cache for reading + listening to this paper offline (see web/sw.js)."""
    _paper_or_404(paper_id)
    try:
        return pipeline.offline_files(paper_id, run)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.get("/api/offline/shell")
def offline_shell() -> dict:
    """The app shell the service worker caches: every page, script, style, font and icon in web/."""
    kinds = {".html", ".js", ".css", ".woff2", ".png", ".webmanifest"}
    static = sorted(f"/static/{p.relative_to(WEB_DIR).as_posix()}" for p in WEB_DIR.rglob("*")
                    if p.is_file() and p.suffix in kinds and p.name != "sw.js")
    return {"urls": ["/", "/voices", "/manifest.webmanifest", *static]}


@app.get("/api/tts/engines")
def tts_engines() -> list[dict]:
    return pipeline.engines_info()


@app.post("/api/tts/samples")
def tts_samples(req: SamplesRequest) -> dict:
    if req.paper_id:
        _paper_or_404(req.paper_id)

    def work(progress):
        samples = pipeline.make_samples(req.engine, req.voices, speed=req.speed, paper_id=req.paper_id,
                                        text=req.text, progress=progress)
        return {"samples": samples}

    return {"job_id": jobs.submit("samples", "gpu", work)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "unknown job") from exc


# ---------------------------------------------------------------- voice lab


@app.get("/api/voicelab/{paper_id}")
def voicelab_view(paper_id: str) -> dict:
    _paper_or_404(paper_id)
    try:
        return voicelab.view(paper_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/voicelab/{paper_id}/render")
def voicelab_render(paper_id: str, req: LabRenderRequest) -> dict:
    _paper_or_404(paper_id)
    work = lambda progress: voicelab.render(paper_id, req.engine, req.voices, req.models, progress)  # noqa: E731
    return {"job_id": jobs.submit("voicelab-render", "gpu", work)}


@app.post("/api/voicelab/{paper_id}/design")
def voicelab_design(paper_id: str, req: LabDesignRequest) -> dict:
    _paper_or_404(paper_id)
    if not req.description.strip() and not req.redesign_of:
        raise HTTPException(400, "describe the voice, or name a voice to redesign")
    work = lambda progress: voicelab.design(paper_id, req.description, req.models, req.redesign_of,  # noqa: E731
                                            progress)
    return {"job_id": jobs.submit("voicelab-design", "gpu", work)}


@app.get("/api/voices/selected")
def get_selected_voice() -> dict:
    return pipeline.selected_voice()


@app.post("/api/voices/selected")
def set_selected_voice(req: SelectVoiceRequest) -> dict:
    try:
        return pipeline.select_voice(req.engine, req.voice, req.model)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.delete("/api/voices/qwen/{voice_id}")
def delete_qwen_voice(voice_id: str) -> dict:
    try:
        voicelab.delete_voice(voice_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, f"no voice {voice_id!r}") from exc
    return {"ok": True}


# ---------------------------------------------------------------- pages + files


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    """Served from the root so it may control every page (offline reading)."""
    return FileResponse(WEB_DIR / "sw.js", media_type="text/javascript",
                        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})


@app.get("/manifest.webmanifest", include_in_schema=False)
def web_manifest():
    return FileResponse(WEB_DIR / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/", include_in_schema=False)
def index_page():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/voices", include_in_schema=False)
def voices_page():
    return FileResponse(WEB_DIR / "voices.html")


@app.get("/reader/{paper_id}", include_in_schema=False)
def reader_page(paper_id: str):
    _paper_or_404(paper_id)
    return FileResponse(WEB_DIR / "reader.html")


pipeline.PAPERS_DIR.mkdir(parents=True, exist_ok=True)
pipeline.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
pipeline.QWEN_VOICES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/papers", StaticFiles(directory=pipeline.PAPERS_DIR), name="papers")
app.mount("/samples", StaticFiles(directory=pipeline.SAMPLES_DIR), name="samples")
app.mount("/voice-refs", StaticFiles(directory=pipeline.QWEN_VOICES_DIR), name="voice-refs")


def serve(host: str = "127.0.0.1", port: int = 8000, *, ssl_certfile: str | None = None,
          ssl_keyfile: str | None = None) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)
