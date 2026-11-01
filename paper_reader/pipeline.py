"""Orchestration: ingest -> rule filter -> LLM clean-up -> estimate; accepted TTS run -> mp3 + manifest.

Everything is stored under DATA_ROOT (default: <project>/data, override with PAPER_READER_DATA):

    papers/<paper_id>/paper.json, estimate.json, figures/, audio/<run_id>/{audio.mp3, manifest.json}
    cache/tts/<sha>.wav + <sha>.json    per-unit clip cache (re-runs only voice what changed)
    samples/<paper_id|default>/<engine>/<voice>.mp3
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .models import AudioManifest, AudioSegment, CleanupStats, Estimate, Paper
from .tts.base import Chunk, SynthResult, TTSEngine, get_engine

Progress = Callable[[float, str], None]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("PAPER_READER_DATA", PROJECT_ROOT / "data"))
PAPERS_DIR = DATA_ROOT / "papers"
CACHE_DIR = DATA_ROOT / "cache" / "tts"
SAMPLES_DIR = DATA_ROOT / "samples"

ENGINE_NAMES = ("kokoro", "qwen3", "elevenlabs")
DEFAULT_MODELS = {"qwen3": "0.6B", "elevenlabs": "eleven_flash_v2_5"}  # used when a request names no model
# The narration voice when nothing is selected in the voice lab (chosen there by the user, 2026-10-31).
DEFAULT_VOICE_SELECTION = {"engine": "qwen3", "voice": "technical_male", "model": "0.6B"}
QWEN_VOICES_DIR = DATA_ROOT / "voices" / "qwen"  # designed voices: voices.json + one reference clip each
SELECTED_VOICE_FILE = DATA_ROOT / "voices" / "selected.json"  # the voice narration uses by default
COST_TOLERANCE_USD = 0.005  # accepted price may differ from the re-computed estimate by rounding only

# Pauses inserted between spoken units (seconds).
PAUSE_BEFORE_SECTION = 0.9
PAUSE_AFTER_TITLE = 0.45
PAUSE_BETWEEN_BLOCKS = 0.35

_PAPER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_MENTION_RE = re.compile(
    r"\b(Figures?|Tables?)\s+(\d+[a-z]?(?:\s*(?:,|and|to|&)\s*\d+[a-z]?)*)", re.IGNORECASE
)


def _noop(_fraction: float, _message: str) -> None:
    pass


def scaled_progress(progress: Progress, lo: float, hi: float) -> Progress:
    """Map a sub-task's 0..1 progress into [lo, hi] of the parent task."""
    return lambda f, msg: progress(lo + (hi - lo) * max(0.0, min(1.0, f)), msg)


# ---------------------------------------------------------------- storage


class CostNotAccepted(Exception):
    """The re-computed estimate is above what the user accepted."""

    def __init__(self, estimated_usd: float, accepted_usd: float | None):
        self.estimated_usd = estimated_usd
        self.accepted_usd = accepted_usd
        super().__init__(
            f"estimated cost ${estimated_usd:.4f} is above the accepted ${accepted_usd or 0:.4f}"
        )


def paper_dir(paper_id: str) -> Path:
    if not _PAPER_ID_RE.match(paper_id) or ".." in paper_id:
        raise ValueError(f"invalid paper id: {paper_id!r}")
    return PAPERS_DIR / paper_id


def load_paper(paper_id: str) -> Paper:
    path = paper_dir(paper_id) / "paper.json"
    if not path.exists():
        raise FileNotFoundError(f"no paper {paper_id!r}")
    return Paper.model_validate_json(path.read_text(encoding="utf-8"))


def _write_json(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(model.model_dump_json(indent=1), encoding="utf-8")
    tmp.replace(path)


def save_paper(paper: Paper) -> None:
    _write_json(paper_dir(paper.paper_id) / "paper.json", paper)


def list_audio_runs(paper_id: str) -> list[AudioManifest]:
    audio_root = paper_dir(paper_id) / "audio"
    manifests = sorted(audio_root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime)
    return [AudioManifest.model_validate_json(p.read_text(encoding="utf-8")) for p in manifests]


def delete_paper(paper_id: str) -> dict:
    """Remove everything stored for one paper: its folder (text, figures, narrations, source PDF,
    clean-up exports), its voice-lab and sample clips and, for an uploaded PDF, the upload. The
    shared TTS clip cache stays: clips are keyed by their text and may serve other papers."""
    folder = paper_dir(paper_id)
    if not (folder / "paper.json").exists():
        raise FileNotFoundError(f"no paper {paper_id!r}")
    targets = [folder, SAMPLES_DIR / folder.name]
    if paper_id.startswith("pdf-"):  # pdf-<first 12 hex of the upload id>
        targets += sorted((DATA_ROOT / "uploads").glob(f"{paper_id[4:]}*"))
    targets = [t for t in targets if t.is_dir()]
    freed = sum(f.stat().st_size for t in targets for f in t.rglob("*") if f.is_file())
    for target in targets:
        shutil.rmtree(target)
    return {"paper_id": paper_id, "deleted": [t.relative_to(DATA_ROOT).as_posix() for t in targets],
            "freed_bytes": freed}


def offline_files(paper_id: str, run_id: str | None = None) -> dict:
    """Everything one paper needs offline: its JSON, figure images and one narration (the given
    run, else the newest), with sizes so the page can show download progress."""
    paper = load_paper(paper_id)
    folder = paper_dir(paper_id)
    rel = [Path("paper.json")] + [Path(img) for fig in paper.figures.values() for img in fig.images]
    runs = list_audio_runs(paper_id)
    run = next((r for r in runs if r.run_id == run_id), runs[-1] if runs else None)
    if run_id and (run is None or run.run_id != run_id):
        raise FileNotFoundError(f"no narration {run_id!r} for {paper_id!r}")
    if run:
        rel += [Path(run.audio_file).parent / "manifest.json", Path(run.audio_file)]
    files = [{"url": f"/papers/{paper_id}/{r.as_posix()}", "bytes": (folder / r).stat().st_size}
             for r in dict.fromkeys(rel) if (folder / r).is_file()]
    return {"paper_id": paper_id, "title": paper.title, "run_id": run.run_id if run else None,
            "files": files, "total_bytes": sum(f["bytes"] for f in files)}


def list_papers() -> list[dict]:
    out = []
    for path in sorted(PAPERS_DIR.glob("*/paper.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        paper = Paper.model_validate_json(path.read_text(encoding="utf-8"))
        out.append({
            "paper_id": paper.paper_id,
            "arxiv_id": paper.arxiv_id,
            "source": paper.source,
            "filename": paper.source_url.removeprefix("upload:") if paper.source.startswith("pdf") else None,
            "title": paper.title,
            "audio_runs": [m.run_id for m in list_audio_runs(paper.paper_id)],
        })
    return out


# ---------------------------------------------------------------- engines

_engines: dict[tuple[str, str | None], TTSEngine] = {}
_engines_lock = threading.Lock()


def engine(name: str, model: str | None = None) -> TTSEngine:
    """Engines are cached per (name, model): Kokoro keeps its model on the GPU between runs."""
    model = model or DEFAULT_MODELS.get(name)
    with _engines_lock:
        if (name, model) not in _engines:
            kwargs = {"model": model} if model else {}
            if name == "qwen3":
                kwargs["voice_dir"] = QWEN_VOICES_DIR
            _engines[(name, model)] = get_engine(name, **kwargs)
        return _engines[(name, model)]


def engine_statuses() -> list[dict]:
    statuses = []
    for name in ENGINE_NAMES:
        try:
            eng = engine(name)
            ok, reason = eng.available()
        except Exception as exc:  # import errors etc. -> engine simply unavailable
            statuses.append({"name": name, "available": False, "reason": str(exc)})
            continue
        status = {"name": name, "available": ok, "reason": reason}
        if hasattr(eng, "device"):
            status["device"] = str(eng.device)
        statuses.append(status)
    return statuses


def engines_info() -> list[dict]:
    """For GET /api/tts/engines: availability + voices (+ models for Qwen3-TTS and ElevenLabs)."""
    from .pricing import ELEVENLABS_MODELS
    from .tts.qwen_engine import MODELS as QWEN_MODELS

    info = []
    for status in engine_statuses():
        entry = dict(status, voices=[], default_voice=None, models=[])
        if status["available"] or status["name"] in ("kokoro", "qwen3"):
            try:
                eng = engine(status["name"])
                entry["voices"] = [v.__dict__ for v in eng.voices()]
                entry["default_voice"] = eng.default_voice()
            except Exception as exc:
                entry.update(available=False, reason=str(exc))
        if status["name"] == "elevenlabs":
            entry["models"] = [{"id": mid, **meta} for mid, meta in ELEVENLABS_MODELS.items()]
        elif status["name"] == "qwen3":
            entry["models"] = [{"id": mid, "label": meta["label"], "usd_per_1k_chars": 0.0}
                               for mid, meta in QWEN_MODELS.items()]
        info.append(entry)
    return info


# ---------------------------------------------------------------- selected voice


def selected_voice() -> dict:
    """{engine, voice, model} chosen in the voice lab, else DEFAULT_VOICE_SELECTION."""
    if not SELECTED_VOICE_FILE.exists():
        return dict(DEFAULT_VOICE_SELECTION)
    return json.loads(SELECTED_VOICE_FILE.read_text(encoding="utf-8"))


def select_voice(engine_name: str, voice: str, model: str | None = None) -> dict:
    if engine_name not in ENGINE_NAMES:
        raise ValueError(f"unknown engine {engine_name!r}")
    model = model or DEFAULT_MODELS.get(engine_name)
    if voice not in {v.id for v in engine(engine_name, model).voices()}:
        raise ValueError(f"unknown {engine_name} voice {voice!r}")
    selection = {"engine": engine_name, "voice": voice, "model": model}
    SELECTED_VOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_VOICE_FILE.write_text(json.dumps(selection), encoding="utf-8")
    return selection


def resolve_voice(engine_name: str, voice: str | None, model: str | None) -> tuple[str | None, str | None]:
    """Fill a missing model/voice from the voice-lab selection (same engine), then the defaults."""
    chosen = selected_voice()
    if chosen["engine"] == engine_name:
        model = model or chosen["model"]
        if model == chosen["model"]:
            voice = voice or chosen["voice"]
    return voice, model or DEFAULT_MODELS.get(engine_name)


# ---------------------------------------------------------------- ingest + estimate


CLEANUP_RUNNERS = ("haiku", "codex")  # LLM clean-up options; None = rule-based text only


def cleanup_runners() -> list[dict]:
    """The LLM clean-up options and whether each can run now (no network calls)."""
    from .cleanup import llm_available
    from .cleanup_codex import CODEX_MODEL, codex_available

    haiku_ok, haiku_reason = llm_available()
    codex_ok, codex_reason = codex_available()
    return [
        {"id": "haiku", "label": "Claude Haiku 5.5 (Anthropic API)", "available": haiku_ok, "reason": haiku_reason},
        {"id": "codex", "label": f"Codex · {CODEX_MODEL} (ChatGPT plan)", "available": codex_ok, "reason": codex_reason},
    ]


def default_cleanup_runner() -> str | None:
    """The first available LLM runner (Haiku when an Anthropic key is set, else Codex), or None."""
    return next((r["id"] for r in cleanup_runners() if r["available"]), None)


def run_cleanup(paper: Paper, llm: str | None, *, model: str | None = None, equations: str = "meaning",
                inline: str = "meaning", progress: Progress = _noop) -> Paper:
    """Rule filter, then the chosen LLM pass ("haiku", "codex" or None; `model` overrides the
    runner's default model, e.g. "gpt-5.6-luna" for Codex). equations="meaning" makes the LLM read
    symbols by what the paper defines them as ("the imitation weight" instead of "omega I"); the
    rule-based text can't do that, so it only applies with an LLM runner. inline does the same for
    symbols inside sentences ("the center-of-mass reward $r^c_t$" -> "the center-of-mass reward"). Saves the paper."""
    from .cleanup import cleanup_paper
    from .cleanup_codex import cleanup_paper_codex
    from .textfilter import apply_filter

    if llm not in (None, *CLEANUP_RUNNERS):
        raise ValueError(f"unknown clean-up runner {llm!r}; choose from {', '.join(CLEANUP_RUNNERS)} or none")
    apply_filter(paper)
    save_paper(paper)  # the reader works from here on, even if the LLM pass fails
    runner = next((r for r in cleanup_runners() if r["id"] == llm), None)
    if runner is None:
        paper.cleanup = CleanupStats(skipped_reason="LLM clean-up disabled")
    elif not runner["available"]:
        paper.cleanup = CleanupStats(skipped_reason=runner["reason"])
    else:
        progress(0.0, f"LLM clean-up: {runner['label']}")
        run = cleanup_paper_codex if llm == "codex" else cleanup_paper
        paper.cleanup = run(paper, progress_cb=progress, equations=equations, inline=inline,
                            **({"model": model} if model else {}))
    save_paper(paper)
    estimate(paper.paper_id, paper=paper)
    return paper


def ingest(url: str, *, include_appendix: bool = False, llm: str | None = "haiku", model: str | None = None,
           equations: str = "meaning", inline: str = "meaning", progress: Progress = _noop) -> Paper:
    """Fetch + parse an arXiv paper, then clean it up (see run_cleanup)."""
    from .arxiv import ingest_arxiv

    progress(0.0, "Fetching arXiv HTML")
    paper = ingest_arxiv(url, PAPERS_DIR, include_appendix=include_appendix,
                         progress_cb=scaled_progress(progress, 0.0, 0.4))
    progress(0.4, "Rule-based text filter")
    run_cleanup(paper, llm, model=model, equations=equations, inline=inline,
                progress=scaled_progress(progress, 0.42, 0.98))
    progress(1.0, "Done")
    return paper


def estimate(paper_id: str, *, paper: Paper | None = None) -> Estimate:
    """Recomputed on every call: engine availability (API keys) can change between calls."""
    from .pricing import build_estimate

    paper = paper or load_paper(paper_id)
    est = build_estimate(paper, engine_statuses())
    _write_json(paper_dir(paper_id) / "estimate.json", est)
    return est


def engine_cost(est: Estimate, engine_name: str, model: str | None) -> float:
    _, model = resolve_voice(engine_name, None, model)
    for option in est.tts:
        if option.engine == engine_name and option.model == model:
            if not option.available:
                raise RuntimeError(f"{option.label} is not available: {option.note or 'unavailable'}")
            return option.cost_usd
    raise ValueError(f"no estimate for engine={engine_name!r} model={model!r}")


# ---------------------------------------------------------------- TTS run


@dataclass
class _Unit:
    unit_id: str
    kind: str  # "section_title" | "block"
    text: str
    fig_refs: list[str]  # ids the parser saw in this block
    pending_figs: list[str]  # unmentioned floats placed right before this unit


def _ordered_units(paper: Paper) -> list[_Unit]:
    """speech_units() in reading order, annotated with figure info for cueing the right pane."""
    from .textfilter import speech_units

    spoken = {u.unit_id: u for u in speech_units(paper)}
    units: list[_Unit] = []
    pending: list[str] = []
    for section in paper.sections:
        if section.id in spoken:
            u = spoken[section.id]
            units.append(_Unit(u.unit_id, u.kind, u.text, [], pending))
            pending = []
        for block in section.blocks:
            if block.type == "figure" and block.figure_id:
                pending.append(block.figure_id)
            elif block.id in spoken:
                u = spoken[block.id]
                units.append(_Unit(u.unit_id, u.kind, u.text, list(block.fig_refs), pending))
                pending = []
    return units


def _figure_index(paper: Paper) -> dict[tuple[str, str], str]:
    index = {}
    for fig in paper.figures.values():
        if fig.number:
            index[(fig.kind, re.sub(r"\D", "", fig.number) or fig.number)] = fig.id
    return index


def mentioned_figures(text: str, index: dict[tuple[str, str], str]) -> list[str]:
    """'Figures 3 and 4' / 'Table 2' in spoken text -> figure ids, in order."""
    found: list[str] = []
    for match in _MENTION_RE.finditer(text):
        kind = "table" if match.group(1).lower().startswith("table") else "figure"
        for num in re.findall(r"\d+", match.group(2)):
            fig_id = index.get((kind, num))
            if fig_id and fig_id not in found:
                found.append(fig_id)
    return found


def _cache_key(engine_name: str, model: str | None, voice: str, speed: float, text: str) -> str:
    raw = json.dumps([engine_name, model, voice, round(speed, 3), text], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


GROUP_SENTENCES = 64  # sentences per synthesize_many call: two full Qwen batches, progress every ~30 s


def _cache_get(key: str) -> SynthResult | None:
    from .tts.audio_utils import read_wav

    wav_path, meta_path = CACHE_DIR / f"{key}.wav", CACHE_DIR / f"{key}.json"
    if not (wav_path.exists() and meta_path.exists()):
        return None
    audio, sr = read_wav(wav_path)
    chunks = [Chunk(**c) for c in json.loads(meta_path.read_text(encoding="utf-8"))["chunks"]]
    return SynthResult(audio=audio, sample_rate=sr, chunks=chunks)


def _cache_put(key: str, text: str, result: SynthResult) -> None:
    from .tts.audio_utils import write_wav

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_wav(CACHE_DIR / f"{key}.wav", result.audio, result.sample_rate)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps({"text": text, "chunks": [c.__dict__ for c in result.chunks]}), encoding="utf-8")


def _voice_units(eng: TTSEngine, model: str | None, voice: str, speed: float, units: list[_Unit],
                 progress: Progress) -> tuple[list[SynthResult], float]:
    """A SynthResult per unit (cache first) and the cost of the newly voiced ones.

    Engines with synthesize_many (Qwen3-TTS) get ~GROUP_SENTENCES sentences per call, pooled
    across paragraphs and sections so the GPU runs full batches. Other engines go one unit at a
    time and get the neighbouring text (ElevenLabs uses it for intonation)."""
    from .tts.kokoro_engine import split_sentences

    keys = [_cache_key(eng.name, model, voice, speed, u.text) for u in units]
    results: list[SynthResult | None] = [_cache_get(k) for k in keys]
    todo = [i for i, r in enumerate(results) if r is None]
    if hasattr(eng, "synthesize_many"):
        groups, group, size = [], [], 0
        for i in todo:
            group.append(i)
            size += max(1, len(split_sentences(units[i].text)))
            if size >= GROUP_SENTENCES:
                groups, group, size = groups + [group], [], 0
        if group and groups and size < GROUP_SENTENCES // 2:
            groups[-1] += group  # a small leftover rides along instead of running a near-empty batch
        elif group:
            groups.append(group)
    else:
        groups = [[i] for i in todo]

    cost, done = 0.0, 0
    for group in groups:
        progress(0.97 * done / len(todo), f"Voicing units {done + 1}-{done + len(group)} of {len(todo)} "
                                          f"({len(units) - len(todo)} cached)")
        if hasattr(eng, "synthesize_many"):
            voiced = eng.synthesize_many([units[i].text for i in group], voice, speed)
        else:
            i = group[0]
            voiced = [eng.synthesize(units[i].text, voice, speed,
                                     previous_text=units[i - 1].text if i else None,
                                     next_text=units[i + 1].text if i + 1 < len(units) else None)]
        for i, result in zip(group, voiced, strict=True):
            _cache_put(keys[i], units[i].text, result)
            results[i] = result
            cost += result.cost_usd
        done += len(group)
    return results, cost


def run_id_for(engine_name: str, voice: str, speed: float, model: str | None) -> str:
    parts = [engine_name] + ([model] if model else []) + [voice, f"{speed:g}x"]
    return re.sub(r"[^A-Za-z0-9._-]+", "_", "-".join(parts))


def run_tts(paper_id: str, *, engine_name: str, voice: str | None = None, speed: float = 1.0,
            model: str | None = None, accepted_cost_usd: float | None = None,
            progress: Progress = _noop) -> AudioManifest:
    """Voice the whole paper. Refuses to start unless the accepted price covers the estimate."""
    voice, model = resolve_voice(engine_name, voice, model)
    from .tts.audio_utils import silence, write_mp3

    paper = load_paper(paper_id)
    est_cost = engine_cost(estimate(paper_id, paper=paper), engine_name, model)
    if accepted_cost_usd is None or est_cost > accepted_cost_usd + COST_TOLERANCE_USD:
        raise CostNotAccepted(est_cost, accepted_cost_usd)

    eng = engine(engine_name, model)
    ok, reason = eng.available()
    if not ok:
        raise RuntimeError(f"{engine_name} unavailable: {reason}")
    voice = voice or eng.default_voice()

    units = _ordered_units(paper)
    if not units:
        raise RuntimeError("nothing to speak: the paper has no speech text")
    fig_index = _figure_index(paper)

    results, cost = _voice_units(eng, model, voice, speed, units, progress)
    pieces, segments = [], []
    offset, sample_rate = 0.0, None
    mentioned: set[str] = set()

    for i, (unit, result) in enumerate(zip(units, results, strict=True)):
        if sample_rate is None:
            sample_rate = result.sample_rate
        elif result.sample_rate != sample_rate:
            raise RuntimeError(f"engine changed sample rate mid-run ({result.sample_rate} != {sample_rate})")

        if i:
            gap = PAUSE_BEFORE_SECTION if unit.kind == "section_title" else (
                PAUSE_AFTER_TITLE if units[i - 1].kind == "section_title" else PAUSE_BETWEEN_BLOCKS)
            pieces.append(silence(gap, sample_rate))
            offset += gap

        segments.extend(_unit_segments(unit, result, offset, fig_index, mentioned))
        pieces.append(result.audio)
        offset += len(result.audio) / sample_rate

    run_id = run_id_for(engine_name, voice, speed, model)
    run_dir = paper_dir(paper_id) / "audio" / run_id
    progress(0.98, "Encoding MP3")
    run_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    write_mp3(np.concatenate(pieces).astype(np.float32), sample_rate, run_dir / "audio.mp3")

    manifest = AudioManifest(
        paper_id=paper_id, run_id=run_id, engine=engine_name, voice=voice, model=model, speed=speed,
        audio_file=f"audio/{run_id}/audio.mp3", duration=round(offset, 3), cost_usd=round(cost, 4),
        segments=segments,
    )
    _write_json(run_dir / "manifest.json", manifest)
    progress(1.0, "Done")
    return manifest


def _unit_segments(unit: _Unit, result: SynthResult, offset: float,
                   fig_index: dict[tuple[str, str], str], mentioned: set[str]) -> Iterator[AudioSegment]:
    """Timed segments for one unit, with the figures each piece should bring up on the right."""
    chunks = result.chunks or [Chunk(unit.text, 0.0, len(result.audio) / result.sample_rate)]
    per_chunk = [mentioned_figures(c.text, fig_index) for c in chunks]

    # Floats nobody mentioned yet appear when the narration passes them, and parser-detected refs
    # that the spoken text no longer names literally (LLM rephrasing) attach to the first piece.
    # Anything this unit names out loud is cued at that sentence instead.
    spoken_anywhere = {f for figs in per_chunk for f in figs}
    lead: list[str] = []
    for f in unit.pending_figs + unit.fig_refs:
        if f not in spoken_anywhere and f not in lead and (f in unit.fig_refs or f not in mentioned):
            lead.append(f)
    per_chunk[0] = lead + per_chunk[0]

    for chunk, figs in zip(chunks, per_chunk):
        mentioned.update(figs)
        yield AudioSegment(
            unit_id=unit.unit_id, kind=unit.kind, start=round(offset + chunk.start, 3),
            end=round(offset + chunk.end, 3), text=chunk.text, figure_ids=figs,
        )


# ---------------------------------------------------------------- voice samples


def sample_text_for(paper_id: str | None, max_chars: int = 420) -> str | None:
    """First sentences of the paper's abstract (as it will actually be spoken)."""
    if not paper_id:
        return None
    paper = load_paper(paper_id)
    for block in paper.iter_blocks():
        if block.speak and block.speech:
            text = block.speech
            if len(text) <= max_chars:
                return text
            cut = text.rfind(". ", 0, max_chars)
            return text[: cut + 1] if cut > 80 else text[:max_chars].rsplit(" ", 1)[0] + "."
    return None


def abstract_text(paper_id: str) -> str | None:
    """The whole abstract as it will be spoken."""
    paper = load_paper(paper_id)
    parts = [b.speech for s in paper.sections if s.kind == "abstract" for b in s.blocks if b.speak and b.speech]
    return " ".join(parts) or None


def _voice_group(voice) -> str:
    accent = "British" if voice.language == "en-gb" else "American"
    return f"{accent} {voice.gender or ''}".strip()


def make_samples(engine_name: str, voices: list[str], *, models: list[str] | None = None, speed: float = 1.0,
                 paper_id: str | None = None, text: str | None = None, label: str | None = None,
                 progress: Progress = _noop) -> list[dict]:
    """Render one passage in several voices (["all"] = every voice) and, for engines with models,
    several models, then write a comparison page (index.html) next to the files.
    `label` keeps a run in its own folder, e.g. "abstract"."""
    from .tts.audio_utils import read_wav, write_mp3
    from .tts.samples import DEFAULT_SAMPLE_TEXT, generate_samples, write_index

    models = models or [DEFAULT_MODELS.get(engine_name)]
    text = text or sample_text_for(paper_id) or DEFAULT_SAMPLE_TEXT
    sub = paper_dir(paper_id).name if paper_id else "default"
    out_dir = SAMPLES_DIR / sub / (f"{engine_name}-{label}" if label else engine_name)
    samples, rows = [], []

    for m, model in enumerate(models):
        eng = engine(engine_name, model)
        ok, reason = eng.available()
        if not ok:
            raise RuntimeError(f"{engine_name} unavailable: {reason}")
        info = {v.id: v for v in eng.voices()}
        chosen = list(info) if voices == ["all"] else voices
        model_label = _model_label(engine_name, model) if model else None
        span = scaled_progress(progress, m / len(models), (m + 1) / len(models))
        for sample in generate_samples(eng, chosen, out_dir / model if model else out_dir,
                                       text=text, speed=speed, progress_cb=span):
            v = info.get(sample["voice"])
            notes = [f"grade {v.grade}" if v and v.grade else "", f"{1 / sample['rtf']:.1f}× RT" if sample["rtf"] else ""]
            rows.append({"group": model_label or (_voice_group(v) if v else "Voices"), "name": sample["voice"],
                         "label": v.label if v else "", "note": " · ".join(n for n in notes if n),
                         "seconds": sample["seconds"], "path": sample["path"]})
            samples.append(dict(sample, model=model))

    # Engines that design a voice from a description (Qwen3-TTS) also get their reference clip on the page.
    if hasattr(engine(engine_name, models[0]), "reference_clip"):
        for voice in {s["voice"] for s in samples}:
            ref_wav, ref_text = engine(engine_name, models[0]).reference_clip(voice)
            audio, sr = read_wav(ref_wav)
            ref_mp3 = out_dir / f"reference_{voice}.mp3"
            write_mp3(audio, sr, ref_mp3)
            rows.append({"group": "Designed reference clip (1.7B VoiceDesign, what the Base models clone)",
                         "name": voice, "label": ref_text, "note": "", "seconds": len(audio) / sr, "path": ref_mp3})

    title = load_paper(paper_id).title if paper_id else "Voice samples"
    write_index(out_dir, rows, text, f"{engine_name} voices: {title}")
    for s in samples:
        path = Path(s["path"]).resolve()
        # ?v= busts the browser cache: the file name is per voice, the text changes per paper.
        s["url"] = f"/samples/{path.relative_to(SAMPLES_DIR.resolve()).as_posix()}?v={int(path.stat().st_mtime)}"
        s["path"] = str(path)
    return samples


def _model_label(engine_name: str, model: str) -> str:
    from .pricing import ELEVENLABS_MODELS
    from .tts.qwen_engine import MODELS as QWEN_MODELS

    table = {"qwen3": QWEN_MODELS, "elevenlabs": ELEVENLABS_MODELS}.get(engine_name, {})
    return table.get(model, {}).get("label", model)
