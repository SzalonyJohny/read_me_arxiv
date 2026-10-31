"""Voice lab: hear the first sentences of a paper's abstract in every voice, design new Qwen3-TTS
voices from descriptions, and pick the voice narration uses by default.

Clips live in data/samples/<paper_id>/lab/{kokoro,qwen3/<model>}/<voice>.mp3, indexed by
lab/state.json. The state is tied to the lab text: when the text changes (e.g. after a new
clean-up), old clips are forgotten and re-rendered on request.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from . import pipeline
from .tts.kokoro_engine import split_sentences

LAB_SENTENCES = 2
# The design model goes first: it also designs missing reference clips, which saves a model swap.
QWEN_MODEL_ORDER = ("1.7B-design", "0.6B", "1.7B")

_state_lock = threading.Lock()


def _noop(_fraction: float, _message: str) -> None:
    pass


def lab_text(paper_id: str) -> str:
    """The first sentences of the abstract, exactly as they will be spoken."""
    paper = pipeline.load_paper(paper_id)
    spoken = [b.speech for s in paper.sections if s.kind == "abstract" for b in s.blocks if b.speak and b.speech]
    spoken = spoken or [b.speech for b in paper.iter_blocks() if b.speak and b.speech][:1]
    if not spoken:
        raise ValueError("the paper has no speech text yet")
    return " ".join(split_sentences(" ".join(spoken))[:LAB_SENTENCES])


def _lab_dir(paper_id: str) -> Path:
    return pipeline.SAMPLES_DIR / pipeline.paper_dir(paper_id).name / "lab"


def _load_state(paper_id: str, text: str) -> dict:
    path = _lab_dir(paper_id) / "state.json"
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("text") == text:
            return state
    return {"text": text, "kokoro": {}, "qwen3": {}}


def _save_state(paper_id: str, state: dict) -> None:
    path = _lab_dir(paper_id) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _url(path: Path, mount: str, root: Path) -> str:
    return f"{mount}/{path.resolve().relative_to(root.resolve()).as_posix()}?v={int(path.stat().st_mtime)}"


def _clip(entry: dict, paper_id: str) -> dict | None:
    path = _lab_dir(paper_id) / entry["file"]
    if not path.exists():
        return None
    return {"url": _url(path, "/samples", pipeline.SAMPLES_DIR), "seconds": entry["seconds"], "rtf": entry["rtf"]}


# ---------------------------------------------------------------- read


def view(paper_id: str) -> dict:
    """Everything the voice-lab page shows for one paper."""
    paper = pipeline.load_paper(paper_id)
    text = lab_text(paper_id)
    state = _load_state(paper_id, text)
    kokoro = pipeline.engine("kokoro")
    qwen = pipeline.engine("qwen3")

    def present(entries: dict) -> dict:
        return {key: clip for key, entry in entries.items() if (clip := _clip(entry, paper_id))}

    qwen_voices = []
    for voice in qwen.registry.all():
        ref = qwen.reference_path(voice["id"])
        qwen_voices.append(dict(voice, reference_url=_url(ref, "/voice-refs", pipeline.QWEN_VOICES_DIR)
                                if ref.exists() else None))
    from .tts.qwen_engine import MODELS

    return {
        "paper_id": paper.paper_id, "title": paper.title, "text": text,
        "selected": pipeline.selected_voice(),
        "kokoro": {"voices": [v.__dict__ for v in kokoro.voices()], "samples": present(state["kokoro"])},
        "qwen3": {
            "models": [{"id": m, "label": MODELS[m]["label"]} for m in QWEN_MODEL_ORDER],
            "voices": qwen_voices,
            "samples": {v: s for v, models in state["qwen3"].items() if (s := present(models))},
        },
    }


# ---------------------------------------------------------------- render / design


def render(paper_id: str, engine_name: str, voices: list[str], models: list[str] | None = None,
           progress=_noop) -> dict:
    """Render the lab text in `voices` (["all"] = every voice) of `engine_name`."""
    from .tts.samples import generate_samples

    text = lab_text(paper_id)
    if engine_name == "kokoro":
        jobs = [(None, pipeline.engine("kokoro"))]
    elif engine_name == "qwen3":
        wanted = models or QWEN_MODEL_ORDER
        unknown = set(wanted) - set(QWEN_MODEL_ORDER)
        if unknown:
            raise ValueError(f"unknown Qwen3-TTS model(s): {', '.join(sorted(unknown))}")
        jobs = [(m, pipeline.engine("qwen3", m)) for m in QWEN_MODEL_ORDER if m in wanted]
    else:
        raise ValueError(f"the voice lab supports kokoro and qwen3, not {engine_name!r}")

    known = [v.id for v in jobs[0][1].voices()]
    chosen = known if voices == ["all"] else voices
    if missing := [v for v in chosen if v not in known]:
        raise ValueError(f"unknown {engine_name} voice(s): {', '.join(missing)}")

    rendered = 0
    for i, (model, eng) in enumerate(jobs):
        out_dir = _lab_dir(paper_id) / engine_name / (model or "")
        span = pipeline.scaled_progress(progress, i / len(jobs), (i + 1) / len(jobs))
        results = generate_samples(eng, chosen, out_dir, text=text, progress_cb=span)
        with _state_lock:
            state = _load_state(paper_id, text)
            for r in results:
                entry = {"file": Path(r["path"]).relative_to(_lab_dir(paper_id)).as_posix(),
                         "seconds": r["seconds"], "rtf": r["rtf"]}
                if model:
                    state["qwen3"].setdefault(r["voice"], {})[model] = entry
                else:
                    state["kokoro"][r["voice"]] = entry
            _save_state(paper_id, state)
        rendered += len(results)
    return {"rendered": rendered}


def design(paper_id: str, description: str = "", models: list[str] | None = None,
           redesign_of: str | None = None, progress=_noop) -> dict:
    """Create a Qwen3-TTS voice (or a new take of one), design its reference clip, render the lab text."""
    lab_text(paper_id)  # fail before designing if the paper has nothing to speak
    qwen = pipeline.engine("qwen3", "1.7B-design")
    voice = qwen.registry.add(description, redesign_of=redesign_of)
    progress(0.02, f"Designing voice: {voice['description'][:60]}")
    qwen.reference_clip(voice["id"])
    render(paper_id, "qwen3", [voice["id"]], models, progress=pipeline.scaled_progress(progress, 0.2, 1.0))
    return {"voice_id": voice["id"]}


def delete_voice(voice_id: str) -> None:
    pipeline.engine("qwen3").registry.delete(voice_id)
    chosen = pipeline.selected_voice()
    if chosen["engine"] == "qwen3" and chosen["voice"] == voice_id:
        pipeline.SELECTED_VOICE_FILE.unlink()  # back to DEFAULT_VOICE_SELECTION
