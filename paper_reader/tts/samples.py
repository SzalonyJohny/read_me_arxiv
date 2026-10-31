"""Voice samples: the same short passage rendered in several voices, for comparing them.

    uv run python -m paper_reader.tts.samples --voices af_heart,am_michael --out data/samples
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from html import escape
from pathlib import Path

from .audio_utils import duration, write_mp3
from .base import TTSEngine, get_engine

# About 25 seconds of speech, with a number, a unit, a Greek letter in words and a figure mention.
DEFAULT_SAMPLE_TEXT = (
    "In this work, we train a small transformer to predict protein stability from sequence alone. "
    "As Figure 2 shows, the prediction error drops sharply during the first two thousand steps "
    "and then levels off. With the temperature tau set to 0.5, the model reaches an accuracy of "
    "87 percent while using only 3.2 gigabytes of memory, roughly a quarter of what the baseline needs."
)


def sample_path(out_dir: Path, voice: str, speed: float) -> Path:
    return out_dir / (f"{voice}.mp3" if speed == 1.0 else f"{voice}_{speed:g}x.mp3")


def generate_samples(engine: TTSEngine, voices: list[str], out_dir: Path, text: str | None = None,
                     speed: float = 1.0, progress_cb: Callable[[float, str], None] | None = None) -> list[dict]:
    """Render `text` (default: DEFAULT_SAMPLE_TEXT) in each voice to <out_dir>/<voice>.mp3.

    Returns [{"voice", "path", "seconds", "rtf"}]; rtf is wall-clock synthesis time divided by
    audio length, so the first voice also pays for any lazy model loading.
    """
    text = text or DEFAULT_SAMPLE_TEXT
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i, voice in enumerate(voices):
        if progress_cb:
            progress_cb(i / len(voices), f"Rendering {voice} ({i + 1}/{len(voices)})")
        started = time.perf_counter()
        result = engine.synthesize(text, voice, speed)
        elapsed = time.perf_counter() - started
        path = sample_path(out_dir, voice, speed)
        write_mp3(result.audio, result.sample_rate, path)
        seconds = duration(result.audio, result.sample_rate)
        samples.append({"voice": voice, "path": path, "seconds": round(seconds, 2),
                        "rtf": round(elapsed / seconds, 3) if seconds else None})
    if progress_cb:
        progress_cb(1.0, "Done")
    return samples


_INDEX_CSS = """
:root { color-scheme: light dark; --bg:#f7f5f0; --fg:#1d1d1f; --muted:#6b6b70; --card:#fff; --line:#e3e0d8; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#ececee; --muted:#9a9aa0; --card:#202024; --line:#333338; } }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
main { max-width:900px; margin:0 auto; }
h1 { font-size:1.4rem; margin:0 0 4px; } h2 { font-size:1.05rem; margin:28px 0 8px; }
.text { color:var(--muted); background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.row { display:grid; grid-template-columns:150px 1fr 96px 56px minmax(220px, 300px); gap:12px; align-items:center;
       padding:8px 12px; background:var(--card); border:1px solid var(--line); border-radius:8px; margin-bottom:6px; }
.row b { font-family:ui-monospace, monospace; font-weight:600; } .muted { color:var(--muted); white-space:nowrap; } audio { width:100%; height:32px; }
@media (max-width:700px) { .row { grid-template-columns:1fr 1fr; } .row audio { grid-column:1 / -1; } }
"""


def write_index(out_dir: Path, rows: list[dict], text: str, title: str) -> Path:
    """A static comparison page (<out_dir>/index.html) with one player per row, in groups.

    rows: [{"group", "name", "label", "note", "seconds", "path"}]; paths may sit in sub-folders."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        src = Path(os.path.relpath(row["path"], out_dir)).as_posix()
        groups.setdefault(row["group"], []).append(
            f'<div class="row"><b>{escape(row["name"])}</b><span>{escape(row.get("label") or "")}</span>'
            f'<span class="muted">{escape(row.get("note") or "")}</span>'
            f'<span class="muted">{row["seconds"]:.0f} s</span>'
            f'<audio controls preload="none" src="{escape(src)}"></audio></div>')
    body = "".join(f"<h2>{escape(name)}</h2>{''.join(items)}" for name, items in groups.items())
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Voice samples</title><style>{_INDEX_CSS}</style></head><body><main>"
            f"<h1>{escape(title)}</h1><p class=\"muted\">{len(rows)} clips · the same passage in each</p>"
            f'<div class="text">{escape(text)}</div>{body}</main></body></html>')
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voices", required=True, help="comma-separated voice ids, e.g. af_heart,am_michael")
    parser.add_argument("--out", type=Path, default=Path("data/samples"), help="output directory")
    parser.add_argument("--engine", default="kokoro", choices=["kokoro", "qwen3", "elevenlabs"])
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--text", help="passage to speak (default: a built-in scientific passage)")
    args = parser.parse_args(argv)

    engine = get_engine(args.engine)
    ok, reason = engine.available()
    if not ok:
        parser.exit(1, f"{args.engine} is unavailable: {reason}\n")
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    for sample in generate_samples(engine, voices, args.out, text=args.text, speed=args.speed,
                                   progress_cb=lambda _f, message: print(message, flush=True)):
        print(f"  {sample['path']}  {sample['seconds']:.2f}s  rtf {sample['rtf']}")


if __name__ == "__main__":
    main()
