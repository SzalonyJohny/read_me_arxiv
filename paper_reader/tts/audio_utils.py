"""Small audio helpers shared by the TTS engines and the pipeline (mono float32 PCM)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf


def silence(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(max(0, round(seconds * sr)), dtype=np.float32)


def concat(arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays])


def duration(audio: np.ndarray, sr: int) -> float:
    return len(audio) / sr


def trim_silence(audio: np.ndarray, sr: int, keep_lead: float, keep_trail: float,
                 threshold: float = 3e-4) -> np.ndarray:
    """Shorten the silence (|sample| <= threshold, default about -70 dBFS) before and after the
    sound to at most keep_lead / keep_trail seconds. Audio that is all silence is returned as is."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    start = max(0, loud[0] - round(keep_lead * sr))
    end = min(len(audio), loud[-1] + 1 + round(keep_trail * sr))
    return audio[start:end]


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """16-bit PCM WAV: lossless enough for clips that end up in a 64 kbit/s MP3, at half the size."""
    sf.write(path, np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


def read_wav(source: Path | BinaryIO) -> tuple[np.ndarray, int]:
    """Read a WAV file (path or file object) as mono float32; multichannel audio is downmixed."""
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    return audio.mean(axis=1, dtype=np.float32), sr


def write_mp3(audio: np.ndarray, sr: int, path: Path, bitrate: str = "64k") -> None:
    """Encode mono float32 PCM to MP3 by piping it through ffmpeg; no temporary files."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; it is needed to write MP3 files")
    pcm = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0).astype("<f4").tobytes()
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
        "-codec:a", "libmp3lame", "-b:a", bitrate, "-ac", "1", "-f", "mp3", "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"ffmpeg failed to encode {path.name}: {detail}")
    path.write_bytes(proc.stdout)
