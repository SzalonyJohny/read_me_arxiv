"""Deleting a paper from the library: its files go, shared data stays, running jobs block it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from paper_reader import pipeline, server
from paper_reader.models import Block, Paper, Section


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "PAPERS_DIR", tmp_path / "papers")
    monkeypatch.setattr(pipeline, "SAMPLES_DIR", tmp_path / "samples")
    monkeypatch.setattr(pipeline, "CACHE_DIR", tmp_path / "cache" / "tts")
    return tmp_path


def _store(paper_id: str, data) -> None:
    paper = Paper(paper_id=paper_id, arxiv_id="", title=f"Paper {paper_id}", source_url="x",
                  sections=[Section(id="S1", title="Intro", blocks=[Block(id="b0001", type="paragraph", text="Hi.")])])
    pipeline.save_paper(paper)
    folder = pipeline.paper_dir(paper_id)
    for rel in ("figures/x1.png", "audio/kokoro-af_heart-1x/audio.mp3", "llm/part_1.json"):
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(b"0" * 100)
    (data / "samples" / paper_id / "lab").mkdir(parents=True)
    (data / "samples" / paper_id / "lab" / "af_heart.mp3").write_bytes(b"0" * 10)


def test_delete_removes_the_paper_its_clips_and_its_upload(data):
    _store("pdf-abcdef012345", data)
    _store("2601.17428v1", data)
    upload = data / "uploads" / "abcdef0123456789"
    upload.mkdir(parents=True)
    (upload / "original.pdf").write_bytes(b"%PDF-1.7")
    cache = data / "cache" / "tts"
    cache.mkdir(parents=True)
    (cache / "clip.wav").write_bytes(b"0")

    result = pipeline.delete_paper("pdf-abcdef012345")
    assert sorted(result["deleted"]) == ["papers/pdf-abcdef012345", "samples/pdf-abcdef012345",
                                         "uploads/abcdef0123456789"]
    assert result["freed_bytes"] > 300
    assert not (data / "papers" / "pdf-abcdef012345").exists() and not upload.exists()
    assert (cache / "clip.wav").exists()  # the shared clip cache stays
    assert [p["paper_id"] for p in pipeline.list_papers()] == ["2601.17428v1"]  # the other paper stays

    with pytest.raises(FileNotFoundError):
        pipeline.delete_paper("pdf-abcdef012345")
    with pytest.raises(ValueError):
        pipeline.delete_paper("../outside")


def test_delete_endpoint(data, monkeypatch):
    _store("2601.17428v1", data)
    client = TestClient(server.app)
    assert client.delete("/api/papers/nope").status_code == 404

    monkeypatch.setattr(server.jobs, "busy", lambda paper_id: paper_id == "2601.17428v1")
    busy = client.delete("/api/papers/2601.17428v1")
    assert busy.status_code == 409 and pipeline.paper_dir("2601.17428v1").exists()

    monkeypatch.setattr(server.jobs, "busy", lambda paper_id: False)
    done = client.delete("/api/papers/2601.17428v1")
    assert done.status_code == 200 and done.json()["paper_id"] == "2601.17428v1"
    assert not pipeline.paper_dir("2601.17428v1").exists()


def test_jobs_know_which_paper_they_work_on():
    import threading

    release = threading.Event()
    manager = server.JobManager()
    job = manager.submit("tts", "gpu", lambda progress: release.wait(5), for_paper="p1")
    assert manager.busy("p1") and not manager.busy("p2")
    release.set()
    for _ in range(100):
        if manager.get(job)["status"] == "done":
            break
        threading.Event().wait(0.02)
    assert not manager.busy("p1")
