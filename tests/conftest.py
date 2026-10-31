"""Test data that is not in git.

The test paper (arXiv 2601.17428v1) is under arXiv's non-exclusive licence, so the repository
does not redistribute it: its PDF and HTML are downloaded from arXiv into tests/fixtures/ before
the first test run. Tests that need them skip when they can't be fetched (e.g. offline).

tests/fixtures/ocr_raw_2601.17428v1.json is the raw DeepSeek-OCR 2 output for that PDF. It needs
a GPU to make, so the tests that use it skip until it exists. Recreate it with:
    uv run python -c "from pathlib import Path; from paper_reader.pdf.ocr import run_worker; \\
        run_worker(Path('tests/fixtures/2601.17428v1.pdf'), Path('tests/fixtures/ocr_raw_2601.17428v1.json'))"
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
ARXIV_FIXTURES = {
    "2601.17428v1.pdf": "https://arxiv.org/pdf/2601.17428v1",
    "2601.17428v1.html": "https://arxiv.org/html/2601.17428v1",
}


def pytest_sessionstart(session) -> None:
    """Runs before collection, so the modules' skip markers see the downloaded files."""
    FIXTURES.mkdir(exist_ok=True)
    for name, url in ARXIV_FIXTURES.items():
        path = FIXTURES / name
        if path.exists():
            continue
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "paper-reader-tests"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            path.with_suffix(".part").write_bytes(data)
            path.with_suffix(".part").replace(path)
        except OSError as exc:
            print(f"\n[tests] could not download {url} ({exc}); the tests that need {name} will skip")
