"""arXiv link -> id/version, HTML fetch, figure download, and the ingest entry point."""

from __future__ import annotations

import mimetypes
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx

from .models import Paper
from .parse_html import parse_latexml

USER_AGENT = "paper-reader/0.1 (personal arXiv paper-to-audio reader; python-httpx)"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)
MAX_PARALLEL_DOWNLOADS = 4
MIN_SPOKEN_CHARS = 500  # less than this means the HTML version has no real content

ProgressCallback = Callable[[float, str], None]

# YYMM.NNNN until 2014, YYMM.NNNNN from 2015 on.
_NEW_ID = r"(?:0[7-9]|1[0-4])\d{2}\.\d{4}|(?:1[5-9]|[2-9]\d)\d{2}\.\d{5}"
_OLD_ID = r"[a-z][a-z\-]*(?:\.[A-Za-z]{2})?/\d{7}"
_ID_VERSION = rf"(?P<id>{_NEW_ID}|{_OLD_ID})(?P<version>v\d+)?"
_BARE_RE = re.compile(rf"(?:arxiv:\s*)?{_ID_VERSION}", re.I)
_PATH_RE = re.compile(rf"/(?:abs|pdf|html|format)/{_ID_VERSION}(?:\.pdf)?(?:/.*)?")
_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}


class ArxivHTMLUnavailable(RuntimeError):
    """arXiv has no usable HTML (LaTeXML) version of the paper."""


def parse_arxiv_url(url: str) -> tuple[str, str | None]:
    """Return (arxiv_id, version or None) for an arXiv link, "arXiv:<id>" or a bare id."""
    text = url.strip().strip("<>\"'")
    bare = _BARE_RE.fullmatch(text)
    if bare:
        return bare["id"], bare["version"]
    if text.lower().startswith(tuple(_HOSTS)):
        text = "https://" + text
    parts = urlsplit(text)
    match = _PATH_RE.fullmatch(parts.path) if (parts.hostname or "").lower() in _HOSTS else None
    if match is None:
        raise ValueError(
            f"Not an arXiv link or id: {url!r}. Expected e.g. https://arxiv.org/abs/2601.17428, "
            "https://arxiv.org/pdf/2601.17428v2, arXiv:2601.17428 or hep-th/9901001."
        )
    return match["id"], match["version"]


def fetch_html(arxiv_id: str, version: str | None) -> tuple[str, str, str]:
    """Fetch arXiv's HTML rendering. Returns (html, final_url, version).

    `final_url` is the URL after redirects, pinned to the served version when arXiv
    answered the unversioned URL (".../html/2601.17428" -> ".../html/2601.17428v1").
    `version` is "" if it cannot be determined (should not happen on arXiv pages).
    Raises ArxivHTMLUnavailable if arXiv has no HTML version of this paper.
    """
    url = f"https://arxiv.org/html/{arxiv_id}{version or ''}"
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True) as client:
        response = client.get(url)
    if response.status_code in (404, 410):
        raise ArxivHTMLUnavailable(_unavailable_message(arxiv_id, version, f"{url} returned {response.status_code}"))
    response.raise_for_status()
    html = response.text
    if "ltx_document" not in html:
        raise ArxivHTMLUnavailable(
            _unavailable_message(arxiv_id, version, "the page is not a LaTeXML document (conversion failed?)")
        )
    final_url = str(response.url)
    version = version or _detect_version(arxiv_id, final_url, html)
    if version and urlsplit(final_url).path.rstrip("/") == f"/html/{arxiv_id}":
        final_url = f"https://arxiv.org/html/{arxiv_id}{version}"  # pin the unversioned URL to what was served
    return html, final_url, version


def _unavailable_message(arxiv_id: str, version: str | None, reason: str) -> str:
    return (
        f"arXiv has no HTML version of {arxiv_id}{version or ''} ({reason}). "
        "Only arXiv's HTML (LaTeXML) rendering is supported; the PDF fallback is not in v1."
    )


def _detect_version(arxiv_id: str, final_url: str, html: str) -> str:
    """The version from the final URL, else from the page's "arXiv:<id>vN" watermark/links."""
    pattern = re.escape(arxiv_id) + r"(v\d+)"
    for source in (urlsplit(final_url).path, html):
        match = re.search(pattern, source)
        if match:
            return match.group(1)
    return ""


# --------------------------------------------------------------------------- figures


def _sanitize_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", unquote(name)).strip("._") or "image"
    stem, dot, ext = name.rpartition(".")
    return f"{stem[:80]}.{ext[:8]}" if dot and stem else name[:80]


def _local_names(urls: list[str]) -> dict[str, str]:
    """Deterministic, collision-free local file name for every distinct URL."""
    names: dict[str, str] = {}
    used: set[str] = set()
    for url in urls:
        if url in names:
            continue
        base = _sanitize_filename(urlsplit(url).path.rsplit("/", 1)[-1])
        stem, dot, ext = base.rpartition(".")
        stem, ext = (stem, f".{ext}") if dot else (base, "")
        candidate, n = base, 1
        while candidate.lower() in used:
            n += 1
            candidate = f"{stem}-{n}{ext}"
        used.add(candidate.lower())
        names[url] = candidate
    return names


def _looks_like_image(content: bytes, content_type: str) -> bool:
    if content_type.startswith("image/"):
        return True
    head = content[:512].lstrip()
    return head.startswith((b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"<svg", b"<?xml")) or head[8:12] == b"WEBP"


def _download(client: httpx.Client, url: str, dest: Path) -> Path:
    """Download one image to `dest` (adding an extension if it has none); returns the final path."""
    response = client.get(url)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if not _looks_like_image(response.content, content_type):
        raise ValueError(f"not an image (content-type {content_type or 'missing'})")
    if not dest.suffix:
        dest = dest.with_suffix(mimetypes.guess_extension(content_type) or ".img")
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return dest


def _existing(dest: Path) -> Path | None:
    """An already-downloaded file for `dest` (exact name, or with an added extension)."""
    candidates = [dest] if dest.suffix else sorted(dest.parent.glob(dest.name + ".*"))
    return next((p for p in candidates if p.is_file() and p.stat().st_size > 0 and p.suffix != ".part"), None)


def download_figures(paper: Paper, paper_dir: Path, *, progress_cb: ProgressCallback | None = None) -> None:
    """Download all Figure.images into paper_dir/figures/ and rewrite them to relative paths.

    Already-downloaded files are reused. Images that fail are dropped from Figure.images
    and reported in paper.warnings.
    """
    fig_dir = paper_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    remote = [url for fig in paper.figures.values() for url in fig.images if url.startswith(("http://", "https://"))]
    names = _local_names(remote)
    results: dict[str, str | Exception] = {}

    def fetch(client: httpx.Client, url: str) -> None:
        dest = fig_dir / names[url]
        try:
            path = _existing(dest) or _download(client, url, dest)
            results[url] = f"figures/{path.name}"
        except (httpx.HTTPError, OSError, ValueError) as exc:
            results[url] = exc

    todo = list(names)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True) as client:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_DOWNLOADS) as pool:
            for done, _ in enumerate(pool.map(lambda u: fetch(client, u), todo), 1):
                if progress_cb:
                    progress_cb(done / len(todo), f"Downloaded {done}/{len(todo)} figure images")

    for fig in paper.figures.values():
        kept = []
        for url in fig.images:
            result = results.get(url, url)  # non-remote entries are already local paths
            if isinstance(result, Exception):
                paper.warnings.append(f"Could not download {fig.label} image {url}: {result}")
            elif result not in kept:
                kept.append(result)
        fig.images = kept


# --------------------------------------------------------------------------- ingest


def ingest_arxiv(
    url: str,
    papers_root: Path,
    include_appendix: bool = False,
    progress_cb: ProgressCallback | None = None,
) -> Paper:
    """arXiv link -> parsed Paper with figures downloaded into papers_root/<paper_id>/figures.

    Does not write paper.json (the pipeline saves it).
    """

    def progress(fraction: float, message: str) -> None:
        if progress_cb:
            progress_cb(fraction, message)

    progress(0.0, "Reading the arXiv link")
    arxiv_id, version = parse_arxiv_url(url)
    progress(0.05, f"Fetching the arXiv HTML of {arxiv_id}{version or ''}")
    html, final_url, version = fetch_html(arxiv_id, version)
    progress(0.3, "Parsing the paper")
    paper = parse_latexml(
        html,
        page_url=final_url,
        arxiv_id=arxiv_id,
        version=version or None,
        source_url=url,
        include_appendix=include_appendix,
    )
    spoken = sum(len(b.text) for b in paper.iter_blocks() if b.speak)
    if spoken < MIN_SPOKEN_CHARS:
        raise ArxivHTMLUnavailable(
            _unavailable_message(arxiv_id, version, f"the HTML has only {spoken} characters of text; "
                                 "the source probably just embeds a PDF")
        )
    n_images = len({img for f in paper.figures.values() for img in f.images})
    progress(0.4, f"Downloading {n_images} figure images")
    download_figures(
        paper,
        papers_root / paper.paper_id,
        progress_cb=lambda f, m: progress(0.4 + 0.6 * f, m),
    )
    progress(1.0, f"Parsed {paper.title!r}")
    return paper
