"""Routes the offline mode depends on: the root-scoped service worker, the manifest and the shell list."""

from fastapi.testclient import TestClient

from paper_reader.server import app

client = TestClient(app)


def test_service_worker_is_root_scoped_and_not_cached():
    r = client.get("/sw.js")
    assert r.status_code == 200 and r.headers["service-worker-allowed"] == "/"
    assert r.headers["cache-control"] == "no-cache" and "javascript" in r.headers["content-type"]


def test_manifest_and_shell():
    assert client.get("/manifest.webmanifest").headers["content-type"].startswith("application/manifest+json")
    urls = client.get("/api/offline/shell").json()["urls"]
    for needed in ("/", "/static/reader.html", "/static/reader.js", "/static/vendor/katex/katex.min.js",
                   "/static/vendor/fonts/source-serif-4.css", "/static/icons/icon-512.png"):
        assert needed in urls
    assert "/static/sw.js" not in urls and all(client.get(u).status_code == 200 for u in urls if u.startswith("/static/"))
