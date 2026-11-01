// Service worker: offline reading.
//  - App shell (pages, scripts, styles, fonts, icons): cached at install, then network-first, so a
//    page never mixes files of two versions while the server is reachable; offline, the cached copies.
//  - Saved papers: web/offline.js stores a paper's JSON, figures and one narration, plus its reader page
//    and narration list, in the cache "paper-<id>". Those files are served cache-first, and the audio
//    honours Range requests so the <audio> element can seek.
//  - Pages and the library/narration APIs are network-first and fall back to the saved copies; any other
//    GET /api/* answers 503 {"detail": "offline"} without the server. Nothing else is cached (no POSTs,
//    no job polling).
// Every response made without the server carries OFFLINE_HEADER, so pages can switch to offline mode.

const SHELL_CACHE = "shell-v11"; // bump to re-cache the shell (old versions are deleted on activate)
const SHELL_PREFIX = "shell-";
const PAPER_PREFIX = "paper-"; // the cache names and index below are shared with web/offline.js
const INDEX_CACHE = "offline-index";
const INDEX_KEY = "/offline-index.json";
const OFFLINE_HEADER = "X-Offline";
const NETWORK_TIMEOUT_MS = 5000; // an unreachable server can hang a request far longer than this
const RETRY_TIMEOUT_MS = 1000; // once the server failed to answer, saved copies come sooner
const TIMED_OUT = Symbol("timed out");

let serverDown = false;

const PAGE_ROUTES = [
  [/^\/reader\/([^/]+)\/?$/, readerPage],
  [/^\/(?:voices)?$/, shellPage],
];
const ROUTES = [
  [/^\/static\/|^\/manifest\.webmanifest$/, shellFile],
  [/^\/papers\/([^/]+)\//, paperFile],
  [/^\/api\/papers\/([^/]+)\/audio$/, narrationList],
  [/^\/api\/papers$/, library],
  [/^\/api\//, (event) => fetch(event.request).catch(offlineResponse)],
];

self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(cacheShell());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(deleteOldShells().then(() => self.clients.claim()));
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== location.origin) return;
  const routes = request.mode === "navigate" ? PAGE_ROUTES : ROUTES;
  for (const [pattern, handle] of routes) {
    const match = pattern.exec(url.pathname);
    if (match) {
      event.respondWith(handle(event, url.pathname, match[1] && decodeURIComponent(match[1])));
      return;
    }
  }
});

// ---------------------------------------------------------------- shell

async function cacheShell() {
  const res = await fetch("/api/offline/shell", { cache: "no-store" });
  if (!res.ok) throw new Error(`Could not list the app shell: HTTP ${res.status}`);
  const { urls } = await res.json();
  const cache = await caches.open(SHELL_CACHE);
  await cache.addAll(urls.map((url) => new Request(url, { cache: "reload" })));
}

async function deleteOldShells() {
  const old = (await caches.keys()).filter((name) => name.startsWith(SHELL_PREFIX) && name !== SHELL_CACHE);
  await Promise.all(old.map((name) => caches.delete(name)));
}

/**
 * Scripts, styles, fonts: the server's copy (which also refreshes the cache), else the cached one.
 * Serving the cache first mixed versions after an update: a new reader.js with an old common.js
 * fails to import, and the page never starts.
 */
function shellFile(event) {
  return networkFirst(event, () => caches.match(event.request, { cacheName: SHELL_CACHE }),
    (response) => store(SHELL_CACHE, event.request, response));
}

/** "/" and "/voices": network-first, falling back to the shell's copy (kept fresh while online). */
function shellPage(event, path) {
  return networkFirst(event, () => caches.match(path, { cacheName: SHELL_CACHE }),
    (response) => store(SHELL_CACHE, path, response));
}

// ---------------------------------------------------------------- saved papers

function paperCache(paperId) {
  return PAPER_PREFIX + paperId;
}

/**
 * /reader/<id>: network-first; offline, the shell's reader.html. Every paper has the same page, and
 * the shell's copy matches the shell's scripts: the copy saved with a paper may be older than them
 * (an old page with a new reader.js broke on its missing elements), so it is only the last resort.
 */
function readerPage(event, path, paperId) {
  const offline = async () => (await caches.match("/static/reader.html", { cacheName: SHELL_CACHE }))
    ?? caches.match(path, { cacheName: paperCache(paperId) });
  return networkFirst(event, offline);
}

/**
 * /papers/<id>/…: cache-first from the paper's cache, else the network. Saving asks for fresh copies
 * with cache: "no-store", which skips the saved ones.
 */
async function paperFile(event, path, paperId) {
  const { request } = event;
  if (request.cache !== "no-store") {
    const cached = await caches.match(request, { cacheName: paperCache(paperId), ignoreVary: true });
    if (cached) return request.headers.has("Range") ? rangeResponse(cached, request.headers.get("Range")) : cached;
  }
  return fetch(request).catch(offlineResponse);
}

/** GET /api/papers/<id>/audio: network-first, falling back to the list saved with the paper. */
function narrationList(event, path, paperId) {
  return networkFirst(event, () => caches.match(path, { cacheName: paperCache(paperId) }));
}

/** GET /api/papers: network-first; offline, the saved papers in the API's shape, newest save first. */
function library(event) {
  return networkFirst(event, async () => {
    const saved = await caches.match(INDEX_KEY, { cacheName: INDEX_CACHE });
    const entries = Object.values(saved ? await saved.json() : {});
    entries.sort((a, b) => b.saved_at.localeCompare(a.saved_at));
    return jsonResponse(entries.map((p) => ({
      paper_id: p.paper_id,
      arxiv_id: p.arxiv_id,
      title: p.title,
      source: p.source,
      filename: p.filename,
      audio_runs: p.run_id ? [p.run_id] : [],
    })), 200);
  });
}

// ---------------------------------------------------------------- range requests

/**
 * A cached response sliced for a "Range: bytes=a-b | a- | -n" request (206), as the <audio> element
 * needs for seeking. Other range forms get the whole response (200); out-of-bounds ones get 416.
 */
async function rangeResponse(cached, header) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(header.trim());
  if (!match || (match[1] === "" && match[2] === "")) return cached;
  const blob = await cached.blob();
  const size = blob.size;
  const [start, end] = match[1] === ""
    ? [Math.max(0, size - Number(match[2])), size - 1] // the last n bytes
    : [Number(match[1]), match[2] === "" ? size - 1 : Math.min(Number(match[2]), size - 1)];
  if (start > end || start >= size) {
    return new Response(null, { status: 416, headers: { "Content-Range": `bytes */${size}` } });
  }
  return new Response(blob.slice(start, end + 1), {
    status: 206,
    headers: {
      "Content-Type": cached.headers.get("Content-Type") ?? "application/octet-stream",
      "Content-Range": `bytes ${start}-${end}/${size}`,
      "Content-Length": String(end - start + 1),
      "Accept-Ranges": "bytes",
    },
  });
}

// ---------------------------------------------------------------- network helpers

/**
 * The server's response; when the network fails or is too slow (NETWORK_TIMEOUT_MS, or RETRY_TIMEOUT_MS
 * while the server seems down), `fallback()` marked as offline. Without a fallback the network gets all
 * the time it needs. `onResponse` (optional) gets a copy of each response the server sends.
 */
async function networkFirst(event, fallback, onResponse) {
  const network = fetch(event.request);
  network.catch(() => {}); // a failure after the fallback was served is expected
  try {
    const timeout = delay(serverDown ? RETRY_TIMEOUT_MS : NETWORK_TIMEOUT_MS, TIMED_OUT);
    const response = await Promise.race([network, timeout]);
    if (response !== TIMED_OUT) {
      // A page can come from the browser's HTTP cache, which says nothing about the server.
      if (event.request.mode !== "navigate") serverDown = false;
      if (onResponse) event.waitUntil(onResponse(response.clone()));
      return response;
    }
  } catch {
    // unreachable server: use the fallback
  }
  serverDown = true;
  const saved = await fallback();
  return saved ? markOffline(saved) : network.catch(offlineResponse);
}

async function store(cacheName, key, response) {
  if (!response.ok) return;
  const cache = await caches.open(cacheName);
  await cache.put(key, response);
}

function delay(ms, value) {
  return new Promise((resolve) => setTimeout(resolve, ms, value));
}

function markOffline(response) {
  const headers = new Headers(response.headers);
  headers.set(OFFLINE_HEADER, "1");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function jsonResponse(data, status) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", [OFFLINE_HEADER]: "1" },
  });
}

function offlineResponse() {
  return jsonResponse({ detail: "offline" }, 503);
}
