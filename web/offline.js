// Offline copies of papers. Saving puts a paper's JSON, figures and one narration, plus its reader page
// and narration list, in the cache "paper-<id>", which the service worker (web/sw.js) serves them from;
// a small index in the cache "offline-index" lists what is saved (also the offline library).

import { api } from "/static/common.js";

const PAPER_PREFIX = "paper-"; // cache names and index key shared with sw.js
const STAGING_SUFFIX = "~saving";
const INDEX_CACHE = "offline-index";
const INDEX_KEY = "/offline-index.json";
const DOWNLOADS_AT_ONCE = 3;

/** Service workers and Cache Storage exist only in secure contexts (HTTPS, localhost, or a flagged origin). */
export function isOfflineSupported() {
  return "serviceWorker" in navigator && "caches" in window;
}

let registering = null;

export function registerServiceWorker() {
  if (!isOfflineSupported()) return;
  registering = navigator.serviceWorker.register("/sw.js", { scope: "/" });
  registering.catch((err) => console.warn(`Offline reading is unavailable: ${err.message}`));
}

/**
 * Resolves once the service worker (which serves saved copies) is registered; rejects with the reason
 * otherwise, e.g. over HTTPS with a certificate the device does not trust.
 */
export function serviceWorkerRegistered() {
  return registering ?? Promise.reject(new Error("the service worker is not registered on this page"));
}

/** What saving would download: {paper_id, title, run_id, files: [{url, bytes}], total_bytes}. */
export function offlinePlan(paperId, runId) {
  const query = runId ? `?run=${encodeURIComponent(runId)}` : "";
  return api(`/api/papers/${encodeURIComponent(paperId)}/offline${query}`);
}

/**
 * Saves a paper with the narration `runId` (else the newest), reporting onProgress({done, total}) in
 * bytes. Downloads into a staging cache that replaces the paper's cache only once complete, so a failed,
 * cancelled (via `signal`) or over-quota save keeps any earlier copy and never looks complete.
 * Resolves with the index entry.
 */
export async function savePaper(paperId, runId, { onProgress = () => {}, signal } = {}) {
  const plan = await offlinePlan(paperId, runId);
  signal?.throwIfAborted();
  const staging = paperCacheName(paperId) + STAGING_SUFFIX;
  await caches.delete(staging); // left over from an interrupted save
  try {
    const cache = await caches.open(staging);
    const total = plan.total_bytes;
    let done = 0;
    await downloadAll(cache, plan.files.map((f) => encodePath(f.url)), signal, (bytes) => {
      done += bytes;
      onProgress({ done: Math.min(done, total), total });
    });
    await cache.put(readerPath(paperId), await fetchFresh(readerPath(paperId), signal));
    await cache.put(narrationsPath(paperId), await savedNarrations(paperId, plan.run_id, signal));
    const paper = await (await cache.match(`/papers/${encodeURIComponent(paperId)}/paper.json`)).json();
    await replaceCache(staging, paperCacheName(paperId));
    const entry = indexEntry(plan, paper, done);
    await updateIndex((index) => {
      index[paperId] = entry;
    });
    return entry;
  } catch (err) {
    throw err.name === "QuotaExceededError"
      ? new Error("There is not enough storage space on this device. Free some space and try again.")
      : err;
  } finally {
    await caches.delete(staging);
  }
}

/** Deletes a paper's offline copy (the index entry first, so it never looks saved half-deleted). */
export async function removePaper(paperId) {
  await updateIndex((index) => {
    delete index[paperId];
  });
  const name = paperCacheName(paperId);
  await Promise.all([caches.delete(name), caches.delete(name + STAGING_SUFFIX)]);
}

/** Index entries {paper_id, title, run_id, bytes, saved_at, source, filename, arxiv_id} of saved papers. */
export async function listSaved() {
  return isOfflineSupported() ? Object.values(await readIndex()) : [];
}

/** Asks the browser not to evict saved papers under storage pressure; true when granted. */
export async function requestPersistence() {
  try {
    return (await navigator.storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}

/** {usage, quota} in bytes for this site, and whether its storage is protected from eviction. */
export async function storageInfo() {
  const [estimate, persisted] = await Promise.all([
    navigator.storage?.estimate?.() ?? {},
    navigator.storage?.persisted?.() ?? false,
  ]);
  return { usage: estimate.usage ?? 0, quota: estimate.quota ?? 0, persisted };
}

// ---------------------------------------------------------------- downloading

function paperCacheName(paperId) {
  return PAPER_PREFIX + paperId;
}

function readerPath(paperId) {
  return `/reader/${encodeURIComponent(paperId)}`;
}

function narrationsPath(paperId) {
  return `/api/papers/${encodeURIComponent(paperId)}/audio`;
}

/** The plan lists plain paths; the reader requests them percent-encoded per segment, and the cache keys must match. */
function encodePath(path) {
  return path.split("/").map(encodeURIComponent).join("/");
}

/** Downloads `urls` into `cache`, a few at a time; the first failure stops the rest. */
async function downloadAll(cache, urls, signal, onBytes) {
  const stop = new AbortController();
  const both = signal ? AbortSignal.any([signal, stop.signal]) : stop.signal;
  const queue = [...urls];
  const worker = async () => {
    for (let url = queue.shift(); url; url = queue.shift()) await download(cache, url, both, onBytes);
  };
  try {
    await Promise.all(Array.from({ length: Math.min(DOWNLOADS_AT_ONCE, queue.length) }, worker));
  } catch (err) {
    stop.abort();
    throw err;
  }
}

/** Streams one file into the cache, counting its bytes as they arrive. */
async function download(cache, url, signal, onBytes) {
  const response = await fetchFresh(url, signal);
  const counted = response.body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      onBytes(chunk.byteLength);
      controller.enqueue(chunk);
    },
  }));
  await cache.put(url, new Response(counted, { status: response.status, headers: response.headers }));
}

/** fetch from the server itself: cache "no-store" also tells the service worker to skip saved copies. */
async function fetchFresh(url, signal) {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) throw new Error(`Could not download ${decodeURI(url)} (HTTP ${response.status}).`);
  return response;
}

/** The narration list as saved: only the saved run, so offline the player never picks one that isn't stored. */
async function savedNarrations(paperId, runId, signal) {
  const runs = await (await fetchFresh(narrationsPath(paperId), signal)).json();
  return jsonResponse(runs.filter((run) => run.run_id === runId));
}

/** Copies every entry of cache `fromName` into `toName`, then drops what only `toName` had (e.g. an older narration). */
async function replaceCache(fromName, toName) {
  const [from, to] = await Promise.all([caches.open(fromName), caches.open(toName)]);
  const requests = await from.keys();
  for (const request of requests) await to.put(request, await from.match(request));
  const kept = new Set(requests.map((r) => r.url));
  for (const request of await to.keys()) {
    if (!kept.has(request.url)) await to.delete(request);
  }
}

/** Mirrors an item of GET /api/papers, so the service worker can list saved papers offline. */
function indexEntry(plan, paper, bytes) {
  const fromPdf = paper.source?.startsWith("pdf");
  return {
    paper_id: plan.paper_id,
    title: plan.title,
    run_id: plan.run_id,
    bytes,
    saved_at: new Date().toISOString(),
    source: paper.source ?? "arxiv-html",
    filename: fromPdf ? paper.source_url?.replace(/^upload:/, "") ?? null : null,
    arxiv_id: paper.arxiv_id ?? null,
  };
}

// ---------------------------------------------------------------- index

async function readIndex() {
  const saved = await caches.match(INDEX_KEY, { cacheName: INDEX_CACHE });
  return saved ? saved.json() : {};
}

async function updateIndex(change) {
  const index = await readIndex();
  change(index);
  const cache = await caches.open(INDEX_CACHE);
  await cache.put(INDEX_KEY, jsonResponse(index));
}

function jsonResponse(data) {
  return new Response(JSON.stringify(data), { headers: { "Content-Type": "application/json" } });
}
