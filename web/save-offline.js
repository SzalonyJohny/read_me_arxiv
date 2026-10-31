// Reader header: the "Save offline" button (size → progress with cancel → "Saved ✓" with a menu to
// update, remove and see the storage used), help where offline saving is unavailable, and the
// "Offline copy" badge while the server is out of reach.

import { connection, h } from "/static/common.js";
import {
  isOfflineSupported, listSaved, offlinePlan, removePaper, requestPersistence, savePaper, serviceWorkerRegistered,
  storageInfo,
} from "/static/offline.js";

const TOAST_MS = 6000;
const FLAG_URL = "chrome://flags/#unsafely-treat-insecure-origin-as-secure";

const $ = (id) => document.getElementById(id);
const ui = {
  root: $("offline"),
  button: $("offline-btn"),
  long: $("offline-long"),
  short: $("offline-short"),
  cancel: $("offline-cancel"),
  panel: $("offline-panel"),
  badge: $("offline-badge"),
  toast: $("offline-toast"),
};

const state = {
  status: "idle", // "unsupported" | "idle" | "saving" | "saved"
  problem: "", // why offline saving is unsupported, when the browser said
  entry: null, // the saved copy's index entry
  loaded: null, // promise: the saved entry has been looked up
  sizes: new Map(), // run id ("" for none) -> bytes a save downloads
  progress: { done: 0, total: 0 },
  abort: null,
};
let ctx; // { paperId, selectedRun: () => AudioManifest | null, runLabel: (runId) => string }
let toastTimer = 0;
let renderQueued = false;

export function initSaveOffline(options) {
  ctx = options;
  ui.button.addEventListener("click", onButtonClick);
  ui.cancel.addEventListener("click", () => state.abort?.abort());
  document.addEventListener("click", (e) => {
    if (!ui.panel.hidden && !ui.root.contains(e.target)) closePanel();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !ui.panel.hidden) closePanel();
  });
  connection.addEventListener("change", () => {
    render();
    fetchSize();
  });
  if (!isOfflineSupported()) state.status = "unsupported";
  state.loaded = loadSavedEntry();
  render();
}

/** The player's narration changed (or loaded): the size to save depends on it. */
export function narrationChanged() {
  fetchSize();
  render();
}

async function loadSavedEntry() {
  if (state.status === "unsupported") return;
  try {
    await serviceWorkerRegistered();
  } catch (err) {
    state.status = "unsupported";
    state.problem = err.message;
    render();
    return;
  }
  try {
    state.entry = (await listSaved()).find((e) => e.paper_id === ctx.paperId) ?? null;
  } catch (err) {
    console.warn("Could not read the offline copies:", err);
  }
  if (state.entry) state.status = "saved";
  render();
}

function selectedRunId() {
  return ctx.selectedRun()?.run_id ?? "";
}

async function fetchSize() {
  await state.loaded;
  const runId = selectedRunId();
  if (state.status !== "idle" || connection.offline || state.sizes.has(runId)) return;
  try {
    state.sizes.set(runId, (await offlinePlan(ctx.paperId, runId)).total_bytes);
  } catch (err) {
    if (!connection.offline) console.warn("Could not get the offline size:", err);
  }
  render();
}

// ---------------------------------------------------------------- button

function onButtonClick() {
  if (state.status === "idle") save();
  else if (ui.panel.hidden) openPanel();
  else closePanel();
}

function render() {
  const { status } = state;
  const [long, short] = labels();
  ui.root.dataset.status = status;
  ui.long.textContent = long;
  ui.short.textContent = short;
  const needsServer = status === "idle" && connection.offline;
  ui.button.disabled = status === "saving" || needsServer;
  ui.button.title = needsServer ? "Saving needs a connection to the server" : "";
  if (status === "idle" || status === "saving") ui.button.removeAttribute("aria-expanded");
  else ui.button.setAttribute("aria-expanded", String(!ui.panel.hidden));
  ui.cancel.hidden = status !== "saving";
  const { done, total } = state.progress;
  ui.root.style.setProperty("--progress", `${total ? (100 * done) / total : 0}%`);
  ui.badge.hidden = !connection.offline;
  ui.badge.textContent = state.entry ? "Offline copy" : "Offline";
  if (!ui.panel.hidden) renderPanel();
}

/** Progress arrives per downloaded chunk: one render per frame is enough. */
function renderSoon() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    render();
  });
}

/** [label, short label for narrow screens] */
function labels() {
  switch (state.status) {
    case "saving": {
      const { done, total } = state.progress;
      const pct = total ? Math.floor((100 * done) / total) : 0;
      return [`Saving ${pct}% · ${megabytes(done, false)}/${megabytes(total)}`, `${pct}%`];
    }
    case "saved":
      return ["Saved ✓", "Saved ✓"];
    case "idle": {
      const size = state.sizes.get(selectedRunId());
      if (size != null) return [`Save offline (${megabytes(size)})`, megabytes(size)];
      return ["Save offline", "Save"];
    }
    default:
      return ["Save offline", "Save"];
  }
}

/** 19338575 -> "19.3 MB" (decimal megabytes, as file managers show them). */
function megabytes(bytes, unit = true) {
  return `${(bytes / 1e6).toFixed(1)}${unit ? " MB" : ""}`;
}

function storageSize(bytes) {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : megabytes(bytes);
}

// ---------------------------------------------------------------- saving and removing

async function save() {
  closePanel();
  const runId = selectedRunId();
  state.abort = new AbortController();
  state.status = "saving";
  state.progress = { done: 0, total: state.sizes.get(runId) ?? 0 };
  render();
  try {
    state.entry = await savePaper(ctx.paperId, runId, {
      signal: state.abort.signal,
      onProgress: (progress) => {
        state.progress = progress;
        renderSoon();
      },
    });
    state.status = "saved";
    showToast(await requestPersistence()
      ? "Saved for offline reading. The browser keeps it until you remove it."
      : "Saved for offline reading. The browser may clear it if the device runs low on space.");
  } catch (err) {
    state.status = state.entry ? "saved" : "idle";
    showToast(state.abort.signal.aborted
      ? `Saving cancelled. ${state.entry ? "The earlier copy is unchanged." : "Nothing was kept."}`
      : `Could not save this paper: ${err.message}`, true);
  } finally {
    state.abort = null;
    render();
  }
}

async function remove() {
  closePanel();
  try {
    await removePaper(ctx.paperId);
    state.entry = null;
    state.status = "idle";
    showToast("The offline copy was removed.");
    fetchSize();
  } catch (err) {
    showToast(`Could not remove the offline copy: ${err.message}`, true);
  }
  render();
}

/** A short note at the bottom of the screen (an always-present live region, emptied to hide it). */
function showToast(text, isError = false) {
  ui.toast.textContent = text;
  ui.toast.classList.toggle("is-error", isError);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    ui.toast.textContent = "";
  }, TOAST_MS);
}

// ---------------------------------------------------------------- menu

function openPanel() {
  ui.panel.hidden = false;
  render();
}

function closePanel() {
  ui.panel.hidden = true;
  render();
}

function renderPanel() {
  ui.panel.replaceChildren(...(state.status === "unsupported" ? helpView() : savedView()).filter(Boolean));
}

function helpView() {
  if (state.problem) {
    return [
      h("p", { class: "ol-title", text: "Offline saving is not available here" }),
      h("p", { text: `The browser refused the offline service: ${state.problem}` }),
      h("p", { class: "muted small", text: "Over HTTPS the device must trust the certificate: with mkcert, install its "
        + "root CA on the phone, or use a certificate from Tailscale." }),
    ];
  }
  return [
    h("p", { class: "ol-title", text: "Offline saving is not available here" }),
    h("p", {}, "Offline saving needs HTTPS, or on Android Chrome the flag ",
      h("code", { text: FLAG_URL }), " with this site's address:"),
    h("code", { class: "ol-origin", text: location.origin }),
    h("p", { class: "muted small", text: "Open the flag, enter the address, set it to Enabled and relaunch Chrome. "
      + "Then reload this page." }),
  ];
}

function savedView() {
  const { entry } = state;
  const usage = h("dd", { text: "…" });
  showStorage(usage);
  const selected = selectedRunId();
  return [
    h("p", { class: "ol-title", text: "Saved for offline reading" }),
    h("dl", { class: "ol-facts" },
      h("dt", { text: "Narration" }), h("dd", { text: entry.run_id ? ctx.runLabel(entry.run_id) : "none" }),
      h("dt", { text: "Size" }), h("dd", { text: megabytes(entry.bytes) }),
      h("dt", { text: "Saved" }), h("dd", { text: new Date(entry.saved_at).toLocaleString(undefined,
        { dateStyle: "medium", timeStyle: "short" }) }),
      h("dt", { text: "Storage" }), usage),
    selected && selected !== entry.run_id
      && h("p", { class: "ol-note", text: "The player's narration is not the saved one: Update saves it instead." }),
    h("div", { class: "ol-actions" },
      h("button", { type: "button", class: "btn", text: "Update", disabled: connection.offline,
                    title: connection.offline ? "Updating needs a connection to the server" : "Download the paper again",
                    onclick: save }),
      h("button", { type: "button", class: "btn danger", text: "Remove offline copy", onclick: remove })),
  ];
}

async function showStorage(el) {
  try {
    const { usage, quota, persisted } = await storageInfo();
    el.textContent = `${storageSize(usage)} used of ${storageSize(quota)} available · ${persisted
      ? "kept until you remove it" : "may be cleared if the device runs low on space"}`;
  } catch {
    el.textContent = "unknown";
  }
}
