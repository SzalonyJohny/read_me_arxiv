// Index page: arXiv link → ingest job (with progress) → reader; PDF upload → analysis with a
// time estimate per extraction method → Continue → ingest job → reader; plus the library list
// (offline: only the papers saved on this device, and no adding).

import { api, connection, fmt, h, pollJob, showProgress } from "/static/common.js";
import { listSaved, registerServiceWorker } from "/static/offline.js";

const $ = (id) => document.getElementById(id);
const ui = {
  card: $("ingest"),
  form: $("ingest-form"),
  url: $("url"),
  submit: $("submit"),
  appendix: $("include-appendix"),
  llm: $("llm-runner"),
  llmNote: $("llm-note"),
  progress: $("ingest-progress"),
  error: $("ingest-error"),
  library: $("library"),
  libraryEmpty: $("library-empty"),
  offlineNote: $("offline-note"),
  // PDF upload
  drop: $("pdf-drop"),
  file: $("pdf-file"),
  pick: $("pdf-pick"),
  busyMsg: $("pdf-busy-msg"),
  pdfCard: $("pdf-card"),
  pdfTitle: $("pdf-title"),
  pdfMeta: $("pdf-meta"),
  chips: $("pdf-chips"),
  pdfForm: $("pdf-form"),
  methods: $("pdf-methods"),
  total: $("pdf-total"),
  actions: $("pdf-actions"),
  continue: $("pdf-continue"),
  reset: $("pdf-reset"),
};

const RULES_ONLY = { id: "none", label: "Off: rule-based filter only", available: true };
const MAX_PDF_BYTES = 100 * 2 ** 20; // the server answers 413 above this; checked here to skip a pointless upload
// "maths-heavy" when the text layer has more than this many maths glyphs/fonts per page on average.
// Same threshold as MATH_HEAVY_PER_PAGE in paper_reader/pdf/ingest.py, where it tips the recommendation to OCR.
const MATH_HEAVY_PER_PAGE = 40;
const NO_PAPERS = "No papers yet. Paste a link or upload a PDF above to start.";
const NO_SAVED_PAPERS = "No papers are saved on this device. While connected, open a paper and choose Save offline.";

const runnerNotes = new Map();
let analysis = null; // the PdfAnalysis of the uploaded PDF waiting for Continue
let busy = false;

// ---------------------------------------------------------------- config + library

// The clean-up runners (Claude Haiku via the API, Codex via its CLI) come from /api/config;
// unavailable ones stay listed but disabled, with the reason as the note.
async function loadConfig() {
  let config = { cleanup_runners: [], default_cleanup: null };
  try {
    config = await api("/api/config");
  } catch (err) {
    ui.llmNote.textContent = `Could not load server config (${err.message}).`;
    ui.llmNote.hidden = false;
  }
  ui.llm.replaceChildren();
  for (const runner of [...config.cleanup_runners, RULES_ONLY]) {
    const label = runner.available ? runner.label : `${runner.label}: unavailable`;
    ui.llm.append(h("option", { value: runner.id, disabled: !runner.available, text: label }));
    runnerNotes.set(runner.id, runner.available ? "" : runner.reason || "not configured");
  }
  ui.llm.value = config.default_cleanup || RULES_ONLY.id;
  showRunnerNote();
}

function showRunnerNote() {
  const unavailable = [...runnerNotes].filter(([, reason]) => reason).map(([id, reason]) => `${id}: ${reason}`);
  const chosen = ui.llm.value === RULES_ONLY.id ? "The rule-based filter runs either way. " : "";
  ui.llmNote.textContent = chosen + (unavailable.length ? `Unavailable: ${unavailable.join("; ")}.` : "");
  ui.llmNote.hidden = !ui.llmNote.textContent;
}

/** The library; offline, the service worker answers with the papers saved on this device. */
async function loadLibrary() {
  let papers = [];
  let problem = "";
  try {
    papers = await api("/api/papers");
  } catch (err) {
    problem = `Could not load the library: ${err.message}`;
  }
  const saved = new Set((await listSaved().catch(() => [])).map((entry) => entry.paper_id));
  ui.library.replaceChildren(...papers.map((paper) => libraryItem(paper, saved.has(paper.paper_id))));
  ui.libraryEmpty.textContent = problem || (connection.offline ? NO_SAVED_PAPERS : NO_PAPERS);
  ui.libraryEmpty.hidden = papers.length > 0;
}

function libraryItem(paper, saved) {
  const runs = paper.audio_runs?.length ?? 0;
  return h("li", {},
    h("a", { href: `/reader/${encodeURIComponent(paper.paper_id)}` },
      h("span", { class: "lib-text" },
        h("span", { class: "lib-title", text: paper.title || paper.paper_id }),
        h("span", { class: "lib-meta", text: sourceLabel(paper) })),
      h("span", { class: "lib-badges" },
        runs > 0 && h("span", { class: "badge", title: paper.audio_runs.join("\n"), text: runs > 1 ? `Audio · ${runs}` : "Audio" }),
        saved && h("span", { class: "badge saved", title: "Readable and playable without a connection", text: "Saved offline" }))));
}

/** "arXiv 2601.17428v1", or for uploads "PDF · paper.pdf" (+ " · OCR" when DeepSeek-OCR read it). */
function sourceLabel(paper) {
  if (paper.source?.startsWith("pdf")) {
    return `PDF · ${paper.filename || paper.paper_id}${paper.source === "pdf-ocr" ? " · OCR" : ""}`;
  }
  return `arXiv ${paper.arxiv_id || paper.paper_id}`;
}

// ---------------------------------------------------------------- offline

/** Adding papers needs the server: offline, the forms are disabled and a note says why. */
function renderConnection() {
  ui.offlineNote.hidden = !connection.offline;
  ui.card.inert = connection.offline;
  ui.card.classList.toggle("is-offline", connection.offline);
}

function onConnectionChange() {
  renderConnection();
  loadLibrary();
  if (!connection.offline) loadConfig();
}

/** Drag-and-drop works on the whole page, so it checks this as well as the (inert) form. */
function canUpload() {
  return !busy && !connection.offline;
}

// ---------------------------------------------------------------- shared state

/** Disables every control while an upload or an ingest job runs (unavailable methods stay disabled). */
function setBusy(value) {
  busy = value;
  for (const el of [ui.url, ui.submit, ui.appendix, ui.llm, ui.pick, ui.continue, ui.reset]) el.disabled = value;
  ui.methods.disabled = value;
  if (!value) updateTotal();
}

function showError(message) {
  ui.error.textContent = message;
  ui.error.hidden = !message;
}

/** Ingest job → progress bar → reader; on failure the error is shown and the controls come back. */
async function runIngest(submitting, request, describeError = (err) => err.message) {
  showError("");
  setBusy(true);
  showProgress(ui.progress, { progress: 0, message: submitting });
  ui.progress.scrollIntoView({ block: "nearest" });
  try {
    const { job_id: jobId, paper_id: submittedId } = await request();
    const job = await pollJob(jobId, (j) => showProgress(ui.progress, j));
    const paperId = job.result?.paper_id ?? submittedId;
    if (!paperId) throw new Error("The server finished without returning a paper id.");
    location.href = `/reader/${encodeURIComponent(paperId)}`;
  } catch (err) {
    ui.progress.hidden = true;
    showError(describeError(err));
    setBusy(false);
  }
}

// ---------------------------------------------------------------- arXiv link

function ingestUrl(event) {
  event.preventDefault();
  const url = ui.url.value.trim();
  if (!url) {
    showError("Paste an arXiv link first, e.g. https://arxiv.org/pdf/2601.17428");
    ui.url.focus();
    return;
  }
  runIngest("Submitting", () => api("/api/papers", {
    method: "POST",
    body: { url, include_appendix: ui.appendix.checked, llm: ui.llm.value },
  }));
}

// ---------------------------------------------------------------- PDF: upload

function isPdf(file) {
  return file.type === "application/pdf" || /\.pdf$/i.test(file.name);
}

async function upload(file) {
  if (!file || !canUpload()) return;
  if (!isPdf(file)) {
    showError(`“${file.name}” is not a PDF. Choose a .pdf file.`);
    return;
  }
  if (file.size > MAX_PDF_BYTES) {
    showError(`“${file.name}” is ${fmt.bytes(file.size)}; PDFs can be up to 100 MB.`);
    return;
  }
  showError("");
  const previous = analysis; // kept if this upload fails
  clearAnalysis();
  setBusy(true);
  ui.busyMsg.textContent = `Uploading ${file.name} (${fmt.bytes(file.size)})…`;
  ui.drop.classList.add("uploading");
  const form = new FormData();
  form.append("file", file, file.name);
  try {
    showAnalysis(await api("/api/pdf", { method: "POST", body: form }));
  } catch (err) {
    if (previous) showAnalysis(previous);
    showError(err.status === 413
      ? `“${file.name}” is too large for the server (PDFs can be up to 100 MB).`
      : err.status === 400 ? `Could not use “${file.name}”: ${err.message}`
      : `Upload failed: ${err.message}`);
  } finally {
    ui.drop.classList.remove("uploading");
    setBusy(false);
  }
  if (analysis) {
    revealAnalysis();
    ui.methods.querySelector("input:checked")?.focus({ preventScroll: true });
  } else {
    ui.pick.focus();
  }
}

/** Scrolls so the card and its Continue button are both visible, or (small screens) the card's top. */
function revealAnalysis() {
  const top = ui.pdfCard.getBoundingClientRect().top;
  const bottom = ui.actions.getBoundingClientRect().bottom;
  if (bottom - top + 48 <= innerHeight) ui.actions.scrollIntoView({ block: "nearest" });
  else ui.pdfCard.scrollIntoView({ block: "start" });
}

/** Back to the drop zone (the upload stays on the server; uploading the same file again is cheap). */
function clearAnalysis() {
  analysis = null;
  ui.pdfCard.hidden = ui.total.hidden = ui.actions.hidden = true;
  ui.drop.hidden = false;
}

// Dragging a file anywhere over the page highlights the drop target, and dropping it anywhere uploads it
// (otherwise the browser would navigate away to show the PDF).
let dragDepth = 0;

function carriesFiles(event) {
  return event.dataTransfer?.types?.includes("Files");
}

function setDragging(on) {
  (analysis ? ui.pdfCard : ui.drop).classList.toggle("dragging", on);
  if (!on) for (const el of [ui.drop, ui.pdfCard]) el.classList.remove("dragging");
}

window.addEventListener("dragenter", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dragDepth += 1;
  if (canUpload()) setDragging(true);
});
window.addEventListener("dragleave", (event) => {
  if (!carriesFiles(event)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) setDragging(false);
});
window.addEventListener("dragover", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  event.dataTransfer.dropEffect = canUpload() ? "copy" : "none";
});
window.addEventListener("drop", (event) => {
  if (!carriesFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  setDragging(false);
  upload(event.dataTransfer.files[0]);
});

// ---------------------------------------------------------------- PDF: analysis + method choice

function showAnalysis(result) {
  analysis = result;
  ui.pdfTitle.textContent = result.title_guess || result.filename;
  ui.pdfMeta.textContent = [
    result.title_guess && result.filename,
    `${result.pages} page${result.pages === 1 ? "" : "s"}`,
    fmt.bytes(result.size_bytes),
  ].filter(Boolean).join(" · ");
  ui.chips.replaceChildren(...analysisChips(result).map(([text, tone]) => h("li", { class: `badge ${tone}`, text })));

  const usable = result.estimates.filter((e) => e.available);
  const preselect = usable.find((e) => e.recommended) ?? usable[0];
  ui.methods.replaceChildren(ui.methods.querySelector("legend"),
    ...result.estimates.map((e) => methodRow(e, e === preselect)));

  ui.drop.hidden = true;
  ui.pdfCard.hidden = ui.total.hidden = ui.actions.hidden = false;
  updateTotal();
}

function analysisChips(a) {
  const chips = [];
  chips.push(a.text_pages > 0
    ? [`text layer on ${a.text_pages}/${a.pages} pages`, "neutral"]
    : ["no text layer", "warn"]);
  if (a.scanned_pages > 0) chips.push([`${a.scanned_pages} scanned page${a.scanned_pages === 1 ? "" : "s"}`, "warn"]);
  if (a.math_hint > MATH_HEAVY_PER_PAGE * Math.max(a.pages, 1)) chips.push(["maths-heavy", "neutral"]);
  if (a.arxiv_id) chips.push([`arXiv ${a.arxiv_id} detected`, ""]);
  return chips;
}

function methodRow(estimate, checked) {
  const noteId = `method-note-${estimate.method}`;
  const note = estimate.available ? estimate.note : `Unavailable: ${estimate.note || "not set up on this server"}`;
  return h("label", { class: `method${estimate.available ? "" : " unavailable"}` },
    h("input", { type: "radio", name: "method", value: estimate.method, checked, disabled: !estimate.available,
                 "aria-describedby": note ? noteId : null }),
    h("span", { class: "m-body" },
      h("span", { class: "m-top" },
        h("span", { class: "m-label", text: estimate.label }),
        estimate.recommended && estimate.available && h("span", { class: "badge", text: "Recommended" }),
        h("span", { class: "m-time", text: fmt.eta(estimate.seconds) })),
      note && h("span", { class: "m-note", id: noteId, text: note })));
}

function selectedEstimate() {
  const value = ui.methods.querySelector("input:checked")?.value;
  return analysis?.estimates.find((e) => e.method === value) ?? null;
}

/** "Estimated total ~34 s: extraction ~8 s + clean-up ~26 s", following the method and the LLM runner. */
function updateTotal() {
  if (!analysis) return;
  const chosen = selectedEstimate();
  ui.continue.disabled = busy || !chosen;
  if (!chosen) {
    ui.total.textContent = "None of the extraction methods can run on this PDF here.";
    return;
  }
  // cleanup_seconds is the server's estimate for its default LLM runner; it is used for any LLM runner.
  const cleanup = ui.llm.value === RULES_ONLY.id ? 0 : analysis.cleanup_seconds;
  ui.total.replaceChildren("Estimated total ",
    h("strong", { text: fmt.eta(chosen.seconds + cleanup) }),
    h("span", { class: "muted", text: `: extraction ${fmt.eta(chosen.seconds)} + clean-up ${
      cleanup ? fmt.eta(cleanup) : "0 s (LLM off)"}` }));
}

function ingestPdf(event) {
  event.preventDefault();
  const chosen = selectedEstimate();
  if (!analysis || !chosen || busy) return;
  runIngest("Starting", () => api(`/api/pdf/${encodeURIComponent(analysis.upload_id)}/ingest`, {
    method: "POST",
    body: { method: chosen.method, include_appendix: ui.appendix.checked, llm: ui.llm.value },
  }), (err) => (err.status === 404
    ? `The server no longer has this upload (${err.message}). Choose the file again.`
    : err.message));
}

// ---------------------------------------------------------------- wiring

ui.form.addEventListener("submit", ingestUrl);
ui.llm.addEventListener("change", () => {
  showRunnerNote();
  updateTotal();
});
ui.pick.addEventListener("click", () => ui.file.click());
ui.file.addEventListener("change", () => {
  const file = ui.file.files[0];
  ui.file.value = ""; // choosing the same file again still fires "change"
  upload(file);
});
ui.methods.addEventListener("change", updateTotal);
ui.pdfForm.addEventListener("submit", ingestPdf);
ui.reset.addEventListener("click", () => {
  showError("");
  clearAnalysis();
  ui.pick.focus();
});
registerServiceWorker();
renderConnection();
connection.addEventListener("change", onConnectionChange);
// The library first: offline it comes from the service worker, and the config is not needed.
loadLibrary().then(() => {
  if (!connection.offline) loadConfig();
});
