// Voice lab: every Kokoro voice and the designed Qwen3-TTS voices read the first two sentences of a
// paper's abstract; select the voice narration uses by default, or design new Qwen voices.

import { api, h, pollJob, showProgress, store } from "/static/common.js";
import { registerServiceWorker } from "/static/offline.js";

const ENGINE_LABELS = { kokoro: "Kokoro", qwen3: "Qwen3-TTS", elevenlabs: "ElevenLabs" };
const KOKORO_GROUPS = [
  ["en-us", "female", "American female"],
  ["en-us", "male", "American male"],
  ["en-gb", "female", "British female"],
  ["en-gb", "male", "British male"],
];
const QWEN_MODEL_KIND = { "0.6B": "clone", "1.7B": "clone", "1.7B-design": "description per sentence" };
const QWEN_MODEL_ORDER = Object.keys(QWEN_MODEL_KIND); // clones first, then per-sentence design
const DEFAULT_DESCRIPTION = "Clear and precise and technical, male middle age man";
const FRESH_MS = 4000;

const $ = (id) => document.getElementById(id);
const ui = {
  paper: $("paper"),
  sampleText: $("sample-text"),
  selectedValue: $("selected-value"),
  narrate: $("narrate"),
  pageError: $("page-error"),
  kokoro: $("kokoro"),
  kkGood: $("kk-good"),
  kkSeq: $("kk-seq"),
  kkRenderAll: $("kk-render-all"),
  kkGroups: $("kk-groups"),
  qwen: $("qwen"),
  qSeq: $("q-seq"),
  designForm: $("design-form"),
  qDesc: $("q-desc"),
  qModels: $("q-models"),
  qDesign: $("q-design"),
  qList: $("q-list"),
};
const sections = {
  kokoro: { jobs: $("kk-jobs"), errors: $("kk-errors") },
  qwen: { jobs: $("q-jobs"), errors: $("q-errors") },
};

const state = {
  papers: [],
  paperId: null,
  lab: null, // GET /api/voicelab/{paper_id}
  loadSeq: 0,
  busy: new Set(), // "<paper>|<job key>" of submitted jobs, until they finish
  pendingDesigns: new Set(), // "<paper>|<description>" being designed
  errors: [], // {id, section, paperId, text}
  errorSeq: 0,
  jobSeq: 0,
  selecting: false,
  fresh: null, // voice id of a just-designed Qwen voice (highlighted briefly)
};

/** One <audio> per clip url (urls carry ?v=), so re-rendering keeps players and their playback. */
const players = new Map();

// ---------------------------------------------------------------- start

registerServiceWorker();
init();

async function init() {
  ui.kkGood.checked = store.get("lab:kokoroGoodOnly", false);
  ui.kkSeq.checked = store.get("lab:kokoroSequence", false);
  ui.qSeq.checked = store.get("lab:qwenSequence", false);

  ui.paper.addEventListener("change", () => switchPaper(ui.paper.value));
  ui.kkGood.addEventListener("change", () => {
    store.set("lab:kokoroGoodOnly", ui.kkGood.checked);
    renderKokoro();
  });
  ui.kkSeq.addEventListener("change", () => store.set("lab:kokoroSequence", ui.kkSeq.checked));
  ui.qSeq.addEventListener("change", () => store.set("lab:qwenSequence", ui.qSeq.checked));
  ui.kkRenderAll.addEventListener("click", renderAllKokoro);
  ui.designForm.addEventListener("submit", designVoice);
  ui.qDesc.addEventListener("input", renderDesignButton);
  ui.qModels.addEventListener("change", () => {
    store.set("lab:qwenModels", checkedModels());
    renderDesignButton();
  });
  document.addEventListener("play", pauseOthers, true);
  for (const [section, toggle] of [[ui.kokoro, ui.kkSeq], [ui.qwen, ui.qSeq]]) {
    section.addEventListener("ended", (e) => toggle.checked && playNext(section, e.target), true);
  }

  try {
    state.papers = await api("/api/papers");
  } catch (err) {
    showPageError(`Could not load the library: ${err.message}`);
  }
  const fromUrl = new URLSearchParams(location.search).get("paper");
  const paperId = fromUrl || state.papers[0]?.paper_id || null;
  renderPaperSelect(paperId);
  if (!paperId) {
    if (!ui.pageError.textContent) {
      showPageError(h("span", {}, "No papers yet. ", h("a", { href: "/", text: "Add one in the library" }), " first."));
    }
    state.lab = null;
    render();
    return;
  }
  if (!fromUrl) setUrlPaper(paperId);
  state.paperId = paperId;
  await loadLab(paperId);
}

function renderPaperSelect(current) {
  const papers = [...state.papers];
  if (current && !papers.some((p) => p.paper_id === current)) papers.unshift({ paper_id: current, title: current });
  ui.paper.replaceChildren(...papers.map((p) =>
    h("option", { value: p.paper_id, text: p.title ? `${p.title} · ${p.arxiv_id || p.paper_id}` : p.paper_id })));
  ui.paper.disabled = papers.length === 0;
  if (current) ui.paper.value = current;
}

function setUrlPaper(paperId) {
  const url = new URL(location.href);
  url.searchParams.set("paper", paperId);
  history.replaceState(null, "", url);
}

function switchPaper(paperId) {
  pauseAll();
  setUrlPaper(paperId);
  state.paperId = paperId;
  state.lab = null;
  showPageError("");
  render();
  loadLab(paperId);
}

/** GET the lab state for a paper and render it. `quiet` keeps the current view while reloading. */
async function loadLab(paperId, { quiet = false } = {}) {
  const seq = ++state.loadSeq;
  try {
    const lab = await api(`/api/voicelab/${encodeURIComponent(paperId)}`);
    if (seq !== state.loadSeq || paperId !== state.paperId) return;
    state.lab = lab;
    showPageError("");
    prefillDescription();
    render();
  } catch (err) {
    if (seq !== state.loadSeq || paperId !== state.paperId) return;
    showPageError(`Could not load the voice lab for ${paperId}: ${err.message}`);
    if (!quiet) {
      state.lab = null;
      render();
    }
  }
}

function prefillDescription() {
  if (ui.qDesc.value.trim() || ui.qDesc.dataset.prefilled) return;
  const preset = state.lab?.qwen3?.voices?.find((v) => v.preset);
  ui.qDesc.value = preset?.description || DEFAULT_DESCRIPTION;
  ui.qDesc.dataset.prefilled = "1";
}

// ---------------------------------------------------------------- render

function render() {
  renderHeader();
  renderKokoro();
  renderQwen();
  renderJobsAndErrors();
  for (const [url, audio] of players) {
    if (!audio.isConnected) players.delete(url);
  }
}

function renderHeader() {
  const lab = state.lab;
  if (lab?.title) document.title = `Voice lab · ${lab.title}`;
  ui.sampleText.replaceChildren(lab
    ? h("p", { text: lab.text || "This paper has no abstract text to read." })
    : h("p", { class: "muted", text: state.paperId ? "Loading…" : "—" }));
  renderSelected();
  ui.narrate.hidden = !state.paperId;
  if (state.paperId) ui.narrate.href = `/reader/${encodeURIComponent(state.paperId)}`;
}

function renderSelected() {
  const sel = state.lab?.selected;
  if (!state.lab) {
    ui.selectedValue.replaceChildren(h("span", { class: "muted", text: state.paperId ? "Loading…" : "—" }));
    return;
  }
  if (!sel) {
    ui.selectedValue.replaceChildren(h("span", { class: "muted", text: "None yet: narration uses the engine's default voice. Pick one below." }));
    return;
  }
  const parts = [h("strong", { text: ENGINE_LABELS[sel.engine] ?? sel.engine })];
  if (sel.model) parts.push(" · ", h("span", { text: sel.model }));
  parts.push(" · ", h("code", { text: sel.voice }));
  const detail = selectionDetail(sel);
  if (detail) parts.push(" ", h("span", { class: "sel-desc", text: detail }));
  ui.selectedValue.replaceChildren(...parts);
}

/** The Kokoro label, or the Qwen description (+ take), of the selected voice. */
function selectionDetail(sel) {
  if (sel.engine === "kokoro") {
    return kokoroVoices().find((v) => v.id === sel.voice)?.label ?? "";
  }
  if (sel.engine === "qwen3") {
    const voice = qwenVoices().find((v) => v.id === sel.voice);
    return voice ? `“${voice.description}”${voice.take > 1 ? ` · take ${voice.take}` : ""}` : "";
  }
  return "";
}

function kokoroVoices() {
  return state.lab?.kokoro?.voices ?? [];
}

function qwenVoices() {
  return state.lab?.qwen3?.voices ?? [];
}

/** The server's Qwen models, in QWEN_MODEL_ORDER (unknown ones last, as sent). */
function qwenModels() {
  const rank = (m) => (QWEN_MODEL_ORDER.includes(m.id) ? QWEN_MODEL_ORDER.indexOf(m.id) : QWEN_MODEL_ORDER.length);
  return [...(state.lab?.qwen3?.models ?? [])].sort((a, b) => rank(a) - rank(b));
}

function isBusy(key) {
  return state.busy.has(`${state.paperId}|${key}`);
}

// ---------------------------------------------------------------- Kokoro

function renderKokoro() {
  const kokoro = state.lab?.kokoro;
  const voices = kokoro?.voices ?? [];
  const samples = kokoro?.samples ?? {};
  const missing = voices.filter((v) => !samples[v.id]).length;
  const allBusy = isBusy("kokoro|all");

  ui.kkRenderAll.disabled = !voices.length || allBusy;
  ui.kkRenderAll.classList.toggle("primary", missing > 0);
  ui.kkRenderAll.textContent = allBusy ? "Rendering voices…"
    : !voices.length ? "Render all voices"
    : missing === 0 ? `Re-render all ${voices.length} voices`
    : missing === voices.length ? `Render all ${voices.length} voices`
    : `Render the ${missing} missing voices`;

  if (!state.lab) {
    ui.kkGroups.replaceChildren(h("p", { class: "muted", text: state.paperId ? "Loading…" : "" }));
    return;
  }
  if (!voices.length) {
    ui.kkGroups.replaceChildren(h("p", { class: "muted", text: "Kokoro is not available on this server." }));
    return;
  }

  const goodOnly = ui.kkGood.checked;
  const shown = voices.filter((v) => !goodOnly || gradeScore(v.grade) >= gradeScore("C"));
  const groups = KOKORO_GROUPS.map(([language, gender, title]) => [title, shown.filter((v) =>
    (v.language || "").toLowerCase() === language && v.gender === gender)]);
  const grouped = new Set(groups.flatMap(([, list]) => list));
  groups.push(["Other voices", shown.filter((v) => !grouped.has(v))]);

  const blocks = groups.filter(([, list]) => list.length).map(([title, list]) =>
    h("div", { class: "kk-group" },
      h("h3", {}, title, h("span", { class: "count", text: String(list.length) })),
      h("ul", { class: "voice-grid" }, sortByGrade(list).map((v) => kokoroCard(v, samples[v.id])))));
  const hidden = voices.length - shown.length;
  if (hidden) blocks.push(h("p", { class: "muted small", text: `${hidden} voices below grade C are hidden.` }));
  ui.kkGroups.replaceChildren(...blocks);
}

function kokoroCard(voice, sample) {
  const sel = state.lab.selected;
  const selected = sel?.engine === "kokoro" && sel.voice === voice.id;
  const busy = isBusy("kokoro|all") || isBusy(`kokoro|${voice.id}`);
  return h("li", { class: `vcard${selected ? " selected" : ""}`, dataset: { voice: voice.id } },
    h("div", { class: "vcard-head" },
      gradeBadge(voice.grade),
      h("div", { class: "vcard-name" },
        h("strong", { text: voice.label || voice.id }),
        h("code", { text: voice.id }))),
    sample
      ? player(sample.url)
      : notRendered(busy, () => renderKokoroVoice(voice.id), `Render ${voice.id}`),
    h("div", { class: "vcard-foot" },
      h("span", { class: "muted small", text: sample ? clipStats(sample) : "" }),
      selectButton(selected, () => select({ engine: "kokoro", voice: voice.id, model: null }), voice.id)));
}

/** All voices, or only the missing ones when some are rendered already. */
function renderAllKokoro() {
  const samples = state.lab?.kokoro?.samples ?? {};
  const missing = kokoroVoices().filter((v) => !samples[v.id]).map((v) => v.id);
  const some = missing.length > 0 && missing.length < kokoroVoices().length;
  runJob({
    section: "kokoro",
    key: "kokoro|all",
    label: some ? `Kokoro · ${missing.length} missing voices` : `Kokoro · all ${kokoroVoices().length} voices`,
    start: (paperId) => kokoroRender(paperId, some ? missing : ["all"]),
  });
}

function renderKokoroVoice(voiceId) {
  runJob({
    section: "kokoro",
    key: `kokoro|${voiceId}`,
    label: `Kokoro · ${voiceId}`,
    start: (paperId) => kokoroRender(paperId, [voiceId]),
  });
}

function kokoroRender(paperId, voices) {
  return api(`/api/voicelab/${encodeURIComponent(paperId)}/render`, { method: "POST", body: { engine: "kokoro", voices } });
}

/** Kokoro grades run A … F with +/-; higher is better. Unknown grades sort last. */
function gradeScore(grade) {
  const m = /^([A-F])([+-]?)$/.exec(grade || "");
  if (!m) return -1;
  return (5 - "ABCDEF".indexOf(m[1])) * 3 + (m[2] === "+" ? 1 : m[2] === "-" ? -1 : 0);
}

function sortByGrade(voices) {
  return [...voices].sort((a, b) => gradeScore(b.grade) - gradeScore(a.grade) || (a.label || a.id).localeCompare(b.label || b.id));
}

function gradeBadge(grade) {
  const letter = (grade || "?")[0].toLowerCase();
  return h("span", { class: `grade grade-${letter}`, title: grade ? `Kokoro grade ${grade}` : "No grade", text: grade || "–" });
}

// ---------------------------------------------------------------- Qwen3-TTS

function renderQwen() {
  const qwen = state.lab?.qwen3;
  renderModelPicks(qwenModels());
  renderDesignButton();
  if (!state.lab) {
    ui.qList.replaceChildren(h("li", { class: "muted", text: state.paperId ? "Loading…" : "" }));
    return;
  }
  if (!qwen) {
    ui.qList.replaceChildren(h("li", { class: "muted", text: "Qwen3-TTS is not available on this server." }));
    return;
  }
  const voices = newestFirst(qwen.voices ?? []);
  ui.qList.replaceChildren(...(voices.length
    ? voices.map((v) => qwenCard(v, qwen))
    : [h("li", { class: "muted", text: "No designed voices yet. Describe one above." })]));
}

/** Newest first: by `created` when the server sends it (presets have none), else reversed server order. */
function newestFirst(voices) {
  return voices.map((voice, index) => ({ voice, index }))
    .sort((a, b) => (b.voice.created ?? 0) - (a.voice.created ?? 0) || b.index - a.index)
    .map(({ voice }) => voice);
}

function qwenCard(voice, qwen) {
  const sel = state.lab.selected;
  const samples = qwen.samples?.[voice.id] ?? {};
  const models = qwenModels();
  const missing = models.filter((m) => !samples[m.id]).map((m) => m.id);
  const redesigning = isBusy(`qwen|redesign|${voice.id}`);
  const rendering = isBusy(`qwen|render|${voice.id}`);
  const busy = redesigning || rendering || models.some((m) => isBusy(`qwen|render|${voice.id}|${m.id}`));
  const classes = ["qcard", sel?.engine === "qwen3" && sel.voice === voice.id && "selected", state.fresh === voice.id && "fresh"];

  return h("li", { class: classes.filter(Boolean).join(" "), dataset: { voice: voice.id } },
    h("div", { class: "qcard-head" },
      h("p", { class: "qdesc", text: voice.description }),
      h("div", { class: "qmeta" },
        h("span", { class: "badge", text: `Take ${voice.take ?? 1}` }),
        voice.preset && h("span", { class: "badge preset", text: "Preset" }),
        h("code", { text: voice.id }))),
    h("div", { class: "clip-rows" },
      h("div", { class: "clip-row reference" },
        h("div", { class: "clip-label" }, h("strong", { text: "Reference" }), h("span", { class: "muted small", text: "designed clip" })),
        voice.reference_url
          ? player(voice.reference_url)
          : h("p", { class: "not-rendered muted small", text: "Not designed yet. Rendering any model designs it first." }),
        h("div", { class: "clip-side" })),
      models.map((m) => modelRow(voice, m, samples[m.id], rendering || redesigning))),
    h("div", { class: "qcard-actions" },
      missing.length > 1 && h("button", {
        type: "button", class: "btn", disabled: rendering,
        text: rendering ? "Rendering…" : `Render ${missing.length} missing`,
        onclick: () => renderQwenClips(voice, missing, true),
      }),
      h("button", {
        type: "button", class: "btn", disabled: redesigning,
        text: redesigning ? "Designing new take…" : "Redesign (new take)",
        title: "Design this description again: a different voice, as a new take",
        onclick: () => redesign(voice),
      }),
      h("button", { type: "button", class: "btn quiet", text: "Edit description", onclick: () => editDescription(voice) }),
      !voice.preset && h("button", {
        type: "button", class: "btn quiet danger", disabled: busy, text: "Delete",
        title: busy ? "Wait for this voice's jobs to finish" : "Delete this voice and its clips",
        onclick: () => deleteVoice(voice),
      })));
}

function modelRow(voice, model, sample, cardBusy) {
  const sel = state.lab.selected;
  const selected = sel?.engine === "qwen3" && sel.voice === voice.id && sel.model === model.id;
  const busy = cardBusy || isBusy(`qwen|render|${voice.id}|${model.id}`);
  return h("div", { class: `clip-row${selected ? " selected" : ""}`, dataset: { model: model.id } },
    h("div", { class: "clip-label", title: model.label || model.id },
      h("strong", { text: model.id }),
      h("span", { class: "muted small", text: QWEN_MODEL_KIND[model.id] ?? model.label ?? "" })),
    sample
      ? player(sample.url)
      : notRendered(busy, () => renderQwenClips(voice, [model.id]), `Render ${model.id} for this voice`),
    h("div", { class: "clip-side" },
      sample && h("span", { class: "muted small", text: clipStats(sample) }),
      selectButton(selected, () => select({ engine: "qwen3", voice: voice.id, model: model.id }), `${voice.id} with ${model.id}`)));
}

function renderQwenClips(voice, models, wholeCard = false) {
  runJob({
    section: "qwen",
    key: wholeCard ? `qwen|render|${voice.id}` : `qwen|render|${voice.id}|${models[0]}`,
    label: `Qwen3 · ${models.join(", ")} · “${ellipsize(voice.description, 48)}”`,
    start: (paperId) => api(`/api/voicelab/${encodeURIComponent(paperId)}/render`, {
      method: "POST", body: { engine: "qwen3", voices: [voice.id], models },
    }),
  });
}

function renderModelPicks(models) {
  const ids = models.map((m) => m.id).join("\n");
  if (ui.qModels.dataset.ids === ids) return;
  ui.qModels.dataset.ids = ids;
  const saved = store.get("lab:qwenModels");
  const legend = ui.qModels.querySelector("legend");
  ui.qModels.replaceChildren(legend, ...models.map((m) =>
    h("label", { class: "check model-pick", title: m.label || m.id },
      h("input", { type: "checkbox", value: m.id, checked: !Array.isArray(saved) || saved.includes(m.id) }),
      h("span", { text: m.id }),
      QWEN_MODEL_KIND[m.id] && h("span", { class: "muted small", text: QWEN_MODEL_KIND[m.id] }))));
}

function checkedModels() {
  return [...ui.qModels.querySelectorAll("input:checked")].map((input) => input.value);
}

function normalized(description) {
  return description.trim().replace(/\s+/g, " ");
}

function renderDesignButton() {
  const description = normalized(ui.qDesc.value);
  const pending = state.pendingDesigns.has(`${state.paperId}|${description}`);
  ui.qDesign.disabled = !state.lab?.qwen3 || !description || !checkedModels().length || pending;
  ui.qDesign.textContent = pending ? "Designing…" : "Design voice";
  ui.qDesign.title = !description ? "Write a description first"
    : !checkedModels().length ? "Pick at least one model to render with"
    : pending ? "This description is already being designed" : "";
}

function designVoice(event) {
  event.preventDefault();
  const description = normalized(ui.qDesc.value);
  const models = checkedModels();
  if (!description || !models.length) return;
  const pendingKey = `${state.paperId}|${description}`;
  state.pendingDesigns.add(pendingKey);
  runJob({
    section: "qwen",
    key: `qwen|design|${++state.jobSeq}`,
    label: `Designing “${ellipsize(description, 60)}”`,
    start: (paperId) => api(`/api/voicelab/${encodeURIComponent(paperId)}/design`, {
      method: "POST", body: { description, models },
    }),
    onDone: (job) => flashVoice(job.result?.voice_id),
    onSettled: () => state.pendingDesigns.delete(pendingKey),
  });
}

function redesign(voice) {
  const models = checkedModels();
  runJob({
    section: "qwen",
    key: `qwen|redesign|${voice.id}`,
    label: `New take of “${ellipsize(voice.description, 56)}”`,
    start: (paperId) => api(`/api/voicelab/${encodeURIComponent(paperId)}/design`, {
      method: "POST",
      body: { description: voice.description, models: models.length ? models : undefined, redesign_of: voice.id },
    }),
    onDone: (job) => flashVoice(job.result?.voice_id),
  });
}

function editDescription(voice) {
  ui.qDesc.value = voice.description;
  renderDesignButton();
  ui.designForm.scrollIntoView({ behavior: "smooth", block: "center" });
  ui.qDesc.focus({ preventScroll: true });
}

async function deleteVoice(voice) {
  const take = voice.take > 1 ? ` (take ${voice.take})` : "";
  if (!confirm(`Delete this voice and its clips?\n\n“${voice.description}”${take}`)) return;
  try {
    await api(`/api/voices/qwen/${encodeURIComponent(voice.id)}`, { method: "DELETE" });
    await loadLab(state.paperId, { quiet: true });
  } catch (err) {
    addError("qwen", `Could not delete ${voice.id}: ${err.message}`);
  }
}

/** Highlights a just-designed card and brings it into view. */
function flashVoice(voiceId) {
  if (!voiceId) return;
  state.fresh = voiceId;
  renderQwen();
  const card = [...ui.qList.children].find((li) => li.dataset.voice === voiceId);
  card?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  setTimeout(() => {
    if (state.fresh !== voiceId) return;
    state.fresh = null;
    card?.classList.remove("fresh");
  }, FRESH_MS);
}

// ---------------------------------------------------------------- selection

function selectButton(selected, onSelect, what) {
  return h("button", {
    type: "button",
    class: selected ? "btn primary sel-btn" : "btn sel-btn",
    "aria-pressed": String(selected),
    "aria-label": selected ? `Selected: ${what}` : `Select ${what}`,
    disabled: state.selecting && !selected,
    text: selected ? "Selected" : "Select",
    onclick: selected ? null : onSelect,
  });
}

async function select(selection) {
  if (state.selecting) return;
  state.selecting = true;
  const section = selection.engine === "kokoro" ? "kokoro" : "qwen";
  try {
    const saved = await api("/api/voices/selected", { method: "POST", body: selection });
    if (state.lab) state.lab.selected = saved ?? selection;
  } catch (err) {
    addError(section, `Could not select ${selection.voice}: ${err.message}`);
  } finally {
    state.selecting = false;
    renderHeader();
    renderKokoro();
    renderQwen();
  }
}

// ---------------------------------------------------------------- jobs and errors

/**
 * Submits a job (start(paperId) → {job_id}) and shows its progress in the section until it ends,
 * then reloads the lab. `key` marks what the job touches so conflicting buttons stay disabled.
 */
async function runJob({ section, key, label, start, onDone, onSettled }) {
  const paperId = state.paperId;
  const busyKey = `${paperId}|${key}`;
  if (!paperId || state.busy.has(busyKey)) return;
  state.busy.add(busyKey);

  const progress = h("div", { class: "progress" }, h("progress", { max: 1, value: 0 }), h("p", { class: "msg" }));
  const row = h("div", { class: "job", dataset: { paper: paperId } }, h("span", { class: "job-label", text: label }), progress);
  sections[section].jobs.append(row);
  showProgress(progress, { progress: 0, message: "Submitting" });
  render();

  let job = null;
  try {
    const { job_id: jobId } = await start(paperId);
    job = await pollJob(jobId, (j) => showProgress(progress, { ...j, message: j.status === "queued" ? "Queued behind another GPU job" : j.message }));
  } catch (err) {
    addError(section, `${label}: ${err.message}`, paperId);
  } finally {
    state.busy.delete(busyKey);
    onSettled?.();
    row.remove();
  }
  if (job && paperId === state.paperId) await loadLab(paperId, { quiet: true });
  render();
  if (job && paperId === state.paperId) onDone?.(job);
}

function addError(section, text, paperId = state.paperId) {
  state.errors.push({ id: ++state.errorSeq, section, paperId, text });
  renderJobsAndErrors();
}

function renderJobsAndErrors() {
  for (const [name, el] of Object.entries(sections)) {
    for (const row of el.jobs.children) row.hidden = row.dataset.paper !== state.paperId;
    el.errors.replaceChildren(...state.errors.filter((e) => e.section === name && e.paperId === state.paperId).map((e) =>
      h("div", { class: "error dismissable", role: "alert" },
        h("span", { text: e.text }),
        h("button", {
          type: "button", class: "btn quiet dismiss", "aria-label": "Dismiss", text: "×",
          onclick: () => {
            state.errors = state.errors.filter((x) => x !== e);
            renderJobsAndErrors();
          },
        }))));
  }
}

function showPageError(message) {
  if (typeof message === "string") ui.pageError.textContent = message;
  else ui.pageError.replaceChildren(message);
  ui.pageError.hidden = !message;
}

// ---------------------------------------------------------------- playback

function player(url) {
  let audio = players.get(url);
  if (!audio) {
    audio = h("audio", { controls: true, preload: "metadata", src: url });
    players.set(url, audio);
  }
  return audio;
}

function notRendered(busy, onRender, label) {
  return h("div", { class: "not-rendered" },
    h("span", { class: "muted small", text: busy ? "Rendering…" : "Not rendered" }),
    h("button", { type: "button", class: "btn mini", disabled: busy, text: "Render", "aria-label": label, onclick: onRender }));
}

function clipStats(sample) {
  const parts = [];
  if (sample.seconds != null) parts.push(`${Number(sample.seconds).toFixed(1)} s`);
  if (sample.rtf != null) parts.push(`RTF ${Number(sample.rtf).toFixed(2)}`);
  return parts.join(" · ");
}

function pauseOthers(event) {
  for (const audio of document.querySelectorAll("audio")) {
    if (audio !== event.target) audio.pause();
  }
}

function pauseAll() {
  for (const audio of document.querySelectorAll("audio")) audio.pause();
}

/** "Play in sequence": when a clip ends, play the next visible clip of the same section. */
function playNext(section, current) {
  const clips = [...section.querySelectorAll("audio")].filter((a) => a.offsetParent !== null);
  const next = clips[clips.indexOf(current) + 1];
  if (!next) return;
  next.closest(".vcard, .qcard")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  next.play().catch(() => {});
}

// ---------------------------------------------------------------- small helpers

function ellipsize(text, max) {
  return text.length > max ? `${text.slice(0, max - 1).trimEnd()}…` : text;
}
