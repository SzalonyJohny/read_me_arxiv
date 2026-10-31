// Listen panel: price estimate → engine/voice choice (+ free Kokoro samples) → accept → TTS job.

import { api, ceilCents, fmt, h, pollJob, showProgress, store } from "/static/common.js";

const MAX_SAMPLES = 6;
const SAMPLE_ENGINE = "kokoro";
const TTS_SPEEDS = [0.8, 0.9, 1, 1.1, 1.2, 1.3];
const ENGINE_LABELS = { kokoro: "Kokoro (local)", qwen3: "Qwen3-TTS (local)", elevenlabs: "ElevenLabs" };

const $ = (id) => document.getElementById(id);
const ui = {
  dialog: $("listen"),
  close: $("listen-close"),
  summary: $("ls-summary"),
  options: $("ls-options"),
  engine: $("ls-engine"),
  modelField: $("ls-model-field"),
  model: $("ls-model"),
  voice: $("ls-voice"),
  speed: $("ls-speed"),
  engineNote: $("ls-engine-note"),
  voiceLab: $("ls-voicelab"),
  tryHint: $("ls-try-hint"),
  tryVoices: $("ls-try-voices"),
  tryGo: $("ls-try-go"),
  tryCount: $("ls-try-count"),
  tryProgress: $("ls-try-progress"),
  tryError: $("ls-try-error"),
  samples: $("ls-samples"),
  warning: $("ls-warning"),
  error: $("ls-error"),
  progress: $("ls-progress"),
  accept: $("ls-accept"),
};

const state = { estimate: null, engines: [], selected: null, loading: null, generating: false, sampling: false };
let ctx; // { paperId, openButton, buttonLabel, onAudioReady }

export function initListen(options) {
  ctx = options;
  ui.voiceLab.href = `/voices?paper=${encodeURIComponent(ctx.paperId)}`;
  const savedSpeed = store.get("ttsSpeed", 1);
  ui.speed.replaceChildren(...TTS_SPEEDS.map((s) => h("option", { value: String(s), text: `${s}×` })));
  ui.speed.value = String(TTS_SPEEDS.includes(savedSpeed) ? savedSpeed : 1);

  ctx.openButton.addEventListener("click", open);
  ui.close.addEventListener("click", () => ui.dialog.close());
  ui.dialog.addEventListener("click", (e) => {
    if (e.target === ui.dialog) ui.dialog.close(); // backdrop
  });
  ui.engine.addEventListener("change", () => selectEngine(ui.engine.value));
  ui.model.addEventListener("change", () => {
    store.set(`model:${ui.engine.value}`, ui.model.value);
    renderOptions();
    renderAccept();
  });
  ui.voice.addEventListener("change", () => store.set(`voice:${ui.engine.value}`, ui.voice.value));
  ui.speed.addEventListener("change", () => store.set("ttsSpeed", Number(ui.speed.value)));
  ui.options.addEventListener("click", onOptionClick);
  ui.tryVoices.addEventListener("change", renderTryCount);
  ui.tryGo.addEventListener("click", renderSamples);
  ui.samples.addEventListener("play", pauseOtherAudio, true);
  ui.accept.addEventListener("click", accept);
}

function open() {
  ui.dialog.showModal();
  if (state.loading) refreshSelected();
  state.loading ??= load();
}

/** Later opens: follow a selection changed in the Voice lab meanwhile (not one made in this panel). */
async function refreshSelected() {
  await state.loading;
  if (!state.engines.length) return;
  const selected = await api("/api/voices/selected").catch(() => state.selected);
  if (JSON.stringify(selected) === JSON.stringify(state.selected) || state.generating) return;
  state.selected = selected;
  const engine = selected && engineByName(selected.engine);
  if (engine?.available) selectEngine(engine.name, selected.model, selected.voice);
}

async function load() {
  try {
    const [, engines, selected] = await Promise.all([
      fetchEstimate(),
      api("/api/tts/engines"),
      api("/api/voices/selected").catch(() => null), // optional: the Voice lab's selection
    ]);
    state.engines = engines;
    state.selected = selected;
    renderEngines();
    renderTryVoices();
  } catch (err) {
    state.loading = null; // retry on the next open
    showMessage(ui.error, `Could not load the estimate: ${err.message}`);
    ui.accept.textContent = "Estimate unavailable";
  }
}

async function fetchEstimate() {
  state.estimate = await api(`/api/papers/${encodeURIComponent(ctx.paperId)}/estimate`);
  renderSummary();
  renderOptions();
  renderAccept();
}

// ---------------------------------------------------------------- estimate

function renderSummary() {
  const est = state.estimate;
  const cleanup = `${fmt.usd(est.cleanup_cost_usd)} ${est.cleanup_cost_is_actual ? "(actual)" : "(estimated)"}`;
  const rows = [
    ["Spoken text", `${fmt.int(est.characters)} characters · ${fmt.int(est.words)} words`],
    ["Audio", `≈ ${fmt.minutes(est.audio_minutes)}`],
    ["Text clean-up", est.cleanup_model ? `${cleanup} · ${est.cleanup_model}` : cleanup],
  ];
  ui.summary.replaceChildren(...rows.flatMap(([term, value]) => [h("dt", { text: term }), h("dd", { text: value })]));
}

function renderOptions() {
  if (!state.estimate) return;
  const current = selectedOption();
  ui.options.replaceChildren(...state.estimate.tts.map((option, i) =>
    h("tr", {
      class: [option.available ? "available" : "unavailable", option === current ? "selected" : ""].join(" "),
      dataset: { index: String(i) },
      title: option.available ? "Choose this option" : option.note ?? "Unavailable",
    },
    h("td", { text: option.label }),
    h("td", { class: "num", text: fmt.int(option.characters) }),
    h("td", { class: "num", text: fmt.minutes(option.audio_minutes) }),
    h("td", { class: "num cost", text: fmt.usd(option.cost_usd) }),
    h("td", { class: "num", text: option.est_processing_minutes == null ? "–" : fmt.minutes(option.est_processing_minutes) }),
    h("td", { class: "status" },
      h("span", { class: option.available ? "ok" : "no", text: option.available ? "Available" : "Unavailable" }),
      option.note && h("small", { text: option.note })))));
}

function onOptionClick(event) {
  const row = event.target.closest("tr.available");
  if (!row || state.generating) return;
  const option = state.estimate.tts[Number(row.dataset.index)];
  if (engineByName(option.engine)?.available) selectEngine(option.engine, option.model);
}

// ---------------------------------------------------------------- engine / model / voice

function engineByName(name) {
  return state.engines.find((e) => e.name === name);
}

function engineLabel(engine) {
  const label = ENGINE_LABELS[engine.name] ?? engine.name;
  return engine.available ? label : `${label} — ${engine.reason || "unavailable"}`;
}

function voiceLabel(voice, voices = []) {
  const extras = [];
  // Qwen3 takes share a description: tell them apart by take (ids end in _t<take>), else by id.
  if (voices.some((v) => v !== voice && v.label === voice.label)) {
    const take = /_t(\d+)$/.exec(voice.id);
    extras.push(take ? `take ${take[1]}` : voice.id);
  }
  if (voice.gender && !voice.label.toLowerCase().includes(voice.gender.toLowerCase())) extras.push(voice.gender);
  if (voice.grade) extras.push(`grade ${voice.grade}`);
  return [voice.label, ...extras].join(" · ");
}

function renderEngines() {
  ui.engine.replaceChildren(...state.engines.map((e) =>
    h("option", { value: e.name, disabled: !e.available, text: engineLabel(e), title: e.reason })));
  // The Voice lab's selection wins over this browser's last choice.
  const chosen = state.selected;
  const labEngine = chosen && state.engines.find((e) => e.available && e.name === chosen.engine);
  if (labEngine) {
    selectEngine(labEngine.name, chosen.model, chosen.voice);
    return;
  }
  const saved = store.get("engine");
  const engine = state.engines.find((e) => e.available && e.name === saved) ?? state.engines.find((e) => e.available);
  if (engine) {
    selectEngine(engine.name);
  } else {
    showMessage(ui.engineNote, "No text-to-speech engine is available on the server.");
    renderAccept();
  }
}

function selectEngine(name, model, voice) {
  const engine = engineByName(name);
  ui.engine.value = name;
  store.set("engine", name);

  const models = engine.models ?? [];
  ui.model.replaceChildren(...models.map((m) =>
    h("option", { value: m.id, text: m.usd_per_1k_chars != null ? `${m.label ?? m.id} · $${m.usd_per_1k_chars}/1k chars` : m.label ?? m.id })));
  ui.modelField.hidden = models.length === 0;
  if (models.length) ui.model.value = firstKnown(models.map((m) => m.id), model, store.get(`model:${name}`));

  renderVoiceSelect(engine, voice);
  showMessage(ui.engineNote, engine.device ? `Runs on ${engine.device}.` : "");
  renderOptions();
  renderAccept();
}

/** The first candidate that is one of `ids`, else the first id. */
function firstKnown(ids, ...candidates) {
  return candidates.find((c) => c && ids.includes(c)) ?? ids[0];
}

function renderVoiceSelect(engine, preferredVoice) {
  const byLanguage = new Map();
  for (const voice of engine.voices) {
    const lang = voice.language || "other";
    if (!byLanguage.has(lang)) byLanguage.set(lang, []);
    byLanguage.get(lang).push(voice);
  }
  const options = (voices) => voices.map((v) => h("option", { value: v.id, text: voiceLabel(v, engine.voices) }));
  ui.voice.replaceChildren(...(byLanguage.size > 1
    ? [...byLanguage].map(([lang, voices]) => h("optgroup", { label: lang }, options(voices)))
    : options(engine.voices)));
  ui.voice.disabled = engine.voices.length === 0;
  if (engine.voices.length) {
    ui.voice.value = firstKnown(engine.voices.map((v) => v.id), preferredVoice, store.get(`voice:${engine.name}`), engine.default_voice);
  }
}

function selection() {
  const engine = engineByName(ui.engine.value);
  return {
    engine: ui.engine.value,
    model: engine?.models?.length ? ui.model.value : null,
    voice: ui.voice.value,
    speed: Number(ui.speed.value),
  };
}

function selectedOption(sel = selection()) {
  return state.estimate?.tts.find((o) => o.engine === sel.engine && (sel.model == null || o.model === sel.model));
}

// ---------------------------------------------------------------- accept → generate

function acceptLabel(option) {
  if (state.generating) return "Generating audio…";
  if (!option) return state.engines.length ? "No estimate for this option" : "Loading…";
  if (!option.available) return `${option.label} is unavailable`;
  if (option.cost_usd === 0) {
    return option.engine === "kokoro" ? "Accept ($0.00, local) and generate audio" : "Accept ($0.00) and generate audio";
  }
  return `Accept ${fmt.usd(ceilCents(option.cost_usd))} and generate audio`;
}

function renderAccept() {
  const option = selectedOption();
  ui.accept.textContent = acceptLabel(option);
  ui.accept.disabled = state.generating || !option?.available || !engineByName(option.engine)?.available || !ui.voice.value;
}

function setGenerating(on) {
  state.generating = on;
  for (const el of [ui.engine, ui.model, ui.voice, ui.speed]) el.disabled = on;
  if (!on) {
    ui.progress.hidden = true;
    ctx.buttonLabel.textContent = "Listen";
    ui.voice.disabled = ui.voice.options.length === 0;
  }
  renderAccept();
}

async function accept() {
  const sel = selection();
  const option = selectedOption(sel);
  if (!option) return;
  const accepted = ceilCents(option.cost_usd);
  const body = { engine: sel.engine, voice: sel.voice, speed: sel.speed, accepted_cost_usd: accepted };
  if (sel.model) body.model = sel.model;

  showMessage(ui.warning, "");
  showMessage(ui.error, "");
  setGenerating(true);
  showProgress(ui.progress, { progress: 0, message: "Submitting" });
  try {
    const { job_id: jobId } = await api(`/api/papers/${encodeURIComponent(ctx.paperId)}/tts`, { method: "POST", body });
    const job = await pollJob(jobId, (j) => {
      showProgress(ui.progress, j);
      ctx.buttonLabel.textContent = `Generating ${Math.round((j.progress ?? 0) * 100)}%`;
    });
    await ctx.onAudioReady(job.result.run_id);
    ui.dialog.close();
  } catch (err) {
    if (err.status === 409) await onPriceChanged(err.detail, accepted);
    else showMessage(ui.error, err.message);
  } finally {
    setGenerating(false);
  }
}

/** 409: the server's estimate is above what was accepted. Show the new price and ask again. */
async function onPriceChanged(detail, accepted) {
  try {
    await fetchEstimate();
  } catch (err) {
    showMessage(ui.error, `Could not reload the estimate: ${err.message}`);
  }
  const now = detail?.estimated_cost_usd ?? selectedOption()?.cost_usd;
  showMessage(ui.warning,
    `The price is now ${fmt.usd(now)}, above the ${fmt.usd(accepted)} you accepted. Check the updated estimate and accept again.`);
}

// ---------------------------------------------------------------- voice samples (Kokoro)

function renderTryVoices() {
  const engine = engineByName(SAMPLE_ENGINE);
  if (!engine?.available) {
    ui.tryHint.textContent = `Voice samples need Kokoro, which is unavailable: ${engine?.reason || "not installed"}.`;
    ui.tryVoices.replaceChildren();
    renderTryCount();
    return;
  }
  const english = engine.voices.filter((v) => (v.language || "").toLowerCase().startsWith("en"));
  ui.tryVoices.replaceChildren(...(english.length ? english : engine.voices).map((v) =>
    h("label", { class: "check voice-option" },
      h("input", { type: "checkbox", value: v.id, checked: v.id === engine.default_voice }),
      h("span", { text: voiceLabel(v) }))));
  renderTryCount();
}

function chosenSampleVoices() {
  return [...ui.tryVoices.querySelectorAll("input:checked")].map((input) => input.value);
}

function renderTryCount() {
  const count = chosenSampleVoices().length;
  for (const box of ui.tryVoices.querySelectorAll("input:not(:checked)")) box.disabled = count >= MAX_SAMPLES;
  ui.tryCount.textContent = ui.tryVoices.childElementCount ? `${count} of ${MAX_SAMPLES} selected` : "";
  ui.tryGo.disabled = count === 0 || state.sampling;
}

async function renderSamples() {
  state.sampling = true;
  renderTryCount();
  showMessage(ui.tryError, "");
  showProgress(ui.tryProgress, { progress: 0, message: "Submitting" });
  try {
    const { job_id: jobId } = await api("/api/tts/samples", {
      method: "POST",
      body: { engine: SAMPLE_ENGINE, voices: chosenSampleVoices(), speed: Number(ui.speed.value), paper_id: ctx.paperId },
    });
    const job = await pollJob(jobId, (j) => showProgress(ui.tryProgress, j));
    showSamples(job.result.samples);
  } catch (err) {
    showMessage(ui.tryError, err.message);
  } finally {
    state.sampling = false;
    ui.tryProgress.hidden = true;
    renderTryCount();
  }
}

function showSamples(samples) {
  const voices = engineByName(SAMPLE_ENGINE).voices;
  ui.samples.replaceChildren(...samples.map((sample) => {
    const voice = voices.find((v) => v.id === sample.voice);
    return h("li", { class: "sample", dataset: { voice: sample.voice } },
      h("div", { class: "sample-head" },
        h("strong", { text: voice ? voiceLabel(voice) : sample.voice }),
        sample.seconds && h("span", { class: "muted small", text: `${sample.seconds.toFixed(1)} s` }),
        h("button", { type: "button", class: "btn", text: "Use this voice", onclick: () => useVoice(sample.voice) })),
      h("audio", { controls: true, preload: "none", src: sample.url }));
  }));
}

function useVoice(voiceId) {
  selectEngine(SAMPLE_ENGINE);
  ui.voice.value = voiceId;
  store.set(`voice:${SAMPLE_ENGINE}`, voiceId);
  for (const li of ui.samples.children) li.classList.toggle("chosen", li.dataset.voice === voiceId);
  renderAccept();
}

function pauseOtherAudio(event) {
  for (const audio of document.querySelectorAll("audio")) {
    if (audio !== event.target) audio.pause();
  }
}

// ---------------------------------------------------------------- small helpers

function showMessage(el, text) {
  el.textContent = text;
  el.hidden = !text;
}
