// Reader page: the paper on the left, the most recently mentioned figure on the right,
// and (when a narration run exists) an audio player that highlights and follows the text.

import { api, connection, fmt, h, pollJob, showProgress, store } from "/static/common.js";
import { initListen } from "/static/listen.js";
import { initMediaSession, setNowPlaying } from "/static/media-session.js";
import { registerServiceWorker } from "/static/offline.js";
import { initSaveOffline, narrationChanged } from "/static/save-offline.js";

const READING_LINE = 0.3; // when scrolling, the block crossing 30% of the text pane is "being read"
const FOLLOW_LINE = 0.35; // auto-follow puts the narrated unit at 35% of the text pane
const TALL_UNIT = 0.6; // units taller than this share of the pane are followed sentence by sentence
const RESUME_AFTER_MS = 8000;
const AUTO_SCROLL_MS = 1200;
const USER_SCROLL_MS = 1000; // scroll events this soon after user input (or each other) are the user's
const SAVE_EVERY_MS = 3000;
const SKIP_SECONDS = 15;
const RATES = [0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2];
const NAV_KEYS = new Set(["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End"]);
const SCROLL_BEHAVIOR = matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
const SPEECH_SOURCES = { filter: "rules", llm: "LLM", llm_rejected: "rules · LLM output rejected" };
const SOURCES = { "arxiv-html": "arXiv HTML", "pdf-text": "PDF text layer", "pdf-ocr": "PDF via OCR (DeepSeek-OCR 2)" };
const FIGURE_MODES = { narration: "following narration", scroll: "following scroll", pinned: "pinned · unpin" };
const LIST_LABEL = /^\s*([•◦▪‣∙·*–-]|\(?[0-9ivxa-zA-Z]{1,4}[.)])\s/;
const KATEX_OPTIONS = {
  delimiters: [
    { left: "\\(", right: "\\)", display: false },
    { left: "\\[", right: "\\]", display: true },
  ],
  throwOnError: false,
  ignoredClasses: ["speech", "katex", "eq-raw"],
};

const $ = (id) => document.getElementById(id);
const ui = {
  pane: $("text-pane"),
  paper: $("paper"),
  toc: $("toc"),
  tocList: $("toc-list"),
  shortTitle: $("short-title"),
  optCites: $("opt-cites"),
  optSpeech: $("opt-speech"),
  optFollow: $("opt-follow"),
  player: $("player"),
  play: $("pl-play"),
  back: $("pl-back"),
  fwd: $("pl-fwd"),
  seek: $("pl-seek"),
  time: $("pl-time"),
  rate: $("pl-rate"),
  run: $("pl-run"),
  nowReading: $("now-reading"),
  resume: $("resume"),
  audio: $("audio"),
  figPane: $("fig-pane"),
  figCollapse: $("fig-collapse"),
  figLabel: $("fig-label"),
  figMode: $("fig-mode"),
  figPrev: $("fig-prev"),
  figNext: $("fig-next"),
  figPos: $("fig-pos"),
  figBody: $("fig-body"),
  lightbox: $("lightbox"),
  lightboxImg: $("lightbox-img"),
  listen: $("listen-open"),
};

const paperId = decodeURIComponent(location.pathname.replace(/^\/reader\//, "").replace(/\/+$/, ""));
// ?t=SECONDS opens the narration at that time (a shareable deep link; also handy for testing).
let startAt = Number.parseFloat(new URLSearchParams(location.search).get("t"));

const state = {
  paper: null,
  blocks: [], // all blocks in reading order
  blockEls: [], // their elements, same order
  sectionOf: new Map(), // block or section id -> its section
  scrollFigure: [], // per block: most recently mentioned figure at or before it
  figureIds: [], // figures and tables in paper order (prev/next)
  figure: { id: undefined, source: null }, // source: "scroll" | "narration" | "pinned"
  runs: [],
  run: null, // active AudioManifest
  segStarts: [],
  segFigure: [], // per segment: last figure mentioned up to and including it
  segFraction: [], // per segment: share of its unit's text spoken before it
  unitFirstSeg: new Map(), // unit id -> index of its first segment
  segIndex: -1,
  readingEl: null,
  pendingSeek: null,
  seekDragging: false,
  ticking: false,
  lastSaved: 0,
};

const follow = { suspended: false, resumeTimer: 0, autoScrolling: false, autoTimer: 0, userScrollAt: -Infinity };

// ---------------------------------------------------------------- helpers

function paperFile(path) {
  return `/papers/${encodeURIComponent(paperId)}/${path.split("/").map(encodeURIComponent).join("/")}`;
}

/** Index of the last item for which `test` holds, assuming true…true false…false; -1 if none. */
function lastIndexWhere(items, test) {
  let lo = 0;
  let hi = items.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (test(items[mid])) {
      found = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return found;
}

/** Top of `el` in the text pane's scroll coordinates. */
function offsetInPane(el) {
  return el.getBoundingClientRect().top - ui.pane.getBoundingClientRect().top + ui.pane.scrollTop;
}

function isTextEntry(el) {
  return Boolean(el.closest?.("input, textarea, select, [contenteditable]"));
}

function hasSelection() {
  const selection = getSelection();
  return Boolean(selection && !selection.isCollapsed && selection.toString().trim());
}

function renderMath(root) {
  for (const el of root.querySelectorAll(".eq-body[data-latex]")) {
    const latex = el.dataset.latex;
    el.removeAttribute("data-latex");
    if (window.katex) window.katex.render(latex, el, { displayMode: true, throwOnError: false });
    else el.textContent = `\\[${latex}\\]`;
  }
  window.renderMathInElement?.(root, KATEX_OPTIONS);
}

/**
 * First words of a plain-text caption (maths as $LaTeX$), without its "Figure 3:" label.
 * Maths becomes \(…\) for KaTeX and is never cut in half.
 */
function excerpt(text, max = 110) {
  const body = (text || "").replace(/^\s*(Figure|Fig\.|Table|Tab\.)\s*[\w.]+\s*[:.]\s*/i, "");
  let out = "";
  for (const part of body.split(/(\$[^$]+\$)/)) {
    const isMath = part.length > 2 && part.startsWith("$") && part.endsWith("$");
    const piece = isMath ? `\\(${part.slice(1, -1)}\\)` : part.replace(/\s+/g, " ");
    if (out.length + piece.length > max) {
      if (!isMath) out += piece.slice(0, max - out.length).replace(/\s+\S*$/, "");
      return `${out.trimEnd()}…`;
    }
    out += piece;
  }
  return out.trim();
}

/** Moves the space before each citation into it, so hidden citations leave no " ." behind. */
function attachCitationSpaces(root) {
  for (const cite of root.querySelectorAll(".cite")) {
    const before = cite.previousSibling;
    if (before?.nodeType !== Node.TEXT_NODE || !/\s$/.test(before.data)) continue;
    before.data = before.data.trimEnd();
    cite.prepend(" ");
  }
}

/** Briefly marks the element a link or the section menu jumped to. */
function flash(el) {
  el.classList.remove("flash");
  void el.offsetWidth; // restart the animation
  el.classList.add("flash");
}

// ---------------------------------------------------------------- rendering the paper

function renderPaper(paper) {
  document.title = `${plainText(paper.title)} · Paper Reader`;
  ui.shortTitle.textContent = mathDelimiters(shortTitle(paper.title));
  ui.shortTitle.title = paper.title;
  ui.paper.replaceChildren(renderFrontMatter(paper), ...paper.sections.map(renderSection));
  for (const link of ui.paper.querySelectorAll('a[href^="http"]')) {
    link.target = "_blank";
    link.rel = "noopener";
  }
  attachCitationSpaces(ui.paper);
  renderToc(paper.sections);
  for (const root of [ui.paper, ui.tocList, ui.shortTitle]) renderMath(root);
}

/** Titles as plain text, e.g. for the tab and the lock screen ($…$ maths keeps its LaTeX). */
function plainText(text) {
  return text.replace(/\$/g, "");
}

/** Titles carry inline maths as $…$; KaTeX auto-render here uses \(…\). */
function mathDelimiters(text) {
  return text.replace(/\$([^$]+)\$/g, "\\($1\\)");
}

function shortTitle(title) {
  const head = title.split(/:\s/)[0];
  return head.length >= 8 ? head : title;
}

function renderFrontMatter(paper) {
  return h("header", { class: "front" },
    h("h1", { class: "paper-title", text: mathDelimiters(paper.title) }),
    paper.authors.length > 0 && h("p", { class: "authors", text: paper.authors.join(", ") }),
    h("p", { class: "source" }, sourceLine(paper)),
    renderAbout(paper));
}

/** arXiv papers link to arXiv; uploads show the file name and link to the original PDF. */
function sourceLine(paper) {
  const fromArxiv = (paper.source ?? "arxiv-html") === "arxiv-html" || (paper.arxiv_id && paper.html_url);
  if (fromArxiv) {
    return h("a", { href: paper.html_url || paper.source_url, target: "_blank", rel: "noopener",
                    text: `arXiv:${paper.arxiv_id}${paper.version ?? ""}` });
  }
  const filename = paper.source_url?.startsWith("upload:") ? paper.source_url.slice("upload:".length) : "";
  return [
    h("span", { class: "source-file", text: filename || "Uploaded PDF" }),
    " · ",
    h("a", { href: paperFile("source.pdf"), target: "_blank", rel: "noopener", text: "Original PDF" }),
  ];
}

function renderAbout(paper) {
  const rows = [
    ["Source", SOURCES[paper.source ?? "arxiv-html"] ?? paper.source],
    ["Appendix", paper.include_appendix ? "included" : "skipped"],
    ...cleanupRows(paper.cleanup),
  ];
  const warnings = paper.warnings ?? [];
  return h("details", { class: "about" },
    h("summary", {}, "About this text",
      warnings.length > 0 && h("span", { class: "badge warn", text: `${warnings.length} warning${warnings.length > 1 ? "s" : ""}` })),
    h("dl", {}, rows.flatMap(([term, value]) => [h("dt", { text: term }), h("dd", { text: value })])),
    warnings.length > 0 && h("ul", { class: "warnings" }, warnings.map((w) => h("li", { text: w }))),
    renderSymbolNames(paper.cleanup),
    renderRecleanup(paper));
}

// Meaning mode: what each symbol is read as (the same name in every equation), rendered with KaTeX.
function renderSymbolNames(cleanup) {
  const names = Object.entries(cleanup?.symbol_names ?? {});
  const byMeaning = cleanup?.equation_mode === "meaning" || cleanup?.inline_mode === "meaning";
  if (!byMeaning || names.length === 0) return null;
  return h("details", { class: "symbol-names" },
    h("summary", { text: `Symbol names (${names.length})` }),
    h("dl", {}, names.flatMap(([latex, name]) => [
      h("dt", { text: `\\(${latex}\\)` }),
      h("dd", { text: name }),
    ])));
}

// Re-run the clean-up of this paper, e.g. to read equations by meaning. Needs the server, so it
// stays hidden offline; the page reloads with the new text when the job is done.
function renderRecleanup(paper) {
  const select = h("select", { "aria-label": "LLM clean-up runner" });
  const meaning = h("input", { type: "checkbox", checked: true }); // reading by meaning is the default
  const inlineMeaning = h("input", { type: "checkbox", checked: true });
  const run = h("button", { class: "btn", type: "button", text: "Re-run clean-up" });
  const progress = h("div", { class: "progress", hidden: true }, h("progress", { max: 1, value: 0 }), h("p", { class: "msg" }));
  const status = h("p", { class: "muted recleanup-note",
    text: "This changes the narration text. Generate the narration again to hear it; unchanged paragraphs come from the cache." });
  const box = h("div", { class: "recleanup", hidden: true },
    h("label", {}, "LLM ", select),
    h("label", { class: "check" }, meaning, " Read equations by meaning"),
    h("label", { class: "check" }, inlineMeaning, " Read symbols in the text by meaning"),
    run, progress, status);
  const sync = () => {
    meaning.disabled = inlineMeaning.disabled = select.value === "none"; // only an LLM can read by meaning
  };
  api("/api/config").then((config) => {
    const runners = [...config.cleanup_runners, { id: "none", label: "Off: rule-based filter only", available: true }];
    for (const r of runners) {
      select.append(h("option", { value: r.id, disabled: !r.available, text: r.available ? r.label : `${r.label}: unavailable` }));
    }
    select.value = config.default_cleanup || "none";
    sync();
    box.hidden = connection.offline;
  }).catch(() => { box.hidden = true; });
  select.addEventListener("change", sync);
  run.addEventListener("click", async () => {
    for (const el of [select, meaning, inlineMeaning, run]) el.disabled = true;
    try {
      const { job_id: jobId } = await api(`/api/papers/${encodeURIComponent(paperId)}/cleanup`, {
        method: "POST", body: { llm: select.value, equations: meaning.checked ? "meaning" : "symbols",
                                inline: inlineMeaning.checked ? "meaning" : "symbols" } });
      await pollJob(jobId, (job) => showProgress(progress, job));
      location.reload();
    } catch (err) {
      status.textContent = `Clean-up failed: ${err.message}`;
      for (const el of [select, run]) el.disabled = false;
      sync();
    }
  });
  return box;
}

function symbolSummary(cleanup) {
  const named = Object.keys(cleanup.symbol_names ?? {}).length;
  const open = (cleanup.inconsistent_symbols ?? []).length;
  const parts = [`${named} named, each read the same way everywhere`];
  if (cleanup.symbols_repaired) parts.push(`${cleanup.symbols_repaired} blocks re-asked to match`);
  if (open) parts[0] = `${named} named; ${open} still read inconsistently (see warnings)`;
  return parts.join(" · ");
}

function cleanupRows(cleanup) {
  if (!cleanup) return [["Clean-up", "rule-based filter only"]];
  if (cleanup.skipped_reason) return [["LLM clean-up", `skipped (${cleanup.skipped_reason}); rule-based filter only`]];
  return [
    ["LLM clean-up", cleanup.model ?? "–"],
    ["Cost", `${fmt.usd(cleanup.cost_usd)} · ${fmt.int(cleanup.input_tokens)} tokens in, ${fmt.int(cleanup.output_tokens)} out`],
    ["Blocks", `${cleanup.blocks_llm} cleaned, ${cleanup.blocks_rejected} rejected by the rewrite guard (rule text kept)`],
    ["Equations", cleanup.equation_mode === "meaning" ? "read by meaning (from the paper's definitions)" : "read symbol by symbol"],
    ["Symbols in the text", cleanup.inline_mode === "meaning" ? "read by meaning" : "read symbol by symbol"],
    ...("meaning" === cleanup.equation_mode || "meaning" === cleanup.inline_mode ? [["Symbols", symbolSummary(cleanup)]] : []),
  ];
}

function renderToc(sections) {
  ui.tocList.replaceChildren(...sections.map((s) =>
    h("a", { class: `toc-item lvl-${s.level}`, href: `#${s.id}`, dataset: { target: s.id } },
      h("span", { class: "toc-num", text: s.number ?? "" }),
      h("span", { class: "toc-title", text: mathDelimiters(s.title) }))));
}

function renderSection(section) {
  const level = Math.min(section.level + 1, 4);
  const heading = h(`h${level}`, { id: section.id, class: "sec-title" },
    section.number && h("span", { class: "sec-num", text: section.number }),
    h("span", { text: mathDelimiters(section.title) }));
  const blocks = section.blocks.map(renderBlock);
  section.blocks.forEach((block, i) => {
    // \paragraph / theorem / proof titles run into the paragraph that follows them.
    if (block.type === "heading_note" && section.blocks[i + 1]?.type === "paragraph") blocks[i].classList.add("run-in");
  });
  return h("section", { class: `sec sec-${section.kind} lvl-${section.level}` },
    heading,
    section.speech_title && speechBox(section.speech_title, null),
    blocks);
}

const BLOCK_RENDERERS = {
  equation: renderEquation,
  figure: renderFigureChip,
  list_item: renderListItem,
};

function renderBlock(block) {
  const el = (BLOCK_RENDERERS[block.type] ?? renderHtml)(block);
  el.id = block.id;
  el.classList.add("blk", `blk-${block.type}`);
  if (block.speak && block.speech) el.append(speechBox(block.speech, block.speech_source));
  if (block.type === "equation" && block.speak && block.speech) {
    // Click (or Enter) shows how the equation is read aloud, in small text under it.
    el.classList.add("has-reading");
    el.tabIndex = 0;
    el.title = "Show how this equation is read aloud";
    el.setAttribute("aria-expanded", "false");
  }
  return el;
}

function toggleReading(equation) {
  const open = equation.classList.toggle("show-reading");
  equation.setAttribute("aria-expanded", String(open));
}

function renderHtml(block) {
  const el = h("div");
  el.innerHTML = block.html;
  return el;
}

function renderListItem(block) {
  const el = renderHtml(block);
  if (LIST_LABEL.test(block.text || el.textContent)) el.classList.add("has-label");
  return el;
}

function renderEquation(block) {
  const body = h("div", { class: "eq-body" });
  if (block.latex) {
    body.dataset.latex = block.latex;
  } else if (state.paper.source?.startsWith("pdf")) {
    // PDF text layer: no LaTeX, just the equation's raw text; shown as is, never fed to KaTeX.
    body.classList.add("eq-raw");
    if (block.html) body.innerHTML = block.html;
    else body.textContent = block.text;
  } else {
    body.innerHTML = block.html;
  }
  return h("div", {}, body, block.eq_number && h("span", { class: "eq-num", text: `(${block.eq_number})` }));
}

function renderFigureChip(block) {
  const fig = state.paper.figures[block.figure_id];
  if (!fig) return h("p", { class: "muted small", text: "Figure not available." });
  const thumb = fig.images.length > 0
    ? h("img", { class: "chip-thumb", src: paperFile(fig.images[0]), alt: "", loading: "lazy" })
    : h("span", { class: "chip-thumb chip-table", "aria-hidden": "true" });
  return h("div", {},
    h("button", { type: "button", class: "fig-chip", dataset: { fig: fig.id }, title: `Show ${fig.label} on the right` },
      thumb,
      h("span", { class: "chip-text" },
        h("strong", { text: fig.label }),
        h("span", { text: excerpt(fig.caption_text) }))));
}

function speechBox(text, source) {
  return h("div", { class: "speech" },
    h("span", { class: `speech-src src-${source ?? "title"}`, text: source ? SPEECH_SOURCES[source] ?? source : "heading" }),
    h("span", { class: "speech-text", text }));
}

// ---------------------------------------------------------------- figure bookkeeping

function indexPaper(paper) {
  state.blocks = paper.sections.flatMap((s) => s.blocks);
  state.blockEls = state.blocks.map((b) => $(b.id));
  state.sectionOf = new Map(paper.sections.flatMap((s) => [[s.id, s], ...s.blocks.map((b) => [b.id, s])]));
  state.scrollFigure = lastMentionedPerBlock(state.blocks, paper.figures);
  state.figureIds = Object.keys(paper.figures);
}

/** Per block: the last figure referenced so far. A float counts as a mention if not yet mentioned. */
function lastMentionedPerBlock(blocks, figures) {
  const mentioned = new Set();
  let current = null;
  return blocks.map((block) => {
    for (const id of block.fig_refs) {
      if (!(id in figures)) continue;
      mentioned.add(id);
      current = id;
    }
    if (block.type === "figure" && block.figure_id in figures && !mentioned.has(block.figure_id)) {
      mentioned.add(block.figure_id);
      current = block.figure_id;
    }
    return current;
  });
}

/** Per segment: the last figure id mentioned in any segment up to and including it. */
function lastFigurePerSegment(segments, figures) {
  let current = null;
  return segments.map((seg) => {
    const known = seg.figure_ids.filter((id) => id in figures);
    if (known.length) current = known.at(-1);
    return current;
  });
}

function figureAtReadingLine() {
  const line = ui.pane.getBoundingClientRect().top + READING_LINE * ui.pane.clientHeight;
  const i = lastIndexWhere(state.blockEls, (el) => el.getBoundingClientRect().top <= line);
  return i < 0 ? null : state.scrollFigure[i];
}

// ---------------------------------------------------------------- figure pane

function showFigure(id, source) {
  if (state.figure.source === "pinned" && source !== "pinned") return;
  state.figure.source = source;
  if (id !== state.figure.id) {
    state.figure.id = id;
    renderFigure(id);
  }
  renderFigureStatus();
}

function pinFigure(id) {
  setFigureCollapsed(false);
  showFigure(id, "pinned");
}

function unpinFigure() {
  state.figure.source = null;
  refreshFigure();
}

function refreshFigure() {
  if (isNarrating()) showFigure(state.segFigure[state.segIndex] ?? null, "narration");
  else showFigure(figureAtReadingLine(), "scroll");
}

function renderFigure(id) {
  const fig = id ? state.paper.figures[id] : null;
  ui.figLabel.textContent = fig ? fig.label : "Figures";
  ui.figBody.replaceChildren(fig ? figureView(fig) : emptyFigureView());
  if (fig) renderMath(ui.figBody);
  ui.figPane.scrollTop = 0;
  for (const chip of ui.paper.querySelectorAll(".fig-chip")) chip.classList.toggle("is-current", chip.dataset.fig === id);
  updateNowPlaying();
}

function figureView(fig) {
  const caption = h("figcaption", { class: "fig-caption" });
  if (fig.caption_html) caption.innerHTML = fig.caption_html;
  else caption.textContent = fig.caption_text;
  if (fig.table_html) {
    const table = h("div", { class: "fig-table" });
    table.innerHTML = fig.table_html;
    return h("figure", { class: "fig fig-has-table" }, caption, table);
  }
  const images = h("div", { class: "fig-images" },
    fig.images.map((path, i) => h("img", {
      src: paperFile(path),
      alt: fig.images.length > 1 ? `${fig.label}, part ${i + 1}` : fig.label,
      title: "Click to enlarge",
      onclick: (e) => openLightbox(e.currentTarget),
    })));
  return h("figure", { class: "fig" }, images, caption);
}

function emptyFigureView() {
  const text = state.figureIds.length
    ? "Figures and tables appear here as the text mentions them."
    : "This paper has no figures or tables.";
  return h("p", { class: "fig-empty muted", text });
}

function renderFigureStatus() {
  const { id, source } = state.figure;
  const pinned = source === "pinned";
  ui.figMode.textContent = FIGURE_MODES[source] ?? FIGURE_MODES.scroll;
  ui.figMode.disabled = !pinned;
  ui.figMode.classList.toggle("pinned", pinned);
  ui.figMode.title = pinned ? "Unpin: follow the text again" : "";
  const count = state.figureIds.length;
  const i = state.figureIds.indexOf(id);
  ui.figPos.textContent = count ? `${i >= 0 ? i + 1 : "–"} / ${count}` : "";
  ui.figPrev.disabled = i <= 0;
  ui.figNext.disabled = count === 0 || i === count - 1;
}

function stepFigure(delta) {
  const i = state.figureIds.indexOf(state.figure.id);
  const next = state.figureIds[Math.min(Math.max(i + delta, 0), state.figureIds.length - 1)];
  if (next) pinFigure(next);
}

function openLightbox(img) {
  ui.lightboxImg.src = img.src;
  ui.lightboxImg.alt = img.alt;
  ui.lightbox.showModal();
}

function setFigureCollapsed(collapsed) {
  ui.figPane.classList.toggle("collapsed", collapsed);
  ui.figCollapse.setAttribute("aria-expanded", String(!collapsed));
  store.set("figCollapsed", collapsed);
}

// ---------------------------------------------------------------- scrolling the text

function scrollToElement(el) {
  follow.userScrollAt = performance.now();
  ui.pane.scrollTo({ top: Math.max(0, offsetInPane(el) - 24), behavior: SCROLL_BEHAVIOR });
  flash(el);
}

/** Programmatic scroll for auto-follow; scroll events it causes don't drive the scroll-figure. */
function autoScrollTo(top) {
  follow.autoScrolling = true;
  clearTimeout(follow.autoTimer);
  follow.autoTimer = setTimeout(endAutoScroll, AUTO_SCROLL_MS);
  ui.pane.scrollTo({ top: Math.max(0, top), behavior: SCROLL_BEHAVIOR });
}

function endAutoScroll() {
  follow.autoScrolling = false;
  clearTimeout(follow.autoTimer);
}

let scrollQueued = false;
function onPaneScroll() {
  const now = performance.now();
  const byUser = now - follow.userScrollAt < USER_SCROLL_MS;
  if (byUser) follow.userScrollAt = now; // a long drag or fling keeps counting as the user's
  if (scrollQueued) return;
  scrollQueued = true;
  requestAnimationFrame(() => {
    scrollQueued = false;
    if (scrollDrivesFigure(byUser)) showFigure(figureAtReadingLine(), "scroll");
  });
}

/**
 * Scrolling picks the figure unless narration does. While paused at a narration position, only the
 * user's own scrolling counts: layout shifts (toggles, scroll anchoring) keep the narrated figure.
 */
function scrollDrivesFigure(byUser) {
  if (follow.autoScrolling || isNarrating()) return false;
  return byUser || state.segIndex < 0;
}

function shouldFollow() {
  return ui.optFollow.checked && !follow.suspended;
}

/** The user scrolls by hand (wheel, touch, keys, scrollbar, jumps). While narration plays, stop following for a while. */
function onManualScroll() {
  follow.userScrollAt = performance.now();
  if (!isNarrating() || !ui.optFollow.checked) return;
  follow.suspended = true;
  endAutoScroll();
  ui.resume.hidden = false;
  clearTimeout(follow.resumeTimer);
  follow.resumeTimer = setTimeout(() => resumeFollow(!ui.audio.paused), RESUME_AFTER_MS);
}

function resumeFollow(scrollBack) {
  clearTimeout(follow.resumeTimer);
  follow.suspended = false;
  ui.resume.hidden = true;
  if (scrollBack && ui.optFollow.checked && state.segIndex >= 0) followSegment(state.segIndex, true);
}

/** Scrolls the narrated unit to FOLLOW_LINE; inside tall units, tracks the sentence position. */
function followSegment(i, unitChanged) {
  const el = $(state.run.segments[i].unit_id);
  if (!el) return;
  const tall = el.offsetHeight > TALL_UNIT * ui.pane.clientHeight;
  if (!unitChanged && !tall) return;
  const within = tall ? state.segFraction[i] * el.offsetHeight : 0;
  autoScrollTo(offsetInPane(el) + within - FOLLOW_LINE * ui.pane.clientHeight);
}

// ---------------------------------------------------------------- narration runs

async function loadRuns(preferredRunId) {
  try {
    state.runs = await api(`/api/papers/${encodeURIComponent(paperId)}/audio`);
  } catch (err) {
    if (!connection.offline) console.warn("Could not load narration runs:", err); // offline: not saved here
    state.runs = [];
  }
  ui.run.replaceChildren(...state.runs.map((r) => h("option", { value: r.run_id, text: runLabel(r) })));
  ui.run.hidden = state.runs.length < 2;
  if (state.runs.length) {
    const wanted = preferredRunId ?? store.get(`run:${paperId}`);
    activateRun(state.runs.find((r) => r.run_id === wanted) ?? state.runs.at(-1));
  }
  narrationChanged();
}

function runLabel(run) {
  return [run.engine, run.model, run.voice, `${run.speed}×`].filter(Boolean).join(" · ");
}

function activateRun(run) {
  savePosition();
  ui.audio.pause();
  markReading(null);
  state.run = run;
  state.segStarts = run.segments.map((s) => s.start);
  state.segFigure = lastFigurePerSegment(run.segments, state.paper.figures);
  state.segFraction = fractionWithinUnit(run.segments);
  state.unitFirstSeg = firstSegmentPerUnit(run.segments);
  state.segIndex = -1;
  markSeekable();
  state.pendingSeek = Number.isFinite(startAt) ? startAt : store.get(positionKey(run), 0);
  startAt = Number.NaN;
  ui.audio.src = paperFile(run.audio_file);
  ui.run.value = run.run_id;
  ui.seek.max = String(run.duration || 0);
  ui.player.hidden = false;
  ui.nowReading.hidden = false;
  ui.nowReading.textContent = "";
  document.body.classList.add("has-audio");
  store.set(`run:${paperId}`, run.run_id);
  updateTimeUI();
  updateNowPlaying();
}

function fractionWithinUnit(segments) {
  const totals = new Map();
  for (const s of segments) totals.set(s.unit_id, (totals.get(s.unit_id) ?? 0) + s.text.length);
  const spoken = new Map();
  return segments.map((s) => {
    const before = spoken.get(s.unit_id) ?? 0;
    spoken.set(s.unit_id, before + s.text.length);
    return totals.get(s.unit_id) ? before / totals.get(s.unit_id) : 0;
  });
}

function firstSegmentPerUnit(segments) {
  const first = new Map();
  segments.forEach((s, i) => {
    if (!first.has(s.unit_id)) first.set(s.unit_id, i);
  });
  return first;
}

function markSeekable() {
  for (const el of ui.paper.querySelectorAll(".seekable")) el.classList.remove("seekable");
  for (const id of state.unitFirstSeg.keys()) $(id)?.classList.add("seekable");
}

function positionKey(run) {
  return `pos:${paperId}:${run.run_id}`;
}

function savePosition() {
  state.lastSaved = performance.now();
  if (state.run && ui.audio.currentTime > 0) store.set(positionKey(state.run), Math.round(ui.audio.currentTime * 10) / 10);
}

// ---------------------------------------------------------------- audio sync

function isNarrating() {
  return state.run !== null && !ui.audio.paused;
}

function mediaDuration() {
  return Number.isFinite(ui.audio.duration) ? ui.audio.duration : state.run?.duration ?? 0;
}

function syncToAudio(seeked = false) {
  updateTimeUI();
  if (!state.run) return;
  const t = ui.audio.currentTime;
  const i = lastIndexWhere(state.segStarts, (start) => start <= t + 0.02);
  if (i !== state.segIndex || seeked) setSegment(i, seeked);
  if (!ui.audio.paused && performance.now() - state.lastSaved > SAVE_EVERY_MS) savePosition();
}

function setSegment(i, seeked) {
  const segments = state.run.segments;
  const previous = segments[state.segIndex];
  const seg = segments[i];
  state.segIndex = i;
  ui.nowReading.textContent = seg?.text ?? "";
  const unitChanged = seg?.unit_id !== previous?.unit_id;
  if (unitChanged) markReading(seg ? $(seg.unit_id) : null);
  if (seg && shouldFollow()) followSegment(i, unitChanged || seeked);
  if (seeked || isNarrating()) showFigure(state.segFigure[i] ?? null, "narration");
  updateNowPlaying();
}

/** Lock-screen metadata: the paper, the narrated section and the figure on the right. */
function updateNowPlaying() {
  if (!state.run) return;
  const seg = state.run.segments[state.segIndex];
  const section = seg && state.sectionOf.get(seg.unit_id);
  const fig = state.figure.id ? state.paper.figures[state.figure.id] : null;
  setNowPlaying({
    title: plainText(state.paper.title),
    section: section ? plainText([section.number, section.title].filter(Boolean).join(" ")) : "",
    images: fig ? fig.images.map(paperFile) : [],
  });
}

function markReading(el) {
  state.readingEl?.classList.remove("reading");
  el?.classList.add("reading");
  state.readingEl = el;
}

function tick() {
  if (ui.audio.paused) {
    state.ticking = false;
    return;
  }
  syncToAudio();
  requestAnimationFrame(tick);
}

function onPlay() {
  renderPlayState();
  if (shouldFollow() && state.segIndex >= 0) followSegment(state.segIndex, true);
  refreshFigure();
  if (!state.ticking) {
    state.ticking = true;
    requestAnimationFrame(tick);
  }
}

function onMetadata() {
  const duration = mediaDuration();
  ui.seek.max = String(duration);
  const target = state.pendingSeek;
  state.pendingSeek = null;
  if (target > 0 && target < duration - 1) ui.audio.currentTime = target;
  updateTimeUI();
}

// ---------------------------------------------------------------- player controls

function togglePlay() {
  if (!state.run) return;
  if (ui.audio.paused) ui.audio.play().catch((err) => showAudioError(err.message));
  else ui.audio.pause();
}

function seekTo(t) {
  ui.audio.currentTime = Math.min(Math.max(t, 0), mediaDuration());
}

function setRate(rate) {
  ui.audio.defaultPlaybackRate = rate;
  ui.audio.playbackRate = rate;
  store.set("rate", rate);
}

function renderPlayState() {
  const playing = !ui.audio.paused;
  ui.play.classList.toggle("is-playing", playing);
  ui.play.setAttribute("aria-label", playing ? "Pause" : "Play");
}

function updateTimeUI() {
  const duration = mediaDuration();
  const t = state.seekDragging ? Number(ui.seek.value) : ui.audio.currentTime;
  if (!state.seekDragging) ui.seek.value = String(t);
  ui.time.textContent = `${fmt.time(t)} / ${fmt.time(duration)}`;
}

function showAudioError(message) {
  ui.nowReading.hidden = false;
  ui.nowReading.textContent = `Audio problem: ${message}`;
}

function initPlayer() {
  const savedRate = store.get("rate", 1);
  const rate = RATES.includes(savedRate) ? savedRate : 1;
  ui.rate.replaceChildren(...RATES.map((r) => h("option", { value: String(r), text: `${r}×` })));
  ui.rate.value = String(rate);
  setRate(rate);

  ui.play.addEventListener("click", togglePlay);
  ui.back.addEventListener("click", () => seekTo(ui.audio.currentTime - SKIP_SECONDS));
  ui.fwd.addEventListener("click", () => seekTo(ui.audio.currentTime + SKIP_SECONDS));
  ui.rate.addEventListener("change", () => setRate(Number(ui.rate.value)));
  ui.run.addEventListener("change", () => {
    activateRun(state.runs.find((r) => r.run_id === ui.run.value));
    narrationChanged();
  });
  ui.seek.addEventListener("input", () => {
    state.seekDragging = true;
    updateTimeUI();
  });
  ui.seek.addEventListener("change", () => {
    state.seekDragging = false;
    seekTo(Number(ui.seek.value));
  });

  const audio = ui.audio;
  audio.addEventListener("loadedmetadata", onMetadata);
  audio.addEventListener("play", onPlay);
  audio.addEventListener("pause", () => {
    renderPlayState();
    savePosition();
  });
  audio.addEventListener("ended", () => store.set(positionKey(state.run), 0));
  audio.addEventListener("timeupdate", () => syncToAudio());
  audio.addEventListener("seeked", () => syncToAudio(true));
  audio.addEventListener("error", () => showAudioError("could not load the audio file."));
  addEventListener("pagehide", savePosition);
  initMediaSession(audio, seekTo, SKIP_SECONDS);
}

// ---------------------------------------------------------------- user input

function onPaneClick(event) {
  const ref = event.target.closest("a.ref");
  if (ref) {
    onRefClick(event, ref);
    return;
  }
  const chip = event.target.closest(".fig-chip");
  if (chip) {
    pinFigure(chip.dataset.fig);
    return;
  }
  const equation = event.target.closest(".blk-equation.has-reading");
  if (equation && !hasSelection()) {
    toggleReading(equation); // equations show their spoken text; paragraphs seek the narration
    return;
  }
  if (!state.run || event.target.closest("a, button, summary, .about") || hasSelection()) return;
  const unit = event.target.closest(".seekable");
  const i = unit ? state.unitFirstSeg.get(unit.id) : undefined;
  if (i === undefined) return;
  resumeFollow(false);
  seekTo(state.run.segments[i].start);
}

function onRefClick(event, ref) {
  event.preventDefault();
  const id = ref.dataset.ref || decodeURIComponent((ref.getAttribute("href") || "").replace(/^#/, ""));
  if (id in state.paper.figures) {
    pinFigure(id);
    return;
  }
  const target = (id && $(id)) || equationByNumber(id, ref.textContent);
  if (target) scrollToElement(target);
}

/**
 * Equation blocks carry no LaTeXML anchor, but "Eq. (4)" links say "4": match that against eq_number
 * ("4", or "6, 7" for a group). Anything else fails silently.
 */
function equationByNumber(id, label) {
  if (!/(^|\.)E\d/.test(id)) return null;
  const number = label.replace(/[()\s]/g, "");
  const block = state.blocks.find((b) => b.type === "equation" && b.eq_number?.split(/,\s*/).includes(number));
  return block ? $(block.id) : null;
}

function onTocClick(event) {
  const item = event.target.closest(".toc-item");
  if (!item) return;
  event.preventDefault();
  ui.toc.open = false;
  const target = $(item.dataset.target);
  if (target) scrollToElement(target);
}

function onKeyDown(event) {
  if (isTextEntry(event.target) || event.target.closest("dialog")) return;
  if (event.key === "Enter" && event.target.matches?.(".blk-equation.has-reading")) {
    event.preventDefault();
    toggleReading(event.target);
  } else if (event.key === " " && state.run) {
    event.preventDefault();
    togglePlay();
  } else if (NAV_KEYS.has(event.key)) {
    onManualScroll();
  }
}

/** Space is play/pause everywhere outside inputs, so a focused button must not also be clicked by it. */
function onKeyUp(event) {
  if (event.key === " " && state.run && !isTextEntry(event.target) && !event.target.closest("dialog")) event.preventDefault();
}

/** A remembered checkbox: restores its state, then calls onChange now and on every change. */
function bindToggle(input, key, onChange) {
  input.checked = store.get(key, input.checked);
  input.addEventListener("change", () => {
    store.set(key, input.checked);
    onChange(input.checked);
  });
  onChange(input.checked);
}

function initControls() {
  bindToggle(ui.optCites, "cites", (on) => document.body.classList.toggle("show-cites", on));
  bindToggle(ui.optSpeech, "speech", (on) => document.body.classList.toggle("show-speech", on));
  bindToggle(ui.optFollow, "follow", (on) => resumeFollow(on && !ui.audio.paused));
  ui.resume.addEventListener("click", () => resumeFollow(true));

  ui.pane.addEventListener("scroll", onPaneScroll, { passive: true });
  ui.pane.addEventListener("scrollend", endAutoScroll);
  ui.pane.addEventListener("wheel", onManualScroll, { passive: true });
  ui.pane.addEventListener("touchmove", onManualScroll, { passive: true });
  ui.pane.addEventListener("pointerdown", (e) => {
    if (e.target === ui.pane && e.offsetX > ui.pane.clientWidth) onManualScroll(); // scrollbar drag
  });
  ui.pane.addEventListener("click", onPaneClick);
  ui.tocList.addEventListener("click", onTocClick);
  document.addEventListener("click", (e) => {
    if (ui.toc.open && !ui.toc.contains(e.target)) ui.toc.open = false;
  });
  document.addEventListener("keydown", onKeyDown);
  document.addEventListener("keyup", onKeyUp);

  setFigureCollapsed(store.get("figCollapsed", false));
  ui.figCollapse.addEventListener("click", () => setFigureCollapsed(!ui.figPane.classList.contains("collapsed")));
  ui.figMode.addEventListener("click", unpinFigure);
  ui.figPrev.addEventListener("click", () => stepFigure(-1));
  ui.figNext.addEventListener("click", () => stepFigure(1));
  ui.lightbox.addEventListener("click", () => ui.lightbox.close());
}

// ---------------------------------------------------------------- offline

/** Without the server, narration can't be generated: Listen hides; reading and playback carry on. */
function renderConnection() {
  ui.listen.hidden = connection.offline;
}

function loadError(err) {
  if (!connection.offline) return h("p", { class: "error", text: `Could not load this paper: ${err.message}` });
  return h("div", { class: "load-offline" },
    h("p", { class: "error", text: "You are offline, and this paper is not saved on this device." }),
    h("p", {}, "Reconnect to the server to read it, or open a paper saved for offline reading from the ",
      h("a", { href: "/", text: "library" }), "."));
}

// ---------------------------------------------------------------- boot

async function boot() {
  registerServiceWorker();
  initControls();
  initPlayer();
  initListen({
    paperId,
    openButton: ui.listen,
    buttonLabel: $("listen-label"),
    onAudioReady: (runId) => loadRuns(runId),
  });
  initSaveOffline({
    paperId,
    selectedRun: () => state.run,
    runLabel: (runId) => {
      const run = state.runs.find((r) => r.run_id === runId);
      return run ? runLabel(run) : runId;
    },
  });
  renderConnection();
  connection.addEventListener("change", renderConnection);
  try {
    state.paper = await api(paperFile("paper.json"));
  } catch (err) {
    ui.shortTitle.textContent = "Paper unavailable";
    ui.paper.replaceChildren(loadError(err));
    return;
  }
  renderPaper(state.paper);
  indexPaper(state.paper);
  refreshFigure();
  await loadRuns();
}

boot();
