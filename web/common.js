// Shared helpers for the index and reader pages: API calls, connectivity, job polling, formatting, storage.

const JOB_POLL_MS = 700;
const STORE_PREFIX = "paper-reader:";
const OFFLINE_HEADER = "X-Offline"; // set by the service worker on what it answers without the server (sw.js)

/**
 * Whether the server is out of reach: the browser is offline, an API call failed, or the service worker
 * answered without the server. Fires "change" when that flips.
 */
class Connection extends EventTarget {
  offline = !navigator.onLine;

  set(offline) {
    if (offline === this.offline) return;
    this.offline = offline;
    this.dispatchEvent(new Event("change"));
  }
}

export const connection = new Connection();
addEventListener("online", () => connection.set(false));
addEventListener("offline", () => connection.set(true));

export class ApiError extends Error {
  constructor(status, detail) {
    super(describeDetail(detail) || `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

/** FastAPI errors come as {detail: str | {message,...} | [{loc, msg}]}. */
function describeDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => (d.loc ? `${d.loc.slice(1).join(".")}: ${d.msg}` : d.msg ?? String(d))).join("; ");
  }
  return detail.message ?? JSON.stringify(detail);
}

/**
 * fetch + JSON. Throws ApiError (with .status and .detail) on non-2xx responses.
 * A FormData body is sent as multipart/form-data (the browser sets the boundary); anything else as JSON.
 * Also keeps `connection` up to date.
 */
export async function api(path, { method = "GET", body } = {}) {
  const init = { method, headers: { Accept: "application/json" } };
  if (body instanceof FormData) {
    init.body = body;
  } else if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const isApi = path.startsWith("/api/");
  const res = await fetch(path, init).catch((err) => {
    if (isApi) connection.set(true);
    throw err;
  });
  // A response the service worker made up means offline; only an API response proves the server is up
  // (other files may come from a saved copy).
  if (res.headers.has(OFFLINE_HEADER)) connection.set(true);
  else if (isApi) connection.set(false);
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, data?.detail ?? data ?? res.statusText);
  return data;
}

/** Polls GET /api/jobs/{id} until done (resolves with the job) or error (rejects). */
export async function pollJob(jobId, onUpdate = () => {}) {
  for (;;) {
    const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
    onUpdate(job);
    if (job.status === "done") return job;
    if (job.status === "error") throw new Error(job.error || job.message || "The job failed");
    await new Promise((resolve) => setTimeout(resolve, JOB_POLL_MS));
  }
}

/** Shows a job's progress in a .progress block (<progress> + .msg). */
export function showProgress(container, job) {
  container.hidden = false;
  container.querySelector("progress").value = job.progress ?? 0;
  const pct = Math.round((job.progress ?? 0) * 100);
  container.querySelector(".msg").textContent = `${job.message || job.status} · ${pct}%`;
}

export const fmt = {
  usd(x) {
    if (!x) return "$0.00";
    return x < 0.01 ? `$${x.toFixed(3)}` : `$${x.toFixed(2)}`;
  },
  int(n) {
    return Math.round(n ?? 0).toLocaleString("en-US");
  },
  minutes(m) {
    if (m == null) return "–";
    return m < 10 ? `${m.toFixed(1)} min` : `${Math.round(m)} min`;
  },
  /** File sizes: "830 KB", "2.4 MB", "48 MB". */
  bytes(n) {
    const mb = (n ?? 0) / 2 ** 20;
    if (mb >= 10) return `${Math.round(mb)} MB`;
    if (mb >= 1) return `${mb.toFixed(1)} MB`;
    return `${Math.max(1, Math.round((n ?? 0) / 1024))} KB`;
  },
  /** Rough durations with sensible precision: "<1 s", "~4 s", "~35 s", "~1 min 20 s", "~12 min", "~1 h 5 min". */
  eta(seconds) {
    const s = Math.max(0, seconds || 0);
    if (s < 1) return "<1 s";
    if (s < 10) return `~${Math.round(s)} s`;
    if (s < 57.5) return `~${Math.round(s / 5) * 5} s`;
    if (s < 595) {
      const r = Math.round(s / 10) * 10;
      return r % 60 ? `~${Math.floor(r / 60)} min ${r % 60} s` : `~${r / 60} min`;
    }
    const m = Math.round(s / 60);
    if (m < 60) return `~${m} min`;
    return m % 60 ? `~${Math.floor(m / 60)} h ${m % 60} min` : `~${m / 60} h`;
  },
  time(seconds) {
    const s = Math.max(0, Math.floor(seconds || 0));
    const h = Math.floor(s / 3600);
    const mm = String(Math.floor((s % 3600) / 60));
    const ss = String(s % 60).padStart(2, "0");
    return h ? `${h}:${mm.padStart(2, "0")}:${ss}` : `${mm}:${ss}`;
  },
};

/** Rounds a price up to whole cents: what the UI shows is what the user accepts. */
export function ceilCents(usd) {
  return Math.ceil(Math.round(usd * 1e6) / 1e4) / 100;
}

/** localStorage with JSON values; never throws (private mode, blocked storage). */
export const store = {
  get(key, fallback = null) {
    try {
      const raw = localStorage.getItem(STORE_PREFIX + key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch {
      return fallback;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(STORE_PREFIX + key, JSON.stringify(value));
    } catch {
      /* storage unavailable: preferences just don't persist */
    }
  },
};

/** document.createElement with props/attributes and children (strings become text nodes). */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key in el && typeof value !== "string") el[key] = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  el.append(...children.flat().filter((c) => c != null && c !== false));
  return el;
}
