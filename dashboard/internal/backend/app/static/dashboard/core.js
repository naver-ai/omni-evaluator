/* OmniEvaluator Internal Dashboard - Core utilities & tab */

const API = "/api";

const TABS = [
  { id: "submission", el: "content-submission" },
  { id: "leaderboard", el: "content-leaderboard" },
  { id: "inference", el: "content-inference" },
];

const TAB_STORAGE_KEY = "omni-internal-tab";

// Per-tab slide distance in percent = 100 / 3 tabs. Mirrors --tab-step in index.html CSS.
const TAB_STEP = 100 / 3;

const _fetchAborts = new Map();

function abortFetchKey(key) {
  const c = _fetchAborts.get(key);
  if (c) { c.abort(); _fetchAborts.delete(key); }
}

function apiFetch(url, opts = {}) {
  const inferredKey = url.includes("/leaderboard")
    ? "leaderboard"
    : url.includes("/inference")
      ? "inference"
      : "default";
  const { abortKey: _ak, timeout: _to, ...fetchOpts } = opts;
  const abortKey = _ak || inferredKey;
  const timeoutMs = _to !== undefined ? _to : 30000;
  const prev = _fetchAborts.get(abortKey);
  if (prev) prev.abort();
  const controller = new AbortController();
  _fetchAborts.set(abortKey, controller);
  let tid;
  if (timeoutMs > 0) {
    tid = setTimeout(() => controller.abort(), timeoutMs);
  }
  const p = fetch(url, { ...fetchOpts, signal: controller.signal });
  if (tid) p.then(() => clearTimeout(tid), () => clearTimeout(tid));
  return p;
}

async function readJsonSafe(response) {
  try {
    return await response.json();
  } catch (_) {
    return {};
  }
}

async function parseJsonResponse(response) {
  const data = await readJsonSafe(response);
  if (!response.ok) {
    const msg = data.detail || data.message || response.statusText || `HTTP ${response.status}`;
    const err = new Error(msg);
    err.status = response.status;
    err.data = data;
    throw err;
  }
  return data;
}

async function fetchJson(url, opts = {}) {
  const r = await fetch(url, opts);
  return parseJsonResponse(r);
}

async function apiFetchJson(url, opts = {}) {
  const r = await apiFetch(url, opts);
  return parseJsonResponse(r);
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function showGlobalLoading() {
  const el = byId("global-loading-overlay");
  if (el) el.classList.remove("hidden");
}

function hideGlobalLoading() {
  const el = byId("global-loading-overlay");
  if (el) el.classList.add("hidden");
}

function setTab(index, save = true) {
  const track = byId("content-track");
  if (track) track.style.transform = `translateX(-${index * TAB_STEP}%)`;
  try {
    document.documentElement.setAttribute("data-tab", String(index));
  } catch (_) {}
  TABS.forEach((t, j) => {
    const btn = qsa(".tab")[j];
    btn.classList.toggle("tab-active", j === index);
    btn.classList.toggle("text-white", j === index);
    btn.classList.toggle("text-slate-500", j !== index);
    btn.classList.toggle("hover:text-slate-900", j !== index);
  });
  if (save) {
    try { localStorage.setItem(TAB_STORAGE_KEY, String(index)); } catch (_) {}
    try { sessionStorage.setItem(TAB_STORAGE_KEY, String(index)); } catch (_) {}
    const tabId = TABS[index]?.id ?? String(index);
    try {
      if (history && history.replaceState) history.replaceState(null, "", `#tab=${tabId}`);
      else location.hash = `#tab=${tabId}`;
    } catch (_) {}
  }
}

function getSavedTab() {
  try {
    const h = (location.hash || "").replace("#", "");
    if (h.startsWith("tab=")) {
      const v = h.slice(4);
      const idx = TABS.findIndex((t) => t.id === v);
      if (idx >= 0) return idx;
      const n = parseInt(v, 10);
      if (!isNaN(n)) return Math.max(0, Math.min(2, n));
    }
    const n = parseInt(localStorage.getItem(TAB_STORAGE_KEY) || "0", 10);
    return Math.max(0, Math.min(2, isNaN(n) ? 0 : n));
  } catch (_) {
    try {
      const n = parseInt(sessionStorage.getItem(TAB_STORAGE_KEY) || "0", 10);
      return Math.max(0, Math.min(2, isNaN(n) ? 0 : n));
    } catch (_) {
      return 0;
    }
  }
}

