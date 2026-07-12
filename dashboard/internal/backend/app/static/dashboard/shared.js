/* OmniEvaluator Internal Dashboard - Shared globals */

/* Source identity palette — single source of truth for the JS layer. Mirrors the
   --source-internal / --source-direct / --source-s3 tokens in dashboard.css
   (violet / cyan / amber). Keep the two in sync if the source hues change. */
const SOURCE_STYLES = { internal: "bg-violet-500", direct: "bg-cyan-500", s3: "bg-amber-500" };
const SOURCE_BADGE_CLASSES = {
  internal: "bg-violet-100 text-violet-600 border-violet-200",
  direct: "bg-cyan-100 text-cyan-600 border-cyan-200",
  s3: "bg-amber-100 text-amber-600 border-amber-200",
};
const SOURCE_EL_IDS = { internal: "internal-models", direct: "direct-models", s3: "s3-models" };
const MSG = {
  requestFailed: "Request failed",
  uploadFailed: "Upload failed",
  downloadFailed: "Download failed",
  unknownError: "Unknown error",
  selectModelBenchmark: "Select at least one model and a benchmark",
};

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function byId(id) {
  return document.getElementById(id);
}

function qs(sel, root = document) {
  return root.querySelector(sel);
}

function qsa(sel, root = document) {
  return root.querySelectorAll(sel);
}

function applyListSearch(listEl, itemSelector, q) {
  if (!listEl) return;
  const query = (q || "").toLowerCase().trim();
  listEl.querySelectorAll(itemSelector).forEach((el) => {
    const match = !query || (el.dataset.search || "").includes(query);
    el.classList.toggle("hidden", !match);
  });
}

const MODEL_MODE_STORAGE_KEY = "omni-model-mode";

function normalizeModelMode(mode) {
  return mode === "debug" ? "debug" : "normal";
}

function getModelMode() {
  try {
    const v = (localStorage.getItem(MODEL_MODE_STORAGE_KEY) || "").toLowerCase();
    if (v) return normalizeModelMode(v);
  } catch (_) {}
  try {
    const v = (sessionStorage.getItem(MODEL_MODE_STORAGE_KEY) || "").toLowerCase();
    if (v) return normalizeModelMode(v);
  } catch (_) {}
  return "normal";
}

function updateModelModeUI(mode) {
  const m = normalizeModelMode(mode);
  qsa(".lb-model-mode-pill").forEach((p) => {
    const val = p.dataset.mode || "normal";
    const active = val === m;
    p.classList.toggle("active", active);
    const label = val === "debug" ? "Debug" : "Normal";
    p.textContent = active ? "✓ " + label : label;
  });
}

function setModelMode(mode, opts = {}) {
  const m = normalizeModelMode((mode || "").toLowerCase());
  try { localStorage.setItem(MODEL_MODE_STORAGE_KEY, m); } catch (_) {}
  try { sessionStorage.setItem(MODEL_MODE_STORAGE_KEY, m); } catch (_) {}
  updateModelModeUI(m);
  if (!opts.silent) {
    _store.emit("model-mode-change", { mode: m });
  }
}

