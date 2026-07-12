/* OmniEvaluator Internal Dashboard - Inference module (requires core.js) */

// Inference Viewer
const infState = {
  modelIds: [],
  benchmark: "",
  total: 0,
  idx: 0,
  benchCache: {},
  benchSetCache: {},
  benchQuery: "",
  benchBound: false,
  benchEngineMap: {},
  benchEngineSetCache: {},
  benchModalityMap: {},
  benchModalitySetCache: {},
  benchModalityFilter: null,
  loaded: false,
  sourceBound: false,
};

// Modality grouping: fixed display order + labels. Mirrors the backend
// classify_benchmark_modality() categories ("text"/"image"/"video"/"audio").
const INF_MODALITY_ORDER = ["text", "image", "video", "audio"];
const INF_MODALITY_LABELS = { text: "Text", image: "Image", video: "Video", audio: "Audio" };

// A model is a "debug" run iff its dir name is "debug__…" (matches the backend's is_debug_model).
function infIsDebugModel(o) {
  return String((o && o.model) || "").toLowerCase().startsWith("debug__");
}
// Render a per-model modality breakdown like "Text 14 · Image 5" (nonzero
// buckets only). Returns "" when there is nothing to show.
function infModalityBreakdown(counts) {
  if (!counts) return "";
  const parts = [];
  INF_MODALITY_ORDER.forEach((m) => {
    const n = Number(counts[m]) || 0;
    if (n > 0) parts.push(`${INF_MODALITY_LABELS[m]} ${n}`);
  });
  if (!parts.length) return "";
  return `<span class="inf-model-mods">${escapeHtml(parts.join(" · "))}</span>`;
}

function infBenchModality(bench) {
  const m = infState.benchModalityMap?.[bench];
  return INF_MODALITY_ORDER.includes(m) ? m : "text";
}

function setInfSectionVisible(id, visible) {
  const el = byId(id);
  if (!el) return;
  el.classList.toggle("is-visible", visible);
}

const INF_SOURCE_KEY = "omni-inf-source";

// Source selector (mode): show one source's model list at a time. Selections
// persist across sources because the checkboxes stay in the DOM and the
// authoritative list lives in infState.modelIds.
function setInferenceSource(src) {
  if (!["internal", "direct", "s3"].includes(src)) src = "internal";
  qsa(".inf-source-pill").forEach((p) => p.classList.toggle("active", p.dataset.source === src));
  qsa(".inf-source-panel").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.source !== src));
  try { localStorage.setItem(INF_SOURCE_KEY, src); } catch (_) {}
}

function bindInferenceSourceTabs() {
  if (infState.sourceBound) return;
  infState.sourceBound = true;
  qsa(".inf-source-pill").forEach((p) => {
    p.addEventListener("click", () => setInferenceSource(p.dataset.source));
  });
  let saved = "internal";
  try { saved = localStorage.getItem(INF_SOURCE_KEY) || "internal"; } catch (_) {}
  setInferenceSource(saved);
}

// Per-source pill badges: total models + a green selected-count badge that
// stays visible even when its source panel is hidden.
function updateInfSourceCounts() {
  ["internal", "direct", "s3"].forEach((src) => {
    const container = byId(`inf-${src}-models`);
    const totalEl = byId(`inf-count-${src}`);
    const selEl = byId(`inf-sel-${src}`);
    const cbs = container ? [...qsa(".inf-model-cb", container)] : [];
    const sel = cbs.filter((c) => c.checked).length;
    if (totalEl) totalEl.textContent = String(cbs.length);
    if (selEl) {
      selEl.textContent = String(sel);
      selEl.classList.toggle("hidden", sel === 0);
    }
  });
}

function resetInfSampleView() {
  abortFetchKey("inf-sample");
  infState.total = 0;
  infState.idx = 0;
  const selected = byId("inf-selected");
  if (selected) selected.textContent = "";
  const sampleWrap = byId("inf-sample");
  if (sampleWrap) sampleWrap.classList.add("hidden");
  const navEl = byId("inf-nav");
  if (navEl) navEl.textContent = "/ 0";
  const sampleInput = byId("inf-sample-input");
  if (sampleInput) {
    sampleInput.value = "";
    sampleInput.max = 0;
  }
  const q = byId("inf-question");
  if (q) q.textContent = "";
  const gt = byId("inf-gt");
  if (gt) gt.textContent = "";
  const choicesWrap = byId("inf-choices-wrap");
  if (choicesWrap) choicesWrap.classList.add("hidden");
  const choicesEl = byId("inf-choices");
  if (choicesEl) choicesEl.textContent = "";
  const media = byId("inf-media");
  if (media) { media.innerHTML = ""; media.classList.add("hidden"); }
  _setInfMediaNote("hide");
  const grid = byId("inf-content-grid");
  if (grid) grid.className = "grid grid-cols-1 gap-4";
  const preds = byId("inf-predictions");
  if (preds) preds.innerHTML = "";
}

function resetInfBenchmarkSelect(label = "Select benchmark") {
  const list = byId("inf-bench-list");
  const search = byId("inf-bench-search");
  const selected = byId("inf-bench-selected");
  if (search) {
    search.value = "";
    search.disabled = true;
  }
  if (list) {
    list.innerHTML = `<div class="text-xs text-slate-500">${label}</div>`;
  }
  if (selected) selected.textContent = label;
  setInfBenchLoading(false);
}

function setInfBenchLoading(loading) {
  const el = byId("inf-bench-loading");
  if (el) el.classList.toggle("hidden", !loading);
  const list = byId("inf-bench-list");
  if (list) list.classList.toggle("is-loading", loading);
}

// Combine the search box query with the active modality-chip filter: an item is
// shown iff it matches the search AND its modality chip is active. Group headers
// are hidden when none of their items are currently visible.
function applyInfBenchFilter(listEl) {
  if (!listEl) return;
  const query = (infState.benchQuery || "").toLowerCase().trim();
  const filter = infState.benchModalityFilter; // Set of active modalities, or null = all
  const visibleByMod = {};
  listEl.querySelectorAll(".inf-bench-item").forEach((el) => {
    const mod = el.dataset.modality || "text";
    const matchSearch = !query || (el.dataset.search || "").includes(query);
    const matchFilter = !filter || filter.has(mod);
    const show = matchSearch && matchFilter;
    el.classList.toggle("hidden", !show);
    if (show) visibleByMod[mod] = true;
  });
  listEl.querySelectorAll(".inf-bench-group-header").forEach((h) => {
    const mod = h.dataset.modality || "text";
    h.classList.toggle("hidden", !visibleByMod[mod]);
  });
}

function updateInfChipStyles(chipsWrap) {
  if (!chipsWrap) return;
  const filter = infState.benchModalityFilter;
  chipsWrap.querySelectorAll(".inf-modality-chip").forEach((chip) => {
    const mod = chip.dataset.modality || "";
    const active = mod === "all" ? !filter : !!(filter && filter.has(mod));
    chip.classList.toggle("active", active);
  });
}

function applyInferenceBenchmarks(benchmarks) {
  const listEl = byId("inf-bench-list");
  const searchEl = byId("inf-bench-search");
  const selectedEl = byId("inf-bench-selected");
  if (!listEl || !searchEl) return;
  const list = [...new Set(benchmarks || [])].filter(Boolean);
  list.sort((a, b) => String(a).localeCompare(String(b), undefined, { sensitivity: "base" }));
  setInfBenchLoading(false);
  if (!list.length) {
    listEl.innerHTML = '<div class="text-xs text-slate-500">No benchmarks</div>';
    searchEl.disabled = true;
    infState.benchmark = "";
    if (selectedEl) selectedEl.textContent = "No benchmarks";
    updateInfFlow();
    return;
  }
  const benchLabel = (bench) => {
    const eng = infState.benchEngineMap?.[bench];
    return eng ? `(${eng}) ${bench}` : bench;
  };
  searchEl.disabled = false;
  if (!list.includes(infState.benchmark)) {
    infState.benchmark = "";
  }
  if (selectedEl) {
    selectedEl.textContent = infState.benchmark ? benchLabel(infState.benchmark) : "Select benchmark";
  }

  // Group benchmarks by modality (already sorted within each group via `list`).
  const groups = {};
  INF_MODALITY_ORDER.forEach((m) => { groups[m] = []; });
  list.forEach((x) => { groups[infBenchModality(x)].push(x); });
  const presentMods = INF_MODALITY_ORDER.filter((m) => groups[m].length);

  // Prune the persisted chip filter to modalities that still exist; null = all shown.
  if (infState.benchModalityFilter) {
    const next = new Set([...infState.benchModalityFilter].filter((m) => groups[m]?.length));
    infState.benchModalityFilter = (next.size && next.size < presentMods.length) ? next : null;
  }

  const itemHtml = (x) => {
    const active = infState.benchmark === x;
    const labelText = benchLabel(x);
    const mod = infBenchModality(x);
    return `<label class="lb-bench-item inf-bench-item flex items-center gap-2 cursor-pointer py-1 px-1.5 rounded hover:bg-slate-50${active ? " active" : ""}" data-bench="${escapeHtml(x)}" data-modality="${escapeHtml(mod)}" data-search="${escapeHtml(labelText.toLowerCase())}">
        <input type="radio" name="inf-bench-radio" class="inf-bench-radio rounded border-slate-300" data-bench="${escapeHtml(x)}" ${active ? "checked" : ""}>
        <span class="truncate">${escapeHtml(labelText)}</span>
      </label>`;
  };

  const chipsHtml = `<div class="inf-modality-chips" id="inf-modality-chips">
      <button type="button" class="lb-pill inf-modality-chip text-xs px-2 py-0.5 rounded-full" data-modality="all">All <span class="inf-modality-count">${list.length}</span></button>
      ${presentMods.map((m) => `<button type="button" class="lb-pill inf-modality-chip text-xs px-2 py-0.5 rounded-full" data-modality="${escapeHtml(m)}">${escapeHtml(INF_MODALITY_LABELS[m])} <span class="inf-modality-count">${groups[m].length}</span></button>`).join("")}
    </div>`;

  const groupsHtml = presentMods.map((m) => {
    const header = `<div class="inf-bench-group-header" data-modality="${escapeHtml(m)}">${escapeHtml(INF_MODALITY_LABELS[m])} <span class="inf-bench-group-count">${groups[m].length}</span></div>`;
    return header + groups[m].map(itemHtml).join("");
  }).join("");

  listEl.innerHTML = chipsHtml + groupsHtml;

  const chipsWrap = byId("inf-modality-chips");
  updateInfChipStyles(chipsWrap);
  searchEl.value = infState.benchQuery || "";
  applyInfBenchFilter(listEl);
  searchEl.oninput = () => {
    infState.benchQuery = searchEl.value || "";
    applyInfBenchFilter(listEl);
  };
  if (chipsWrap) {
    chipsWrap.querySelectorAll(".inf-modality-chip").forEach((chip) => {
      chip.onclick = () => {
        const mod = chip.dataset.modality || "all";
        // Single-select (the benchmark itself is single-select, so one modality at a time
        // is the natural mental model): "All" clears the filter; a modality selects just
        // that one; clicking the already-active modality again returns to "All".
        const active = infState.benchModalityFilter;
        if (mod === "all" || (active && active.size === 1 && active.has(mod))) {
          infState.benchModalityFilter = null;
        } else {
          infState.benchModalityFilter = new Set([mod]);
        }
        updateInfChipStyles(chipsWrap);
        applyInfBenchFilter(listEl);
      };
    });
  }
  listEl.querySelectorAll(".inf-bench-radio").forEach((rb) => {
    rb.onchange = () => {
      const next = rb.dataset.bench || "";
      if (infState.benchmark !== next) {
        infState.benchmark = next;
        resetInfSampleView();
      }
      listEl.querySelectorAll(".inf-bench-item").forEach((el) => {
        el.classList.toggle("active", el.dataset.bench === infState.benchmark);
      });
      if (selectedEl) selectedEl.textContent = infState.benchmark ? benchLabel(infState.benchmark) : "Select benchmark";
      updateInfFlow();
      if (infState.benchmark && infState.modelIds.length) {
        loadInferenceSamples();
      }
    };
  });
}

function updateInfFlow() {
  const hasModels = infState.modelIds.length > 0;
  setInfSectionVisible("inf-bench-wrap", hasModels);
  setInfSectionVisible("inf-viewer-wrap", hasModels && !!infState.benchmark);
  const hint = byId("inf-empty-hint");
  if (hint) hint.classList.toggle("hidden", hasModels);
}

function resetInferenceSelection() {
  infState.modelIds = [];
  infState.benchmark = "";
  qsa(".inf-model-cb").forEach((cb) => { cb.checked = false; });
  qsa(".inf-model-item").forEach((label) => label.classList.remove("active"));
  updateInfSourceCounts();
  resetInfSampleView();
  resetInfBenchmarkSelect();
  updateInfFlow();
}

async function loadInferenceModels() {
  try {
    infState.modelIds = [];
    infState.benchmark = "";
    infState.benchCache = {};
    infState.benchSetCache = {};
    infState.benchQuery = "";
    infState.benchEngineMap = {};
    infState.benchEngineSetCache = {};
    infState.benchModalityMap = {};
    infState.benchModalitySetCache = {};
    infState.benchModalityFilter = null;
    resetInfSampleView();
    resetInfBenchmarkSelect();
    updateInfFlow();

    const d = await apiFetchJson(`${API}/inference/models`, { abortKey: "inf-models" });
    // Normal/Debug split mirrors the leaderboard's model-mode (shared global, persisted).
    // Debug runs are model dirs named "debug__…"; Normal hides them, Debug shows only them.
    const mode = (typeof getModelMode === "function") ? getModelMode() : "normal";
    const opts = (d.options || []).filter((o) => (mode === "debug") === infIsDebugModel(o));
    const internal = opts.filter((o) => o.source === "internal");
    const direct = opts.filter((o) => o.source === "direct");
    const s3 = opts.filter((o) => o.source === "s3");

    const render = (el, list) => {
      if (!el) return;
      if (!list.length) {
        el.innerHTML = '<div class="text-xs text-slate-500">No models.</div>';
        return;
      }
      el.innerHTML = list
        .map((o, i) => `<label class="inf-model-item motion-stagger flex items-center gap-2 cursor-pointer text-xs text-slate-500" data-id="${o.id}" style="--i: ${i}"><input type="checkbox" class="inf-model-cb" data-id="${o.id}"> <span class="inf-model-text">${escapeHtml(o.model || "")}</span>${o.bench_count > 0 ? `<span class="inf-model-count">(${o.bench_count})</span>` : ""}${infModalityBreakdown(o.bench_counts)}</label>`)
        .join("");
    };
    render(byId("inf-internal-models"), internal);
    render(byId("inf-direct-models"), direct);
    render(byId("inf-s3-models"), s3);
    bindInferenceSourceTabs();
    updateInfSourceCounts();

    (opts || []).forEach((o) => {
      if (o?.id && Array.isArray(o.benchmarks)) {
        infState.benchCache[o.id] = o.benchmarks;
      }
    });

    const updateModelItemStyles = () => {
      qsa(".inf-model-item").forEach((label) => {
        const id = label.dataset.id || "";
        const active = infState.modelIds.includes(id);
        label.classList.toggle("active", active);
      });
    };

    qsa(".inf-model-cb").forEach((cb) => {
      cb.onchange = () => {
        infState.modelIds = [...qsa(".inf-model-cb:checked")].map((c) => c.dataset.id);
        updateModelItemStyles();
        updateInfSourceCounts();
        infState.benchmark = "";
        infState.benchEngineMap = {};
        infState.benchModalityMap = {};
        resetInfSampleView();
        resetInfBenchmarkSelect();
        updateInfFlow();
        if (!infState.modelIds.length) return;
        const key = [...infState.modelIds].sort().join("|");
        const cachedSet = infState.benchSetCache[key];
        const cachedEng = infState.benchEngineSetCache[key];
        const cachedMod = infState.benchModalitySetCache[key];
        // A modality map is only a valid cache hit if it's non-empty: an empty {} can be
        // left over from a fetch made before the backend served benchmark_modality_map
        // (an empty object is truthy), which would pin every benchmark to "text".
        if (cachedSet && cachedEng && cachedMod && Object.keys(cachedMod).length) {
          infState.benchEngineMap = cachedEng;
          infState.benchModalityMap = cachedMod;
          applyInferenceBenchmarks(cachedSet);
          return;
        }
        let missing = false;
        const merged = [];
        for (const id of infState.modelIds) {
          const list = infState.benchCache[id];
          if (!Array.isArray(list) || list.length === 0) {
            missing = true;
            break;
          }
          merged.push(...list);
        }
        if (!missing) {
          const uniq = [...new Set(merged)];
          if (uniq.length && cachedEng && cachedMod && Object.keys(cachedMod).length) {
            infState.benchSetCache[key] = uniq;
            infState.benchEngineMap = cachedEng;
            infState.benchModalityMap = cachedMod;
            applyInferenceBenchmarks(uniq);
            return;
          }
        }
        loadInferenceBenchmarks();
      };
    });

    const resetBtn = byId("inf-reset");
    if (resetBtn) resetBtn.onclick = resetInferenceSelection;
    bindInfBenchmarkPanel();
    infState.loaded = true;
  } catch (e) {
    if (e.name !== "AbortError") console.error(e);
  }
}

function bindInfBenchmarkPanel() {
  if (infState.benchBound) return;
  const benchPanelWrap = byId("inf-bench-panel-wrap");
  if (!benchPanelWrap) return;
  infState.benchBound = true;
  benchPanelWrap.classList.add("open");
}

async function loadInferenceBenchmarks() {
  if (!infState.modelIds.length) return;
  const list = byId("inf-bench-list");
  if (!list) return;
  resetInfBenchmarkSelect("Select benchmark");
  setInfBenchLoading(true);
  const key = [...infState.modelIds].sort().join("|");
  try {
    const d = await apiFetchJson(`${API}/inference/benchmarks?model_ids=${infState.modelIds.join(",")}`, { abortKey: "inf-bench" });
    const benchmarks = d.benchmarks || [];
    if ([...infState.modelIds].sort().join("|") !== key) return;
    infState.benchEngineMap = d.benchmark_engine_map || {};
    infState.benchModalityMap = d.benchmark_modality_map || {};
    infState.benchSetCache[key] = benchmarks;
    infState.benchEngineSetCache[key] = infState.benchEngineMap;
    infState.benchModalitySetCache[key] = infState.benchModalityMap;
    applyInferenceBenchmarks(benchmarks);
  } catch (e) {
    if (e.name === "AbortError") return;
    setInfBenchLoading(false);
    if (list) {
      list.innerHTML = '<div class="text-xs text-slate-500">Failed to load</div>';
    }
  }
}

async function loadInferenceSamples() {
  infState.modelIds = [...qsa(".inf-model-cb:checked")].map((c) => c.dataset.id);

  if (!infState.modelIds.length || !infState.benchmark) {
    alert(MSG.selectModelBenchmark);
    return;
  }

  resetInfSampleView();
  updateInfFlow();
  const statusEl = byId("inf-selected");
  if (statusEl) statusEl.textContent = "Loading samples...";
  try {
    const d = await apiFetchJson(`${API}/inference/sample/0?model_ids=${infState.modelIds.join(",")}&benchmark=${encodeURIComponent(infState.benchmark)}`, { abortKey: "inf-sample" });
    infState.total = d.total_samples || 0;
    infState.idx = 0;
    // Sync the jump input's max as soon as the total is known.
    const sampleInput = byId("inf-sample-input");
    if (sampleInput) sampleInput.max = infState.total;
    if (statusEl) {
      if (infState.total > 0) {
        statusEl.textContent = `Models: ${infState.modelIds.length}, Samples: ${infState.total}`;
      } else {
        statusEl.textContent = "No samples found for this benchmark.";
      }
    }

    if (infState.total > 0) {
      byId("inf-sample").classList.remove("hidden");
      showInferenceSample();
    } else {
      byId("inf-sample").classList.add("hidden");
    }
  } catch (e) {
    if (e.name !== "AbortError") alert("Error: " + e.message);
  }
}

function sourceBadgeClass(src) {
  return SOURCE_BADGE_CLASSES[src] || SOURCE_BADGE_CLASSES.s3;
}

// Sync the position indicators (counter, jump input, slider) to infState.idx
// WITHOUT fetching — cheap, so slider drags get instant position feedback.
function syncInfNav() {
  const navEl = byId("inf-nav");
  const sampleInput = byId("inf-sample-input");
  const slider = byId("inf-sample-slider");
  if (navEl) navEl.textContent = `${infState.idx + 1} / ${infState.total}`;
  if (sampleInput) {
    sampleInput.value = infState.total > 0 ? String(infState.idx + 1) : "";
    sampleInput.max = infState.total;
  }
  if (slider) {
    slider.max = String(Math.max(1, infState.total));
    slider.value = String(infState.idx + 1);
    slider.disabled = infState.total <= 1;
  }
}

async function showInferenceSample() {
  syncInfNav();
  const loadingEl = byId("inf-loading-overlay");
  let _spinTimer = null;
  if (loadingEl) _spinTimer = setTimeout(() => loadingEl.classList.remove("hidden"), 180);
  try {
    const d = await apiFetchJson(`${API}/inference/sample/${infState.idx}?model_ids=${infState.modelIds.join(",")}&benchmark=${encodeURIComponent(infState.benchmark)}`, { abortKey: "inf-sample" });
    byId("inf-question").textContent = formatQuestionText(d.question) || "-";
    byId("inf-gt").textContent = (d.ground_truth ?? "") !== "" ? String(d.ground_truth) : "-";
    const choicesWrap = byId("inf-choices-wrap");
    const choicesEl = byId("inf-choices");
    if (choicesEl && choicesWrap) {
      const choicesText = d.choices ? String(d.choices).trim() : "";
      choicesEl.textContent = choicesText || "-";
      choicesWrap.classList.toggle("hidden", !choicesText);
    }
    renderInfMedia(d.media || {});
    renderInfMediaNote(d.media || {});
    byId("inf-predictions").innerHTML = (d.predictions || [])
      .map((p) => {
        const scores = p.scores && typeof p.scores === "object" ? Object.entries(p.scores) : [];
        const badges = scores.length ? scores.map(([k, v]) => {
          const cls = metricBadgeClass(k, v);
          return `<span class="inf-metric-badge ${cls}">${escapeHtml(k)}: ${escapeHtml(String(v))}</span>`;
        }).join("") : "";
        return `<div class="p-2 rounded border border-slate-200 bg-white">
          <div class="flex items-center gap-2 flex-wrap mb-1">
            <span class="text-xs px-1.5 py-0.5 rounded border ${sourceBadgeClass(p.source)}">${escapeHtml(p.source)}</span>
            <span class="text-xs text-slate-500">${escapeHtml(p.model)}</span>
          </div>
          <p class="text-sm mt-1 whitespace-pre-wrap text-slate-700">${escapeHtml(p.prediction || "-")}</p>
          ${badges ? `<div class="mt-1.5 flex flex-wrap gap-1">${badges}</div>` : ""}
        </div>`;
      })
      .join("");
  } catch (e) {
    if (e.name !== "AbortError") console.error(e);
  } finally {
    if (_spinTimer) clearTimeout(_spinTimer);
    if (loadingEl) loadingEl.classList.add("hidden");
  }
}

const MEDIA_EXTS = new Set([
  "jpg", "jpeg", "png", "gif", "webp", "bmp",
  "mp4", "mov", "webm", "mkv",
  "mp3", "wav", "flac", "ogg", "m4a",
]);

function metricBadgeClass(key, val) {
  if ((key || "").toString().toLowerCase() === "index") return "";
  if (val == null || val === "-" || val === "") return "";
  const n = parseFloat(String(val));
  if (!Number.isFinite(n) || n < 0 || n > 1) return "";
  const eps = 1e-9;
  if (Math.abs(n - 1) < eps) return "inf-metric-one";
  if (Math.abs(n) < eps) return "inf-metric-zero";
  if (n <= 0.5 + eps) return "inf-metric-low";
  if (n < 1 - eps) return "inf-metric-mid";
  return "";
}

function formatQuestionText(q) {
  if (typeof q !== "string") return "";
  const s = q.trim();
  if (!s) return "";
  const lines = s.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  if (lines.length <= 1) {
    return formatQuestionLine(s);
  }
  const cleaned = lines.filter((line) => !isMediaLine(line));
  if (!cleaned.length) {
    return formatQuestionLine(s);
  }
  return cleaned.join("\n");
}

function formatQuestionLine(s) {
  const lower = s.toLowerCase();
  const extMatch = lower.match(/\.([a-z0-9]+)(\?.*)?$/);
  const ext = extMatch ? extMatch[1] : "";
  const looksLikePath =
    s.includes("/") ||
    s.includes("\\") ||
    lower.startsWith("http://") ||
    lower.startsWith("https://") ||
    lower.startsWith("s3://") ||
    lower.startsWith("file://");
  if (looksLikePath && MEDIA_EXTS.has(ext)) {
    return basenameFromPath(s);
  }
  return s;
}

function isMediaLine(line) {
  const lower = line.toLowerCase();
  const extMatch = lower.match(/\.([a-z0-9]+)(\?.*)?$/);
  const ext = extMatch ? extMatch[1] : "";
  if (!MEDIA_EXTS.has(ext)) return false;
  if (!/\s/.test(line)) return true;
  return (
    lower.startsWith("http://") ||
    lower.startsWith("https://") ||
    lower.startsWith("s3://") ||
    lower.startsWith("file://") ||
    lower.startsWith("/") ||
    line.includes("\\")
  );
}

function basenameFromPath(path) {
  const clean = path.split("?")[0].split("#")[0];
  const parts = clean.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || clean;
}

function renderInfMedia(media) {
  const wrap = byId("inf-media");
  const grid = byId("inf-content-grid");
  if (!wrap) return;
  const uniq = (arr) => [...new Set((Array.isArray(arr) ? arr : []).filter(Boolean))];
  const images = uniq(media?.images);
  const videos = uniq(media?.videos);
  const audios = uniq(media?.audios);
  const items = [
    { kind: "image", label: "Image", values: images },
    { kind: "video", label: "Video", values: videos },
    { kind: "audio", label: "Audio", values: audios },
  ];
  const hasAny = items.some((i) => i.values.length);
  // Toggle media column visibility and grid layout
  wrap.classList.toggle("hidden", !hasAny);
  if (grid) {
    grid.className = hasAny
      ? "grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-4"
      : "grid grid-cols-1 gap-4";
  }
  if (!hasAny) {
    wrap.innerHTML = "";
    return;
  }
  wrap.innerHTML = items
    .filter((i) => i.values.length)
    .map((i) => {
      const count = i.values.length;
      const label = count > 1 ? `${i.label} (${count})` : i.label;
      const bodies = i.values.map((v) => {
        const sv = String(v);
        // Backend emits this sentinel when embedded media exceeds the inline cap (memory mgmt).
        if (sv.startsWith("omni://oversized")) return _infMediaWarn(_oversizedReason(sv));
        const src = resolveMediaUrl(v, i.kind);
        return src
          ? renderMediaElement(i.kind, src)
          : _infMediaWarn(_unresolvableReason(sv));
      }).join("");
      return `
        <div class="inf-media-card">
          <div class="inf-media-title">${label}</div>
          <div class="inf-media-body space-y-2">${bodies}</div>
        </div>
      `;
    })
    .join("");
}

function resolveMediaUrl(val, kind) {
  if (!val) return "";
  const s = String(val);
  if (s.startsWith("http://") || s.startsWith("https://")) return s;
  if (s.startsWith("data:")) return s;
  if (s.startsWith("s3://")) return "";
  if (s.startsWith("/")) {
    const enc = btoa(String.fromCharCode(...new TextEncoder().encode(s)))
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");
    if (kind === "image") return `/api/local-image?path=${enc}`;
    return `/api/local-media?path=${enc}`;
  }
  return s;
}

function _infMediaWarn(msg) {
  return `<div class="inf-media-warning">⚠ ${escapeHtml(msg)}</div>`;
}

function _oversizedReason(sentinel) {
  // omni://oversized?kind=image&bytes=12345678
  const m = /bytes=(\d+)/.exec(sentinel);
  const mb = m ? (parseInt(m[1], 10) / (1024 * 1024)).toFixed(1) : "?";
  return `Media too large to display (${mb} MB — exceeds the inline memory-safety cap)`;
}

function _unresolvableReason(v) {
  if (v.startsWith("s3://")) return "S3 media cannot be previewed";
  return "Media unavailable (referenced by path only)";
}

// Single explanatory banner above the content grid for ALL media-absence reasons:
//  - "info": the benchmark is media-based but this sample has no media to show;
//  - "warn": media was referenced but failed to load (404 / 413 / 403 / network).
// Unifies what used to be a silent empty column + a buried per-image warning.
function _setInfMediaNote(kind, msg) {
  const note = byId("inf-media-note");
  if (!note) return;
  if (kind === "hide") {
    note.className = "inf-media-note hidden";
    note.innerHTML = "";
    return;
  }
  note.className = `inf-media-note ${kind === "warn" ? "is-warn" : "is-info"}`;
  const icon = kind === "warn" ? "⚠" : "ℹ";  // monochrome glyphs to match the SVG/text icon set
  note.innerHTML = `<span class="inf-media-note-icon">${icon}</span><span>${escapeHtml(msg)}</span>`;
}

// Show the info banner when an image/video/audio benchmark has NO media for this sample
// (the most common cause is that the eval engine didn't store the media bytes).
function renderInfMediaNote(media) {
  const mod = infBenchModality(infState.benchmark);
  if (!["image", "video", "audio"].includes(mod)) { _setInfMediaNote("hide"); return; }
  const uniq = (a) => [...new Set((Array.isArray(a) ? a : []).filter(Boolean))];
  const present = uniq(media?.images).length + uniq(media?.videos).length + uniq(media?.audios).length;
  if (present) { _setInfMediaNote("hide"); return; }  // media exists; load errors go to "warn" via _infMediaErr
  const label = INF_MODALITY_LABELS[mod] || "media";
  const low = label.toLowerCase();
  const eng = infState.benchEngineMap?.[infState.benchmark] || "";
  // Case A — NO media is referenced by this sample (the record carries no path/URL), so there
  // is simply nothing to load. This is distinct from Case B (media referenced but absent from
  // object storage), which _infMediaErr surfaces as a "warn" banner. Keep the two worded apart.
  const msg = eng === "lmms_eval"
    ? `No ${low} is referenced for this sample — the lmms_eval engine doesn't store ${low} paths in its output, so there's nothing to load. Open the same dataset under a built-in / vlm_eval_kit model to view the ${low}.`
    : `No ${low} is referenced for this sample — the record contains no ${low} path, so there's nothing to load.`;
  _setInfMediaNote("info", msg);
}

function _infMediaErr(el) {
  // The <img>/<video>/<audio> failed to load. Don't show a separate inline marker — the reason
  // is surfaced once, in the unified banner above the Question (avoids a duplicate message).
  // Drop the broken element and collapse the media column when nothing renders in it anymore.
  const src = el.getAttribute("src") || "";
  // Keep the enclosing media card's count honest: a failed item shouldn't leave the
  // title reading "Image (2)" when only one image actually renders.
  const card = el.closest ? el.closest(".inf-media-card") : null;
  el.remove();
  if (card) {
    if (!card.querySelector("img,video,audio,.inf-media-warning")) {
      card.remove();
    } else {
      const titleEl = card.querySelector(".inf-media-title");
      const n = card.querySelectorAll("img,video,audio").length;
      if (titleEl) {
        const base = titleEl.textContent.replace(/\s*\(\d+\)\s*$/, "");
        titleEl.textContent = n > 1 ? `${base} (${n})` : base;
      }
    }
  }
  const wrap = byId("inf-media");
  const grid = byId("inf-content-grid");
  const stillHasMedia = !!(wrap && wrap.querySelector("img,video,audio"));
  if (wrap && !stillHasMedia) {
    wrap.innerHTML = "";
    wrap.classList.add("hidden");
    if (grid) grid.className = "grid grid-cols-1 gap-4";
  }
  // Only surface the failure banner when NOTHING renders. If a sibling media still shows
  // (e.g. one model's image loaded while another model's S3 copy 404'd), a global
  // "couldn't load" banner is misleading — drop the broken one silently instead.
  if (stillHasMedia) return;
  // Case B — media WAS referenced but failed to load. Name the storage backend so the banner
  // says exactly why it's missing, instead of a vague "unavailable". This is the counterpart to
  // Case A (no media referenced) above; the two must read as clearly different situations.
  const isLocal = src.indexOf("/api/local-") === 0;
  if (!isLocal) {
    // Object-storage media (a presigned https URL). An onerror here means the object isn't in
    // the bucket — the dataset's media hasn't been uploaded to object storage yet (404). The URL
    // is cross-origin, so we can't fetch-probe it; the load failure itself is the signal.
    _setInfMediaNote("warn", "Media is referenced, but it isn't in object storage (S3) yet — the file hasn't been uploaded to the bucket, so it can't be loaded (404).");
    return;
  }
  _setInfMediaNote("warn", "Media is referenced but couldn't be loaded — it may be missing or temporarily unavailable.");
  fetch(src).then((r) => {
    let reason = "";
    if (r.status === 404) reason = "Media is referenced, but the file isn't on the server (404).";
    else if (r.status === 413) reason = "Media is referenced, but it's too large to preview (over 50MB, 413).";
    else if (r.status === 403) reason = "Media is referenced, but the path isn't allowed (403).";
    else if (!r.ok) reason = `Media is referenced, but it failed to load (HTTP ${r.status}).`;
    // Re-check: another media may have finished loading while this probe was in flight.
    const w = byId("inf-media");
    if (reason && !(w && w.querySelector("img,video,audio"))) _setInfMediaNote("warn", reason);
  }).catch(() => {});
}

// Re-filter the model list when the (shared) Normal/Debug mode changes. setModelMode emits
// this from either panel's pills; we rebuild only if the viewer has loaded at least once.
// loadInferenceModels resets the current model/benchmark selection, which is correct here:
// the normal and debug model sets are disjoint, so the prior selection no longer applies.
if (typeof _store !== "undefined" && _store && typeof _store.on === "function") {
  _store.on("model-mode-change", () => {
    if (infState.loaded) loadInferenceModels();
  });
}

function renderMediaElement(kind, src) {
  if (kind === "image") {
    return `<img class="inf-media-img" src="${escapeHtml(src)}" alt="image" style="display:none" onload="this.style.display=''" onerror="_infMediaErr(this)">`;
  }
  if (kind === "video") {
    return `<video class="inf-media-video" src="${escapeHtml(src)}" controls style="display:none" onloadedmetadata="this.style.display=''" onerror="_infMediaErr(this)"></video>`;
  }
  if (kind === "audio") {
    return `<audio class="inf-media-audio" src="${escapeHtml(src)}" controls style="display:none" onloadedmetadata="this.style.display=''" onerror="_infMediaErr(this)"></audio>`;
  }
  return "";
}
