/* OmniEvaluator Internal Dashboard - Entry (requires core.js + modules) */

function bindSubmissionEvents() {
  const downloadBtn = byId("download-mymodel");
  if (downloadBtn) {
    downloadBtn.addEventListener("click", async () => {
      try {
        const r = await fetch(`${API}/submission/example/download`);
        if (!r.ok) throw new Error(MSG.downloadFailed);
        const blob = await r.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "MyModel.zip";
        a.click();
        URL.revokeObjectURL(a.href);
      } catch (e) {
        alert(MSG.downloadFailed + ": " + (e.message || MSG.unknownError));
      }
    });
  }

  byId("model-search").oninput = debounce(loadModels, 300);

  byId("content-submission").addEventListener("click", async (e) => {
    const b = e.target.closest(".del-btn");
    if (!b) return;
    e.preventDefault();
    // Inline two-step confirm: first click → "Sure?", second click → delete
    if (!b.dataset.confirming) {
      b.dataset.confirming = "1";
      b.textContent = "Sure?";
      b.classList.add("!bg-red-600", "!text-white");
      setTimeout(() => {
        if (b.dataset.confirming) {
          delete b.dataset.confirming;
          b.textContent = "Delete";
          b.classList.remove("!bg-red-600", "!text-white");
        }
      }, 2500);
      return;
    }
    delete b.dataset.confirming;
    const m = b.dataset.m;
    const src = b.dataset.src || "direct";
    try {
      const r = await fetch(`${API}/submission/model/${encodeURIComponent(src)}/${encodeURIComponent(m)}`, { method: "DELETE" });
      if (r.ok) {
        showGlobalLoading();
        try {
          await loadModels();
          await loadLeaderboard(true);
        } finally {
          hideGlobalLoading();
        }
      } else {
        await loadModels();
      }
    } catch (_) {
      loadModels();
    }
  });

  byId("upload-zone").onclick = () => byId("file-input").click();
  byId("upload-zone").ondragover = (e) => {
    e.preventDefault();
    byId("upload-zone").classList.add("border-cyan-500", "bg-cyan-50");
  };
  byId("upload-zone").ondragleave = () => {
    byId("upload-zone").classList.remove("border-cyan-500", "bg-cyan-50");
  };
  byId("upload-zone").ondrop = (e) => {
    e.preventDefault();
    byId("upload-zone").classList.remove("border-cyan-500", "bg-cyan-50");
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith(".zip")) uploadFile(f);
  };
  byId("file-input").onchange = (e) => {
    const f = e.target.files[0];
    if (f) uploadFile(f);
  };
}

const SUB_SOURCE_KEY = "omni-sub-source";

function setSubmissionSource(src) {
  if (!["internal", "direct", "s3"].includes(src)) src = "internal";
  qsa(".sub-source-pill").forEach((p) => p.classList.toggle("active", p.dataset.source === src));
  qsa(".sub-panel").forEach((panel) => panel.classList.toggle("hidden", panel.dataset.source !== src));
  try { localStorage.setItem(SUB_SOURCE_KEY, src); } catch (_) {}
}

function bindSubmissionSourceTabs() {
  qsa(".sub-source-pill").forEach((p) => {
    p.addEventListener("click", () => setSubmissionSource(p.dataset.source));
  });
  let saved = "internal";
  try { saved = localStorage.getItem(SUB_SOURCE_KEY) || "internal"; } catch (_) {}
  setSubmissionSource(saved);
}

function bindTabEvents() {
  byId("tab-submission").onclick = () => {
    setTab(0);
    stopLbAutoRefresh();
    loadModels();
  };
  byId("tab-leaderboard").onclick = () => {
    setTab(1);
    loadLeaderboard(true);
    startLbAutoRefresh(false);
  };
  byId("tab-inference").onclick = () => {
    setTab(2);
    stopLbAutoRefresh();
    if (!infState.loaded) loadInferenceModels();
  };
}

function bindLeaderboardEvents() {
  byId("lb-refresh").onclick = () => { resetLbFilters(); loadLeaderboard(); };
  const modePills = qsa(".lb-model-mode-pill");
  if (modePills.length) {
    setModelMode("normal", { silent: true });
    modePills.forEach((p) => {
      p.onclick = () => {
        const mode = p.dataset.mode || "normal";
        setModelMode(mode);
      };
    });
  }
  _store.on("model-mode-change", () => {
    lbState.page = 0;
    loadLeaderboard(true);
  });
  qsa(".lb-source-pill").forEach((p) => {
    p.onclick = () => {
      const src = p.dataset.source;
      const active = p.classList.toggle("active");
      const label = src.charAt(0).toUpperCase() + src.slice(1);
      p.textContent = active ? "✓ " + label : label;
      lbState.sources = [...qsa(".lb-source-pill.active")].map((b) => b.dataset.source);
      loadLeaderboard();
    };
  });
  qsa(".lb-modality-pill").forEach((p) => {
    p.onclick = () => {
      const mod = p.dataset.modality || "";
      lbState.modalityFilter = togglePillFilter(lbState.modalityFilter, mod, 3);
      applyPillGroup(".lb-modality-pill", lbState.modalityFilter, 3, "modality", (m) => (m ? m.charAt(0).toUpperCase() + m.slice(1) : "All"));
      loadLeaderboard();
    };
  });
  const VALUE_PRIMARY_SUBS = ["inference", "evaluation", "total"];
  const VALUE_SECONDARY_SUBS = ["latency", "throughput", "benchmarks"];
  qsa(".lb-inference-pill").forEach((p) => {
    p.onclick = () => {
      const inf = p.dataset.inference || "";
      lbState.inferenceEngineFilter = togglePillFilter(lbState.inferenceEngineFilter, inf, INFERENCE_OPTS.length);
      applyPillGroup(".lb-inference-pill", lbState.inferenceEngineFilter, INFERENCE_OPTS.length, "inference", (i) => i || "All");
      loadLeaderboard();
    };
  });
  qsa(".lb-evaluation-pill").forEach((p) => {
    p.onclick = () => {
      const ev = p.dataset.evaluation || "";
      lbState.evaluationEngineFilter = togglePillFilter(lbState.evaluationEngineFilter, ev, EVALUATION_OPTS.length);
      applyPillGroup(".lb-evaluation-pill", lbState.evaluationEngineFilter, EVALUATION_OPTS.length, "evaluation", (e) => e || "All");
      loadLeaderboard();
    };
  });
  qsa(".lb-value-pill").forEach((p) => {
    p.onclick = () => {
      const val = p.dataset.value || "score";
      if (lbState.valueType === val) return;
      lbState.valueType = val;
      const defaultPrimary = val === "time" ? "" : val === "coverage" ? "evaluation" : "";
      const defaultSecondary = val === "time" ? "benchmarks" : "";
      lbState.valueSubtypePrimary = defaultPrimary;
      lbState.valueSubtypeSecondary = defaultSecondary;
      qsa(".lb-value-pill").forEach((q) => {
        const isActive = (q.dataset.value || "score") === val;
        q.classList.toggle("active", isActive);
        const label = q.dataset.value === "time" ? "Time" : q.dataset.value === "coverage" ? "Coverage" : "Score";
        q.textContent = isActive ? "✓ " + label : label;
      });
      const subWrap = byId("lb-value-sub-wrap");
      const totalBtn = qs(".lb-value-sub-total");
      const latencyBtn = qs('.lb-value-sub-pill[data-subtype="latency"]');
      const throughputBtn = qs('.lb-value-sub-pill[data-subtype="throughput"]');
      const benchmarksBtn = qs('.lb-value-sub-pill[data-subtype="benchmarks"]');
      const evaluationBtn = qs('.lb-value-sub-pill[data-subtype="evaluation"]');
      if (val === "time" || val === "coverage") {
        subWrap?.classList.remove("hidden");
        if (totalBtn) totalBtn.classList.toggle("hidden", val !== "time");
        const showTimeExtras = val === "time";
        latencyBtn?.classList.toggle("hidden", !showTimeExtras);
        throughputBtn?.classList.toggle("hidden", !showTimeExtras);
        benchmarksBtn?.classList.toggle("hidden", !showTimeExtras);
        evaluationBtn?.classList.toggle("hidden", val === "time");
        qsa(".lb-value-sub-pill").forEach((q) => {
          const sub = q.dataset.subtype || "";
          const isPrimary = VALUE_PRIMARY_SUBS.includes(sub);
          const isSecondary = VALUE_SECONDARY_SUBS.includes(sub);
          const isActive = isPrimary
            ? sub === defaultPrimary
            : isSecondary
              ? sub === defaultSecondary
              : false;
          q.classList.toggle("active", isActive);
          const label = sub ? (sub.charAt(0).toUpperCase() + sub.slice(1)) : "";
          q.textContent = isActive ? "✓ " + label : label;
        });
      } else {
        subWrap?.classList.add("hidden");
      }
      loadLeaderboard();
    };
  });
  qsa(".lb-value-sub-pill").forEach((p) => {
    p.onclick = () => {
      if (lbState.valueType !== "time" && lbState.valueType !== "coverage") return;
      const sub = p.dataset.subtype || "";
      if (lbState.valueType === "time" && sub === "evaluation") return;
      const isPrimary = VALUE_PRIMARY_SUBS.includes(sub);
      const isSecondary = VALUE_SECONDARY_SUBS.includes(sub);
      if (!isPrimary && !isSecondary) return;
      if (lbState.valueType === "coverage" && (!isPrimary || sub === "total")) return;
      if (isPrimary) {
        if (lbState.valueType === "time") {
          lbState.valueSubtypePrimary = lbState.valueSubtypePrimary === sub ? "" : sub;
        } else {
          if (lbState.valueSubtypePrimary === sub) return;
          lbState.valueSubtypePrimary = sub;
        }
      } else if (isSecondary) {
        if (lbState.valueSubtypeSecondary === sub) return;
        lbState.valueSubtypeSecondary = sub;
      }
      qsa(".lb-value-sub-pill").forEach((q) => {
        const s = q.dataset.subtype || "";
        const sIsPrimary = VALUE_PRIMARY_SUBS.includes(s);
        const sIsSecondary = VALUE_SECONDARY_SUBS.includes(s);
        const isActive = sIsPrimary
          ? s === lbState.valueSubtypePrimary
          : sIsSecondary
            ? s === lbState.valueSubtypeSecondary
            : false;
        q.classList.toggle("active", isActive);
        const label = s ? (s.charAt(0).toUpperCase() + s.slice(1)) : "";
        q.textContent = isActive ? "✓ " + label : label;
      });
      loadLeaderboard();
    };
  });
  const benchPanelWrap = byId("lb-bench-panel-wrap");
  const benchToggleArea = byId("lb-bench-toggle-area");
  const toggleBench = () => benchPanelWrap?.classList.toggle("open");
  benchToggleArea?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleBench();
  });
  benchToggleArea?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      toggleBench();
    }
  });
  document.addEventListener("click", (e) => {
    if (benchPanelWrap?.contains(e.target)) return;
    benchPanelWrap?.classList.remove("open");
  });
  byId("lb-avg-toggle")?.addEventListener("click", () => {
    lbState.showAvg = !lbState.showAvg;
    const btn = byId("lb-avg-toggle");
    btn.classList.toggle("active", lbState.showAvg);
    btn.textContent = lbState.showAvg ? "✓ Avg" : "Avg";
    loadLeaderboard();
  });
  byId("lb-omniscore-toggle")?.addEventListener("click", () => {
    lbState.omniScore = !lbState.omniScore;
    const btn = byId("lb-omniscore-toggle");
    btn.classList.toggle("active", lbState.omniScore);
    btn.textContent = lbState.omniScore ? "✓ OmniScore" : "OmniScore";
    if (lbState.omniScore) {
      lbState.metricMapPrev = { ...(lbState.metricMap || {}) };
      const next = {};
      (lbState.allColumns || []).forEach((b) => { next[b] = "judge_rating"; });
      lbState.metricMap = { ...lbState.metricMap, ...next };
    } else if (lbState.metricMapPrev) {
      lbState.metricMap = { ...lbState.metricMapPrev };
      lbState.metricMapPrev = null;
    }
    loadLeaderboard();
  });
  byId("lb-merge-inference-toggle")?.addEventListener("click", () => {
    lbState.mergeInference = !lbState.mergeInference;
    const btn = byId("lb-merge-inference-toggle");
    btn.classList.toggle("active", lbState.mergeInference);
    btn.textContent = lbState.mergeInference ? "✓ Merge Inference" : "Merge Inference";
    loadLeaderboard();
  });
  byId("lb-normalized-toggle")?.addEventListener("click", () => {
    lbState.normalized = !lbState.normalized;
    const btn = byId("lb-normalized-toggle");
    btn.classList.toggle("active", lbState.normalized);
    btn.textContent = lbState.normalized ? "✓ Normalized" : "Normalized";
    if (lbState.lastData) {
      renderLeaderboardTable(lbState.lastData);
    }
  });
  qsa(".lb-chart-mode-pill").forEach((p) => {
    p.addEventListener("click", () => {
      const m = p.dataset.mode || "bar";
      qsa(".lb-chart-mode-pill").forEach((q) => {
        const isActive = (q.dataset.mode || "") === m;
        q.classList.toggle("active", isActive);
      });
      if (lbState.lastData) updateLbChart(lbState.lastData);
    });
  });
  ["lb-chart-models-wrap", "lb-chart-benchmarks-wrap"].forEach((wrapId) => {
    const wrap = byId(wrapId);
    const toggle = byId(wrapId.replace("-wrap", "-toggle"));
    if (!wrap || !toggle) return;
    const toggleFn = () => wrap.classList.toggle("open");
    toggle.addEventListener("click", (e) => { e.stopPropagation(); toggleFn(); });
    toggle.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggleFn(); }
    });
  });
  byId("lb-thead")?.addEventListener("click", (e) => {
    const th = e.target.closest(".lb-sortable");
    if (!th || !th.dataset.sort) return;
    const key = th.dataset.sort;
    if (lbState.sort === key) {
      if (lbState.sortDir === "asc") lbState.sortDir = "desc";
      else { lbState.sort = ""; lbState.sortDir = "asc"; }
    } else {
      lbState.sort = key;
      lbState.sortDir = "asc";
    }
    lbState.page = 0;
    loadLeaderboard(false, { skipChart: true });
  });
}

function bindLbFilterCollapse() {
  const toggle = byId("lb-filters-toggle");
  const panel = byId("lb-filters");
  if (!toggle || !panel) return;
  const KEY = "omni-lb-filters-collapsed";
  try { if (localStorage.getItem(KEY) === "1") panel.classList.add("collapsed"); } catch (_) {}
  const fn = () => {
    const collapsed = panel.classList.toggle("collapsed");
    try { localStorage.setItem(KEY, collapsed ? "1" : "0"); } catch (_) {}
  };
  toggle.addEventListener("click", fn);
  toggle.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fn(); }
  });
}

function bindInferenceEvents() {
  // (removed dead #inf-load binding — that button no longer exists; sample loading is
  //  triggered by the benchmark radio onchange in inference.js)
  byId("inf-first")?.addEventListener("click", () => {
    infState.idx = 0;
    showInferenceSample();
  });
  byId("inf-prev")?.addEventListener("click", () => {
    if (infState.idx > 0) {
      infState.idx--;
      showInferenceSample();
    }
  });
  byId("inf-next")?.addEventListener("click", () => {
    if (infState.idx < infState.total - 1) {
      infState.idx++;
      showInferenceSample();
    }
  });
  byId("inf-last")?.addEventListener("click", () => {
    infState.idx = Math.max(0, infState.total - 1);
    showInferenceSample();
  });
  const infSampleInput = byId("inf-sample-input");
  if (infSampleInput) {
    // Brief red-border cue, then restore the current sample number so the
    // input never stays stuck on an invalid value (no silent no-op).
    const flagInfSampleInvalid = () => {
      infSampleInput.classList.add("border-rose-400", "text-rose-600");
      setTimeout(() => {
        infSampleInput.classList.remove("border-rose-400", "text-rose-600");
        infSampleInput.value = infState.total > 0 ? String(infState.idx + 1) : "";
      }, 800);
    };
    infSampleInput.onchange = () => {
      const raw = infSampleInput.value.trim();
      if (raw === "" || infState.total <= 0) {
        flagInfSampleInvalid();
        return;
      }
      const n = parseInt(raw, 10);
      if (isNaN(n) || n < 1 || n > infState.total) {
        flagInfSampleInvalid();
        return;
      }
      infState.idx = n - 1;
      showInferenceSample();
    };
    infSampleInput.onkeydown = (e) => {
      if (e.key === "Enter") {
        infSampleInput.onchange();
      }
    };
  }
  const infSlider = byId("inf-sample-slider");
  if (infSlider) {
    // Dragging fires "input" continuously. Update the position label live for
    // instant feedback, but DEBOUNCE the actual sample fetch so a fast drag
    // across thousands of samples doesn't hammer the server — load only when
    // the drag pauses (~200ms) or is released ("change").
    let sliderTimer = null;
    infSlider.addEventListener("input", () => {
      if (infState.total <= 0) return;
      let n = parseInt(infSlider.value, 10);
      if (isNaN(n)) return;
      n = Math.min(Math.max(n, 1), infState.total);
      infState.idx = n - 1;
      syncInfNav();
      clearTimeout(sliderTimer);
      sliderTimer = setTimeout(showInferenceSample, 200);
    });
    infSlider.addEventListener("change", () => {
      clearTimeout(sliderTimer);
      if (infState.total > 0) showInferenceSample();
    });
  }
  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
    if (document.documentElement.getAttribute("data-tab") === "2" && infState.total > 0) {
      if (e.key === "ArrowLeft") {
        if (infState.idx > 0) { infState.idx--; showInferenceSample(); }
      } else if (e.key === "ArrowRight") {
        if (infState.idx < infState.total - 1) { infState.idx++; showInferenceSample(); }
      }
    }
  });
}

// Event bindings (run when DOM ready)
function init() {
  bindSubmissionEvents();
  bindSubmissionSourceTabs();
  bindTabEvents();
  bindLeaderboardEvents();
  bindLbFilterCollapse();
  bindInferenceEvents();
  // Reset all leaderboard filters to defaults on every page load so a reload starts clean
  // (lbState is in-memory; this also syncs the filter pills + Model Mode to those defaults).
  resetLbFilters();

  const savedTab = getSavedTab();
  setTab(savedTab, false);
  if (savedTab === 0) loadModels();
  else if (savedTab === 1) {
    loadLeaderboard(true);
    startLbAutoRefresh(false);
  } else if (savedTab === 2) loadInferenceModels();
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      stopLbAutoRefresh();
    } else if (getSavedTab() === 1) {
      startLbAutoRefresh(true);
    }
  });

  // Load config paths from server
  fetch("/health").then(async (r) => {
    if (!r.ok) return;
    const d = await r.json();
    const intEl = byId("internal-path-info");
    if (intEl) {
      const exists = d.internal_path_exists;
      const dot = exists
        ? '<span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1"></span>'
        : '<span class="inline-block w-1.5 h-1.5 rounded-full bg-rose-500 mr-1"></span>';
      intEl.innerHTML = `${dot}<span class="text-slate-500">Now:</span> <span class="${exists ? "text-slate-700" : "text-rose-600"}">${escapeHtml(d.internal_path || "-")}</span>`
        + (exists ? "" : ' <span class="text-rose-600">(not found)</span>');
    }
    const s3El = byId("s3-path-info");
    if (s3El) {
      if (d.s3_configured) {
        const dot = '<span class="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1"></span>';
        s3El.innerHTML = `${dot}<span class="text-slate-500">Now:</span> <span class="text-slate-700">${escapeHtml(d.s3_path || "-")}</span>`;
      } else {
        const dot = '<span class="inline-block w-1.5 h-1.5 rounded-full bg-slate-300 mr-1"></span>';
        s3El.innerHTML = `${dot}<span class="text-slate-400">Not configured</span>`;
      }
    }
  }).catch(() => {});

  requestAnimationFrame(() => {
    if (document.body) document.body.classList.add("motion-ready");
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
