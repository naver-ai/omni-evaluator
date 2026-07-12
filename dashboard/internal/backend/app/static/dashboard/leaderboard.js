/* OmniEvaluator Internal Dashboard - Leaderboard orchestrator (requires leaderboard-state.js, leaderboard-api.js, leaderboard-render.js, core.js) */

const INFERENCE_OPTS = ["api", "huggingface", "vllm", "sglang"];
const EVALUATION_OPTS = ["built-in", "lmms_eval", "vlm_eval_kit", "audio_bench"];

function resetPillGroup(selector, stateArr, allOpts, dataAttr, labelFn) {
  applyPillGroup(selector, stateArr, allOpts.length, dataAttr, labelFn);
}

function applyPillGroup(selector, stateArr, allCount, dataAttr, labelFn) {
  qsa(selector).forEach((p) => {
    const val = p.dataset[dataAttr] || "";
    const isAll = val === "";
    const active = isAll
      ? (stateArr.length === 0 || stateArr.length === allCount)
      : stateArr.includes(val);
    p.classList.toggle("active", active);
    p.textContent = isAll ? "✓ All" : (active ? "✓ " : "") + (labelFn ? labelFn(val) : val || "All");
  });
}

function togglePillFilter(current, val, allCount) {
  if (val === "") return [];
  const arr = [...current];
  const idx = arr.indexOf(val);
  if (idx >= 0) arr.splice(idx, 1);
  else arr.push(val);
  arr.sort();
  return arr.length === allCount ? [] : arr;
}

function resetLbFilters() {
  lbState.sources = ["internal", "direct", "s3"];
  lbState.hiddenBenchmarks = [];
  lbState.hiddenRows = [];
  lbState.modalityFilter = [];
  lbState.inferenceEngineFilter = [];
  lbState.evaluationEngineFilter = [];
  lbState.valueType = "score";
  lbState.valueSubtypePrimary = "";
  lbState.valueSubtypeSecondary = "";
  lbState.showAvg = false;
  lbState.normalized = true;
  lbState.mergeInference = true;
  lbState.sort = "";
  lbState.sortDir = "asc";
  lbState.page = 0;
  lbState.metricKeysMap = {};
  lbState.metricMap = {};
  lbState.omniScore = false;
  lbState.metricMapPrev = null;
  lbState.chartModelIndices = [];
  lbState.chartBenchmarks = [];
  lbState.chartModelsManual = false;
  lbState.chartBenchmarksManual = false;
  qsa(".lb-source-pill").forEach((p) => {
    const active = lbState.sources.includes(p.dataset.source);
    p.classList.toggle("active", active);
    p.textContent = (active ? "✓ " : "") + p.dataset.source.charAt(0).toUpperCase() + p.dataset.source.slice(1);
  });
  setModelMode("normal", { silent: true });
  resetPillGroup(".lb-modality-pill", lbState.modalityFilter, ["text", "vision", "audio"], "modality", (m) => (m ? m.charAt(0).toUpperCase() + m.slice(1) : "All"));
  resetPillGroup(".lb-inference-pill", lbState.inferenceEngineFilter, INFERENCE_OPTS, "inference", (i) => i || "All");
  resetPillGroup(".lb-evaluation-pill", lbState.evaluationEngineFilter, EVALUATION_OPTS, "evaluation", (e) => e || "All");
  qsa(".lb-value-pill").forEach((p) => {
    const active = (p.dataset.value || "score") === "score";
    p.classList.toggle("active", active);
    p.textContent = active ? "✓ Score" : (p.dataset.value === "time" ? "Time" : "Coverage");
  });
  byId("lb-value-sub-wrap")?.classList.add("hidden");
  const avgBtn = byId("lb-avg-toggle");
  if (avgBtn) {
    avgBtn.classList.remove("active");
    avgBtn.textContent = "Avg";
  }
  const normBtn = byId("lb-normalized-toggle");
  if (normBtn) {
    normBtn.classList.add("active");
    normBtn.textContent = "✓ Normalized";
  }
  const mergeBtn = byId("lb-merge-inference-toggle");
  const omniBtn = byId("lb-omniscore-toggle");
  if (omniBtn) {
    omniBtn.classList.remove("active");
    omniBtn.textContent = "OmniScore";
  }
  if (mergeBtn) {
    mergeBtn.classList.add("active");
    mergeBtn.textContent = "✓ Merge Inference";
  }
  const benchList = byId("lb-bench-list");
  if (benchList) {
    benchList.querySelectorAll(".lb-bench-cb").forEach((cb) => (cb.checked = true));
  }
  // Pure reset: sync lbState + filter UI to defaults. Callers (Reset button, page init)
  // trigger loadLeaderboard() themselves so this can run on load without a duplicate fetch.
}
