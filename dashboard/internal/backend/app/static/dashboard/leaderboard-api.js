/* OmniEvaluator Internal Dashboard - Leaderboard API/data functions (requires leaderboard-state.js, core.js) */

function getLbSources() {
  return lbState.sources || [];
}

function buildLbParams() {
  const params = new URLSearchParams();
  params.set("sources", getLbSources().join(","));
  params.set("model_mode", getModelMode());
  params.set("page", String(lbState.page));
  params.set("page_size", String(lbState.pageSize));
  if (lbState.sort) params.set("sort", lbState.sort);
  if (lbState.sortDir) params.set("sort_dir", lbState.sortDir);
  if (lbState.hiddenBenchmarks.length) params.set("hidden_benchmarks", lbState.hiddenBenchmarks.join(","));
  const mods = lbState.modalityFilter;
  if (mods.length > 0 && mods.length < 3) params.set("modality", mods.join(","));
  const infs = lbState.inferenceEngineFilter;
  if (infs.length > 0 && infs.length < 4) params.set("inference_engine", infs.join(","));
  const evs = lbState.evaluationEngineFilter;
  if (evs.length > 0 && evs.length < 4) params.set("evaluation_engine", evs.join(","));
  if (lbState.valueType) params.set("value_type", lbState.valueType);
  if (lbState.valueType === "time") {
    const subs = [lbState.valueSubtypePrimary, lbState.valueSubtypeSecondary].filter(Boolean);
    if (subs.length) params.set("value_subtype", subs.join(","));
  } else if (lbState.valueType === "coverage") {
    if (lbState.valueSubtypePrimary) params.set("value_subtype", lbState.valueSubtypePrimary);
  }
  if (lbState.showAvg) params.set("show_avg", "true");
  params.set("merge_inference", lbState.mergeInference ? "true" : "false");
  if (lbState.valueType === "score" && lbState.metricMap && Object.keys(lbState.metricMap).length) {
    // Only send metrics the user actually changed from the per-benchmark default
    // (metricKeysMap[b][0]). The full ~237-entry map URL-encodes to ~10KB, which
    // overflows nginx's 8KB header buffer: it silently RSTs the HTTP/2 stream and the
    // browser surfaces it as "Failed to fetch" (most visibly when None also appends the
    // full hidden_benchmarks list). The backend treats a partial map as override-only,
    // so omitting defaults is safe. See apply_metric_override().
    const km = lbState.metricKeysMap || {};
    const diff = {};
    for (const [b, m] of Object.entries(lbState.metricMap)) {
      if (!m) continue;
      const def = (km[b] && km[b][0]) || "";
      if (m !== def) diff[b] = m;
    }
    if (Object.keys(diff).length) params.set("metric_map", JSON.stringify(diff));
  }
  return params;
}

let _scanPollTimer = null;

function _updateScanStatus(scanning) {
  const el = byId("lb-scan-status");
  if (!el) return;
  if (scanning) {
    el.classList.remove("hidden");
    el.classList.add("flex");
  } else {
    el.classList.add("hidden");
    el.classList.remove("flex");
  }
}

function _applyLeaderboardData(d, opts = {}) {
  const silent = !!opts.silent;
  const onlyIfChanged = !!opts.onlyIfChanged;
  const tableWrap = byId("lb-table-wrap");
  const countsEl = byId("lb-counts");

  const sig = _leaderboardSignature(d);
  if (onlyIfChanged && sig === _lbLastSignature) return;
  _lbLastSignature = sig;
  const prevData = lbState.lastData;
  lbState.lastData = d;

  // Show/hide scan spinner
  const scanning = !!d.scanning;
  _updateScanStatus(scanning);

  // If scanning, poll again in 5s to pick up new results
  if (_scanPollTimer) clearTimeout(_scanPollTimer);
  if (scanning) {
    _scanPollTimer = setTimeout(() => {
      loadLeaderboard(false, { silent: true });
    }, 5000);
  }

  if (!d.rows || d.rows.length === 0) {
    if ((d.total || 0) > 0 && lbState.page > 0) {
      lbState.page = 0;
      loadLeaderboard();
      return;
    }
    if (silent) return;
    if (scanning) {
      setLbEmptyState('<div class="flex items-center justify-center gap-3"><div class="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div><span>Scanning models... results will appear shortly.</span></div>', false, true);
    } else {
      setLbEmptyState("No models.");
    }
    return;
  }

  const cols = d.columns || [];
  const allCols = d.all_columns ?? cols;
  lbState.allColumns = allCols;

  syncMetricKeysMap(d);
  renderLeaderboardTable(d);

  tableWrap.classList.remove("hidden");
  const benchWrap = byId("lb-bench-above-table");
  if (benchWrap) benchWrap.classList.remove("hidden");
  countsEl.classList.add("hidden");
  updateLbDims(d.total, cols.length);

  const pageState = updateLbPagination(d.total);
  if (pageState.needsReload) {
    loadLeaderboard();
    return;
  }

  rebuildLbBenchmarkList(cols, allCols, d.benchmark_engine_map || {});
  // Sort / pagination only change the table; leave the chart untouched so it
  // does not re-animate when nothing chart-relevant changed.
  if (!opts.skipChart) {
    rebuildChartSelectors(d, prevData);
    updateLbChart(d);
  }
}

async function loadLeaderboard(bypassCache = false, opts = {}) {
  const silent = !!opts.silent;
  const loading = byId("lb-loading");
  const tableWrap = byId("lb-table-wrap");
  const empty = byId("lb-empty");

  if (!silent) {
    empty.classList.add("hidden");
    // Only show the "Loading..." text on the genuine first load (nothing on screen
    // yet). On tab re-entry / sort / filter we already have a table, so keep it in
    // place and update quietly — no flash.
    if (!lbState.lastData) {
      loading.classList.remove("hidden");
      tableWrap.classList.add("hidden");
      byId("lb-pagination")?.classList.add("hidden");
    }
  }

  const sources = getLbSources();
  if (!sources.length) {
    setLbEmptyState("Select at least one source (Internal/Direct/S3).", true);
    return;
  }

  const refresh = bypassCache;

  try {
    const params = buildLbParams();
    if (refresh) params.set("refresh", "true");
    const d = await apiFetchJson(`${API}/leaderboard?${params}`, { abortKey: "leaderboard" });
    if (!silent) loading.classList.add("hidden");

    const showAvg = d.show_avg || false;
    const valueType = d.value_type || lbState.valueType || "score";
    if (showAvg && valueType === "score") await loadWeightCache();

    _applyLeaderboardData(d, opts);
  } catch (e) {
    if (e.name !== "AbortError") {
      if (silent) { console.error(e); return; }
      setLbEmptyState("Error: " + (e.message || MSG.requestFailed));
    }
  }
}

async function loadWeightCache(force = false) {
  if (lbState.weightCacheLoaded && !force) return;
  try {
    const weights = await fetchJson(`${API}/leaderboard/weights`);
    lbState.weightCache = weights && typeof weights === "object" ? weights : {};
  } catch (_) {
    lbState.weightCache = {};
  }
  lbState.weightCacheLoaded = true;
}

let _lbAutoTimer = null;
function startLbAutoRefresh(immediate = true) {
  if (_lbAutoTimer) return;
  if (immediate) loadLeaderboard(true, { silent: true, onlyIfChanged: true });
  _lbAutoTimer = setInterval(() => {
    if (document.hidden) return;
    loadLeaderboard(true, { silent: true, onlyIfChanged: true });
  }, 60000);
}

function stopLbAutoRefresh() {
  if (_lbAutoTimer) {
    clearInterval(_lbAutoTimer);
    _lbAutoTimer = null;
  }
}
