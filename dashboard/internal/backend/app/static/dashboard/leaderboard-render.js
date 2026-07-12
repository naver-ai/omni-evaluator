/* OmniEvaluator Internal Dashboard - Leaderboard rendering (requires leaderboard-state.js, leaderboard-api.js, core.js) */

function cellClass(val, minmax, col, valueType) {
  if (val == null || val === "-" || val === "") return "";
  const mm = minmax?.[col];
  if (!mm || mm[0] == null || mm[1] == null) return "";
  const mn = Number(mm[0]);
  const mx = Number(mm[1]);
  if (mn === mx || Math.abs(mn - mx) < 1e-9) return ""; // min === max: no highlight (applies to Score/Time/Coverage alike)
  const fMin = formatCellValue(mn, col, valueType);
  const fMax = formatCellValue(mx, col, valueType);
  if (fMin === fMax) return ""; // no highlight when the displayed (formatted) values are identical
  const n = parseFloat(String(val));
  if (Math.abs(n - mx) < 1e-9) return "lb-cell-max";
  if (Math.abs(n - mn) < 1e-9) return "lb-cell-min";
  return "";
}


function formatModelName(name) {
  if (name == null || name === "") return "-";
  const s = String(name);
  if (s.toLowerCase().startsWith("debug__")) {
    const rest = s.slice("debug__".length);
    const idx = rest.indexOf("__");
    return idx >= 0 ? "debug__" + rest.slice(0, idx) : s;
  }
  const idx = s.indexOf("__");
  return idx >= 0 ? s.slice(0, idx) : s;
}

// Instant cell tooltip: a title= attribute is OS-delayed (~1s). For truncated Model/
// Checkpoint/Inference cells we want the full value immediately, so delegate one set of
// listeners over [data-tip] and position a small fixed box at the cursor.
(function setupCellTooltip() {
  if (typeof document === "undefined" || window.__cellTipInit) return;
  window.__cellTipInit = true;
  let tip = null;
  document.addEventListener("mouseover", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (!el) return;
    const txt = el.getAttribute("data-tip") || "";
    if (!txt) return;
    if (!tip) {
      tip = document.createElement("div");
      tip.className = "app-cell-tip";
      document.body.appendChild(tip);
    }
    tip.textContent = txt;
    tip.style.display = "block";
  });
  document.addEventListener("mousemove", (e) => {
    if (!tip || tip.style.display !== "block") return;
    tip.style.left = (e.clientX + 12) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  document.addEventListener("mouseout", (e) => {
    const el = e.target.closest && e.target.closest("[data-tip]");
    if (el && tip) tip.style.display = "none";
  });
})();

const BENCH_SUFFIX_LABELS = {
  "__runtime_inference": "Inference",
  "__runtime_total": "Total",
  "__latency": "Latency",
  "__throughput": "Throughput",
  "__benchmark_time": "Benchmarks",
  "__coverage_inference": "Inference",
  "__coverage_evaluation": "Evaluation",
};

function splitBenchKey(col) {
  if (!col) return { base: col, suffix: "", subLabel: "" };
  const s = String(col);
  for (const [suffix, label] of Object.entries(BENCH_SUFFIX_LABELS)) {
    if (s.endsWith(suffix)) {
      return { base: s.slice(0, -suffix.length), suffix, subLabel: label };
    }
  }
  return { base: s, suffix: "", subLabel: "" };
}

function benchDisplayMeta(col, valueType) {
  const meta = splitBenchKey(col);
  const useBase = valueType !== "score" && meta.suffix;
  return {
    base: meta.base,
    display: useBase ? meta.base : String(col),
    subLabel: useBase ? meta.subLabel : "",
  };
}

function benchLabelText(col, valueType, benchEngMap = {}) {
  const meta = benchDisplayMeta(col, valueType);
  const eng = benchEngMap[col] || benchEngMap[meta.base];
  const base = meta.display;
  const label = meta.subLabel ? `${base} (${meta.subLabel})` : base;
  return eng && eng !== "-" ? `(${eng}) ${label}` : label;
}

function normalizeScoreValue(n) {
  if (!Number.isFinite(n)) return null;
  if (n < 0) return n;
  if (n <= 1) return n * 100;
  if (n <= 100) return n;
  if (n <= 1000) return n / 10;
  return n;
}

function formatScore(val, col) {
  if (val == null || val === "-" || val === "") return "-";
  const n = parseFloat(String(val));
  if (isNaN(n)) return String(val);
  if (lbState.normalized && n >= 0) {
    const nn = normalizeScoreValue(n);
    if (nn != null) return nn.toFixed(1);
  }
  return n.toFixed(1);
}

function formatTime(seconds) {
  if (seconds == null || seconds === "-" || seconds === "") return "-";
  const n = parseFloat(String(seconds));
  if (isNaN(n) || n < 0) return String(seconds);
  const h = Math.floor(n / 3600);
  const m = Math.floor((n % 3600) / 60);
  const s = Math.floor(n % 60);
  if (h > 0) return `${String(h).padStart(2, "0")}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${String(m).padStart(2, "0")}m`;
  return `${s}s`;
}

function formatBenchmarkTime(seconds) {
  if (seconds == null || seconds === "-" || seconds === "") return "-";
  const n = parseFloat(String(seconds));
  if (isNaN(n) || n < 0) return String(seconds);
  const h = Math.floor(n / 3600);
  const m = Math.floor((n % 3600) / 60);
  const s = Math.floor(n % 60);
  if (h > 0) return `${String(h).padStart(2, "0")}h ${String(m).padStart(2, "0")}m`;
  if (m > 0) return `${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
  return `${n.toFixed(1)}s`;
}

function normalizeLatencyThroughput(val) {
  if (val == null || val === "-" || val === "") return "-";
  const s = String(val).trim();
  if (!s) return s;
  if (s.endsWith("s")) {
    const core = s.slice(0, -1).trim();
    if (core && !isNaN(parseFloat(core))) return core;
  }
  return s;
}

function isTimeRawMode() {
  return lbState.valueType === "time"
    && (lbState.valueSubtypeSecondary === "latency" || lbState.valueSubtypeSecondary === "throughput");
}

function formatValueForDisplay(val, col, valueType, opts = {}) {
  const empty = opts.empty ?? "-";
  const nanFallback = Object.prototype.hasOwnProperty.call(opts, "nan") ? opts.nan : null;
  if (val == null || val === "-" || val === "") return empty;
  const n = parseFloat(String(val));
  if (isNaN(n)) return nanFallback != null ? nanFallback : String(val);
  if (col && (col.endsWith("__latency") || col.endsWith("__throughput"))) {
    return normalizeLatencyThroughput(val);
  }
  if (isTimeRawMode() && col && (
    col.endsWith("__benchmark_time")
    || col.endsWith("__runtime_inference")
    || col.endsWith("__runtime_evaluation")
    || col.endsWith("__runtime_total")
  )) {
    return String(val);
  }
  if (col && (col.endsWith("__benchmark_time") || (col === "_sum" && valueType === "time"))) {
    return formatBenchmarkTime(val);
  }
  if (col && (col.endsWith("__runtime_inference") || col.endsWith("__runtime_evaluation") || col.endsWith("__runtime_total"))) {
    return formatTime(val);
  }
  if (valueType === "coverage" || (col && (col.endsWith("__coverage_inference") || col.endsWith("__coverage_evaluation")))) {
    const clamped = Math.max(0, Math.min(1, n));
    const pct = (clamped * 100).toFixed(1);
    return opts.coveragePercent === false ? pct : `${pct}%`;
  }
  if (valueType === "score") {
    return opts.scoreRaw ? n.toFixed(1) : formatScore(val, col);
  }
  return n.toFixed(1);
}

function formatCellValue(val, col, valueType) {
  return formatValueForDisplay(val, col, valueType);
}

function hideLbChart() {
  const wrap = byId("lb-chart-wrap");
  if (wrap) wrap.classList.add("hidden");
  if (window.lbChartInstance) {
    window.lbChartInstance.destroy();
    window.lbChartInstance = null;
  }
}

function syncMetricKeysMap(d) {
  const keysMap = d?.metric_keys_map || {};
  lbState.metricKeysMap = keysMap;
  const cols = d?.columns || [];
  cols.forEach((b) => {
    const keys = keysMap[b] || [];
    const current = lbState.metricMap[b];
    if (!current || (keys.length && !keys.includes(current))) {
      lbState.metricMap[b] = keys[0] || "";
    }
  });
  if (lbState.omniScore) {
    const updated = {};
    cols.forEach((b) => { updated[b] = "judge_rating"; });
    lbState.metricMap = { ...lbState.metricMap, ...updated };
  }
}

// Lead label for the secondary header rows (Metric / Weight). The label sits in a sticky
// colspan=2 cell that stays frozen over the #+Model columns; the remaining lead columns
// (Inference/Checkpoint/Sum/Avg) follow in a plain colspan cell that scrolls with the table.
function frozenLeadCells(labelHtml, leadCount) {
  const rest = leadCount - 2;
  return `<th class="lb-hdr-lead px-2 py-1 text-left text-slate-500 font-normal" colspan="2">${labelHtml}</th>` +
    (rest > 0 ? `<th class="px-2 py-1" colspan="${rest}"></th>` : "");
}

function buildMetricSelectHeader(cols, showAvg, showSum, valueType) {
  if (valueType !== "score" || !cols?.length) return "";
  const leadCount = 4 + (showSum ? 1 : 0) + (showAvg ? 1 : 0);
  return `<tr class="lb-metric-row text-[0.6rem] font-normal">
      ${frozenLeadCells("Metric", leadCount)}
      ${cols.map((b) => {
        const keys = lbState.metricKeysMap?.[b] || [];
        const selected = lbState.metricMap?.[b] || (keys[0] || "");
        if (!keys.length) {
          return `<th class="px-2 py-1 text-center text-slate-500 font-normal">-</th>`;
        }
        const options = keys.map((k) => {
          const sel = k === selected ? "selected" : "";
          return `<option value="${escapeHtml(k)}" ${sel}>${escapeHtml(k)}</option>`;
        }).join("");
        return `<th class="px-2 py-1 text-center font-normal">
          <select class="lb-metric-sel w-full max-w-[7.5rem] px-1 py-0.5 rounded text-[0.6rem] bg-white border border-slate-300 text-center" data-bench="${escapeHtml(b)}">${options}</select>
        </th>`;
      }).join("")}
    </tr>`;
}

function bindMetricSelects() {
  const selects = qsa(".lb-metric-sel");
  if (!selects.length) return;
  selects.forEach((sel) => {
    sel.onchange = () => {
      const bench = sel.dataset.bench;
      lbState.metricMap[bench] = sel.value || "";
      if (lbState.omniScore && sel.value !== "judge_rating") {
        lbState.omniScore = false;
        const omniBtn = byId("lb-omniscore-toggle");
        if (omniBtn) {
          omniBtn.classList.remove("active");
          omniBtn.textContent = "OmniScore";
        }
      }
      loadLeaderboard();
    };
  });
}

function setLbEmptyState(message, clearData = false, isHtml = false) {
  const loading = byId("lb-loading");
  const tableWrap = byId("lb-table-wrap");
  const empty = byId("lb-empty");
  const pagination = byId("lb-pagination");
  const countsEl = byId("lb-counts");
  if (loading) loading.classList.add("hidden");
  if (tableWrap) tableWrap.classList.add("hidden");
  if (pagination) pagination.classList.add("hidden");
  if (empty) {
    empty.classList.remove("hidden");
    if (isHtml) empty.innerHTML = message;
    else empty.textContent = message;
  }
  if (countsEl) countsEl.classList.add("hidden");
  hideLbChart();
  const ww = byId("lb-weight-wrap");
  if (ww) ww.classList.add("hidden");
  const bw = byId("lb-bench-above-table");
  if (bw) bw.classList.add("hidden");
  const dimsEl = byId("lb-table-dims");
  if (dimsEl) dimsEl.textContent = "";
  if (clearData) {
    lbState.lastData = null;
    _lbLastSignature = "";
  }
}

function updateLbDims(total, colsLen) {
  const dimsEl = byId("lb-table-dims");
  if (!dimsEl) return;
  const mergeNote = lbState.mergeInference ? " · Best-of merged" : "";
  dimsEl.textContent = `${total} models × ${colsLen} benchmarks${mergeNote}`;
  dimsEl.title = lbState.mergeInference ? "Scores are best-of across inferences" : "";
}

function updateLbPagination(total) {
  const pagination = byId("lb-pagination");
  if (!pagination) return { needsReload: false };
  if (total <= lbState.pageSize) {
    lbState.page = 0;
    pagination.classList.add("hidden");
    return { needsReload: false };
  }
  const totalPages = Math.ceil(total / lbState.pageSize);
  const safePage = Math.min(lbState.page, totalPages - 1);
  if (safePage !== lbState.page) {
    lbState.page = safePage;
    return { needsReload: true };
  }
  pagination.innerHTML =
    `<button class="lb-page-prev btn btn-ghost px-3 py-1 text-sm" ${lbState.page <= 0 ? "disabled" : ""}>Prev</button>` +
    `<span class="px-3 text-slate-500 text-sm">Page ${lbState.page + 1} / ${totalPages}</span>` +
    `<button class="lb-page-next btn btn-ghost px-3 py-1 text-sm" ${lbState.page >= totalPages - 1 ? "disabled" : ""}>Next</button>`;
  pagination.classList.remove("hidden");
  pagination.querySelector(".lb-page-prev")?.addEventListener("click", () => {
    if (lbState.page > 0) {
      lbState.page--;
      loadLeaderboard(false, { skipChart: true });
    }
  });
  pagination.querySelector(".lb-page-next")?.addEventListener("click", () => {
    if (lbState.page < totalPages - 1) {
      lbState.page++;
      loadLeaderboard(false, { skipChart: true });
    }
  });
  return { needsReload: false };
}

function clampWeight(v) {
  if (!Number.isFinite(v)) return 0;
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

let _weightSaveTimer = null;

function buildWeightHeaderHtml(cols, showAvg, showSum, valueType) {
  if (valueType !== "score" || !showAvg || !cols?.length) return "";
  const leadCount = 4 + (showSum ? 1 : 0) + (showAvg ? 1 : 0);
  const weightLabel = `<span>Weight</span><button type="button" id="lb-weight-reset" class="ml-2 text-[0.6rem] text-slate-500 hover:text-slate-700">Reset</button>`;
  const weightRow = `<tr class="lb-weight-row text-[0.6rem] font-normal">
      ${frozenLeadCells(weightLabel, leadCount)}
      ${cols.map((b) => {
        const raw = parseFloat(lbState.weightCache?.[b]);
        const v = Number.isFinite(raw) ? clampWeight(raw) : 1;
        return `<th class="px-2 py-1 text-center font-normal">
          <input type="text" inputmode="decimal" class="lb-weight-inp w-14 px-1.5 py-0.5 rounded text-[0.6rem] bg-white border border-slate-300 text-center" data-bench="${escapeHtml(b)}" value="${v}">
        </th>`;
      }).join("")}
    </tr>`;
  const normRow = `<tr class="lb-weight-row text-[0.6rem] font-normal" id="lb-weight-norm-row">
      ${frozenLeadCells("Weight (normalized)", leadCount)}
      ${cols.map((b) => `<th class="px-2 py-1 text-center text-slate-500 font-normal" data-bench="${escapeHtml(b)}">-</th>`).join("")}
    </tr>`;
  return weightRow + normRow;
}

function updateWeightNormalizedRow() {
  const normRow = byId("lb-weight-norm-row");
  if (!normRow) return;
  const inputs = qsa(".lb-weight-inp");
  const raw = [];
  inputs.forEach((inp) => {
    const v = clampWeight(parseFloat(inp.value));
    raw.push({ bench: inp.dataset.bench, val: v });
  });
  const sum = raw.reduce((a, b) => a + b.val, 0);
  raw.forEach(({ bench, val }) => {
    const cell = normRow.querySelector(`th[data-bench="${CSS.escape(bench)}"]`);
    if (!cell) return;
    const n = sum > 0 ? (val / sum) : 0;
    cell.textContent = n.toFixed(3);
  });
}

function bindWeightHeaderInputs() {
  const inputs = qsa(".lb-weight-inp");
  if (!inputs.length) return;
  const resetBtn = byId("lb-weight-reset");
  updateWeightNormalizedRow();
  if (resetBtn) {
    resetBtn.onclick = () => {
      const w = { ...(lbState.weightCache || {}) };
      inputs.forEach((i) => {
        i.value = "1";
        w[i.dataset.bench] = 1;
      });
      updateWeightNormalizedRow();
      clearTimeout(_weightSaveTimer);
      _weightSaveTimer = setTimeout(async () => {
        try {
          await fetchJson(`${API}/leaderboard/weights`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(w),
          });
          lbState.weightCache = w;
          lbState.weightCacheLoaded = true;
          loadLeaderboard();
        } catch (_) {}
      }, 200);
    };
  }
  inputs.forEach((inp) => {
    const commit = () => {
      const v = clampWeight(parseFloat(inp.value));
      inp.value = Number.isFinite(v) ? String(v) : "0";
      updateWeightNormalizedRow();
      clearTimeout(_weightSaveTimer);
      _weightSaveTimer = setTimeout(async () => {
        const w = {};
        qsa(".lb-weight-inp").forEach((i) => {
          const val = clampWeight(parseFloat(i.value));
          w[i.dataset.bench] = val;
        });
        try {
          await fetchJson(`${API}/leaderboard/weights`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(w),
          });
          lbState.weightCache = w;
          lbState.weightCacheLoaded = true;
          loadLeaderboard();
        } catch (_) {}
      }, 400);
    };
    inp.oninput = () => {
      updateWeightNormalizedRow();
      clearTimeout(_weightSaveTimer);
      _weightSaveTimer = setTimeout(commit, 600);
    };
    inp.onblur = commit;
  });
}

function renderLeaderboardTable(d) {
  if (!d) return;
  const cols = d.columns || [];
  const minmax = d.minmax || {};
  const showAvg = d.show_avg || false;
  const showSum = d.show_sum || false;
  const valueType = d.value_type || lbState.valueType || "score";
  const sortArrow = (key) => (lbState.sort !== key ? "" : lbState.sortDir === "asc" ? " ↑" : " ↓");
  const benchEngMap = d.benchmark_engine_map || {};

  const rowKeyOf = (r) => `${r?.model || ""}||${r?.checkpoint || ""}||${r?.inference || ""}`;
  const allRows = d.rows || [];
  const hiddenSet = new Set(lbState.hiddenRows || []);
  const visibleRows = allRows.filter((r) => !hiddenSet.has(rowKeyOf(r)));
  const hiddenRowObjs = allRows.filter((r) => hiddenSet.has(rowKeyOf(r)));

  let theadHtml = "<tr>" +
    `<th class="lb-col-rank px-2 py-1.5 text-center text-slate-500 font-normal" title="Rank in current sort">#</th>` +
    `<th class="lb-sortable lb-col-model px-2 py-1.5 text-left" data-sort="model">Model${sortArrow("model")}</th>` +
    `<th class="lb-sortable lb-col-inference px-2 py-1.5 text-left" data-sort="inference">Inference${sortArrow("inference")}</th>` +
    `<th class="lb-sortable lb-col-checkpoint px-2 py-1.5 text-left" data-sort="checkpoint">Checkpoint${sortArrow("checkpoint")}</th>`;
  if (showSum) theadHtml += `<th class="lb-sortable lb-col-sum px-2 py-1.5 text-center" data-sort="_sum">Sum${sortArrow("_sum")}</th>`;
  if (showAvg) theadHtml += `<th class="lb-sortable px-2 py-1.5 text-center" data-sort="avg">Avg${sortArrow("avg")}</th>`;
  theadHtml += cols.map((c) => {
    const meta = benchDisplayMeta(c, valueType);
    const eng = benchEngMap[c] || benchEngMap[meta.base];
    const nameHtml = meta.subLabel
      ? `${escapeHtml(meta.display)} <span class="text-[0.55rem] text-slate-500">(${escapeHtml(meta.subLabel)})</span>`
      : escapeHtml(meta.display);
    const title = meta.subLabel ? `${meta.display} (${meta.subLabel})` : meta.display;
    const label = eng && eng !== "-"
      ? `<span class="block"><span class="text-[0.6rem] text-slate-500">(${escapeHtml(eng)})</span><span class="block font-normal text-slate-500">${nameHtml}</span></span>`
      : `<span class="font-normal text-slate-500">${nameHtml}</span>`;
    return `<th class="lb-sortable lb-bench-col px-2 py-1.5 text-center font-normal text-slate-500" data-sort="${escapeHtml(c)}" title="${escapeHtml(title)}">${label}${sortArrow(c)}</th>`;
  }).join("") + "</tr>";
  theadHtml += buildMetricSelectHeader(cols, showAvg, showSum, valueType);
  theadHtml += buildWeightHeaderHtml(cols, showAvg, showSum, valueType);

  const thead = byId("lb-thead");
  if (thead) {
    thead.innerHTML = theadHtml;
    bindMetricSelects();
    bindWeightHeaderInputs();
  }

  const tbody = byId("lb-tbody");
  if (!tbody) return;
  tbody.innerHTML = visibleRows
    .map((row, i) => {
      const _rank = i + 1;
      const _medals = lbState.sortDir === "desc" && lbState.sort && !["model", "inference", "checkpoint"].includes(lbState.sort);
      const _rankCell = _medals && _rank <= 3 ? ["🥇", "🥈", "🥉"][_rank - 1] : _rank;
      const _rkey = rowKeyOf(row);
      let tr = `<tr class="lb-row">`;
      tr += `<td class="lb-col-rank px-2 py-1.5 text-center text-slate-500 lb-rank-cell"><span class="lb-rank-num">${_rankCell}</span><button type="button" class="lb-row-hide" data-rowkey="${escapeHtml(_rkey)}" title="Hide row" aria-label="Hide row">✕</button></td>`;
      tr += `<td class="lb-col-model px-2 py-1.5" data-tip="${escapeHtml(row.model || "")}">${escapeHtml(formatModelName(row.model))}</td>`;
      tr += `<td class="lb-col-inference px-2 py-1.5" data-tip="${escapeHtml(row.inference || "")}">${escapeHtml(row.inference)}</td>`;
      tr += `<td class="lb-col-checkpoint px-2 py-1.5" data-tip="${escapeHtml(row.checkpoint || "")}">${escapeHtml(row.checkpoint)}</td>`;
      if (showSum) {
        const sumVal = formatCellValue(row._sum != null ? row._sum : "-", "_sum", valueType);
        const sumCls = cellClass(row._sum, minmax, "_sum", valueType);
        tr += `<td class="lb-col-sum px-2 py-1.5 text-center ${sumCls}">${escapeHtml(sumVal)}</td>`;
      }
      if (showAvg) {
        const avgVal = formatCellValue(row._avg != null ? row._avg : "-", "_avg", valueType);
        const avgCls = cellClass(row._avg, minmax, "_avg", valueType);
        tr += `<td class="px-2 py-1.5 text-center ${avgCls}">${escapeHtml(avgVal)}</td>`;
      }
      cols.forEach((c) => {
        const val = row.scores?.[c] ?? "-";
        const cls = cellClass(val, minmax, c, valueType);
        tr += `<td class="px-2 py-1.5 text-center ${cls}">${escapeHtml(formatCellValue(val, c, valueType))}</td>`;
      });
      tr += "</tr>";
      return tr;
    })
    .join("");

  tbody.querySelectorAll(".lb-row-hide").forEach((btn) => {
    btn.onclick = (e) => {
      e.stopPropagation();
      const k = btn.dataset.rowkey;
      if (k && !lbState.hiddenRows.includes(k)) {
        lbState.hiddenRows.push(k);
        renderLeaderboardTable(lbState.lastData);
      }
    };
  });
  renderHiddenRows(hiddenRowObjs, rowKeyOf);
}

// Bottom "hidden rows" tray. Clicking a chip restores that row. State is lbState.hiddenRows
// (model||checkpoint||inference keys); hide and restore both just re-render from
// lbState.lastData — no server round-trip needed since it's a display-only toggle.
function renderHiddenRows(hiddenRowObjs, rowKeyOf) {
  const box = byId("lb-hidden-rows");
  if (!box) return;
  if (!hiddenRowObjs || !hiddenRowObjs.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  const chips = hiddenRowObjs
    .map((r) => {
      const key = rowKeyOf(r);
      const label = [formatModelName(r.model), r.checkpoint, r.inference].filter(Boolean).join(" · ");
      return `<button type="button" class="lb-hidden-chip" data-rowkey="${escapeHtml(key)}" title="Click to restore">↩ ${escapeHtml(label)}</button>`;
    })
    .join("");
  box.innerHTML = `<span class="lb-hidden-label">${hiddenRowObjs.length} hidden · click to restore</span><span class="lb-hidden-chips">${chips}</span><button type="button" class="lb-hidden-restore-all" title="Restore all hidden rows">Restore all</button>`;
  box.querySelectorAll(".lb-hidden-chip").forEach((chip) => {
    chip.onclick = () => {
      const k = chip.dataset.rowkey;
      lbState.hiddenRows = (lbState.hiddenRows || []).filter((x) => x !== k);
      renderLeaderboardTable(lbState.lastData);
    };
  });
  const restoreAll = box.querySelector(".lb-hidden-restore-all");
  if (restoreAll) restoreAll.onclick = () => {
    lbState.hiddenRows = [];
    renderLeaderboardTable(lbState.lastData);
  };
}

function updateBenchAllNoneState(selectAllBtn, deselectAllBtn, hiddenCount, totalCount) {
  if (!selectAllBtn || !deselectAllBtn) return;
  selectAllBtn.classList.toggle("active", totalCount > 0 && hiddenCount === 0);
  deselectAllBtn.classList.toggle("active", totalCount > 0 && hiddenCount === totalCount);
}

function rebuildLbBenchmarkList(visibleCols, allCols, benchEngMap = {}) {
  const list = byId("lb-bench-list");
  const searchEl = byId("lb-bench-search");
  const selectAllBtn = byId("lb-bench-select-all");
  const deselectAllBtn = byId("lb-bench-deselect-all");
  if (!list) return;

  const cols = visibleCols || [];
  const available = allCols && allCols.length ? allCols : cols;
  const hidden = (lbState.hiddenBenchmarks || []).filter((h) => available.includes(h));
  const listCols = available;
  const valueType = lbState.lastData?.value_type || lbState.valueType || "score";
  list.innerHTML = listCols
    .map((b) => {
      const meta = benchDisplayMeta(b, valueType);
      const eng = benchEngMap[b] || benchEngMap[meta.base];
      const engLabel = eng && eng !== "-" ? ` <span class="text-[0.6rem] text-slate-500">(${escapeHtml(eng)})</span>` : "";
      const displayHtml = meta.subLabel
        ? `${escapeHtml(meta.display)} <span class="text-[0.55rem] text-slate-500">(${escapeHtml(meta.subLabel)})</span>`
        : escapeHtml(meta.display);
      const searchText = `${meta.display} ${b} ${meta.subLabel || ""}${eng ? " " + eng : ""}`.toLowerCase();
      return `<label class="lb-bench-item flex items-center gap-2 cursor-pointer py-1 px-1.5 rounded hover:bg-slate-50" data-bench="${escapeHtml(b)}" data-search="${escapeHtml(searchText)}"><input type="checkbox" class="lb-bench-cb rounded border-slate-300" data-bench="${escapeHtml(b)}" ${hidden.includes(b) ? "" : "checked"}> <span class="truncate">${displayHtml}</span>${engLabel}</label>`;
    })
    .join("");

  const applySearch = (q) => applyListSearch(list, ".lb-bench-item", q);

  const applyAndReload = () => {
    const hiddenNow = [...list.querySelectorAll(".lb-bench-cb:not(:checked)")].map((c) => c.dataset.bench);
    lbState.hiddenBenchmarks = hiddenNow;
    updateBenchAllNoneState(selectAllBtn, deselectAllBtn, hiddenNow.length, listCols.length);
    loadLeaderboard();
  };

  updateBenchAllNoneState(selectAllBtn, deselectAllBtn, hidden.length, listCols.length);

  if (selectAllBtn) selectAllBtn.onclick = () => {
    list.querySelectorAll(".lb-bench-cb").forEach((cb) => { cb.checked = true; });
    applyAndReload();
  };
  if (deselectAllBtn) deselectAllBtn.onclick = () => {
    list.querySelectorAll(".lb-bench-cb").forEach((cb) => { cb.checked = false; });
    lbState.hiddenBenchmarks = [...listCols];
    updateBenchAllNoneState(selectAllBtn, deselectAllBtn, listCols.length, listCols.length);
    loadLeaderboard();
  };

  list.querySelectorAll(".lb-bench-cb").forEach((cb) => {
    cb.onchange = () => applyAndReload();
  });

  if (searchEl) {
    searchEl.value = lbState.benchSearchQuery || "";
    applySearch(searchEl.value);
    searchEl.oninput = () => {
      lbState.benchSearchQuery = searchEl.value || "";
      applySearch(lbState.benchSearchQuery);
    };
  }
}

function computeChartDefaults(rows, cols) {
  if (!rows?.length || !cols?.length) return { modelIndices: [], benchmarks: [] };
  const hasScore = (r, c) => {
    const v = r.scores?.[c];
    return v != null && v !== "-" && v !== "" && !isNaN(parseFloat(v));
  };
  const N = 7;
  // Bidirectional density refinement: choose models/benchmarks so the selected N×N grid is
  // as filled as possible (dense = fewer blank bars = prettier, richer comparison). Seed with
  // coverage-ranked models, then alternate: the benchmarks the current models cover most ->
  // the models that cover those benchmarks most. Converges in ~2 passes.
  let modelIdx = rows
    .map((r, i) => ({ i, count: cols.filter((c) => hasScore(r, c)).length }))
    .sort((a, b) => b.count - a.count)
    .slice(0, N)
    .map((x) => x.i);
  let benchmarks = [];
  for (let pass = 0; pass < 2; pass++) {
    const curModels = modelIdx.map((i) => rows[i]);
    benchmarks = cols
      .map((c) => ({ c, count: curModels.filter((r) => hasScore(r, c)).length }))
      .sort((a, b) => b.count - a.count)
      .slice(0, N)
      .map((x) => x.c);
    modelIdx = rows
      .map((r, i) => ({ i, count: benchmarks.filter((c) => hasScore(r, c)).length }))
      .sort((a, b) => b.count - a.count)
      .slice(0, N)
      .map((x) => x.i);
  }
  return { modelIndices: modelIdx, benchmarks };
}

function rebuildChartSelectors(d, prevData) {
  const rows = d?.chart_rows || d?.rows || [];
  const cols = d?.columns || [];
  const benchEngMap = d?.benchmark_engine_map || {};
  const modelsEl = byId("lb-chart-models");
  const benchmarksEl = byId("lb-chart-benchmarks");
  if (!modelsEl || !benchmarksEl) return;

  const { modelIndices, benchmarks } = computeChartDefaults(rows, cols);
  const prevRows = prevData?.chart_rows || prevData?.rows || [];
  const rowKey = (r) => `${r?.model || ""}||${r?.checkpoint || ""}||${r?.inference || ""}`;
  const prevKeys = (lbState.chartModelIndices || [])
    .map((i) => prevRows[i])
    .filter(Boolean)
    .map(rowKey);
  const preservedIndices = prevKeys
    .map((k) => rows.findIndex((r) => rowKey(r) === k))
    .filter((i) => i >= 0);
  const preservedBenchmarks = (lbState.chartBenchmarks || []).filter((c) => cols.includes(c));
  const desiredModelCount = Math.min(7, rows.length);
  const desiredBenchCount = Math.min(7, cols.length);
  // Auto-pick unless the user manually edited the chart selectors (tracked by flags).
  // This replaces the old "count >= 7" heuristic, which silently overwrote a user's
  // deliberate 7-item selection on every data refresh.
  const useAutoModels = !lbState.chartModelsManual;
  const useAutoBenches = !lbState.chartBenchmarksManual;

  if (useAutoModels) {
    lbState.chartModelIndices = modelIndices.length
      ? modelIndices
      : rows.map((_, i) => i).slice(0, desiredModelCount);
  } else {
    lbState.chartModelIndices = preservedIndices.length
      ? preservedIndices
      : (modelIndices.length ? modelIndices : rows.map((_, i) => i).slice(0, desiredModelCount));
  }

  if (useAutoBenches) {
    lbState.chartBenchmarks = benchmarks.length
      ? benchmarks
      : cols.slice(0, desiredBenchCount);
  } else {
    lbState.chartBenchmarks = preservedBenchmarks.length
      ? preservedBenchmarks
      : (benchmarks.length ? benchmarks : cols.slice(0, desiredBenchCount));
  }

  const rowLabel = (r) => {
    const m = formatModelName(r.model) || r.inference || "-";
    const inf = r.inference && r.inference !== "-" ? `(${r.inference}) ` : "";
    const ck = r.checkpoint && r.checkpoint !== "-" ? ` (${r.checkpoint})` : "";
    return inf + m + ck;
  };

  const valueType = d?.value_type || lbState.valueType || "score";
  const benchLabel = (c) => benchLabelText(c, valueType, benchEngMap);

  modelsEl.innerHTML = rows
    .map(
      (r, i) =>
        `<label class="flex items-center gap-2 cursor-pointer text-xs text-slate-500">
          <input type="checkbox" class="lb-chart-model-cb" data-idx="${i}" ${lbState.chartModelIndices.includes(i) ? "checked" : ""}>
          <span class="truncate" title="${escapeHtml(rowLabel(r))}">${escapeHtml(rowLabel(r))}</span>
        </label>`
    )
    .join("");
  benchmarksEl.innerHTML = cols
    .map(
      (c) =>
        `<label class="flex items-center gap-2 cursor-pointer text-xs text-slate-500">
          <input type="checkbox" class="lb-chart-bench-cb" data-bench="${escapeHtml(c)}" ${lbState.chartBenchmarks.includes(c) ? "checked" : ""}>
          <span class="truncate" title="${escapeHtml(benchLabel(c))}">${escapeHtml(benchLabel(c))}</span>
        </label>`
    )
    .join("");

  modelsEl.querySelectorAll(".lb-chart-model-cb").forEach((cb) => {
    cb.onchange = () => {
      lbState.chartModelIndices = [...modelsEl.querySelectorAll(".lb-chart-model-cb:checked")].map((x) => parseInt(x.dataset.idx, 10));
      lbState.chartModelsManual = true;
      if (lbState.lastData) updateLbChart(lbState.lastData);
    };
  });
  benchmarksEl.querySelectorAll(".lb-chart-bench-cb").forEach((cb) => {
    cb.onchange = () => {
      lbState.chartBenchmarks = [...benchmarksEl.querySelectorAll(".lb-chart-bench-cb:checked")].map((x) => x.dataset.bench);
      lbState.chartBenchmarksManual = true;
      if (lbState.lastData) updateLbChart(lbState.lastData);
    };
  });
}

function updateLbChart(d) {
  const wrap = byId("lb-chart-wrap");
  const canvas = byId("lb-chart");
  const allRows = d?.chart_rows || d?.rows || [];
  if (!wrap || !canvas || !allRows.length || !d?.columns?.length) {
    wrap?.classList.add("hidden");
    return;
  }
  wrap.classList.remove("hidden");

  const rows = allRows;
  const cols = d.columns;
  const selModels = (lbState.chartModelIndices || []).filter((i) => i >= 0 && i < rows.length);
  const selBenchmarks = (lbState.chartBenchmarks || []).filter((c) => cols.includes(c));
  const chartRows = selModels.length ? selModels.map((i) => rows[i]) : rows.slice(0, 7);
  const chartCols = selBenchmarks.length ? selBenchmarks : cols.slice(0, 7);

  const modelsCountEl = byId("lb-chart-models-count");
  if (modelsCountEl) modelsCountEl.textContent = `(${chartRows.length}/${rows.length})`;
  const benchCountEl = byId("lb-chart-benchmarks-count");
  if (benchCountEl) benchCountEl.textContent = `(${chartCols.length}/${cols.length})`;

  if (!chartRows.length || !chartCols.length) {
    wrap.classList.add("hidden");
    return;
  }

  const modeEl = qs(".lb-chart-mode-pill.active");
  const mode = modeEl?.dataset?.mode || "bar";
  const chartRowLabel = (r) => {
    const m = formatModelName(r.model) || r.inference || "-";
    const inf = r.inference && r.inference !== "-" ? `(${r.inference}) ` : "";
    return inf + m;
  };
  const labels = chartRows.map(chartRowLabel);
  const colors = ["#4f46e5", "#0ea5e9", "#8b5cf6", "#06b6d4", "#6366f1", "#0284c7", "#7c3aed", "#0891b2", "#a855f7", "#2563eb"];
  const valueType = d.value_type || lbState.valueType || "score";
  const formatChartLabel = (val, col) => formatValueForDisplay(val, col, valueType, {
    empty: "", nan: "", coveragePercent: false, scoreRaw: true,
  });
  const datasets = chartCols.map((c, i) => ({
    label: benchLabelText(c, valueType, d.benchmark_engine_map || {}),
    data: chartRows.map((r) => {
      const v = r.scores?.[c];
      const n = parseFloat(v);
      if (isNaN(n)) return null; // unevaluated -> null so Chart.js draws no bar (vs a misleading 0)
      if (valueType === "score" && lbState.normalized) {
        const nn = normalizeScoreValue(n);
        return nn == null ? null : nn;
      }
      return n;
    }),
    backgroundColor: colors[i % colors.length] + "80",
    borderColor: colors[i % colors.length],
    borderWidth: 1,
  }));

  // In-place update if same chart type — no destroy/recreate, no flicker
  if (window.lbChartInstance && window.lbChartInstance.config.type === mode) {
    window.lbChartInstance.data.labels = labels;
    window.lbChartInstance.data.datasets = datasets;
    window.lbChartInstance.update();
    return;
  }

  // Type changed or first render
  if (window.lbChartInstance) {
    window.lbChartInstance.destroy();
    window.lbChartInstance = null;
  }
  const ctx = canvas.getContext("2d");
  const dataLabelsOpt = {
    display: true, align: "top", anchor: "end",
    formatter: (val, ctx) => formatChartLabel(val, chartCols[ctx.datasetIndex]),
    color: "#475569", font: { size: 9 },
  };
  const axisStyle = {
    ticks: { color: "#64748b", font: { size: 10 } },
    grid: { color: "rgba(15, 23, 42, 0.08)" },
  };
  window.lbChartInstance = new Chart(ctx, {
    type: mode,
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { position: "bottom", labels: { color: "#475569", font: { size: 10 } } },
        datalabels: dataLabelsOpt,
      },
      scales: { x: axisStyle, y: { ...axisStyle, beginAtZero: true } },
    },
  });
}
