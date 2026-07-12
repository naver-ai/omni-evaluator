/* OmniEvaluator Internal Dashboard - Leaderboard state (requires core.js) */

// Leaderboard state
let lbState = {
  sources: ["internal", "direct", "s3"],
  hiddenBenchmarks: [],
  modalityFilter: [], // [] or ["text","vision","audio"] - [] or all 3 = All
  inferenceEngineFilter: [], // [] or ["api","huggingface","vllm","sglang"] - [] or all 4 = All
  evaluationEngineFilter: [], // [] or ["built-in","lmms_eval","vlm_eval_kit","audio_bench"] - [] or all 4 = All
  valueType: "score",
  valueSubtypePrimary: "",
  valueSubtypeSecondary: "",
  showAvg: false,
  normalized: true,
  mergeInference: true,
  sort: "",
  sortDir: "asc",
  page: 0,
  pageSize: 100000, // no pagination: render all rows in one scroll
  hiddenRows: [], // user-hidden row keys (model||checkpoint||inference); shown at table bottom
  allColumns: [],
  lastData: null,
  chartModelIndices: [],
  chartBenchmarks: [],
  chartModelsManual: false, // user manually changed chart model selection (vs auto-pick)
  chartBenchmarksManual: false,
  weightCache: {},
  weightCacheLoaded: false,
  benchSearchQuery: "",
  metricKeysMap: {},
  metricMap: {},
  omniScore: false,
  metricMapPrev: null,
};

let _lbLastSignature = "";

function _leaderboardSignature(d) {
  if (!d) return "";
  const cols = d.columns || [];
  const rows = d.rows || [];
  const eng = d.benchmark_engine_map || {};
  return JSON.stringify({
    total: d.total,
    page: d.page,
    page_size: d.page_size,
    value_type: d.value_type,
    value_subtype: d.value_subtype,
    show_avg: d.show_avg,
    show_sum: d.show_sum,
    columns: cols,
    rows: rows,
    benchmark_engine_map: eng,
  });
}
