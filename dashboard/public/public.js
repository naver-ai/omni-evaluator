const MODALITIES = ["text", "vision", "audio"];

// Real leaderboard data, loaded from the generated data.json snapshot.
// Shape: { text|vision|audio: { cols:[...], rows:[{model, scores:[...], overall, coverage}] },
//          _meta: { models, benchmarks } }
let DATA = null;

async function loadData() {
  const res = await fetch(`data.json?v=${Date.now()}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`data.json ${res.status}`);
  DATA = await res.json();
}

// Escape strings before interpolating into innerHTML. Benchmark/model names in the snapshot
// derive from evaluation output filenames, so they are treated as untrusted (defense-in-depth,
// mirroring the internal dashboard which escapes every interpolated value).
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

// One illustrative sample per modality (text / image / video / audio), shown 2x2.
// Image/Video/Audio carry a small bundled media file so the modalities are actually visualized.
// IMPORTANT: these are SYNTHETIC examples for layout/preview only — the predictions and scores
// are fabricated. Model identities are therefore ANONYMIZED (Model A/B/C) so no invented output
// is ever attributed to a real, named third-party model. Replace with real, provenance-backed
// samples (and real model names) before treating this section as factual.
const EXAMPLE_ITEMS = [
  {
    task: "Text",
    benchmark: "GSM8K",
    title: "Math Word Problem",
    prompt: "Question: If x + 3 = 11, what is x?",
    ground_truth: "8",
    predictions: [
      { model: "Model A", prediction: "8", score: 100.0, is_correct: true },
      { model: "Model B", prediction: "8", score: 100.0, is_correct: true },
      { model: "Model C", prediction: "5", score: 0.0, is_correct: false },
    ],
  },
  {
    task: "Image",
    benchmark: "AI2D",
    title: "Science Diagram QA",
    media: { kind: "image", src: "assets/examples/image.jpg" },
    prompt: "Question: Which tissue type includes the vessel and tracheid?",
    ground_truth: "Xylem",
    predictions: [
      { model: "Model A", prediction: "Xylem", score: 100.0, is_correct: true },
      { model: "Model B", prediction: "Phloem", score: 0.0, is_correct: false },
      { model: "Model C", prediction: "Parenchyma", score: 0.0, is_correct: false },
    ],
  },
  {
    task: "Video",
    benchmark: "MVBench",
    title: "Video Understanding",
    media: { kind: "video", src: "assets/examples/video.mp4" },
    prompt: "Question: What is the main action taking place in the clip?",
    ground_truth: "A person assembling an object",
    predictions: [
      { model: "Model A", prediction: "A person assembling an object", score: 95.0, is_correct: true },
      { model: "Model B", prediction: "A person cooking a meal", score: 18.0, is_correct: false },
      { model: "Model C", prediction: "A person assembling an object", score: 90.5, is_correct: true },
    ],
  },
  {
    task: "Audio",
    benchmark: "ChartQA (audio)",
    title: "Spoken Chart QA",
    media: { kind: "audio", src: "assets/examples/audio.wav" },
    prompt: "Question (read aloud): What value does the highlighted bar represent?",
    ground_truth: "42",
    predictions: [
      { model: "Model A", prediction: "42", score: 100.0, is_correct: true },
      { model: "Model B", prediction: "40", score: 0.0, is_correct: false },
      { model: "Model C", prediction: "42", score: 100.0, is_correct: true },
    ],
  },
];

const state = {
  modality: "text",
  sortKey: "overall", // "model" | "overall" | <benchmark column index>
  sortDesc: true,
};

function initReveal() {
  const elements = document.querySelectorAll("[data-reveal]");
  if (!elements.length) return;
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12 }
  );
  elements.forEach((el) => observer.observe(el));
}

function renderHeroStats() {
  const meta = (DATA && DATA._meta) || {};
  const statMap = {
    models: meta.models ?? "--",
    benchmarks: meta.benchmarks ?? "--",
    modalities: MODALITIES.length,
  };
  document.querySelectorAll("[data-stat]").forEach((el) => {
    const key = el.dataset.stat;
    if (key && Object.prototype.hasOwnProperty.call(statMap, key)) {
      el.textContent = String(statMap[key]);
    }
  });
}

function getLeaderboardData() {
  const block = (DATA && DATA[state.modality]) || { cols: [], rows: [] };
  return { cols: block.cols || [], rows: block.rows || [] };
}

function sortRows(rows) {
  const { sortKey, sortDesc } = state;
  const dir = sortDesc ? -1 : 1;
  return [...rows].sort((a, b) => {
    if (sortKey === "model") return dir * a.model.localeCompare(b.model);
    if (sortKey === "overall") return dir * (a.overall - b.overall);
    if (sortKey === "coverage") return dir * ((a.coverage || 0) - (b.coverage || 0));
    const av = parseFloat(a.scores[sortKey]);
    const bv = parseFloat(b.scores[sortKey]);
    const an = Number.isNaN(av);
    const bn = Number.isNaN(bv);
    if (an && bn) return 0;
    if (an) return 1; // missing scores always sink
    if (bn) return -1;
    return dir * (av - bv);
  });
}

function columnMaxes(rows, numCols) {
  const maxes = Array(numCols).fill(-Infinity);
  rows.forEach((row) => {
    row.scores.forEach((v, i) => {
      const n = parseFloat(v);
      if (!Number.isNaN(n) && n > maxes[i]) maxes[i] = n;
    });
  });
  return maxes;
}

function rankCell(pos, showMedals) {
  if (showMedals) {
    if (pos === 0) return '<span class="rank-medal">🥇</span>';
    if (pos === 1) return '<span class="rank-medal">🥈</span>';
    if (pos === 2) return '<span class="rank-medal">🥉</span>';
  }
  return String(pos + 1);
}

function sortIndicator(key) {
  return state.sortKey === key ? `<span class="sort-indicator">${state.sortDesc ? "▾" : "▴"}</span>` : "";
}

function ariaSort(key) {
  if (state.sortKey !== key) return "none";
  return state.sortDesc ? "descending" : "ascending";
}

function sortBy(rawKey) {
  const key = rawKey === "model" || rawKey === "overall" || rawKey === "coverage" ? rawKey : parseInt(rawKey, 10);
  if (state.sortKey === key) {
    state.sortDesc = !state.sortDesc;
  } else {
    state.sortKey = key;
    state.sortDesc = key !== "model"; // model: asc by default, scores/overall: desc
  }
  renderLeaderboard();
}

function renderLeaderboard() {
  const { cols, rows: dataRows } = getLeaderboardData();
  const table = document.getElementById("leaderboard-table");
  if (!table) return;

  document.querySelectorAll(".modality-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.modality === state.modality);
  });

  const colMax = columnMaxes(dataRows, cols.length);
  const overallMax = dataRows.length ? Math.max(...dataRows.map((r) => r.overall)) : 0;
  const rows = sortRows(dataRows);
  // Medals only make sense when the order puts the best on top.
  const showMedals = state.sortDesc && state.sortKey !== "model";

  const header = `
    <thead>
      <tr>
        <th class="col-rank" scope="col">#</th>
        <th class="col-model" scope="col" aria-sort="${ariaSort("model")}"><button class="sort-btn" type="button" data-key="model">Model${sortIndicator("model")}</button></th>
        <th class="col-overall" scope="col" aria-sort="${ariaSort("overall")}"><button class="sort-btn" type="button" data-key="overall">Overall${sortIndicator("overall")}</button></th>
        <th class="col-coverage" scope="col" aria-sort="${ariaSort("coverage")}" title="Number of benchmarks this model was scored on"><button class="sort-btn" type="button" data-key="coverage">Benchmarks${sortIndicator("coverage")}</button></th>
        ${cols
          .map(
            (c, i) =>
              `<th scope="col" aria-sort="${ariaSort(i)}"><button class="sort-btn" type="button" data-key="${i}">${escapeHtml(c)}${sortIndicator(i)}</button></th>`
          )
          .join("")}
      </tr>
    </thead>
  `;

  const body = rows
    .map((row, pos) => {
      const overallBest = Math.abs(row.overall - overallMax) < 1e-9 ? " cell-best" : "";
      const cells = row.scores
        .map((v, i) => {
          const n = parseFloat(v);
          if (Number.isNaN(n)) return `<td class="cell-empty">–</td>`;
          const best = Math.abs(n - colMax[i]) < 1e-9 ? "cell-best" : "";
          return `<td class="${best}">${v}</td>`;
        })
        .join("");
      return `
        <tr>
          <td class="col-rank">${rankCell(pos, showMedals)}</td>
          <td class="col-model">${escapeHtml(row.model)}</td>
          <td class="col-overall${overallBest}">${row.overall.toFixed(1)}</td>
          <td class="col-coverage">${row.coverage ?? "–"}</td>
          ${cells}
        </tr>
      `;
    })
    .join("");

  const caption = `<caption class="sr-only">${state.modality} leaderboard: benchmark scores per model.</caption>`;
  table.innerHTML = `${caption}${header}<tbody>${body}</tbody>`;

  table.querySelectorAll(".sort-btn").forEach((btn) => {
    btn.addEventListener("click", () => sortBy(btn.dataset.key));
  });
}

function renderExamples() {
  const list = document.getElementById("examples-list");
  if (!list) return;

  list.innerHTML = EXAMPLE_ITEMS
    .map((item) => {
      let media = "";
      if (item.media) {
        const { kind, src } = item.media;
        if (kind === "image") media = `<img class="example-media" src="${src}" alt="example image" loading="lazy">`;
        else if (kind === "video") media = `<video class="example-media" src="${src}" controls preload="metadata"></video>`;
        else if (kind === "audio") media = `<audio class="example-media example-audio" src="${src}" controls preload="none"></audio>`;
      }
      const preds = (item.predictions || [])
        .map((p) => {
          const cls = p.is_correct ? "green" : "red";
          const mark = p.is_correct ? "Correct" : "Incorrect";
          const score = typeof p.score === "number" ? p.score.toFixed(1) : "";
          return `
            <div class="example-pred">
              <div class="example-pred-head">
                <span class="example-pred-model">${p.model}</span>
                <span class="example-pred-meta">
                  ${score !== "" ? `<span class="example-pred-score">${score}</span>` : ""}
                  <span class="badge ${cls}">${mark}</span>
                </span>
              </div>
              <div class="example-pred-text">${p.prediction}</div>
            </div>
          `;
        })
        .join("");
      return `
        <div class="example-card">
          <div class="example-header">
            <span class="badge blue">${item.task}</span>
            <span class="example-bench">${item.benchmark}</span>
          </div>
          ${media}
          <div class="example-title">${item.title}</div>
          <div class="example-label">Prompt</div>
          <div class="example-text">${item.prompt}</div>
          <div class="example-label">Ground Truth</div>
          <div class="example-text">${item.ground_truth}</div>
          <div class="example-label">Predictions</div>
          <div class="example-preds">${preds}</div>
        </div>
      `;
    })
    .join("");
}

function initLeaderboardTabs() {
  document.querySelectorAll(".modality-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mod = btn.dataset.modality;
      if (!MODALITIES.includes(mod)) return;
      state.modality = mod;
      state.sortKey = "overall";
      state.sortDesc = true;
      renderLeaderboard();
    });
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  initReveal();
  initLeaderboardTabs();
  renderExamples();
  try {
    await loadData();
  } catch (err) {
    const table = document.getElementById("leaderboard-table");
    if (table) table.innerHTML = `<caption>Failed to load leaderboard data (${err.message}).</caption>`;
    return;
  }
  renderLeaderboard();
  renderHeroStats();
});
