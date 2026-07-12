/* OmniEvaluator Internal Dashboard - Submission module (requires core.js) */

const SOURCE_TITLES = { internal: "Internal Path", direct: "Direct Upload", s3: "S3" };

function updateModelListHeaderCounts() {
  const getCount = (id) => {
    const el = byId(id);
    const n = parseInt(el?.dataset?.count || "0", 10);
    return Number.isFinite(n) ? n : 0;
  };
  const counts = {
    internal: getCount(SOURCE_EL_IDS.internal),
    direct: getCount(SOURCE_EL_IDS.direct),
    s3: getCount(SOURCE_EL_IDS.s3),
  };
  const total = counts.internal + counts.direct + counts.s3;

  const totalEl = byId("model-list-header");
  if (totalEl) totalEl.textContent = `${total} model${total === 1 ? "" : "s"}`;

  Object.keys(SOURCE_EL_IDS).forEach((src) => {
    const badge = byId(`sub-count-${src}`);
    if (badge) badge.textContent = String(counts[src]);
    const head = byId(SOURCE_EL_IDS[src])?.previousElementSibling;
    if (head && head.classList.contains("sub-models-head")) {
      head.textContent = `${SOURCE_TITLES[src]} (${counts[src]})`;
    }
  });
}

async function loadModels() {
  const loadingEl = byId("model-list-loading");
  if (loadingEl) {
    loadingEl.classList.remove("hidden");
    loadingEl.classList.add("flex");
  }
  try {
    const [internal, direct, s3] = await Promise.all([
      fetchJson(`${API}/submission/internal/models`).catch((e) => ({ models: [], _error: e.message })),
      fetchJson(`${API}/submission/direct/models`).catch((e) => ({ models: [], _error: e.message })),
      fetchJson(`${API}/submission/s3/models`).catch(() => ({ models: [] })),
    ]);
    const query = (byId("model-search")?.value || "").toLowerCase().match(/[a-z0-9]+/g) || [];
    const filter = (arr) => (!query.length ? arr : arr.filter((n) => query.every((t) => n.toLowerCase().includes(t))));
    const sortAsc = (arr) => [...arr].sort((a, b) => String(a).localeCompare(String(b), undefined, { sensitivity: "base" }));
    const sourceDot = (src) => {
      const cls = SOURCE_STYLES[src] || SOURCE_STYLES.internal;
      return `<span class="w-1.5 h-1.5 rounded-full ${cls} shrink-0 mt-1.5"></span>`;
    };
    const internalList = internal.models || [];
    const directList = direct.models || [];
    const s3List = s3.models || [];

    const render = (el, list, allowDelete, errorMsg, source, fullList) => {
      if (!el) return;
      if (errorMsg) {
        el.innerHTML = `<p class="text-xs text-amber-600">${escapeHtml(errorMsg)}</p>`;
        el.dataset.count = "0";
        return;
      }
      const dot = source ? sourceDot(source) : "";
      const isEmptyByFilter = fullList && fullList.length > 0 && list.length === 0;
      const emptyMsg = isEmptyByFilter ? "No results" : "No models.";
      el.innerHTML = list.length
        ? list
            .map(
              (n, i) =>
                `<div class="model-item motion-stagger flex justify-between items-center gap-2 py-1.5 px-2 rounded-lg border border-slate-200 mb-1" style="--i: ${i}">
                  ${dot}
                  <span class="text-xs text-slate-700 truncate flex-1">${escapeHtml(n)}</span>
                  ${allowDelete && source === "direct" ? `<button class="del-btn btn btn-danger" data-src="${source}" data-m="${escapeHtml(n)}">Delete</button>` : ""}
                </div>`
            )
            .join("")
        : `<p class="text-xs text-slate-500">${emptyMsg}</p>`;
      const count = fullList ? fullList.length : list.length;
      el.dataset.count = String(count);
    };

    render(byId("internal-models"), sortAsc(filter(internalList)), false, internal._error, "internal", internalList);
    render(byId("direct-models"), sortAsc(filter(directList)), true, direct._error, "direct", directList);
    render(byId("s3-models"), sortAsc(filter(s3List)), false, null, "s3", s3List);

    updateModelListHeaderCounts();
  } catch (e) {
    if (e.name !== "AbortError") console.error(e);
  } finally {
    if (loadingEl) loadingEl.classList.add("hidden");
  }
}

async function uploadFile(file) {
  const stEl = byId("upload-status");
  if (!stEl) console.warn("upload-status element missing");
  // Fall back to a no-op stub so a markup change can't make uploadFile throw.
  const st = stEl || { classList: { remove() {}, add() {} }, set innerHTML(_) {}, set textContent(_) {} };
  st.classList.remove("hidden");
  st.innerHTML = '<span class="upload-spinner"></span>Uploading...';
  const fd = new FormData();
  fd.append("file", file);
  try {
    const r = await fetch(`${API}/submission/direct/upload`, { method: "POST", body: fd });
    const d = await readJsonSafe(r);
    st.textContent = r.ok ? "Uploaded: " + (d.filename || d.model || "OK") : "Error: " + (d.detail || d.message || r.statusText || "Unknown");
    if (r.ok) {
      await loadModels();
      if (typeof loadLeaderboard === "function") loadLeaderboard(true);
    }
  } catch (e) {
    st.textContent = "Error: " + (e.message || MSG.uploadFailed);
  }
}
