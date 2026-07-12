# tests/pipelines — infer→eval e2e test rules

General operating rules (markers / conftest / execution / naming §7 / granularity §8) are in `tests/CLAUDE.md`,
and the structural design intent is in `tests/DESIGN.md §6`. This document contains only the rules
that apply specifically to **infer → eval pipeline** tests.

---

## 1. Core principles

### 1.1 In-process entry points — not subprocess
Pipeline tests **call the production entry points (`evaluate.main(args)` / `infer.main(args)`) directly as functions**.
This contrasts with `tests/entrypoints/`, which spawns `python -m omni_evaluator` as a subprocess and inspects
exit code/stdout — here we inspect the **intermediate artifacts** the pipeline writes to disk, in-process.
Args are constructed with `get_parser` + validation identical to production (= same parsing path as CLI, but not subprocess).

### 1.2 yaml drives every stage — default always-on / optional opt-in
A single `tests/configs/pipeline/<name>.yaml` file drives every stage: download → inference → postprocess → evaluation → score
(the SoT for engine/model/benchmark). The dataset local root (`LOCAL_DATA_DIR`), output/cache (tmp),
and debug are injected by the test. When a config changes, the test follows (`tests/DESIGN.md §8`).

The config directory splits into two execution policies — `test_pipeline.py` parses the directory and
dynamically parametrizes `(config, benchmark)` via `pytest_generate_tests`:

```
tests/configs/pipeline/
├── default/      # always-on — every yaml dropped here is collected without any options
└── optional/     # opt-in only — excluded by default due to env conflict risk, etc.
```

- **No options** → run all yaml files in `default/`.
- **`--pipeline-config=<path>`** (repeatable) → run *only* those yaml files; `default/` is **excluded**.
  The value is the yaml path (absolute, or relative to the working directory) — not restricted to `optional/`.
- **`--pipeline-config=all`** → run all of `optional/` (default excluded).
- A non-existent path causes a `UsageError` at the collection stage.

Options live in `tests/conftest.py` (global options home); selection/parametrize logic is in `test_pipeline.py`
(one test per area — `tests/CLAUDE.md §3`). The key invariant is "if optional is collected, default does not run" —
only the opted-in configs are run in isolation.

### 1.3 super-test — stages = methods, verified via intermediate artifacts
Expensive inference is run **only once** (`pipeline_run`, module-scope: `evaluate.main` called once → all
artifacts written to disk). On top of that, each pipeline stage is verified as *one method*, parametrized per
benchmark — inspecting the *intermediate output* and *artifact completeness* of each stage:

| Stage | Intermediate artifact verified |
|---|---|
| Download/resolve | Inference record count matches debug cap; multimodal items resolve to existing local files |
| Inference | All records have a non-empty prediction |
| Postprocess | Postprocess function is composable + application trace present (`prediction_postprocessed`) |
| Evaluation | Non-empty metric dict |
| Score | Metric values are native numbers, finite, and within normal range (ratio metrics in [0,1]) |
| Artifacts | Output JSON (config/inference/evaluation) + (conditional) submission_output |

> Exact method names, artifact dict shapes, and metric keys: source and `test_pipeline.py` are the SoT.

### 1.4 Debug samples are capped, but can be increased for confidence
The pipeline caps samples with `--debug` (`tests/CLAUDE.md §5` safety default) — to prevent runaway live charges/time.
To increase verification confidence beyond unit smoke (3), raise `NUM_DEBUG_SAMPLES` in the test.
Since this constant is **bound at import time** in each module (source `omni_evaluator/inference`),
both the *source module* and the *actual consumer (inference engine module)* attributes must be set/restored
for the change to actually take effect — patching only one side leaves the engine using the old value.

### 1.5 Builtin data is gated by `LOCAL_DATA_DIR` (SSRF)
Multimodal items in builtin benchmarks use internal storage presigned URLs (private IPs) that are blocked by SSRF
defenses. They are bypassed by resolving to local files via `--local_dirpath` (same reason as
`tests/entrypoints/CLAUDE.md §1.7`); the local root is taken from the `LOCAL_DATA_DIR` env var
(no default; skipped if unset).

---

## 2. Markers

| Branch | Marker | Notes |
|---|---|---|
| HF inference pipeline | `slow` + `requires_gpu` | Long due to model loading and multi-sample inference; `--runslow` opt-in |
| (light/submission branch) | See time budget table in `tests/DESIGN.md §6` | Add `requires_env`/`very_slow` as needed |

If `LOCAL_DATA_DIR` is unset, the `local_data_dir` fixture skips (§1.5). Increase
`@pytest.mark.timeout(...)` sufficiently when the sample count is large.

---

## 3. Checklist for adding a new pipeline test

1. Add a config yaml — engine/model/benchmark are SoT; do not include node-dependent keys.
   - Safe config to always run → `tests/configs/pipeline/default/<name>.yaml` (auto-collected without options).
   - Opt-in only due to env conflict risk, etc. → `tests/configs/pipeline/optional/<name>.yaml`
     (run only with `--pipeline-config=<that yaml path>`, §1.2).
2. Wrap expensive runs in a module/session-scope fixture for **exactly one** execution (`pipeline_run` pattern).
3. One method per stage, parametrized per benchmark. Assert on intermediate artifacts (disk JSON) (§1.3).
4. To increase the debug cap, patch both source + consumer as described in §1.4.
5. Markers: `slow` + the appropriate environment gate for the inference engine (`requires_gpu`/`requires_env`). For builtin data, `LOCAL_DATA_DIR`.
