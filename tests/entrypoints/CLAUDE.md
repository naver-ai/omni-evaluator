# tests/entrypoints — CLI entrypoint test rules

Overall operating rules (markers / conftest / execution / naming §7 / granularity §8) are in `tests/CLAUDE.md`,
and the structural design intent is in `tests/DESIGN.md §7`. This document contains only
rules that apply specifically to **`python -m omni_evaluator` CLI** tests.

---

## 1. Core principles

### 1.1 The subject under test is the *process*; assertions target *observable surfaces*
This area does not import functions and call them directly — it spins up a real shell subprocess
via the `run_cli` fixture and asserts only on **exit code / stdout / stderr / disk artifacts**.
The argument-parsing *logic* itself is covered by `tests/test_args.py` at the namespace level.
This area focuses on "what actually happens when the user types that command." Therefore the
"directly import the function under test" rule from `tests/CLAUDE.md §9` does not apply here
(because the subject under test is the CLI surface, not a symbol). However, config
paths/constants that e2e tests read are still directly imported per §9 (`INFERENCE_CONFIG_DIR`).

### 1.2 There is no `infer` subcommand — infer.py is verified through `evaluate`
The CLI surface subcommands are only `list` and `evaluate`. The inference step is not a
separate entrypoint but a step called internally by `evaluate`. To observe infer.py /
evaluate.py *separately*, use step toggle flags (the exact flag names are sourced from
`omni_evaluator/args.py` and the e2e tests as SoT):

| What to observe | How |
|---|---|
| infer.py only | Run `evaluate` with inference enabled and evaluation disabled → verify inference artifacts |
| evaluate.py only | Run `evaluate` with inference disabled on top of existing inference artifacts → verify evaluation/submission artifacts |
| Both in sequence | Run `evaluate` without toggles → verify both inference + evaluation artifacts |

### 1.3 Two layers — deterministic layer / live e2e layer
Two types of tests coexist within this area, distinguished by markers (same pattern as `tests/inference/CLAUDE.md §1.3`).

- **Deterministic layer** (zero external resources, always runs / `smoke`): `__main__.py` dispatch and exit code contracts.
  Each branch of `list`, exit code on missing/misspelled args, `--help`, etc. A regression safety net that runs in any environment.
- **Live e2e layer** (`slow` + `requires_env`): Whether `evaluate` actually reaches the real inference API and
  infer.py + evaluate.py run to completion. Non-deterministic, so opt-in only (`--runslow`).

### 1.4 Live e2e safety defaults — `--debug` required, one benchmark
Every `evaluate` call in e2e is launched **always with `--debug`**. Debug mode caps the number
of records processed to the operational constant (`tests/CLAUDE.md §5`, `tests/inference/CLAUDE.md §1.4`),
so even if the config/dataset grows, live API/GPU calls cannot exceed that limit → prevents billing/time blowups.
Additionally, only **one** benchmark from the config's benchmark list is run. These two defaults are
baked into the `build_evaluate_argv` builder (conftest) so that all e2e cells automatically follow them.

### 1.5 e2e configs use this area's dedicated copies as SoT; discover output paths via glob, don't predict them
e2e reads `tests/configs/entrypoints/{inference,evaluation}/*.yaml` via OmegaConf to build
engine/model/benchmark flags — when configs change, e2e follows (`tests/DESIGN.md §8`). This
area does not cross-read config directories from other test areas (`tests/configs/inference`, etc.);
it maintains **dedicated copies** (eliminating cross-area coupling). Path constants are sourced from
`tests/entrypoints/conftest.py` (`INFERENCE_CONFIG_DIR`/`EVALUATION_CONFIG_DIR`). When adding a
new engine cell, also copy its yaml into this directory (§3). However, keys that create node
dependencies (`hf_home`/`hf_hub_cache`) and non-arg keys (`reference_name`) are not copied over.
Since `--output_dirpath` is internally transformed into a `exp_name/...` subdirectory (exact rule
is sourced from `utils/io`), **do not reconstruct the exact output path in tests** — instead, find
it via glob under the tmp root and assert on that.

### 1.6 Engine verification uses a cross matrix, not a full grid
**Do not take the full product of** "multiple inference × multiple evaluation". Evaluation harness
extras are mutually exclusive and cannot all be installed in one env (`tests/CLAUDE.md §1.2`),
and inference engines have varying infrastructure requirements, making a full Cartesian product
mostly infeasible and redundant. Instead, examine the two axes as a cross:

- **Inference axis**: vary the inference engine (evaluation fixed to builtin, 1 benchmark) to verify infer.py dispatch.
- **Evaluation axis**: vary the evaluation harness (inference fixed to hf) to verify evaluate.py dispatch.
- The intersection (hf+builtin) is already covered by the inference axis, so it is excluded from the evaluation axis.

Each cell carries **cell-specific gates** (key/GPU/extra) via `pytest.param(marks=[...])`, so cells
without the required infrastructure are automatically skipped. **Parallelism across harnesses is
delegated to the tox+uv env matrix** (one env = one harness extra) — it is not reinvented here.
(Which inference/evaluation engines to add as cells is sourced from `test_evaluate.py`.)

### 1.7 Builtin datasets are resolved locally via `--local_dirpath` (SSRF gate)
Multimodal items in production builtin benchmarks arrive as presigned URLs from internal storage.
Those URLs resolve to private IPs and are blocked by `_validate_url_safe` (SSRF defense), so
without `--local_dirpath`, image fetching is blocked, predictions are empty, and the pipeline fails.
**Only cells that use builtin** pass the dataset's local root via `--local_dirpath` to resolve items
as **local file paths** (which bypass URL validation). External harnesses (lmms_eval, etc.) fetch
data from HF hub on their own and do not require this gate. Since the local root differs per node
and there is no canonical value in the repo, **no default is set** — builtin e2e only runs when
`LOCAL_DATA_DIR` is configured; otherwise it is skipped (§2). The SSRF defense itself is never
bypassed — placing production data locally is the correct approach.

### 1.8 vllm is a served endpoint — url/version/key come from env; skip if server is absent
The inference vllm cell hits an in-house OpenAI-compatible **served endpoint** (not in-process loading).
Therefore, the gate is `requires_env("VLLM_URL", "VLLM_API_VERSION", "VLLM_API_KEY")` rather than
a GPU requirement, and the builder reads those env vars and injects them as `--url`/`--vllm_api_version`/`--vllm_api_key`
(same reasoning as §1.5's "node dependency keys are not hardcoded in yaml"). The vllm validator
requires `--url`, so without this injection the cell would fail — env is the SoT. Each cell also
performs a healthcheck to the url before running; if the server is down, instead of waiting for
the subprocess to fail after long retries, it **skips with a Warning** (`skip_if_vllm_down`,
`tests/conftest.py`) — preventing server absence from being misread as a code regression.

---

## 2. Markers

| Layer | Marker |
|---|---|
| Deterministic (list / error / help) | `smoke` (always runs) |
| Live e2e (common) | `slow` (all cells) |
| Live e2e (per-cell gate) | inference api/vllm→`requires_env(...)`, inference hf→`requires_gpu`, evaluation external harness→`requires_extra("<import_pkg>")` |

For an e2e cell to actually run, it must pass all gates attached to that cell:
1. `slow` → skipped unless opted in with `--runslow` (common to all cells).
2. Inference axis cells: api requires `requires_env("OPENAI_API_KEY")`, vllm requires `requires_env("VLLM_URL", "VLLM_API_VERSION", "VLLM_API_KEY")` (§1.8, builder injects env→`--url` etc.) + url healthcheck skip, hf requires `requires_gpu` (skipped if no CUDA). `requires_env` passes only when both .env.test and the relevant env var are present.
3. Evaluation axis cells: inference is hf so `requires_gpu` + harness extra `requires_extra(...)` (skipped if not installed; takes the import package name as argument).
4. Builtin cells: the `local_data_dir` fixture skips if `LOCAL_DATA_DIR` is not set or absent (§1.7).
   No default — must be explicitly enabled on a node that has the data. (External harness cells do not have this gate.)

---

## 3. Checklist for adding new CLI tests

1. Is it a deterministic branch (new subcommand/flag/error path)? → `smoke`, assert exit code/stdout with `run_cli`.
   One test per branch (`tests/CLAUDE.md §8`). Bundle inputs that fall into the same exit code into one test.
2. Do you need to verify that a new step actually runs? → e2e. Use `build_evaluate_argv` to build config→flags
   (`--debug` and single benchmark are automatic, §1.4). Find artifacts via glob (§1.5). Attach `slow`.
3. For a new engine cell: ① copy its yaml to `tests/configs/entrypoints/{inference,evaluation}/`
   (area-dedicated copy is SoT, §1.5), ② add a `pytest.param(marks=[...])` entry to the inference/evaluation
   axis param list in `test_evaluate.py` and attach per-cell gates (§2) — follow the cross matrix principle (§1.6).
4. To batch multiple calls into one, use a module-scoped fixture like `inferred_dir` to share artifacts
   and split step verification across separate tests.
