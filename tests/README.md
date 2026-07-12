# omni_evaluator Tests

Test suite for `omni_evaluator`. Tests run on isolated [tox](https://tox.wiki) environments,
each pinned to the transformers / submodule versions a given engine or adapter needs.

## 📦 Installation

- Driven by `tox` (config: [`tox.ini`](../tox.ini)) — each env = its own `uv` venv + needed submodules + pinned deps.
- **Prerequisite:** [`uv`](https://docs.astral.sh/uv/). Python 3.10 (per `requires-python`) is auto-provisioned by `uv` — no manual install.

```bash
# 0. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. Install tox with the uv backend (one-time)
uv tool install tox --with tox-uv

# 2. Sanity-check the install — list environments without running anything
tox -l
```

- This installs the runner only. Bare `tox` runs **every** env below (takes a while).
- To run tests: [Quick Start](#-quick-start) (full suite) · [Partial Tests](#-partial-tests) (quick subset).
- Worktrees: `/mnt/tmp/.tox-omni` (override with `TOX_WORK_DIR`).

| Environment | What it covers |
|---|---|
| `omni_tf457` | **Modern stack** (`transformers>=4.57,<5`): unit tests + builtin eval + **all 3 engines in one venv** (`lmms_eval` / `lm_eval` / `vlmeval`) + BASE models (hyperclovax_vision, qwen2_vl, qwen2_omni, qwen3(_omni), whisper_v3, voxtral) + submodule-data benchmarks + builtin pipelines (incl. the optional hyperclovax_vision pipeline) |
| `omni_tf444` | Dedicated models `emu3` / `janus_pro` / `deepseek_vl` adapters on `transformers==4.44.0` |
| `omni_vllm` | _(optional; not in `env_list`)_ vLLM serving backend — run with `tox -e omni_vllm` |

> Minimized from 8 envs to 2 (+1 optional): E2E testing showed the only hard split is **transformers version** — the 3 eval engines co-install in one venv (`omni_tf457`), and only the dedicated models' `transformers==4.44.0` can't share. minicpmo (`transformers==4.51.0`) is a separate ad-hoc venv, not a tox env.

<details>
<summary>📂 How <code>tox.ini</code> is wired (and what's not yet enabled)</summary>

The only hard split is `transformers` version: the modern stack (`>=4.57,<5`) hosts all engines + BASE models in one venv (`omni_tf457`); the dedicated models pin `transformers==4.44.0` and can't share, so they get their own venv (`omni_tf444`).

```ini
[tox]
env_list =
    omni_tf457              # modern: all engines + BASE models + submodules → transformers>=4.57,<5
    omni_tf444              # dedicated: emu3 / janus_pro / deepseek_vl       → transformers==4.44.0
# omni_vllm (optional, not in env_list) — vLLM serving; `tox -e omni_vllm`
work_dir = {env:TOX_WORK_DIR:/mnt/tmp/.tox-omni}   # venvs/worktrees here
```

`[commands]` holds reusable `commands_pre` snippets, referenced as `{[commands]<name>}`:

```ini
[commands]
opencv = uv pip install opencv-python-headless     # every env needs it
clone_janus = bash -c 'test -d .../submodules/Janus || git clone <url> ... && git checkout <pinned-sha>'
# clone_deepseek_vl / clone_emu3 / clone_lmms_eval / clone_lm_eval / clone_vlm_eval — same idempotent clone-at-pinned-sha pattern
```

`[testenv]` is the shared base every env inherits; each `[testenv:<name>]` only adds its clone + pin:

```ini
[testenv]                                  # ── inherited by all envs ──
package = editable                         # pip install -e . with the `test` extra
pass_env = CUDA_VISIBLE_DEVICES HF_* OPENAI_API_KEY ...   # host env vars passed in
commands_pre = {[commands]opencv}          # install opencv-python-headless
commands = pytest {posargs}                # {posargs} = whatever you put after `--`

[testenv:omni_tf444]                        # ── thin override example ──
commands_pre =
    {[commands]clone_janus}                 # reusable snippet from [commands]
    uv pip install -e submodules/Janus
    uv pip install transformers==4.44.0     # pin just for this env
commands =
    pytest tests/inference/.../test_emu3.py {posargs} --runslow   # narrowed paths
```
</details>


## 🚀 Quick Start

### 1. `.env.test` — opt-in to live tests

Tests that hit a real API or dataset are **skipped unless `.env.test` exists**. Copy the template, fill in only the keys you need:

```bash
cp .env.test.example .env.test
```

- Existence of `.env.test` = the opt-in switch (loaded by `tests/conftest.py`).
- A `requires_env(...)` test skips if its var is blank — even if your shell already exports it.
- Only the vars below affect tests; blank → that test skips, nothing breaks.

```bash
# ── Live API keys — each gates its provider's live tests (blank → skip) ──
OPENAI_API_KEY=          # OpenAI live tests (tests/inference/api/, OpenAI eval)
OPENAI_ORGANIZATION=     # optional OpenAI org id
ANTHROPIC_API_KEY=       # Anthropic live tests (tests/api/anthropic/)
GOOGLE_API_KEY=          # Google live tests (tests/api/google/)

# ── vLLM live endpoint — all three consumed together; skip if blank or server is down ──
VLLM_URL=                # vLLM smoke endpoint URL
VLLM_API_KEY=            # endpoint key
VLLM_API_VERSION=        # e.g. v1

# ── HuggingFace ──
HF_TOKEN=                # gated model downloads (requires_hf_token tests; blank → skip)
HF_MODEL_ID=             # override the model an HF adapter test loads (blank → adapter DEFAULT_MODEL_ID)
HF_HUB_CACHE=            # HF Hub cache dir for downloaded models (blank → ~/.cache/huggingface/hub)

# ── S3 live round-trip — all four consumed together; blank → skip ──
S3_BUCKET_NAME=          # sandbox bucket for the live upload/download smoke
S3_ACCESS_KEY=           # access key
S3_SECRET_KEY=           # secret key
S3_ENDPOINT_URL=         # S3-compatible endpoint URL

# ── Local dataset root ──
LOCAL_DATA_DIR=          # builtin pipeline / entrypoint e2e (tests/pipelines, tests/entrypoints; blank → skip)
```

### 2. Collect tests (run nothing)

`--collect-only` builds the env once but runs no test — confirms the env builds and tests are discovered:

```bash
tox -e omni_tf457 -- tests/ --collect-only -q
```

### 3. Run everything

```bash
# Full suite across all tox environments (after the install steps above)
tox
```

> ⚠️ The full suite takes a while. For a quick check, run a [Partial Test](#-partial-tests) instead.


## Partial Tests

### By environment

Run a single `tox` environment with `-e`:

```bash
tox -e omni_tf457      # modern stack: unit tests + builtin eval + all 3 engines + BASE models
tox -e omni_tf444      # dedicated models (emu3 / janus_pro / deepseek_vl) only
tox -e omni_vllm       # vLLM serving backend (optional)
```

`omni_tf457` already bundles `lm_eval` and `vlmeval` — no separate engine env. To narrow to one engine, pass its test path after `--` (e.g. `tox -e omni_tf457 -- tests/evaluation/vlm_eval_kit/ --runslow`).

Add a **new** environment — copy a `[testenv:...]` block in `tox.ini`; it inherits base `[testenv]`:

- `commands_pre`: clone its submodule + pin its transformers version.
- `commands`: narrow to its test paths.
- e.g. `omni_tf444` = base + Janus / DeepSeek-VL / Emu3 clones + `transformers==4.44.0`.

### By file

Anything after `--` is passed through to `pytest`, so target a file or test directly:

```bash
tox -e omni_tf457 -- tests/utils_/test_string.py
tox -e omni_tf457 -- tests/schemas/test_task.py -k roundtrip
tox -e omni_tf457 -- tests/api/openai/             # whole directory
```

`slow` / `very_slow` tests are skipped by default. Opt in with:

```bash
tox -e omni_tf457 -- tests/ --runslow        # 1~10 min tests
tox -e omni_tf457 -- tests/ --runveryslow    # 10 min+ tests
```

### Pipelines (opt-in configs)

Pipeline e2e (`tests/pipelines/test_pipeline.py`):

- Default: runs configs under `tests/configs/pipeline/default/`.
- `--pipeline-config <path>` (repeatable): runs optional configs instead — default configs then **not** run.
- Needs `LOCAL_DATA_DIR` (else skip); external-engine configs (`lm_eval` / `lmms_eval` / `vlm_eval_kit`) run only where that extra is installed.

```bash
# A single optional config (hyperclovax_vision builtin — bundled in omni_tf457)
tox -e omni_tf457 -- tests/pipelines/test_pipeline.py \
    --pipeline-config=tests/configs/pipeline/optional/hyperclovax_vision_builtin.yaml --runslow

# Every config under optional/
tox -e omni_tf457 -- tests/pipelines/test_pipeline.py --pipeline-config=all --runslow
```

## Details

### Path layout

`tests/<X>/` mirrors `omni_evaluator/<X>/` 1:1, plus dedicated folders for cross-cutting concerns:

```
tests/
├── conftest.py        # cross-cutting only — markers, options, .env.test, skip hooks
├── configs/           # operational yaml, single source of truth (live smoke reads these)
│
│   # ── source mirror (each module on its own) ──
├── api/  clients/  schemas/  submission/  utils_/  postprocess/  modules/
├── inference/         # per inference engine (api / vllm / sglang / huggingface)
├── evaluation/        # per eval harness (builtin / lm_eval / lmms_eval / vlm_eval_kit)
│
│   # ── cross-cutting ──
├── interfaces/        # sibling implementations honor the same contract
├── engine_parity/     # same input → same output across engines
├── pipelines/         # infer → eval e2e
└── entrypoints/       # `python run.py` CLI as a subprocess
```

### Design principles

- **Source 1:1 mirroring** — `omni_evaluator/<X>/` → `tests/<X>/`; unit checks mock out external resources.
- **Cross-cutting → own folders** — interfaces / parity / pipelines / entrypoints kept apart from the mirror.
- **Fixtures in the narrowest conftest** — promote only when 2+ sibling areas use them; top-level holds only markers / options / `.env.test` / skip hooks.
- **Sibling impls share a base test class** — verified against one contract (e.g. `EvaluationEngineCommonTests`); children override fixtures only.

### Markers

Markers are split into three categories that never overlap.

**Domain taxonomy** (search/filter tags, never auto-skip):

| Marker | Values |
|---|---|
| `inference_engine(name)` | `api` \| `vllm` \| `sglang` \| `hf` |
| `eval_engine(name)` | `builtin` \| `lm_eval_harness` \| `lmms_eval` \| `vlm_eval_kit` |
| `model_size(name)` | `small` \| `medium` \| `large` \| `xl` |

**Environment requirements** (auto-skip when the environment doesn't match):

| Marker | Skips when |
|---|---|
| `requires_gpu` | CUDA unavailable |
| `requires_multi_gpu(n=2)` | fewer than `n` GPUs |
| `requires_multi_node` | not a multi-node environment |
| `requires_env(*vars)` | `.env.test` missing, or any listed var is unset |
| `requires_hf_token` | `HF_TOKEN` not set |
| `requires_extra(*pkgs)` | a listed import-name package is not installed |
| `available_gpu(*names)` | current GPU model isn't one of `names` |

**Runtime characteristics** (skipped by default, opt-in):

| Marker | Meaning |
|---|---|
| `slow` | 1~10 min; run with `--runslow` |
| `very_slow` | 10 min+; run with `--runveryslow` |
