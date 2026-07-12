# tests/evaluation — Harness test conventions

Overall operating rules (markers / conftest / execution / naming §7 / area documentation §10) are in `tests/CLAUDE.md`; structural design intent is in `tests/DESIGN.md`. This document contains only the additional rules that apply exclusively to **evaluation area tests**.

This area's tests fall into two branches — they differ in what they verify, their mock policy, and their residency rules:

| Branch | Location | What is verified | Rules |
|---|---|---|---|
| **Harness engine** | `tests/evaluation/<harness>/test_engine.py` | Are the 3-stage entry points of `engine.py` exported + does 1 cycle run with a real dataset | §1 ~ §3 |
| **metric functions** | `tests/evaluation/metrics/test_*.py` | Do the metric functions in `evaluation/metrics/` produce *accurate scores* | §4 |

The two branches are connected by the "no deep mock" rule in §1.3 — engine tests do *not patch* the metrics inside `evaluate_task`; metric accuracy is the *direct* responsibility of metric tests.

---

## 1. Harness engine tests — core principles

### 1.1 Call the original implementation as-is
Tests **import and call the functions** from `omni_evaluator/evaluation/<harness>/engine.py` without reimplementing them. When production code changes, those changes should immediately be reflected in the tests for natural maintainability (same principle as `tests/inference/CLAUDE.md §1.1`).

### 1.2 The evaluation harness has a 3-stage entry point — not a single `main()`
The inference engine has a single `engine.main()` entry point, but the evaluation harness `engine.py` **exports the same 3-stage role across multiple harnesses (`builtin`, `lmms_eval`, `lm_eval_harness`, `vlm_eval_kit`)** — however, *the roles are the same but the signatures are not identical*. Some argument names and details differ per harness.

| Role (stage) | Responsibility | Output |
|---|---|---|
| task_config creation | Assembles benchmark metadata (task_config) — internal helper, not called directly by tests | task_config object |
| record yield | Dataset loading → Record iterator + task_config | Record iterator and task_config |
| metric aggregation | record + prediction → metric aggregation | run output (+ per-record output) |

The three roles accept arguments for *task identity* (which benchmark), *prompt injection*, *aggregation method*, *debug*, etc. However, some argument names differ per harness — some harnesses have different names for the task identifier argument or task_config argument, and some harnesses accept additional arguments. **These variants are not absorbed by the base contract** (§1.6) — the only place that actually calls the variants is each harness's live smoke, where the argument names are directly hardcoded.

> The exact function names, signatures, argument names, and which harness uses which variant are single source of truth in the source (`omni_evaluator/evaluation/<harness>/engine.py`) and the corresponding `test_engine.py`.

### 1.3 There is no single mock boundary — responsibility is split between "export contract" and "live smoke"
Inference had a clean single boundary like `batch_chat_completion_*` where monkeypatching just that one line sufficed, but evaluation's `evaluate_task` has scattered metric functions / external packages (`lmms-eval`, `lm-eval-harness`, `vlmeval`) / judge LLM calls, so **a single boundary does not exist**. Patching deep into metric functions makes tests fragile and hard to maintain.

So verification is split into two layers:

| Layer | What is verified | Cost | Who |
|---|---|---|---|
| **export contract** | Are the 3-stage entry points (`get_data_iterator` / `evaluate_task`) exposed as callable attributes | Free, always passes | Base (`test_engine_common.py`) |
| **Live smoke** | Dataset loading + evaluate 1 cycle with the yaml benchmark (§3) | Requires extras + dataset + tokens | Child (`<harness>/test_engine.py`), `slow` gate |

> ⚠️ **The static contract goes only as far as "whether exported".** In the past, the base pinned signature arguments and return annotations, but that static check was intentionally removed (commit *"evaluation common tests simplified"*) — static markdown/annotation pinning produces false alarms on every refactor, whereas **live smoke transitively guarantees signatures and return types through actual execution**. What the base verifies is only "have the two entry points not disappeared".

**Do not mock-patch metric functions inside `evaluate_task`.** If you need to verify metric accuracy, unit-test the metric function directly in `tests/evaluation/metrics/` (§4) — bypassing `evaluate_task` is the right approach.

> ⚠️ **Scope of "no deep mock"** — this rule means *do not monkeypatch metric functions / judge LLM / external lib calls inside `evaluate_task`*. **It does not mean "do not unit-test pure helper functions in `engine.py`"** — unit tests for those helpers are covered separately in §1.7.

**Exception — `builtin` harness goes one layer deeper (§1.3.1).** `builtin` is not coupled to an SDK, and the *dataset loading point* and *judge evaluation point* are cleanly exposed as module attributes, so monkeypatching just those two points makes record iteration / per-record metric assembly / final aggregation run deterministically. **Deterministic text metrics (e.g., `exact_match`) are run for real without mocking** — this is to verify orchestration and metric accuracy in one pass.

### 1.3.1 Deep mock boundary for `builtin` harness
The `builtin` harness test monkeypatches only *two points* on the `engine.py` module — to simultaneously verify deterministic metrics and orchestration while cutting off external nondeterminism:

| Mocked point | Reason for mocking |
|---|---|
| dataset loading | Cuts off the entire dataset download / multimodal reconstruction stage. Replaced with a fake loader that yields the fixture's record list as-is (`conftest.py::fake_dataset_iterator_factory`). |
| judge evaluation (LLM judge) | Blocks LLM judge calls. When judge-family metrics are not requested, the mock should not be reached — verified with a call counter. |

**The text metric evaluation point is not mocked** — it is left as a deterministic metric, and two cases created by the fixture — *prediction matches / does not match the answer* — verify deterministic scores.

The path that re-reads task_config from yaml is also monkeypatched to return the fixture's synthesized task_config as-is, cutting off yaml I/O (paired with `conftest.py::evaluation_task_config_factory`).

> The exact module attribute paths (what is being patched) and patch fixture names are single source of truth in `tests/evaluation/builtin/test_engine.py`.

### 1.4 Environment isolation — extras conflicts are the key constraint
As the "Optional Dependencies" table in `README.md` shows, each harness's extras (`lmms_eval`, `lm_eval`, `vlmeval`) **cannot be installed simultaneously due to version conflicts**. An environment with only one harness installed at a time is normal, and tests must accommodate this.

Two mechanisms are used together:

| Mechanism | Purpose | Location |
|---|---|---|
| `pytest.importorskip("<pkg>")` | Collection safety net at module import — skip instead of `ImportError` when extras not installed | Top of each harness `test_engine.py` |
| `@pytest.mark.requires_extra("<pkg>")` | For CI / local filtering (`-m "requires_extra('lmms_eval')"`) | Class decorator |

`requires_extra(*pkg_names)` is the environment requirements marker from `tests/CLAUDE.md §1.2` — auto-skips if `importlib.util.find_spec(pkg) is None`. If `pytest.importorskip` is the primary safety net for module collection, the marker is the **core mechanism for CI matrix branching via marker filtering**.

The argument passed to the marker is the **Python package name** (argument for `find_spec`), not the extras name in `pyproject.toml`. Mapping based on the README table:

| extras name (`pyproject.toml`) | Python pkg (`find_spec` argument) | Applied harness |
|---|---|---|
| `lmms_eval` | `lmms_eval` | lmms_eval |
| `lm_eval` | `lm_eval` | lm_eval_harness |
| `vlmeval` | `vlmeval` | vlm_eval_kit |

The `builtin` harness has no extras (importable with base installation only) — both `pytest.importorskip` and `requires_extra` are unnecessary.

> Metric tests (§4) use the same mechanism — e.g., `importorskip` at the top of a file to skip at collection stage when the CLIP stack (`clip_benchmark` / `open_clip`) for image metrics is not installed.

### 1.5 Record yield / evaluate calls always cut with `debug=True`
Whether in mock-based deep tests or live smoke, calls where records flow are processed **only up to the first `NUM_DEBUG_SAMPLES` (=3)** — to prevent live smoke from unintentionally exhausting tokens/time or judge LLM from running out of control (`tests/CLAUDE.md §5`).

Two approaches vary by harness:
- Calls that **accept** a `debug` argument → `debug=True`.
- Variants where the record yield stage **does not accept** a `debug` argument → the caller cuts directly with `itertools.islice(iterator, NUM_DEBUG_SAMPLES)`.

Whether a harness accepts the `debug` argument follows the source signature.

### 1.6 Common contract is an abstract base class — but *export only*
Whether the 3-stage entry points are exported is the common contract for all harnesses. This contract is collected in the `EvaluationEngineCommonTests` base class in `tests/evaluation/test_engine_common.py`, and each harness's `test_engine.py` inherits it so that **overriding just the `engine_module` fixture** provides the same regression safety net.

```python
# tests/evaluation/<harness>/test_engine.py
import pytest
pytest.importorskip(                              # §1.4 collection safety net
    "<pkg_name>",
    reason="install with `pip install -e \".[<extra>]\"`",
)
from omni_evaluator.evaluation.<harness> import engine
from tests.evaluation.test_engine_common import EvaluationEngineCommonTests


@pytest.mark.eval_engine("<harness>")
@pytest.mark.requires_extra("<pkg_name>")         # §1.4 CI filter
@pytest.mark.timeout(60)
class Test<Harness>EvaluationEngine(EvaluationEngineCommonTests):
    @pytest.fixture
    def engine_module(self):
        return engine

    # Base automatically provides export contract. Children only add live smoke (§3) + harness-specific.
```

Because the base class name does **not** start with `Test`, pytest does not collect the base directly — only children are collected. The child's `@pytest.mark.eval_engine` / `requires_extra` markers automatically apply to inherited base methods as well (same mechanism as `tests/inference/CLAUDE.md §1.5`).

**What goes in the base vs. what goes in children:**
- **Base** → export contract only (have the entry points not disappeared).
- **Children** → live smoke (§3), harness-specific branches (`builtin`'s deep mock §1.3.1, `lmms_eval`'s `task_manager` flow, etc.), helper unit tests (§1.7).

> ⚠️ **Argument name variants (`vlm_eval_kit`'s `dataset_name` / `benchmark_config`, etc.) are not absorbed by the base.** In the past, the base contract handled dynamic key lookup via fixture overrides (`evaluate_task_identifier_arg` / `task_config_arg`), but that mechanism was removed when it was simplified to export-only. The only place that actually calls variants is the child's live smoke — it calls with the correct argument names **directly hardcoded** there. (Some child test file docstrings' "Fixture source maps" may retain old fixture names, but the base no longer consumes those fixtures — clean them up when found.)

### 1.7 Helper functions in `engine.py` — test only the *minimum contract*
Beyond the 3-stage functions, helpers in `engine.py` (sample → record conversion, multichoice post-processing, etc.) also have regression safety net value, but **these are *sub-role*** — the main focus is entry point export contract + live smoke. The purpose of helper unit tests is *blocking trivial bugs like NameError / type breakage at the collection stage*, not locking down internal branches in fine detail.

**Does not conflict with §1.3's "no deep mock"** — §1.3 refers only to monkeypatching metric functions / judge / external lib calls inside `evaluate_task`. Helper unit tests are outside that scope. However, keep them as *shallow units* only.

**Core principle — helper internals change frequently:**
- Multimodal content processing, fallback priority, options resolution, generation_kwargs mapping — all are *internal implementation* and are frequently refactored as production datasets change.
- Testing these in fine detail causes **false alarms on every refactor**. Since entry point live smoke transitively goes through all of these, *the effective regression safety net already exists*.

**Writing rules:**
1. **Do not create new classes** — use module-level test functions. Attach `@pytest.mark.eval_engine("<harness>")` directly to functions.
2. **1~2 functions per helper** only. Usually one happy path is enough.
3. **What to verify** — *does the function run* (no NameError) + *returns Record / dict type* + *input identity fields (benchmark / index, metric_name, etc.) are echoed in the output*.
4. **What not to verify** — internal transformation mappings, fallback priority, options resolution result values, exact conversion of multimodal content.
5. **Location** — place as module-level in each harness's `test_engine.py`. Do not create separate `test_<helper>.py` files (source mirroring principle, `tests/CLAUDE.md §6.1`).

**Fixture pattern — use `types.SimpleNamespace` to structurally fake external objects.** No need to import the external lib itself or do deep mocking. Helpers only *read a few attributes / methods* in their body, so a structural fake suffices.

---

## 2. Checklist for adding a new harness engine test

1. Create `tests/evaluation/<harness>/__init__.py` (if it doesn't exist).
2. Define a `Test<Harness>EvaluationEngine` class inheriting `EvaluationEngineCommonTests` in `tests/evaluation/<harness>/test_engine.py`.
3. **Add `pytest.importorskip("<pkg>")` at the top of the module** (§1.4) — skip at collection stage if the external pkg is absent.
4. **Fixture override**: `engine_module` **alone is sufficient** (§1.6). Even if there are argument name variants, do not create additional fixtures for the base — call variants directly hardcoded in the live smoke call site.
5. Attach markers to the class:
   - `@pytest.mark.eval_engine("<harness>")` — domain classification (no auto-skip, for filtering/searching)
   - `@pytest.mark.requires_extra("<pkg>")` — environment isolation + CI matrix filter (§1.4)
6. **Do not go deep into mocks** — do not directly patch metrics / SDK / judge LLM inside `evaluate_task` (§1.3). Metric accuracy is the responsibility of §4.
7. **Add live smoke** (§3) — `slow` + `requires_hf_token` + `requires_extra` gates.
8. If harness-specific fixtures are reused across two or more cases, extract them into that harness's `conftest.py` (`tests/CLAUDE.md §3` residency principle).
9. **Path/file constants should be gathered at the top of the file** (yaml location, etc.). Only one place to edit when the directory moves.
10. **Put a "Fixture source map" in the class docstring** — for each fixture, note whether it comes from (a) this class override, (b) base default, (c) which `conftest.py`, or (d) pytest builtin. The CLI for checking is `pytest --fixtures <file>`. **Do not list fixtures that the base no longer consumes** (§1.6 warning).

---

## 3. Live smoke contract

Each harness child class's live smoke is the only execution path that verifies "does this harness run 1 cycle with a *real dataset*" (the effective guarantee of signatures and return types after the static signature pinning was removed in §1.3).

**Gates** — attach all three markers (triple protection against external resource consumption):
- `@pytest.mark.slow` — deselected from default execution, only active with `--runslow` (`tests/CLAUDE.md §1.3`).
- `@pytest.mark.requires_extra("<pkg>")` — only in environments with that harness's extras installed (§1.4).
- `@pytest.mark.requires_hf_token` — dataset download token required.
- `@pytest.mark.timeout(...)` — increase timeout since sequential benchmark verification exceeds the class default of 60s.

**Target** — **iterate over all benchmarks listed** in `benchmarks` of the yaml (`tests/configs/evaluation/<harness>.yaml`) (not just the first item). The yaml is the single source of truth for benchmarks — do not hardcode benchmark names in tests.

**Cycle per benchmark**:
1. `get_data_iterator(...)` → `(iterator, task_config)`. If the variant does not accept a `debug` argument, cut with `itertools.islice` to `NUM_DEBUG_SAMPLES` (§1.5).
2. Generate dummy records of the truncated length with `fake_inference_record_factory` → `evaluate_task(..., debug=True)`.

**Verification items (this is all — do not assert metric values themselves)**:
1. `task_config.num_records > 0` — is the dataset metadata non-empty.
2. Did the iterator yield ≥1 record (truncated length ≥ 1).
3. Is the run output's `.metrics` from `evaluate_task` a non-empty dict.

The accuracy of metric *values* is the responsibility of §4, so it is not verified here — live smoke only checks "does the pipeline flow to the end and return the promised shape".

> Argument name variants (§1.2) are hardcoded directly at this call site — for example, a harness with a different task identifier argument or task_config argument name calls it with its own argument names in its live smoke. The exact argument names are SoT in the source and corresponding `test_engine.py`.
