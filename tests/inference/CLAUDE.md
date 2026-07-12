# tests/inference — Engine test conventions

For overall operating rules (markers / conftest / running / naming §7), see `tests/CLAUDE.md`; for structural design intent, see `tests/DESIGN.md`. This document contains only the additional rules that apply exclusively to **inference engine tests**.

---

## 1. Core principles

### 1.1 Call the original implementation directly
Tests **import and call `main(...)` from `omni_evaluator/inference/<engine>/engine.py` directly without reimplementing it**. This ensures that when orchestration logic changes, those changes are immediately reflected in tests, making maintenance natural.

### 1.2 Fake responses are created at exactly one "boundary" line
To avoid reaching actual network/models, only the following *single point* is `monkeypatched`. Deeper internals (provider SDK, transformers, vLLM internals) are not touched.

| Engine | Boundary | Call unit |
|---|---|---|
| api | Per-engine batch inference boundary (sync/async two branches) | batch (entire messages_list at once) |
| vllm | Batch inference boundary (chat ↔ completion, sync ↔ async branches). The entry `healthcheck` is also monkeypatched in an autouse fixture to block side effects. | batch (entire messages_list at once) |
| huggingface | Per-record inferencer (class instance) | per-record (1 call per record) |
| sglang | Batch inference boundary (future) | batch |

The key distinction is **batch boundary (entire messages_list in one call) vs per-record boundary (one call per record)** — this shape difference is intentional and should be preserved, but the exact boundary symbol name, signature, and monkeypatch target are the source (`omni_evaluator/inference/`) and the corresponding `test_engine.py` as single source of truth.

Since each engine has a different boundary shape, child classes provide the `patch_boundary(output_fn=None) -> calls` fixture to abstract this difference so the base test does not need to know about it (see §1.5). The *shape* of fake responses is built exactly as the original expects — the dict format that `engine.main` writes back to the record (prediction / tool_calls / latency etc. — exact fields: source is SoT).

```python
# Example patch_boundary implementation for API/vLLM child — monkeypatches the batch function
# and records calls by expanding messages_list into per-record call logs
@pytest.fixture
def patch_boundary(self, monkeypatch, fake_inference_output):
    def _patch(output_fn=None):
        if output_fn is None:
            def output_fn(idx, messages):
                return fake_inference_output(prediction=f"pred-{idx}")
        calls = []
        def _fake_sync(*a, **kw):
            messages_list = kw["messages_list"]
            results = []
            for i, msgs in enumerate(messages_list):
                calls.append({"messages": msgs, "kwargs": kw})
                try:
                    results.append(output_fn(i, msgs))
                except Exception:
                    results.append(None)
            return results
        monkeypatch.setattr(engine, "<batch boundary>", _fake_sync)  # actual attr name: see source/test_engine.py
        return calls
    return _patch
```

HF child monkeypatches the per-record inferencer class itself as a fake instead of a batch function — one instance call = one record. The code pattern follows the same `patch_boundary` signature (`(output_fn=None) -> calls`).

### 1.3 Live calls coexist in the same file with markers
Within the same `test_engine.py`, mock unit tests and live smokes marked with `@pytest.mark.requires_env("OPENAI_API_KEY")` can coexist. Mock cases are deterministic regression safety nets that run in any environment; smokes are connectivity verifications that run only when the environment is set up. (Marker behavior: `tests/CLAUDE.md §1.2`)

Since live smokes are often meant for humans to see how responses actually come back, **each smoke method should `print` prediction / latency / api_name etc.**. They are hidden by pytest's default capture on pass, so run with `-s` when you want to see them:

```bash
pytest -s tests/inference/api/test_engine.py::TestApiEngineMain::test_openai_smoke_text
```

On failure, the captured output is automatically exposed along with the error, so no extra flags are needed.

**Live smoke image/video inputs use URL fixtures.** Mock unit tests have monkeypatched boundaries that do not reach schema conversion, so the PIL/ndarray-based `image_record_factory` / `video_record_factory` (`tests/inference/conftest.py`) are appropriate for verifying content type preservation up to the boundary. Live smokes go through the actual flow from schema → provider/endpoint, so it is standard to use the URL-based record factory (same conftest) that matches the form where production datasets put a string (URL or file path) in the value of image/video content.

URL selection note: provider image fetch backends have per-domain policies, and even the same https URL may be rejected by some domains (observed: COCO/Wikimedia rejected, Unsplash allowed — on OpenAI). When swapping URLs, quickly verify with curl that all four endpoints (OpenAI/Anthropic/Google/vLLM) can fetch the URL before applying.

### 1.4 `engine.main()` is always called with `debug=True`
Regardless of mock-based unit tests, live smokes, parity, or e2e — always call `engine.main()`'s `debug` argument as **True**. In debug mode, only up to the maximum number of records (a production constant) are processed (exact constant name/value: `omni_evaluator/inference/` is SoT):

- **Live smokes**: even if yaml/records bloat or a mock leaks, token consumption cannot exceed that upper bound → prevents API billing accidents.
- **Mock tests**: a runaway safety net against regressions where a misaligned monkeypatch allows external calls to leak out.

Baking `debug=True` into the `main_kwargs_factory` baseline means all inherited / self tests automatically follow — this baseline pattern is the mechanism for consistent rule application (`tests/CLAUDE.md §5`).

Tests that verify debug behavior itself (`test_debug_truncates`) explicitly restate `debug=True` at the call site to expose the intent "this test is looking at the debug path".

### 1.5 Common contracts are consolidated in an abstract base class
`engine.main()` follows **common contracts** that are engine-agnostic. These contracts are collected in the `EngineMainCommonTests` base class in `tests/inference/test_engine_common.py`, and each engine's `test_engine.py` inherits from it, overriding only fixtures to **automatically obtain an identical regression safety net**.

The base does not depend on boundary shape — as long as the child provides the boundary abstraction via a `patch_boundary(output_fn=None) -> calls` fixture, all contract tests in the base work identically whether it is a batch engine or a per-record engine (boundary comparison table in §1.2).

**6 universal contracts verified by the base** (regardless of batch/per-record):
1. `prediction`/`tool_calls`/`latency` are written back to records
2. `debug=True` → boundary is reached only up to the debug limit
3. Partial failure (some slots are `None`/raise) → only valid slots are filled and total record count is preserved
4. Input preservation in output dict (`benchmark`/`index`/`messages`) + common fields to be filled (`prediction`/`tool_calls`/`latency`)
5. image+text content reaches the boundary without type loss
6. video+text content reaches the boundary without type loss

**Contracts meaningful only for batch engines** are left out of the base and placed as individual methods in each batch engine (cannot occur in per-record engines like HF):
- Empty batch response → `engine.main()` returns `None`
- Entire batch function raises → `engine.main()` returns `None`

```python
# tests/inference/<engine>/test_engine.py
import pytest
from omni_evaluator.inference.<engine> import engine
from tests.inference.test_engine_common import EngineMainCommonTests


@pytest.mark.inference_engine("<engine>")
class Test<Engine>EngineMain(EngineMainCommonTests):
    @pytest.fixture
    def engine_module(self):
        return engine

    @pytest.fixture
    def main_kwargs_factory(self):
        def _f(records, task_config, **overrides):
            base = dict(..., debug=True)   # §1.4 safe default
            base.update(overrides)
            return base
        return _f

    @pytest.fixture
    def patch_boundary(self, monkeypatch, fake_inference_output):
        """Monkeypatches the engine's boundary with a fake. (output_fn=None) -> calls.
        If output_fn(idx, messages) returns None/raises, that slot fails.
        calls is a per-record call log — batch engines expand messages_list and record per record.
        """
        ...  # see §1.2 example

    # Engine-specific behavior / batch-only contracts / live smokes are added as methods in the same class
    def test_empty_batch_returns_none(self, ...):  # batch engines only
        ...
    def test_<engine_specific_behavior>(self, ...):
        ...
```

Because the base class name does **not** start with `Test`, pytest does not collect the base directly. Only child classes are collected, and inherited common tests run in the child's context — the child's `@pytest.mark.inference_engine("<engine>")` marker is also automatically applied to inherited methods.

**What goes in the base vs what goes in the child:**
- **Add to base** → universal contracts that all engines (batch/per-record) must satisfy equally (the 6 above).
- **Put in child** → (a) contracts meaningful only for batch engines (empty batch/batch raise → `None`), (b) engine-specific branches (API's async ↔ sync, vLLM's chat ↔ completion and healthcheck etc. — exact argument names that trigger branches: source is SoT), (c) live smokes.

If an engine-specific response dict has additional fields beyond the common keys (e.g., vLLM/HF's reasoning / perplexity types — exact keys: source is SoT), just override `fake_inference_output` in the child class.

---

## 2. Checklist for adding a new engine test

1. Create `tests/inference/<engine>/__init__.py` (if absent).
2. Define `Test<Engine>EngineMain` class inheriting from `EngineMainCommonTests` in `tests/inference/<engine>/test_engine.py`.
3. **3 required fixture overrides**: `engine_module`, `main_kwargs_factory`, `patch_boundary`. (Also `fake_inference_output` if the response dict shape differs.) **`debug=True` must be baked into the `main_kwargs_factory` baseline** (§1.4). The `patch_boundary` signature is `(output_fn=None) -> calls` — all base contracts operate on top of this abstraction (§1.2 example, §1.5 explanation).
4. Attach `@pytest.mark.inference_engine("<engine>")` to the class (automatically applied to all inherited/self methods). Also attach `@pytest.mark.requires_gpu` if GPU is needed.
5. **Monkeypatch only one boundary line** — do not go deeper to directly patch SDK / external libraries.
6. Engine-specific behavior (async branches, per-modality generate, etc.) is added as methods in the same class.
7. If live smokes exist, add them as `@pytest.mark.requires_env(...)` or `@pytest.mark.requires_gpu` methods in the same class. Do not split into a separate file.
8. If an engine-specific fixture is reused across two or more cases, extract it to that engine's conftest (`tests/CLAUDE.md §3` residence principle).
9. **Path/file constants go at the top of the file** (yaml locations, smoke queries, token limits, etc.). Only one place to update when directories move.
10. **Put a "Fixture source map" in the class docstring** — where each fixture comes from: (a) this class override, (b) base class default implementation, (c) which `conftest.py`, or (d) pytest builtin. Method signatures alone are not traceable, so collect them in one place per class. CLI for verification: `pytest --fixtures <file>`.
