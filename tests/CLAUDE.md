# tests/ — Test operating guide

For the design intent behind the test *structure* (folder split, role separation of cross-cutting concerns), see `tests/DESIGN.md`. This document defines the operating rules to follow when **adding / running / maintaining** tests on top of that structure. Area-specific rules are in each subdirectory's `CLAUDE.md` (e.g., `tests/inference/CLAUDE.md`).

---

## 1. Markers

Registered in `tests/conftest.py`, strictly separated into three categories. A single marker must not carry the meaning of two categories simultaneously.

### 1.1 Domain classification — no automatic skip
Used as tags. For search and filtering.

| Marker | Values | Purpose |
|---|---|---|
| `inference_engine(name)` | `"api"` / `"vllm"` / `"sglang"` / `"hf"` | Which inference engine |
| `eval_engine(name)` | `"builtin"` / `"lm_eval_harness"` / `"lmms_eval"` / `"vlm_eval_kit"` | Which evaluation harness |
| `model_size(name)` | `"small"` (<3B) / `"medium"` (3-13B) / `"large"` (13-70B) / `"xl"` (>70B) | Model size |

Multi-value allowed: parity cases like `@pytest.mark.inference_engine("vllm", "hf")`.

> **`modality` marker is deprecated.** Modality is not a manually written tag but is determined from the **single source of truth in the source**. HF adapter tests let `MODULE_CLS.ENGINE_FEATURES`
> (`support_*_understanding` flags in `omni_evaluator/schemas/inference.py`) decide which
> generate-* tests to *collect* (`__init_subclass__` in `tests/inference/huggingface/test_huggingface_adapter_common.py`).
> Since modality is not redundantly declared on the test side, source-test
> drift is prevented at the root.

### 1.2 Environment requirements — automatic skip
If the environment does not match, conftest automatically skips at collection time.

| Marker | Behavior |
|---|---|
| `requires_gpu` | Skip if `torch.cuda.is_available()` is false |
| `requires_multi_gpu(n=2)` | Skip if GPU count < n |
| `requires_env(*env_var_names)` | Skip if `.env.test` is absent or any of the specified env vars is empty. Takes **raw env var names** as arguments (e.g., `"OPENAI_API_KEY"`). |
| `requires_hf_token` | Skip if `HF_TOKEN` is not set |
| `requires_extra(*pkg_names)` | Skip if any of the specified Python packages fails `importlib.util.find_spec`. For environments where optional extras in `pyproject.toml` (`lmms_eval`, `lm_eval`, `vlmeval`, model-specific extras, etc.) conflict and cannot be installed simultaneously — serves both to auto-skip tests for uninstalled extras and as a CI matrix filter (`-m "requires_extra('lmms_eval')"`). Arguments are **Python package names to import**, not extras names. It is recommended to also add `pytest.importorskip("<pkg>")` at the top of the file as a module collection safety net. |

**The primary gate for live API tests is the mere existence of the `.env.test` file.** Even if `OPENAI_API_KEY` is exported in the shell, tests are unconditionally skipped if `.env.test` is absent — this prevents accidental live calls with personal keys.

```bash
cp .env.test.example .env.test
# Fill in only the required keys in .env.test
pytest tests/inference/api/
```

### 1.3 Execution characteristics — skipped by default, opt-in
| Marker | Duration | How to activate |
|---|---|---|
| `slow` | 1~10 min | `pytest --runslow` |
| `very_slow` | 10 min+ | `pytest --runveryslow` |

---

## 2. Test structure (summary)

For detailed design and intent, see `tests/DESIGN.md`. One-page summary from an operational perspective:

```
tests/
├── conftest.py          ← markers / options / .env.test loading / skip hooks (no domain fixtures)
│
│   # Source 1:1 mirroring — does each module work on its own
├── api/  clients/  schemas/  submission/  utils_/  postprocess/
├── inference/  evaluation/
│
│   # Cross-cutting concerns
├── engine_parity/       ← Does a different engine produce the same output for the same input
├── pipelines/           ← infer → eval e2e (light / heavy / submission)
└── entrypoints/         ← python -m omni_evaluator CLI
```

Area-specific additional rules are defined in each area's `CLAUDE.md`. This document covers only what applies commonly across all areas. **Rules to follow when *writing* an area `CLAUDE.md` are in §9**.

---

## 3. conftest hierarchy — where fixtures live

**Principle**: fixtures go in the *narrowest conftest*. Promote upward only when two or more sibling areas have actually started using that fixture.

```
tests/conftest.py                        # cross-cutting only
                                         #   - markers, options, skip hooks
                                         #   - .env.test loading
                                         #   - no domain fixtures
tests/<area>/conftest.py                 # shared by sibling tests within that area
tests/<area>/<sub>/conftest.py           # fixtures used only by that sub
```

Examples:
- `record_factory`, `task_config_factory` → `tests/inference/conftest.py` (shared by all inference engines)
- vLLM's `SamplingParams` fake → `tests/inference/vllm/conftest.py`
- When a vision-language record is needed in two or more engines, promote to `tests/inference/conftest.py`

**Anti-pattern**: collecting domain fixtures (Record / TaskConfig / fake responses, etc.) in the top-level conftest. It becomes a god-file and a change in one area breaks tests in another.

---

## 4. Running tests

### Basic
```bash
source ~/.omni/bin/activate
pytest tests/                            # fast unit tests only
```

### Time options
```bash
pytest tests/ --runslow                  # includes slow tests
pytest tests/ --runveryslow             # includes all very_slow tests
```

### Live API tests
```bash
cp .env.test.example .env.test           # once at the start
# Fill in OPENAI_API_KEY etc. in .env.test
pytest tests/inference/api/              # requires_env tests become active
```

### Marker filter
```bash
pytest tests/ -m "inference_engine('api')"
pytest tests/ -m "requires_gpu"
pytest tests/ -m "not slow and not very_slow"
```

---

## 5. Safety defaults

Any call that may touch external resources (API / GPU / disk / tokens) must always be made **with safety knobs enabled**. Area-specific baseline fixtures (e.g., the factory in `tests/inference/conftest.py`) bake in these values by default so they apply automatically even if an individual test forgets.

| Area | Knob | Meaning |
|---|---|---|
| inference — `engine.main()` | `debug=True` **always** | Process records only up to `omni_evaluator.inference.NUM_DEBUG_SAMPLES` (=3). Even if a mock leaks or records balloon, external calls will not exceed 3. See `tests/inference/CLAUDE.md §1.4` for details. |

When a new area needs a safety knob, add it to that area's `CLAUDE.md` and register a row in this table.

---

## 6. Adding new tests

1. **Is it a source-mirroring area?** (unit verification of a module under `omni_evaluator/<X>/`)
   - Mirror the same path: `omni_evaluator/inference/api/engine.py` → `tests/inference/api/test_engine.py`
   - Area-specific rules (original call / mock boundary / verification scope) are defined in that area's `CLAUDE.md`.

2. **Is it a cross-cutting concern?**
   - Inter-engine equivalence → `tests/engine_parity/`
   - infer → eval e2e → `tests/pipelines/`
   - CLI entry points → `tests/entrypoints/`

3. **Fixture placement**: follow the residence principle in §3. Place in the narrowest location and promote when needed.

4. **Attaching markers**:
   - Domain classification (§1.1): attach if it helps with search and filtering.
   - If external resources (GPU / API key / multi-node / HF token) are required, §1.2 markers are *mandatory*.
   - If it takes more than 1 minute, also attach `slow` / `very_slow` (§1.3).

5. **Test function naming**: *must* follow the rules in §7.

---

## 7. Test function naming & docstrings

Function names should be **short**; intent and verification scope go in the **docstring**. The output must fit cleanly on one line of `pytest -v` to maintain readability.

### 7.1 Naming
- Pattern: `test_<core_action_or_aspect>` — **1~3 words**.
- Context already provided by the class / file / directory should be **omitted** from the function name.
- Do not use suffixes indicating *depth / thoroughness* (`_minimal`, `_basic`, `_simple`, `_complete`, `_thorough`, etc.) — a single test verifies the contract of that function by definition. Only split when a case of genuinely different depth is added later.

### 7.2 Docstrings
- **First line (required)**: one sentence stating what is verified (present tense, declarative). For happy path / contract verification this is often *all that is needed*.
- **Body (optional)**: non-obvious fixture / mock boundaries, specific incident regression guards, verifications intentionally omitted — only when truly needed, in one line.
- **If there are multiple verification items**, list them as bullets.

**Two prohibitions:**
- Do not rewrite content that already exists elsewhere (policy / design intent / regression cases) — just reference the single source of truth (CLAUDE.md / commit / class docstring) in one line.
- Do not write content that is self-evident from the code (argument meaning / call pattern / data structure shape).

Docstring length is proportional to the *number of branches being verified* — not to the *amount of context being explained*.

### 7.3 Refactoring existing tests
Existing tests with long names / bloated docstrings should be cleaned up according to §7 rules when touching the same file. New tests follow §7 from the start.

---

## 8. Test granularity — one test per branch ⭐

**Principle**: the number of *branches* in a function is the natural upper bound on the number of tests for that function. Do not repeatedly verify the same branch with different inputs.

### 8.1 Why
- **Avoid false confidence**: five tests for a five-line function may *look like sufficient coverage*, but if they re-enter the same branch with different inputs, branch omissions remain intact (test inflation).
- **Maintenance burden**: when a branch disappears or merges, its tests must disappear too — if branches and tests are not 1:1 mapped, it is unclear which tests to delete.
- **Reader cognitive load**: whether 6 tests cover 6 branches or cover 2 branches 6 times — the latter leaves only the question "why 6 times?" which is poorly answered even by a docstring.

### 8.2 Rules
1. **Identify branches first, then write tests**. Enumerate the if / else / early-return / exception paths in the function mentally and create one test for each.
2. **Inputs that fall into the same branch are grouped within one test** — typically one docstring + multiple `assert` lines. Parametrize is often unnecessary.
3. **Trivial tests like `test_imports` are generally unnecessary**. If module import fails, all other tests die at the collection stage, so no separate safety net is needed. *Exception*: the pattern of using `pytest.importorskip` to normally skip environments where extras are not installed (§1.2's `requires_extra`) — this is an environment gate, not an import safety net.
4. **If a different *form* of input creates a different branch, that is a new test**. Example: if empty dict vs missing key both fall into the *same None branch*, they can be grouped in one test. But if they represent *different branches* (e.g., extra processing only when the dict is empty), separate them.

### 8.3 Examples (good / bad)

5-line function — 2 branches:
```python
def get_system_prompt(task_name, system_prompt_map):
    if task_name in system_prompt_map and system_prompt_map[task_name] in SYSTEM_PROMPTS:
        return SYSTEM_PROMPTS[system_prompt_map[task_name]]
    return None
```

❌ Bad — 5 tests (test inflation):
```python
def test_imports(): ...                  # redundant — caught automatically at collection stage
def test_task_not_in_map(): ...          # miss branch 1
def test_mapped_key_missing(): ...       # miss branch 2 — same None branch
def test_resolves(): ...                 # hit branch
def test_empty_map(): ...               # miss branch 3 — same None branch
```

✅ Good — 2 tests (branches = 2):
```python
def test_miss_returns_none():
    """Any lookup failure returns None — three miss paths grouped in one assertion."""
    assert get_system_prompt("missing", {"other": "key"}) is None
    assert get_system_prompt("k", {"k": "__no_such__"}) is None
    assert get_system_prompt("any", {}) is None

def test_resolves():
    """Returns the value from SYSTEM_PROMPTS when both lookups hit."""
    ...
```

### 8.4 What unit to use for branches
A "branch" is identified at the unit of *external variability received as input via the function signature*. The following count as branches:
- if/else / early return / exception paths
- Presence or absence of key fields in an input dict / list (e.g., whether `options` exists)
- Each step in an external fallback chain (e.g., query → doc["question"] → default)
- Per-key conversions in a dict like generation_kwargs (`until → stop_words` is one branch, `max_gen_toks → max_new_tokens` is one branch — but grouping 3-4 keys in one dict assertion is usually cleaner)

The following do *not* count as branches:
- Re-entering the same branch with a different input value (e.g., label="A" vs label="B")
- Whether import is possible (the collection stage verifies this automatically)
- Self-evident type conversion (str → str)

---

## 9. Test function import style & group separation ⭐

Test target functions should be **imported directly** — do not import the whole module and call it as `module.func(...)`. When verifying multiple functions / classes in one file, separate groups with **comment headers**.

### 9.1 Why
- **Declares the function as a public surface**. `from <module> import <func>` is an explicit statement that the symbol is the interface under verification. On rename / move / privatization, the import breaks immediately and signals the regression. `module.func` calls are `getattr` lookups that neither IDE nor import-time catches.
- **Reduces reader cognitive load**. `get_api_group("gpt-4o")` is shorter than `routing.get_api_group("gpt-4o")`, and which symbol is the *verification target* is visible at a glance.
- **Makes the moment of holding a module reference explicit**. The *legitimate* reason to hold a module as an alias is only when **late binding is needed** — i.e., when a module attribute *must change during the test*:
  (a) `monkeypatch.setattr(module, "_internal_helper", ...)` — module attribute patching (helper / dispatch table / mutable cache, etc.)
  (b) A cross-harness base class receives the module via an `engine_module` fixture for dynamic dispatch (`tests/evaluation/CLAUDE.md §1.6`, `tests/inference/CLAUDE.md §1.5`)

  **If a constant, function, or class is being *read only*, import it directly even if it is private (`_FOO`)**. Early binding means no alias is needed.

### 9.2 Rules
1. **Import verification target functions directly**.
   ```python
   from omni_evaluator.api.routing import get_api_group
   assert get_api_group("gpt-4o") == ApiGroup.openai
   ```
   Forbidden:
   ```python
   from omni_evaluator.api import routing
   assert routing.get_api_group("gpt-4o") == ApiGroup.openai
   ```

2. **If a module reference is needed for the late-binding reason in §9.1, add it as an alias** — and use it only for that reason. Function calls use the *directly imported name*:
   ```python
   from omni_evaluator.api import chat_completions as cc_mod                    # for internal dispatch patching
   from omni_evaluator.api.chat_completions import chat_completion_sync         # verification target

   def test_dispatch_openai(monkeypatch):
       monkeypatch.setattr(cc_mod, "chat_completion_sync_openai", _fake)
       chat_completion_sync(api_name="gpt-4o", messages=[...])                  # ← directly imported name
   ```

3. **Separate groups by function / class with comment headers**. When a single file contains multiple verification targets, visually distinguish which tests belong to which target. Follow the conventions already used in the area.
   ```python
   from omni_evaluator.api.routing import get_api_group, get_client, get_engine_features

   # ── get_api_group ────────────────────────────────────────────────
   def test_api_group_openai(): ...
   def test_api_group_unknown_raises(): ...

   # ── get_client ───────────────────────────────────────────────────
   def test_get_client_openai_sync(): ...

   # ── get_engine_features ──────────────────────────────────────────
   def test_engine_features_keys_present(): ...
   ```

### 9.3 Cleaning up existing tests
Existing files with the `module.func()` pattern should be cleaned up according to §9 rules when touching that file. If a module is held as a whole without the late-binding justification from §9.1, replace with direct imports across the board. Only keep a module alias when the monkeypatch in the same file *genuinely* touches module attributes.

---

## 10. Rules for writing area `CLAUDE.md` — *behavior/branches*, not symbol names ⭐

When writing an area-specific `CLAUDE.md` (e.g., `tests/schemas/CLAUDE.md`), **describe the *behavior, branches, and contracts* to verify — do not write concrete source symbols (method names, signatures, exact output dict shapes)**. The single source of truth for concrete values is *the source and the tests* — not `CLAUDE.md`. This applies the docstring rule from §7.2 ("do not rewrite content that already exists") to *area documents*.

### 10.1 Why
- **Documentation does not stay in sync with source.** A compiler does not cross-check markdown against code, so symbol names embedded in prose **silently become lies** the moment a refactoring (rename), unimplemented feature, or signature change occurs. Only a breaking test catches that lie — but documentation never breaks.
- In practice, one area document in this repo had `from_engine_dispatch(engine_name=…)` (actual: `from_engine(evaluation_engine=…)`), `from_dict(api_name, …)` (actual dispatch keys are `api_group`/`inference_engine`), `Message.from_dict(mode=…)` (not implemented in source), and a simple-case output dict shape — **all diverged from the source** — and every single one came from a line that *duplicated a concrete symbol in prose*.

### 10.2 Write two layers separately
| Layer | Write in document? | Example |
|---|---|---|
| **(A) Principles, contracts, branches** | ✅ Yes (the purpose of documentation) | Area responsibility, verification depth (L1/L2), mock boundaries, folder mirroring, pitfalls (§), "this function has dispatch routing + unknown error path" |
| **(B) Concrete source values** | No (source/tests are SoT) | Exact method names, argument order, key names, return dict shape |

### 10.3 Examples (good / bad)

Bad — embedding symbols and signatures reserves drift:
```markdown
- `TaskConfig.from_engine_dispatch(engine_name="builtin", ...)` — engine-specific factory routing
- `ApiGenerationOptions.from_dict(api_name, obj)` — provider-based child selection
```

✅ Good — with *behavior/branches*, delegating concrete values to source and tests:
```markdown
- `TaskConfig` has dispatch routing from engine string → factory and an unknown-engine error path.
- API generation options dispatch a child class via provider identifier (engine/group).
  (Exact method names, arguments, and dispatch keys: source and the corresponding `test_*.py` are the single source of truth.)
```

### 10.4 Two caveats
1. **One anchor symbol for orientation is allowed** — linking one line to help the reader find the source is OK. But that is *orientation*, not *spec*. The executable spec is the tests. Once you start *listing* signatures, keys, and output shapes, you have crossed into (B).
2. **Write "contracts the code must satisfy", but pin them with regression tests.** Example: "statistics average values must be native types, not numpy" is an **intent/contract** worth keeping in the document — if the source violates it, *that is a bug, not documentation drift*. However, do not leave that contract only in prose; also add a corresponding test (`type(...) is float` style) so the test, not the document, catches violations.

### 10.5 Cleaning up existing area documents
Existing area `CLAUDE.md` files with embedded concrete symbols and signatures should be cleaned up by separating (A)/(B) according to §10 rules when touching those files. Replace (B) with behavioral descriptions or delete them, and move live contracts to regression tests.
