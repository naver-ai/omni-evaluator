# tests/utils_ — Utility module test conventions

For overall operating rules see `tests/CLAUDE.md`, and for structural design intent see `tests/DESIGN.md`. This document contains additional rules that apply **only** to the **`omni_evaluator/utils/`** area (shared utilities across 9 modules).

> **Folder name**: `tests/utils_/` (trailing underscore). `utils` conflicts with Python module names in some environments, so this avoids that.

---

## 1. Area responsibilities

Modules under `omni_evaluator/utils/` are **primarily pure functions**. External resources are limited to a few modules (`multimodal.py`, healthcheck/dynamic loading in `common.py`).

The table below lists only the **responsibility category** and **external resources each module touches** (mock boundary intent). Exact function names, signatures, and I/O are the sole responsibility of the source (`omni_evaluator/utils/`) and the corresponding `test_*.py` files.

| Module | Responsibility category | External resources |
|---|---|---|
| `io.py` | Multi-format file I/O (JSON/YAML/CSV/Excel/WAV/Pickle), path helpers | Filesystem |
| `common.py` | Engine/task enumeration, custom module dynamic loading, healthcheck, seed fixing | `importlib`, network (`requests`) |
| `string.py` | URL/number validation, JSON & literal parsing, base64, format key extraction | None |
| `data.py` | Function argument filtering, option parsing & cycling, distributed data splitting, prompt formatting | None |
| `multimodal.py` | Image/audio/video load, convert, normalize + format detection (magic bytes) + SSRF security | `PIL`, `librosa`, `av`, `pydub`, network (`requests`) |
| `resource.py` | CPU/GPU memory queries, CUDA device info, distributed resource splitting | `torch.cuda`, `psutil` |
| `torch.py` | dtype interpretation, CUDA capability, tensor comparison | `torch` |
| `patches.py` | Temporary patching of modules/functions/methods/environment variables (context manager) | Direct `sys.modules` mutation |

---

## 2. Mock boundaries — differ per module

Most are pure functions. Mock boundaries are only clearly defined for modules with external resources. The following is an intent mapping of *which resource is blocked/replaced with what* — the exact patch target symbols are the sole responsibility of the source and `test_*.py` files.

| Module | Mock boundary (external resource) | Recommended tool |
|---|---|---|
| `io.py` | Filesystem read/write | `pytest tmp_path` fixture (real temporary directory) |
| `common.py` (healthcheck) | Network requests | Replace HTTP client with `monkeypatch` |
| `common.py` (dynamic loading) | `importlib` module loading | Inject fake module into `sys.modules` or use monkeypatch |
| `multimodal.py` (URL loading) | Network requests | `monkeypatch` + byte fixture |
| `multimodal.py` (codec) | `PIL` / `librosa` / `av` / `pydub` | Where possible, *call the real library* + small fixture bytes (fast since processing is in-memory only) |
| `resource.py` | CUDA device count / system memory queries | `monkeypatch` + fixed values |
| `torch.py` | CUDA capability queries | `monkeypatch` + fixed values |
| `patches.py` | Its own behavior (state restoration) | Verify real patch/restore cycles |

`tmp_path` and small byte fixtures are the primary tools — deterministic unit verification is often possible without any mocks.

---

## 3. Verification depth — 2 layers

| Layer | What is examined | Mock | Time |
|---|---|---|---|
| **L1 deterministic unit** | Pure functions (string / data / some io / patches) | Almost none | < 100 ms |
| **L2 mock boundary unit** | External resources (network / GPU / codec) | monkeypatch / tmp_path | < 1 s |

Some functions in `multimodal.py` are fast enough to call the real codec with *small fixture bytes* (in-memory). In that case, use `@pytest.mark.requires_extra("PIL", ...)` to skip when the lib is absent.

---

## 4. Folder structure — 1:1 source mirroring

```
tests/utils_/
├── conftest.py                # shared fixtures (tmp_path helper, byte fixtures)
├── test_io.py                 # multi-format file I/O + path helpers
├── test_common.py             # engine/task enumeration / healthcheck / seed fixing
├── test_string.py             # URL/number validation / parsing / format key extraction
├── test_data.py               # argument filtering / distributed splitting / option parsing
├── test_multimodal.py         # image conversion / format detection / MIME / SSRF
├── test_resource.py           # memory queries / CUDA device count / resource splitting
├── test_torch.py              # dtype interpretation / capability / tensor comparison
└── test_patches.py            # context manager + cleanup verification
```
