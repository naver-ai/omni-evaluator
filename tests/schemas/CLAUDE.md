# tests/schemas — Dataclass schema test conventions

For overall operating rules see `tests/CLAUDE.md`, and for structural design intent see `tests/DESIGN.md`. This document contains additional rules that apply **only** to the **`omni_evaluator/schemas/`** area (5 dataclass-based schemas).

Schemas are shared data structures that *every* area depends on, so regressions spread most widely here — this must be the most *deterministic and fast* unit test area.

---

## 1. Area responsibilities

All files under `omni_evaluator/schemas/` are **dataclass + serialization helpers**. The domain each module covers:

| Module | Domain |
|---|---|
| Base | Serialization / dict-like interface shared by all schemas |
| `chat.py` | Conversation messages + content types (text / image / audio / video) + tool calls |
| `task.py` | Task metadata / dataset / config / evaluation definitions + per-engine factories |
| `evaluation.py` | Evaluation run output + statistics |
| `inference.py` | Inference records + engine capability flags |
| `generation_options.py` | Generation/sampling options base + per-engine/provider children |

All are **pure function areas** — almost no external resource calls (exceptions in §2).

The exact class names and method lists are single source of truth in the source (`omni_evaluator/schemas/`) and the corresponding `test_*.py`.

---

## 2. Mock boundaries — almost none

The schemas themselves do not touch external resources. Verification is fundamentally about deterministic mapping of *input → output*.

Rare cases where mocking is needed:

| Behavior | External boundary | Mock |
|---|---|---|
| Multimodal content value extraction (image/audio/video) | Codec conversion calls (PIL / audio bytes etc.) | Byte fixtures or monkeypatch utils |
| Record serialization + optional tokenizer | Tokenizer calls | Lambda or fake tokenizer |
| Engine string → factory dispatch | Dynamic import boundary (engine subclass / builtin loading) | `sys.modules` stub or monkeypatch |

Exact symbols, signatures, and values: source (`omni_evaluator/schemas/`) and the corresponding `test_*.py` are the single source of truth.

L1 deterministic unit tests alone can cover most branches — mocking should be *minimized*.

---

## 3. Verification depth — L1 single layer

| Layer | What is verified | Mock | Time |
|---|---|---|---|
| **L1 deterministic unit** | Dataclass creation / serialization / deserialization / dispatch / field transformation | Almost none | < 100 ms |

L2/L3 (mock boundary / live) are unnecessary for the schemas area — no external resources.

---

## 4. Folder structure — 1:1 source mirroring

```
tests/schemas/
├── conftest.py                    
├── test_schemas_common.py         ← Common contract base (uncollected mixin) + its self-verification probe — one file (§5)
├── test_chat.py                   
├── test_task.py                   
├── test_evaluation.py             
├── test_inference.py              
└── test_generation_options.py     
```

One `test_<name>.py` per source file. The largest `chat.py` is consolidated in one file, but class separation per content type is recommended.

`test_schemas_common.py` is not a source mirror but *cross-cutting common infrastructure* — other `test_*.py` files import contract bases from here. Bases (prefixed with `_`) are not collected, and only the probe self-verifications (`Test*`) in the same file are collected (§5).

---

## 5. Common contract tests — base inheritance per interface

Inherits interfaces shared across multiple schemas (base serialization / content access / generation options / inference output etc. — the list is SoT in source). Whether *every implementation honors the same contract* is written once in **contract base classes** in `tests/schemas/test_schemas_common.py` rather than scattered across individual `test_*.py`, so each `TestX` inherits (or parametrizes a sibling list). When a new implementation is added, just filling in the fixture automatically attaches contract verification (extension safety net).

**Self-verification probes** (`Test*`) for the bases are also kept in the same file — since the bases assume behavior of the actual interfaces (`SchemaInterface` / `ContentInterface`), if an interface changes the probe breaks and forces the base to be updated as well (contract ↔ interface coupling). Base classes are prefixed with `_` so pytest does not collect them as classes; only the probe's `Test*` are collected and executed.
