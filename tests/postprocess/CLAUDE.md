# tests/postprocess — Prediction post-processor test conventions

For overall operating rules, see `tests/CLAUDE.md`; for structural design intent, see `tests/DESIGN.md §3.2 (flat mirroring)`. This document contains additional rules that apply **only** to the **`omni_evaluator/postprocess/`** area (model prediction normalization / answer extraction).

---

## 1. Area responsibilities

`omni_evaluator/postprocess/` extracts *evaluable answers* from *raw prediction text*.

> **Language scope**: Tests in this area cover **Korean / English** only. Chinese is not a verification target (even if a normalizer exists in the source, no variants are added in tests).

| Module | Processor class | Responsibility |
|---|---|---|
| `_interface.py` | `ProcessorInterface` | Base for all children — `extract(prediction, query, version, api_name, ...)` signature |
| `asr/__init__.py` | `AsrProcessor` | ASR result normalization (Whisper-style English / Korean / Chinese / Qwen). Active verification: default(English) extract / Korean extract / `normalize_korean` / `normalize_chinese` (code-switch Chinese CJK-spacing pipeline) / dispatcher branching. `normalize_default` and `normalize_chinese` (which chains it) require external libraries / resources — same dependency guard applies (see §5.5) |
| `binary/__init__.py` | `BinaryProcessor` | true / false extraction (Korean/English regex) |
| `code/__init__.py` | `CodeProcessor` | Code block extraction + continuation handling |
| `freeform/__init__.py` | `FreeformProcessor` | Free-form answer extraction (cue phrase regex + API fallback) |
| `multichoice/__init__.py` | `MultichoiceProcessor` | Multiple-choice option extraction (A/B/C, 1/2/3, ①②③ + API fallback) |
| `spatial_grounding/__init__.py` | `SpatialGroundingProcessor` | 2D region extraction for image/video grounding — bracket-arity source detection (quad > bbox > point), cross-shape conversion. Parsing-only: emits RAW parsed coords (coordinate-space normalization is the metric's job). API fallback intentionally unimplemented (raises). |
| `temporal_grounding/__init__.py` | `TemporalGroundingProcessor` | `[start, end]` interval extraction from prose — separator-pair last-wins regex, MM:SS / HH:MM:SS token conversion, invariant guard. API fallback intentionally unimplemented (raises). |
| `custom.py` | 4 functions | `parse_think`, `parse_boxed_format`, `parse_last_pattern`, `parse_circled_answer` |
| `__init__.py` | `get_postprocess_functions(...)` | Pipeline builder (partial function OrderedDict) |

Most are **regex-based → pure function areas**. Two exceptions implement an LLM-API fallback branch when regex fails:
- `MultichoiceProcessor._extract_multichoice_api(...)`
- `FreeformProcessor._extract_freeform_api(...)`

The two grounding processors **do not** implement an API fallback — verifying their `api_name` branch is therefore "raises `NotImplementedError`", not a mock-and-assert (see §2).

---

## 2. Mock boundaries — API fallback in one place

Most have no external resources. Mock boundaries are only the two *_api methods:

| Method | Mock target | Isolation |
|---|---|---|
| `MultichoiceProcessor._extract_multichoice_api` | `omni_evaluator.api.chat_completion_sync` | Unit mock (fake return value only — no actual API call) |
| `FreeformProcessor._extract_freeform_api` | same | same |
| `SpatialGroundingProcessor` / `TemporalGroundingProcessor` | (none) | API fallback is unimplemented; the `api_name` branch is verified by asserting `NotImplementedError` is raised — no mock surface |

> **No actual API calls are made in this area.** The behavior of Chat Completion itself is enforced by the `batch_chat_completion` smoke in `tests/inference/`, so here we only verify *whether the fallback branch is taken + whether arguments are correctly passed* via fake return. If a live smoke (`@requires_env`) is needed, that is the responsibility of the inference area.

ASR's `normalize_default` (Whisper-style English normalization) requires the `transformers` + `jiwer` packages and the `asr/resources/english.json` resource file. Dependency guards:
- `@pytest.mark.requires_extra("transformers", "jiwer")` — auto-skip when packages are absent.
- `@pytest.mark.skipif(not os.path.exists(_ENGLISH_SPELLING_MAPPING_PATH), ...)` — auto-skip when resource file is absent.
- `@pytest.mark.timeout(30)` — overrides the file-level `timeout(1)` because the first `transformers` import is heavy (not a regex verification).

`_extract_default` (English) / `_extract_default_korean` / `normalize_korean` / dispatcher branching can be verified without external dependencies and require no guards.

---

## 3. Verification depth — 2 layers

| Layer | What is verified | Mock | Time |
|---|---|---|---|
| **L1 deterministic unit (regex)** | Korean/English / various format variants of each Processor.extract() | None | < 100 ms |
| **L2 API fallback** | LLM call branching when regex fails | `chat_completion_sync` fake return | < 1 s |

L1 is the core. L2 verifies only *branching + argument passing* via fake return. The behavior of Chat Completion itself is the responsibility of `batch_chat_completion` smoke in `tests/inference/`, so no live API calls are made in this area (see §2).

---

## 4. Folder structure — per-Processor subfolder + custom at root

The structure where each Processor in the source has its own folder (`binary/__init__.py`, etc.) is mirrored as-is. The single-file `custom.py` and pipeline builder `__init__.py` are placed at the root. Each subfolder has its **own `_cases.py`** so case data is isolated per Processor.

```
tests/postprocess/
├── CLAUDE.md
├── __init__.py
├── conftest.py                 # cross-cutting fixture (patch_api_chat_completion, task_config_factory)
├── test_custom.py              ← omni_evaluator/postprocess/custom.py            (single file → placed directly at root)
│
├── asr/
│   ├── __init__.py
│   ├── _cases.py               # ASR_PASSTHROUGH / ASR_DEFAULT_* / ASR_KOREAN_* / ASR_NORMALIZE_KOREAN
│   └── test_asr.py             ← omni_evaluator/postprocess/asr/__init__.py      (§5.5 — normalize_default has dependency guard)
│
├── binary/
│   ├── __init__.py
│   ├── _cases.py               # BINARY_POSITIVE / BINARY_NEGATIVE / BINARY_LONG_* etc.
│   └── test_binary.py          ← omni_evaluator/postprocess/binary/__init__.py
│
├── code/
│   ├── __init__.py
│   ├── _cases.py               # CODE_EMPTY_OR_WHITESPACE / CODE_EXPLICIT_BLOCK
│   └── test_code.py            ← omni_evaluator/postprocess/code/__init__.py
│
├── freeform/
│   ├── __init__.py
│   ├── _cases.py               # FREEFORM_EN_CUE / FREEFORM_KO_CUE / FREEFORM_*_COMPLEX etc.
│   └── test_freeform.py        ← omni_evaluator/postprocess/freeform/__init__.py
│
├── multichoice/
│   ├── __init__.py
│   ├── _cases.py               # MC_LETTER_PREDICTIONS / MC_NUMBER_* / MC_CIRCLED_* / MC_*_COMPLEX
│   └── test_multichoice.py     ← omni_evaluator/postprocess/multichoice/__init__.py
│
├── spatial_grounding/
│   ├── __init__.py
│   ├── _cases.py               # SG_BBOX_BRACKETS / SG_POINT_* / SG_QUAD_* / SG_CROSS_SHAPE / SG_OUTPUT_SCALE / SG_INPUT_AUTO etc.
│   └── test_spatial_grounding.py  ← omni_evaluator/postprocess/spatial_grounding/__init__.py
│
└── temporal_grounding/
    ├── __init__.py
    ├── _cases.py               # TG_PAIR_PROSE / TG_TIMESTAMP_* / TG_LAST_WINS / TG_INVARIANT_* etc.
    └── test_temporal_grounding.py ← omni_evaluator/postprocess/temporal_grounding/__init__.py
```

Design intent:
- **Source 1:1 mirroring**: each Processor's tests are isolated in its own folder. Cases for one Processor do not mix with tests for another Processor, so the *scope of change impact* is clear at the folder level.
- **`_cases.py` always lives inside that Processor's folder**: importing case data from another Processor's folder is prevented at the root.
- **`conftest.py` only at root**: cross-cutting fixtures like `patch_api_chat_completion` / `task_config_factory` are automatically accessible by all subfolders via pytest's parent-conftest discovery.
- **`custom.py` and `__init__.py` are single files**: splitting them into separate folders would mean *one file inside a folder*, which is noise. `test_custom.py` / `test__init__.py` at the root verify them directly.

Verification of `ProcessorInterface`'s own contract (whether children have overridden extract, etc.) is handled by **`tests/interfaces/test_postprocess_interface.py`** — this area only looks at *each Processor's behavior*.
