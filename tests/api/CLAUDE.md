# tests/api — API provider test conventions

For overall test operating rules, see `tests/CLAUDE.md`; for structural design intent, see `tests/DESIGN.md`. This document covers additional rules that apply **only** to the **`omni_evaluator/api/`** area (provider routing + message serialization + response parsing).

Difference in responsibilities between `tests/api/` and `tests/inference/api/`:
- `tests/api/` (this document) — **per-provider serialization / SDK call / response parsing units**
- `tests/inference/api/` — `engine.main()` orchestration (batch calls / debug branching / per-record loop) units

Because inference's `engine.main()` *calls* this area's single entry point, the output of this area becomes the input to the inference area.

---

## 1. Area responsibilities

`omni_evaluator/api/` handles 5 kinds of responsibilities:

| Module | Responsibility |
|---|---|
| `routing.py` | model name → provider branching (openai / anthropic / google) |
| `chat_completions.py` | provider-agnostic single entry point (sync / async / batch) |
| `completions.py` | text-only (openai only) entry point (sync / async / batch) |
| `model_supporting.py` | yaml-based capability matrix lookup |
| `<provider>/` | provider SDK wrapping + message serialization + response parsing |

The exact entry point function names and signatures are governed by the source (`omni_evaluator/api/`) and the corresponding `test_*.py` as the single source of truth.

Provider directory structure:
```
omni_evaluator/api/
├── openai/      chat_completions.py + responses.py (structured output)
├── anthropic/   chat_completions.py + completions.py (empty, chat-only)
└── google/      chat_completions.py + completions.py (empty, chat-only)
                + file upload shared cache (LRU limit, §5.6)
```

---

## 2. Mock boundary — provider SDK layer

The native SDK call site for each provider is the single mock boundary. Nothing deeper (HTTP / TLS / SDK internals) is touched. The boundary differs per provider, and in some cases like OpenAI's structured output, even within the same provider there can be a separate call site for the chat path.

**Inside the boundary (what is verified)**:
- Message serialization — per-provider format conversion
- Response parsing (provider response → our standard output)
- Retry logic
- Timeout handling

**Outside the boundary (not verified)**:
- Actual HTTP / network — limited to live smoke tests isolated by the `requires_env("<PROVIDER>_API_KEY")` marker
- SDK-internal serialization (e.g., pydantic models enforced by the SDK)

The exact mock target call names, module locations, and symbols are governed by the source (`omni_evaluator/api/`) and the corresponding `test_*.py` as the single source of truth.

---

## 3. Verification depth — 3 layers

| Layer | What is checked | Mock depth | Time |
|---|---|---|---|
| **L1 deterministic unit** | provider routing branching, message serialization, generation_options conversion | no mock | < 100 ms |
| **L2 mock boundary unit** | single entry point dispatch (retry / response parsing / error handling) | provider SDK object mock | < 1 s |
| **L3 live smoke** | actual OpenAI / Anthropic / Google endpoint call | no mock | 1~5 s |

Since `tests/inference/api/test_engine.py` already has 9 provider × modality smokes (`test_<provider>_smoke_<modality>`), this area focuses on **L1 / L2**. There is no need to duplicate the same live calls.

---

## 4. Folder structure — 1:1 source mirroring

```
tests/api/
├── conftest.py                 # provider-agnostic fixtures (e.g., fake_response_factory)
├── test_routing.py             # routing.py units (provider branching / client selection / capability lookup)
├── test_chat_completions.py    # chat_completions.py — provider-agnostic dispatch
├── test_completions.py         # completions.py
├── test_model_supporting.py    # model_supporting.py yaml load
├── openai/
│   ├── conftest.py             # OpenAI-specific fixtures (e.g., fake_chat_completion fixture)
│   ├── test_chat_completions.py
│   └── test_responses.py       # structured output (pydantic schema)
├── anthropic/
│   ├── conftest.py
│   └── test_chat_completions.py
└── google/
    ├── conftest.py             # file upload cache fixture
    └── test_chat_completions.py
```

---

## 5. Key verification items

### 5.1 Routing — deterministic branching
- model name → provider group branching (openai / anthropic / google); unknown models follow the production code's choice (error or default)
- provider group + sync/async → correct SDK client object, key loaded from env var
- capability lookup → result returned with per-provider support flags populated

The exact method names, arguments, and return shapes are governed by the source (`omni_evaluator/api/`) and the corresponding `test_*.py` as the single source of truth.

### 5.2 Message serialization (per-provider)
Message structure differs per provider (system message placement, tool representation, image representation), and serialization is verified to produce *exactly the format each provider expects*. The serialized output is compared 1:1 against the shape the provider SDK expects, so that if the SDK later requires a new field, it is caught as a regression.

The exact serialization functions, template arguments, and per-provider key shapes are governed by the source (`omni_evaluator/api/`) and the corresponding `test_*.py` as the single source of truth.

### 5.3 Response parsing
Does the conversion from provider response object (mock) → our standard output work correctly:
- Text prediction extraction
- Tool call conversion (provider's tool representation → our standard form)
- Reasoning content extraction (for reasoning-class models / when thinking is supported)
- Latency measurement
- Empty response / refusal handling

The exact output types and field names are governed by the source (`omni_evaluator/api/`) and the corresponding `test_*.py` as the single source of truth.

### 5.4 Retry / timeout
- Retry branching — returns a failure value after the configured number of retries (mock raises continuously)
- Appropriate exception or failure value when timeout is exceeded
- Distinguishing transient vs permanent errors — transient retries, permanent fails immediately

The exact retry counts, timeout constants, and exception classes used for distinction are governed by the source (`omni_evaluator/api/`) and the corresponding `test_*.py` as the single source of truth.

### 5.5 generation_options normalization
Per-provider normalization / dispatch of generation_options is the responsibility of the schemas area — see `tests/schemas/CLAUDE.md §5.6` and `tests/schemas/test_generation_options.py`.

### 5.6 Google file upload cache — thread / process safety
The file upload cache in the Google area is a shared cache with an LRU limit. When inference runs with multiple workers (`world_size > 1`) or is called concurrently via async, multiple workers read/write the cache simultaneously. This regression is not exposed by single-process mock units, so it is verified explicitly:
- Same input arrives as a cache miss simultaneously → upload happens **exactly once** (idempotent), both workers receive the same result
- Read race during LRU eviction → cache invariant is maintained (no over-limit, no duplicate keys, no leaked lookup exceptions)
- Verification method: simulate concurrent calls with threads + count upload calls via mock. The cache itself can be caught with unit tests without going all the way to live smoke.

The exact cache data structure, limit value, and symbols are governed by the source (`omni_evaluator/api/`) and the corresponding `test_*.py` as the single source of truth.

---

## 6. Fixture patterns

Fixtures are layered in three tiers following the residency principle from §3 (`tests/CLAUDE.md`):

- **Provider-agnostic** (`tests/api/conftest.py`) — standard response builder shared by all providers. Per-provider conftest overrides as needed.
- **Per-provider** (`tests/api/<provider>/conftest.py`) — factory that mimics the shape of that provider SDK's response objects. Use lightweight stubs instead of real SDK types (see §8).
- **Mock helper** — fixture that monkeypatches the provider's native call site and returns a call argument log for use in argument verification.

The exact fixture names, response object field shapes, and patch targets are governed by the corresponding `conftest.py` as the single source of truth.

---

## 7. Live smoke placement

Live smokes *may* be added to `tests/api/<provider>/test_chat_completions.py`, but **since `tests/inference/api/test_engine.py` already has 9 provider × modality smokes, duplication should be avoided.**

Cases where live smoke is needed in this area:
- When confirming that a provider SDK's new feature / new model (e.g., OpenAI o1 reasoning, Anthropic thinking) *response shape* is compatible with our parsing
- Paths not covered by inference area smokes, such as structured output (pydantic schema)

In these cases, `@pytest.mark.requires_env("<PROVIDER>_API_KEY")` + `@pytest.mark.slow` markers are mandatory.

---

## 8. Common pitfalls

- **Mock unit tests that pass PIL Image / numpy array directly into message conversion functions** — in the normal production flow, these are converted to strings (URL/base64) during the `prepare_dataset` stage before arriving. Feeding *different* inputs into mock units than what the production flow provides means real regressions are missed. Verify using the same input mode as the inference-side image_record_url_factory.
- **Do not construct provider SDK objects directly** — lightweight stubs / mocks are sufficient. Building real SDK response types breaks on SDK version upgrades.
- **yaml change regression in the capability matrix** — when yaml is modified, a schema unit test is needed to verify that the support flag key set is not broken.
- **Thread-safety of the Google file upload cache** — whether the shared cache is safe under concurrent calls. Verification items are in §5.6 (elevated from mock unit).
- **The primary gate for `requires_env` is the existence of `.env.test` itself** (`tests/CLAUDE.md §1.2`) — shell exports alone do not work. This is an intentional safety gate.
