# tests/clients — External data client conventions

For overall test operating rules see `tests/CLAUDE.md`, and for structural design intent see `tests/DESIGN.md`. This document covers additional rules that apply **only** to the **`omni_evaluator/clients/`** area (S3-compatible object storage clients).

---

## 1. Area responsibilities

`omni_evaluator/clients/` handles only **client abstractions** for external data stores.

| Module | Responsibility |
|---|---|
| `__init__.py` | Export endpoint constants (`OCR_ENDPOINT`, `OBJECT_DETECTION_ENDPOINT`, `LENS_ENDPOINT`) |
| `s3_client.py` | `S3Client` — upload/download/list/delete combining boto3 + `pyarrow.fs.S3FileSystem` |

`S3Client` is used in production as the permanent storage for datasets / model checkpoints / result artifacts, so it is a **risk point where accidentally touching a real bucket is possible if not protected by mocks**. moto or a stub backend is required.

---

## 2. Mock boundary — moto (S3 stub) + both boto3 / pyarrow

S3Client uses **two SDKs** simultaneously:

- `boto3` (batch upload/download, multiprocessing worker)
- `pyarrow.fs.S3FileSystem` (single-file I/O, fast stream)

Mock strategy:

| Verification scenario | Mock tool | Notes |
|---|---|---|
| Deterministic unit (path normalization / error handling) | Fake boto/pyarrow client objects via `monkeypatch` | Fastest |
| Integration (both boto + pyarrow) | `moto.mock_aws` decorator | S3 backend simulation |
| Live (actual NAVER Cloud / AWS) | `@requires_env("NAVER_CLOUD_ACCESS_KEY", "NAVER_CLOUD_SECRET_KEY")` | Isolated in a separate file, shared sandbox bucket |

Prefer `moto` as the first choice — since boto + pyarrow share the same backend (in-memory S3), consistency between the two libraries is verified at the same time.

---

## 3. Verification depth — 3 layers

| Layer | What it checks | Mock depth | Time |
|---|---|---|---|
| **L1 deterministic unit** | Path normalization (PosixPath vs str), credential env var loading, batch_io result aggregation | boto3 object mock | < 100 ms |
| **L2 moto integration** | upload/download/list/delete round-trips, error injection (`NoCredentialsError`, `BucketNotFoundError`) | `@moto.mock_aws` | < 1 s |
| **L3 live (optional)** | Actual NAVER Cloud S3 sandbox bucket | `@requires_env` | 1~10 s |

Only L1 / L2 run automatically in CI. L3 is in a separate file + `@slow` marker.

---

## 4. Folder structure

```
tests/clients/
├── conftest.py                 # moto fixture (mock_aws + temporary bucket)
└── test_s3_client.py             # S3Client unit (L1 + L2) + (optional) live smoke (L3, slow)
```

Since the source is a single `s3_client.py`, mirroring is straightforward. When new clients (R2 / GCS) are added in the future, follow the same pattern with `test_<name>_client.py`.
