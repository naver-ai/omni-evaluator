# OmniEvaluator Builtin Datasets Builder

_Helpers for assembling raw resources → `data.jsonl` + `resources/` layout → S3 mirror → task registration._

> **⚠️ Reference helpers, not a turnkey pipeline.** Every raw dataset dump
> has its own field names, directory layout, and label conventions, so this
> module is meant as **assembly parts**, not a one-shot script. Write a thin
> driver per benchmark that composes these helpers to your source data shape.

Converts a raw resource pile (image / audio / video files + metadata in
CSV / JSON / ...) into the layout omni_evaluator builtin tasks expect
(`data.jsonl` + `resources/`), optionally mirrors it to an S3-compatible
bucket, and walks you through registering the corresponding builtin task.

---

## 1. Full Task Build Process

Steps to register a new benchmark as a builtin task. All of ①–⑤ must be
completed for the task to be runnable.

### ① Collect raw data

- Original image / audio / video files (before layout placement)
- Source file(s) with QA / labels / metadata (json, jsonl, csv, tsv, parquet, ...)

### ② Build `data.jsonl` + `resources/`

Compose helpers in `build_dataset.py` to produce:

```
./out/omni-evaluator/datasets/<benchmark>/<split>/
├── data.jsonl
└── resources/
    ├── images/…
    ├── audios/…
    └── videos/<variant>/…
```

Minimal example (image + MC QA):

```python
from scripts.data_builder.build_dataset import build_content, build_sample, build_dataset

def _sample(raw, idx):
    return build_sample(
        raw={}, index=idx,
        user_contents=[
            build_content("image", raw["image_file"]),
            build_content("text",  raw["question"]),
        ],
        label=raw["answer"],
        options=list("ABCD"),
        option_contents=raw["choices"],
        meta={"category": raw["subject"]},
    )

build_dataset(
    raw_items=iter_raw("/data/raw/foo/all.jsonl"),
    benchmark="foo", split="test",
    source_root="/data/raw/foo/images",
    output_root="./out",
    sample_builder=_sample,
)
```

### ③ (Optional) Upload to S3 mirror

Required only when the task's `config.yaml` uses `dataset.source: "s3"`.
For local-only runs, use `dataset.source: "local"` with `local_dirpath`.

```python
from scripts.data_builder.build_dataset import upload_to_s3
upload_to_s3(
    local_root="./out/omni-evaluator/datasets/foo/test",
    remote_prefix="omni-evaluator/datasets/foo/test",
    bucket_name=os.environ["S3_BUCKET_NAME"],
    access_key=os.environ["S3_ACCESS_KEY"],
    secret_key=os.environ["S3_SECRET_KEY"],
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
)
```

### ④ Create the task directory

Under `omni_evaluator/evaluation/builtin/tasks/<task_name>/`, add three files:

```
tasks/<task_name>/
├── __init__.py         # can be empty
├── config.yaml         # dataset / prompts / postprocess / evaluation
└── custom.py           # sample → Record conversion (task-specific schema)
```

#### `config.yaml` example (image MC benchmark)

```yaml
meta:
  benchmark_name: "foo"
  split: "test"
  lang: "en"
  input_modality: ["image", "text"]
  output_modality: ["text"]
  task_type: ["visual_question_answering"]
  subtask_type: ["multiple_choice"]

dataset:
  source: "s3"                                                    # or "local"
  data_filepath: "omni-evaluator/datasets/foo/test/data.jsonl"
  image_dirpath: "omni-evaluator/datasets/foo/test/resources/images"
  # add audio_dirpath / video_dirpath only for the modalities the task needs

prompts:
  direct:
    system_prompt:
    task_prompt: "{query}\nAnswer with the option's letter from the given choices directly."
  reasoning:
    system_prompt:
    task_prompt: "{}\nGive step by step reasoning."

inference:
  generation_options:
    max_new_tokens: 4096

postprocess:
  direct: {}
  reasoning:
    pipeline: ['think']

evaluation:
  method: "generation"
  target_metrics:
    text_evaluator:
      exact_match: {}
    judge_evaluator:
      judge_binary:
        lang: "en"
        judge_model: "gpt-5-mini"
        judge_prompt: "..."
        max_tokens: 16
        temperature: 1.0
  display_metrics: ["judge_binary", "exact_match"]
```

Key points:

- `dataset.data_filepath` / `image_dirpath` must match paths produced in ②/③.
- `prompts.direct` and `prompts.reasoning` are switched by the `--reasoning`
  CLI flag.
- Only metrics listed in `evaluation.target_metrics` are computed;
  `display_metrics` controls the top-of-report ordering.

#### `custom.py` skeleton

Converts one data.jsonl line → a `Record`. Forwarding the `messages`
content as-is and populating label / options / meta usually suffices.

```python
from typing import Any, Dict, List, Optional
from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent as ChatTextContent,
    ImageContent as ChatImageContent,
    # add AudioContent, VideoContent as needed
)
from omni_evaluator.schemas.inference import Record


def sample_to_record(
    task_name: str,
    task_config,
    sample_idx: int,
    sample: Dict[str, Any],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    task_prompt_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Record:
    # Forward sample.messages verbatim as ChatMessage entries.
    messages: List[ChatMessage] = []
    for msg in sample.get("messages", []) or []:
        contents = []
        for c in (msg.get("content") or []):
            if c["type"] == "text":
                contents.append(ChatTextContent(type="text", value=c["value"]))
            elif c["type"] == "image":
                contents.append(ChatImageContent(type="image", value=c["value"]))
            # handle audio / video similarly if the benchmark uses them
        messages.append(ChatMessage(role=msg.get("role", "user"), content=contents))

    return Record(
        messages=messages,
        label=sample.get("label", []) or [],
        options=sample.get("options"),
        option_contents=sample.get("option_contents"),
        meta={
            "question_id": str(sample.get("index", sample_idx)),
            # forward sample.meta.category through so group_metrics can slice on it
            "category": (sample.get("meta") or {}).get("category"),
        },
    )
```

- `Record` is defined in `omni_evaluator/schemas/inference.py`.
- Fast reference — copy an existing task's `custom.py`:
  - MC: `tasks/mmmu_validation/custom.py`
  - VQA (freeform): `tasks/chartqa_test/custom.py`
  - Video: `tasks/video_mme_test/custom.py`

### ⑤ Smoke-test the run

```bash
python run.py evaluate \
    --evaluation_engine=builtin \
    --benchmarks=<task_name> \
    --inference_engine=huggingface \
    --model_name_or_path=<model> \
    --debug \
    --output_dirpath=./out \
    --cache_dirpath=/mnt/tmp
```

`--debug` runs only `NUM_DEBUG_SAMPLES` (=3) samples — strongly recommended
when a task is registered for the first time, so schema mistakes surface
fast. Once that succeeds, re-run without `--debug` for the full split.

---

## 2. Dataset Layout

### 2.1 Path convention

Standard tree referenced by builtin tasks (aligned with the `dataset` block
in `config.yaml`):

```
omni-evaluator/datasets/<benchmark>/<split>/
├── data.jsonl                              # one line == one sample (shared across all video variants)
└── resources/
    ├── images/<filename>.png|jpg|...       # image_dirpath
    ├── audios/<filename>.wav|mp3|...       # audio_dirpath
    └── videos/                             # video_dirpath — branch by variant subdir
        ├── base/<filename>.mp4|...            #   original mp4
        ├── 8_frames/<filename>...             #   pre-extracted 8-frame variant
        ├── 64_frames/<filename>...            #   pre-extracted 64-frame variant
        └── 128_frames/<filename>...           #   pre-extracted 128-frame variant
```

- Samples carry filenames only; the actual bytes live under the tree above.
  At load time `prepare_dataset.py` resolves each filename to a real path / URL.
- The same tree exists both on the local filesystem mirror and on the
  S3-compatible mirror. Tasks with `dataset.source: "s3"` fetch from S3;
  `"local"` reads from the local mirror.

#### `config.yaml` reference pattern

```yaml
dataset:
  source: "s3"
  data_filepath: "omni-evaluator/datasets/<benchmark>/<split>/data.jsonl"
  image_dirpath: "omni-evaluator/datasets/<benchmark>/<split>/resources/images"
  # audio_dirpath, video_dirpath follow the same pattern
```

#### Video variants (base / N_frames)

Variant tasks of the same benchmark **share `data.jsonl`** and differ only
in `video_dirpath`:

| task | `data_filepath` | `video_dirpath` |
|---|---|---|
| `video_mmmu_test` | `.../video_mmmu/test/data.jsonl` | `.../videos/base` |
| `video_mmmu_test_8frames` | same | `.../videos/8_frames` |
| `video_mmmu_test_64frames` | same | `.../videos/64_frames` |
| `video_mmmu_test_128frames` | same | `.../videos/128_frames` |

So **`data.jsonl` is built once per benchmark**, the base variant goes under
`resources/videos/base/`, and each N-frame variant is produced by a
frame-extraction pass placed under `resources/videos/<N>_frames/`.
For the standard N-frame variants (default `8` / `64` / `128`) two focused
helpers ship alongside `build_dataset.py`:
`preprocess_video.py` (container unify + integrity check) and
`build_frame_variants.py` (N-frame variants build) — see **§4 Video
preprocessing utilities** below. For non-standard sampling strategies
(keyframe, stride, task-specific decoder) use your own tool and drop the
output into `resources/videos/<variant>/`. `partition_resources()` can
target a variant subdir via
`modality_target_subdirs={"video": "videos/64_frames"}`.

### 2.2 `data.jsonl` schema

One line per sample:

```jsonc
{
  "index": 0,                                     // unique id within the split (int or str)
  "messages": [                                   // chat-format turns
    {
      "role": "user",
      "content": [
        {"type": "image", "value": "00000000.png", "ocr": [ ... ]},
        {"type": "text",  "value": "How many food item is shown in the bar graph?"}
      ]
    }
    // prepend/append role="system" / "assistant" for multi-turn samples
  ],
  "label": ["14"],                                // list of answer(s); always a list
  "options": ["A", "B", "C", "D"],                // MC letters (optional)
  "option_contents": ["Antenna", "Simple eye", ...],  // MC choice texts (optional)
  "captions": {"en": [...], "ko": [...]},         // image captions (only some datasets, optional)
  "questions": [...],                             // multi-question (optional)
  "meta": {"category": "Accounting", "image_size": {...}}
}
```

#### Field details

| Field | Type | Required | Description |
|---|---|:---:|---|
| `index` | `int` \| `str` | ✅ | Unique within the split |
| `messages` | `list[dict]` | ✅ | Chat-format; see the content-type table below |
| `label` | `list[str]` | ✅ | Answer(s). Always a list, even for a single answer. Use `[]` if none. |
| `options` | `list[str]` |  | MC letters, e.g. `["A","B","C","D"]` |
| `option_contents` | `list[str]` |  | MC choice texts, index-aligned with `options` |
| `captions` | `dict[str, list[str]]` |  | Language code → captions |
| `questions` | `list[str]` |  | Multi-question sample |
| `meta` | `dict` |  | Free-form. **`category` is the primary group_metrics key** (see §2.3) |

#### `messages[*].content` types

| `type` | `value` | Extra fields |
|---|---|---|
| `text` | Natural-language string | — |
| `image` | `<filename>.png/jpg/...` (under `resources/images/`) | `ocr` (list of `{id, bbox, text, confidence}`), `entity`, ... |
| `audio` | `<filename>.wav/mp3/...` (under `resources/audios/`) | — |
| `video` | `<filename>.mp4/...` (under `resources/videos/<variant>/`) | `duration`, `fps`, `width`, `height`, `codec`, `subtitle`, ... |

### 2.3 `meta` — primary group_metrics key

`meta.category` is the **primary grouping axis** used by `group_metrics`.
Downstream evaluation partitions samples by `category` and auto-computes
per-category accuracy and similar aggregate stats.

- **`category` is the only axis that is conventionally uniform** — other
  names (`subject`, `subfield`, `domain`, ...) are handled inconsistently
  across tasks. Any axis you want auto-sliced in group_metrics should be
  placed at `meta.category`.
- You can freely add other metadata to `meta` (e.g. `image_size`,
  `topic_difficulty`), but they are **not** auto-sliced — handle them
  explicitly in the task's `custom.py` if you need them.

> Task-specific schema requirements vary. **Always inspect the task's
> `custom.py:sample_to_record` first** and shape `build_sample()` output
> to match what it reads.

---

## 3. Builder API (`build_dataset.py`)

Six helpers. Import from the project root:
`from scripts.data_builder.build_dataset import ...`.

### 3.1 `build_content(type_, value, **extras)` — one content piece

```python
build_content("image", "00000000.png", ocr=[...])
build_content("text", "How many food items?")
build_content("video", "abc.mp4", duration=60.0, fps=30)
```

### 3.2 `build_sample(raw, index, user_contents, ...)` — one sample line

Turns one raw dict into a standard sample dict. Multimodal filenames stay
as they are in the source dump; `partition_resources()` later moves the
actual files into the target tree.

```python
from scripts.data_builder.build_dataset import build_content, build_sample

sample = build_sample(
    raw={},                                             # placeholder / kept only for traceability
    index=0,
    user_contents=[
        build_content("image", "00000000.png"),
        build_content("text", "How many bars?"),
    ],
    label="14",                                         # normalized to ["14"]
    meta={"category": "chart_qa", "image_size": {"00000000.png": [850, 600]}},
)
```

MC example:

```python
build_sample(
    raw={}, index=42,
    user_contents=[
        build_content("image", "diagram_42.jpg"),
        build_content("text", "What is between the head and abdomen?"),
    ],
    label="D",
    options=["A", "B", "C", "D"],
    option_contents=["Antenna", "Simple eye", "Spiracle", "Thorax"],
    meta={"category": "biology"},
)
```

### 3.3 `partition_resources(samples, source_root, target_root, ...)` — place resources

Walks `sample.messages[*].content`, picks items with `type` in
`image`/`audio`/`video`, and materializes the referenced files under the
correct modality subdir of the target tree.

- Defaults: `images/`, `audios/`, `videos/base/`
- `copy_mode`: `"hardlink"` (fastest on the same filesystem — recommended) /
  `"symlink"` / `"copy"`
- Variant target: `modality_target_subdirs={"video": "videos/64_frames"}`

### 3.4 `write_data_jsonl(samples, target_root)` — dump data.jsonl

Writes one JSON object per line to `target_root/data.jsonl`. Uses
`ensure_ascii=False` so non-ASCII text is preserved verbatim.

### 3.5 `upload_to_s3(local_root, remote_prefix, ...)` — mirror to S3

Uploads the local tree wholesale. **Uses the repo's `S3Client`**, so
S3-compatible storage quirks (e.g. forcing
`AWS_REQUEST_CHECKSUM_CALCULATION=when_required`) are handled inside.
Do **not** call `boto3` directly.

```python
upload_to_s3(
    local_root="./out/omni-evaluator/datasets/foo/test",
    remote_prefix="omni-evaluator/datasets/foo/test",
    bucket_name=os.environ["S3_BUCKET_NAME"],
    access_key=os.environ["S3_ACCESS_KEY"],
    secret_key=os.environ["S3_SECRET_KEY"],
    endpoint_url=os.environ["S3_ENDPOINT_URL"],
)
```

### 3.6 `build_dataset(...)` — orchestration example

Reference glue that chains the four helpers above. Supply a
`sample_builder=lambda raw, idx: build_sample(raw, ...)` callback so you
own only the raw → sample mapping; the rest is handled for you.

---

## 4. Video preprocessing utilities

Two focused CLIs handle the video side of the pipeline. Both are
directory-scoped (`--video_dirpath`) and idempotent — safe to re-run.

### 4.1 Stage 1 — `preprocess_video.py`: unify container + validate integrity

```bash
python scripts/create_task/preprocess_video.py \
    --video_dirpath /path/to/raw/videos \
    --workers 8
```

- **Container unification** — every non-`.mp4` video
  (`.mkv/.webm/.avi/.mov/…`) is stream-copy remuxed to `.mp4` with
  `+faststart` alongside the source, then the source is removed. No
  re-encode: codec / GOP / fps / resolution / pixel format are preserved
  verbatim. Existing `.mp4` files are left untouched.
- **Integrity validation** — `ffprobe` extracts stream metadata
  (codec / w / h / fps / duration / nb_frames); `ffmpeg -err_detect
  explode -f null -` walks the whole stream to catch truncation or
  corrupted frames.
- **Outputs** — sidecars at the directory root:
  - `.metadata.jsonl` — per-video:
    `{path, action, meta:{codec,w,h,fps,duration,nb_frames}}`
  - `.errors.jsonl` — files that failed remux or full-decode probe (does
    not block the run)

### 4.2 Stage 2 — `build_frame_variants.py`: N-frame compressed variants

Only run after Stage 1 (assumes the directory is a flat set of validated
`*.mp4`).

```bash
python scripts/create_task/build_frame_variants.py \
    --video_dirpath /path/to/videos \
    --frames 8 64 128 \
    --workers 16
```

- **Layout rewrite (in place)**:

  ```
  <video_dirpath>/*.mp4
    →
  <video_dirpath>/base/*.mp4          # moved verbatim; no re-encode
  <video_dirpath>/8_frames/*.mp4      # built
  <video_dirpath>/64_frames/*.mp4     # built
  <video_dirpath>/128_frames/*.mp4    # built
  ```

- **Encoding** — `ffmpeg`'s `fps` filter resamples the video stream to
  `N / duration` fps, preserving original duration. `libx264 -preset fast
  -crf 23 -pix_fmt yuv420p -g 1` makes every output frame a keyframe
  (cheap seek; favorable for vLLM's sequential `cap.grab()` loop). Audio
  is stream-copied (`-c:a copy`) so sync stays intact for audio-aware
  models.
- **Short-clip edge case** — if `total_frames ≤ target N`, hardlink the
  base file instead of re-encoding (same content, zero extra disk).
- **`--frames` default: `[8, 64, 128]`** — the frame counts consumed by
  the inference paths. Pass a subset (`--frames 64`) to build only what
  a task needs.
- **Outputs** — `.base_moved` sentinel + `.variant_errors.jsonl` on
  failure.

### 4.3 Why this split

Stage 1 is **lossless** (stream copy) and runs once per source. Stage 2
is **lossy but disposable** — variants can be regenerated any time from
`base/` without re-downloading. Keeping them separate also lets tasks
that don't need compressed variants stop after Stage 1.

### 4.4 Placement into the omni_evaluator tree

After Stage 2 the directory already matches the omni_evaluator convention
(`<video_dirpath>/base/`, `<video_dirpath>/<N>_frames/`). Point the
benchmark's build script at that directory as `source_root` for
`partition_resources()`, or copy the whole `base/` and `<N>_frames/`
subdirs into
`omni-evaluator/datasets/<benchmark>/<split>/resources/videos/`.

---

## 5. Gotchas

- **These are reference helpers** — write a thin driver per dataset that
  imports and composes them, rather than running this file directly.
- **Check `custom.py:sample_to_record` first** — which fields it reads
  differs per benchmark. Shape your schema to match, and the loader
  absorbs it as-is.
- **Stick to the path convention** — keep the
  `omni-evaluator/datasets/<bench>/<split>/` tree intact so `config.yaml`
  needs no edits. Modality subdir names (`images` / `audios` / `videos/base`
  or `videos/<N>_frames`) must match what `config.yaml` references.
- **`meta.category` is the group_metrics convention** — other names
  (subject, subfield, domain, ...) are handled inconsistently, so put any
  axis you want auto-sliced under `meta.category`.
- **`copy_mode="hardlink"` only works on the same filesystem** — falls
  back to copy on cross-device, so crossing a mount boundary means real
  file duplication; budget disk accordingly.
- **Always use `omni_evaluator.clients.s3_client.S3Client` for uploads** —
  S3-compatible storage checksum quirks are baked into it. Calling
  `boto3` directly leads to intermittent failures.
- **`data.jsonl` is UTF-8 with `ensure_ascii=False`** — non-ASCII text
  (e.g. Korean, CJK) escaped as `\uXXXX` can produce silent tokenizer
  differences downstream.
- **Rerun safety** — `partition_resources()` skips files that already
  exist, so partial-failure reruns are idempotent. `write_data_jsonl()`
  overwrites, so sort raw items deterministically (e.g. by `index`)
  before feeding them in for reproducible reruns.
- **Video variants are a separate pipeline after base** — for the
  standard N-frame variants, run `preprocess_video.py` then
  `build_frame_variants.py` (§4). For non-standard sampling strategies,
  drop the output into `resources/videos/<variant>/` yourself.

---

## 6. Related files

| Path | Role |
|---|---|
| `scripts/create_task/build_dataset.py` | Builder API: `build_content`, `build_sample`, `partition_resources`, `write_data_jsonl`, `upload_to_s3`, `build_dataset` |
| `scripts/create_task/preprocess_video.py` | Stage 1: video container unify (→ `.mp4`) + integrity validation |
| `scripts/create_task/build_frame_variants.py` | Stage 2: N-frame variants (`base/`, `<N>_frames/`) |
| `omni_evaluator/evaluation/prepare_dataset.py` | Load data.jsonl → Record iterator |
| `omni_evaluator/evaluation/builtin/tasks/<task>/config.yaml` | Paths / split / prompts / metrics |
| `omni_evaluator/evaluation/builtin/tasks/<task>/custom.py` | Sample → Record conversion (task-specific) |
| `omni_evaluator/clients/s3_client.py` | Standard S3 client with S3-compatible storage quirks baked in |
| `omni_evaluator/schemas/inference.py` | `Record`, `Message`, `*Content` type definitions |
| `omni_evaluator/schemas/task.py` | `TaskConfig`, `TaskDataset` and other config schemas |
