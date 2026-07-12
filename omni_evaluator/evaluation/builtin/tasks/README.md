# Task Creation Guide

## In-house Benchmarks
1. Obtain a dataset (local dataset or Hugging Face)
    - Datasets sourced from huggingface_hub can be used directly without the formatting described below. However, conversion and S3 upload in the format below are required in the following cases:
        - When multimodal items in the dataset are not URLs
            - Local file paths that require separate multimodal item downloads
            - Bytes data: Since vLLM and some models do not support direct bytes input for audio/video, path conversion must be performed each time, which adds overhead.
                - Video in particular is burdensome to keep in memory per sample
    - A local dataset consists of `data.jsonl` (one JSON object per line) and the individual multimodal items it references.
        - The format mirrors the OpenAI chat format, except every content item carries its payload under the `value` key.
        - `data.jsonl` example — one record, pretty-printed here for readability (in the actual file each record is a single line):
        ```json
        {
          "index": "42",
          "messages": [
            {
              "role": "user",
              "content": [
                {"type": "image", "value": "images/triangle_0042.png"},
                {"type": "text", "value": "In the figure, triangle ABC is isosceles with AB = AC and angle A = 40 degrees. What is the measure of angle B?\nA. 40\nB. 60\nC. 70\nD. 100"}
              ]
            }
          ],
          "label": ["C"],
          "meta": {
            "category": "geometry",
            "question_id": "geo_0042"
          }
        }
        ```
        - Field notes:
            - `index` — **string** id (quote numbers: `"42"`).
            - `messages` — chat turns (`system`/`user`/`assistant`). The system prompt normally lives in `config.yaml` (`prompts.*.system_prompt`, auto-prepended), so most tasks use a single `user` turn; text-only tasks just omit the `image` item.
            - `value` — text, or a multimodal item's **relative path** (e.g. `images/triangle_0042.png`), resolved against the resource dir. Don't use absolute paths (`/mnt/...`).
            - `label` — list of strings, multiple answers allowed: `["C"]`, `["70"]`.
            - `meta` — free-form; keys like `category` feed group metrics.
        - Multimodal items live under `resources/{images,audio,videos}/...` as the config points.

2. Upload the dataset to S3.
- Uploaded datasets can be accessed remotely via S3 or downloaded locally for evaluation stability and speed.
```
# The dataset path contains the following files/directories
# data.jsonl, resources/audio, resources/images, resources/videos
dataset_dirpath = "sample_dataset/test"
# Must follow this format
# f'omni-evaluator/datasets/{benchmark_name}/{split}'
remote_dirpath = "omni-evaluator/datasets/sample_dataset/test"

s3_client.upload_file(
    filepath="sample_dataset/test/data.json",
    remote_dirpath=remote_dirpath,
)
s3_client.upload_dir(
    dirpath=dataset_dirpath,
    remote_dirpath=remote_dirpath,
)
```

3. Create a task directory (`evaluation/builtin/tasks/{task_name}`)
- Write config.yaml
  - Refer to `schemas/task.py` and existing task configurations
    - Regarding dataset paths:
        - If the source is local, all multimodal item values in messages must exist within `{local_dirpath}/{OO_dirpath}`
- (Optional) Write custom.py
    - Currently, the following 3 custom cases are supported:
    - sample_to_record: If the dataset format from step 1) is not followed, write task-specific logic in the `sample_to_record` method to import data into the common Record schema
    - postprocess: To add postprocessing that applies only to a specific task beyond the package's common postprocessing logic, write the method and specify the method name in `evaluation.postprocess.pipeline` within `config.yaml`
    - (WIP) metric: To add a metric used only for a specific task, write the method and specify the method name in `evaluation.target_metrics` within `config.yaml`
        - If the metric could potentially be shared across other modules, it is recommended to write it in `evaluation/metrics/evaluator.py`


## Dataset Source (`config.yaml`)

`dataset.source` selects how data loads (full field list: `schemas/task.py` → `TaskDataset`):

| `source` | When | Key fields |
|---|---|---|
| `huggingface_hub` | Already on HF Hub, items are URLs/standard features — used directly | `path`, `split`, `name`, `combine`, `trust_remote_code`, `audio_decode`, `audio_column` |
| `s3` | Converted to `data.jsonl` + `resources/` and uploaded to S3 | `data_filepath`, `image_dirpath`, `audio_dirpath`, `video_dirpath` |
| `local` | Same layout, on the local filesystem | `local_dirpath` + the `*_dirpath` above |
| `package` / `resources` | Bundled in the package / task dir | see `schemas/task.py` |

Common to all: `options`, `subset` (filter samples by `meta[key]`).

```yaml
# huggingface_hub
dataset: {source: "huggingface_hub", path: "princeton-nlp/CharXiv", split: "validation"}
# multi-subset / audio: add  name: [...], combine: {method: concatenate},
#                             audio_decode: false, audio_column: [...], trust_remote_code: true

# s3 — each path is an S3 object key
dataset:
  source: "s3"
  data_filepath: "omni-evaluator/datasets/chartqa/test/data.jsonl"
  image_dirpath: "omni-evaluator/datasets/chartqa/test/resources/images"

# local — values resolve under {local_dirpath}/{*_dirpath}
dataset:
  source: "local"
  local_dirpath: "/data/datasets/chartqa/test"
  data_filepath: "data.jsonl"
  image_dirpath: "resources/images"
```

> Adding custom tasks for the **lm-eval-harness** or **lmms-eval** engines is documented in those engines' own READMEs:
> [`../../lm_eval_harness/README.md`](../../lm_eval_harness/README.md) and [`../../lmms_eval/README.md`](../../lmms_eval/README.md).
