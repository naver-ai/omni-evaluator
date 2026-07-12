# LMMs-Eval Evaluation Engine

A vision-language benchmark evaluation engine based on the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework.

## Features

| Item | Description |
|------|-------------|
| GPU Usage | **NO** — Only performs data preparation and metric computation |
| External API | Not required |
| Number of Tasks | 747 |
| Supported Modalities | text, image, video |

Supports perplexity-based evaluation for multiple choice tasks, as well as few-shot learning and per-category group metrics.

## Required Environment Variables

| Environment Variable | Description |
|----------------------|-------------|
| `HF_TOKEN` | Hugging Face Hub authentication token (for dataset downloads) |
| `HF_HOME` | Hugging Face cache directory |
| `HF_HUB_CACHE` | Hugging Face Hub cache directory |

## Engine-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_fewshot` | `-1` | Number of few-shot examples; uses task defaults if `-1` |

## Examples

### Using with the Hugging Face Inference Engine
```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="huggingface" \
    --model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct" \
    --exp_name="qwen2.5-vl-3b" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="textvqa_val" \
    --torch_dtype="float16"
```

### Using with the vLLM Inference Engine
```bash
python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="vllm" \
    --url="http://localhost:8000" \
    --exp_name="my-experiment" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="videomme"
```

## Adding Custom Tasks

Add a task folder under `resources/custom_tasks/<task_name>/` in standard
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) format. At startup the engine **copies these
folders into the installed `lmms_eval/tasks/`** (same-name files are overwritten; `example_*` folders are
skipped).

A folder has up to 3 files — see `activitynetqa/` for a full example:

- **`_default_template_yaml`** — shared dataset/prompt config (`dataset_path`, `dataset_kwargs`, prompts)
- **`<task>.yaml`** — the task: `task`, `test_split`, `output_type`, `doc_to_visual/text/target`,
  `metric_list`, `include: _default_template_yaml`, `generation_kwargs`
- **`utils.py`** — the functions referenced as `!function utils.<fn>` (`doc_to_*`, `process_results`,
  aggregations); LLM-judge calls live here

```yaml
# <task>.yaml (abridged)
task: "activitynetqa"
test_split: test
output_type: generate_until
include: _default_template_yaml
doc_to_visual: !function utils.activitynetqa_doc_to_visual
doc_to_text: !function utils.activitynetqa_doc_to_text
doc_to_target: !function utils.activitynetqa_doc_to_answer
process_results: !function utils.activitynetqa_process_results
metric_list:
  - metric: gpt_eval_accuracy
    aggregation: !function utils.activitynetqa_aggregate_accuracy
    higher_is_better: true
```

Run: `--evaluation_engine="lmms_eval" --benchmarks="<task>"`.

## Custom Parquets

For heavy datasets (typically video), load from a local parquet instead of the Hub:

1. Put the file at `resources/custom_parquets/<task_name>/<file>.parquet`.
2. In the task's `dataset_kwargs`, point to it (path relative to `custom_parquets/`):

```yaml
# custom_tasks/lvbench/lvbench.yaml (abridged)
dataset_path: gwkrsrch2/lvbench
dataset_kwargs:
  custom_parquet_path: lvbench/test-00000-of-00001-b27242e870ecc6da.parquet
test_split: test   # the parquet is always loaded as the `test` split
```

One `custom_parquet_path` = one file. **Alternative:** to pin a local HF snapshot instead, add the repo
id to `resources/custom_snapshots.yaml` (`repo_id: [paths...]` — first existing path wins, else normal Hub download).
