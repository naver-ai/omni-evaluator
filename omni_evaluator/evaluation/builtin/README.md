# Builtin Evaluation Engine

A self-implemented evaluation framework that supports various vision/language benchmarks based on YAML configuration.

## Features

| Item | Description |
|------|-------------|
| GPU Usage | **Usually NO** — only data preparation and metric computation. *Exception:* the optional verifier metric with `--verifier_engine huggingface` loads a local judge model on GPU. |
| External API | `OPENAI_API_KEY` required for LLM-judge metrics and the verifier's `api/*` backend (default) |
| Number of Tasks | 82 |
| Supported Modalities | text, image, video, multi-image |

Each task is defined in `tasks/{task_name}/config.yaml`, and custom data loading and evaluation logic can be specified through `custom.py`.

## Required Environment Variables

| Environment Variable | Description |
|----------------------|-------------|
| `OPENAI_API_KEY` | Required for LLM-judge-based metrics and the verifier's `api/*` backend (default `gpt-5-mini`) |

## Engine-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--local_dirpath` | `None` | Local dataset path; downloads from remote storage if `None` |
| `--subtask_type` | `None` | Type to override the subtask defined in the task config |
| `--num_fewshot` | `0` | Number of few-shot examples |
| `--fewshot_image_max_size` | `224` | Maximum size of few-shot images (px) |
| `--do_cot` | `false` | [deprecated] Add chain-of-thought prompts |

## Examples

### Using with the Hugging Face Inference Engine
```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="huggingface" \
    --model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct" \
    --exp_name="qwen2.5-vl-3b" \
    --evaluation_engine="builtin" \
    --benchmarks="mmbench_en_dev" \
    --torch_dtype="bfloat16"
```

### Using with the SGLang Inference Engine
```bash
python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="sglang" \
    --url="http://localhost:30000" \
    --exp_name="my-experiment" \
    --evaluation_engine="builtin" \
    --benchmarks="mmbench_en_dev"
```
