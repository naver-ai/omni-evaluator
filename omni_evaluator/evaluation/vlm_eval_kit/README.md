# VLMEvalKit Evaluation Engine

A vision-language benchmark evaluation engine based on the [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) framework.

## Features

| Item | Description |
|------|-------------|
| GPU Usage | **NO** — Only performs dataset construction and evaluation |
| External API | LLM-judge API required for some benchmarks (e.g., GPT-4o) |
| Number of Tasks | 490 |
| Supported Modalities | text, image, video |

Different judge models are used per benchmark:
- GPT-4o: Complex reasoning benchmarks
- GPT-4-turbo: Video benchmarks
- ChatGPT-0125: MCQ benchmarks

## Required Environment Variables

| Environment Variable | Description |
|----------------------|-------------|
| `HF_TOKEN` | Hugging Face Hub authentication token (for dataset downloads) |
| `HF_HOME` | Hugging Face cache directory (+ auto-configures VLMEvalKit's `LMUData`) |
| `HF_HUB_CACHE` | Hugging Face Hub cache directory |
| `OPENAI_API_KEY` | Required for LLM-judge evaluation (optional depending on benchmark) |

Optional environment variables:

| Environment Variable | Description |
|----------------------|-------------|
| `EVAL_PROXY` | Proxy for evaluation requests |
| `HTTP_PROXY` | HTTP proxy setting |

## Engine-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `None` | Path to VLMEvalKit config JSON file |
| `--judge` | `None` | Model name for LLM-as-judge evaluation |
| `--judge_args` | `None` | JSON-encoded arguments for the judge model |
| `--retry` | `None` | Maximum number of retries for judge API calls |
| `--api_nproc` | `None` | Number of parallel processes for judge API |

## Examples

### Using with the Hugging Face Inference Engine
```bash
CUDA_VISIBLE_DEVICES=0 python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="huggingface" \
    --model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct" \
    --exp_name="qwen2.5-vl-3b" \
    --evaluation_engine="vlm_eval_kit" \
    --benchmarks="MMBench_DEV_EN_V11" \
    --torch_dtype="bfloat16"
```

### Using with the vLLM Inference Engine
```bash
python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="vllm" \
    --url="http://localhost:8000" \
    --exp_name="my-experiment" \
    --evaluation_engine="vlm_eval_kit" \
    --benchmarks="MME"
```
