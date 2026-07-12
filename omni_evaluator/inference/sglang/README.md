# SGLang Inference Engine

Performs inference by sending API requests to a pre-launched SGLang server.

## Features

| Item | Description |
|------|-------------|
| GPU Usage | **NO** — The external SGLang server occupies the GPU |
| External API | **YES** — An SGLang server must be running beforehand |
| Async Support | Parallel requests via `--do_async` |
| Supported Modalities | text, image, video, audio (depends on the server model) |

## Required Environment Variables

No environment variables are required.

## Prerequisites

You must start the SGLang server first:
```bash
python -m sglang.launch_server \
    --model-path="Qwen/Qwen2.5-VL-3B-Instruct" \
    --port=30000
```

## Engine-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | **(required)** | SGLang server URL |

## Generation Options

The following generation options from `GenerationOptionArgs` are supported by this engine:

| Argument | Default | Description |
|----------|---------|-------------|
| `--temperature` | `None` | Sampling temperature |
| `--top_p` | `None` | Top-p (nucleus) sampling threshold |
| `--top_k` | `None` | Top-k filtering value |
| `--max_new_tokens` | `None` | Maximum tokens to generate (mapped to `max_tokens`) |
| `--repetition_penalty` | `None` | Repetition penalty |
| `--stop_words` | `None` | Comma-separated stop sequences (mapped to `stop`) |
| `--n` | `None` | Number of independent output sequences to generate |
| `--logprobs` | `None` | Number of log probability tokens to return per step |
| `--seed` | `None` | Random seed for reproducibility |
| `--do_sample` | `None` | Enable sampling |

> `--num_beams`, `--length_penalty`, `--frequency_penalty`, `--presence_penalty`, `--top_logprobs` are not supported by this engine and will be ignored.

## Examples

### With builtin Evaluation Engine
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

### With lmms_eval Evaluation Engine
```bash
python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="sglang" \
    --url="http://localhost:30000" \
    --exp_name="my-experiment" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="videomme"
```
