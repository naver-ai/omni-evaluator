# vLLM Inference Engine

Performs inference by sending API requests to a pre-launched vLLM server.

## Features

| Item | Description |
|------|-------------|
| GPU Usage | **NO** — The external vLLM server occupies the GPU |
| External API | **YES** — A vLLM server must be running beforehand |
| Async Support | Parallel requests via `--do_async` |
| Supported Modalities | text, image, video, audio (depends on the server model) |

## Required Environment Variables

| Environment Variable | Description |
|----------------------|-------------|
| `VLLM_API_KEY` | vLLM server authentication key (required when `--api-key` is set on the server) |

## Prerequisites

You must start the vLLM server first:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model="Qwen/Qwen2.5-VL-3B-Instruct" \
    --port=8000
```

## Engine-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--url` | **(required)** | vLLM server URL |
| `--vllm_api_version` | `v1` | vLLM API version |
| `--vllm_model_name` | `None` | Model name registered on the server |
| `--vllm_api_key` | `None` | Server authentication API key |
| `--model_name_or_path` | `None` | Model path for local vllm.LLM inference |
| `--trust_remote_code` | `false` | Allow remote code execution from the Hub |
| `--skip_chat_template` | `false` | Skip applying the chat template |
| `--add_generation_prompt` | `false` | Add a generation prompt when applying the chat template |
| `--chat_template_kwargs` | `None` | Extra kwargs for the chat template (JSON string) |
| `--mm_processor_kwargs` | `None` | Kwargs for the multimodal processor (JSON string) |

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
    --inference_engine="vllm" \
    --url="http://localhost:8000" \
    --exp_name="my-experiment" \
    --evaluation_engine="builtin" \
    --benchmarks="mmbench_en_dev"
```

### With lmms_eval Evaluation Engine
```bash
python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="vllm" \
    --url="http://localhost:8000" \
    --exp_name="my-experiment" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="textvqa_val"
```
