# API Inference Engine

Performs inference by calling external AI service APIs (OpenAI, Anthropic, Google).

## Features

| Item | Description |
|------|-------------|
| GPU Usage | **NO** — Uses external cloud APIs |
| External API | **YES** — Requires an API key from each provider |
| Async Support | Parallel requests via `--do_async` (except Google) |
| Supported Modalities | text, image (varies by provider) |

Set the `inference_engine` value to one of `api/openai`, `api/anthropic`, or `api/google`.

## Required Environment Variables

You must set the appropriate key depending on the API provider you are using:

| Environment Variable | Provider | Description |
|----------------------|----------|-------------|
| `OPENAI_API_KEY` | OpenAI | OpenAI API authentication key |
| `ANTHROPIC_API_KEY` | Anthropic | Anthropic API authentication key |
| `GOOGLE_API_KEY` | Google | Google AI API authentication key |

## Engine-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--api_name` | **(required)** | API model name (e.g., `gpt-4o`, `claude-3-opus`, `gemini-pro`) |

## Generation Options

The following generation options from `GenerationOptionArgs` are available. Support varies by provider:

| Argument | Default | OpenAI | Anthropic | Google | Description |
|----------|---------|--------|-----------|--------|-------------|
| `--temperature` | `None` | ✅ | ✅ | ✅ | Sampling temperature |
| `--top_p` | `None` | ✅ | ✅ (exclusive w/ temperature) | ✅ | Top-p sampling threshold |
| `--top_k` | `None` | ❌ | ✅ | ✅ | Top-k filtering value |
| `--max_new_tokens` | `None` | ✅ → `max_tokens` | ✅ → `max_tokens` | ✅ → `maxOutputTokens` | Max tokens to generate |
| `--stop_words` | `None` | ✅ → `stop` | ✅ → `stop_sequences` | ✅ → `stopSequences` | Stop sequences |
| `--frequency_penalty` | `None` | ✅ | ❌ | ✅ → `frequencyPenalty` | Frequency penalty |
| `--repetition_penalty` | `None` | ✅ → `frequency_penalty` | ❌ | ✅ → `frequencyPenalty` | Mapped to frequency penalty |
| `--presence_penalty` | `None` | ✅ | ❌ | ✅ → `presencePenalty` | Presence penalty |
| `--n` | `None` | ✅ | ❌ | ❌ | Number of output sequences |
| `--logprobs` | `None` | ✅ → `bool` + `top_logprobs` | ❌ | ✅ (enables `responseLogprobs`) | Log probabilities |
| `--top_logprobs` | `None` | ✅ | ❌ | ❌ | Top token log probabilities (OpenAI only) |

> For reasoning models (e.g., `o1`, `o3`), `--temperature`, `--top_p`, `--frequency_penalty`, `--presence_penalty`, `--logprobs`, `--stop_words` are automatically removed. Use `--reasoning_effort` and `--max_new_tokens` instead.
>
> For Anthropic extended thinking models, use `--thinking_budget`. `--temperature` and `--top_p` are automatically removed when thinking is enabled.

## Examples

### OpenAI + builtin Evaluation Engine
```bash
python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="api/openai" \
    --api_name="gpt-4o" \
    --exp_name="gpt4o-eval" \
    --evaluation_engine="builtin" \
    --benchmarks="mmbench_en_dev"
```

### Anthropic + lmms_eval Evaluation Engine
```bash
python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="api/anthropic" \
    --api_name="claude-3-opus-20240229" \
    --exp_name="claude3-eval" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="textvqa_val"
```

## Notes

- The Google API does not support asynchronous (`--do_async`) requests.
- The Anthropic API handles system messages separately.
- Image encoding methods differ by provider (URL vs base64).
