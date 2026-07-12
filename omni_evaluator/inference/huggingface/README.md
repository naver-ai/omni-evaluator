# Hugging Face Inference Engine

Performs inference by loading models directly onto local GPUs.

## Features

| Item | Description |
|------|-------------|
| GPU Usage | **YES** — Loads the model directly into GPU memory |
| External API | Not required |
| Distributed Inference | Multi-GPU support via `device_map="auto"` + `world_size` |
| Supported Modalities | text, image, video, audio |

Supports over 30 model architectures (Qwen2-VL, LLaVA, Phi, HyperCLOVA-VLM, etc.).

## Required Environment Variables

| Environment Variable | Description |
|----------------------|-------------|
| `HF_TOKEN` | Hugging Face Hub token (for model downloads) |
| `CUDA_VISIBLE_DEVICES` | Specifies the GPU devices to use |

## Engine-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name_or_path` | **(required)** | Hugging Face model name or local path |
| `--device_map` | `auto` | Model device placement strategy |
| `--trust_remote_code` | `false` | Allow remote code execution from the Hub |
| `--low_cpu_mem_usage` | `false` | CPU memory saving mode during `from_pretrained` |
| `--skip_chat_template` | `false` | Skip `apply_chat_template` and only perform tokenization |
| `--model_kwargs` | `null` | Additional kwargs passed to the huggingface Module `__init__` (JSON string or dict) |

## Generation Options

The following generation options from `GenerationOptionArgs` are supported by this engine:

| Argument | Default | Description |
|----------|---------|-------------|
| `--temperature` | `None` | Sampling temperature |
| `--top_p` | `None` | Top-p (nucleus) sampling threshold |
| `--top_k` | `None` | Top-k filtering value |
| `--num_beams` | `None` | Number of beams for beam search |
| `--max_new_tokens` | `None` | Maximum number of new tokens to generate |
| `--repetition_penalty` | `None` | Repetition penalty |
| `--length_penalty` | `None` | Length penalty |
| `--stop_words` | `None` | Comma-separated stop sequences |
| `--do_sample` | `None` | Enable sampling; `None` uses model default |

> `--frequency_penalty`, `--presence_penalty`, `--n`, `--logprobs`, `--top_logprobs`, `--seed` are not supported by this engine and will be ignored.

## Adding a New Model

To add support for a new model, modify the following files in order:

### 1. Register the model group — `omni_evaluator/__init__.py`

Add a new entry to `HuggingfaceModelGroup`:

```python
class HuggingfaceModelGroup(str, Enum):
    ...
    your_model: str = "your_model"
```

### 2. Implement the module — `omni_evaluator/inference/huggingface/your_model.py`

Create a new file that inherits from `HuggingfaceModule` and implements `ENGINE_FEATURES` and `generate_text()`.
Use an existing implementation as a reference (e.g., `qwen2_vl.py` for image+text, `qwen2.py` for text-only).

```python
from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.inference import InferenceEngineFeatures, HuggingfaceInferenceOutput

class YourModelModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_text_understanding=True,
        support_image_understanding=True,   # set per modality
        support_video_understanding=False,
        support_audio_understanding=False,
        support_text_generation=True,
        support_compute_perplexity=False,
        support_image_generation=False,
        support_audio_generation=False,
    ).to_dict()

    def __init__(self, *args, **kwargs):
        HuggingfaceModule.__init__(self, *args, **kwargs)
        self.model = ...      # load model
        self.processor = ...  # load processor/tokenizer

    def generate_text(self, messages, generation_options=None, **kwargs):
        ...
        return HuggingfaceInferenceOutput(prediction="...", ...)

    # Optional — only needed if support_compute_perplexity=True
    def compute_perplexity(self, messages, options=None, **kwargs):
        ...
        return HuggingfaceInferenceOutput(perplexities=[...], prediction=0)
```

### 3. Wire up the engine — `omni_evaluator/inference/huggingface/engine.py`

**3-A.** Add an import at the top of the file:
```python
from omni_evaluator.inference.huggingface.your_model import YourModelModule
```

**3-B.** Add a pattern-matching branch in `get_model_group()` (check for keyword conflicts with existing branches):
```python
elif "your_model_keyword" in model_name_or_path.lower():
    return HuggingfaceModelGroup.your_model
```

**3-C.** Add an instantiation branch in `__init__()`:
```python
elif self.model_group == HuggingfaceModelGroup.your_model:
    self.module = YourModelModule(**common_kwargs, **model_kwargs)
```

### 4. (Optional) Add model-specific CLI arguments — `omni_evaluator/args.py`

If your model requires additional parameters (e.g., `min_pixels`), add them to `HuggingfaceInferenceEngineArgs` and pass them through `common_kwargs` or `model_kwargs` in `engine.py`.

---

## Examples

### With builtin Evaluation Engine
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

### With lmms_eval Evaluation Engine
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
