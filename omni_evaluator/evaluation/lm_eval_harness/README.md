# LM-Eval-Harness Evaluation Engine

A text-only benchmark evaluation engine based on the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework.

## Features

| Item | Description |
|------|-------------|
| GPU Usage | **NO** — Only performs data preparation and metric computation |
| External API | Not required |
| Number of Tasks | 7,778 |
| Supported Modalities | text only |

Supports two output types: `generate_until` (generation) and `multiple_choice` (selection), and provides per-category group metrics and bootstrap confidence intervals.

## Required Environment Variables

| Environment Variable | Description |
|----------------------|-------------|
| `HF_TOKEN` | Hugging Face Hub authentication token (for dataset downloads) |
| `HF_HOME` | Hugging Face cache directory |
| `HF_HUB_CACHE` | Hugging Face Hub cache directory |
| `HF_ALLOW_CODE_EVAL` | Code evaluation permission flag (required) |

## Engine-specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--num_fewshot` | `None` | Number of few-shot examples; uses task defaults if `None` |

## Examples

### Using with the Hugging Face Inference Engine (Multi-GPU)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="huggingface" \
    --model_name_or_path="naver-hyperclovax/HyperCLOVAX-SEED-Vision-Instruct-3B" \
    --exp_name="my-experiment" \
    --evaluation_engine="lm_eval_harness" \
    --benchmarks="humaneval" \
    --torch_dtype="float16" \
    --world_size=4
```

### Using with the vLLM Inference Engine
```bash
python run.py evaluate \
    --do_async \
    --verbose \
    --inference_engine="vllm" \
    --url="http://localhost:8000" \
    --exp_name="my-experiment" \
    --evaluation_engine="lm_eval_harness" \
    --benchmarks="mmlu"
```

## Adding Custom Tasks

Add a task folder under `resources/custom_tasks/<task_name>/` in standard
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) format (text-only). At startup
the engine **copies these folders into the installed `lm_eval/tasks/`** (same-name files are overwritten;
`example_*` folders are skipped). A folder has a `<task>.yaml` and, if scoring needs custom logic, a
`utils.py`. See `example_aime25/`.

```yaml
# <task>.yaml — doc_to_* are Jinja templates over the dataset columns
task: aime25
dataset_path: math-ai/aime25          # HF Hub repo
output_type: generate_until           # or: multiple_choice (logprob selection)
test_split: test
doc_to_text: "Solve the problem. Put your final answer in \\boxed{}.\n\n{{problem}}"
doc_to_target: "{{answer}}"
process_results: !function utils.process_results   # optional custom scorer
metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
generation_kwargs: { until: ["<|endoftext|>"], do_sample: false, max_gen_toks: 32768 }
```

```python
# utils.py (optional) — only when built-in metrics aren't enough
from math_verify import parse, verify

def process_results(doc, results):
    gold = parse("\\boxed{" + str(doc["answer"]) + "}")
    return {"exact_match": int(verify(gold, parse(results[0])))}
```

Run: `--evaluation_engine="lm_eval_harness" --benchmarks="<task>"`.
