<div align="center">

# OmniEvaluator

A unified evaluation toolkit for LLMs and multimodal models.

Supports on-device and API-based inference with **Hugging Face**, **vLLM**, and **SGLang**, and enables evaluation across **text**, **image**, **video**, and **audio** benchmarks via multiple evaluation engines.

</div>


## Overview

OmniEvaluator provides a single interface to evaluate language and multimodal models across a wide range of benchmarks. It integrates with multiple inference backends and evaluation frameworks, allowing consistent and reproducible evaluation workflows.

**Inference Engines**: `huggingface`, `vllm`, `sglang`, `api/openai`, `api/anthropic`, `api/google`

**Evaluation Engines**: `builtin`, `lmms_eval`, `lm_eval_harness`, `vlm_eval_kit`, `audio_bench`


## Quick Start

### Set environment variables

Create a `.env` file in the project root:
```dotenv
HF_TOKEN={your-hf-token}
HF_HOME=/path/to/huggingface/cache
HF_HUB_CACHE=${HF_HOME}/hub
OPENAI_API_KEY={your-openai-api-key}  # for LLM-judge benchmarks
```

### Example: Hugging Face inference
```bash
CUDA_VISIBLE_DEVICES=0 python -m omni_eval evaluate \
    --do_async \
    --verbose \
    --inference_engine="huggingface" \
    --model_name_or_path="your-model-name-or-path" \
    --exp_name="my-experiment" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="textvqa_val" \
    --torch_dtype="float16"
```

### Example: vLLM inference
```bash
python -m omni_eval evaluate \
    --do_async \
    --verbose \
    --inference_engine="vllm" \
    --url="http://localhost:8000" \
    --exp_name="my-experiment" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="videomme"
```

### Example: SGLang inference
```bash
python -m omni_eval evaluate \
    --do_async \
    --verbose \
    --inference_engine="sglang" \
    --url="http://localhost:30000" \
    --exp_name="my-experiment" \
    --evaluation_engine="lmms_eval" \
    --benchmarks="mmmu_val"
```

### Example: lm_eval_harness (text-only)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m omni_eval evaluate \
    --do_async \
    --verbose \
    --inference_engine="huggingface" \
    --model_name_or_path="your-model-name-or-path" \
    --exp_name="my-experiment" \
    --evaluation_engine="lm_eval_harness" \
    --benchmarks="humaneval" \
    --torch_dtype="float16" \
    --world_size=4
```

### CLI utilities
```bash
# List available inference engines
python -m omni_eval list --inference_engines

# List available evaluation engines
python -m omni_eval list --evaluation_engines

# List supported tasks for an evaluation engine
python -m omni_eval list --tasks --evaluation_engine="lmms_eval"
```


## Supported Benchmarks

### Builtin (82 tasks)
Vision, document understanding, math, multi-image, and Korean benchmarks.

`docvqa`, `infovqa`, `chartqa`, `mathvista`, `mathverse`, `scienceqa`, `textvqa`, `vqav2`, `seedbench`, `seedbench_ko`, `mmmu`, `gqa`, `pope`, `mmbench`, `ai2d`, `mme`, `hallusion`, `amber`, `mmvet`, `refcoco_*`, `mia_bench`, ...

### LMMs-Eval (747 tasks)
Extensive vision-language benchmarks from the [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework.

`ai2d`, `chartqa`, `docvqa`, `mathvista`, `mmmu`, `mmbench`, `videomme`, `realworldqa`, `mmstar`, `ocrbench`, `scienceqa`, `textvqa`, `vqav2`, `wildvision`, ...

### Lm-Eval-Harness (7,778 tasks)
Comprehensive text-only benchmarks from [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

`mmlu`, `mmlu_pro`, `gpqa`, `humaneval`, `mbpp`, `gsm8k`, `math`, `arc_challenge`, `hellaswag`, `winogrande`, `ifeval`, `bbh`, `kmmlu`, ...

### VLMEvalKit (490 tasks)
Vision-language benchmarks from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

`MME`, `MMBench`, `POPE`, `HallusionBench`, `AMBER`, `COCO_VAL`, `OCRBench`, `MathVista`, `RealWorldQA`, `SEEDBench`, `Video-MME`, ...

### AudioBench (76 tasks)
Audio understanding benchmarks from [AudioBench](https://github.com/AudioLLMs/AudioBench).

`librispeech_test_clean`, `common_voice_15_en_test`, `covost2_*`, `fleurs_*`, `clotho_aqa_test`, `mmau_test`, `vocal_sound_test`, `gtzan_test`, ...