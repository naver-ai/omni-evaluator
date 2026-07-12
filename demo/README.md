<div align="center">

# OmniEvaluator Demo

OmniEvaluator is a composable evaluation system for reproducible omni-modal foundation model evaluation. The demo video below walks through three core workflows of OmniEvaluator.

▶️ [Watch Demo Video](./demo.mp4)

</div>

---

## ⚡ Colab Quickstart

No local setup — run the whole pipeline (install → evaluate → results) in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naver-ai/omni-evaluator/blob/main/demo/Colab_Quickstart.ipynb)

Runs `Qwen/Qwen2.5-VL-3B-Instruct` × `vlm_eval_kit` on `AI2D_TEST` in `--debug` mode (3 samples) on a free T4 GPU — just run every cell top to bottom. ([notebook](./Colab_Quickstart.ipynb))

---

## Scenario 1 — Local Evaluation

CLI-based evaluation on a locally configured environment. The user specifies the inference engine, evaluation framework, and benchmarks through a single `python run.py evaluate` command.

**Example: evaluate with a running vLLM server**

```bash
python run.py evaluate \
    --inference_engine="vllm" \
    --url="http://localhost:8000" \
    --exp_name="my-model-eval" \
    --evaluation_engine="vlm_eval_kit" \
    --benchmarks="MMStar,TextVQA_VAL,MMMU_DEV_VAL" \
    --do_async \
    --verbose
```

**Example: evaluate with a local HuggingFace model**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py evaluate \
    --inference_engine="huggingface" \
    --model_name_or_path="Qwen/Qwen2.5-VL-7B-Instruct" \
    --exp_name="my-model-eval" \
    --evaluation_engine="vlm_eval_kit" \
    --benchmarks="MMStar,TextVQA_VAL" \
    --torch_dtype="float16" \
    --world_size=4 \
    --do_async \
    --verbose
```

Upon completion, OmniEvaluator writes a provenance-rich evaluation artifact (JSON) capturing the full configuration, per-sample predictions, and aggregated metrics—enabling exact reproduction of any run.

---

## Scenario 2 — Remote Evaluation

Server-based evaluation via REST API. An administrator launches a persistent evaluation server once, and team members submit evaluation requests without any local environment setup—ensuring identical inference conditions across all users.

**Step 1: Start the evaluation server (administrator)**

```bash
python launch_server.py \
    --host 0.0.0.0 \
    --port 8080 \
    --base "python evaluate.py" \
    --max_concurrent 1 \
    --log_dir "./logs"
```

**Step 2: Submit an evaluation job (team member)**

```bash
# Submit a job
curl -X POST http://<server>:8080/add_job \
    -H "Content-Type: application/json" \
    -d '{
        "arguments": "--inference_engine=vllm --url=http://localhost:8000 --exp_name=my-experiment --evaluation_engine=vlm_eval_kit --benchmarks=MMStar --do_async"
    }'
# → returns {"pid": "<job_pid>"}

# Poll job status: pending → inprogress → completed / failed
curl -X POST http://<server>:8080/get_state \
    -H "Content-Type: application/json" \
    -d '{"pid": "<job_pid>"}'

# List all jobs
curl http://<server>:8080/jobs
```

---

## Scenario 3 — Dashboard

An integrated web dashboard hosted at **[omni-evaluator.info](https://omni-evaluator.info)** that reads evaluation artifacts and provides:

- **Cross-modal benchmark visualization** — compare scores across image, video, audio, and text benchmarks in a unified view
- **Omni Score** — a single composite score for model selection across heterogeneous benchmarks
- **Side-by-side sample comparison** — inspect per-sample predictions from multiple experiments simultaneously

The dashboard synchronizes with evaluation artifacts automatically as new results are written.
