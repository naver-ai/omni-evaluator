# Demo

[![OmniEvaluator Demo](https://github.com/naver-ai/omni-evaluator/blob/main/demo/thumbnail.png)]

**OmniEvaluator** is a composable system for reproducible omni-modal (text, image, video, audio) model evaluation. The demo video walks through three core workflows.

▶️ [Watch Demo Video](https://www.youtube.com/watch?v=4Z5VZZWyXqY)

## 1. Local Evaluation

CLI-based evaluation on a local environment. A single `python run.py evaluate` command specifies the inference engine, evaluation framework, and benchmarks, and runs everything on local GPUs—ideal for rapid, isolated testing.

## 2. Remote Evaluation

Server-based evaluation via REST API. An administrator launches a persistent evaluation server once, and team members submit jobs with a lightweight `curl` request—no local setup required. Evaluation runs on a decoupled server, so it can run in parallel with active training without consuming the training cluster's GPU resources.

## 3. Dashboard

An integrated web dashboard that reads evaluation artifacts and provides a cross-modal **Leaderboard** for comparing scores across benchmarks and an **Inference Viewer** for inspecting per-sample predictions and metrics.
