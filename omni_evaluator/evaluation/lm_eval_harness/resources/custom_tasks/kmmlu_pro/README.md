# KMMLU-Pro

### Paper

Title: `KMMLU-Pro: Korean Massive Multi-task Language Understanding Pro`

Paper: [https://arxiv.org/pdf/2507.08924](https://arxiv.org/pdf/2507.08924)

KMMLU-Pro is an enhanced version of the KMMLU benchmark featuring more diverse Korean domains and improved data quality. It provides rigorous evaluation of large language models' understanding of Korean language and culture across multiple domains.

Homepage: [https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro](https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro)

### Dataset Information

KMMLU-Pro consists of Korean multiple-choice questions covering various domains including:
- Science and technology
- Humanities
- Social sciences
- Professional knowledge
- And more

### Citation

```bibtex
@dataset{kmmlu_pro,
  title={KMMLU-Pro: Korean Massive Multi-task Language Understanding Pro},
  author={LGAI-EXAONE},
  year={2024},
  url={https://huggingface.co/datasets/LGAI-EXAONE/KMMLU-Pro}
}
```

### Groups and Tasks

#### Groups

* `kmmlu_pro`: Main KMMLU-Pro evaluation task

#### Tasks

* `kmmlu_pro`: Evaluates Korean multiple-choice question answering

### Evaluation

- **Metric**: Exact Match (EM)
- **Answer Format**: Models generate answers which are then extracted using Korean-aware regex patterns
- **Scoring**: Case-insensitive and punctuation-insensitive exact match aggregated by mean
