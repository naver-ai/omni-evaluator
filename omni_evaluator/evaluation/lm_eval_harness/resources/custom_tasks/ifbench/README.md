# IFBench

### Paper

Title: IFBench: Instruction Following Benchmark for Large Language Models
Abstract: [https://arxiv.org/abs/2507.02833](https://arxiv.org/abs/2507.02833)

IFBench is a comprehensive instruction-following benchmark developed by Allen Institute for AI. It evaluates how well Large Language Models (LLMs) can follow complex instructions with multiple constraints. The benchmark includes various instruction types and provides both strict and loose evaluation metrics to assess instruction-following capabilities.

Homepage: [https://huggingface.co/datasets/allenai/IFBench_test](https://huggingface.co/datasets/allenai/IFBench_test)

### Dataset Information

The benchmark evaluates models on their ability to follow verifiable instructions such as:
- Word count constraints (e.g., "write more than 400 words")
- Keyword mentions (e.g., "mention the keyword 'AI' at least 3 times")
- Format constraints (e.g., "use bullet points")
- And various other instruction types

### Citation

```bibtex
@dataset{ifbench,
  title={IFBench: Instruction Following Benchmark},
  author={Allen Institute for AI},
  year={2024},
  url={https://huggingface.co/datasets/allenai/IFBench_test}
}
```

### Groups and Tasks

#### Groups

* Not part of a group yet

#### Tasks

* `ifbench`: Main instruction-following evaluation task

### Metrics

The benchmark provides multiple evaluation metrics:

* `prompt_level_strict_acc`: Strict accuracy at prompt level (all instructions must be followed)
* `inst_level_strict_acc`: Strict accuracy at instruction level (individual instruction compliance)
* `prompt_level_loose_acc`: Loose accuracy at prompt level (partial credit for partial compliance)
* `inst_level_loose_acc`: Loose accuracy at instruction level (lenient instruction compliance)

The "strict" metrics require all instructions to be perfectly followed, while "loose" metrics allow for more lenient interpretation of instruction compliance, making them more suitable for evaluating practical instruction-following capabilities of chat models.
