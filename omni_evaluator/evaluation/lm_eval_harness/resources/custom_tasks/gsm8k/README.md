# GSM8K

This is a modified version of the GSM8K benchmark optimized for generative evaluation of recent instruct-tuned chat models using mathematical verification.

## Overview

Grade School Math 8K (GSM8K) is a dataset of 8.5K high quality linguistically diverse grade school math word problems. The problems take between 2 and 8 steps to solve, and solutions involve arithmetic operations on whole numbers.

## Key Modifications

- **Output Format**: Models are instructed to put answers in `\boxed{}` format, making evaluation more robust and consistent
- **Scoring Method**: Uses `math_verify` library instead of standard string matching, enabling accurate evaluation of mathematically equivalent answers
- **Design**: Optimized for evaluating recent instruct-tuned chat models where mathematical verification is more appropriate than exact string matching

## Dataset Information

### Paper

Title: `Training Verifiers to Solve Math Word Problems`

Abstract: State-of-the-art language models can match or exceed human performance on many tasks, but they still struggle with basic math. We investigate how to train a verifier to check if a proposed solution to a math word problem is correct. We demonstrate that a verifier trained on relatively simple, synthetic data can achieve high accuracy.

Homepage: [https://huggingface.co/datasets/openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)

### Citation

```bibtex
@misc{cobbe2021training,
      title={Training Verifiers to Solve Math Word Problems},
      author={Karl Cobbe and Vineet Kosaraju and Mohammad Bavarian and Mark Chen and Heewoo Jun and Lukas Kaiser and Matthias Plappert and Jerry Tworek and Jacob Hilton and Reiichiro Nakano and Christopher Hesse and Jie Tang},
      year={2021},
      eprint={2110.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Groups and Tasks

* `gsm8k`: Main task using `math_verify` for scoring

### Scoring

The benchmark uses `math_verify` library to evaluate answers:
- Parses the gold answer and model-generated answer mathematically
- Verifies if both answers are mathematically equivalent
- Reports exact_match metric aggregated by mean

This approach is more suitable than string-based matching for evaluating instruction-following chat models on mathematical problems.
