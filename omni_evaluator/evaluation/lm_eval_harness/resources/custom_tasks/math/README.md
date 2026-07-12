# MATH

## Overview

This is a modified version of the `hendrycks_math` benchmark that uses `math_verify` for parsing and evaluation. The task structure and categories are identical to the original Hendrycks MATH benchmark, but the answer extraction and verification are handled through mathematical parsing rather than string matching, making it more robust for evaluating recent instruction-tuned models.

## Paper
Measuring Mathematical Problem Solving With the MATH Dataset
https://arxiv.org/abs/2103.03874

Many intellectual endeavors require mathematical problem solving, but this skill remains beyond the capabilities of computers. To measure this ability in machine learning models, we introduce MATH, a new dataset of 12,500 challenging competition mathematics problems. Each problem in MATH has a full step-by-step solution which can be used to teach models to generate answer derivations and explanations.

## Key Differences from Standard MATH

- **Answer Parsing**: Uses `math_verify` library to parse answers mathematically instead of string-based extraction
- **Verification**: Employs mathematical equivalence verification instead of exact string matching
- **Evaluation**: More suitable for evaluating chain-of-thought responses and instruction-tuned models where mathematical correctness is paramount

NOTE: This task is based on the MATH (`hendrycks_math`) implementation at https://github.com/EleutherAI/lm-evaluation-harness/tree/master, but with `math_verify`-based evaluation.


## Citation
```
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
```

### Groups and Tasks

#### Groups

- `math`: Math benchmark with `math_verify`-based evaluation (modified version of Hendrycks MATH).

#### Tasks

- `math_algebra`
- `math_counting_and_prob`
- `math_geometry`
- `math_intermediate_algebra`
- `math_num_theory`
- `math_prealgebra`
- `math_precalc`
