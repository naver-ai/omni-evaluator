# AIME (American Invitational Mathematics Examination) Benchmark

## Overview

Custom AIME benchmark implementation for LM Evaluation Harness, featuring American Invitational Mathematics Examination problems from 2024 and 2025.

## Usage

Use the following task names in your evaluation:

- `aime` - Combined AIME problems
- `aime24` - AIME 2024
- `aime25` - AIME 2025

## Key Difference: Math Verify Scoring

Unlike standard LM-Eval, this implementation uses **Math Verify** for robust scoring:

- Handles mathematically equivalent answers (e.g., `1/2` vs `0.5`)
- Reduces false negatives from formatting variations
- Ensures mathematical correctness beyond simple string matching

This provides fairer and more accurate evaluation of mathematical reasoning abilities.

