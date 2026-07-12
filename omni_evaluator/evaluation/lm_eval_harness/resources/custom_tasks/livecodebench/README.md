# LiveCodeBench

LiveCodeBench is a benchmark for evaluating code generation models on live programming problems with realistic and up-to-date test cases.

## Dataset Information

- **Path**: `lighteval/code_generation_lite`
- **Content**: Programming problems with starter code and test cases

## Prompt Format

### With Starter Code:
```
### Question:
{problem_description}

### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.
```python
{starter_code}
```

### Answer: (use the provided format with backticks)
```

### Without Starter Code:
```
### Question:
{problem_description}

### Format: Read the inputs from stdin solve the problem and write the answer to stdout...
```python
# YOUR CODE HERE
```

### Answer: (use the provided format with backticks)
```

## Code Extraction

Regular expression: `(?<=```python\n)((?:\n|.)+?)(?=\n```)`

Extracts code blocks enclosed in python markdown format.

## Evaluation Metrics

- **pass@k**: Percentage of problems where at least one of k generations passes all test cases
- Default: k=[1]

## Configuration

- **Temperature**: 0.0 (deterministic)
- **Max tokens**: 2048
- **Do sample**: false
