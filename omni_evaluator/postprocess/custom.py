# OmniEvaluator
# Copyright (c) 2026-present NAVER Cloud Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import logging
import re
from typing import List, Tuple, Dict, Any, Optional, Union

from omni_evaluator.evaluation.metrics.constants import (
    vqa__circledNumbersMap,
)

logger = logging.getLogger(__name__)


def parse_think(
    prediction: str,
    think_start_pattern: Union[str, bool] = "<think>",
    think_end_pattern: Union[str, List[str], bool] = "</think>",
    eot_token: Union[str, List[str]] = "<|im_end|>",
    **kwargs,
):
    if (
        think_start_pattern
        and not isinstance(think_start_pattern, str)
    ):
        think_start_pattern = "<think>"
    if isinstance(think_end_pattern, str):
        think_end_pattern = [think_end_pattern]
    elif not isinstance(think_end_pattern, list):
        think_end_pattern = ["</think>"]
    if isinstance(eot_token, str):
        eot_token = [eot_token]
    elif not isinstance(eot_token, list):
        eot_token = ["<|im_end|>"]

    matched_end_pattern = None
    matched_eot_token = None
    for _end_pattern, _eot in itertools.product(think_end_pattern, eot_token):
        if re.search(rf"{_end_pattern}", prediction):
            matched_end_pattern = _end_pattern
            matched_eot_token = _eot
            break

    reasoning_content = None
    if matched_end_pattern is not None:
        reasoning_content = re.split(
            pattern=rf"{matched_end_pattern}",
            string=prediction,
            flags=re.DOTALL,
        )[0].strip()
        if isinstance(reasoning_content, str):
            reasoning_content = reasoning_content.replace(think_start_pattern, "")
        _prediction = parse_last_pattern(
            prediction,
            prefix=rf"{matched_end_pattern}",
            suffix=rf"{matched_eot_token}",
            dotall=True,
            return_all=False,
            return_last=False,
            verbose=False,
        )
        if isinstance(_prediction, str):
            prediction = _prediction.strip()

    return {
        "prediction": prediction,
        "reasoning_content": reasoning_content,
    }

def parse_boxed_format(
    prediction: str,
    verbose: Optional[bool] = False,
    **kwargs,
):
    # Extract the content of the LAST top-level ``\boxed{...}``, brace-balanced.
    # Reasoning models often emit intermediate ``\boxed{}`` during the trace and
    # the final answer at the end; paper protocols (MathVista, WeMath, KMMMU,
    # Korean SAT, etc.) treat the trailing one as canonical. The depth-aware
    # scan also handles nested braces (e.g. ``\boxed{\frac{1}{2}}``).
    output = None
    if isinstance(prediction, str):
        _key = "\\boxed{"
        _start = prediction.rfind(_key)
        if _start != -1:
            _i = _start + len(_key)
            _depth = 1
            _j = _i
            while _j < len(prediction):
                _ch = prediction[_j]
                if _ch == "{":
                    _depth += 1
                elif _ch == "}":
                    _depth -= 1
                    if _depth == 0:
                        output = prediction[_i:_j]
                        break
                _j += 1
    logger.debug(f'parse_boxed_format: {prediction} -> {output}')
    return output

def parse_last_pattern(
    prediction: str,
    prefix: str,
    suffix: str,
    dotall: bool = True,
    return_all: Optional[bool] = False,
    return_last: Optional[bool] = True,
    verbose: Optional[bool] = False,
    **kwargs,
) -> Optional[str]:
    """
    Return the inner text of the last occurrence enclosed by `prefix` ... `suffix`.
    Example: text='x <|im_start|>A<|im_end|> y <|im_start|>B<|im_end|>' -> 'B'

    Args:
        text: Source string.
        prefix: Opening delimiter (literal).
        suffix: Closing delimiter (literal).
        dotall: If True, '.' matches newlines (re.DOTALL).

    Returns:
        The inner string of the last match, or None if no pair is found.

    Notes:
        - This is non-greedy and picks the *last* complete match.
        - It does not handle *nested* delimiter structures. For nested braces,
          use a stack-based parser.
    """
    if not prefix or not suffix:
        raise ValueError("prefix and suffix must be non-empty")
    
    if not prediction.endswith(suffix):
        prediction = f'{prediction}{suffix}'
    
    pattern = re.escape(prefix) + r"(.*?)" + re.escape(suffix)
    flags = re.DOTALL if dotall else 0

    output = list()
    for _match in re.finditer(pattern, prediction, flags):
        if not _match:
            continue
        _output = _match.group(1)
        if prefix in _output:
            output.extend(_output.split(prefix))
        else:
            output.append(_output)
    
    output = [e.strip() for e in output if e.strip()]
    if len(output) < 1:
        output = None
    elif return_all:
        pass
    elif return_last:
        output = output[-1]
    else:
        output = output[0]
    
    logger.debug(f'parse_last_pattern: {prediction} -> {output}')
    return output

def parse_circled_answer(
    prediction: str,
    verbose: Optional[bool] = False,
    **kwargs,
):
    output = vqa__circledNumbersMap[prediction]

    logger.debug(f'parse_circled_answer: {prediction} -> {output}')
    return output