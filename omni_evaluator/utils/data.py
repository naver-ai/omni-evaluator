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

import copy
import fractions
import inspect
import itertools
import json
import logging
import numpy as np
import queue
import re
import threading
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Sized, Iterable, Iterator
import urllib

logger = logging.getLogger(__name__)


def filter_arguments(
    func: Callable,
    *args, **kwargs,
) -> Dict[str, Any]:
    func_kwargs = dict()
    func_parameters = inspect.signature(func).parameters.keys()

    # update args to func_kwargs
    _arg_names = list(func_parameters)[:len(args)]
    for _arg_name, _arg_value in zip(_arg_names, args):
        func_kwargs[_arg_name] = _arg_value
    # update kwargs to func_kwargs
    func_kwargs.update({
        k: v for k, v in kwargs.items() if k in func_parameters
    })
    return func_kwargs


def find_field(
    obj: Dict[str, Any],
    candidate_keys: List[str],
    default: Any = None,
    ignore_values: List[str] = None,
) -> Any:
    field = default
    doc_keys_lower = {k.lower(): k for k in obj.keys()}
    for k in candidate_keys:
        _value = None
        if k in obj: 
            _value = obj[k]
        elif k.lower() in doc_keys_lower:
            _value = obj[doc_keys_lower[k]]
        else: 
            continue
        
        if (
            _value is None 
            or (isinstance(_value, Sized) and len(_value) < 1)
        ):
            continue
        elif (
            ignore_values is not None
            and _value in ignore_values
        ):
            continue 
        else:
            field = _value
            break
    return field


def generator_factory(
    producer: Callable[..., Any],
    *args,
    **kwargs,
) -> Callable[[], Iterator]:
    """Bind ``producer(*args, **kwargs)`` into a zero-arg factory.

    Each call to the returned factory invokes ``producer(*args, **kwargs)``
    afresh, yielding an independent iterator instance. Used to safely share
    one logical sequence across multiple consumers (e.g. per-rank threads
    doing contiguous-block split) where each consumer must own a separate
    iterator state.

    If ``producer`` returns a tuple, the first element is extracted as the
    iterator. Common case: ``load_dataset(...)`` returns ``(iterator, size)``.
    """
    def factory() -> Iterator:
        result = producer(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result
    return factory


def split_iterator(
    obj: Union[Iterable, Callable[[], Iterator]],
    total_size: int,
    world_size: int,
) -> Tuple[List[Iterator], List[int]]:
    """Split ``obj`` into ``world_size`` contiguous-block slices.

    ``obj`` may be either:
      - A re-iterable (list/tuple/HF Dataset/etc.): each rank receives an
        ``itertools.islice`` view; safe because ``iter(seq)`` returns a fresh
        iterator on every call.
      - A factory callable (zero-arg, returns a fresh iterator per call): each
        rank calls the factory and gets an independent generator. Avoids
        materialization while still being safe across threads.

    A bare single-use generator is unsafe to pass here when ``world_size > 1``;
    wrap it with :func:`generator_factory` (or any equivalent) first.

    Returns ``(slices, sizes)`` — two parallel lists of length ``world_size``.
    """
    if not (isinstance(world_size, int) and world_size > 0):
        raise ValueError(f'Invalid world_size: {world_size}')
    if not (isinstance(total_size, int) and total_size >= 0):
        raise ValueError(f'Invalid total_size: {total_size}')

    base, leftover = divmod(total_size, world_size)
    sizes = [base + (1 if r < leftover else 0) for r in range(world_size)]
    starts = list(itertools.accumulate([0] + sizes[:-1]))

    if callable(obj):
        slices = [
            itertools.islice(obj(), starts[r], starts[r] + sizes[r])
            for r in range(world_size)
        ]
    else:
        slices = [
            itertools.islice(obj, starts[r], starts[r] + sizes[r])
            for r in range(world_size)
        ]
    return slices, sizes


def rename_dict(
    obj: Dict[str, Any],
    rename_map: Dict[str, str],
) -> Dict[str, Any]:
    for _from, _to in rename_map.items():
        if _from not in obj:
            continue
        obj[_to] = obj.pop(_from)
    return obj


def format_task_prompt(
    task_prompt: str,
    query: str,
    **kwargs,
) -> str:
    if (
        not isinstance(task_prompt, str)
        or len(task_prompt) < 1
    ):
        return query
    
    if query is None:
        _placeholder_pattern = re.compile(r"^\s*\{[^{}]*\}")
        query = _placeholder_pattern.sub("", task_prompt, count=1)
        query = query.strip()
        return query
    
    if "{}" in task_prompt:
        query = task_prompt.format(query)
    else:
        _matches = re.findall(r"{\s*([A-Za-z_]\w*)\s*}", task_prompt)
        if len(_matches) > 0:
            _format_kwargs = dict()
            for _match in _matches:
                if _match == "query":
                    _format_kwargs["query"] = query
                else:
                    _format_kwargs[_match] = kwargs[_match]
            query = task_prompt.format(**_format_kwargs)
        else:
            query = f'{query}\n{task_prompt}'
    return query


def align_tag(
    text: str,
    tag: str,
    num_attach: int,
    attach_head: Optional[bool] = True,
    re_escape: Optional[bool] = False,
) -> str:
    tag_pattern = tag
    if re_escape:
        tag_pattern = re.escape(tag)
    
    tag_matches = list(re.finditer(tag_pattern, text))
    if num_attach > len(tag_matches): # make up for the shortage
        _tags = tag * (num_attach - len(tag_matches))
        if attach_head:
            text = f'{_tags}{text}' 
        else:
            text = f'{text}{_tags}' 
    elif num_attach < len(tag_matches): # remove tags exceeding
        for _match in reversed(tag_matches):
            if num_attach == len(tag_matches):
                break
            text = text[:_match.start()] + text[_match.end():]
            num_attach += 1
    else: # equal number 
        pass
    return text

def normalize_unit(
    x: Union[float, int],
    unit: Union[float, int],
    max_denominator: int = 100000,
) -> float:
    nom = fractions.Fraction(x).limit_denominator(max_denominator)
    denom = fractions.Fraction(unit).limit_denominator(max_denominator)
    output = float(nom / denom)
    return output


def safe_percentage(
    numerator: Union[int, float],
    denominator: Union[int, float],
    default: float = 0.0,
) -> float:
    """Return ``(numerator / denominator) * 100.0``, returning ``default`` when
    the denominator is 0/None.

    Used by per-class aggregations where an empty cohort is paper-defined as
    undefined (e.g. wemath's RM ratio when the multi_score=1 cohort is empty,
    or mathverse's per-category mean when a category has zero samples).
    """
    if not denominator:
        return default
    return numerator / denominator * 100.0


def extract_options(
    text: str,
) -> List[Tuple[str, str]]:
    """
    Extract (letter, option_text) pairs from a line like:
      "Options: A: ..., B: ..., C: ..., D: ..."
    Returns: [("A", "..."), ...]
    """
    option_pattern = re.compile(
        r"""
        (?is)
        (?:^|[\n,]\s*)              # start, newline, or comma boundary
        \(?                         # optional opening parenthesis
        ([A-Z])\s*                  # option letter
        (?:[:\.\)])\s*              # separator (: . ))
        (.*?)                       # option content (non-greedy)
        (?=                         # stop when next option begins or end
            (?:[\n,]\s*\(?[A-Z]\s*[:\.\)]) |
            \Z
        )
        """,
        re.VERBOSE,
    )
    
    # 1) Find the "Options:" segment (can span lines)
    _match = re.search(r"(?is)\bOptions\s*:\s*(.+?)(?:\n\s*\n|$)", text)
    if _match:
        text = _match.group(1).strip()    

    # 2) Extract each option using a non-greedy match up to the next ", X:" or end
    options = list()
    option_contents = list()
    for _option, _option_content in option_pattern.findall(text):
        _option_content = " ".join(_option_content.split())  # normalize whitespace/newlines
        _option_content = _option_content.strip()
        options.append(_option)
        option_contents.append(_option_content)
    return options, option_contents

def shift_options(
    options: List[Any],
    option_contents: List[Any],
    label: List[str],
    run_index: int,
) -> Tuple[List[Any], List[Any], List[str]]:
    if not (isinstance(label, (list, tuple)) and len(label) > 0):
        raise ValueError(f'Label should be list or tuple: {label}')
    label = copy.deepcopy(label)
    
    # circular evaluation
    if run_index > 0:
        _option_index = run_index % len(option_contents)
        option_contents = option_contents[-_option_index:] + option_contents[:-_option_index]
        
        for _label_idx, _label in enumerate(label):
            if isinstance(_label, int):
                _label = _label + _option_index
            if _label in options:
                _label = options[options.index(_label) + _option_index]
            elif _label in option_contents:
                continue
            else: # (A) Man
                _label_option, _label_content = extract_options(text=_label)
                if (
                    len(_label_option) == 1 
                    and len(_label_content) == 1
                    and _label_option[0] in options
                    and _label_content[0] in option_contents
                ):
                    _label_option = options[(options.index(_label_option[0]) + _option_index) % len(options)]
                    _label_content = _label_content[0]
                    _label = f'{_label_option}. {_label_content}'
                else:
                    raise ValueError(f'invalid type of label to apply circular_evaluation: {_label}')
            label[_label_idx] = _label
            
    return options, option_contents, label