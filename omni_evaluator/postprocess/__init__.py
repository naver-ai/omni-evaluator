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

from collections import OrderedDict
from enum import Enum
from functools import partial
import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Iterable

from omni_evaluator import EvaluationEngine

logger = logging.getLogger(__name__)
from omni_evaluator.postprocess.asr import AsrProcessor
from omni_evaluator.postprocess.binary import BinaryProcessor
from omni_evaluator.postprocess.code import CodeProcessor
from omni_evaluator.postprocess.freeform import FreeformProcessor
from omni_evaluator.postprocess.spatial_grounding import SpatialGroundingProcessor
from omni_evaluator.postprocess.multichoice import MultichoiceProcessor
from omni_evaluator.postprocess.temporal_grounding import TemporalGroundingProcessor
from omni_evaluator.postprocess.custom import (
    parse_boxed_format, parse_last_pattern, parse_think, parse_circled_answer,
)
from omni_evaluator.schemas.task import (
    TaskConfig, TaskMeta,
    TaskPrompts, TaskDataset,
    TaskInference, TaskInferenceGenerationOptions,
    TaskEvaluation, TaskEvaluationPostprocess, TaskEvaluationJudge,
    TaskPostprocess, TaskPostprocessLogic,
)
from omni_evaluator.utils.common import get_custom_module


PostprocessLogic = {
    "binary": BinaryProcessor.extract,
    "code": CodeProcessor.extract,
    "freeform": FreeformProcessor.extract,
    "multichoice": MultichoiceProcessor.extract,
    "spatial_grounding": SpatialGroundingProcessor.extract,
    "temporal_grounding": TemporalGroundingProcessor.extract,
    "asr": AsrProcessor.extract,
    "boxed": parse_boxed_format,
    "think": partial(
        parse_think,
        think_start_pattern="<think>",
        think_end_pattern="</think>",
        eot_token="<|im_end|>",
    ),

}

def _resolve_conditional_postprocess_pipeline(
    pipeline: Union[List[str], Dict[str, List[str]], None],
    sample: Dict[str, Any],
    conditional_on: Optional[str],
) -> Optional[List[str]]:
    """Pick the pipeline variant for one sample.

    Args: pipeline - list (passthrough) or dict keyed by meta value,
          conditional_on - meta field name (None => passthrough).
    """
    if not isinstance(pipeline, dict) or not conditional_on:
        return pipeline
    _meta = sample.get("meta", {}) or {}
    _variant_key = _meta.get(conditional_on, None)
    return pipeline.get(_variant_key, None)


def get_postprocess_functions(
    evaluation_engine: str,
    task_name: str,
    task_config: Union[Dict[str, Any], TaskConfig],
    postprocess_pipeline: Optional[Union[List[str], Dict[str, List[str]]]] = None,
    postprocess_version: Optional[List[str]] = None,
    postprocess_api_name: Optional[str] = None,
    postprocess_allow_api: Optional[bool] = None,
    parse_boxed: Optional[bool] = False,
    verbose: Optional[bool] = False,
    **kwargs,
) -> Tuple[Dict[Optional[str], "OrderedDict[str, Callable]"], Optional[str]]:
    """Build postprocess chains, optionally per-sample conditional.

    Returns ``(variants, conditional_on)``:
      - variants: ``Dict[variant_key, OrderedDict]`` of step name -> partial fn.
        When conditional_on is None, variants is a single entry keyed by ``None``.
        When conditional_on is set, variants has one entry per dict key in pipeline.
      - conditional_on: meta field name for per-sample variant selection, or None.
    """
    task_config = TaskConfig.ensure(task_config)

    # TaskPostprocess (top-level, new structure) is the canonical source. Its
    # `chain` holds per-processor TaskPostprocessLogic kwargs; conditional_on
    # selects per-sample variants. Legacy evaluation.postprocess is mirrored
    # by the schema, so this is populated for legacy configs too.
    _pp_new = task_config.postprocess

    conditional_on = _pp_new.conditional_on if _pp_new is not None else None
    if not postprocess_allow_api and not (_pp_new is not None and _pp_new.allow_api):
        # Master switch is off — kill api_name everywhere so processors take their
        # regex / non-api fallback. This covers (a) the CLI override and (b) every
        # TaskPostprocessLogic in the chain, including api_name values that came
        # from _DEFAULT_POSTPROCESS_LOGICS or from a YAML override.
        postprocess_api_name = None
        if _pp_new is not None and _pp_new.chain:
            for _entry in _pp_new.chain.values():
                if isinstance(_entry, TaskPostprocessLogic):
                    _entry.api_name = None
                elif isinstance(_entry, dict):
                    for _logic in _entry.values():
                        if isinstance(_logic, TaskPostprocessLogic):
                            _logic.api_name = None

    # Resolve pipeline order. CLI override wins; otherwise derive from chain keys.
    if not postprocess_pipeline and _pp_new is not None and _pp_new.chain:
        if conditional_on:
            postprocess_pipeline = {
                _variant_key: list(_variant_chain.keys())
                for _variant_key, _variant_chain in _pp_new.chain.items()
                if isinstance(_variant_chain, dict)
            }
        else:
            postprocess_pipeline = list(_pp_new.chain.keys())

    def _lookup_logic(proc_name: str, variant_key: Optional[str] = None) -> Optional[TaskPostprocessLogic]:
        if _pp_new is None or not _pp_new.chain:
            return None
        if variant_key is not None:
            _variant = _pp_new.chain.get(variant_key)
            if isinstance(_variant, dict):
                _logic = _variant.get(proc_name)
                if isinstance(_logic, TaskPostprocessLogic):
                    return _logic
            return None
        _logic = _pp_new.chain.get(proc_name)
        if isinstance(_logic, TaskPostprocessLogic):
            return _logic
        return None

    def _build_chain(pipeline_list: Optional[List[str]], variant_key: Optional[str] = None) -> "OrderedDict[str, Callable]":
        chain: "OrderedDict[str, Callable]" = OrderedDict()
        if parse_boxed:
            chain["parse_boxed"] = partial(parse_boxed_format, verbose=verbose)
        if not isinstance(pipeline_list, (list, tuple)):
            return chain
        for _idx, _postprocess_logic in enumerate(pipeline_list):
            if _postprocess_logic in chain:
                continue

            # Per-processor kwargs come from TaskPostprocess.chain; the CLI
            # overrides (postprocess_api_name, postprocess_version) act as
            # task-wide fallbacks when no per-processor logic is registered.
            _logic_entry = _lookup_logic(_postprocess_logic, variant_key=variant_key)
            _bind_kwargs: Dict[str, Any] = {}
            if _logic_entry is not None:
                # Bind only static (non-None) entries; None-valued extras stay in
                # postprocess.chain[proc].extra for per-record meta lookup in
                # evaluate.py.
                for _k, _v in _logic_entry.to_kwargs().items():
                    if _v is not None or _k == "verbose":
                        _bind_kwargs[_k] = _v
                _bind_kwargs.setdefault("verbose", verbose)
            else:
                # Legacy path — single global api_name / version
                _postprocess_version = None
                if isinstance(postprocess_version, str):
                    _postprocess_version = postprocess_version
                elif isinstance(postprocess_version, (list, tuple)) and _idx < len(postprocess_version):
                    _postprocess_version = postprocess_version[_idx]
                _bind_kwargs = {
                    "version_name": _postprocess_version,
                    "api_name": postprocess_api_name,
                    "verbose": verbose,
                }

            if _postprocess_logic in PostprocessLogic:
                _fn = PostprocessLogic[_postprocess_logic]
                chain[_postprocess_logic] = partial(_fn, **_bind_kwargs)
            else:  # custom postprocess
                _custom_module = get_custom_module(
                    evaluation_engine=evaluation_engine,
                    task_name=task_name,
                )
                if _custom_module and hasattr(_custom_module, _postprocess_logic):
                    _fn = getattr(_custom_module, _postprocess_logic, None)
                    chain[_postprocess_logic] = partial(_fn, **_bind_kwargs)
                else:
                    raise ValueError(
                        f'Given custom postprocess_logic not implemented: {_postprocess_logic}'
                    )
        return chain

    variants: Dict[Optional[str], "OrderedDict[str, Callable]"] = {}

    if conditional_on and isinstance(postprocess_pipeline, dict):
        # Conditional: one chain per variant key — propagate variant_key so per-processor
        # kwargs lookup hits TaskPostprocess.chain[variant_key][proc_name].
        for _variant_key, _pipeline_list in postprocess_pipeline.items():
            variants[_variant_key] = _build_chain(_pipeline_list, variant_key=_variant_key)
    elif isinstance(postprocess_pipeline, (list, tuple)):
        # Single chain
        variants[None] = _build_chain(postprocess_pipeline)
    else:
        # Default engine-specific paths (only when no explicit pipeline given)
        default_chain: "OrderedDict[str, Callable]" = OrderedDict()
        if parse_boxed:
            default_chain["parse_boxed"] = partial(parse_boxed_format, verbose=verbose)
        if evaluation_engine == EvaluationEngine.lm_eval_harness:
            if task_name in [
                "humaneval", "humaneval_instruct",
                "humaneval_64", "humaneval_64_instruct",
                "humaneval_plus", "humaneval_x",
                "mbpp", "mbpp_plus",
            ]:
                default_chain["code/python"] = partial(
                    CodeProcessor.extract,
                    language="python",
                    extract_continuation=True,
                    version_name=postprocess_version,
                    api_name=postprocess_api_name,
                    verbose=verbose,
                )
        variants[None] = default_chain

    for _variant_key, _chain in variants.items():
        if len(_chain) > 0:
            pipeline_str = ', '.join(
                f'{_postprocess_logic}: {_postprocess_func}'
                for _postprocess_logic, _postprocess_func in _chain.items()
            )
            label = f'[{_variant_key}] ' if _variant_key is not None else ''
            logger.info(f'Postprocess pipeline {label}{pipeline_str}')

    return variants, conditional_on