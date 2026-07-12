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

import argparse
import asyncio
import contextlib
import copy
import json
import numpy as np
import PIL
from PIL import Image
import sys
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence, Callable, Iterable

from omni_evaluator.api import get_api_group
from omni_evaluator.api.chat_completions import (
    batch_chat_completion_async,
    batch_chat_completion_sync,
)
from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.evaluation.metrics.judge_evaluator import JudgeEvaluator
from omni_evaluator.schemas.chat import (
    OcrToken, EntityToken,
    Message as ChatMessage, 
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
)
from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from omni_evaluator.schemas.generation_options import ApiGenerationOptions
from omni_evaluator.schemas.inference import Record
from omni_evaluator.schemas.task import TaskConfig, TaskInference, TaskInferenceGenerationOptions
from omni_evaluator.utils.data import format_task_prompt, normalize_unit
from omni_evaluator.utils.string import is_numeric

@contextlib.contextmanager
def charxiv_patch():
    import charxiv
    from charxiv import constants
    sys.modules["constants"] = constants

    try:
        yield

    finally:
        sys.modules.pop("constants")
        del constants
        # del model_src

def sample_to_record(
    task_name: str,
    task_config: Dict[str, Any],
    sample_idx: int,
    sample: Dict[str, Any],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    run_index: Optional[int] = 0,
    **kwargs,
):
    if not sample["subplot_loc"]:
        sample["subplot_loc"] = [sample["subplot_row"], sample["subplot_col"]]
            
    queries, labels = list(), list()
    question_ids, categories = list(), list()
    with charxiv_patch():
        from charxiv.score_utils import (
            DOMAIN2ABBR, NUM2YEAR, QNUM2QTYPE,
            NUMSUBPLOTS2SUBPLOTTYPE, D_TEMPLATE, R_TEMPLATE,
            IDX2ANSTYPE, IDX2SRC,
        )
        if task_config.inference.config["mode"] == "descriptive":
            from charxiv import descriptive_utils
            
            keys = sample.keys()
            for _key in sorted(keys):
                if not _key.startswith("descriptive_q"):
                    continue
                _query_key = _key
                _label_key = _key.replace("_q", "_a")
                _question_id = sample[_query_key]
                _query = descriptive_utils.descriptive_query_helper(
                    qid=_question_id,
                    subplot_loc=sample["subplot_loc"],
                )
                _label = sample[_label_key]
                _category = QNUM2QTYPE(_question_id)
                queries.append(_query)
                labels.append(_label)
                question_ids.append(_question_id)
                categories.append(_category)
        
        else:
            from charxiv import reasoning_utils
            from constants import REASONING_RESP_INST, REASONING_GRADING_PREFIX, REASONING_GRADING_INST

            # 1: text-in-chart, 2: text-in-general, 3: number-in-chart
            _question_id = sample["reasoning_q_source"]
            _query = sample["reasoning_q"]
            _answer = sample["reasoning_a"]
            _answer_type = sample["reasoning_a_type"]
            _category = IDX2ANSTYPE[_answer_type]
            if _answer_type in [1, 2, 3]:
                _query = REASONING_RESP_INST[_answer_type].format(_query)
            # 4: number-in-general -> need to specify the number of decimal places
            elif _answer_type == 4:
                _query = REASONING_RESP_INST[_answer_type].format(
                    _query, 
                    reasoning_utils.get_number_instruction(_answer),
                )
            else:
                raise ValueError(f"Invalid instruction category: {_answer_type}")
            
            queries.append(_query)
            labels.append(_answer)
            question_ids.append(_question_id)
            categories.append(_category)

    sample_meta = {
        "original_id": sample["original_id"],
        "question_ids": question_ids,
        "year": sample["year"],
        "category": sample["category"],
        "category_original": sample["category"],
        "figure_path": sample["figure_path"],
        "original_figure_path": sample["original_figure_path"],
        "num_subplots": sample["num_subplots"],
        "subplot_row": sample["subplot_row"],
        "subplot_col": sample["subplot_col"],
        "subplot_loc": sample["subplot_loc"],
        "descriptive_q1": sample["descriptive_q1"],
        "descriptive_q2": sample["descriptive_q2"],
        "descriptive_q3": sample["descriptive_q3"],
        "descriptive_q4": sample["descriptive_q4"],
        "descriptive_a1": sample["descriptive_a1"],
        "descriptive_a2": sample["descriptive_a2"],
        "descriptive_a3": sample["descriptive_a3"],
        "descriptive_a4": sample["descriptive_a4"],
        "reasoning_q": sample["reasoning_q"],
        "reasoning_q_source": sample["reasoning_q_source"],
        "reasoning_a": sample["reasoning_a"],
        "reasoning_a_type": sample["reasoning_a_type"],
    }
    
    messages = [
        {
            "role": "user",
            "content": list(),
        },
    ]
    if isinstance(sample["image"], PIL.Image.Image):
        messages[-1]["content"].append({
            "type": "image",
            "value": sample["image"],
        })
    elif isinstance(sample["image"], (list, tuple)):
        for _image in sample["image"]:
            messages[-1]["content"].append({
                "type": "image",
                "value": _image,
            })
    else:
        raise ValueError(f'invalid image type: {sample["image"]}')
    
    records = list()
    for _query, _label, _question_id, _category in zip(
        queries, labels, question_ids, categories,
    ):
        _messages = copy.deepcopy(messages)
        _labels = copy.deepcopy(_label)
        _sample_meta = copy.deepcopy(sample_meta)
        
        _messages[-1]["content"].append({
            "type": "text",
            "value": _query,
        })
        if isinstance(_labels, str):
            _labels = [_labels, ]
        _sample_meta["question_id"] = _question_id
        _sample_meta["category"] = _category
        
        _record = default_sample_to_record(
            task_name=task_config.task_name,
            task_config=task_config,
            sample_idx=sample_idx,
            sample={
                "index": f'{sample_idx}_{_question_id}',
                "messages": _messages,
                "options": None,
                "option_contents": None,
                "label": _labels,
                "meta": _sample_meta,
            },
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            num_ocr_tokens=num_ocr_tokens,
            num_entity_tokens=num_entity_tokens,
            run_index=run_index,
        )
        records.append(_record)
        
    return records

def evaluate(
    args: argparse.Namespace,
    evaluation_method: str,
    task_name: str,
    task_config: TaskConfig,
    records: List[Dict[str, Any]],
    **kwargs,
):
    metrics = dict()
    group_metrics = dict()
    sample_metrics = [dict() for _ in range(0, len(records))]
    
    for _target_metric in task_config.evaluation.target_metrics:
        judge_messages_list, generation_options_list = list(), list()
        with charxiv_patch():
            for _record_idx, _record in enumerate(records):
                _judge_prompt = None
                if task_config.inference.config["mode"] == "descriptive":
                    from charxiv import descriptive_utils
                    from constants import (
                        DESCRIPTIVE_RESP_INST, DESCRIPTIVE_GRADING_PREFIX, 
                        DESCRIPTIVE_GRADING_QMAP, DESCRIPTIVE_GRADING_ICL,
                    )
                    # construct judge_message
                    _judge_query = DESCRIPTIVE_GRADING_QMAP[_record["meta"]["question_id"]]
                    _rubric_icl = descriptive_utils.get_rubric(_record["meta"]["question_id"])
                    _json_keys = descriptive_utils.build_json_keys(1) # len(batch)
                    # populate batch size, question, and json keys spec
                    _judge_prefix = DESCRIPTIVE_GRADING_PREFIX\
                        .replace("<|NUM_TRIPLETS|>", str(1))\
                        .replace("<|OVERARCHING_QUESTION|>", _judge_query)\
                        .replace("<|JSON_KEYS|>", _json_keys)
                    # add in-context grading example based on the template id
                    # prompt + example + model responses
                    # Coerce a missing prediction (e.g. failed inference, judge
                    # cannot read None) to an empty string so the judge receives
                    # a well-formed prompt and naturally rates it as incorrect.
                    _prediction_text = _record["prediction"] if isinstance(_record["prediction"], str) else ""
                    _judge_input = descriptive_utils.populate_grading_inputs(
                        batch=[ # resp_key, response, answer
                            (_record["index"], _prediction_text, _record["label"][0]),
                        ],
                    )
                    _judge_prompt = _judge_prefix + _rubric_icl + _judge_input

                else:
                    from constants import (
                        REASONING_RESP_INST, REASONING_GRADING_PREFIX, REASONING_GRADING_INST,
                    )
                    _query = _record["meta"]["reasoning_q"]
                    _answer = _record["label"][0]
                    _answer_type = _record["meta"]["reasoning_a_type"]
                    # Same None-coerce as the descriptive branch: ``str.replace``
                    # requires a str second arg, so a None prediction otherwise
                    # raises ``TypeError: replace() argument 2 must be str, not None``.
                    _prediction = _record["prediction"] if isinstance(_record["prediction"], str) else ""
                    # get query for answer type (inst_category), then
                    # populate the query with the question, ground truth, and response
                    _judge_prompt = REASONING_GRADING_PREFIX + copy.deepcopy(\
                        REASONING_GRADING_INST[_answer_type])\
                        .replace("<|question|>", _query)\
                        .replace("<|ground_truth|>", _answer)\
                        .replace("<|response|>", _prediction)
                    
                _judge_messages = list()
                _judge_messages.append(ChatMessage(
                    role="user",
                    content=[
                        ChatTextContent(
                            type="text",
                            value=_judge_prompt,
                        )
                    ]
                ))
                judge_messages_list.append(_judge_messages)
                
                _api_group = get_api_group(api_name=task_config.evaluation.judge_evaluator.metrics[_target_metric].judge_model)
                _generation_options = ApiGenerationOptions.from_dict(
                    api_name=task_config.evaluation.judge_evaluator.metrics[_target_metric].judge_model,
                    obj=task_config.evaluation.judge_evaluator.metrics[_target_metric].to_dict(),
                    api_group=_api_group,
                ).to_dict()
                generation_options_list.append(_generation_options)
            
        responses = None
        if args.do_async:
            responses = asyncio.run(batch_chat_completion_async(
                api_name=task_config.evaluation.judge_evaluator.metrics[_target_metric].judge_model,
                messages_list=judge_messages_list,
                generation_options_list=generation_options_list,
                response_format={"type": "json_object"},
                semaphore_size=args.inference_concurrency,
            ))
        else:
            responses = batch_chat_completion_sync(
                api_name=task_config.evaluation.judge_evaluator.metrics[_target_metric].judge_model,
                messages_list=judge_messages_list,
                generation_options_list=generation_options_list,
                response_format={"type": "json_object"},
            )
        
        _sample_metrics = list()
        _group_metrics = dict()
        for _record_idx, (_record, _response) in enumerate(zip(records, responses)):
            _group_name = _record["meta"]["category"]
            _score = None
            # When a response_format="json_object" response fails json.loads,
            # the prediction can remain a raw string — defensive parse.
            _pred = _response.get("prediction") if isinstance(_response, dict) else None
            if isinstance(_pred, str):
                try:
                    _pred = json.loads(_pred)
                except Exception:
                    _pred = None
            if isinstance(_pred, dict):
                if task_config.inference.config["mode"] == "descriptive":
                    _score = _pred.get("score_T1")
                else:
                    _score = _pred.get("score")
            if not isinstance(_score, (int, float)):
                _score = None
            _sample_metrics.append(_score)
            sample_metrics[_record_idx][_target_metric] = _score
            if _group_name not in _group_metrics:
                _group_metrics[_group_name] = list()
            _group_metrics[_group_name].append(_score)
        
        # filter None (parse-failed samples) before averaging — np.mean cannot
        # handle Python None and raises TypeError.
        _valid = [s for s in _sample_metrics if isinstance(s, (int, float))]
        metrics[_target_metric] = float(np.mean(_valid)) if _valid else 0.0
        for _group_name, _metric_values in _group_metrics.items():
            if _group_name not in group_metrics:
                group_metrics[_group_name] = dict()
            _valid_g = [s for s in _metric_values if isinstance(s, (int, float))]
            group_metrics[_group_name][_target_metric] = float(np.mean(_valid_g)) if _valid_g else 0.0
    
    # omni_evaluator
    evaluation_output = EvaluationRunOutput.from_task(
        args,
        task_name,
        task_config,
        records,
        metrics,
        group_metrics=group_metrics,
        sample_metrics=sample_metrics,
        default_metric_keys=["reward", "pass^1"],
    )
    return evaluation_output, sample_metrics