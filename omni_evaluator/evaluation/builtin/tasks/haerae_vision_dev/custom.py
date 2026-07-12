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
import fractions
import json
import numpy as np
import re
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

from omni_evaluator.evaluation.builtin.tasks.haerae_vision_dev.prompts import JUDGE_PROMPT

SCORE_FORMAT = r"""<score>\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*(?:/\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*)?</score>"""

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
    query = sample["question_original"]
    _question_type = task_config.inference.config.get("question_type", None)
    if _question_type == "explicit":
        query = sample["question_explicit"]

    sample_meta = {
        "question_idx": sample["question_idx"],
        "category": sample["category"],
        "source": sample["source"],
        "checklist": sample["checklist"],
        "question_original": sample["question_original"],
        "question_explicit": sample["question_explicit"],
    }
    
    messages = [
        {
            "role": "user",
            "content": list()
        }
    ]
    for _image in sample["images"]:
        messages[-1]["content"].append({
            "type": "image",
            "image": _image,
        })
    messages[-1]["content"].append({
        "type": "text",
        "text": query,
    })

    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample.get("question_idx", sample_idx),
            "messages": messages,
            "options": None,
            "option_contents": None,
            "label": None,
            "meta": sample_meta,
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
    return record

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
        _sample_metrics = list()
        _group_metrics = dict()
        if "judge" in _target_metric:
            judge_messages_list, generation_options_list = list(), list()
            for _record_idx, _record in enumerate(records):
                # skip if prediction is empty
                if not _record["prediction"]:
                    continue 

                _last_user_message = ChatMessage.get_user_messages(
                    messages=_record["messages"],
                )[-1]
                _query = ChatMessage.get_query(message=_last_user_message)
                _judge_prompt = JUDGE_PROMPT.format(
                    QUESTION=_query,
                    RESPONSE=_record["prediction"],
                    CHECKLIST=_record["meta"]["checklist"],
                )
                _judge_messages = list()
                _judge_messages.append(ChatMessage(
                    role="user",
                    content=[
                        ChatTextContent(
                            type="text", 
                            value=_judge_prompt,
                        ),
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
                    semaphore_size=args.inference_concurrency,
                ))
            else:
                responses = batch_chat_completion_sync(
                    api_name=task_config.evaluation.judge_evaluator.metrics[_target_metric].judge_model,
                    messages_list=judge_messages_list,
                    generation_options_list=generation_options_list,
                )

            for _sample_idx, _response in enumerate(responses):
                _match = re.search(SCORE_FORMAT, _response["prediction"])
                _rating, _max_rating = None, None
                if _match:
                    try:
                        _rating = _match.group(1)
                        _rating = is_numeric(x=_rating)
                    except Exception as ex:
                        pass
                    try:
                        _max_rating = _match.group(2)
                        _max_rating = is_numeric(x=_max_rating)
                    except Exception as ex:
                        pass

                _score = _rating
                if _rating and _max_rating:
                    _score = normalize_unit(
                        x=_score,
                        unit=_max_rating,
                    )
                
                _sample_metrics.append(_score)
                _group_name = records[_sample_idx]["meta"]["category"]
                if _group_name not in _group_metrics:
                    _group_metrics[_group_name] = list()
                _group_metrics[_group_name].append(_score)
        
            metrics[_target_metric] = np.nanmean(_sample_metrics)
            for _group_name, _metric_values in _group_metrics.items():
                if _group_name not in group_metrics:
                    group_metrics[_group_name] = dict()
                group_metrics[_group_name][_target_metric] = np.nanmean(_metric_values)
            for _sample_idx, _sample_metric in enumerate(_sample_metrics):
                sample_metrics[_sample_idx][_target_metric] = _sample_metric
            
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