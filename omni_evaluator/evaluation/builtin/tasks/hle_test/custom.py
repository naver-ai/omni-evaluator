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
from collections import defaultdict
import logging
import pydantic
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence, Callable, Iterable, Literal

logger = logging.getLogger(__name__)

from omni_evaluator.api import get_api_group
from omni_evaluator.api.chat_completions import (
    batch_chat_completion_sync as batch_chat_completion_sync_api,
    batch_chat_completion_async as batch_chat_completion_async_api,
)
from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record
)
from omni_evaluator.evaluation.metrics.text_evaluator import TextEvaluator
from omni_evaluator.schemas.chat import (
    Message as ChatMessage, 
    TextContent as ChatTextContent,
)
from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from omni_evaluator.schemas.generation_options import ApiGenerationOptions
from omni_evaluator.schemas.task import TaskConfig
from omni_evaluator.utils.data import format_task_prompt
from omni_evaluator.utils.multimodal import to_pil_image


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
    _user_content = list()
    _user_content.append({"type": "text", "value": sample["question"]})
    if sample["image"]:
        _image = to_pil_image(image=sample["image"])
        _user_content.append({"type": "image", "value": _image})

    _sample = {
        "index": sample["id"],
        "messages": [
            {
                "role": "user",
                "content": _user_content   
            }
        ],
        "label": sample["answer"],
        "meta": {
            "question": sample["question"],
            "answer_type": sample["answer_type"],
            "author_name": sample["author_name"],
            "rationale": sample["rationale"],
            "rationale_image": None, # sample["rationale_image"]
            "raw_subject": sample["raw_subject"],
            "category": sample["category"],
            "canary": sample["canary"],
        }
    }
    
    record = default_sample_to_record(
        task_name=task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample=_sample,
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
    class JudgeResponseFormat(pydantic.BaseModel):
        extracted_final_answer: str
        reasoning: str
        correct: Literal["yes", "no"]
        confidence: int
        strict: Literal[True]
    
    messages_list = list()
    generation_options_list = list()
    for _idx, _record in tqdm(
        enumerate(records),
        initial=0,
        total=len(records),
        desc=f'Evaluating judge_format'
    ):
        _prediction = _record["prediction"]
        if _record.get("prediction_postprocessed", None):
            _prediction = _record["prediction_postprocessed"]
        _user_messages = ChatMessage.get_user_messages(messages=_record["messages"])
        _query = ChatMessage.get_query(message=_user_messages[-1])
        _label = _record["label"]
        if isinstance(_label, (list, tuple)):
            _label = _label[0]
        
        if not _prediction:
            logger.warning(f'Invalid prediction in record: {_prediction}')
            continue
        if not _query:
            logger.warning(f'Invalid query in record: {_query}')
            continue
        if not _label:
            logger.warning(f'Invalid label in record: {_label}')
            continue
        
        messages = list()
        if task_config.evaluation.judges["judge_format"].system_prompt:
            messages.append(ChatMessage(
                role="user",
                content=[
                    ChatTextContent(type="text", value=task_config.evaluation.judges["judge_format"].system_prompt), 
                ],
            ))

        judge_prompt = format_task_prompt(
            task_prompt=task_config.evaluation.judges["judge_format"].judge_prompt,
            prediction=_prediction,
            query=_query,
            label=_label,
        )
        messages.append(ChatMessage(
            role="user",
            content=[ChatTextContent(type="text", value=judge_prompt), ],
        ))
        messages_list.append(messages)
        
        generation_options = ApiGenerationOptions(
            temperature=task_config.evaluation.judges["judge_format"].temperature,
            max_tokens=task_config.evaluation.judges["judge_format"].max_tokens,
        )                
        generation_options_list.append(generation_options)
    
    responses = None
    api_group = get_api_group(api_name=task_config.evaluation.judges["judge_format"].judge_model)
    if task_config.evaluation.judges["judge_format"].do_async and api_group != "google":
        responses = asyncio.run(batch_chat_completion_async_api(
            api_name=task_config.evaluation.judges["judge_format"].judge_model,
            messages_list=messages_list,
            generation_options_list=generation_options_list,
            options_list=None,
            tools_list=None,
            response_format=JudgeResponseFormat,
        ))
    else:
        responses = batch_chat_completion_sync_api(
            api_name=task_config.evaluation.judges["judge_format"].judge_model,
            messages_list=messages_list,
            generation_options_list=generation_options_list,
            options_list=None,
            tools_list=None,
            response_format=JudgeResponseFormat,
        )
    
    metrics = None
    group_metrics = dict()
    # sample_metrics must be parallel to ``records`` so that downstream
    # evaluate.py can index by ``_record_idx`` from enumerate(records).
    # Previously ``zip(records, responses)`` truncated to the shorter side
    # when the judge dropped/failed responses, causing IndexError later.
    sample_metrics = [{"exact_match": None} for _ in records]
    corrects, corrects_group = list(), defaultdict(list)
    confidences, confidences_group = list(), defaultdict(list)
    for _idx, _record in enumerate(records):
        _response = responses[_idx] if _idx < len(responses) else None
        # `batch_chat_completion_*` returns an ``*InferenceOutput`` dataclass
        # when called with ``messages_list`` (legacy paths still returned a
        # raw dict), so accept both shapes — and ``None`` from failed calls.
        if _response is None:
            _prediction = None
        elif isinstance(_response, dict):
            _prediction = _response.get("prediction")
        else:
            _prediction = getattr(_response, "prediction", None)
        _correct, _confidence = None, None
        if (
            isinstance(_prediction, dict)
            and _prediction.get("correct")
            and _prediction.get("confidence")
        ):
            _correct = "yes" in _prediction["correct"]
            _confidence = _prediction["confidence"]
            corrects.append(_correct)
            confidences.append(_confidence)
            if _record["meta"]["category"]:
                corrects_group[_record["meta"]["category"]].append(_correct)
                confidences_group[_record["meta"]["category"]].append(_confidence)
        sample_metrics[_idx] = {"exact_match": _correct}
        
    metrics = TextEvaluator.compute_calibration_error(
        predictions=corrects,
        labels=None,
        confidences=confidences,
    )
    for _group_name in corrects_group.keys():
        group_metrics[_group_name] = TextEvaluator.compute_calibration_error(
            predictions=corrects_group[_group_name],
            labels=None,
            confidences=confidences_group[_group_name],
        )        
    
    # omni_evaluator
    evaluation_output = EvaluationRunOutput.from_task(
        args,
        task_name,
        task_config,
        records,
        metrics,
        group_metrics=group_metrics,
        sample_metrics=sample_metrics,
        num_valid_evaluation=len(confidences),
    )
    return evaluation_output, sample_metrics