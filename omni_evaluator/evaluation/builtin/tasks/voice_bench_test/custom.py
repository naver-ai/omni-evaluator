# Reference from https://github.com/MatthewCYM/VoiceBench (Apache-2.0)

# Modifications Copyright (c) 2026-present NAVER Cloud Corp.
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
import copy
from functools import partialmethod
import importlib
import json
from litellm import completion
import math
import numpy as np
import os
from pathlib import Path
import PIL
from PIL import Image
import random
import time
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence, Callable, Iterable

from omni_evaluator import EvaluationEngine, InferenceEngine
from omni_evaluator.api import get_api_group
from omni_evaluator.api.chat_completions import (
    batch_chat_completion_async,
    batch_chat_completion_sync,
)
from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record
)
from omni_evaluator.schemas.chat import (
    OcrToken, EntityToken,
    Message as ChatMessage, 
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
)
from omni_evaluator.infer import infer_record
from omni_evaluator.inference import NUM_DEBUG_SAMPLES
from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from omni_evaluator.schemas.generation_options import ApiGenerationOptions
from omni_evaluator.schemas.inference import Record
from omni_evaluator.schemas.task import TaskConfig
from omni_evaluator.utils.patches import ClassPatcher
from omni_evaluator.utils.multimodal import to_pil_image
from omni_evaluator.evaluation.metrics.image_evaluator import ImageEvaluator

from omni_evaluator.evaluation.builtin.tasks.voice_bench_test.prompts import JUDGE_RATING_PROMPT, JUDGE_BINARY_PROMPT


# Reference from https://github.com/MatthewCYM/VoiceBench (Apache-2.0) - src/evaluator/__init__.py (evaluator_mapping)
EVALUATOR_MAAPING = { 
    "advbench": "harm", 
    "alpacaeval": "open", 
    "alpacaeval_full": "open", 
    "alpacaeval_speaker": "open", 
    "bbh": "bbh", 
    "commoneval": "open",
    "ifeval": "ifeval",
    "mmsu": "mcq", 
    "mtbench": "mcq", # TODO: should find
    "openbookqa": "mcq", 
    "sd-qa": "qa",
    "wildvoice": "open",
}

# Reference from https://github.com/MatthewCYM/VoiceBench (Apache-2.0) - api_judge.py (judge model & sampling params)
JUDGE_API_NAME = "gpt-4o-mini"


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
    # if sample["dataset_name"] in ["advbench", "alpacaeval", "alpacaeval_full", "bbh", "commoneva", ]
    _sample_index = f'{sample["dataset_name"]}__{sample_idx:08}'
    if sample.get("id", None):
        _sample_index = sample["id"]
    elif sample.get("question_id", None):
        _sample_index = sample["question_id"]
        
    labels = sample.get("reference", None)
    if isinstance(labels, str):
        labels = [labels, ]
        
    user_contents = list()
    if sample["dataset_name"] in [
        "mtbench",
    ]:
        user_contents.append({"type": "audio", "value": sample["audio1"]["bytes"]})
        user_contents.append({"type": "audio", "value": sample["audio2"]["bytes"]})
        # user_contents.append({"type": "text", "value": sample["turns"]})
    else:
        user_contents.append({"type": "audio", "value": sample["audio"]["bytes"]})
        user_contents.append({"type": "text", "value": sample["prompt"]})
        
    _sample = {
        "index": _sample_index,
        "messages": [
            {
                "role": "user",
                "content": user_contents,
            },
        ],
        "label": labels,
        "meta": {
            "id": sample.get("id", None),
            "question_id": sample.get("question_id", None),
            "category": sample["dataset_name"],
            "name": sample["dataset_name"],
            "revision": sample["dataset_revision"],
            "split": sample["dataset_split"],
            "prompt": sample.get("prompt", None),
            "turns": sample.get("turns", None),
            "reference": sample.get("reference", None),
            "key": sample.get("key", None),
            "instruction_id_list": sample.get("instruction_id_list", None),
            "kwargs": sample.get("kwargs", None),
            "audio_path": sample["audio"]["path"],
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
    from voice_bench import evaluator # from voice_bench import api_judge
    
    records_by_group = dict()
    for _record in records:
        _group_name = _record["meta"]["category"]
        if _group_name not in records_by_group:
            records_by_group[_group_name] = list()
        records_by_group[_group_name].append(_record)

    metrics = {
        "average": list()
    }
    group_metrics = dict()
    sample_metrics = None
    _num_valid_evaluation = 0
    for _group_name, _records in records_by_group.items():        
        # api_judge: generate score given prompt, reference, and response
        # Reference from https://github.com/MatthewCYM/VoiceBench (Apache-2.0) - api_judge.py (judge-scoring flow)
        judge_scores = [None, ] * len(_records)
        if _group_name in [
            "alpacaeval",
            "alpacaeval_full",
            "alpacaeval_speaker",
            "commoneval",
            "wildvoice",
            "sd-qa",
        ]:
            _judge_messages_list, _generation_options_list= list(), list()
            for _record_idx, _record in enumerate(_records):
                # skip if prediction is empty
                if not _record["prediction"]:
                    continue 

                _last_user_message = ChatMessage.get_user_messages(
                    messages=_record["messages"],
                )[-1]
                _query = ChatMessage.get_query(message=_last_user_message)
                _judge_prompt = None
                if _record["label"] is None:
                    _judge_prompt = JUDGE_RATING_PROMPT.format(
                        prompt=_query,
                        response=_record["prediction"],
                    )
                else:
                    _judge_prompt = JUDGE_BINARY_PROMPT.format(
                        prompt=_query,
                        response=_record["prediction"],
                        reference=_record["label"][0],
                    )
                _judge_messages = list()
                _judge_messages.append(ChatMessage(
                    role="system",
                    content=[
                        ChatTextContent(
                            type="text", 
                            value="You are a helpful assistant who tries to help answer the user's question.",
                        ),
                    ]
                ))
                _judge_messages.append(ChatMessage(
                    role="user",
                    content=[
                        ChatTextContent(
                            type="text", 
                            value=_judge_prompt,
                        ),
                    ]
                ))
                _judge_messages_list.append(_judge_messages)
                
                _api_group = get_api_group(api_name=JUDGE_API_NAME)
                _generation_options = ApiGenerationOptions.from_dict(
                    api_name=JUDGE_API_NAME,
                    obj={
                        "max_tokens": 1024,
                        "frequency_penalty": 0.0,
                        "presence_penalty": 0.0,
                        "stop": None,
                        "temperature": 0.5,
                        "top_p": 0.95,
                        "n": 3,
                    },
                    api_group=_api_group,
                ).to_dict()
                _generation_options_list.append(_generation_options)
            
            _responses = None
            if args.do_async:
                _responses = asyncio.run(batch_chat_completion_async(
                    api_name=JUDGE_API_NAME,
                    messages_list=_judge_messages_list,
                    generation_options_list=_generation_options_list,
                    semaphore_size=args.inference_concurrency,
                ))
            else:
                _responses = batch_chat_completion_sync(
                    api_name=JUDGE_API_NAME,
                    messages_list=_judge_messages_list,
                    generation_options_list=_generation_options_list,
                )
            
            # parse judge_scores from responses
            judge_scores = [
                _response.prediction
                for _response in _responses
            ]
         
        _evaluator_type = EVALUATOR_MAAPING[_group_name]
        _evaluation_inputs = list()
        for _record_idx, (_record, _judge_scores) in enumerate(zip(_records, judge_scores)):
            _prediction = _record["prediction"]
            if _record["prediction_postprocessed"]:
                _prediction = _record["prediction_postprocessed"]
            if not _prediction:
                continue
            
            _evaluation_input = {
                "id": _record["meta"]["id"],
                "prompt": _record["meta"]["prompt"],
                "reference": _record["meta"]["reference"],
                "response": _record["prediction"],
            }
            if _record["meta"].get("key", None): # ifeval
                _evaluation_input["key"] = _record["meta"]["key"]
            if _record["meta"].get("instruction_id_list", None): # ifeval
                _evaluation_input["instruction_id_list"] = _record["meta"]["instruction_id_list"]
            if _record["meta"].get("kwargs", None): # ifeval
                _evaluation_input["kwargs"] = _record["meta"]["kwargs"]
            if _judge_scores: 
                _evaluation_input["score"] = _judge_scores
            _evaluation_inputs.append(_evaluation_input)   
            
        # Reference from https://github.com/MatthewCYM/VoiceBench (Apache-2.0) - src/evaluator/*.py (evaluator dispatch & metric scaling)
        _metric_score = None
        _group_metrics = None
        if _evaluator_type == "open":
            _group_metrics = evaluator.open.OpenEvaluator().evaluate(
                data=_evaluation_inputs,
            )
            _group_metrics["judge_rating"] = _group_metrics["gpt"] * 20.0 * 1e-2
            _metric_score = _group_metrics["judge_rating"]
        elif _evaluator_type == "qa":
            _group_metrics = evaluator.qa.QAEvaluator().evaluate(
                data=_evaluation_inputs,
            )
            _group_metrics["panda"] = _group_metrics["panda"] * 1e-2
            _group_metrics["judge_binary"] = _group_metrics["gpt"] * 1e-2
            _metric_score = _group_metrics["judge_binary"]
        elif _evaluator_type == "harm":
            _group_metrics = evaluator.harm.HarmEvaluator().evaluate(
                data=_evaluation_inputs,
            )
            _metric_score = _group_metrics["refusal_rate"]
        elif _evaluator_type == "ifeval":
            _group_metrics = evaluator.ifeval.IFEvaluator().evaluate(
                data=_evaluation_inputs,
            )
            _metric_score = _group_metrics["final"]
        elif _evaluator_type == "mcq":
            _group_metrics = evaluator.mcq.MCQEvaluator().evaluate(
                data=_evaluation_inputs,
            )
            _group_metrics["acc"] = _group_metrics["acc"] * 1e-2
            _metric_score = _group_metrics["acc"]
        elif _evaluator_type == "bbh":
            _group_metrics = evaluator.bbh.BBHEvaluator().evaluate(
                data=_evaluation_inputs,
            )
            _group_metrics["acc"] = _group_metrics["acc"] * 1e-2
            _metric_score = _group_metrics["acc"]
        group_metrics[_group_name] = _group_metrics
        metrics["average"].append(_metric_score)
        _num_valid_evaluation += len(_evaluation_inputs)
    metrics["average"] = np.mean(metrics["average"])
    
    # omni_evaluator
    evaluation_output = EvaluationRunOutput.from_task(
        args,
        task_name,
        task_config,
        records,
        metrics,
        group_metrics=group_metrics,
        sample_metrics=sample_metrics,
        num_valid_evaluation=_num_valid_evaluation,
        default_metric_keys=["reward", "pass^1"],
    )
    return evaluation_output, sample_metrics