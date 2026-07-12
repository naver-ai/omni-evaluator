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
from omni_evaluator.utils.multimodal import to_audio_bytes, to_nparray_audio, to_pil_image, to_nparray_video
from omni_evaluator.utils.string import is_numeric


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
    sample_index = sample.get("index", sample_idx)
    sample_meta = {
        "index": sample["index"],
        "category": sample["audio type"],
        # "category": sample["task type"],
        "task_type": sample["task type"],
        "question": sample["question"],
        "answer": sample["answer"],
        "options": sample["options"],
        "audio_type": sample["audio type"],
        "audio_content": sample["audio content"],
        "image_content": sample["image content"],
        "audio_path": sample["audio_path"],
        "image_path": sample["image_path"], 
        "num_added_source": 0,
    }
    
    option_contents = sample["options"]
    options = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:len(option_contents)]
        
    messages = list()
    user_contents = list()
    # add audios
    if (
        isinstance(task_config.inference.config, dict)
        and not task_config.inference.config.get("audio", False)
    ): # do not use audio input
        pass
    elif isinstance(sample["audio"], (list, tuple)):
        for _audio in sample["audio"]:
            # HF Audio feature unwrap: cast_column("audio", Audio(decode=False))
            # yields {"bytes": ..., "path": ...}; decode=True yields
            # {"array": np.ndarray, "sampling_rate": int, "path": ...}.
            # Match the same key precedence used elsewhere in this repo's audio tasks.
            if isinstance(_audio, dict):
                _audio = _audio.get("bytes") or _audio.get("array") or _audio.get("path")
            user_contents.append({
                "type": "audio",
                "value": to_audio_bytes(audio=_audio),
            })
    elif sample["audio"] is not None:
        _audio = sample["audio"]
        if isinstance(_audio, dict):
            _audio = _audio.get("bytes") or _audio.get("array") or _audio.get("path")
        user_contents.append({
            "type": "audio",
            "value": to_audio_bytes(audio=_audio),
        })
    else:
        raise ValueError(f'invalid audio type: {sample["audio"]}')
    
    # add images
    if (
        isinstance(task_config.inference.config, dict)
        and not task_config.inference.config.get("image", False)
    ): # do not use image input
        pass
    elif isinstance(sample["image"], (list, tuple)):
        for _image in sample["image"]:
            user_contents.append({
                "type": "image",
                "value": to_pil_image(image=_image),
            })
    elif sample["image"] is not None:
        user_contents.append({
            "type": "image",
            "value": to_pil_image(image=sample["image"]),
        })
    else:
        raise ValueError(f'invalid image type: {sample["image"]}')
    
    # paper-strict (aligned with lmms-eval omni_bench utils.py):
    #   audio / image modalities are attached as message content, so there is no separate
    #   text preamble (removes our arbitrary wording like "Please answer the following
    #   question based on ..."). Only audio_transcript / image_caption modes add a
    #   prefix block (exactly the same format as lmms-eval's _audio_transcript /
    #   _image_caption variant prompts).
    _num_added_source = 0
    _context_prefix = ""
    if isinstance(task_config.inference.config, dict):
        if (
            task_config.inference.config.get("audio", False)
            and sample.get("audio", None)
        ):
            _num_added_source += 1
        if (
            task_config.inference.config.get("audio_transcript", False)
            and sample.get("audio transcript", None)
        ):
            _context_prefix += f'Audio context for the images:\n{sample["audio transcript"]}\n\n'
            _num_added_source += 1
        if (
            task_config.inference.config.get("image", False)
            and sample.get("image", None)
        ):
            _num_added_source += 1
        if (
            task_config.inference.config.get("image_caption", False)
            and sample.get("image caption", None)
        ):
            _context_prefix += f'Visual Context for the audio:\n{sample["image content"]}\n\n'
            _num_added_source += 1
        sample_meta["num_added_source"] = _num_added_source
    _query = f'{_context_prefix}Question:\n{sample["question"]}'

    task_prompt_kwargs = dict()
    _task_prompt = task_prompt
    if not _task_prompt:
        _task_prompt = getattr(task_config.prompts, "task_prompt", None)
    # task_prompt may be set to null in the yaml (the omni_bench paper uses the raw
    # question; no template). Fall back to an empty string for the placeholder check.
    if _task_prompt and "{option}" in _task_prompt:
        task_prompt_kwargs["options"] = options
    if _task_prompt and "{option_contents}" in _task_prompt:
        task_prompt_kwargs["option_contents"] = "\n".join([
            f'{_option}. {_option_content}'
            for _option, _option_content in zip(options, option_contents)
        ])

    user_contents.append({
        "type": "text",
        "value": _query,
    })
    messages.append({
        "role": "user",
        "content": user_contents,
    })
    
    labels = list()
    _label_option_content = sample["answer"]
    _label_option = options[option_contents.index(_label_option_content)]
    labels.append(_label_option)    
    labels.append(_label_option_content)
    labels.append(f'{_label_option}. {_label_option_content}')
    
    record = default_sample_to_record(
        task_name=task_config.task_name,
        task_config=task_config,
        sample_idx=sample_idx,
        sample={
            "index": sample_index,
            "messages": messages,
            "options": options,
            "option_contents": option_contents,
            "label": labels,
            "meta": sample_meta,
        },
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        task_prompt_kwargs=task_prompt_kwargs,
        num_ocr_tokens=num_ocr_tokens,
        num_entity_tokens=num_entity_tokens,
        run_index=run_index,
    )
    return record

# def evaluate(
#     args: argparse.Namespace,
#     evaluation_method: str,
#     task_name: str,
#     task_config: TaskConfig,
#     records: List[Dict[str, Any]],
#     **kwargs,
# ):
#     metrics = dict()
#     group_metrics = dict()
#     sample_metrics = [dict() for _ in range(0, len(records))]
    
#     for _target_metric in task_config.evaluation.target_metrics:
#         judge_messages_list, generation_options_list = list(), list()
#         with charxiv_patch():
#             for _record_idx, _record in enumerate(records):
#                 _judge_prompt = None
#                 if task_config.inference.config["mode"] == "descriptive":
#                     from charxiv import descriptive_utils
#                     from constants import (
#                         DESCRIPTIVE_RESP_INST, DESCRIPTIVE_GRADING_PREFIX, 
#                         DESCRIPTIVE_GRADING_QMAP, DESCRIPTIVE_GRADING_ICL,
#                     )
#                     # construct judge_message
#                     _judge_query = DESCRIPTIVE_GRADING_QMAP[_record["meta"]["question_id"]]
#                     _rubric_icl = descriptive_utils.get_rubric(_record["meta"]["question_id"])
#                     _json_keys = descriptive_utils.build_json_keys(1) # len(batch)
#                     # populate batch size, question, and json keys spec
#                     _judge_prefix = DESCRIPTIVE_GRADING_PREFIX\
#                         .replace("<|NUM_TRIPLETS|>", str(1))\
#                         .replace("<|OVERARCHING_QUESTION|>", _judge_query)\
#                         .replace("<|JSON_KEYS|>", _json_keys)
#                     # add in-context grading example based on the template id                
#                     # prompt + example + model responses
#                     _judge_input = descriptive_utils.populate_grading_inputs(
#                         batch=[ # resp_key, response, answer
#                             (_record["index"], _record["prediction"], _record["label"][0]), 
#                         ],
#                     )
#                     _judge_prompt = _judge_prefix + _rubric_icl + _judge_input
                    
#                 else:
#                     from constants import (
#                         REASONING_RESP_INST, REASONING_GRADING_PREFIX, REASONING_GRADING_INST,
#                     )
#                     _query = _record["meta"]["reasoning_q"]
#                     _answer = _record["label"][0]
#                     _answer_type = _record["meta"]["reasoning_a_type"]
#                     _prediction = _record["prediction"]
#                     # get query for answer type (inst_category), then
#                     # populate the query with the question, ground truth, and response
#                     _judge_prompt = REASONING_GRADING_PREFIX + copy.deepcopy(\
#                         REASONING_GRADING_INST[_answer_type])\
#                         .replace("<|question|>", _query)\
#                         .replace("<|ground_truth|>", _answer)\
#                         .replace("<|response|>", _prediction)
                    
#                 _judge_messages = list()
#                 _judge_messages.append(ChatMessage(
#                     role="user",
#                     content=[
#                         ChatTextContent(
#                             type="text",
#                             value=_judge_prompt,
#                         )
#                     ]
#                 ))
#                 judge_messages_list.append(_judge_messages)
                
#                 _api_group = get_api_group(api_name=task_config.evaluation.judge_evaluator.metrics[_target_metric].judge_model)
#                 _generation_options = ApiGenerationOptions.from_dict(
#                     obj=task_config.evaluation.judge_evaluator.metrics[_target_metric].to_dict(),
#                     api_group=_api_group,
#                 ).to_dict()
#                 generation_options_list.append(_generation_options)
            
#         responses = None
#         if args.do_async:
#             responses = asyncio.run(batch_chat_completion_async(
#                 api_name=task_config.evaluation.judge_evaluator.metrics[_target_metric].judge_model,
#                 messages_list=judge_messages_list,
#                 system_message_list=None,
#                 generation_options_list=generation_options_list,
#                 response_format={"type": "json_object"},
#                 semaphore_size=args.inference_concurrency,
#             ))
#         else:
#             responses = batch_chat_completion_sync(
#                 api_name=task_config.evaluation.judge_evaluator.metrics[_target_metric].judge_model,
#                 messages_list=judge_messages_list,
#                 system_message_list=None,
#                 generation_options_list=generation_options_list,
#                 response_format={"type": "json_object"},
#             )
        
#         _sample_metrics = list()
#         _group_metrics = dict()
#         for _record_idx, (_record, _response) in enumerate(zip(records, responses)):
#             _group_name = _record["meta"]["category"]
#             _score = None
#             if task_config.inference.config["mode"] == "descriptive":
#                 _score = _response["prediction"]["score_T1"]
#             else:
#                 _score = _response["prediction"]["score"]
#             if not isinstance(_score, (int, float)):
#                 _score = None
#             _sample_metrics.append(_score)
#             sample_metrics[_record_idx][_target_metric] = _score
#             if _group_name not in _group_metrics:
#                 _group_metrics[_group_name] = list()
#             _group_metrics[_group_name].append(_score)
        
#         metrics[_target_metric] = np.mean(_sample_metrics)
#         for _group_name, _metric_values in _group_metrics.items():
#             if _group_name not in group_metrics:
#                 group_metrics[_group_name] = dict()
#             group_metrics[_group_name][_target_metric] = np.mean(_metric_values)
    
#     # omni_evaluator  
#     # aggregate evaluation_output
#     num_samples = task_config.num_records
#     _num_valid_inferences = sum([
#         True if _record["prediction"] else False
#         for _record in records
#     ])
#     num_empty_predictions = len(records) - _num_valid_inferences
#     coverage_inference = _num_valid_inferences / len(records)
#     _num_valid_evaluation = len([_sample_metric for _sample_metric in sample_metrics if _sample_metric])
#     coverage_evaluation = _num_valid_evaluation / num_samples

#     evaluation_output = EvaluationRunOutput(
#         inference_engine=args.inference_engine,
#         evaluation_engine=args.evaluation_engine,
#         task_name=task_name,
#         evaluation_method=task_config["evaluation"]["method"],
#         num_samples=num_samples,
#         num_empty_predictions=num_empty_predictions,
#         coverage_inference=coverage_inference,
#         coverage_evaluation=coverage_evaluation,
#         runtime_inference=None,
#         runtime_evaluation=None,
#         metric_keys=task_config["evaluation"]["display_metrics"]
#         if task_config["evaluation"]["display_metrics"] else ["reward", "pass^1", ],
#         metrics=metrics,
#         group_metrics=group_metrics,
#         sample_metrics=sample_metrics,
#     )
#     return evaluation_output, sample_metrics