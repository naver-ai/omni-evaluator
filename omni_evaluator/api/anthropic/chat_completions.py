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

import asyncio
import json
from omegaconf import ListConfig, DictConfig
import PIL
from PIL import Image
import pydantic
import time
import traceback
from typing import Optional, Union, Any, List, Dict, Tuple, Callable

import logging
logger = logging.getLogger(__name__)

try: # anthropic
    import anthropic
except Exception as ex:
    logger.warning('Could not import dependencies: Anthropic api')

from omni_evaluator.api.anthropic import get_engine_features
from omni_evaluator.inference import (
    TIMEOUT, MAX_RETRY, WAIT_BETWEEN_RETRY,
    NUM_MAX_COROUTINES,
)
from omni_evaluator.utils.common import remove_stop_words
from omni_evaluator.utils.response import summarize_payload_shape, is_valid_response
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.generation_options import ApiAnthropicGenerationOptions


def chat_completion_sync(
    client,
    api_name: str,
    messages: List[Union[Dict[str, Any], ChatMessage]],
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    reasoning_options=None,
    timeout: Optional[int] = TIMEOUT,
    max_retry: Optional[int] = MAX_RETRY,
    wait_between_retry: Optional[int] = WAIT_BETWEEN_RETRY,
) -> Optional[Dict[str, Any]]:
    # Convert generation_options
    if not isinstance(generation_options, dict) or len(generation_options) < 1:
        generation_options = dict()
    generation_options = ApiAnthropicGenerationOptions.from_dict(
        obj=generation_options,
        api_name=api_name,
        reasoning_options=reasoning_options,
    ).to_dict()

    # Extract system messages and convert ChatMessage → internal dict (no template)
    _sys_msgs = []
    _converted = []
    for _message in messages:
        if isinstance(_message, ChatMessage):
            if _message.role == "system":
                _sys = ChatMessage.get_query(_message)
                if isinstance(_sys, str):
                    _sys_msgs.append(_sys)
                continue
        _converted.append(_message)
    messages = _converted
    system_message = "\n".join(_sys_msgs) if _sys_msgs else None

    # Normalize media formats
    engine_features = get_engine_features(api_name=api_name)
    messages = [
        ChatMessage.preprocess_message(
            message=_message,
            remove_audio=not engine_features["support_audio_understanding"],
            content_fields_audio=None,
            remove_image=not engine_features["support_image_understanding"],
            content_fields_image=None,
            remove_video=not engine_features["support_video_understanding"],
            content_fields_video=None,
        )
        for _message in messages
    ]

    # Convert internal dict → Anthropic format
    messages = [
        ChatMessage.to_template(obj=_message, template="anthropic")
        for _message in messages
    ]

    if "stop_sequences" not in generation_options:
        generation_options["stop_sequences"] = list()
    if tools:
        generation_options["tools"] = tools

    _response_format_original = response_format
    if response_format:
        try: # Pydantic v2
            response_format = response_format.model_json_schema()
        except AttributeError: # Pydantic v1
            response_format = response_format.schema()
        response_format = json.dumps(
            response_format, ensure_ascii=False, indent=2,
        )
        response_format = f'''\n\nYou must output a single JSON object only.
Do not include any extra text, explanations, or markdown formatting.

Your response must strictly follow the following JSON Schema:
{response_format}'''
        if system_message:
            system_message += response_format
        else:
            system_message = response_format.lstrip()

    for cur_try in range(1, max_retry+1):
        response, latency = None, None
        try:
            _start_time = time.time()
            response = client.with_options(timeout=timeout).messages.create(
                model=api_name,
                system=system_message if system_message else list(),
                messages=messages,
                **generation_options,
            )
            latency = time.time() - _start_time
            response = json.loads(response.json())
            response["latency"] = latency
            response.update(parse_response(
                response,
                response_format=_response_format_original,
                stop_words=generation_options.get("stop_sequences", None),
            ))
            if not is_valid_response(response):
                # content-contract failure — identical retry cannot help; surface once and stop.
                logger.error(
                    f'anthropic response unusable, not retrying (model={api_name}); '
                    f'no usable text/tool content (raise --max_new_tokens or set '
                    f'--thinking_budget) | response={response}'
                )
                return None
            return response
        except Exception as ex:
            _status = (
                getattr(ex, 'status_code', None)
                or getattr(getattr(ex, 'response', None), 'status_code', None)
            )
            if isinstance(_status, int) and 400 <= _status < 500:
                summarize_payload_shape(
                    {'model': api_name, 'system': system_message or [], 'messages': messages, **generation_options},
                    status=_status, logger=logger,
                )
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request anthropic: {response}')
            traceback.print_exc()
            time.sleep(wait_between_retry * cur_try)
            continue

    logger.error(f'Failed after {max_retry} tries')
    return None


async def chat_completion_async(
    client,
    api_name: str,
    messages: List[Union[Dict[str, Any], ChatMessage]],
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    reasoning_options=None,
    semaphore: Optional[asyncio.locks.Semaphore] = None,
    timeout: Optional[int] = TIMEOUT,
    max_retry: Optional[int] = MAX_RETRY,
    wait_between_retry: Optional[int] = WAIT_BETWEEN_RETRY,
) -> Optional[Dict[str, Any]]:
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    # Convert generation_options
    if not isinstance(generation_options, dict) or len(generation_options) < 1:
        generation_options = dict()
    generation_options = ApiAnthropicGenerationOptions.from_dict(
        obj=generation_options,
        api_name=api_name,
        reasoning_options=reasoning_options,
    ).to_dict()

    # Extract system messages and convert ChatMessage → internal dict (no template)
    _sys_msgs = []
    _converted = []
    for _message in messages:
        if isinstance(_message, ChatMessage):
            if _message.role == "system":
                _sys = ChatMessage.get_query(_message)
                if isinstance(_sys, str):
                    _sys_msgs.append(_sys)
                continue
        _converted.append(_message)
    messages = _converted
    system_message = "\n".join(_sys_msgs) if _sys_msgs else None

    # Normalize media formats
    engine_features = get_engine_features(api_name=api_name)
    messages = [
        ChatMessage.preprocess_message(
            message=_message,
            remove_audio=not engine_features["support_audio_understanding"],
            content_fields_audio=None,
            remove_image=not engine_features["support_image_understanding"],
            content_fields_image=None,
            remove_video=not engine_features["support_video_understanding"],
            content_fields_video=None,
        )
        for _message in messages
    ]

    # Convert internal dict → Anthropic format
    messages = [
        ChatMessage.to_template(obj=_message, template="anthropic")
        for _message in messages
    ]

    if "stop_sequences" not in generation_options:
        generation_options["stop_sequences"] = list()
    if tools:
        generation_options["tools"] = tools

    _response_format_original = response_format
    if response_format:
        try: # Pydantic v2
            response_format = response_format.model_json_schema()
        except AttributeError: # Pydantic v1
            response_format = response_format.schema()
        response_format = json.dumps(
            response_format, ensure_ascii=False, indent=2,
        )
        response_format = f'''\n\nYou must output a single JSON object only.
Do not include any extra text, explanations, or markdown formatting.

Your response must strictly follow the following JSON Schema:
{response_format}'''
        if system_message:
            system_message += response_format
        else:
            system_message = response_format.lstrip()

    for cur_try in range(1, max_retry+1):
        response, latency = None, None
        try:
            async with semaphore:
                _start_time = time.time()
                response = await client.with_options(timeout=timeout).messages.create(
                    model=api_name,
                    system=system_message if system_message else list(),
                    messages=messages,
                    **generation_options,
                )
                latency = time.time() - _start_time
            response = json.loads(response.json())
            response["latency"] = latency
            response.update(parse_response(
                response,
                response_format=_response_format_original,
                stop_words=generation_options.get("stop_sequences", None),
            ))
            if not is_valid_response(response):
                # content-contract failure — identical retry cannot help; surface once and stop.
                logger.error(
                    f'anthropic response unusable, not retrying (model={api_name}); '
                    f'no usable text/tool content (raise --max_new_tokens or set '
                    f'--thinking_budget) | response={response}'
                )
                return None
            return response
        except Exception as ex:
            _status = (
                getattr(ex, 'status_code', None)
                or getattr(getattr(ex, 'response', None), 'status_code', None)
            )
            if isinstance(_status, int) and 400 <= _status < 500:
                summarize_payload_shape(
                    {'model': api_name, 'system': system_message or [], 'messages': messages, **generation_options},
                    status=_status, logger=logger,
                )
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request anthropic: {response}')
            traceback.print_exc()
            await asyncio.sleep(wait_between_retry * cur_try)
            continue

    logger.error(f'Failed after {max_retry} tries')
    return None


def parse_response(
    response,
    response_format=None,
    stop_words=None,
) -> Dict[str, Any]:
    generated_text, prediction, reasoning = None, None, None
    tool_calls, function_call = None, None
    annotations, finish_reason, error_message = list(), None, None

    if not isinstance(response, dict):
        error_message = 'failed to request API'
    elif "content" in response and isinstance(response.get("content", None), list):
        generated_text = [
            _block["text"]
            for _block in response["content"]
            if _block.get("type") == "text"
        ]
        _thinking_blocks = [
            _block.get("thinking", "")
            for _block in response["content"]
            if _block.get("type") == "thinking"
        ]
        if _thinking_blocks:
            reasoning = "\n".join(_thinking_blocks)
        tool_calls = [
            _block
            for _block in response["content"]
            if _block.get("type") == "tool_use"
        ] or None
        finish_reason = response.get("stop_reason", None)
    else:
        error_message = 'unexpected response format'

    if not generated_text:
        pass
    elif response_format:
        try:
            prediction = [json.loads(_text) for _text in generated_text]
        except Exception:
            pass
    else:
        prediction = [
            remove_stop_words(text=_text.strip(), stop_words=stop_words)
            if _text else None
            for _text in generated_text
        ]

    if isinstance(prediction, (list, tuple)) and len(prediction) == 1:
        prediction = prediction[0]

    return {
        "generated_text": generated_text,
        "prediction": prediction,
        "reasoning_content": reasoning,
        "tool_calls": tool_calls,
        "function_call": function_call,
        "annotations": annotations,
        "finish_reason": finish_reason,
        "error_message": error_message,
    }