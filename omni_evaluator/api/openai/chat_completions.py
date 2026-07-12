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

try: # openai
    import openai
except Exception as ex:
    logger.warning('Could not import dependencies: OpenAI api')

from omni_evaluator.api.openai import get_engine_features
from omni_evaluator.inference import (
    TIMEOUT, MAX_RETRY, WAIT_BETWEEN_RETRY,
    NUM_MAX_COROUTINES, 
)
from omni_evaluator.utils.common import remove_stop_words
from omni_evaluator.utils.response import summarize_payload_shape, is_valid_response
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.generation_options import ApiOpenaiGenerationOptions


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
):
    # Convert generation_options
    if not isinstance(generation_options, dict) or len(generation_options) < 1:
        generation_options = dict()
    generation_options = ApiOpenaiGenerationOptions.from_dict(
        obj=generation_options,
        api_name=api_name,
        reasoning_options=reasoning_options,
    ).to_dict()

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

    # Convert internal dict → OpenAI format
    messages = [
        ChatMessage.to_template(obj=_message, template="openai")
        for _message in messages
    ]

    for cur_try in range(1, max_retry+1):
        response, latency = None, None
        try:
            _start_time = time.time()
            response = client.chat.completions.create(
                model=api_name,
                messages=messages,
                **generation_options,
                tools=tools if tools else list(),
                timeout=timeout,
            )
            latency = time.time() - _start_time
            response = json.loads(response.json())
            response["latency"] = latency
            response.update(parse_response(
                response,
                response_format=None,
                stop_words=generation_options.get("stop", None),
            ))
            if not is_valid_response(response):
                # content-contract failure — identical retry cannot help; surface once and stop.
                logger.error(
                    f'openai response unusable, not retrying (model={api_name}); '
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
                    {'model': api_name, 'messages': messages, 'tools': tools or [], **generation_options},
                    status=_status, logger=logger,
                )
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request openai: {response}')
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
):
    # Convert generation_options
    if not isinstance(generation_options, dict) or len(generation_options) < 1:
        generation_options = dict()
    generation_options = ApiOpenaiGenerationOptions.from_dict(
        obj=generation_options,
        api_name=api_name,
        reasoning_options=reasoning_options,
    ).to_dict()

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

    # Convert internal dict → OpenAI format
    messages = [
        ChatMessage.to_template(obj=_message, template="openai")
        for _message in messages
    ]

    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    for cur_try in range(1, max_retry+1):
        response, latency = None, None
        try:
            async with semaphore:
                _start_time = time.time()
                response = await client.chat.completions.create(
                    model=api_name,
                    messages=messages,
                    **generation_options,
                    tools=tools if tools else list(),
                    timeout=timeout,
                )
                latency = time.time() - _start_time
            response = json.loads(response.json())
            response["latency"] = latency
            response.update(parse_response(
                response,
                response_format=None,
                stop_words=generation_options.get("stop", None),
            ))
            if not is_valid_response(response):
                # content-contract failure — identical retry cannot help; surface once and stop.
                logger.error(
                    f'openai response unusable, not retrying (model={api_name}); '
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
                    {'model': api_name, 'messages': messages, 'tools': tools or [], **generation_options},
                    status=_status, logger=logger,
                )
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request openai: {response}')
            traceback.print_exc()
            await asyncio.sleep(wait_between_retry * cur_try)
            continue

    logger.error(f'Failed after {max_retry} tries')
    return None


def parse_response(
    response,
    response_format=None,
    stop_words=None,
):
    generated_text, prediction, reasoning = None, None, None
    tool_calls, function_call = None, None
    annotations, finish_reason, error_message = list(), None, None

    if not isinstance(response, dict):
        error_message = 'failed to request API'
    elif "choices" in response and len(response.get("choices", list())) > 0:
        _last_message = response["choices"][-1].get("message", dict())
        generated_text = [
            _choice["message"]["content"]
            for _choice in response["choices"]
        ]
        if _last_message.get("reasoning", None):
            reasoning = _last_message["reasoning"]
        elif _last_message.get("reasoning_content", None):
            reasoning = _last_message["reasoning_content"]
        tool_calls = _last_message.get("tool_calls", None)
        function_call = _last_message.get("function_call", None)
        annotations = _last_message.get("annotations", list())
        finish_reason = response["choices"][-1].get("finish_reason", None)
    elif "output" in response:
        _message_items = [
            _item for _item in response["output"]
            if _item.get("type") == "message"
        ]
        if _message_items:
            generated_text = [
                _content["text"]
                for _content in _message_items[-1].get("content", list())
                if _content.get("type") == "output_text"
            ]
        _reasoning_summaries = [
            _summary.get("text", "")
            for _item in response["output"]
            if _item.get("type") == "reasoning"
            for _summary in _item.get("summary", list())
        ]
        if _reasoning_summaries:
            reasoning = "\n".join(_reasoning_summaries)
        finish_reason = response.get("status", None)
    else:
        error_message = 'unexpected response format'

    if not generated_text:
        pass
    elif response_format:
        # response_format requested structured JSON — parse per-choice.
        # On failure for any choice, fall back to that choice's raw text;
        # callers reading prediction as a dict should still defensively
        # handle str/None to absorb invalid-JSON edge cases.
        # generated_text can arrive as a single string (one choice) or a
        # list of strings (n>1); iterating a str would split per char and
        # mis-parse, so wrap singletons explicitly.
        _texts = [generated_text] if isinstance(generated_text, str) else list(generated_text)
        prediction = []
        for _text in _texts:
            try:
                prediction.append(json.loads(_text))
            except Exception:
                prediction.append(_text)
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