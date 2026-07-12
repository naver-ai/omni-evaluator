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
import logging
from omegaconf import ListConfig, DictConfig
import PIL
from PIL import Image
import pydantic
import time
import traceback
from typing import Optional, Union, Any, List, Dict, Tuple, Callable

logger = logging.getLogger(__name__)

try: # openai
    import openai
except Exception as ex:
    logger.warning(f'Could not import dependencies: OpenAI api')

from omni_evaluator.api.openai import get_engine_features
from omni_evaluator.inference import (
    TIMEOUT, MAX_RETRY, WAIT_BETWEEN_RETRY,
    NUM_MAX_COROUTINES, 
)
from omni_evaluator.utils.common import remove_stop_words
from omni_evaluator.schemas.chat import Message as ChatMessage


def completion_sync(
    client,
    api_name: str,
    prompt: str,
    system_message: Optional[str] = None, # not used
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    timeout: Optional[int] = TIMEOUT,
    max_retry: Optional[int] = MAX_RETRY,
    wait_between_retry: Optional[int] = WAIT_BETWEEN_RETRY,
) -> Optional[Dict[str, Any]]:
    if generation_options is None or len(generation_options) < 1:
        generation_options = dict()

    for cur_try in range(1, max_retry+1):
        response, latency = None, None
        try:
            _start_time = time.time()
            response = client.completions.create(
                model=api_name,
                prompt=prompt,
                **generation_options,
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
            return response
        except Exception as ex:
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request openai: {response}')
            traceback.print_exc()
            time.sleep(wait_between_retry * cur_try)
            continue

    logger.error(f'Failed after {max_retry} tries')
    return None


async def completion_async(
    client,
    api_name: str,
    prompt: str,
    system_message: Optional[str] = None, # not used
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    semaphore: Optional[asyncio.locks.Semaphore] = None, 
    timeout: Optional[int] = TIMEOUT,
    max_retry: Optional[int] = MAX_RETRY,
    wait_between_retry: Optional[int] = WAIT_BETWEEN_RETRY,
) -> Optional[Dict[str, Any]]:
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)
    
    if generation_options is None or len(generation_options) < 1:
        generation_options = dict()

    for cur_try in range(1, max_retry+1):
        response, latency = None, None            
        try:
            async with semaphore:
                _start_time = time.time()
                response = await client.completions.create(
                    model=api_name,
                    prompt=prompt,
                    **generation_options,
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
            return response
        except Exception as ex:
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
) -> Dict[str, Any]:
    generated_text, prediction, reasoning = None, None, None
    tool_calls, function_call = None, None
    annotations, finish_reason, error_message = list(), None, None

    if not isinstance(response, dict):
        error_message = 'failed to request API'
    elif "choices" in response and len(response.get("choices", list())) > 0:
        generated_text = [
            _choice["text"]
            for _choice in response["choices"]
        ]
        tool_calls = response["choices"][-1].get("tool_calls", None)
        function_call = response["choices"][-1].get("function_call", None)
        annotations = response["choices"][-1].get("annotations", list())
        finish_reason = response["choices"][-1].get("finish_reason", None)
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