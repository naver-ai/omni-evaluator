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
import aiohttp
import base64
import copy
import json
import io
import numpy as np
from omegaconf import ListConfig, DictConfig
import os
from pathlib import Path
import PIL
from PIL import Image
import pydantic
import requests
import random
import time
import traceback
from typing import Any, Dict, List, Union, Tuple, Optional, Callable
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from omni_evaluator import ApiGroup, EvaluationMethod
from omni_evaluator.inference import (
    TIMEOUT, MAX_RETRY, WAIT_BETWEEN_RETRY,
    NUM_MAX_COROUTINES, 
)
from omni_evaluator.api import get_api_group, get_client, get_engine_features
from omni_evaluator.api.openai.completions import (
    completion_sync as completion_sync_openai,
    completion_async as completion_async_openai,
)
from omni_evaluator.api.anthropic.completions import (
    completion_sync as completion_sync_anthropic,
    completion_async as completion_async_anthropic,
)
from omni_evaluator.api.google.completions import (
    completion_sync as completion_sync_google,
    completion_async as completion_async_google,
)
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import ApiInferenceOutput
from omni_evaluator.schemas.generation_options import ApiGenerationOptions


def completion_sync(
    api_name: str,
    prompt: List[str],
    system_message: Optional[str] = None,
    generation_options: Optional[Union[Dict[str, Any], ApiGenerationOptions]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    semaphore: Optional[asyncio.locks.Semaphore] = None, 
    timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    return_dict: Optional[bool] = False,
) -> Union[ApiInferenceOutput, Any, None]:
    if not isinstance(timeout, (int, float)):
        timeout = TIMEOUT
    if not isinstance(max_retry, int):
        max_retry = MAX_RETRY
    
    _completion_sync = None
    api_group = get_api_group(api_name=api_name)
    if api_group == ApiGroup.openai:
        _completion_sync = completion_sync_openai
    elif api_group == ApiGroup.anthropic:
        _completion_sync = completion_sync_anthropic
    elif api_group == ApiGroup.google:
        _completion_sync = completion_sync_google
    else:
        raise ValueError(f'unsupported api_group: {api_name}')

    if isinstance(generation_options, dict):
        generation_options = ApiGenerationOptions.from_dict(
            api_name=api_name,
            api_group=api_group,
            obj=generation_options,
        ).to_dict()
    elif isinstance(generation_options, ApiGenerationOptions):
        generation_options = generation_options.to_dict()
    
    client = get_client(api_name=api_name, do_async=False)
    _response = _completion_sync(
        client=client,
        api_name=api_name, 
        prompt=prompt,
        system_message=system_message,
        generation_options=generation_options,
        tools=tools,
        response_format=response_format,
        timeout=timeout,
        max_retry=max_retry,
    )
    if hasattr(client, "close"):
        client.close()

    if isinstance(_response, dict):
        if return_dict:
            return ApiInferenceOutput(
                prediction=_response.get("prediction", None),
                reasoning_content=_response.get("reasoning_content", None),
                generated_text=_response.get("generated_text", None),
                tool_calls=_response.get("tool_calls", None),
                function_call=_response.get("function_call", None),
                annotations=_response.get("annotations", list()),
                finish_reason=_response.get("finish_reason", None),
                latency=_response.get("latency", None),
            )
        else:
            return _response.get("prediction", None)
    else:
        return ApiInferenceOutput() if return_dict else None
    

def batch_completion_sync(
    api_name: str,
    prompt_list: List[List[str]],
    system_message_list: Optional[List[str]] = None,
    generation_options_list: Optional[List[Union[Dict[str, Any], ApiGenerationOptions]]] = None,
    options_list: Optional[List[List[str]]] = None,
    tools_list: Optional[List[List[Dict[str, Any]]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    evaluation_method: Optional[str] = None,
    timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    wait_between_retry: Optional[Union[int, float]] = None,
) -> List[ApiInferenceOutput]:
    if not isinstance(timeout, (int, float)):
        timeout = TIMEOUT
    if not isinstance(max_retry, int):
        max_retry = MAX_RETRY
    if not isinstance(wait_between_retry, (int, float)):
        wait_between_retry = WAIT_BETWEEN_RETRY

    _completion_sync = None
    api_group = get_api_group(api_name=api_name)
    if api_group == ApiGroup.openai:
        _completion_sync = completion_sync_openai
    elif api_group == ApiGroup.anthropic:
        _completion_sync = completion_sync_anthropic
    elif api_group == ApiGroup.google:
        _completion_sync = completion_sync_google
    else:
        raise ValueError(f'unsupported api_group: {api_name}')

    if (
        isinstance(generation_options_list, (list, tuple))
        and len(generation_options_list) == len(prompt_list)
    ):
        generation_options_list = [
            _generation_options.to_dict(template=api_group)
            if isinstance(_generation_options, ApiGenerationOptions) else _generation_options
            for _generation_options in generation_options_list
        ]
    else:
        generation_options_list = [None, ] * len(prompt_list)
    
    if (
        isinstance(tools_list, (list, tuple))
        and len(tools_list) == len(prompt_list)
    ):
        pass
    else:
        tools_list = [None, ] * len(prompt_list)
    
    output = None
    if evaluation_method == EvaluationMethod.perplexity:
        raise AssertionError(f'evaluation_method `perplexity` not supported for API')
    
    else:
        responses = list()
        client = get_client(api_name=api_name, do_async=False)
        for _idx, (_prompt, _generation_options, _tools, _system_message) in tqdm(enumerate(zip(
            prompt_list, generation_options_list, tools_list, system_message_list,
        )), initial=0, total=len(prompt_list)):
            responses.append(_completion_sync(
                client=client,
                api_name=api_name, 
                prompt=_prompt,
                system_message=_system_message,
                generation_options=_generation_options,
                tools=_tools,
                response_format=response_format,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
            ))
        if hasattr(client, "close"):
            client.close()

        output = list()
        for _response in responses:
            if isinstance(_response, dict):
                _output = ApiInferenceOutput(
                    prediction=_response.get("prediction", None),
                    reasoning_content=_response.get("reasoning_content", None),
                    generated_text=_response.get("generated_text", None),
                    finish_reason=_response.get("finish_reason", None),
                    latency=_response.get("latency", None),
                )
            else:
                _output = ApiInferenceOutput()
            output.append(_output)
    return output


async def batch_completion_async(
    api_name: str,
    prompt_list: List[List[str]],
    system_message_list: Optional[List[str]] = None,
    generation_options_list: Optional[List[Union[Dict[str, Any], ApiGenerationOptions]]] = None,
    options_list: Optional[List[List[str]]] = None,
    tools_list: Optional[List[List[Dict[str, Any]]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    evaluation_method: Optional[str] = None,
    semaphore_size: Optional[int] = None,
    timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    wait_between_retry: Optional[Union[int, float]] = None,
) -> List[ApiInferenceOutput]:
    if (
        not isinstance(semaphore_size, int)
        or semaphore_size < 1
    ):
        semaphore_size = NUM_MAX_COROUTINES
    semaphore_size = min(semaphore_size, NUM_MAX_COROUTINES)
    if not isinstance(timeout, (int, float)):
        timeout = TIMEOUT
    if not isinstance(max_retry, int):
        max_retry = MAX_RETRY
    if not isinstance(wait_between_retry, (int, float)):
        wait_between_retry = WAIT_BETWEEN_RETRY
    
    api_group = get_api_group(api_name=api_name)
    _completion_async = None
    if api_group == ApiGroup.openai:
        _completion_async = completion_async_openai
    elif api_group == ApiGroup.anthropic:
        _completion_async = completion_async_anthropic
    elif api_group == ApiGroup.google:
        _completion_async = completion_async_google
    else:
        raise ValueError(f'unsupported api_name: {api_name}')
    
    if (
        isinstance(generation_options_list, (list, tuple))
        and len(generation_options_list) == len(prompt_list)
    ):
        generation_options_list = [
            _generation_options.to_dict(template=api_group)
            if isinstance(_generation_options, ApiGenerationOptions) else _generation_options
            for _generation_options in generation_options_list
        ]
    else:
        generation_options_list = [None, ] * len(prompt_list)

    if (
        isinstance(tools_list, (list, tuple))
        and len(tools_list) == len(prompt_list)
    ):
        pass
    else:
        tools_list = [None, ] * len(prompt_list)

    semaphore = asyncio.Semaphore(semaphore_size)
    
    output = None
    if evaluation_method == EvaluationMethod.perplexity:
        raise AssertionError(f'evaluation_method `perplexity` not supported for API')
    
    else:
        responses = None
        async with get_client(api_name=api_name, do_async=True) as client:
            responses = await tqdm_asyncio.gather(*[
                _completion_async(
                    client=client,
                    api_name=api_name, 
                    prompt=_prompt,
                    system_message=_system_message,
                    generation_options=_generation_options,
                    tools=_tools,
                    response_format=response_format,
                    semaphore=semaphore, 
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                ) # , await asyncio.sleep(SEC_BETWEEN_RETRY)]
                for _idx, (_prompt, _generation_options, _tools, _system_message) in enumerate(zip(
                    prompt_list, generation_options_list, tools_list, system_message_list,
                ))
            ], initial=0, total=len(prompt_list))
        
        output = list()
        for _response in responses:
            if isinstance(_response, dict):
                _output = ApiInferenceOutput(
                    prediction=_response.get("prediction", None),
                    reasoning_content=_response.get("reasoning_content", None),
                    generated_text=_response.get("generated_text", None),
                    finish_reason=_response.get("finish_reason", None),
                    latency=_response.get("latency", None),
                )
            else:
                _output = ApiInferenceOutput()
            output.append(_output)
    return output