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
from omni_evaluator.api.openai.chat_completions import (
    chat_completion_sync as chat_completion_sync_openai,
    chat_completion_async as chat_completion_async_openai,
)
from omni_evaluator.api.openai.responses import (
    response_create_sync as response_create_sync_openai,
    response_create_async as response_create_async_openai,
)
from omni_evaluator.api.anthropic.chat_completions import (
    chat_completion_sync as chat_completion_sync_anthropic,
    chat_completion_async as chat_completion_async_anthropic,
)
from omni_evaluator.api.google.chat_completions import (
    chat_completion_sync as chat_completion_sync_google,
    chat_completion_async as chat_completion_async_google,
)
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.generation_options import GenerationOptions
from omni_evaluator.schemas.inference import ApiInferenceOutput, InferenceOutput, Record

import logging
logger = logging.getLogger(__name__)


def _merge_response_into_record(record: Record, response: Dict[str, Any]) -> None:
    # Wrap raw response dict in ApiInferenceOutput (type-safe + parity with
    # huggingface/vllm/sglang engines), then merge onto record.
    _inference_output = ApiInferenceOutput(
        prediction=response.get("prediction"),
        reasoning_content=response.get("reasoning_content"),
        generated_text=response.get("generated_text"),
        finish_reason=response.get("finish_reason"),
        tool_calls=response.get("tool_calls"),
        function_call=response.get("function_call"),
        annotations=response.get("annotations"),
        perplexities=response.get("perplexities"),
        error_message=response.get("error_message"),
        latency=response.get("latency"),
    )
    record.merge_inference_output(_inference_output)


def chat_completion_sync(
    api_name: str,
    messages: Optional[List[Union[Dict[str, Any], ChatMessage]]] = None,
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    reasoning_options=None,
    record: Optional[Record] = None,
    verbose: bool = False,
    timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    wait_between_retry: Optional[Union[int, float]] = None,
    return_dict: Optional[bool] = False,
) -> Union[Record, ApiInferenceOutput, Optional[str]]:
    """Send a synchronous chat completion request to an external API.

    Accepts either `messages` directly OR `record` (Record). A Record is
    materialized at the entrance so the body always works against `record.*`.

    Returns:
        - If `record` was given by the caller: returns the same `record`
          (in-place mutated with prediction/reasoning_content/etc.).
        - Else if `return_dict` is True: returns an :class:`ApiInferenceOutput`.
        - Else: returns the prediction string (or None on failure).
    """
    _record_arg = record   # the original record reference given by the caller (or None)
    if record is None:
        record = Record(
            messages=messages,
            generation_options=generation_options,
            tools=tools,
        )
    if record.messages is None:
        raise ValueError('chat_completion_sync: provide `messages` or `record.messages`')
    if not isinstance(timeout, (int, float)):
        timeout = TIMEOUT
    if not isinstance(max_retry, int):
        max_retry = MAX_RETRY
    if not isinstance(wait_between_retry, (int, float)):
        wait_between_retry = WAIT_BETWEEN_RETRY

    _chat_completion_sync = None
    api_group = get_api_group(api_name=api_name)
    if api_group == ApiGroup.openai:
        if (
            response_format
            or (isinstance(record.generation_options, dict) and record.generation_options.get("reasoning"))
            or (reasoning_options and reasoning_options.get("reasoning_effort"))
        ):
            _chat_completion_sync = response_create_sync_openai
        else:
            _chat_completion_sync = chat_completion_sync_openai
    elif api_group == ApiGroup.anthropic:
        _chat_completion_sync = chat_completion_sync_anthropic
    elif api_group == ApiGroup.google:
        _chat_completion_sync = chat_completion_sync_google
    else:
        raise ValueError(f'Unsupported api_group: {api_name}')

    client = get_client(api_name=api_name, do_async=False)
    response = _chat_completion_sync(
        client=client,
        api_name=api_name,
        messages=list(record.messages),
        generation_options=record.generation_options,
        tools=record.tools,
        response_format=response_format,
        reasoning_options=reasoning_options,
        timeout=timeout,
        max_retry=max_retry,
        wait_between_retry=wait_between_retry,
    )
    if hasattr(client, "close"):
        client.close()

    if isinstance(response, dict):
        _merge_response_into_record(record=record, response=response)
        if verbose:
            record.verbose(prefix="\t")
        if _record_arg is not None:
            return record
        if return_dict:
            return ApiInferenceOutput(
                prediction=response.get("prediction", None),
                reasoning_content=response.get("reasoning_content", None),
                generated_text=response.get("generated_text", None),
                tool_calls=response.get("tool_calls", None),
                function_call=response.get("function_call", None),
                annotations=response.get("annotations", list()),
                finish_reason=response.get("finish_reason", None),
                latency=response.get("latency", None),
            )
        return response.get("prediction", None)
    else:
        # unified fail path: removed the record-aware branch. Always ApiInferenceOutput / None
        # (the return_dict branch is kept for legacy caller compatibility).
        return ApiInferenceOutput(error_message='API call failed (non-dict response)') if return_dict else None


async def chat_completion_async(
    api_name: str,
    messages: Optional[List[Union[Dict[str, Any], ChatMessage]]] = None,
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    reasoning_options=None,
    record: Optional[Record] = None,
    verbose: bool = False,
    semaphore: Optional[asyncio.locks.Semaphore] = None,
    timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    wait_between_retry: Optional[Union[int, float]] = None,
    return_dict: Optional[bool] = False,
) -> Union[Record, ApiInferenceOutput, Optional[str]]:
    """Async counterpart of :func:`chat_completion_sync`."""
    _record_arg = record   # the original record reference given by the caller (or None)
    if record is None:
        record = Record(
            messages=messages,
            generation_options=generation_options,
            tools=tools,
        )
    if record.messages is None:
        raise ValueError('chat_completion_async: provide `messages` or `record.messages`')
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    if not isinstance(timeout, (int, float)):
        timeout = TIMEOUT
    if not isinstance(max_retry, int):
        max_retry = MAX_RETRY
    if not isinstance(wait_between_retry, (int, float)):
        wait_between_retry = WAIT_BETWEEN_RETRY

    _chat_completion_async = None
    api_group = get_api_group(api_name=api_name)
    if api_group == ApiGroup.openai:
        # Responses API requires a Pydantic BaseModel (calls .model_json_schema()).
        # Plain-dict response_format (e.g. {"type": "json_object"}) must go through
        # legacy chat.completions which accepts the raw dict.
        if (
            (
                response_format is not None 
                and hasattr(response_format, "model_json_schema")
            )
            or (
                isinstance(record.generation_options, dict) 
                and record.generation_options.get("reasoning")
            )
            or (
                reasoning_options 
                and reasoning_options.get("reasoning_effort")
            )
        ):
            _chat_completion_async = response_create_async_openai
        else:
            _chat_completion_async = chat_completion_async_openai
    elif api_group == ApiGroup.anthropic:
        _chat_completion_async = chat_completion_async_anthropic
    elif api_group == ApiGroup.google:
        _chat_completion_async = chat_completion_async_google
    else:
        raise ValueError(f'Unsupported api_group: {api_name}')

    response = None
    async with get_client(api_name=api_name, do_async=True) as client:
        response = await _chat_completion_async(
            client=client,
            api_name=api_name,
            messages=list(record.messages),
            generation_options=record.generation_options,
            tools=record.tools,
            response_format=response_format,
            reasoning_options=reasoning_options,
            semaphore=semaphore,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
        )

    if isinstance(response, dict):
        _merge_response_into_record(record=record, response=response)
        if verbose:
            record.verbose(prefix="\t")
        if _record_arg is not None:
            return record
        if return_dict:
            return ApiInferenceOutput(
                prediction=response.get("prediction", None),
                reasoning_content=response.get("reasoning_content", None),
                generated_text=response.get("generated_text", None),
                tool_calls=response.get("tool_calls", None),
                function_call=response.get("function_call", None),
                annotations=response.get("annotations", list()),
                finish_reason=response.get("finish_reason", None),
                latency=response.get("latency", None),
            )
        return response.get("prediction", None)
    else:
        # unified fail path: removed the record-aware branch. Always ApiInferenceOutput / None
        # (the return_dict branch is kept for legacy caller compatibility).
        return ApiInferenceOutput(error_message='API call failed (non-dict response)') if return_dict else None


def batch_chat_completion_sync(
    api_name: str,
    messages_list: Optional[List[List[Union[Dict[str, Any], ChatMessage]]]] = None,
    generation_options_list: Optional[List[Dict[str, Any]]] = None,
    options_list: Optional[List[List[str]]] = None,
    tools_list: Optional[List[List[Dict[str, Any]]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    reasoning_options=None,
    evaluation_method: Optional[str] = None,
    records: Optional[List[Optional[Record]]] = None,
    verbose: bool = False,
    timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    wait_between_retry: Optional[Union[int, float]] = None,
) -> Union[List[Optional[Record]], List[ApiInferenceOutput]]:
    """Run synchronous chat completions for a batch of message lists.

    If `records` is given, iterate over it and pass each Record to
    chat_completion_sync — returns List[Record] (in-place mutated).
    Otherwise iterate over per-request lists — returns List[ApiInferenceOutput].
    """
    if records is None and messages_list is None:
        raise ValueError('batch_chat_completion_sync: provide either `messages_list` or `records`')
    if not isinstance(timeout, (int, float)):
        timeout = TIMEOUT
    if not isinstance(max_retry, int):
        max_retry = MAX_RETRY
    if not isinstance(wait_between_retry, (int, float)):
        wait_between_retry = WAIT_BETWEEN_RETRY

    if evaluation_method == EvaluationMethod.perplexity:
        raise AssertionError(f'Evaluation method `perplexity` not supported for API')

    outputs = list()
    if records is not None:
        for _record_idx, _record in tqdm(
            enumerate(records), initial=0, total=len(records),
        ):
            _result = chat_completion_sync(
                api_name=api_name,
                record=_record,
                response_format=response_format,
                reasoning_options=reasoning_options,
                verbose=verbose,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
            )
            if isinstance(_result, InferenceOutput):
                _record.merge_inference_output(_result)
            outputs.append(_record)
        return outputs

    if not (
        isinstance(generation_options_list, (list, tuple))
        and len(generation_options_list) == len(messages_list)
    ):
        generation_options_list = [None] * len(messages_list)
    if not (
        isinstance(tools_list, (list, tuple))
        and len(tools_list) == len(messages_list)
    ):
        tools_list = [None] * len(messages_list)

    for message_idx, (_messages, _generation_options, _tools) in tqdm(
        enumerate(zip(messages_list, generation_options_list, tools_list)),
        initial=0,
        total=len(messages_list),
    ):
        outputs.append(chat_completion_sync(
            api_name=api_name,
            messages=_messages,
            generation_options=_generation_options,
            tools=_tools,
            response_format=response_format,
            reasoning_options=reasoning_options,
            verbose=verbose,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
            return_dict=True,
        ))
    return outputs


async def batch_chat_completion_async(
    api_name: str,
    messages_list: Optional[List[List[Union[Dict[str, Any], ChatMessage]]]] = None,
    generation_options_list: Optional[List[Dict[str, Any]]] = None,
    options_list: Optional[List[List[str]]] = None,
    tools_list: Optional[List[List[Dict[str, Any]]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    reasoning_options=None,
    evaluation_method: Optional[str] = None,
    records: Optional[List[Optional[Record]]] = None,
    verbose: bool = False,
    semaphore_size: Optional[int] = None,
    timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    wait_between_retry: Optional[Union[int, float]] = None,
) -> Union[List[Optional[Record]], List[ApiInferenceOutput]]:
    """Async counterpart of :func:`batch_chat_completion_sync`."""
    if records is None and messages_list is None:
        raise ValueError('batch_chat_completion_async: provide either `messages_list` or `records`')
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

    if evaluation_method == EvaluationMethod.perplexity:
        raise AssertionError(f'Evaluation method `perplexity` not supported for API')

    semaphore = asyncio.Semaphore(semaphore_size)

    if records is not None:
        _results = await tqdm_asyncio.gather(*[
            chat_completion_async(
                api_name=api_name,
                record=_record,
                response_format=response_format,
                reasoning_options=reasoning_options,
                verbose=verbose,
                semaphore=semaphore,
                timeout=timeout,
                max_retry=max_retry,
                wait_between_retry=wait_between_retry,
            )
            for _record in records
        ], initial=0, total=len(records))
        outputs = list()
        for _record, _result in zip(records, _results):
            if isinstance(_result, InferenceOutput):
                _record.merge_inference_output(_result)
            outputs.append(_record)
        return outputs

    if not (
        isinstance(generation_options_list, (list, tuple))
        and len(generation_options_list) == len(messages_list)
    ):
        generation_options_list = [None] * len(messages_list)
    if not (
        isinstance(tools_list, (list, tuple))
        and len(tools_list) == len(messages_list)
    ):
        tools_list = [None] * len(messages_list)
    outputs = await tqdm_asyncio.gather(*[
        chat_completion_async(
            api_name=api_name,
            messages=_messages,
            generation_options=_generation_options,
            tools=_tools,
            response_format=response_format,
            reasoning_options=reasoning_options,
            verbose=verbose,
            semaphore=semaphore,
            timeout=timeout,
            max_retry=max_retry,
            wait_between_retry=wait_between_retry,
            return_dict=True,
        )
        for _messages, _generation_options, _tools in zip(
            messages_list, generation_options_list, tools_list,
        )
    ], initial=0, total=len(messages_list))
    return list(outputs)
