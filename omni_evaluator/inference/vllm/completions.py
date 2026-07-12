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
import copy
import httpx
import logging
import numpy as np
from omegaconf import ListConfig, DictConfig
import os
import time
import traceback
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, Iterable
from urllib.parse import urljoin
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger(__name__)

from omni_evaluator import EvaluationMethod
from omni_evaluator.inference import (
    TIMEOUT, SOCKET_TIMEOUT, MAX_RETRY, WAIT_BETWEEN_RETRY,
    NUM_MAX_COROUTINES,
)
from omni_evaluator.postprocess import parse_think
from omni_evaluator.inference.vllm.chat_completions import (
    _force_logprobs_options,
    _nll_from_prompt_logprobs,
)
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.generation_options import GenerationOptions, VllmGenerationOptions
from omni_evaluator.schemas.inference import InferenceOutput, Record, VllmInferenceOutput
from omni_evaluator.utils.response import summarize_payload_shape
from omni_evaluator.utils.common import remove_stop_words


def _resolve_generation_options(record: Record) -> Dict[str, Any]:
    # Normalize record.generation_options to a plain dict regardless of input form.
    if isinstance(record.generation_options, GenerationOptions):
        return record.generation_options.to_dict()
    if isinstance(record.generation_options, dict):
        return dict(record.generation_options)
    return dict()


def _resolve_prompt(record: Record, tokenizer: Optional[Any]) -> Optional[str]:
    # Derive the prompt string from `record`. Order:
    #   1. record.prompt if explicitly provided
    #   2. tokenizer.apply_chat_template over record.messages (if tokenizer is given)
    #   3. concatenated ChatMessage.get_prompt across record.messages
    if record.prompt is not None:
        return record.prompt
    if record.messages is None:
        return None
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            [_m.to_dict(template="openai") for _m in record.messages],
            tokenize=False,
        )
    return "\n\n".join(
        ChatMessage.get_prompt(message=_m) for _m in record.messages
    )


def completion_sync(
    url: str,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    generation_options: Optional[VllmGenerationOptions] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    reasoning: Optional[Union[str, bool]] = False,
    api_version: Optional[str] = None,
    record: Optional[Record] = None,
    tokenizer: Optional[Any] = None,
    verbose: bool = False,
    timeout: Optional[int] = TIMEOUT,
    max_retry: Optional[int] = MAX_RETRY,
    wait_between_retry: Optional[int] = WAIT_BETWEEN_RETRY,
) -> Optional[Record]:
    # Send synchronous text completion request to vLLM server with retry logic.
    # Accepts either explicit `prompt` (+ generation_options/tools) OR a `record`.
    # On entry, a Record is materialized so the body always works against `record.*`.
    # The response fields are written back onto `record` in-place. Returns the same
    # `record` (or None on failure).
    if record is None:
        record = Record(
            prompt=prompt,
            generation_options=generation_options,
            tools=tools,
        )
    _prompt = _resolve_prompt(record=record, tokenizer=tokenizer)
    if _prompt is None:
        raise ValueError('completion_sync: provide `prompt` or `record.prompt` / `record.messages`')

    _generation_options = _resolve_generation_options(record=record)

    headers = {
        "Content-Type": "application/json",
    }
    if os.getenv("VLLM_API_KEY", None):
        headers["Authorization"] = f'Bearer {os.environ["VLLM_API_KEY"]}'

    payload = {
        "prompt": _prompt,
        "tools": record.tools,
        **_generation_options,
    }
    if model_name:
        payload["model"] = model_name

    if api_version:
        url = urljoin(url, api_version)

    for cur_try in range(1, max_retry+1):
        try:
            response, latency = None, None
            _start_time = time.time()
            with httpx.Client(
                headers=headers,
                timeout=timeout,
                http2=True,
            ) as client:
                response = client.post(
                    f'{url}/completions',
                    json=payload,
                    timeout=timeout,
                )
            latency = time.time() - _start_time
            if response.status_code == 200:
                response = response.json()
                if isinstance(response, (dict, DictConfig)):
                    response["latency"] = latency
                    response.update(parse_response(
                        response=response,
                        reasoning=reasoning,
                        stop_words=_generation_options.get("stop", None),
                    ))
                    if response["generated_text"] is None:
                        logger.warning(f'Received null output from vLLM: {response}')
                        if (
                            "is not a multimodal model" in response.get("error_message", "")
                            or "is not a multimodal model" in response.get("error", dict()).get("message", "")
                        ):
                            pass
                        else:
                            raise AssertionError(f'unexpected inference error while vLLM: {response}')
                    _inference_output = VllmInferenceOutput(
                        prediction=response.get("prediction"),
                        reasoning_content=response.get("reasoning_content"),
                        generated_text=response.get("generated_text"),
                        finish_reason=response.get("finish_reason"),
                        tool_calls=response.get("tool_calls"),
                        function_call=response.get("function_call"),
                        annotations=response.get("annotations"),
                        perplexities=response.get("perplexities"),
                        prompt_logprobs=response.get("prompt_logprobs"),
                        error_message=response.get("error_message"),
                        latency=response.get("latency"),
                    )
                    record.merge_inference_output(_inference_output)
                    if verbose:
                        record.verbose(prefix="\t")
                    return record
            else:
                logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request vllm: {response.content}')
                time.sleep(wait_between_retry)
                continue

        except Exception as ex:
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to parse vllm output')
            traceback.print_exc()
            time.sleep(wait_between_retry)
            continue

    logger.error(f'Failed after {max_retry} tries')
    # unified fail path: always return InferenceOutput (batch_*_completion does the record merge).
    return VllmInferenceOutput(error_message=f'Failed after {max_retry} tries')


async def completion_async(
    url: str,
    prompt: Optional[str] = None,
    model_name: Optional[str] = None,
    generation_options: Optional[VllmGenerationOptions] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    reasoning: Optional[Union[str, bool]] = False,
    api_version: Optional[str] = None,
    record: Optional[Record] = None,
    tokenizer: Optional[Any] = None,
    verbose: bool = False,
    semaphore: Optional[asyncio.locks.Semaphore] = None,
    timeout: Optional[int] = TIMEOUT,
    socket_timeout: Optional[int] = SOCKET_TIMEOUT,
    max_retry: Optional[int] = MAX_RETRY,
    wait_between_retry: Optional[int] = WAIT_BETWEEN_RETRY,
) -> Optional[Record]:
    # Send async text completion request to vLLM server with semaphore-based concurrency control.
    # Same record-entry contract as completion_sync.
    if record is None:
        record = Record(
            prompt=prompt,
            generation_options=generation_options,
            tools=tools,
        )
    _prompt = _resolve_prompt(record=record, tokenizer=tokenizer)
    if _prompt is None:
        raise ValueError('completion_async: provide `prompt` or `record.prompt` / `record.messages`')
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)

    _generation_options = _resolve_generation_options(record=record)

    headers = {
        "Content-Type": "application/json",
    }
    if os.getenv("VLLM_API_KEY", None):
        headers["Authorization"] = f'Bearer {os.environ["VLLM_API_KEY"]}'

    payload = {
        "prompt": _prompt,
        "tools": record.tools,
        **_generation_options,
    }
    if model_name:
        payload["model"] = model_name

    if api_version:
        url = urljoin(url, api_version)

    for cur_try in range(1, max_retry+1):
        try:
            response, latency = None, None
            async with semaphore:
                _start_time = time.time()
                async with aiohttp.ClientSession(
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout, sock_read=socket_timeout),
                ) as session:
                    async with session.post(
                        f'{url}/completions',
                        json=payload,
                    ) as response:
                        if response.status == 200:
                            response = await response.json(content_type=None)
                        else:
                            _text = await response.text()
                            if 400 <= response.status < 500:
                                summarize_payload_shape(
                                    payload, status=response.status, logger=logger,
                                    messages_key=None,
                                )
                            raise Exception(f'HTTP {response.status}: {_text[:500]}')
                latency = time.time() - _start_time
            if isinstance(response, (dict, DictConfig)):
                response["latency"] = latency
                response.update(parse_response(
                    response=response,
                    reasoning=reasoning,
                    stop_words=_generation_options.get("stop", None),
                ))
                if response["generated_text"] is None:
                    logger.warning(f'Received null output from vLLM: {response}')
                    if (
                        "is not a multimodal model" in response.get("error_message", "")
                        or "is not a multimodal model" in response.get("error", dict()).get("message", "")
                    ):
                        pass
                    else:
                        raise AssertionError(f'unexpected inference error while vLLM: {response}')
                _inference_output = VllmInferenceOutput(
                    prediction=response.get("prediction"),
                    reasoning_content=response.get("reasoning_content"),
                    generated_text=response.get("generated_text"),
                    finish_reason=response.get("finish_reason"),
                    tool_calls=response.get("tool_calls"),
                    function_call=response.get("function_call"),
                    annotations=response.get("annotations"),
                    perplexities=response.get("perplexities"),
                    prompt_logprobs=response.get("prompt_logprobs"),
                    error_message=response.get("error_message"),
                    latency=response.get("latency"),
                )
                record.merge_inference_output(_inference_output)
                if verbose:
                    record.verbose(prefix="\t")
                return record
            else:
                logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request vllm: {response.content}')
                await asyncio.sleep(wait_between_retry)
                continue

        except Exception as ex:
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to parse vllm output')
            traceback.print_exc()
            await asyncio.sleep(wait_between_retry)
            continue

    logger.error(f'Failed after {max_retry} tries')
    # unified fail path: always return InferenceOutput (batch_*_completion does the record merge).
    return VllmInferenceOutput(error_message=f'Failed after {max_retry} tries')


def batch_completion_sync(
    url: str,
    prompts: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    generation_options_list: Optional[List[Optional[VllmGenerationOptions]]] = None,
    options_list: Optional[List[List[str]]] = None,
    tools_list: Optional[List[List[Dict[str, Any]]]] = None,
    evaluation_method: Optional[str] = None,
    reasoning: Optional[Union[str, bool]] = False,
    api_version: Optional[str] = None,
    records: Optional[List[Optional[Record]]] = None,
    tokenizer: Optional[Any] = None,
    verbose: bool = False,
    timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    wait_between_retry: Optional[Union[int, float]] = None,
) -> Optional[List[Optional[Record]]]:
    # Run synchronous text completions for a batch of prompts sequentially.
    # If `records` is given, iterate over it and pass each Record to completion_sync.
    # Otherwise iterate over `prompts` (legacy path).
    # Returns: list of Record objects (or None entries on per-request failure)
    if records is None and prompts is None:
        raise ValueError('batch_completion_sync: provide either `prompts` or `records`')
    if not isinstance(timeout, (int, float)):
        timeout = TIMEOUT
    if not isinstance(max_retry, int):
        max_retry = MAX_RETRY
    if not isinstance(wait_between_retry, (int, float)):
        wait_between_retry = WAIT_BETWEEN_RETRY

    outputs = None
    if evaluation_method == EvaluationMethod.perplexity:
        outputs = list()
        if records is not None:
            for _record_idx, _record in tqdm(
                enumerate(records), initial=0, total=len(records),
            ):
                _options = _record.options if _record.options else _record.option_contents
                if not _options:
                    outputs.append(_record)
                    continue
                _force_logprobs_options(generation_options=_record.generation_options)
                _base_prompt = _record.prompt
                if _base_prompt is None and _record.messages is not None:
                    _base_prompt = "\n\n".join(
                        ChatMessage.get_prompt(message=_m) for _m in _record.messages
                    )
                _perplexities = list()
                _latency = 0.0
                for _option in _options:
                    _result = completion_sync(
                        url=url,
                        prompt=f'{_base_prompt} {_option}',
                        generation_options=_record.generation_options,
                        model_name=model_name,
                        reasoning=reasoning,
                        api_version=api_version,
                        timeout=timeout,
                        max_retry=max_retry,
                        wait_between_retry=wait_between_retry,
                    )
                    if _result is None or _result.prompt_logprobs is None:
                        continue
                    _perplexities.append(_nll_from_prompt_logprobs(prompt_logprobs=_result.prompt_logprobs, option=_option))
                    _latency += _result.latency or 0
                _record.prediction = _options[int(np.argmin(_perplexities))]
                _record.perplexities = _perplexities
                _record.latency = _latency
                if verbose:
                    _record.verbose(prefix="\t")
                outputs.append(_record)
        else:
            for _idx in range(0, len(generation_options_list)):
                generation_options_list[_idx].max_tokens = 1
                generation_options_list[_idx].logprobs = 1
                generation_options_list[_idx].prompt_logprobs = True
            for prompt_idx, (prompt, generation_options, options) in tqdm(enumerate(
                zip(prompts, generation_options_list, options_list)
            ), initial=0, total=len(prompts)):
                _perplexities = list()
                _latency = 0.0
                for _option in options:
                    _result = completion_sync(
                        url=url,
                        prompt=f'{prompt} {_option}',
                        generation_options=generation_options,
                        reasoning=reasoning,
                        api_version=api_version,
                        timeout=timeout,
                        max_retry=max_retry,
                        wait_between_retry=wait_between_retry,
                    )
                    if _result is None or _result.prompt_logprobs is None:
                        continue
                    _perplexities.append(_nll_from_prompt_logprobs(prompt_logprobs=_result.prompt_logprobs, option=_option))
                    _latency += _result.latency or 0
                outputs.append(Record(
                    prompt=prompt,
                    options=options,
                    generation_options=generation_options,
                    prediction=options[int(np.argmin(_perplexities))],
                    perplexities=_perplexities,
                    latency=_latency,
                ))

    else:
        outputs = list()
        if records is not None:
            for _record_idx, _record in tqdm(
                enumerate(records), initial=0, total=len(records),
            ):
                _result = completion_sync(
                    url=url,
                    record=_record,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    reasoning=reasoning,
                    api_version=api_version,
                    verbose=verbose,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                )
                if isinstance(_result, InferenceOutput):
                    _record.merge_inference_output(_result)
                outputs.append(_record)
        else:
            if (
                isinstance(generation_options_list, (list, tuple))
                and len(generation_options_list) == len(prompts)
            ):
                pass
            else:
                generation_options_list = [None, ] * len(prompts)
            if (
                isinstance(tools_list, (list, tuple))
                and len(tools_list) == len(prompts)
            ):
                pass
            else:
                tools_list = [None, ] * len(prompts)
            for prompt_idx, (prompt, generation_options, tools) in tqdm(
                enumerate(zip(prompts, generation_options_list, tools_list)),
                initial=0,
                total=len(prompts),
            ):
                outputs.append(completion_sync(
                    url=url,
                    prompt=prompt,
                    model_name=model_name,
                    generation_options=generation_options,
                    tools=tools,
                    reasoning=reasoning,
                    api_version=api_version,
                    verbose=verbose,
                    timeout=timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                ))
    return outputs


async def batch_completion_async(
    url: str,
    prompts: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    generation_options_list: Optional[List[Optional[VllmGenerationOptions]]] = None,
    options_list: Optional[List[List[str]]] = None,
    tools_list: Optional[List[List[Dict[str, Any]]]] = None,
    evaluation_method: Optional[str] = None,
    reasoning: Optional[Union[str, bool]] = False,
    api_version: Optional[str] = None,
    records: Optional[List[Optional[Record]]] = None,
    tokenizer: Optional[Any] = None,
    verbose: bool = False,
    semaphore_size: Optional[int] = None,
    timeout: Optional[Union[int, float]] = None,
    socket_timeout: Optional[Union[int, float]] = None,
    max_retry: Optional[int] = None,
    wait_between_retry: Optional[Union[int, float]] = None,
) -> Optional[List[Optional[Record]]]:
    # Run async text completions for a batch of prompts with concurrency control.
    if records is None and prompts is None:
        raise ValueError('batch_completion_async: provide either `prompts` or `records`')
    if (
        not isinstance(semaphore_size, int)
        or semaphore_size < 1
    ):
        semaphore_size = NUM_MAX_COROUTINES
    semaphore_size = min(semaphore_size, NUM_MAX_COROUTINES)
    if not isinstance(timeout, (int, float)):
        timeout = TIMEOUT
    if not isinstance(socket_timeout, (int, float)):
        socket_timeout = SOCKET_TIMEOUT
    if not isinstance(max_retry, int):
        max_retry = MAX_RETRY
    if not isinstance(wait_between_retry, (int, float)):
        wait_between_retry = WAIT_BETWEEN_RETRY

    semaphore = asyncio.Semaphore(semaphore_size)

    outputs = None
    if evaluation_method == EvaluationMethod.perplexity:
        outputs = list()
        if records is not None:
            # Pre-compute the per-record base prompt + option list once and
            # flatten coroutines so all (record × option) requests share the
            # same semaphore.
            _base_prompts: List[Optional[str]] = list()
            _options_per_record: List[Optional[List[str]]] = list()
            _coros = list()
            for _record in records:
                _options = _record.options if _record.options else _record.option_contents
                _options_per_record.append(_options)
                if not _options:
                    _base_prompts.append(None)
                    continue
                _force_logprobs_options(generation_options=_record.generation_options)
                _base_prompt = _record.prompt
                if _base_prompt is None and _record.messages is not None:
                    _base_prompt = "\n\n".join(
                        ChatMessage.get_prompt(message=_m) for _m in _record.messages
                    )
                _base_prompts.append(_base_prompt)
                for _option in _options:
                    _coros.append(completion_async(
                        url=url,
                        prompt=f'{_base_prompt} {_option}',
                        model_name=model_name,
                        generation_options=_record.generation_options,
                        reasoning=reasoning,
                        api_version=api_version,
                        semaphore=semaphore,
                        timeout=timeout,
                        socket_timeout=socket_timeout,
                        max_retry=max_retry,
                        wait_between_retry=wait_between_retry,
                    ))
            _results = await tqdm_asyncio.gather(*_coros, initial=0, total=len(_coros))
            _cursor = 0
            for _record, _options in zip(records, _options_per_record):
                if not _options:
                    outputs.append(_record)
                    continue
                _slice = _results[_cursor:_cursor + len(_options)]
                _cursor += len(_options)
                _perplexities = list()
                _latency = 0.0
                for _option, _result in zip(_options, _slice):
                    if _result is None or _result.prompt_logprobs is None:
                        continue
                    _perplexities.append(_nll_from_prompt_logprobs(prompt_logprobs=_result.prompt_logprobs, option=_option))
                    _latency += _result.latency or 0
                _record.prediction = _options[int(np.argmin(_perplexities))]
                _record.perplexities = _perplexities
                _record.latency = _latency
                if verbose:
                    _record.verbose(prefix="\t")
                outputs.append(_record)
        else:
            for _idx in range(0, len(generation_options_list)):
                generation_options_list[_idx].max_tokens = 1
                generation_options_list[_idx].logprobs = 1
                generation_options_list[_idx].prompt_logprobs = True
            _coros = list()
            for prompt_idx, (prompt, generation_options, options) in enumerate(
                zip(prompts, generation_options_list, options_list)
            ):
                for _option in options:
                    _coros.append(completion_async(
                        url=url,
                        prompt=f'{prompt} {_option}',
                        model_name=model_name,
                        generation_options=generation_options,
                        reasoning=reasoning,
                        api_version=api_version,
                        semaphore=semaphore,
                        timeout=timeout,
                        socket_timeout=socket_timeout,
                        max_retry=max_retry,
                        wait_between_retry=wait_between_retry,
                    ))
            _results = await tqdm_asyncio.gather(*_coros, initial=0, total=len(_coros))
            _cursor = 0
            for prompt_idx, (prompt, generation_options, options) in enumerate(
                zip(prompts, generation_options_list, options_list)
            ):
                _slice = _results[_cursor:_cursor + len(options)]
                _cursor += len(options)
                _perplexities = list()
                _latency = 0.0
                for _option, _result in zip(options, _slice):
                    if _result is None or _result.prompt_logprobs is None:
                        continue
                    _perplexities.append(_nll_from_prompt_logprobs(prompt_logprobs=_result.prompt_logprobs, option=_option))
                    _latency += _result.latency or 0
                outputs.append(Record(
                    prompt=prompt,
                    options=options,
                    generation_options=generation_options,
                    prediction=options[int(np.argmin(_perplexities))],
                    perplexities=_perplexities,
                    latency=_latency,
                ))

    else:
        if records is not None:
            _results = await tqdm_asyncio.gather(*[
                completion_async(
                    url=url,
                    record=_record,
                    tokenizer=tokenizer,
                    reasoning=reasoning,
                    api_version=api_version,
                    verbose=verbose,
                    semaphore=semaphore,
                    timeout=timeout,
                    socket_timeout=socket_timeout,
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
        else:
            if (
                isinstance(generation_options_list, (list, tuple))
                and len(generation_options_list) == len(prompts)
            ):
                pass
            else:
                generation_options_list = [None, ] * len(prompts)
            if (
                isinstance(tools_list, (list, tuple))
                and len(tools_list) == len(prompts)
            ):
                pass
            else:
                tools_list = [None, ] * len(prompts)
            outputs = await tqdm_asyncio.gather(*[
                completion_async(
                    url=url,
                    prompt=prompt,
                    generation_options=generation_options,
                    tools=tools,
                    reasoning=reasoning,
                    api_version=api_version,
                    verbose=verbose,
                    semaphore=semaphore,
                    timeout=timeout,
                    socket_timeout=socket_timeout,
                    max_retry=max_retry,
                    wait_between_retry=wait_between_retry,
                )
                for prompt_idx, (prompt, generation_options, tools) in enumerate(
                    zip(prompts, generation_options_list, tools_list)
                )
            ], initial=0, total=len(prompts))
    return outputs


def parse_response(
    response: Dict[str, Any],
    reasoning: Optional[Union[str, bool]] = False,
    stop_words: Optional[List[str]] = None,
) -> Dict[str, Any]:
    # Parse vLLM text completion response into standardized fields.
    # Args: response - raw JSON response from vLLM, reasoning - pattern or flag for chain-of-thought extraction,
    #   stop_words - tokens to strip from generated text
    # Returns: dict with generated_text, prediction, reasoning_content, finish_reason, error_message, prompt_logprobs
    generated_text, prediction, reasoning_content = None, None, None
    finish_reason, error_message = None, None
    prompt_logprobs = None

    if not isinstance(response, (dict, DictConfig)):
        error_message = f'failed to request API'

    elif len(response.get("choices", list())) < 1:
        # 'choices not exist' indicates that request has failed
        # and usually when cvs_id is invalid (i.e. failed to upload image to cvs)
        _error = response.get("error", dict())
        error_message = f'code: {_error.get("code", 500)}'
        _error_message = None
        if _error and _error.get("message", None):
            _error_message = _error["message"]
        elif response.get("detail", None):
            _error_message = response["detail"]
        if _error_message:
            error_message += f' - {_error_message}'

    else:
        _last_choice = response["choices"][-1]
        finish_reason = _last_choice.get("finish_reason", None)
        generated_text = _last_choice.get("text", None)
        if isinstance(generated_text, str):
            generated_text = generated_text.strip()
        else:
            error_message = f'text not exist in choices'
        if _last_choice.get("reasoning", None):
            reasoning_content = _last_choice["reasoning"]
        elif _last_choice.get("reasoning_content", None):
            reasoning_content = _last_choice["reasoning_content"]
        prompt_logprobs = _last_choice.get("prompt_logprobs", None)

    prediction = generated_text
    if not generated_text:
        pass
    else:
        prediction = remove_stop_words(
            text=prediction,
            stop_words=stop_words,
        )
        if reasoning:
            _output = parse_think(
                prediction=prediction,
                think_end_pattern=reasoning,
                eot_token="<|im_end|>",
            )
            if _output["prediction"]:
                prediction = _output["prediction"]
            if (
                not reasoning_content
                and _output["reasoning_content"]
            ):
                reasoning_content = _output["reasoning_content"]

    return {
        "generated_text": generated_text,
        "prediction": prediction,
        "reasoning_content": reasoning_content,
        "finish_reason": finish_reason,
        "error_message": error_message,
        "prompt_logprobs": prompt_logprobs,
    }
