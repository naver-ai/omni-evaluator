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


def _build_response_format_text(
    response_format: Optional[Any],
) -> Optional[Dict[str, Any]]:
    """Coerce a caller-supplied ``response_format`` into the Responses-API
    ``text`` shape.

    Callers may pass either a ``pydantic.BaseModel`` (with ``model_json_schema``)
    or a Chat-Completions style dict (``{"type": "json_object"}`` /
    ``{"type": "json_schema", ...}``). Returns ``None`` when the input is
    falsy or yields no usable schema.

    Note: the Responses API expects ``text = {"format": {...}}`` (one extra
    level of nesting compared to Chat Completions' ``response_format``).
    Putting ``type`` directly under ``text`` raises
    ``BadRequestError: Unknown parameter: 'text.type'``.

    The schema is normalized in-place for OpenAI strict structured-outputs
    mode (``additionalProperties: false`` on every object, every property
    name listed in ``required``, recursively into ``properties``/``items``/
    ``$defs``/``anyOf``/``allOf``/``oneOf``). pydantic's ``model_json_schema``
    omits these by default, so without this OpenAI returns
    ``Invalid schema for response_format ... 'additionalProperties' is
    required to be supplied and to be false``.
    """
    def _enforce_strict(_schema):
        if not isinstance(_schema, dict):
            return
        if _schema.get("type") == "object":
            _props = _schema.get("properties") or {}
            _schema["additionalProperties"] = False
            _schema["required"] = list(_props.keys())
        for _key in ("properties", "$defs"):
            _child = _schema.get(_key)
            if isinstance(_child, dict):
                for _v in _child.values():
                    _enforce_strict(_v)
        _items = _schema.get("items")
        if isinstance(_items, dict):
            _enforce_strict(_items)
        elif isinstance(_items, list):
            for _v in _items:
                _enforce_strict(_v)
        for _key in ("anyOf", "allOf", "oneOf"):
            _arr = _schema.get(_key)
            if isinstance(_arr, list):
                for _v in _arr:
                    _enforce_strict(_v)

    if not response_format:
        return None
    if hasattr(response_format, "model_json_schema"):
        _schema = response_format.model_json_schema()
        _enforce_strict(_schema)
        _name = _schema.get("title") or getattr(response_format, "__name__", "response_format")
        return {"format": {
            "type": "json_schema",
            "name": _name,
            "schema": _schema,
            "strict": True,
        }}
    if isinstance(response_format, dict):
        _response_format_type = response_format.get("type")
        if _response_format_type == "json_object":
            return {"format": {"type": "json_object"}}
        if _response_format_type == "json_schema":
            # Chat-Completions structured outputs nest the schema one level
            # deep (``json_schema.schema``); the flat form puts it directly
            # under ``schema``. Accept both.
            _schema = response_format.get("schema")
            _name = response_format.get("name")
            if _schema is None:
                _inner_json_schema = response_format.get("json_schema")
                if isinstance(_inner_json_schema, dict):
                    _schema = _inner_json_schema.get("schema", _inner_json_schema)
                    _name = _name or _inner_json_schema.get("name")
            if _schema is not None:
                _enforce_strict(_schema)
                _format = {"type": "json_schema", "schema": _schema, "strict": True}
                if _name:
                    _format["name"] = _name
                else:
                    _format["name"] = _schema.get("title", "response_format")
                return {"format": _format}
        # Unknown shape — forward under `format` and let the SDK validate.
        return {"format": response_format}
    return None


def response_create_sync(
    client,
    api_name: str,
    messages: List[Union[Dict[str, Any], ChatMessage]],
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    reasoning_options=None,
    instructions: Optional[str] = None,
    previous_response_id: Optional[str] = None,
    store: Optional[bool] = None,
    timeout: Optional[int] = TIMEOUT,
    max_retry: Optional[int] = MAX_RETRY,
    wait_between_retry: Optional[int] = WAIT_BETWEEN_RETRY,
) -> Optional[Dict[str, Any]]:
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

    # Convert internal dict → OpenAI Responses format
    messages = [
        ChatMessage.to_template(obj=_message, template="openai_response")
        for _message in messages
    ]

    # build Responses API-specific kwargs
    kwargs = dict()
    _response_format_text = _build_response_format_text(response_format)
    if _response_format_text is not None:
        kwargs["text"] = _response_format_text
    if instructions is not None:
        kwargs["instructions"] = instructions
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id
    if store is not None:
        kwargs["store"] = store

    for cur_try in range(1, max_retry+1):
        response, latency = None, None
        try:
            _start_time = time.time()
            response = client.responses.create(
                model=api_name,
                input=messages,
                **generation_options,
                **kwargs,
                tools=tools if tools else list(),
                timeout=timeout,
            )
            latency = time.time() - _start_time
            response = json.loads(response.json())
            response["latency"] = latency
            response.update(parse_response(
                response,
                response_format=response_format,
                stop_words=generation_options.get("stop", None),
            ))
            if not is_valid_response(response):
                # content-contract failure — identical retry cannot help; surface once and stop.
                logger.error(
                    f'openai responses unusable, not retrying (model={api_name}); '
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
                    {'model': api_name, 'input': messages, 'tools': tools or [], **generation_options, **kwargs},
                    status=_status, logger=logger, messages_key='input',
                )
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request openai responses: {response}')
            traceback.print_exc()
            time.sleep(wait_between_retry * cur_try)
            continue

    logger.error(f'Failed after {max_retry} tries')
    return None


async def response_create_async(
    client,
    api_name: str,
    messages: List[Union[Dict[str, Any], ChatMessage]],
    generation_options: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[pydantic.BaseModel] = None,
    reasoning_options=None,
    instructions: Optional[str] = None,
    previous_response_id: Optional[str] = None,
    store: Optional[bool] = None,
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

    # Convert internal dict → OpenAI Responses format
    messages = [
        ChatMessage.to_template(obj=_message, template="openai_response")
        for _message in messages
    ]

    # build Responses API-specific kwargs
    kwargs = dict()
    _response_format_text = _build_response_format_text(response_format)
    if _response_format_text is not None:
        kwargs["text"] = _response_format_text
    if instructions is not None:
        kwargs["instructions"] = instructions
    if previous_response_id is not None:
        kwargs["previous_response_id"] = previous_response_id
    if store is not None:
        kwargs["store"] = store

    for cur_try in range(1, max_retry+1):
        response, latency = None, None
        try:
            async with semaphore:
                _start_time = time.time()
                logger.debug(
                    "responses.create payload — text=%r, has_tools=%s, generation_keys=%s",
                    kwargs.get("text"), bool(tools), list(generation_options.keys()),
                )
                response = await client.responses.create(
                    model=api_name,
                    input=messages,
                    **generation_options,
                    **kwargs,
                    tools=tools if tools else list(),
                    timeout=timeout,
                )
                latency = time.time() - _start_time
            response = json.loads(response.json())
            response["latency"] = latency
            response.update(parse_response(
                response,
                response_format=response_format,
                stop_words=generation_options.get("stop", None),
            ))
            if not is_valid_response(response):
                # content-contract failure — identical retry cannot help; surface once and stop.
                logger.error(
                    f'openai responses unusable, not retrying (model={api_name}); '
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
                    {'model': api_name, 'input': messages, 'tools': tools or [], **generation_options, **kwargs},
                    status=_status, logger=logger, messages_key='input',
                )
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request openai responses: {response}')
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
        # Normalize Responses-API tool calls to the OpenAI tool_call shape
        # ({"id", "type": "function", "function": {"name", "arguments": <json str>}})
        # so the standardized output carries them like the chat/anthropic/google engines.
        tool_calls = [
            {
                "id": _item.get("call_id") or _item.get("id"),
                "type": "function",
                "function": {
                    "name": _item.get("name"),
                    "arguments": _item["arguments"] if isinstance(_item.get("arguments"), str)
                    else json.dumps(_item.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for _item in response["output"]
            if _item.get("type") in ("function_call", "custom_tool_call", "tool_call")
        ] or None
        finish_reason = response.get("status", None)
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
