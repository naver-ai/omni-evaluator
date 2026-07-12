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
from collections import OrderedDict
import json
from omegaconf import ListConfig, DictConfig
import os
from pathlib import Path
import PIL
from PIL import Image
import pydantic
import tempfile
import time
import traceback
from typing import Optional, Union, Any, List, Dict, Tuple, Callable

import logging
logger = logging.getLogger(__name__)

# Silence ``google_genai.models``'s per-call INFO line
# ``AFC is enabled with max remote calls: 10`` — it floods stdout and
# clobbers tqdm progress bars without conveying anything actionable.
# The SDK does not surface this as a kwarg; logger-level is the only knob.
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

try: # google
    from google import genai
except Exception as ex:
    logger.warning('Could not import dependencies: Google-genai api')

from omni_evaluator.api.google import get_engine_features
from omni_evaluator.inference import (
    TIMEOUT, MAX_RETRY, WAIT_BETWEEN_RETRY,
    NUM_MAX_COROUTINES, 
)
from omni_evaluator.utils.common import remove_stop_words
from omni_evaluator.utils.response import summarize_payload_shape, is_valid_response
from omni_evaluator.utils.multimodal import (
    to_pil_image, to_image_bytes, to_audio_bytes, to_nparray_audio,
    image_mime_type, audio_mime_type, media_mime_type,
    safe_request_get,
)
from omni_evaluator.utils.string import is_url
from omni_evaluator.schemas.generation_options import ApiGoogleGenerationOptions
from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
)


_UPLOADED_FILES_CACHE_MAX_SIZE = 1024
UPLOADED_FILES_CACHE = OrderedDict()

# --- model-name self-healing -------------------------------------------------
# The Gemini API rejects an unknown/mis-ordered model id with a 404 (e.g.
# `gemini-flash-3.1-lite` instead of the real `gemini-3.1-flash-lite`). On a 404
# we resolve the requested name to a real available model by EXACT token-set match
# (split on '-'/'.'), then retry once. ListModels is fetched lazily and cached.
_GOOGLE_MODEL_IDS = None      # cache: available model ids (generateContent), no "models/" prefix


def _is_model_not_found(ex) -> bool:
    return getattr(ex, "code", None) == 404 or "NOT_FOUND" in str(ex) or "is not found" in str(ex)


def _model_name_tokens(name: str) -> frozenset:
    import re
    return frozenset(t for t in re.split(r"[-.]", str(name).lower()) if t)


def _available_google_models(client) -> list:
    global _GOOGLE_MODEL_IDS
    if _GOOGLE_MODEL_IDS is None:
        ids = []
        try:
            for m in client.models.list():
                actions = (getattr(m, "supported_actions", None)
                           or getattr(m, "supported_generation_methods", None) or [])
                if "generateContent" in actions:
                    ids.append(str(getattr(m, "name", "")).replace("models/", ""))
        except Exception as ex:
            logger.warning(f'Google ListModels failed ({ex}); cannot resolve model name')
        _GOOGLE_MODEL_IDS = ids
    return _GOOGLE_MODEL_IDS


def _resolve_google_model(client, api_name: str) -> Optional[str]:
    """Map a mis-ordered/aliased gemini name to a real available model id by exact token-set
    match (gemini-flash-3.1-lite -> gemini-3.1-flash-lite). None if no match; the shortest id
    wins when several share the token set (prefers the base over -preview/-latest variants)."""
    avail = _available_google_models(client)
    if api_name in avail:
        return api_name
    want = _model_name_tokens(api_name)
    matches = [m for m in avail if _model_name_tokens(m) == want]
    return sorted(matches, key=len)[0] if matches else None


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
    generation_options = ApiGoogleGenerationOptions.from_dict(
        obj=generation_options,
        api_name=api_name,
        reasoning_options=reasoning_options,
    ).to_dict()

    # drop unsupported type
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
    messages = [
        ChatMessage.to_template(obj=_message, template="google")
        for _message in messages
    ]

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

    user_contents = list()
    _response_format_added = False
    for _message in messages:
        if (
            _message["role"] == "system"
            and response_format
        ):
            _message["content"].append({
                "type": "text",
                "text": response_format.lstrip(),
            })
            _response_format_added = True

        for _content in _message["content"]:
            if _content["type"] == "text":
                user_contents.append(ChatTextContent.get_value(_content))
            elif _content["type"] == "audio":
                _val = ChatAudioContent.get_value(_content)
                if isinstance(_val, bytes):
                    user_contents.append(genai.types.Part.from_bytes(
                        data=_val,
                        mime_type=audio_mime_type(_val),
                    ))
                elif isinstance(_val, str):
                    if os.path.exists(_val):
                        _file = file_upload_sync(
                            client=client,
                            filepath=_val,
                            interval=10,
                            timeout=timeout,
                        )
                        if not _file:
                            return None
                        user_contents.append(_file)
                    elif is_url(_val):
                        with tempfile.NamedTemporaryFile(delete=True) as fp:
                            _response = safe_request_get(_val, stream=True, timeout=timeout)
                            for chunk in _response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    fp.write(chunk)
                            fp.flush()
                            _file = file_upload_sync(
                                client=client,
                                filepath=fp.name,
                                interval=10,
                                timeout=timeout,
                            )
                            if not _file:
                                return None
                            user_contents.append(_file)
                    else: # base64_encoded string
                        _audio_bytes = to_audio_bytes(audio=_val)
                        user_contents.append(genai.types.Part.from_bytes(
                            data=_audio_bytes,
                            mime_type=audio_mime_type(_audio_bytes),
                        ))
                else:
                    raise ValueError(f'Invalid audio content: {type(_val)}')
            elif _content["type"] == "image":
                _image = to_pil_image(image=ChatImageContent.get_value(_content))
                _image_raw = to_image_bytes(image=_image, encode_base64=False)
                _image_part = genai.types.Part.from_bytes(data=_image_raw, mime_type=image_mime_type(_image_raw))
                user_contents.append(_image_part)
            elif _content["type"] == "video":
                _val = ChatVideoContent.get_value(_content)
                if os.path.exists(_val):
                    _file = file_upload_sync(
                        client=client,
                        filepath=_val,
                        interval=10,
                        timeout=timeout,
                    )
                    if not _file:
                        return None
                    user_contents.append(_file)
                else:
                    with tempfile.NamedTemporaryFile(delete=True) as fp:
                        _response = safe_request_get(_val, stream=True, timeout=timeout)
                        for chunk in _response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fp.write(chunk)
                        fp.flush()
                        _file = file_upload_sync(
                            client=client,
                            filepath=fp.name,
                            interval=10,
                            timeout=timeout,
                        )
                        if not _file:
                            return None
                        user_contents.append(_file)
    
    if (
        response_format
        and not _response_format_added
    ):
        user_contents.insert(0, response_format.lstrip())

    resolved_model = None
    for cur_try in range(1, max_retry+1):
        response, latency = None, None
        try:
            _start_time = time.time()
            response = client.models.generate_content(
                model=resolved_model or api_name,
                contents=user_contents,
                config=genai.types.GenerateContentConfig(
                    **generation_options,
                    tools=tools,
                ),
            )
            latency = time.time() - _start_time
            response = json.loads(response.json())
            response["latency"] = latency
            response.update(parse_response(
                response,
                response_format=_response_format_original,
                stop_words=generation_options.get("stopSequences", None),
            ))
            if not is_valid_response(response):
                # content-contract failure — identical retry cannot help; surface once and stop.
                logger.error(
                    f'google response unusable, not retrying (model={resolved_model or api_name}); '
                    f'no usable text/tool content (raise --max_new_tokens or set '
                    f'--thinking_budget) | response={response}'
                )
                return None
            return response
        except Exception as ex:
            # self-heal a wrong/mis-ordered model id once (gemini-flash-3.1-lite ->
            # gemini-3.1-flash-lite) and retry immediately with the resolved id.
            if resolved_model is None and _is_model_not_found(ex):
                cand = _resolve_google_model(client, api_name)
                if cand and cand != api_name:
                    resolved_model = cand
                    logger.warning(f"google model '{api_name}' not found -> resolved to '{cand}'")
                    continue
            _status = (
                getattr(ex, 'status_code', None)
                or getattr(ex, 'code', None)
                or getattr(getattr(ex, 'response', None), 'status_code', None)
            )
            if isinstance(_status, int) and 400 <= _status < 500:
                summarize_payload_shape(
                    {
                        'model': resolved_model or api_name,
                        'contents': user_contents,
                        'tools': tools or [],
                        **generation_options,
                    },
                    status=_status, logger=logger,
                    messages_key='contents', content_key='parts',
                )
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request google: {response}')
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
    generation_options = ApiGoogleGenerationOptions.from_dict(
        obj=generation_options,
        api_name=api_name,
        reasoning_options=reasoning_options,
    ).to_dict()

    # drop unsupported type
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
    messages = [
        ChatMessage.to_template(obj=_message, template="google")
        for _message in messages
    ]

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

    user_contents = list()
    _response_format_added = False
    async with semaphore:
        for _message in messages:
            if (
                _message["role"] == "system"
                and response_format
            ):
                _message["content"].append({
                    "type": "text",
                    "text": response_format.lstrip(),
                })
                _response_format_added = True
            
            for _content in _message["content"]:
                if _content["type"] == "text":
                    user_contents.append(ChatTextContent.get_value(_content))
                elif _content["type"] == "audio":
                    _val = ChatAudioContent.get_value(_content)
                    if isinstance(_val, bytes):
                        user_contents.append(genai.types.Part.from_bytes(
                            data=_val,
                            mime_type=audio_mime_type(_val),
                        ))
                    elif isinstance(_val, str):
                        if os.path.exists(_val):
                            _file = await file_upload_async(
                                client=client,
                                filepath=_val,
                                interval=10,
                                timeout=timeout,
                            )
                            if not _file:
                                return None
                            user_contents.append(_file)
                        elif is_url(_val):
                            with tempfile.NamedTemporaryFile(delete=True) as fp:
                                _response = safe_request_get(_val, stream=True, timeout=timeout)
                                for chunk in _response.iter_content(chunk_size=1024 * 1024):
                                    if chunk:
                                        fp.write(chunk)
                                fp.flush()
                                _file = await file_upload_async(
                                    client=client,
                                    filepath=fp.name,
                                    interval=10,
                                    timeout=timeout,
                                )
                                if not _file:
                                    return None
                                user_contents.append(_file)
                        else: # base64_encoded string
                            _audio_bytes = to_audio_bytes(audio=_val)
                            user_contents.append(genai.types.Part.from_bytes(
                                data=_audio_bytes,
                                mime_type=audio_mime_type(_audio_bytes),
                            ))
                    else:
                        raise ValueError(f'Invalid audio content: {type(_val)}')
                elif _content["type"] == "image":
                    _image = to_pil_image(image=ChatImageContent.get_value(_content))
                    _image_raw = to_image_bytes(image=_image, encode_base64=False)
                    _image_part = genai.types.Part.from_bytes(data=_image_raw, mime_type=image_mime_type(_image_raw))
                    user_contents.append(_image_part)
                elif _content["type"] == "video":
                    _val = ChatVideoContent.get_value(_content)
                    if os.path.exists(_val):
                        _file = await file_upload_async(
                            client=client,
                            filepath=_val,
                            interval=10,
                            timeout=timeout,
                        )
                        if not _file:
                            return None
                        user_contents.append(_file)
                    else:
                        with tempfile.NamedTemporaryFile(delete=True) as fp:
                            _response = safe_request_get(_val, stream=True, timeout=timeout)
                            for chunk in _response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    fp.write(chunk)
                            fp.flush()
                            _file = await file_upload_async(
                                client=client,
                                filepath=fp.name,
                                interval=10,
                                timeout=timeout,
                            )
                            if not _file:
                                return None
                            user_contents.append(_file)

    if (
        response_format
        and not _response_format_added
    ):
        user_contents.insert(0, response_format.lstrip())

    resolved_model = None
    for cur_try in range(1, max_retry+1):
        try:
            response, latency = None, None
            async with semaphore:
                _start_time = time.time()
                response = await client.models.generate_content(
                    model=resolved_model or api_name,
                    contents=user_contents,
                    config=genai.types.GenerateContentConfig(
                        **generation_options,
                        tools=tools,
                    ),
                )
                latency = time.time() - _start_time
            response = json.loads(response.json())
            response["latency"] = latency
            response.update(parse_response(
                response,
                response_format=_response_format_original,
                stop_words=generation_options.get("stopSequences", None),
            ))
            if not is_valid_response(response):
                # content-contract failure — identical retry cannot help; surface once and stop.
                logger.error(
                    f'google response unusable, not retrying (model={resolved_model or api_name}); '
                    f'no usable text/tool content (raise --max_new_tokens or set '
                    f'--thinking_budget) | response={response}'
                )
                return None
            return response
        except Exception as ex:
            # self-heal a wrong/mis-ordered model id once (see sync path).
            if resolved_model is None and _is_model_not_found(ex):
                cand = _resolve_google_model(client, api_name)
                if cand and cand != api_name:
                    resolved_model = cand
                    logger.warning(f"google model '{api_name}' not found -> resolved to '{cand}'")
                    continue
            _status = (
                getattr(ex, 'status_code', None)
                or getattr(ex, 'code', None)
                or getattr(getattr(ex, 'response', None), 'status_code', None)
            )
            if isinstance(_status, int) and 400 <= _status < 500:
                summarize_payload_shape(
                    {
                        'model': resolved_model or api_name,
                        'contents': user_contents,
                        'tools': tools or [],
                        **generation_options,
                    },
                    status=_status, logger=logger,
                    messages_key='contents', content_key='parts',
                )
            logger.error(f'({cur_try:02d}/{max_retry:02d}) Failed to request google: {response}')
            traceback.print_exc()
            await asyncio.sleep(wait_between_retry * cur_try)
            continue

    logger.error(f'Failed after {max_retry} tries')
    return None

def file_upload_sync(
    client,
    filepath: str,
    *,
    interval: float = 2.0,
    timeout: int = 600,
) -> Union[Any, bool]:
    file = None
    deadline = time.time() + timeout
    dispaly_name = f'{Path(filepath).parent.name}/{Path(filepath).name}'

    # detect MIME type from file content, not filename
    try:
        with open(filepath, "rb") as _f:
            _header = _f.read(12)
        mime_type = media_mime_type(_header, fallback="video/mp4")
    except Exception:
        mime_type = "video/mp4"

    # Check process-local cache first; skip per-sample `files.list(...)` to
    # avoid accumulating Files API load (transient 503s on long audio runs).
    # See the async counterpart for the full rationale.
    cached_name = UPLOADED_FILES_CACHE.get(dispaly_name)
    if cached_name:
        try:
            file = client.files.get(name=cached_name)
        except Exception as ex:
            logger.debug(f'cached file lookup failed; will re-upload ({ex})')
            UPLOADED_FILES_CACHE.pop(dispaly_name, None)
            file = None

    # upload if not in cache (or cached entry is invalid)
    if file is None:
        file = client.files.upload(
            file=filepath,
            config=genai.types.UploadFileConfig(
                display_name=dispaly_name,
                mime_type=mime_type,
            ),
        )
        UPLOADED_FILES_CACHE[dispaly_name] = file.name
        if len(UPLOADED_FILES_CACHE) > _UPLOADED_FILES_CACHE_MAX_SIZE:
            UPLOADED_FILES_CACHE.popitem(last=False)

    while True:
        _state = getattr(file, "state", None)
        _status = getattr(file, "status", None)
        _state_name = getattr(_state, "name", None) or str(_state)
        if "ACTIVE" in _state_name:
            return file
        if "FAILED" in _state_name:
            logger.error(f'File processing failed in Genai client - filepath_or_url: {filepath}, state: {_state}, status: {_status}')
            return False
        if time.time() >= deadline:
            logger.error(f'File state failed to be ACTIVE within {timeout}s - filepath_or_url: {filepath}, state: {_state}')
            return False
        time.sleep(interval)
        file = client.files.get(name=file.name)

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
    elif (
        isinstance(response.get("candidates", None), list)
        and len(response.get("candidates", list())) > 0
        and isinstance(response["candidates"][-1].get("content", None), dict)
        and isinstance(response["candidates"][-1]["content"].get("parts", None), (list, tuple))
        and len(response["candidates"][-1]["content"]["parts"]) > 0
        and response["candidates"][-1]["content"]["parts"][-1].get("text", None)
    ):
        generated_text = [
            _candidate["content"]["parts"][-1]["text"]
            for _candidate in response["candidates"]
        ]
        _thinking_parts = [
            _part.get("text", "")
            for _part in response["candidates"][-1]["content"]["parts"]
            if _part.get("thought", False)
        ]
        if _thinking_parts:
            reasoning = "\n".join(_thinking_parts)
        tool_calls = response["candidates"][-1]["content"]["parts"][-1].get("function_call", None)
        function_call = response["candidates"][-1]["content"]["parts"][-1].get("function_response", None)
        finish_reason = response["candidates"][-1].get("finish_reason", None)
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


async def file_upload_async(
    client,
    filepath: str,
    *,
    interval: float = 2.0,
    timeout: int = 600,
) -> Union[Any, bool]:
    file = None
    deadline = asyncio.get_running_loop().time() + timeout
    dispaly_name = f'{Path(filepath).parent.name}/{Path(filepath).name}'

    # detect MIME type from file content, not filename
    try:
        with open(filepath, "rb") as _f:
            _header = _f.read(12)
        mime_type = media_mime_type(_header, fallback="video/mp4")
    except Exception:
        mime_type = "video/mp4"

    # Check process-local cache first. Calling `files.list(...)` per sample
    # accumulates load against the Gemini Files API (paginated async iter on
    # every call) and triggers transient 503s on long runs (e.g. 3K audio
    # samples). The in-memory cache is sufficient — each sample's display_name
    # is derived from filepath, so a hit means we already uploaded it this
    # process. On miss, just upload; the API tolerates duplicates by
    # display_name (returns a new file_id).
    cached_name = UPLOADED_FILES_CACHE.get(dispaly_name)
    if cached_name:
        try:
            file = await client.files.get(name=cached_name)
        except Exception as ex:
            logger.debug(f'cached file lookup failed; will re-upload ({ex})')
            UPLOADED_FILES_CACHE.pop(dispaly_name, None)
            file = None

    # upload if not in cache (or cached entry is invalid)
    if file is None:
        file = await client.files.upload(
            file=filepath,
            config=genai.types.UploadFileConfig(
                display_name=dispaly_name,
                mime_type=mime_type,
            ),
        )
        UPLOADED_FILES_CACHE[dispaly_name] = file.name
        if len(UPLOADED_FILES_CACHE) > _UPLOADED_FILES_CACHE_MAX_SIZE:
            UPLOADED_FILES_CACHE.popitem(last=False)

    while True:
        _state = getattr(file, "state", None)
        _status = getattr(file, "status", None)
        _state_name = getattr(_state, "name", None) or str(_state)
        if "ACTIVE" in _state_name:
            return file
        if "FAILED" in _state_name:
            logger.error(f'File processing failed in Genai client - filepath_or_url: {filepath}, state: {_state}, status: {_status}')
            return False
        if asyncio.get_running_loop().time() >= deadline:
            logger.error(f'File state failed to be ACTIVE within {timeout}s - filepath_or_url: {filepath}, state: {_state}')
            return False
        await asyncio.sleep(interval)
        file = await client.files.get(name=file.name)