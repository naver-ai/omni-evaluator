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

import base64
from collections import defaultdict
import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass, MISSING
import json
import logging
import numpy as np
from omegaconf import ListConfig, DictConfig
import os
import PIL
from PIL import Image
import re
import tempfile
from typing import ClassVar, Union, Any, Callable, Iterator, Tuple, List, Literal, Dict, Optional

from omni_evaluator.enums.media import AudioFormat, ImageFormat, Modality, VideoFormat
from omni_evaluator.schemas import SchemaInterface

logger = logging.getLogger(__name__)
from omni_evaluator.utils.data import align_tag
from omni_evaluator.utils.multimodal import (
    to_pil_image, to_image_bytes, resize_image,
    to_audio_bytes, to_nparray_audio, detect_audio_format,
    to_video_bytes, to_nparray_video,
    detect_image_format, detect_video_format,
    image_mime_type, audio_mime_type, video_mime_type,
)
from omni_evaluator.utils.string import is_url

OCR_PREFIX = "Reference OCR token:"
ENTITY_PREFIX = "Detected entities:"
SUBTITLE_PREFIX = "Video subtitles:"

# Query-parameter signals for cloud-storage presigned URLs. Their presence
# indicates a private-bucket access token, which third-party fetchers
# (OpenAI / Anthropic image downloaders) cannot reliably resolve — region
# restrictions, ``Content-Disposition: attachment`` overrides, or expiring
# signatures all cause silent ``invalid_image_url`` rejections. When matched,
# the caller side downloads the bytes itself and inlines base64.
_PRESIGNED_QUERY_PARAMS = (
    "X-Amz-Signature",   # AWS S3 SigV4 (used by AWS, NCP, MinIO, Wasabi, …)
    "X-Goog-Signature",  # Google Cloud Storage
)


def _is_presigned_url(url: Any) -> bool:
    """Return True when *url* looks like a presigned cloud-storage URL.

    Strict signal: only matches when one of the well-known SigV4-style
    query parameters is present. Plain public URLs (HTTP/HTTPS with no
    signature) pass through unchanged so the downstream API can fetch them
    directly without an extra client-side download.
    """
    if not isinstance(url, str) or "?" not in url:
        return False
    if not url.lower().startswith(("http://", "https://")):
        return False
    _query = url.split("?", 1)[1]
    return any(_param in _query for _param in _PRESIGNED_QUERY_PARAMS)


_RAW_BASE64_RE = re.compile(r"[A-Za-z0-9+/]+={0,2}")


def _is_raw_base64(value: Any, min_len: int = 64) -> bool:
    """Return True when *value* is a bare base64 string (no ``data:`` prefix).

    OpenAI / vLLM reject ``image_url.url`` payloads that are raw base64 —
    they require the ``data:<mime>;base64,`` prefix. When a record carries
    a prefix-less base64 image, the caller forces ``to_pil=True`` so
    ``_convert_image`` routes through the PIL branch and re-emits the
    bytes with the proper data URI prefix.

    Strict signal: alphabet-only, padded-correct length (``len % 4 == 0``),
    not a known URL / file URI scheme, not an existing local path. Plain
    text and URLs pass through unchanged.
    """
    if not isinstance(value, str) or len(value) < min_len:
        return False
    if value.startswith(("http://", "https://", "file://", "data:")):
        return False
    if len(value) % 4 != 0:
        return False
    if os.path.exists(value):
        return False
    return bool(_RAW_BASE64_RE.fullmatch(value))


class ContentInterface:
    """Mixin for accessing content values across different API formats (OpenAI, vLLM, SGLang, HF, etc.)."""
    VALUE_KEYS: ClassVar[List[str]] = ["value"]
    NESTED_KEYS: ClassVar[Dict[str, str]] = {}

    @classmethod
    def get_key(cls, content: dict) -> Optional[str]:
        """Find which key holds the value in content dict."""
        for key in cls.VALUE_KEYS:
            if key in content:
                return key
        return None

    @classmethod
    def get_value(cls, content: dict) -> Any:
        """Extract the actual value, handling nested dicts."""
        key = cls.get_key(content)
        if key is None:
            return None
        value = content[key]
        nested_key = cls.NESTED_KEYS.get(key)
        if nested_key and isinstance(value, dict):
            return value.get(nested_key, value)
        return value

    @classmethod
    def set_value(cls, content: dict, new_value) -> None:
        """Set the actual value, preserving nested dict structure."""
        key = cls.get_key(content)
        if key is None:
            return
        nested_key = cls.NESTED_KEYS.get(key)
        if nested_key and isinstance(content[key], dict):
            content[key][nested_key] = new_value
        else:
            content[key] = new_value

    @classmethod
    def set_key(cls, content: dict, new_key: str) -> None:
        """Rename the value key (e.g., 'value' -> 'image')."""
        old_key = cls.get_key(content)
        if old_key and old_key != new_key:
            content[new_key] = content.pop(old_key)


@dataclass(kw_only=True)
class AudioContent(ContentInterface, SchemaInterface):
    VALUE_KEYS: ClassVar[List[str]] = ["value", "audio", "input_audio", "audio_url", "url"]
    NESTED_KEYS: ClassVar[Dict[str, str]] = {"input_audio": "data", "audio_url": "url"}
    # openai chat_content
    type: Modality = Modality.audio
    value: Union[str, bytes, np.ndarray]
        
    def to_dict(
        self,
        template: Optional[str] = None,
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        output = super().to_dict()
        output = AudioContent.to_template(
            obj=output,
            template=template,
        )
        return output

    @classmethod
    def to_template(
        cls,
        obj: Dict[str, Any],
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        output = None
        if template == "openai":
            output = AudioContent.to_openai_template(obj=obj)
        elif template == "openai_response":
            output = AudioContent.to_openai_response_template(obj=obj)
        elif template == "anthropic":
            output = AudioContent.to_anthropic_template(obj=obj)
        elif template == "google":
            output = obj
        elif template == "hf":
            output = AudioContent.to_hf_template(obj=obj)
        elif template == "json":
            output = AudioContent.to_serializable(obj=obj, remove_unserializable=True)
        else: # default
            output = AudioContent.to_serializable(obj=obj, remove_unserializable=False)
        return output
    
    @classmethod
    def to_openai_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Convert audio content dict to OpenAI input_audio format with base64-encoded data.
        # Args: obj - audio content dict with "value" key holding raw audio data
        # Returns: modified obj with "input_audio" key containing base64 data and format
        value = obj.pop("value", None)
        audio_format = detect_audio_format(audio_bytes=value)
        if not audio_format:
            audio_format = "wav"
        else:
            audio_format = audio_format.lower()
        value = to_audio_bytes(
            audio=value,
            encode_base64=True,
        )
        obj["type"] = "input_audio"
        obj["input_audio"] = {
            "data": value,
            "format": audio_format if audio_format else "wav",
        }
        return obj
    
    @classmethod
    def to_openai_response_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        value = cls._convert_audio(
            audio=value,
        )
        obj["type"] = "input_audio"
        obj["audio_url"] = value
        return obj

    @classmethod
    def to_anthropic_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        value = cls._convert_audio(
            audio=value,
        )
        obj["source"] = {
            "url": value,
        }
        return obj

    @classmethod
    def to_hf_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        obj["audio"] = value
        return obj

    @classmethod
    def to_serializable(
        cls,
        obj: Dict[str, Any],
        remove_unserializable: bool = False,
    ) -> Dict[str, Any]:
        # remove_unserializable=True is the "dump to disk" path. Drop the
        # payload unconditionally — base64 strings are json-serializable in
        # form but binary in spirit, and inflating them to disk was
        # ballooning audio benchmarks to multi-GB per file. All evaluation
        # engines re-load multimodal items from the dataset (builtin via
        # load_dataset overwrite; lmms_eval/lm_eval_harness/vlm_eval_kit
        # via their own task/dataset builders), so the value is redundant
        # on disk.
        if remove_unserializable:
            obj["value"] = None
            return obj
        try:
            json.dumps(obj)
        except Exception as ex:
            obj["value"] = to_audio_bytes(
                audio=obj["value"],
                encode_base64=True,
            )
        return obj

    @classmethod
    def _convert_audio(
        cls,
        audio: Union[str, bytes, np.ndarray],
    ) -> str:
        if isinstance(audio, str):
            if os.path.exists(audio): # path
                audio = f'file://{audio}'
            elif is_url(audio): # url
                pass
            else: # base64_str
                pass
        elif isinstance(audio, bytes):
            raise AssertionError("not implemented audio as bytes")
        elif isinstance(audio, np.ndarray):
            raise AssertionError("not implemented audio as np.ndarray")
        else:
            raise ValueError(f'invalid audio type: {type(audio)}')
        return audio
    
@dataclass
class OcrToken(SchemaInterface):
    id: str
    text: str
    bbox: Optional[List[Any]] = None
    confidence: Optional[float] = None

    def __init__(self, **kwargs) -> None:
        for _field in fields(self):
            if _field.name in kwargs:
                setattr(self, _field.name, kwargs[_field.name])
            elif _field.default != MISSING:
                setattr(self, _field.name, _field.default)
            else:
                raise TypeError(f'{self.__class__.__name__}.__init__() missing 1 required positional argument: {_field.name}')

@dataclass
class EntityToken(SchemaInterface):
    id: str
    text: str
    bbox: Optional[List[Any]] = None
    confidence: Optional[float] = None

    def __init__(self, **kwargs) -> None:
        for _field in fields(self):
            if _field.name in kwargs:
                setattr(self, _field.name, kwargs[_field.name])
            elif _field.default != MISSING:
                setattr(self, _field.name, _field.default)
            else:
                raise TypeError(f'{self.__class__.__name__}.__init__() missing 1 required positional argument: {_field.name}')


@dataclass
class SubtitleCue(SchemaInterface):
    """One time-aligned subtitle segment (WebVTT/SRT cue).

    A "cue" is the standard term in W3C WebVTT / common SRT libraries
    (pysubs2 et al.) for one self-contained subtitle entry. Unlike OcrToken,
    a cue is utterance-level, not sub-word.
    """
    start: float                                    # seconds from video start
    end: float
    text: str
    speaker: Optional[str] = None                   # optional, when multi-speaker

    def __init__(self, **kwargs) -> None:
        for _field in fields(self):
            if _field.name in kwargs:
                setattr(self, _field.name, kwargs[_field.name])
            elif _field.default != MISSING:
                setattr(self, _field.name, _field.default)
            else:
                raise TypeError(f'{self.__class__.__name__}.__init__() missing 1 required positional argument: {_field.name}')

@dataclass(kw_only=True)
class ImageContent(ContentInterface, SchemaInterface):
    VALUE_KEYS: ClassVar[List[str]] = ["value", "image", "image_url", "image_aux", "url"]
    NESTED_KEYS: ClassVar[Dict[str, str]] = {"value": "image", "image": "image", "image_url": "url", "image_aux": "image"}
    # openai chat_content
    type: Modality = Modality.image
    value: Union[str, PIL.Image.Image, np.ndarray]
    ocr: Optional[List[Union[Dict[str, Any], OcrToken]]] = None
    entity: Optional[List[Union[Dict[str, Any], EntityToken]]] = None
        
    def to_dict(
        self,
        template: Optional[str] = None,
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        output = super().to_dict()
        output = ImageContent.to_template(
            obj=output, 
            template=template,
            to_pil=to_pil,
        )
        return output
        
    @classmethod
    def to_template(
        cls,
        obj: Dict[str, Any],
        template: Optional[str] = None,
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        # Dispatch image content dict to the appropriate provider template and serialize OCR/entity tokens.
        # Args: obj - image content dict, template - provider name ("openai"|"anthropic"|"hf"|"json"|None)
        # Returns: transformed image content dict with provider-specific keys
        output = None
        if template == "openai":
            output = ImageContent.to_openai_template(obj=obj, to_pil=to_pil)
            output.pop("ocr", None)
            output.pop("entity", None)
        elif template == "openai_response":
            output = ImageContent.to_openai_response_template(obj=obj)
            output.pop("ocr", None)
            output.pop("entity", None)
        elif template == "anthropic":
            output = ImageContent.to_anthropic_template(obj=obj, to_pil=to_pil)
            output.pop("ocr", None)
            output.pop("entity", None)
        elif template == "google":
            output = obj
            output.pop("ocr", None)
            output.pop("entity", None)
        elif template == "hf":
            output = ImageContent.to_hf_template(obj=obj)
            output.pop("ocr", None)
            output.pop("entity", None)
        elif template == "json":
            output = ImageContent.to_serializable(obj=obj, remove_unserializable=True)
        else: # default
            output = ImageContent.to_serializable(obj=obj, remove_unserializable=False)
            
        if isinstance(output.get("ocr", None), (list, tuple)):
            output["ocr"] = [
                _ocr_token.to_dict() 
                if isinstance(_ocr_token, OcrToken) else _ocr_token
                for _ocr_token in output["ocr"]
            ]
        if isinstance(output.get("entity", None), (list, tuple)):
            output["entity"] = [
                _entity_token.to_dict() 
                if isinstance(_entity_token, EntityToken) else _entity_token
                for _entity_token in output["entity"]
            ]
        return output
    
    @classmethod
    def to_openai_template(
        cls,
        obj: Dict[str, Any],
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        # Convert image content dict to OpenAI chat-completions format (image_url or image_aux).
        # Args: obj - image content dict, to_pil - whether to convert value to PIL first
        # Returns: obj with "image_url" or "image_aux" key for OpenAI API consumption
        value = obj.pop("value", None)
        if (
            isinstance(value, (dict, DictConfig))
            and len(value) > 1
        ): # which include more than image_url such as ocr / lens_keywords for vllm
            _img = value["image"]
            # OpenAI Chat Completions API rejects file:// URIs; convert local paths to base64
            if (
                isinstance(value["image"], str) 
                and os.path.exists(value["image"])
            ):
                to_pil = True
            value["image"] = cls._convert_image(
                image=value["image"], 
                to_pil=to_pil,
            )
            obj["type"] = "image_aux"
            obj["image_aux"] = value
        else:
            # OpenAI Chat Completions API rejects file:// URIs; convert local paths to base64.
            # Also force base64 inline when the URL is a presigned cloud-storage URL — the
            # OpenAI fetcher cannot reliably download those (NCP/AWS attach
            # ``Content-Disposition: attachment``, signatures expire, or outbound IP rules
            # block the fetch), returning ``invalid_image_url``. Public URLs pass through.
            # Raw base64 (no ``data:`` prefix) also gets rejected as ``invalid_image_url`` —
            # force PIL so the bytes are re-emitted with the proper data URI prefix.
            if isinstance(value, str) and (
                os.path.exists(value) or _is_presigned_url(value) or _is_raw_base64(value)
            ):
                to_pil = True
            value = cls._convert_image(
                image=value,
                to_pil=to_pil,
            )
            obj["type"] = "image_url"
            obj["image_url"] = {"url": value}
        return obj

    @classmethod
    def to_openai_response_template(
        cls,
        obj: Dict[str, Any],
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        # Convert image content dict to OpenAI response-API format using "input_image" type.
        # Args: obj - image content dict, to_pil - whether to convert value to PIL first
        # Returns: obj with "input_image" type and image data under "image_url" or "image_aux"
        value = obj.pop("value", None)
        if (
            isinstance(value, (dict, DictConfig))
            and len(value) > 1
        ): # which include more than image_url such as ocr / lens_keywords for vllm
            _img = value["image"]
            # OpenAI Responses API rejects file:// URIs; convert local paths to base64
            if (
                isinstance(value["image"], str) 
                and os.path.exists(value["image"])
            ):
                to_pil = True
            value["image"] = cls._convert_image(
                image=value["image"], 
                to_pil=to_pil,
            )
            obj["type"] = "input_image"
            obj["image_aux"] = value
        else:
            # OpenAI Responses API rejects file:// URIs; convert local paths to base64.
            # Same presigned-URL caveat as ``to_openai_template`` — force base64 inline.
            # Raw base64 (no ``data:`` prefix) also triggers ``invalid_image_url`` —
            # force PIL so the bytes are re-emitted with the proper data URI prefix.
            if isinstance(value, str) and (
                os.path.exists(value) or _is_presigned_url(value) or _is_raw_base64(value)
            ):
                to_pil = True
            value = cls._convert_image(
                image=value,
                to_pil=to_pil,
            )
            obj["type"] = "input_image"
            obj["image_url"] = value
        return obj
    
    @classmethod
    def to_anthropic_template(
        cls,
        obj: Dict[str, Any],
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        # Convert image content dict to Anthropic Messages API format with base64 or URL source.
        # Args: obj - image content dict, to_pil - whether to force PIL conversion
        # Returns: obj with "source" key containing base64 data or URL for Anthropic API
        value = obj.pop("value", None)
        if (
            isinstance(value, (dict, DictConfig))
            and len(value) > 1
        ): # which include more than image_url such as ocr / lens_keywords for vllm
            # only openai & private-vLLM supports "image_aux" type
            value = value["image"]

        # Presigned cloud-storage URLs aren't reliably fetchable from Anthropic
        # either — same root cause as OpenAI (attachment headers, region/IP
        # restrictions, signature expiry). Force base64 inline.
        # Raw base64 (no ``data:`` prefix) also needs to take the base64-source
        # branch — Anthropic's URL source can't parse a bare base64 string.
        if (
            not isinstance(value, str)
            or to_pil
            or _is_presigned_url(value)
            or _is_raw_base64(value)
            or os.path.exists(value)  # local path: Anthropic can't fetch file:// → inline base64 (parity w/ OpenAI template)
        ):
            _raw = to_image_bytes(image=value, encode_base64=False)
            _mime = image_mime_type(_raw)
            _b64 = base64.standard_b64encode(_raw).decode("utf-8")
            obj["source"] = {
                "type": "base64",
                "media_type": _mime,
                "data": _b64,
            }
        else:
            value = cls._convert_image(
                image=value,
                to_pil=to_pil,
                attach_bytes_prefix=False,
            ) # image_url (maybe image_path)
            # Anthropic Messages API: URL source uses the ``url`` key
            # (base64 source uses ``data``). Mixing them up returns
            # ``messages.X.content.Y.image.source.url.url: Field required``.
            obj["source"] = {
                "type": "url",
                "url": value,
            }
        return obj
    
    @classmethod
    def to_hf_template(cls, obj: Dict[str, Any]) -> Dict[str, Any]:
        value = obj.pop("value", None)
        if (
            isinstance(value, (dict, DictConfig))
            and len(value) > 1
        ): # which include more than image_url such as ocr / lens_keywords for vllm
            # only openai & private-vLLM supports "image_aux" type
            value = value["image"]
        obj["image"] = value
        return obj

    @classmethod
    def to_serializable(cls, obj: Dict[str, Any], remove_unserializable: bool = False) -> Dict[str, Any]:
        # See AudioContent.to_serializable — remove_unserializable=True is
        # the disk-dump path; drop the payload unconditionally so base64
        # images don't bloat the artifact (engines re-load from dataset).
        if remove_unserializable:
            obj["value"] = None
            return obj
        try:
            json.dumps(obj)
        except Exception as ex:
            obj["value"] = to_image_bytes(
                image=obj["value"],
                encode_base64=True,
            )
        return obj

    @classmethod
    def _convert_image(
        cls,
        image: Union[str, PIL.Image.Image],
        to_pil: Optional[bool] = False,
        attach_bytes_prefix: Optional[bool] = True,
    ) -> str:
        # Normalize an image (path, URL, or PIL) to a string representation (file URI, URL, or base64).
        # Args: image - file path, URL, or PIL Image; attach_bytes_prefix - prepend data URI prefix
        # Returns: string form of the image suitable for API payloads
        if to_pil:
            image = to_pil_image(image=image)
        if isinstance(image, str):
            if os.path.exists(image): # path
                image = f'file://{image}'
            elif image.startswith("data:") or is_url(image): # data URI or real URL — pass through
                pass
            else: # raw base64 image data (e.g. builtin datasets store images inline)
                # Without a `data:<mime>;base64,` prefix the OpenAI/Anthropic image
                # fetchers treat the bare base64 as a literal URL and reject it
                # ("Failed to download image from /9j/..."). Promote to a data URI
                # when a prefix is wanted (attach_bytes_prefix); otherwise leave the
                # bare base64 (Anthropic-style source.data consumers expect it raw).
                try:
                    _raw = base64.b64decode(image, validate=True)
                    if attach_bytes_prefix:
                        image = f'data:{image_mime_type(_raw)};base64,{image}'
                except Exception:
                    pass # not base64 → leave unchanged (preserve prior behavior)
        elif isinstance(image, PIL.Image.Image):
            _raw = to_image_bytes(image=image, encode_base64=False)
            _mime = image_mime_type(_raw)
            _b64 = base64.standard_b64encode(_raw).decode("utf-8")
            if attach_bytes_prefix:
                image = f'data:{_mime};base64,{_b64}'
            else:
                image = _b64
        else:
            raise ValueError(f'invalid image type: {type(image)}')
        return image
    
@dataclass(kw_only=True)
class TextContent(ContentInterface, SchemaInterface):
    VALUE_KEYS: ClassVar[List[str]] = ["value", "text"]
    NESTED_KEYS: ClassVar[Dict[str, str]] = {}
    # openai chat_content
    type: Modality = Modality.text
    value: str
        
    def to_dict(
        self,
        template: Optional[str] = None,
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        output = super().to_dict()
        output = TextContent.to_template(obj=output, template=template)
        return output

    @classmethod
    def to_template(
        cls,
        obj: Dict[str, Any],
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        output = None
        if template == "openai":
            output = TextContent.to_openai_template(obj=obj)
        elif template == "openai_response":
            output = TextContent.to_openai_response_template(obj=obj)
        elif template == "anthropic":
            output = TextContent.to_anthropic_template(obj=obj)
        elif template == "google":
            output = TextContent.to_google_template(obj=obj)
        elif template == "hf":
            output = TextContent.to_hf_template(obj=obj)
        elif template == "json":
            output = TextContent.to_serializable(obj=obj, remove_unserializable=True)
        else: # default
            output = TextContent.to_serializable(obj=obj, remove_unserializable=False)
        return output

    @classmethod
    def to_openai_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        if not obj.get("text", None) or value:
            obj["text"] = value
        return obj

    @classmethod
    def to_openai_response_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        obj["type"] = "input_text"
        if not obj.get("text", None) or value:
            obj["text"] = value
        return obj

    @classmethod
    def to_anthropic_template(
        cls,
        obj: Dict[str, Any],
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        if not obj.get("text", None) or value:
            obj["text"] = value
        return obj

    @classmethod
    def to_google_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        if not obj.get("text", None) or value:
            obj["text"] = value
        return obj

    @classmethod
    def to_hf_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        if not obj.get("text", None) or value:
            obj["text"] = value
        return obj

    @classmethod
    def to_serializable(cls, obj: Dict[str, Any], remove_unserializable: bool = False) -> Dict[str, Any]:
        try:
            json.dumps(obj)
        except Exception as ex:
            if remove_unserializable:
                obj["value"] = None
        return obj
    
@dataclass(kw_only=True)
class VideoContent(ContentInterface, SchemaInterface):
    VALUE_KEYS: ClassVar[List[str]] = ["value", "video", "video_url", "url"]
    NESTED_KEYS: ClassVar[Dict[str, str]] = {"video_url": "url"}
    # openai chat_content
    type: Modality = Modality.video
    value: Union[str, np.ndarray]
    subtitle: Optional[List[Union[Dict[str, Any], SubtitleCue]]] = None

    def to_dict(
        self,
        template: Optional[str] = None,
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        output = super().to_dict()
        output = VideoContent.to_template(obj=output, template=template)
        return output

    @classmethod
    def to_template(
        cls,
        obj: Dict[str, Any],
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Dispatch video content dict to the appropriate provider template formatter.
        # Args: obj - video content dict, template - provider name ("openai"|"anthropic"|"hf"|"json"|None)
        # Returns: transformed video content dict with provider-specific keys
        output = None
        if template == "openai":
            output = VideoContent.to_openai_template(obj=obj)
            output.pop("subtitle", None)
        elif template == "openai_response":
            output = VideoContent.to_openai_response_template(obj=obj)
            output.pop("subtitle", None)
        elif template == "anthropic":
            output = VideoContent.to_anthropic_template(obj=obj)
            output.pop("subtitle", None)
        elif template == "google":
            output = obj
            output.pop("subtitle", None)
        elif template == "hf":
            output = VideoContent.to_hf_template(obj=obj)
            output.pop("subtitle", None)
        elif template == "json":
            output = VideoContent.to_serializable(obj=obj, remove_unserializable=True)
        else: # default
            output = VideoContent.to_serializable(obj=obj, remove_unserializable=False)

        if isinstance(output.get("subtitle", None), (list, tuple)):
            output["subtitle"] = [
                _cue.to_dict()
                if isinstance(_cue, SubtitleCue) else _cue
                for _cue in output["subtitle"]
            ]
        return output
    
    @classmethod
    def to_openai_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        value = cls._convert_video(
            video=value,
        )
        obj["type"] = "video_url"
        obj["video_url"] = {"url": value}
        return obj
    
    @classmethod
    def to_openai_response_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        value = cls._convert_video(
            video=value,
        )
        obj["type"] = "input_video"
        obj["video_url"] = value
        return obj

    @classmethod
    def to_anthropic_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        value = cls._convert_video(
            video=value,
        )
        obj["source"] = {"url": value}
        return obj

    @classmethod
    def to_hf_template(
        cls,
        obj: Dict[str, Any],
    ) -> Dict[str, Any]:
        value = obj.pop("value", None)
        obj["video"] = value
        return obj

    @classmethod
    def to_serializable(cls, obj: Dict[str, Any], remove_unserializable: bool = False) -> Dict[str, Any]:
        # See AudioContent.to_serializable — remove_unserializable=True is
        # the disk-dump path; drop the payload unconditionally so base64
        # video doesn't bloat the artifact (engines re-load from dataset).
        if remove_unserializable:
            obj["value"] = None
            return obj
        try:
            json.dumps(obj)
        except Exception as ex:
            raise AssertionError(f'not implemented yet')
        return obj

    @classmethod
    def _convert_video(
        cls,
        video: Union[str, np.ndarray],
    ) -> str:
        if isinstance(video, str):
            if os.path.exists(video): # path
                video = f'file://{video}'
            else: # url
                pass
        elif isinstance(video, np.ndarray):
            raise AssertionError("not implemented video as a np.ndarray")
        else:
            raise ValueError(f'invalid video type: {type(video)}')
        return video

CONTENT_ACCESSOR_MAP = {
    "text": TextContent,
    "audio": AudioContent,
    "image": ImageContent,
    "video": VideoContent,
}


@dataclass
class ToolCallFunction(SchemaInterface):
    name: str
    arguments: str
    
    def __post_init__(self) -> None:
        if isinstance(self.arguments, dict):
            self.arguments = json.dumps(self.arguments, ensure_ascii=False)

@dataclass
class ToolCall(SchemaInterface):
    id: str
    type: Literal["function"]
    function: Union[ToolCallFunction, Dict[str, Any]]

@dataclass
class Message(SchemaInterface):        
    role: Literal["system", "user", "assistant"]
    content: List[Union[
        Dict[str, Any], 
        AudioContent, 
        ImageContent, 
        TextContent, 
        VideoContent,
    ]]
    name: Optional[str] = None
    tool_call_id: Optional[Union[str, int]] = None
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    function_call: Optional[Any] = None
    annotations: Optional[List[str]] = None
    
    def __post_init__(self) -> None:
        if isinstance(self.content, str):
            self.content = [{
                "type": "text", 
                "value": self.content,
            }]
        elif isinstance(self.content, (dict, DictConfig)):
            self.content = [self.content, ]
            
    def to_dict(
        self,
        template: Optional[str] = None,
        to_pil: Optional[bool] = False,
    ) -> Dict[str, Any]:
        # Serialize a Message to dict, converting each content item to the target provider template.
        # Args: template - provider format ("openai"|"anthropic"|"hf"|"json"|None), to_pil - force PIL images
        # Returns: dict with role, content list, and optional fields (tool_calls, name, etc.)
        output = super().to_dict()
        for field in fields(self):
            if (
                field.name == "content"
                or output.get(field.name, None) is not None
            ):
                continue
            output.pop(field.name, None)

        if output["content"] is None:
            pass
        elif isinstance(output["content"], str):
            output["content"] = output["content"] # system message
        else:
            _contents = list()
            if isinstance(output["content"], (dict, AudioContent, ImageContent, TextContent, VideoContent)):
                output["content"] = [output["content"], ]
            if not isinstance(output["content"], (list, tuple)):
                raise TypeError(f'Invalid content type: {output["content"]}')
            for _idx, _content in enumerate(output["content"]):
                if _content["type"] == "audio":
                    _contents.append(AudioContent.to_template(
                        obj=_content, 
                        template=template,
                    ))
                elif _content["type"] == "image":
                    _content_ocr, _content_entity = None, None
                    if isinstance(_content["value"], dict):
                        _content_ocr = _content["value"].pop("ocr", None) # maybe str
                        _content_entity = _content["value"].pop("entity", None) # maybe str
                
                    _contents.append(ImageContent.to_template(
                        obj=_content, 
                        template=template,
                        to_pil=to_pil,
                    ))
                        
                    if _content_ocr:
                        if isinstance(_content_ocr, (list, tuple)):
                            _content_ocr = ",".join([
                                _ocr_token["text"]
                                if isinstance(_ocr_token, dict) else _ocr_token
                                for _ocr_token in _content_ocr
                            ])
                        _content_ocr = TextContent(**{
                            "type": "text", 
                            "value": f'{OCR_PREFIX} {_content_ocr}',
                        }).to_dict()
                        _contents.append(TextContent.to_template(
                            obj=_content_ocr,
                            template=template,
                        ))
                    if _content_entity:
                        if isinstance(_content_entity, (list, tuple)):
                            _content_entity = ",".join([
                                _entity_token["text"]
                                if isinstance(_entity_token, dict) else _entity_token
                                for _entity_token in _content_entity
                            ])
                        _content_entity = TextContent(**{
                            "type": "text", 
                            "value": f'{ENTITY_PREFIX} {_content_entity}',
                        }).to_dict()
                        _contents.append(TextContent.to_template(
                            obj=_content_entity,
                            template=template,
                        ))
                elif _content["type"] == "text":
                    _contents.append(TextContent.to_template(
                        obj=_content, 
                        template=template,
                    ))
                elif _content["type"] == "video":
                    _content_subtitle = None
                    if isinstance(_content.get("value", None), dict):
                        _content_subtitle = _content["value"].pop("subtitle", None)
                    if _content_subtitle is None:
                        _content_subtitle = _content.pop("subtitle", None)

                    _contents.append(VideoContent.to_template(
                        obj=_content,
                        template=template,
                    ))

                    if _content_subtitle:
                        if isinstance(_content_subtitle, (list, tuple)):
                            _rendered_cues = list()
                            for _cue in _content_subtitle:
                                if isinstance(_cue, SubtitleCue):
                                    _start = _cue.start
                                    _end = _cue.end
                                    _text = _cue.text
                                elif isinstance(_cue, dict):
                                    _start = _cue.get("start", 0.0)
                                    _end = _cue.get("end", 0.0)
                                    _text = _cue.get("text", "")
                                else:
                                    continue
                                _rendered_cues.append(
                                    f'[{float(_start):.1f}-{float(_end):.1f}] {_text}'
                                )
                            _content_subtitle = "\n".join(_rendered_cues)
                        if _content_subtitle:
                            _content_subtitle = TextContent(**{
                                "type": "text",
                                "value": f'{SUBTITLE_PREFIX} {_content_subtitle}',
                            }).to_dict()
                            _contents.append(TextContent.to_template(
                                obj=_content_subtitle,
                                template=template,
                            ))
                output["content"] = _contents
        
        if template in ("anthropic", "google"):
            _name = output.get("name", None)
            if _name and output.get("content"):
                _first_text_idx = next(
                    (i for i, c in enumerate(output["content"]) if isinstance(c, dict) and c.get("type") == "text"),
                    None,
                )
                if _first_text_idx is not None:
                    output["content"][_first_text_idx]["text"] = f"[{_name}]: " + output["content"][_first_text_idx].get("text", "")
                else:
                    output["content"].insert(0, {"type": "text", "text": f"[{_name}]"})
            output.pop("name", None)
        return output
    
    @classmethod
    def to_template(
        cls,
        obj: Dict[str, Any],
        template: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Convert a raw message dict's content items in-place to the target provider template.
        # Args: obj - message dict with "content" list, template - provider name or "datalake"
        # Returns: obj with each content item transformed to the specified template format
        if template == "datalake":
            obj = cls.to_datalake_template(
                obj=obj,
                **kwargs,
            )
        else:
            if template in [
                "anthropic", 
                "google",
            ]:
                _name = obj.pop("name", None)
                if (
                    _name 
                    and obj.get("content", None)
                ):
                    if isinstance(obj["content"], str):
                        obj["content"] = f'[{_name}]: {obj["content"]}'
                    elif (
                        isinstance(obj["content"], (dict, TextContent))
                        and obj["content"]["type"] == "text"
                    ):
                        _text = obj["content"].pop("value", None) or obj["content"].get("text", "")
                        obj["content"]["value"] = f'[{_name}]: {_text}'
                    elif isinstance(obj["content"], (list, tuple)):
                        _index = None
                        for _content_idx, _content in enumerate(obj["content"]):
                            if _content["type"] == "text":
                                _index = _content_idx
                        if _index is None:
                            obj["content"].append({
                                "type": "text",
                                "value": f'[{_name}]',
                            })
                        else:
                            _value = obj["content"][_index].pop("value", "")
                            obj["content"][_index]["value"] = f'[{_name}]: {_value}'
            
            if obj["content"]:
                if isinstance(obj["content"], str):
                    obj["content"] = [{
                        "type": "text",
                        "value": obj["content"],
                    }, ]
                elif isinstance(obj["content"], dict):
                    obj["content"] = [obj["content"], ]
                for _idx, _content in enumerate(obj["content"]):
                    if _content["type"] == "audio":
                        _content = AudioContent.to_template(obj=_content, template=template)
                    elif _content["type"] == "image":
                        _content = ImageContent.to_template(obj=_content, template=template)
                    elif _content["type"] == "text":
                        _content = TextContent.to_template(obj=_content, template=template)
                    elif _content["type"] == "video":
                        _content = VideoContent.to_template(obj=_content, template=template)
                    obj["content"][_idx] = _content

        return obj
    
    @classmethod
    def to_datalake_template(
        cls,
        obj: Dict[str, Any],
        image_tag: Optional[str] = "<|image|>",
        video_tag: Optional[str] = "<|video|>",
        audio_tag: Optional[str] = "<|audio|>",
        is_first_message: Optional[bool] = False,
        is_last_message: Optional[bool] = False,
    ) -> Dict[str, Any]:
        # Convert a message dict to datalake training format with separated media URLs and aligned tags.
        # Args: obj - message dict, image/video/audio_tag - placeholder tokens, is_last_message - marks end-of-turn
        # Returns: datalake-format dict with content, *_urls, *_metas, and alignment tags inserted
        message = {
            "role": obj["role"],
            "content_type": "text",
            "content": "",
            "candidates": list(),
            "stop": False,
            "trainable_role": False,
            "trainable_content": False,
            "endofturn": is_last_message,
            "audio_urls": list(),
            "audio_metas": list(),
            "image_urls": list(),
            "image_metas": list(),
            "video_urls": list(),
            "video_metas": list(),
            "debuggingInfo": dict(),
            "meta": dict(),
        }
        if (
            obj["role"] == "assistant"
            and is_last_message
        ):
            message["stop"] = True      

        if obj["content"]:
            _ocr_idx, _entity_idx = 0, 0
            for _idx, _content in enumerate(obj["content"]):
                _content_cls = CONTENT_ACCESSOR_MAP.get(_content["type"])
                _value_key = _content_cls.get_key(_content) if _content_cls else None
                if not _value_key:
                    raise ValueError(f'no valid field in content: {list(_content.keys())}')
                _value = _content[_value_key]
                
                if _content["type"] == "text":
                    if _value.startswith(OCR_PREFIX) : # ocr prompt
                        if len(message["image_metas"]) <= _ocr_idx:
                            message["image_metas"].append(dict())
                        _ocr_tokens = _value.replace(OCR_PREFIX, "").strip().split(",")
                        message["image_metas"][_ocr_idx]["words"] = [
                            OcrToken(
                                id=_idx,
                                text=_token,
                            ).to_dict()
                            for _token_idx, _token in enumerate(_ocr_tokens)
                        ]
                        _ocr_idx += 1
                    elif _value.startswith(ENTITY_PREFIX) : # ocr prompt
                        if len(message["image_metas"]) <= _entity_idx:
                            message["image_metas"].append(dict())
                        _entity_tokens = _value.replace(ENTITY_PREFIX, "").strip().split(",")
                        message["image_metas"][_entity_idx]["words"] = [
                            EntityToken(
                                id=_idx,
                                text=_token,
                            ).to_dict()
                            for _token_idx, _token in enumerate(_entity_tokens)
                        ]
                        _entity_idx += 1
                    else:
                        message["content"] += f'{_value}'
                elif _content["type"] == "audio":
                    message["audio_urls"].append(_value)
                    message["audio_metas"].append({
                        "format": "wav", 
                        "note": "api_omni audio input",
                    })
                    # TODO: modify if interleaved any-to-any implemented
                    # message["content"] += audio_tag 
                elif _content["type"] == "image":
                    if isinstance(_value, (dict, DictConfig)):
                        _image = to_pil_image(image=_value["image"])
                        message["image_urls"].append(_image)
                        _image_meta = dict()
                        if "ocr" in _value:
                            _ocr_tokens = _value["ocr"]
                            _image_meta["words"] = _ocr_tokens
                        if "entity" in _value:
                            _entity_tokens = _value["entity"]
                            _image_meta["lens"] = _entity_tokens
                        message["image_metas"].append(_image_meta)
                    else:
                        _value = to_pil_image(image=_value)
                        message["image_urls"].append(_value)
                    # TODO: modify if interleaved any-to-any implemented
                    # message["content"] += image_tag 
                elif _content["type"] == "video":
                    message["video_urls"].append(_value)
                
        # alig_tags: image
        message["content"] = re.sub(r"<image_[0-9]{1,5}>", image_tag, message["content"])
        message["content"] = re.sub(re.escape(image_tag), "", message["content"])
        if len(message["image_urls"]) > 0:
            message["content"] = align_tag(
                text=message["content"],
                tag=image_tag,
                num_attach=len(message["image_urls"]),
                attach_head=True,
                re_escape=True,
            )
        # alig_tags: video
        message["content"] = re.sub(r"<video_[0-9]{1,5}>", video_tag, message["content"])
        message["content"] = re.sub(re.escape(video_tag), "", message["content"])
        if len(message["video_urls"]) > 0:
            message["content"] = align_tag(
                text=message["content"],
                tag=video_tag,
                num_attach=len(message["video_urls"]),
                attach_head=True,
                re_escape=True,
            )
        # alig_tags: audio
        message["content"] = re.sub(r"<audio_[0-9]{1,5}>", audio_tag, message["content"])
        message["content"] = re.sub(re.escape(audio_tag), "", message["content"])
        if len(message["audio_urls"]) > 0:
            message["content"] = align_tag(
                text=message["content"],
                tag=audio_tag,
                num_attach=len(message["audio_urls"]),
                attach_head=False,
                re_escape=True,
            )
        message["content"] = message["content"].rstrip()
        return message
    
    @classmethod
    def get_prompt(
        cls,
        message: Dict[str, Any],
    ) -> str:
        # Build a human-readable prompt string from a message dict in "{role}: {content}" format.
        # Args: message - message dict with "role", optional "name", and "content"
        # Returns: formatted string like "user: What is in this image?"
        name = message["role"]
        if getattr(message, "name", None):
            name = message["name"]
            name = name.title()
        
        query = ""
        if isinstance(message["content"] ,str):
            query +=  f'{message["content"]}\n'
        else:
            _contents = copy.deepcopy(message["content"])
            if isinstance(_contents, (dict, AudioContent, ImageContent, TextContent, VideoContent)):
                _contents = [_contents, ]
            if not isinstance(_contents, (list, tuple)):
                raise ValueError(f'invalid content: {_contents}')
            for _content in _contents:
                if isinstance(_content, str):
                    query += f'{_content}\n'
                elif isinstance(_content, (dict, AudioContent, ImageContent, TextContent, VideoContent)):
                    _content_cls = CONTENT_ACCESSOR_MAP.get(_content["type"])
                    _val = _content_cls.get_value(_content) if _content_cls else None
                    if _val is None:
                        raise ValueError(f'invalid content: {_content}')
                    query += f'{_val}\n'
        query = query.rstrip()
        prompt = f'{name}: {query}'
        return prompt
    
    @classmethod
    def get_name(
        cls,
        message: Dict[str, Any],
    ) -> str:
        output = message["role"]
        if getattr(message, "name", None):
            output = message["name"]
        return output
    
    @classmethod
    def get_user_messages(
        cls,
        messages: List[Union[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        messages = [
            _message.to_dict() if not isinstance(_message, dict) else _message
            for _message in messages
        ]

        user_messages = list()
        for _message in messages:
            if _message["role"] != "user":
                continue
            user_messages.append(_message)
        user_messages = copy.deepcopy(user_messages)
        return user_messages
    
    @classmethod
    def get_query(
        cls,
        message: Union[Dict[str, Any], "Message"],
    ) -> str:
        # Extract concatenated text-only content from a message, excluding OCR/entity prefixes.
        # Args: message - Message instance or message dict with "content" list
        # Returns: concatenated text query string with trailing whitespace stripped
        if not isinstance(message, dict):
            message = message.to_dict()

        query = ""
        for _c in message["content"]:
            if not isinstance(_c, dict):
                _c = _c.to_dict()
            if _c["type"] != "text":
                continue
            
            _query = TextContent.get_value(_c)
            if _query is None:
                raise ValueError(f'no valid value field in ChatContent: {_c}')
            if (
                _query.startswith(OCR_PREFIX)
                or _query.startswith(ENTITY_PREFIX)
                or _query.startswith(SUBTITLE_PREFIX)
            ):
                continue
            query += f'{_query}\n'
        
        query = query.rstrip()
        return query
        
    @classmethod
    def iter_multimodal_contents(
        cls,
        messages: List[Union[Dict[str, Any], "Message"]],
    ) -> Iterator[Dict[str, Any]]:
        # Yield every audio/image/video content dict across all messages in dialogue order.
        # Args: messages - list of Message instances or message dicts
        # Returns: iterator over dict-form multimodal content items (mutating a yielded
        #   item updates the underlying dict for dict-form messages)
        if not messages:
            return
        for _message in messages:
            if not isinstance(_message, dict):
                _message = _message.to_dict()
            _contents = _message.get("content", None)
            if not isinstance(_contents, list):
                continue
            for _content in _contents:
                if not isinstance(_content, dict):
                    _content = _content.to_dict()
                if _content.get("type", None) not in ("audio", "image", "video"):
                    continue
                yield _content

    @classmethod
    def get_images(
        cls,
        message: Union[Dict[str, Any], "Message"],
        to_pil: bool = True,
    ) -> List[Any]:
        # Extract all image values from a message's content list.
        # Args: message - Message instance or dict, to_pil - convert each image to PIL.Image
        # Returns: list of images (PIL.Image if to_pil, otherwise raw value)
        if not isinstance(message, dict):
            message = message.to_dict()
        _content = message["content"]
        images = list()
        for _c in _content:
            if not isinstance(_c, dict):
                _c = _c.to_dict()
            if _c["type"] != "image":
                continue
            _image = ImageContent.get_value(_c)
            if not _image:
                continue
            if to_pil:
                _image = to_pil_image(image=_image)
            images.append(_image)
        return images
    
    @classmethod
    def resize_images(
        cls,
        message: Union[Dict[str, Any], "Message"],
        max_size: Optional[int] = 224,
        to_pil: bool = True,
    ) -> Union[Dict[str, Any], "Message"]:
        # Resize all images in a message's content in-place to fit within max_size dimensions.
        # Args: message - message dict or Message, max_size - maximum height/width in pixels
        # Returns: the same message object with images resized
        for _content_idx, _content in enumerate(message["content"]):
            if _content["type"] != "image":
                continue
            _image = ImageContent.get_value(_content)
            if _image is not None and isinstance(_image, PIL.Image.Image):
                ImageContent.set_value(_content, resize_image(
                    image=_image,
                    max_height=max_size,
                    max_width=max_size,
                ))
        return message
    
    @classmethod
    def get_videos(
        cls,
        message: Union[Dict[str, Any], "Message"],
    ) -> List[Any]:
        # Extract all video values from a message's content list.
        # Args: message - Message instance or message dict
        # Returns: list of video values (paths, URLs, or ndarrays)
        if not isinstance(message, dict):
            message = message.to_dict()
        _content = message["content"]
        videos = list()
        for _c in _content:
            if not isinstance(_c, dict):
                _c = _c.to_dict()
            if _c["type"] != "video":
                continue
            _video = VideoContent.get_value(_c)
            videos.append(_video)
        return videos
    
    @classmethod
    def get_audios(
        cls,
        message: Union[Dict[str, Any], "Message"],
    ) -> List[Any]:
        # Extract all audio values from a message's content list.
        # Args: message - Message instance or message dict
        # Returns: list of audio values (paths, bytes, or ndarrays)
        if not isinstance(message, dict):
            message = message.to_dict()
        _content = message["content"]
        audios = list()
        for _c in _content:
            if not isinstance(_c, dict):
                _c = _c.to_dict()
            if _c["type"] != "audio":
                continue
            _audio = AudioContent.get_value(_c)
            audios.append(_audio)
        return audios
    
    @classmethod
    def preprocess_message(
        cls,
        message: Dict[str, Any],
        remove_audio: bool = False,
        content_fields_audio: Optional[Dict[str, Any]] = None,
        allowed_audio_format: Optional[List[Union[AudioFormat, str]]] = None,
        remove_image: bool = False,
        content_fields_image: Optional[Dict[str, Any]] = None,
        allowed_image_format: Optional[List[Union[ImageFormat, str]]] = None,
        remove_text: bool = False,
        content_fields_text: Optional[Dict[str, Any]] = None,
        remove_video: bool = False,
        content_fields_video: Optional[Dict[str, Any]] = None,
        allowed_video_format: Optional[List[Union[VideoFormat, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Filter and convert content items by modality: drop unsupported types, convert media formats, and add extra fields.
        # Args: remove_* - flags to strip modalities, allowed_*_format - convert media to first matching format, content_fields_* - extra dict fields to inject
        # Returns: message dict with filtered and format-converted content list
        if isinstance(message, cls):
            message = message.to_dict()
        if message["content"] is None:
            # tool_calls, etc
            pass
        elif isinstance(message["content"], str):
            message["content"] = [{"type": "text", "value": message["content"]}]
        else:
            _new_contents = list()
            for _content in message["content"]:
                if "image" in _content["type"]:
                    if remove_image:
                        continue
                    if allowed_image_format:
                        if ImageContent.get_key(_content):
                            _image_value = ImageContent.get_value(_content)
                            _detected_format = None
                            if isinstance(_image_value, np.ndarray):
                                _detected_format = ImageFormat.NPARRAY
                            elif isinstance(_image_value, bytes):
                                _detected_format = ImageFormat.BYTES
                            elif isinstance(_image_value, str):
                                if _image_value.startswith("file://"):
                                    _image_value = _image_value.replace("file://", "")
                                if os.path.exists(_image_value):
                                    _detected_format = ImageFormat.FILEPATH
                                elif is_url(_image_value):
                                    _detected_format = ImageFormat.URL
                                elif _image_value.startswith("data:"):
                                    _detected_format = ImageFormat.BASE64_STR_PREFIX
                                else:
                                    _detected_format = ImageFormat.BASE64_STR
                            if _detected_format not in allowed_image_format:
                                _converted = None
                                for _target_format in allowed_image_format:
                                    if _target_format == ImageFormat.BASE64_STR:
                                        _image_raw = to_image_bytes(image=_image_value, encode_base64=False)
                                        _mime = image_mime_type(_image_raw)
                                        _converted = base64.standard_b64encode(_image_raw).decode("utf-8")
                                        break
                                    elif _target_format == ImageFormat.BASE64_STR_PREFIX:
                                        _image_raw = to_image_bytes(image=_image_value, encode_base64=False)
                                        _mime = image_mime_type(_image_raw)
                                        _converted = base64.standard_b64encode(_image_raw).decode("utf-8")
                                        _converted = f'data:{_mime};base64,{_converted}'
                                        break
                                    elif _target_format == ImageFormat.BYTES:
                                        _converted = to_image_bytes(image=_image_value, encode_base64=False)
                                        break
                                    elif _target_format == ImageFormat.PIL:
                                        _converted = to_pil_image(image=_image_value)
                                        break
                                    elif _target_format == ImageFormat.FILEPATH:
                                        if isinstance(_image_value, str) and os.path.exists(_image_value):
                                            _converted = _image_value
                                            _content["is_tempfile"] = False
                                        else:
                                            _image_bytes = to_image_bytes(image=_image_value, encode_base64=False)
                                            _ext = (detect_image_format(_image_bytes) or "JPEG").lower()
                                            with tempfile.NamedTemporaryFile(suffix=f'.{_ext}', delete=False) as fp:
                                                fp.write(_image_bytes)
                                                _converted = fp.name
                                            _content["is_tempfile"] = True
                                        break
                                if _converted is None:
                                    _image_raw = to_image_bytes(image=_image_value, encode_base64=False)
                                    _mime = image_mime_type(_image_raw)
                                    _converted = base64.standard_b64encode(_image_raw).decode("utf-8")
                                    _converted = f'data:{_mime};base64,{_converted}'
                                ImageContent.set_value(_content, _converted)
                    if isinstance(content_fields_image, dict):
                        for _field_name, _field_value in content_fields_image.items():
                            _content[_field_name] = _field_value
                
                if "text" in _content["type"]:
                    if remove_text: 
                        continue
                    if isinstance(content_fields_text, dict):
                        for _field_name, _field_value in content_fields_text.items():
                            _content[_field_name] = _field_value
                
                if "video" in _content["type"]:
                    if remove_video:
                        continue
                    if allowed_video_format:
                        if VideoContent.get_key(_content):
                            _video_value = VideoContent.get_value(_content)
                            _detected_format = None
                            if isinstance(_video_value, np.ndarray):
                                _detected_format = VideoFormat.NPARRAY
                            elif isinstance(_video_value, bytes):
                                _detected_format = VideoFormat.BYTES
                            elif isinstance(_video_value, str):
                                if _video_value.startswith("file://"):
                                    _video_value = _video_value.replace("file://", "")
                                if os.path.exists(_video_value):
                                    _detected_format = VideoFormat.FILEPATH
                                elif is_url(_video_value):
                                    _detected_format = VideoFormat.URL
                                elif _video_value.startswith("data:"):
                                    _detected_format = VideoFormat.BASE64_STR_PREFIX
                                else:
                                    _detected_format = VideoFormat.BASE64_STR
                            if _detected_format not in allowed_video_format:
                                _converted = None
                                for _target_format in allowed_video_format:
                                    if _target_format == VideoFormat.BASE64_STR:
                                        _video_raw = to_video_bytes(video=_video_value)
                                        _mime = video_mime_type(_video_raw)
                                        _converted = base64.standard_b64encode(_video_raw).decode("utf-8")
                                        break
                                    elif _target_format == VideoFormat.BASE64_STR_PREFIX:
                                        _video_raw = to_video_bytes(video=_video_value)
                                        _mime = video_mime_type(_video_raw)
                                        _converted = base64.standard_b64encode(_video_raw).decode("utf-8")
                                        _converted = f'data:{_mime};base64,{_converted}'
                                        break
                                    elif _target_format == VideoFormat.BYTES:
                                        _converted = to_video_bytes(video=_video_value)
                                        break
                                    elif _target_format == VideoFormat.NPARRAY:
                                        _converted, _, _ = to_nparray_video(video=_video_value)
                                        break
                                    elif _target_format == VideoFormat.FILEPATH:
                                        if isinstance(_video_value, str) and os.path.exists(_video_value):
                                            _converted = _video_value
                                            _content["is_tempfile"] = False
                                        else:
                                            _video_bytes = to_video_bytes(video=_video_value)
                                            _ext = (detect_video_format(_video_bytes) or "mp4").lower()
                                            with tempfile.NamedTemporaryFile(suffix=f'.{_ext}', delete=False) as fp:
                                                fp.write(_video_bytes)
                                                _converted = fp.name
                                            _content["is_tempfile"] = True
                                        break
                                if _converted is None:
                                    _video_raw = to_video_bytes(video=_video_value)
                                    _mime = video_mime_type(_video_raw)
                                    _converted = base64.standard_b64encode(_video_raw).decode("utf-8")
                                    _converted = f'data:{_mime};base64,{_converted}'
                                VideoContent.set_value(_content, _converted)
                    if isinstance(content_fields_video, dict):
                        for _field_name, _field_value in content_fields_video.items():
                            _content[_field_name] = _field_value
                    
                if "audio" in _content["type"]:
                    if remove_audio: 
                        continue
                    if allowed_audio_format:
                        if AudioContent.get_key(_content):
                            _audio_value = AudioContent.get_value(_content)
                            _detected_format = None
                            if isinstance(_audio_value, np.ndarray):
                                _detected_format = AudioFormat.NPARRAY
                            elif isinstance(_audio_value, bytes):
                                _detected_format = AudioFormat.BYTES
                            elif isinstance(_audio_value, str):
                                if _audio_value.startswith("file://"):
                                    _audio_value = _audio_value.replace("file://", "")
                                if os.path.exists(_audio_value):
                                    _detected_format = AudioFormat.FILEPATH
                                elif is_url(_audio_value):
                                    _detected_format = AudioFormat.URL
                                elif _audio_value.startswith("data:"):
                                    _detected_format = AudioFormat.BASE64_STR_PREFIX
                                else:
                                    _detected_format = AudioFormat.BASE64_STR

                            if _detected_format not in allowed_audio_format:
                                _converted = None
                                for _target_format in allowed_audio_format:
                                    if _target_format == AudioFormat.BASE64_STR:
                                        _audio_raw = to_audio_bytes(audio=_audio_value, encode_base64=False)
                                        _mime = audio_mime_type(_audio_raw)
                                        _converted = base64.standard_b64encode(_audio_raw).decode("utf-8")
                                        break
                                    elif _target_format == AudioFormat.BASE64_STR_PREFIX:
                                        _audio_raw = to_audio_bytes(audio=_audio_value, encode_base64=False)
                                        _mime = audio_mime_type(_audio_raw)
                                        _converted = base64.standard_b64encode(_audio_raw).decode("utf-8")
                                        _converted = f'data:{_mime};base64,{_converted}'
                                        break
                                    elif _target_format == AudioFormat.BYTES:
                                        _converted = to_audio_bytes(audio=_audio_value, encode_base64=False)
                                        break
                                    elif _target_format == AudioFormat.NPARRAY:
                                        _converted, _, _ = to_nparray_audio(audio=_audio_value)
                                        break
                                    elif _target_format == AudioFormat.FILEPATH:
                                        if isinstance(_audio_value, str) and os.path.exists(_audio_value):
                                            _converted = _audio_value
                                            _content["is_tempfile"] = False
                                        else:
                                            _audio_bytes = to_audio_bytes(audio=_audio_value, encode_base64=False)
                                            _ext = (detect_audio_format(_audio_bytes) or "wav").lower()
                                            with tempfile.NamedTemporaryFile(suffix=f'.{_ext}', delete=False) as fp:
                                                fp.write(_audio_bytes)
                                                _converted = fp.name
                                            _content["is_tempfile"] = True
                                        break
                                if _converted is None:
                                    _audio_raw = to_audio_bytes(audio=_audio_value, encode_base64=False)
                                    _mime = audio_mime_type(_audio_raw)
                                    _converted = base64.standard_b64encode(_audio_raw).decode("utf-8")
                                    _converted = f'data:{_mime};base64,{_converted}'
                                AudioContent.set_value(_content, _converted)
                    if isinstance(content_fields_audio, dict):
                        for _field_name, _field_value in content_fields_audio.items():
                            _content[_field_name] = _field_value
                
                _new_contents.append(_content)
            message["content"] = _new_contents
        return message