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

from enum import Enum


class Modality(str, Enum):
    audio: str = "audio"
    image: str = "image"
    text: str = "text"
    video: str = "video"


class ImageFormat(str, Enum):
    BASE64_STR = "base64_str"
    BASE64_STR_PREFIX = "base64_str_prefix"
    BYTES = "bytes"
    PIL = "pil"
    FILEPATH = "filepath"
    URL = "url"
    NPARRAY = "nparray"


class VideoFormat(str, Enum):
    BASE64_STR = "base64_str"
    BASE64_STR_PREFIX = "base64_str_prefix"
    BYTES = "bytes"
    NPARRAY = "nparray"
    FILEPATH = "filepath"
    URL = "url"


class AudioFormat(str, Enum):
    BASE64_STR = "base64_str"
    BASE64_STR_PREFIX = "base64_str_prefix"
    BYTES = "bytes"
    NPARRAY = "nparray"
    FILEPATH = "filepath"
    URL = "url"
