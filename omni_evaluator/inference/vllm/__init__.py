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

from omni_evaluator import AudioFormat, ImageFormat, VideoFormat
from omni_evaluator.schemas.inference import InferenceEngineFeatures

ENGINE_FEATURES = InferenceEngineFeatures(
    support_audio_understanding=True,
    support_image_understanding=True,
    support_text_understanding=True,
    support_video_understanding=True,
    support_audio_generation=False,
    support_text_generation=True,
    support_compute_perplexity=True,  # has a prompt_logprobs path — completions.py:327-399
).to_dict()

# Allowed media formats in priority order (first = highest priority for conversion).
# FILEPATH is appended at call-time when allowed_local_media_path is provided.
ALLOWED_AUDIO_FORMAT = [AudioFormat.URL, AudioFormat.BASE64_STR_PREFIX]
ALLOWED_IMAGE_FORMAT = [ImageFormat.URL, ImageFormat.BASE64_STR_PREFIX]
ALLOWED_VIDEO_FORMAT = [VideoFormat.URL, VideoFormat.BASE64_STR_PREFIX]

from .engine import main
