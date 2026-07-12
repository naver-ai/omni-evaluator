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

from omni_evaluator.api.model_supporting import get_model_supporting
from omni_evaluator.schemas.inference import InferenceEngineFeatures


def get_engine_features(api_name: str):
    caps = get_model_supporting(provider="openai", api_name=api_name)
    return InferenceEngineFeatures(**caps).to_dict()

def is_chat_model(api_name: str):
    is_chat_model = True
    if api_name in ["gpt-3.5-turbo-instruct", ]:
        is_chat_model = False
    return is_chat_model