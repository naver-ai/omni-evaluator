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

import os
from typing import Any, Dict

from omni_evaluator import ApiGroup
from omni_evaluator.api.anthropic import get_engine_features as get_engine_features_anthropic
from omni_evaluator.api.google import get_engine_features as get_engine_features_google
from omni_evaluator.api.openai import get_engine_features as get_engine_features_openai


def get_api_group(
    api_name: str,
) -> str:
    """Determine the API provider group (openai, anthropic, google) from the model name."""
    _name = api_name.lower()
    if (
        _name.startswith(("gpt-", "chatgpt-"))
        or _name.startswith(("o1", "o3", "o4"))
    ):
        return ApiGroup.openai
    elif _name.startswith("claude"):
        return ApiGroup.anthropic
    elif _name.startswith("gemini"):
        return ApiGroup.google
    else:
        raise ValueError(f'unsupported api_name: {api_name}')


def get_client(
    api_name: str,
    do_async: bool = False,
) -> Any:
    """Create and return an API client for the given provider."""
    client = None
    api_group = get_api_group(api_name=api_name)
    if api_group == ApiGroup.openai:
        import openai
        if do_async:
            client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY", None),
                organization=os.getenv("OPENAI_ORGANIZATION", None),
            )
        else:
            client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", None),
                organization=os.getenv("OPENAI_ORGANIZATION", None),
            )
    elif api_group == ApiGroup.anthropic:
        import anthropic
        if do_async:
            client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY", None),
            )
        else:
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY", None),
            )
    elif api_group == ApiGroup.google:
        from google import genai
        client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY", None),
        )
        if do_async:
            client = client.aio
    else:
        raise ValueError(f'unsupported api_group: {api_group}')

    return client


def get_engine_features(
    api_name: str,
) -> Dict:
    """Return a dict of engine feature flags for the given API model."""
    api_group = get_api_group(api_name=api_name)
    if api_group == ApiGroup.anthropic:
        return get_engine_features_anthropic(
            api_name=api_name,
        )
    elif api_group == ApiGroup.google:
        return get_engine_features_google(
            api_name=api_name,
        )
    elif api_group == ApiGroup.openai:
        return get_engine_features_openai(
            api_name=api_name,
        )
    else:
        raise ValueError(f'unsupported api_group: {api_group}')
