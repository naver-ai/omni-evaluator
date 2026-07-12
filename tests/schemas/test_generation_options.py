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

"""Unit tests for omni_evaluator/schemas/generation_options.py — engine-specific fields, from_dict key conversion, provider dispatch, and to_dict None removal."""

import pytest

from omni_evaluator.schemas.generation_options import (
    ApiAnthropicGenerationOptions,
    ApiGenerationOptions,
    ApiGoogleGenerationOptions,
    ApiOpenaiGenerationOptions,
    HuggingfaceGenerationOptions,
    SglangGenerationOptions,
    VllmGenerationOptions,
)

from .test_schemas_common import _GenerationOptionsMixin, _SchemaMixin


# ============================================================================
# Each leaf child — common mixin contract (_GenerationOptionsMixin + _SchemaMixin) + engine-specific branching.
# ============================================================================


class TestHuggingfaceGenerationOptions(_GenerationOptionsMixin, _SchemaMixin):
    @pytest.fixture
    def options_cls(self):
        return HuggingfaceGenerationOptions

    @pytest.fixture
    def schema_instance(self):
        return HuggingfaceGenerationOptions()

    def test_fields(self):
        """Specialized fields (beam/sampling + image-gen) exist and default to None."""
        o = HuggingfaceGenerationOptions()
        assert o.inference_engine == "huggingface"
        assert o.num_beams is None and o.do_sample is None and o.top_k is None
        assert o.stop_words is None and o.max_new_tokens is None
        # image generation exclusive fields
        assert o.height is None and o.width is None and o.num_inference_steps is None

    def test_from_dict(self):
        """from_dict: temperature==0.0 → do_sample=False, unsupported keys (frequency_penalty, etc.) are dropped."""
        out = HuggingfaceGenerationOptions.from_dict(
            {"temperature": 0.0, "max_new_tokens": 10, "frequency_penalty": 1, "seed": 7}
        ).to_dict()
        assert out == {"do_sample": False, "temperature": 0.0, "max_new_tokens": 10}


class TestVllmGenerationOptions(_GenerationOptionsMixin, _SchemaMixin):
    @pytest.fixture
    def options_cls(self):
        return VllmGenerationOptions

    @pytest.fixture
    def schema_instance(self):
        return VllmGenerationOptions()

    def test_to_dict_drops_none(self):
        """`to_dict()` removes `inference_engine` and None fields, and retains empty list default (`stop`)."""
        out = VllmGenerationOptions(temperature=0.5).to_dict()
        assert out == {"temperature": 0.5, "stop": []}  # inference_engine removed, None fields removed, stop=[] retained

    def test_from_dict_filters(self):
        """`from_dict` keeps only items with valid field names and non-None values (unknown keys and None values are ignored)."""
        out = VllmGenerationOptions.from_dict(
            {"temperature": 0.5, "top_p": None, "bogus": 1}
        ).to_dict()
        assert out == {"temperature": 0.5, "stop": []}

    def test_fields(self):
        """Specialized fields exist, and only `stop` has an empty list default."""
        o = VllmGenerationOptions()
        assert o.inference_engine == "vllm"
        assert o.n is None and o.max_tokens is None and o.prompt_logprobs is None and o.seed is None
        assert o.stop == []  # default_factory=list

    def test_from_dict(self):
        """from_dict: max_new_tokens→max_tokens, stop_words→stop, num_beams/length_penalty are dropped."""
        out = VllmGenerationOptions.from_dict(
            {"max_new_tokens": 20, "stop_words": ["x"], "num_beams": 3, "length_penalty": 1.0}
        ).to_dict()
        assert out == {"max_tokens": 20, "stop": ["x"]}


class TestSglangGenerationOptions(_GenerationOptionsMixin, _SchemaMixin):
    @pytest.fixture
    def options_cls(self):
        return SglangGenerationOptions

    @pytest.fixture
    def schema_instance(self):
        return SglangGenerationOptions()

    def test_fields(self):
        """Specialized fields exist, and only `stop` has an empty list default (a separate class with the same shape as vllm)."""
        o = SglangGenerationOptions()
        assert o.inference_engine == "sglang"
        assert o.n is None and o.max_tokens is None and o.prompt_logprobs is None and o.seed is None
        assert o.stop == []

    def test_from_dict(self):
        """from_dict: same conversion as vLLM (max_new_tokens→max_tokens, stop_words→stop) via a separate code path."""
        out = SglangGenerationOptions.from_dict(
            {"max_new_tokens": 20, "stop_words": ["x"], "num_beams": 3, "length_penalty": 1.0}
        ).to_dict()
        assert out == {"max_tokens": 20, "stop": ["x"]}


class TestApiOpenaiGenerationOptions(_GenerationOptionsMixin, _SchemaMixin):
    @pytest.fixture
    def options_cls(self):
        return ApiOpenaiGenerationOptions

    @pytest.fixture
    def schema_instance(self):
        return ApiOpenaiGenerationOptions()

    def test_fields(self):
        """Specialized fields (frequency/presence_penalty, max_output_tokens, reasoning) default to None."""
        o = ApiOpenaiGenerationOptions()
        assert o.inference_engine == "api/openai"
        assert o.frequency_penalty is None and o.presence_penalty is None
        assert o.max_output_tokens is None and o.reasoning is None

    def test_from_dict(self):
        """from_dict: repetition_penalty→frequency_penalty, max_new_tokens→max_tokens, logprobs int→bool(+top_logprobs), top_k dropped."""
        out = ApiOpenaiGenerationOptions.from_dict(
            {"repetition_penalty": 1.2, "max_new_tokens": 5, "logprobs": 3, "top_k": 9}
        ).to_dict()
        assert out == {
            "frequency_penalty": 1.2,
            "max_tokens": 5,
            "logprobs": True,
            "top_logprobs": 3,
        }


class TestApiAnthropicGenerationOptions(_GenerationOptionsMixin, _SchemaMixin):
    @pytest.fixture
    def options_cls(self):
        return ApiAnthropicGenerationOptions

    @pytest.fixture
    def schema_instance(self):
        return ApiAnthropicGenerationOptions()

    def test_fields(self):
        """Specialized fields (stop_sequences, thinking, top_k) default to None."""
        o = ApiAnthropicGenerationOptions()
        assert o.inference_engine == "api/anthropic"
        assert o.top_k is None and o.stop_sequences is None and o.thinking is None

    def test_from_dict(self):
        """from_dict: max_new_tokens→max_tokens, stop_words→stop_sequences, top_p is dropped when temperature and top_p are both specified."""
        out = ApiAnthropicGenerationOptions.from_dict(
            {"max_new_tokens": 5, "stop_words": ["a"], "temperature": 0.7, "top_p": 0.9}
        ).to_dict()
        assert out == {"temperature": 0.7, "max_tokens": 5, "stop_sequences": ["a"]}


class TestApiGoogleGenerationOptions(_GenerationOptionsMixin, _SchemaMixin):
    @pytest.fixture
    def options_cls(self):
        return ApiGoogleGenerationOptions

    @pytest.fixture
    def schema_instance(self):
        return ApiGoogleGenerationOptions()

    def test_fields(self):
        """camelCase specialized fields (topK/topP/maxOutputTokens/stopSequences/thinking_config) default to None."""
        o = ApiGoogleGenerationOptions()
        assert o.inference_engine == "api/google"
        assert o.topK is None and o.topP is None and o.maxOutputTokens is None
        assert o.stopSequences is None and o.thinking_config is None and o.responseLogprobs is None

    def test_from_dict(self):
        """from_dict: top_p→topP, top_k→topK, repetition_penalty→frequencyPenalty, max_new_tokens→maxOutputTokens, logprobs int→responseLogprobs=True."""
        out = ApiGoogleGenerationOptions.from_dict(
            {"top_p": 0.8, "top_k": 5, "repetition_penalty": 1.1,
             "max_new_tokens": 7, "logprobs": 2, "presence_penalty": 0.3}
        ).to_dict()
        assert out == {
            "logprobs": 2,
            "responseLogprobs": True,
            "frequencyPenalty": 1.1,
            "presencePenalty": 0.3,
            "topK": 5,
            "topP": 0.8,
            "maxOutputTokens": 7,
            "stopSequences": [],
        }


# ============================================================================
# ApiGenerationOptions — dispatcher (not a leaf, _GenerationOptionsMixin not applied, §5).
# provider dispatch + per-provider to_dict template.
# ============================================================================


class TestApiGenerationOptions:
    def test_dispatch_by_api_group(self):
        """api_group string routes to each provider child class."""
        assert isinstance(
            ApiGenerationOptions.from_dict({}, api_name="gpt-4o", api_group="openai"),
            ApiOpenaiGenerationOptions,
        )
        assert isinstance(
            ApiGenerationOptions.from_dict({}, api_name="claude", api_group="anthropic"),
            ApiAnthropicGenerationOptions,
        )
        assert isinstance(
            ApiGenerationOptions.from_dict({}, api_name="gemini", api_group="google"),
            ApiGoogleGenerationOptions,
        )

    def test_dispatch_by_inference_engine(self):
        """inference_engine string ("api/openai", etc.) also routes to the same provider child."""
        assert isinstance(
            ApiGenerationOptions.from_dict({}, api_name="gpt-4o", inference_engine="api/openai"),
            ApiOpenaiGenerationOptions,
        )
        assert isinstance(
            ApiGenerationOptions.from_dict({}, api_name="claude", inference_engine="api/anthropic"),
            ApiAnthropicGenerationOptions,
        )
        assert isinstance(
            ApiGenerationOptions.from_dict({}, api_name="gemini", inference_engine="api/google"),
            ApiGoogleGenerationOptions,
        )

    def test_dispatch_requires_selector(self):
        """Raises ValueError when neither api_group nor inference_engine is provided."""
        with pytest.raises(ValueError):
            ApiGenerationOptions.from_dict({}, api_name="x")

    def test_dispatch_unsupported(self):
        """Raises ValueError for a selector that matches no provider."""
        with pytest.raises(ValueError):
            ApiGenerationOptions.from_dict({}, api_name="x", inference_engine="vllm")

    def test_to_dict_template(self):
        """to_dict(template) serializes with per-provider key names (openai/anthropic/google)."""
        api = ApiGenerationOptions(temperature=0.5, top_p=0.9, top_k=4, max_tokens=10, stop=["z"])
        assert api.to_dict("openai") == {
            "temperature": 0.5, "top_p": 0.9, "max_tokens": 10, "stop": ["z"],
        }
        assert api.to_dict("anthropic") == {
            "temperature": 0.5, "top_k": 4, "top_p": 0.9, "max_tokens": 10, "stop_sequences": ["z"],
        }
        assert api.to_dict("google") == {
            "temperature": 0.5, "topK": 4, "topP": 0.9, "maxOutputTokens": 10, "stopSequences": ["z"],
        }
