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

"""Unit tests for omni_evaluator/schemas/inference.py — Record creation, __post_init__ validation, to_dict serialization, and InferenceEngineFeatures default validation."""

from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import pytest

from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import (
    ApiInferenceOutput,
    HuggingfaceInferenceOutput,
    InferenceEngineFeatures,
    InferenceOutput,
    Record,
    SglangInferenceOutput,
    VllmInferenceOutput,
)

from .test_schemas_common import _InferenceOutputMixin, _SchemaMixin


@pytest.fixture
def tiny_image():
    """2x2 solid-color PIL image — for prediction serialization validation (codec call is in-memory so deterministic)."""
    return Image.new("RGB", (2, 2), (255, 0, 0))


# ============================================================================
# Per-schema — common mixin contract + specialized branches.
# ============================================================================


class TestRecord(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return Record(messages=[])  # empty list required since to_dict iterates over messages

    def test_post_init_native_conversion(self):
        """__post_init__ converts numpy scalars and OmegaConf containers to Python native types."""
        record = Record(
            perplexities=np.int64(7),
            latency=np.float64(1.5),
            options=OmegaConf.create([1, 2, 3]),
            metrics=OmegaConf.create({"score": 0.9}),
        )
        assert type(record.perplexities) is int and record.perplexities == 7
        assert type(record.latency) is float and record.latency == 1.5
        assert type(record.options) is list and record.options == [1, 2, 3]
        assert type(record.metrics) is dict and record.metrics == {"score": 0.9}

    def test_post_init_tools_valid(self):
        """Hermes-format tools (tool=str, arguments=dict) pass through unchanged."""
        record = Record(tools=[{"tool": "search", "arguments": {"q": "x"}}])
        assert record.tools == [{"tool": "search", "arguments": {"q": "x"}}]

    def test_post_init_tools_invalid(self):
        """Non-Hermes tools raise ValueError — tool name not str / arguments not dict, two paths."""
        with pytest.raises(ValueError):
            Record(tools=[{"tool": 123}])
        with pytest.raises(ValueError):
            Record(tools=[{"tool": "f", "arguments": [1, 2]}])

    def test_to_dict_messages(self):
        """to_dict serializes both dict and ChatMessage message forms to the same template dict."""
        expected = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        from_obj = Record(messages=[ChatMessage(role="user", content="hi")])
        from_dict = Record(messages=[{"role": "user", "content": [{"type": "text", "value": "hi"}]}])
        assert from_obj.to_dict(template="hf")["messages"] == expected
        assert from_dict.to_dict(template="hf")["messages"] == expected

    def test_to_dict_template(self):
        """The template argument determines the key name for message content (None→value, hf/openai→text)."""
        record = Record(messages=[ChatMessage(role="user", content="hi")])
        assert record.to_dict()["messages"][0]["content"][0] == {"type": "text", "value": "hi"}
        assert record.to_dict(template="hf")["messages"][0]["content"][0] == {"type": "text", "text": "hi"}
        assert record.to_dict(template="openai")["messages"][0]["content"][0] == {"type": "text", "text": "hi"}

    def test_to_dict_prediction(self, tiny_image):
        """to_dict encodes PIL images inside list and dict predictions to base64, leaving non-images unchanged."""
        list_pred = Record(messages=[], prediction=[tiny_image]).to_dict()["prediction"]
        dict_pred = Record(messages=[], prediction={"img": tiny_image}).to_dict()["prediction"]
        scalar_pred = Record(messages=[], prediction="answer").to_dict()["prediction"]
        assert isinstance(list_pred[0], str)
        assert isinstance(dict_pred["img"], str)
        assert scalar_pred == "answer"


class TestInferenceEngineFeatures(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return InferenceEngineFeatures()

    def test_defaults(self):
        """When not specified, only text understanding/generation are True; all other support_* flags are False."""
        features = InferenceEngineFeatures()
        assert features.support_text_understanding is True
        assert features.support_text_generation is True
        assert features.support_audio_understanding is False
        assert features.support_image_understanding is False
        assert features.support_video_understanding is False
        assert features.support_reasoning is False
        assert features.support_ocr is False


# ============================================================================
# 5 InferenceOutput children — no specialized branches; parametrize repeats common contract 5 times.
# ============================================================================


@pytest.mark.parametrize(
    "_output_cls",
    [
        InferenceOutput,
        HuggingfaceInferenceOutput,
        VllmInferenceOutput,
        SglangInferenceOutput,
        ApiInferenceOutput,
    ],
    ids=lambda c: c.__name__,
)
class TestInferenceOutput(_InferenceOutputMixin, _SchemaMixin):
    @pytest.fixture
    def output_cls(self, _output_cls):
        return _output_cls

    @pytest.fixture
    def schema_instance(self, _output_cls):
        return _output_cls(prediction="x")
