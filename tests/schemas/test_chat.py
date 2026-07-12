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

"""Unit tests for schemas/chat.py — Content / Message / ToolCall serialization."""
from PIL import Image
import pytest

import omni_evaluator.schemas.chat as chat
from omni_evaluator.schemas.chat import (
    AudioContent,
    ENTITY_PREFIX,
    EntityToken,
    ImageContent,
    Message,
    OCR_PREFIX,
    OcrToken,
    TextContent,
    ToolCall,
    ToolCallFunction,
    VideoContent,
)

from .test_schemas_common import _ContentMixin, _SchemaMixin


# ============================================================================
# Content classes — common contract (_ContentMixin/_SchemaMixin) + provider shape
# ============================================================================


class TestTextContent(_ContentMixin, _SchemaMixin):
    @pytest.fixture
    def content_cls(self):
        return TextContent

    @pytest.fixture
    def sample_content(self):
        return {"type": "text", "value": "Hello"}

    @pytest.fixture
    def schema_instance(self):
        return TextContent(value="Hello")

    def test_provider_templates(self):
        """openai/anthropic/google/hf map value→text key, default/json preserve value, openai_response uses input_text."""
        for template in ("openai", "anthropic", "google", "hf"):
            assert TextContent(value="Hi").to_dict(template=template) == {
                "type": "text",
                "text": "Hi",
            }
        assert TextContent(value="Hi").to_dict(template="openai_response") == {
            "type": "input_text",
            "text": "Hi",
        }
        assert TextContent(value="Hi").to_dict() == {"type": "text", "value": "Hi"}


class TestImageContent(_ContentMixin, _SchemaMixin):
    @pytest.fixture
    def content_cls(self):
        return ImageContent

    @pytest.fixture
    def sample_content(self):
        return {"type": "image", "value": "https://example.com/i.png"}

    @pytest.fixture
    def schema_instance(self):
        return ImageContent(value="https://example.com/i.png")

    def test_url_templates(self):
        """url source: openai→image_url.url, anthropic→source(url), hf→image, google/default→value preserved."""
        url = "https://example.com/i.png"
        assert ImageContent(value=url).to_dict(template="openai") == {
            "type": "image_url",
            "image_url": {"url": url},
        }
        assert ImageContent(value=url).to_dict(template="anthropic") == {
            "type": "image",
            "source": {"type": "url", "url": url},
        }
        assert ImageContent(value=url).to_dict(template="hf") == {"type": "image", "image": url}
        assert ImageContent(value=url).to_dict(template="google") == {"type": "image", "value": url}

    def test_pil_to_base64(self, tiny_image):
        """PIL source: openai encodes as data-uri base64, anthropic encodes as source.data + media_type."""
        openai = ImageContent(value=tiny_image).to_dict(template="openai")
        assert openai["type"] == "image_url"
        assert openai["image_url"]["url"].startswith("data:image/")

        anthropic = ImageContent(value=tiny_image).to_dict(template="anthropic")
        src = anthropic["source"]
        assert src["type"] == "base64"
        assert src["media_type"].startswith("image/")
        assert isinstance(src["data"], str) and src["data"]

    def test_nested_value_get(self):
        """When value is a dict({'image':...}), NESTED_KEYS('image') extracts the inner value (vLLM image_aux path)."""
        content = {"type": "image", "value": {"image": "u", "lens": "k"}}
        assert ImageContent.get_value(content) == "u"


class TestAudioContent(_ContentMixin, _SchemaMixin):
    @pytest.fixture
    def content_cls(self):
        return AudioContent

    @pytest.fixture
    def sample_content(self):
        return {"type": "audio", "value": "https://example.com/a.wav"}

    @pytest.fixture
    def schema_instance(self):
        return AudioContent(value="https://example.com/a.wav")

    def test_url_templates(self):
        """url source: anthropic→source.url, hf→audio, google/default→value preserved."""
        url = "https://example.com/a.wav"
        assert AudioContent(value=url).to_dict(template="anthropic") == {
            "type": "audio",
            "source": {"url": url},
        }
        assert AudioContent(value=url).to_dict(template="hf") == {"type": "audio", "audio": url}
        assert AudioContent(value=url).to_dict(template="google") == {"type": "audio", "value": url}

    def test_openai_structure(self, monkeypatch):
        """openai: assembles codec result into input_audio{data, format} (codec itself is covered by utils tests)."""
        monkeypatch.setattr(chat, "detect_audio_format", lambda audio_bytes: "wav")
        monkeypatch.setattr(chat, "to_audio_bytes", lambda audio, encode_base64: "B64")
        assert AudioContent(value=b"raw").to_dict(template="openai") == {
            "type": "input_audio",
            "input_audio": {"data": "B64", "format": "wav"},
        }


class TestVideoContent(_ContentMixin, _SchemaMixin):
    @pytest.fixture
    def content_cls(self):
        return VideoContent

    @pytest.fixture
    def sample_content(self):
        return {"type": "video", "value": "https://example.com/v.mp4"}

    @pytest.fixture
    def schema_instance(self):
        return VideoContent(value="https://example.com/v.mp4")

    def test_url_templates(self):
        """url source: openai→video_url.url, anthropic→source.url, hf→video, google/default→value preserved."""
        url = "https://example.com/v.mp4"
        assert VideoContent(value=url).to_dict(template="openai") == {
            "type": "video_url",
            "video_url": {"url": url},
        }
        assert VideoContent(value=url).to_dict(template="anthropic") == {
            "type": "video",
            "source": {"url": url},
        }
        assert VideoContent(value=url).to_dict(template="hf") == {"type": "video", "video": url}
        assert VideoContent(value=url).to_dict(template="google") == {"type": "video", "value": url}


# ============================================================================
# OcrToken / EntityToken — required-arg branching in custom __init__
# ============================================================================


class TestOcrToken(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return OcrToken(id="1", text="foo")

    def test_missing_required_raises(self):
        """When a field without a default (id) is omitted, the custom __init__ raises TypeError."""
        with pytest.raises(TypeError):
            OcrToken(text="foo")


class TestEntityToken(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return EntityToken(id="1", text="foo")

    def test_missing_required_raises(self):
        """When a field without a default (id) is omitted, the custom __init__ raises TypeError."""
        with pytest.raises(TypeError):
            EntityToken(text="foo")


# ============================================================================
# ToolCall — Hermes {name, arguments} serialization
# ============================================================================


class TestToolCallFunction(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return ToolCallFunction(name="f", arguments={"a": 1})

    def test_arguments_normalized(self):
        """If arguments is a dict it is converted to a JSON string; if already a str it is kept as-is."""
        assert ToolCallFunction(name="f", arguments={"a": 1}).arguments == '{"a": 1}'
        assert ToolCallFunction(name="f", arguments='{"a": 1}').arguments == '{"a": 1}'


class TestToolCall(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return ToolCall(
            id="c1",
            type="function",
            function=ToolCallFunction(name="f", arguments={"a": 1}),
        )


# ============================================================================
# Message — content normalization / template dispatch / extraction / preprocess
# ============================================================================


class TestMessage(_SchemaMixin):
    @pytest.fixture
    def schema_instance(self):
        return Message(role="user", content="hi")

    def test_str_content_wrapped(self):
        """str content is wrapped into text content by __post_init__, and the value key is preserved without a template."""
        assert Message(role="user", content="hi").to_dict() == {
            "role": "user",
            "content": [{"type": "text", "value": "hi"}],
        }

    def test_drops_none_fields(self):
        """Optional fields other than content that are None (name/tool_calls etc.) are removed from the output dict."""
        assert set(Message(role="user", content="hi").to_dict()) == {"role", "content"}

    def test_delegates_content_template(self, sample_text_message_dict):
        """Message.to_dict(template=...) delegates content conversion to the Content class (verified with one representative)."""
        out = Message(**sample_text_message_dict).to_dict(template="openai")
        assert out["content"] == [{"type": "text", "text": "Hello"}]

    def test_name_prefix(self):
        """anthropic/google prepend name as `[name]: ` before the first text content and remove the name key."""
        out = Message(
            role="user", content=[{"type": "text", "value": "hi"}], name="bob"
        ).to_dict(template="anthropic")
        assert out["content"][0]["text"] == "[bob]: hi"
        assert "name" not in out

    def test_image_ocr_entity_split(self):
        """ocr/entity inside an image value (dict) are split into separate text content entries prefixed with OCR_PREFIX/ENTITY_PREFIX."""
        msg = Message(
            role="user",
            content=[
                {"type": "image", "value": {"image": "https://x/i.png", "ocr": ["a", "b"], "entity": ["e1"]}},
            ],
        )
        values = [c["value"] for c in msg.to_dict()["content"] if c["type"] == "text"]
        assert any(v.startswith(OCR_PREFIX) for v in values)
        assert any(v.startswith(ENTITY_PREFIX) for v in values)

    def test_tool_calls_preserved(self):
        """tool_calls are preserved even when content is None, and ToolCallFunction.arguments (dict) is serialized as a JSON string."""
        tc = ToolCall(id="c1", type="function", function=ToolCallFunction(name="f", arguments={"a": 1}))
        out = Message(role="assistant", content=None, tool_calls=[tc]).to_dict(template="openai")
        assert out["content"] is None
        assert out["tool_calls"][0] == {
            "id": "c1",
            "type": "function",
            "function": {"name": "f", "arguments": '{"a": 1}'},
        }

    def test_get_query_text_only(self):
        """Joins only text content with newlines, skipping non-text content."""
        msg = Message(
            role="user",
            content=[
                {"type": "image", "value": "https://x/i.png"},
                {"type": "text", "value": "q1"},
                {"type": "text", "value": "q2"},
            ],
        )
        assert Message.get_query(msg) == "q1\nq2"

    def test_get_query_skips_ocr_entity(self):
        """Text starting with OCR_PREFIX / ENTITY_PREFIX is excluded from the query."""
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "value": f"{OCR_PREFIX} foo"},
                {"type": "text", "value": f"{ENTITY_PREFIX} bar"},
                {"type": "text", "value": "real"},
            ],
        }
        assert Message.get_query(msg) == "real"

    def test_preprocess_remove_modalities(self):
        """remove_* flags remove the corresponding modality content (each branch verified with an independent input)."""
        base = lambda: {
            "role": "user",
            "content": [
                {"type": "image", "value": "https://x/i.png"},
                {"type": "text", "value": "t"},
                {"type": "audio", "value": "https://x/a.wav"},
                {"type": "video", "value": "https://x/v.mp4"},
            ],
        }
        types = lambda m: [c["type"] for c in m["content"]]
        assert types(Message.preprocess_message(base(), remove_image=True)) == ["text", "audio", "video"]
        assert types(Message.preprocess_message(base(), remove_text=True)) == ["image", "audio", "video"]
        assert types(Message.preprocess_message(base(), remove_audio=True)) == ["image", "text", "video"]
        assert types(Message.preprocess_message(base(), remove_video=True)) == ["image", "text", "audio"]

    def test_preprocess_injects_content_fields(self):
        """content_fields_* injects additional keys into the corresponding modality content dict."""
        msg = {"role": "user", "content": [{"type": "text", "value": "t"}]}
        out = Message.preprocess_message(msg, content_fields_text={"x": 1})
        assert out["content"][0] == {"type": "text", "value": "t", "x": 1}

    def test_preprocess_str_content_normalized(self):
        """str content is normalized into a single-element text content list."""
        out = Message.preprocess_message({"role": "user", "content": "hi"})
        assert out["content"] == [{"type": "text", "value": "hi"}]

    def test_preprocess_none_content_passthrough(self):
        """When content is None (e.g., tool_calls only), it passes through unchanged."""
        out = Message.preprocess_message({"role": "assistant", "content": None})
        assert out["content"] is None
