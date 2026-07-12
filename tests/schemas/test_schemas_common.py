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

"""schemas area common test base — interface contract mixins and their self-verification probes collected in one file."""
from __future__ import annotations

from dataclasses import dataclass, fields
import json
from typing import Any, ClassVar, Dict, List, Optional

import pytest

from omni_evaluator.schemas import SchemaInterface
from omni_evaluator.schemas.chat import ContentInterface


# ============================================================================
# Common contract base — activated only when a child `TestX` inherits + fills fixtures (not collected).
# ============================================================================


class _SchemaMixin:
    """`SchemaInterface` child common contract — verifies only the surface that children do not override."""

    @pytest.fixture
    def schema_instance(self):
        raise NotImplementedError("Child TestX must provide the schema_instance fixture")

    def test_dict_like_access(self, schema_instance):
        """__getitem__/__setitem__/__contains__ accept only declared fields; anything else raises KeyError/False."""
        fname = fields(schema_instance)[0].name
        assert fname in schema_instance
        schema_instance[fname] = schema_instance[fname]  # reassign same value — field type independent
        assert "__nope__" not in schema_instance
        with pytest.raises(KeyError):
            schema_instance["__nope__"]
        with pytest.raises(KeyError):
            schema_instance["__nope__"] = 1

    def test_pop(self, schema_instance):
        """pop returns the field value and resets to default/None; unknown key without argument raises KeyError."""
        fname = fields(schema_instance)[0].name
        expected = schema_instance[fname]
        assert schema_instance.pop(fname) == expected
        assert schema_instance.pop("__nope__", "fallback") == "fallback"
        with pytest.raises(KeyError):
            schema_instance.pop("__nope__")

    def test_from_kwargs_drops_unknown(self, schema_instance):
        """from_kwargs accepts only declared fields and silently drops unknown keys."""
        cls = type(schema_instance)
        kwargs = {f.name: getattr(schema_instance, f.name) for f in fields(schema_instance)}
        kwargs["__junk__"] = object()
        obj = cls.from_kwargs(**kwargs)
        assert isinstance(obj, cls)
        assert not hasattr(obj, "__junk__")

    def test_to_dict_returns_dict(self, schema_instance):
        """to_dict() returns a dict regardless of whether template is overridden."""
        assert isinstance(schema_instance.to_dict(), dict)


class _ContentMixin:
    """Common contract for content classes (Text/Image/Audio/Video) — verifies value access and template serialization."""

    @pytest.fixture
    def content_cls(self):
        raise NotImplementedError("Child TestX must provide the content_cls fixture")

    @pytest.fixture
    def sample_content(self):
        raise NotImplementedError("Child TestX must provide the sample_content fixture")

    def test_get_key(self, content_cls, sample_content):
        """get_key returns a key present in VALUE_KEYS, or None if no value key exists."""
        assert content_cls.get_key(sample_content) in content_cls.VALUE_KEYS
        assert content_cls.get_key({"type": sample_content["type"]}) is None

    def test_get_set_value_roundtrip(self, content_cls, sample_content):
        """get_value returns the value immediately after set_value (NESTED_KEYS dict structure also preserved)."""
        content_cls.set_value(sample_content, "__sentinel__")
        assert content_cls.get_value(sample_content) == "__sentinel__"

    def test_set_key_renames(self, content_cls, sample_content):
        """set_key moves the current value key to the new key (preserving raw value, removing old key)."""
        old_key = content_cls.get_key(sample_content)
        old_raw = sample_content[old_key]
        content_cls.set_key(sample_content, "renamed")
        assert sample_content["renamed"] == old_raw
        assert old_key not in sample_content

    def test_default_templates_serializable(self, content_cls, sample_content):
        """default(None)/json template produces a JSON-serializable dict and preserves type."""
        for template in (None, "json"):
            out = content_cls.to_template(obj=dict(sample_content), template=template)
            assert out["type"] == sample_content["type"]
            json.dumps(out)  # must be serializable (no raise)


class _GenerationOptionsMixin:
    """Verifies common contract for `GenerationOptions` engine/provider leaf children."""

    @pytest.fixture
    def options_cls(self):
        raise NotImplementedError("Child TestX must provide the options_cls fixture")

    def test_to_dict_drops_inference_engine_and_none(self, options_cls):
        """to_dict() removes inference_engine and None fields (empty list from default_factory is kept)."""
        out = options_cls().to_dict()
        assert "inference_engine" not in out
        assert all(v is not None for v in out.values())

    def test_from_dict_drops_unknown(self, options_cls):
        """from_dict creates a cls instance and leaves neither unknown keys nor inference_engine in the output."""
        out = options_cls.from_dict({"__bogus__": 1})
        assert isinstance(out, options_cls)
        serialized = out.to_dict()
        assert "__bogus__" not in serialized
        assert "inference_engine" not in serialized


class _InferenceOutputMixin:
    """Common contract for `InferenceOutput` engine-specific children — verifies that to_dict removes None fields."""

    @pytest.fixture
    def output_cls(self):
        raise NotImplementedError("Child TestX must provide the output_cls fixture")

    def test_to_dict_drops_none(self, output_cls):
        """to_dict() removes all None fields and retains only set fields."""
        assert output_cls().to_dict() == {}
        assert output_cls(prediction="x", finish_reason="stop").to_dict() == {
            "prediction": "x",
            "finish_reason": "stop",
        }


# ============================================================================
# Contract self-verification — probes make minimal calls to the actual interface to pin base ↔ interface coupling.
# ============================================================================


@dataclass
class _NestedProbe(SchemaInterface):
    x: int = 1


@dataclass
class _SchemaProbe(SchemaInterface):
    a: int = 1
    b: Optional[str] = None
    nested: Optional[Any] = None


class _ContentProbe(ContentInterface):
    """Minimal probe with NESTED_KEYS — directly verifies the nested branch of ContentInterface."""

    VALUE_KEYS: ClassVar[List[str]] = ["value"]
    NESTED_KEYS: ClassVar[Dict[str, str]] = {"value": "inner"}


class TestSchemaInterface(_SchemaMixin):
    """Verifies SchemaInterface dict-like/pop/from_kwargs contracts using a probe."""

    @pytest.fixture
    def schema_instance(self):
        return _SchemaProbe(a=1, b="x", nested=_NestedProbe(x=2))

    def test_base_to_dict_flattens(self, schema_instance):
        """base to_dict recursively flattens nested dataclasses into dicts."""
        out = schema_instance.to_dict()
        assert out["a"] == 1 and out["b"] == "x"
        assert isinstance(out["nested"], dict) and out["nested"] == {"x": 2}

    def test_base_to_dict_keeps_none(self):
        """base to_dict retains None fields as keys (dropping them is the responsibility of child overrides)."""
        out = _SchemaProbe(a=1).to_dict()
        assert "b" in out and out["b"] is None


class TestContentInterface:
    """Verifies ContentInterface value key lookup, replacement, renaming, and NESTED_KEYS branching using a probe."""

    def test_get_set_key(self):
        """get_key returns an existing key from VALUE_KEYS (or None if absent); set_key renames while preserving raw value."""
        assert _ContentProbe.get_key({"type": "probe", "value": {"inner": "v"}}) == "value"
        assert _ContentProbe.get_key({"type": "probe"}) is None
        content = {"type": "probe", "value": {"inner": "v"}}
        _ContentProbe.set_key(content, "renamed")
        assert content["renamed"] == {"inner": "v"} and "value" not in content

    def test_nested_unwrap_preserves_dict(self):
        """When NESTED_KEYS is present and value is a dict, get_value extracts the inner key and set_value preserves the dict structure."""
        content = {"type": "probe", "value": {"inner": "v"}}
        assert _ContentProbe.get_value(content) == "v"
        _ContentProbe.set_value(content, "w")
        assert content["value"] == {"inner": "w"}
