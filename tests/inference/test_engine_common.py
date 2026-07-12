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

"""Base class for `engine.main()` contract tests common to all inference engines."""
from __future__ import annotations

from typing import Any, Dict

import pytest

from omni_evaluator.inference import NUM_DEBUG_SAMPLES


def _get_content_list(message):
    """Returns the content list regardless of whether the message received by the boundary is a Message instance or a dict."""
    if hasattr(message, "content"):
        return message.content
    return message["content"]


def _apply_output_to_record(record, output):
    """Applies the output dict (prediction, etc.) from the boundary fake to the record in-place.

    After merging, the batch boundary receives `records=` and fills each record with inference
    results, then returns the records list as-is, and `engine.main()` returns that list of Records.
    To mimic this contract in the fake, each key in the output dict is set on the record's
    corresponding field.
    """
    for _key, _value in output.items():
        if hasattr(record, _key):
            setattr(record, _key, _value)


class EngineMainCommonTests:
    """Contract tests for `engine.main()` independent of engine type.
    """

    # ── fixtures to be overridden by child classes ─────────────────────────

    @pytest.fixture
    def engine_module(self):
        raise NotImplementedError(
            "subclass must override `engine_module` fixture to return the engine module"
        )

    @pytest.fixture
    def main_kwargs_factory(self):
        raise NotImplementedError(
            "subclass must override `main_kwargs_factory` fixture "
            "→ (records, task_config, **overrides) returns engine.main() kwargs dict"
        )

    @pytest.fixture
    def patch_boundary(self):
        """Boundary monkeypatch helper. Child implements to match its own boundary shape."""
        raise NotImplementedError(
            "subclass must override `patch_boundary` fixture "
            "→ (output_fn=None) returns calls list after monkeypatching boundary"
        )

    @pytest.fixture
    def fake_inference_output(self):
        """Single-record response dict builder. Override if additional keys are needed per engine."""
        def _factory(prediction: str = "x") -> Dict[str, Any]:
            return {"prediction": prediction, "tool_calls": None, "latency": 0.0}
        return _factory

    # ── common tests ────────────────────────────────────────────

    def test_fills_predictions(
        self,
        engine_module,
        main_kwargs_factory, fake_inference_output, patch_boundary,
        record_factory, task_config_factory,
    ):
        """prediction/tool_calls/latency returned by the boundary are reflected in the record."""
        records = record_factory(n=2)

        def _output(idx, messages):
            return fake_inference_output(prediction=f"pred-{idx}")

        patch_boundary(output_fn=_output)

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=2),
        ))

        assert isinstance(out, list) and len(out) == 2
        assert [r["prediction"] for r in out] == ["pred-0", "pred-1"]

    def test_debug_truncates(
        self,
        engine_module,
        main_kwargs_factory, patch_boundary,
        record_factory, task_config_factory,
    ):
        """With `debug=True`, only the first `NUM_DEBUG_SAMPLES` records reach the boundary."""
        n = NUM_DEBUG_SAMPLES + 2
        records = record_factory(n=n)
        calls = patch_boundary()

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=n),
            debug=True,
        ))

        assert len(calls) == NUM_DEBUG_SAMPLES
        assert isinstance(out, list) and len(out) == NUM_DEBUG_SAMPLES

    def test_partial_failure(
        self,
        engine_module,
        main_kwargs_factory, fake_inference_output, patch_boundary,
        record_factory, task_config_factory,
    ):
        """Even when the boundary returns None for some slots, only valid slots are filled and the total record count is preserved.
        """
        n = NUM_DEBUG_SAMPLES  # exactly matches debug=True
        records = record_factory(n=n)

        def _output(idx, messages):
            if idx == 1:
                return None
            return fake_inference_output(prediction=f"ok-{idx}")

        patch_boundary(output_fn=_output)

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=n),
        ))

        assert isinstance(out, list) and len(out) == n
        assert out[0]["prediction"] == "ok-0"
        assert out[1]["prediction"] is None
        assert out[2]["prediction"] == "ok-2"

    def test_output_schema(
        self,
        engine_module,
        main_kwargs_factory, fake_inference_output, patch_boundary,
        record_factory, task_config_factory,
    ):
        """The output dict preserves input fields and contains all common fields that `main` must populate.
        """
        records = record_factory(n=1)
        patch_boundary(output_fn=lambda idx, m: fake_inference_output(prediction="hello"))

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=1),
        ))

        assert isinstance(out, list) and len(out) == 1
        rec = out[0].to_dict(template="json")          # main returns Record objects — validate after serialization
        expected_keys = {
            "benchmark", "index", "messages",         # preserved from input
            "prediction", "tool_calls", "latency",    # set by main (common)
        }
        missing = expected_keys - rec.keys()
        assert not missing, f"output dict missing keys: {missing}"
        assert rec["benchmark"] == "dummy"
        assert rec["index"] == "0"
        assert rec["prediction"] == "hello"

    # ── multimodal input → text output verification ─────────────────────
    # Verifies that content types reach the boundary without loss. Semantics are out of scope.

    def test_image_input(
        self,
        engine_module,
        main_kwargs_factory, fake_inference_output, patch_boundary,
        image_record_factory, task_config_factory,
    ):
        """image+text content reaches the boundary without type loss and a text prediction is filled into the record."""
        records = image_record_factory(n=1)
        calls = patch_boundary(
            output_fn=lambda idx, m: fake_inference_output(prediction="a red square")
        )

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=1),
        ))

        assert len(calls) == 1
        content_types = [c["type"] for c in _get_content_list(calls[0]["messages"][0])]
        assert "image" in content_types, f"expected image, got {content_types}"
        assert "text" in content_types

        assert isinstance(out, list) and len(out) == 1
        assert out[0]["prediction"] == "a red square"

    def test_video_input(
        self,
        engine_module,
        main_kwargs_factory, fake_inference_output, patch_boundary,
        video_record_factory, task_config_factory,
    ):
        """video+text content reaches the boundary without type loss and a text prediction is filled into the record.
        """
        records = video_record_factory(n=1)
        calls = patch_boundary(
            output_fn=lambda idx, m: fake_inference_output(prediction="a black screen")
        )

        out = engine_module.main(**main_kwargs_factory(
            records=records,
            task_config=task_config_factory(num_records=1),
        ))

        assert len(calls) == 1
        content_types = [c["type"] for c in _get_content_list(calls[0]["messages"][0])]
        assert "video" in content_types, f"expected video, got {content_types}"
        assert "text" in content_types

        assert isinstance(out, list) and len(out) == 1
        assert out[0]["prediction"] == "a black screen"
