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

"""Regression tests for app/services/inference_builder.py.

Focus: build_inference_output_from_eval ground-truth resolution
(lines ~93-98). The key correctness property is that "falsy but
present" ground-truth values (0, False, [], "") are PRESERVED and
NOT coerced to '' by an over-eager truthiness check. Resolution must
use `is None` (the actual code), falling back ground_truth -> label
-> ''.
"""

import unittest

from app.services.inference_builder import (
    build_inference_output_from_eval,
    message_text_from_messages,
)


def _wrap(item: dict) -> dict:
    """Wrap a single eval-item dict in the {'inference': [...]} envelope."""
    return {"inference": [item]}


def _first(out: dict) -> dict:
    """Return the first output record from a build result."""
    assert out is not None
    return out["output"][0]


class BuildInferenceGroundTruthTests(unittest.TestCase):
    def test_ground_truth_int_zero_preserved(self):
        out = build_inference_output_from_eval(_wrap({"ground_truth": 0}))
        rec = _first(out)
        self.assertEqual(rec["ground_truth"], 0)
        self.assertNotEqual(rec["ground_truth"], "")
        # "answer" mirrors ground_truth in the current implementation.
        self.assertEqual(rec["answer"], 0)

    def test_ground_truth_bool_false_preserved(self):
        out = build_inference_output_from_eval(_wrap({"ground_truth": False}))
        rec = _first(out)
        self.assertIs(rec["ground_truth"], False)
        self.assertEqual(rec["answer"], False)

    def test_ground_truth_empty_list_preserved(self):
        out = build_inference_output_from_eval(_wrap({"ground_truth": []}))
        rec = _first(out)
        self.assertEqual(rec["ground_truth"], [])
        self.assertIsInstance(rec["ground_truth"], list)

    def test_ground_truth_empty_string_preserved(self):
        # An explicitly-present empty string stays an empty string (it is
        # not None, so no fallback to label happens).
        out = build_inference_output_from_eval(
            _wrap({"ground_truth": "", "label": "should-not-be-used"})
        )
        rec = _first(out)
        self.assertEqual(rec["ground_truth"], "")

    def test_missing_ground_truth_falls_back_to_label(self):
        out = build_inference_output_from_eval(_wrap({"label": "L"}))
        rec = _first(out)
        self.assertEqual(rec["ground_truth"], "L")
        self.assertEqual(rec["answer"], "L")

    def test_label_zero_used_when_ground_truth_missing(self):
        # label is also resolved with `is None`, so a falsy label (0) is kept.
        out = build_inference_output_from_eval(_wrap({"label": 0}))
        rec = _first(out)
        self.assertEqual(rec["ground_truth"], 0)

    def test_missing_both_falls_back_to_empty_string(self):
        out = build_inference_output_from_eval(_wrap({}))
        rec = _first(out)
        self.assertEqual(rec["ground_truth"], "")
        self.assertEqual(rec["answer"], "")

    def test_ground_truth_takes_precedence_over_label(self):
        out = build_inference_output_from_eval(
            _wrap({"ground_truth": "GT", "label": "LB"})
        )
        rec = _first(out)
        self.assertEqual(rec["ground_truth"], "GT")


class BuildInferenceStructureTests(unittest.TestCase):
    def test_returns_none_for_missing_inference(self):
        self.assertIsNone(build_inference_output_from_eval({}))

    def test_returns_none_for_empty_inference_list(self):
        self.assertIsNone(build_inference_output_from_eval({"inference": []}))

    def test_returns_none_when_no_dict_items(self):
        # A list of non-dicts yields an empty out_list -> None.
        self.assertIsNone(build_inference_output_from_eval({"inference": ["x", 1]}))

    def test_single_nested_list_is_unwrapped(self):
        # inference = [[item]] should be flattened to one record.
        out = build_inference_output_from_eval(
            {"inference": [[{"ground_truth": "a"}, {"ground_truth": "b"}]]}
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(out["output"]), 2)
        self.assertEqual(out["output"][0]["ground_truth"], "a")
        self.assertEqual(out["output"][1]["ground_truth"], "b")

    def test_multiple_chunked_lists_are_merged(self):
        out = build_inference_output_from_eval(
            {"inference": [[{"ground_truth": "a"}], [{"ground_truth": "b"}]]}
        )
        self.assertIsNotNone(out)
        self.assertEqual(len(out["output"]), 2)

    def test_prediction_resolution_prefers_postprocessed(self):
        out = build_inference_output_from_eval(
            _wrap({
                "prediction_postprocessed": "PP",
                "prediction": "P",
                "answer": "A",
            })
        )
        rec = _first(out)
        self.assertEqual(rec["prediction"], "PP")

    def test_prediction_zero_preserved(self):
        # prediction resolved with `is None`, so 0 survives.
        out = build_inference_output_from_eval(
            _wrap({"prediction_postprocessed": 0})
        )
        rec = _first(out)
        self.assertEqual(rec["prediction"], 0)

    def test_prediction_falls_back_through_chain(self):
        out = build_inference_output_from_eval(_wrap({"answer": "A"}))
        rec = _first(out)
        self.assertEqual(rec["prediction"], "A")

    def test_question_falls_back_to_prompt(self):
        out = build_inference_output_from_eval(_wrap({"prompt": "the prompt"}))
        rec = _first(out)
        self.assertEqual(rec["question"], "the prompt")
        # prompt field is `prompt or question`.
        self.assertEqual(rec["prompt"], "the prompt")

    def test_question_falls_back_to_input_then_messages(self):
        out = build_inference_output_from_eval(_wrap({"input": "from input"}))
        rec = _first(out)
        self.assertEqual(rec["question"], "from input")

    def test_question_from_messages_when_nothing_else(self):
        out = build_inference_output_from_eval(
            _wrap({"messages": [{"role": "user", "content": "hi there"}]})
        )
        rec = _first(out)
        self.assertEqual(rec["question"], "hi there")

    def test_record_carries_index_and_defaults(self):
        out = build_inference_output_from_eval(_wrap({"index": 7}))
        rec = _first(out)
        self.assertEqual(rec["index"], 7)
        self.assertEqual(rec["options"], [])
        self.assertEqual(rec["metrics"], {})
        self.assertEqual(rec["meta"], {})

    def test_metrics_non_dict_coerced_to_empty(self):
        out = build_inference_output_from_eval(
            _wrap({"metrics": "not-a-dict"})
        )
        rec = _first(out)
        self.assertEqual(rec["metrics"], {})


class MessageTextTests(unittest.TestCase):
    def test_non_list_returns_empty(self):
        self.assertEqual(message_text_from_messages("nope"), "")

    def test_user_string_content(self):
        msgs = [{"role": "user", "content": "hello"}]
        self.assertEqual(message_text_from_messages(msgs), "hello")

    def test_media_parts_skipped(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "value": "BASE64BLOB"},
                {"type": "text", "text": "describe this"},
            ],
        }]
        self.assertEqual(message_text_from_messages(msgs), "describe this")

    def test_text_parts_joined_with_newline(self):
        msgs = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "line1"},
                {"type": "text", "text": "line2"},
            ],
        }]
        self.assertEqual(message_text_from_messages(msgs), "line1\nline2")

    def test_falls_back_to_non_user_role(self):
        # No user message with text -> second pass over any role.
        msgs = [{"role": "system", "content": "sys text"}]
        self.assertEqual(message_text_from_messages(msgs), "sys text")


if __name__ == "__main__":
    unittest.main()
