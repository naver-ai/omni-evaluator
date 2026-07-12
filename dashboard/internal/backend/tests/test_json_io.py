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

"""Characterization + correctness tests for app/services/json_io.py.

Targets the brace/string state machine (_find_string_end, _find_object_end) and the
head/tail field extractor (_parse_head_tail). All fixtures are built inline as plain
strings; no files, network, or S3.
"""

import json
import unittest

from app.services.json_io import (
    _find_object_end,
    _find_string_end,
    _parse_head_tail,
)


class FindStringEndTests(unittest.TestCase):
    def test_plain_string(self):
        text = '"hello"'
        self.assertEqual(_find_string_end(text, 0), 6)

    def test_escaped_quote_not_treated_as_terminator(self):
        # rest = "abc\"def"  -> the \" in the middle must be skipped, closing quote is the last char.
        text = '"abc\\"def"'
        end = _find_string_end(text, 0)
        # The matching close quote is the final character, not the escaped one.
        self.assertEqual(end, len(text) - 1)
        # And the slice between quotes round-trips through json.loads intact.
        self.assertEqual(json.loads(text[: end + 1]), 'abc"def')

    def test_escaped_backslash_before_quote(self):
        # "a\\" -> backslash is escaped, the following quote really closes the string.
        text = '"a\\\\"'
        end = _find_string_end(text, 0)
        self.assertEqual(end, len(text) - 1)
        self.assertEqual(json.loads(text[: end + 1]), "a\\")

    def test_unterminated_returns_minus_one(self):
        self.assertEqual(_find_string_end('"never closes', 0), -1)


class FindObjectEndTests(unittest.TestCase):
    def test_simple_object(self):
        text = '{"a": 1}'
        self.assertEqual(_find_object_end(text, 0), len(text))

    def test_nested_object(self):
        text = '{"a": {"b": {"c": 1}}}'
        self.assertEqual(_find_object_end(text, 0), len(text))

    def test_literal_open_brace_in_string_does_not_inflate_depth(self):
        text = '{"note": "has a { brace"}'
        end = _find_object_end(text, 0)
        self.assertEqual(end, len(text))
        self.assertEqual(json.loads(text[:end]), {"note": "has a { brace"})

    def test_literal_close_brace_in_string_does_not_close_object(self):
        # A bare '}' inside the string would prematurely close the object if not string-aware.
        text = '{"note": "end } here", "x": 1}'
        end = _find_object_end(text, 0)
        self.assertEqual(end, len(text))
        self.assertEqual(json.loads(text[:end]), {"note": "end } here", "x": 1})

    def test_escaped_quote_inside_value_keeps_string_state(self):
        text = '{"q": "say \\"hi\\" now"}'
        end = _find_object_end(text, 0)
        self.assertEqual(end, len(text))
        self.assertEqual(json.loads(text[:end]), {"q": 'say "hi" now'})

    def test_trailing_content_after_object(self):
        # Object end is just past the matching '}', trailing junk is excluded.
        text = '{"a": 1}, "next": 2'
        end = _find_object_end(text, 0)
        self.assertEqual(text[:end], '{"a": 1}')
        self.assertEqual(json.loads(text[:end]), {"a": 1})

    def test_unterminated_object_returns_minus_one(self):
        self.assertEqual(_find_object_end('{"a": 1', 0), -1)


class ParseHeadTailTests(unittest.TestCase):
    def test_scalar_string_with_escaped_quote_not_truncated(self):
        # task_name value contains an escaped quote: the embedded \" must NOT terminate
        # the string early. The scalar branch returns the RAW (still JSON-escaped) slice
        # between the outer quotes -- it does not json-decode -- so the captured value is
        # the literal `foo\"bar`, the important point being it is the FULL value, not the
        # truncated `foo`.
        head = '{"task_name": "foo\\"bar", "config": {"model": "m"}}'
        tail = "}"
        result = _parse_head_tail(head, tail)
        self.assertIsNotNone(result)
        self.assertEqual(result["task_name"], 'foo\\"bar')
        # Not truncated at the escaped quote.
        self.assertNotEqual(result["task_name"], "foo")
        # And the raw slice still round-trips once wrapped back in quotes and decoded.
        self.assertEqual(json.loads('"' + result["task_name"] + '"'), 'foo"bar')

    def test_object_value_with_literal_brace_in_string_parses(self):
        # config string value contains a literal '{' and '}' -> depth must not corrupt.
        head = '{"config": {"model": "x", "note": "a{b}c"}, "task_name": "t"}'
        tail = "}"
        result = _parse_head_tail(head, tail)
        self.assertEqual(result["config"], {"model": "x", "note": "a{b}c"})
        self.assertEqual(result["task_name"], "t")

    def test_evaluation_object_with_brace_in_string(self):
        head = '{"config": {"a": 1}}'
        tail = '"evaluation": {"summary": "score is {high}", "n": 3}}'
        result = _parse_head_tail(head, tail)
        self.assertEqual(result["evaluation"], {"summary": "score is {high}", "n": 3})

    def test_unterminated_object_leaves_field_unset_and_warns(self):
        # config object never closes within the head slice => field left unset + warning.
        head = '{"config": {"model": "x", "deep": {"a": 1}'  # missing closing braces
        tail = "}"
        with self.assertLogs("app.services.json_io", level="WARNING") as cm:
            result = _parse_head_tail(head, tail)
        # config must NOT be present because the object is unterminated.
        if result is not None:
            self.assertNotIn("config", result)
        self.assertTrue(
            any("unterminated" in m and "config" in m for m in cm.output),
            cm.output,
        )

    def test_wellformed_head_tail_regression(self):
        # A realistic head (config/meta/scalars) + tail (evaluation/metrics) split.
        head = (
            '{"task_name": "vqa_eval", '
            '"evaluation_engine": "engineA", '
            '"config": {"model": "model-x", "shots": 5}, '
            '"meta": {"num_records": 100}, '
            '"inference": [[{"id": 0}'  # head cuts off mid-array, as for a large file
        )
        tail = (
            '{"id": 99}]], '
            '"evaluation": {"acc": 0.81, "f1": 0.77}, '
            '"metrics": {"bleu": 0.42}}'
        )
        result = _parse_head_tail(head, tail)
        self.assertIsNotNone(result)
        # Scalars from head.
        self.assertEqual(result["task_name"], "vqa_eval")
        self.assertEqual(result["evaluation_engine"], "engineA")
        # Objects from head.
        self.assertEqual(result["config"], {"model": "model-x", "shots": 5})
        self.assertEqual(result["meta"], {"num_records": 100})
        # evaluation from tail.
        self.assertEqual(result["evaluation"], {"acc": 0.81, "f1": 0.77})
        # top-level metrics fallback fires because evaluation has no "metrics" key.
        self.assertEqual(result["metrics"], {"bleu": 0.42})

    def test_metrics_fallback_skipped_when_evaluation_has_metrics(self):
        # Characterizes the guard: top-level metrics is only pulled when evaluation lacks it.
        head = '{"config": {"a": 1}}'
        tail = (
            '"evaluation": {"metrics": {"acc": 0.5}}, '
            '"metrics": {"f1": 0.9}}'
        )
        result = _parse_head_tail(head, tail)
        self.assertEqual(result["evaluation"], {"metrics": {"acc": 0.5}})
        # Because evaluation already carries metrics, the top-level "metrics" is NOT extracted.
        self.assertNotIn("metrics", result)

    def test_returns_none_when_nothing_extractable(self):
        self.assertIsNone(_parse_head_tail("no markers here", "still nothing"))


if __name__ == "__main__":
    unittest.main()
