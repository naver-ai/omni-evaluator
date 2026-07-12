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

"""Characterization + correctness tests for app/services/leaderboard_filters.py.

Hermetic: pure-function tests over small inline dict fixtures. No network/S3/files.
Run from internal/backend with:
    python -m unittest tests.test_leaderboard_filters -v
"""

import unittest

from app.services.leaderboard_filters import (
    apply_metric_override,
    compute_averages,
    compute_minmax,
    merge_inference_rows,
    parse_float,
    sort_rows,
    _METRIC_MAP_MAX_BYTES,
    _METRIC_MAP_MAX_ENTRIES,
)


# --------------------------------------------------------------------------
# compute_averages
# --------------------------------------------------------------------------
class TestComputeAverages(unittest.TestCase):
    def test_dash_placeholders_excluded_from_mean(self):
        # '-' columns must be skipped entirely (parse_float -> None), so the
        # mean is over the numeric columns only, and the denominator is the
        # count of *numeric* columns, not len(visible_cols).
        rows = [{"scores": {"a": "10", "b": "-", "c": "20"}}]
        show_sum = compute_averages(
            rows, visible_cols=["a", "b", "c"],
            weights={}, value_type="score", value_subs=[],
        )
        self.assertFalse(show_sum)
        # (10 + 20) / 2 == 15.0  -- the '-' in b is NOT counted as a column.
        self.assertAlmostEqual(rows[0]["_avg"], 15.0)
        self.assertIsNone(rows[0]["_sum"])

    def test_empty_string_and_none_also_excluded(self):
        rows = [{"scores": {"a": "4", "b": "", "c": None, "d": "8"}}]
        compute_averages(
            rows, visible_cols=["a", "b", "c", "d"],
            weights={}, value_type="score", value_subs=[],
        )
        self.assertAlmostEqual(rows[0]["_avg"], 6.0)  # (4 + 8) / 2

    def test_no_numeric_cols_does_not_divide_by_zero(self):
        # All placeholders -> no numeric vals -> _avg is None (no ZeroDivisionError).
        rows = [{"scores": {"a": "-", "b": "", "c": None}}]
        try:
            compute_averages(
                rows, visible_cols=["a", "b", "c"],
                weights={}, value_type="score", value_subs=[],
            )
        except ZeroDivisionError:
            self.fail("compute_averages divided by zero on all-placeholder row")
        self.assertIsNone(rows[0]["_avg"])
        self.assertIsNone(rows[0]["_sum"])

    def test_missing_column_in_scores_treated_as_none(self):
        # visible_cols references a benchmark the row never ran; .get -> None.
        rows = [{"scores": {"a": "10"}}]
        compute_averages(
            rows, visible_cols=["a", "missing"],
            weights={}, value_type="score", value_subs=[],
        )
        self.assertAlmostEqual(rows[0]["_avg"], 10.0)

    def test_weights_applied_only_for_score_value_type(self):
        # weight a=3, b=1 -> weighted mean = (10*3 + 20*1) / (3 + 1) = 50/4 = 12.5
        weights = {"a": 3.0, "b": 1.0}
        rows = [{"scores": {"a": "10", "b": "20"}}]
        compute_averages(
            rows, visible_cols=["a", "b"],
            weights=weights, value_type="score", value_subs=[],
        )
        self.assertAlmostEqual(rows[0]["_avg"], 12.5)

    def test_weights_ignored_for_non_score_value_type(self):
        # Same weights, but value_type != 'score' -> every weight forced to 1.0,
        # so this is a plain arithmetic mean: (10 + 20) / 2 = 15.0.
        weights = {"a": 3.0, "b": 1.0}
        rows = [{"scores": {"a": "10", "b": "20"}}]
        compute_averages(
            rows, visible_cols=["a", "b"],
            weights=weights, value_type="coverage", value_subs=[],
        )
        self.assertAlmostEqual(rows[0]["_avg"], 15.0)

    def test_show_sum_true_for_time_without_latency_throughput(self):
        rows = [{"scores": {"a__runtime_total": "5", "b__runtime_inference": "7"}}]
        show_sum = compute_averages(
            rows, visible_cols=["a__runtime_total", "b__runtime_inference"],
            weights={}, value_type="time", value_subs=["total", "inference"],
        )
        self.assertTrue(show_sum)
        self.assertAlmostEqual(rows[0]["_sum"], 12.0)  # 5 + 7
        self.assertAlmostEqual(rows[0]["_avg"], 6.0)   # mean of the two

    def test_show_sum_false_when_latency_or_throughput_present(self):
        rows = [{"scores": {"a__latency": "5"}}]
        show_sum = compute_averages(
            rows, visible_cols=["a__latency"],
            weights={}, value_type="time", value_subs=["latency"],
        )
        self.assertFalse(show_sum)
        self.assertIsNone(rows[0]["_sum"])


# --------------------------------------------------------------------------
# sort_rows
# --------------------------------------------------------------------------
class TestSortRows(unittest.TestCase):
    def test_numeric_column_descending(self):
        rows = [
            {"model": "m1", "scores": {"a": "10"}},
            {"model": "m2", "scores": {"a": "30"}},
            {"model": "m3", "scores": {"a": "20"}},
        ]
        out = sort_rows(rows, sort_col="a", sort_dir="desc", visible_cols=["a"])
        self.assertEqual([r["model"] for r in out], ["m2", "m3", "m1"])

    def test_numeric_column_ascending(self):
        rows = [
            {"model": "m1", "scores": {"a": "10"}},
            {"model": "m2", "scores": {"a": "30"}},
            {"model": "m3", "scores": {"a": "20"}},
        ]
        out = sort_rows(rows, sort_col="a", sort_dir="asc", visible_cols=["a"])
        self.assertEqual([r["model"] for r in out], ["m1", "m3", "m2"])

    def test_missing_values_sort_below_present_in_desc(self):
        # sort_key is (v is not None, v): present (True, x) outranks missing
        # (False, -1e9). With reverse=True, present values come first.
        rows = [
            {"model": "has", "scores": {"a": "5"}},
            {"model": "dash", "scores": {"a": "-"}},
            {"model": "none", "scores": {}},
        ]
        out = sort_rows(rows, sort_col="a", sort_dir="desc", visible_cols=["a"])
        self.assertEqual(out[0]["model"], "has")
        # The two missing-value rows both map to (False, -1e9).
        self.assertEqual({out[1]["model"], out[2]["model"]}, {"dash", "none"})

    def test_stable_sort_preserves_input_order_on_ties(self):
        # Equal numeric values must keep original relative order (stable sort).
        rows = [
            {"model": "first", "scores": {"a": "10"}},
            {"model": "second", "scores": {"a": "10"}},
            {"model": "third", "scores": {"a": "10"}},
        ]
        out = sort_rows(rows, sort_col="a", sort_dir="desc", visible_cols=["a"])
        self.assertEqual([r["model"] for r in out], ["first", "second", "third"])

    def test_string_model_column_case_insensitive_asc(self):
        rows = [
            {"model": "Zebra"},
            {"model": "alpha"},
            {"model": "Mango"},
        ]
        out = sort_rows(rows, sort_col="model", sort_dir="asc", visible_cols=[])
        self.assertEqual([r["model"] for r in out], ["alpha", "Mango", "Zebra"])

    def test_default_sort_no_sort_col(self):
        # No sort_col -> model asc, inference asc, checkpoint numeric asc.
        rows = [
            {"model": "b", "inference": "vllm", "checkpoint": "checkpoint-200"},
            {"model": "a", "inference": "vllm", "checkpoint": "checkpoint-100"},
            {"model": "a", "inference": "vllm", "checkpoint": "checkpoint-50"},
        ]
        out = sort_rows(rows, sort_col="", sort_dir="asc", visible_cols=[])
        self.assertEqual(
            [(r["model"], r["checkpoint"]) for r in out],
            [("a", "checkpoint-50"), ("a", "checkpoint-100"), ("b", "checkpoint-200")],
        )

    def test_checkpoint_none_sorts_first_as_minus_one(self):
        rows = [
            {"model": "a", "inference": "x", "checkpoint": "checkpoint-5"},
            {"model": "a", "inference": "x", "checkpoint": "checkpoint-none"},
        ]
        out = sort_rows(rows, sort_col="checkpoint", sort_dir="asc", visible_cols=[])
        self.assertEqual(out[0]["checkpoint"], "checkpoint-none")  # -1 < 5


# --------------------------------------------------------------------------
# compute_minmax
# --------------------------------------------------------------------------
class TestComputeMinmax(unittest.TestCase):
    def test_minmax_over_all_rows(self):
        rows = [
            {"scores": {"a": "10"}},
            {"scores": {"a": "50"}},
            {"scores": {"a": "30"}},
        ]
        mm = compute_minmax(rows, visible_cols=["a"], show_sum=False)
        self.assertEqual(mm["a"], (10.0, 50.0))

    def test_minmax_stable_regardless_of_pagination_slice(self):
        # compute_minmax must be fed the FULL row set, not a paginated page,
        # so the scale is identical whether or not the caller slices the list.
        full = [
            {"scores": {"a": "10"}},
            {"scores": {"a": "50"}},
            {"scores": {"a": "30"}},
            {"scores": {"a": "5"}},
        ]
        page = full[:2]  # a paginated slice, NOT what we pass in
        mm_full = compute_minmax(full, visible_cols=["a"], show_sum=False)
        mm_page = compute_minmax(page, visible_cols=["a"], show_sum=False)
        self.assertEqual(mm_full["a"], (5.0, 50.0))
        # Demonstrates the bug pagination would introduce if you sliced first:
        self.assertEqual(mm_page["a"], (10.0, 50.0))
        self.assertNotEqual(mm_full["a"], mm_page["a"])

    def test_equal_min_max_column_omitted(self):
        # When min == max the column is dropped (no useful color scale).
        rows = [{"scores": {"a": "7"}}, {"scores": {"a": "7"}}]
        mm = compute_minmax(rows, visible_cols=["a"], show_sum=False)
        self.assertNotIn("a", mm)

    def test_no_numeric_values_column_omitted(self):
        rows = [{"scores": {"a": "-"}}, {"scores": {"a": ""}}]
        mm = compute_minmax(rows, visible_cols=["a"], show_sum=False)
        self.assertNotIn("a", mm)
        self.assertEqual(mm, {})

    def test_avg_included_and_sum_only_when_show_sum(self):
        rows = [
            {"scores": {}, "_avg": 1.0, "_sum": 5.0},
            {"scores": {}, "_avg": 9.0, "_sum": 50.0},
        ]
        mm_no_sum = compute_minmax(rows, visible_cols=[], show_sum=False)
        self.assertEqual(mm_no_sum["_avg"], (1.0, 9.0))
        self.assertNotIn("_sum", mm_no_sum)

        mm_sum = compute_minmax(rows, visible_cols=[], show_sum=True)
        self.assertEqual(mm_sum["_sum"], (5.0, 50.0))


# --------------------------------------------------------------------------
# apply_metric_override
# --------------------------------------------------------------------------
class TestApplyMetricOverride(unittest.TestCase):
    def test_valid_small_override_applied(self):
        rows = [{
            "scores": {"benchA": "1"},
            "metrics": {"benchA": {"f1": "0.88", "acc": "0.99"}},
        }]
        apply_metric_override(rows, '{"benchA": "f1"}')
        # scores[benchA] should now reflect the f1 metric value.
        self.assertEqual(rows[0]["scores"]["benchA"], "0.88")

    def test_missing_metric_key_becomes_dash(self):
        rows = [{
            "scores": {"benchA": "1"},
            "metrics": {"benchA": {"acc": "0.99"}},
        }]
        apply_metric_override(rows, '{"benchA": "f1"}')  # f1 absent
        self.assertEqual(rows[0]["scores"]["benchA"], "-")

    def test_empty_string_value_becomes_dash(self):
        rows = [{
            "scores": {"benchA": "1"},
            "metrics": {"benchA": {"f1": ""}},
        }]
        apply_metric_override(rows, '{"benchA": "f1"}')
        self.assertEqual(rows[0]["scores"]["benchA"], "-")

    def test_oversized_json_string_ignored_no_raise(self):
        # A JSON string just over the byte cap must be ignored silently
        # (DoS guard) -- override NOT applied, no exception.
        rows = [{
            "scores": {"benchA": "orig"},
            "metrics": {"benchA": {"f1": "0.88"}},
        }]
        # Build a valid-JSON string whose raw length exceeds the byte cap by
        # padding the value with a very long (but unused) key string.
        big_value = "x" * (_METRIC_MAP_MAX_BYTES + 10)
        oversized = '{"benchA": "f1", "pad": "' + big_value + '"}'
        self.assertGreater(len(oversized), _METRIC_MAP_MAX_BYTES)
        try:
            apply_metric_override(rows, oversized)
        except Exception as e:
            self.fail(f"oversized override raised: {e!r}")
        self.assertEqual(rows[0]["scores"]["benchA"], "orig")  # untouched

    def test_too_many_entries_ignored(self):
        # A parsed dict with > entry cap keys is ignored (no override applied).
        entries = {f"b{i}": "f1" for i in range(_METRIC_MAP_MAX_ENTRIES + 1)}
        import json as _json
        payload = _json.dumps(entries)
        # Keep it under the byte cap so we isolate the entry-count guard.
        self.assertLessEqual(len(payload), _METRIC_MAP_MAX_BYTES)
        rows = [{
            "scores": {"b0": "orig"},
            "metrics": {"b0": {"f1": "0.5"}},
        }]
        apply_metric_override(rows, payload)
        self.assertEqual(rows[0]["scores"]["b0"], "orig")  # untouched

    def test_entry_count_at_cap_is_applied(self):
        # Exactly at the cap (200) is allowed (guard is strictly '>').
        entries = {f"b{i}": "f1" for i in range(_METRIC_MAP_MAX_ENTRIES)}
        import json as _json
        payload = _json.dumps(entries)
        rows = [{
            "scores": {"b0": "orig"},
            "metrics": {"b0": {"f1": "0.5"}},
        }]
        apply_metric_override(rows, payload)
        self.assertEqual(rows[0]["scores"]["b0"], "0.5")

    def test_empty_or_invalid_json_is_noop(self):
        rows = [{"scores": {"a": "1"}, "metrics": {"a": {"f1": "2"}}}]
        apply_metric_override(rows, "")          # falsy -> early return
        apply_metric_override(rows, "not json")  # JSONDecodeError -> swallowed
        apply_metric_override(rows, "[1, 2, 3]")  # not a dict -> return
        self.assertEqual(rows[0]["scores"]["a"], "1")

    def test_no_scores_key_means_no_write(self):
        # apply only writes when 'scores' in r; a row without it is left alone.
        rows = [{"metrics": {"a": {"f1": "2"}}}]
        apply_metric_override(rows, '{"a": "f1"}')
        self.assertNotIn("scores", rows[0])


# --------------------------------------------------------------------------
# merge_inference_rows  (and the ordering dependency with apply_metric_override)
# --------------------------------------------------------------------------
class TestMergeInferenceRows(unittest.TestCase):
    def test_merge_drops_metrics_key(self):
        # merge_inference_rows builds brand-new dicts with only
        # model/inference/checkpoint/source/scores -> 'metrics' is dropped.
        # THIS is why apply_metric_override (which reads r['metrics']) MUST run
        # BEFORE merge: after merge there is no 'metrics' left to override from.
        rows = [{
            "model": "m",
            "checkpoint": "checkpoint-1",
            "source": "s",
            "inference": "vllm",
            "scores": {"a": "1"},
            "metrics": {"a": {"f1": "0.9"}},
        }]
        merged = merge_inference_rows(rows)
        self.assertEqual(len(merged), 1)
        self.assertNotIn("metrics", merged[0])
        self.assertEqual(
            set(merged[0].keys()),
            {"model", "inference", "checkpoint", "source", "scores"},
        )

    def test_merge_keeps_best_higher_for_normal_scores(self):
        # Two inference engines for the same (model, ckpt, source) merge into
        # one row; for a normal score column the higher value wins.
        rows = [
            {"model": "m", "checkpoint": "checkpoint-1", "source": "s",
             "inference": "vllm", "scores": {"a": "10"}},
            {"model": "m", "checkpoint": "checkpoint-1", "source": "s",
             "inference": "sglang", "scores": {"a": "30"}},
        ]
        merged = merge_inference_rows(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["scores"]["a"], "30")
        # inferences are sorted+joined.
        self.assertEqual(merged[0]["inference"], "sglang / vllm")

    def test_merge_keeps_lowest_for_runtime_columns(self):
        # runtime/latency columns are "lower is better" -> min wins.
        rows = [
            {"model": "m", "checkpoint": "checkpoint-1", "source": "s",
             "inference": "vllm", "scores": {"a__runtime_total": "9.0"}},
            {"model": "m", "checkpoint": "checkpoint-1", "source": "s",
             "inference": "sglang", "scores": {"a__runtime_total": "4.0"}},
        ]
        merged = merge_inference_rows(rows)
        self.assertEqual(merged[0]["scores"]["a__runtime_total"], "4.0")

    def test_distinct_keys_not_merged(self):
        rows = [
            {"model": "m", "checkpoint": "checkpoint-1", "source": "s",
             "inference": "vllm", "scores": {"a": "1"}},
            {"model": "m", "checkpoint": "checkpoint-2", "source": "s",
             "inference": "vllm", "scores": {"a": "2"}},
        ]
        merged = merge_inference_rows(rows)
        self.assertEqual(len(merged), 2)

    def test_override_then_merge_pipeline_order(self):
        # End-to-end demonstration of the required ordering:
        # override reads 'metrics' to rewrite 'scores'; if we merged first the
        # metrics would be gone and the override would silently no-op.
        rows = [{
            "model": "m", "checkpoint": "checkpoint-1", "source": "s",
            "inference": "vllm",
            "scores": {"benchA": "raw"},
            "metrics": {"benchA": {"f1": "0.77"}},
        }]
        apply_metric_override(rows, '{"benchA": "f1"}')  # must come first
        merged = merge_inference_rows(rows)
        self.assertEqual(merged[0]["scores"]["benchA"], "0.77")


# --------------------------------------------------------------------------
# parse_float (helper underpinning the above)
# --------------------------------------------------------------------------
class TestParseFloat(unittest.TestCase):
    def test_placeholders_return_none(self):
        for v in (None, "-", ""):
            self.assertIsNone(parse_float(v))

    def test_comma_stripped(self):
        self.assertEqual(parse_float("1,234.5"), 1234.5)

    def test_non_numeric_returns_none(self):
        self.assertIsNone(parse_float("abc"))


if __name__ == "__main__":
    unittest.main()
