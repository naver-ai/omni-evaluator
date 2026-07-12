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

"""Regression tests for app/services/metric_extraction.py.

Covers:
  * extract_benchmark_from_filename basic cases.
  * The #28 fix: when latency lives only at evaluation.run_outputs[0]
    (not at evaluation top level), BOTH runtime_inference (via
    _extract_runtime_and_coverage / extract_metrics_map) and
    benchmark_time are populated and CONSISTENT (both = latency *
    num_samples).
  * A literal 0 latency is resolved via `is not None` (not dropped by a
    falsy check) and surfaced as "latency": "0.0000".
"""

import unittest

from app.services.metric_extraction import (
    extract_benchmark_from_filename,
    _resolve_latency_and_samples,
    _extract_runtime_and_coverage,
    extract_metrics_map,
)


class BenchmarkFromFilenameTests(unittest.TestCase):
    def test_double_underscore_split(self):
        self.assertEqual(
            extract_benchmark_from_filename("mmbench__cot.json"), "mmbench"
        )

    def test_no_double_underscore_returns_stem(self):
        self.assertEqual(
            extract_benchmark_from_filename("plainbench.json"), "plainbench"
        )

    def test_single_underscore_is_not_a_split(self):
        # Only "__" splits; a single underscore stays in the stem.
        self.assertEqual(
            extract_benchmark_from_filename("bench_v2.json"), "bench_v2"
        )

    def test_only_first_segment_kept(self):
        self.assertEqual(
            extract_benchmark_from_filename("a__b__c.json"), "a"
        )

    def test_full_path_uses_stem_only(self):
        self.assertEqual(
            extract_benchmark_from_filename("/x/y/docvqa__greedy.json"),
            "docvqa",
        )

    def test_no_extension(self):
        self.assertEqual(
            extract_benchmark_from_filename("ocrbench__m"), "ocrbench"
        )


class ResolveLatencyAndSamplesTests(unittest.TestCase):
    def test_top_level_preferred(self):
        ev = {
            "latency": 1.5,
            "num_samples": 4,
            "run_outputs": [{"latency": 99.0, "num_samples": 999}],
        }
        lat, n = _resolve_latency_and_samples(ev)
        self.assertEqual(lat, 1.5)
        self.assertEqual(n, 4)

    def test_falls_back_to_run_outputs_zero(self):
        ev = {"run_outputs": [{"latency": 2.0, "num_samples": 8}]}
        lat, n = _resolve_latency_and_samples(ev)
        self.assertEqual(lat, 2.0)
        self.assertEqual(n, 8)

    def test_zero_latency_at_top_level_not_dropped(self):
        # 0 is not None, so the top-level 0 wins (does NOT fall through
        # to run_outputs).
        ev = {
            "latency": 0,
            "num_samples": 0,
            "run_outputs": [{"latency": 5.0, "num_samples": 5}],
        }
        lat, n = _resolve_latency_and_samples(ev)
        self.assertEqual(lat, 0)
        self.assertEqual(n, 0)

    def test_empty_returns_none_none(self):
        self.assertEqual(_resolve_latency_and_samples({}), (None, None))


class Issue28RuntimeBenchmarkConsistencyTests(unittest.TestCase):
    """#28: latency only at run_outputs[0] must drive BOTH
    runtime_inference and benchmark_time, consistently."""

    def _data(self):
        # latency/num_samples live ONLY inside run_outputs[0],
        # nowhere at evaluation top level.
        return {
            "evaluation": {
                "run_outputs": [{"latency": 2.0, "num_samples": 10}],
            }
        }

    def test_runtime_inference_derived_from_run_outputs_latency(self):
        rc = _extract_runtime_and_coverage(self._data())
        self.assertEqual(rc["runtime_inference"], 20.0)  # 2.0 * 10
        self.assertEqual(rc["runtime_total"], 20.0)

    def test_metrics_map_has_consistent_runtime_and_benchmark_time(self):
        m = extract_metrics_map(self._data())
        # Both populated...
        self.assertIn("runtime_inference", m)
        self.assertIn("benchmark_time", m)
        # ...and consistent: both equal latency * num_samples = 20.0.
        self.assertEqual(m["runtime_inference"], "20.00")
        self.assertEqual(m["benchmark_time"], "20.00")
        # latency itself is surfaced with 4 decimals.
        self.assertEqual(m["latency"], "2.0000")

    def test_explicit_runtime_inference_takes_precedence_over_derived(self):
        # If run_outputs[0] also carries an explicit runtime_inference,
        # it should be used instead of latency*num_samples.
        data = {
            "evaluation": {
                "run_outputs": [
                    {"latency": 2.0, "num_samples": 10, "runtime_inference": 7.0}
                ],
            }
        }
        rc = _extract_runtime_and_coverage(data)
        self.assertEqual(rc["runtime_inference"], 7.0)
        # benchmark_time still comes from latency * num_samples (separate path).
        m = extract_metrics_map(data)
        self.assertEqual(m["benchmark_time"], "20.00")
        self.assertEqual(m["runtime_inference"], "7.00")


class LiteralZeroLatencyTests(unittest.TestCase):
    def test_zero_latency_surfaced_not_dropped(self):
        # latency at top-level evaluation == 0 must appear as "0.0000",
        # proving resolution uses `is not None`, not a truthiness check.
        data = {"evaluation": {"latency": 0, "num_samples": 10}}
        m = extract_metrics_map(data)
        self.assertEqual(m["latency"], "0.0000")

    def test_zero_latency_yields_zero_benchmark_time(self):
        data = {"evaluation": {"latency": 0, "num_samples": 10}}
        m = extract_metrics_map(data)
        self.assertEqual(m["benchmark_time"], "0.00")

    def test_zero_latency_at_run_outputs_level(self):
        data = {"evaluation": {"run_outputs": [{"latency": 0, "num_samples": 4}]}}
        m = extract_metrics_map(data)
        self.assertEqual(m["latency"], "0.0000")
        self.assertEqual(m["benchmark_time"], "0.00")


class MetricsMapCoverageTests(unittest.TestCase):
    def test_coverage_clamped_into_unit_range(self):
        data = {
            "evaluation": {
                "coverage_inference": 1.5,
                "coverage_evaluation": -0.2,
            }
        }
        m = extract_metrics_map(data)
        self.assertEqual(m["coverage_inference"], "1.0000")
        self.assertEqual(m["coverage_evaluation"], "0.0000")

    def test_numeric_metrics_formatted(self):
        data = {"metrics": {"accuracy": 0.8567, "count": 12}}
        m = extract_metrics_map(data)
        self.assertEqual(m["accuracy"], "0.86")
        self.assertEqual(m["count"], "12")

    def test_throughput_surfaced(self):
        data = {"evaluation": {"throughput": 3.14159}}
        m = extract_metrics_map(data)
        self.assertEqual(m["throughput"], "3.1416")


if __name__ == "__main__":
    unittest.main()
