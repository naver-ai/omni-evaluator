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

"""Regression tests for scan_cache.get_model_benchmark_counts.

Seeds fake entries into the warm caches (_file_cache, _zip_cache, _s3_cache)
matching the real tuple shapes and asserts the (source, model) -> count map.
The keys must match sc_get_models output:
  * internal -> first path component under INTERNAL_OUTPUTS_PATH
  * direct   -> zip stem under DIRECT_SUBMISSION_DIR
  * s3       -> the _s3_cache model_name
Counts are DISTINCT real benchmarks (synthetic suffixes and "-" excluded).
"""

import unittest

from app.services import scan_cache
from app.config import INTERNAL_OUTPUTS_PATH, DIRECT_SUBMISSION_DIR


class ModelBenchmarkCountTests(unittest.TestCase):
    def setUp(self):
        # _file_cache value: (mtime, entry, size); entry has a "benchmarks" dict.
        # The internal model name is the first path component under
        # INTERNAL_OUTPUTS_PATH, so put files two levels deep to prove the
        # .parts[0] derivation (matches sc_get_models("internal")).
        self.f1 = str(INTERNAL_OUTPUTS_PATH / "modelA" / "evaluation_output" / "mmbench.json")
        self.f2 = str(INTERNAL_OUTPUTS_PATH / "modelA" / "evaluation_output" / "docvqa.json")
        scan_cache._file_cache[self.f1] = (
            1.0,
            {"benchmarks": {"mmbench": "0.5", "mmbench__latency": "1.0"}},
            10,
        )
        scan_cache._file_cache[self.f2] = (
            1.0,
            {"benchmarks": {"docvqa": "0.8", "missing_bench": "-"}},
            10,
        )

        # _zip_cache value: (mtime, rows, size). Internal zip (under
        # INTERNAL_OUTPUTS_PATH) -> source "internal", model = zip stem.
        self.iz = str(INTERNAL_OUTPUTS_PATH / "zipmodel.zip")
        scan_cache._zip_cache[self.iz] = (
            1.0,
            [{"scores": {"chartqa": "0.7", "ai2d": "0.6", "ai2d__runtime_total": "9"}}],
            10,
        )

        # Direct zip (under DIRECT_SUBMISSION_DIR) -> source "direct", model = stem.
        self.dz = str(DIRECT_SUBMISSION_DIR / "submitted.zip")
        scan_cache._zip_cache[self.dz] = (
            1.0,
            [{"scores": {"vqa": "0.9", "gone": "-"}}],
            10,
        )

        # _s3_cache value: (ts, rows). Key is the model_name.
        scan_cache._s3_cache["s3model"] = (
            1.0,
            [
                {"scores": {"benchX": "0.1", "benchY": "0.2"}},
                {"scores": {"benchY": "0.3", "benchZ": "-"}},
            ],
        )

    def tearDown(self):
        scan_cache._file_cache.clear()
        scan_cache._zip_cache.clear()
        scan_cache._s3_cache.clear()

    def test_counts_and_keys(self):
        counts = scan_cache.get_model_benchmark_counts()
        # internal modelA: mmbench + docvqa (synthetic __latency and "-" excluded)
        self.assertEqual(counts.get(("internal", "modelA"), {}).get("total"), 2)
        # internal zip: chartqa + ai2d (synthetic __runtime_total excluded)
        self.assertEqual(counts.get(("internal", "zipmodel"), {}).get("total"), 2)
        # direct zip: vqa only ("-" excluded)
        self.assertEqual(counts.get(("direct", "submitted"), {}).get("total"), 1)
        # s3: benchX + benchY distinct ("-" excluded, dup benchY counted once)
        self.assertEqual(counts.get(("s3", "s3model"), {}).get("total"), 2)

    def test_keys_match_sc_get_models_shape(self):
        counts = scan_cache.get_model_benchmark_counts()
        # Every key is a (source, model) tuple; the value is a breakdown dict.
        for (src, model), bd in counts.items():
            self.assertIn(src, {"internal", "direct", "s3"})
            self.assertIsInstance(model, str)
            self.assertGreater(bd["total"], 0)
            # The four modality buckets must sum to the reported total.
            self.assertEqual(
                bd["text"] + bd["image"] + bd["video"] + bd["audio"], bd["total"]
            )

    def test_modality_split(self):
        # mmbench/docvqa/chartqa/ai2d are image benchmarks by name heuristic;
        # benchX/benchY/vqa have no image/video/audio name signal -> text.
        counts = scan_cache.get_model_benchmark_counts()
        a = counts[("internal", "modelA")]
        self.assertEqual(a["image"], 2)  # mmbench, docvqa
        self.assertEqual(a["text"], 0)
        z = counts[("internal", "zipmodel")]
        self.assertEqual(z["image"], 2)  # chartqa, ai2d
        s = counts[("s3", "s3model")]
        self.assertEqual(s["text"], 2)  # benchX, benchY -> text
        self.assertEqual(s["image"], 0)

    def test_explicit_video_audio_map(self):
        # An explicit modality map routes benchmarks to video/audio buckets.
        scan_cache._bench_modalities["benchX"] = "audio"
        scan_cache._bench_modalities["benchY"] = "video"
        try:
            counts = scan_cache.get_model_benchmark_counts()
            s = counts[("s3", "s3model")]
            self.assertEqual(s["audio"], 1)
            self.assertEqual(s["video"], 1)
            self.assertEqual(s["text"], 0)
        finally:
            scan_cache._bench_modalities.clear()


if __name__ == "__main__":
    unittest.main()
