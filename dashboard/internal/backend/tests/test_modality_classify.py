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

"""Tests for classify_benchmark_modality in app/services/leaderboard_filters.py.

Hermetic: pure-function tests over inline fixtures. No network/S3/files.
Run from internal/backend with:
    python -m unittest tests.test_modality_classify -v
"""

import unittest

from app.services.leaderboard_filters import classify_benchmark_modality


class TestExplicitModalityMapWins(unittest.TestCase):
    def test_audio_from_map(self):
        self.assertEqual(classify_benchmark_modality("foo", {"foo": "audio"}), "audio")

    def test_video_from_map(self):
        self.assertEqual(classify_benchmark_modality("foo", {"foo": "video"}), "video")

    def test_vision_maps_to_image(self):
        self.assertEqual(classify_benchmark_modality("foo", {"foo": "vision"}), "image")

    def test_image_maps_to_image(self):
        self.assertEqual(classify_benchmark_modality("foo", {"foo": "image"}), "image")

    def test_text_from_map(self):
        self.assertEqual(classify_benchmark_modality("foo", {"foo": "text"}), "text")

    def test_map_overrides_name_heuristic(self):
        # Name says image (chartqa), but explicit map says audio -> map wins.
        self.assertEqual(classify_benchmark_modality("chartqa", {"chartqa": "audio"}), "audio")

    def test_map_value_is_case_insensitive(self):
        self.assertEqual(classify_benchmark_modality("foo", {"foo": "VISION"}), "image")

    def test_unknown_map_value_falls_through_to_heuristic(self):
        # An unrecognized map value must not short-circuit; fall back to name heuristic.
        self.assertEqual(classify_benchmark_modality("librispeech", {"librispeech": "weird"}), "audio")

    def test_empty_map_value_falls_through(self):
        self.assertEqual(classify_benchmark_modality("videomme", {"videomme": ""}), "video")

    def test_coarse_vision_map_loses_to_video_name(self):
        # 'longvideobench' is clearly VIDEO, but the auto modality_map only tags
        # it as coarse 'vision' (cannot tell image vs video). The name-based
        # video signal must override the coarse 'vision' map value.
        self.assertEqual(
            classify_benchmark_modality("longvideobench_val_i", {"longvideobench_val_i": "vision"}),
            "video",
        )

    def test_explicit_video_map_still_wins(self):
        # An explicit 'video' map value remains authoritative.
        self.assertEqual(classify_benchmark_modality("foo", {"foo": "video"}), "video")

    def test_coarse_vision_map_kept_for_plain_vision_bench(self):
        # An ordinary vision benchmark (name matches no audio/video/image term)
        # still resolves to "image" via the coarse map value.
        self.assertEqual(classify_benchmark_modality("foo", {"foo": "vision"}), "image")


class TestNameHeuristicAudio(unittest.TestCase):
    def test_clear_audio(self):
        for b in ["librispeech_test_clean", "fleurs_zh_test", "covost2_en2zh_test",
                  "clotho_v1_test", "gtzan_test", "meld_emotion_test", "mmau_test",
                  "cochlscene_test", "vocal_sound_test", "voice_bench_test"]:
            with self.subTest(b=b):
                self.assertEqual(classify_benchmark_modality(b), "audio")


class TestNameHeuristicVideo(unittest.TestCase):
    def test_clear_video(self):
        for b in ["videomme", "mvbench", "MVBench", "lvbench", "longvideobench_val_i"]:
            with self.subTest(b=b):
                self.assertEqual(classify_benchmark_modality(b), "video")

    def test_video_before_image_ordering(self):
        # 'video_mmmu' contains both a video term and the image term 'mmmu';
        # video MUST win because it is checked first.
        self.assertEqual(classify_benchmark_modality("video_mmmu"), "video")


class TestNameHeuristicImage(unittest.TestCase):
    def test_clear_image(self):
        for b in ["chartqa", "ai2d", "ai2d_test", "mmmu_val", "mathvista",
                  "mathvision", "mathverse_testmini", "ocrbench", "hallusion",
                  "docvqa_test", "infovqa_test", "textvqa_val", "mmstar",
                  "mmvet", "seedbench", "realworldqa", "mmbench_en_dev",
                  "llava_in_the_wild", "konet_2024_test"]:
            with self.subTest(b=b):
                self.assertEqual(classify_benchmark_modality(b), "image")


class TestNameHeuristicText(unittest.TestCase):
    def test_clear_text(self):
        for b in ["arc_challenge_generative", "bbh_zeroshot", "aime25", "gsm8k",
                  "mmlu_pro", "humaneval", "mbpp", "ifeval", "gpqa_cot",
                  "livecodebench"]:
            with self.subTest(b=b):
                self.assertEqual(classify_benchmark_modality(b), "text")

    def test_unknown_defaults_to_text(self):
        self.assertEqual(classify_benchmark_modality("some_random_benchmark"), "text")


class TestBaseNameSplit(unittest.TestCase):
    def test_base_name_split_on_double_underscore(self):
        # No exact match, but base name 'videomme' is in the map.
        m = {"videomme": "video"}
        self.assertEqual(classify_benchmark_modality("videomme__0", m), "video")

    def test_exact_key_preferred_over_base(self):
        m = {"chartqa__0": "audio", "chartqa": "vision"}
        self.assertEqual(classify_benchmark_modality("chartqa__0", m), "audio")

    def test_base_split_then_heuristic_when_not_in_map(self):
        # Neither exact nor base in map -> heuristic on full name.
        self.assertEqual(classify_benchmark_modality("librispeech__3", {}), "audio")


class TestReturnValuesAlwaysValid(unittest.TestCase):
    def test_always_one_of_four(self):
        allowed = {"audio", "video", "image", "text"}
        for b in ["", "x", "videomme", "chartqa", "librispeech", "gsm8k"]:
            with self.subTest(b=b):
                self.assertIn(classify_benchmark_modality(b), allowed)


if __name__ == "__main__":
    unittest.main()
