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

"""Characterization tests for the LOCAL + ZIP sample read+merge dispatch in app/api/inference.py.

These pin the CURRENT, correct behavior of the read+merge path so the upcoming refactor
(unifying the local/zip/S3 helpers into shared code) cannot silently change results.

SEAMS UNDER TEST (and why):
  * ``_get_one_sample(source, model, benchmark, idx)`` for a local *directory* source, and
    ``_get_one_sample_zip(zip_path, base_bench, idx)`` for a *zip member* source. These are the
    real per-source read seams: each resolves the output file for (model, benchmark), streams
    exactly the idx-th record with ijson (never loading the whole file), and returns
    ``(record, total_samples, media_base)``. They are fully hermetic — driven off a tempfile
    directory / temp ``.zip``, no network, no S3, no internal mounts, no demo data.
  * ``get_sample_detail(...)`` for the LOCAL path — the top-level dispatch — exercised with
    ``app.api.inference.INTERNAL_OUTPUTS_PATH`` monkeypatched to a temp dir and the scan
    inference-index + target caches cleared (resolution requires both). This proves the public
    contract (total_samples, predictions, GT, question, out-of-range 404) over the real seam.
  * ``_merge_one_record`` and ``_record_ground_truth`` directly — the merge/fold logic that both
    read paths feed into.

LOCAL vs ZIP EQUIVALENCE: the same eval JSON is read as a local dir file and as a zip member,
and the two are asserted to yield identical record / prediction / score / ground-truth / question
results. That equivalence is exactly what the refactor must preserve.

IMPORTANT current-behavior note pinned by these tests: the read path streams records with ijson,
which decodes JSON numbers that have a fractional part as ``decimal.Decimal``. ``_merge_one_record``
extracts scores via ``_as_score``, which coerces ``Decimal`` back to ``float``, so *fractional*
top-level/metric scores read through the streaming path are kept (as ``float``) alongside *integer*
scores. An explicit test (`test_local_fractional_scores_decode_as_decimal_and_are_kept_as_float`)
pins this. When ``_merge_one_record`` is driven directly with a native Python dict (no ijson),
floats are real ``float`` and likewise appear in scores — that path is pinned separately.
"""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import app.api.inference as inf
from app.services import scan


# A base64-looking blob (>=64 chars of base64 alphabet) prefixed with the PNG magic so the
# media extractor treats it as embedded image media and inlines it as a data: URI rather than
# echoing it into the question text. Used to prove base64 media never leaks into `question`.
B64_PNG = "iVBORw0KGgo" + "A" * 80
EXPECTED_IMG_DATA_URI = f"data:image/png;base64,{B64_PNG}"


def _record_0() -> dict:
    """Realistic first record: full prediction chain, integer scores, a metrics dict,
    ground_truth == 0 (falsy-but-valid), and a multimodal `messages` question carrying
    a text part plus a base64 image part."""
    return {
        "prediction_postprocessed": "PP0",
        "prediction": "P0",
        "answer": "A0",
        "label": 0,            # falsy-but-valid GT; int so it survives the score filter
        "exact_match": 1,      # int top-level score
        "ground_truth": 0,     # falsy-but-valid; preferred GT key
        "metrics": {"count": 3, "note": "not-a-score"},  # int metric survives; str dropped
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": "What is shown in the image?"},
                    {"type": "image", "value": B64_PNG},
                ],
            }
        ],
    }


def _record_1() -> dict:
    """Second/last record: no postprocessed pred (falls back to `prediction`), a `question`
    field, and label == False (falsy-but-valid bool GT)."""
    return {
        "prediction": "P1",
        "answer": "A1",
        "label": False,
        "score": 7,  # int top-level score
        "question": "Second question text",
    }


def _eval_json_with_header() -> dict:
    """Eval-output shape: records under the `inference` list-of-lists path
    (matches _RECORD_PATHS[0] 'inference.item.item'); header config carries num_records."""
    return {"config": {"num_records": 2}, "inference": [[_record_0(), _record_1()]]}


def _eval_json_no_header() -> dict:
    """Same records but NO num_records in the header, so total_samples must come from the
    streaming record count (the count_records / count_records_stream fallback)."""
    return {"inference": [[_record_0(), _record_1()]]}


def _empty_merge_result() -> dict:
    return {
        "question": "",
        "ground_truth": "",
        "choices": "",
        "predictions": [],
        "total_samples": 0,
    }


def _clear_caches() -> None:
    """Resolution for a local dir source goes through scan's inference-index TTL cache and
    inference.py's per-target LRU cache. Clear both so each test resolves fresh against its
    own temp dir (otherwise a prior test's path could be served)."""
    with scan._INF_INDEX_LOCK:
        scan._INF_INDEX_CACHE.clear()
    with inf._target_lock:
        inf._target_cache.clear()


class LocalDirReadMergeTests(unittest.TestCase):
    """Drives `_get_one_sample` over a local DIRECTORY source built in a tempdir, then folds
    each streamed record through `_merge_one_record` — the real read+merge dispatch."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.model = "mymodel"
        # Layout: <base>/<model>/inference_output/<bench>__<method>.json
        # _resolve_local_path returns <base>/<model>; find_inference_output rglobs
        # inference_output/*.json and keys by base benchmark ('mybench').
        out = self.base / self.model / "inference_output"
        out.mkdir(parents=True)
        (out / "mybench__builtin.json").write_bytes(
            json.dumps(_eval_json_with_header()).encode("utf-8")
        )
        self._orig_outputs = inf.INTERNAL_OUTPUTS_PATH
        inf.INTERNAL_OUTPUTS_PATH = self.base
        _clear_caches()

    def tearDown(self):
        inf.INTERNAL_OUTPUTS_PATH = self._orig_outputs
        _clear_caches()
        self._tmp.cleanup()

    def _read_and_merge(self, idx):
        rec, total, media_base = inf._get_one_sample("internal", self.model, "mybench", idx)
        self.assertIsNotNone(rec, f"record {idx} should resolve")
        result = _empty_merge_result()
        if total:
            result["total_samples"] = total
        inf._merge_one_record(result, rec, "internal", self.model, media_base)
        return rec, total, media_base, result

    def test_total_samples_from_header(self):
        _, total, _, _ = self._read_and_merge(0)
        self.assertEqual(total, 2)  # from config.num_records, no record walk

    def test_record_at_idx0_returned(self):
        rec, _, _, _ = self._read_and_merge(0)
        self.assertEqual(rec.get("prediction_postprocessed"), "PP0")
        self.assertEqual(rec.get("answer"), "A0")

    def test_record_at_last_index_returned(self):
        rec, _, _, result = self._read_and_merge(1)
        self.assertEqual(rec.get("prediction"), "P1")
        # No postprocessed pred on record 1 -> prediction wins the fallback chain.
        self.assertEqual(result["predictions"][0]["prediction"], "P1")

    def test_prediction_resolves_postprocessed_first(self):
        _, _, _, result = self._read_and_merge(0)
        self.assertEqual(result["predictions"][0]["prediction"], "PP0")

    def test_scores_include_integer_toplevel_and_metrics(self):
        _, _, _, result = self._read_and_merge(0)
        scores = result["predictions"][0]["scores"]
        # Integer top-level scores + integer metric survive; the string metric is excluded.
        self.assertEqual(scores, {"label": 0, "exact_match": 1, "ground_truth": 0, "count": 3})

    def test_ground_truth_preserves_falsy_zero(self):
        _, _, _, result = self._read_and_merge(0)
        self.assertEqual(result["ground_truth"], "0")  # GT == 0 must NOT be dropped

    def test_ground_truth_preserves_falsy_false(self):
        # Record 1 has only label == False as GT -> rendered "False", not "".
        _, _, _, result = self._read_and_merge(1)
        self.assertEqual(result["ground_truth"], "False")

    def test_question_text_extracted_from_messages(self):
        _, _, _, result = self._read_and_merge(0)
        self.assertEqual(result["question"], "What is shown in the image?")

    def test_base64_media_not_leaked_into_question(self):
        _, _, _, result = self._read_and_merge(0)
        self.assertNotIn(B64_PNG, result["question"])
        # And the base64 image is inlined as a data: URI under media.images instead.
        self.assertIn(EXPECTED_IMG_DATA_URI, result["media"]["images"])

    def test_total_samples_from_record_count_when_no_header(self):
        # Rewrite the file without num_records so the count fallback is exercised.
        out = self.base / self.model / "inference_output" / "mybench__builtin.json"
        out.write_bytes(json.dumps(_eval_json_no_header()).encode("utf-8"))
        _clear_caches()
        _, total, _, _ = self._read_and_merge(0)
        self.assertEqual(total, 2)

    def test_local_fractional_scores_decode_as_decimal_and_are_kept_as_float(self):
        """Floats read via ijson become Decimal; `_as_score` coerces them back to float
        so fractional top-level/metric scores survive the merge (alongside int scores)."""
        rec = {
            "prediction": "P",
            "accuracy": 0.9,                 # fractional -> Decimal -> coerced to float
            "exact_match": 1,                # int -> kept
            "metrics": {"f1": 0.5, "n": 2},  # f1 fractional -> float; n int -> kept
        }
        out = self.base / self.model / "inference_output" / "mybench__builtin.json"
        out.write_bytes(json.dumps({"config": {"num_records": 1}, "inference": [[rec]]}).encode())
        _clear_caches()
        _, _, _, result = self._read_and_merge(0)
        scores = result["predictions"][0]["scores"]
        self.assertEqual(scores, {"accuracy": 0.9, "exact_match": 1, "f1": 0.5, "n": 2})
        # Coerced Decimals are real floats (JSON-serializable), not Decimal.
        self.assertIsInstance(scores["accuracy"], float)
        self.assertIsInstance(scores["f1"], float)

    def test_out_of_range_index_array_matched_returns_none(self):
        # The array is found but idx is beyond it -> _get_one_sample returns (None, total, base).
        rec, total, _ = inf._get_one_sample("internal", self.model, "mybench", 99)
        self.assertIsNone(rec)
        self.assertEqual(total, 2)


class ZipReadMergeTests(unittest.TestCase):
    """Drives `_get_one_sample_zip` over a temp .zip containing the SAME eval JSON as a member."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.zpath = Path(self._tmp.name) / "mymodel.zip"
        with zipfile.ZipFile(self.zpath, "w") as zf:
            zf.writestr(
                "mymodel/inference_output/mybench__builtin.json",
                json.dumps(_eval_json_with_header()),
            )

    def tearDown(self):
        self._tmp.cleanup()

    def _read_and_merge(self, idx):
        rec, total, media_base = inf._get_one_sample_zip(self.zpath, "mybench", idx)
        self.assertIsNotNone(rec, f"zip record {idx} should resolve")
        result = _empty_merge_result()
        if total:
            result["total_samples"] = total
        inf._merge_one_record(result, rec, "internal", "mymodel", media_base)
        return rec, total, media_base, result

    def test_total_samples_from_header(self):
        _, total, _, _ = self._read_and_merge(0)
        self.assertEqual(total, 2)

    def test_media_base_is_none_for_zip(self):
        # Zip members have no resolvable filesystem base dir; embedded media still inlines.
        _, _, media_base, _ = self._read_and_merge(0)
        self.assertIsNone(media_base)

    def test_total_samples_from_record_count_when_no_header(self):
        zp = Path(self._tmp.name) / "noheader.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr(
                "mymodel/inference_output/mybench__builtin.json",
                json.dumps(_eval_json_no_header()),
            )
        rec, total, _ = inf._get_one_sample_zip(zp, "mybench", 0)
        self.assertIsNotNone(rec)
        self.assertEqual(total, 2)

    def test_out_of_range_index_returns_none_with_total(self):
        rec, total, _ = inf._get_one_sample_zip(self.zpath, "mybench", 99)
        self.assertIsNone(rec)
        self.assertEqual(total, 2)

    def test_missing_benchmark_returns_empty(self):
        rec, total, base = inf._get_one_sample_zip(self.zpath, "nosuchbench", 0)
        self.assertIsNone(rec)
        self.assertEqual(total, 0)
        self.assertIsNone(base)


class ZipOversizeGuardTests(unittest.TestCase):
    """Pins the per-sample size guard: a member over INFERENCE_SAMPLE_MAX_BYTES is NOT
    materialised to preview one record (memory protection on a 1GB box); the viewer gets a
    'Preview unavailable' sentinel plus the head-derived total instead, and an under-cap
    member still reads normally."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.zpath = Path(self._tmp.name) / "mymodel.zip"
        with zipfile.ZipFile(self.zpath, "w") as zf:
            zf.writestr(
                "mymodel/inference_output/mybench__builtin.json",
                json.dumps(_eval_json_with_header()),
            )
        self._orig_cap = inf.INFERENCE_SAMPLE_MAX_BYTES

    def tearDown(self):
        inf.INFERENCE_SAMPLE_MAX_BYTES = self._orig_cap
        self._tmp.cleanup()

    def test_oversized_member_returns_preview_unavailable_with_head_total(self):
        inf.INFERENCE_SAMPLE_MAX_BYTES = 1  # force the guard (member is bigger than 1 byte)
        rec, total, base = inf._get_one_sample_zip(self.zpath, "mybench", 0)
        self.assertIsNotNone(rec)
        self.assertIn("Preview unavailable", rec.get("prediction", ""))
        self.assertEqual(total, 2)  # still derived from the head slice, no body read
        self.assertIsNone(base)
        # The sentinel folds into the viewer response as the model's prediction text.
        result = _empty_merge_result()
        result["total_samples"] = total
        inf._merge_one_record(result, rec, "internal", "mymodel", base)
        self.assertIn("Preview unavailable", result["predictions"][0]["prediction"])

    def test_under_cap_member_reads_normally(self):
        inf.INFERENCE_SAMPLE_MAX_BYTES = self._orig_cap  # default 64 MiB; member is tiny
        rec, total, _ = inf._get_one_sample_zip(self.zpath, "mybench", 0)
        self.assertEqual(rec.get("prediction_postprocessed"), "PP0")
        self.assertEqual(total, 2)


class LocalZipEquivalenceTests(unittest.TestCase):
    """The crux for the refactor: reading the SAME eval JSON via the local-dir seam and via the
    zip-member seam must produce identical record / prediction / score / GT / question results.
    Media differs only in base (local resolves a base_dir, zip has None) but inlined base64 is
    identical, so we compare the inlined image set too."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        payload = json.dumps(_eval_json_with_header())

        # Local dir source
        self.model = "mymodel"
        out = root / self.model / "inference_output"
        out.mkdir(parents=True)
        (out / "mybench__builtin.json").write_bytes(payload.encode("utf-8"))
        self._orig_outputs = inf.INTERNAL_OUTPUTS_PATH
        inf.INTERNAL_OUTPUTS_PATH = root
        _clear_caches()

        # Zip source
        self.zpath = root / "mymodel.zip"
        with zipfile.ZipFile(self.zpath, "w") as zf:
            zf.writestr("mymodel/inference_output/mybench__builtin.json", payload)

    def tearDown(self):
        inf.INTERNAL_OUTPUTS_PATH = self._orig_outputs
        _clear_caches()
        self._tmp.cleanup()

    def _merge_local(self, idx):
        rec, total, base = inf._get_one_sample("internal", self.model, "mybench", idx)
        result = _empty_merge_result()
        result["total_samples"] = total
        inf._merge_one_record(result, rec, "internal", self.model, base)
        return result

    def _merge_zip(self, idx):
        rec, total, base = inf._get_one_sample_zip(self.zpath, "mybench", idx)
        result = _empty_merge_result()
        result["total_samples"] = total
        inf._merge_one_record(result, rec, "internal", self.model, base)
        return result

    def _assert_equivalent(self, idx):
        local = self._merge_local(idx)
        zipd = self._merge_zip(idx)
        self.assertEqual(local["total_samples"], zipd["total_samples"])
        self.assertEqual(
            local["predictions"][0]["prediction"], zipd["predictions"][0]["prediction"]
        )
        self.assertEqual(
            local["predictions"][0]["scores"], zipd["predictions"][0]["scores"]
        )
        self.assertEqual(local["ground_truth"], zipd["ground_truth"])
        self.assertEqual(local["question"], zipd["question"])
        # Inlined embedded media must be identical across both seams.
        self.assertEqual(
            (local.get("media") or {}).get("images"),
            (zipd.get("media") or {}).get("images"),
        )

    def test_equivalent_at_idx0(self):
        self._assert_equivalent(0)

    def test_equivalent_at_last_index(self):
        self._assert_equivalent(1)


class GetSampleDetailLocalTests(unittest.TestCase):
    """Exercises the public top-level dispatch `get_sample_detail` over the LOCAL seam."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.model = "mymodel"
        out = self.base / self.model / "inference_output"
        out.mkdir(parents=True)
        (out / "mybench__builtin.json").write_bytes(
            json.dumps(_eval_json_with_header()).encode("utf-8")
        )
        self._orig_outputs = inf.INTERNAL_OUTPUTS_PATH
        inf.INTERNAL_OUTPUTS_PATH = self.base
        _clear_caches()

    def tearDown(self):
        inf.INTERNAL_OUTPUTS_PATH = self._orig_outputs
        _clear_caches()
        self._tmp.cleanup()

    def test_detail_idx0(self):
        detail = inf.get_sample_detail(0, model_ids="internal:mymodel", benchmark="mybench")
        self.assertEqual(detail["total_samples"], 2)
        self.assertEqual(len(detail["predictions"]), 1)
        pred = detail["predictions"][0]
        self.assertEqual(pred["model"], "mymodel")
        self.assertEqual(pred["source"], "internal")
        self.assertEqual(pred["prediction"], "PP0")
        self.assertEqual(pred["scores"], {"label": 0, "exact_match": 1, "ground_truth": 0, "count": 3})
        self.assertEqual(detail["ground_truth"], "0")
        self.assertEqual(detail["question"], "What is shown in the image?")
        self.assertNotIn(B64_PNG, detail["question"])
        self.assertEqual(detail["media"]["images"], [EXPECTED_IMG_DATA_URI])

    def test_detail_last_index(self):
        detail = inf.get_sample_detail(1, model_ids="internal:mymodel", benchmark="mybench")
        self.assertEqual(detail["predictions"][0]["prediction"], "P1")
        self.assertEqual(detail["ground_truth"], "False")

    def test_detail_out_of_range_raises_404(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as ctx:
            inf.get_sample_detail(99, model_ids="internal:mymodel", benchmark="mybench")
        self.assertEqual(ctx.exception.status_code, 404)


class MergeOneRecordDirectTests(unittest.TestCase):
    """`_merge_one_record` driven directly with native Python dicts (no ijson). Here floats are
    real `float` and DO land in scores — distinct from the streaming path's Decimal behavior."""

    def test_prediction_fallback_chain(self):
        # postprocessed > prediction > answer > '-'
        r = _empty_merge_result()
        inf._merge_one_record(r, {"prediction_postprocessed": "PP", "prediction": "P", "answer": "A"}, "s", "m", None)
        self.assertEqual(r["predictions"][0]["prediction"], "PP")

        r = _empty_merge_result()
        inf._merge_one_record(r, {"prediction": "P", "answer": "A"}, "s", "m", None)
        self.assertEqual(r["predictions"][0]["prediction"], "P")

        r = _empty_merge_result()
        inf._merge_one_record(r, {"answer": "A"}, "s", "m", None)
        self.assertEqual(r["predictions"][0]["prediction"], "A")

        r = _empty_merge_result()
        inf._merge_one_record(r, {"foo": "bar"}, "s", "m", None)
        self.assertEqual(r["predictions"][0]["prediction"], "-")

    def test_scores_dict_native_floats_and_metrics(self):
        r = _empty_merge_result()
        rec = {
            "prediction": "P",
            "accuracy": 0.9,   # native float -> kept (no ijson here)
            "cnt": 3,          # int -> kept
            "label": 0,        # int -> kept
            "tag": "x",        # str -> dropped
            "metrics": {"f1": 0.5, "k": 2, "s": "y"},  # f1,k kept; s dropped
        }
        inf._merge_one_record(r, rec, "s", "m", None)
        self.assertEqual(
            r["predictions"][0]["scores"],
            {"accuracy": 0.9, "cnt": 3, "label": 0, "f1": 0.5, "k": 2},
        )

    def test_multi_model_merge_gt_from_first_record_that_has_it(self):
        """N-model merge: predictions accumulate; question/GT/choices are set once, from the
        first record that actually supplies a (truthy) question. A record with no usable
        question/GT leaves them empty so a later model can fill them."""
        result = _empty_merge_result()
        # Model 1: no usable prediction (-> '-'), no question, no GT keys.
        inf._merge_one_record(result, {"prediction": None, "answer": None}, "internal", "M1", None)
        # Model 2: real prediction, question, and GT == 0 (falsy-but-valid).
        inf._merge_one_record(result, {"prediction": "M2pred", "question": "qq", "ground_truth": 0}, "internal", "M2", None)

        self.assertEqual(len(result["predictions"]), 2)
        self.assertEqual(result["predictions"][0]["prediction"], "-")
        self.assertEqual(result["predictions"][1]["prediction"], "M2pred")
        self.assertEqual(result["question"], "qq")
        self.assertEqual(result["ground_truth"], "0")  # 0 from M2 preserved

    def test_first_record_with_question_wins_and_locks_gt(self):
        # Once a truthy question is set, a later record's GT/question are NOT overwritten.
        result = _empty_merge_result()
        inf._merge_one_record(result, {"prediction": "P1", "question": "first", "ground_truth": "GT1"}, "s", "M1", None)
        inf._merge_one_record(result, {"prediction": "P2", "question": "second", "ground_truth": "GT2"}, "s", "M2", None)
        self.assertEqual(result["question"], "first")
        self.assertEqual(result["ground_truth"], "GT1")


class RecordGroundTruthTests(unittest.TestCase):
    """`_record_ground_truth`: KEY PRESENCE (value is not None), not truthiness, so falsy-but-valid
    0 / False / '' survive. Tries keys in priority order, skipping any whose value is None."""

    def test_falsy_zero_survives(self):
        self.assertEqual(inf._record_ground_truth({"label": 0}), "0")

    def test_falsy_false_survives(self):
        self.assertEqual(inf._record_ground_truth({"label": False}), "False")

    def test_empty_string_present_survives(self):
        # Present (not None) -> returned as the empty string, not skipped.
        self.assertEqual(inf._record_ground_truth({"ground_truth": ""}), "")

    def test_none_value_is_skipped_to_next_key(self):
        self.assertEqual(inf._record_ground_truth({"ground_truth": None, "label": 0}), "0")

    def test_priority_ground_truth_over_label(self):
        self.assertEqual(inf._record_ground_truth({"ground_truth": "GT", "label": 9}), "GT")

    def test_falls_through_key_order(self):
        # No ground_truth/label -> answer is the next key consulted.
        self.assertEqual(inf._record_ground_truth({"answer": "ANS"}), "ANS")

    def test_no_known_key_returns_empty(self):
        self.assertEqual(inf._record_ground_truth({"foo": 1}), "")


if __name__ == "__main__":
    unittest.main()
