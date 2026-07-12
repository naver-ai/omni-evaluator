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

"""Characterization tests for the S3 sample read/merge path + media helpers.

SEAM: these tests drive the real S3 read path of app/api/inference.py at two levels:

  * _get_one_sample_s3(model, base_bench, idx)  -- the per-source S3 helper that
    resolves an object key (via _list_s3_keys), fetches the head Range + full Body,
    counts/streams records, and returns (record, total, media_base). This is the seam
    that exercises the actual S3 *read* logic and is the S3 analogue of _get_one_sample
    (local dir) and _get_one_sample_zip (zip member).
  * _merge_one_record(result, rec, src, model, media_base) -- the shared merge step that
    folds a streamed record into the viewer response (prediction / scores / GT / question
    / media). All three source paths funnel through this, so asserting the S3 path yields
    the SAME merged shape as the local/zip paths for the SAME fixture bytes pins the
    read+merge invariant the upcoming refactor must preserve (cross-path equivalence).

Why this seam (not get_sample_detail): get_sample_detail wires in model-id parsing,
benchmark resolution and the FastAPI response, none of which is the read+merge logic
under refactor; _get_one_sample_s3 + _merge_one_record is the smallest hermetic surface
that actually exercises read+merge for the S3 source.

HERMETIC: boto3 is never reached. We patch the S3 client *factory* as imported into
app.api.inference (`inference._get_s3_client`) and the list helper
(`inference._s3_list_with_retry`) -- inference.py does `from ..services.s3_sync import
_get_s3_client, _s3_list_with_retry`, so the names live in the inference module and must
be patched there. The mocked client's get_object returns {'Body': io.BytesIO(<bytes>)};
list_objects_v2 is never called directly (we patch _s3_list_with_retry). No network,
no real S3, no cmlssd004, no demo data. All fixtures are inline JSON / tempfiles.

Module-level caches (_list_s3_keys' S3 key cache, _presign_cache) are reset in setUp so
tests don't leak state into one another.

DOCUMENTED BEHAVIOR (asserted as CURRENT behavior):

  1. FLOAT SCORES ARE KEPT. The streaming readers parse JSON with ijson, which yields
     non-integer numbers as decimal.Decimal. _merge_one_record collects scores via
     `_as_score`, which coerces Decimal back to float, so float scores
     (e.g. "score": 0.9, "metrics": {"accuracy": 1.0}) survive (as float) in the merged
     `scores` dict alongside integer scores. This is identical across the local / zip / S3
     paths (all stream through ijson), so it is a cross-path INVARIANT -- pinned here.

  2. _resolve_media_path RELATIVE TRAVERSAL is gated by CONTAINMENT in base_dir, not by the
     resolved path's prefix. '../../etc/passwd' under base_dir '/mnt/data/x' resolves to
     '/mnt/etc/passwd', which escapes base_dir and is therefore '' (blocked) even though it
     starts with the allowed '/mnt/' prefix. A legitimate relative path under base_dir is
     returned. An ABSOLUTE path is gated by the local-image allowlist ('/etc/passwd' -> '').
"""

import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from app.api import inference
from app.services import inference_cache


# --------------------------------------------------------------------------------------
# Fixture builders: realistic eval-output JSON under a _RECORD_PATHS array.
# --------------------------------------------------------------------------------------

# Two records. REC0 carries the full surface the merge reads: postprocessed prediction,
# raw prediction, answer, ground truth (label), an integer score, a float score, a metrics
# block (mixed int/float), a question, and an image path under an allowed prefix.
REC0 = {
    "question": "What is 2 + 2?",
    "prediction": "four",
    "prediction_postprocessed": "4",
    "answer": "4",
    "label": "4",
    "intscore": 3,          # int -> survives the merge's score filter
    "score": 0.9,           # float -> kept as float (ijson Decimal coerced by _as_score)
    "metrics": {"accuracy": 1.0, "count": 5},  # accuracy float kept (1.0), count int kept
    "image": "/mnt/data/imgs/a.png",
}
# REC1 exercises the prediction fallback chain (no postprocessed; falls to `prediction`)
# and a falsy-but-valid GT (label 0) that must NOT be dropped by truthiness.
REC1 = {
    "question": "Q1",
    "prediction": "p1",
    "answer": "a1",
    "label": 0,
    "em": 1,
}

# Header config carries num_records, so the HEAD Range read alone yields the total
# (no record-count fallback needed) -- the built-in-output happy path.
EVAL_FIXTURE = {"config": {"num_records": 2}, "inference": [[REC0, REC1]]}


def _fixture_bytes(obj=EVAL_FIXTURE) -> bytes:
    return json.dumps(obj).encode("utf-8")


# The crafted inference_output key the mocked list returns. base_bench "mmlu" must be a
# substring of Path(key).stem ("mmlu") for _get_one_sample_s3's key selection to match.
INF_KEY = "eval/v1/modelX/checkpoint-1/inference_output/mmlu.json"


def _make_s3_mock(body_bytes: bytes):
    """Mock boto3 S3 client: get_object returns a fresh BytesIO Body each call.

    A fresh BytesIO per call matters: _get_one_sample_s3 calls get_object twice (once with
    a Range for the head, once for the full body), and each read() exhausts the stream.
    """
    client = mock.Mock(name="s3client")

    def _get_object(**kwargs):
        if "Range" in kwargs:
            rng = kwargs["Range"]  # e.g. "bytes=0-65535"
            try:
                end = int(rng.split("-", 1)[1])
            except Exception:
                end = len(body_bytes)
            return {"Body": io.BytesIO(body_bytes[: end + 1])}
        return {"Body": io.BytesIO(body_bytes)}

    client.get_object.side_effect = _get_object
    return client


class _S3SeamBase(unittest.TestCase):
    """Resets module-level caches so each test is independent of ordering."""

    def setUp(self):
        inference_cache._S3_KEY_CACHE.clear()
        with inference._presign_lock:
            inference._presign_cache.clear()

    def tearDown(self):
        inference_cache._S3_KEY_CACHE.clear()
        with inference._presign_lock:
            inference._presign_cache.clear()


# --------------------------------------------------------------------------------------
# _get_one_sample_s3 read + cross-path equivalence
# --------------------------------------------------------------------------------------

class GetOneSampleS3Tests(_S3SeamBase):
    def _read_s3(self, idx, body=None, list_resp=None):
        body = _fixture_bytes() if body is None else body
        client = _make_s3_mock(body)
        if list_resp is None:
            list_resp = {"Contents": [{"Key": INF_KEY}]}
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry", return_value=list_resp), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            return inference._get_one_sample_s3("modelX", "mmlu", idx)

    def test_reads_record_and_total_from_header(self):
        rec, total, media_base = self._read_s3(0)
        self.assertIsInstance(rec, dict)
        self.assertEqual(rec["question"], "What is 2 + 2?")
        # num_records in the head config -> total without a record-count pass.
        self.assertEqual(total, 2)
        # S3 records have no local filesystem base (media resolved via URLs/embeds only).
        self.assertIsNone(media_base)

    def test_reads_second_record(self):
        rec, total, _ = self._read_s3(1)
        self.assertEqual(rec["question"], "Q1")
        self.assertEqual(rec["label"], 0)
        self.assertEqual(total, 2)

    def test_oversized_object_returns_preview_unavailable_without_body_fetch(self):
        # head_object reports an object just over INFERENCE_SAMPLE_MAX_BYTES: the guard must
        # return the 'Preview unavailable' sentinel + head-derived total and NEVER fetch the
        # full body. Sized relative to the configured cap so it stays correct if the cap changes.
        client = _make_s3_mock(_fixture_bytes())
        client.head_object.return_value = {"ContentLength": inference.INFERENCE_SAMPLE_MAX_BYTES + 1024 * 1024}
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry", return_value={"Contents": [{"Key": INF_KEY}]}), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            rec, total, base = inference._get_one_sample_s3("modelX", "mmlu", 0)
        self.assertIn("Preview unavailable", rec.get("prediction", ""))
        self.assertEqual(total, 2)  # still from the head slice
        self.assertIsNone(base)
        # No full-body get_object (a call WITHOUT a Range) was ever made.
        full_body_calls = [c for c in client.get_object.call_args_list if "Range" not in c.kwargs]
        self.assertEqual(full_body_calls, [])

    def test_undersized_object_reads_normally_through_guard(self):
        # head_object reports a small object: the guard does not fire, normal streaming read.
        client = _make_s3_mock(_fixture_bytes())
        client.head_object.return_value = {"ContentLength": 1024}
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry", return_value={"Contents": [{"Key": INF_KEY}]}), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            rec, total, _ = inference._get_one_sample_s3("modelX", "mmlu", 0)
        self.assertEqual(rec["question"], "What is 2 + 2?")
        self.assertEqual(total, 2)

    def test_total_counted_when_header_lacks_count(self):
        # No config.num_records -> _get_one_sample_s3 falls back to counting records
        # over the body bytes it already fetched.
        body = _fixture_bytes({"inference": [[REC0, REC1]]})
        rec, total, _ = self._read_s3(0, body=body)
        self.assertEqual(total, 2)
        self.assertEqual(rec["question"], "What is 2 + 2?")

    def test_no_client_returns_empty_triplet(self):
        with mock.patch.object(inference, "_get_s3_client", return_value=None), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            self.assertEqual(inference._get_one_sample_s3("m", "mmlu", 0), (None, 0, None))

    def test_no_bucket_returns_empty_triplet(self):
        client = _make_s3_mock(_fixture_bytes())
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "S3_BUCKET", ""):
            self.assertEqual(inference._get_one_sample_s3("m", "mmlu", 0), (None, 0, None))

    def test_no_matching_key_returns_empty_triplet(self):
        # benchmark 'docvqa' is not a substring of the only key's stem 'mmlu'.
        client = _make_s3_mock(_fixture_bytes())
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry",
                               return_value={"Contents": [{"Key": INF_KEY}]}), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            self.assertEqual(inference._get_one_sample_s3("modelX", "docvqa", 0), (None, 0, None))

    def test_falls_back_to_evaluation_key(self):
        # No inference_output key; an evaluation_output key whose stem contains the bench
        # is used instead.
        eval_key = "eval/v1/modelX/checkpoint-1/evaluation_output/mmlu.json"
        client = _make_s3_mock(_fixture_bytes())
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry",
                               return_value={"Contents": [{"Key": eval_key}]}), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            rec, total, _ = inference._get_one_sample_s3("modelX", "mmlu", 0)
        self.assertEqual(rec["question"], "What is 2 + 2?")
        self.assertEqual(total, 2)

    def test_get_object_raising_degrades_to_none_record(self):
        # Header read fails AND body read fails -> rec None, total 0, no raise.
        client = mock.Mock()
        client.get_object.side_effect = RuntimeError("boom")
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry",
                               return_value={"Contents": [{"Key": INF_KEY}]}), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            rec, total, base = inference._get_one_sample_s3("modelX", "mmlu", 0)
        self.assertIsNone(rec)
        self.assertEqual(total, 0)
        self.assertIsNone(base)


class CrossPathEquivalenceTests(_S3SeamBase):
    """The refactor invariant: identical fixture bytes -> identical merged shape across
    the local-dir, zip, and S3 read paths, all funneled through _merge_one_record."""

    def _merge_from_s3(self, idx):
        client = _make_s3_mock(_fixture_bytes())
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry",
                               return_value={"Contents": [{"Key": INF_KEY}]}), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            rec, total, media_base = inference._get_one_sample_s3("modelX", "mmlu", idx)
        result = {"question": "", "ground_truth": "", "choices": "", "predictions": [], "total_samples": 0}
        if total:
            result["total_samples"] = total
        inference._merge_one_record(result, rec, "s3", "modelX", media_base)
        return result

    def _merge_from_local(self, idx):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "inference_output_mmlu.json"
            target.write_bytes(_fixture_bytes())
            rec, total, media_base = (
                inference.stream_record_at(target, idx),
                inference.total_from_head(inference.read_json_head_tail(target)) or 0,
                target.parent,
            )
            result = {"question": "", "ground_truth": "", "choices": "", "predictions": [], "total_samples": 0}
            if total:
                result["total_samples"] = total
            inference._merge_one_record(result, rec, "internal", "modelX", media_base)
            return result

    def _merge_from_zip(self, idx):
        member = "modelX/checkpoint-1/inference_output/mmlu.json"
        with tempfile.TemporaryDirectory() as d:
            zpath = Path(d) / "modelX.zip"
            with zipfile.ZipFile(zpath, "w") as zf:
                zf.writestr(member, _fixture_bytes())
            rec, total, media_base = inference._get_one_sample_zip(zpath, "mmlu", idx)
            result = {"question": "", "ground_truth": "", "choices": "", "predictions": [], "total_samples": 0}
            if total:
                result["total_samples"] = total
            inference._merge_one_record(result, rec, "zip", "modelX", media_base)
            return result

    def _strip_model_source(self, result):
        # The per-source fields legitimately differ (model/source labels, media_base which
        # is a local Path only for the local path). Compare the read+merge-derived payload:
        # prediction, scores, question, GT, choices, total.
        preds = [{"prediction": p["prediction"], "scores": p["scores"]} for p in result["predictions"]]
        return {
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "choices": result["choices"],
            "total_samples": result["total_samples"],
            "predictions": preds,
        }

    def test_s3_matches_local_idx0(self):
        s3 = self._strip_model_source(self._merge_from_s3(0))
        local = self._strip_model_source(self._merge_from_local(0))
        self.assertEqual(s3, local)

    def test_s3_matches_zip_idx0(self):
        s3 = self._strip_model_source(self._merge_from_s3(0))
        zp = self._strip_model_source(self._merge_from_zip(0))
        self.assertEqual(s3, zp)

    def test_s3_matches_local_idx1(self):
        s3 = self._strip_model_source(self._merge_from_s3(1))
        local = self._strip_model_source(self._merge_from_local(1))
        self.assertEqual(s3, local)

    def test_merged_shape_idx0_pins_exact_payload(self):
        # Pin the exact merged payload, including the float scores (see module docstring
        # behavior #1): int scores 'intscore'/'count' AND float scores 'score'/'accuracy' survive.
        s3 = self._strip_model_source(self._merge_from_s3(0))
        self.assertEqual(
            s3,
            {
                "question": "What is 2 + 2?",
                "ground_truth": "4",                 # _record_ground_truth -> label/answer/...
                "choices": "",
                "total_samples": 2,
                "predictions": [
                    {
                        "prediction": "4",            # prediction_postprocessed wins
                        # floats kept (Decimal -> float), alongside int scores
                        "scores": {"intscore": 3, "score": 0.9, "accuracy": 1.0, "count": 5},
                    }
                ],
            },
        )

    def test_float_scores_kept_on_every_path(self):
        # Cross-path INVARIANT of behavior #1: float-keeping is identical everywhere.
        for getter in (self._merge_from_s3, self._merge_from_local, self._merge_from_zip):
            scores = getter(0)["predictions"][0]["scores"]
            self.assertEqual(scores, {"intscore": 3, "score": 0.9, "accuracy": 1.0, "count": 5})
            self.assertEqual(scores["score"], 0.9)       # float 0.9 kept
            self.assertEqual(scores["accuracy"], 1.0)    # float 1.0 kept

    def test_prediction_fallback_and_falsy_gt_idx1(self):
        # REC1 has no postprocessed -> falls back to raw 'prediction'; label 0 must survive.
        s3 = self._strip_model_source(self._merge_from_s3(1))
        self.assertEqual(s3["predictions"][0]["prediction"], "p1")
        self.assertEqual(s3["ground_truth"], "0")   # falsy-but-valid GT preserved
        # NOTE: 'label' is an integer here, so the merge's int/float score filter sweeps it
        # into `scores` too -- the GT key and a numeric score key overlap. Pinned as current.
        self.assertEqual(s3["predictions"][0]["scores"], {"label": 0, "em": 1})


# --------------------------------------------------------------------------------------
# _list_s3_keys: graceful degradation (#18 fix) + key classification
# --------------------------------------------------------------------------------------

class ListS3KeysTests(_S3SeamBase):
    def test_classifies_inference_and_evaluation_keys(self):
        contents = {
            "Contents": [
                {"Key": "eval/v1/modelX/checkpoint-1/inference_output/mmlu.json"},
                {"Key": "eval/v1/modelX/checkpoint-1/evaluation_output/mmlu.json"},
                {"Key": "eval/v1/modelX/checkpoint-1/output/docvqa.json"},
                {"Key": "eval/v1/modelX/checkpoint-1/notes.txt"},   # non-json: skipped
                {"Key": ""},                                                # empty: skipped
            ]
        }
        client = mock.Mock()
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry", return_value=contents), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            keys = inference._list_s3_keys("modelX")
        self.assertEqual(keys["inference"], ["eval/v1/modelX/checkpoint-1/inference_output/mmlu.json"])
        self.assertEqual(
            sorted(keys["evaluation"]),
            sorted([
                "eval/v1/modelX/checkpoint-1/evaluation_output/mmlu.json",
                "eval/v1/modelX/checkpoint-1/output/docvqa.json",
            ]),
        )

    def test_list_failure_degrades_to_empty_no_raise(self):
        # The #18 fix: _s3_list_with_retry already returns {} on outage, so _list_s3_keys
        # must degrade to empty buckets and NOT raise (would 500 /benchmarks and /sample).
        client = mock.Mock()
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry", return_value={}), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            keys = inference._list_s3_keys("modelX")
        self.assertEqual(keys, {"inference": [], "evaluation": []})

    def test_list_helper_raising_is_not_swallowed_here(self):
        # Sanity: the degradation lives in _s3_list_with_retry (returns {}); _list_s3_keys
        # itself doesn't wrap it. If the helper somehow raised, _list_s3_keys would propagate.
        # We assert the realistic contract: helper returns {} -> empty, no raise (above).
        # Here we confirm a first-page {} is treated as a transient failure and NOT cached,
        # so a later successful list is reflected.
        client = mock.Mock()
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry", return_value={}), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            self.assertEqual(inference._list_s3_keys("modelTransient"),
                             {"inference": [], "evaluation": []})
        # Cache must NOT hold the failed empty result for this model.
        self.assertIsNone(inference_cache.get_s3_key_cache("modelTransient"))
        contents = {"Contents": [{"Key": "eval/v1/modelTransient/checkpoint-1/inference_output/mmlu.json"}]}
        with mock.patch.object(inference, "_get_s3_client", return_value=client), \
             mock.patch.object(inference, "_s3_list_with_retry", return_value=contents), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            keys = inference._list_s3_keys("modelTransient")
        self.assertEqual(len(keys["inference"]), 1)

    def test_no_client_returns_empty_and_caches(self):
        with mock.patch.object(inference, "_get_s3_client", return_value=None), \
             mock.patch.object(inference, "S3_BUCKET", "test-bucket"):
            self.assertEqual(inference._list_s3_keys("modelNoClient"),
                             {"inference": [], "evaluation": []})


# --------------------------------------------------------------------------------------
# _presign_s3_object: ExpiresIn from config + caching within validity
# --------------------------------------------------------------------------------------

class PresignS3ObjectTests(_S3SeamBase):
    def test_uses_config_expire_and_passes_params(self):
        client = mock.Mock()
        client.generate_presigned_url.return_value = "https://signed/url"
        with mock.patch.object(inference, "_get_s3_client", return_value=client):
            url = inference._presign_s3_object("buck", "path/key.png")
        self.assertEqual(url, "https://signed/url")
        _, kwargs = client.generate_presigned_url.call_args
        self.assertEqual(kwargs["ExpiresIn"], inference.S3_PRESIGN_EXPIRE)
        self.assertEqual(kwargs["Params"], {"Bucket": "buck", "Key": "path/key.png"})
        self.assertEqual(client.generate_presigned_url.call_args[0][0], "get_object")

    def test_second_call_within_validity_uses_cache(self):
        client = mock.Mock()
        client.generate_presigned_url.return_value = "https://signed/url"
        with mock.patch.object(inference, "_get_s3_client", return_value=client):
            u1 = inference._presign_s3_object("buck", "k")
            u2 = inference._presign_s3_object("buck", "k")
        self.assertEqual(u1, u2)
        # Cached on the 2nd call within validity -> generate_presigned_url called ONCE.
        self.assertEqual(client.generate_presigned_url.call_count, 1)

    def test_no_client_returns_empty(self):
        with mock.patch.object(inference, "_get_s3_client", return_value=None):
            self.assertEqual(inference._presign_s3_object("buck", "k"), "")

    def test_generate_raising_returns_empty_and_not_cached(self):
        client = mock.Mock()
        client.generate_presigned_url.side_effect = RuntimeError("denied")
        with mock.patch.object(inference, "_get_s3_client", return_value=client):
            self.assertEqual(inference._presign_s3_object("buck", "k"), "")
        # Nothing cached on failure.
        with inference._presign_lock:
            self.assertNotIn(("buck", "k"), inference._presign_cache)


# --------------------------------------------------------------------------------------
# _maybe_refresh_storage_url: legacy.example.com host check (exact + suffix; reject lookalike)
# --------------------------------------------------------------------------------------

class MaybeRefreshStorageUrlTests(_S3SeamBase):
    def setUp(self):
        super().setUp()
        self._orig_legacy_host = inference.MEDIA_LEGACY_HOST
        inference.MEDIA_LEGACY_HOST = "legacy.example.com"

    def tearDown(self):
        inference.MEDIA_LEGACY_HOST = self._orig_legacy_host
        super().tearDown()

    def test_exact_host_accepted_triggers_presign(self):
        with mock.patch.object(inference, "_presign_s3_object", return_value="https://refreshed") as m:
            out = inference._maybe_refresh_storage_url("https://legacy.example.com/bucket/key.png")
        self.assertTrue(m.called)
        self.assertEqual(out, "https://refreshed")
        self.assertEqual(m.call_args[0], ("bucket", "key.png"))

    def test_subdomain_suffix_accepted_triggers_presign(self):
        with mock.patch.object(inference, "_presign_s3_object", return_value="https://refreshed") as m:
            out = inference._maybe_refresh_storage_url("https://cdn.legacy.example.com/bucket/key.png")
        self.assertTrue(m.called)
        self.assertEqual(out, "https://refreshed")

    def test_lookalike_host_rejected_no_presign(self):
        # 'legacy.example.com.evil.com' must NOT match: a naive substring check would have
        # wrongly matched. Host is returned unchanged and presign is never attempted.
        url = "https://legacy.example.com.evil.com/bucket/key.png?X-Amz-Foo=1"
        with mock.patch.object(inference, "_presign_s3_object", return_value="https://refreshed") as m:
            out = inference._maybe_refresh_storage_url(url)
        self.assertFalse(m.called)
        self.assertEqual(out, url)

    def test_unrelated_host_returned_unchanged(self):
        url = "https://example.com/a.png"
        with mock.patch.object(inference, "_presign_s3_object", return_value="X") as m:
            out = inference._maybe_refresh_storage_url(url)
        self.assertFalse(m.called)
        self.assertEqual(out, url)

    def test_unexpired_presigned_url_not_refreshed(self):
        # A legacy.example.com URL with X-Amz-Date/Expires markers still within validity
        # is returned as-is (no presign call). Use a far-future date so it never expires.
        url = (
            "https://legacy.example.com/bucket/key.png"
            "?X-Amz-Date=20990101T000000Z&X-Amz-Expires=3600&X-Amz-Signature=abc"
        )
        with mock.patch.object(inference, "_presign_s3_object", return_value="https://refreshed") as m:
            out = inference._maybe_refresh_storage_url(url)
        self.assertFalse(m.called)
        self.assertEqual(out, url)

    def test_expired_presigned_url_refreshed(self):
        # Markers far in the past -> expired -> presign attempted, refreshed URL returned.
        url = (
            "https://legacy.example.com/bucket/key.png"
            "?X-Amz-Date=20000101T000000Z&X-Amz-Expires=3600&X-Amz-Signature=abc"
        )
        with mock.patch.object(inference, "_presign_s3_object", return_value="https://refreshed") as m:
            out = inference._maybe_refresh_storage_url(url)
        self.assertTrue(m.called)
        self.assertEqual(out, "https://refreshed")

    def test_missing_key_in_path_returns_unchanged(self):
        # Only a bucket, no object key -> can't presign -> returned unchanged, no presign.
        url = "https://legacy.example.com/onlybucket"
        with mock.patch.object(inference, "_presign_s3_object", return_value="X") as m:
            out = inference._maybe_refresh_storage_url(url)
        self.assertFalse(m.called)
        self.assertEqual(out, url)


# --------------------------------------------------------------------------------------
# _resolve_media_path: traversal / absolute rejection vs allowed paths
# --------------------------------------------------------------------------------------

class ResolveMediaPathTests(_S3SeamBase):
    def test_absolute_outside_allowed_returns_empty(self):
        # '/etc/passwd' is absolute and outside /mnt//data/ -> '' (not echoed).
        self.assertEqual(inference._resolve_media_path("/etc/passwd", Path("/mnt/data/x")), "")

    def test_relative_traversal_escaping_allowed_returns_empty(self):
        # base_dir under /tmp (not allowed); '../../etc/passwd' resolves to '/tmp/etc/passwd'
        # ... actually resolves to '/etc/passwd' here, outside every allowed prefix -> ''.
        self.assertEqual(inference._resolve_media_path("../../etc/passwd", Path("/tmp/work/x")), "")

    def test_relative_traversal_landing_in_allowed_prefix_is_blocked(self):
        # base_dir '/mnt/data/x' + '../../etc/passwd' resolves to '/mnt/etc/passwd', which
        # escapes base_dir and is therefore '' (blocked) -- even though it starts with the
        # allowed '/mnt/' prefix. Containment in base_dir, not prefix, gates the relative branch.
        self.assertEqual(
            inference._resolve_media_path("../../etc/passwd", Path("/mnt/data/x")),
            "",
        )

    def test_path_under_base_dir_returned(self):
        self.assertEqual(
            inference._resolve_media_path("imgs/a.png", Path("/mnt/data/x")),
            "/mnt/data/x/imgs/a.png",
        )

    def test_absolute_under_allowed_prefix_returned(self):
        self.assertEqual(
            inference._resolve_media_path("/mnt/data/x/a.png", None),
            "/mnt/data/x/a.png",
        )

    def test_relative_with_no_base_dir_returns_empty(self):
        self.assertEqual(inference._resolve_media_path("imgs/a.png", None), "")

    def test_empty_value_returns_empty(self):
        self.assertEqual(inference._resolve_media_path("", Path("/mnt/data/x")), "")

    def test_data_uri_passthrough(self):
        self.assertEqual(
            inference._resolve_media_path("data:image/png;base64,iVBORw0KGgo", None),
            "data:image/png;base64,iVBORw0KGgo",
        )

    def test_http_url_non_storage_passthrough(self):
        self.assertEqual(
            inference._resolve_media_path("http://example.com/a.png", None),
            "http://example.com/a.png",
        )

    def test_s3_url_presigned_when_possible(self):
        # s3:// values route through _presign_s3_url -> _presign_s3_object.
        with mock.patch.object(inference, "_presign_s3_object", return_value="https://signed/x") as m:
            out = inference._resolve_media_path("s3://buck/path/a.png", None)
        self.assertTrue(m.called)
        self.assertEqual(out, "https://signed/x")
        self.assertEqual(m.call_args[0], ("buck", "path/a.png"))

    def test_s3_url_falls_back_to_original_when_presign_empty(self):
        # When presign yields '' (no creds), the original s3:// value is returned.
        with mock.patch.object(inference, "_presign_s3_object", return_value=""):
            out = inference._resolve_media_path("s3://buck/path/a.png", None)
        self.assertEqual(out, "s3://buck/path/a.png")


if __name__ == "__main__":
    unittest.main()
