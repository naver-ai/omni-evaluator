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

"""Characterization + correctness tests for app/services/sample_reader.py.

Covers the BytesIO-rewind streaming readers (count_records_stream /
stream_record_at_stream) across every _RECORD_PATHS shape, plus the LRU bound of the
module-level _count_cache. All fixtures are small inline JSON wrapped in io.BytesIO;
no files, network, or S3.

NOTE (genuine bug, see module summary): with the ijson yajl2_c backend installed in
this environment, the streaming readers wrap ``ijson.items(...)`` in
``contextlib.closing(...)``, but the C-backend iterator has no ``.close()`` method.
The resulting AttributeError is swallowed by the broad ``except Exception``, so every
prefix attempt aborts: count_records_stream always returns 0 and
stream_record_at_stream always returns (None, False), regardless of the data. The tests
below assert the CORRECT intended behavior, so they FAIL until the bug is fixed
(removing/relaxing the contextlib.closing wrapper). This is intentional per the task
rules: keep the failing test and document the bug rather than force it green.
"""

import io
import json
import unittest

from app.services import sample_reader
from app.services.sample_reader import (
    _RECORD_PATHS,
    count_records_stream,
    stream_record_at_stream,
)


def _bytes(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


# Each fixture pairs a JSON shape with the _RECORD_PATHS entry it should match.
# Crucially, several shapes only match a *later* candidate path, so a successful read
# proves the buffer is rewound (seek(0)) before each prefix attempt.
RECORDS = [{"id": 0, "v": "a"}, {"id": 1, "v": "b"}, {"id": 2, "v": "c"}]

# inference list-of-lists -> matched by the FIRST path "inference.item.item"
SHAPE_INFERENCE_NESTED = {"inference": [RECORDS]}
# inference flat list -> "inference.item.item" yields nothing, matched by "inference.item"
SHAPE_INFERENCE_FLAT = {"inference": RECORDS}
# output -> first two inference.* candidates yield nothing, matched by "output.item"
SHAPE_OUTPUT = {"output": RECORDS}
# outputs -> only the 4th/last candidate "outputs.item" matches (strongest rewind proof)
SHAPE_OUTPUTS = {"outputs": RECORDS}


class RecordPathsConstantTests(unittest.TestCase):
    def test_record_paths_constant(self):
        # Guard the assumption the rewind tests rely on: ordering + membership.
        self.assertEqual(
            _RECORD_PATHS,
            ("inference.item.item", "inference.item", "output.item", "outputs.item"),
        )


class CountRecordsStreamTests(unittest.TestCase):
    """Asserts intended behavior. FAILS while the contextlib.closing bug stands."""

    def setUp(self):
        with sample_reader._count_lock:
            sample_reader._count_cache.clear()

    def test_count_inference_nested(self):
        self.assertEqual(count_records_stream(_bytes(SHAPE_INFERENCE_NESTED)), 3)

    def test_count_inference_flat(self):
        # inference.item.item yields 0 here; rewind lets inference.item find the 3 records.
        self.assertEqual(count_records_stream(_bytes(SHAPE_INFERENCE_FLAT)), 3)

    def test_count_output(self):
        self.assertEqual(count_records_stream(_bytes(SHAPE_OUTPUT)), 3)

    def test_count_outputs_matches_last_candidate(self):
        # Three earlier prefixes yield nothing; only after three rewinds does outputs.item hit.
        self.assertEqual(count_records_stream(_bytes(SHAPE_OUTPUTS)), 3)

    def test_count_accepts_filelike(self):
        self.assertEqual(count_records_stream(io.BytesIO(_bytes(SHAPE_OUTPUTS))), 3)

    def test_count_no_records_returns_zero(self):
        # No matching array -> 0. (Passes even with the bug, but pins the contract.)
        self.assertEqual(count_records_stream(_bytes({"something_else": [1, 2]})), 0)


class StreamRecordAtStreamTests(unittest.TestCase):
    """Asserts intended behavior. FAILS while the contextlib.closing bug stands."""

    def setUp(self):
        with sample_reader._count_lock:
            sample_reader._count_cache.clear()

    def _check_all_indices(self, shape):
        raw = _bytes(shape)
        for i, expected in enumerate(RECORDS):
            rec, matched = stream_record_at_stream(raw, i)
            self.assertTrue(matched, f"array not matched for idx {i} in {shape!r}")
            self.assertEqual(rec, expected, f"wrong record at idx {i}")

    def test_inference_nested_all_indices(self):
        self._check_all_indices(SHAPE_INFERENCE_NESTED)

    def test_inference_flat_all_indices(self):
        self._check_all_indices(SHAPE_INFERENCE_FLAT)

    def test_output_all_indices(self):
        self._check_all_indices(SHAPE_OUTPUT)

    def test_outputs_all_indices_proves_rewind(self):
        # outputs.item is the last candidate; returning the right record at each index
        # only works if the BytesIO is rewound before that final attempt.
        self._check_all_indices(SHAPE_OUTPUTS)

    def test_specific_middle_index(self):
        rec, matched = stream_record_at_stream(_bytes(SHAPE_OUTPUTS), 1)
        self.assertTrue(matched)
        self.assertEqual(rec, {"id": 1, "v": "b"})

    def test_index_out_of_range_matched_true(self):
        # Array found but idx beyond its length -> (None, True).
        rec, matched = stream_record_at_stream(_bytes(SHAPE_OUTPUT), 99)
        self.assertIsNone(rec)
        self.assertTrue(matched)

    def test_no_matching_array_matched_false(self):
        # No recognised array -> (None, False). (Passes even with the bug.)
        rec, matched = stream_record_at_stream(_bytes({"nope": [1, 2, 3]}), 0)
        self.assertIsNone(rec)
        self.assertFalse(matched)

    def test_filelike_input(self):
        rec, matched = stream_record_at_stream(io.BytesIO(_bytes(SHAPE_INFERENCE_NESTED)), 2)
        self.assertTrue(matched)
        self.assertEqual(rec, {"id": 2, "v": "c"})

    def test_non_dict_record_returns_none_but_matched(self):
        # A scalar at the requested index isn't a dict -> record None, but array matched.
        rec, matched = stream_record_at_stream(_bytes({"output": ["scalar", "x"]}), 0)
        self.assertIsNone(rec)
        self.assertTrue(matched)


class CountCacheLRUTests(unittest.TestCase):
    """Independent of ijson — these pin the bounded-LRU contract and pass."""

    def setUp(self):
        with sample_reader._count_lock:
            sample_reader._count_cache.clear()

    def tearDown(self):
        with sample_reader._count_lock:
            sample_reader._count_cache.clear()

    def _evict_to_bound(self):
        # Mirror the eviction loop count_records runs under its lock.
        with sample_reader._count_lock:
            while len(sample_reader._count_cache) > sample_reader._COUNT_CACHE_MAX:
                sample_reader._count_cache.popitem(last=False)

    def test_cache_capped_and_lru_evicted(self):
        maxsize = sample_reader._COUNT_CACHE_MAX
        cache = sample_reader._count_cache
        # Insert more than maxsize entries, evicting after each insert as count_records does.
        total = maxsize + 10
        for i in range(total):
            with sample_reader._count_lock:
                cache[f"/path/{i}"] = (float(i), i)
                cache.move_to_end(f"/path/{i}")
            self._evict_to_bound()

        # Size is capped at the bound.
        self.assertEqual(len(cache), maxsize)
        # The oldest (least-recently inserted) keys were evicted.
        for i in range(10):
            self.assertNotIn(f"/path/{i}", cache)
        # The most-recent keys survive.
        for i in range(total - maxsize, total):
            self.assertIn(f"/path/{i}", cache)

    def test_move_to_end_protects_from_eviction(self):
        maxsize = sample_reader._COUNT_CACHE_MAX
        cache = sample_reader._count_cache
        with sample_reader._count_lock:
            for i in range(maxsize):
                cache[f"/path/{i}"] = (float(i), i)
            # Touch the oldest key so it becomes most-recently used.
            cache.move_to_end("/path/0")
        # Add one more, forcing a single eviction.
        with sample_reader._count_lock:
            cache["/path/new"] = (999.0, 999)
            cache.move_to_end("/path/new")
        self._evict_to_bound()

        self.assertEqual(len(cache), maxsize)
        # /path/0 was protected; /path/1 (now the true LRU) was evicted instead.
        self.assertIn("/path/0", cache)
        self.assertNotIn("/path/1", cache)


class FactoryModeTests(unittest.TestCase):
    """The zip/S3 read path passes a re-openable FACTORY (callable -> fresh file-like) so the
    member is streamed, never buffered whole. Verify factory mode matches bytes mode, re-opens
    a fresh stream per prefix attempt, and closes each stream it opens (no FD leak)."""

    @staticmethod
    def _factory(data: bytes):
        calls = []

        def make():
            calls.append(1)
            return io.BytesIO(data)

        make.calls = calls
        return make

    def test_factory_stream_record_matches_bytes(self):
        # SHAPE_OUTPUTS matches only the LAST prefix -> several re-opens before the hit.
        f = self._factory(_bytes(SHAPE_OUTPUTS))
        rec, matched = stream_record_at_stream(f, 1)
        self.assertTrue(matched)
        self.assertEqual(rec, RECORDS[1])

    def test_factory_count_matches_bytes(self):
        f = self._factory(_bytes(SHAPE_INFERENCE_FLAT))
        self.assertEqual(count_records_stream(f), len(RECORDS))

    def test_factory_reopened_fresh_per_prefix(self):
        # SHAPE_OUTPUTS only matches the 4th prefix, so the factory MUST be re-invoked for
        # each earlier attempt — proving no single buffer is silently reused across prefixes.
        f = self._factory(_bytes(SHAPE_OUTPUTS))
        stream_record_at_stream(f, 0)
        self.assertGreaterEqual(len(f.calls), 4)

    def test_factory_streams_are_closed(self):
        closed = []

        def make():
            b = io.BytesIO(_bytes(SHAPE_OUTPUT))
            orig = b.close

            def c():
                closed.append(1)
                orig()

            b.close = c
            return b

        stream_record_at_stream(make, 0)
        self.assertTrue(closed, "each factory-opened stream should be closed (owns=True)")

    def test_factory_out_of_range_matched(self):
        f = self._factory(_bytes(SHAPE_INFERENCE_NESTED))
        rec, matched = stream_record_at_stream(f, 99)
        self.assertIsNone(rec)
        self.assertTrue(matched)  # correct array found, index just out of range


class NaNToleranceTest(unittest.TestCase):
    """Eval JSONs are written with Python's json.dump, which emits bare `NaN` literals (e.g.
    vlm_eval_kit pads `option_contents` with NaN). Those are invalid JSON, so the strict ijson
    stream ABORTS at the first one and every record after it goes invisible. The readers must
    fall back to a tolerant full parse and recover the whole array. (Live bug: one early NaN
    blanked entire MMStar/MMMU_DEV_VAL files in the Inference Viewer.)"""

    def _nan_doc_bytes(self) -> bytes:
        recs = [
            {"index": 0, "prediction": "a"},
            # record 1 carries the bare NaN — record 2 must still be reachable past it
            {"index": 1, "prediction": "b", "option_contents": ["x", float("nan"), float("nan")]},
            {"index": 2, "prediction": "c"},
        ]
        raw = json.dumps({"inference": [recs]}).encode("utf-8")  # default allow_nan -> bare NaN
        assert b"NaN" in raw  # guard: the fixture really exercises the NaN path
        return raw

    def test_stream_recovers_record_after_nan(self):
        rec, matched = stream_record_at_stream(self._nan_doc_bytes(), 2)
        self.assertTrue(matched)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.get("prediction"), "c")

    def test_stream_count_recovers_full_total(self):
        self.assertEqual(count_records_stream(self._nan_doc_bytes()), 3)

    def test_local_file_recovers_after_nan(self):
        import os
        import tempfile
        from pathlib import Path

        from app.services.sample_reader import count_records, stream_record_at

        with tempfile.NamedTemporaryFile("wb", suffix=".json", delete=False) as fh:
            fh.write(self._nan_doc_bytes())
            p = Path(fh.name)
        try:
            rec = stream_record_at(p, 2)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.get("prediction"), "c")
            self.assertEqual(count_records(p), 3)
        finally:
            os.unlink(p)


if __name__ == "__main__":
    unittest.main()
