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

"""Tests for the two small-box hardening guards added in final QA:

  1. submission._dir_total_bytes — the direct-upload disk-quota accountant.
  2. scan._read_member_if_small — the per-member read cap now also applied to the
     eval (score) member in scan_eval_zip, so a pathological huge member is skipped
     instead of spiking scan-peak RAM on the 1 GiB box.
"""

import io
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import app.services.scan as scan_mod
from app.api.submission import _dir_total_bytes
from app.services.scan import _read_member_if_small


class DirTotalBytesTest(unittest.TestCase):
    def test_sums_only_zip_files(self):
        with TemporaryDirectory() as d:
            base = Path(d)
            (base / "a.zip").write_bytes(b"x" * 100)
            (base / "b.zip").write_bytes(b"y" * 50)
            (base / "notes.txt").write_bytes(b"z" * 1000)  # ignored (not .zip)
            self.assertEqual(_dir_total_bytes(base), 150)

    def test_exclude_skips_one_path(self):
        with TemporaryDirectory() as d:
            base = Path(d)
            (base / "a.zip").write_bytes(b"x" * 100)
            dest = base / "b.zip"
            dest.write_bytes(b"y" * 50)
            self.assertEqual(_dir_total_bytes(base, exclude=dest), 100)

    def test_missing_dir_is_zero(self):
        with TemporaryDirectory() as d:
            self.assertEqual(_dir_total_bytes(Path(d) / "nope"), 0)


class ReadMemberCapTest(unittest.TestCase):
    def setUp(self):
        self._orig = scan_mod.INFERENCE_SAMPLE_MAX_BYTES

    def tearDown(self):
        scan_mod.INFERENCE_SAMPLE_MAX_BYTES = self._orig

    def _zip_with(self, payload: bytes) -> zipfile.ZipFile:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("m/ckpt/lmms_eval/evaluation_output/MMStar.json", payload)
        buf.seek(0)
        return zipfile.ZipFile(buf, "r")

    def test_under_cap_returns_bytes(self):
        scan_mod.INFERENCE_SAMPLE_MAX_BYTES = 1024
        zf = self._zip_with(b'{"ok": 1}')
        got = _read_member_if_small(zf, "m/ckpt/lmms_eval/evaluation_output/MMStar.json")
        self.assertEqual(got, b'{"ok": 1}')

    def test_over_cap_returns_none(self):
        scan_mod.INFERENCE_SAMPLE_MAX_BYTES = 4  # smaller than the member
        zf = self._zip_with(b'{"ok": 1}')
        got = _read_member_if_small(zf, "m/ckpt/lmms_eval/evaluation_output/MMStar.json")
        self.assertIsNone(got)

    def test_oversized_eval_member_skipped_not_crashed(self):
        # scan_eval_zip must skip an over-cap eval member gracefully (no benchmark, no raise).
        scan_mod.INFERENCE_SAMPLE_MAX_BYTES = 4
        zf = self._zip_with(b'{"results": {"MMStar": {"acc": 0.5}}}')
        benchmarks, rows, eng_map, mod_map = scan_mod.scan_eval_zip(zf, "m", "direct")
        self.assertEqual(benchmarks, set())


if __name__ == "__main__":
    unittest.main()
