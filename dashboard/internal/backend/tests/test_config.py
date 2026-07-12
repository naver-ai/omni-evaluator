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

"""Tests asserting the 8 new app.config constants exist with the right types."""

import unittest

import app.config as config


class NewConfigConstantsTest(unittest.TestCase):
    def test_dashboard_api_key_is_str(self):
        self.assertTrue(hasattr(config, "DASHBOARD_API_KEY"))
        self.assertIsInstance(config.DASHBOARD_API_KEY, str)

    def test_int_byte_and_count_constants(self):
        for name in (
            "UPLOAD_MAX_BYTES",
            "ZIP_MEMBER_MAX_BYTES",
            "ZIP_TOTAL_MAX_BYTES",
            "ZIP_MAX_RATIO",
            "S3_PRESIGN_EXPIRE",
            "MEDIA_INLINE_MAX_BYTES",
        ):
            with self.subTest(constant=name):
                self.assertTrue(hasattr(config, name), name + " missing")
                value = getattr(config, name)
                # bool is a subclass of int; explicitly reject it.
                self.assertIsInstance(value, int, name + " not int")
                self.assertNotIsInstance(value, bool, name + " is bool")

    def test_s3_has_creds_is_bool(self):
        self.assertTrue(hasattr(config, "S3_HAS_CREDS"))
        self.assertIsInstance(config.S3_HAS_CREDS, bool)


if __name__ == "__main__":
    unittest.main()
