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

"""Tests for app/auth.py require_api_key (called directly, no TestClient).

require_api_key reads the module-level name app.auth.DASHBOARD_API_KEY
(imported into app.auth at import time), so we monkeypatch THAT name, not
app.config.DASHBOARD_API_KEY.
"""

import unittest

import app.auth as auth_mod
from app.auth import require_api_key
from fastapi import HTTPException


class RequireApiKeyDisabledTest(unittest.TestCase):
    """When DASHBOARD_API_KEY is empty, auth is a no-op for any input."""

    def setUp(self):
        self._orig = auth_mod.DASHBOARD_API_KEY
        auth_mod.DASHBOARD_API_KEY = ""

    def tearDown(self):
        auth_mod.DASHBOARD_API_KEY = self._orig

    def test_empty_key_no_headers_returns_none(self):
        self.assertIsNone(require_api_key(x_api_key="", authorization=""))

    def test_empty_key_ignores_any_provided_token(self):
        # Even with garbage tokens, disabled auth returns None (no raise).
        self.assertIsNone(
            require_api_key(x_api_key="whatever", authorization="Bearer nope")
        )


class RequireApiKeyEnabledTest(unittest.TestCase):
    """When DASHBOARD_API_KEY is set, the token must match."""

    KEY = "s3cr3t-key"

    def setUp(self):
        self._orig = auth_mod.DASHBOARD_API_KEY
        auth_mod.DASHBOARD_API_KEY = self.KEY

    def tearDown(self):
        auth_mod.DASHBOARD_API_KEY = self._orig

    def _assert_401(self, **kwargs):
        with self.assertRaises(HTTPException) as ctx:
            require_api_key(**kwargs)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_empty_token_raises_401(self):
        self._assert_401(x_api_key="", authorization="")

    def test_wrong_x_api_key_raises_401(self):
        self._assert_401(x_api_key="wrong", authorization="")

    def test_wrong_bearer_raises_401(self):
        self._assert_401(x_api_key="", authorization="Bearer wrong")

    def test_matching_x_api_key_returns_none(self):
        self.assertIsNone(require_api_key(x_api_key=self.KEY, authorization=""))

    def test_matching_bearer_returns_none(self):
        self.assertIsNone(
            require_api_key(x_api_key="", authorization="Bearer " + self.KEY)
        )

    def test_bearer_scheme_is_case_insensitive(self):
        # auth.py lowercases the header before the "bearer " check.
        self.assertIsNone(
            require_api_key(x_api_key="", authorization="bearer " + self.KEY)
        )

    def test_x_api_key_takes_precedence_over_bearer(self):
        # token = x_api_key first; bearer is only consulted when x_api_key empty.
        self.assertIsNone(
            require_api_key(x_api_key=self.KEY, authorization="Bearer ignored")
        )
        # ...and a present-but-wrong x_api_key is not rescued by a valid bearer.
        self._assert_401(x_api_key="wrong", authorization="Bearer " + self.KEY)

    def test_bearer_without_scheme_prefix_raises_401(self):
        # A bare key in Authorization (no "Bearer ") is not extracted -> token
        # stays empty -> 401.
        self._assert_401(x_api_key="", authorization=self.KEY)


if __name__ == "__main__":
    unittest.main()
