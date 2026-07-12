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

"""Tests for the S3 source whitelist (OMNI_ENABLED_SOURCES) and the media path
rewrite (OMNI_LOCAL_MEDIA_PREFIX_MAP).

These guard two related behaviors added so S3 credentials can be present purely for
re-signing presigned media URLs from a legacy host WITHOUT the dashboard scanning S3 as a data
source (which hangs a 1 GB box):

  * a disabled source is never listed/scanned, at every entry point;
  * a configured path prefix is rewritten before the realpath + allowlist check, and
    that rewrite cannot widen access (traversal still 403s).
"""

import base64
import os
import unittest

import app.services.scan_cache as scan_cache
import app.api.inference as inference
import app.api.local_image as local_image
from app.config import _parse_prefix_map


class ParsePrefixMapTest(unittest.TestCase):
    def test_empty_and_garbage(self):
        self.assertEqual(_parse_prefix_map(""), ())
        self.assertEqual(_parse_prefix_map("   "), ())
        self.assertEqual(_parse_prefix_map("noequals"), ())
        self.assertEqual(_parse_prefix_map("=;=;a="), ())  # missing halves dropped

    def test_basic_pair_and_whitespace(self):
        self.assertEqual(
            _parse_prefix_map("  /mnt/a/ = /data/a/  "),
            (("/mnt/a/", "/data/a/"),),
        )

    def test_multiple_pairs(self):
        self.assertEqual(
            _parse_prefix_map("/mnt/a/=/data/a/;/mnt/b/=/data/b/"),
            (("/mnt/a/", "/data/a/"), ("/mnt/b/", "/data/b/")),
        )

    def test_trailing_slash_is_forced(self):
        # The pitfall the normalization closes: without the slash, "/mnt/ds" would also
        # match "/mnt/ds_evil/...". Both ends must be normalized to directory prefixes.
        self.assertEqual(
            _parse_prefix_map("/mnt/ds=/data/ds"),
            (("/mnt/ds/", "/data/ds/"),),
        )

    def test_value_may_contain_equals(self):
        # split("=", 1) keeps any '=' in the path value intact.
        self.assertEqual(
            _parse_prefix_map("/mnt/a/=/data/a=b/"),
            (("/mnt/a/", "/data/a=b/"),),
        )


class EnabledSourcesGatingTest(unittest.TestCase):
    def setUp(self):
        self._orig = scan_cache.ENABLED_SOURCES
        scan_cache.ENABLED_SOURCES = ("internal", "direct")

    def tearDown(self):
        scan_cache.ENABLED_SOURCES = self._orig

    def test_get_models_disabled_source_returns_empty_without_listing(self):
        called = {"hit": False}

        def _boom():
            called["hit"] = True
            raise AssertionError("S3 must not be listed when disabled")

        orig = scan_cache._scan_s3_models
        scan_cache._scan_s3_models = _boom
        try:
            self.assertEqual(scan_cache.get_models("s3"), [])
            self.assertFalse(called["hit"])
        finally:
            scan_cache._scan_s3_models = orig

    def test_scan_all_sources_default_excludes_disabled(self):
        called = {"hit": False}

        def _boom():
            called["hit"] = True
            raise AssertionError("S3 must not be listed when disabled")

        orig = scan_cache._scan_s3_models
        scan_cache._scan_s3_models = _boom
        try:
            # Default sources (None) must collapse to ENABLED_SOURCES; even an explicit
            # request for s3 is intersected away.
            scan_cache.scan_all_sources(quick=True)
            scan_cache.scan_all_sources({"s3"}, quick=False)
            self.assertFalse(called["hit"])
        finally:
            scan_cache._scan_s3_models = orig


class InferenceSourceGateTest(unittest.TestCase):
    """model_ids reach /benchmarks and /sample as a client-supplied query param, so the
    per-model resolvers must gate disabled sources themselves."""

    def setUp(self):
        self._orig = inference.ENABLED_SOURCES
        inference.ENABLED_SOURCES = ("internal", "direct")

    def tearDown(self):
        inference.ENABLED_SOURCES = self._orig

    def test_benchmarks_resolver_skips_disabled_source(self):
        def _boom(model):
            raise AssertionError("S3 benchmark listing must not run when disabled")

        orig = inference._benchmarks_from_s3_model
        inference._benchmarks_from_s3_model = _boom
        try:
            self.assertEqual(
                inference._get_benchmarks_for_model_with_engine("s3", "anymodel"),
                ([], {}),
            )
        finally:
            inference._benchmarks_from_s3_model = orig

    def test_sample_resolver_skips_disabled_source(self):
        def _boom(model, base_bench, idx):
            raise AssertionError("S3 sample listing must not run when disabled")

        orig = inference._get_one_sample_s3
        inference._get_one_sample_s3 = _boom
        try:
            self.assertEqual(
                inference._get_one_sample("s3", "anymodel", "MMStar", 0),
                (None, 0, None),
            )
        finally:
            inference._get_one_sample_s3 = orig


def _b64url(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode()).decode()


class _FakeQueryParams:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, key, default=None):
        return self._m.get(key, default)


class _FakeRequest:
    def __init__(self, **params):
        self.query_params = _FakeQueryParams(params)


class MediaPrefixRewriteTest(unittest.TestCase):
    OLD = "/mnt/source_images/"
    NEW = "/data/datasets/images/"

    def setUp(self):
        self._orig = local_image.LOCAL_MEDIA_PREFIX_MAP
        local_image.LOCAL_MEDIA_PREFIX_MAP = ((self.OLD, self.NEW),)

    def tearDown(self):
        local_image.LOCAL_MEDIA_PREFIX_MAP = self._orig

    def test_mapped_path_is_rewritten(self):
        req = _FakeRequest(path=_b64url(self.OLD + "MMStar/0.jpg"))
        real_path, mime, err = local_image._decode_local_path(req)
        self.assertIsNone(err)
        self.assertEqual(real_path, "/data/datasets/images/MMStar/0.jpg")
        self.assertEqual(mime, "image/jpeg")

    def test_rewrite_cannot_escape_allowlist(self):
        # Traversal after rewrite must still resolve outside the allowlist and 403.
        req = _FakeRequest(path=_b64url(self.OLD + "../../../../etc/passwd"))
        real_path, mime, err = local_image._decode_local_path(req)
        self.assertIsNotNone(err)
        self.assertEqual(err.status_code, 403)

    def test_unmapped_path_passes_through_unchanged(self):
        target = "/data/outputs/x.png"
        req = _FakeRequest(path=_b64url(target))
        real_path, mime, err = local_image._decode_local_path(req)
        self.assertIsNone(err)
        self.assertEqual(real_path, os.path.realpath(target))


class MediaS3PrefixRewriteTest(unittest.TestCase):
    """A local media path matching MEDIA_S3_PREFIX_MAP is rewritten to an s3:// URL and then
    presigned. With no creds in the test env, _presign_s3_url returns "" so _resolve_media_path
    falls back to the rewritten s3:// string — which is what we assert."""

    OLD = "/mnt/source_images/"
    NEW = "s3://your-bucket/media/images/"

    def setUp(self):
        self._orig_map = inference.MEDIA_S3_PREFIX_MAP
        self._orig_client = inference._get_s3_client
        inference.MEDIA_S3_PREFIX_MAP = ((self.OLD, self.NEW),)
        inference._get_s3_client = lambda: None  # force presign to no-op

    def tearDown(self):
        inference.MEDIA_S3_PREFIX_MAP = self._orig_map
        inference._get_s3_client = self._orig_client

    def test_local_path_rewritten_to_s3_url(self):
        out = inference._resolve_media_path(self.OLD + "MMStar/0.jpg", None)
        self.assertEqual(out, self.NEW + "MMStar/0.jpg")

    def test_unmapped_local_path_not_rewritten(self):
        # An absolute path outside the map is gated by the local-image allowlist instead; an
        # allowed prefix resolves to itself, not an s3:// URL.
        out = inference._resolve_media_path("/data/outputs/x.png", None)
        self.assertFalse(out.startswith("s3://"))


class MediaRebucketTest(unittest.TestCase):
    """With MEDIA_LEGACY_HOST + MEDIA_REBUCKET set, a legacy presigned URL is re-presigned
    against the configured bucket on the active S3 client, using the SAME object key."""

    URL = ("https://legacy.example.com/legacy-bucket/media/datasets/ai2d/test/"
           "resources/images/345802.jpg?X-Amz-Date=20260206T170222Z&X-Amz-Expires=604800")

    def setUp(self):
        self._orig_host = inference.MEDIA_LEGACY_HOST
        self._orig_rebucket = inference.MEDIA_REBUCKET
        self._orig_presign = inference._presign_s3_object
        self.calls = []
        inference.MEDIA_LEGACY_HOST = "legacy.example.com"
        inference.MEDIA_REBUCKET = "new-bucket"
        inference._presign_s3_object = lambda b, k: self.calls.append((b, k)) or f"https://new/{b}/{k}?sig"

    def tearDown(self):
        inference.MEDIA_LEGACY_HOST = self._orig_host
        inference.MEDIA_REBUCKET = self._orig_rebucket
        inference._presign_s3_object = self._orig_presign

    def test_legacy_url_repointed_to_rebucket_same_key(self):
        out = inference._maybe_refresh_storage_url(self.URL)
        # Key is everything after the embedded bucket segment (legacy-bucket/).
        self.assertEqual(self.calls, [("new-bucket",
                                       "media/datasets/ai2d/test/resources/images/345802.jpg")])
        self.assertTrue(out.startswith("https://new/new-bucket/"))

    def test_non_legacy_url_untouched(self):
        out = inference._maybe_refresh_storage_url("https://example.com/x.jpg")
        self.assertEqual(out, "https://example.com/x.jpg")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
