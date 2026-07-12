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

"""Deterministic unit tests for clients/s3_client.py L1."""
from __future__ import annotations

import json
import os
import pickle
import time
import uuid
import warnings
from pathlib import PosixPath
from unittest.mock import MagicMock

import pytest
import yaml

from omni_evaluator.clients import s3_client as s3_module
from omni_evaluator.clients.s3_client import (
    S3Client,
    _download_file_boto,
    _init_boto_client,
    _upload_file_boto,
)
from tests.clients.conftest import ByteStoreFs


# ============================================================================
# _accumulate_result — None passthrough + aggregation (passed/failed/skipped/bytes)
# ============================================================================


def _zero_output() -> dict:
    return {
        "num_passed": 0,
        "num_failed": 0,
        "num_skipped": 0,
        "total_bytes": 0.0,
        "error_message": None,
    }


def test_accumulate_result_none_is_noop():
    """result=None does not touch any counter."""
    output = _zero_output()
    S3Client._accumulate_result(output, None)
    assert output == _zero_output()


def test_accumulate_result_aggregates():
    """status / skipped / error_message / file_size are each aggregated into the correct accumulator."""
    output = _zero_output()
    S3Client._accumulate_result(
        output, {"status": True, "skipped": False, "file_size": 10}
    )
    S3Client._accumulate_result(
        output, {"status": True, "skipped": True, "file_size": 7}
    )
    S3Client._accumulate_result(
        output,
        {"status": False, "skipped": False, "file_size": 3, "error_message": "boom"},
    )
    assert output["num_passed"] == 2
    assert output["num_failed"] == 1
    assert output["num_skipped"] == 1
    assert output["total_bytes"] == 20
    assert output["error_message"] == ["boom"]


# ============================================================================
# list — trailing slash normalization + CommonPrefixes/Contents merge + prefix strip
# ============================================================================


def test_list_normalizes_and_merges(make_s3_client):
    """Trailing slash is added automatically; subdirectory and file keys are merged with the prefix stripped."""
    boto = MagicMock()
    boto.get_paginator.return_value.paginate.return_value = [
        {
            "CommonPrefixes": [{"Prefix": "root/data/2025-09-14/"}],
            "Contents": [{"Key": "root/info.json"}, {"Key": "root/x/"}],
        }
    ]
    client = make_s3_client(boto_client=boto)
    assert client.list("root") == ["data/2025-09-14", "info.json", "x"]


# ============================================================================
# upload_obj / download_obj — round-trip branching by extension
# ============================================================================


@pytest.mark.parametrize(
    "remote_filename, payload",
    [
        ("a.txt", "héllo"),
        ("a.pickle", {"k": [1, 2]}),
        ("a.json", {"한글": 1}),
        ("a.jsonl", [{"a": 1}, {"b": 2}]),
        ("a.yaml", {"k": "한글"}),
        ("a.bin", b"\x00\x01raw"),
    ],
)
def test_upload_download_roundtrip(make_s3_client, remote_filename, payload):
    """Serialization/deserialization is symmetric per extension (.txt/.pickle/.json/.jsonl/.yaml/raw)."""
    fs = ByteStoreFs()
    client = make_s3_client(pyarrow_client=fs)
    client.upload_obj(payload, remote_filename)
    assert client.download_obj(remote_filename) == payload


def test_upload_obj_jsonl_rejects_non_list(make_s3_client):
    """.jsonl raises TypeError for obj that is not a list/tuple."""
    client = make_s3_client(pyarrow_client=ByteStoreFs())
    with pytest.raises(TypeError):
        client.upload_obj({"not": "a list"}, "a.jsonl")


def test_upload_obj_format_details(make_s3_client):
    """Non-obvious serialization details: json indent=2/ensure_ascii=False, yaml allow_unicode."""
    fs = ByteStoreFs()
    client = make_s3_client(pyarrow_client=fs)
    client.upload_obj({"한글": 1}, "a.json")
    client.upload_obj({"k": "한글"}, "a.yaml")
    json_raw = fs.store["bucket/a.json"].decode("utf-8")
    yaml_raw = fs.store["bucket/a.yaml"].decode("utf-8")
    assert "한글" in json_raw  # ensure_ascii=False
    assert "한글" in yaml_raw  # allow_unicode=True


def test_upload_obj_posix_path(make_s3_client):
    """PosixPath remote_filepath is normalized to a string and becomes the store key."""
    fs = ByteStoreFs()
    client = make_s3_client(pyarrow_client=fs)
    client.upload_obj("x", PosixPath("dir") / "p.txt")
    assert "bucket/dir/p.txt" in fs.store


# ============================================================================
# upload_file / download_file — local file <-> remote stream
# ============================================================================


def test_upload_file_reads_and_joins_remote_path(make_s3_client, tmp_path):
    """Reads a local file and uploads it to remote_dirpath/<basename>."""
    fs = ByteStoreFs()
    client = make_s3_client(pyarrow_client=fs)
    local = tmp_path / "hello.json"
    local.write_text(json.dumps({"x": 1}), encoding="utf-8")
    client.upload_file(str(local), remote_dirpath="remote/sub")
    raw = fs.store["bucket/remote/sub/hello.json"].decode("utf-8")
    assert json.loads(raw) == {"x": 1}


def test_download_file_creates_parent_and_copies(make_s3_client, tmp_path):
    """Creates parent directories if they do not exist, then copies the remote stream to local."""
    fs = ByteStoreFs()
    fs.store["bucket/x/y.bin"] = b"payload"
    client = make_s3_client(pyarrow_client=fs)
    dst = tmp_path / "nested" / "deep" / "y.bin"
    client.download_file(str(dst), remote_filepath="x/y.bin")
    assert dst.read_bytes() == b"payload"


# ============================================================================
# upload_dir — file enumeration / hidden skip / non-directory rejection
# ============================================================================


def test_upload_dir_single_process(make_s3_client, tmp_path):
    """Files are uploaded under remote_dirpath via the single-process path, and results are aggregated."""
    (tmp_path / "a.txt").write_text("hi", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("bye", encoding="utf-8")

    boto = MagicMock()
    client = make_s3_client(boto_client=boto)
    # resume=False: skips head_object comparison. Because MagicMock.__int__ returns 1,
    # with resume=True a 1-byte file would incorrectly match the head size and the upload would be skipped.
    out = client.upload_dir(
        str(tmp_path), remote_dirpath="remote", num_process=1, resume=False
    )

    assert out["num_files"] == 2
    assert out["num_passed"] == 2
    assert out["success_rate"] == 1.0
    keys = sorted(c.kwargs["Key"] for c in boto.upload_file.call_args_list)
    assert keys == ["remote/a.txt", "remote/sub/b.txt"]


def test_upload_dir_skips_hidden(make_s3_client, tmp_path):
    """With include_hidden=False, dotfiles and everything under dot-dirs are excluded."""
    (tmp_path / "visible.txt").write_text("v", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("h", encoding="utf-8")
    (tmp_path / ".dotdir").mkdir()
    (tmp_path / ".dotdir" / "inner.txt").write_text("i", encoding="utf-8")

    boto = MagicMock()
    client = make_s3_client(boto_client=boto)
    out = client.upload_dir(
        str(tmp_path), remote_dirpath="r", num_process=1, resume=False
    )

    assert out["num_files"] == 1
    keys = [c.kwargs["Key"] for c in boto.upload_file.call_args_list]
    assert keys == ["r/visible.txt"]


def test_upload_dir_rejects_non_directory(make_s3_client, tmp_path):
    """Raises ValueError if dirpath is not a directory."""
    f = tmp_path / "not_a_dir.txt"
    f.write_text("x", encoding="utf-8")
    client = make_s3_client()
    with pytest.raises(ValueError):
        client.upload_dir(str(f), remote_dirpath="r", num_process=1)


# ============================================================================
# download_dir — paginator listing + directory marker / prefix mismatch filter
# ============================================================================


def test_download_dir_filters_and_downloads(make_s3_client, tmp_path):
    """Directory markers and prefix mismatches among paginator keys are excluded; the rest are downloaded via single process."""
    boto = MagicMock()
    boto.get_paginator.return_value.paginate.return_value = [
        {
            "Contents": [
                {"Key": "remote/a.txt", "Size": 1},
                {"Key": "remote/sub/", "Size": 0},  # directory marker -> excluded
                {"Key": "other/skip.txt", "Size": 9},  # prefix mismatch -> excluded
                {"Key": "remote/sub/b.txt", "Size": 1},
            ]
        }
    ]

    # Write dummy files via side_effect so that downloaded files actually appear on disk
    def _write_dummy(Bucket, Key, Filename):
        from pathlib import Path

        Path(Filename).parent.mkdir(parents=True, exist_ok=True)
        Path(Filename).write_bytes(b"x")

    boto.download_file.side_effect = _write_dummy

    client = make_s3_client(boto_client=boto)
    out = client.download_dir(
        str(tmp_path / "local"), remote_dirpath="remote", num_process=1
    )

    assert out["num_files"] == 2
    assert out["num_passed"] == 2
    keys = sorted(c.kwargs["Key"] for c in boto.download_file.call_args_list)
    assert keys == ["remote/a.txt", "remote/sub/b.txt"]


# ============================================================================
# delete_file / delete_dir — single / batch deletion delegation
# ============================================================================


def test_delete_file_joins_bucket(make_s3_client):
    """bucket is prepended to remote_filepath and delegated to pyarrow.delete_file."""
    pa = MagicMock()
    client = make_s3_client(pyarrow_client=pa)
    client.delete_file("foo/bar.txt")
    pa.delete_file.assert_called_once_with("bucket/foo/bar.txt")


def test_delete_dir_iterates_paginator_keys(make_s3_client):
    """Directory markers among paginator keys are skipped; delete_object is called for all others."""
    boto = MagicMock()
    boto.get_paginator.return_value.paginate.return_value = [
        {
            "Contents": [
                {"Key": "d/a.txt"},
                {"Key": "d/sub/"},  # directory marker -> skip
                {"Key": "d/sub/b.txt"},
            ]
        }
    ]
    client = make_s3_client(boto_client=boto)
    client.delete_dir("d")
    keys = sorted(c.kwargs["Key"] for c in boto.delete_object.call_args_list)
    assert keys == ["d/a.txt", "d/sub/b.txt"]


# ============================================================================
# get_presigned_url — argument passthrough + return value relay
# ============================================================================


def test_get_presigned_url_passes_params(make_s3_client):
    """Bucket/Key/Disposition/ExpiresIn are passed through as-is and the URL is relayed."""
    boto = MagicMock()
    boto.generate_presigned_url.return_value = "https://fake/url"
    client = make_s3_client(boto_client=boto)

    url = client.get_presigned_url("foo/bar.bin", duration=120)
    assert url == "https://fake/url"
    call = boto.generate_presigned_url.call_args
    assert call.args == ("get_object",)
    assert call.kwargs["Params"]["Bucket"] == "bucket"
    assert call.kwargs["Params"]["Key"] == "foo/bar.bin"
    assert "foo/bar.bin" in call.kwargs["Params"]["ResponseContentDisposition"]
    assert call.kwargs["ExpiresIn"] == 120


# ============================================================================
# Module helpers — _init_boto_client / _upload_file_boto / _download_file_boto
# ============================================================================


def test_init_boto_client_global_assignment():
    """Module GLOBAL_BOTO_CLIENT is updated only when update_global_worker=True."""
    client, config = _init_boto_client(
        access_key="ak",
        secret_key="sk",
        endpoint_url=None,
        region="kr-standard",
        update_global_worker=False,
    )
    assert s3_module.GLOBAL_BOTO_CLIENT is None  # no change (autouse resets to None)
    assert client is not None and config is not None

    client2, _ = _init_boto_client(
        access_key="ak",
        secret_key="sk",
        endpoint_url=None,
        region="kr-standard",
        update_global_worker=True,
    )
    assert s3_module.GLOBAL_BOTO_CLIENT is client2


def test_upload_file_boto_missing_local(tmp_path):
    """If the local file does not exist, immediately returns status=False + 'local file not found'."""
    out = _upload_file_boto(
        bucket_name="bucket",
        key="k",
        filepath=str(tmp_path / "nope.txt"),
        client=MagicMock(),
    )
    assert out["status"] is False
    assert out["skipped"] is False
    assert out["error_message"] == "local file not found"


def test_upload_file_boto_skipped_on_resume_match(tmp_path):
    """If resume is set and the remote head size matches the local size, skipped=True is returned without uploading."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"abcde")
    boto = MagicMock()
    boto.head_object.return_value = {"ContentLength": 5}

    out = _upload_file_boto(
        bucket_name="bucket", key="k", filepath=str(f), resume=True, client=boto
    )
    assert out["skipped"] is True
    boto.upload_file.assert_not_called()


def test_upload_file_boto_uploads_on_size_mismatch(tmp_path):
    """Even with resume, upload is called if the head size differs from the local size."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"abcde")
    boto = MagicMock()
    boto.head_object.return_value = {"ContentLength": 99}

    out = _upload_file_boto(
        bucket_name="bucket", key="k", filepath=str(f), resume=True, client=boto
    )
    assert out["status"] is True
    assert out["skipped"] is False
    boto.upload_file.assert_called_once()


def test_upload_file_boto_captures_upload_error(tmp_path):
    """If the upload call raises an exception, status=False + error_message is captured."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    boto = MagicMock()
    boto.upload_file.side_effect = RuntimeError("S3 boom")

    out = _upload_file_boto(
        bucket_name="bucket", key="k", filepath=str(f), resume=False, client=boto
    )
    assert out["status"] is False
    assert "S3 boom" in out["error_message"]


def test_download_file_boto_skipped_on_resume_match(tmp_path):
    """If resume is set and a local file of the same size already exists, skipped=True is returned without downloading."""
    f = tmp_path / "x.bin"
    f.write_bytes(b"abcde")
    boto = MagicMock()

    out = _download_file_boto(
        bucket_name="bucket",
        key="k",
        filepath=str(f),
        file_size=5,
        resume=True,
        client=boto,
    )
    assert out["skipped"] is True
    boto.download_file.assert_not_called()


def test_download_file_boto_downloads_when_missing(tmp_path):
    """Calls download_file if the local file does not exist."""
    boto = MagicMock()
    out = _download_file_boto(
        bucket_name="bucket",
        key="k",
        filepath=str(tmp_path / "deep" / "x.bin"),
        file_size=5,
        resume=True,
        client=boto,
    )
    assert out["status"] is True
    assert out["skipped"] is False
    boto.download_file.assert_called_once()


def test_download_file_boto_captures_error(tmp_path):
    """If the download call raises an exception, status=False + error_message is captured."""
    boto = MagicMock()
    boto.download_file.side_effect = RuntimeError("network down")

    out = _download_file_boto(
        bucket_name="bucket",
        key="k",
        filepath=str(tmp_path / "x.bin"),
        file_size=5,
        resume=False,
        client=boto,
    )
    assert out["status"] is False
    assert "network down" in out["error_message"]


# ============================================================================
# L3 live smoke — actual NAVER Cloud S3 round-trip
# Disabled by default in CI via double gate: slow + requires_env.
# ============================================================================


@pytest.mark.slow
@pytest.mark.requires_env(
    "S3_BUCKET_NAME", "S3_ACCESS_KEY", "S3_SECRET_KEY", "S3_ENDPOINT_URL"
)
def test_live_roundtrip():
    """Validates end-to-end: upload -> list check -> download round-trip -> cleanup."""
    live = S3Client(
        bucket_name=os.environ["S3_BUCKET_NAME"],
        access_key=os.environ["S3_ACCESS_KEY"],
        secret_key=os.environ["S3_SECRET_KEY"],
        endpoint_url=os.environ["S3_ENDPOINT_URL"],
        verbose=False,
    )

    prefix = "tests/clients/_smoke"
    payload = {"hello": "world", "한글": [1, 2, 3]}
    max_attempts = 5
    list_poll = 20  # number of list polls per attempt (x0.5s)

    def _cleanup(key: str) -> None:
        # Cleanup failures are not failed but surfaced as warnings — prevents lingering objects in shared bucket.
        try:
            live.delete_file(key)
        except Exception as ex:
            warnings.warn(
                f"smoke cleanup failed for key={key}: {ex}. "
                f"Please manually clean up any remaining objects under `{prefix}/`.",
                stacklevel=1,
            )

    def _attempt() -> None:
        """Runs a single end-to-end cycle; raises an exception if list is not reflected or upload fails."""
        key = f"{prefix}/{uuid.uuid4().hex}.json"
        try:
            live.upload_obj(payload, key)

            basename = os.path.basename(key)
            for _ in range(list_poll):
                if basename in live.list(prefix):
                    break
                time.sleep(0.5)
            else:
                raise AssertionError(
                    f"{basename} not visible in list({prefix}) after polling"
                )

            assert live.download_obj(key) == payload
        finally:
            _cleanup(key)

    last_error = None
    for _attempt_idx in range(max_attempts):
        try:
            _attempt()
            return  # success
        except Exception as ex:  # noqa: BLE001 — all failures are retry candidates
            last_error = ex
            time.sleep(1.0)

    raise AssertionError(
        f"live roundtrip failed after {max_attempts} end-to-end attempts; "
        f"last error: {last_error!r}"
    )
