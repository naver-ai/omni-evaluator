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

"""Common fixtures for S3Client L1 deterministic unit tests — only two client attrs are replaced with fakes on a real instance."""
from __future__ import annotations

import io
from typing import Any, Callable, Dict, List, Optional

import pytest


# ─────────────────────────────────────────────────────────────
# byte-store pyarrow fake for upload_obj / download_obj round-trip
# ─────────────────────────────────────────────────────────────


class _CapturingStream(io.BytesIO):
    """`open_output_stream` return stream — captures to store at close() time."""

    def __init__(self, store: Dict[str, bytes], path: str) -> None:
        super().__init__()
        self._store = store
        self._path = path
        self._captured = False

    def __enter__(self) -> "_CapturingStream":
        return self

    def __exit__(self, *args: Any) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if not self._captured and not self.closed:
            self._store[self._path] = self.getvalue()
            self._captured = True
        super().close()


class ByteStoreFs:
    """In-memory fake for `pyarrow.fs.S3FileSystem` byte-roundtrip only."""

    def __init__(self) -> None:
        self.store: Dict[str, bytes] = {}
        self.deleted: List[str] = []

    def open_output_stream(self, path: str) -> _CapturingStream:
        return _CapturingStream(self.store, path)

    def open_input_stream(self, path: str) -> io.BytesIO:
        if path not in self.store:
            raise FileNotFoundError(path)
        return io.BytesIO(self.store[path])

    def delete_file(self, path: str) -> None:
        self.deleted.append(path)
        self.store.pop(path, None)


# ─────────────────────────────────────────────────────────────
# S3Client factory — creates a real instance then overwrites only two client attrs
# ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_global_boto_client(monkeypatch) -> None:
    """Resets the module-global `GLOBAL_BOTO_CLIENT` to None for each test."""
    import omni_evaluator.clients.s3_client as s3_module

    monkeypatch.setattr(s3_module, "GLOBAL_BOTO_CLIENT", None)


@pytest.fixture
def make_s3_client() -> Callable[..., Any]:
    """Factory that creates an `S3Client` and overwrites two client attrs with the given arguments."""
    from omni_evaluator.clients.s3_client import S3Client

    def _make(
        pyarrow_client: Optional[Any] = None,
        boto_client: Optional[Any] = None,
        bucket_name: str = "bucket",
    ) -> S3Client:
        client = S3Client(
            bucket_name=bucket_name,
            access_key="ak",
            secret_key="sk",
            endpoint_url="http://fake",
            region="kr-standard",
            verbose=False,
        )
        if pyarrow_client is not None:
            client.pyarrow_client = pyarrow_client
        if boto_client is not None:
            client.boto_client = boto_client
        return client

    return _make
