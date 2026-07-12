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

"""Caches for inference benchmark discovery and S3 listings."""

import time
from collections import OrderedDict
from threading import Lock

_MAX_ENTRIES = 1000  # LRU cap per cache
_S3_TTL = 300  # 5 minutes, matches scan_cache's S3 TTL

_BENCH_CACHE: "OrderedDict[tuple[str, str], list[str]]" = OrderedDict()
_BENCH_ENGINE_CACHE: "OrderedDict[tuple[str, str], dict[str, str]]" = OrderedDict()
# S3 key cache stores (timestamp, value); entries older than _S3_TTL are misses.
_S3_KEY_CACHE: "OrderedDict[str, tuple[float, dict[str, list[str]]]]" = OrderedDict()
_LOCK = Lock()


def _lru_put(cache: OrderedDict, key, value) -> None:
    """Insert/refresh key as most-recent and evict oldest beyond _MAX_ENTRIES.

    Caller must hold _LOCK.
    """
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _MAX_ENTRIES:
        cache.popitem(last=False)


def get_bench_cache(key: tuple[str, str]) -> tuple[list[str], dict[str, str]] | None:
    with _LOCK:
        if key in _BENCH_CACHE and key in _BENCH_ENGINE_CACHE:
            _BENCH_CACHE.move_to_end(key)
            _BENCH_ENGINE_CACHE.move_to_end(key)
            return _BENCH_CACHE[key], _BENCH_ENGINE_CACHE[key]
    return None


def set_bench_cache(key: tuple[str, str], benches: list[str], eng_map: dict[str, str]) -> None:
    with _LOCK:
        _lru_put(_BENCH_CACHE, key, benches)
        _lru_put(_BENCH_ENGINE_CACHE, key, eng_map)


def get_s3_key_cache(model: str) -> dict[str, list[str]] | None:
    now = time.time()
    with _LOCK:
        cached = _S3_KEY_CACHE.get(model)
        if cached and (now - cached[0]) < _S3_TTL:
            _S3_KEY_CACHE.move_to_end(model)
            return cached[1]
    return None


def set_s3_key_cache(model: str, keys: dict[str, list[str]]) -> None:
    with _LOCK:
        _lru_put(_S3_KEY_CACHE, model, (time.time(), keys))
