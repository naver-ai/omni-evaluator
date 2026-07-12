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

"""Admin API (cache reset, debug, etc.)."""

from fastapi import APIRouter, Depends

from ..auth import require_api_key
from ..services.scan_cache import get_benchmark_engines, clear_cache

router = APIRouter()


@router.post("/reset", dependencies=[Depends(require_api_key)])
def reset_cache():
    """Clear all scan caches. Next request triggers fresh scan."""
    clear_cache()
    return {"ok": True, "message": "Cache cleared"}


@router.get("/debug/benchmark_engine", dependencies=[Depends(require_api_key)])
def debug_benchmark_engine():
    """Debug: return benchmark engine map count and sample."""
    m = get_benchmark_engines()
    by_eng = {}
    for b, e in m.items():
        by_eng.setdefault(e, []).append(b)
    return {
        "count": len(m),
        "by_engine": {e: len(lst) for e, lst in sorted(by_eng.items())},
        "sample": dict(list(m.items())[:10]),
    }
