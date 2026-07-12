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

"""Optional shared-secret auth. No-op unless DASHBOARD_API_KEY is set (preserves current internal-only behavior)."""

from fastapi import Header, HTTPException

from .config import DASHBOARD_API_KEY


def require_api_key(x_api_key: str = Header(default=""), authorization: str = Header(default="")):
    if not DASHBOARD_API_KEY:
        return
    token = x_api_key
    if not token and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    if token != DASHBOARD_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
