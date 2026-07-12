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

import importlib.util
from typing import Optional


def require_package(
    name: str,
    *,
    extras: Optional[str] = None,
    feature: Optional[str] = None,
) -> None:
    # Raise ImportError with an install hint when an optional dependency is missing.
    # Call from inside a model adapter's __init__ (or first method that needs the dep)
    # so that importing the adapter module itself stays free of side-effects.
    if importlib.util.find_spec(name) is not None:
        return
    install_target = extras or name
    label = feature or name
    raise ImportError(
        f"{label} requires `{name}`. Install with: pip install {install_target}"
    )
