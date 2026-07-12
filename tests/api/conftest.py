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

"""tests/api/ shared conftest — home for provider-agnostic fixtures."""
from __future__ import annotations

from typing import Optional

import pytest


@pytest.fixture
def parsed_response_factory():
    """Builder for the standard dict shape returned by provider parse_response() — for expected-value comparisons."""
    def _factory(
        prediction: str = "hello",
        reasoning_content: Optional[str] = None,
        tool_calls=None,
        function_call=None,
        annotations=None,
        finish_reason: str = "stop",
        latency: float = 0.1,
        error_message: Optional[str] = None,
        generated_text=None,
    ):
        return {
            "prediction": prediction,
            "reasoning_content": reasoning_content,
            "tool_calls": tool_calls,
            "function_call": function_call,
            "annotations": annotations if annotations is not None else [],
            "finish_reason": finish_reason,
            "latency": latency,
            "error_message": error_message,
            "generated_text": generated_text if generated_text is not None else prediction,
        }
    return _factory
