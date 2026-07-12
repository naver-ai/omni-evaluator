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

"""Case data for `omni_evaluator/postprocess/code` tests."""

import pytest


# Empty / whitespace-only prediction — if the extracted result has length 0 after strip, return None.
CODE_EMPTY_OR_WHITESPACE = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="spaces"),
    pytest.param("\n\t\n", id="newlines_and_tab"),
]

# Explicit fence blocks — all lang alternatives of `PATTERN__CODE_BLOCK` and the no-tag case.
# 6 named langs from `r'```(?:python|java|javascript|sh|cpp|json)?\n((?:(?!```)[\s\S])+)```'`
# + empty match of the optional group (no-tag). Each entry: (prediction, language_arg, expected_code).
CODE_EXPLICIT_BLOCK = [
    pytest.param("```python\ndef foo(): return 42\n```", "python", "def foo(): return 42\n", id="python"),
    pytest.param("```java\nclass Foo {}\n```", "java", "class Foo {}\n", id="java"),
    pytest.param(
        "```javascript\nfunction foo() {}\n```", "javascript", "function foo() {}\n",
        id="javascript",
    ),
    pytest.param("```sh\necho hi\n```", "sh", "echo hi\n", id="sh"),
    pytest.param("```cpp\nint main() {}\n```", "cpp", "int main() {}\n", id="cpp"),
    pytest.param('```json\n{"a": 1}\n```', "json", '{"a": 1}\n', id="json"),
    # `(?:...)?` in PATTERN__CODE_BLOCK matches empty — extracts even when fence has no lang tag.
    pytest.param("```\nplain code\n```", "python", "plain code\n", id="no_lang_tag_in_fence"),
]
