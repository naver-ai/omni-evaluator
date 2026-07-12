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

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# When a question (problem specification) is given, generate a correct Python program that matches the specification and passes all tests.
SYSTEM_PROMPTS = {
    "code/general": None,
    "code/python": f"""You are an expert Python programmer.
Always produce complete, executable Python code that fulfills the given requirements.
If a partial implementation is provided, extend it appropriately rather than rewriting from scratch.
Unless explicitly requested, return only the code — omit any explanatory text, markdown formatting, or test cases.""",
}
