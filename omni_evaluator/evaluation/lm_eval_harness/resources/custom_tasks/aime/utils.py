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

import re
from typing import Dict, List
from math_verify import parse, verify

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    retval = 0

    # answer maybe either in answer / Answer field, so we check both
    if "answer" in doc:
        answer_field = "answer"
    elif "Answer" in doc:
        answer_field = "Answer"
    else:
        raise ValueError("No answer field found in document")

    gold = parse("\\boxed{"+str(doc[answer_field])+"}")
    answer = parse(results[0])
    if verify(gold, answer):
        retval = 1
    results = {
        "exact_match": retval,
    }
    return results

