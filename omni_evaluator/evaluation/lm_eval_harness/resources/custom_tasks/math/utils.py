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

from typing import Dict, List

import datasets
from math_verify import parse, verify


# parsing_timeout (sec) override. Upstream math_verify.parse default = 5s
# is too short for some predictions whose latex normalization regex
# (latex2sympy2_extended._fix_malformed_operators) backtracks for tens of
# seconds. 30s recovers cases that just need more time; pathological
# divergence still aborts but is swallowed by the BaseException catch below
# so evaluation continues.
_PARSE_TIMEOUT_SEC = 30


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        parsed = parse(doc["solution"], parsing_timeout=_PARSE_TIMEOUT_SEC)[0]
        parsed = str(parsed)
        out_doc = {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": parsed,
        }
        return out_doc

    return dataset.map(_process_doc)


def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    # math_verify.parse triggers a SIGALRM-based timeout deep inside the
    # latex normalizer; on divergent regex paths (default 5s) the
    # TimeoutException propagates and kills the whole evaluation run.
    #
    # Two-layered safety:
    #   1) parsing_timeout=_PARSE_TIMEOUT_SEC (>>5) recovers slow-but-finite
    #      normalizations.
    #   2) BaseException catch returns 0 for the remaining pathological cases
    #      so a single bad pred can't abort the run.
    #
    # NOTE: math_verify.errors.TimeoutException subclasses *BaseException*
    # directly (not Exception), so a plain `except Exception` misses it.
    # We catch BaseException and explicitly re-raise the two interpreter-
    # control signals (KeyboardInterrupt, SystemExit) so the user can still
    # Ctrl-C / exit. We deliberately do NOT `from math_verify.errors import
    # TimeoutException` — that submodule isn't present in some math_verify
    # release tarballs, and the BaseException catch covers it regardless.
    retval = 0
    try:
        gold = parse(doc["solution"], parsing_timeout=_PARSE_TIMEOUT_SEC)
        answer = parse(results[0], parsing_timeout=_PARSE_TIMEOUT_SEC)
        if verify(gold, answer):
            retval = 1
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        retval = 0
    return {"exact_match": retval}
