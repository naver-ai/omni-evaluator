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

from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class HarpResults:
    pattern: Union[str, list]
    num_repeats: int

    def __len__(self):
        # total repeated length = pattern length * repetition count
        return len(self.pattern) * self.num_repeats


class Harp:
    def __call__(self, s: str) -> Optional[HarpResults]:
        s_rev = s[::-1]
        best = None
        for i in range(1, len(s_rev)):
            count = 1
            for j in range(i, len(s_rev), i):
                if s_rev[j : j + i] != s_rev[:i]:
                    break
                count += 1
            if count > 1:
                pattern = s_rev[:i][::-1]
                res = HarpResults(pattern, count)
                if best is None or len(res) > len(best):
                    best = res
        return best


def harp_tokens(tokens):
    """
    tokens: List[str]
    Returns: (pattern token list, repetition count)
    """
    best_score = 0
    best_pattern = []
    best_count = 0
    L = len(tokens)
    for n in range(1, L // 2 + 1):
        count = 1
        # in the suffix region of the token list,
        # compare previous block (prev) with the last block (last), each of length n
        while n * (count + 1) <= L:
            prev_start = L - n * (count + 1)
            prev_end = L - n * count
            last_start = L - n * count
            last_end = L - n * (count - 1)
            if tokens[prev_start:prev_end] == tokens[last_start:last_end]:
                count += 1
            else:
                break
        if count > 1:
            score = n * count
            if score > best_score:
                best_score = score
                best_pattern = tokens[-n:]
                best_count = count
    return best_pattern, best_count
