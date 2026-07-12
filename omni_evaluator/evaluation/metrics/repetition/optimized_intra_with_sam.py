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
from collections import Counter, defaultdict
from typing import Union, List, Tuple

from transformers import AutoTokenizer


def remove_code_blocks(text: str) -> str:
    pattern = r"```.*?```"
    return re.sub(pattern, "", text, flags=re.DOTALL)


def is_not_markdown_or_url(text: str) -> bool:
    url_pattern = r"https?://\S+|www\.\S+"
    if re.search(url_pattern, text):
        return False
    if re.fullmatch(r"[\W\n\r_]+", text):
        return False
    return True


class _State:
    __slots__ = ("next", "link", "len", "first_pos", "occ")

    def __init__(self):
        self.next = dict()  # edge: token -> state id
        self.link = -1  # suffix link
        self.len = 0  # longest length in this state
        self.first_pos = -1  # index of first end‑position
        self.occ = 0  # end‑pos count (to be propagated)


class _SuffixAutomaton:
    def __init__(self):
        self.states = [_State()]  # state 0 = root
        self.last = 0  # id of the state representing whole string so far

    def extend(self, token: int, pos: int):
        st_new = len(self.states)
        self.states.append(_State())
        self.states[st_new].len = self.states[self.last].len + 1
        self.states[st_new].first_pos = pos
        self.states[st_new].occ = 1

        p = self.last
        while p >= 0 and token not in self.states[p].next:
            self.states[p].next[token] = st_new
            p = self.states[p].link

        if p == -1:
            self.states[st_new].link = 0
        else:
            q = self.states[p].next[token]
            if self.states[p].len + 1 == self.states[q].len:
                self.states[st_new].link = q
            else:
                # clone state q
                clone = len(self.states)
                self.states.append(_State())
                self.states[clone].next = self.states[q].next.copy()
                self.states[clone].len = self.states[p].len + 1
                self.states[clone].link = self.states[q].link
                self.states[clone].first_pos = self.states[q].first_pos
                # no occ init – will be accumulated
                while p >= 0 and self.states[p].next.get(token, -1) == q:
                    self.states[p].next[token] = clone
                    p = self.states[p].link
                self.states[q].link = self.states[st_new].link = clone
        self.last = st_new

    def propagate_occurrences(self):
        # sort states by length descending for end‑pos propagation
        order = sorted(range(1, len(self.states)), key=lambda s: self.states[s].len, reverse=True)
        for s in order:
            link = self.states[s].link
            if link >= 0:
                self.states[link].occ += self.states[s].occ


def get_soft_repeat_score_sam(
    tokens: List[int], tokenizer: AutoTokenizer, min_length: int = 4
) -> Tuple[tuple, float, int]:
    """Measure soft repetition using a Suffix Automaton. Returns (pattern, score, count)."""
    if len(tokens) < min_length * 2:  # too short to contain repetition
        return (), 0.0, 0

    # 1) build Suffix Automaton
    sam = _SuffixAutomaton()
    for idx, tok in enumerate(tokens):
        sam.extend(tok, idx)
    sam.propagate_occurrences()

    best_pattern = ()
    best_score = 0.0
    best_count = 0

    half = len(tokens) // 2
    for state in sam.states[1:]:  # skip state 0 (root)
        n = state.len
        if n < min_length or n > half:
            continue
        cnt = state.occ
        if cnt < 2:
            continue
        score = (cnt - 1) * n
        if score <= best_score:
            continue

        # extract candidate pattern: end index = first_pos, length = n
        end_idx = state.first_pos
        patt_tokens = tuple(tokens[end_idx - n + 1 : end_idx + 1])
        decoded = tokenizer.decode(patt_tokens)

        if is_not_markdown_or_url(decoded):
            best_pattern = patt_tokens
            best_score = score
            best_count = cnt

    return best_pattern, best_score, best_count


def check_soft_repetition(texts: List[str], tokenizer: AutoTokenizer, threshold: float = 33.0, min_length: int = 4):
    """Check a list of texts for soft repetition using a Suffix Automaton."""
    flagged_indices = []
    flagged_info = []

    for i, text in enumerate(texts):
        cleaned = remove_code_blocks(text)
        tokens = tokenizer.encode(cleaned, add_special_tokens=False)

        pattn, score, cnt = get_soft_repeat_score_sam(tokens, tokenizer, min_length)

        if score > threshold:
            flagged_indices.append(i)
            flagged_info.append(
                {
                    "index": i,
                    "original_text": text,
                    "score": score,
                    "pattern_tokens": pattn,
                    "pattern_str": tokenizer.decode(pattn),
                    "count": cnt,
                }
            )

    return flagged_indices, flagged_info


if __name__ == "__main__":
    sample_texts = [
        "안녕하세요 안녕하세요 안녕하세요 안녕하세요 제가 말했잖아요 안녕하세요",
        "이 텍스트는 반복이 거의 없어요.",
        "```python\nprint('hello')\n```\n이 코드는 제외됩니다.",
        "URL 테스트 http://example.com",
        "특수문자만!!!!!!!???"
    ]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    result_idx, result_info = check_soft_repetition(sample_texts, tokenizer, threshold=33.0)
    print("=== results ===")
    print("Flagged:", result_idx)
    for info in result_info:
        print(info)


