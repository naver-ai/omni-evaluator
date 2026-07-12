# Reference from https://github.com/gmftbyGMFTBY/Rep-Dropout (MIT)

# Modifications Copyright (c) 2026-present NAVER Cloud Corp.
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

import os
import json
import pandas as pd
from nltk import ngrams
from tqdm import tqdm

from omni_evaluator.utils.data import safe_percentage


# —————————————————————————————————————
# per-sample repetition metric functions
# —————————————————————————————————————
# reference: https://github.com/gmftbyGMFTBY/Rep-Dropout/blob/main/repetition_dropout/utils/evaluation.py
def eval_text(text, n):
    tokens = text.strip().split()
    ngram_list = list(ngrams(tokens, n))
    seen = set()
    dup_count = 0
    for ng in ngram_list:
        if ng in seen:
            dup_count += 1
        else:
            seen.add(ng)
    total = len(ngram_list)
    return dup_count, total


def rep_n_single(text, n):
    dup, total = eval_text(text, n)
    return safe_percentage(dup, total)


def rep_r_single(text):
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    # 2-gram counting
    counter = {}
    for j in range(len(tokens) - 1):
        gm = f"{tokens[j]} {tokens[j+1]}"
        counter[gm] = counter.get(gm, 0) + 1
    # generate labels
    label = [0] * len(tokens)
    for i in range(1, len(tokens)):
        if counter[f"{tokens[i-1]} {tokens[i]}"] > 1:
            label[i - 1] = label[i] = 1
    return safe_percentage(sum(label), len(label))


def rep_w_single(text, w=16):
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    rep_w = 0
    for idx in range(1, len(tokens)):
        if tokens[idx] in set(tokens[max(0, idx - w) : idx]):
            rep_w += 1
    return rep_w / (len(tokens) - 1) * 100
