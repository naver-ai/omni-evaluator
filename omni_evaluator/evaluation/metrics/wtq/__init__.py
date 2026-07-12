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
import unicodedata
from math import isnan, isinf

_EPS = 1e-6

# Each cell is a (type, key, canonical) tuple:
#   type      : 'str' | 'num' | 'date'
#   key       : str | float | (year, month, day)
#   canonical : normalized string for fallback comparison
_WILDCARD = -1


def _normalize(text):
    if not isinstance(text, str):
        text = text.decode("utf-8", errors="ignore")
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if unicodedata.category(ch) != "Mn"
    )
    text = re.sub("[‘’´`]", "'", text)
    text = re.sub("[“”]", '"', text)
    text = re.sub("[‐‑‒–—−]", "-", text)
    while True:
        prev = text
        text = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", text.strip())
        text = re.sub(r"(?<!^)( \([^)]*\))*$", "", text.strip())
        text = re.sub(r'^"([^"]*)"$', r"\1", text.strip())
        if text == prev:
            break
    if text.endswith("."):
        text = text[:-1]
    return re.sub(r"\s+", " ", text, flags=re.U).lower().strip()


def _try_parse_num(text):
    try:
        return int(text)
    except ValueError:
        pass
    try:
        val = float(text)
        if not isnan(val) and not isinf(val):
            return val
    except ValueError:
        pass
    return None


def _try_parse_date(text):
    try:
        parts = text.lower().split("-")
        assert len(parts) == 3
        y = _WILDCARD if parts[0] in ("xx", "xxxx") else int(parts[0])
        m = _WILDCARD if parts[1] == "xx" else int(parts[1])
        d = _WILDCARD if parts[2] == "xx" else int(parts[2])
        assert not (y == m == d == _WILDCARD)
        assert m == _WILDCARD or 1 <= m <= 12
        assert d == _WILDCARD or 1 <= d <= 31
        return (y, m, d)
    except Exception:
        return None


def _make_cell(raw, hint=None):
    ref = hint if hint else raw
    canonical = _normalize(raw)

    num = _try_parse_num(ref)
    if num is not None:
        key = int(num) if isinstance(num, float) and abs(num - round(num)) < _EPS else num
        return ("num", float(key), canonical)

    ymd = _try_parse_date(ref)
    if ymd is not None:
        y, m, d = ymd
        if m == d == _WILDCARD:
            return ("num", float(y), canonical)
        return ("date", ymd, canonical)

    return ("str", _normalize(raw), canonical)


def _cell_matches(a, b):
    type_a, key_a, canonical_a = a
    type_b, key_b, canonical_b = b
    if canonical_a == canonical_b:
        return True
    if type_a != type_b:
        return False
    if type_a == "num":
        return abs(key_a - key_b) < _EPS
    if type_a == "date":
        return key_a == key_b
    return False


def to_value_list(strings, hints=None):
    assert isinstance(strings, (list, tuple, set))
    if hints is not None:
        assert isinstance(hints, (list, tuple, set))
        assert len(strings) == len(hints)
        return list(set(_make_cell(s, h) for s, h in zip(strings, hints)))
    return list(set(_make_cell(s) for s in strings))


def check_denotation(expected, predicted):
    if len(expected) != len(predicted):
        return False
    return all(any(_cell_matches(e, p) for p in predicted) for e in expected)


def _tsv_unescape(token):
    return token.replace(r"\n", "\n").replace(r"\p", "|").replace("\\\\", "\\")


def tsv_unescape_list(field):
    return [_tsv_unescape(tok) for tok in field.split("|")]
