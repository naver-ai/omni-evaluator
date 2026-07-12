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

"""Unit tests for utils/string.py — URL/integer/numeric validation, parsing, format keys, base64."""
import numpy as np

from omni_evaluator.utils.string import (
    decode_base64_string,
    extract_format_keys,
    is_integer,
    is_numeric,
    is_url,
    parse_string,
)


# ─────────────────────────────────────────────────────────────
# decode_base64_string — standard / url-safe / missing-padding decoding
# ─────────────────────────────────────────────────────────────

def test_decode_base64_string():
    """bytes pass through as-is; str is decoded for both standard and url-safe (with padding correction)."""
    import base64

    raw = b"hello world"
    assert decode_base64_string(raw) is raw
    assert decode_base64_string(base64.b64encode(raw).decode()) == raw
    # url-safe string with stripped padding is also restored
    no_pad = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    assert decode_base64_string(no_pad) == raw


# ─────────────────────────────────────────────────────────────
# extract_format_keys — extraction of str.format named placeholders
# ─────────────────────────────────────────────────────────────

def test_extract_format_keys():
    """Returns only named fields recognized by str.Formatter, in order."""
    assert extract_format_keys("{a} hello {b}") == ["a", "b"]
    assert extract_format_keys("no placeholder") == []


# ─────────────────────────────────────────────────────────────
# is_integer — int/np.integer/integer strings return int, everything else returns None
# ─────────────────────────────────────────────────────────────

def test_is_integer():
    """int/np.integer/integer strings return int; others (decimal strings/non-numeric/type mismatch) return None."""
    assert is_integer(42) == 42
    assert is_integer(np.int64(7)) == 7
    assert is_integer(" 13 ") == 13
    assert is_integer("4.2") is None
    assert is_integer("abc") is None
    assert is_integer([1]) is None


# ─────────────────────────────────────────────────────────────
# is_numeric — numeric values/strings (including %) return float, everything else returns None
# ─────────────────────────────────────────────────────────────

def test_is_numeric():
    """Numeric values/strings (with %, whitespace) return float; non-numeric/type mismatches return None."""
    assert is_numeric(3) == 3.0
    assert is_numeric("12.5") == 12.5
    assert is_numeric(" 50% ") == 50.0
    assert is_numeric("abc") is None
    assert is_numeric([1]) is None


# ─────────────────────────────────────────────────────────────
# is_url — http/https + netloc check
# ─────────────────────────────────────────────────────────────

def test_is_url():
    """Returns True only when both http/https and netloc are present."""
    assert is_url("https://example.com/path") is True
    assert is_url("http://host:8080") is True
    assert is_url("ftp://example.com") is False
    assert is_url("not a url") is False


# ─────────────────────────────────────────────────────────────
# parse_string — JSON → literal_eval → original fallback chain
# ─────────────────────────────────────────────────────────────

def test_parse_string():
    """Tries JSON → literal_eval → original in order; returns input as-is if all fail."""
    assert parse_string('{"a": 1}') == {"a": 1}
    assert parse_string("[1, 2]") == [1, 2]
    assert parse_string("not parsable") == "not parsable"
