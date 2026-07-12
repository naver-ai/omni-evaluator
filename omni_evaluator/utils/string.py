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

import ast
import base64
import json
import numpy as np
import re
import string
from typing import List, Tuple, Dict, Any, Union, Optional, Callable
import urllib


def is_url(string: str) -> bool:
    """Return True if *string* is a valid http/https URL, False otherwise."""
    try:
        result = urllib.parse.urlparse(string)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False

def is_integer(x: Any) -> Optional[int]:
    """Return *x* as int if it can be interpreted as an integer, otherwise None."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    elif isinstance(x, str):
        try:
            return int(x.strip())
        except (ValueError, TypeError):
            return None
    else:
        return None

def is_numeric(x: Any) -> Optional[float]:
    """Return *x* as float if it can be interpreted as a number (including '12.3%'), otherwise None."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    elif isinstance(x, str):
        x = x.strip()
        if x.endswith("%"):
            x = x[:-1]
        try:
            return float(x)
        except (ValueError, TypeError):
            return None
    else:
        return None

def parse_string(string: str) -> Any:
    """Try to parse *string* as JSON or a Python literal; return the original string on failure."""
    try:
        string = json.loads(string)
    except Exception as ex:
        try:
            string = ast.literal_eval(string)
        except Exception as ex:
            pass
    return string

def sanitize_name(name: str) -> str:
    """Collapse a name/identifier into a single filesystem- and key-safe token: ``.`` -> ``_``,
    path separators -> ``__``, whitespace -> ``_``. Shared by exp_name / version_name (output
    dir names) and the verifier metric alias so they normalize identically. Keeps the publisher
    in a hub id (``'Qwen/Qwen3.5-0.8B'`` -> ``'Qwen__Qwen3_5-0_8B'``)."""
    name = str(name).strip().replace(".", "_").replace("/", "__").replace("\\", "__")
    name = re.sub(r"\s+", "_", name)
    return name


def extract_format_keys(template: str) -> list[str]:
    output = list()
    # (literal_text, field_name, format_spec, conversion)
    for literal, field, spec, conv in string.Formatter().parse(template):
        if field is None:
            continue
        output.append(field)
    return output

def decode_base64_string(
    text: Union[str, bytes],
) -> bytes:
    if isinstance(text, bytes):
        return text
    
    try:
        return base64.b64decode(text, validate=True)
    except Exception as ex:
        # if url-safe base64 string or broken in padding
        padding = (-len(text)) % 4
        if padding:
            text += "=" * padding
        return base64.urlsafe_b64decode(text)
