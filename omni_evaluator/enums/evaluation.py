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

from enum import Enum


class NullPredictionPolicy(str, Enum):
    """How TextEvaluator handles null / non-string predictions before scoring.

    Applied uniformly at the record-iteration stage so every downstream metric
    sees a normalized string prediction.
    """
    miss: str = "miss"          # invalid → ""  (label="no" → FP / label="yes" → TP via "yes" mapping)
    skip: str = "skip"          # drop sample from predictions/labels list
    fallback: str = "fallback"  # replace with TaskTextEvaluator.fallback_value (None → "")
