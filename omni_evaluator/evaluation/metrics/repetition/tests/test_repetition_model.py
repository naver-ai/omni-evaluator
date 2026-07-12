# file location: omni_evaluator/evaluation/metrics/repetition/tests/test_repetition_model.py

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

import pytest
from ..repetition_model import RepetitionModel

@pytest.fixture(scope="module")
def repetition_model():
    """Create a RepetitionModel instance once for the entire test session."""
    try:
        return RepetitionModel()
    except Exception as e:
        pytest.skip(f"Failed to load RepetitionModel. Check the resources directory: {e}")

def test_predicts_high_for_repetitive_text(repetition_model):
    """Verify that a high probability is predicted for obviously repetitive text."""
    # text with severe repetition
    repetitive_text = "이 문장은 반복됩니다. 이 문장은 반복됩니다. 이 문장은 반복됩니다. 이 문장은 반복됩니다. 이 문장은 반복됩니다. 이 문장은 반복됩니다. 이 문장은 반복됩니다. 이 문장은 반복됩니다. 이 문장은 반복됩니다. 이 문장은 반복됩니다."
    
    probability = repetition_model.predict_proba(repetitive_text)
    
    # predicted probability should be >= 0.9
    assert probability > 0.9

def test_predicts_low_for_normal_text(repetition_model):
    """Verify that a low probability is predicted for normal non-repetitive text."""
    # normal sentence
    normal_text = "오늘은 날씨가 좋아서 공원에 산책하러 가기 좋은 날입니다."
    
    probability = repetition_model.predict_proba(normal_text)
    
    # predicted probability should be <= 0.1
    assert probability < 0.1

def test_handles_short_text(repetition_model):
    """Verify that very short text is handled without error and returns a low probability."""
    # edge case: short text
    short_text = "단문"
    
    probability = repetition_model.predict_proba(short_text)
    
    # predicted probability should be <= 0.1
    assert probability < 0.1