# Enumerable value namespaces grouped by domain. Imported here so callers can
# do `from omni_evaluator.enums import EvaluationEngine`. The legacy
# `from omni_evaluator import EvaluationEngine` re-export in the package root
# is preserved for backward compatibility.

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

from omni_evaluator.enums.dataset import CombineMethod, DatasetSource
from omni_evaluator.enums.engine import (
    ApiGroup,
    EvaluationEngine,
    EvaluationMethod,
    HuggingfaceModelGroup,
    InferenceEngine,
    T2IGeneratorType,
)
from omni_evaluator.enums.evaluation import NullPredictionPolicy
from omni_evaluator.enums.media import AudioFormat, ImageFormat, Modality, VideoFormat
from omni_evaluator.enums.task import SpatialGroundingType, SubtaskType, TaskType

__all__ = [
    # dataset
    "CombineMethod",
    "DatasetSource",
    # engine
    "ApiGroup",
    "EvaluationEngine",
    "EvaluationMethod",
    "HuggingfaceModelGroup",
    "InferenceEngine",
    "T2IGeneratorType",
    # evaluation
    "NullPredictionPolicy",
    # media
    "AudioFormat",
    "ImageFormat",
    "Modality",
    "VideoFormat",
    # task
    "SpatialGroundingType",
    "SubtaskType",
    "TaskType",
]
