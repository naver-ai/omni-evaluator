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

import argparse
import asyncio
import contextlib
import copy
import json
import numpy as np
import PIL
from PIL import Image
import sys
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence, Callable, Iterable

from omni_evaluator.api import get_api_group
from omni_evaluator.api.chat_completions import (
    batch_chat_completion_async,
    batch_chat_completion_sync,
)
from omni_evaluator.evaluation.prepare_dataset import (
    sample_to_record as default_sample_to_record,
)
from omni_evaluator.evaluation.metrics.judge_evaluator import JudgeEvaluator
from omni_evaluator.schemas.chat import (
    OcrToken, EntityToken,
    Message as ChatMessage, 
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
)
from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from omni_evaluator.schemas.generation_options import ApiGenerationOptions
from omni_evaluator.schemas.inference import Record
from omni_evaluator.schemas.task import TaskConfig, TaskInference, TaskInferenceGenerationOptions
from omni_evaluator.utils.data import format_task_prompt, normalize_unit
from omni_evaluator.utils.string import is_numeric

from omni_evaluator.evaluation.builtin.tasks.charxiv_descriptive_validation.custom import *