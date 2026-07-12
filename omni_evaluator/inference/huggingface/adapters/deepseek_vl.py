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

import copy
import logging
import numpy as np
from omegaconf import DictConfig, ListConfig
import os
from pathlib import Path
import PIL
import soundfile
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel,
    BitsAndBytesConfig,
)
from typing import List, Tuple, Dict, Any, Union, Optional

from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.inference.huggingface.adapters.janus import JanusModule
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes
from omni_evaluator.utils.torch import get_compute_capability

logger = logging.getLogger(__name__)


class DeepSeekVlModule(JanusModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=False,
        support_audio_generation=False,
        support_image_generation=True,
        support_text_generation=True,
        support_video_generation=False,
        support_compute_perplexity=True,
    ).to_dict()
    AUDIO_PLACEHOLDER = "<audio_placeholder>\n"
    IMAGE_PLACEHOLDER = "<image_placeholder>\n"
    VIDEO_PLACEHOLDER = "<video_placeholder>\n"
    
    def __init__(
        self,
        *args, **kwargs,
    ):
        # Inherit JanusModule's methods (generate_text / generate_image / compute_perplexity),
        # but load the model with deepseek_vl's own class. JanusModule.__init__ registers and
        # loads janus's MultiModalityCausalLM, but a deepseek-vl checkpoint has the empty gen_*
        # submodules that janus requires, so it breaks with `class_name  is invalid`
        # (and it would be double loading since we overwrite with the deepseek loader below).
        # So skip the parent JanusModule.__init__ and call only the grandparent HuggingfaceModule.__init__.
        HuggingfaceModule.__init__(self, *args, **kwargs)

        try:
            # from transformers import AutoProcessor # should import here again
            from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
            # from deepseek_vl.utils.io import load_pil_images
        except Exception as ex:
            logger.warning('Could not import dependencies for `deepseek_vl`')
            raise

        # fallback to fp32 if fp16 is given
        if get_compute_capability() >= 8.0:
            self.model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            self.model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **self.model_kwargs,
        ).eval()
        self.processor = VLChatProcessor.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = self.processor.tokenizer