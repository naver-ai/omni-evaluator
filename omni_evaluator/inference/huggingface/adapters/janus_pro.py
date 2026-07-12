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


class JanusProModule(JanusModule):
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
        super().__init__(*args, **kwargs)
        
        try:
            # from transformers import AutoProcessor # should import here again
            from janus.models import MultiModalityCausalLM, VLChatProcessor
        except Exception as ex:
            logger.warning('Could not import dependencies for `janus`')
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
    
    @classmethod
    def _to_janus_template(
        cls,
        messages: List[Union[Dict[str, Any], ChatMessage]],
        assistant_content: Optional[str] = None,
    ):
        conversation = list()
        _last_role = None
        for _message_idx, _message in enumerate(messages):
            _role = None
            _last_role = _message["role"]
            if _message["role"] == "system":
                _role = "<|System|>"
            elif _message["role"] == "assistant":
                _role = "<|Assistant|>"
            elif _message["role"] == "user":
                _role = "<|User|>"
            else:
                raise ValueError(f'invalid role: {_message["role"]}')

            _query = ""
            _audios, _images, _videos = list(), list(), list()
            for _content in _message["content"]:
                if _content["type"] == "text":
                    _query += _content["text"]
                elif (
                    _content["type"] == "audio" 
                    and _content["audio"]
                ):
                    _audios.append(_content["audio"])
                elif (
                    _content["type"] == "image" 
                    and _content["image"]
                ):
                    _images.append(_content["image"])
                elif (
                    _content["type"] == "video" 
                    and _content["video"]
                ):
                    _videos.append(_content["video"])
            if len(_audios) > 0:
                _query = f'{cls.AUDIO_PLACEHOLDER * len(_audios)}{_query}'
            if len(_images) > 0:
                _query = f'{cls.IMAGE_PLACEHOLDER * len(_images)}{_query}'
            if len(_videos) > 0:
                _query = f'{cls.VIDEO_PLACEHOLDER * len(_videos)}{_query}'
                
            conversation.append({
                "role": _role,
                "content": _query,
                # "audios": _audios,
                "images": _images,
                # "videos": _videos,
            })
            
        if _last_role != "assistant":
            if not assistant_content:
                assistant_content = ""
            conversation.append({
                "role": "<|Assistant|>",
                "content": assistant_content,
            })
        return conversation