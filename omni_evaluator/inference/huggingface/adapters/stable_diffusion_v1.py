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

from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes
from omni_evaluator.utils.torch import get_compute_capability


class StableDiffusionV1Module(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=False,
        support_text_understanding=True,
        support_video_understanding=False,
        support_audio_generation=False,
        support_text_generation=False,
        support_image_generation=True,
        support_video_generation=False,
        support_compute_perplexity=False,
    ).to_dict()
    
    def __init__(
        self,
        use_safetensors: Optional[bool] = True, 
        variant: Optional[str] = "fp16",
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_safetensors = use_safetensors
        self.variant = variant
        
        if self.model_name_or_path in [
            "stabilityai/stable-diffusion-xl-base-1.0",
        ]:
            from diffusers import DiffusionPipeline
            self.model = DiffusionPipeline.from_pretrained(
                self.model_name_or_path, 
                use_safetensors=self.use_safetensors,
                variant=self.variant,
                **self.model_kwargs,
            )
            self.model.enable_xformers_memory_efficient_attention()
        else:
            from diffusers import StableDiffusionPipeline
            self.model = StableDiffusionPipeline.from_pretrained(
                self.model_name_or_path,
                **self.model_kwargs,
            )
        self.model.enable_attention_slicing()
        self.image_processor = getattr(self.model, "image_processor", None)
        self.tokenizer = getattr(self.model, "tokenizer", None)

    def generate_image(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        return_audio: Optional[bool] = None,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()
        user_messages = ChatMessage.get_user_messages(messages=messages)
        prompt = ChatMessage.get_query(message=user_messages[-1])
        
        # inference: generate via hf_model
        _output = self.model(
            prompt,
            **generation_options,
        )
        generated_images = list()
        for _image in _output.images:
            _image = to_image_bytes(image=_image, encode_base64=True)
            generated_images.append(_image)
        prediction = generated_images
        
        output = HuggingfaceInferenceOutput(
            prediction_ids=None,
            prediction=prediction,
            generated_ids=None,
            generated_text=None,
            generated_images=generated_images,
            temp_paths=None,
        )
        return output