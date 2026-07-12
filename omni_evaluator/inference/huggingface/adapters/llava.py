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
from omegaconf import DictConfig, ListConfig
import os
from pathlib import Path
import PIL
import torch
from typing import List, Tuple, Dict, Any, Union, Optional
import warnings

from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.chat import (
    OCR_PREFIX, ENTITY_PREFIX,
    Message as ChatMessage, 
    TextContent as ChatTextContent, 
)
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes
from omni_evaluator.utils.torch import get_compute_capability

logger = logging.getLogger(__name__)


class LlavaModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_image_understanding=True,
        support_video_understanding=False,
        support_audio_understanding=False,
        support_compute_perplexity=False,
    ).to_dict()
    
    def __init__(
        self,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates
        except Exception as ex:
            logger.warning('Could not import dependencies for `llava`')
        
        warnings.filterwarnings("ignore")
        
        self.model_kwargs.pop("low_cpu_mem_usage", None)
        self.model_kwargs["attn_implementation"] = "sdpa"
        cc = get_compute_capability()
        if cc is not None and cc >= 8.0:
            self.model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.model_name, self.prompt_version = None, None
        if "qwen" in self.model_name_or_path.lower():
            self.model_name = "llava_qwen"
            self.prompt_version = "qwen_1_5"
        else: # default # if "vicuna" in self.model_name_or_path.lower():
            self.model_name = "llava_vicuna"
            self.prompt_version = "v1"
        logger.info(f'Set llava prompt_version: {self.prompt_version}')
        
        self.conv_template = conv_templates[self.prompt_version]  # Make sure you use correct chat template for different models
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path=self.model_name_or_path, 
            model_base=None, 
            model_name=self.model_name, 
            **self.model_kwargs,
        )
        self.model = self.model.eval()
    

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX

        messages = [
            ChatMessage.preprocess_message(
                message=_message,
                remove_image=not self.ENGINE_FEATURES["support_image_understanding"],
                content_fields_image=None,
                remove_video=not self.ENGINE_FEATURES["support_video_understanding"],
                content_fields_video=None,
                remove_audio=not self.ENGINE_FEATURES["support_audio_understanding"],
                content_fields_audio=None,
            )
            for _message in messages
        ]

        user_message = ChatMessage.get_user_messages(messages=messages)[-1]
        query = ChatMessage.get_query(message=user_message)
        images = ChatMessage.get_images(message=user_message)

        IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        if self.model.config.mm_use_im_start_end:
            IMAGE_TOKEN = f'{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}'
        query = IMAGE_TOKEN * len(images) + f'\n{query}'

        conv = copy.deepcopy(self.conv_template)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # preprocess
        input_ids = tokenizer_image_token(
            prompt=prompt,
            tokenizer=self.tokenizer,
            image_token_index=IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(device=self.model.device)

        image_tensor, image_sizes = None, None
        if len(images) > 0:
            image_sizes = [image.size for image in images]
            image_tensor = process_images(
                images=images,
                image_processor=self.image_processor,
                model_cfg=self.model.config,
            ).to(device=self.model.device, dtype=self.model.dtype)

        stopping_criteria = self._resolve_stopping_criteria(
            generation_options=generation_options,
            input_ids=input_ids,
        )

        # inference: generate via hf_model
        generated_ids = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            **generation_options,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )[0].detach().cpu()
        prediction_ids = generated_ids[input_ids.shape[1]:]
        # decode
        generation = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prediction = self.tokenizer.decode(
            prediction_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        output = HuggingfaceInferenceOutput(
            prediction_ids=prediction_ids,
            prediction=prediction,
            generated_ids=generated_ids,
            generated_text=generation,
            temp_paths=None,
        )
        return output