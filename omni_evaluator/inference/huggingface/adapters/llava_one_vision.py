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
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from typing import List, Tuple, Dict, Any, Union, Optional

try:
    from transformers import AutoTokenizer, AutoProcessor
except Exception as ex:
    logger.warning('Could not import dependencies for `llava_one_vision`')
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


class LlavaOneVisionModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=False,
        support_text_generation=True,
        support_compute_perplexity=False,
    ).to_dict()
    
    def __init__(
        self,
        *args, **kwargs,
    ):
        from transformers import LlavaOnevisionForConditionalGeneration
        super().__init__(*args, **kwargs)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            **self.model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
    

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()
        # add system if no system turn
        if not any([_message["role"] == "system" for _message in messages]):
            _message = ChatMessage(
                role="system",
                content=ChatTextContent(type="text", value="You are a helpful assistant."),
            ).to_dict(template="hf")
            messages.insert(0, _message)

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
        images = ChatMessage.get_images(message=user_message)

        # preprocess
        prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self.processor(
            images=list(images.values()),
            text=prompt,
            return_tensors="pt",
        ).to(device=self.model.device)

        stopping_criteria = self._resolve_stopping_criteria(
            generation_options=generation_options,
            input_ids=model_inputs["input_ids"],
        )

        # inference: generate via hf_model
        generated_ids = self.model.generate(
            **model_inputs,
            **generation_options,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )[0].detach().cpu()
        prediction_ids = generated_ids[model_inputs["input_ids"].shape[1]:]
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