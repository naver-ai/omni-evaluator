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

from omegaconf import DictConfig, ListConfig
import os
from pathlib import Path
import PIL
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel,
    BitsAndBytesConfig,
)
from typing import List, Tuple, Dict, Any, Union, Optional

from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes


class LlamaVisionModule(HuggingfaceModule):
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
        
        from transformers import MllamaForConditionalGeneration
        
        self.model = MllamaForConditionalGeneration.from_pretrained(
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
        label: Optional[Union[Any, List[Any]]] = None,
        compute_perplexity: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()
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
        
        user_message = ChatMessage.get_user_messages(messages=messages)
        images = ChatMessage.get_images(message=user_message)
        # remove image since already collcted images
        for _message in messages:
            for _content in _message["content"]:
                if _content["type"] == "image":
                    _content.pop("image", None)

        if compute_perplexity:
            loss = None
            output = HuggingfaceInferenceOutput(
                loss=loss,
            )
            
        else:
            # preprocess
            model_inputs = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            )
            model_inputs = self.processor(
                images,
                model_inputs,
                add_special_tokens=False,
                return_tensors="pt"
            )

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
            prediction_ids = generated_ids[len(model_inputs.input_ids[0]):]
            # decode
            generation = self.processor.decode(
                generated_ids, 
                skip_special_tokens=False, 
                clean_up_tokenization_spaces=False
            )
            prediction = self.processor.decode(
                prediction_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
                
            output = HuggingfaceInferenceOutput(
                prediction_ids=prediction_ids,
                prediction=prediction,
                generated_ids=generated_ids,
                generated_text=generation,
                temp_paths=None,
            )

        return output