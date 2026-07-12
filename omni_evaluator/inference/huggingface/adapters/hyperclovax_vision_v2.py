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
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import (
        AutoTokenizer, AutoProcessor,
        AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )
except Exception as ex:
    logger.warning('Could not import dependencies for `hyperclovax_vision_v2`')

from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes


class HyperclovaxSeedVisionV2Module(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=True,
        support_audio_generation=False,
        support_image_generation=False,
        support_text_generation=True,
        support_video_generation=False,
        support_compute_perplexity=True,
    ).to_dict()
    
    def __init__(
        self,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.model = AutoModelForImageTextToText.from_pretrained(
        # self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, 
            **self.model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            hf_cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            hf_cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
    
    def _generate(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
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

        skip_reasoning = True
        if self.reasoning:
            skip_reasoning = False

        # preprocess
        model_inputs = self.processor.apply_chat_template(   
            messages, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt", 
            add_generation_prompt=True,
            skip_reasoning=skip_reasoning,
        ).to(device=self.model.device)
        
        stopping_criteria = self._resolve_stopping_criteria(
            generation_options=generation_options,
            input_ids=model_inputs["input_ids"],
        )

        # inference: generate via hf_model            
        output = self.model.generate(
            **model_inputs,
            **generation_options,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )
        output = output[0]
        prompt = self.tokenizer.decode(model_inputs["input_ids"][0])
        
        return {
            "output": output,
            "prompt": prompt,
            "prompt_length": model_inputs["input_ids"].shape[-1],
        }
    
    def generate_text(
        self,
        *args, 
        **kwargs,
    ):
        generation_options = kwargs.pop("generation_options", dict())
        _output = self._generate(
            *args, 
            **kwargs,
            generation_options=generation_options,
        )
        
        generated_ids = _output["output"].detach().cpu()
        prediction_ids = generated_ids[_output["prompt_length"]:]
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
    
    def compute_perplexity(
        self,
        messages: List[Dict[str, Any]],
        options: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
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
        
        skip_reasoning = True
        if self.reasoning:
            skip_reasoning = False
        
        # preprocess
        model_inputs = self.processor.apply_chat_template(   
            messages, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt", 
            add_generation_prompt=True,
            skip_reasoning=skip_reasoning,
        ).to(device=self.model.device)

        perplexities = list()
        for _option in options:
            _model_inputs = copy.deepcopy(model_inputs)
            _option_token_ids = torch.tensor(
                self.tokenizer.encode(f'{_option}<|im_end|>'),
                dtype=torch.long,
                device="cpu",
            ) # e.g. One<|im_end|>
            _num_tokens = len(_option_token_ids) - 1
            
            _input_ids = torch.cat([
                _model_inputs["input_ids"][0],
                _option_token_ids[:-1].to(device=_model_inputs["input_ids"].device),
            ], dim=0)
            _labels = -100 * torch.ones(_input_ids.shape).to(dtype=torch.long, device="cpu")
            _labels[-len(_option_token_ids):] = _option_token_ids
            _model_inputs["input_ids"] = _input_ids.unsqueeze(dim=0)
            _model_inputs.pop("labels", None)
            
            with torch.no_grad():
                _output = self.model(
                    **_model_inputs,
                )
            _logits = _output["logits"].detach().to(device="cpu")
            _nll = F.cross_entropy(
                _logits.view(-1, _logits.shape[-1]),
                _labels,
                ignore_index=-100,
                reduction="sum",
            )
            _nll = _nll / _num_tokens
            _nll = _nll.item()
            # _nll = np.exp(_nll) # perplexity
            perplexities.append(_nll)
            
        prediction = np.argmin(perplexities)
        output = HuggingfaceInferenceOutput(
            prediction=prediction,
            perplexities=perplexities,
            temp_paths=None,
        )
        return output