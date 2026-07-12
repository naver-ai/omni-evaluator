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
import re
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


class HyperclovaxModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=False,
        support_text_understanding=True,
        support_video_understanding=False,
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

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, 
            **self.model_kwargs,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            hf_cache_dir=self.cache_dir,
            trust_remote_code=self.model_kwargs.get("trust_remote_code", None),
        )
        self.stop_strings: List[str] = ["<|endofturn|>", "<|stop|>"]
    
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
        for _message_idx, _message in enumerate(messages):
            if isinstance(_message["content"], str):
                continue
            if (
                isinstance(_message["content"], dict)
                and _message["content"]["type"] == "text"
            ):
                messages[_message_idx]["content"] = _message["content"]["text"]
            elif (
                isinstance(_message["content"], (list, tuple))
            ):
                messages[_message_idx]["content"] = "\n".join([
                    _content["text"]
                    for _content in _message["content"]
                    if _content["type"] == "text"
                ])

        # preprocess
        force_reasoning = False
        skip_reasoning = True
        if self.reasoning:
            force_reasoning = True
            skip_reasoning = False

        model_inputs = self.tokenizer.apply_chat_template(   
            messages, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt", 
            add_generation_prompt=True,
            force_reasoning=force_reasoning,
            skip_reasoning=skip_reasoning,
        ).to(device=self.model.device)
        
        stopping_criteria = self._resolve_stopping_criteria(
            generation_options=generation_options,
            input_ids=model_inputs["input_ids"],
        )

        stop_strings = copy.deepcopy(self.stop_strings)
        if isinstance(generation_options.get("stop_strings", None), (list, tuple)):
            stop_strings += generation_options.pop("stop_strings", list())

        # inference: generate via hf_model
        output = self.model.generate(
            **model_inputs,
            **generation_options,
            stopping_criteria=stopping_criteria,
            stop_strings=stop_strings,
            tokenizer=self.tokenizer,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )
        output = output[0]
        prompt = model_inputs.get("prompt", None)
        if prompt:
            prompt = prompt[0]
        
        return {
            "output": output,
            "prompt": prompt,
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
        
        prediction_ids = generated_ids = _output["output"].detach().cpu()
        # decode
        generation = self.tokenizer.decode(
            generated_ids, 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )
        prediction = self.tokenizer.decode(
            prediction_ids, 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )

        reasoning_content = None
        _think_matches = re.findall(
            r'<\|im_start\|>assistant/think\n(.*?)<\|im_end\|>',
            prediction, 
            re.DOTALL,
        )
        if _think_matches:
            reasoning_content = _think_matches[-1]
        
        _assistant_matches = re.findall(
            r'<\|im_start\|>assistant\n(.*?)<\|im_end\|>',
            prediction,
            re.DOTALL,
        )
        if _assistant_matches:
            prediction = _assistant_matches[-1]
        elif "assistant\n" in prediction:
            prediction = prediction.split("assistant\n", maxsplit=1)[-1]
        if prediction.endswith("<|im_end|>"):
            prediction = prediction[:-len("<|im_end|>")]
            
        if reasoning_content:
            prediction = "</think>".join([reasoning_content, prediction])
            
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
        
        # preprocess
        force_reasoning = False
        skip_reasoning = True
        if self.reasoning:
            force_reasoning = True
            skip_reasoning = False
        
        # preprocess
        model_inputs = self.tokenizer.apply_chat_template(   
            messages, 
            tokenize=True, 
            return_dict=True, 
            return_tensors="pt", 
            add_generation_prompt=True,
            force_reasoning=force_reasoning,
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
            _model_inputs["attention_mask"] = torch.ones_like(
                _model_inputs["input_ids"],
            ).to(dtype=torch.long, device=_model_inputs["input_ids"].device)
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