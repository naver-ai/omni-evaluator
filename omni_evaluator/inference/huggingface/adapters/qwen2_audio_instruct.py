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

import base64
import copy
import logging
import numpy as np
from omegaconf import DictConfig, ListConfig
import os
from pathlib import Path
import PIL
import soundfile
import tempfile
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel,
    BitsAndBytesConfig,
)
from typing import List, Tuple, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes, to_nparray_audio, to_audio_bytes
from omni_evaluator.utils.torch import get_compute_capability


class Qwen2AudioInstructModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=True,
        support_image_understanding=False,
        support_text_understanding=True,
        support_video_understanding=False,
        support_audio_generation=False,
        support_text_generation=True,
        support_compute_perplexity=True,
    ).to_dict()
    
    def __init__(
        self,
        audio_samplerate: Optional[int] = 24_000,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        # self.model_kwargs["torch_dtype"] = torch.float16
        # if get_compute_capability() >= 8.0:
        #     self.model_kwargs["torch_dtype"] = torch.bfloat16
        #     self.model_kwargs["attn_implementation"] = "flash_attention_2"

        self.audio_samplerate = audio_samplerate
        
        try:
            # from transformers import AutoProcessor # should import here again
            from transformers import Qwen2AudioForConditionalGeneration
        except Exception as ex:
            logger.warning(f'Could not import dependencies for "{self.model_group}"')
            raise
        
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name_or_path, 
            **self.model_kwargs,
        )
        self.model = self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, 
            cache_dir=self.cache_dir,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
    

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
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

        # preprocess
        model_inputs_str = None
        if self.skip_chat_template:
            model_inputs_str = list()
            for _message_idx, _message in enumerate(messages):
                _query = ChatMessage.get_query(message=_message)
                model_inputs_str.append(f'{_query}')
            model_inputs_str = "\n\n".join(model_inputs_str)
        else:
            model_inputs_str = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            )

        audios = list()
        for message in messages:
            contents = message["content"]
            if isinstance(contents, dict):
                contents = [contents, ]
            for _content in contents:
                if "audio" not in _content["type"]: 
                    continue
                _audio_array, _sampling_rate, _audio_format = to_nparray_audio(
                    audio=_content["audio"],
                    sampling_rate=self.processor.feature_extractor.sampling_rate,
                )
                # audios.append((_audio_array, _sampling_rate))
                audios.append(_audio_array)

        model_inputs = self.processor(
            text=model_inputs_str, 
            audio=audios, 
            return_tensors="pt", 
            padding=True,
        ).to(device=self.model.device, dtype=self.torch_dtype)

        stopping_criteria = self._resolve_stopping_criteria(
            generation_options=generation_options,
            input_ids=model_inputs["input_ids"],
        )

        # inference: generate via hf_model
        generated = self.model.generate(
            **model_inputs,
            **generation_options,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )
        prompt_length = len(model_inputs.input_ids[0])
        return generated, prompt_length
    
    def generate_text(
        self,
        *args, 
        **kwargs,
    ):
        _generated, _prompt_length = self._generate(
            *args, 
            **kwargs,
        )
        generated_ids = _generated    
        generated_ids = generated_ids[0].detach().cpu()
        prediction_ids = generated_ids[_prompt_length:]
        
        # decode
        generated_text = self.processor.decode(
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
            generated_audios=None,
            generated_text=generated_text,
            temp_paths=None,
        )
        return output
     
    def compute_perplexity(
        self,
        messages: List[Dict[str, Any]],
        options: Optional[List[str]] = None,
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
        model_inputs_str = None
        if self.skip_chat_template:
            model_inputs_str = list()
            for _message_idx, _message in enumerate(messages):
                _query = ChatMessage.get_query(message=_message)
                model_inputs_str.append(f'{_query}')
            model_inputs_str = "\n\n".join(model_inputs_str)
        else:
            model_inputs_str = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
            )

        audios = list()
        for message in messages:
            contents = message["content"]
            if isinstance(contents, dict):
                contents = [contents, ]
            for _content in contents:
                if "audio" not in _content["type"]: 
                    continue
                _audio_array, _sampling_rate, _audio_format = to_nparray_audio(
                    audio=_content["audio"],
                    sampling_rate=self.processor.feature_extractor.sampling_rate,
                )
                audios.append((_audio_array, _sampling_rate))

        model_inputs = self.processor(
            text=model_inputs_str, 
            audios=audios, 
            return_tensors="pt", 
            padding=True,
        )
        model_inputs.input_ids = model_inputs.input_ids.to(device=self.model.device)

        perplexities = list()
        for _option in options:
            _model_inputs = copy.deepcopy(model_inputs)
            _option_input_ids = torch.tensor(
                self.tokenizer.encode(f'{_option}{self.tokenizer.eos_token}'),
                dtype=torch.long,
                device="cpu",
            ) # e.g. One<|im_end|>
            _num_tokens = len(_option_input_ids) - 1
            
            _input_ids = torch.cat([
                _model_inputs["input_ids"][0],
                _option_input_ids[:-1].to(device=_model_inputs["input_ids"].device),
            ], dim=0)
            _attention_mask = torch.cat([
                _model_inputs["attention_mask"][0],
                torch.ones(_option_input_ids[:-1].shape).to(device=_model_inputs["attention_mask"].device),
            ])
            _labels = -100 * torch.ones(_input_ids.shape).to(dtype=torch.long, device="cpu")
            _labels[-len(_option_input_ids):] = _option_input_ids
        
            _model_inputs["input_ids"] = _input_ids.unsqueeze(dim=0)
            _model_inputs["attention_mask"] = _attention_mask.unsqueeze(dim=0)
            _output = self.model(
                **_model_inputs,
                use_cache=use_cache, 
            )
            _logits = _output["logits"].detach().to(device="cpu")
            del _output
        
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