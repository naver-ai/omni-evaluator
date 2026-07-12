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
from transformers import (
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel,
    BitsAndBytesConfig,
)
from typing import List, Tuple, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)

from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes
from omni_evaluator.utils.optional_import import require_package
from omni_evaluator.utils.torch import get_compute_capability


class Qwen3VlModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=True,
        support_text_generation=True,
        support_compute_perplexity=True,
    ).to_dict()
    
    def __init__(
        self,
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 2048 * 28 * 28,
        use_audio_in_video: Optional[bool] = False,
        *args, **kwargs,
    ):
        require_package("qwen_vl_utils", extras="qwen-vl-utils", feature="Qwen3-VL")
        from qwen_vl_utils import process_vision_info
        self._process_vision_info = process_vision_info

        super().__init__(*args, **kwargs)

        if get_compute_capability() >= 8.0:
            self.model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            self.model_kwargs["torch_dtype"] = torch.float16
        
        if not isinstance(min_pixels, int):
            min_pixels = 256 * 28 * 28
        if not isinstance(max_pixels, int):
            max_pixels = 2048 * 28 * 28
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.min_visual_tokens = min_pixels // (28 * 28) # 256
        self.max_visual_tokens = max_pixels // (28 * 28) # 2048
        if not isinstance(self.max_video_frames, int):
            self.max_video_frames = 128
        if not isinstance(self.fps, (int, float)):
            self.fps = 2.0
        self.use_audio_in_video = use_audio_in_video
        
        try:
            # from transformers import AutoProcessor # should import here again
            from transformers import Qwen3VLForConditionalGeneration
        except Exception as ex:
            logger.warning(f'Could not import dependencies for "{self.model_group}"')
            raise
        
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path, 
            **self.model_kwargs,
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path, 
            cache_dir=self.cache_dir,
            min_pixels=self.min_pixels, 
            max_pixels=self.max_pixels,
            fps=self.fps,
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
        messages = [
            ChatMessage.preprocess_message(
                message=_message,
                remove_image=not self.ENGINE_FEATURES["support_image_understanding"],
                content_fields_image={
                    "min_pixels": kwargs.get("min_pixels", self.min_pixels),
                    "max_pixels": kwargs.get("max_pixels", self.max_pixels),
                },
                remove_video=not self.ENGINE_FEATURES["support_video_understanding"],
                content_fields_video={
                    "min_pixels": kwargs.get("min_pixels", self.min_pixels),
                    "max_pixels": kwargs.get("max_pixels", self.max_pixels),
                },
                remove_audio=not self.ENGINE_FEATURES["support_audio_understanding"],
                content_fields_audio=None,
            )
            for _message in messages
        ]

        if "omni" in self.model_name_or_path:
            generation_options["use_audio_in_video"] = False
            generation_options["return_audio"] = False

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
        _model_inputs_image, _model_inputs_video = self._process_vision_info(messages)

        model_inputs = self.processor(
            text=[model_inputs_str, ],
            images=_model_inputs_image,
            videos=_model_inputs_video,
            padding=True,
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

    def batched_generate_text(
        self,
        messages_list: List[List[Dict[str, Any]]],
        generation_options: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ) -> List[HuggingfaceInferenceOutput]:
        """Batched text generation for TEXT-ONLY inputs: chat string via the processor,
        left-pad + a single ``model.generate``. Any conversation with media (or
        skip_chat_template) falls back to the per-item base loop, which routes through
        generate_text and its vision processing."""
        if not messages_list:
            return []
        if self._messages_list_has_media(messages_list) or self.skip_chat_template:
            return super().batched_generate_text(
                messages_list, generation_options=generation_options, use_cache=use_cache, **kwargs)
        if generation_options is None:
            generation_options = dict()
        generation_options = dict(generation_options)
        _stop_words = generation_options.get("stop_words")
        if "omni" in self.model_name_or_path:
            generation_options["use_audio_in_video"] = False
            generation_options["return_audio"] = False

        model_inputs_strs = list()
        for _messages in messages_list:
            _messages = [
                ChatMessage.preprocess_message(
                    message=_message,
                    remove_image=not self.ENGINE_FEATURES["support_image_understanding"],
                    content_fields_image={"min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                    remove_video=not self.ENGINE_FEATURES["support_video_understanding"],
                    content_fields_video={"min_pixels": self.min_pixels, "max_pixels": self.max_pixels},
                    remove_audio=not self.ENGINE_FEATURES["support_audio_understanding"],
                    content_fields_audio=None,
                )
                for _message in _messages
            ]
            model_inputs_strs.append(self.processor.apply_chat_template(
                _messages, tokenize=False, add_generation_prompt=True))

        # text-only: encode the templated strings with the tokenizer, left-padded
        _tokenizer = self.tokenizer if self.tokenizer is not None else getattr(self.processor, "tokenizer", None)
        _prev_side = getattr(_tokenizer, "padding_side", "right")
        _tokenizer.padding_side = "left"
        model_inputs = _tokenizer(
            model_inputs_strs, return_tensors="pt", padding=True,
        ).to(device=self.model.device)
        _tokenizer.padding_side = _prev_side

        stopping_criteria = self._resolve_stopping_criteria(
            generation_options=generation_options,
            input_ids=model_inputs["input_ids"],
        )
        generated_ids = self.model.generate(
            **model_inputs,
            **generation_options,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )
        prompt_length = model_inputs["input_ids"].shape[1]   # uniform after left-pad

        outputs = list()
        for _idx in range(len(model_inputs_strs)):
            _generated_ids = generated_ids[_idx].detach().cpu()
            _prediction_ids = _generated_ids[prompt_length:]
            outputs.append(HuggingfaceInferenceOutput(
                prediction_ids=_prediction_ids,
                prediction=self._apply_generation_postprocess(self.processor.decode(
                    _prediction_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False), _stop_words),
                generated_ids=_generated_ids,
                generated_text=self.processor.decode(
                    _generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                temp_paths=None,
            ))
        return outputs

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
                content_fields_image={
                    "min_pixels": kwargs.get("min_pixels", self.min_pixels),
                    "max_pixels": kwargs.get("max_pixels", self.max_pixels),
                },
                remove_video=not self.ENGINE_FEATURES["support_video_understanding"],
                content_fields_video={
                    "min_pixels": kwargs.get("min_pixels", self.min_pixels),
                    "max_pixels": kwargs.get("max_pixels", self.max_pixels),
                },
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
        _model_inputs_image, _model_inputs_video = self._process_vision_info(messages)

        model_inputs = self.processor(
            text=[model_inputs_str, ],
            images=_model_inputs_image,
            videos=_model_inputs_video,
            padding=True,
            return_tensors="pt",
        ).to(device=self.model.device)

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