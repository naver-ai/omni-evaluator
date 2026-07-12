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
    BitsAndBytesConfig, GenerationConfig,
)
from typing import List, Tuple, Dict, Any, Union, Optional

from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes, to_nparray_video
from omni_evaluator.utils.torch import get_compute_capability


class XOmniModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=False,
        support_audio_generation=False,
        support_image_generation=True,
        support_text_generation=True,
        support_video_generation=False,
        support_compute_perplexity=False,
    ).to_dict()
    
    def __init__(
        self,
        flux_model_name_or_path: str = "black-forest-labs/FLUX.1-dev",
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.flux_model_name_or_path = flux_model_name_or_path
        
        # fallback to fp32 if fp16 is given
        if get_compute_capability() >= 8.0:
            self.model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            self.model_kwargs["torch_dtype"] = torch.float32

        self.model_kwargs["trust_remote_code"] = True
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path, 
            **self.model_kwargs,
        )
        self.model.init_vision(flux_model_name_or_path)
        self.model.set_generation_mode("text")
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, 
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            use_fast=True,
        )

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()
        self.model.set_generation_mode("text")
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
        
        # uncomment if error raise
        messages = self._to_x_omni_template(
            messages=messages,
        )
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt",
        ).to(device=self.model.device)
        do_sample = generation_options.get("do_sample", False)
        temperature = generation_options.get("temperature", 0.0)
        max_new_tokens = generation_options.get("max_new_tokens", 512)
        top_p = generation_options.get("top_p", None)
        num_beams = generation_options.get("num_beams", 1)
        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            num_beams=num_beams,
            use_cache=True, 
            eos_token_id=self.tokenizer.encode("<|im_end|>")[0],
            pad_token_id=0,
        )

        # inference: generate via hf_model
        generated_ids = self.model.generate(
            input_ids, 
            generation_config=generation_config,
        )
        prediction_ids = generated_ids[:, input_ids.shape[1]:-1]

        generated_ids = generated_ids[0]
        generated_text = self.tokenizer.decode(generated_ids)
        prediction, _ = self.model.mmdecode(
            self.tokenizer, 
            prediction_ids,
        )
        prediction_ids = prediction_ids[0]
        prediction = prediction[0]

        output = HuggingfaceInferenceOutput(
            prediction_ids=prediction_ids,
            prediction=prediction,
            generated_audios=None,
            generated_images=None,
            generated_text=generated_text,
            generated_videos=None,
            generated_ids=generated_ids,
            prompt=None,
            temp_paths=None,
        )
        return output
    
    def generate_image(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        image_size: int = 1152,
        downsample_size: int = 16,
        cfg_scale: float = 1.0,
        min_p: float = 0.03,
        seed: int = 1234,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()
        self.model.set_generation_mode("image")
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
        _user_messages = ChatMessage.get_user_messages(messages=messages)
        prompt = ChatMessage.get_query(message=_user_messages[-1])
        prompt = ChatMessage.get_prompt(message=_user_messages[-1])

        # image_size: (width, height)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif len(image_size) == 1:
            image_size = (image_size[0], image_size[1])

        token_width = image_size[0] // downsample_size
        token_height = image_size[1] // downsample_size
        image_prefix = f'<SOM>{token_height} {token_width}<IMAGE>'
        
        temperature = generation_options.get("temperature", 1.0)
        top_p = generation_options.get("top_p", 1.0)
        cfg_scale = generation_options.get("cfg_scale", 1.0)
        min_p = generation_options.get("min_p", 0.03)
        seed = generation_options.get("seed", seed)
        suppress_tokens = self.tokenizer.convert_tokens_to_ids(self.model.config.mm_special_tokens)
        generation_config = GenerationConfig(
            max_new_tokens=token_height * token_width,
            do_sample=True,
            temperature=temperature,
            min_p=min_p,
            top_p=top_p,
            guidance_scale=cfg_scale,
            suppress_tokens=suppress_tokens,
        )

        # Sample inputs:
        model_inputs = self.tokenizer(
            [prompt+ image_prefix],
            return_tensors="pt", 
            padding="longest", 
            padding_side="left",
        )
        negative_ids = self.tokenizer.encode(
            image_prefix, 
            add_special_tokens=False, 
            return_tensors="pt",
        ).expand(1, -1).to(device=self.model.device)

        # inference: generate via hf_model
        torch.manual_seed(seed)
        generated_ids = self.model.generate(
            inputs=model_inputs.input_ids.to(device=self.model.device), 
            attention_mask=model_inputs.attention_mask.to(device=self.model.device),
            generation_config=generation_config,
            negative_prompt_ids=negative_ids,
        )
        generated_ids = torch.nn.functional.pad(
            input=generated_ids, 
            pad=(0, 1), 
            value=self.tokenizer.convert_tokens_to_ids("<EOM>"),
        )
        
        torch.manual_seed(seed)
        generated_ids = generated_ids[0]
        _, generated_images = self.model.mmdecode(self.tokenizer, generated_ids, skip_special_tokens=False)
        prediction = generated_images

        output = HuggingfaceInferenceOutput(
            prediction_ids=None,
            prediction=prediction,
            generated_audios=None,
            generated_images=generated_images,
            generated_text=None,
            generated_videos=None,
            generated_ids=None,
            prompt=prompt,
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

        # preprocess # conversation
        # do not need to implement skip_chat_template since Janus has not chat_template
        janus_messages = self._to_janus_template(
            messages=messages,
        )
        images = list()
        for _message in janus_messages:
            if "images" in _message:
                for _image in _message["images"]:
                    _image = to_pil_image(image=_image)
                    images.append(_image)

        model_inputs = self.processor(
            conversations=janus_messages, 
            images=images, 
            force_batchify=True,
        ).to(device=self.model.device, dtype=self.model.dtype)
        
        perplexities = list()
        for _option in options:
            _images = list()
            _janus_messages = self._to_janus_template(
                messages=messages,
                assistant_content=_option,
            )
            for _message in _janus_messages:
                if "images" in _message:
                    for _image in _message["images"]:
                        _image = to_pil_image(image=_image)
                        _images.append(_image)

            _model_inputs = self.processor(
                conversations=_janus_messages,
                images=_images,
                force_batchify=True,
            ).to(device=self.model.device, dtype=self.model.dtype)
            _input_ids = _model_inputs["input_ids"].squeeze(0)
            _option_input_ids = self.processor.tokenizer.encode(
                _option, add_special_tokens=False, return_tensors="pt",
            ).squeeze(0)
            _num_tokens = len(_option_input_ids)
            _labels = -100 * torch.ones(_input_ids.shape).to(dtype=torch.long, device="cpu")
            _labels[-len(_option_input_ids):] = _option_input_ids

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
    
    def _to_x_omni_template(
        self,
        messages: List[Union[Dict[str, Any], ChatMessage]],
    ):
        query = ""
        for _message_idx, _message in enumerate(messages):
            if _message["role"] != "user":
                continue
            for _content in _message["content"]:
                if _content["type"] == "image":
                    _image = to_pil_image(image=_content["image"])
                    _image_query = self.model.tokenize_image(_image)
                    query = f'{_image_query}\n{query}'
                elif _content["type"] == "text":
                    query = f'{query}{_content["text"]}'
                    
        output = [
            {
                "role": "user",
                "content": query,
            }
        ]
        return output