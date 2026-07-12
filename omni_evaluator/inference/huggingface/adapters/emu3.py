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
from PIL import Image
import soundfile
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoProcessor, AutoImageProcessor, AutoModelForCausalLM, AutoModel,
    BitsAndBytesConfig, GenerationConfig,
)
from transformers.generation import (
    LogitsProcessorList, PrefixConstrainedLogitsProcessor, 
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from typing import List, Tuple, Dict, Any, Union, Optional

from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes
from omni_evaluator.utils.torch import get_compute_capability

logger = logging.getLogger(__name__)


class Emu3Module(HuggingfaceModule):
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
        vq_model_name_or_path: str = "BAAI/Emu3-VisionTokenizer",
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        try:
            # from transformers import AutoProcessor # should import here again
            from emu3.mllm.processing_emu3 import Emu3Processor
        except Exception as ex:
            logger.warning('Could not import dependencies for `emu3`')
            raise

        self.vq_model_name_or_path = vq_model_name_or_path

        # fallback to fp32 if fp16 is given
        if get_compute_capability() >= 8.0:
            self.model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            self.model_kwargs["torch_dtype"] = torch.float32

        trust_remote_code = True
        self.model_kwargs["trust_remote_code"] = trust_remote_code

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **self.model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=self.cache_dir,
            padding_side="left",
        )
        
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.vq_model_name_or_path, 
            trust_remote_code=trust_remote_code,
            cache_dir=self.cache_dir,
        )
        self.image_tokenizer = AutoModel.from_pretrained(
            self.vq_model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=self.cache_dir,
        ).eval()
        self.processor = Emu3Processor(self.image_processor, self.image_tokenizer, self.tokenizer)
    

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
                content_fields_image=None,
                remove_video=not self.ENGINE_FEATURES["support_video_understanding"],
                content_fields_video=None,
                remove_audio=not self.ENGINE_FEATURES["support_audio_understanding"],
                content_fields_audio=None,
            )
            for _message in messages
        ]

        # preprocess # conversation
        _user_messages = ChatMessage.get_user_messages(messages=messages)
        prompt = ChatMessage.get_query(message=_user_messages[-1])
        images = list()
        for _message in messages:
            if _message["role"] != "user":
                continue
            for _content in _message["content"]:
                if _content["type"] == "image":
                    _image = _content["image"]
                    _image = to_pil_image(image=_image)
                    images.append(_image)

        model_inputs = self.processor(
            text=prompt,
            image=images[0],
            mode="U", # understanding
            return_tensors="pt",
            padding="longest",
        )

        max_new_tokens = generation_options.get("max_new_tokens", 40960)
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # inference: generate via hf_model
        # set use_cache=False since Emu3 raise error
        # transformers_modules/BAAI/Emu3_hyphen_Chat/d9cb6ffe11d3f62b73d08d40672c243e77344d51/modeling_emu3.py", line 1286, in prepare_inputs_for_generation
        #     past_length = past_key_values.seen_tokens
        # AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids.to(device=self.model.device),
            attention_mask=model_inputs.attention_mask.to(device=self.model.device),
            generation_config=generation_config,
            use_cache=use_cache,
        )
        generated_ids = generated_ids[0]
        prediction_ids = generated_ids[model_inputs.input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
        ).strip()
        prediction = self.tokenizer.decode(
            prediction_ids,
            skip_special_tokens=True,
        ).strip()

        output = HuggingfaceInferenceOutput(
            prediction_ids=prediction_ids,
            prediction=prediction,
            generated_audios=None,
            generated_images=None,
            generated_text=generated_text,
            generated_videos=None,
            generated_ids=generated_ids,
            prompt=prompt,
            temp_paths=None,
        )
        return output
    
    def generate_image(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        positive_prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        classifier_free_guidance: float = 3.0,
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

        # preprocess # conversation
        _user_messages = ChatMessage.get_user_messages(messages=messages)
        prompt = ChatMessage.get_query(message=_user_messages[-1])
        if not positive_prompt:
            positive_prompt = ""
        if not negative_prompt:
            negative_prompt = ""

        prompt = f'{prompt}{positive_prompt}'
        _processor_kwargs = dict(
            mode="G", # generation
            ratio="1:1",
            image_area=self.model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )
        positive_model_inputs = self.processor(
            text=prompt, 
            **_processor_kwargs,
        )
        negative_model_inputs = self.processor(
            text=negative_prompt, 
            **_processor_kwargs,
        )

        # prepare hyper parameters
        do_sample = generation_options.get("do_sample", True)
        max_new_tokens = generation_options.get("max_new_tokens", 40960)
        top_k = generation_options.get("top_k", 2048)
        num_beams = generation_options.get("num_beams", 1)
        generation_config = GenerationConfig(
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
        )

        _constrained_fn = self.processor.build_prefix_constrained_fn(
            positive_model_inputs.image_size[:, 0], # height
            positive_model_inputs.image_size[:, 1], # width
        )
        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                classifier_free_guidance,
                self.model,
                unconditional_ids=negative_model_inputs.input_ids.to(device=self.model.device),
            ),
            PrefixConstrainedLogitsProcessor(
                _constrained_fn ,
                num_beams=num_beams,
            ),
        ])

        # inference: generate via hf_model
        generated_ids = self.model.generate(
            input_ids=positive_model_inputs.input_ids.to(device=self.model.device),
            attention_mask=positive_model_inputs.attention_mask.to(device=self.model.device),
            generation_config=generation_config,
            logits_processor=logits_processor,
            
        )
        generated_ids = generated_ids[0]
        _generated_images = self.processor.decode(generated_ids)
        generated_images = list()
        for _idx, _image in enumerate(_generated_images):
            if not isinstance(_image, PIL.Image.Image):
                continue
            generated_images.append(_image)
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
                _query = f'{[cls.AUDIO_PLACEHOLDER, ] * len(_audios)}{_query}'
            if len(_images) > 0:
                _query = f'{[cls.IMAGE_PLACEHOLDER, ] * len(_images)}{_query}'
            if len(_videos) > 0:
                _query = f'{[cls.VIDEO_PLACEHOLDER, ] * len(_videos)}{_query}'
                
            conversation.append({
                "role": _role,
                "content": _query,
                # "_audios": _audios,
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