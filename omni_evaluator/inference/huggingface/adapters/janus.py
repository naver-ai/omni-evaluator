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

logger = logging.getLogger(__name__)


class JanusModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=False,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=False,
        support_audio_generation=False,
        support_image_generation=True,
        support_text_generation=True,
        support_video_generation=False,
        support_compute_perplexity=True,
    ).to_dict()
    AUDIO_PLACEHOLDER = "<audio_placeholder>\n"
    IMAGE_PLACEHOLDER = "<image_placeholder>\n"
    VIDEO_PLACEHOLDER = "<video_placeholder>\n"
    
    def __init__(
        self,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        try:
            # from transformers import AutoProcessor # should import here again
            from janus.models import MultiModalityCausalLM, VLChatProcessor
        except Exception as ex:
            logger.warning('Could not import dependencies for `janus`')
            raise

        # fallback to fp32 if fp16 is given
        if get_compute_capability() >= 8.0:
            self.model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            self.model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **self.model_kwargs,
        ).eval()
        self.processor = VLChatProcessor.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = self.processor.tokenizer
    

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
        # do not need to implement skip_chat_template since Janus has not chat_template
        images = list()
        janus_messages = self._to_janus_template(
            messages=messages,
        )
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
        prompt = model_inputs['sft_format'][0]

        stopping_criteria = self._resolve_stopping_criteria(
            generation_options=generation_options,
            input_ids=model_inputs["input_ids"],
        )

        # inference: generate via hf_model
        if "max_new_tokens" not in generation_options:
            generation_options["max_new_tokens"] = self.model.language_model.config.max_position_embeddings
        inputs_embeds = self.model.prepare_inputs_embeds(**model_inputs)
        generated_ids = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=model_inputs.attention_mask,
            **generation_options,
            stopping_criteria=stopping_criteria,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id, # use eos_token_id instead of pad_token_id
            use_cache=use_cache,
        )
        generated_ids = generated_ids[0].detach().cpu()
        prediction_ids = generated_ids
        # decode
        generated_text = self.tokenizer.decode(
            generated_ids, 
            skip_special_tokens=False, 
            clean_up_tokenization_spaces=False
        )
        prediction = self.tokenizer.decode(
            prediction_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

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
        parallel_size: int = 1,
        cfg_weight: float = 5,
        image_token_num_per_image: int = 576,
        img_size: int = 384,
        patch_size: int = 16,
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
        # do not need to implement skip_chat_template since Janus has not chat_template
        images = list()
        janus_messages = self._to_janus_template(
            messages=messages,
        )
        for _message in janus_messages:
            if "images" in _message:
                for _image in _message["images"]:
                    _image = to_pil_image(image=_image)
                    images.append(_image)

        prompt = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=janus_messages,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt = f'{prompt}{self.processor.image_start_tag}'
        
        input_ids = torch.tensor(
            self.tokenizer.encode(prompt), 
            dtype=torch.long,
        )
        tokens = torch.zeros(
            (parallel_size*2, len(input_ids)), 
            dtype=torch.int,
        ).to(device=self.model.device)
        for _idx in range(0, parallel_size*2):
            tokens[_idx, :] = input_ids
            if _idx % 2 != 0:
                tokens[_idx, 1:-1] = self.processor.pad_id
        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)

        generated_tokens = torch.zeros(
            (parallel_size, image_token_num_per_image), 
            dtype=torch.int,
        ).to(device=self.model.device)

        temperature = generation_options.get("temperature", 1.0)
        for _idx in range(0, image_token_num_per_image):
            _outputs = self.model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=_outputs.past_key_values if _idx != 0 else None)
            _hidden_states = _outputs.last_hidden_state
            _logits = self.model.gen_head(_hidden_states[:, -1, :])
            _logit_cond = _logits[0::2, :]
            _logit_uncond = _logits[1::2, :]
            _logits = _logit_uncond + cfg_weight * (_logit_cond-_logit_uncond)
            _probs = torch.softmax(_logits / temperature, dim=-1)
            _next_token = torch.multinomial(_probs, num_samples=1)
            generated_tokens[:, _idx] = _next_token.squeeze(dim=-1)
            _next_token = torch.cat([_next_token.unsqueeze(dim=1), _next_token.unsqueeze(dim=1)], dim=1).view(-1)
            _image_embeds = self.model.prepare_gen_img_embeds(_next_token)
            inputs_embeds = _image_embeds.unsqueeze(dim=1)

        _decoded_code = self.model.gen_vision_model.decode_code(
            generated_tokens.to(dtype=torch.int), 
            shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size],
        )
        _decoded_code = _decoded_code.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
        _decoded_code = np.clip((_decoded_code + 1) / 2 * 255, 0, 255)
        _image_array = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
        _image_array[:, :, :] = _decoded_code

        generated_images = list()
        for _idx in range(0, parallel_size):
            _image = PIL.Image.fromarray(_image_array[_idx])
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
            
            _num_tokens = _model_inputs["input_ids"].shape[1] - model_inputs["input_ids"].shape[1] - 1
            _labels = -100 * torch.ones(_model_inputs["input_ids"].shape[1]).to(dtype=torch.long, device="cpu")
            _labels[-(_num_tokens + 1):] = _model_inputs["input_ids"][0, -(_num_tokens + 1):]
            
            _inputs_embeds = self.model.prepare_inputs_embeds(**_model_inputs)
            _output = self.model.language_model(
                inputs_embeds=_inputs_embeds,
                attention_mask=_model_inputs.attention_mask,
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
                _role = "System"
            elif _message["role"] == "assistant":
                _role = "Assistant"
            elif _message["role"] == "user":
                _role = "User"
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
                _query = f'{cls.AUDIO_PLACEHOLDER * len(_audios)}{_query}'
            if len(_images) > 0:
                _query = f'{cls.IMAGE_PLACEHOLDER * len(_images)}{_query}'
            if len(_videos) > 0:
                _query = f'{cls.VIDEO_PLACEHOLDER * len(_videos)}{_query}'
                
            conversation.append({
                "role": _role,
                "content": _query,
                # "audios": _audios,
                "images": _images,
                # "videos": _videos,
            })
            
        if _last_role != "assistant":
            if not assistant_content:
                assistant_content = ""
            conversation.append({
                "role": "Assistant",
                "content": assistant_content,
            })
        return conversation