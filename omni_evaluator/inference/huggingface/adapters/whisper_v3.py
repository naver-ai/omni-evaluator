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
    AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoModel,
    BitsAndBytesConfig,
)
from typing import List, Tuple, Dict, Any, Union, Optional

from omni_evaluator import AudioFormat
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.inference import (
    InferenceEngineFeatures, HuggingfaceInferenceOutput,
)
from omni_evaluator.utils.multimodal import (
    to_pil_image, to_image_bytes,
    to_nparray_audio, to_audio_bytes,
)
from omni_evaluator.utils.torch import get_compute_capability


class WhisperV3Module(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=True,
        support_image_understanding=False,
        support_text_understanding=False,
        support_video_understanding=False,
        support_audio_generation=False,
        support_text_generation=True,
        support_image_generation=False,
        support_video_generation=False,
        support_compute_perplexity=False,
    ).to_dict()
    
    def __init__(
        self,
        use_safetensors: Optional[bool] = True, 
        variant: Optional[str] = "fp16",
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.default_generation_options = {
            # "max_new_tokens": 448,
            "num_beams": 1,
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "logprob_threshold": -1.0,
            # "no_speech_threshold": 0.6,
            "return_timestamps": True,
        }

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name_or_path,
            **self.model_kwargs,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        return_audio: Optional[bool] = None,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()
        messages = [
            ChatMessage.preprocess_message(
                message=_message,
                remove_audio=not self.ENGINE_FEATURES["support_audio_understanding"],
                content_fields_audio=None,
                remove_image=not self.ENGINE_FEATURES["support_image_understanding"],
                content_fields_image=None,
                allowed_audio_format=[AudioFormat.BYTES],
                remove_text=not self.ENGINE_FEATURES["support_text_understanding"],
                content_fields_text=None,
                remove_video=not self.ENGINE_FEATURES["support_video_understanding"],
                content_fields_video=None,
            )
            for _message in messages
        ]
        
        _audio_bytes = messages[-1]["content"][-1]["audio"]
        _audio_array, _sampling_rate, _audio_format = to_nparray_audio(audio=_audio_bytes)
        
        model_inputs = self.processor(
            _audio_array,
            sampling_rate=_sampling_rate,
            return_tensors="pt",
            # truncation=False,
            # padding="longest",
            return_attention_mask=True,
        ).to(device=self.model.device, dtype=self.torch_dtype)

        for _k, _v in self.default_generation_options.items():
            if _k not in generation_options:
                generation_options[_k] = _v

        generated_ids = self.model.generate(
            **model_inputs,
            **generation_options,
            use_cache=use_cache,
        )[0].detach().cpu()
        prediction_ids = generated_ids
        
        # decode
        generated_text = self.tokenizer.decode(
            generated_ids, 
            skip_special_tokens=False, 
            decode_with_timestamps=False,
        )
        prediction = self.tokenizer.decode(
            generated_ids, 
            skip_special_tokens=True, 
            decode_with_timestamps=False,
        ).strip()
        
        output = HuggingfaceInferenceOutput(
            prediction_ids=prediction_ids,
            prediction=prediction,
            generated_ids=generated_ids,
            generated_text=generated_text,
            generated_images=None,
            temp_paths=None,
        )
        return output