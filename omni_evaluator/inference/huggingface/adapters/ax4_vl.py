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
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
except Exception as ex:
    logger.warning('Could not import dependencies for `ax4_vl`')

from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import InferenceEngineFeatures, HuggingfaceInferenceOutput
from omni_evaluator.utils.multimodal import to_nparray_video, to_pil_image
from omni_evaluator.utils.torch import get_compute_capability


class Ax4VlModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_text_understanding=True,
        support_image_understanding=True,
        support_video_understanding=True,
        support_audio_understanding=False,
        support_text_generation=True,
        support_compute_perplexity=True,
        support_image_generation=False,
        support_audio_generation=False,
    ).to_dict()

    def __init__(
        self,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # trust_remote_code is required for this model
        self.model_kwargs["trust_remote_code"] = True

        if get_compute_capability() >= 8.0:
            self.model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **self.model_kwargs,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

    def _extract_pil_images_and_build_content(
        self,
        messages: List[Dict[str, Any]],
    ):
        """Extract PIL images from messages and rebuild content with {"type": "image"} placeholders.
        Video inputs are decomposed into frames via to_nparray_video and treated as images.
        """
        rebuilt_messages = []
        all_images = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", [])

            if isinstance(content, str):
                rebuilt_messages.append({"role": role, "content": content})
                continue

            new_content = []
            for item in content:
                item_type = item.get("type", "")
                if item_type == "image":
                    pil = to_pil_image(item.get("image"))
                    all_images.append(pil)
                    new_content.append({"type": "image"})
                elif item_type == "video":
                    frames_np, _, _ = to_nparray_video(
                        video=item.get("video"),
                        max_frames=self.max_video_frames or 16,
                    )
                    for frame in frames_np:
                        all_images.append(to_pil_image(frame))
                        new_content.append({"type": "image"})
                elif item_type == "text":
                    new_content.append({"type": "text", "text": item.get("text", "")})

            rebuilt_messages.append({"role": role, "content": new_content})

        return rebuilt_messages, all_images

    def _build_model_inputs(
        self,
        messages: List[Dict[str, Any]],
    ):
        """Common pipeline: preprocess messages → extract images → build processor input tensors."""
        messages = [
            ChatMessage.preprocess_message(
                message=_message,
                remove_image=not self.ENGINE_FEATURES["support_image_understanding"],
                remove_video=not self.ENGINE_FEATURES["support_video_understanding"],
                remove_audio=True,
            )
            for _message in messages
        ]

        rebuilt_messages, pil_images = self._extract_pil_images_and_build_content(messages)

        text = self.processor.apply_chat_template(
            rebuilt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.processor(
            images=pil_images if pil_images else None,
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(device=self.model.device)

        return model_inputs

    def generate_text(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()

        stop_words = generation_options.pop("stop_words", None)
        model_inputs = self._build_model_inputs(messages)

        stopping_criteria = None
        if (
            self.stopping_criteria is not None
            and isinstance(stop_words, (list, tuple))
            and len(stop_words) > 0
        ):
            stopping_criteria = self.get_stopping_criteria(
                stop_words=stop_words,
                eos_token=self.eos_token,
                input_ids=model_inputs["input_ids"],
            )

        # use greedy decoding when temperature is 0
        temperature = generation_options.get("temperature", None)
        if temperature is not None:
            if temperature > 0:
                generation_options["do_sample"] = True
            else:
                generation_options.pop("temperature")
                generation_options.pop("top_p", None)
                generation_options["do_sample"] = False

        generated_ids = self.model.generate(
            **model_inputs,
            **generation_options,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
        )[0].detach().cpu()

        prediction_ids = generated_ids[len(model_inputs.input_ids[0]):]
        generation = self.processor.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prediction = self.processor.decode(
            prediction_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return HuggingfaceInferenceOutput(
            prediction_ids=prediction_ids,
            prediction=prediction,
            generated_ids=generated_ids,
            generated_text=generation,
        )

    def compute_perplexity(
        self,
        messages: List[Dict[str, Any]],
        options: Optional[List[str]] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        model_inputs = self._build_model_inputs(messages)

        perplexities = []
        for _option in options:
            _model_inputs = copy.deepcopy(model_inputs)
            _option_input_ids = torch.tensor(
                self.tokenizer.encode(f'{_option}{self.tokenizer.eos_token}'),
                dtype=torch.long,
                device="cpu",
            )
            _num_tokens = len(_option_input_ids) - 1

            _input_ids = torch.cat([
                _model_inputs["input_ids"][0],
                _option_input_ids[:-1].to(device=_model_inputs["input_ids"].device),
            ], dim=0)
            _attention_mask = torch.cat([
                _model_inputs["attention_mask"][0],
                torch.ones(_option_input_ids[:-1].shape).to(device=_model_inputs["attention_mask"].device),
            ])
            _labels = -100 * torch.ones(_input_ids.shape, dtype=torch.long, device="cpu")
            _labels[-len(_option_input_ids):] = _option_input_ids

            _model_inputs["input_ids"] = _input_ids.unsqueeze(0)
            _model_inputs["attention_mask"] = _attention_mask.unsqueeze(0)

            _output = self.model(**_model_inputs, use_cache=use_cache)
            _logits = _output["logits"].detach().to(device="cpu")
            del _output

            _nll = F.cross_entropy(
                _logits.view(-1, _logits.shape[-1]),
                _labels,
                ignore_index=-100,
                reduction="sum",
            )
            _nll = (_nll / _num_tokens).item()
            perplexities.append(_nll)

        prediction = np.argmin(perplexities)
        return HuggingfaceInferenceOutput(
            prediction=prediction,
            perplexities=perplexities,
        )
