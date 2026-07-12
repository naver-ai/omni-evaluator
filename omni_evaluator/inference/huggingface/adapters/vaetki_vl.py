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
import functools
import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
except Exception as ex:
    logger.warning('Could not import dependencies for `vaetki_vl`')

_transformers_compat_patched = False


def _apply_transformers_compat_patches():
    # Idempotent compat shims for transformers < 4.55 used by VAETKI remote code.
    # Deferred to first VAETKI instantiation so importing this module does not
    # mutate transformers global state for unrelated models.
    global _transformers_compat_patched
    if _transformers_compat_patched:
        return

    # TransformersKwargs was introduced in transformers 4.55+.
    import transformers.utils as _tu
    if not hasattr(_tu, "TransformersKwargs"):
        from typing import TypedDict
        class _TransformersKwargs(TypedDict, total=False):
            pass
        _tu.TransformersKwargs = _TransformersKwargs

    # VAETKI passes position_ids to create_causal_mask, but that kwarg does not
    # exist in transformers 4.52.4 (added in 4.55+).
    import transformers.masking_utils as _mu
    _orig_create_causal = _mu.create_causal_mask
    _orig_create_sliding = _mu.create_sliding_window_causal_mask

    @functools.wraps(_orig_create_causal)
    def _patched_create_causal(*args, **kwargs):
        kwargs.pop("position_ids", None)
        return _orig_create_causal(*args, **kwargs)

    @functools.wraps(_orig_create_sliding)
    def _patched_create_sliding(*args, **kwargs):
        kwargs.pop("position_ids", None)
        return _orig_create_sliding(*args, **kwargs)

    _mu.create_causal_mask = _patched_create_causal
    _mu.create_sliding_window_causal_mask = _patched_create_sliding

    _transformers_compat_patched = True


from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import InferenceEngineFeatures, HuggingfaceInferenceOutput
from omni_evaluator.utils.multimodal import to_nparray_video, to_pil_image
from omni_evaluator.utils.optional_import import require_package
from omni_evaluator.utils.torch import get_compute_capability


class VaetkiVlModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_text_understanding=True,
        support_image_understanding=True,
        support_video_understanding=True,
        support_audio_understanding=False,
        support_text_generation=True,
        support_compute_perplexity=True,   # AutoModelForCausalLM + standard HF inputs
        support_image_generation=False,
        support_audio_generation=False,
    ).to_dict()

    def __init__(
        self,
        min_pixels: Optional[int] = 256 * 28 * 28,
        max_pixels: Optional[int] = 1605632,
        video_max_pixels: Optional[int] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        *args, **kwargs,
    ):
        require_package("qwen_vl_utils", extras="qwen-vl-utils", feature="VAETKI-VL")
        _apply_transformers_compat_patches()
        HuggingfaceModule.__init__(self, *args, **kwargs)

        self.model_kwargs["trust_remote_code"] = True
        self.model_kwargs["attn_implementation"] = "eager"

        # VAETKI remote code looks up ALL_ATTENTION_FUNCTIONS["eager"] directly,
        # so it must always be registered regardless of attn_implementation.
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        if "eager" not in ALL_ATTENTION_FUNCTIONS._global_mapping:
            from transformers.models.altclip.modeling_altclip import eager_attention_forward
            ALL_ATTENTION_FUNCTIONS._global_mapping["eager"] = eager_attention_forward

        if get_compute_capability() >= 8.0:
            self.model_kwargs["attn_implementation"] = "flash_attention_2"

            # Monkey-patch: VAETKI-VL has a typo in _supports_flash_attn that causes FA2
            # validation to fail, even though RiceFlashAttention2 calls flash_attn_varlen_func
            # directly and works fine. Bypass only the validation, then restore after load.
            # The method name differs by transformers version:
            #   < 4.55: _check_and_enable_flash_attn_2 (classmethod, returns config)
            #   >= 4.55: get_correct_attn_implementation (instance method, returns str)
            import transformers.modeling_utils as _modeling_utils
            _ptm = _modeling_utils.PreTrainedModel
            _fa2_patch_attr = None

            if hasattr(_ptm, '_check_and_enable_flash_attn_2'):
                _fa2_patch_attr = '_check_and_enable_flash_attn_2'
                _fa2_orig = getattr(_ptm, _fa2_patch_attr)

                @classmethod
                def _force_fa2(cls, config, *args, **kwargs):
                    try:
                        return _fa2_orig.__func__(cls, config, *args, **kwargs)
                    except ValueError:
                        config._attn_implementation = "flash_attention_2"
                        return config

                setattr(_ptm, _fa2_patch_attr, _force_fa2)

            elif hasattr(_ptm, 'get_correct_attn_implementation'):
                _fa2_patch_attr = 'get_correct_attn_implementation'
                _fa2_orig = getattr(_ptm, _fa2_patch_attr)

                def _force_fa2(self, requested_attention, is_init_check=False):
                    try:
                        return _fa2_orig(self, requested_attention, is_init_check)
                    except ValueError:
                        return "flash_attention_2"

                setattr(_ptm, _fa2_patch_attr, _force_fa2)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            **self.model_kwargs,
        ).eval()

        if get_compute_capability() >= 8.0 and _fa2_patch_attr is not None:
            # Restore the original method so other model loads are not affected.
            setattr(_ptm, _fa2_patch_attr, _fa2_orig)

        if not isinstance(min_pixels, int):
            min_pixels = 256 * 28 * 28
        if not isinstance(max_pixels, int):
            max_pixels = 1605632
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # Per-frame pixel budget: keep total vision tokens equivalent to the 16-frame baseline.
        _max_frames = self.max_video_frames or 32
        if video_max_pixels is not None:
            self.video_max_pixels = video_max_pixels
        elif _max_frames > 16:
            self.video_max_pixels = max(int(self.max_pixels * 16 / _max_frames), self.min_pixels)
        else:
            self.video_max_pixels = self.max_pixels

        self.system_prompt = system_prompt

        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            max_pixels=self.max_pixels,
            min_pixels=self.min_pixels,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

    def _build_model_inputs(self, messages: List[Dict[str, Any]]):
        """Convert messages to Qwen2.5-VL-compatible format and return processor input tensors.

        - Images: to_pil_image() → {"type":"image","image":PIL,"max_pixels":...,"min_pixels":...}
        - Videos: extract frames with to_nparray_video(), then treat each frame as an image (video_max_pixels applied)
        - Use process_vision_info() from qwen_vl_utils to extract image_inputs/video_inputs
        - Remove second_per_grid_ts (not accepted by model.generate())
        """
        messages = [
            ChatMessage.preprocess_message(
                message=_message,
                remove_image=not self.ENGINE_FEATURES["support_image_understanding"],
                remove_video=not self.ENGINE_FEATURES["support_video_understanding"],
                remove_audio=True,
            )
            for _message in messages
        ]

        rebuilt_messages = []
        if self.system_prompt:
            rebuilt_messages.append({"role": "system", "content": self.system_prompt})

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
                    new_content.append({
                        "type": "image",
                        "image": to_pil_image(item.get("image")),
                        "max_pixels": self.max_pixels,
                        "min_pixels": self.min_pixels,
                    })
                elif item_type == "video":
                    frames_np, _, _ = to_nparray_video(
                        video=item.get("video"),
                        max_frames=self.max_video_frames or 32,
                    )
                    for frame in frames_np:
                        new_content.append({
                            "type": "image",
                            "image": to_pil_image(frame),
                            "max_pixels": self.video_max_pixels,
                            "min_pixels": self.min_pixels,
                        })
                elif item_type == "text":
                    new_content.append({"type": "text", "text": item.get("text", "")})

            rebuilt_messages.append({"role": role, "content": new_content})

        text = self.processor.apply_chat_template(
            rebuilt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(rebuilt_messages)

        model_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        model_inputs.pop("second_per_grid_ts", None)
        model_inputs = model_inputs.to(device=self.model.device)

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

        # Switch to greedy decoding when temperature is 0.
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
