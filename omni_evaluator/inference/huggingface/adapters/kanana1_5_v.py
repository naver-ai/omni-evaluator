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

import logging
import types
import torch
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
except Exception as ex:
    logger.warning('Could not import dependencies for `kanana1_5_v`')


def _vision_attn_fa2_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
    """Replaces VisionAttention.forward with a direct flash_attn_varlen_func call.
    Skips O(n²) attention mask construction to prevent OOM on long video inputs."""
    from flash_attn import flash_attn_varlen_func as _fa2_varlen
    from transformers.models.qwen2_vl.modeling_qwen2_vl import apply_rotary_pos_emb_vision

    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = emb.cos(), emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
    attn_output = _fa2_varlen(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
    attn_output = attn_output.reshape(seq_length, -1)
    attn_output = self.proj(attn_output)
    return attn_output


from omni_evaluator.inference.huggingface._interface import HuggingfaceModule
from omni_evaluator.schemas.chat import Message as ChatMessage
from omni_evaluator.schemas.inference import InferenceEngineFeatures, HuggingfaceInferenceOutput
from omni_evaluator.utils.multimodal import to_nparray_video, to_pil_image
from omni_evaluator.utils.torch import get_compute_capability
from PIL import Image


class Kanan1_5VModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_text_understanding=True,
        support_image_understanding=True,
        support_video_understanding=True,
        support_audio_understanding=False,
        support_text_generation=True,
        support_compute_perplexity=False,  # non-standard generate() signature prevents standard forward pass
        support_image_generation=False,
        support_audio_generation=False,
    ).to_dict()

    # default values from preprocessor_config.json
    _BASE_MAX_PIXELS = 1_254_400
    _MIN_PIXELS = 78_400

    def __init__(
        self,
        video_max_pixels: Optional[int] = None,
        *args, **kwargs,
    ):
        HuggingfaceModule.__init__(self, *args, **kwargs)

        self.model_kwargs["trust_remote_code"] = True

        if get_compute_capability() >= 8.0:
            self.model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            self.model_kwargs["attn_implementation"] = "eager"
            # On cc < 8.0, eager attention outputs float32, causing dtype mismatches,
            # so load the entire model in float32.
            self.model_kwargs["torch_dtype"] = torch.float32

        # Kanana hardcodes _attn_implementation="flash_attention_2" in vision_config,
        # which is not propagated to sub-configs inside from_pretrained, causing FA2
        # validation to fail. Patch the FA2 validation method to fall back to eager
        # on ImportError/ValueError.
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
            def _safe_fa2(cls, config, *args, **kwargs):
                try:
                    return _fa2_orig.__func__(cls, config, *args, **kwargs)
                except (ImportError, ValueError):
                    config._attn_implementation = "eager"
                    return config

            setattr(_ptm, _fa2_patch_attr, _safe_fa2)

        elif hasattr(_ptm, 'get_correct_attn_implementation'):
            _fa2_patch_attr = 'get_correct_attn_implementation'
            _fa2_orig = getattr(_ptm, _fa2_patch_attr)

            def _safe_fa2(self, requested_attention, is_init_check=False):
                try:
                    return _fa2_orig(self, requested_attention, is_init_check)
                except (ImportError, ValueError):
                    return "eager"

            setattr(_ptm, _fa2_patch_attr, _safe_fa2)

        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name_or_path,
            **self.model_kwargs,
        ).eval()

        if _fa2_patch_attr is not None:
            setattr(_ptm, _fa2_patch_attr, _fa2_orig)

        if get_compute_capability() >= 8.0:
            # cc >= 8.0: replace VisionAttention with a direct flash_attn_varlen_func call
            # to work around FA2 dispatcher incompatibility
            for module in self.model.modules():
                if module.__class__.__name__ == "VisionAttention":
                    module.forward = types.MethodType(_vision_attn_fa2_forward, module)
        else:
            # cc < 8.0: from_pretrained's torch_dtype conversion may not apply to
            # sub-models created via _from_config, so cast explicitly to float32.
            self.model = self.model.to(dtype=torch.float32)

        # scale pixel budget according to the number of video frames
        _max_frames = self.max_video_frames or 32
        if video_max_pixels is not None:
            self.video_max_pixels = video_max_pixels
        elif _max_frames > 16:
            self.video_max_pixels = max(
                int(self._BASE_MAX_PIXELS * 16 / _max_frames),
                self._MIN_PIXELS,
            )
        else:
            self.video_max_pixels = self._BASE_MAX_PIXELS

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

    def _resize_for_video(self, image: Image.Image) -> Image.Image:
        """Resize a video frame to fit within the video_max_pixels budget."""
        w, h = image.size
        if w * h <= self.video_max_pixels:
            return image
        scale = (self.video_max_pixels / (w * h)) ** 0.5
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return image.resize((new_w, new_h), Image.LANCZOS)

    def _build_kanana_inputs(self, messages: List[Dict[str, Any]]):
        """Convert omni_evaluator HF-format messages to the Kanana processor format
        and return model input tensors via batch_encode_collate().

        Kanana processor input format:
            [{"conv": [{"role": ..., "content": "<image> ... text"}], "image": [PIL, ...]}]
        - <image> markers are inserted at each image position, prepended before the text
        - returns pixel_values, image_metas, input_ids, attention_mask, seq_length
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

        conv = []
        all_images = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", [])

            if isinstance(content, str):
                conv.append({"role": role, "content": content})
                continue

            images_in_msg = []
            text_parts = []

            for item in content:
                item_type = item.get("type", "")
                if item_type == "image":
                    images_in_msg.append(to_pil_image(item.get("image")))
                elif item_type == "video":
                    frames_np, _, _ = to_nparray_video(
                        video=item.get("video"),
                        max_frames=self.max_video_frames or 32,
                    )
                    for frame in frames_np:
                        pil = self._resize_for_video(to_pil_image(frame))
                        images_in_msg.append(pil)
                elif item_type == "text":
                    text_parts.append(item.get("text", ""))

            all_images.extend(images_in_msg)

            # prepend <image> markers before the text content
            msg_text = " ".join(text_parts)
            if images_in_msg:
                msg_content = "<image> " * len(images_in_msg) + msg_text
            else:
                msg_content = msg_text

            conv.append({"role": role, "content": msg_content})

        data = [{"conv": conv, "image": all_images if all_images else None}]

        inputs = self.processor.batch_encode_collate(
            data,
            padding="longest",
            padding_side="right",
            max_length=32768,
            add_generation_prompt=True,
        )
        return inputs

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
        inputs = self._build_kanana_inputs(messages)

        device = self.model.device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        seq_length = inputs.get("seq_length")
        if seq_length is not None:
            seq_length = seq_length.to(device)

        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(device=device, dtype=self.model.dtype)

        image_metas = inputs.get("image_metas")

        stopping_criteria = None
        if (
            self.stopping_criteria is not None
            and isinstance(stop_words, (list, tuple))
            and len(stop_words) > 0
        ):
            stopping_criteria = self.get_stopping_criteria(
                stop_words=stop_words,
                eos_token=self.eos_token,
                input_ids=input_ids,
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

        cont = self.model.generate(
            pixel_values=pixel_values,
            image_metas=image_metas,
            input_ids=input_ids,
            attention_mask=attention_mask,
            seq_length=seq_length,
            **generation_options,
            stopping_criteria=stopping_criteria,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=use_cache,
        )

        # Kanana's generate() uses inputs_embeds internally and may return only
        # newly generated tokens; detect this case by comparing output shape.
        if cont.shape[1] <= input_ids.shape[1]:
            generated_ids = cont[0].detach().cpu()
        else:
            generated_ids = cont[0][len(input_ids[0]):].detach().cpu()

        prediction = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return HuggingfaceInferenceOutput(
            prediction_ids=generated_ids,
            prediction=prediction,
        )
