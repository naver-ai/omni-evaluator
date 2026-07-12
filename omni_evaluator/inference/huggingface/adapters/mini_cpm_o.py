# Reference from https://huggingface.co/openbmb/MiniCPM-o-2_6 (Apache-2.0)

# Modifications Copyright (c) 2026-present NAVER Cloud Corp.
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
import math
import moviepy
import numpy as np
from omegaconf import DictConfig, ListConfig
import os
from pathlib import Path
import PIL
from PIL import Image
import soundfile
import tempfile
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
from omni_evaluator.utils.multimodal import to_pil_image, to_image_bytes, to_nparray_audio, to_audio_bytes, to_nparray_video
from omni_evaluator.utils.torch import get_compute_capability

logger = logging.getLogger(__name__)

_check_imports_patched = False


def _apply_check_imports_patch():
    # MiniCPM-o remote modeling file has a top-level `import minicpmo`/`from minicpmo...`
    # that does not exist on PyPI; transformers.dynamic_module_utils.get_imports flags it
    # since the import is outside any try/except. Filter `minicpmo` out so dynamic loading
    # proceeds (the actual usage of the module is handled by trust_remote_code at runtime).
    # Deferred to MiniCPM-o instantiation; idempotent.
    global _check_imports_patched
    if _check_imports_patched:
        return
    import transformers.dynamic_module_utils as _dmu
    _orig_get_imports = _dmu.get_imports

    def _filtered_get_imports(filename):
        return [imp for imp in _orig_get_imports(filename) if imp != "minicpmo"]

    _dmu.get_imports = _filtered_get_imports
    _check_imports_patched = True


def _patch_cross_device_embedding_merge(model):
    """Make MiniCPM-o's vision/audio embedding merges device-safe.

    The remote modeling code writes vision/audio features into the LLM token
    embeddings via in-place indexed assignment without moving the source tensor
    onto the destination device. Under device_map="auto" the modality towers and
    embed_tokens can sit on different GPUs, which raises a device-mismatch error
    in `get_vllm_embedding` (scatter_) and `get_omni_embedding` (index assign).

    We wrap both methods so the source tensors are moved onto the destination
    (input-embedding) device just before the merge, then restore the embeddings'
    original device on return so the rest of the forward pass is unaffected.
    Idempotent: a model is patched at most once.
    """
    import types

    if getattr(model, "_cross_device_merge_patched", False):
        return

    # The remote modeling file computes vision_hidden_states inline (there is no
    # `get_vision_embedding` method on this checkpoint) and merges them into the
    # LLM token embeddings. We reproduce the original body faithfully and add an
    # explicit device move on the source tensor before the scatter so the merge
    # works under device_map="auto".
    # Adapted from openbmb/MiniCPM-o-2_6 modeling_minicpmo.py::get_vllm_embedding - https://huggingface.co/openbmb/MiniCPM-o-2_6
    def _patched_get_vllm_embedding(self, data):
        import math

        if "vision_hidden_states" not in data:
            dtype = self.llm.model.embed_tokens.weight.dtype
            device = self.llm.model.embed_tokens.weight.device
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.type(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx],
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
                    ).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:  # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes = torch.Tensor(
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)
        else:
            vision_hidden_states = data["vision_hidden_states"]

        if hasattr(self.llm.config, "scale_emb"):
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

        new_vllm_embedding = vllm_embedding.clone()

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
            for i in vision_hidden_states
        ]

        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        # move the vision source onto the embedding device
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]).to(cur_vllm_emb.device),
                    )
                elif self.training:
                    new_vllm_embedding[i] += cur_vs_hs[0].mean() * 0

        return new_vllm_embedding, vision_hidden_states

    # Adapted from openbmb/MiniCPM-o-2_6 modeling_minicpmo.py::get_omni_embedding - https://huggingface.co/openbmb/MiniCPM-o-2_6
    def _patched_get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        if stream_input:
            audio_embeddings = self.get_audio_embedding_streaming(data)
        else:
            audio_embeddings = self.get_audio_embedding(data, chunk_length)

        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]

                # Mirror upstream `get_omni_embedding`: branch on
                # `config.chunk_input`, NOT `config.stream_input`. With
                # `chunk_input=True` (the chat() default) the processor reserves
                # audio placeholders as PER-1s chunk blocks, so `audio_bounds[i]`
                # holds many short bounds while `get_audio_embedding` returns one
                # pooled tensor per clip. The upstream chunk_input path flattens
                # the per-clip embeddings and slices them across the bounds
                # sequentially; the zip()-1:1 path is only valid when bounds and
                # embeddings are emitted one-to-one (non-chunk_input). The prior
                # patch gated the flatten/slice logic on `stream_input` (default
                # False), so a long standalone audio always hit the zip path and
                # crashed with e.g. "embeddings of shape [500, 3584] to input
                # indices of length 25".
                if getattr(self.config, "chunk_input", True):
                    for i in range(bs):
                        if not audio_embeddings[i]:
                            continue
                        # move the audio source onto the embedding device
                        audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                            device=input_embeddings.device, dtype=input_embeddings.dtype
                        )
                        audio_start_pos = 0
                        for bound in audio_bounds[i]:
                            audio_len = bound[1] - bound[0]
                            input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                                audio_start_pos : audio_start_pos + audio_len, :
                            ]
                            audio_start_pos += audio_len
                else:
                    for i in range(bs):
                        audio_embs = audio_embeddings[i]
                        bounds = audio_bounds[i]
                        for embs, bound in zip(audio_embs, bounds):
                            audio_indices = torch.arange(bound[0], bound[1], dtype=torch.long).to(
                                input_embeddings.device
                            )

                            if embs.shape[0] != len(audio_indices):
                                raise ValueError(
                                    f"Shape mismatch: Trying to assign embeddings of shape {embs.shape} "
                                    f"to input indices of length {len(audio_indices)}"
                                )
                            # move the audio source onto the embedding device
                            input_embeddings[i, audio_indices] = embs.to(
                                device=input_embeddings.device, dtype=input_embeddings.dtype
                            )
        elif self.training:
            for i in range(bs):
                # dummy audio_embedings
                input_embeddings += audio_embeddings[0].mean() * 0

        return input_embeddings

    model.get_vllm_embedding = types.MethodType(_patched_get_vllm_embedding, model)
    model.get_omni_embedding = types.MethodType(_patched_get_omni_embedding, model)
    model._cross_device_merge_patched = True
    logger.info(
        "Patched MiniCPM-o get_vllm_embedding/get_omni_embedding for "
        "cross-device (multi-GPU device_map) embedding merges."
    )


class MiniCPMoModule(HuggingfaceModule):
    ENGINE_FEATURES = InferenceEngineFeatures(
        support_audio_understanding=True,
        support_image_understanding=True,
        support_text_understanding=True,
        support_video_understanding=True,
        support_audio_generation=True,
        support_text_generation=True,
        support_compute_perplexity=True,
    ).to_dict()

    CONTENT_TYPE_PRIORITY = {
        "image": 0,
        "video": 0,
        "text": 1,
        "audio": 2,
    }
    
    def __init__(
        self,
        *args, **kwargs,
    ):
        # uv pip install peft==0.17.1 transformers==4.51.0 vector_quantize_pytorch vocos minicpmo-utils[all]
        _apply_check_imports_patch()
        super().__init__(*args, **kwargs)
        
        self.model_kwargs["torch_dtype"] = torch.float16
        self.model_kwargs["attn_implementation"] = "sdpa"
        self.model_kwargs["trust_remote_code"] = True
        cc = get_compute_capability()
        if cc is not None and cc >= 8.0:
            self.model_kwargs["torch_dtype"] = torch.bfloat16
            self.model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            init_vision=True,
            init_audio=True,
            init_tts=True,
            **self.model_kwargs,
        )
        self.model = self.model.eval()
        self.model.init_tts()

        # When the model is split across GPUs (device_map="auto"), the remote
        # modeling code merges vision/audio embeddings into the LLM token
        # embeddings with manual indexed writes that do NOT move the source
        # tensor onto the destination device:
        #   - get_vllm_embedding: `cur_vllm_emb.scatter_(0, idx, cur_vs_hs...)`
        #   - get_omni_embedding: `input_embeddings[i, idx] = embs.to(dtype)`
        # The vision tower (vpm/resampler) and audio tower (apm) can land on a
        # different GPU than embed_tokens, raising
        # "Expected all tensors to be on the same device" (cuda:0 vs cuda:1).
        # accelerate's dispatch hooks only move tensors at module boundaries,
        # not inside these hand-written merges. Patch them to move the source
        # onto the destination device, which keeps multi-GPU splitting (and
        # thus the memory headroom that motivated allocating 2 GPUs).
        _patch_cross_device_embedding_merge(self.model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, 
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

    # Upper bound on the number of `<unit>` blocks emitted for a video in omni
    # mode. Each unit costs ~85 prompt tokens (a 64-token image placeholder plus
    # a ~12-token 1-second audio placeholder plus the `<unit>` marker), so the
    # cap keeps even a long video (≈ MAX_OMNI_UNITS seconds) comfortably under
    # `max_inp_length` (32768) with ample room left for the question text. This
    # mirrors `to_nparray_video`'s 128-frame cap: without it a multi-minute clip
    # produces thousands of units, overflows `max_inp_length`, and the
    # processor's `_convert` truncation cuts mid-image — leaving one more
    # image-start than image-end token and crashing `torch.hstack` in the image
    # bound code.
    MAX_OMNI_UNITS = 120

    def _build_omni_video_units(self, frames, wav, sampling_rate):
        """Build MiniCPM-o omni-format video contents (one `<unit>` per second).

        MiniCPM-o's omni video mode expects the video fed as repeated `<unit>`
        blocks, each pairing a frame with the matching slice of audio. The audio
        encoder pools mel features with `AvgPool1d(kernel=audio_pool_step=5)`, so
        a chunk shorter than ~9 mel frames (~0.09s at 16kHz) collapses to length
        0 and raises "Output size is too small".

        The previous logic split the waveform into one chunk per sampled frame
        (up to `to_nparray_video`'s 128 frames). For any video shorter than
        ~11s that makes every chunk far too short and the pooler underflows.
        Instead we rebuild units at 1 fps following the upstream openbmb omni
        convention: one 1-second chunk of audio, paired with the frame nearest
        that timestamp. The trailing partial second is zero-padded up to 1s so it
        never underflows the pooler, and because each chunk gets its own audio
        placeholder, the per-chunk placeholder length matches the per-chunk
        feature length (see the `merge_audio_from_same_content` note in
        `_generate`).

        For videos longer than `MAX_OMNI_UNITS` seconds we keep the chunks at 1s
        but sample their start times uniformly across the whole timeline rather
        than emitting one unit per second; this bounds the prompt length (see
        `MAX_OMNI_UNITS`) at the cost of dropping the audio between sampled
        windows, analogous to uniform frame subsampling for the visual stream."""
        has_wav = wav is not None and getattr(wav, "size", 0) > 0
        if not has_wav:
            return [
                token
                for frame in frames
                for token in ("<unit>", Image.fromarray(frame.astype(np.uint8)))
            ]

        sr = int(sampling_rate) if sampling_rate else 16_000
        num_frames = len(frames)
        total_seconds = max(1, math.ceil(wav.shape[0] / sr))
        # Last valid 1-second window start, so a window never runs past the wav.
        last_start = max(0, wav.shape[0] - sr)

        if total_seconds <= self.MAX_OMNI_UNITS:
            num_units = total_seconds
            unit_starts = [u * sr for u in range(num_units)]
        else:
            num_units = self.MAX_OMNI_UNITS
            logger.warning(
                "MiniCPM-o omni video is %ds long; subsampling audio to %d "
                "uniformly-spaced 1s windows to fit max_inp_length.",
                total_seconds,
                num_units,
            )
            unit_starts = [
                int(round(i * last_start / (num_units - 1))) if num_units > 1 else 0
                for i in range(num_units)
            ]

        contents = []
        for unit_idx, start in enumerate(unit_starts):
            if num_frames <= 1:
                frame_idx = 0
            else:
                # frame nearest the temporal midpoint of this 1-second window
                mid_frac = (start + sr / 2) / wav.shape[0]
                frame_idx = min(num_frames - 1, int(round(mid_frac * (num_frames - 1))))
            image = Image.fromarray(frames[frame_idx].astype(np.uint8))
            chunk = wav[start : start + sr]
            if chunk.shape[0] < sr:
                chunk = np.pad(chunk, (0, sr - chunk.shape[0]))
            contents += ["<unit>", image, chunk]
        return contents

    def _generate(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        use_cache: Optional[bool] = True,
        return_audio: Optional[bool] = None,
        **kwargs,
    ):
        if generation_options is None:
            generation_options = dict()
        if return_audio is None:
            return_audio = getattr(self, "return_audio", False)

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

        if self.skip_chat_template:
            raise ValueError(f'MiniCPMoModule not support `skip_chat_template`')

        # preprocess
        user_message = {
            "role": "user",
            "content": list(),
        }
        video_contents = list()
        _has_video = False
        for _message in messages:
            _message["content"] = sorted(
                _message["content"], 
                key=lambda content: self.CONTENT_TYPE_PRIORITY[content["type"]],
            )
            
            _contents = list()
            if isinstance(_message["content"], str):
                _contents.append(_message["content"])

            elif isinstance(_message["content"], dict):
                if _message["content"]["type"] == "text":
                    if "value" in _message["content"]:
                        _contents.append(_message["content"]["value"])
                    elif "text" in _message["content"]:
                        _contents.append(_message["content"]["value"])
                    else:
                        raise ValueError(f'invalid content, Neither `value` nor `text` not exist: {_message["content"]}')

                elif _message["content"]["type"] == "audio":
                    if "value" in _message["content"]:
                        _audio, _sampling_rate, _audio_format = to_nparray_audio(audio=_message["content"]["value"])
                        _contents.append(_audio)
                    elif "audio" in _message["content"]:
                        _audio, _sampling_rate, _audio_format = to_nparray_audio(audio=_message["content"]["audio"])
                        _contents.append(_audio)
                
                elif _message["content"]["type"] == "image":
                    if "value" in _message["content"]:
                        _image = to_pil_image(image=_message["content"]["value"])
                        _contents.append(_image)
                    elif "image" in _message["content"]:
                        _image = to_pil_image(image=_message["content"]["image"])
                        _contents.append(_image)
                
                elif _message["content"]["type"] == "video":
                    _has_video = True
                    _frames, _wav, _sampling_rate = None, None, None
                    if "value" in _message["content"]:
                        _frames, _wav, _sampling_rate = to_nparray_video(
                            video=_message["content"]["value"],
                        )
                    elif "video" in _message["content"]:
                        _frames, _wav, _sampling_rate = to_nparray_video(
                            video=_message["content"]["video"],
                        )
                    _contents += self._build_omni_video_units(_frames, _wav, _sampling_rate)

            elif isinstance(_message["content"], (list, tuple)):
                for _content in _message["content"]:
                    if _content["type"] == "text":
                        if "value" in _content:
                            _contents.append(_content["value"])
                        elif "text" in _content:
                            _contents.append(_content["text"])
                        else:
                            raise ValueError(f'invalid content, Neither `value` nor `text` not exist: {_content}')
                    
                    elif _content["type"] == "audio":
                        if "value" in _content:
                            _audio, _sampling_rate, _audio_format = to_nparray_audio(audio=_content["value"])
                            _contents.append(_audio)
                        elif "audio" in _content:
                            _audio, _sampling_rate, _audio_format = to_nparray_audio(audio=_content["audio"])
                            _contents.append(_audio)
                    
                    elif _content["type"] == "image":
                        if "value" in _content:
                            _image = to_pil_image(image=_content["value"])
                            _contents.append(_image)
                        elif "image" in _content:
                            _image = to_pil_image(image=_content["image"])
                            _contents.append(_image)
                    
                    elif _content["type"] == "video":
                        _has_video = True
                        _frames, _wav, _sampling_rate = None, None, None
                        if "value" in _content:
                            _frames, _wav, _sampling_rate = to_nparray_video(
                                video=_content["value"],
                            )
                        elif "video" in _content:
                            _frames, _wav, _sampling_rate = to_nparray_video(
                                video=_content["video"],
                            )
                        _contents += self._build_omni_video_units(_frames, _wav, _sampling_rate)

            else:
                raise ValueError(f'invalid content: {_message["content"]}')
            
            user_message["content"] += _contents
        
        messages = [user_message, ]
        if len(video_contents) > 0:
            messages.append({
                "role": "user",
                "content": video_contents,
            })
        
        # Video frames are fed as an interleaved `<unit>`/image/audio-chunk
        # sequence (omni format). For that path the remote modeling defaults are
        # wrong and trigger a crash in the processor:
        #   - max_inp_length defaults to 8192. With up to 128 frames the prompt
        #     blows past it, and `processing_minicpmo._convert` truncates with a
        #     naive `input_ids[:max_inp_length]` that can cut through the middle
        #     of an image region. That leaves one more image-start token than
        #     image-end token, so `torch.hstack([starts, ends])` raises
        #     "Sizes of tensors must match ... Expected size N but got size N-1".
        #   - default slicing (max_slice_nums>1) multiplies the per-frame token
        #     count (each slice adds its own start/end markers), making the
        #     overflow far worse.
        # Use the omni-video configuration: no slicing, no image ids, omni token
        # joining, and a large input budget so frames are never truncated. Only
        # set as defaults so an explicit caller override still wins.
        if _has_video:
            generation_options.setdefault("omni_mode", True)
            generation_options.setdefault("use_image_id", False)
            generation_options.setdefault("max_slice_nums", 1)
            generation_options.setdefault("max_inp_length", 32768)

        generation_options_audio = dict()
        if return_audio:
            generation_options_audio["sampling"] = True
            generation_options_audio["use_tts_template"] = True
            generation_options_audio["generate_audio"] = True
            generation_options_audio["output_audio_path"] = True

        # `chat()` defaults to `merge_audio_from_same_content=True`, which hstacks
        # every audio clip that shares a message turn into a SINGLE waveform
        # before mel extraction. Because we collapse all messages into one user
        # turn, that merges otherwise-independent clips. The audio placeholders,
        # however, are emitted PER clip (`get_audio_placeholder(len(a))` per audio
        # in the processor), so the number of reserved `<unk>` tokens is the sum
        # of per-clip frame counts, while the merged waveform yields one feature
        # segment whose frame count differs (center-padding is applied once for
        # the whole waveform instead of once per clip). The two diverge and the
        # embedding merge crashes with e.g.
        #   "The expanded size of the tensor (10) must match the existing size (7)".
        # Disabling the merge keeps one feature segment per clip, so each clip's
        # embeddings line up with its own placeholder region. Only set as a
        # default so an explicit caller override still wins.
        generation_options.setdefault("merge_audio_from_same_content", False)

        generated_text = self.model.chat(
            msgs=messages,
            tokenizer=self.tokenizer,
            **generation_options,
            stopping_criteria=None,
            eos_token_id=self.eos_token_id,
            use_cache=use_cache,
            **generation_options_audio,
        )
        
        prediction = generated_text
        if return_audio:
            pass # TODO: add audio
        
        output = HuggingfaceInferenceOutput(
            prediction_ids=None,
            prediction=prediction,
            generated_ids=None,
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
        # Mirror of the qwen2_omni compute_perplexity pattern, adapted for the
        # MiniCPM-o stack which only exposes high-level `model.chat()`. We
        # route through the lower-level path used by chat() internally:
        #   processor → get_vllm_embedding (+ get_omni_embedding) → llm.forward
        # The base prompt is multimodal-embedded ONCE; each option is embedded
        # via the LLM's text embedder and concatenated onto the base embeds,
        # then a single forward yields logits for NLL over option-token positions.
        #
        # Scope: text + image only. Video/audio prompts are rejected here —
        # perplexity-scored evals are almost always MCQA over (image, text), and
        # the audio/video conversion in _generate (`<unit>`/wav-chunk interleave)
        # bypasses our embed concat strategy.
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

        if self.skip_chat_template:
            raise ValueError(f'MiniCPMoModule not support `skip_chat_template`')

        # 1. Build chat()-format msgs (text + image only).
        user_message = {"role": "user", "content": list()}
        for _message in messages:
            _content = _message.get("content")
            if isinstance(_content, str):
                user_message["content"].append(_content)
                continue
            if isinstance(_content, dict):
                _content = [_content]
            _content = sorted(
                _content,
                key=lambda c: self.CONTENT_TYPE_PRIORITY[c["type"]],
            )
            for _c in _content:
                _ctype = _c.get("type")
                if _ctype == "text":
                    _val = _c.get("value", _c.get("text"))
                    user_message["content"].append(_val)
                elif _ctype == "image":
                    _val = _c.get("value", _c.get("image"))
                    user_message["content"].append(to_pil_image(image=_val))
                else:
                    raise ValueError(
                        f'MiniCPMoModule.compute_perplexity only supports text+image '
                        f'content; got type={_ctype!r}'
                    )
        chat_msgs = [user_message]

        # 2. msgs → processor inputs (mirror modeling_minicpmo.chat() lines 919-980).
        if getattr(self.model, "processor", None) is None:
            self.model.processor = AutoProcessor.from_pretrained(
                self.model.config._name_or_path,
                trust_remote_code=True,
            )
        processor = self.model.processor

        copy_msgs = copy.deepcopy(chat_msgs)
        images = list()
        audios = list()
        audio_parts = list()
        for _i, _msg in enumerate(copy_msgs):
            _content = _msg["content"] if isinstance(_msg["content"], list) else [_msg["content"]]
            _cur_msgs = list()
            for _c in _content:
                if isinstance(_c, PIL.Image.Image):
                    images.append(_c)
                    _cur_msgs.append("(<image>./</image>)")
                elif isinstance(_c, np.ndarray):
                    audios.append(_c)
                    audio_parts.append(_i)
                    _cur_msgs.append("(<audio>./</audio>)")
                elif isinstance(_c, str):
                    _cur_msgs.append(_c)
            _msg["content"] = "\n".join(_cur_msgs)

        prompts_lists = [
            processor.tokenizer.apply_chat_template(
                copy_msgs,
                tokenize=False,
                add_generation_prompt=True,
            ),
        ]
        inputs = processor(
            prompts_lists,
            [images],
            [audios],
            [audio_parts],
            max_slice_nums=None,
            use_image_id=None,
            chunk_input=True,
            return_tensors="pt",
            max_length=32768,
        ).to(device=self.model.device)
        inputs.pop("image_sizes", None)

        # 3. Base inputs_embeds (multimodal-integrated, computed once).
        with torch.inference_mode():
            base_embeds, _ = self.model.get_vllm_embedding(inputs)
            if getattr(self.model.config, "init_audio", False):
                base_embeds = self.model.get_omni_embedding(
                    inputs,
                    input_embeddings=base_embeds,
                    chunk_length=self.model.config.audio_chunk_length,
                )
        base_attention_mask = inputs["attention_mask"]  # [1, base_len]

        # 4. Per-option: text-embed option tokens, concat, single forward, NLL.
        perplexities = list()
        for _option in options:
            _option_input_ids = torch.tensor(
                self.tokenizer.encode(f'{_option}{self.tokenizer.eos_token}'),
                dtype=torch.long,
                device="cpu",
            )  # e.g. tokens + eos
            _num_tokens = len(_option_input_ids) - 1

            _option_embeds = self.model.llm.get_input_embeddings()(
                _option_input_ids[:-1].to(device=base_embeds.device)
            ).unsqueeze(0).to(dtype=base_embeds.dtype)

            _full_embeds = torch.cat([base_embeds, _option_embeds], dim=1)
            _full_attention_mask = torch.cat([
                base_attention_mask,
                torch.ones(
                    (1, _option_input_ids[:-1].shape[0]),
                    dtype=base_attention_mask.dtype,
                    device=base_attention_mask.device,
                ),
            ], dim=1)

            _labels = -100 * torch.ones(_full_embeds.shape[1], dtype=torch.long, device="cpu")
            _labels[-len(_option_input_ids):] = _option_input_ids

            with torch.inference_mode():
                _output = self.model.llm(
                    inputs_embeds=_full_embeds,
                    attention_mask=_full_attention_mask,
                    use_cache=use_cache,
                )
            _logits = _output.logits.detach().to(device="cpu")
            del _output

            _nll = F.cross_entropy(
                _logits.view(-1, _logits.shape[-1]),
                _labels,
                ignore_index=-100,
                reduction="sum",
            )
            _nll = _nll / _num_tokens
            perplexities.append(_nll.item())

        prediction = np.argmin(perplexities)
        output = HuggingfaceInferenceOutput(
            prediction=prediction,
            perplexities=perplexities,
            temp_paths=None,
        )

        return output

    def generate_text(
        self,
        *args,
        **kwargs,
    ):
        return self._generate(
            *args,
            **kwargs,
            return_audio=False,
        )
    
    def generate_audio(
        self,
        *args, 
        **kwargs,
    ):
        return self._generate(
            *args, 
            **kwargs,
            return_audio=True,
        )