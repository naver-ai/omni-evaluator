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
from abc import ABC, abstractmethod
from pathlib import Path
import torch
from typing import List, Tuple, Dict, Any, Union, Optional

from omni_evaluator.modules.image_generation import T2IGenerator
from omni_evaluator.postprocess import parse_think
from omni_evaluator.utils.common import remove_stop_words
from omni_evaluator.utils.torch import resolve_torch_dtype

logger = logging.getLogger(__name__)


class HuggingfaceModule(ABC):
    # Optional-dependency declarations live in the single source of truth
    # ``model_groups.MODULE_REQUIRED_PACKAGES`` (keyed by model group), not on the
    # adapter class. engine.py calls ``model_groups.require_group_dependencies``
    # before model load, so deps are declared in exactly one place.

    def __init__(
        self,
        model_name_or_path: str,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        max_memory: Optional[Any] = None,
        low_cpu_mem_usage: Optional[bool] = True,
        trust_remote_code: Optional[bool] = None,
        reasoning: Optional[bool] = False,
        cache_dir: Optional[str] = None,
        skip_chat_template: Optional[bool] = False,
        max_video_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        stopping_criteria: Optional[Any] = None,
        temp_dirpath: Optional[str] = "./temp",
        t2i_generator: Optional[T2IGenerator] = None,
        **kwargs,
    ):
        # vars to be updated in child class
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.image_processor = None
        # updated vars in this class
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.skip_chat_template = skip_chat_template
        self.max_video_frames = max_video_frames
        self.fps = fps
        self.stopping_criteria = stopping_criteria
        self.temp_dirpath = temp_dirpath
        if not Path(temp_dirpath).exists():
            Path(temp_dirpath).mkdir(parents=True, exist_ok=True)

        if device_map is None:
            device_map = "auto"
        if torch_dtype is None:
            torch_dtype = "auto"
        self.torch_dtype = resolve_torch_dtype(torch_dtype=torch_dtype)
        self.model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "cache_dir": self.cache_dir,
            "low_cpu_mem_usage": low_cpu_mem_usage,
            "device_map": device_map,
            "max_memory": max_memory,
        }
        if trust_remote_code is not None:
            self.model_kwargs["trust_remote_code"] = trust_remote_code
        self.reasoning = reasoning
        
        self.t2i_generator = t2i_generator

    def generate_audio(
        self,
        *args, **kwargs,
    ):
        raise NotImplementedError('generate_audio() should be implemented in child class')

    def generate_image(
        self,
        *args, **kwargs,
    ):
        raise NotImplementedError('generate_image() should be implemented in child class')

    def generate_text(
        self,
        *args, **kwargs,
    ):
        raise NotImplementedError('generate_text() should be implemented in child class')

    def batched_generate_text(
        self,
        messages_list: List[List[Dict[str, Any]]],
        generation_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Any]:
        """Generate text for a batch of conversations, one output per item.

        **Placeholder default**: run ``generate_text`` once per conversation — correct
        for every adapter but NOT batched (no speedup). Adapters that can fuse a single
        ``model.generate`` over the batch (e.g. Qwen text / vl / omni) override this;
        when this default runs, a warning fires once so the missing override is visible.
        ``generation_options`` is copied per item because callees may pop keys (e.g.
        ``stop_words``) from it in place.
        """
        if not getattr(self, "_warned_unbatched_generate", False):
            logger.warning(
                "batched_generate_text is not implemented for %s; falling back to a "
                "per-item generate_text loop (no batching speedup).",
                type(self).__name__,
            )
            self._warned_unbatched_generate = True
        # generate_text returns the RAW prediction (its __call__ wrapper is what
        # post-processes); here it is called directly, so clean each output to keep
        # batched_generate_text's "cleaned prediction" contract consistent.
        _stop_words = copy.deepcopy(generation_options.get("stop_words")) if isinstance(generation_options, dict) else None
        outputs = list()
        for _messages in messages_list:
            _output = self.generate_text(
                messages=_messages,
                generation_options=copy.deepcopy(generation_options) if isinstance(generation_options, dict) else generation_options,
                **kwargs,
            )
            if hasattr(_output, "prediction"):
                _output.prediction = self._apply_generation_postprocess(_output.prediction, stop_words=_stop_words)
            outputs.append(_output)
        return outputs

    def _apply_generation_postprocess(self, prediction: Optional[str], stop_words: Optional[List[str]] = None) -> Optional[str]:
        """Post-process a generated string exactly as ``HuggingfaceInferencer.__call__``
        does after ``generate_text`` (rstrip + stop-word trim; strip the ``<think>`` span
        when ``self.reasoning``). ``batched_generate_text`` is called directly (bypassing
        ``__call__``), so its implementations run this on each decoded prediction to stay
        consistent with the single-sample path. cf. ``engine.py`` __call__."""
        if not isinstance(prediction, str):
            return prediction
        prediction = prediction.rstrip()
        prediction = remove_stop_words(text=prediction, stop_words=stop_words)
        if self.reasoning:
            _parsed = parse_think(
                prediction=prediction,
                think_end_pattern=["</think>", "</think>\n<answer>"],
                eot_token=["<|im_end|>", "</answer>"],
            )
            if _parsed.get("prediction"):
                prediction = _parsed["prediction"]
        return prediction

    @staticmethod
    def _messages_list_has_media(messages_list: List[List[Dict[str, Any]]]) -> bool:
        """True if any conversation in the batch carries image/audio/video content.
        Multimodal adapters use this to fall back to the per-item path (their batched
        fast path is text-only).

        Tolerant of both serialized dicts and schema objects (``ImageContent`` etc.):
        the content item / message may be a dict OR an object, and ``type`` may be a
        ``Modality`` (str-Enum) or a plain string. Detecting media is safety-critical
        (a miss silently drops media in the text-only batch path), so read defensively.
        """
        _MEDIA = ("image", "audio", "video")
        for _messages in messages_list:
            for _message in _messages:
                _contents = _message.get("content", None) if isinstance(_message, dict) else getattr(_message, "content", None)
                if not isinstance(_contents, (list, tuple)):
                    continue
                for _content in _contents:
                    _type = _content.get("type", None) if isinstance(_content, dict) else getattr(_content, "type", None)
                    _type = getattr(_type, "value", _type)   # Modality(str,Enum) -> "image"; plain str unchanged
                    if _type in _MEDIA:
                        return True
        return False

    def generate_video(
        self,
        *args, **kwargs,
    ):
        raise NotImplementedError('generate_video() should be implemented in child class')

    def compute_perplexity(
        self,
        *args, **kwargs,
    ):
        raise NotImplementedError('compute_perplexity() should be implemented in child class')
    
    def get_stopping_criteria(
        self,
        stop_words: List[str],
        eos_token: Optional[str] = None,
        input_ids: Optional[torch.Tensor] = None,
    ):
        if (
            isinstance(eos_token, str)
            and eos_token not in stop_words
        ):
            stop_words.append(eos_token)

        # SHOULD HAVE when evaluation_engine in ["lm_eval_harness", ]
        stopping_criteria = self.stopping_criteria(
            self.tokenizer if self.tokenizer is not None else self.processor,
            stop_words,
            input_ids.shape[1],
            input_ids.shape[0],
        )
        return stopping_criteria

    def _resolve_stopping_criteria(
        self,
        generation_options: Dict[str, Any],
        input_ids: torch.Tensor,
    ) -> Optional[Any]:
        """Pop stop_words from generation_options and build StoppingCriteria if needed.

        Returns the stopping_criteria object, or None if no stop_words were given
        or if self.stopping_criteria factory is not set.
        Modifies generation_options in-place by removing the 'stop_words' key.
        """
        stop_words = generation_options.pop("stop_words", None)
        if (
            self.stopping_criteria is None
            or not isinstance(stop_words, (list, tuple))
            or len(stop_words) == 0
        ):
            return None
        return self.get_stopping_criteria(
            stop_words=list(stop_words),
            eos_token=self.eos_token,
            input_ids=input_ids,
        )
    
    @functools.cached_property
    def pad_token_id(
        self,
    ):
        pad_token_id = None
        if pad_token_id is None and self.tokenizer is not None:
            pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None and self.processor is not None:
            pad_token_id = getattr(self.processor, "pad_token_id", None)
        return pad_token_id
    
    @functools.cached_property
    def bos_token_id(
        self,
    ):
        bos_token_id = None
        if bos_token_id is None and self.tokenizer is not None:
            bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is None and self.processor is not None:
            bos_token_id = getattr(self.processor, "bos_token_id", None)
        return bos_token_id
    
    @functools.cached_property
    def eos_token(
        self,
    ):
        eos_token = None
        if eos_token is None and self.tokenizer is not None:
            eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token is None and self.processor is not None:
            eos_token = getattr(self.processor, "eos_token", None)
        return eos_token
    
    @functools.cached_property
    def eos_token_id(
        self,
    ):
        eos_token_id = None
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None and self.processor is not None:
            eos_token_id = getattr(self.processor, "eos_token_id", None)
        return eos_token_id