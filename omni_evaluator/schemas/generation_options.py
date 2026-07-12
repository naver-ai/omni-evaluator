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

from collections import defaultdict
import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
import logging
from typing import Union, Any, Callable, Tuple, List, Literal, Dict, Optional

from omni_evaluator.enums.engine import ApiGroup, InferenceEngine
from omni_evaluator.schemas import SchemaInterface

logger = logging.getLogger(__name__)


@dataclass
class GenerationOptions(SchemaInterface):
    """Base class for generation/sampling options across inference engines."""

    inference_engine: InferenceEngine
    
    def to_dict(self):
        output = super().to_dict()
        # remove if field has not been set
        output.pop("inference_engine", None)
        for field in fields(self): 
            if output.get(field.name, None) is not None:
                continue
            output.pop(field.name, None)
        return output
    
    @classmethod
    def from_dict(cls, obj: Dict[str, Any]):
        valid_field_names = [field.name for field in fields(cls)]
        obj = {
            k: v 
            for k, v in obj.items()
            if (
                k in valid_field_names
                and v is not None
            )
        }
        return cls(**obj)
    
@dataclass
class HuggingfaceGenerationOptions(GenerationOptions):
    inference_engine: InferenceEngine = InferenceEngine.huggingface
    num_beams: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    # text generation
    max_new_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    stop_words: Optional[List[str]] = None
    reasoning_effort: Optional[str] = None
    # image generation
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[List[str]] = None
    
    @classmethod
    def from_dict(cls, obj: Dict[str, Any]):
        if obj.get("temperature", 1.0) == 0.0:
            obj["do_sample"] = False  # greedy decoding (deterministic)
        # drop fields not supported by HuggingFace generate()
        for key in ("frequency_penalty", "presence_penalty", "n", "logprobs", "top_logprobs", "seed"):
            obj.pop(key, None)
        return super().from_dict(obj=obj)

    
@dataclass
class VllmGenerationOptions(GenerationOptions):
    inference_engine: InferenceEngine = InferenceEngine.vllm
    n: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = field(default_factory=list)
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]):
        obj.pop("num_beams", None)  # vLLM does not support beam search via API
        obj["max_tokens"] = obj.pop("max_new_tokens", None)
        obj["stop"] = obj.pop("stop_words", list())
        # drop fields not supported by vLLM SamplingParams
        for key in ("length_penalty", "frequency_penalty", "presence_penalty", "top_logprobs"):
            obj.pop(key, None)
        return super().from_dict(obj=obj)

@dataclass
class SglangGenerationOptions(GenerationOptions):
    inference_engine: InferenceEngine = InferenceEngine.sglang
    n: Optional[int] = None
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    repetition_penalty: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = field(default_factory=list)
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]):
        obj.pop("num_beams", None)  # SGLang does not support beam search
        obj["max_tokens"] = obj.pop("max_new_tokens", None)
        obj["stop"] = obj.pop("stop_words", list())
        # drop fields not supported by SGLang
        for key in ("length_penalty", "frequency_penalty", "presence_penalty", "top_logprobs"):
            obj.pop(key, None)
        return super().from_dict(obj=obj)


@dataclass
class ApiGenerationOptions(GenerationOptions):
    inference_engine: Optional[InferenceEngine] = None
    temperature: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    logprobs: Optional[Union[int, bool]] = None
    top_logprobs: Optional[int] = None
    response_logprobs: Optional[bool] = None
    n: Optional[int] = None

    def to_dict(self, template: str):
        if (
            template == "openai"
            or template == "api/openai"
        ):
            return ApiOpenaiGenerationOptions(
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop=self.stop,
                logprobs=self.logprobs,
                top_logprobs=self.top_logprobs,
                n=self.n,
            ).to_dict()
        elif (
            template == "anthropic"
            or template == "api/anthropic"
        ):
            return ApiAnthropicGenerationOptions(
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stop_sequences=self.stop,
            ).to_dict()
        elif (
            template == "google"
            or template == "api/google"
        ):
            return ApiGoogleGenerationOptions(
                logprobs=self.logprobs,
                responseLogprobs=self.response_logprobs,
                temperature=self.temperature,
                frequencyPenalty=self.frequency_penalty,
                presencePenalty=self.presence_penalty,
                topK=self.top_k,
                topP=self.top_p,
                maxOutputTokens=self.max_tokens,
                stopSequences=self.stop,
            ).to_dict()

    @classmethod
    def from_dict(
        cls,
        obj: Dict[str, Any],
        api_name: str,
        api_group: Optional[str] = None,
        inference_engine: Optional[str] = None,
        reasoning_options=None,
    ):
        if (
            not api_group
            and not inference_engine
        ):
            raise ValueError(f'Either `api_group` or `inference_engine` should be given')
        if (
            api_group == ApiGroup.openai
            or inference_engine == InferenceEngine.api__openai
        ):
            return ApiOpenaiGenerationOptions.from_dict(
                obj=obj,
                api_name=api_name,
                reasoning_options=reasoning_options,
            )
        elif (
            api_group == ApiGroup.anthropic
            or inference_engine == InferenceEngine.api__anthropic
        ):
            return ApiAnthropicGenerationOptions.from_dict(
                obj=obj,
                api_name=api_name,
                reasoning_options=reasoning_options,
            )
        elif (
            api_group == ApiGroup.google
            or inference_engine == InferenceEngine.api__google
        ):
            return ApiGoogleGenerationOptions.from_dict(
                obj=obj,
                api_name=api_name,
                reasoning_options=reasoning_options,
            )
        else:
            raise ValueError(f'unsupported inference_engine: {inference_engine}')

@dataclass
class ApiOpenaiGenerationOptions(GenerationOptions):
    inference_engine: InferenceEngine = InferenceEngine.api__openai
    temperature: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    n: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(
        cls,
        obj: Dict[str, Any],
        api_name: Optional[str] = None,
        reasoning_options=None,
    ):
        # drop fields not supported by OpenAI API
        for key in ("num_beams", "do_sample", "top_k", "length_penalty"):
            obj.pop(key, None)
        _repetition_penalty = obj.pop("repetition_penalty", None)
        if "frequency_penalty" not in obj:
            obj["frequency_penalty"] = _repetition_penalty
        _max_new_tokens = obj.pop("max_new_tokens", None)  # max_new_tokens -> max_tokens in OpenAI
        if "max_tokens" not in obj:
            obj["max_tokens"] = _max_new_tokens
        _stop = obj.pop("stop_words", list())
        if "stop" not in obj:
            obj["stop"] = _stop
        if not obj.get("stop"):  # empty list should not be included for OpenAI
            obj.pop("stop", None)
        # logprobs: int (GenerationOptionArgs) -> bool (OpenAI API takes bool)
        _logprobs = obj.get("logprobs")
        if isinstance(_logprobs, int):
            obj["logprobs"] = _logprobs > 0
            if _logprobs > 0 and "top_logprobs" not in obj:
                obj["top_logprobs"] = _logprobs

        from omni_evaluator.api.model_supporting import get_model_supporting
        _caps = get_model_supporting(provider="openai", api_name=api_name) if api_name else {}
        if _caps.get("support_reasoning"):
            _options_to_replace = {
                "max_tokens": "max_completion_tokens", # max_completion_tokens not works as like max_tokens
                "temperature": None, # reasoning model not support temperature
                "top_p": None, # reasoning model not support top_p
                "frequency_penalty": None, # reasoning model not support frequency_penalty
                "presence_penalty": None, # reasoning model not support presence_penalty
                "logprobs": None, # reasoning model not support logprobs
                "stop": None, # reasoning model not support stop
            }
            for _from, _to in _options_to_replace.items():
                _value = obj.pop(_from, None)
                if _value and _to:
                    obj[_to] = _value
            if reasoning_options and reasoning_options.get("reasoning_effort"):
                obj["reasoning"] = {"effort": reasoning_options.get("reasoning_effort")}
        return super().from_dict(obj=obj)
    

@dataclass
class ApiAnthropicGenerationOptions(GenerationOptions):
    inference_engine: InferenceEngine = InferenceEngine.api__anthropic
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    thinking: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(
        cls,
        obj: Dict[str, Any],
        api_name: Optional[str] = None,
        reasoning_options=None,
    ):
        # drop fields not supported by Anthropic API
        for key in ("num_beams", "do_sample", "repetition_penalty", "frequency_penalty",
                    "presence_penalty", "length_penalty", "n", "logprobs", "top_logprobs", "seed"):
            obj.pop(key, None)
        _max_new_tokens = obj.pop("max_new_tokens", None)  # max_new_tokens -> max_tokens in Anthropic
        if "max_tokens" not in obj:
            obj["max_tokens"] = _max_new_tokens
        # Anthropic Messages API requires ``max_tokens`` (SDK raises
        # ``TypeError: Missing required arguments`` if absent / None). Some
        # lm_eval tasks (math, mmlu_redux, mgsm, ...) don't set it, leaving
        # None here — fall back to a safe default so the call doesn't fail.
        if not obj.get("max_tokens"):
            obj["max_tokens"] = 8192
        obj["stop_sequences"] = obj.pop("stop_words", list())
        # Anthropic rejects ``stop_sequences`` entries that are empty or
        # whitespace-only (HTTP 400 ``invalid_request_error: each stop
        # sequence must contain non-whitespace``). lm_eval tasks sometimes
        # pass ``""`` / ``"\n"`` — filter them out before the API call.
        if isinstance(obj.get("stop_sequences"), list):
            obj["stop_sequences"] = [
                _s for _s in obj["stop_sequences"]
                if isinstance(_s, str) and _s.strip()
            ]
        from omni_evaluator.api.model_supporting import get_model_supporting
        _caps = get_model_supporting(provider="anthropic", api_name=api_name) if api_name else {}
        # Fable 5 / Opus 4.7+ reject temperature/top_p/top_k server-side
        # (HTTP 400). model_supporting.yaml flags these models with
        # ``support_sampling_params: false``; drop the params before they
        # reach the API. Default is true, so Sonnet / Haiku / older Opus
        # keep their existing handling below.
        if _caps.get("support_sampling_params", True) is False:
            for _k in ("temperature", "top_p", "top_k"):
                obj.pop(_k, None)
        if (
            reasoning_options
            and reasoning_options.get("thinking_budget")
            and _caps.get("support_reasoning")
        ):
            obj["thinking"] = {
                "type": "enabled",
                "budget_tokens": reasoning_options.get("thinking_budget"),
            }
            # extended thinking requires temperature=1; remove to use API default
            obj.pop("temperature", None)
            obj.pop("top_p", None)
        elif (
            obj.get("temperature", None)
            and obj.get("top_p", None)
        ): # can not use temperature and top_p at the same time
            obj.pop("top_p", None)
        return super().from_dict(obj=obj)
    

@dataclass
class ApiGoogleGenerationOptions(GenerationOptions):
    inference_engine: InferenceEngine = InferenceEngine.api__google
    logprobs: Optional[int] = None
    responseLogprobs: Optional[bool] = None
    temperature: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    presencePenalty: Optional[float] = None
    topK: Optional[int] = None
    topP: Optional[float] = None
    maxOutputTokens: Optional[int] = None
    stopSequences: Optional[List[str]] = None
    thinking_config: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(
        cls,
        obj: Dict[str, Any],
        api_name: Optional[str] = None,
        reasoning_options=None,
    ):
        # drop fields not supported by Google API
        for key in ("num_beams", "do_sample", "n", "length_penalty", "top_logprobs", "seed"):
            obj.pop(key, None)
        if isinstance(obj.get("logprobs"), int):
            obj["responseLogprobs"] = True
        obj["topP"] = obj.pop("top_p", None)
        obj["topK"] = obj.pop("top_k", None)
        if obj.get("repetition_penalty", None):
            obj["frequencyPenalty"] = obj.pop("repetition_penalty", None)
        elif obj.get("frequency_penalty", None):
            obj["frequencyPenalty"] = obj.pop("frequency_penalty", None)
        obj["presencePenalty"] = obj.pop("presence_penalty", None)
        obj["maxOutputTokens"] = obj.pop("max_new_tokens", None)
        obj["stopSequences"] = obj.pop("stop_words", list())
        from omni_evaluator.api.model_supporting import get_model_supporting
        _caps = get_model_supporting(provider="google", api_name=api_name) if api_name else {}
        if (
            reasoning_options
            and reasoning_options.get("thinking_budget")
            and _caps.get("support_reasoning")
        ):
            obj["thinking_config"] = {"thinking_budget": reasoning_options.get("thinking_budget")}
        return super().from_dict(obj=obj)