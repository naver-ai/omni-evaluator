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
from dataclasses import asdict, dataclass, fields, is_dataclass
import importlib
import json
import numpy as np
from omegaconf import ListConfig, DictConfig, OmegaConf
import os
from pathlib import Path
import PIL
from PIL import Image
import torch
from typing import Union, Any, Callable, Tuple, List, Literal, Dict, Optional

from omni_evaluator.schemas import SchemaInterface

from omni_evaluator.schemas.chat import (
    Message as ChatMessage, 
    ToolCall,
)
from omni_evaluator.schemas.generation_options import GenerationOptions
from omni_evaluator.utils.multimodal import to_image_bytes


@dataclass
class InferenceEngineFeatures(SchemaInterface):
    """Feature flags describing the capabilities of an inference engine."""

    support_audio_understanding: bool = False
    support_text_understanding: bool = True
    support_image_understanding: bool = False
    support_video_understanding: bool = False
    support_audio_generation: bool = False
    support_text_generation: bool = True
    support_image_generation: bool = False
    support_video_generation: bool = False
    support_compute_perplexity: bool = False
    support_ocr: bool = False
    support_entities: bool = False
    support_reasoning: bool = False
    # True by default; flipped to false in ``model_supporting.yaml`` for
    # models that reject temperature/top_p/top_k server-side (e.g. Claude
    # Fable 5 / Opus 4.7+). Read by the generation_options sanitizer.
    support_sampling_params: bool = True

@dataclass
class Record(SchemaInterface):
    """A single evaluation record containing prompt, messages, label, and prediction."""

    benchmark: Optional[str] = None
    index: Optional[str] = None
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    generation_options: Optional[Union[Dict[str, Any], GenerationOptions]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    label: Optional[Any] = None
    options: Optional[List[Union[int, str]]] = None
    option_contents: Optional[List[str]] = None
    prediction: Optional[Any] = None
    prediction_postprocessed: Optional[Any] = None
    reasoning_content: Optional[str] = None
    generation: Optional[str] = None
    finish_reason: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Any] = None
    annotations: Optional[List[Any]] = None
    perplexities: Optional[List[Union[int, float]]] = None
    prompt_logprobs: Optional[List[Any]] = None
    error_message: Optional[str] = None
    latency: Optional[Union[int, float]] = None
    metrics: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        for _field in fields(self):
            _field_value = getattr(self, _field.name)
            if isinstance(_field_value, np.integer):
                setattr(self, _field.name, int(_field_value))
            elif isinstance(_field_value, np.floating):
                setattr(self, _field.name, float(_field_value))
            elif isinstance(_field_value, ListConfig):
                setattr(self, _field.name, OmegaConf.to_container(_field_value))
            elif isinstance(_field_value, DictConfig):
                setattr(self, _field.name, OmegaConf.to_container(_field_value))
                
        if self.tools: # validate if tools are in 'Hermes' format
            for _tool in self.tools:
                if not isinstance(_tool.get("tool", None), str):
                    raise ValueError(
                        f'invalid tool format, tool should be in a `Hermes` format: {_tool}'
                    )
                if _tool.get("arguments", None) and not isinstance(_tool["arguments"], dict):
                    raise ValueError(
                        f'invalid tool format, tool should be in a `Hermes` format: {_tool}'
                    )
    
    def merge_inference_output(self, output: Optional[Union[Dict[str, Any], "InferenceOutput"]]) -> None:
        """Copy inference engine response fields onto this Record in-place.

        Accepts either a raw response dict (the result of parse_response) or an
        InferenceOutput dataclass instance (a type-safe object wrapped by vllm/sglang/api,
        etc.). For a dataclass, it is converted to a dict via asdict, then goes through the same key mapping.
        """
        if not output:
            return
        if not isinstance(output, dict):
            # InferenceOutput dataclass — convert to plain dict (asdict-style).
            # SchemaInterface.to_dict() handles recursive None-aware emit; but for
            # merge purposes we want the raw shallow dict including None values
            # so callers can distinguish "not provided" (skip) vs "explicit None".
            output = {f.name: getattr(output, f.name) for f in fields(output)}
        if "prediction" in output:
            self.prediction = output["prediction"]
        if "reasoning_content" in output:
            self.reasoning_content = output["reasoning_content"]
        if "generated_text" in output:
            self.generation = output["generated_text"]
        if "finish_reason" in output:
            self.finish_reason = output["finish_reason"]
        if "tool_calls" in output:
            self.tool_calls = output["tool_calls"]
        if "function_call" in output:
            self.function_call = output["function_call"]
        if "annotations" in output:
            self.annotations = output["annotations"]
        if "perplexities" in output:
            self.perplexities = output["perplexities"]
        if "prompt_logprobs" in output:
            self.prompt_logprobs = output["prompt_logprobs"]
        if "error_message" in output:
            self.error_message = output["error_message"]
        if "latency" in output:
            self.latency = output["latency"]

    def to_dict(
        self,
        template: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Serialize Record to dict, converting messages to the target template and PIL predictions to base64.
        # Args: template - provider format passed through to Message.to_dict ("openai"|"anthropic"|etc.)
        # Returns: dict representation with serialized messages and predictions
        messages = [
            ChatMessage.to_template(obj=_message, template=template)
            if isinstance(_message, dict) else _message.to_dict(template=template)
            for _message in self.messages
        ]
        prediction = self.prediction
        if isinstance(prediction, (list, tuple)):
            prediction = list(prediction)
            for _idx, _prediction in enumerate(prediction):
                if isinstance(_prediction, PIL.Image.Image):
                    prediction[_idx] = to_image_bytes(
                        image=_prediction,
                        encode_base64=True,
                    )
        elif isinstance(prediction, dict):
            prediction = dict(prediction)
            for _key, _prediction in prediction.items():
                if isinstance(_prediction, PIL.Image.Image):
                    prediction[_key] = to_image_bytes(
                        image=_prediction,
                        encode_base64=True,
                    )
        output = super().to_dict()
        output["messages"] = messages
        output["prediction"] = prediction
        return output

    @classmethod
    def from_dict(
        cls,
        obj: Dict[str, Any] = None,
        **kwargs,
    ):
        if obj is not None:
            kwargs = obj
        valid_field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_field_names}
        return cls(**filtered)

    def verbose(
        self,
        prefix: str = "",
        full_messages: bool = False,
    ) -> None:
        # Print a human-readable summary of this Record including prompt, label, prediction, and metrics.
        # Args: prefix - string prepended to each output line for indentation
        # Returns: None (prints to stdout)
        if isinstance(self.prompt, str):
            print(f'{prefix}- {"Prompt":<15}: {self.prompt}')
        if self.messages is not None:
            if full_messages:
                for _message_idx, _message in enumerate(self.messages):
                    if _message["role"] == "system":
                        continue
                    _turn_prefix = f'"{_message.name}"' if _message.name else f'"Turn_{_message_idx}"' 
                    _turn_content = ChatMessage.get_query(message=_message)
                    print(f'{prefix}- {_turn_prefix:<15}: {_turn_content}')
            else:
                _query = ChatMessage.get_query(message=self.messages[-1])
                print(f'{prefix}- {"Query":<15}: {_query}')
        if self.label is not None:
            print(f'{prefix}- {"Label":<15}: {self.label}')   
        if isinstance(self.prediction, (list, tuple, dict, bytes)):
            print(f'{prefix}- {"Prediction":<15}: type {type(self.prediction)}, size {len(self.prediction)}')
        elif self.prediction:
            print(f'{prefix}- {"Prediction":<15}: {self.prediction}')
        if isinstance(self.prediction_postprocessed, str):
            print(f'{prefix}- {"Postprocessed":<15}: {self.prediction_postprocessed}')
        if self.reasoning_content:
            print(f'{prefix}- {"ReasoningContent":<15}: {self.reasoning_content}')
        if self.perplexities is not None:
            print(f'{prefix}- {"Perplexities":<15}: {self.perplexities}')
        if self.options is not None:
            print(f'{prefix}- {"Options":<15}: {self.options}')
        if self.option_contents is not None:
            print(f'{prefix}- {"OptionContents":<15}: {self.option_contents}')
        if isinstance(self.latency, (int, float)):
            print(f'{prefix}- {"Latency":<15}: {self.latency}')
        if isinstance(self.metrics, (dict, DictConfig)):
            print(f'{prefix}- {"Metrics":<15}:')
            for _metric_name, _metric_value in self.metrics.items():
                if not isinstance(_metric_value, (int, float)):
                    continue
                print(f"{prefix}{prefix}- {_metric_name:<40}: {float(_metric_value):<.04f}")
        print()

@dataclass
class InferenceOutput(SchemaInterface):
    prediction: Optional[Any] = None
    reasoning_content: Optional[str] = None
    generated_audios: Optional[List[Any]] = None
    generated_text: Optional[str] = None
    generated_images: Optional[List[Any]] = None
    generated_videos: Optional[List[Any]] = None
    perplexities: Optional[List[Union[int, float]]] = None
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    function_call: Optional[Any] = None
    annotations: Optional[List[str]] = None
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    embeddings: Optional[torch.Tensor] = None
    latency: Optional[float] = None
    prompt: Optional[str] = None

    def to_dict(self):
        output = super().to_dict()
        # drop field with None
        output = {k: v for k, v in output.items() if v is not None}
        return output

@dataclass
class HuggingfaceInferenceOutput(InferenceOutput):
    prediction: Optional[Any] = None
    reasoning_content: Optional[str] = None
    prediction_ids: Optional[torch.Tensor] = None
    generated_audios: Optional[List[Any]] = None
    generated_text: Optional[str] = None
    generated_images: Optional[List[Any]] = None
    generated_videos: Optional[List[Any]] = None
    generated_ids: Optional[torch.Tensor] = None
    perplexities: Optional[List[Union[int, float]]] = None
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    function_call: Optional[Any] = None
    annotations: Optional[List[str]] = None
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    loss: Optional[Union[int, float, torch.Tensor]] = None
    embeddings: Optional[torch.Tensor] = None
    latency: Optional[float] = None
    prompt: Optional[str] = None
    temp_paths: Optional[List[str]] = None

@dataclass
class VllmInferenceOutput(InferenceOutput):
    prediction: Optional[Any] = None
    reasoning_content: Optional[str] = None
    generated_audios: Optional[List[Any]] = None
    generated_text: Optional[str] = None
    generated_images: Optional[List[Any]] = None
    generated_videos: Optional[List[Any]] = None
    perplexities: Optional[List[Union[int, float]]] = None
    prompt_logprobs: Optional[List[Any]] = None
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    function_call: Optional[Any] = None
    annotations: Optional[List[str]] = None
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    embeddings: Optional[torch.Tensor] = None
    latency: Optional[float] = None
    prompt: Optional[str] = None

@dataclass
class SglangInferenceOutput(InferenceOutput):
    prediction: Optional[Any] = None
    reasoning_content: Optional[str] = None
    generated_audios: Optional[List[Any]] = None
    generated_text: Optional[str] = None
    generated_images: Optional[List[Any]] = None
    generated_videos: Optional[List[Any]] = None
    perplexities: Optional[List[Union[int, float]]] = None
    prompt_logprobs: Optional[List[Any]] = None
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    function_call: Optional[Any] = None
    annotations: Optional[List[str]] = None
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    embeddings: Optional[torch.Tensor] = None
    latency: Optional[float] = None
    prompt: Optional[str] = None

@dataclass
class ApiInferenceOutput(InferenceOutput):
    prediction: Optional[Union[str, Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None
    generated_audios: Optional[List[Any]] = None
    generated_text: Optional[str] = None
    generated_images: Optional[List[Any]] = None
    generated_videos: Optional[List[Any]] = None
    perplexities: Optional[List[Union[int, float]]] = None
    tool_calls: Optional[List[Union[ToolCall, Dict[str, Any]]]] = None
    function_call: Optional[Any] = None
    annotations: Optional[List[str]] = None
    finish_reason: Optional[str] = None
    error_message: Optional[str] = None
    embeddings: Optional[torch.Tensor] = None
    latency: Optional[float] = None
    prompt: Optional[str] = None