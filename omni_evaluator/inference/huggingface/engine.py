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
import gc
import itertools
import logging
import os
import time
import threading
import torch
import traceback
from tqdm import tqdm
from typing import Any, Dict, List, Union, Optional, Iterable

logger = logging.getLogger(__name__)

from omni_evaluator import EvaluationMethod, EvaluationEngine, HuggingfaceModelGroup
from omni_evaluator.utils.resource import get_num_cuda_devices
from omni_evaluator.modules.image_generation import T2IGenerator
from omni_evaluator.inference import NUM_DEBUG_SAMPLES
from omni_evaluator.inference.huggingface.adapters.ax4_vl import Ax4VlModule
from omni_evaluator.inference.huggingface.adapters.deepseek_vl import DeepSeekVlModule
from omni_evaluator.inference.huggingface.adapters.kanana1_5_v import Kanan1_5VModule
from omni_evaluator.inference.huggingface.adapters.emu3 import Emu3Module
from omni_evaluator.inference.huggingface.adapters.hyperclovax import HyperclovaxModule
from omni_evaluator.inference.huggingface.adapters.hyperclovax_vision import HyperclovaxSeedVisionModule
from omni_evaluator.inference.huggingface.adapters.hyperclovax_vision_v2 import HyperclovaxSeedVisionV2Module
from omni_evaluator.inference.huggingface.adapters.janus import JanusModule
from omni_evaluator.inference.huggingface.adapters.janus_pro import JanusProModule
from omni_evaluator.inference.huggingface.adapters.llava import LlavaModule
from omni_evaluator.inference.huggingface.adapters.llava_one_vision import LlavaOneVisionModule
from omni_evaluator.inference.huggingface.adapters.mini_cpm_o import MiniCPMoModule
from omni_evaluator.inference.huggingface.adapters.phi4_multimodal import Phi4MultimodalModule
from omni_evaluator.inference.huggingface.adapters.qwen2 import Qwen2Module
from omni_evaluator.inference.huggingface.adapters.qwen2_audio_instruct import Qwen2AudioInstructModule
from omni_evaluator.inference.huggingface.adapters.qwen2_audio import Qwen2AudioModule
from omni_evaluator.inference.huggingface.adapters.qwen2_omni import Qwen2OmniModule
from omni_evaluator.inference.huggingface.adapters.qwen2_vl import Qwen2VlModule
from omni_evaluator.inference.huggingface.adapters.qwen3 import Qwen3Module
from omni_evaluator.inference.huggingface.adapters.qwen3_omni import Qwen3OmniModule
from omni_evaluator.inference.huggingface.adapters.qwen3_vl import Qwen3VlModule
from omni_evaluator.inference.huggingface.adapters.stable_diffusion_v1 import StableDiffusionV1Module
from omni_evaluator.inference.huggingface.adapters.vaetki_vl import VaetkiVlModule
from omni_evaluator.inference.huggingface.adapters.voxtral import VoxtralModule
from omni_evaluator.inference.huggingface.adapters.whisper_v3 import WhisperV3Module
from omni_evaluator.inference.huggingface.adapters.x_omni import XOmniModule
from omni_evaluator.postprocess import parse_think
from omni_evaluator.utils.common import remove_stop_words
from omni_evaluator.utils.resource import split_resources, get_dynamic_max_memory
from omni_evaluator.schemas.generation_options import HuggingfaceGenerationOptions
from omni_evaluator.schemas.inference import HuggingfaceInferenceOutput
from omni_evaluator.schemas.task import TaskConfig


# Model group -> adapter class. Module-level (not rebuilt per __init__).
# Keep in sync with HuggingfaceModelGroup / the adapter imports above.
MODULE_DISPATCH = {
    HuggingfaceModelGroup.ax4_vl: Ax4VlModule,
    HuggingfaceModelGroup.kanana1_5_v: Kanan1_5VModule,
    HuggingfaceModelGroup.deepseek_vl: DeepSeekVlModule,
    HuggingfaceModelGroup.emu3: Emu3Module,
    HuggingfaceModelGroup.hyperclovax: HyperclovaxModule,
    HuggingfaceModelGroup.hyperclovax_seed: HyperclovaxModule,
    HuggingfaceModelGroup.hyperclovax_seed_vision: HyperclovaxSeedVisionModule,
    HuggingfaceModelGroup.hyperclovax_seed_vision_v2: HyperclovaxSeedVisionV2Module,
    HuggingfaceModelGroup.janus: JanusModule,
    HuggingfaceModelGroup.janus_pro: JanusProModule,
    HuggingfaceModelGroup.llava: LlavaModule,
    HuggingfaceModelGroup.llava_onevision_hf: LlavaOneVisionModule,
    HuggingfaceModelGroup.mini_cpm_o: MiniCPMoModule,
    HuggingfaceModelGroup.phi4_multimodal: Phi4MultimodalModule,
    HuggingfaceModelGroup.qwen2: Qwen2Module,
    HuggingfaceModelGroup.qwen2_audio_instruct: Qwen2AudioInstructModule,
    HuggingfaceModelGroup.qwen2_audio: Qwen2AudioModule,
    HuggingfaceModelGroup.qwen2_omni: Qwen2OmniModule,
    HuggingfaceModelGroup.qwen2_vl: Qwen2VlModule,
    HuggingfaceModelGroup.qwen3: Qwen3Module,
    HuggingfaceModelGroup.qwen3_omni: Qwen3OmniModule,
    HuggingfaceModelGroup.qwen3_vl: Qwen3VlModule,
    HuggingfaceModelGroup.stable_diffusion_v1: StableDiffusionV1Module,
    HuggingfaceModelGroup.vaetki_vl: VaetkiVlModule,
    HuggingfaceModelGroup.voxtral: VoxtralModule,
    HuggingfaceModelGroup.whisper_v3: WhisperV3Module,
    HuggingfaceModelGroup.x_omni: XOmniModule,
}


class HuggingfaceInferencer:

    def __init__(
        self,
        evaluation_engine: str,
        model_name_or_path: str,
        model_group: Optional[Union[str, HuggingfaceModelGroup]] = None,
        reasoning: Optional[Union[str, bool]] = False,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        max_memory: Optional[Any] = None,
        low_cpu_mem_usage: Optional[bool] = True,
        trust_remote_code: Optional[bool] = None,
        cache_dir: Optional[str] = None,
        use_safetensors: Optional[bool] = None,
        skip_chat_template: Optional[bool] = False,
        max_video_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
        use_audio_in_video: Optional[bool] = False,
        temp_dirpath: Optional[str] = "./temp",
        rank: int = 0,
        world_size: int = 1,
        **model_kwargs,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        if model_group is not None:
            self.model_group = HuggingfaceModelGroup(model_group)
        else:
            from omni_evaluator.inference.huggingface.model_groups import get_model_group
            self.model_group = get_model_group(model_name_or_path=model_name_or_path)
        self.reasoning = reasoning
        self.use_audio_in_video = use_audio_in_video

        self.evaluation_engine = evaluation_engine
        self.stopping_criteria = None
        if not isinstance(self.evaluation_engine, str):
            pass
        elif self.evaluation_engine == EvaluationEngine.builtin:
            pass
        elif self.evaluation_engine == EvaluationEngine.lmms_eval:
            pass
        elif self.evaluation_engine == EvaluationEngine.lm_eval_harness:
            from lm_eval.models.utils_hf import stop_sequences_criteria
            self.stopping_criteria = stop_sequences_criteria
        elif self.evaluation_engine == EvaluationEngine.vlm_eval_kit:
            pass

        self.rank = rank
        self.world_size = world_size
        if self.world_size > 1:
            _num_per_rank = split_resources(
                num_resources=get_num_cuda_devices(),
                world_size=self.world_size,
            )
            _device_indices = list(range(sum(_num_per_rank[:self.rank]), sum(_num_per_rank[:self.rank+1])))
            max_memory_per_rank = get_dynamic_max_memory(
                device_indices=_device_indices,
                skip_device_under=0.2,
                rank=self.rank,
                world_size=self.world_size,
            )
            device_map = "auto"
            max_memory = max_memory_per_rank

        common_kwargs = dict(
            model_name_or_path=model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            skip_chat_template=skip_chat_template,
            max_video_frames=max_video_frames,
            fps=fps,
            stopping_criteria=self.stopping_criteria,
            temp_dirpath=temp_dirpath,
            reasoning=reasoning,
            use_audio_in_video=use_audio_in_video,
            use_safetensors=use_safetensors,
        )

        module_cls = MODULE_DISPATCH.get(self.model_group)
        if module_cls is None:
            raise ValueError(f'Unsupported model_group: {self.model_group}')
        # Fail fast on a missing optional dependency before any weight download.
        # Single source of truth: model_groups.MODULE_REQUIRED_PACKAGES.
        from omni_evaluator.inference.huggingface.model_groups import require_group_dependencies
        require_group_dependencies(self.model_group)
        self.module = module_cls(**common_kwargs, **model_kwargs)


    @torch.no_grad()
    def __call__(
        self,
        messages: List[Dict[str, Any]],
        generation_options: Optional[Dict[str, Any]] = None,
        options: Optional[List[str]] = None,
        evaluation_method: Optional[str] = EvaluationMethod.generation,
        output_modality: Optional[List[str]] = None,
        **kwargs,
    ) -> HuggingfaceInferenceOutput:
        if not isinstance(generation_options, dict):
            generation_options = dict()
        stop_words = copy.deepcopy(generation_options.get("stop_words", None))

        output = None
        _start_time = time.time()
        if evaluation_method == EvaluationMethod.perplexity:
            output = self.module.compute_perplexity(
                messages=messages,
                options=options,
                **kwargs,
            )
        elif evaluation_method == EvaluationMethod.generation:
            if (
                self.module.ENGINE_FEATURES["support_text_generation"]
                and (
                    output_modality is None
                    or "text" in output_modality
                )
            ): # default
                output = self.module.generate_text(
                    messages=messages,
                    generation_options=generation_options,
                    **kwargs,
                )
            
            if (
                self.module.ENGINE_FEATURES["support_audio_generation"]
                and output_modality is not None
                and "audio" in output_modality
            ):
                output = self.module.generate_audio(
                    messages=messages,
                    generation_options=generation_options,
                    **kwargs,
                )

            if (
                self.module.ENGINE_FEATURES["support_image_generation"]
                and output_modality is not None
                and "image" in output_modality
            ):
                output = self.module.generate_image(
                    messages=messages,
                    generation_options=generation_options,
                    **kwargs,
                )

            if (
                self.module.ENGINE_FEATURES["support_video_generation"]
                and output_modality is not None
                and "video" in output_modality
            ):
                # TODO: placeholder, update when generate_video is updated
                # output = self.module.generate_video(
                #     messages=messages,
                #     generation_options=generation_options,
                #     **kwargs,
                # )
                pass
        else:
            raise ValueError(f'Invalid evaluation_method: {evaluation_method}')
        
        latency = time.time() - _start_time
        output.latency = latency

        if (
            isinstance(output, HuggingfaceInferenceOutput)
            and isinstance(output.temp_paths, (tuple, list))
        ):
            for temp_path in output.temp_paths:
                os.remove(temp_path) # remove temp files
            output.temp_paths = None

        if evaluation_method == EvaluationMethod.generation:
            if isinstance(output.prediction, str):
                output.prediction = output.prediction.rstrip()
                output.prediction = remove_stop_words(
                    text=output.prediction,
                    stop_words=stop_words,
                )
                if self.reasoning:
                    _output = parse_think(
                        prediction=output.prediction,
                        think_end_pattern=[
                            "</think>",
                            "</think>\n<answer>",
                        ],
                        eot_token=[
                            "<|im_end|>",
                            "</answer>",
                        ],
                    )
                    if _output["prediction"]:
                        output.prediction = _output["prediction"]
                    if _output["reasoning_content"]:
                        output.reasoning_content = _output["reasoning_content"]
        return output

    def __del__(self) -> None:
        if self.module is not None:
            if self.module.model is not None:
                del self.module.model
            del self.module

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def main(
    rank: int,
    world_size: int,
    run_index: int,
    num_runs: int,
    inference_engine: str,
    evaluation_engine: str,
    benchmark: str,
    evaluation_method: str,
    benchmark_idx: int,
    num_benchmarks: int,
    benchmark_dataset: Iterable,
    task_config: TaskConfig,
    default_generation_options: Dict[str, Any],
    model_name_or_path: str,
    benchmark_dataset_size: Optional[int] = None,
    model_group: Optional[Union[str, HuggingfaceModelGroup]] = None,
    reasoning: Optional[Union[str, bool]] = False,
    skip_chat_template: Optional[bool] = False,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    low_cpu_mem_usage: Optional[bool] = None,
    trust_remote_code: Optional[bool] = None,
    hf_hub_cache: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
    max_video_frames: Optional[int] = None,
    fps: Optional[float] = None,
    use_audio_in_video: Optional[bool] = False,
    t2i_generator: Optional[T2IGenerator] = None,
    debug: Optional[bool] = False,
    verbose: Optional[bool] = True,
    thread_lock: Optional[threading.Lock] = None,
    thread_barrier: Optional[threading.Barrier] = None,
    *args, **kwargs,
) -> Optional[List[Dict[str, Any]]]:
    # Orchestrate per-sample inference for a benchmark dataset using local HuggingFace models.
    # Args: benchmark_dataset - iterable of evaluation records, task_config - benchmark task configuration,
    #   default_generation_options - sampling params, model_name_or_path - HF model identifier or local path,
    #   model_group - override automatic model group detection (HuggingfaceModelGroup or its string value),
    #   reasoning - enable chain-of-thought parsing, evaluation_method - "generation" or "perplexity"
    # Returns: list of record dicts with predictions, or None on complete failure
    if (
        world_size > 1
        and thread_lock is not None
    ):
        thread_lock.acquire()

    hf_inferencer = HuggingfaceInferencer(
        evaluation_engine=evaluation_engine,
        model_name_or_path=model_name_or_path,
        model_group=model_group,
        reasoning=reasoning,
        torch_dtype=torch_dtype,
        device_map=device_map,
        max_memory=None,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
        cache_dir=hf_hub_cache,
        skip_chat_template=skip_chat_template,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_video_frames=max_video_frames,
        fps=fps,
        use_audio_in_video=use_audio_in_video,
        t2i_generator=t2i_generator,
        rank=rank,
        world_size=world_size,
        **kwargs,
    )
    if evaluation_method == EvaluationMethod.perplexity:
        if not hf_inferencer.module.ENGINE_FEATURES["support_compute_perplexity"]:
            raise RuntimeError(f'Evaluation method `perplexity` not implemented in hugginface module: {hf_inferencer.module}')
    
    if world_size > 1:
        if thread_lock is not None:
            thread_lock.release()
        if thread_barrier is not None:
            logger.info(f'(rank: {rank+1:02}/{world_size:02}) waiting for other thread to load model')
            thread_barrier.wait()

    # benchmark_dataset is the pre-sliced per-rank iterator from infer.py
    # (split_iterator). When called with a non-sliced input (e.g. world_size=1
    # path that bypasses split), fall back to the full-dataset size for tqdm.
    if benchmark_dataset_size is None:
        benchmark_dataset_size = task_config["num_records"]

    # initialize output_bin
    output = list()
    for record_idx, record in tqdm(
        enumerate(benchmark_dataset),
        initial=0,
        total=benchmark_dataset_size,
        desc=f'({benchmark_idx+1:02}/{num_benchmarks:02}) (rank: {rank+1:02}/{world_size:02}) (run: {run_index+1:02}/{num_runs:02}) Inference {inference_engine} - {benchmark}/{evaluation_method}'
    ):
        if debug and record_idx >= NUM_DEBUG_SAMPLES:
            break
        if record is None:
            logger.warning(f"({rank+1:03}/{world_size:03}) invalid record: {benchmark}/{record_idx}")
            continue

        _record = copy.deepcopy(record) # to prevent pointer intervention
        generation_options = copy.deepcopy(default_generation_options)
        if isinstance(_record.generation_options, dict):
            for _k, _v in _record.generation_options.items():
                if (
                    _k not in generation_options
                    or generation_options[_k] is None
                ):
                     generation_options[_k] = _v
            
        _record.generation_options = HuggingfaceGenerationOptions.from_dict(obj=generation_options).to_dict()
        
        options = None
        if evaluation_method == EvaluationMethod.perplexity:
            options = ["A", "B", "C", "D"] # default
            if _record.option_contents is not None:
                options = _record.option_contents
            elif _record.options is not None:
                options = _record.options


        try:
            _start_time = time.time()
            _inference_output = hf_inferencer(
                messages=[
                    _message.to_dict(template="hf")
                    for _message in _record.messages
                ],
                generation_options=_record.generation_options,
                options=options,
                evaluation_method=evaluation_method,
                output_modality=task_config["meta"]["output_modality"],
            )

            if evaluation_method == EvaluationMethod.generation:
                pass
            elif evaluation_method == EvaluationMethod.perplexity:
                if _record.options:
                    _inference_output["prediction"] = _record.options[_inference_output["prediction"]]
                elif _record.option_contents:
                    _inference_output["prediction"] = _record.option_contents[_inference_output["prediction"]]
                else:
                    _inference_output["prediction"] = options[_inference_output["prediction"]]

            # hf_inferencer may return either a dict or a HuggingfaceInferenceOutput.
            # merge_inference_output handles both (for a dataclass, asdict-shallow then
            # the same key mapping). A wrap pattern consistent with vllm/sglang/api —
            # unifying per-engine dataclass usage.
            if _inference_output is not None:
                _record.merge_inference_output(_inference_output)
            _record.latency = time.time() - _start_time   # engine-level wall time wins
        except Exception as ex:
            logger.warning(f'{benchmark}/{evaluation_method} failed `{inference_engine}` - {type(ex)}: {str(ex)}')
            traceback.print_exc()
            # preserve error_message so error context flows to the caller
            # (engine.main -> infer.py jsonl).
            _record.error_message = f'{type(ex).__name__}: {ex}'
            if world_size > 1:
                raise

        output.append(_record)
        logger.debug(f'[{benchmark}/{evaluation_method}] - instance_id: {getattr(_record, "index", record_idx)}')
        if verbose:
            _record.verbose(prefix="\t")
    
    if hf_inferencer is not None:
        del hf_inferencer

    if (
        world_size > 1
        and thread_barrier is not None
    ): 
        logger.info(f'(idx: {benchmark_idx+1:02}/{num_benchmarks:02}) (rank: {rank+1:02}/{world_size:02}) waiting for other thread')
        thread_barrier.wait()

    return output