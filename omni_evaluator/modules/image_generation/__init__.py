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

from dataclasses import dataclass
import os
import PIL
from PIL import Image
import torch
import torch.nn as nn
from typing import Any, Dict, List, Literal, Tuple, Union, Optional, Callable, Iterable

from omni_evaluator import T2IGeneratorType
from omni_evaluator.utils.resource import get_num_cuda_devices
from omni_evaluator.modules.image_generation._interface import ImageGeneratorInterface
# from omni_evaluator.modules.image_generation.hyperclova_vdm.t2i import HyperClovaVdmT2IGeneartor
from omni_evaluator.utils.patches import patch_envs, patch_module
from omni_evaluator.utils.resource import split_resources, get_dynamic_max_memory
from omni_evaluator.utils.torch import resolve_torch_dtype


class T2IGenerator(ImageGeneratorInterface):
    def __init__(
        self,
        generator_type: Literal[
            T2IGeneratorType.ta_tok,
            T2IGeneratorType.hyperclova_vdm,
        ],
        torch_dtype: Optional[Union[str, torch.dtype]] = "float32",
        hf_cache_dir: Optional[str] = None,
        cache_dirpath: Optional[str] = None,
        rank: Optional[int] = 0,
        world_size: Optional[int] = 1,
        *args, 
        **kwargs,
    ):
        self.generator_type = generator_type
        self.generator = None
        self.rank = rank
        self.world_size = world_size
        self.max_memory_per_rank = None
        if self.world_size > 1:
            _num_per_rank = split_resources(
                num_resources=get_num_cuda_devices(),
                world_size=self.world_size,
            )
            _device_indices = list(range(sum(_num_per_rank[:self.rank]), sum(_num_per_rank[:self.rank+1])))
            self.max_memory_per_rank = get_dynamic_max_memory(
                device_indices=_device_indices,
                skip_device_under=0.2,
                rank=self.rank,
                world_size=self.world_size,
            )
        
        if generator_type == T2IGeneratorType.ta_tok:
            import omni_evaluator.modules.image_generation.ta_tok.t2i as t2i
            with patch_module(
                module_name="tok",
                module=t2i,
            ):
                from omni_evaluator.modules.image_generation.ta_tok.t2i import TextToImageInference
                self.generator = TextToImageInference(
                    torch_dtype=torch_dtype,
                    hf_cache_dir=hf_cache_dir,
                    max_memory=self.max_memory_per_rank,
                    *args, 
                    **kwargs,
                )
                
        elif generator_type == T2IGeneratorType.hyperclova_vdm:
            self.generator = HyperClovaVdmT2IGeneartor(
                torch_dtype=torch_dtype,
                hf_cache_dir=hf_cache_dir,
                cache_dirpath=cache_dirpath,
                max_memory=self.max_memory_per_rank,
                *args, 
                **kwargs,
            )
            
        else:
            raise ValueError(f'unsupported t2i generator: {generator_type}')
        
    def generate_images(
        self,
        *args, 
        **kwargs,
    ):
        if self.generator_type == T2IGeneratorType.ta_tok:
            return self.generator.generate_images(
                *args,
                **kwargs,
            )
            
        elif self.generator_type == T2IGeneratorType.hyperclova_vdm:
            return self.generator.generate_images(
                *args,
                **kwargs,
            )

        else:
            raise ValueError(f'unsupported generator_type: {self.generator_type}')

    def __del__(self):
        if self.generator is not None:
            del self.generator