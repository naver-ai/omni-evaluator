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
from huggingface_hub import hf_hub_download
import logging
import os
import PIL
from PIL import Image
import shutil
import sys
import torch
import torch.nn as nn
from typing import Any, Dict, List, Literal, Tuple, Union, Optional, Callable, Iterable

from omni_evaluator.clients.s3_client import S3Client
from omni_evaluator.utils.io import is_sub_path
from omni_evaluator.utils.patches import patch_envs, patch_module
from omni_evaluator.utils.torch import get_cuda_indexes, resolve_torch_dtype

logger = logging.getLogger(__name__)


class HyperClovaVdmT2IGenerator:
    def __init__(
        self, 
        model_path: str,
        width: Optional[int] = 768,
        height: Optional[int] = 768,
        num_inference_steps: Optional[int] = 50,
        discrete_token_length: Optional[int] = 729,
        density: Optional[int] = 768 ** 2,
        factor: Optional[int] = 16,
        ratio_list: Optional[List[Tuple[int]]] = [
            (1, 1), (1, 2), (3, 4), (3, 5), (4, 5), (6, 9), (9, 16),
        ],
        required_vram: Optional[int] = 16, # GiB
        torch_dtype: Optional[Union[str, torch.dtype]] = "float32",
        max_memory: Optional[Dict[str, Any]] = None,
        hf_cache_dir: Optional[str] = None,
        cache_dirpath: Optional[str] = None,
    ):  
        if os.path.exists(model_path):
            pass
        elif (
            hf_cache_dir
            and os.path.exists(hf_cache_dir)
        ):
            # use checkpoint in hf_cache_dir
            model_path = os.path.join(hf_cache_dir, model_path)
        elif (
            cache_dirpath
            and os.path.exists(cache_dirpath)
        ):
            # download from S3 storage
            s3_client = S3Client(
                bucket_name=os.environ["S3_BUCKET_NAME"],
                access_key=os.environ["S3_ACCESS_KEY"],
                secret_key=os.environ["S3_SECRET_KEY"],
                endpoint_url=os.environ["S3_ENDPOINT_URL"],
                region=os.environ["S3_REGION"],
            )
            s3_client.download_dir(
                dirpath=cache_dirpath,
                remote_dirpath=model_path,
            )
            model_path = os.path.join(cache_dirpath, model_path)
        else:
            raise ValueError(f'`model_path` not exists: {model_path}')
                
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
            if max_memory is None:
                _cuda_indexes = get_cuda_indexes()
                _vram_per_device = int(required_vram // len(_cuda_indexes))
                max_memory = {
                    _index: f'{_vram_per_device}GiB' 
                    for _index in _cuda_indexes
                }
            else:
                _cuda_indexes = [_key for _key in max_memory.keys() if _key != "cpu"]
                _vram_per_device = int(required_vram // len(_cuda_indexes))
                max_memory = {
                    _index: f'{min(_vram_per_device, _available_vram)}GiB' 
                    for _index, _available_vram in max_memory.items()
                }
            # max_memory["cpu"] = "16GiB" # cpu offloading
        torch_dtype = resolve_torch_dtype(torch_dtype)
        
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device = device
        self.hf_cache_dir = hf_cache_dir
        self.cache_dirpath = cache_dirpath
        self.width = width
        self.height = height
        self.num_inference_steps = num_inference_steps
        self.discrete_token_length = discrete_token_length
        self.density = density
        self.factor = factor
        self.ratio_list = ratio_list
        self.resolution_map = dict()
        for _width_ratio, _height_ratio in list(set(ratio_list)):
            _aspect_ratio = _height_ratio / _width_ratio
            _width = int((((density / _aspect_ratio) ** 0.5) // factor) * factor)
            _height = int((((density * _aspect_ratio) ** 0.5) // factor) * factor)
            self.resolution_map[(_width_ratio, _height_ratio)] = (_width, _height)
        
        import pipeline
        
        with patch_module(
            module_name="pipeline",
            module=pipeline,
        ):
            self.pipeline = pipeline.VisionTokenToImagePipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map="balanced",
                max_memory=max_memory,
            )
        logger.info(f'Loaded t2i generator - hyperclova_vdm: {model_path}')
        
    @torch.no_grad()
    def generate_images(
        self, 
        input_ids: torch.Tensor,
        width: Optional[int] = None,
        height: Optional[int] = None,
        width_ratio: Optional[int] = None,
        height_ratio: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs
    ) -> PIL.Image.Image:
        """
        Args:
            input_ids: 2d torch.Tensor
        """
        
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        
        if (
            width is not None
            and height is not None
        ): # use given resolution
            pass
        elif (
            width_ratio is not None
            and height_ratio is not None
            and (width_ratio, height_ratio) in self.resolution_map
        ): # get resolution by ratio
            width, height = self.resolution_map[(width_ratio, height_ratio)]
        else: # use default resolution
            width = self.width
            height = self.height
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        
        _output = self.pipeline(
            vision_tokens=input_ids[:, :self.discrete_token_length], # expected length
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )

        images = _output.images
        return images
    
    def __del__(
        self,
    ):
        try:
            if is_sub_path(
                parent_path=self.cache_dirpath,
                child_path=self.model_path,
            ):
                shutil.rmtree(self.model_path)
                logger.info(f'Removed `model_path` in cache_dirpath: {self.model_path}')
        except Exception:
            pass