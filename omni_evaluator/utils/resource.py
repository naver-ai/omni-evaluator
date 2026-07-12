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
import sys
import torch
from typing import Any, Dict, List, Union, Tuple, Optional, Callable, Iterable

logger = logging.getLogger(__name__)

_MIN_CPU_MEMORY: int = 200 * 1024**3
# A100 session - gpu: 85051572224, cpu: 2164179365888
# V100 session - gpu: 34089926656, cpu: 404351586304
# P40 session  - gpu: 25633226752, cpu: 270080565248


def get_total_cpu_memory() -> int:
    import psutil
    return psutil.virtual_memory().total


def get_available_cpu_memory() -> int:
    total = get_total_cpu_memory()
    return int(max(min(_MIN_CPU_MEMORY, total * 0.80), total * 0.25))


def get_num_cuda_devices() -> int:
    return torch.cuda.device_count()


def get_dynamic_max_memory(
    device_indices: List[int],
    skip_device_under: Optional[float] = None,
    rank: int = 0,
    world_size: int = 1,
    margin_fraction: float = 1.0,
) -> Dict[Union[int, str], int]:
    memory = dict()
    logger.info(f"(rank: {rank+1:02}/{world_size:02}) dynamic device memory")
    for device_index in device_indices:
        available_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        if (
            isinstance(skip_device_under, float)
            and available_bytes / total_bytes < skip_device_under
        ):
            logger.info(f'Skip using cuda:{device_index} since already allocated: {available_bytes}/{total_bytes}')
            continue
        device_name = f'cuda:{device_index}'
        total_bytes = int(total_bytes * margin_fraction)
        bytes_to_allocated = int(available_bytes * margin_fraction)
        memory[device_index] = bytes_to_allocated
        logger.info(f'(rank: {rank+1:02}/{world_size:02}) {device_name}: {bytes_to_allocated / (1024**3):.2f}/{total_bytes / (1024**3):.2f}GiB')
    if not (len(memory) >= 1):
        raise ValueError(f'At least one cuda device should be allocated: {memory}')
    # cpu_per_rank = get_available_cpu_memory()
    cpu_per_rank = get_total_cpu_memory() // world_size
    memory["cpu"] = cpu_per_rank
    logger.info(f'cpu: {cpu_per_rank / (1024**3):.2f}GiB')
    return memory


def split_resources(
    num_resources: int,
    world_size: int,
) -> List[int]:
    _num_per_rank, _num_left = divmod(num_resources, world_size)
    _num_per_rank = [
        (_num_per_rank+1) if _rank < _num_left else _num_per_rank 
        for _rank in range(0, world_size)
    ]
    return _num_per_rank