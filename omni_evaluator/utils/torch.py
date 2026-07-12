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

from contextlib import contextmanager
from functools import lru_cache
import logging
import torch
from typing import List, Tuple, Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def is_torchcodec_loadable() -> bool:
    # Probe whether torchcodec's native .so actually loads. Bare `import torchcodec`
    # is not enough — the C-extension load happens lazily on first decoder
    # construction, so any wheel-vs-PyTorch ABI mismatch (e.g. missing
    # `aoti_torch_create_device_guard` symbol) or libavutil version mismatch
    # only surfaces there. We trigger that path once at probe time and cache
    # the result, so callers can pre-emptively coerce decode=False on
    # datasets.Audio/Video columns when the runtime is broken — without that
    # guard, datasets' lazy decode on first sample iteration explodes
    # mid-evaluation.
    try:
        import torchcodec  # noqa: F401
        from torchcodec.decoders import AudioDecoder  # noqa: F401
        return True
    except Exception as ex:
        logger.warning(
            'torchcodec native load failed; downstream callers should fall back to '
            'raw bytes / librosa-based decoders. Underlying error: %s',
            ex,
        )
        return False


def get_compute_capability(
    device_index: int = 0,
) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device_index)
    cc_version = float(f'{major}.{minor}')
    return cc_version

def get_cuda_indexes() -> Optional[List[int]]:
    output = None
    if torch.cuda.is_available():
        output = list(range(torch.cuda.device_count()))
    return output

def is_cpu(device_map: Optional[Union[str, Dict[str, Any]]] = None) -> bool:
    # True when a HF backend would run on CPU: explicit device_map=cpu (str or
    # all-cpu dict), or no CUDA available at all.
    if device_map == "cpu" or (
        isinstance(device_map, dict) and device_map
        and all(str(v) == "cpu" for v in device_map.values())
    ):
        return True
    return not torch.cuda.is_available()

@contextmanager
def torch_num_threads(num_threads: Optional[int] = None):
    # Temporarily cap torch intra-op threads to num_threads for the block, restoring
    # the previous value on exit. num_threads falsy / <= 0 -> no-op (leave unchanged).
    if not num_threads or num_threads <= 0:
        yield
        return
    prev_num_threads = torch.get_num_threads()
    torch.set_num_threads(int(num_threads))
    try:
        yield
    finally:
        torch.set_num_threads(prev_num_threads)

def resolve_torch_dtype(
    torch_dtype: Union[str, torch.dtype],
) -> Union[str, torch.dtype]:
    # Alias branches first — keep ``int`` aliases (16/32) reachable. The earlier
    # ``elif not isinstance(torch_dtype, str)`` was a generic catch-all that
    # consumed integer inputs before the ``16 in [..., 16]`` / ``32 in [..., 32]``
    # checks, so ``resolve_torch_dtype(16)`` silently flipped to bf16 on A100/H100
    # (cc>=8.0) instead of returning float16.
    if isinstance(torch_dtype, torch.dtype):
        pass
    elif torch_dtype in [
        "auto",
    ]:
        pass
    elif torch_dtype in [
        "torch.float32",
        "float32",
        "fp32",
        "32",
        32,
    ]:
        torch_dtype = torch.float32
    elif torch_dtype in [
        "torch.bfloat16",
        "bfloat16",
        "bf16",
    ]:
        torch_dtype = torch.bfloat16
    elif torch_dtype in ["torch.float16", "float16", "fp16", "16", 16]:
        torch_dtype = torch.float16
    elif not isinstance(torch_dtype, str):
        # Fallback for unrecognized non-string inputs (None / unknown ints):
        # cc-based default — bf16 on Ampere+, fp16 otherwise.
        torch_dtype = torch.float16
        cc = get_compute_capability()
        if cc is not None and cc >= 8.0:
            torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"invalid torch_dtype: {torch_dtype}")

    return torch_dtype

def find_first_difference(
    a: torch.Tensor, 
    b: torch.Tensor,
) -> int:
    min_len = min(len(a), len(b))
    diffs = (a[:min_len] != b[:min_len]).nonzero(as_tuple=False)
    return diffs[0].item() if len(diffs) > 0 else min_len