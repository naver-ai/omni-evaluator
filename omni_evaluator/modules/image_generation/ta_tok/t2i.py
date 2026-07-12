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
import torch
import torch.nn as nn
from typing import Any, Dict, List, Literal, Tuple, Union, Optional, Callable, Iterable

from omni_evaluator.utils.patches import patch_envs, patch_module
from omni_evaluator.utils.torch import resolve_torch_dtype

logger = logging.getLogger(__name__)


@dataclass
class T2IConfig:
    model_path: str
    ar_path: str
    encoder_path: str
    decoder_path: str
    device: str
    dtype: torch.dtype
    scale: Literal[0, 1, 2] = 0
    seq_len: Literal[729, 169, 81] = 729
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 1200
    cfg_scale: float = 4.0


class TextToImageInference:
    def __init__(
        self, 
        model_path: str,
        ar_path: str,
        encoder_path: str,
        decoder_path: str,
        scale: Optional[int] = 0,
        seq_len: Optional[int] = 729,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.95,
        top_k: Optional[int] = 1200,
        cfg_scale: Optional[float] = 4.0,
        torch_dtype: Optional[Union[str, torch.dtype]] = "float32",
        max_memory: Optional[Dict[str, Any]] = None,
        hf_cache_dir: Optional[str] = None,
    ):
        _device = "cpu"
        if torch.cuda.is_available():
            _device = "cuda"
            if max_memory:
                for _key in max_memory.keys():
                    if _key == "cpu":
                        continue
                    _device = f'cuda:{_key}'
        _torch_dtype = resolve_torch_dtype(torch_dtype)
        
        if not os.path.exists(ar_path):
            ar_path = hf_hub_download(
                "csuhan/TA-Tok", 
                ar_path, 
                cache_dir=hf_cache_dir,
            )
        if not os.path.exists(encoder_path):
            encoder_path = hf_hub_download(
                "csuhan/TA-Tok", 
                encoder_path, 
                cache_dir=hf_cache_dir,
            )
        if not os.path.exists(decoder_path):
            decoder_path = hf_hub_download(
                "peizesun/llamagen_t2i", 
                decoder_path, 
                cache_dir=hf_cache_dir,
            )
        
        self.config = T2IConfig(
            model_path=model_path,
            ar_path=ar_path,
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            scale=scale,
            seq_len=seq_len,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            cfg_scale=cfg_scale,
            device=_device,
            dtype=_torch_dtype,
        )        
        # torch.set_grad_enabled(False)
        
        with patch_envs(envs={
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
        }):
            import ta_tok
            with patch_module(
                module=ta_tok,
                module_name="tok",
            ):
                from ta_tok.mm_autoencoder import MMAutoEncoder
                self.visual_tokenizer = MMAutoEncoder(
                    ar_path=self.config.ar_path,
                    encoder_path=self.config.encoder_path,
                    decoder_path=self.config.decoder_path,
                    encoder_args={"input_type": "rec"},
                    decoder_args={},
                ).eval().to(dtype=self.config.dtype, device=self.config.device)
            self.visual_tokenizer.ar_model.cls_token_num = self.config.seq_len
            self.visual_tokenizer.encoder.pool_scale = self.config.scale + 1
        logger.info(f'Loaded t2i generator - ta_tok: {model_path}')

    @torch.no_grad()
    def generate_images(
        self, 
        input_ids: torch.Tensor,
        **kwargs,
    ) -> PIL.Image.Image:
        """
        Args:
            input_ids: 2d torch.Tensor
        """
        _padding_matrix = [0, ] * self.config.seq_len
        input_ids = torch.cat([
            input_ids, 
            torch.tensor([_padding_matrix, ] * len(input_ids), dtype=torch.long, device=input_ids.device),
        ], dim=1)[:, :self.config.seq_len]
        
        generated_ids = self.visual_tokenizer.decode_from_encoder_indices(
            input_ids.to(self.config.device),
            {"cfg_scale": self.config.cfg_scale}
        ).detach().cpu()
        
        images = list()
        for _generated_ids in generated_ids:
            _image = Image.fromarray(_generated_ids.numpy())
            images.append(_image)
        return images