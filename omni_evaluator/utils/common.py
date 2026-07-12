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

import argparse
import importlib
import inspect
import json
import logging
import numpy as np
import os
from pathlib import Path
import random
import requests
import secrets
import sys
import time
import types
from typing import Any, Dict, List, Union, Tuple, Optional, Callable, Iterable
import urllib

from omni_evaluator import InferenceEngine, EvaluationEngine

logger = logging.getLogger(__name__)


def list_inference_engines() -> List[str]:
    """Return all registered inference engine names."""
    output = [engine.value for engine in InferenceEngine]
    return output


def list_evaluation_engines() -> List[str]:
    """Return all registered evaluation engine names."""
    output = [engine.value for engine in EvaluationEngine]
    return output

def list_tasks(
    evaluation_engine: str,
) -> List[str]:
    """Return all available task names for the given *evaluation_engine*."""
    if evaluation_engine == EvaluationEngine.builtin:
        available_tasks = list()
        for _task in importlib.resources.files(
            f'omni_evaluator.evaluation.builtin.tasks',
        ).iterdir():
            if not os.path.isdir(_task):
                continue
            _task_name = Path(_task).name
            available_tasks.append(_task_name)
        return available_tasks
    
    elif evaluation_engine == EvaluationEngine.lmms_eval:
        from lmms_eval.tasks import TaskManager
        task_manager = TaskManager(
            verbosity="INFO", 
            include_path=None, 
            model_name=None,
        )
        available_tasks = task_manager.all_tasks
        # available_tasks = task_manager.all_subtasks
        return available_tasks
    
    elif evaluation_engine == EvaluationEngine.lm_eval_harness:
        from lm_eval.tasks import TaskManager
        task_manager = TaskManager(
            verbosity="INFO",
            # metadata=None,
        )
        # ``all_tasks`` lists top-level entries only; ``all_subtasks`` covers
        # task-group children (e.g. siqa appears under a parent group on some
        # lm-evaluation-harness commits). Union both so the validation in
        # args.py accepts any registered task name.
        _tasks = set(getattr(task_manager, "all_tasks", []) or [])
        _tasks.update(getattr(task_manager, "all_subtasks", []) or [])
        return sorted(_tasks)
    
    elif evaluation_engine == EvaluationEngine.vlm_eval_kit:
        from vlmeval.dataset import DATASET_CLASSES
        from vlmeval.dataset.video_dataset_config import supported_video_datasets
        
        available_tasks = list()
        for cls in DATASET_CLASSES:
            available_tasks += cls.supported_datasets()
        available_tasks += list(supported_video_datasets.keys())
        return available_tasks
    
    else:
        raise ValueError(f'invalid evaluation_engine: {evaluation_engine}')

def get_custom_module(
    evaluation_engine: str,
    task_name: Optional[str] = None,
    module_path: Optional[str] = None,
) -> Optional[types.ModuleType]:
    """Import and return a custom task module, or None if not found."""
    if not (task_name or module_path):
        raise ValueError(f'Task_name or module_path should be given')
    custom_module = None
    if evaluation_engine == EvaluationEngine.builtin:
        if not module_path:
            module_path = f'omni_evaluator.evaluation.builtin.tasks.{task_name}.custom'
        try:
            if importlib.util.find_spec(module_path) is not None:
                custom_module = importlib.import_module(module_path)
        except ModuleNotFoundError as ex:
            pass
        except Exception as ex:
            raise
        return custom_module
    
    elif evaluation_engine == EvaluationEngine.lmms_eval:
        if not module_path:
            module_path = f'omni_evaluator.evaluation.lmms_eval.resources.custom_tasks.{task_name}.utils'
        try:
            if importlib.util.find_spec(module_path) is not None:
                custom_module = importlib.import_module(module_path)
        except ModuleNotFoundError as ex:
            pass
        except Exception as ex:
            raise
        return custom_module
    
    elif evaluation_engine == EvaluationEngine.lm_eval_harness:
        if not module_path:
            module_path = f'omni_evaluator.evaluation.lm_eval_harness.resources.custom_tasks.{task_name}.utils'
        custom_module = None
        try:
            if importlib.util.find_spec(module_path) is not None:
                custom_module = importlib.import_module(module_path)
        except ModuleNotFoundError as ex:
            pass
        except Exception as ex:
            raise
        return custom_module
    
    elif evaluation_engine == EvaluationEngine.vlm_eval_kit:
        return None

    else:
        raise ValueError(f'invalid evaluation_engine: {evaluation_engine}')

def validate_url(
    url: str, 
    protocol: str = "http",
    correction: bool = False
) -> str:
    """
    validate url whether url is complete
    correct url if correction is True

    Parameters:
        url (str): The URL to validate.
        protocol (str): The default protocol to use if the URL has no protocol. Default is "http".

    Returns:
        str: The formatted URL with the proper scheme.
    """
    
    parsed = urllib.parse.urlparse(url)

    # prepend the default protocol if no protocol is provided
    if parsed.scheme in ["http", "https", "tcp", ]:
        pass
    elif correction:
        logger.info(f'Append protocol since url is missing protocol: {protocol}')
        url = f'{protocol}://{url}'
        parsed = urllib.parse.urlparse(url)
    else:
        raise ValueError(f'url is missing protocol: {url}')

    # validate the url has a netloc (domain or host)
    if parsed.netloc:
        pass
    else:
        raise ValueError(f'url is missing domain or host: {url}')

    # reconstruct the url with the proper scheme
    url = urllib.parse.urlunparse(parsed)
    return url

def healthcheck(
    url: str,
    max_retries: int = 20,
    interval: int = 30,
    token: Optional[str] = None,
):
    """
    Sends a healthcheck request to the specified URL.

    Parameters:
        url (str): The URL to check.
        max_retries (int): Maximum number of retries before giving up. Default is 20.
        interval (int): Time (in seconds) to wait between retries. Default is 30.
        token (str, optional): Bearer token for authentication. Default is None.
    """

    url = validate_url(
        url=url,
        protocol="http",
        correction=True,
    )

    headers = None
    if token:
        headers = {"Authorization": f"Bearer {token}"}

    logger.info(f'Run healthcheck: {url}')
    cur_try = 1
    while cur_try <= max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=(5, 60))
            if response.status_code == 200:
                logger.info(f'Healthcheck succeeded: {url}')
                return True
            else:
                logger.warning(f'({cur_try:03}/{max_retries:03}) No response from {url}. Retrying in {interval} seconds.')
        except requests.RequestException as ex:
            logger.warning(f'({cur_try:03}/{max_retries:03}) Exception while healthcheck: {ex}. Retrying in {interval} seconds.')

        cur_try += 1
        time.sleep(interval)

    logger.error(f'Healthcheck failed: {url}')
    return False

def remove_stop_words(
    text: str,
    stop_words: List[str] = None,
):
    if isinstance(stop_words, (list, tuple)):
        for _stop_word in stop_words:
            if _stop_word not in text:
                continue
            text = text.split(_stop_word)[0]
            text = text.rstrip()
    return text

def set_seed(
    seed: Optional[int] = None,
    *,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    warn_only: bool = True,
) -> int:
    if seed is None:
        seed = secrets.randbits(32)
    elif not isinstance(seed, int):
        seed = int(seed)

    # 1) Python standard library
    random.seed(seed)
    # 2) Hash-based randomness (best applied before process start)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 3) NumPy
    try:
        np.random.seed(seed)
    except Exception:
        pass

    # 4) PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        # Fix seed for all CUDA devices when available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # CUDA matmul determinism (required in some environments)
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # cuDNN settings
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        torch.backends.cudnn.deterministic = bool(deterministic)
        # Force PyTorch deterministic algorithms
        # warn_only=True allows non-deterministic ops to emit warnings instead of errors
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(warn_only))
        # Disable TF32 for reproducibility (TF32 can introduce numerical differences)
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

    return seed