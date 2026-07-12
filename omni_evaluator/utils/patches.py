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

import contextlib
import importlib
import inspect
import logging
import os
from pathlib import Path, PosixPath
import shutil
import sys
import types
from typing import Any, Dict, Generator, List, Union, Tuple, Optional, Callable, Iterable

logger = logging.getLogger(__name__)


def update_package_resources(
    package_name: str,
    source_dirpath: Union[str, PosixPath],
    target_dirpath: Optional[str] = None,
) -> int:        
    _engine_spec = importlib.util.find_spec(package_name)
    if (
        _engine_spec is None 
        or len(_engine_spec.submodule_search_locations) < 1
    ):
        raise ImportError(f'package not installed: {package_name}')
    package_dirpath = _engine_spec.submodule_search_locations[0]
    if isinstance(target_dirpath, str):
        package_dirpath = os.path.join(package_dirpath, target_dirpath)
    
    if isinstance(source_dirpath, PosixPath):
        source_dirpath = Path.as_posix(source_dirpath)
    dirnames = list()
    if os.path.exists(source_dirpath):
        dirnames = os.listdir(source_dirpath)
    
    num_copied = 0
    if len(dirnames) > 0:
        logger.info(f'Update package resources: {package_name}')
        for _dirname in dirnames:
            _source_dirpath = os.path.join(source_dirpath, _dirname)
            if not os.path.isdir(_source_dirpath):
                continue
            if _dirname.startswith('example_'):
                logger.debug(f'Skipping example directory: {_dirname}')
                continue
            _target_dirpath = os.path.join(package_dirpath, _dirname)
            shutil.copytree(
                _source_dirpath, 
                _target_dirpath,
                dirs_exist_ok=True,
            )
            logger.info(f'Copied `{_dirname}`: {_target_dirpath}')
            num_copied += 1
    else:
        logger.info(f'No package resources to update: {package_name}')
    return num_copied


def _get_module(
    module_name: str,
    parent_module: Callable = None,
    full_path: Optional[str] = None,
):
    # ``full_path`` preserves the original dotted name through recursion so the
    # leaf step can fall back to ``sys.modules`` when an ``__init__.py``
    # re-exports a callable that shadows a submodule name (e.g. a package's
    # ``__init__.py: from .evaluate import evaluate`` makes ``pkg.evaluate``
    # resolve to a function rather than the submodule, blocking setattr).
    if full_path is None:
        full_path = module_name
    module_name = module_name.split(".")
    cur_module_name, left_module_name = module_name[0], module_name[1:]

    cur_module = None
    if parent_module is None:
        cur_module = sys.modules.get(cur_module_name, None)
    else:
        cur_module = getattr(parent_module, cur_module_name, None)

    if cur_module is None:
        # Fallback: dotted path may be a submodule already loaded in sys.modules
        if full_path in sys.modules:
            return sys.modules[full_path]
        raise AssertionError(f'module not exist: {full_path}')
    elif len(left_module_name) < 1:
        # Leaf step: if attribute traversal landed on a non-module (e.g. function
        # re-exported by parent __init__.py), prefer sys.modules' actual submodule.
        import types
        if not isinstance(cur_module, types.ModuleType) and full_path in sys.modules:
            return sys.modules[full_path]
        return cur_module
    else:
        return _get_module(
            module_name=".".join(module_name[1:]).strip(),
            parent_module=cur_module,
            full_path=full_path,
        )
        

@contextlib.contextmanager
def patch_module(
    module_name: str,
    module: Callable,
) -> Generator[None, None, None]:
    module_name = module_name.split(".")
    if len(module_name) < 1:
        raise ValueError(f'Invalid module_name: {module_name}')
    
    elif len(module_name) == 1: # top-level module
        module_name = module_name[0]
        original_module = sys.modules.get(module_name, None)
        sys.modules[module_name] = module
        try:
            yield
        finally:
            sys.modules[module_name] = original_module
    
    else: # submodule
        parent_module = _get_module(
            module_name=".".join(module_name[:-1]).strip(),
            parent_module=None,
        )
        original_module = getattr(parent_module, module_name[-1], None)
        setattr(parent_module, module_name[-1], module)
        try:
            yield
        finally:
            setattr(parent_module, module_name[-1], original_module)

@contextlib.contextmanager
def patch_function(
    module_name: str,
    func_name: str,
    func: Callable,
) -> Generator[None, None, None]:
    module = _get_module(module_name)
    if module is None:
        raise RuntimeError(f'Module not exist: {module_name}')
    original_func = getattr(module, func_name, None)
    setattr(module, func_name, func)
    try:
        yield
    finally:
        setattr(module, func_name, original_func)

@contextlib.contextmanager
def patch_instance_method(
    instance: Any,
    method_name: str,
    method: Callable,
) -> Generator[None, None, None]:
    original_method = getattr(instance, method_name)
    setattr(instance, method_name, types.MethodType(method, instance))
    try:
        yield
    finally:
        setattr(instance, method_name, original_method)

@contextlib.contextmanager
def patch_envs(envs: Dict[str, str]) -> Generator[None, None, None]:
    envs_backup = {k: os.environ.get(k, None) for k in envs}
    os.environ.update(envs)
    try:
        yield
    finally:
        for k, v in envs_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

@contextlib.contextmanager
def inject_partial_local(name: str, new_partial_func: Any, scope: dict) -> Generator[None, None, None]:
    """
    Temporarily override a name binding in the given scope with a partial function.
    scope: globals() or locals()
    """
    original = scope.get(name)
    scope[name] = new_partial_func
    try:
        yield
    finally:
        scope[name] = original

class ClassPatcher:
    """
    patcher = ClassPatcher(A)
    with patcher.patch_temporarily('greet', new_greet):
        c = A()
        print(c.greet()) 
    """
    def __init__(self, cls: type) -> None:
        self.cls = cls
        self._original_methods = dict()

    def patch(self, method_name: str, new_method: Callable) -> None:
        if method_name not in self._original_methods:
            self._original_methods[method_name] = getattr(self.cls, method_name)
        setattr(self.cls, method_name, new_method)

    def undo(self) -> None:
        for method_name, original_method in self._original_methods.items():
            setattr(self.cls, method_name, original_method)
        self._original_methods.clear()

    @contextlib.contextmanager
    def patch_temporarily(self, method_name: str, new_method: Callable) -> Generator[None, None, None]:
        self.patch(method_name, new_method)
        try:
            yield
        finally:
            self.undo()
            
def is_function(obj: Any) -> bool:
    return (
        inspect.isfunction(obj) 
        or inspect.ismethod(obj) 
        or inspect.isbuiltin(obj)
    )

def is_variable(obj: Any) -> bool:
    return (
        not callable(obj) 
        and not inspect.ismodule(obj) 
        and not inspect.isclass(obj)
    )