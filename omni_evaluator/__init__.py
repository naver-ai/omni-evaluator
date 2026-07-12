# datasets ≥ 4.x forwards `num_channels` to torchcodec's `AudioDecoder.__init__`,
# but some installed torchcodec builds (or stale module copies) don't accept it
# and raise `TypeError: AudioDecoder.__init__() got an unexpected keyword
# argument 'num_channels'`. Patch the shim lazily — defer until datasets is
# first imported, so `import omni_evaluator` does not pull in datasets/torch.

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

import importlib.util
import sys


def _apply_torchcodec_compat_patch() -> None:
    try:
        from datasets.features import _torchcodec as _mod
        _orig = _mod.AudioDecoder

        class _CompatibleAudioDecoder(_orig):
            def __init__(self, *args, **kwargs):
                try:
                    super().__init__(*args, **kwargs)
                except TypeError as ex:
                    if "num_channels" in str(ex) and "num_channels" in kwargs:
                        kwargs.pop("num_channels", None)
                        super().__init__(*args, **kwargs)
                    else:
                        raise

        _mod.AudioDecoder = _CompatibleAudioDecoder
    except Exception:
        pass


class _DatasetsPatchFinder:
    """sys.meta_path hook: applies the torchcodec compat patch once, right after datasets loads."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "datasets":
            return None
        # Remove ourselves first to avoid recursion when find_spec re-enters.
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None:
            return None
        real_loader = spec.loader

        class _PatchingLoader:
            def create_module(self, s):
                cm = getattr(real_loader, "create_module", None)
                return cm(s) if cm is not None else None

            def exec_module(self, module):
                real_loader.exec_module(module)
                _apply_torchcodec_compat_patch()

        spec.loader = _PatchingLoader()
        return spec


if "datasets" in sys.modules:
    # datasets was already imported before omni_evaluator — patch immediately.
    _apply_torchcodec_compat_patch()
else:
    sys.meta_path.insert(0, _DatasetsPatchFinder())


# Re-export Enum namespaces from `omni_evaluator.enums.*` so legacy callers can
# still do `from omni_evaluator import EvaluationEngine, ...`. New code should
# import from the leaf modules (e.g. `omni_evaluator.enums.engine`) directly.
from omni_evaluator.enums.dataset import CombineMethod, DatasetSource
from omni_evaluator.enums.engine import (
    ApiGroup,
    EvaluationEngine,
    EvaluationMethod,
    HuggingfaceModelGroup,
    InferenceEngine,
    T2IGeneratorType,
)
from omni_evaluator.enums.evaluation import NullPredictionPolicy
from omni_evaluator.enums.media import AudioFormat, ImageFormat, Modality, VideoFormat
from omni_evaluator.enums.task import SubtaskType, TaskType
