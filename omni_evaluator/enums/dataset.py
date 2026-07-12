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

from enum import Enum


class DatasetSource(str, Enum):
    local: str = "local"            # located in local file system
    s3: str = "s3"
    huggingface_hub: str = "huggingface_hub"
    package: str = "package"        # included in python package
    resources: str = "resources"    # included in repo


class CombineMethod(str, Enum):
    """Strategy for combining multiple HF subsets into a single dataset."""
    concatenate: str = "concatenate"   # row-wise concat (vertical), like pandas pd.concat(axis=0)
    join: str = "join"                 # key-based column join, like pandas pd.merge(on=...)
