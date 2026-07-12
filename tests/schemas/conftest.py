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

"""schemas area common fixtures — baseline for provider template conversion, multimodal serialization, and yaml round-trip validation."""

import pytest


@pytest.fixture
def tiny_image():
    """2x2 solid-color PIL image — baseline for PIL source serialization (base64 conversion)."""
    from PIL import Image

    return Image.new("RGB", (2, 2), (255, 0, 0))


@pytest.fixture
def sample_text_message_dict():
    """Simplest user text message — baseline for provider template conversion."""
    return {"role": "user", "content": [{"type": "text", "value": "Hello"}]}


@pytest.fixture
def sample_image_message_dict():
    """image+text user message — baseline for multimodal serialization."""
    return {
        "role": "user",
        "content": [
            {"type": "image", "value": "https://example.com/img.png"},
            {"type": "text", "value": "Describe."},
        ],
    }


@pytest.fixture
def sample_task_config_dict():
    """Minimal dict form of TaskConfig — baseline for yaml round-trip validation."""
    return {
        "task_name": "dummy",
        "evaluation_engine": "builtin",
        "num_records": 1,
        "meta": {"benchmark_name": "dummy", "output_modality": ["text"]},
        "dataset": {"source": "local"},
        "evaluation": {"method": "generation", "target_metrics": ["exact_match"]},
    }
