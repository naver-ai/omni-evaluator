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

"""Common fixtures shared by all inference engine tests."""
from __future__ import annotations

from typing import List

import pytest

LIVE_SMOKE_IMAGE_URL = (
    "https://images.unsplash.com/photo-1506744038136-46273834b3fb"
    "?auto=format&fit=crop&w=1200&q=80"
)
LIVE_SMOKE_VIDEO_URL = "https://www.w3schools.com/html/mov_bbb.mp4"

@pytest.fixture
def record_factory():
    """Minimal Record list builder consisting of a single text-only user turn."""
    from omni_evaluator.schemas.chat import Message
    from omni_evaluator.schemas.inference import Record

    def _factory(n: int = 1, query: str = "hello") -> List:
        records = []
        for i in range(n):
            records.append(Record(
                benchmark="dummy",
                index=str(i),
                messages=[Message(role="user", content=query)],
            ))
        return records
    return _factory


@pytest.fixture
def image_record_factory():
    """Record builder with image+text content."""
    from PIL import Image
    from omni_evaluator.schemas.chat import Message
    from omni_evaluator.schemas.inference import Record

    def _factory(n: int = 1, query: str = "Describe this image."):
        records = []
        for i in range(n):
            img = Image.new("RGB", (32, 32), color="red")
            records.append(Record(
                benchmark="dummy", index=str(i),
                messages=[Message(role="user", content=[
                    {"type": "image", "value": img},
                    {"type": "text", "value": query},
                ])],
            ))
        return records
    return _factory



@pytest.fixture
def image_record_url_factory():
    """Record builder with image+text content — value is a public URL string."""
    from omni_evaluator.schemas.chat import Message
    from omni_evaluator.schemas.inference import Record

    def _factory(n: int = 1, query: str = "Describe this image."):
        records = []
        for i in range(n):
            records.append(Record(
                benchmark="dummy", index=str(i),
                messages=[Message(role="user", content=[
                    {"type": "image", "value": LIVE_SMOKE_IMAGE_URL},
                    {"type": "text", "value": query},
                ])],
            ))
        return records
    return _factory

@pytest.fixture
def video_record_factory():
    """Record builder with video+text content."""
    import numpy as np
    from omni_evaluator.schemas.chat import Message
    from omni_evaluator.schemas.inference import Record

    def _factory(n: int = 1, query: str = "Describe this video."):
        video = np.zeros((4, 32, 32, 3), dtype=np.uint8)  # 4-frame dummy
        records = []
        for i in range(n):
            records.append(Record(
                benchmark="dummy", index=str(i),
                messages=[Message(role="user", content=[
                    {"type": "video", "value": video},
                    {"type": "text", "value": query},
                ])],
            ))
        return records
    return _factory

@pytest.fixture
def video_record_url_factory():
    """Record builder with video+text content — value is a public URL string."""
    from omni_evaluator.schemas.chat import Message
    from omni_evaluator.schemas.inference import Record

    def _factory(n: int = 1, query: str = "Describe this video."):
        records = []
        for i in range(n):
            records.append(Record(
                benchmark="dummy", index=str(i),
                messages=[Message(role="user", content=[
                    {"type": "video", "value": LIVE_SMOKE_VIDEO_URL},
                    {"type": "text", "value": query},
                ])],
            ))
        return records
    return _factory


@pytest.fixture
def task_config_factory():
    """Build a minimal TaskConfig."""
    from omni_evaluator import DatasetSource
    from omni_evaluator.schemas.task import TaskConfig, TaskDataset, TaskMeta

    def _factory(num_records: int = 1, evaluation_engine: str = "builtin"):
        return TaskConfig(
            task_name="dummy",
            evaluation_engine=evaluation_engine,
            num_records=num_records,
            meta=TaskMeta(benchmark_name="dummy"),
            dataset=TaskDataset(source=DatasetSource.local),
        )
    return _factory
