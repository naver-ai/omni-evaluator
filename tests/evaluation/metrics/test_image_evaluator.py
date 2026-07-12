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

"""Validates the purely deterministic surfaces of ImageEvaluator (compute_relative_position, ImageTransformDataset, compute_gen_eval) and actual model smoke tests."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("clip_benchmark", reason="image_evaluator requires the CLIP eval stack")
pytest.importorskip("open_clip", reason="image_evaluator requires open_clip")

from PIL import Image

from omni_evaluator.evaluation.metrics.image_evaluator import (
    ImageEvaluator,
    ImageTransformDataset,
)

pytestmark = pytest.mark.eval_engine("builtin")


# ─────────────────────────────────────────────────────────────────────────────
#  (B) compute_relative_position — pure numpy geometry (self not used)
# ─────────────────────────────────────────────────────────────────────────────

RELATIVE_POSITION = [
    pytest.param([10, 0, 12, 2], [0, 0, 2, 2], {"right of"}, id="right_of"),
    pytest.param([-12, 0, -10, 2], [0, 0, 2, 2], {"left of"}, id="left_of"),
    pytest.param([0, -12, 2, -10], [0, 0, 2, 2], {"above"}, id="above"),
    pytest.param([0, 10, 2, 12], [0, 0, 2, 2], {"below"}, id="below"),
    pytest.param([0, 0, 2, 2], [0, 0, 2, 2], set(), id="same_bbox_no_relation"),
]


@pytest.fixture
def evaluator():
    """Returns an ImageEvaluator instance without loading any model."""
    ev = ImageEvaluator.__new__(ImageEvaluator)
    ev.colors = ["red", "blue"]
    return ev


@pytest.mark.parametrize("target_bbox, reference_bbox, expected", RELATIVE_POSITION)
def test_compute_relative_position(evaluator, target_bbox, reference_bbox, expected):
    """Computes the left/right/above/below relation set from the sign of the center offset between two bboxes."""
    assert (
        evaluator.compute_relative_position(
            target_bbox=target_bbox, reference_bbox=reference_bbox
        )
        == expected
    )


# ─────────────────────────────────────────────────────────────────────────────
#  (B) ImageTransformDataset — pure PIL (no model/transform)
# ─────────────────────────────────────────────────────────────────────────────

def _image(width=10, height=8):
    return Image.new("RGB", (width, height))


def _mask(height=8, width=10):
    return np.zeros((height, width), dtype=np.uint8)


def test_dataset_len_matches_bbox_count():
    """multiplier = number of bboxes/masks; 1 if neither is provided."""
    ds = ImageTransformDataset(image=_image(), bbox_list=[[0, 0, 4, 4], [2, 2, 6, 6]], mask_list=[_mask(), _mask()])
    assert len(ds) == 2
    assert len(ImageTransformDataset(image=_image())) == 1


def test_dataset_getitem_crops_to_bbox():
    """`__getitem__` returns (image, 0) cropped to bbox (transform=None)."""
    ds = ImageTransformDataset(image=_image(), bbox_list=[[0, 0, 4, 4]], mask_list=[_mask()])
    cropped, label = ds[0]
    assert cropped.size == (4, 4)
    assert label == 0


def test_dataset_mask_shape_mismatch_raises():
    """Raises ValueError if mask shape differs from (image.height, image.width)."""
    with pytest.raises(ValueError):
        ImageTransformDataset(image=_image(), mask_list=[np.zeros((3, 3), dtype=np.uint8)])


def test_dataset_bbox_mask_length_mismatch_raises():
    """Raises ValueError if bbox_list and mask_list have different lengths."""
    with pytest.raises(ValueError):
        ImageTransformDataset(image=_image(), bbox_list=[[0, 0, 4, 4]], mask_list=[_mask(), _mask()])


# ─────────────────────────────────────────────────────────────────────────────
#  (C) compute_gen_eval — requirement matching (only detect_objects/detect_colors boundaries are mocked)
# ─────────────────────────────────────────────────────────────────────────────

def _detected_object(bbox):
    """Creates an object matching the shape returned by detect_objects."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    return {"bbox": np.array(bbox, dtype=float), "mask": mask, "binary_mask": mask > 0, "confidence": 0.9}


def _gen_eval(*, detected, include=None, exclude=None, colors_result=None):
    """Mocks only the detect_objects/detect_colors boundaries and runs compute_gen_eval once."""
    ev = ImageEvaluator.__new__(ImageEvaluator)
    ev.colors = ["red", "blue"]
    ev.detect_objects = lambda image, confidence_threshold=None: detected()
    if colors_result is not None:
        ev.detect_colors = lambda **kwargs: colors_result
    return ev.compute_gen_eval(
        image=Image.new("RGB", (20, 20)),
        include_requirements=include,
        exclude_requirements=exclude,
    )


def test_gen_eval_image_none_short_circuits():
    """Returns correct=False early without calling the model when image is falsy."""
    out = ImageEvaluator.__new__(ImageEvaluator).compute_gen_eval(image=None)
    assert out == {"correct": False, "reason": "failed to generate image while inference"}


@pytest.mark.parametrize(
    "count, expected_correct",
    [pytest.param(1, True, id="count_satisfied"), pytest.param(2, False, id="count_short")],
)
def test_gen_eval_count(count, expected_correct):
    """Whether include count is satisfied → correct."""
    out = _gen_eval(
        detected=lambda: {"cat": [_detected_object([0, 0, 2, 2])]},
        include=[{"class": "cat", "count": count}],
    )
    assert out["correct"] is expected_correct


@pytest.mark.parametrize(
    "exclude_class, expected_correct",
    [pytest.param("dog", False, id="excluded_present"), pytest.param("elephant", True, id="excluded_absent")],
)
def test_gen_eval_exclude(exclude_class, expected_correct):
    """correct=False when an excluded class appears at or above the threshold."""
    out = _gen_eval(
        detected=lambda: {"dog": [_detected_object([0, 0, 2, 2])]},
        exclude=[{"class": exclude_class, "count": 1}],
    )
    assert out["correct"] is expected_correct


@pytest.mark.parametrize(
    "relation, expected_correct",
    [pytest.param("right of", True, id="position_matches"), pytest.param("left of", False, id="position_wrong")],
)
def test_gen_eval_position(relation, expected_correct):
    """Position requirement is evaluated by the actual compute_relative_position (cat is to the right of table)."""
    out = _gen_eval(
        detected=lambda: {"table": [_detected_object([0, 0, 2, 2])], "cat": [_detected_object([10, 0, 12, 2])]},
        include=[
            {"class": "table", "count": 1},
            {"class": "cat", "count": 1, "position": (relation, 0)},
        ],
    )
    assert out["correct"] is expected_correct


@pytest.mark.parametrize(
    "detected_color, expected_correct",
    [pytest.param("red", True, id="color_matches"), pytest.param("blue", False, id="color_wrong")],
)
def test_gen_eval_color(detected_color, expected_correct):
    """Color requirement is compared against detect_colors (boundary mocked) result."""
    out = _gen_eval(
        detected=lambda: {"cat": [_detected_object([0, 0, 2, 2])]},
        include=[{"class": "cat", "count": 1, "color": "red"}],
        colors_result=[detected_color],
    )
    assert out["correct"] is expected_correct


# ─────────────────────────────────────────────────────────────────────────────
#  Real model path smoke — runs Mask2Former/CLIP for real without mocks.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_evaluator():
    """Returns an ImageEvaluator with real models loaded; skips if loading fails."""
    try:
        return ImageEvaluator()
    except Exception as exc:  # noqa: BLE001 — environment gate for network/cache/weight load failures
        pytest.skip(f"could not load ImageEvaluator models: {type(exc).__name__}: {exc}")


def _photo():
    """Returns a minimal RGB image with a rectangle drawn on a solid background."""
    from PIL import ImageDraw

    img = Image.new("RGB", (320, 240), (120, 120, 120))
    ImageDraw.Draw(img).rectangle([40, 40, 200, 200], fill=(200, 30, 30))
    return img


def test_detect_objects_smoke_returns_dict(real_evaluator):
    """detect_objects runs to completion and returns a dict (empty dict is also valid)."""
    out = real_evaluator.detect_objects(image=_photo())
    assert isinstance(out, dict)
    for _cls, instances in out.items():
        assert isinstance(instances, list)
        for inst in instances:  # check structure only when detections exist
            assert set(inst) >= {"bbox", "mask", "binary_mask", "confidence"}
            assert np.asarray(inst["bbox"]).shape == (4,)
            assert isinstance(float(inst["confidence"]), float)


def test_detect_colors_smoke_returns_color_list(real_evaluator):
    """detect_colors (CLIP) runs to completion and returns a label list within the colors candidates."""
    out = real_evaluator.detect_colors(
        image=_photo(), classname="object", bbox_list=None, mask_list=None
    )
    assert isinstance(out, list) and len(out) >= 1
    assert all(color in real_evaluator.colors for color in out)


def test_compute_gen_eval_smoke_real_pipeline(real_evaluator):
    """compute_gen_eval passes through actual detect_objects and returns the promised result dict."""
    out = real_evaluator.compute_gen_eval(
        image=_photo(), include_requirements=[{"class": "person", "count": 1}]
    )
    assert set(out) >= {"correct", "reason", "detected_objects"}
    assert isinstance(out["correct"], bool)
    assert isinstance(out["reason"], list)
    assert isinstance(out["detected_objects"], list)
