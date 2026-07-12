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

"""Unit tests for omni_evaluator/postprocess/spatial_grounding/__init__.py."""

import pytest

from omni_evaluator.enums import SpatialGroundingType
from omni_evaluator.postprocess.spatial_grounding import SpatialGroundingProcessor

from ._cases import (
    SG_BARE_FALLBACK_BBOX,
    SG_BBOX_BRACKETS,
    SG_CROSS_SHAPE,
    SG_EMPTY,
    SG_GUI_POINT,
    SG_NFKC_FULLWIDTH,
    SG_NON_STRING,
    SG_NO_ANSWER,
    SG_POINT_BRACKETS,
    SG_POINT_NO_EXTENT,
    SG_QUAD_BRACKETS,
    SG_REFCOCO_BBOX,
    SG_RICHEST_FIRST,
    SG_THINK_STRIPPED,
    SG_VERSION_NAME_INVALID,
    SG_VERSION_NAME_VALID,
)


# `tests/postprocess/CLAUDE.md §8` — regex catastrophic-backtracking guard.
pytestmark = pytest.mark.timeout(1)


# ============================================================================
# Source detection — bracket arity decides source shape (richest-first).
# ============================================================================


@pytest.mark.parametrize("text", SG_BBOX_BRACKETS)
def test_bbox_bracket_detected(text):
    """4-num bracket inside prose / paren / brace / spacing variants → bbox source."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name="bbox") == "[0.1, 0.2, 0.3, 0.4]"


@pytest.mark.parametrize("text", SG_POINT_BRACKETS)
def test_point_bracket_detected(text):
    """2-num bracket → point source (no shape conversion needed for point target)."""
    out = SpatialGroundingProcessor.extract(prediction=text, version_name="point")
    assert out is not None and out.startswith("[") and out.endswith("]")


@pytest.mark.parametrize("text", SG_QUAD_BRACKETS)
def test_quad_bracket_detected(text):
    """8-num bracket → quad source preserved when target=quad."""
    out = SpatialGroundingProcessor.extract(prediction=text, version_name="quad")
    assert out is not None and out.count(",") == 7   # 8 nums separated by 7 commas


@pytest.mark.parametrize("text,winning_source", SG_RICHEST_FIRST)
def test_richest_first_wins(text, winning_source):
    """When both quad and bbox brackets exist, quad wins (richest source → more info downstream)."""
    # quad → bbox convert yields a different aabb than the embedded bbox.
    out = SpatialGroundingProcessor.extract(prediction=text, version_name="bbox")
    assert out == "[0.0, 0.0, 1.0, 1.0]"   # aabb of the quad, not the embedded bbox


@pytest.mark.parametrize("text", SG_BARE_FALLBACK_BBOX)
def test_bare_number_fallback(text):
    """No bracket → last `target_arity` bare numbers taken AS target shape."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name="bbox") == "[0.1, 0.2, 0.3, 0.4]"


@pytest.mark.parametrize("text", SG_NO_ANSWER)
def test_no_answer_returns_none(text):
    """No bracket and not enough bare numbers → None."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name="bbox") is None


# ============================================================================
# Non-string / empty input — early returns.
# ============================================================================


@pytest.mark.parametrize("value", SG_NON_STRING)
def test_non_string_passthrough(value):
    """Non-string input is returned unchanged (no extraction attempted)."""
    assert SpatialGroundingProcessor.extract(prediction=value) == value


@pytest.mark.parametrize("text", SG_EMPTY)
def test_empty_returns_none(text):
    """Empty / whitespace-only string → None."""
    assert SpatialGroundingProcessor.extract(prediction=text) is None


# ============================================================================
# NFKC normalize + `<think>` strip — text preprocessing.
# ============================================================================


@pytest.mark.parametrize("text", SG_NFKC_FULLWIDTH)
def test_nfkc_normalizes_fullwidth(text):
    """Fullwidth digits / brackets / comma are halfwidth-normalized before regex."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name="bbox") == "[0.1, 0.2, 0.3, 0.4]"


@pytest.mark.parametrize("text", SG_THINK_STRIPPED)
def test_think_trace_stripped(text):
    """Coordinates inside `<think>...</think>` are ignored; the post-trace answer wins."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name="bbox") == "[0.1, 0.2, 0.3, 0.4]"


# ============================================================================
# Cross-shape conversion — source vs target.
# ============================================================================


@pytest.mark.parametrize("text,version,expected", SG_CROSS_SHAPE)
def test_cross_shape_convert(text, version, expected):
    """bbox/quad/point cross-shape conversion produces canonical metric-ready output."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name=version) == expected


@pytest.mark.parametrize("text,version", SG_POINT_NO_EXTENT)
def test_point_to_extent_returns_none(text, version):
    """point → bbox / point → quad lacks extent information → None."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name=version) is None


# ============================================================================
# version_name parsing — enum / str / invalid / None.
# ============================================================================


@pytest.mark.parametrize("version_name", SG_VERSION_NAME_VALID)
def test_version_name_resolves(version_name):
    """None / str (any case) / enum value all resolve to a SpatialGroundingType."""
    # Use a bbox-source text that is valid for every target shape.
    out = SpatialGroundingProcessor.extract(prediction="[0.0, 0.0, 1.0, 1.0]", version_name=version_name)
    assert out is not None


def test_version_name_enum_passthrough():
    """SpatialGroundingType enum passed directly is honored."""
    out = SpatialGroundingProcessor.extract(
        prediction="[0.0, 0.0, 1.0, 1.0]", version_name=SpatialGroundingType.POINT,
    )
    assert out == "[0.5, 0.5]"


@pytest.mark.parametrize("version_name", SG_VERSION_NAME_INVALID)
def test_version_name_invalid_falls_back_to_bbox(version_name):
    """Unknown version_name string → warning + BBOX default (still emits a 4-num bbox)."""
    out = SpatialGroundingProcessor.extract(prediction="[0.1, 0.2, 0.3, 0.4]", version_name=version_name)
    assert out == "[0.1, 0.2, 0.3, 0.4]"


# ============================================================================
# Raw-coordinate emission — the processor no longer rescales.
# ============================================================================


def test_emits_raw_coords():
    """Parsing-only: coords are emitted in the model's native scale (no rescale).
    Normalization moved to the metric, so a pixel-scale input passes through unchanged."""
    assert SpatialGroundingProcessor.extract(
        prediction="[100, 200, 300, 400]", version_name="bbox",
    ) == "[100.0, 200.0, 300.0, 400.0]"


# ============================================================================
# Dataset-grounded post-processing — real model output forms.
# ============================================================================


@pytest.mark.parametrize("text,expected", SG_REFCOCO_BBOX)
def test_refcoco_bbox_postprocess(text, expected):
    """refcoco* (version_name=bbox): a normalized [0,1] bbox is emitted unchanged."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name="bbox") == expected


@pytest.mark.parametrize("text,expected", SG_GUI_POINT)
def test_gui_point_postprocess(text, expected):
    """gui_*_test (version_name=point): pixel coords in any shape reduce to a RAW
    point (bbox/box → center); the metric normalizes later via meta.image_w/h."""
    assert SpatialGroundingProcessor.extract(prediction=text, version_name="point") == expected


# ============================================================================
# api_name — LLM fallback is intentionally unimplemented.
# ============================================================================


def test_api_name_raises_not_implemented():
    """Non-empty api_name raises NotImplementedError to prevent silent fallthrough."""
    with pytest.raises(NotImplementedError):
        SpatialGroundingProcessor.extract(prediction="[0.1, 0.2, 0.3, 0.4]", api_name="gpt-4o-mini")
