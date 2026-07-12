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

"""Test case data for `omni_evaluator/postprocess/spatial_grounding`."""

import pytest


# ── Source detection (bracket group, richest-first) ────────────────────────
# 4-num bracket → bbox source. Prose / parentheses / braces / spacing variants.
SG_BBOX_BRACKETS = [
    pytest.param("[0.1, 0.2, 0.3, 0.4]", id="en_plain_brackets"),
    pytest.param("The bounding box is approximately [0.1, 0.2, 0.3, 0.4].", id="en_prose_brackets"),
    pytest.param("정답: [0.1, 0.2, 0.3, 0.4]", id="ko_prose_brackets"),
    pytest.param("(0.1, 0.2, 0.3, 0.4)", id="en_parens"),
    pytest.param("{0.1, 0.2, 0.3, 0.4}", id="en_braces"),
    pytest.param("[ 0.1 , 0.2 , 0.3 , 0.4 ]", id="en_extra_spacing"),
]

# 2-num bracket → point source.
SG_POINT_BRACKETS = [
    pytest.param("[0.5, 0.5]", id="en_plain"),
    pytest.param("The point is at [0.2, 0.8].", id="en_prose"),
    pytest.param("좌표: [0.5, 0.5]", id="ko_prose"),
]

# 8-num bracket → quad source.
SG_QUAD_BRACKETS = [
    pytest.param("[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]", id="en_unit_square"),
    pytest.param("Quad: [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]", id="en_prose"),
]

# Richest-first preference — text contains BOTH a quad and a bbox; quad wins.
SG_RICHEST_FIRST = [
    pytest.param(
        "axis-aligned box [0.1, 0.2, 0.3, 0.4] vs quad [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]",
        "quad",
        id="quad_over_bbox",
    ),
]

# ── Bare-number fallback (no bracket group present) ────────────────────────
# When NO bracket match exists, the last `target_arity` bare numbers are taken
# AS the target shape (no shape detection).
SG_BARE_FALLBACK_BBOX = [
    pytest.param("the answer is 0.1 0.2 0.3 0.4", id="en_bare_4nums"),
    pytest.param("답은 0.1, 0.2, 0.3, 0.4", id="ko_bare_4nums"),
]

# ── No answer / non-string / empty ──────────────────────────────────────────
SG_NO_ANSWER = [
    pytest.param("I cannot determine the bounding box.", id="en_refusal"),
    pytest.param("모르겠습니다.", id="ko_refusal"),
]

SG_NON_STRING = [
    pytest.param(None, id="none"),
    pytest.param(123, id="int"),
    pytest.param([], id="list"),
]

SG_EMPTY = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace"),
]

# ── NFKC normalize / `<think>` strip ────────────────────────────────────────
SG_NFKC_FULLWIDTH = [
    # Fullwidth digits → halfwidth via NFKC; brackets/comma also fullwidth.
    pytest.param("［０．１，０．２，０．３，０．４］", id="fullwidth"),
]

SG_THINK_STRIPPED = [
    pytest.param(
        "<think>let me think 0.9 0.9 0.9 0.9</think>\n[0.1, 0.2, 0.3, 0.4]",
        id="cot_trace_then_answer",
    ),
]

# ── Cross-shape conversion (text + version_name → expected output) ──────────
# Each case picks a source-shape text and a target version_name; the expected
# output string is the canonical form the metric parses.
SG_CROSS_SHAPE = [
    # bbox source
    pytest.param("[0.1, 0.2, 0.3, 0.4]", "bbox", "[0.1, 0.2, 0.3, 0.4]", id="bbox_to_bbox"),
    pytest.param("[0.0, 0.0, 1.0, 1.0]", "point", "[0.5, 0.5]", id="bbox_to_point_center"),
    pytest.param("[0.0, 0.0, 1.0, 1.0]", "quad", "[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]", id="bbox_to_quad_corners"),
    # quad source
    pytest.param("[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]", "bbox", "[0.0, 0.0, 1.0, 1.0]", id="quad_to_bbox_aabb"),
    pytest.param("[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]", "point", "[0.5, 0.5]", id="quad_to_point_centroid"),
    pytest.param("[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]", "quad", "[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]", id="quad_to_quad"),
    # point source — identity only
    pytest.param("[0.5, 0.5]", "point", "[0.5, 0.5]", id="point_to_point"),
]

# point → bbox / point → quad: no extent info → None.
SG_POINT_NO_EXTENT = [
    pytest.param("[0.5, 0.5]", "bbox", id="point_to_bbox_none"),
    pytest.param("[0.5, 0.5]", "quad", id="point_to_quad_none"),
]

# ── version_name parsing ────────────────────────────────────────────────────
SG_VERSION_NAME_VALID = [
    pytest.param(None, id="default_bbox"),
    pytest.param("bbox", id="str_bbox"),
    pytest.param("BBOX", id="upper_bbox"),
    pytest.param("point", id="str_point"),
    pytest.param("quad", id="str_quad"),
]

SG_VERSION_NAME_INVALID = [
    pytest.param("polygon", id="unknown"),
    pytest.param("BBOX2D", id="unknown_upper"),
]

# Note: the processor is parsing-only and emits RAW parsed coords (no input-
# scale auto-detect, no output_scale rescale). Coordinate-space normalization
# moved to the metric (`compute_iou` / `compute_click_dist_accuracy` via
# `_grounding_norm_axis`), so those cases are tested there, not here.

# ── Dataset-grounded post-processing (real model output forms) ──────────────
# Prediction strings below are the shapes models actually emit on each task
# family; they assert the postprocess output the metric will receive.

# refcoco* (version_name="bbox"): Qwen2.5-Omni-3B emits a normalized [0,1] bbox,
# sometimes wrapped in prose / bare. Output is the raw bbox, unchanged.
SG_REFCOCO_BBOX = [
    pytest.param("[0.41, 0.33, 0.65, 0.77]", "[0.41, 0.33, 0.65, 0.77]", id="plain_unit_bbox"),
    pytest.param("[0.34, 0.29, 0.51, 0.56]", "[0.34, 0.29, 0.51, 0.56]", id="plain_unit_bbox2"),
    pytest.param("The bounding box is [0.24, 0.16, 0.42, 0.71].", "[0.24, 0.16, 0.42, 0.71]", id="prose_bbox"),
    pytest.param("0.5, 0.5, 0.8, 0.9", "[0.5, 0.5, 0.8, 0.9]", id="bare_bbox_fallback"),
]

# gui_*_test (version_name="point"): models emit PIXEL coords in many shapes;
# bbox/box are reduced to their center. Output is RAW pixels (metric divides by
# meta.image_w/h downstream). Shapes mirror the gui_360 / groundcua / gta1 /
# jedi / scalecua debug runs.
SG_GUI_POINT = [
    pytest.param('<point x1="179" y1="255" alt="select AutoNum">...</point>', "[179.0, 255.0]", id="point_attr_singular"),
    pytest.param('<points x1="481" y1="427" alt="connect with us.">...</points>', "[481.0, 427.0]", id="point_attr_plural"),
    pytest.param("<point>(786, 179)</point>", "[786.0, 179.0]", id="point_tag_paren"),
    pytest.param("click(x=156, y=100)", "[156.0, 100.0]", id="action_click"),
    pytest.param("<bbox>(466, 175, 510, 198)</bbox>", "[488.0, 186.5]", id="bbox_to_center"),
    pytest.param("<box>[[124, 82, 222, 157]]</box>", "[173.0, 119.5]", id="scalecua_box_to_center"),
    pytest.param('{"coordinate": [123, 100]}', "[123.0, 100.0]", id="tool_call_coord"),
    pytest.param("<point><x>51, 315</x></point>", "[51.0, 315.0]", id="point_nested_x"),
]
