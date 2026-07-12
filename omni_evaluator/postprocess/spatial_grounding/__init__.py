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

import functools
import logging
import re
import unicodedata
from typing import List, Optional, Tuple, Union

from omni_evaluator.enums import SpatialGroundingType
from omni_evaluator.postprocess._interface import ProcessorInterface

logger = logging.getLogger(__name__)


# arity (number of floats) per grounding shape
_ARITY = {
    SpatialGroundingType.BBOX: 4,
    SpatialGroundingType.POINT: 2,
    SpatialGroundingType.QUAD: 8,
}


class SpatialGroundingProcessor(ProcessorInterface):
    """Extract a spatial grounding answer and emit it in the requested output shape.

    Models answering grounding tasks (refcoco / refcocog / refcoco+ / Kosmos /
    custom pointing) wrap the predicted coordinates in prose, e.g.
        "The bounding box coordinates ... are approximately [0.05, 0.35, 0.26, 0.99]."
    while the downstream ``compute_iou`` parses ``box2`` with a strict
    ``box2.strip("[]").split(", ")`` and silently scores 0 (Exception → iou=0)
    when the string is anything but a bare ``[v1, v2, ...]``. This processor
    pulls coordinates out and re-emits the canonical form ``compute_iou`` can
    parse.

    **Cross-shape conversion:** the model may output a *different* shape
    than the task expects — e.g. a 4-point quad while the task scores against
    a bbox. ``version_name`` (a value of :class:`SpatialGroundingType`,
    default ``BBOX``) is the shape the metric expects. The source shape is
    auto-detected from the text (richest match wins: quad → bbox → point) and
    converted to the target before emission.

    Conversion matrix (``None`` = insufficient information, returns ``None``):

    ====================  ====================================
    source → target       conversion
    ====================  ====================================
    bbox  → bbox          identity
    bbox  → point         center: ``((x1+x2)/2, (y1+y2)/2)``
    bbox  → quad          4 corners (CW from top-left)
    quad  → bbox          axis-aligned bbox: ``min/max`` of x,y
    quad  → point         centroid: ``(mean(xs), mean(ys))``
    quad  → quad          identity
    point → point         identity
    point → bbox          **None** — no extent information
    point → quad          **None** — no extent information
    ====================  ====================================

    Spatial only — temporal grounding (Charades-STA etc.) uses
    :class:`TemporalGroundingProcessor`.
    """

    _NUM = r"[-+]?(?:\d+\.\d+|\.\d+|\d+)"

    # Pattern factories — each compiles its ``re.Pattern`` on first call and
    # then caches it (functools.lru_cache).  Keyed by (cls, *args), so
    # subclasses that override ``_NUM`` get their own entries.
    @classmethod
    @functools.lru_cache(maxsize=None)
    def _bracket_pattern(cls, n: int) -> re.Pattern:
        body = r"\s*,\s*".join([rf"({cls._NUM})"] * n)
        return re.compile(r"[\[\(\{]\s*" + body + r"\s*[\]\)\}]")

    # Non-bracket point-producing patterns

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _point_attr_pattern(cls) -> re.Pattern:
        """HTML/SVG-attribute style — ui-tars-1.5-7b, Qwen2.5-Omni-3B:
        ``<points x1='580' y1='701' alt='...'>...</points>`` /
        ``<point x1='..' y1='..'>``."""
        return re.compile(
            rf"<points?\b[^>]*?\bx1\s*=\s*['\"]?({cls._NUM})['\"]?"
            rf"[^>]*?\by1\s*=\s*['\"]?({cls._NUM})['\"]?",
            re.IGNORECASE,
        )

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _point_nested_x_pattern(cls) -> re.Pattern:
        """Nested-tag variant: ``<point><x>N, N</x></point>``."""
        return re.compile(
            rf"<point>\s*<x>\s*({cls._NUM})\s*[,\s]\s*({cls._NUM})\s*</x>",
            re.IGNORECASE,
        )

    @classmethod
    @functools.lru_cache(maxsize=None)
    def _action_click_pattern(cls) -> re.Pattern:
        """Pyautogui-style action call:
        ``click(x=N, y=N)`` / ``pyautogui.click(x=N, y=N)`` /
        ``left_click(...)`` / ``tap(...)``."""
        return re.compile(
            rf"(?:click|pyautogui\.click|left_click|tap)"
            rf"\s*\(\s*x\s*=\s*({cls._NUM})\s*,\s*y\s*=\s*({cls._NUM})\s*\)"
        )

    @classmethod
    def _resolve_target(
        cls,
        version_name: Optional[Union[str, SpatialGroundingType]],
    ) -> SpatialGroundingType:
        """Resolve *version_name* to a canonical ``SpatialGroundingType``.

        Accepts the Enum directly, or its string value (``"bbox"`` / ``"point"``
        / ``"quad"``). Unknown / None values fall back to ``BBOX``.
        """
        if isinstance(version_name, SpatialGroundingType):
            return version_name
        if isinstance(version_name, str):
            try:
                return SpatialGroundingType(version_name.lower())
            except ValueError:
                logger.warning(
                    "SpatialGroundingProcessor: unknown version_name %r, falling back to bbox",
                    version_name,
                )
        return SpatialGroundingType.BBOX

    @classmethod
    def _detect_source(
        cls,
        text: str,
        target: SpatialGroundingType,
    ) -> Optional[Tuple[List[float], SpatialGroundingType]]:
        """Find the best ``(nums, source_type)`` in *text*.

        Stages (first match wins):
          0. Non-bracket POINT-producing patterns
             (``POINT_ATTR_RE``/``POINT_NESTED_X_RE``/``ACTION_CLICK_RE``) —
             these label their numbers with ``x1=``/``y1=``/``x=``/``y=``
             tokens and would either fail the bracket parser or be
             mis-counted by the bare-num fallback.
          1. Bracket groups (arity 8/4/2 — quad/bbox/point) — richest wins,
             so a model that emits both a quad and its center is treated as
             quad source, giving cross-shape conversion the most information.
          2. Bare numbers — take the last ``_ARITY[target]`` numbers and
             treat them as the target shape (no conversion). Covers models
             that omit brackets entirely.
        """
        # Stage 0 — non-bracket point-producing patterns. POINT only; bbox
        # targets that need extent info pass through to the bracket /
        # bare-num stages below.
        for _pat in (cls._point_attr_pattern(),
                     cls._point_nested_x_pattern(),
                     cls._action_click_pattern()):
            m = _pat.search(text)
            if m:
                try:
                    nums = [float(m.group(1)), float(m.group(2))]
                except (TypeError, ValueError):
                    continue
                return nums, SpatialGroundingType.POINT

        # Stage 1 — bracket groups, richest-first.
        for arity, src in ((8, SpatialGroundingType.QUAD),
                           (4, SpatialGroundingType.BBOX),
                           (2, SpatialGroundingType.POINT)):
            matches = list(cls._bracket_pattern(arity).finditer(text))
            if matches:
                try:
                    nums = [float(_g) for _g in matches[-1].groups()]
                except (TypeError, ValueError):
                    continue
                return nums, src

        # Stage 2 — bare nums. Take the last ``target_arity`` and treat
        # them as the target shape (no conversion). Last-resort for models
        # that omit brackets entirely.
        target_arity = _ARITY[target]
        bare = re.findall(cls._NUM, text)
        if len(bare) >= target_arity:
            try:
                nums = [float(_n) for _n in bare[-target_arity:]]
            except (TypeError, ValueError):
                return None
            return nums, target
        return None

    @classmethod
    def _convert(
        cls,
        nums: List[float],
        source: SpatialGroundingType,
        target: SpatialGroundingType,
    ) -> Optional[List[float]]:
        if source == target:
            return nums
        if source == SpatialGroundingType.BBOX:
            x1, y1, x2, y2 = nums
            if target == SpatialGroundingType.POINT:
                return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            if target == SpatialGroundingType.QUAD:
                return [x1, y1, x2, y1, x2, y2, x1, y2]
        if source == SpatialGroundingType.QUAD:
            xs = nums[0::2]
            ys = nums[1::2]
            if target == SpatialGroundingType.BBOX:
                return [min(xs), min(ys), max(xs), max(ys)]
            if target == SpatialGroundingType.POINT:
                return [sum(xs) / len(xs), sum(ys) / len(ys)]
        # point → bbox / point → quad: insufficient information
        return None

    @classmethod
    def extract(
        cls,
        prediction: str,
        query: Optional[str] = None,
        version_name: Optional[Union[str, SpatialGroundingType]] = None,
        api_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ) -> Optional[str]:
        """Return canonical ``"[v1, v2, ...]"`` string in *version_name* shape.

        This processor is PARSING-ONLY: it extracts the coordinate group from
        free-form text and reshapes it to *version_name*, emitting the RAW
        parsed numbers in whatever scale the model produced. Coordinate-space
        normalization is the metric's responsibility (it has the per-record
        image size); ``compute_iou`` / ``compute_click_dist_accuracy`` apply
        ``_grounding_norm_axis`` downstream.

        Args:
            prediction:   model output (free-form text).
            query:        upstream prompt (unused; kept for ProcessorInterface
                          signature parity).
            version_name: desired output shape (Enum value or string ``"bbox"``
                          / ``"point"`` / ``"quad"``). Defaults to ``"bbox"``.
            api_name:     LLM fallback identifier. **Not implemented** for
                          grounding — passing a non-None value raises
                          ``NotImplementedError`` to avoid silent fallthrough.

        Stages:
          1) NFKC normalize + strip ``<think>...</think>`` CoT trace.
          2) Auto-detect source shape (quad > bbox > point) from bracket
             groups; fallback to bare numbers at target arity.
          3) Convert source → target via :meth:`_convert`.
          4) Re-emit as ``"[v1, v2, ...]"`` parseable by ``compute_iou``.

        Returns ``None`` when no parseable answer is found or conversion is
        impossible (e.g. point → bbox).
        """
        if isinstance(api_name, str) and len(api_name) > 0:
            raise NotImplementedError(
                'SpatialGroundingProcessor._extract_spatial_grounding_api() is not implemented'
            )

        if not isinstance(prediction, str) or not prediction.strip():
            return prediction if not isinstance(prediction, str) else None

        target = cls._resolve_target(version_name)

        text = unicodedata.normalize("NFKC", prediction)
        _think_end = text.rfind("</think>")
        if _think_end != -1:
            text = text[_think_end + len("</think>"):]
        text = text.strip()

        detected = cls._detect_source(text=text, target=target)
        if detected is None:
            if verbose:
                logger.debug("SpatialGroundingProcessor: no grounding answer in %r", prediction)
            return None
        nums, source = detected

        # Shape convert (raw coords, no scale change).
        converted = cls._convert(nums=nums, source=source, target=target)
        if converted is None:
            if verbose:
                logger.debug(
                    "SpatialGroundingProcessor: cannot convert %s → %s (%r)",
                    source.value, target.value, prediction,
                )
            return None

        # canonical form parsed by compute_iou: "[v1, v2, ...]" (", "-joined).
        output = "[" + ", ".join(repr(_v) for _v in converted) + "]"
        if verbose:
            logger.debug(
                "SpatialGroundingProcessor: %r -> %s (source=%s → target=%s)",
                prediction, output, source.value, target.value,
            )
        return output
