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

import os
from pathlib import Path
import PIL
from typing import Any, Dict, List, Optional, Union, Literal


class EvaluatorInterface:
    @staticmethod
    def _grounding_norm_axis(v: float, dim: Optional[int]) -> float:
        """Normalize one coordinate component using the image dimension *dim*.

        - ``dim`` falsy (None / 0): no info to normalize against → return *v*
          unchanged (caller compares whatever space the inputs arrived in).
        - value already in ``[-0.05, 1.5]``: treated as already-normalized and
          passed through (lets a normalized GT label coexist with a pixel
          prediction in the same call).
        - otherwise: a pixel value → divided by *dim*.
        """
        if not dim:
            return v
        if -0.0001 <= v <= 1.0001:
            return v
        return v / dim

    @classmethod
    def compute_iou(
        cls,
        box1: str,
        box2: str,
        unpad: bool = False,
        image: Optional[PIL.Image.Image] = None,
    ):
        """
        box1: ground-truth bbox assumed to be in padded coordinate space.
        box2: predicted bbox, which may be in either unpadded or padded space depending on the model;
              when do_eval_unpad is set, box2 must be converted to padded form (equivalently, convert box1 to unpadded).

        Coordinate normalization: the spatial_grounding postprocessor now emits
        RAW parsed coords (no scale change), so box2 is normalized here. When an
        *image* is given, each axis is normalized per-axis via
        `_grounding_norm_axis` using the image's width/height (a pixel coord is
        divided by the dimension; an already-normalized value passes through).
        Without an image, box2 is assumed to already be in [0, 1].
        """
        try:
            box1 = box1.strip("[]").split(", ")
            box1 = [float(element) for element in box1]

            box2 = box2.strip("[]").split(", ")
            box2 = [float(element) for element in box2]

            if image is not None and len(box2) == 4:
                width, height = image.size
                box2 = [
                    cls._grounding_norm_axis(box2[0], width),
                    cls._grounding_norm_axis(box2[1], height),
                    cls._grounding_norm_axis(box2[2], width),
                    cls._grounding_norm_axis(box2[3], height),
                ]
            if unpad:
                width, height = image.size
                box2 = cls._pad_bbox(box2, width, height)

            if len(box2) != 4:
                return 0

            # box = (x1, y1, x2, y2)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

            # obtain x1, y1, x2, y2 of the intersection
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            # compute the width and height of the intersection
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)

            inter = w * h
            iou = inter / (box1_area + box2_area - inter)
        except Exception as ex:
            iou = 0
        return iou

    @classmethod
    def _pad_bbox(
        cls, 
        bbox: List[Union[int, float]], 
        width: int, 
        height: int, 
        detection_precision: Optional[int] = 2,
    ):

        bbox[0], bbox[2] = bbox[0] * width, bbox[2] * width
        bbox[1], bbox[3] = bbox[1] * height, bbox[3] * height

        processed_bbox = []
        if height > width:
            for idx, point in enumerate(bbox):  # x1 y1 x2 y2
                if idx % 2 == 0:
                    processed_bbox.append(round(((height - width) // 2 + point) / height, detection_precision))
                else:
                    processed_bbox.append(round(point / height, detection_precision))
        elif width > height:
            for idx, point in enumerate(bbox):  # x1 y1 x2 y2
                if idx % 2 == 0:
                    processed_bbox.append(round(point / width, detection_precision))
                else:
                    processed_bbox.append(round(((width - height) // 2 + point) / width, detection_precision))
        else:
            processed_bbox = [round(point / width, 2) for point in bbox]

        return processed_bbox