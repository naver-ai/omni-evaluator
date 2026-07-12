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

import clip_benchmark
from clip_benchmark import metrics as clip_benchmark_metrics
from clip_benchmark.metrics import zeroshot_classification as clip_benchmark_zeroshot_classification
import logging
import numpy as np
import open_clip
import os
from pathlib import Path
import PIL
from PIL import Image
import torch
import torchvision
import threading
from tqdm import tqdm
import traceback
from transformers import AutoConfig, AutoImageProcessor, Mask2FormerForUniversalSegmentation
from typing import Any, Callable, Dict, List, Optional, Union, Literal

logger = logging.getLogger(__name__)

from omni_evaluator.evaluation.metrics._interface import EvaluatorInterface
from omni_evaluator.utils.multimodal import to_pil_image
from omni_evaluator.utils.torch import get_compute_capability


DEFAULT_COLORS = [
    "red", "orange", "yellow", "green", "blue", 
    "purple", "pink", "brown", "black", "white",
]

COCO_OBJECT_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "computer mouse",
    "tv remote",
    "computer keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

class ImageTransformDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        image: PIL.Image.Image, 
        transform: Optional[torchvision.transforms.transforms.Compose] = None, 
        bbox_list: Optional[List[List[float]]] = None,
        mask_list: Optional[List[List[float]]] = None,
        background_color: str = "#999",
        crop_style: str = "1",
    ):
        self.image = image.convert("RGB")
        self.multiplier = 1 # single image
        self.transform = transform
        self.mask_list = mask_list
        if isinstance(self.mask_list, (list, tuple)):
            self.multiplier = len(self.mask_list)
            for _idx, _mask in enumerate(self.mask_list):
                if not (tuple([image.height, image.width]) == tuple(_mask.shape)):
                    raise ValueError(f'`mask_list[{_idx}]` has different shape with (image.height, image.width): ({image.height}, {image.width}) vs. {tuple(_mask.shape)}')
        self.bbox_list = bbox_list
        if isinstance(self.bbox_list, (list, tuple)):
            if not (len(self.bbox_list) == len(self.mask_list)):
                raise ValueError(f'bbox_list should have the same length as mask_list: {self.bbox_list} vs. {self.mask_list}')
            self.multiplier = len(self.bbox_list)
        self.background_color = background_color
        self.background = Image.new("RGB", image.size, color=background_color)
        if self.background_color == "original":
            self.background = self.image.copy()
        self.crop_style = crop_style

    def __len__(self):
        return self.multiplier

    def __getitem__(self, index):
        image = self.image
        if self.mask_list is not None:
            image = Image.composite(
                self.image,
                self.background,
                Image.fromarray(self.mask_list[index]),
            )
        if self.bbox_list is not None:
            if self.crop_style == '1':
                image = image.crop(self.bbox_list[index])
            
        if isinstance(self.transform, Callable):
            image = self.transform(image)
        return (image, 0)


class ImageEvaluator(EvaluatorInterface):
    COLOR_CLASSIFIERS = dict()
    
    def __init__(
        self,
        object_detector_name: str = "facebook/mask2former-swin-small-coco-instance",
        clip_model_name: str = "ViT-L-14",
        clip_model_pretrained: str = "openai",
        colors: Optional[List[str]] = None,
        hf_cache_dir: Optional[str] = None,
    ):
        # define object_detectors
        self.object_detector_name = object_detector_name
        self.object_detection_classnames = COCO_OBJECT_NAMES
        if "mask2former" in object_detector_name:
            self.object_detector = Mask2FormerForUniversalSegmentation.from_pretrained(
                object_detector_name,
                cache_dir=hf_cache_dir,
            )
            self.object_detector.processor = AutoImageProcessor.from_pretrained(
                object_detector_name,
                cache_dir=hf_cache_dir,
            )
            self.object_detector.config = AutoConfig.from_pretrained(
                object_detector_name,
                cache_dir=hf_cache_dir,
            )
            # self.object_detection_classnames = list(self.object_detector.config.id2label.values())
        else:
            raise ValueError(f'not implemented `object_detector_name`: {object_detector_name}')
        
        # define clip vars
        self.device = "cpu"
        if get_compute_capability():
            self.device = "cuda"
        # move object detector onto the selected device (was left on CPU regardless)
        self.object_detector = self.object_detector.to(self.device)
        self.clip_model_name = clip_model_name
        self.clip_model_pretrained = clip_model_pretrained
        self.clip, _, self.clip_transform = open_clip.create_model_and_transforms(
            self.clip_model_name, 
            pretrained=clip_model_pretrained, 
            device=self.device,
        )
        self.clip_tokenizer = open_clip.get_tokenizer(self.clip_model_name)
        self.clip_batch_size = 16
        self.num_workers = 1
        
        self.colors = colors
        if not self.colors:
            self.colors = DEFAULT_COLORS
            
    def detect_colors(
        self,
        image,
        classname: str,
        bbox_list: Optional[List[List[float]]] = None,
        mask_list: Optional[List[List[float]]] = None,
        background_color: Optional[str] = None,
        crop_style: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ):
        if not background_color:
            background_color = "#999"
        if not crop_style:
            crop_style = "1"
        if not colors:
            colors = self.colors
        
        if classname not in self.COLOR_CLASSIFIERS:
            self.COLOR_CLASSIFIERS[classname] = clip_benchmark.metrics.zeroshot_classification.zero_shot_classifier(
                model=self.clip, 
                tokenizer=self.clip_tokenizer, 
                classnames=colors,
                templates=[
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object"
                ],
                device=self.device,
            )

        dataloader = torch.utils.data.DataLoader(
            ImageTransformDataset(
                image=image, 
                transform=self.clip_transform,
                bbox_list=bbox_list,
                mask_list=mask_list,
                background_color=background_color,
                crop_style=crop_style,
            ),
            batch_size=self.clip_batch_size, 
            num_workers=self.num_workers,
        )
        
        output = None
        with torch.no_grad():
            _prediction, _ = clip_benchmark.metrics.zeroshot_classification.run_classification(
                model=self.clip, 
                classifier=self.COLOR_CLASSIFIERS[classname], 
                dataloader=dataloader,
                device=self.device,
            ) # _prediction = _prediction.detach().cpu()
            output = list()
            for index in _prediction.argmax(dim=1):
                output.append(colors[index.item()])
        return output
    
    def compute_relative_position(
        self,
        target_bbox: List[float], 
        reference_bbox: List[float],
        position_threshold: Optional[float] = None,
        offset_threshold: Optional[float] = None,
    ):
        """
        target_bbox is `right of` reference_bbox
        """
        if not position_threshold:
            position_threshold = 0.1
        if not offset_threshold:
            offset_threshold = 1e-3
        else:
            offset_threshold = float(offset_threshold)
            
        if not isinstance(target_bbox, np.ndarray):
            target_bbox = np.array(target_bbox)
        target_bbox = target_bbox.reshape(2, 2)
        if not isinstance(reference_bbox, np.ndarray):
            reference_bbox = np.array(reference_bbox)
        reference_bbox = reference_bbox.reshape(2, 2)
        
        # center: (avg(x1, x2), avg(y1, y2))
        center_target = target_bbox.mean(axis=0)
        center_reference = reference_bbox.mean(axis=0)
        # dim: (abs(x2 - x1), abs(y2 - y1))
        dim_target = np.abs(np.diff(target_bbox, axis=0))[0]
        dim_reference = np.abs(np.diff(reference_bbox, axis=0))[0]

        # offset: diff between centers
        offset = center_target - center_reference
        revised_offset = np.maximum(
            0,
            np.abs(offset) - position_threshold * (dim_target + dim_reference), 
        ) * np.sign(offset)
        
        relations = set()
        # if center of two bboxes are similar
        if np.all(np.abs(revised_offset) < offset_threshold):
            return relations

        delta_x, delta_y = revised_offset / np.linalg.norm(offset)
        if delta_x < -0.5: 
            relations.add("left of")
        if delta_x > 0.5: 
            relations.add("right of")
        if delta_y < -0.5: 
            relations.add("above")
        if delta_y > 0.5: 
            relations.add("below")
        return relations
    
    def detect_objects(
        self,
        image: PIL.Image.Image,
        confidence_threshold: Optional[float] = None,
    ):
        if not confidence_threshold:
            confidence_threshold = 0.3

        if "mask2former" in self.object_detector_name:
            _inputs = self.object_detector.processor(images=image, return_tensors="pt")
            _inputs = _inputs.to(self.device)
            with torch.no_grad():
                _outputs = self.object_detector(
                    **_inputs,
                )
        result = self.object_detector.processor.post_process_instance_segmentation(
            _outputs, 
            target_sizes=[
                (image.height, image.width),
            ],
            return_binary_maps=True,
        )[0]
        
        output = dict()
        if len(result["segments_info"]) < 1:
            return output
        
        bboxes = torchvision.ops.masks_to_boxes(result["segmentation"])
        bboxes = bboxes.detach().cpu().numpy()
        result["segmentation"] = result["segmentation"].detach().cpu().numpy()
        for _bbox, _segmentation, _segment_info in zip(
            bboxes, result["segmentation"], result["segments_info"],
        ):
            if (
                confidence_threshold
                and _segment_info["score"] < confidence_threshold
            ):
                continue
            _semtantic_label = self.object_detection_classnames[_segment_info["label_id"]]
            if _semtantic_label not in output:
                output[_semtantic_label] = list()
            output[_semtantic_label].append({
                "bbox": _bbox,
                "mask": _segmentation,
                "binary_mask": _segmentation > 0,
                "confidence": _segment_info["score"],
            })
        del result 
        return output
    
    def compute_gen_eval(
        self,
        image: Union[str, bytes, PIL.Image.Image], 
        include_requirements: Optional[List[Dict[str, Any]]] = None,
        exclude_requirements: Optional[List[Dict[str, Any]]] = None,
        confidence_threshold: Optional[float] = None,
        position_threshold: Optional[float] = None,
        offset_threshold: Optional[float] = None,
        background_color: Optional[str] = None,
        crop_style: Optional[str] = None,
        colors: Optional[List[str]] = None,
    ):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        if not image:
            return {
                "correct": False, 
                "reason": "failed to generate image while inference",
            }
        
        correct = True
        reason = list()
        
        if not include_requirements:
            include_requirements = list()
        if not exclude_requirements:
            exclude_requirements = list()
 
        if not isinstance(image, PIL.Image.Image):
            image = to_pil_image(image=image)
        detected_objects = self.detect_objects(
            image=image,
            confidence_threshold=confidence_threshold,
        )

        # Check for expected objects
        matched_objects = dict()
        found_objects = None
        found_colors = None
        actual_relative_positions = list()
        for _requirement_idx, _requirement in enumerate(include_requirements):
            target_classname = _requirement["class"]
            target_count = _requirement["count"]
            target_color = _requirement.get("color", None)
            target_position = _requirement.get("position", None)
            matched = True

            found_objects = list()
            if target_classname in detected_objects:
                found_objects = detected_objects[target_classname]
            found_objects = found_objects[:target_count]
            
            # check count
            if len(found_objects) < target_count:
                correct = False
                matched = False
                reason.append(f'expected {target_classname} >= {target_count}, found {len(found_objects)}')
            
            # check color
            if target_color: 
                _bbox_list = list()
                _mask_list = list()
                for _object in found_objects:
                    _bbox_list.append(_object["bbox"])
                    _mask_list.append(_object["binary_mask"])
                found_colors = self.detect_colors(
                    image=image,
                    classname=target_classname,
                    bbox_list=_bbox_list if _bbox_list else None,
                    mask_list=_mask_list if _mask_list else None,
                    background_color=background_color,
                    crop_style=crop_style,
                    colors=colors,
                )
                if found_colors.count(target_color) < target_count:
                    correct = False
                    matched = False
                    reason.append(f'expected {target_classname} with {target_color} >= {target_count}, found {found_colors}')
                        
            # check relative position
            if target_position:
                target_relative_position, target_group = _requirement["position"]
                if target_group not in matched_objects:
                    correct = False
                    matched = False
                else:
                    for _found_object in found_objects:
                        for _matched_object in matched_objects[target_group]:
                            _actual_relative_positions = self.compute_relative_position(
                                target_bbox=_found_object["bbox"],
                                reference_bbox=_matched_object["bbox"],
                                position_threshold=position_threshold,
                                offset_threshold=offset_threshold,
                            )
                            actual_relative_positions.append(_actual_relative_positions)
                            if target_relative_position not in _actual_relative_positions:
                                correct = False
                                matched = False
                                reason.append(f'expected {target_classname} {target_relative_position}, found {_actual_relative_positions}')
                                break
                        if not matched:
                            break
                    
            if matched:
                matched_objects[_requirement_idx] = found_objects
        
        # Check for non-expected objects
        for _requirement_idx, _requirement in enumerate(exclude_requirements):
            target_classname = _requirement["class"]
            if (
                target_classname in detected_objects
                and len(detected_objects[target_classname]) >= _requirement["count"]
            ):
                correct = False
                reason.append(f'expected {target_classname} < {_requirement["count"]}, found {len(detected_objects[target_classname])}')
        
        if found_objects:
            for _idx, _found_object in enumerate(found_objects):
                found_objects[_idx]["bbox"] = found_objects[_idx]["bbox"].tolist()
                found_objects[_idx]["mask"] = found_objects[_idx]["mask"].tolist()
                found_objects[_idx]["binary_mask"] = found_objects[_idx]["binary_mask"].tolist()
        
        return {
            "correct": correct,
            "reason": reason,
            "detected_objects": list(detected_objects.keys()),
            "found_objects": found_objects,
            "found_colors": found_colors,
            "actual_relative_positions": actual_relative_positions,
        }
