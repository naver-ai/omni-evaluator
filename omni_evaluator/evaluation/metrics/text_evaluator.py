# Reference from https://github.com/huggingface/evaluate (Apache-2.0)
# Reference from https://github.com/hendrycks/outlier-exposure (Apache-2.0)
# Reference from https://github.com/haotian-liu/LLaVA (Apache-2.0)
# Reference from https://github.com/yuliang-liu/MultimodalOCR (MIT)
# Reference from https://github.com/open-compass/VLMEvalKit (Apache-2.0)
# Reference from https://github.com/facebookresearch/mmf (BSD-3-Clause)

# Modifications Copyright (c) 2026-present NAVER Cloud Corp.
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

import ast
from collections import Counter, defaultdict
import copy
import dateparser
import evaluate as hf_evaluate
import jiwer
import json
import logging
import math
from nltk import edit_distance
from nltk.util import ngrams
import numpy as np
import os
from pathlib import Path
import PIL
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import re
import regex
import sacrebleu
import string
import sys
import threading
from tqdm import tqdm
import traceback
from typing import Any, Dict, List, Optional, Union, Literal, Tuple
import yaml

logger = logging.getLogger(__name__)

from omni_evaluator import NullPredictionPolicy
from omni_evaluator.enums import SpatialGroundingType
from omni_evaluator.evaluation.metrics._interface import EvaluatorInterface
from omni_evaluator.evaluation.metrics.constants import (
    vqa__contractions, vqa__manualMap, vqa__articles,
    vqa__periodStrip, vqa__commaStrip, vqa__punct, vqa__circledNumbersMap,
)
from omni_evaluator.evaluation.metrics.html import (
    parse_html, parse_html_table, parse_html_math, remove_html_tag,
    create_html_tree,
    _get_insert_cost_html, _get_remove_cost_html, _get_update_cost_html,
)
import zss as _zss
from omni_evaluator.evaluation.metrics.repetition import RepetitionModel
from omni_evaluator.evaluation.metrics.wtq import (
    to_value_list, tsv_unescape_list, check_denotation, _normalize as _wtq_normalize,
)
from omni_evaluator.evaluation.metrics.pier import pier as hike_pier
from omni_evaluator.evaluation.metrics.mmmu_accuracy import score_row as mmmu_score_row
from omni_evaluator.postprocess.custom import (
    parse_circled_answer,
)
from omni_evaluator.schemas.chat import (
    OcrToken, EntityToken,
    Message as ChatMessage, 
)
from omni_evaluator.schemas.inference import Record

_ARITY = {
    SpatialGroundingType.BBOX: 4,
    SpatialGroundingType.POINT: 2,
    SpatialGroundingType.QUAD: 8,
}

class TextEvaluator(EvaluatorInterface):
    nlgeval = None
    available_metrics = [
        "exact_match",
        "f1", # f1, recall, precision
        "binary_f1",
        "calibration_error",
        "jaccard_distance",
        "edit_distance", # "levenshtein_distance", "ned",
        "anls",
        "nlgeval",
        "bleu",
        "rouge",
        "meteor",
        "cider",
        "spice",
        "bert_score",
        "error_rate",
        "wer",
        "cer",
        "iou",
        "unpad_iou",
        "click_dist_accuracy",
        "mmau_string_match",
        "mmmu_accuracy",
        "temporal_iou",
        "tree_edit_score",
        "ocrbench_vqaeval",
        "wtq_vqaeval",
        "vqaeval",
        "fintabnet_vqaeval",
        "repetition",
    ]
    _repetition_model_instance = None  # class variable for model caching
    _model_lock = threading.Lock()  # Lock for thread-safe model access

    @classmethod
    def evaluate(
        cls,
        target_metrics: Union[List[str], Dict[str, Dict[str, Any]]],
        records: List[Union[Dict[str, Any], Record]],
        group_field: Optional[str] = "category",
        sources: Optional[List[str]] = None, # comet
        confidences: Optional[List[float]] = None, # calibration_error
        # Task-level pre-normalize trigger for table-eval methods
        # ("squad"/"fintabnet"/"wtq"). Affects sample-prep (predictions/labels
        # rewritten before any metric dispatch). Per-metric ``do_normalize``
        # (e.g. wer/cer/mer) is read from yaml metric kwargs separately.
        do_normalize: Optional[Union[bool, str]] = False,
        null_prediction_policy: NullPredictionPolicy = NullPredictionPolicy.miss,
        fallback_value: Optional[str] = None,
        # Task-level resolved do_async — caller (``engine.py`` dispatch) is
        # expected to inline-OR yaml task-level + CLI runtime override before
        # calling. Per-metric yaml override (``_target_metrics_kwargs[<metric>]
        # .get("do_async")``) is consulted inside the relevant dispatch branches
        # (tree_edit_score).
        do_async: bool = False,
    ):
        # Coerce stringly-typed input from non-typed callers (raw kwargs / YAML).
        if isinstance(null_prediction_policy, str) and not isinstance(null_prediction_policy, NullPredictionPolicy):
            null_prediction_policy = NullPredictionPolicy(null_prediction_policy)
        # target_metrics is either a flat list (legacy) or
        # {metric_name: kwargs} dict (new). Normalize to both forms internally.
        if isinstance(target_metrics, dict):
            _target_metrics_kwargs = {k: (v or {}) for k, v in target_metrics.items()}
            target_metrics = list(_target_metrics_kwargs.keys())
        elif isinstance(target_metrics, (list, tuple)):
            _target_metrics_kwargs = {m: {} for m in target_metrics}
            target_metrics = list(target_metrics)
        else:
            _target_metrics_kwargs = {}
            target_metrics = []
        """
        Args:
            predictions (List[str]): _description_
            labels (Union[List[str], List[List[str]]]):
                - List[str]: single-references
                - List[List[str]]: multi-references
            metrics (List[str], optional): _description_. Defaults to None.
        """
        if target_metrics is None:
            target_metrics = cls.available_metrics

        # Skip the table-eval pre-normalize when do_normalize is bool/None or an
        # ASR-method string ("default"/"korean"/"chinese"); compute_wer/compute_cer
        # handle ASR normalization downstream.
        _TABLE_EVAL_NORMALIZE_METHODS = {"squad", "fintabnet", "wtq"}
        _apply_table_eval_normalize = (
            isinstance(do_normalize, str)
            and do_normalize in _TABLE_EVAL_NORMALIZE_METHODS
        )

        predictions, labels, queries = list(), list(), list()
        option_contents_list, categories = list(), list()
        options_list, question_types_list = list(), list()
        for _record_idx, _record in enumerate(records):
            # prediction
            _prediction = _record["prediction"]
            if _record.get("prediction_postprocessed", None):
                _prediction = _record["prediction_postprocessed"]
            # null prediction policy — apply uniformly so downstream metrics
            # only ever see strings.
            #   skip     : drop sample entirely (predictions/labels shorter)
            #   fallback : replace with fallback_value (None → "")
            #   else     : default "miss" → "" (binary_f1 treats empty token
            #              list as positive, so label="no" samples become FP /
            #              label="yes" stay TP — i.e. not a free TN bonus).
            if not isinstance(_prediction, str):
                if null_prediction_policy == NullPredictionPolicy.skip:
                    continue
                elif null_prediction_policy == NullPredictionPolicy.fallback:
                    _prediction = fallback_value if fallback_value is not None else ""
                else:
                    _prediction = ""
            if _apply_table_eval_normalize:
                _prediction = cls.normalize(text=_prediction, method=do_normalize)
            # label
            _labels = _record["label"]
            if isinstance(_labels, str):
                _labels = [_labels, ]
            if _apply_table_eval_normalize:
                _labels = [
                    cls.normalize(text=_label, method=do_normalize)
                    for _label in _labels
                ]
            # query
            _user_message = ChatMessage.get_user_messages(messages=_record["messages"])[-1]
            _query = ChatMessage.get_query(message=_user_message)
            # options / option_contents
            _options = _record.get("options") or None
            _option_contents = None
            if _record.get("option_contents", None):
                _option_contents = _record["option_contents"]
            elif _record.get("options", None):
                _option_contents = _record["options"]
            # category
            _category = None
            if (
                group_field
                and group_field in _record.get("meta", {})
            ):
                _category = _record["meta"][group_field]
            # question_type (used by mmmu_accuracy to route MC vs open)
            _question_type = None
            _meta = _record.get("meta") or dict()
            if isinstance(_meta, dict):
                _question_type = _meta.get("question_type")

            predictions.append(_prediction)
            labels.append(_labels)
            queries.append(_query)
            option_contents_list.append(_option_contents)
            options_list.append(_options)
            categories.append(_category)
            question_types_list.append(_question_type)
        
        if "comet" in target_metrics:
            if not sources:
                sources = [
                    _record["meta"]["source"]
                    for _record in records
                ]
                
            if len(sources) != len(records):
                raise ValueError(f'`sources` should be same length with predictions to compute comet')
        if "calibration_error" in target_metrics:
            if not (
                confidences
                and len(confidences) and len(records)
            ):
                raise ValueError(f'`confidences` should be same length with predictions to compute calibration_error')

        # evaluate by sample
        metrics = dict()
        sample_metrics = dict()
        for _idx, (_prediction, _labels) in tqdm(
            enumerate(zip(predictions, labels)),
            initial=0,
            total=len(predictions),
            desc="Evaluating sample-wise metrics",
        ):
            if isinstance(_prediction, (list, tuple)):
                _prediction = _prediction[0]
            
            if (
                not _labels 
                or len(_labels) < 1 
                or all([not _label for _label in _labels])
            ):
                logger.warning(f'Label for {_idx}th sample not given: {_labels}')
                continue

            # Sample-wise metric dispatch — every branch follows the same convention:
            #   _kwargs_<metric> = _target_metrics_kwargs.get("<metric>") or {}
            # then forward to compute_<metric>(...)'s optional args via ``.get("<arg>", <default>)``.
            # Even a metric with no args to forward now keeps the ``_kwargs_<metric>`` extraction
            # itself for future extension.
            _sample_metrics = dict()
            if "edit_distance" in target_metrics:
                _kwargs_edit_distance = _target_metrics_kwargs.get("edit_distance") or {}
                _sample_metrics["normalized_edit_distance"] = min(list(map(
                    lambda _label: cls.compute_ned(
                        label=_label,
                        prediction=_prediction,
                        uncased=_kwargs_edit_distance.get("uncased", True),
                    ),
                    _labels,
                )))
                _sample_metrics["normalized_levenshtein_distance"] = min(list(map(
                    lambda _label: cls.compute_levenshtein_distance(
                        label=_label,
                        prediction=_prediction,
                        uncased=_kwargs_edit_distance.get("uncased", True),
                    ),
                    _labels,
                )))
            if "jaccard_distance" in target_metrics:
                _kwargs_jaccard_distance = _target_metrics_kwargs.get("jaccard_distance") or {}  # noqa: F841
                _sample_metrics["jaccard_distance_unigram"] = cls.compute_jaccard_distance(
                    labels=_labels,
                    prediction=_prediction,
                    n=1,
                )
                _sample_metrics["jaccard_distance_bigram"] = cls.compute_jaccard_distance(
                    labels=_labels,
                    prediction=_prediction,
                    n=2,
                )
            if "anls" in target_metrics:
                _kwargs_anls = _target_metrics_kwargs.get("anls") or {}
                _sample_metrics["anls"] = cls.compute_anls(
                    labels=_labels,
                    prediction=_prediction,
                    uncased=True,
                    threshold=_kwargs_anls.get("threshold", 0.5),
                )
                _sample_metrics["anls_cased"] = cls.compute_anls(
                    labels=_labels,
                    prediction=_prediction,
                    uncased=False,
                    threshold=_kwargs_anls.get("threshold", 0.5),
                )
            if "exact_match" in target_metrics:
                _kwargs_exact_match = _target_metrics_kwargs.get("exact_match") or {}
                _sample_metrics["exact_match"] = max(list(map(
                    lambda _label: cls.compute_exact_match(
                        label=_label,
                        prediction=_prediction,
                        relative_tolerance=_kwargs_exact_match.get("relative_tolerance"),
                        absolute_tolerance=_kwargs_exact_match.get("absolute_tolerance", 1e-6),
                    ),
                    _labels,
                )))
            if "string_match" in target_metrics:
                _kwargs_string_match = _target_metrics_kwargs.get("string_match") or {}  # noqa: F841
                _sample_metrics["string_match"] = max(list(map(
                    lambda _label: cls.compute_string_match(label=_label, prediction=_prediction),
                    _labels,
                )))
            if "substring_match" in target_metrics:
                _kwargs_substring_match = _target_metrics_kwargs.get("substring_match") or {}
                _sample_metrics["substring_match"] = max(list(map(
                    lambda _label: cls.compute_substring_match(
                        label=_label,
                        prediction=_prediction,
                        normalize=_kwargs_substring_match.get("normalize", "squad"),
                    ),
                    _labels,
                )))

            if (
                "f1" in target_metrics
                or "recall" in target_metrics
                or "precision" in target_metrics
            ): # word level f1
                _kwargs_f1 = _target_metrics_kwargs.get("f1") or {}
                _scores = cls.compute_f1(
                    labels=_labels,
                    prediction=_prediction,
                    normalize=_kwargs_f1.get("normalize", "squad"),
                    aggregate=_kwargs_f1.get("aggregate", "max"),
                )
                for _score_name, _score_value in _scores.items(): # "precision", "recall", "f1"
                    _sample_metrics[_score_name] = _score_value
            if "mer" in target_metrics:
                _kwargs_mer = _target_metrics_kwargs.get("mer") or {}
                _mer, _stat = cls.compute_mer(
                    labels=_labels,
                    prediction=_prediction,
                    do_normalize=_kwargs_mer.get("do_normalize", True),
                )
                _sample_metrics["mer"] = _mer
            if "pier" in target_metrics:
                _kwargs_pier = _target_metrics_kwargs.get("pier") or {}
                _pier, _poi, _rest = cls.compute_pier(
                    labels=_labels,
                    prediction=_prediction,
                    dummy_token=_kwargs_pier.get("dummy_token", "뷁"),
                )
                _sample_metrics["pier"] = _pier
            if "iou" in target_metrics or "unpad_iou" in target_metrics:
                # Load the record's image once; compute_iou uses its size to
                # per-axis normalize the (now RAW) prediction coords. None when
                # the record carries no image → compute_iou assumes [0, 1].
                _images = list()
                for _message in records[_idx]["messages"]:
                    _images = ChatMessage.get_images(
                        message=_message,
                        to_pil=True,
                    )
                _image = _images[0] if _images else None
            if "iou" in target_metrics:
                _kwargs_iou = _target_metrics_kwargs.get("iou") or {}  # noqa: F841
                iou = cls.compute_iou(
                    box1=_labels[0],
                    box2=_prediction,
                    unpad=False,
                    image=_image,
                )
                th_iou = 1 if iou > 0.5 else 0 # IoU threshold scoring method
                _sample_metrics["iou"] = iou
                _sample_metrics["th_iou"] = th_iou
            if "unpad_iou" in target_metrics:
                iou = cls.compute_iou(
                    box1=_labels[0],
                    box2=_prediction,
                    unpad=True,
                    image=_image,
                )
                th_iou = 1 if iou > 0.5 else 0
                _sample_metrics["unpad_iou"] = iou
                _sample_metrics["unpad_th_iou"] = th_iou
            if "click_dist_accuracy" in target_metrics:
                _meta = records[_idx].get("meta") or dict()
                _click_dist_acc_kwargs = _target_metrics_kwargs.get("click_dist_accuracy") or dict()
                _sample_metrics["click_dist_accuracy"] = cls.compute_click_dist_accuracy(
                    label=_labels[0],
                    prediction=_prediction,
                    image_w=_meta.get("image_w"),
                    image_h=_meta.get("image_h"),
                    threshold=_click_dist_acc_kwargs.get("threshold"),
                )
            if "vqaeval" in target_metrics:
                _kwargs_vqaeval = _target_metrics_kwargs.get("vqaeval") or {}  # noqa: F841
                _sample_metrics["vqaeval"] = cls.compute_vqaeval(labels=_labels, prediction=_prediction)
            if "wtq_vqaeval" in target_metrics:
                _kwargs_wtq_vqaeval = _target_metrics_kwargs.get("wtq_vqaeval") or {}  # noqa: F841
                _sample_metrics["wtq_vqaeval"] = cls.compute_wtq_vqaeval(
                    label=_labels[0],
                    prediction=_prediction,
                )
            if "ocrbench_vqaeval" in target_metrics:
                _kwargs_ocrbench_vqaeval = _target_metrics_kwargs.get("ocrbench_vqaeval") or {}
                _sample_metrics["ocrbench_vqaeval"] = cls.compute_ocrbench_vqaeval(
                    label=_labels[0],
                    prediction=_prediction,
                    lang=_kwargs_ocrbench_vqaeval.get("lang", "en"),
                    uncased=_kwargs_ocrbench_vqaeval.get("uncased", False),
                )
            if "fintabnet_vqaeval" in target_metrics:
                _kwargs_fintabnet_vqaeval = _target_metrics_kwargs.get("fintabnet_vqaeval") or {}  # noqa: F841
                _sample_metrics["fintabnet_vqaeval"] = cls.compute_fintabnet_vqaeval(
                    label=_labels[0],
                    predictin=_prediction,
                )
            if "mmau_string_match" in target_metrics:
                _kwargs_mmau_string_match = _target_metrics_kwargs.get("mmau_string_match") or {}  # noqa: F841
                _score = None
                for _label in _labels:
                    _score_ = cls.compute_mmau_string_match(
                        label=_label,
                        prediction=_prediction,
                        option_contents=option_contents_list[_idx],
                        options=options_list[_idx],
                    )
                    if (
                        _score is None
                        or _score_ > _score
                    ):
                        _score = _score_
                _sample_metrics["mmau_string_match"] = _score
            if "mmmu_accuracy" in target_metrics:
                _kwargs_mmmu_accuracy = _target_metrics_kwargs.get("mmmu_accuracy") or {}  # noqa: F841
                _sample_metrics["mmmu_accuracy"] = cls.compute_mmmu_accuracy(
                    prediction=_prediction,
                    labels=_labels,
                    options=options_list[_idx],
                    option_contents=option_contents_list[_idx],
                    question_type=question_types_list[_idx],
                )
            if "temporal_iou" in target_metrics:
                _kwargs_temporal_iou = _target_metrics_kwargs.get("temporal_iou") or {}  # noqa: F841
                # Per-sample dispatch emits mIoU input + recall flags at three
                # thresholds. Each averaged across the run by the aggregator
                # below: ``temporal_iou`` → mIoU, ``temporal_recall@X`` → R@1@X.
                _r = None
                for _label in _labels:
                    _r_ = cls.compute_temporal_iou(label=_label, prediction=_prediction)
                    if _r is None or _r_["iou"] > _r["iou"]:
                        _r = _r_
                _sample_metrics["temporal_iou"] = _r["iou"]
                _sample_metrics["temporal_recall@0.3"] = _r["r@0.3"]
                _sample_metrics["temporal_recall@0.5"] = _r["r@0.5"]
                _sample_metrics["temporal_recall@0.7"] = _r["r@0.7"]
            # ``tree_edit_score`` is dispatched batch-wise below (next to wer/cer)
            # — optional stage-1 LLM conversion is a single batch call rather
            # than a per-sample sync call.
            if "repetition" in target_metrics:
                _kwargs_repetition = _target_metrics_kwargs.get("repetition") or {}  # noqa: F841
                _repetition_prob = cls.compute_repetition_prob(pred=_prediction)
                _is_repetition = 1 if _repetition_prob > 0.5 else 0 # threshold: repetition(1) if > 0.5
                _sample_metrics["repetition_prob"] = _repetition_prob
                _sample_metrics["is_repetition"] = _is_repetition

            for _score_name, _score_value in _sample_metrics.items():
                if _score_name not in sample_metrics:
                    sample_metrics[_score_name] = [None, ] * len(predictions)
                sample_metrics[_score_name][_idx] = _score_value
            
        for _score_name, _score_values in sample_metrics.items():
            if _score_name == "is_repetition":
                continue
            metrics[_score_name] = np.nanmean(np.array(_score_values, dtype=float))
        if (
            "is_repetition" in sample_metrics 
            and len(sample_metrics["is_repetition"]) > 0
        ):
            _is_repetitions = np.array(sample_metrics["is_repetition"], dtype=float)
            metrics["repetition_count"] = float(np.nansum(_is_repetitions)) # (1) repetition sample count
            metrics["repetition_ratio"] = float(np.nanmean(_is_repetitions)) # (2) repetition ratio (mean)

        # Batch-wise metric dispatch — same convention as sample-wise:
        #   _kwargs_<metric> = _target_metrics_kwargs.get("<metric>") or {}
        # then forward to compute_<metric>(...)'s optional args via ``.get("<arg>", <default>)``.
        if "binary_f1" in target_metrics:
            _kwargs_binary_f1 = _target_metrics_kwargs.get("binary_f1") or {}
            _scores = cls.compute_binary_f1(
                labels=labels,
                predictions=predictions,
                **_kwargs_binary_f1,
            )
            metrics.update(_scores)
        if "calibration_error" in target_metrics:
            _kwargs_calibration_error = _target_metrics_kwargs.get("calibration_error") or {}
            _scores = cls.compute_calibration_error(
                labels=_labels,
                prediction=_prediction,
                confidences=confidences,
                p_norm=_kwargs_calibration_error.get("p_norm", 2),
                bin_size=_kwargs_calibration_error.get("bin_size", 100),
            )
            metrics.update(_scores)
        if "rouge" in target_metrics:
            _kwargs_rouge = _target_metrics_kwargs.get("rouge") or {}  # noqa: F841
            _scores = cls.compute_rouge(
                labels=labels,
                predictions=predictions,
            )
            metrics.update(_scores)
        if "bleu" in target_metrics:
            _kwargs_bleu = _target_metrics_kwargs.get("bleu") or {}
            _scores, _sample_metrics = cls.compute_bleu(
                labels=labels,
                predictions=predictions,
                method=_kwargs_bleu.get("method", "bleu"),
                max_orders=_kwargs_bleu.get("max_orders", [1, 2, 3, 4]),
            )
            metrics.update(_scores)
            sample_metrics.update(_sample_metrics)
        if "comet" in target_metrics:
            _kwargs_comet = _target_metrics_kwargs.get("comet") or {}
            _scores, _sample_metrics = cls.compute_comet(
                labels=labels,
                predictions=predictions,
                sources=sources,
                model_name=_kwargs_comet.get("model_name", "wmt20-comet-da"),
            )
            metrics.update(_scores)
            sample_metrics.update(_sample_metrics)
        if "meteor" in target_metrics:
            _kwargs_meteor = _target_metrics_kwargs.get("meteor") or {}  # noqa: F841
            _scores = cls.compute_meteor(labels=labels, predictions=predictions)
            metrics.update(_scores)
        if "spider" in target_metrics:
            _kwargs_spider = _target_metrics_kwargs.get("spider") or {}  # noqa: F841
            _scores, _sample_metrics = cls.compute_spider(labels=labels, predictions=predictions)
            metrics.update(_scores)
            sample_metrics.update(_sample_metrics)
        else:
            if "cider" in target_metrics:
                _kwargs_cider = _target_metrics_kwargs.get("cider") or {}  # noqa: F841
                _scores, _sample_metrics = cls.compute_cider(labels=labels, predictions=predictions)
                metrics.update(_scores)
                sample_metrics.update(_sample_metrics)
            if "spice" in target_metrics:
                _kwargs_spice = _target_metrics_kwargs.get("spice") or {}  # noqa: F841
                _scores, _sample_results = cls.compute_spice(labels=labels, predictions=predictions)
                metrics.update(_scores)
        if "bert_score" in target_metrics:
            _kwargs_bert_score = _target_metrics_kwargs.get("bert_score") or {}
            _scores = cls.compute_bert_score(
                labels=labels,
                predictions=predictions,
                model_type=_kwargs_bert_score.get("model_type", "distilbert-base-uncased"),
            )
            metrics.update(_scores)
        if "nlgeval" in target_metrics:
            _kwargs_nlgeval = _target_metrics_kwargs.get("nlgeval") or {}  # noqa: F841
            _scores = cls.compute_nlgeval(labels=labels, predictions=predictions)
            metrics.update(_scores)
        if "wer" in target_metrics:
            _kwargs_wer = _target_metrics_kwargs.get("wer") or {}
            _scores, _sample_metrics = cls.compute_wer(
                labels=labels,
                predictions=predictions,
                do_normalize=_kwargs_wer.get("do_normalize", do_normalize),
            )
            metrics.update(_scores)
            sample_metrics.update(_sample_metrics)
        if "cer" in target_metrics:
            _kwargs_cer = _target_metrics_kwargs.get("cer") or {}
            _scores, _sample_metrics = cls.compute_cer(
                labels=labels,
                predictions=predictions,
                do_normalize=_kwargs_cer.get("do_normalize", do_normalize),
            )
            metrics.update(_scores)
            sample_metrics.update(_sample_metrics)
        if "tree_edit_score" in target_metrics:
            _kwargs_tree_edit_score = _target_metrics_kwargs.get("tree_edit_score") or {}
            # group: prefer the yaml override (a single string for all); otherwise auto-extract
            # per-sample (legacy ``meta.info_key`` first prefix -> ``meta.category``).
            _yaml_group = _kwargs_tree_edit_score.get("group")
            _groups: List[Optional[str]] = []
            for _record in records:
                if _yaml_group is not None:
                    _groups.append(_yaml_group)
                    continue
                _meta_record = _record.get("meta", {}) or {}
                _info_key = _meta_record.get("info_key") or ""
                if isinstance(_info_key, str) and "_" in _info_key:
                    _groups.append(_info_key.split("_")[0])
                elif isinstance(_meta_record.get("category"), str) and _meta_record["category"]:
                    _groups.append(_meta_record["category"])
                else:
                    _groups.append(None)
            _scores, _sample_metrics = cls.compute_tree_edit(
                labels=labels,
                predictions=predictions,
                groups=_groups,
                api_name=_kwargs_tree_edit_score.get("api_name"),
                source_format=_kwargs_tree_edit_score.get("source_format"),
                max_tokens=_kwargs_tree_edit_score.get("max_tokens", 8192),
                temperature=_kwargs_tree_edit_score.get("temperature", 0.0),
                # Per-metric yaml override falls back to the resolved task-level
                # ``do_async`` (caller has already inline-OR'd args.do_async).
                do_async=_kwargs_tree_edit_score.get("do_async", do_async),
                semaphore_size=_kwargs_tree_edit_score.get("semaphore_size", 4),
            )
            metrics.update(_scores)
            sample_metrics.update(_sample_metrics)
        
        group_metrics = dict()
        if (
            isinstance(categories, (list, tuple))
            and len(categories) == len(predictions)
        ):
            for _metric_name, _metric_values in sample_metrics.items():
                for _record_idx, _metric_value in enumerate(_metric_values):
                    _group_names = categories[_record_idx]
                    if not _group_names:
                        continue
                    if isinstance(_group_names, str):
                        _group_names = [_group_names, ]
                    for _group_name in _group_names:
                        if not _group_name:
                            continue
                        if _group_name not in group_metrics:
                            group_metrics[_group_name] = defaultdict(list)
                        group_metrics[_group_name][_metric_name].append(_metric_value)
            for _group_name, _group_metrics in group_metrics.items():
                # ``num_samples`` is the count of records contributing to this
                # group — surfaced alongside the per-metric means so callers can
                # judge statistical weight (matches task-custom group_metrics
                # convention in htmlbench/mia/llavaw/kovisit).
                _group_num_samples = max(
                    (len(_v) for _v in _group_metrics.values() if isinstance(_v, list)),
                    default=0,
                )
                for _metric_name, _metric_values in _group_metrics.items():
                    _metric_values = [
                        _v if isinstance(_v, (int, float, np.floating)) else np.nan
                        for _v in _metric_values
                    ]
                    group_metrics[_group_name][_metric_name] = np.nanmean(_metric_values)
                group_metrics[_group_name] = dict(group_metrics[_group_name]) # defaultdict to dict for serialization
                group_metrics[_group_name]["num_samples"] = _group_num_samples
        
        return {
            "scores": metrics,
            "sample_scores": sample_metrics,
            "group_metrics": group_metrics if group_metrics else None,
        }

    @classmethod
    def compute_rouge(
        cls, 
        labels: List[List[str]], 
        predictions: List[str],
    ) -> Dict[str, float]:
        # keys: "rouge-1", "rouge-2", "rouge-L"
        metric = hf_evaluate.load("rouge")
        scores = metric.compute(predictions=predictions, references=labels)
        return scores

    @classmethod
    def compute_bleu(
        cls,
        labels: List[List[str]],
        predictions: List[str],
        method: Optional[str] = "bleu", # "bleu", "sacrebleu"
        max_orders: Optional[List[int]] = [1,2,3,4],
    ) -> Dict[str, float]:
        metric = hf_evaluate.load(method)

        output = dict()
        sample_metrics = dict()
        if method in [
            "bleu",
        ]:
            # keys: "bleu-1", "bleu-2", "bleu-3", "bleu-4"
            for max_order in max_orders:
                _scores = metric.compute(
                    predictions=predictions, 
                    references=labels, 
                    max_order=max_order,
                )
                output[f'bleu-{max_order}'] = _scores["bleu"]
        elif method in [
            "sacrebleu",
        ]:
            _scores = metric.compute(
                predictions=predictions, 
                references=labels, 
            )
            _bleus = list()
            for _prediction, _labels in zip(predictions, labels):
                _bleu = sacrebleu.sentence_bleu(_prediction, _labels)
                _bleus.append(_bleu.score)
            output[f'bleu'] = _scores["score"]
            sample_metrics[f'bleu'] = _bleus
        else:
            raise ValueError(f'invalid method for `compute_bleu`: {method}')
        return output, sample_metrics
    
    @classmethod
    def compute_meteor(
        cls, 
        labels: List[List[str]], 
        predictions: List[str],
    ) -> Dict[str, float]:
        metric = hf_evaluate.load("meteor")
        scores = metric.compute(predictions=predictions, references=labels)
        return scores
    
    @classmethod
    def compute_cider(
        cls, 
        labels: List[List[str]], 
        predictions: List[str],
    ) -> Dict[str, float]:
        labels = {
            _idx: _labels
            for _idx, _labels in enumerate(copy.deepcopy(labels))
        }
        predictions = {
            _idx: [_prediction, ]
            for _idx, _prediction in enumerate(copy.deepcopy(predictions))
        }
        score, sample_scores = Cider().compute_score(
            gts=labels,
            res=predictions,
        )
        scores = {"cider": score}
        sample_scores = {"cider": sample_scores}
        return scores, sample_scores
    
    # SPICE relies on the Stanford CoreNLP PCFG parser, whose memory/time cost
    # is ~O(n^3) in the number of tokens of a single sentence. Degenerate model
    # outputs (e.g. a phrase repeated hundreds of times within one sentence)
    # blow up the parser's heap and crash the whole `java` subprocess, aborting
    # the entire evaluation. Cap each caption to a sane word count to stay safe.
    # Verified safe at the hardcoded -Xmx8G heap: a 50-word cap tokenizes to
    # well under the length where the exhaustive PCFG parser exhausts the heap,
    # even with degenerate comma-separated repetition. (100 words still OOMs.)
    SPICE_MAX_WORDS = 50

    @classmethod
    def _truncate_for_spice(cls, text: str) -> str:
        if not isinstance(text, str):
            return text
        words = text.split()
        if len(words) <= cls.SPICE_MAX_WORDS:
            return text
        return " ".join(words[:cls.SPICE_MAX_WORDS])

    @classmethod
    def compute_spice(
        cls,
        labels: List[List[str]],
        predictions: List[str],
    ) -> Dict[str, float]:
        labels = {
            _idx: [cls._truncate_for_spice(_label) for _label in _labels]
            for _idx, _labels in enumerate(copy.deepcopy(labels))
        }
        predictions = {
            _idx: [cls._truncate_for_spice(_prediction), ]
            for _idx, _prediction in enumerate(copy.deepcopy(predictions))
        }
        score, sample_results = Spice().compute_score(
            gts=labels,
            res=predictions,
        )
        scores = {"spice": score}
        return scores, sample_results
    
    @classmethod
    def compute_spider(
        cls, 
        labels: List[List[str]], 
        predictions: List[str],
    ) -> Dict[str, float]:
        scores = dict()
        sample_scores = dict()
        _scores, _sample_scores = cls.compute_cider(
            labels=labels,
            predictions=predictions,
        )
        scores.update(_scores)
        sample_scores.update(_sample_scores)
        
        _scores, _sample_results = cls.compute_spice(
            labels=labels,
            predictions=predictions,
        )
        scores.update(_scores)
                
        if (
            scores.get("cider", None)
            and scores.get("spice", None)
        ):
            scores["spider"] = (scores["cider"] + scores["spice"]) / 2.0
        elif scores.get("cider", None):
            scores["spider"] = scores["cider"] / 2.0
        elif scores.get("spice", None):
            scores["spider"] = scores["spice"] / 2.0
        return scores, sample_scores
    
    @classmethod
    def compute_comet(
        cls,
        labels: List[List[str]],
        predictions: List[str],
        sources: List[str],
        model_name: Optional[str] = "wmt20-comet-da",
    ) -> Dict[str, float]:
        output = dict()
        sample_metrics = None

        # multi-reference support: flatten each sample's references, evaluate at once, then
        # aggregate per-sample by max afterward (the standard for multi-reference MT evaluation).
        _flat_srcs, _flat_preds, _flat_refs, _group_sizes = [], [], [], []
        for _src, _pred, _refs in zip(sources, predictions, labels):
            _refs_list = _refs if isinstance(_refs, (list, tuple)) else [_refs]
            _group_sizes.append(len(_refs_list))
            for _ref in _refs_list:
                _flat_srcs.append(_src)
                _flat_preds.append(_pred)
                _flat_refs.append(_ref)

        def _aggregate_max(flat_scores):
            agg, _idx = [], 0
            for _n in _group_sizes:
                if _n == 0:
                    agg.append(float("nan"))
                    continue
                agg.append(max(flat_scores[_idx:_idx + _n]))
                _idx += _n
            return agg

        try:
            import comet
            _model_path = comet.download_model(model_name)
            _model = comet.load_from_checkpoint(_model_path)
            _model_inputs = [
                {"src": _s, "mt": _p, "ref": _r}
                for _s, _p, _r in zip(_flat_srcs, _flat_preds, _flat_refs)
            ]
            _scores = _model.predict(
                _model_inputs,
                batch_size=8,
                gpus=1,
            )
            _sample_scores = _aggregate_max(_scores["scores"])
            _valid = [s for s in _sample_scores if s == s]
            output["comet"] = sum(_valid) / max(1, len(_valid))
            if sample_metrics is None:
                sample_metrics = defaultdict(list)
            sample_metrics["comet"] = _sample_scores

        except Exception as ex_unbabel:
            # Fallback to HF evaluate (which itself imports `comet`); if that
            # also fails — typically because ``unbabel-comet`` isn't installed
            # — degrade gracefully so other text metrics (bleu/wer/cer/...)
            # still surface. paper-strict comet requires ``pip install unbabel-comet``.
            try:
                metric = hf_evaluate.load("comet")
                _scores = metric.compute(
                    predictions=_flat_preds,
                    references=_flat_refs,
                    sources=_flat_srcs,
                )
                _sample_scores = _aggregate_max(_scores["scores"])
                _valid = [s for s in _sample_scores if s == s]
                output["comet"] = sum(_valid) / max(1, len(_valid))
                if sample_metrics is None:
                    sample_metrics = defaultdict(list)
                sample_metrics["comet"] = _sample_scores
            except Exception as ex_hf:
                logger.warning(
                    "compute_comet: failed to load COMET in both unbabel-comet "
                    "and hf_evaluate paths; returning NaN. Install "
                    "`unbabel-comet` (with model checkpoint download) to "
                    "enable paper-strict translation scoring. "
                    "unbabel-comet error=%r ; hf_evaluate error=%r",
                    ex_unbabel, ex_hf,
                )
                _sample_scores = [float("nan")] * len(predictions)
                output["comet"] = float("nan")
                if sample_metrics is None:
                    sample_metrics = defaultdict(list)
                sample_metrics["comet"] = _sample_scores
        return output, sample_metrics

    @staticmethod
    def _resolve_asr_normalizer(do_normalize: Optional[Union[bool, str]]):
        """Map a do_normalize flag to an AsrProcessor normalizer callable.

        - falsy (False / None / "") -> None (skip normalization)
        - True (bool) -> normalize_default (English; backward compatible)
        - "default" / "english" / "en" -> normalize_default
        - "korean" / "ko" -> normalize_korean
        - "chinese" / "zh" -> normalize_chinese
        - any other str -> None (let table-eval method strings like "wtq" pass through)
        """
        if not do_normalize:
            return None
        from omni_evaluator.postprocess.asr import AsrProcessor
        if isinstance(do_normalize, str):
            _key = do_normalize.lower()
            if _key in {"default", "english", "en"}:
                return AsrProcessor.normalize_default
            if _key in {"korean", "ko"}:
                return AsrProcessor.normalize_korean
            if _key in {"chinese", "zh", "zh_cn", "zh_hans"}:
                return AsrProcessor.normalize_chinese
            return None
        return AsrProcessor.normalize_default

    @classmethod
    def compute_wer(
        cls,
        labels: List[List[str]],
        predictions: List[str],
        do_normalize: Optional[Union[bool, str]] = False,
    ) -> float:
        metrics = {
            "wer": 1.0, # default: error_rate 100%
        }
        sample_metrics = defaultdict(list)

        _normalizer = cls._resolve_asr_normalizer(do_normalize)
        if _normalizer is not None:
            predictions = [
                _normalizer(text=_p) if isinstance(_p, str) else _p
                for _p in predictions
            ]
            labels = copy.deepcopy(labels)
            for _idx, _label in enumerate(labels):
                if isinstance(_label, str):
                    labels[_idx] = _normalizer(text=_label)
                elif isinstance(_label, (list, tuple)):
                    labels[_idx] = [
                        _normalizer(text=_l) if isinstance(_l, str) else _l
                        for _l in _label
                    ]

        num_edits, denom = 0, 0
        for _idx, (_prediction, _label) in enumerate(zip(predictions, labels)):
            if (
                _prediction is None
                or len(_prediction.split()) < 1
            ):
                _prediction = ""

            _wer, _num_edits, _denom = None, 0, 0
            if isinstance(_label, str):
                _label = [_label, ]
            for _label_ in _label:
                if (
                    _label_ is None
                    or len(_label_.strip()) < 1
                ):
                    continue

                _measures = jiwer.process_words(
                    reference=_label_,
                    hypothesis=_prediction,
                )
                _num_edits_ = _measures.substitutions + _measures.deletions + _measures.insertions
                _denom_ = _measures.substitutions + _measures.deletions + _measures.hits
                _wer_ = 1.0
                if _denom_ > 0:
                    _wer_ = _num_edits_ / _denom_
                if (
                    _wer is None
                    or _wer_ < _wer
                ):
                    _num_edits_word = _num_edits_
                    _denom_word = _denom_
                    _wer = _wer_

            num_edits += _num_edits_word
            denom += _denom_word
            sample_metrics["wer"].append(_wer)

        if denom > 0:
            metrics["wer"] = num_edits / denom
        return metrics, sample_metrics

    @classmethod
    def compute_cer(
        cls,
        labels: List[List[str]],
        predictions: List[str],
        do_normalize: Optional[Union[bool, str]] = False,
    ) -> float:
        metrics = {
            "cer": 1.0, # default: error_rate 100%
        }
        sample_metrics = defaultdict(list)

        _normalizer = cls._resolve_asr_normalizer(do_normalize)
        if _normalizer is not None:
            predictions = [
                _normalizer(text=_p) if isinstance(_p, str) else _p
                for _p in predictions
            ]
            labels = copy.deepcopy(labels)
            for _idx, _label in enumerate(labels):
                if isinstance(_label, str):
                    labels[_idx] = _normalizer(text=_label)
                elif isinstance(_label, (list, tuple)):
                    labels[_idx] = [
                        _normalizer(text=_l) if isinstance(_l, str) else _l
                        for _l in _label
                    ]

        num_edits, denom = 0, 0
        for _idx, (_prediction, _label) in enumerate(zip(predictions, labels)):
            if (
                _prediction is None
                or len(_prediction.split()) < 1
            ):
                _prediction = ""

            _cer, _num_edits, _denom = None, 0, 0
            if isinstance(_label, str):
                _label = [_label, ]
            for _label_ in _label:
                if (
                    _label_ is None
                    or len(_label_.strip()) < 1
                ):
                    continue

                _measures = jiwer.process_characters(
                    reference=_label_,
                    hypothesis=_prediction,
                )
                _num_edits_ = _measures.substitutions + _measures.deletions + _measures.insertions
                _denom_ = _measures.substitutions + _measures.deletions + _measures.hits
                _cer_ = 1.0
                if _denom_ > 0:
                    _cer_ = _num_edits_ / _denom_
                if (
                    _cer is None
                    or _cer_ < _cer
                ):
                    _num_edits = _num_edits_
                    _denom = _denom_
                    _cer = _cer_

            num_edits += _num_edits
            denom += _denom
            sample_metrics["cer"].append(_cer)

        if denom > 0:
            metrics["cer"] = num_edits / denom
        return metrics, sample_metrics

    @classmethod
    def compute_mer(
        cls,
        labels: List[str], 
        prediction: str,
        do_normalize: Optional[bool] = True,
    ):
        """
        Compute Mixed Error Rate (MER) for Korean-English code-switching.
        Exactly matches HiKE/src/metrics/mer.py mer function.
        """
        
        # Add space after all Korean characters to consider Korean characters as a single token.
        class SpaceKoreanChars:
            def __call__(self, text: str) -> str:
                text = re.sub(r'([\uAC00-\uD7A3])', r' \1 ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        
        def _normalize(text: str):
            _pipeline = jiwer.transforms.Compose([
                SpaceKoreanChars(),
                jiwer.transforms.Strip(),
            ])
            text = _pipeline(text)
            return text

        wer = 1.0
        stat = {
            "substitutions": 0,
            "insertions": 0,
            "deletions": 0,
            "num_tokens": 0,
        }

        if (
            prediction is None
            or len(prediction.split()) < 1
        ):
            return wer * 100, stat
        
        if do_normalize:
            prediction = _normalize(text=prediction)
            labels = [_normalize(text=_label) for _label in labels]
        
        for _label in labels:
            if (
                _label is None
                or len(_label.strip()) < 1
            ):
                continue
            
            _measures = jiwer.process_words(
                reference=_label, 
                hypothesis=prediction,
            )
            if _measures.wer < wer:
                stat["substitutions"] = _measures.substitutions
                stat["insertions"] = _measures.insertions
                stat["deletions"] = _measures.deletions
                stat["num_tokens"] = _label
                wer = _measures.wer

        return wer * 100, stat

    @classmethod
    def compute_pier(
        cls,
        labels: List[str], 
        prediction: str,
        dummy_token: str = "뷁",
    ):
        """
        Compute PIER (Point of Interest Error Rate) score.
        Exactly matches HiKE/src/metrics/__init__.py pier_fixed function.
        """
        # If there's no "rest" in reference, PIER always returns 0.
        # To prevent this, add dummy token out of poi to reference
        score, poi,rest = 1.0, None, None
        
        if (
            prediction is None
            or len(prediction.split()) < 1
        ):
            return score, poi, rest
            
        if not prediction.endswith(dummy_token):
            prediction = f'{prediction} {dummy_token}'

        for _label in labels:
            if not _label.endswith(dummy_token):
                _label = f'{_label} {dummy_token}' 
            try:
                _result = hike_pier(
                    reference=_label, 
                    hypothesis=prediction,
                )
                if _result["poi"]["PIER"] < score:
                    score = _result["poi"]["PIER"]
                    poi = _result["poi"]
                    rest = _result["rest"]
            except Exception as ex:
                pass
            
        return score, poi, rest

    @classmethod
    def compute_nlgeval(
        cls, 
        labels: Union[List[str], List[List[str]]], 
        predictions: List[str],
    ) -> float:
        from nlgeval import NLGEval

        if cls.nlgeval is None:
            try:
                cls.nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)
            except Exception as ex:
                cls.nlgeval = None
                logger.warning(f"Failed to load nlgeval package: {ex}")
                return dict()

        if isinstance(labels[0], str):  # nlgeval takes multi-references as an input
            labels = [[gt, ] for gt in labels]
        num_max_refs = max([len(gt) for gt in labels])
        new_labels = [["" for _ in range(0, len(labels))] for _ in range(0, num_max_refs)]
        for row_idx, gt in enumerate(labels):
            for col_idx, _label in enumerate(gt):
                new_labels[col_idx][row_idx] = _label
        _scores = cls.nlgeval.compute_metrics(ref_list=new_labels, hyp_list=predictions)

        keys = [
            "Bleu_1",
            "Bleu_2",
            "Bleu_3",
            "Bleu_4",
            "ROUGE_L",
            "METEOR",
            "CIDEr",
            # "SkipThoughtCS", "EmbeddingAverageCosineSimilarity", "VectorExtremaCosineSimilarity", "GreedyMatchingScore",
        ]
        new_keys = [
            "nlgeval_bleu-1",
            "nlgeval_bleu-2",
            "nlgeval_bleu-3",
            "nlgeval_bleu-4",
            "nlgeval_rouge-L",
            "nlgeval_meteor",
            "nlgeval_cider",
        ]
        scores = dict()
        for key, new_key in zip(keys, new_keys):
            if key not in _scores:
                logger.warning(f"Metric {key} not exists in nlgeval results: {_scores.keys()}")
                continue
            scores[new_key] = float(_scores.pop(key))
        return scores
    
    @classmethod
    def _get_repetition_model(cls) -> "RepetitionModel":
        """[Thread-Safe] Create the RepetitionModel once and return the cached instance."""
        if cls._repetition_model_instance is None:
            with cls._model_lock:
                if cls._repetition_model_instance is None:
                    # Initialize class without arguments
                    cls._repetition_model_instance = RepetitionModel()

        return cls._repetition_model_instance

    @classmethod
    def compute_repetition_prob(cls, pred: str) -> float:
        """Compute the repetition probability score using the ML model.
        The model is loaded only once per process.
        """
        model = cls._get_repetition_model()
        return model.predict_proba(pred)
    
    @classmethod
    def compute_bert_score(
        cls,
        labels: List[List[str]],
        predictions: List[str],
        model_type: str = "distilbert-base-uncased",
    ) -> Dict[str, List[float]]:
        metric = hf_evaluate.load("bertscore")
        for _idx, _prediction in enumerate(predictions):
            if _prediction is None:
                predictions[_idx] = ""

        output = metric.compute(
            predictions=predictions, 
            references=labels,
            model_type=model_type,
        )
        output = {
            f'bert_score_{k}': np.mean(v)
            for k, v in output.items()
            if isinstance(v, (list, tuple))
        }
        return output

    @classmethod
    def compute_jaccard_distance(
        cls,
        labels: List[str],
        prediction: str,
        n: int = 1,
    ):
        score = 1.0

        prediction = prediction.strip()
        if n == 1:
            prediction = set(prediction.split(" "))
        else:
            prediction = set(" ".join(ngram) for ngram in ngrams(prediction.split(" "), n=n))
        if len(prediction) < 1:
            return score

        for gt in labels:
            gt = gt.strip()
            if n == 1:
                gt = set(gt.split(" "))
            else:
                gt = set(" ".join(ngram) for ngram in ngrams(gt.split(" "), n=n))
            if len(gt) < 1:
                continue

            _score = 1.0 - len(prediction.intersection(gt)) / len(prediction.union(gt))
            if _score <= score:
                score = _score
        return score

    @classmethod
    def compute_ned(
        cls, 
        label: str, 
        prediction: str, 
        uncased: bool = True,
    ) -> float:
        if prediction is None:
            prediction = ""
        elif isinstance(prediction, (int, float)):
            prediction = str(prediction)
        if isinstance(label, (int, float)):
            label = str(label)
        prediction = prediction.strip()
        label = label.strip()
        if uncased:
            return float(edit_distance(prediction.lower(), label.lower()) / max(len(prediction), len(label)))
        else:
            return float(edit_distance(prediction, label) / max(len(prediction), len(label)))

    @classmethod
    def compute_levenshtein_distance(
        cls, 
        label: str, 
        prediction: str, 
        uncased: bool = True,
    ):
        # refernce: https://github.com/huggingface/evaluate/pull/412
        if prediction is None:
            prediction = ""
        elif isinstance(prediction, (int, float)):
            prediction = str(prediction)
        if isinstance(label, (int, float)):
            label = str(label)
        prediction = prediction.strip()
        label = label.strip()
        if uncased:
            label = label.lower()
            prediction = prediction.lower()

        m = len(label)
        n = len(prediction)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if label[i - 1] == prediction[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
        l = max(m, n)
        return float(d[m][n]) / l

    @classmethod
    def compute_anls(
        cls, 
        labels: str, 
        prediction: str,
        uncased: bool = True,
        threshold: float = 0.5,
    ) -> float:
        nl = min([
            cls.compute_ned(label=gt, prediction=prediction, uncased=uncased)
            for gt in labels
        ])
        # convert distance to similarity only when distance is less than threshold
        if nl < threshold:
            return 1 - nl
        else:
            return 0

    @classmethod
    def compute_exact_match(
        cls,
        label: str,
        prediction: str,
        relative_tolerance: Optional[float] = None,
        absolute_tolerance: Optional[float] = 1e-6,
    ) -> float:
        """
        # TODO: wtq
        if cls._is_date_string(gt) and cls._is_date_string(pred):
            gt, pred = dateparser.parse(gt), dateparser.parse(pred)
            if gt == pred: # when date values are equal
                return 1.0
            else: # when date values are not equal
                return 0.0
        """
        if TextEvaluator._is_numeric_string(label) and TextEvaluator._is_circled_number(prediction):
            prediction = parse_circled_answer(prediction)
        if (
            cls._is_numeric_string(label)
            and cls._is_numeric_string(prediction)
        ):
            label, prediction = float(label), float(prediction)
            if absolute_tolerance is None and relative_tolerance is None:
                return int(label == prediction)
            else:
                tolerance = relative_tolerance * abs(label) if relative_tolerance is not None else absolute_tolerance
                if abs(label - prediction) < tolerance:  # when numeric values are equal
                    return 1.0
                else:  # when numeric values are not equal
                    return 0.0

        elif str(label).strip().lower().replace(".", "") == str(prediction).strip().lower().replace(
            ".", ""
        ):  # when non-numeric values are equal
            return 1.0

        else:  # when non-numeric values are not equal
            return 0.0
        
    @classmethod
    def compute_substring_match(
        cls,
        label: str,
        prediction: str, 
        normalize: str = "squad",
    ):
        if normalize == "squad":
            prediction = cls.normalize_squad(prediction)
            label = cls.normalize_squad(label)
            
        prediction = SimpleTokenizer.tokenize(prediction, uncased=True)
        label = SimpleTokenizer.tokenize(label, uncased=True)

        for i in range(0, len(prediction) - len(label) + 1):
            if label == prediction[i:i+len(label)]:
                return 1.0
        return 0.0
    
    @classmethod
    def compute_string_match(
        cls,
        label: str,
        prediction: str, 
    ):
        if not label:
            label = ""
        if not prediction:
            prediction = ""
            
        label = label.strip().replace("\n"," ")
        prediction = prediction.strip().replace("\n"," ")
        if label in prediction:
            return 1.0
        else:
            return 0.0

    @classmethod
    def compute_f1(
        cls,
        labels: List[str],
        prediction: str,
        normalize: str = "squad",
        aggregate: str = "max",
    ) -> Dict[str, float]:
        def _compute_f1(prediction: str, ground_truth: str):
            if isinstance(normalize, str):
                prediction = cls.normalize(text=prediction, method=normalize)
                ground_truth = cls.normalize(text=ground_truth, method=normalize)
            pred_tokens = prediction.split()
            gt_tokens = ground_truth.split()

            precision, recall, f1 = None, None, None
            if len(pred_tokens) == 0 or len(gt_tokens) == 0:
                precision = 0 if len(pred_tokens) == 0 else 1  # If prediction is empty than precision is 0.
                recall = 1 if len(gt_tokens) == 0 else 0  # If reference is empty than recall is one.
                f1 = int(pred_tokens == gt_tokens)  # If either is empty, then F1 is 1 if they agree, 0 otherwise.
            else:
                common_tokens = Counter(pred_tokens) & Counter(gt_tokens)
                num_common = sum(common_tokens.values())
                if num_common == 0:
                    precision = 0
                    recall = 0
                    f1 = 0
                else:
                    precision = 1.0 * num_common / len(pred_tokens)
                    recall = 1.0 * num_common / len(gt_tokens)
                    f1 = (2 * precision * recall) / (precision + recall)

            return {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }

        try:
            temp_labels = ast.literal_eval(labels[0])
            if isinstance(temp_labels, list):
                labels = temp_labels  # convert to list when gts[0] is a list
        except Exception:
            # print(f'[warn] Can not convert str to list:{gts}')
            pass

        precisions, recalls, f1s = list(), list(), list()
        for gt in labels:
            _scores = _compute_f1(prediction, gt)
            precisions.append(_scores["precision"])
            recalls.append(_scores["recall"])
            f1s.append(_scores["f1"])
        if aggregate == "avg":
            return {
                "precision": np.mean(precisions),
                "recall": np.mean(recalls),
                "f1": np.mean(f1s),
            }
        else:  # default: "max":
            return {
                "precision": max(precisions),
                "recall": max(recalls),
                "f1": max(f1s),
            }

    @classmethod
    def compute_binary_f1(
        cls,
        labels: List[List[str]],
        predictions: List[str],
        **_kwargs,
    ) -> Dict[str, float]:
        """Binary Yes/No F1.

        ``predictions`` are expected to be strings; null-prediction handling is
        done upstream in :meth:`evaluate` via ``null_prediction_policy``.
        ``**_kwargs`` absorbs unrelated per-metric kwargs forwarded from
        ``target_metrics[<metric>]`` dispatch dict.
        """
        predictions = copy.deepcopy(predictions)
        labels = copy.deepcopy(labels)

        pos, neg = 1, 0
        for _idx, (_pred, _gts) in enumerate(zip(predictions, labels)):
            # Only keep the first sentence
            _pred = _pred.split(".")[0]
            _pred = _pred.replace(",", "")
            _tokens = [_token.lower() for _token in _pred.split(" ")]
            if (
                "no" in _tokens
                or "not" in _tokens
            ):
                predictions[_idx] = neg
            else:
                predictions[_idx] = pos

            # use the first label
            _gt = _gts
            if isinstance(_gts, (list, tuple)):
                _gt = _gts[0]
            if _gt == "no":
                labels[_idx] = neg
            else:
                labels[_idx] = pos

        if not predictions:
            return {"f1": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, label in zip(predictions, labels):
            if pred == pos and label == pos:
                TP += 1
            elif pred == pos and label == neg:
                FP += 1
            elif pred == neg and label == neg:
                TN += 1
            elif pred == neg and label == pos:
                FN += 1

        precision = float(TP) / float(TP + FP) if (TP + FP) > 0 else 0.0
        recall = float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        return {
            "f1": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
        
    @classmethod
    def compute_calibration_error(
        cls,
        predictions: List[str],
        labels: List[List[str]], 
        confidences: List[float],
        p_norm: Literal[1, "1", 2, "2", "inf", "infinity", ] = 2,
        bin_size: Optional[int] = 100,
    ) -> Dict[str, float]:
        """
        reference: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
        """
        corrects = None
        if not labels:
            corrects = copy.deepcopy(predictions)
        else:
            corrects = list()
            for _prediction, _labels in zip(predictions, labels):
                _correct = max(list(map(lambda _label: cls.compute_exact_match(
                    label=_label, prediction=_prediction,
                ), _labels)))
                corrects.append(_correct)
        accuracy = np.mean(corrects)
        confidence_interval_095 = 1.96 * np.sqrt(accuracy * (1 - accuracy) / len(corrects))
        
        confidences = copy.deepcopy(confidences)
        confidences = np.array(confidences)
        
        cur_idx = 0
        calibration_error = 0
        while cur_idx < len(confidences):
            corrects_bin = corrects[cur_idx:cur_idx+bin_size]
            confidences_bin = confidences[cur_idx:cur_idx+bin_size]
            difference = np.abs(np.nanmean(confidences_bin) - np.nanmean(corrects_bin))

            if p_norm in [1, "1", ]:
                _calibration_error = len(confidences_bin) / len(confidences) * difference
                calibration_error += _calibration_error
            elif p_norm in [2, "2", ]:
                _calibration_error = len(confidences_bin) / len(confidences) * np.square(difference)
                calibration_error += _calibration_error
            elif p_norm in ["inf", "infinity", ]:
                calibration_error = np.maximum(calibration_error, difference)
            else:
                raise ValueError(f'`p_norm` should be one of [1, "1", 2, "2", "inf", "infinity"]')
            
            cur_idx += bin_size
            
        if p_norm in [2, "2", ]:
            calibration_error = np.sqrt(calibration_error)
        
        return {
            "accuracy": accuracy, 
            "confidence_interval": confidence_interval_095,
            "calibration_error": calibration_error,
        }
    
    @classmethod
    def compute_vqaeval(
        cls, 
        labels: List[str], 
        prediction: str,
    ):
        if prediction is None:
            prediction = ""
        # preproess pred
        # pred = pred.replace("\n", " ").replace("\t", " ").strip()

        # from https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/llava/eval/m4c_evaluator.py#L181
        prediction = prediction.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()
        prediction = cls._vqa__processPunctuation(prediction)
        prediction = cls._vqa__processDigitArticle(prediction)

        score = 0.0

        try:
            temp_labels = ast.literal_eval(labels[0])
            if isinstance(temp_labels, list):
                labels = temp_labels  # convert to list when gts[0] is a list (e.g., TextVQA with multiple free-form answers)
        except Exception:
            # print(f'[warn] Can not convert str to list:{gts}')
            pass

        for gt in labels:
            # preproess gt
            # gt = gt.replace("\n", " ").replace("\t", " ").strip()
            gt = gt.lower().replace(",", "").replace("?", "").replace("'s", " 's").strip()
            gt = cls._vqa__processPunctuation(gt)
            gt = cls._vqa__processDigitArticle(gt)
            if gt == prediction:
                score += 1.0
            if score >= 3:
                return 1.0
        return 0 if score == 0 else score / 3
    
    @classmethod
    def compute_ocrbench_vqaeval(
        cls, 
        labels: List[str], 
        prediction: str,
        lang: str = "en",
        uncased: bool = False,
    ):
        """
        reference: https://github.com/Yuliang-Liu/MultimodalOCR/blob/main/OCRBench_v2/eval_scripts/vqa_metric.py
        """
        if prediction is None:
            prediction = ""

        if lang == "zh":
            prediction = prediction.strip().replace("\n"," ").replace(" ", "")
            labels = [
                _label.strip().replace("\n"," ").replace(" ", "")
                for _label in labels
            ]
        else:
            prediction = prediction.strip().replace("\n"," ")
            labels = [
                _label.strip().replace("\n"," ")
                for _label in labels
            ]
        if uncased:
            prediction = prediction.lower()
            labels = [
                _label.lower()
                for _label in labels
            ]

        score = 0.0
        for _label in labels:
            if (
                lang == "zh"
                and len(_label.split()) < 4
            ) or (
                lang != "zh"
                and len(_label.split()) < 5
            ):
                if _label in prediction:
                    score = 1
                    break
            else:
                _distance = cls.compute_levenshtein_distance(
                    label=_label,
                    prediction=prediction,
                    uncased=uncased,
                )
                _anls_score = 1 - _distance
                if _anls_score >= 0.5:
                    score = max(score, _anls_score)
        
        return score

    @classmethod
    def compute_wtq_vqaeval(
        cls,
        label: str, 
        prediction: str,
    ):
        original_strings = tsv_unescape_list(label)
        target_values = to_value_list(original_strings)

        predicted_strings = tsv_unescape_list(prediction)
        predicted_values = to_value_list(predicted_strings)
        correct = check_denotation(target_values, predicted_values)
        if correct: 
            return 1
        else: 
            return 0

    @classmethod
    def compute_fintabnet_vqaeval(
        cls,
        label: str, 
        predictin: str,
    ):
        predictin, preds = cls.normalize(text=predictin, method="fintabnet")
        label, gts = cls.normalize(text=label, method="fintabnet")

        correct = any(_pred == _label for _pred in preds for _label in gts)
        if correct:
            score = 1
        else:
            score = 0

        return score

    @classmethod
    def compute_mmmu_accuracy(
        cls,
        prediction: str,
        labels: List[str],
        options: Optional[List[str]] = None,
        option_contents: Optional[List[str]] = None,
        question_type: Optional[str] = None,
    ) -> float:
        """Upstream-equivalent MMMU accuracy (0.0 / 1.0).

        Delegates to ``omni_evaluator.evaluation.metrics.mmmu_accuracy.score_row``,
        which routes between multi-choice parsing (parse_multi_choice_response
        + eval_multi_choice) and open-ended parsing (parse_open_response +
        eval_open) based on ``question_type``. Falls back to MC when options
        are present and question_type is missing.
        """
        return mmmu_score_row(
            prediction=prediction,
            labels=labels,
            options=options,
            option_contents=option_contents,
            question_type=question_type,
        )

    _TEMPORAL_PAIR_RE = re.compile(
        r"(-?\d+(?:\.\d+)?)\s*(?:-|–|—|to|,)\s*(-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    _TEMPORAL_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")

    @classmethod
    def compute_temporal_iou(
        cls,
        label: str,
        prediction: str,
    ) -> Dict[str, float]:
        """Temporal IoU + threshold-recall flags for a single sample.

        Returns a dict so the caller can emit multiple ``sample_metrics`` keys
        (``temporal_iou``, ``temporal_recall@0.3/0.5/0.7``) from one dispatch.
        Aggregating each across the run yields mIoU and R@1@IoU=X.

        Unparseable prediction or label is treated as a miss (all zeros) rather
        than skipped, matching upstream Charades-STA / eval_tvg semantics.
        """
        def _parse_time_interval(text: str) -> Optional[Tuple[float, float]]:
            # Accept decimals/integers with separators ``-``, ``–``, ``—``,
            # ``to`` or comma (e.g. ``"24.3 - 30.4"``, ``"24.3 to 30.4"``,
            # ``"[24.3, 30.4]"``). Falls back to the first two floats in the
            # string when no separator pattern matches.
            if not isinstance(text, str) or not text.strip():
                return None
            # strip CoT trace; only post-think output is searched.
            think_end = text.rfind("</think>")
            if think_end != -1:
                text = text[think_end + len("</think>"):].strip()
            for s, e in cls._TEMPORAL_PAIR_RE.findall(text):
                try:
                    s_f, e_f = float(s), float(e)
                except ValueError:
                    continue
                if 0.0 <= s_f < e_f <= 1e6:
                    return (s_f, e_f)
            floats = cls._TEMPORAL_FLOAT_RE.findall(text)
            if len(floats) >= 2:
                try:
                    s_f, e_f = float(floats[0]), float(floats[1])
                except ValueError:
                    return None
                if 0.0 <= s_f < e_f <= 1e6:
                    return (s_f, e_f)
            return None

        def _interval_iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
            union = max(a[1], b[1]) - min(a[0], b[0])
            return inter / union if union > 0 else 0.0

        gt = _parse_time_interval(label)
        pred = _parse_time_interval(prediction)
        if gt is None or pred is None:
            return {"iou": 0.0, "r@0.3": 0.0, "r@0.5": 0.0, "r@0.7": 0.0}
        iou = _interval_iou(gt, pred)
        return {
            "iou": iou,
            "r@0.3": float(iou >= 0.3),
            "r@0.5": float(iou >= 0.5),
            "r@0.7": float(iou >= 0.7),
        }

    @classmethod
    def compute_click_dist_accuracy(
        cls,
        label,
        prediction,
        image_w: Optional[int] = None,
        image_h: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> int:
        """Spatial-grounding click accuracy for a single sample.

        Returns ``1`` iff the predicted click point lies within *threshold*
        euclidean distance of the GT, else ``0``.

        GT labels: point ``[x, y]`` used as-is, bbox ``[x1, y1, x2, y2]`` reduced
        to its center.  A prediction that does not parse scores ``0``.
        """
        if threshold is None:
            threshold = 0.05

        def _to_point(value):
            """Coerce a ``[x, y]`` or ``[x1, y1, x2, y2]`` literal (or its JSON
            string form) to a single ``[x, y]`` point. None on any failure."""
            try:
                if isinstance(value, str):
                    value = json.loads(value)
                if not isinstance(value, (list, tuple)):
                    return None
                if len(value) == _ARITY.get(SpatialGroundingType.BBOX, 4):
                    return [(float(value[0]) + float(value[2])) / 2.0,
                            (float(value[1]) + float(value[3])) / 2.0]
                if len(value) >= _ARITY.get(SpatialGroundingType.POINT, 2):
                    return [float(value[0]), float(value[1])]
            except (ValueError, TypeError, IndexError, KeyError,
                    json.JSONDecodeError):
                return None
            return None

        gt_pt = _to_point(label)
        pred_pt = _to_point(prediction)
        if gt_pt is None or pred_pt is None:
            return 0
        gx = cls._grounding_norm_axis(gt_pt[0], image_w)
        gy = cls._grounding_norm_axis(gt_pt[1], image_h)
        px = cls._grounding_norm_axis(pred_pt[0], image_w)
        py = cls._grounding_norm_axis(pred_pt[1], image_h)
        return 1 if math.hypot(px - gx, py - gy) <= threshold else 0

    @classmethod
    def compute_html_tree_distance(cls, gt: str, pred: str) -> float:
        """zss tree-edit distance between two HTML strings.

        Uses ``create_tree`` + cost fns from ``omni_evaluator.evaluation.metrics.html``.
        Degenerate (empty/None tree) → ``max(len(gt), len(pred))`` so the caller's
        normalize denominator still produces a meaningful 0-score, matching the
        legacy ``_get_dist`` semantics.
        """
        gt_tree = create_html_tree(gt)
        pred_tree = create_html_tree(pred)
        if gt_tree is None or pred_tree is None:
            return float(max(len(gt or ""), len(pred or "")))
        return _zss.distance(
            pred_tree,
            gt_tree,
            get_children=_zss.Node.get_children,
            insert_cost=_get_insert_cost_html,
            remove_cost=_get_remove_cost_html,
            update_cost=_get_update_cost_html,
        )

    @classmethod
    def compute_tree_edit(
        cls,
        labels: List[List[str]],
        predictions: List[str],
        groups: Optional[List[Optional[str]]] = None,
        api_name: Optional[str] = None,
        source_format: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        do_async: bool = False,
        semaphore_size: int = 4,
    ) -> Tuple[Dict[str, float], Dict[str, List[Optional[float]]]]:
        """Batch HTML tree-edit-distance score in [0, 1] with optional stage-1
        LLM conversion — equivalent to legacy ``eval_html.get_score``.

        - Direct mode (no ``api_name``/``source_format``): predictions consumed
          as-is — htmlbench_ko semantics. Multi-label ``max`` per sample.
        - Stage-1 mode (``api_name`` + ``source_format`` both set): predictions
          first converted via ``metrics.html.generate_html`` in a single batch
          call, then scored — latex/markdown semantics.

        Per-sample ``group`` ∈ {"table", "equation", None}:
          table    → keep <table> only on both sides
          equation → keep <math> only on both sides
          None     → no extra parser (parse_html + remove_html_tag still applied)

        Sample score is ``1 - dist / max(gt_dist_from_empty, pred_dist_from_empty)``
        clipped to [0, 1] and NaN-safe. Tree-construction helpers + zss cost
        functions live in ``omni_evaluator.evaluation.metrics.html`` (same
        sub-module split as ``mmmu_accuracy`` / ``wtq`` / ``pier``).

        Returns the standard batch-metric tuple ``({"tree_edit_score": mean},
        {"tree_edit_score": per-sample list})`` matching ``compute_bleu`` /
        ``compute_wer`` etc.
        """
        def _score_pair(label: str, prediction: str, group: Optional[str]) -> float:
            gt = parse_html(label) if isinstance(label, str) else ""
            pred = parse_html(prediction) if isinstance(prediction, str) else ""
            if group == "table":
                gt, pred = parse_html_table(gt), parse_html_table(pred)
            elif group == "equation":
                gt, pred = parse_html_math(gt), parse_html_math(pred)
            gt, pred = remove_html_tag(gt), remove_html_tag(pred)
            try:
                dist = cls.compute_html_tree_distance(gt, pred)
                gt_dist = cls.compute_html_tree_distance(gt, "")
                pred_dist = cls.compute_html_tree_distance(pred, "")
                denom = max(gt_dist, pred_dist, 1.0)
                score = max(1.0 - dist / denom, 0.0)
                return 0.0 if score != score else float(score)  # NaN-safe
            except Exception:
                return 0.0

        if api_name and source_format:
            # Defer the import: avoids a circular dependency at module load
            # time and keeps the optional LLM client out of the metric's
            # cold-start path when stage-1 isn't used.
            from omni_evaluator.evaluation.metrics.html import generate_html
            predictions = generate_html(
                predictions=list(predictions),
                groups=groups,
                api_name=api_name,
                source_format=source_format,
                max_tokens=max_tokens,
                temperature=temperature,
                do_async=do_async,
                semaphore_size=semaphore_size,
            )
        if groups is None:
            groups = [None] * len(predictions)

        _sample_scores: List[Optional[float]] = []
        for _labels, _prediction, _group in zip(labels, predictions, groups):
            if not _labels or not _prediction:
                _sample_scores.append(None)
                continue
            _best: Optional[float] = None
            for _label in _labels:
                _s = _score_pair(_label, _prediction, _group)
                if _best is None or _s > _best:
                    _best = _s
            _sample_scores.append(_best)

        _valid = [_s for _s in _sample_scores if _s is not None]
        _mean = float(np.nanmean(_valid)) if _valid else 0.0
        return (
            {"tree_edit_score": _mean},
            {"tree_edit_score": _sample_scores},
        )

    @classmethod
    def compute_mmau_string_match(
        cls,
        label: str,
        prediction: str,
        option_contents: List[str],
        options: Optional[List[str]] = None,
    ):
        """Content-token overlap score for multiple-choice audio QA.

        Returns 1.0 iff the prediction (a) reproduces every content word of
        the gold option and (b) contains no content word that is unique to a
        wrong option. Returns 0.0 otherwise; ``False`` on an empty prediction.

        If the whole prediction is a bare option letter (``A`` / ``(A)`` /
        ``A.``) and an aligned ``options`` / ``option_contents`` mapping is
        provided, the corresponding option content is spliced in first so a
        letter-only answer can still be scored on its content.
        """
        _WORD_RE = re.compile(r"\b\w+\b")
        _BARE_LETTER_RE = re.compile(r"\s*[\(\[]?\s*([A-Za-z])\s*[\)\].:]?\s*")
        _OPTION_PREFIX_RE = re.compile(r"^\s*\(?[A-Z]\)?\s*[:\.\)]\s*")

        def _content_words(text: str) -> set:
            return set(_WORD_RE.findall(text.lower()))

        # Numeric answers stringify identically on both sides.
        if isinstance(prediction, (int, float)):
            prediction = str(prediction)
        if isinstance(label, (int, float)):
            label = str(label)

        # Letter-only prediction → append the chosen option's content so the
        # word-overlap check below has real content to match against. Only when
        # the ENTIRE prediction is one letter — a content answer starting with
        # "A"/"An" (e.g. "A woman ...") must not trigger this substitution.
        if (
            isinstance(prediction, str)
            and options
            and option_contents
            and len(options) == len(option_contents)
        ):
            _letter_match = _BARE_LETTER_RE.fullmatch(prediction)
            if _letter_match:
                _letter = _letter_match.group(1).upper()
                _option_letters_upper = [str(o).strip().upper() for o in options]
                if _letter in _option_letters_upper:
                    _idx = _option_letters_upper.index(_letter)
                    prediction = f"{prediction} {option_contents[_idx]}"

        # Drop any leading "(A) " / "A) " / "A. " marker from the gold label
        # and from each option so the tokenization below sees content only.
        # Detect-then-strip (not a blind slice) — some upstream callers already
        # deliver stripped values.
        label = _OPTION_PREFIX_RE.sub("", label)
        option_contents = [_OPTION_PREFIX_RE.sub("", oc) for oc in option_contents]

        pred_words = _content_words(prediction)
        gold_words = _content_words(label)

        if not pred_words:
            return False

        # Words appearing in a wrong option but NOT in the gold option —
        # touching any of them means the prediction leaked a distractor.
        distractor_words: set = set()
        for oc in option_contents:
            oc_words = _content_words(oc)
            if oc_words != gold_words:
                distractor_words |= oc_words - gold_words

        covers_gold = gold_words.issubset(pred_words)
        avoids_distractors = pred_words.isdisjoint(distractor_words)
        return 1.0 if (covers_gold and avoids_distractors) else 0.0

    @classmethod
    def _is_numeric_string(cls, text: str) -> bool:
        if not isinstance(text, str):
            return False
        try:
            float(text)
            return True
        except ValueError as ex:
            return False

    @classmethod
    def _is_circled_number(cls, text: str) -> bool:
        return text in vqa__circledNumbersMap

    @classmethod
    def _is_date_string(cls, text: str) -> bool:
        try:
            date_str = dateparser.parse(text)
            if date_str is not None:
                return True
            else:
                return False
        except Exception as ex:
            return False

    @classmethod
    def normalize(
        cls,
        text: str,
        method: Optional[str] = "wtq",
        **kwargs,
    ):
        if method == "squad":
            return cls._normalize_squad(
                text=text,
                **kwargs
            )
        elif method == "fintabnet":
            return cls._normalize_fintabnet(
                text=text,
                **kwargs
            )
        else: # "wtq" # default
            return cls._normalize_wtq(
                text=text,
                **kwargs
            )

    @classmethod
    def _normalize_wtq(
        cls,
        text: str,
        **kwargs,
    ):
        return _wtq_normalize(text)
    
    @classmethod
    def _normalize_squad(
        cls,
        text: str,
        **kwargs,
    ):
        text = text.lower()
        text = "".join(ch for ch in text if ch not in set(string.punctuation))
        text = regex.sub(r"\b(a|an|the)\b", " ", text)
        text = " ".join(text.split())
        return text

    @classmethod
    def _normalize_fintabnet(
        cls,
        text: str,
        **kwargs,
    ):
        remove_words = [
            "dollar",
            "gallons",
            "square feet",
            "shares",
            "mbtu",
            "mbpd",
            "mbbls",
            "mmbtu",
            "unit",
            "gwh",
            "year",
            "mmcf",
            "mile",
            "mboe",
        ]

        # Data specific filtering using regular expressions
        # Remove special characters like $, (, and )
        text = re.sub(r"[\$\(\),]", "", text)

        # Replace "dollar" with empty string if it's not part of another word
        # Build regex pattern; append 's?' to handle optional plural suffix
        pattern = r"\b(" + "|".join(remove_words) + r")s?\b"

        # Iterate strings and remove specified words
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Unit conversion dictionary with regex patterns for flexibility
        unit_conversion = {
            r" \bthousand\b": "e3",
            r" \bmillion\b": "e6",
            r" \bbillion\b": "e9",
            r"\bthousand\b": "e3",
            r"\bmillion\b": "e6",
            r"\bbillion\b": "e9",
            r" ?%": "e-2",
        }

        # Convert percentages to their decimal representation.
        # Applying this after unit_conversion prevents "percent" from being processed
        # in cases like "million %", which would be incorrect.
        # s = re.sub(r' ?%', 'e-2', s)
        # s_percent = re.sub(r' ?%', '', s_percent)

        s_unit_free = text

        # Iterate over unit_conversion and apply transformations
        for pattern, value in unit_conversion.items():
            text = re.sub(pattern, value, text)
            s_unit_free = re.sub(pattern, "", s_unit_free)

        # Attempt to convert to float
        try:
            return float(text), [float(text), float(s_unit_free)]
        except ValueError as e:
            # Return the original string and the error for debugging purposes
            return text, [text, s_unit_free]

    @classmethod
    # Reference from https://github.com/facebookresearch/mmf (BSD-3-Clause) - EvalAIAnswerProcessor.processPunctuation
    def _vqa__processPunctuation(cls, text: str):
        outText = text
        for p in vqa__punct:
            if (p + " " in text or " " + p in text) or (re.search(vqa__commaStrip, text) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        # Remove all periods not connected to digits
        outText = vqa__periodStrip.sub("", outText, re.UNICODE)
        return outText

    @classmethod
    # Reference from https://github.com/facebookresearch/mmf (BSD-3-Clause) - EvalAIAnswerProcessor.processDigitArticle
    def _vqa__processDigitArticle(cls, text: str):
        outText = []
        tempText = text.lower().split()
        for word in tempText:
            word = vqa__manualMap.setdefault(word, word)
            if word not in vqa__articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in vqa__contractions:
                outText[wordId] = vqa__contractions[word]
        outText = " ".join(outText)
        return outText
    
class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'
    REGEXP = regex.compile(
        '(%s)|(%s)' % (ALPHA_NUM, NON_WS),
        flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
    )

    @classmethod
    def tokenize(
        cls, 
        text: str, 
        uncased: bool = False,
    ):
        matches = [m for m in cls.REGEXP.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens     