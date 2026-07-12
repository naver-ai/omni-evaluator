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

import importlib
import logging
import pandas as pd
from typing import Any, List, Tuple, Dict, Union, Optional

from omni_evaluator import EvaluationEngine
from omni_evaluator.utils.string import is_integer

logger = logging.getLogger(__name__)


class LeaderboardFormatter:
    leaderboards = [
        "docvqa",
        "infovqa",
        "chartqa",
        "stvqa",
        "textvqa",
        "vqav2",
        "vizwiz",
        "mmbench",
        "mmau_test",
    ]
    
    @classmethod
    def format(
        cls,
        benchmark: str,
        records: List[Dict[str, Any]],
    ):
        bench_lower = benchmark.lower()

        # Skip benchmarks/splits that don't have a public submission flow.
        # - chartqa: no official leaderboard.
        # - non-test splits (validation/dev/train): local debugging artifacts,
        #   their sample IDs follow internal conventions (not server-side IDs).
        if "chartqa" in bench_lower:
            return None
        if any(
            _marker in bench_lower
            for _marker in ("validation", "_dev", "dev_", "_train", "train_")
        ):
            return None

        # rows = list()
        output = None
        if (
            "docvqa" in bench_lower
            or "infovqa" in bench_lower
        ):  # RRC format
            output = cls._format_rrc(
                benchmark=benchmark,
                records=records,
            )

        elif (
            "stvqa" in benchmark.lower()
            or "textvqa" in benchmark.lower()
            or "vqav2" in benchmark.lower()
        ): # EvalAI format
            output = cls._format_evalai(
                benchmark=benchmark,
                records=records,
            )
        
        elif (
            "mmau_test" in benchmark.lower()
        ): # EvalAI format
            output = cls._format_evalai_mmau(
                benchmark=benchmark,
                records=records,
            )

        elif "vizwiz" in benchmark.lower():
            output = cls._format_vizwiz(
                benchmark=benchmark,
                records=records,
            )
            
        elif "mmbench" in benchmark.lower():
            output = cls._format_mmbench(
                benchmark=benchmark,
                records=records,
            )

        # else:
        #     raise ValueError(f'# [error] no releated learderboard for benchmark {benchmark}\n\tavailable leaderboards: {cls.leaderboards}')

        if output is not None:
            logger.info(f'Formatted to submit {benchmark}: {len(output)}')
        return output
    
    @classmethod
    def _format_rrc(
        cls,
        benchmark: str,
        records: List[Dict[str, Any]],
    ):
        output = list()
        for _record in records:
            _index = _record["index"]
            _prediction = _record["prediction"]
            if _record["prediction_postprocessed"]:
                _prediction = _record["prediction_postprocessed"]
            
            if _prediction is None:
                _prediction = "" # failed prediction
            if not (isinstance(_index, (int, float)) or (isinstance(_index, str) and is_integer(_index))):
                raise TypeError(f'Invalid questionId: {_index}')

            if isinstance(_index, str):
                _index = int(_index)
            output.append({
                "questionId": _index,
                "answer": _prediction,
            })
        return output
    
    @classmethod
    def _format_chartqa(
        cls,
        benchmark: str,
        records: List[Dict[str, Any]],
    ):        
        output = list()
        for _record in records:
            _index = _record["index"]
            _prediction = _record["prediction"]
            if _record["prediction_postprocessed"]:
                _prediction = _record["prediction_postprocessed"]
            _label = _record["label"][0]
            
            if _prediction is None:
                _prediction = "" # failed prediction
            if not (isinstance(_index, (int, float)) or (isinstance(_index, str) and is_integer(_index))):
                raise TypeError(f'Invalid questionId: {_index}')

            if isinstance(_index, str):
                _index = int(_index)
            output.append({
                "questionId": _index,
                "prediction": _prediction,
                "gt": _label,
            })
        return output
    
    @classmethod
    def _format_evalai(
        cls,
        benchmark: str,
        records: List[Dict[str, Any]],
    ):
        output = list()
        for _record in records:
            _index = _record["index"]
            _prediction = _record["prediction"]
            if _record["prediction_postprocessed"]:
                _prediction = _record["prediction_postprocessed"]
            
            if _prediction is None:
                _prediction = "" # failed prediction
            if not isinstance(_index, (int, float, str)):
                raise TypeError(f'Invalid questionId: {_index}')

            if (
                isinstance(_index, str)
                and is_integer(x=_index)
            ):
                _index = int(_index)

            _output = {
                "question_id": _index,
                "answer": _prediction,
            }
            output.append(_output)
        return output
    
    @classmethod
    def _format_evalai_mmau(
        cls,
        benchmark: str,
        records: List[Dict[str, Any]],
    ):
        output = list()
        for _record in records:
            _index = _record["index"]
            _prediction = _record["prediction"]
            if _record["prediction_postprocessed"]:
                _prediction = _record["prediction_postprocessed"]
            
            if _prediction is None:
                _prediction = "" # failed prediction
            if not isinstance(_index, (int, float, str)):
                raise TypeError(f'Invalid questionId: {_index}')

            if (
                isinstance(_index, str)
                and is_integer(x=_index)
            ):
                _index = int(_index)

            _output = {
                "id": _index,
                "question": _record["meta"].get("question", ""),
                "model_prediction": _prediction,
                "answer": _record["meta"].get("answer", ""),
                "choices": _record["meta"].get("choices", ""),
                "task": _record["meta"].get("task", ""),
                "difficulty": _record["meta"].get("difficulty", ""),
            }
            output.append(_output)
        return output
    
    @classmethod
    def _format_vizwiz(
        cls,
        benchmark: str,
        records: List[Dict[str, Any]],
    ):        
        output = list()
        for _record in records:
            _index = _record["index"]
            _prediction = _record["prediction"]
            if _record["prediction_postprocessed"]:
                _prediction = _record["prediction_postprocessed"]
            
            if _prediction is None:
                _prediction = "" # failed prediction
            
            if not _index.startswith("VizWiz_test_"):
                if not is_integer(_index):
                    raise ValueError(f'Invalid questionId: {_index}')
                _index = f'VizWiz_test_{int(_index):08d}.jpg'
            _prediction = _prediction.lower()
            output.append({
                "image": _index,
                "answer": _prediction,
            })
        return output
        
    @classmethod
    def _format_mmbench(
        cls,
        benchmark: str,
        records: List[Dict[str, Any]],
    ):
        output = None
        with importlib.resources.files(
            "omni_evaluator.evaluation.resources.data",
        ).joinpath("MMBench_TEST_EN.tsv").open("r") as fp:
            output = pd.read_csv(fp, sep="\t")
                        
        _prediction_map = dict()
        for _record in records:
            _index = _record["index"]
            _prediction = _record["prediction"]
            if _record["prediction_postprocessed"]:
                _prediction = _record["prediction_postprocessed"]
            
            if _prediction is None:
                _prediction = "" # failed prediction

            _prediction_map[_index] = _prediction

        _predictions = list()
        for _index in output["index"]:
            _prediction = _prediction_map.get(str(_index), "")
            _predictions.append(_prediction)
        output.insert(10, "prediction", _predictions)
        output = output.drop("image", axis=1)
        output = output.to_dict(orient="records")
        return output
    
    