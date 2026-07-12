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

from collections import defaultdict, OrderedDict
import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import partial
import importlib
import json
import logging
import numpy as np
from typing import Union, Any, Callable, Tuple, List, Dict, Optional

from omni_evaluator.inference.prompts import SYSTEM_PROMPTS
from omni_evaluator.postprocess import CodeProcessor, MultichoiceProcessor
from omni_evaluator.schemas import SchemaInterface
from omni_evaluator.enums.engine import EvaluationEngine, EvaluationMethod, InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class EvaluationStatistics(SchemaInterface):
    """statistics for a single evaluation run."""
    avg_num_chars: Optional[int] = None
    avg_num_tokens: Optional[int] = None
    avg_num_words: Optional[int] = None

@dataclass
class EvaluationRunOutput(SchemaInterface):
    """Metrics and metadata for a single evaluation run."""

    inference_engine: Optional[InferenceEngine] = None
    evaluation_engine: Optional[EvaluationEngine] = None
    task_name: Optional[str] = None
    evaluation_method: Optional[EvaluationMethod] = None
    run_index: Optional[int] = 1
    num_runs: Optional[int] = 1
    num_samples: Optional[int] = None
    num_empty_predictions: Optional[int] = None
    coverage_inference: Optional[float] = None
    coverage_evaluation: Optional[float] = None
    latency: Optional[float] = None
    throughput: Optional[float] = None
    runtime_inference: Optional[float] = None
    runtime_evaluation: Optional[float] = None
    runtime_postprocess: Optional[float] = None
    statistics: Optional[Dict[str, Any]] = None
    metric_keys: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    group_metrics: Optional[Dict[str, Any]] = None
    sample_metrics: Optional[List[Dict[str, Any]]] = None
    outputs: Optional[Dict[str, Any]] = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'
    
    def __post_init__(self):
        if isinstance(self.metrics, dict):
            for _k, _v in self.metrics.items():
                if isinstance(_v, np.floating):
                    self.metrics[_k] = float(_v)
                elif isinstance(_v, np.integer):
                    self.metrics[_k] = int(_v)

    @classmethod
    def from_task(
        cls,
        args,
        task_name: str,
        task_config,
        records: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        group_metrics: Optional[Dict[str, Any]] = None,
        sample_metrics: Optional[List[Dict[str, Any]]] = None,
        num_valid_evaluation: Optional[int] = None,
        default_metric_keys: Optional[List[str]] = None,
        **overrides: Any,
    ) -> "EvaluationRunOutput":
        """Build ``EvaluationRunOutput`` from a finished task ``evaluate()`` run.

        Computes the per-run boilerplate that every task otherwise repeats inline
        (coverage_inference / coverage_evaluation / num_empty_predictions /
        num_samples / metric_keys fallback) from ``records`` + sample_metrics,
        then assembles the dataclass. ``overrides`` go straight to the ctor —
        use them to set non-default fields like ``runtime_*`` when a task needs them.

        Args:
          args: parsed CLI ``argparse.Namespace`` or ``None``. When provided,
            ``inference_engine`` / ``evaluation_engine`` attributes are read;
            otherwise both default to ``None`` and the outer loop fills them.
          task_name: task identifier used in the output.
          task_config: ``TaskConfig`` — its ``num_records`` and ``evaluation``
            attributes drive sample counting and ``display_metrics`` fallback.
          records: per-sample input records. ``record["prediction"]`` truthiness
            decides ``num_empty_predictions``.
          metrics / group_metrics / sample_metrics: as returned by the task's
            metric computation.
          num_valid_evaluation: number of samples that produced a valid score.
            Defaults to the count of truthy entries in ``sample_metrics`` —
            tasks whose ``sample_metrics`` is ``None`` or whose validity count
            differs from the truthy count (e.g., voice_bench accumulates over
            subtasks) must pass this explicitly.
          default_metric_keys: metric keys to surface when the task's
            ``display_metrics`` config is empty.
        """
        if sample_metrics is None and num_valid_evaluation is None:
            raise ValueError(
                "from_task: either sample_metrics or num_valid_evaluation must be provided. "
                "sample_metrics=None silently yields coverage_evaluation=0.0 otherwise."
            )
        _num_samples = (task_config.num_records if task_config else None) or len(records)
        _num_valid_inferences = sum(1 for _r in records if _r.get("prediction"))
        _num_empty_predictions = len(records) - _num_valid_inferences
        _coverage_inference = _num_valid_inferences / len(records) if records else 0.0
        if num_valid_evaluation is None:
            num_valid_evaluation = sum(1 for sm in sample_metrics if sm)
        _coverage_evaluation = (num_valid_evaluation / _num_samples) if _num_samples else 0.0
        _display_metrics = (
            (task_config.evaluation.display_metrics if task_config else None)
            or default_metric_keys or []
        )
        _inference_engine = overrides.pop(
            "inference_engine", getattr(args, "inference_engine", None)
        )
        _evaluation_engine = overrides.pop(
            "evaluation_engine", getattr(args, "evaluation_engine", None)
        )
        _evaluation_method = overrides.pop(
            "evaluation_method",
            (task_config.evaluation.method if task_config else None),
        )
        return cls(
            inference_engine=_inference_engine,
            evaluation_engine=_evaluation_engine,
            task_name=task_name,
            evaluation_method=_evaluation_method,
            num_samples=_num_samples,
            num_empty_predictions=_num_empty_predictions,
            coverage_inference=_coverage_inference,
            coverage_evaluation=_coverage_evaluation,
            runtime_inference=None,
            runtime_evaluation=None,
            metric_keys=_display_metrics,
            metrics=metrics,
            group_metrics=group_metrics if group_metrics else None,
            sample_metrics=sample_metrics,
            **overrides,
        )

    def update_statistics(
        self,
        inference_run_outputs: List[Dict[str, Any]],
        tokenizer: Optional[Callable] = None,
    ):
        statistics = defaultdict(list)
        for _record in inference_run_outputs:
            _record = copy.deepcopy(_record)
            
            _prediction = list()
            if isinstance(_record["prediction"], str):
                _prediction = [_record["prediction"], ]
            elif isinstance(_record["prediction"], (list, tuple)):
                for _v in _record["prediction"]:
                    if (
                        isinstance(_v, dict) 
                        and _v.get("value", None)
                    ):
                        _v = _v["value"]
                    _prediction.append(_v)
            elif isinstance(_record["prediction"], dict):
                for _k, _v in _record["prediction"].items():
                    if (
                        isinstance(_v, dict) 
                        and _v.get("value", None)
                    ):
                        _v = _v["value"]
                    _prediction.append(_v)
            if not _prediction:
                continue

            _avg_num_chars = list()
            _avg_num_tokens = list()
            _avg_num_words = list()
            for _prediction_ in _prediction:
                if not isinstance(_prediction_, str):
                    continue
                _prediction_ = _prediction_.strip()
                _avg_num_chars.append(len(_prediction_))
                if tokenizer:
                    _avg_num_tokens.append(len(tokenizer.tokenize(_prediction_)))
                _avg_num_words.append(len(_prediction_.split()))
            
            if len(_avg_num_chars) > 0:
                statistics["avg_num_chars"].append(np.mean(_avg_num_chars))
            if len(_avg_num_tokens) > 0:
                statistics["avg_num_tokens"].append(np.mean(_avg_num_tokens))
            if len(_avg_num_words) > 0:
                statistics["avg_num_words"].append(np.mean(_avg_num_words))
            
        statistics = {
            _stat_key: np.nanmean(_stat_value)
            for _stat_key, _stat_value in statistics.items()
        }
        self.statistics = EvaluationStatistics(**statistics)
    
    def verbose(self, _print=print) -> None:
        """Log a detailed summary of a single evaluation run including coverage, latency, and per-group metrics.

        ``_print`` lets the caller swap in ``tqdm.write`` (or any printer
        with the same signature) so output co-exists cleanly with an active
        tqdm progress bar without lines getting overwritten/truncated.
        """
        _print(f'# (run: {self.run_index+1:02}/ {self.num_runs:02}) EvaluationRunOutput [{self.task_name}] ({self.inference_engine}/{self.evaluation_engine})')
        _print(f'- {"evaluation_method":<25}: {self.evaluation_method}')
        _print(f'- {"num_samples":<25}: {self.num_samples}')
        _print(f'- {"num_empty_predictions":<25}: {self.num_empty_predictions}')
        if isinstance(self.coverage_inference, (int, float)):
            _print(f'- {"coverage_inference":<25}: {self.coverage_inference:.4f}')
        if isinstance(self.coverage_evaluation, (int, float)):
            _print(f'- {"coverage_evaluation":<25}: {self.coverage_evaluation:.4f}')
        # always _print these timing fields even when None — null surfaces missing measurements
        # runtime_* are wall-clock seconds; rendered in minutes for readability when measured.
        _latency_repr = f'{self.latency:.4f} sec' if isinstance(self.latency, (int, float)) else f'{self.latency}'
        _throughput_repr = f'{self.throughput:.4f} req/sec' if isinstance(self.throughput, (int, float)) else f'{self.throughput}'
        _runtime_inference_repr = f'{self.runtime_inference / 60:.4f} min' if isinstance(self.runtime_inference, (int, float)) else f'{self.runtime_inference}'
        _runtime_postprocess_repr = f'{self.runtime_postprocess / 60:.4f} min' if isinstance(self.runtime_postprocess, (int, float)) else f'{self.runtime_postprocess}'
        _runtime_evaluation_repr = f'{self.runtime_evaluation / 60:.4f} min' if isinstance(self.runtime_evaluation, (int, float)) else f'{self.runtime_evaluation}'
        _print(f'- {"latency":<25}: {_latency_repr}')
        _print(f'- {"throughput":<25}: {_throughput_repr}')
        _print(f'- {"runtime_inference":<25}: {_runtime_inference_repr}')
        _print(f'- {"runtime_postprocess":<25}: {_runtime_postprocess_repr}')
        _print(f'- {"runtime_evaluation":<25}: {_runtime_evaluation_repr}')

        if self.statistics and len(self.statistics) > 0:
            _print(f'- {"statistics":<25}:')
            for _stat_key, _stat_value in self.statistics.items():
                if _stat_value is None:
                    continue
                if isinstance(_stat_value, float):
                    _print(f'\t- {f"{_stat_key}":<15}: {_stat_value:.4f}')
                else:
                    _print(f'\t- {f"{_stat_key}":<15}: {_stat_value}')

        if self.metrics and len(self.metrics) > 0:
            _print(f'- {"metrics (overall)":<25}:')
            for _metric_name, _metric_value in self.metrics.items():
                if isinstance(_metric_value, float):
                    _print(f'\t- {f"{_metric_name}":<15}: {_metric_value:.4f}')
                else:
                    _print(f'\t- {f"{_metric_name}":<15}: {_metric_value}')

        if (
            self.group_metrics is not None
            and len(self.group_metrics) > 0
        ):
            for _group_name, _group_metrics in self.group_metrics.items():
                if len(_group_metrics) < 1:
                    continue
                _print(f'\t- {f"metrics ({_group_name})":<35}:')
                for _metric_name, _metric_value in _group_metrics.items():
                    if isinstance(_metric_value, float):
                        _print(f'\t\t- {f"{_metric_name}":<15}: {_metric_value:.4f}')
                    else:
                        _print(f'\t\t- {f"{_metric_name}":<15}: {_metric_value}')

@dataclass
class EvaluationOutput(SchemaInterface):
    """Aggregated evaluation output containing run results and metadata."""

    inference_engine: Optional[InferenceEngine] = None
    evaluation_engine: Optional[EvaluationEngine] = None
    task_name: Optional[str] = None
    evaluation_method: Optional[EvaluationMethod] = None
    num_runs: Optional[int] = 1
    num_samples: Optional[int] = None
    coverage_inference: Optional[float] = None
    coverage_evaluation: Optional[float] = None
    latency: Optional[float] = None
    throughput: Optional[float] = None
    runtime_inference: Optional[float] = None
    runtime_evaluation: Optional[float] = None
    runtime_postprocess: Optional[float] = None
    statistics: Optional[Dict[str, Any]] = None
    metric_keys: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    group_metrics: Optional[Dict[str, Any]] = None
    run_outputs: Optional[List[Dict[str, Any]]] = field(default_factory=list)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.__dict__})'

    def __post_init__(self):
        if isinstance(self.metrics, dict):
            for _k, _v in self.metrics.items():
                if isinstance(_v, np.floating):
                    self.metrics[_k] = float(_v)
                elif isinstance(_v, np.integer):
                    self.metrics[_k] = int(_v)
    
    @classmethod
    def validate(
        cls,
        obj: Union[Dict[str, Any], "EvaluationOutput"],
    ) -> bool:
        # Validate that an EvaluationOutput has required fields (engines, task_name) and non-empty metrics.
        # Args: obj - EvaluationOutput instance or dict to validate
        # Returns: True if valid, False otherwise (logs errors)
        output = None
        if isinstance(obj, cls):
            output = obj
        elif isinstance(obj, dict):
            output = cls.from_dict(**obj)
        if not isinstance(output, cls):
            logger.error('Invalid EvaluationOutput')
            return False

        if (
            not output.inference_engine
            or not output.evaluation_engine
            or not output.task_name
        ):
            logger.error('Invalid EvaluationOutput')
            return False
        if (
            not isinstance(output.metrics, dict)
            or len(output.metrics) < 1
        ):
            logger.error('Invalid EvaluationOutput - invalid metrics')
            return False
        if (
            not isinstance(output.group_metrics, dict)
            or len(output.group_metrics) < 1
        ):
            logger.error('Invalid EvaluationOutput - invalid group_metrics')
            return False
        return True
    
    @classmethod
    def aggregate_run_outputs(
        cls,
        run_outputs: List[Union[EvaluationRunOutput, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        # Aggregate metrics across multiple evaluation runs, computing mean, std, any-correct, and all-correct.
        # Args: run_outputs - list of per-run result dicts containing metrics and sample_metrics
        # Returns: dict with averaged coverage, latency, throughput, and aggregated metrics with -std/-any/-all suffixes
        # collect run_outputs
        coverage_inference, coverage_evaluation = list(), list()
        latency, throughput = list(), list()
        runtime_inference, runtime_evaluation, runtime_postprocess = list(), list(), list()
        statistics = dict()
        metric_keys = list()
        _metrics = defaultdict(list)
        group_metrics = dict()
        for _run_index, _run_output in enumerate(run_outputs):
            coverage_inference.append(_run_output["coverage_inference"])
            coverage_evaluation.append(_run_output["coverage_evaluation"])
            latency.append(_run_output["latency"])
            throughput.append(_run_output["throughput"])
            runtime_inference.append(_run_output["runtime_inference"])
            runtime_evaluation.append(_run_output["runtime_evaluation"])
            runtime_postprocess.append(_run_output["runtime_postprocess"])
            for _field_name, _field_value in _run_output["statistics"].items():
                if _field_name not in statistics:
                    statistics[_field_name] = list()
                statistics[_field_name].append(_field_value)
            metric_keys += _run_output["metric_keys"]
            for _metric_name, _metric_value in _run_output["metrics"].items():
                _metrics[_metric_name].append(_metric_value)
            if _run_output["group_metrics"]:
                for _group_name, _group_metrics in _run_output["group_metrics"].items():
                    if _group_name not in group_metrics:
                        group_metrics[_group_name] = defaultdict(list)
                    for _metric_name, _metric_value in _group_metrics.items():
                        group_metrics[_group_name][_metric_name].append(_metric_value)
        
        # aggregate run_outputs
        for _field_name, _field_value in statistics.items():
            statistics[_field_name][f'{_field_name}'] = np.mean(_field_value)
            statistics[_field_name][f'{_field_name}-std'] = np.std(_field_value)
        
        metrics = dict()
        for _metric_name, _metric_value in dict(_metrics).items():
            metrics[f'{_metric_name}'] = np.mean(_metric_value)
            _any, _all = list(), list()
            for _sample_idx in range(0, run_outputs[0]["num_samples"]):
                _any_, _all_ = False, True
                if "sample_metrics" not in run_outputs[0]:
                    _any, _all = None, None
                    break
                if _sample_idx >= len(run_outputs[0]["sample_metrics"]):
                    break
                for _run_output in run_outputs:
                    if "sample_metrics" not in _run_output:
                        _any_, _all_ = None, None
                        break 
                    if _metric_name not in _run_output["sample_metrics"][_sample_idx]:
                        _any_, _all_ = None, None
                        break
                    if _run_output["sample_metrics"][_sample_idx][_metric_name]:
                        _any_ = True
                    else:
                        _all_ = False
                if (
                    _any_ is None
                    or _all_ is None
                ): # skip if one of metric is not computed
                    _any, _all = None, None
                    break
                _any.append(_any_)
                _all.append(_all_)
            if isinstance(_any, (list, tuple)):
                metrics[f'{_metric_name}-any'] = np.mean(_any)
            if isinstance(_all, (list, tuple)):
                metrics[f'{_metric_name}-all'] = np.mean(_all)
            metrics[f'{_metric_name}-std'] = np.std(_metric_value)

        if group_metrics:
            for _group_name, _group_metrics in group_metrics.items():
                group_metrics[_group_name] = dict()
                for _metric_name, _metric_value in dict(_group_metrics).items():
                    group_metrics[_group_name][f'{_metric_name}'] = np.mean(_metric_value)
                    group_metrics[_group_name][f'{_metric_name}-std'] = np.std(_metric_value)
            
        return {
            "coverage_inference": np.mean(coverage_inference),
            "coverage_evaluation": np.mean(coverage_evaluation),
            "latency": np.mean(latency),
            "throughput": np.mean(throughput),
            "runtime_inference": np.mean(runtime_inference),
            "runtime_evaluation": np.mean(runtime_evaluation),
            "runtime_postprocess": np.mean(runtime_postprocess),
            "statistics": statistics,
            "metric_keys": list(set(metric_keys)),
            "metrics": metrics,
            "group_metrics": group_metrics,
        }
    
    def add_run_output(
        self,
        run_output: Dict[str, Any],
    ) -> None:
        # Append a run output and re-aggregate overall metrics (uses first run directly, aggregates on subsequent runs).
        # Args: run_output - single run result dict with metrics, coverage, etc.
        # Returns: None (mutates self in-place)
        self.run_outputs.append(run_output)

        if len(self.run_outputs) == 1:
            self.num_samples = run_output["num_samples"]
            self.coverage_inference = run_output["coverage_inference"]
            self.coverage_evaluation = run_output["coverage_evaluation"]
            self.latency = run_output["latency"]
            self.throughput = run_output["throughput"]
            self.runtime_inference = run_output["runtime_inference"]
            self.runtime_evaluation = run_output["runtime_evaluation"]
            self.runtime_postprocess = run_output["runtime_postprocess"]
            self.statistics = run_output["statistics"]
            self.metric_keys = run_output["metric_keys"]
            self.metrics = run_output["metrics"]
            self.group_metrics = run_output["group_metrics"]
        else:
            _aggregated_output = self.aggregate_run_outputs(
                run_outputs=self.run_outputs,
            )
            self.coverage_inference = _aggregated_output["coverage_inference"]
            self.coverage_evaluation = _aggregated_output["coverage_evaluation"]
            self.latency = _aggregated_output["latency"]
            self.throughput = _aggregated_output["throughput"]
            self.runtime_inference = _aggregated_output["runtime_inference"]
            self.runtime_evaluation = _aggregated_output["runtime_evaluation"]
            self.runtime_postprocess = _aggregated_output["runtime_postprocess"]
            self.statistics = _aggregated_output["statistics"]
            self.metric_keys = _aggregated_output["metric_keys"]
            self.metrics = _aggregated_output["metrics"]
            self.group_metrics = _aggregated_output["group_metrics"]
    
    def verbose(
        self,
        verbose_group_output: bool = True,
        _print=print,
    ) -> None:
        # Print a summary of the aggregated evaluation including per-group metrics if enabled.
        # Args: verbose_group_output - whether to print per-group (sub-category) metric breakdowns
        #       _print - callable swap-in (e.g. ``tqdm.write``) so this output stays
        #         on its own lines instead of fighting an active tqdm progress bar
        # Returns: None (prints to stdout via the supplied callable)
        _print(f'# EvaluationOutput [{self.task_name}] ({self.inference_engine}/{self.evaluation_engine})')
        _print(f'- {"evaluation_method":<25}: {self.evaluation_method}')
        _print(f'- {"num_runs":<25}: {self.num_runs}')
        if isinstance(self.num_samples, (int, float)):
            _print(f'- {"num_samples":<25}: {self.num_samples:.4f}')
        if isinstance(self.coverage_inference, (int, float)):
            _print(f'- {"coverage_inference":<25}: {self.coverage_inference:.4f}')
        if isinstance(self.coverage_evaluation, (int, float)):
            _print(f'- {"coverage_evaluation":<25}: {self.coverage_evaluation:.4f}')
        # always _print these timing fields even when None — null surfaces missing measurements
        # runtime_* are wall-clock seconds; rendered in minutes for readability when measured.
        _latency_repr = f'{self.latency:.4f} sec' if isinstance(self.latency, (int, float)) else f'{self.latency}'
        _throughput_repr = f'{self.throughput:.4f} req/sec' if isinstance(self.throughput, (int, float)) else f'{self.throughput}'
        _runtime_inference_repr = f'{self.runtime_inference / 60:.4f} min' if isinstance(self.runtime_inference, (int, float)) else f'{self.runtime_inference}'
        _runtime_postprocess_repr = f'{self.runtime_postprocess / 60:.4f} min' if isinstance(self.runtime_postprocess, (int, float)) else f'{self.runtime_postprocess}'
        _runtime_evaluation_repr = f'{self.runtime_evaluation / 60:.4f} min' if isinstance(self.runtime_evaluation, (int, float)) else f'{self.runtime_evaluation}'
        _print(f'- {"latency":<25}: {_latency_repr}')
        _print(f'- {"throughput":<25}: {_throughput_repr}')
        _print(f'- {"runtime_inference":<25}: {_runtime_inference_repr}')
        _print(f'- {"runtime_postprocess":<25}: {_runtime_postprocess_repr}')
        _print(f'- {"runtime_evaluation":<25}: {_runtime_evaluation_repr}')

        if self.statistics and len(self.statistics) > 0:
            _print(f'- {"statistics":<25}:')
            for _stat_key, _stat_value in self.statistics.items():
                if _stat_value is None:
                    continue
                if isinstance(_stat_value, float):
                    _print(f'\t- {f"{_stat_key}":<15}: {_stat_value:.4f}')
                else:
                    _print(f'\t- {f"{_stat_key}":<15}: {_stat_value}')

        if self.metrics and len(self.metrics) > 0:
            _print(f'- {"metrics (overall)":<25}:')
            for _metric_name, _metric_value in self.metrics.items():
                if isinstance(_metric_value, float):
                    _print(f'\t- {f"{_metric_name}":<15}: {_metric_value:.4f}')
                else:
                    _print(f'\t- {f"{_metric_name}":<15}: {_metric_value}')

        if (
            verbose_group_output
            and self.group_metrics is not None
            and len(self.group_metrics) > 0
        ):
            for _group_name, _group_metrics in self.group_metrics.items():
                if len(_group_metrics) < 1:
                    continue
                _print(f'\t- {f"metrics ({_group_name})":<35}:')
                for _metric_name, _metric_value in _group_metrics.items():
                    if isinstance(_metric_value, float):
                        _print(f'\t\t- {f"{_metric_name}":<15}: {_metric_value:.4f}')
                    else:
                        _print(f'\t\t- {f"{_metric_name}":<15}: {_metric_value}')

        _print("")
        
    @classmethod
    def from_dict(
        cls,
        obj: Dict[str, Any] = None,
        **kwargs,
    ):
        if obj is not None:
            kwargs = obj
        valid_field_names = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in valid_field_names}
        return cls(**filtered)