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
from omegaconf import ListConfig, DictConfig
import PIL
from PIL import Image
import torch
from typing import Union, Any, Callable, Tuple, List, Dict, Optional, Literal

from omni_evaluator.enums.dataset import CombineMethod, DatasetSource
from omni_evaluator.enums.engine import EvaluationEngine, EvaluationMethod, InferenceEngine
from omni_evaluator.enums.evaluation import NullPredictionPolicy
from omni_evaluator.enums.media import Modality
from omni_evaluator.enums.task import SubtaskType, TaskType
from omni_evaluator.schemas import SchemaInterface

logger = logging.getLogger(__name__)

# Prompts live in ``omni_evaluator.evaluation.metrics.prompts.verifier``.


@dataclass(kw_only=True)
class TaskMeta(SchemaInterface):
    benchmark_name: str
    split: Optional[str] = None
    lang: Optional[List[str]] = None
    input_modality: Optional[List[Literal[
        Modality.audio,
        Modality.image,
        Modality.text,
        Modality.video,
    ]]] = None
    output_modality: Optional[List[Literal[
        Modality.audio,
        Modality.image,
        Modality.text,
        Modality.video,
    ]]] = None
    task_type: Optional[List[TaskType]] = None
    subtask_type: Optional[List[SubtaskType]] = None
    num_runs: Optional[int] = None
    num_sample_repetition: Optional[int] = None
    num_fewshot: Optional[int] = None

    def __post_init__(self):
        if self.task_type is not None:
            self.task_type = [
                v if isinstance(v, TaskType) else TaskType(v)
                for v in (self.task_type if isinstance(self.task_type, list) else [self.task_type])
            ]
        if self.subtask_type is not None:
            self.subtask_type = [
                v if isinstance(v, SubtaskType) else SubtaskType(v)
                for v in (self.subtask_type if isinstance(self.subtask_type, list) else [self.subtask_type])
            ]
        # ``num_runs`` is Optional in the field declaration (yamls usually omit
        # it) but every downstream consumer treats it as a positive int
        # (per-run loops, runtime scaffolding, EvaluationOutput.num_runs).
        # Normalize at the single entry point so yaml fresh-build and JSON
        # round-trip both land on a valid value.
        if not isinstance(self.num_runs, int) or self.num_runs < 1:
            self.num_runs = 1

@dataclass(kw_only=True)
class TaskDatasetCombine(SchemaInterface):
    """Strategy for combining multiple HF subsets. method=concatenate is row-wise
    union; method=join performs a column-wise join across subsets matched by the
    `key` column. (`key` is used instead of pandas-style `on` because YAML 1.1
    parses unquoted `on:` as boolean True.)"""
    method: CombineMethod = CombineMethod.concatenate
    key: Optional[str] = None   # join key column name; only used when method=join

    def __post_init__(self):
        if isinstance(self.method, str):
            self.method = CombineMethod(self.method)
        if self.method == CombineMethod.join and not self.key:
            raise ValueError(
                "dataset.combine.method='join' requires 'key' (the join key column name)"
            )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TaskDatasetCombine":
        # Lenient hydration: accept the legacy `on` alias (pre-rename JSON outputs)
        # and silently drop unknown fields so reloads survive schema evolution.
        if data is None:
            return cls()
        data = dict(data)
        if "on" in data and "key" not in data:
            data["key"] = data.pop("on")
        _valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in _valid})


@dataclass(kw_only=True)
class TaskDataset(SchemaInterface):
    source: Literal[
        DatasetSource.local,
        DatasetSource.s3,
        DatasetSource.package,
        DatasetSource.huggingface_hub,
        DatasetSource.resources,
    ]
    options: Optional[List[str]] = None
    # s3, local
    data_filepath: Optional[str] = None
    local_dirpath: Optional[str] = None
    audio_dirpath: Optional[str] = None
    image_dirpath: Optional[str] = None
    video_dirpath: Optional[str] = None
    # Subset filter: pick samples whose sample["meta"][key] matches the allowed value(s).
    # AND across keys, OR within each value list. Single str is auto-wrapped to [str].
    # Meta value that is itself a list (e.g. img_type) is matched by intersection.
    # Missing meta key → reject (sample dropped).
    subset: Optional[Dict[str, List[str]]] = None
    # huggingface_hub
    path: Optional[str] = None
    split: Optional[Union[str, List[str]]] = None
    name: Optional[Union[str, List[str]]] = None
    revision: Optional[Union[str, List[str]]] = None
    trust_remote_code: Optional[bool] = None
    config: Optional[Union[str, List[str]]] = None
    combine: Optional[TaskDatasetCombine] = None
    # `hf_load_dataset` verification_mode. None mirrors the HF datasets library
    # default (basic split/checksum verification). Set to "no_checks" per-task
    # when upstream dataset_infos.json is stale (e.g. HAERAE-VISION 165→653).
    verification_mode: Optional[str] = None
    # Default ``False`` — datasets' Audio feature is cast to ``decode=False``
    # so iteration yields raw ``{"bytes": ..., "path": ...}`` dicts that flow
    # through our librosa-based multimodal helpers. This avoids the unstable
    # datasets ↔ torchcodec ABI surface (e.g. AudioDecoder kwarg drift).
    # Set ``audio_decode: true`` in the yaml only when the task genuinely
    # wants datasets' decoded ``AudioDecoder`` objects.
    audio_decode: Optional[bool] = False
    audio_column: Optional[Union[str, List[str]]] = "audio"

    def __post_init__(self):
        # Hydrate nested combine spec from raw dict (YAML / JSON round-trip).
        # from_dict handles legacy aliases (e.g. pre-rename `on` → `key`) and
        # drops unknown fields, so reloads survive schema evolution.
        # Default to concatenate when omitted so single-subset tasks need no config.
        if self.combine is None:
            self.combine = TaskDatasetCombine()
        elif isinstance(self.combine, dict):
            self.combine = TaskDatasetCombine.from_dict(self.combine)

        if self.subset is None:
            return
        if not isinstance(self.subset, dict):
            raise ValueError(
                f"dataset.subset must be a dict mapping meta-key to allowed value(s), "
                f"got {type(self.subset).__name__}"
            )
        _normalized: Dict[str, List[str]] = {}
        for _key, _value in self.subset.items():
            # None or empty value → drop the key (== "no filter on this field")
            if _value is None:
                continue
            if isinstance(_value, str):
                if _value:
                    _normalized[_key] = [_value]
                continue
            if isinstance(_value, (list, tuple)):
                if not all(isinstance(_x, str) for _x in _value):
                    raise ValueError(
                        f"dataset.subset['{_key}'] must be a str or list[str], "
                        f"got element types {[type(_x).__name__ for _x in _value]}"
                    )
                _filtered = [_x for _x in _value if _x]
                if _filtered:
                    _normalized[_key] = _filtered
                continue
            raise ValueError(
                f"dataset.subset['{_key}'] must be a str or list[str] (or null to ignore), "
                f"got {type(_value).__name__}"
            )
        # All keys dropped → no filter at all
        self.subset = _normalized if _normalized else None

@dataclass(kw_only=True)
class TaskPrompts(SchemaInterface):
    # Either a single string (applied to every record) or a dict keyed by the
    # value of `sample.meta[conditional_on]`. When a dict is used,
    # `conditional_on` must name the meta field that selects the variant.
    system_prompt: Optional[Union[str, Dict[str, Any]]] = None
    task_prompt: Optional[Union[str, Dict[str, Any]]] = None
    conditional_on: Optional[str] = None

    def __post_init__(self):
        if self.conditional_on is None:
            for _name, _value in (
                ("system_prompt", self.system_prompt),
                ("task_prompt", self.task_prompt),
            ):
                if isinstance(_value, dict):
                    raise ValueError(
                        f"prompts.{_name} is a dict but prompts.conditional_on is unset; "
                        f"set conditional_on to the meta field name that selects the variant, "
                        f"or replace the dict with a single string."
                    )
        else:
            if not isinstance(self.conditional_on, str) or not self.conditional_on:
                raise ValueError(
                    f"prompts.conditional_on must be a non-empty string, got "
                    f"{type(self.conditional_on).__name__}={self.conditional_on!r}"
                )
            for _name, _value in (
                ("system_prompt", self.system_prompt),
                ("task_prompt", self.task_prompt),
            ):
                if _value is None:
                    continue
                if not isinstance(_value, dict):
                    raise ValueError(
                        f"prompts.conditional_on='{self.conditional_on}' but prompts.{_name} "
                        f"is {type(_value).__name__}; provide a dict mapping meta values to "
                        f"prompts (or set to null if this prompt is not needed)."
                    )


@dataclass(kw_only=True)
class TaskInferenceGenerationOptions(SchemaInterface):
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None
    # image generation
    height: Optional[int] = None
    width: Optional[int] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    negative_prompt: Optional[List[str]] = None

@dataclass(kw_only=True)
class TaskInference(SchemaInterface):
    generation_options: Optional[TaskInferenceGenerationOptions] = None
    config: Optional[Dict[str, Any]] = None
    # Auxiliary modality gates. Resolution: CLI args > this task_config value.
    # Semantics for all three: None or 0 → disabled, positive N → first N,
    # negative → use all available.
    num_ocr_tokens: Optional[int] = None
    num_entity_tokens: Optional[int] = None
    num_subtitle_cues: Optional[int] = None
    # Per-task overrides forwarded to vLLM via OpenAI ``extra_body``. Same
    # shape as vLLM's protocol fields. CLI args (max_video_frames, fps,
    # min_pixels, max_pixels) merge in on top of these task-config values;
    # CLI wins when non-None.
    media_io_kwargs: Optional[Dict[str, Dict[str, Any]]] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    
    def __post_init__(
        self,
    ):
        if (
            self.generation_options
            and isinstance(self.generation_options, dict)
        ):
            self.generation_options = TaskInferenceGenerationOptions(**self.generation_options)
       
@dataclass(kw_only=True)
class TaskEvaluationJudge(SchemaInterface):
    lang: str
    judge_model: Union[str, List[str]]
    judge_prompt: Optional[str] = None # judge task_prompt to format
    system_prompt: Optional[str] = None # judge system_prompt
    instruction: Optional[str] = None # additional instruction to format judge_prompt template
    baseline: Optional[str] = None # required when judge_pairwise
    rubrics: Optional[Dict[str, Any]] = None
    response_format: Optional[type] = None
    # Marker that flips ``_postprocess_judge_rating_record`` into reason-extraction
    # mode (parses ``Reason:``/``Explanation:`` lines from the judge response into
    # ``sample_metrics[i]["{target_metric}/reasons"]``). Truthy enables, falsy skips.
    reason_format: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    process_message_kwargs: Optional[Dict[str, Any]] = None
    do_async: Optional[bool] = False
    max_rating: Optional[Union[int, float]] = None  # upper bound of the judge_rating scale; used as divisor to normalize ratings to [0, 1] (e.g. 10 for a 0-10 scale)


@dataclass(kw_only=True)
class TaskVerifier(SchemaInterface):
    """Per-task verifier override — config.yaml ``verifier:`` block, sibling of
    ``inference``/``postprocess``/``evaluation``.

    Fields mirror the ``verifier_*`` surface of ``omni_evaluator.args.VerifierArgs``
    (prefix stripped). Every field is ``Optional`` and defaults to ``None`` =
    "not specified for this task": evaluate.py then falls back to the
    corresponding VerifierArgs (CLI/default) value, PER FIELD. A field set here
    takes precedence over the CLI arg for that task.

    Intentionally NOT exposed (decided upstream): ``prompt`` (always
    prompts/verifier.py, toggled only by ``reasoning``), ``lang`` (fixed "en"),
    ``reason_format`` (internal postprocess gate in ``Verifier``).
    """
    engine: Optional[str] = None
    api_name: Optional[str] = None
    model_name_or_path: Optional[str] = None
    model_group: Optional[str] = None
    device_map: Optional[str] = None
    gguf_filename: Optional[str] = None
    alias: Optional[str] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    reasoning: Optional[bool] = None
    num_concurrency: Optional[int] = None
    num_cpu_threads: Optional[int] = None
    max_seq_len: Optional[int] = None


# ────────────────────────────────────────────────────────────────────
# New per-entry "Logic" classes — kwargs holders used under postprocess
# and target_metrics.{text,judge}_evaluator.
#
# Each Logic class declares known kwargs as explicit fields plus a
# catch-all `extra` dict for custom-task / future kwargs. `to_kwargs()`
# produces a flat dict ready for ** unpacking at call time.
#
# Kept as three distinct classes (rather than one shared one) so each
# can diverge independently in the future, even though some fields
# currently overlap.
# ────────────────────────────────────────────────────────────────────

@dataclass(kw_only=True)
class TaskPostprocessLogic(SchemaInterface):
    """Per-entry kwargs for one postprocess logic (think, boxed, multichoice,
    freeform, asr, spatial_grounding, or a task-local custom function)."""
    api_name: Optional[str] = None                          # multichoice, freeform, asr
    version_name: Optional[str] = None                      # multichoice, asr
    output_scale: Optional[Union[float, List[float]]] = None  # spatial_grounding: scalar=max ([0,v]) or [lo,hi]
    verbose: Optional[bool] = False
    extra: Optional[Dict[str, Any]] = None  # custom processor's task-specific kwargs (e.g. mathvista data_info)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TaskPostprocessLogic":
        if data is None:
            return cls()
        data = dict(data)
        api_name = data.pop("api_name", None)
        version_name = data.pop("version_name", None)
        output_scale = data.pop("output_scale", None)
        verbose = data.pop("verbose", False)
        # Round-trip path: serialized form (asdict) emits an explicit `extra` key
        # holding the per-record kwargs dict. yaml form stores those kwargs flat
        # under the processor entry. Support both: pop `extra` if present and use
        # it as the canonical value; any leftover keys are treated as yaml-form
        # kwargs and merged in (yaml form keys take precedence on collision).
        explicit_extra = data.pop("extra", None)
        if explicit_extra is not None and not isinstance(explicit_extra, dict):
            explicit_extra = None
        if explicit_extra is None:
            extra = data or None
        else:
            extra = {**explicit_extra, **data} if data else explicit_extra
        return cls(
            api_name=api_name,
            version_name=version_name,
            output_scale=output_scale,
            verbose=verbose,
            extra=extra,
        )

    def to_kwargs(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"verbose": self.verbose}
        if self.api_name is not None:
            out["api_name"] = self.api_name
        if self.version_name is not None:
            out["version_name"] = self.version_name
        if self.output_scale is not None:
            out["output_scale"] = self.output_scale
        if self.extra:
            out.update(self.extra)
        return out


@dataclass(kw_only=True)
class TaskJudgeEvaluatorLogic(SchemaInterface):
    """Per-judge kwargs for one JudgeEvaluator metric. Currently mirrors
    TaskEvaluationJudge field-for-field; kept as a distinct class so
    judge-evaluator-specific parameters can diverge in the future."""
    lang: str = "en"
    judge_model: Union[str, List[str]] = "gpt-5-mini"
    judge_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    instruction: Optional[str] = None
    baseline: Optional[str] = None
    rubrics: Optional[Dict[str, Any]] = None
    response_format: Optional[type] = None
    # Mirrors TaskEvaluationJudge field-for-field; `reason_format` had once been
    # silently dropped. Keeping it lets a config.yaml `judge_evaluator` block that
    # carries `reason_format` hydrate without `TypeError: unexpected keyword
    # argument 'reason_format'`. Semantics mirror TaskEvaluationJudge.reason_format
    # — truthy flips judge response parsing into reason-extraction mode.
    reason_format: Optional[Union[str, Dict[str, Any]]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None
    process_message_kwargs: Optional[Dict[str, Any]] = None
    do_async: Optional[bool] = False
    max_rating: Optional[Union[int, float]] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TaskJudgeEvaluatorLogic":
        if data is None:
            return cls()
        return cls.from_kwargs(**data)

    def to_kwargs(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ────────────────────────────────────────────────────────────────────
# Wrapper classes — top-level entries under TaskConfig.postprocess and
# TaskEvaluation.target_metrics. Each holds a dict of name → Logic plus
# task-level reserved/default kwargs.
# ────────────────────────────────────────────────────────────────────

_TASK_POSTPROCESS_RESERVED = {"conditional_on", "allow_api"}

# Per-processor default kwargs auto-applied to each TaskPostprocess.chain entry
# when YAML omits them. Lets config.yaml drop boilerplate like
# `freeform: {api_name: gpt-4o-mini}`. The defaults are recorded on the logic
# instance unconditionally; the master switch `--postprocess_allow_api`
# (or `TaskPostprocess.allow_api`) is enforced in `get_postprocess_functions`,
# which forces `api_name=None` at chain construction time when the switch is off.
_DEFAULT_POSTPROCESS_LOGICS: Dict[str, Dict[str, Any]] = {
    "freeform":    {"api_name": "gpt-4o-mini"},
    "multichoice": {"api_name": "gpt-4o-mini"},
    "binary":      {"api_name": "gpt-4o-mini"},
}


@dataclass(kw_only=True)
class TaskPostprocess(SchemaInterface):
    """Top-level postprocess chain (sibling to inference, evaluation).

    YAML form (simple)::

        postprocess:
          think: {}
          boxed: {}
          multichoice:
            api_name: gpt-4o-mini-2024-07-18
          custom_fn:
            data_info: null

    YAML form (conditional)::

        postprocess:
          conditional_on: subcategory     # reserved meta key
          multiple_choice:
            think: {}
            multichoice: {api_name: ...}
          freeform:
            think: {}
            freeform: {}

    `chain` is `Dict[processor_name, TaskPostprocessLogic]` in the simple
    case, or `Dict[variant_key, Dict[processor_name, TaskPostprocessLogic]]`
    when `conditional_on` is set.
    """
    conditional_on: Optional[str] = None
    allow_api: Optional[bool] = False
    chain: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def _merge_defaults(cls, proc_name: str, proc_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge per-processor default kwargs onto YAML-provided kwargs.
        YAML override wins; absent keys fall back to ``_DEFAULT_POSTPROCESS_LOGICS``.
        The master switch (``--postprocess_allow_api`` / ``postprocess.allow_api``)
        is enforced later in the chain builder, not here — the schema only
        records the default api_name; the builder decides whether to honor it."""
        _defaults = _DEFAULT_POSTPROCESS_LOGICS.get(proc_name, {})
        if not _defaults:
            return dict(proc_kwargs) if isinstance(proc_kwargs, dict) else (proc_kwargs or {})
        _user = dict(proc_kwargs) if isinstance(proc_kwargs, dict) else {}
        return {**_defaults, **_user}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TaskPostprocess":
        if data is None:
            return cls()
        data = dict(data)
        # Legacy form has explicit `pipeline` list (or dict-of-list for conditional)
        # along with task-wide api_name/kwargs/version; translate to new chain.
        if "pipeline" in data:
            return cls._from_legacy(data)
        conditional_on = data.pop("conditional_on", None)
        allow_api = data.pop("allow_api", False)
        # Round-trip path: serialized form (asdict) emits an explicit `chain` key
        # holding the nested processor dict, so unwrap it back to flat form.
        if "chain" in data and isinstance(data["chain"], dict) and len(data) == 1:
            data = data.pop("chain")
        # Remaining keys are processor entries (or variant entries when conditional)
        chain: Dict[str, Any] = {}
        if conditional_on:
            for variant_key, variant_chain in data.items():
                if not isinstance(variant_chain, dict):
                    continue
                chain[variant_key] = {
                    proc_name: proc_kwargs
                    if isinstance(proc_kwargs, TaskPostprocessLogic)
                    else TaskPostprocessLogic.from_dict(cls._merge_defaults(proc_name, proc_kwargs))
                    for proc_name, proc_kwargs in variant_chain.items()
                }
        else:
            for proc_name, proc_kwargs in data.items():
                chain[proc_name] = (
                    proc_kwargs
                    if isinstance(proc_kwargs, TaskPostprocessLogic)
                    else TaskPostprocessLogic.from_dict(cls._merge_defaults(proc_name, proc_kwargs))
                )
        return cls(conditional_on=conditional_on, allow_api=allow_api, chain=chain)

    @classmethod
    def _from_legacy(cls, data: Dict[str, Any]) -> "TaskPostprocess":
        """Translate legacy {pipeline, api_name, version, kwargs, allow_api,
        conditional_on} form to the new chain dict."""
        pipeline = data.pop("pipeline", None)
        api_name = data.pop("api_name", None)
        version = data.pop("version", None)
        per_record_kwargs = data.pop("kwargs", None) or {}
        conditional_on = data.pop("conditional_on", None)
        allow_api = data.pop("allow_api", False)

        def _build_logic(_proc_name: str, _kwargs: Dict[str, Any]) -> TaskPostprocessLogic:
            # Apply per-processor schema defaults (e.g. api_name=gpt-4o-mini for
            # freeform/multichoice/binary) so the LLM fallback has a model name
            # when `allow_api` is on. Mirrors the new-form path in `from_dict`.
            return TaskPostprocessLogic.from_dict(cls._merge_defaults(_proc_name, _kwargs))

        # Merge legacy fields into each processor entry's kwargs.
        def _entry_kwargs() -> Dict[str, Any]:
            kw: Dict[str, Any] = dict(per_record_kwargs)
            if api_name is not None:
                kw["api_name"] = api_name
            if version is not None:
                # legacy supported either str or list-of-str; pass through
                kw["version_name"] = version
            return kw

        chain: Dict[str, Any] = {}
        if isinstance(pipeline, list):
            for proc_name in pipeline:
                chain[proc_name] = _build_logic(proc_name, _entry_kwargs())
        elif isinstance(pipeline, dict):
            for variant_key, variant_list in pipeline.items():
                chain[variant_key] = {
                    proc_name: _build_logic(proc_name, _entry_kwargs())
                    for proc_name in (variant_list or [])
                }
        return cls(conditional_on=conditional_on, allow_api=allow_api, chain=chain)


_TASK_TEXT_EVALUATOR_RESERVED = {
    "do_normalize",
    "group_field",
    "null_prediction_policy",
    "fallback_value",
    "do_async",
}


@dataclass(kw_only=True)
class TaskTextEvaluator(SchemaInterface):
    """Value class for ``target_metrics.text_evaluator``.

    YAML form::

        text_evaluator:
          group_field: category        # reserved task-level default
          exact_match:                 # metric name → kwargs dict
            relative_tolerance: 0.05
          tree_edit_score:
            api_name: "gpt-5-mini"
            source_format: "latex"

    Per-metric kwargs are stored as ``Dict[str, Any]`` (no inner schema class)
    — text_evaluator dispatch already reads them via ``_kwargs.get(<key>, default)``
    so static typing isn't required. New metric kwargs are added by writing
    yaml only, with no schema changes.
    """
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Task-level pre-normalize for table-eval methods ("squad"/"fintabnet"/"wtq")
    # — rewrites predictions/labels in sample-prep before any metric dispatch.
    # Also serves as the fallback when wer/cer's per-metric ``do_normalize``
    # is unset.
    do_normalize: Optional[Union[bool, str]] = None
    group_field: Optional[str] = "category"
    # null-prediction handling (single source of truth for all text metrics).
    # See ``NullPredictionPolicy`` for behavior of each value.
    null_prediction_policy: Optional[NullPredictionPolicy] = None
    fallback_value: Optional[str] = None
    # Task-level default for async LLM calls within text-evaluator metrics
    # (currently only consumed by ``tree_edit_score`` stage-1 conversion).
    # Mirrors TaskJudgeEvaluator.do_async. Callers fold this with the
    # per-metric value and the runtime override via an inline ``or`` cascade.
    do_async: Optional[bool] = None

    def __post_init__(self):
        # Coerce YAML string ("miss"/"skip"/"fallback") → enum.
        if isinstance(self.null_prediction_policy, str):
            self.null_prediction_policy = NullPredictionPolicy(self.null_prediction_policy)
        # Coerce ``metrics`` entries to plain dicts (yaml gives dicts; round-trip
        # may give None for empty entries).
        if isinstance(self.metrics, dict):
            self.metrics = {
                _name: (dict(_kwargs) if isinstance(_kwargs, dict) else {})
                for _name, _kwargs in self.metrics.items()
            }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TaskTextEvaluator":
        if data is None:
            return cls()
        data = dict(data)
        # Allow explicit `metrics:` nesting, but flat form is canonical.
        if "metrics" in data and isinstance(data["metrics"], dict):
            metrics_dict = data.pop("metrics")
        else:
            metrics_dict = {
                k: data.pop(k) for k in list(data.keys()) if k not in _TASK_TEXT_EVALUATOR_RESERVED
            }
        task_defaults = {k: data[k] for k in _TASK_TEXT_EVALUATOR_RESERVED if k in data}
        # Each metric entry is a plain dict (or None → empty dict).
        metrics = {
            _name: (dict(_kwargs) if isinstance(_kwargs, dict) else {})
            for _name, _kwargs in metrics_dict.items()
        }
        return cls(metrics=metrics, **task_defaults)


_TASK_JUDGE_EVALUATOR_RESERVED = {"do_async"}

# Default judge entries auto-populated when not overridden by config.yaml.
_DEFAULT_JUDGE_LOGICS = {
    # NOTE: the verifier (``verifier_score``) is no longer configured here — it is
    # built entirely from ``VerifierArgs`` via ``Verifier`` in evaluate.py
    # (prompt = prompts/verifier.py selected by ``verifier_reasoning``; lang fixed
    # "en"; reason_format is an internal postprocess gate). These remaining entries
    # are the generic LLM-judge defaults used by ``JudgeEvaluator``.
    "judge_binary":   {"lang": "en", "judge_model": "gpt-5-mini", "max_tokens": 1024, "temperature": 0.0},
    "judge_rating":   {"lang": "en", "judge_model": "gpt-5-mini", "max_tokens": 1024, "temperature": 0.0},
    "judge_pairwise": {"lang": "en", "judge_model": "gpt-5-mini", "max_tokens": 1024, "temperature": 0.0},
}


@dataclass(kw_only=True)
class TaskJudgeEvaluator(SchemaInterface):
    """Value class for ``target_metrics.judge_evaluator``.

    YAML form::

        judge_evaluator:
          do_async: true               # reserved task-level default
          judge_binary:                # metric name → TaskJudgeEvaluatorLogic
            lang: ko
            judge_model: gpt-4o-mini-2024-07-18
            judge_prompt: |
              ...
    """
    metrics: Dict[str, TaskJudgeEvaluatorLogic] = field(default_factory=dict)
    # Task-level default for async LLM calls. Callers fold this with the
    # per-metric value and the runtime override via an inline ``or`` cascade
    # — matching the other yaml/runtime resolutions in ``engine.py`` rather
    # than introducing a dedicated helper method.
    do_async: Optional[bool] = None

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None) -> "TaskJudgeEvaluator":
        if data is None:
            return cls()
        data = dict(data)
        if "metrics" in data and isinstance(data["metrics"], dict):
            metrics_dict = data.pop("metrics")
        else:
            metrics_dict = {
                k: data.pop(k) for k in list(data.keys()) if k not in _TASK_JUDGE_EVALUATOR_RESERVED
            }
        task_defaults = {k: data[k] for k in _TASK_JUDGE_EVALUATOR_RESERVED if k in data}
        metrics = {}
        for name, kw in metrics_dict.items():
            if isinstance(kw, TaskJudgeEvaluatorLogic):
                metrics[name] = kw
            elif isinstance(kw, dict):
                metrics[name] = TaskJudgeEvaluatorLogic.from_dict(kw)
        return cls(metrics=metrics, **task_defaults)


@dataclass(kw_only=True)
class TaskEvaluationPostprocess(SchemaInterface):
    # pipeline may be a single list (applied to every record) or a dict keyed by
    # the value of `sample.meta[conditional_on]`. When dict, conditional_on must
    # be set (mirrors TaskPrompts).
    pipeline: Optional[Union[List[str], Dict[str, List[str]]]] = None
    version: Optional[List[str]] = None
    api_name: Optional[str] = None
    allow_api: Optional[bool] = False
    kwargs: Optional[Dict[str, Any]] = None
    conditional_on: Optional[str] = None

    def __post_init__(self):
        if self.conditional_on is None:
            if isinstance(self.pipeline, dict):
                raise ValueError(
                    "postprocess.pipeline is a dict but postprocess.conditional_on is unset; "
                    "set conditional_on to the meta field name that selects the variant, "
                    "or replace the dict with a single list."
                )
        else:
            if not isinstance(self.conditional_on, str) or not self.conditional_on:
                raise ValueError(
                    f"postprocess.conditional_on must be a non-empty string, got "
                    f"{type(self.conditional_on).__name__}={self.conditional_on!r}"
                )
            if self.pipeline is not None and not isinstance(self.pipeline, dict):
                raise ValueError(
                    f"postprocess.conditional_on='{self.conditional_on}' but postprocess.pipeline "
                    f"is {type(self.pipeline).__name__}; provide a dict mapping meta values to "
                    f"pipeline variants (or set pipeline to null if not needed)."
                )

@dataclass(kw_only=True)
class TaskEvaluation(SchemaInterface):
    method: Literal[
        EvaluationMethod.generation,
        EvaluationMethod.perplexity,
    ]
    # `target_metrics` accepts two forms:
    #   - Legacy list:  ["exact_match", "judge_binary"]
    #   - New dict:     {"text_evaluator": {...}, "judge_evaluator": {...}}
    # __post_init__ normalizes to a flat List[str] for legacy access while
    # populating `text_evaluator` / `judge_evaluator` for new code paths.
    target_metrics: Any = field(default_factory=list)
    display_metrics: Optional[List[str]] = None
    do_normalize: Optional[bool] = False
    # Numeric tolerance forwarded to metrics that accept (relative_tolerance,
    # absolute_tolerance) — currently only compute_exact_match. Default is STRICT:
    # relative_tolerance=None (no proportional margin) + a tiny absolute_tolerance
    # for float-representation equality (e.g. 0.5 == 0.50). This keeps large-magnitude
    # categorical numbers (years/IDs/counts) exact (|Δ|≥1 ≫ 1e-6).
    # Opt in to a relative margin ONLY for tasks whose OFFICIAL metric defines one
    # (e.g. ChartQA relaxed accuracy @5%, arXiv:2203.10244 §5.1): set it in that
    # task's config.yaml under target_metrics.text_evaluator.exact_match.relative_tolerance.
    relative_tolerance: Optional[float] = None
    absolute_tolerance: Optional[float] = 1e-6
    postprocess: Optional[TaskEvaluationPostprocess] = None
    judges: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None

    # New structured access (auto-derived; engine/custom.py may migrate to these)
    text_evaluator: Optional[TaskTextEvaluator] = None
    judge_evaluator: Optional[TaskJudgeEvaluator] = None

    def __post_init__(
        self,
    ):
        if (
            self.postprocess
            and isinstance(self.postprocess, dict)
        ):
            self.postprocess = TaskEvaluationPostprocess(**self.postprocess)

        # Step 1: normalize target_metrics — accept dict (new) or list (legacy)
        _new_text_input: Optional[Dict[str, Any]] = None
        _new_judge_input: Optional[Dict[str, Any]] = None
        if isinstance(self.target_metrics, dict):
            # New dict form: {"text_evaluator": {...}, "judge_evaluator": {...}}
            _new_text_input = self.target_metrics.get("text_evaluator")
            _new_judge_input = self.target_metrics.get("judge_evaluator")
            # Flatten to legacy list for backward-compat readers
            _flat: List[str] = []
            if isinstance(_new_text_input, dict):
                # exclude reserved task-level keys
                _flat.extend(
                    k for k in _new_text_input.keys()
                    if k not in _TASK_TEXT_EVALUATOR_RESERVED and k != "metrics"
                )
            if isinstance(_new_judge_input, dict):
                _flat.extend(
                    k for k in _new_judge_input.keys()
                    if k not in _TASK_JUDGE_EVALUATOR_RESERVED and k != "metrics"
                )
            self.target_metrics = _flat
        elif self.target_metrics is None:
            self.target_metrics = []

        # Step 2: hydrate legacy `judges` dict entries from raw kwargs into
        # TaskEvaluationJudge instances. Default-fill happens later (Step 5)
        # so values from the new dict form take precedence.
        if self.judges is None:
            self.judges = dict()
        if isinstance(self.judges, dict):
            for _metric_name, _judge_kwargs in self.judges.items():
                if isinstance(_judge_kwargs, TaskEvaluationJudge):
                    pass
                elif isinstance(_judge_kwargs, dict):
                    self.judges[_metric_name] = TaskEvaluationJudge(**_judge_kwargs)
        else:
            self.judges = dict()

        # Step 3 prelude: hydrate raw dicts from JSON round-trip (asdict emits
        # nested `metrics: {...}` which TaskEvaluation accepts as raw, otherwise
        # `is None` check below would leave it as a plain dict).
        if isinstance(self.text_evaluator, dict):
            self.text_evaluator = TaskTextEvaluator.from_dict(self.text_evaluator)
        if isinstance(self.judge_evaluator, dict):
            self.judge_evaluator = TaskJudgeEvaluator.from_dict(self.judge_evaluator)

        # Step 3: build new structured target_metrics (text_evaluator / judge_evaluator)
        # If new dict form was given, use it; otherwise derive from legacy fields.
        if self.text_evaluator is None:
            if _new_text_input is not None:
                self.text_evaluator = TaskTextEvaluator.from_dict(_new_text_input)
            else:
                _text_names = [n for n in self.target_metrics if not n.startswith("judge")]
                if _text_names:
                    _metrics_text: Dict[str, Dict[str, Any]] = {}
                    for _name in _text_names:
                        _kw: Dict[str, Any] = {}
                        if _name == "exact_match":
                            if self.relative_tolerance is not None:
                                _kw["relative_tolerance"] = self.relative_tolerance
                            if self.absolute_tolerance is not None:
                                _kw["absolute_tolerance"] = self.absolute_tolerance
                        _metrics_text[_name] = _kw
                    self.text_evaluator = TaskTextEvaluator(metrics=_metrics_text)

        if self.judge_evaluator is None:
            if _new_judge_input is not None:
                self.judge_evaluator = TaskJudgeEvaluator.from_dict(_new_judge_input)
            else:
                _judge_names = [n for n in self.target_metrics if n.startswith("judge")]
                if _judge_names:
                    _metrics_judge: Dict[str, TaskJudgeEvaluatorLogic] = {}
                    for _name in _judge_names:
                        _legacy = self.judges.get(_name)
                        if isinstance(_legacy, TaskEvaluationJudge):
                            _metrics_judge[_name] = TaskJudgeEvaluatorLogic.from_dict(asdict(_legacy))
                        elif isinstance(_legacy, dict):
                            _metrics_judge[_name] = TaskJudgeEvaluatorLogic.from_dict(_legacy)
                        else:
                            _metrics_judge[_name] = TaskJudgeEvaluatorLogic()
                    self.judge_evaluator = TaskJudgeEvaluator(metrics=_metrics_judge)

        # Step 3 epilogue: strip judge_evaluator.metrics entries that aren't in
        # target_metrics. Prior schema revisions auto-injected `judge_score` (etc.)
        # via `_add_default_judges`, and those got serialized into inference output
        # JSONs. Re-loading such JSONs must not re-activate those stale defaults.
        if (
            self.judge_evaluator is not None
            and isinstance(self.judge_evaluator.metrics, dict)
        ):
            _allowed_judges = {
                n for n in self.target_metrics
                if isinstance(n, str)
                and n.startswith(("judge_rating", "judge_score", "judge_binary", "judge_pairwise"))
            }
            if _allowed_judges:
                self.judge_evaluator.metrics = {
                    k: v for k, v in self.judge_evaluator.metrics.items() if k in _allowed_judges
                }

        # Note: legacy fields (relative_tolerance, absolute_tolerance, judges,
        # evaluation.postprocess) are now INPUT-only — consumed by Step 1-3
        # to translate old YAML into the new structures. Engine, postprocess
        # pipeline, and custom.py all read from text_evaluator / judge_evaluator /
        # task_config.postprocess; no new→legacy mirror is needed.

@dataclass(kw_only=True)
class TaskConfig(SchemaInterface):
    task_name: str
    inference_engine: Optional[InferenceEngine] = None
    evaluation_engine: EvaluationEngine
    num_records: Optional[int] = None
    meta: TaskMeta
    dataset: TaskDataset
    prompts: Optional[TaskPrompts] = field(default_factory=TaskPrompts)
    inference: Optional[TaskInference] = field(default_factory=TaskInference)
    # New top-level postprocess (sibling to inference / evaluation).
    # Legacy nested location (evaluation.postprocess) is translated one-way
    # (legacy → new) in __post_init__ for backward yaml compatibility. There
    # is no reverse mirror: engine / postprocess pipeline / evaluate.py all
    # read task_config.postprocess directly.
    postprocess: Optional[TaskPostprocess] = None
    evaluation: Optional[TaskEvaluation] = None
    # Per-task verifier override (config.yaml ``verifier:`` block). None = not
    # specified -> evaluate.py falls back to VerifierArgs (CLI/default) per field.
    verifier: Optional[TaskVerifier] = None
    multithreading: Optional[Dict[str, Any]] = None
    arguments: Optional[Dict[str, Any]] = None

    def __post_init__(
        self,
    ):
        # Lenient hydration: when nested fields come in as raw dicts (e.g. from
        # JSON inference outputs written by an older schema revision), drop keys
        # the current dataclass no longer accepts so a stale field like
        # `meta.category` doesn't break re-evaluation. The dropped keys are
        # logged at debug level for traceability.
        def _hydrate(cls, value):
            if not isinstance(value, dict):
                return value
            _valid = {f.name for f in fields(cls)}
            _unknown = [k for k in value.keys() if k not in _valid]
            if _unknown:
                logger.debug(
                    "Dropping unknown fields %s when constructing %s",
                    _unknown, cls.__name__,
                )
            return cls(**{k: v for k, v in value.items() if k in _valid})

        self.meta = _hydrate(TaskMeta, self.meta)
        self.dataset = _hydrate(TaskDataset, self.dataset)
        if self.prompts:
            self.prompts = _hydrate(TaskPrompts, self.prompts)
        if self.inference:
            self.inference = _hydrate(TaskInference, self.inference)
        if self.evaluation:
            self.evaluation = _hydrate(TaskEvaluation, self.evaluation)
        if self.verifier:
            self.verifier = _hydrate(TaskVerifier, self.verifier)

        # Hydrate top-level postprocess from dict if needed
        if isinstance(self.postprocess, dict):
            self.postprocess = TaskPostprocess.from_dict(self.postprocess)

        # Legacy nested location (evaluation.postprocess) is consumed only as an
        # INPUT here: when an old config sets it, translate to the new top-level
        # TaskPostprocess. No reverse mirror — engine/postprocess/evaluate.py
        # all read task_config.postprocess directly.
        if (
            self.postprocess is None
            and self.evaluation is not None
            and self.evaluation.postprocess is not None
        ):
            _legacy_dict: Dict[str, Any] = {}
            if isinstance(self.evaluation.postprocess, TaskEvaluationPostprocess):
                for _f in fields(TaskEvaluationPostprocess):
                    _v = getattr(self.evaluation.postprocess, _f.name, None)
                    if _v is not None:
                        _legacy_dict[_f.name] = _v
            elif isinstance(self.evaluation.postprocess, dict):
                _legacy_dict = dict(self.evaluation.postprocess)
            if _legacy_dict:
                self.postprocess = TaskPostprocess.from_dict(_legacy_dict)

    @staticmethod
    def _select_mode_in_dict(data: Dict[str, Any], mode: str) -> Dict[str, Any]:
        """Resolve the build-time ``direct``/``reasoning`` variant *in place*
        on a raw yaml-style task_config dict — before schema hydration, since
        ``TaskPrompts``/``TaskPostprocess``/``TaskInference`` don't know about
        these mode keys.

        Acts on three sections (each may carry ``conditional_on`` at the parent
        level, which is propagated to the selected child when the child has
        none of its own):
          - ``data["prompts"]``
          - ``data["postprocess"]``
          - ``data["inference"]``

        Sections in already-flat form (no ``direct``/``reasoning`` key) are
        passed through untouched, so already-unwrapped JSON-hydrate dicts
        are a no-op.
        """
        def _unwrap(parent):
            if not isinstance(parent, dict):
                return None
            _child = parent.get(mode, None)
            if not isinstance(_child, dict):
                return None
            _condition = parent.get("conditional_on", None)
            if _condition and "conditional_on" not in _child:
                _child["conditional_on"] = _condition
            return _child

        for _section_key in ("prompts", "postprocess", "inference"):
            _unwrapped = _unwrap(data.get(_section_key))
            if _unwrapped is not None:
                data[_section_key] = _unwrapped
        return data

    @classmethod
    def ensure(cls, value, mode: str = "direct") -> "TaskConfig":
        """Coerce *value* to a TaskConfig instance and apply build-time
        defaults that depend on the *mode* ("direct" or "reasoning").

        - ``TaskConfig`` instance → returned as-is (mode unwrap is a no-op
          on already-hydrated instances; ``apply_reasoning_defaults`` is
          idempotent)
        - ``dict`` → mode unwrap on the raw dict, then ``cls(**value)``
        - other → ``TypeError``

        Centralizes the ``if isinstance(value, dict): value = TaskConfig(**value)``
        boilerplate and the build-time mode unwrap that previously lived
        scattered across each caller (evaluate.py, _build_task_config, …).
        """
        if isinstance(value, cls):
            instance = value
        elif isinstance(value, dict):
            cls._select_mode_in_dict(value, mode=mode)
            instance = cls(**value)
        else:
            raise TypeError(
                f"TaskConfig.ensure: expected TaskConfig or dict, got {type(value).__name__}"
            )
        instance.apply_reasoning_defaults(reasoning=(mode == "reasoning"))
        return instance

    def apply_reasoning_defaults(self, reasoning: bool = False) -> None:
        """Apply reasoning-mode defaults in place. Currently raises
        ``inference.generation_options.max_new_tokens`` to >= 8192 so the
        chain-of-thought has enough room. No-op when *reasoning* is falsy
        or the inference block is absent.

        Caller-driven (not invoked from ``__post_init__``) because
        ``reasoning`` is a runtime flag not part of the persisted schema.
        Invoke once per (task_config, reasoning) pairing — typically right
        after ``_build_task_config`` or after JSON-hydrating a stale config.
        """
        if not reasoning or self.inference is None:
            return
        _gen = self.inference.generation_options
        if not isinstance(_gen, TaskInferenceGenerationOptions):
            return
        _mnt = getattr(_gen, "max_new_tokens", None)
        if _mnt is None or (isinstance(_mnt, (int, float)) and _mnt <= 8192):
            _gen.max_new_tokens = 8192

    @classmethod
    def from_engine(cls, evaluation_engine, **kwargs):
        """Dispatch to engine-specific factory method."""
        if isinstance(evaluation_engine, str):
            evaluation_engine = EvaluationEngine(evaluation_engine)
        _dispatch = {
            EvaluationEngine.builtin: cls.from_builtin,
            EvaluationEngine.lm_eval_harness: cls.from_lm_eval_harness,
            EvaluationEngine.lmms_eval: cls.from_lmms_eval,
            EvaluationEngine.vlm_eval_kit: cls.from_vlm_eval_kit,
        }
        factory = _dispatch.get(evaluation_engine)
        if factory is None:
            raise ValueError(f"Unsupported evaluation engine: {evaluation_engine}")
        return factory(**kwargs)

    @classmethod
    def from_builtin(cls, task_name, reasoning=False):
        from omni_evaluator.evaluation.builtin import _build_task_config
        return _build_task_config(task_name=task_name, reasoning=reasoning)

    @classmethod
    def from_lm_eval_harness(cls, task_name, task, num_records, system_prompt=None, task_prompt=None):
        from omni_evaluator.evaluation.lm_eval_harness import _build_task_config
        return _build_task_config(
            task_name=task_name, task=task, num_records=num_records,
            system_prompt=system_prompt, task_prompt=task_prompt,
        )

    @classmethod
    def from_lmms_eval(cls, task_name, task, num_records, system_prompt=None, task_prompt=None):
        from omni_evaluator.evaluation.lmms_eval import _build_task_config
        return _build_task_config(
            task_name=task_name, task=task, num_records=num_records,
            system_prompt=system_prompt, task_prompt=task_prompt,
        )

    @classmethod
    def from_vlm_eval_kit(cls, task_name, dataset, num_records, system_prompt=None, task_prompt=None):
        from omni_evaluator.evaluation.vlm_eval_kit import _build_task_config
        return _build_task_config(
            task_name=task_name, dataset=dataset, num_records=num_records,
            system_prompt=system_prompt, task_prompt=task_prompt,
        )

