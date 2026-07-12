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

import argparse
import dataclasses
from dataclasses import dataclass, field, fields
from datetime import datetime
import huggingface_hub
import importlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Callable, Tuple, Union, Any, Optional, List, Dict
from uuid import uuid4

from omni_evaluator import ApiGroup, InferenceEngine, EvaluationEngine, EvaluationMethod, T2IGeneratorType
from omni_evaluator.utils.common import list_tasks
from omni_evaluator.utils.io import get_output_dirpath, read_file
from omni_evaluator.utils.string import sanitize_name
from omni_evaluator.utils.patches import update_package_resources

logger = logging.getLogger(__name__)


class NestedGroupAction(argparse.Action):
    def __init__(
        self, 
        option_strings: str, 
        dest: str, 
        group_name: str, 
        attr_name: str, 
        **kwargs,
    ):
        self.group_name = group_name
        self.attr_name = attr_name
        super().__init__(option_strings, dest, **kwargs)

    def __call__(
        self, 
        parser: argparse.ArgumentParser, 
        namespace: str, 
        values: str, 
        option_string: Optional[str] =None,
    ):
        nested_group = getattr(namespace, self.group_name, None)
        if nested_group is None:
            nested_group = argparse.Namespace()
            setattr(namespace, self.group_name, nested_group)
        attr_name = self.attr_name
        if attr_name.startswith(self.group_name):
            attr_name = attr_name[len(self.group_name)+1:] # "_"
        setattr(nested_group, attr_name, values)
        delattr(namespace, self.attr_name)
        
class CustomArgumentParser(argparse.ArgumentParser):
    """Argument parser with nested-group support and artifact restore/save."""

    def parse_args(
        self, 
        args: Optional[argparse.Namespace] = None, 
        namespace: Optional[argparse.Namespace] = None,
    ):
        namespaces = super().parse_args(args, namespace)
        self._apply_nested_defaults(namespaces)
        return namespaces

    def _apply_nested_defaults(
        self, 
        namespace: argparse.Namespace,
    ):
        for action in self._actions:
            if isinstance(action, NestedGroupAction):
                nested_group = getattr(namespace, action.group_name, None)
                if nested_group is None:
                    nested_group = argparse.Namespace()
                    setattr(namespace, action.group_name, nested_group)
                attr_name = action.attr_name
                if attr_name.startswith(action.group_name):
                    attr_name = attr_name[len(action.group_name)+1:] # "_"
                if not hasattr(nested_group, attr_name):
                    # set default in nested_group if not set in NestedGroupAction.__call__
                    setattr(nested_group, attr_name, action.default)
                    delattr(namespace, action.attr_name)

    @classmethod
    def restore_arguments(
        cls,
        artifact_filepath: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        ignore_arguments: Optional[List[str]] = None,
    ) -> None:
        # Restore CLI arguments from a previous experiment artifact file and inject them into sys.argv.
        # Args: artifact_filepath - path to a JSON artifact containing saved arguments; ignore_arguments - argument names to skip during restoration
        # Returns: None (modifies sys.argv in place)
        if ignore_arguments is None:
            ignore_arguments = [
                "artifact_filepath",
                "cache_dirpath",
                "output_dirpath",
                "remote_output_dirpath",
                "local_dirpath",
            ]
        arguments = None
        if artifact_filepath:
            if not os.path.exists(artifact_filepath):
                raise FileNotFoundError(f'Artifact_filepath not exist: {artifact_filepath}')
            artifact_meta = read_file(artifact_filepath)["meta"]
            arguments = artifact_meta["arguments"]
        
        if not arguments:
            raise ValueError(f'Artifact_filepath or arguments should be set to restore arguments')
        
        command_arguments = list()
        _visited = list()
        argument_start_idx = 0
        for _idx, _arg in enumerate(sys.argv):
            if (
                _arg.startswith("--")
                and argument_start_idx == 0
            ):
                argument_start_idx = _idx
            
            if argument_start_idx < 1:
                continue
            
            if (
                _idx >= len(sys.argv) - 1
                or sys.argv[_idx+1].startswith("--")
            ): # arg value is included in current arg
                _splited = _arg.split("=")
                _arg_name = _splited[0]
                _arg_name = _arg_name.replace("--", "").strip()
                if len(_splited) > 1:
                    _arg_value = _splited[1]
                    _arg_value = _arg_value.strip()
                command_arguments.append(_arg)
                _visited.append(_arg_name)
            else:
                _arg_name = _arg.replace("--", "").strip()
                _arg_value = sys.argv[_idx+1].strip()
                command_arguments.append(f'--{_arg_name}={_arg_value}')
                _visited.append(_arg)
        
        for _arg_name, _arg_value in arguments.items():
            _command_argument = None
            if (
                _arg_name in ignore_arguments
                or _arg_name in _visited
            ):
                continue 
            elif _arg_value is None:
                pass
            elif isinstance(_arg_value, bool):
                if _arg_value:
                    _command_argument = f'--{_arg_name}'
            elif isinstance(_arg_value, (int, float, str)):
                _command_argument = f'--{_arg_name}={_arg_value}'
            elif isinstance(_arg_value, (list, tuple)):
                _arg_value = ",".join(_arg_value)
                _command_argument = f'--{_arg_name}={_arg_value}'
            
            if _command_argument:
                command_arguments.append(_command_argument)
        sys.argv = sys.argv[:argument_start_idx] + command_arguments

def _add_dataclass_args(
    parser: argparse.ArgumentParser,
    dataclass_cls: type,
    group_name: Optional[str] = None,
) -> None:
    # Register all fields of a dataclass as argparse arguments on the given parser.
    # Args: dataclass_cls - dataclass whose fields become CLI arguments; group_name - if set, arguments are nested under a NestedGroupAction namespace
    # Returns: None (mutates parser in place)
    for _field in fields(dataclass_cls):
        _field_name = _field.name
        _field_name = _field_name.replace('-','_')
        _arg_name = f"--{_field_name}"
        if _arg_name in parser._option_string_actions:
            continue
        _kwargs = dict()
        _field_type = _field.type
        # Unwrap Optional[X] → X so argparse receives a callable type converter
        _type_origin = getattr(_field_type, "__origin__", None)
        _type_args = getattr(_field_type, "__args__", None)
        if _type_origin is Union and _type_args and type(None) in _type_args:
            _non_none = [a for a in _type_args if a is not type(None)]
            _field_type = _non_none[0] if len(_non_none) == 1 else str
        if _field_type is bool:
            if not _field.default:
                _kwargs["action"] = "store_true"
            else:
                _kwargs["action"] = "store_false"
        else:
            _kwargs["type"] = _field_type
            if _field.default is not dataclasses.MISSING:
                _kwargs["default"] = _field.default
        if group_name:
            _kwargs["action"] = NestedGroupAction
            _kwargs["group_name"] = group_name
            _kwargs["attr_name"] = _field_name

        parser.add_argument(
            _arg_name, 
            **_kwargs,
        )

def get_parser(
    parser: Optional[argparse.ArgumentParser] = None,
    argv: Optional[List[str]] = None,
) -> Tuple[argparse.ArgumentParser, List[Callable]]:
    # Build the argument parser by registering all dataclass arg groups and engine-specific args.
    # Args: argv - optional list of CLI strings to pre-parse for conditional arg groups (inference/evaluation engine)
    # Returns: tuple of (configured ArgumentParser, list of validation callables to run after parsing)
    if parser is None:
        parser = argparse.ArgumentParser()

    validations = list()
    _add_dataclass_args(parser, CommonArgs)
    validations.append(CommonArgs.validate)
    _add_dataclass_args(parser, PathArgs)
    validations.append(PathArgs.validate)
    _add_dataclass_args(parser, HuggingfaceArgs)
    validations.append(HuggingfaceArgs.validate)
    _add_dataclass_args(parser, SecurityArgs)
    validations.append(SecurityArgs.validate)
    _add_dataclass_args(parser, ApiArgs)
    validations.append(ApiArgs.validate)
    _add_dataclass_args(parser, InferenceArgs)
    validations.append(InferenceArgs.validate)
    _add_dataclass_args(parser, GenerationOptionArgs)
    validations.append(GenerationOptionArgs.validate)
    _add_dataclass_args(parser, PostprocessArgs)
    validations.append(PostprocessArgs.validate)
    _add_dataclass_args(parser, EvaluationArgs)
    validations.append(EvaluationArgs.validate)
    _add_dataclass_args(parser, VerifierArgs)
    validations.append(VerifierArgs.validate)
    _add_dataclass_args(parser, T2IGeneratorArgs)
    validations.append(T2IGeneratorArgs.validate)
    _add_dataclass_args(parser, S3ClientArgs)
    validations.append(S3ClientArgs.validate)
    
    _args = None
    if argv is not None:
        _args, _ = parser.parse_known_args(args=argv) # known_args, unknown_args
    else:
        _args, _ = parser.parse_known_args() # known_args, unknown_args
    
    if _args.use_cvs:
        _add_dataclass_args(parser, CvsClientArgs)
        validations.append(CvsClientArgs.validate)
    if _args.use_obs:
        _add_dataclass_args(parser, ObsClientArgs)
        validations.append(ObsClientArgs.validate)
    
    # inference_engine
    if _args.inference_engine == InferenceEngine.huggingface:
        _add_dataclass_args(parser, HuggingfaceInferenceEngineArgs)
        validations.append(HuggingfaceInferenceEngineArgs.validate)
    elif _args.inference_engine == InferenceEngine.vllm:
        _add_dataclass_args(parser, VllmInferenceEngineArgs)
        validations.append(VllmInferenceEngineArgs.validate)
    elif _args.inference_engine == InferenceEngine.sglang:
        _add_dataclass_args(parser, SglangInferenceEngineArgs)
        validations.append(SglangInferenceEngineArgs.validate)
    elif _args.inference_engine in [
        InferenceEngine.api__openai,
        InferenceEngine.api__anthropic,
        InferenceEngine.api__google,
    ]:
        _add_dataclass_args(parser, ApiInferenceEngineArgs)
        validations.append(ApiInferenceEngineArgs.validate)
    else:
        raise ValueError(f'Not supported `inference_engine`: {_args.inference_engine}')

    # evaluation_engine
    if _args.evaluation_engine == EvaluationEngine.builtin:
        _add_dataclass_args(parser, BuiltinEvaluationEngineArgs)
        validations.append(BuiltinEvaluationEngineArgs.validate)
    elif _args.evaluation_engine == EvaluationEngine.lmms_eval:
        _add_dataclass_args(parser, LmmsEvalEvaluationEngineArgs)
        validations.append(LmmsEvalEvaluationEngineArgs.validate)
    elif _args.evaluation_engine == EvaluationEngine.lm_eval_harness:
        _add_dataclass_args(parser, LmEvalHarnessEvaluationEngineArgs)
        validations.append(LmEvalHarnessEvaluationEngineArgs.validate)
    elif _args.evaluation_engine == EvaluationEngine.vlm_eval_kit:
        _add_dataclass_args(parser, VlmEvalKitEvaluationEngineArgs)
        validations.append(VlmEvalKitEvaluationEngineArgs.validate)
    else:
        raise ValueError(f'Not supported `evaluation_engine`: {_args.evaluation_engine}')

    if not _args.t2i_generator_type:
        pass
    elif _args.t2i_generator_type == T2IGeneratorType.ta_tok:
        _add_dataclass_args(parser, TaTokT2IGeneratorArgs, group_name="t2i_generator")
        validations.append(TaTokT2IGeneratorArgs.validate)
    elif _args.t2i_generator_type == T2IGeneratorType.hyperclova_vdm:
        _add_dataclass_args(parser, VdmT2IGeneratorArgs, group_name="t2i_generator")
        validations.append(VdmT2IGeneratorArgs.validate)
    else:
        raise ValueError(f'Not supported `t2i_generator_type`: {_args.t2i_generator_type}')

    return parser, validations


def _map_generation_kwargs(kwargs: dict) -> dict:
    # Map lmms_eval/lm_eval_harness generation_kwargs keys to omni_evaluator internal keys.
    result = {}
    for k, v in kwargs.items():
        if k == "until":
            if isinstance(v, str):
                v = [v]
            result["stop_words"] = v
        elif k == "max_gen_toks":
            result["max_new_tokens"] = v
        else:
            result[k] = v
    return result


@dataclass(kw_only=True)
class ArgsInterface:
    @classmethod
    def validate(cls, args):
        return args
    
@dataclass(kw_only=True)
class CommonArgs(ArgsInterface):
    exp_name: str = field(
        metadata={"help": "experiment name used for saving results and display"}
    )
    version_name: str = field(
        default=None,
        metadata={"help": "version name to distinguish between experiment runs"}
    )
    benchmarks: str = field(
        metadata={"help": "comma-separated list of benchmark names to evaluate"}
    )
    dataset_subset: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Filter dataset by sample meta values (overrides config-level dataset.subset). "
            "JSON dict, e.g. '{\"category\":[\"Math\",\"Algebra\"],\"question_type\":\"multiple_choice\"}', "
            "or shorthand 'category=Math|Algebra;question_type=multiple_choice'. "
            "AND across keys, OR within values. Empty/`{}`/null clears the config subset."
        )},
    )
    inference_engine: InferenceEngine = field(
        metadata={"help": "inference backend to use (e.g., huggingface, vllm, api/openai)"}
    )
    evaluation_engine: EvaluationEngine = field(
        default=EvaluationEngine.builtin,
        metadata={"help": "evaluation framework to use (default: %(default)s)"}
    )
    reasoning: bool = field(
        default=False,
        metadata={"help": "enable reasoning mode for inference and postprocessing"}
    )
    reasoning_effort: Optional[str] = field(
        default=None,
        metadata={"help": "reasoning effort level (low/medium/high) for OpenAI o-series models"}
    )
    thinking_budget: Optional[int] = field(
        default=None,
        metadata={"help": "thinking token budget for Anthropic/Google thinking models"}
    )
    skip_inference: bool = field(
        default=False,
        metadata={"help": "skip inference and load previous results"}
    )
    skip_evaluation: bool = field(
        default=False,
        metadata={"help": "skip evaluation and load previous results"}
    )
    rank: int = field(
        default=0,
        metadata={"help": "worker rank for multi-process inference (do not set manually)"}
    )
    world_size: int = field(
        default=1,
        metadata={"help": "number of parallel workers for multi-process inference (default: %(default)s)"}
    )
    use_cvs: bool = field(
        default=False,
        metadata={"help": "upload image vectors to CVS and replace IMG_LOC with cvs_id"}
    )
    use_obs: bool = field(
        default=False,
        metadata={"help": "fetch images from Object Storage by URL"}
    )
    do_async: bool = field(
        default=False,
        metadata={"help": "enable asynchronous inference requests"}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "enable debug mode with reduced dataset and verbose logging"}
    )
    resume: bool = field(
        default=False,
        metadata={"help": "resume from previously saved inference results"}
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "enable verbose logging (DEBUG level)"}
    )
    artifact_filepath: str = field(
        default=None,
        metadata={"help": "path to artifact file for reproducing a previous experiment"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Validate and normalize common args: set exp_name, parse benchmarks, enforce world_size, and load engine defaults.
        # Args: args - parsed namespace with exp_name, benchmarks, inference_engine, evaluation_engine, and flags
        # Returns: validated and mutated args namespace

        logging.basicConfig(
            level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if isinstance(args.inference_engine, InferenceEngine):
            args.inference_engine = args.inference_engine.value # Enum -> str
        if isinstance(args.evaluation_engine, EvaluationEngine):
            args.evaluation_engine = args.evaluation_engine.value # Enum -> str

        if (
            not isinstance(args.exp_name, str)
            or len(args.exp_name) < 1
        ):
            args.exp_name = f'untitled_exp_{uuid4()}__{datetime_str}'
        
        if (
            not isinstance(args.artifact_filepath, str)
            or not os.path.exists(args.artifact_filepath)
        ): # append suffix if not resuemd experiment
            args.exp_name = f'{args.exp_name}__{args.inference_engine}'
            if args.reasoning:
                args.exp_name = f'{args.exp_name}__reasoning'
            if args.debug:
                args.exp_name = f'debug__{args.exp_name}'
                    
            # Auxiliary modality suffixes — only added when the modality is
            # explicitly attached at the CLI level. Format:
            #   __<label><N>   when count is a positive int (e.g. __ocr512)
            #   __<label>      when count is negative (use-all sentinel)
            #   (no suffix)    when count is None or 0 (both mean off)
            # External-API flags (update_ocr / update_lens) substitute the label
            # (ocr_api / lens_api) so a single suffix carries both signals.
            for _label, _value, _api_flag, _api_label in (
                ("ocr",      args.num_ocr_tokens,    args.update_ocr,  "ocr_api"),
                ("entity",   args.num_entity_tokens, args.update_lens, "lens_api"),
                ("subtitle", args.num_subtitle_cues, False,            None),
            ):
                _resolved_label = _api_label if _api_flag else _label
                if isinstance(_value, int) and _value != 0:
                    if _value > 0:
                        args.exp_name += f'__{_resolved_label}{_value}'
                    else:                                  # negative = use all
                        args.exp_name += f'__{_resolved_label}'
                elif _api_flag and _api_label:
                    # external API on but no explicit count override
                    args.exp_name += f'__{_api_label}'
            
            # prevent unexpected filename stemming # e.g. "meta-llama/Llama-3.2-11B-Vision-Instruct" -> "meta-llama__Llama-3_2-11B-Vision-Instruct"
            args.exp_name = sanitize_name(args.exp_name)
            logger.info(f'Set `exp_name`: {args.exp_name}')
        # version_name shares the same output-dir namespace -> normalize identically (None -> skip).
        if isinstance(args.version_name, str) and args.version_name.strip():
            args.version_name = sanitize_name(args.version_name)
        
        if not (isinstance(args.world_size, int) and args.world_size > 0):
            raise ValueError(f'World_size should be greater than 0: {args.world_size}')
        if (
            args.world_size > 1
            and args.inference_engine not in [
                "huggingface",
            ]
        ):
            logger.warning(f'multiprocess is not supported for inference_engien: {args.inference_engine}')
            logger.warning(f'set `world_size`: {args.world_size} -> 0')
            args.world_size = 1
        
        # load default_benchmarks and copy custom_resources into installed package
        if args.evaluation_engine == EvaluationEngine.builtin:
            from omni_evaluator.evaluation.builtin import DEFAULT_BENCHMARKS
        elif args.evaluation_engine == EvaluationEngine.lmms_eval:
            from omni_evaluator.evaluation.lmms_eval import DEFAULT_BENCHMARKS
            _resource_dirpath = importlib.resources.files(
                "omni_evaluator.evaluation.lmms_eval",
            ).joinpath("resources/custom_tasks")
            update_package_resources(
                package_name="lmms_eval",
                source_dirpath=_resource_dirpath,
                target_dirpath="tasks", # "lmms-eval/lmms_eval/tasks"
            )
        elif args.evaluation_engine == EvaluationEngine.lm_eval_harness:
            from omni_evaluator.evaluation.lm_eval_harness import DEFAULT_BENCHMARKS
            _resource_dirpath = importlib.resources.files(
                "omni_evaluator.evaluation.lm_eval_harness",
            ).joinpath("resources/custom_tasks")
            update_package_resources(
                package_name="lm_eval",
                source_dirpath=_resource_dirpath,
                target_dirpath="tasks", # "lm-evaluation-harness/lm_eval/tasks"
            )
        elif args.evaluation_engine == EvaluationEngine.vlm_eval_kit:
            from omni_evaluator.evaluation.vlm_eval_kit import DEFAULT_BENCHMARKS
            # _resource_dirpath = importlib.resources.files(
            #     "omni_evaluator.evaluation.vlm_eval_kit",
            # ).joinpath("resources/custom_tasks")
            # update_package_resources(
            #     package_name="vlmeval",
            #     source_dirpath=_resource_dirpath,
            #     target_dirpath="tasks", # "VLMEvalKit/vlmeval/tasks"
            # )
        # set defulat_benchmarks if not specified
        if not isinstance(args.benchmarks, str) or len(args.benchmarks) < 1:
            args.benchmarks = DEFAULT_BENCHMARKS
            logger.info(f'Set default benchmarks: {args.benchmarks}')
        elif isinstance(args.benchmarks, str):
            args.benchmarks = args.benchmarks.split(",")
        if not isinstance(args.benchmarks, (list, tuple)):
            raise TypeError(f"Invalid benchmarks given: {args.benchmarks}")
        # Dedup preserving first-occurrence order — downstream OrderedDicts
        # keyed by task_name would collapse duplicates and trip evaluate.py's
        # length check.
        args.benchmarks = list(dict.fromkeys(e.strip() for e in args.benchmarks))

        # check if all benchmarks supported by evaluation_engine
        available_benchmarks = list_tasks(evaluation_engine=args.evaluation_engine)
        for benchmark in args.benchmarks:
            if benchmark not in available_benchmarks:
                raise ValueError(f'Unsupported benchmark for {args.evaluation_engine}: `{benchmark}`')

        # Parse --dataset_subset: accept JSON dict OR shorthand
        # 'k1=a|b;k2=c'. Empty/'null'/'{}' → empty dict = clear config subset.
        # None (CLI omitted) → leave config subset alone.
        if isinstance(args.dataset_subset, str):
            _raw = args.dataset_subset.strip()
            _parsed: Dict[str, List[str]] = {}
            if _raw in ("", "null", "{}"):
                _parsed = {}
            else:
                _looks_json = _raw.startswith("{") and _raw.endswith("}")
                if _looks_json:
                    import json as _json
                    try:
                        _obj = _json.loads(_raw)
                    except _json.JSONDecodeError as _e:
                        raise ValueError(f"--dataset_subset JSON parse error: {_e}")
                    if not isinstance(_obj, dict):
                        raise ValueError(f"--dataset_subset must decode to a dict, got {type(_obj).__name__}")
                    _parsed = _obj
                else:
                    # shorthand 'k1=a|b;k2=c'
                    for _pair in _raw.split(";"):
                        _pair = _pair.strip()
                        if not _pair:
                            continue
                        if "=" not in _pair:
                            raise ValueError(f"--dataset_subset shorthand pair missing '=': {_pair!r}")
                        _k, _v = _pair.split("=", 1)
                        _k, _v = _k.strip(), _v.strip()
                        if not _k:
                            raise ValueError(f"--dataset_subset shorthand has empty key: {_pair!r}")
                        if _v == "" or _v.lower() == "null":
                            _parsed[_k] = None
                        else:
                            _parsed[_k] = [x.strip() for x in _v.split("|") if x.strip()]
            args.dataset_subset = _parsed
            logger.info(f"Set `dataset_subset`: {args.dataset_subset}")

        return args
    
@dataclass(kw_only=True)
class PathArgs(ArgsInterface):
    cache_dirpath: str = field(
        default=None,
        metadata={"help": "directory for caching intermediate results during inference (overrides CACHE_DIRPATH env var; fallback: ./cache)"}
    )
    output_dirpath: str = field(
        default=None,
        metadata={"help": "local output directory for inference and evaluation results (overrides OUTPUT_DIRPATH env var; fallback: ./output)"}
    )
    remote_output_dirpath: str = field(
        default="evaluator/v1.0",
        metadata={"help": "remote output directory on S3-compatible storage for inference and evaluation results"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Resolve cache_dirpath / output_dirpath with priority: CLI > env > hardcoded fallback.
        # Dataclass defaults are None so that "not provided" is distinguishable from "explicit value".
        if not (isinstance(args.cache_dirpath, str) and len(args.cache_dirpath) > 0):
            args.cache_dirpath = os.getenv("CACHE_DIRPATH") or None
        if not (isinstance(args.output_dirpath, str) and len(args.output_dirpath) > 0):
            args.output_dirpath = os.getenv("OUTPUT_DIRPATH") or None

        # cache_dirpath
        if (
            not isinstance(args.cache_dirpath, str)
            or len(args.cache_dirpath) < 1
        ):
            args.cache_dirpath = os.path.abspath("./cache")
        args.cache_dirpath = get_output_dirpath(
            output_dirpath=args.cache_dirpath,
            evaluation_engine=args.evaluation_engine,
            exp_name=args.exp_name,
            version_name=args.version_name,
        )
        os.environ["CACHE_DIRPATH"] = args.cache_dirpath
        logger.info(f'Set `cache_dirpath`: {args.cache_dirpath}')
        if not Path(args.cache_dirpath).exists():
            Path(args.cache_dirpath).mkdir(exist_ok=True, parents=True)
            logger.info(f'Created cache_dirpath: {args.cache_dirpath}')

        # output_dirpath
        if (
            not isinstance(args.output_dirpath, str)
            or len(args.output_dirpath) < 1
        ):
            args.output_dirpath = "./output/"
        else:
            args.output_dirpath = get_output_dirpath(
                output_dirpath=args.output_dirpath,
                evaluation_engine=args.evaluation_engine,
                exp_name=args.exp_name,
                version_name=args.version_name,
            )
            logger.info(f'Set `output_dirpath`: {args.output_dirpath}')
            if not Path(args.output_dirpath).exists():
                Path(args.output_dirpath).mkdir(exist_ok=True, parents=True)
                logger.info(f'Created `output_dirpath`: {args.output_dirpath}')
        os.environ["OUTPUT_DIRPATH"] = args.output_dirpath
        
        # remote_output_dirpath
        if (
            not isinstance(args.remote_output_dirpath, str)
            or len(args.remote_output_dirpath) < 1
        ):
            args.remote_output_dirpath = None
        else:
            args.remote_output_dirpath = get_output_dirpath(
                output_dirpath=args.remote_output_dirpath,
                evaluation_engine=args.evaluation_engine,
                exp_name=args.exp_name,
                version_name=args.version_name,
            )
            logger.info(f'Set `remote_output_dirpath`: {args.remote_output_dirpath}')

        return args
    
@dataclass(kw_only=True)
class HuggingfaceArgs(ArgsInterface):
    hf_token: str = field(
        default=None,
        metadata={"help": "Hugging Face API token (overrides HF_TOKEN env var)"}
    )
    hf_home: str = field(
        default=None,
        metadata={"help": "Hugging Face home directory (overrides HF_HOME env var; fallback: ~/.cache/huggingface)"}
    )
    hf_hub_cache: str = field(
        default=None,
        metadata={"help": "Hugging Face Hub cache directory (overrides HF_HUB_CACHE env var; fallback: ~/.cache/huggingface/hub)"}
    )
    hf_allow_code_eval: str = field(
        default="1",
        metadata={"help": "allow code evaluation in Hugging Face evaluate (sets HF_ALLOW_CODE_EVAL)"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Sync HuggingFace env vars (HF_TOKEN, HF_HOME, HF_HUB_CACHE) with args and reload huggingface_hub.
        # Args: args - parsed namespace with hf_token, hf_home, hf_hub_cache, and hf_allow_code_eval
        # Returns: validated args with HuggingFace paths resolved and env vars set
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logger.info(f'Set `TOKENIZERS_PARALLELISM`: {os.getenv("TOKENIZERS_PARALLELISM", None)}')
        
        if (
            isinstance(args.hf_token, str) 
            and len(args.hf_token) > 0
        ):
            os.environ["HF_TOKEN"] = args.hf_token # legacy
            os.environ["HUGGINGFACE_TOKEN"] = args.hf_token # up-to-date
        elif (
            os.getenv("HF_TOKEN", None)
            or os.getenv("HUGGINGFACE_TOKEN", None)
        ):
            args.hf_token = os.environ["HF_TOKEN"] or os.environ["HUGGINGFACE_TOKEN"]
        logger.info(f'Set `HF_TOKEN`: {os.getenv("HF_TOKEN", None)}')
        logger.info(f'Set `HUGGINGFACE_TOKEN`: {os.getenv("HUGGINGFACE_TOKEN", None)}')
        
        if (
            isinstance(args.hf_allow_code_eval, str)
            and len(args.hf_allow_code_eval) > 0
        ):
            os.environ["HF_ALLOW_CODE_EVAL"] = args.hf_allow_code_eval
        elif os.getenv("HF_ALLOW_CODE_EVAL", None):
            args.hf_allow_code_eval = "1"
        logger.info(f'Set `HF_ALLOW_CODE_EVAL`: {os.getenv("HF_ALLOW_CODE_EVAL", None)}')
        
        # hf_home — priority: CLI > env > fallback (~/.cache/huggingface)
        if isinstance(args.hf_home, str) and len(args.hf_home) > 0:
            os.environ["HF_HOME"] = args.hf_home
        elif os.getenv("HF_HOME"):
            args.hf_home = os.environ["HF_HOME"]
        else:
            args.hf_home = os.path.expanduser("~/.cache/huggingface")
            os.environ["HF_HOME"] = args.hf_home

        # hf_hub_cache — priority: CLI > env > derived from hf_home
        if isinstance(args.hf_hub_cache, str) and len(args.hf_hub_cache) > 0:
            os.environ["HF_HUB_CACHE"] = args.hf_hub_cache
            os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_hub_cache
        elif (
            os.getenv("HF_HUB_CACHE", None)
            or os.getenv("HUGGINGFACE_HUB_CACHE", None)
        ):
            args.hf_hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        else:
            args.hf_hub_cache = args.hf_home
            os.environ["HF_HUB_CACHE"] = args.hf_hub_cache
            os.environ["HUGGINGFACE_HUB_CACHE"] = args.hf_hub_cache
            
        if (
            isinstance(args.hf_hub_cache, str)
            and Path(args.hf_hub_cache).name != "hub"
        ): # ~/.cache/huggingface -> ~/.cache/huggingface/hub
            args.hf_hub_cache = os.path.join(args.hf_hub_cache, "hub")

        logger.info(f'Set `HF_HOME`: {os.getenv("HF_HOME", None)}')
        logger.info(f'Set `HF_HUB_CACHE`: {os.getenv("HF_HUB_CACHE", None)}')
        logger.info(f'Set `HUGGINGFACE_HUB_CACHE`: {os.getenv("HUGGINGFACE_HUB_CACHE", None)}')
        
        # reload huggingface_hub to update env_vars
        if "huggingface_hub" in sys.modules:
            del sys.modules["huggingface_hub"]
            import huggingface_hub
        if "huggingface_hub.constants" in sys.modules:
            del sys.modules["huggingface_hub.constants"]
            from huggingface_hub import constants
        logger.info(f'reloaded "huggingface_hub"')
        
        return args
 
@dataclass(kw_only=True)
class SecurityArgs(ArgsInterface):
    allowed_hosts: str = field(
        default=None,
        metadata={"help": "comma-separated hostnames or CIDR ranges allowed to bypass the SSRF private-IP check in `_validate_url_safe` (overrides ALLOWED_HOSTS env var)"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Sync ALLOWED_HOSTS env with CLI arg. Priority: CLI > env > unset (strict SSRF guard).
        # Args: args - parsed namespace with allowed_hosts
        if (
            isinstance(args.allowed_hosts, str)
            and len(args.allowed_hosts) > 0
        ):
            os.environ["ALLOWED_HOSTS"] = args.allowed_hosts
        elif os.getenv("ALLOWED_HOSTS", None):
            args.allowed_hosts = os.environ["ALLOWED_HOSTS"]
        logger.info(f'Set `ALLOWED_HOSTS`: {os.getenv("ALLOWED_HOSTS", None)}')
        return args


@dataclass(kw_only=True)
class ApiArgs(ArgsInterface):
    openai_api_url: str = field(
        default=None,
        metadata={"help": "OpenAI API base URL"}
    )
    openai_api_key: str = field(
        default=None,
        metadata={"help": "OpenAI API key (overrides OPENAI_API_KEY env var)"}
    )
    openai_api_organization: str = field(
        default=None,
        metadata={"help": "OpenAI organization ID (overrides OPENAI_ORGANIZATION env var)"}
    )
    azure_endpoint: str = field(
        default="https://api.cognitive.microsoft.com/sts/v1.0/issueToken",
        metadata={"help": "Azure API endpoint URL"}
    )
    azure_api_key: str = field(
        default=None,
        metadata={"help": "Azure API key (overrides AZURE_API_KEY env var)"}
    )
    anthropic_api_url: str = field(
        default="https://api.anthropic.com/v1/complete",
        metadata={"help": "Anthropic API base URL"}
    )
    anthropic_api_key: str = field(
        default=None,
        metadata={"help": "Anthropic API key (overrides ANTHROPIC_API_KEY env var)"}
    )
    google_api_key: str = field(
        default=None,
        metadata={"help": "Google API key (overrides GOOGLE_API_KEY env var)"}
    )
    
    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Sync API keys and URLs (OpenAI, Anthropic, Google, Azure) between args and environment variables.
        # Args: args - parsed namespace with openai/anthropic/google/azure API keys and URLs
        # Returns: validated args with all API env vars set
        # OPENAI_API_URL canonical form is the BASE URL (e.g. https://api.openai.com/v1).
        # New-style providers (lmms-eval llm_judge.providers.openai) pass this as OpenAI SDK
        # base_url; the SDK auto-appends "/chat/completions", so a full endpoint URL here
        # produces a duplicated path. Any inbound value ending in "/chat/completions" is
        # stripped back to base form. Legacy tasks using requests.post(URL) must re-append
        # "/chat/completions" themselves.
        if (
            isinstance(args.openai_api_url, str)
            and len(args.openai_api_url) > 0
        ):
            _url = args.openai_api_url.rstrip("/")
            if _url.endswith("/chat/completions"):
                _url = _url[: -len("/chat/completions")]
            args.openai_api_url = _url
            os.environ["OPENAI_API_URL"] = _url
        elif os.getenv("OPENAI_API_URL", None):
            _url = os.environ["OPENAI_API_URL"].rstrip("/")
            if _url.endswith("/chat/completions"):
                _url = _url[: -len("/chat/completions")]
                os.environ["OPENAI_API_URL"] = _url
            args.openai_api_url = _url
        logger.info(f'Set `OPENAI_API_URL`: {os.getenv("OPENAI_API_URL", None)}')

        if (
            isinstance(args.openai_api_key, str)
            and len(args.openai_api_key) > 0
        ):
            os.environ["OPENAI_API_KEY"] = args.openai_api_key
        elif os.getenv("OPENAI_API_KEY", None):
            args.openai_api_key = os.environ["OPENAI_API_KEY"]
        logger.info(f'Set `OPENAI_API_KEY`: {os.getenv("OPENAI_API_KEY", None)}')

        if (
            isinstance(args.openai_api_organization, str)
            and len(args.openai_api_organization) > 0
        ):
            os.environ["OPENAI_ORGANIZATION"] = args.openai_api_organization
        elif os.getenv("OPENAI_ORGANIZATION", None):
            args.openai_api_organization = os.environ["OPENAI_ORGANIZATION"]
        logger.info(f'Set `OPENAI_ORGANIZATION`: {os.getenv("OPENAI_ORGANIZATION", None)}')

        if (
            isinstance(args.anthropic_api_url, str)
            and len(args.anthropic_api_url) > 0
        ):
            os.environ["ANTHROPIC_API_URL"] = args.anthropic_api_url
        elif os.getenv("ANTHROPIC_API_URL", None):
            args.anthropic_api_url = os.environ["ANTHROPIC_API_URL"]
        logger.info(f'Set `ANTHROPIC_API_URL`: {os.getenv("ANTHROPIC_API_URL", None)}')

        if (
            isinstance(args.anthropic_api_key, str)
            and len(args.anthropic_api_key) > 0
        ):
            os.environ["ANTHROPIC_API_KEY"] = args.anthropic_api_key
        elif os.getenv("ANTHROPIC_API_KEY", None):
            args.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"]
        logger.info(f'Set `ANTHROPIC_API_KEY`: {os.getenv("ANTHROPIC_API_KEY", None)}')

        if (
            isinstance(args.google_api_key, str)
            and len(args.google_api_key) > 0
        ):
            os.environ["GOOGLE_API_KEY"] = args.google_api_key
        elif os.getenv("GOOGLE_API_KEY", None):
            args.google_api_key = os.environ["GOOGLE_API_KEY"]
        logger.info(f'Set `GOOGLE_API_KEY`: {os.getenv("GOOGLE_API_KEY", None)}')

        if (
            isinstance(args.azure_endpoint, str)
            and len(args.azure_endpoint) > 0
        ):
            os.environ["AZURE_ENDPOINT"] = args.azure_endpoint
        elif os.getenv("AZURE_ENDPOINT", None):
            args.azure_endpoint = os.environ["AZURE_ENDPOINT"]
        logger.info(f'Set `AZURE_ENDPOINT`: {os.getenv("AZURE_ENDPOINT", None)}')

        if (
            isinstance(args.azure_api_key, str)
            and len(args.azure_api_key) > 0
        ):
            os.environ["AZURE_API_KEY"] = args.azure_api_key
        elif os.getenv("AZURE_API_KEY", None):
            args.azure_api_key = os.environ["AZURE_API_KEY"]
        logger.info(f'Set `AZURE_API_KEY`: {os.getenv("AZURE_API_KEY", None)}')
        
        return args

@dataclass(kw_only=True)
class GenerationOptionArgs(ArgsInterface):
    do_sample: Optional[bool] = field(
        default=None,
        metadata={"help": "enable sampling during generation; None uses model default (HuggingFace only)"}
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "sampling temperature for generation"}
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={"help": "top-p (nucleus) sampling threshold for generation"}
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"help": "top-k filtering value for generation (not supported by OpenAI)"}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={"help": "number of beams for beam search (HuggingFace only)"}
    )
    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "maximum number of new tokens to generate"}
    )
    repetition_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "repetition penalty; maps to frequency_penalty for OpenAI/Google"}
    )
    length_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "length penalty for generation (HuggingFace only)"}
    )
    stop_words: Optional[str] = field(
        default=None,
        metadata={"help": "comma-separated list of stop words/sequences to halt generation"}
    )
    frequency_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "frequency penalty for generation (OpenAI, Google)"}
    )
    presence_penalty: Optional[float] = field(
        default=None,
        metadata={"help": "presence penalty for generation (OpenAI, Google)"}
    )
    n: Optional[int] = field(
        default=None,
        metadata={"help": "number of output sequences to generate (vLLM, SGLang, OpenAI)"}
    )
    logprobs: Optional[int] = field(
        default=None,
        metadata={"help": "number of log probabilities to return per token (vLLM, SGLang, OpenAI)"}
    )
    top_logprobs: Optional[int] = field(
        default=None,
        metadata={"help": "number of top token log probabilities to return (OpenAI only)"}
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "random seed for generation reproducibility (vLLM, SGLang)"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        if isinstance(args.stop_words, str) and len(args.stop_words) > 0:
            args.stop_words = [s.strip() for s in args.stop_words.split(",") if s.strip()]
        else:
            args.stop_words = None
        return args


@dataclass(kw_only=True)
class InferenceArgs(ArgsInterface):
    num_runs: int = field(
        default=1,
        metadata={"help": "number of inference runs; uses task config value if not set"}
    )
    system_prompt: str = field(
        default=None,
        metadata={"help": "system prompt to prepend to each request"}
    )
    task_prompt: str = field(
        default=None,
        metadata={"help": "task prompt or template string to format per sample"}
    )
    inference_concurrency: int = field(
        default=64,
        metadata={"help": "concurrent inference operations: GPU batch size (HF local), or max in-flight requests (vLLM/sglang/api async) (default: %(default)s)"}
    )
    request_timeout: float = field(
        default=1800,
        metadata={"help": "timeout in seconds for HTTP requests (default: %(default)s)"}
    )
    socket_timeout: float = field(
        default=900,
        metadata={"help": "per-chunk socket read timeout in seconds for asynchronous requests (default: %(default)s)"}
    )
    max_retry: int = field(
        default=5,
        metadata={"help": "max number of retries on request failure (default: %(default)s)"}
    )
    wait_between_retry: int = field(
        default=5,
        metadata={"help": "wait time in seconds between retries (default: %(default)s)"}
    )
    torch_dtype: str = field(
        default="float16",
        metadata={"help": "torch data type for model weights (default: %(default)s)"}
    )
    update_ocr: bool = field(
        default=False,
        metadata={"help": "update OCR tokens using external OCR API"}
    )
    update_lens: bool = field(
        default=False,
        metadata={"help": "update entity tokens using external Lens API"}
    )
    num_ocr_tokens: int = field(
        default=None,
        metadata={"help": "number of OCR tokens to include; None or 0 disables, negative uses all available, positive N keeps first N"}
    )
    num_entity_tokens: int = field(
        default=None,
        metadata={"help": "number of entity tokens to include; None or 0 disables, negative uses all available, positive N keeps first N"}
    )
    num_subtitle_cues: int = field(
        default=None,
        metadata={"help": "number of subtitle cues to include; None or 0 disables, negative uses all available, positive N keeps first N"}
    )
    entity_keyword_threshold: float = field(
        default=3,
        metadata={"help": "minimum score threshold for entity keywords (default: %(default)s)"}
    )
    entity_keyword_fashion_threshold: float = field(
        default=15,
        metadata={"help": "minimum score threshold for fashion entity keywords (default: %(default)s)"}
    )
    # None defaults below mean "let the model / server use its own default".
    # Non-None values are forwarded as overrides:
    #   HF engine → processor kwargs (Qwen-style models fall back to their own
    #               hard-coded defaults when None is passed in).
    #   vLLM engine → media_io_kwargs (max_video_frames, fps) + mm_processor_kwargs
    #                 (min_pixels, max_pixels) on the OpenAI extra_body field.
    min_pixels: int = field(
        default=None,
        metadata={"help": "minimum pixel count for image resizing; None → model/server default"}
    )
    max_pixels: int = field(
        default=None,
        metadata={"help": "maximum pixel count for image resizing; None → model/server default"}
    )
    max_video_frames: int = field(
        default=None,
        metadata={"help": "max video frames to extract; None → model/server default. Forwards to vLLM as media_io_kwargs.video.num_frames"}
    )
    fps: float = field(
        default=None,
        metadata={"help": "video sampling fps; None → model/server default. Forwards to vLLM as media_io_kwargs.video.fps"}
    )
    use_audio_in_video: bool = field(
        default=False,
        metadata={"help": "use audio track from video inputs during inference"}
    )

    @classmethod
    def validate(cls, args):
        from omni_evaluator.utils.torch import resolve_torch_dtype
        args.torch_dtype = resolve_torch_dtype(torch_dtype=args.torch_dtype)
        args.torch_dtype = str(args.torch_dtype)
        logger.info(f'Set torch_dtype: {args.torch_dtype}')

        # Numeric gates for auxiliary modalities. Semantics across all three:
        #   None or negative → use everything available
        #   0               → disable
        #   positive N      → keep first N
        for _attr in ("num_ocr_tokens", "num_entity_tokens", "num_subtitle_cues"):
            _val = getattr(args, _attr, None)
            if _val is not None and not isinstance(_val, int):
                raise ValueError(f'`{_attr}` should be None or int: {_val!r}')
        return args
        
@dataclass(kw_only=True)
class PostprocessArgs(ArgsInterface):
    postprocess_pipeline: str = field(
        default=None,
        metadata={"help": "comma-separated sequence of postprocess logic steps to apply"}
    )
    postprocess_version: str = field(
        default=None,
        metadata={"help": "comma-separated versions for each postprocess step (must match pipeline length)"}
    )
    postprocess_allow_api: bool = field(
        default=False,
        metadata={"help": "allow external API calls during postprocessing"}
    )
    postprocess_api_name: str = field(
        default="gpt-4o-mini-2024-07-18",
        metadata={"help": "API model name for postprocessing (requires --postprocess_allow_api)"}
    )
    parse_boxed: bool = field(
        default=False,
        metadata={"help": r"extract \boxed{} answers during postprocessing"}
    )
    postprocess_verbose: bool = field(
        default=False,
        metadata={"help": "enable verbose logging for postprocessing results"}
    )
    
    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Parse and validate the postprocess pipeline, version list, and API permissions.
        # Args: args - parsed namespace with postprocess_pipeline, postprocess_version, and postprocess_allow_api
        # Returns: validated args with pipeline/version parsed into lists
        if args.postprocess_pipeline is None:
            pass
        elif isinstance(args.postprocess_pipeline, str):
            args.postprocess_pipeline = args.postprocess_pipeline.split(",")
            args.postprocess_pipeline = [e.strip() for e in args.postprocess_pipeline if len(e.strip()) > 0]
        if not (args.postprocess_pipeline is None or isinstance(args.postprocess_pipeline, (list, tuple))):
            raise TypeError(f'Argument `postprocess_pipeline` should be None, string, or list type: {args.postprocess_pipeline}')
        if isinstance(args.postprocess_pipeline, (list, tuple)):
            for _postprocess_logic in args.postprocess_pipeline:
                from omni_evaluator.postprocess import PostprocessLogic
                if _postprocess_logic not in PostprocessLogic:
                    raise ValueError(f'Postprocess logic should be one of {list(PostprocessLogic.keys())}: {_postprocess_logic}')
        
        if args.postprocess_version is None:
            pass
        elif isinstance(args.postprocess_version, str):
            if isinstance(args.postprocess_pipeline, (list, tuple)):
                args.postprocess_version = args.postprocess_version.split(",")
                args.postprocess_version = [e.strip() for e in args.postprocess_version if len(e.strip()) > 0]
        if not (
            args.postprocess_version is None
            or (
                args.postprocess_pipeline is None
                and isinstance(args.postprocess_version, str)
            )
            or (
                isinstance(args.postprocess_pipeline, (list, tuple))
                and isinstance(args.postprocess_version, (list, tuple))
                and len(args.postprocess_pipeline) == len(args.postprocess_version)
            )
        ):
            raise ValueError(f'Argument `postprocess_version` should be None, string, or list with same length of postprocess_pipeline: {args.postprocess_version}')
        
        if args.postprocess_allow_api:
            if not (isinstance(args.postprocess_api_name, str) and len(args.postprocess_api_name) > 0):
                raise ValueError(f'`postprocess_api_name` should be set when `postprocess_allow_api` is True: got {args.postprocess_api_name!r}')
            from omni_evaluator.api import get_api_group
            api_group = get_api_group(api_name=args.postprocess_api_name)
            if api_group == ApiGroup.openai:
                if not args.openai_api_key:
                    raise ValueError(f'`openai_api_key` should be set when `postprocess_allow_api` and postprocess_api_name is `{args.postprocess_api_name}`')
            elif api_group == ApiGroup.anthropic:
                if not args.anthropic_api_key:
                    raise ValueError(f'`anthropic_api_key` should be set when `postprocess_allow_api` and postprocess_api_name is `{args.postprocess_api_name}`')
            elif api_group == ApiGroup.google:
                if not args.google_api_key:
                    raise ValueError(f'`google_api_key` should be set when `postprocess_allow_api` and postprocess_api_name is `{args.postprocess_api_name}`')
            logger.info(f'allow api while postprocessing: {args.postprocess_api_name}')
            
        return args
    
@dataclass(kw_only=True)
class EvaluationArgs(ArgsInterface):
    evaluation_methods: str = field(
        default=EvaluationMethod.generation.value,
        metadata={"help": "comma-separated evaluation methods (e.g., generation, perplexity)"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Parse evaluation_methods string into a list and ensure it matches the number of benchmarks.
        # Args: args - parsed namespace with evaluation_methods (comma-separated string) and benchmarks list
        # Returns: validated args with evaluation_methods as a list aligned to benchmarks
        if not isinstance(args.evaluation_methods, str):
            args.evaluation_methods = len(args.benchmarks) * [
                EvaluationMethod.generation.value, 
            ]
        elif "," in args.evaluation_methods:
            args.evaluation_methods = args.evaluation_methods.split(",")
        else:
            args.evaluation_methods = len(args.benchmarks) * [args.evaluation_methods, ]
        if not isinstance(args.evaluation_methods, (list, tuple)):
            raise TypeError(f"Invalid evaluation_methods given: {args.evaluation_methods}")
        args.evaluation_methods = [e.strip() for e in args.evaluation_methods]
        valid = [EvaluationMethod.generation.value, EvaluationMethod.perplexity.value]
        invalid = [e for e in args.evaluation_methods if e not in valid]
        if invalid:
            raise ValueError(
                f"Invalid `evaluation_method`: {invalid!r}. "
                f"Valid values: {valid!r}. "
                f"Got evaluation_methods={args.evaluation_methods!r}"
            )
        if len(args.evaluation_methods) != len(args.benchmarks):
            raise ValueError(f"Length not match between benchmarks and evaluation_methods: {len(args.benchmarks)} vs. {len(args.evaluation_methods)}")
        return args


@dataclass(kw_only=True)
class VerifierArgs(ArgsInterface):
    """Verifier judge configuration.

    Replaces the legacy ``--compute-judge-score`` boolean flag with a richer
    surface that exposes the underlying backend (HF model / API) and the
    generation params that used to be hardcoded in
    ``_DEFAULT_JUDGE_LOGICS["judge_score"]``. Defaults preserve the legacy
    behavior — ``api/openai`` engine + ``gpt-5-mini`` + ``max_tokens=1024`` +
    ``temperature=0.0`` — so flipping only ``--compute-verifier-score`` yields
    output identical to the old ``--compute-judge-score`` path.
    """

    # master switch
    enable_verifier: bool = field(
        default=False,
        metadata={"help": "compute the verifier judge score (replaces the legacy "
                          "--compute-judge-score flag). Default backend is "
                          "api/openai + gpt-5-mini for behavioral parity."}
    )

    # backend selection (shares InferenceEngine enum values)
    verifier_engine: str = field(
        default=InferenceEngine.api__openai.value,
        metadata={"help": "verifier inference engine: huggingface | llama_cpp | "
                          "api/openai | api/anthropic | api/google"}
    )
    verifier_api_name: str = field(
        default="gpt-5-mini",
        metadata={"help": "API model name when verifier_engine is api/* "
                          "(e.g., gpt-5-mini, claude-haiku-4-5). Mirrors "
                          "ApiInferenceEngineArgs.api_name naming. Default "
                          "reproduces the legacy judge_score path."}
    )
    verifier_model_name_or_path: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "HF id / local path when verifier_engine=huggingface. "
                          "Mirrors HuggingfaceInferenceEngineArgs.model_name_or_path "
                          "naming. Accepts HF hub id, full-model dir, or LoRA "
                          "adapter dir (adapter_config.json detected automatically)."}
    )
    verifier_model_group: str = field(
        default="qwen3",
        metadata={"help": "HF adapter model_group for the verifier checkpoint "
                          "(huggingface engine). Default qwen2_omni (the trained "
                          "verifier is a full-omni Qwen2.5-Omni checkpoint). Set "
                          "explicitly because the local checkpoint path can't be "
                          "auto-detected by get_model_group."}
    )
    verifier_device_map: Optional[str] = field(
        default=None,
        metadata={"help": "device_map for the verifier HF backend (huggingface engine). "
                          "Default None -> HuggingfaceInferencer default. For "
                          "verifier_engine=llama_cpp it is forced to cpu (llama_cpp picks "
                          "the device via n_gpu_layers, not device_map). Set auto / cuda:N "
                          "to place the HF verifier on GPU."}
    )
    verifier_gguf_filename: str = field(
        default="*Q8_0.gguf",
        metadata={"help": "GGUF filename glob selecting the quant for the llama_cpp engine "
                          "(resolved inside verifier_model_name_or_path's local dir, or passed "
                          "as Llama.from_pretrained's filename for a hub repo). Default Q8_0; "
                          "set e.g. *f16.gguf for fp16."}
    )
    verifier_alias: Optional[str] = field(
        default=None,
        metadata={"help": "label namespacing the recorded metric as verifier_score/{alias} so "
                          "multiple verifiers / checkpoints stay distinct. Default None -> derived "
                          "+ sanitized in validate: api_name for api/*, else the tail of "
                          "verifier_model_name_or_path."}
    )

    # generation params (was hardcoded in _DEFAULT_JUDGE_LOGICS["judge_score"])
    verifier_max_new_tokens: int = field(
        default=512,
        metadata={"help": "verifier generation max new tokens"}
    )
    verifier_temperature: float = field(
        default=0.0,
        metadata={"help": "verifier sampling temperature (0.0 = greedy)"}
    )
    verifier_reasoning: bool = field(
        default=False,
        metadata={"help": "use the CoT verifier prompt (VERIFIER_COT_PROMPT) and "
                          "chain-of-thought parsing instead of VERIFIER_PROMPT. Only "
                          "this flag toggles the COT variant."}
    )
    verifier_num_concurrency: int = field(
        default=1,
        metadata={"help": "verifier degree of parallelism. huggingface-GPU -> model.generate "
                          "batch size; api/* -> concurrent in-flight requests (asyncio.Semaphore); "
                          "llama_cpp -> worker processes (clamped to the NUMA node count). "
                          "huggingface-CPU ignores it (always bs=1). Default 1."}
    )
    verifier_max_seq_len: Optional[int] = field(
        default=5120,
        metadata={"help": "token budget for the verifier prompt (huggingface / llama_cpp). When "
                          "set, reference/prediction/question are head+tail truncated (middle "
                          "dropped) so prompt + generated output fit this budget — mirrors the "
                          "training-time truncation for train/inference parity. None -> no "
                          "truncation (rely on the model's context window)."}
    )
    verifier_num_cpu_threads: int = field(
        default=16,
        metadata={"help": "torch intra-op threads during huggingface-CPU verifier "
                          "inference, restored afterwards (LLM CPU inference is "
                          "memory-bandwidth bound; more threads past the saturation "
                          "point hurt). 0/negative -> leave unchanged. Default 16."}
    )
    verifier_verbose: bool = field(
        default=False,
        metadata={"help": "print each verifier row (query / reference / prediction / "
                          "rating / explanation), like Record.verbose. Debug aid."}
    )
    # NOTE: HF-engine-specific knobs are intentionally NOT exposed here:
    #   - HF cache       -> reuse global ``args.hf_hub_cache`` (HuggingfaceArgs).
    #   - torch_dtype    -> reuse global ``args.torch_dtype`` (InferenceArgs).
    #   - max_seq_len    -> hardcoded default inside JudgeEvaluator's HF dispatch
    #                       (added when the HF backend lands).
    #   - attn_impl      -> ditto, default ``"sdpa"`` inside the HF dispatch.
    #   - multimodal     -> reuse global ``args.use_audio_in_video`` + task yaml
    #                       video_dirpath; not exposed as a separate verifier flag.
    # Add knobs here only when there's a real reason for verifier to diverge
    # from the main inference path.

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        if not getattr(args, "enable_verifier", False):
            return args
        api_engines = {
            InferenceEngine.api__openai.value,
            InferenceEngine.api__anthropic.value,
            InferenceEngine.api__google.value,
        }
        valid_engines = api_engines | {
            InferenceEngine.huggingface.value,
            InferenceEngine.llama_cpp.value,
        }
        if args.verifier_engine not in valid_engines:
            raise ValueError(
                f"Invalid --verifier-engine: {args.verifier_engine!r}. "
                f"Valid values: {sorted(valid_engines)!r}"
            )
        # Ensure the model-id arg required by each engine is provided.
        # api/* -> verifier_api_name (mirrors ApiInferenceEngineArgs.api_name).
        # huggingface / llama_cpp -> verifier_model_name_or_path (mirrors
        # HuggingfaceInferenceEngineArgs.model_name_or_path).
        if args.verifier_engine in [
            InferenceEngine.huggingface.value,
            InferenceEngine.llama_cpp.value,
        ]:
            if not (isinstance(args.verifier_model_name_or_path, str)
                    and args.verifier_model_name_or_path.strip()):
                raise ValueError(
                    "--verifier-model-name-or-path is required when "
                    "--verifier-engine=huggingface"
                )
        else:  # api/*
            if not (isinstance(args.verifier_api_name, str)
                    and args.verifier_api_name.strip()):
                raise ValueError(
                    f"--verifier-api-name is required when "
                    f"--verifier-engine={args.verifier_engine}"
                )
        if args.verifier_max_new_tokens is not None and args.verifier_max_new_tokens <= 0:
            raise ValueError(f"--verifier-max-new-tokens must be > 0; got {args.verifier_max_new_tokens}")
        if args.verifier_temperature is not None and args.verifier_temperature < 0:
            raise ValueError(f"--verifier-temperature must be >= 0; got {args.verifier_temperature}")
        # Metric-namespacing alias (verifier_score/{alias}). Derive when unset — api_name for
        # api/*, else the checkpoint-path tail — then sanitize to a single safe key segment.
        _alias = getattr(args, "verifier_alias", None)
        if not (isinstance(_alias, str) and _alias.strip()):
            if args.verifier_engine in api_engines:
                _alias = args.verifier_api_name
            else:
                # compact alias from the checkpoint path: keep the last two components
                # (publisher/model or run/checkpoint) so a deep path collapses while the hub
                # publisher survives.
                _parts = [_p for _p in str(args.verifier_model_name_or_path).replace("\\", "/").split("/") if _p]
                _alias = "__".join(_parts[-2:])
        args.verifier_alias = sanitize_name(_alias or args.verifier_engine)
        return args


@dataclass(kw_only=True)
class HuggingfaceInferenceEngineArgs(ArgsInterface):
    model_name_or_path: str = field(
        metadata={"help": "Hugging Face pretrained model name or local path"}
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "device placement strategy for model loading (default: %(default)s)"}
    )
    trust_remote_code: Optional[bool] = field(
        default=None,
        metadata={"help": "trust remote code when loading model from Hugging Face Hub"}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={"help": "enable low CPU memory usage mode for from_pretrained"}
    )
    skip_chat_template: bool = field(
        default=False,
        metadata={"help": "skip apply_chat_template and use tokenization only"}
    )
    model_group: Optional[str] = field(
        default=None,
        metadata={"help": "override automatic model group detection; must be a valid HuggingfaceModelGroup value (e.g. 'hyperclovax_seed', 'qwen2_vl')"}
    )
    model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "additional kwargs passed to the huggingface Module __init__ (JSON string)"}
    )

    @classmethod
    def validate(cls, args):
        if not isinstance(args.hf_token, str) or len(args.hf_token) < 1:
            args.hf_token = os.getenv("HF_TOKEN", None)
        if not (
            (isinstance(args.hf_token, str) and len(args.hf_token) > 0)
            or os.getenv("HF_TOKEN", None)
            or os.getenv("HUGGINGFACE_TOKEN", None)
        ):
            raise ValueError(f'`hf_token` should be set for inference_engine: {args.inference_engine}')

        if not (isinstance(args.model_name_or_path, str) and len(args.model_name_or_path) > 0):
            raise ValueError(f"`model_name_or_path` should be given when using huggingface: {args.model_name_or_path}")

        # Normalize HF repo id that was accidentally prefixed with `/` (e.g. "/Qwen/Qwen2.5-Omni-3B").
        # Only strip when the path doesn't exist locally — preserves real local-path checkpoints.
        if (
            args.model_name_or_path.startswith("/")
            and not os.path.exists(args.model_name_or_path)
        ):
            _stripped = args.model_name_or_path.lstrip("/")
            logger.warning(
                f'`model_name_or_path` starts with "/" but does not exist locally; '
                f'treating as HF repo id: {args.model_name_or_path!r} -> {_stripped!r}'
            )
            args.model_name_or_path = _stripped

        if args.model_kwargs is None:
            args.model_kwargs = dict()
        elif isinstance(args.model_kwargs, str):
            args.model_kwargs = json.loads(args.model_kwargs)
        return args
    
@dataclass(kw_only=True)
class VllmInferenceEngineArgs(ArgsInterface):
    url: str = field(
        default=None,
        metadata={"help": "vLLM server URL for inference requests"}
    )
    vllm_api_version: str = field(
        default="v1",
        metadata={"help": "API version for vLLM server (e.g., 'v1')"}
    )
    vllm_model_name: str = field(
        default=None,
        metadata={"help": "model name to use for vLLM server requests"}
    )
    vllm_api_key: str = field(
        default=None,
        metadata={"help": "API key for vLLM server authentication (if --api-key was set on server)"}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "model name or path for local vllm.LLM inference"}
    )
    trust_remote_code: Optional[bool] = field(
        default=None,
        metadata={"help": "trust remote code when loading model from Hugging Face Hub"}
    )
    skip_chat_template: bool = field(
        default=False,
        metadata={"help": "skip apply_chat_template and use tokenization only"}
    )
    add_generation_prompt: bool = field(
        default=False,
        metadata={"help": "append generation prompt when applying chat template"}
    )
    chat_template_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string of extra kwargs passed to vLLM chat template (e.g., '{\"skip_reasoning\": false}')"}
    )
    mm_processor_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string of kwargs passed to vLLM multimodal processor (e.g., '{\"num_crops\": 4}')"}
    )
    allowed_local_media_path: Optional[str] = field(
        default=None,
        metadata={"help": "local directory path where media files reside; if valid, filepath format is allowed in preprocess_message"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Validate vLLM server URL or local model path, normalize protocol, and set API key env var.
        # Also parse chat_template_kwargs/mm_processor_kwargs JSON strings and apply reasoning defaults.
        # Args: args - parsed namespace with url, model_name_or_path, and vllm_api_key
        # Returns: validated args with url normalized and VLLM_API_KEY env var set
        if isinstance(args.allowed_local_media_path, str) and os.path.exists(args.allowed_local_media_path):
            pass  # valid path — keep as str
        else:
            args.allowed_local_media_path = None

        if isinstance(args.chat_template_kwargs, str):
            args.chat_template_kwargs = json.loads(args.chat_template_kwargs)
        if args.chat_template_kwargs is None:
            args.chat_template_kwargs = dict()
        if "skip_reasoning" not in args.chat_template_kwargs:
            args.chat_template_kwargs["skip_reasoning"] = not args.reasoning
        if "use_audio_in_video" not in args.chat_template_kwargs:
            args.chat_template_kwargs["use_audio_in_video"] = args.use_audio_in_video
        if args.add_generation_prompt and "add_generation_prompt" not in args.chat_template_kwargs:
            args.chat_template_kwargs["add_generation_prompt"] = True

        if isinstance(args.mm_processor_kwargs, str):
            args.mm_processor_kwargs = json.loads(args.mm_processor_kwargs)
        if args.mm_processor_kwargs is None:
            args.mm_processor_kwargs = dict()
        if "use_audio_in_video" not in args.mm_processor_kwargs:
            args.mm_processor_kwargs["use_audio_in_video"] = args.use_audio_in_video

        if not (isinstance(args.url, str) and len(args.url) > 0):
            raise ValueError(f"`url` should be given when using vllm: {args.url}")
        if isinstance(args.url, str):
            if (
                not args.url.startswith("http://")
                and not args.url.startswith("https://")
            ):
                args.url = f'http://{args.url}'
                logger.info(f'add protocol to given url: {args.url}')
        elif isinstance(args.model_name_or_path, str):
            os.environ["RANK"] = "0"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_ADDR"] = "node0"
            os.environ["MASTER_PORT"] = "23457"
            args.url = None
        else:
            raise ValueError(
                f'`model_name_or_path` or `url` should be given when using vllm: '
                f'model_name_or_path={args.model_name_or_path!r}, url={args.url!r}'
            )

        if (
            isinstance(args.vllm_api_key, str)
            and len(args.vllm_api_key.strip()) > 0
        ):
            os.environ["VLLM_API_KEY"] = args.vllm_api_key
        return args
    
@dataclass(kw_only=True)
class SglangInferenceEngineArgs(ArgsInterface):
    url: str = field(
        metadata={"help": "SGLang server URL for inference requests"}
    )

    @classmethod
    def validate(cls, args):
        if not (isinstance(args.url, str) and len(args.url) > 0):
            raise ValueError(f"`url` should be given when using sglang: {args.url}")
        if (
            not args.url.startswith("http://")
            and not args.url.startswith("https://")
        ):
            args.url = f'http://{args.url}'
            logger.info(f'add protocol to given url: {args.url}')
        return args

@dataclass(kw_only=True)
class ApiInferenceEngineArgs(ArgsInterface):
    api_name: str = field(
        metadata={"help": "API model name to evaluate (e.g., gpt-4o, claude-3-opus)"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Validate API model name and ensure the required API key is set for the selected provider.
        # Args: args - parsed namespace with api_name, inference_engine, and provider-specific API keys
        # Returns: validated args
        if not (isinstance(args.api_name, str) and len(args.api_name) > 0):
            raise ValueError(f"Api_name should be given when using api: {args.api_name}")

        if args.inference_engine == InferenceEngine.api__openai:
            if not (
                (isinstance(args.openai_api_key, str) and len(args.openai_api_key) > 0)
                or os.getenv("OPENAI_API_KEY", None)
            ):
                raise ValueError(f"Openai_api_key should be set when using api/openai")
        
        elif args.inference_engine == InferenceEngine.api__anthropic:
            if not (
                (isinstance(args.anthropic_api_key, str) and len(args.anthropic_api_key) > 0)
                or os.getenv("ANTHROPIC_API_KEY", None)
            ):
                raise ValueError(f"Anthropic_api_key should be set when using api/anthropic")
        
        elif args.inference_engine == InferenceEngine.api__google:
            if not (
                (isinstance(args.google_api_key, str) and len(args.google_api_key) > 0)
                or os.getenv("GOOGLE_API_KEY", None)
            ):
                raise ValueError(f"Google_api_key should be set when using api/google")
        return args
       
@dataclass(kw_only=True)
class BuiltinEvaluationEngineArgs(ArgsInterface):
    local_dirpath: str = field(
        default=None,
        metadata={"help": "local dataset directory path; None to download from remote storage"}
    )
    subtask_type: str = field(
        default=None,
        metadata={"help": "subtask type to override the predefined subtask in task config"}
    )
    num_fewshot: int = field(
        default=0,
        metadata={"help": "number of few-shot examples in prompt (default: %(default)s)"}
    )
    fewshot_image_max_size: int = field(
        default=224,
        metadata={"help": "max image dimension in pixels for few-shot examples (default: %(default)s)"}
    )
    do_cot: bool = field(
        default=False,
        metadata={"help": "[deprecated] add chain-of-thought task prompt"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Validate builtin engine args: check local_dirpath existence and num_fewshot/fewshot_image_max_size bounds.
        # Args: args - parsed namespace with local_dirpath, num_fewshot, and fewshot_image_max_size
        # Returns: validated args with local_dirpath set to None if path does not exist
        if (
            isinstance(args.local_dirpath, str)
            and not os.path.exists(args.local_dirpath)
        ): # given but not exists
            logger.warning(f'set `local_dirpath` null since given path not exist: {args.local_dirpath}')
            args.local_dirpath = None
            
        if not (args.local_dirpath is None or (isinstance(args.local_dirpath, str) and os.path.exists(args.local_dirpath))):
            raise FileNotFoundError(f'Specified `local_dirpath` not exists: {args.local_dirpath}')
        
        if not (isinstance(args.num_fewshot, int) and args.num_fewshot >= -1):
            raise ValueError(f'`num_fewshot` should be integer equal or greater than -1: {args.num_fewshot!r}')
        
        if not (
            isinstance(args.fewshot_image_max_size, int)
            and (
                args.fewshot_image_max_size == -1
                or args.fewshot_image_max_size > 10
            )
        ):
            raise ValueError(f'`fewshot_image_max_size` should be -1 or integer greater than 10: {args.fewshot_image_max_size!r}')
        return args

@dataclass(kw_only=True)
class LmmsEvalEvaluationEngineArgs(ArgsInterface):
    num_fewshot: int = field(
        default=-1,
        metadata={"help": "number of few-shot examples; -1 to use task default (default: %(default)s)"}
    )
    generation_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "generation kwargs as JSON string to override task defaults with highest priority (e.g. '{\"max_new_tokens\": 512, \"temperature\": 0.7}'); default: %(default)s"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Verify that HF_TOKEN, HF_HOME, and HF_HUB_CACHE are available for lmms-eval engine.
        # Args: args - parsed namespace with hf_token and hf_home
        # Returns: validated args
        if isinstance(args.generation_kwargs, str):
            _parsed = json.loads(args.generation_kwargs)
            args.generation_kwargs = _map_generation_kwargs(_parsed)
        if not isinstance(args.hf_token, str) or len(args.hf_token) < 1:
            args.hf_token = os.getenv("HF_TOKEN", None)
        if not (
            (isinstance(args.hf_token, str) and len(args.hf_token) > 0)
            or os.getenv("HF_TOKEN", None)
            or os.getenv("HUGGINGFACE_TOKEN", None)
        ):
            raise ValueError(f'`HF_TOKEN` or `HUGGINGFACE_TOKEN` should be set for evaluation_engine: {args.evaluation_engine}')

        if not isinstance(args.hf_home, str) or len(args.hf_home) < 1:
            args.hf_home = os.getenv("HF_HOME", None)
        if not (
            (isinstance(args.hf_home, str) and len(args.hf_home) > 0)
            or os.getenv("HF_HOME", None)
        ):
            raise ValueError(f'`HF_HOME` should be set for evaluation_engine: {args.evaluation_engine}')

        if not (
            os.getenv("HF_HUB_CACHE", None)
            or os.getenv("HUGGINGFACE_HUB_CACHE", None)
        ):
            raise ValueError(f'`HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE` should be set for evaluation_engine: {args.evaluation_engine}')
        return args

@dataclass(kw_only=True)
class LmEvalHarnessEvaluationEngineArgs(ArgsInterface):
    num_fewshot: int = field(
        default=None,
        metadata={"help": "number of few-shot examples; None to use task default"}
    )
    generation_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "generation kwargs as JSON string to override task defaults with highest priority (e.g. '{\"max_new_tokens\": 512, \"temperature\": 0.7}'); default: %(default)s"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Verify that HF_TOKEN, HF_HOME, HF_HUB_CACHE, and HF_ALLOW_CODE_EVAL are available for lm-eval-harness.
        # Args: args - parsed namespace with hf_token and hf_home
        # Returns: validated args
        if isinstance(args.generation_kwargs, str):
            _parsed = json.loads(args.generation_kwargs)
            args.generation_kwargs = _map_generation_kwargs(_parsed)
        if not isinstance(args.hf_token, str) or len(args.hf_token) < 1:
            args.hf_token = os.getenv("HF_TOKEN", None)
        if not (
            (isinstance(args.hf_token, str) and len(args.hf_token) > 0)
            or os.getenv("HF_TOKEN", None)
            or os.getenv("HUGGINGFACE_TOKEN", None)
        ):
            raise ValueError(f'`HF_TOKEN` or `HUGGINGFACE_TOKEN` should be set for evaluation_engine: {args.evaluation_engine}')

        if not isinstance(args.hf_home, str) or len(args.hf_home) < 1:
            args.hf_home = os.getenv("HF_HOME", None)
        if not (
            (isinstance(args.hf_home, str) and len(args.hf_home) > 0)
            or os.getenv("HF_HOME", None)
        ):
            raise ValueError(f'`HF_HOME` should be set for evaluation_engine: {args.evaluation_engine}')

        if not (
            os.getenv("HF_HUB_CACHE", None)
            or os.getenv("HUGGINGFACE_HUB_CACHE", None)
        ):
            raise ValueError(f'`HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE` should be set for evaluation_engine: {args.evaluation_engine}')

        if not os.getenv("HF_ALLOW_CODE_EVAL", None):
            raise ValueError(f'`HF_ALLOW_CODE_EVAL` should be set for evaluation_engine: {args.evaluation_engine}')

        return args

@dataclass(kw_only=True)
class VlmEvalKitEvaluationEngineArgs(ArgsInterface):
    config: str = field(
        default=None,
        metadata={"help": "path to the VLMEvalKit config JSON file"}
    )
    judge: str = field(
        default=None,
        metadata={"help": "judge model name for LLM-as-judge evaluation"}
    )
    judge_args: str = field(
        default=None,
        metadata={"help": "JSON-encoded arguments for the judge model"}
    )
    retry: int = field(
        default=None,
        metadata={"help": "max retries for judge API calls"}
    )
    api_nproc: int = field(
        default=None,
        metadata={"help": "number of parallel processes for judge API calls"}
    )

    @classmethod
    def validate(cls, args: argparse.Namespace) -> argparse.Namespace:
        # Verify HF credentials for VLMEvalKit and set LMUData env var from hf_home.
        # Args: args - parsed namespace with hf_token and hf_home
        # Returns: validated args with LMUData env var set
        if not isinstance(args.hf_token, str) or len(args.hf_token) < 1:
            args.hf_token = os.getenv("HF_TOKEN", None)
        if not (
            (isinstance(args.hf_token, str) and len(args.hf_token) > 0)
            or os.getenv("HF_TOKEN", None)
            or os.getenv("HUGGINGFACE_TOKEN", None)
        ):
            raise ValueError(f'`HF_TOKEN` or `HUGGINGFACE_TOKEN` should be set for evaluation_engine: {args.evaluation_engine}')

        if not isinstance(args.hf_home, str) or len(args.hf_home) < 1:
            args.hf_home = os.getenv("HF_HOME", None)
        if not (
            (isinstance(args.hf_home, str) and len(args.hf_home) > 0)
            or os.getenv("HF_HOME", None)
        ):
            raise ValueError(f'`HF_HOME` should be set for evaluation_engine: {args.evaluation_engine}')

        if not (
            os.getenv("HF_HUB_CACHE", None)
            or os.getenv("HUGGINGFACE_HUB_CACHE", None)
        ):
            raise ValueError(f'`HF_HUB_CACHE` or `HUGGINGFACE_HUB_CACHE` should be set for evaluation_engine: {args.evaluation_engine}')

        if (
            (isinstance(args.hf_home, str) and len(args.hf_home) > 0)
            or os.getenv("HF_HOME", None)
        ):
            os.environ["LMUData"] = args.hf_home or os.environ["HF_HOME"]
            logger.info(f'Set `LMUData`: {os.getenv("LMUData", None)}')
        return args
    
@dataclass(kw_only=True)
class T2IGeneratorArgs(ArgsInterface):
    t2i_generator_type: str = field(
        default=None,
        metadata={"help": "text-to-image generator type (e.g., ta_tok, hyperclova_vdm)"}
    )
    
@dataclass(kw_only=True)
class TaTokT2IGeneratorArgs(ArgsInterface):
    t2i_generator_model_path: str = field(
        default="csuhan/Tar-1.5B",
        metadata={"help": "model name or path for Ta-Tok image generation model"}
    )
    t2i_generator_ar_path: str = field(
        default="ar_dtok_lp_256px.pth",
        metadata={"help": "AR detokenizer checkpoint path for Ta-Tok"}
    )
    t2i_generator_encoder_path: str = field(
        default="ta_tok.pth",
        metadata={"help": "encoder checkpoint path for Ta-Tok"}
    )
    t2i_generator_decoder_path: str = field(
        default="vq_ds16_t2i.pt",
        metadata={"help": "decoder checkpoint path for Ta-Tok"}
    )
    t2i_generator_scale: int = field(
        default=0,
        metadata={"help": "generation scale for Ta-Tok (default: %(default)s)"}
    )
    t2i_generator_seq_len: int = field(
        default=729,
        metadata={"help": "sequence length for Ta-Tok generation (default: %(default)s)"}
    )
    t2i_generator_temperature: float = field(
        default=1.0,
        metadata={"help": "sampling temperature for image generation (default: %(default)s)"}
    )
    t2i_generator_top_p: float = field(
        default=0.95,
        metadata={"help": "top-p (nucleus) sampling threshold for image generation (default: %(default)s)"}
    )
    t2i_generator_top_k: int = field(
        default=1200,
        metadata={"help": "top-k filtering value for image generation (default: %(default)s)"}
    )
    t2i_generator_cfg_scale: float = field(
        default=4.0,
        metadata={"help": "classifier-free guidance scale for Ta-Tok (default: %(default)s)"}
    )
    t2i_generator_torch_dtype: str = field(
        default="float32",
        metadata={"help": "torch data type for image generation model (default: %(default)s)"}
    )
    
@dataclass(kw_only=True)
class VdmT2IGeneratorArgs(ArgsInterface):
    t2i_generator_model_path: str = field(
        default="",
        metadata={"help": "model name or path for the vision diffusion model (text-to-image)"}
    )
    t2i_generator_width: int = field(
        default=768,
        metadata={"help": "output image width in pixels for VDM (64-2048, default: %(default)s)"}
    )
    t2i_generator_height: int = field(
        default=768,
        metadata={"help": "output image height in pixels for VDM (64-2048, default: %(default)s)"}
    )
    t2i_generator_num_inference_steps: int = field(
        default=50,
        metadata={"help": "number of diffusion inference steps for VDM (1-200, default: %(default)s)"}
    )
   
@dataclass(kw_only=True)
class CvsClientArgs(ArgsInterface):
    cvs_host: str = field(
        default=None,
        metadata={"help": "CVS server host address (overrides CVS_HOST env var)"}
    )
    cvs_min_image_size: int = field(
        default=2,
        metadata={"help": "minimum allowed image dimension in pixels for CVS (default: %(default)s)"}
    )
    cvs_max_image_size: int = field(
        default=2240,
        metadata={"help": "maximum allowed image dimension in pixels for CVS (default: %(default)s)"}
    )

    @classmethod
    def validate(cls, args):
        if isinstance(args.cvs_host, str) and len(args.cvs_host) > 0:
            os.environ["CVS_HOST"] = args.cvs_host
        elif os.getenv("CVS_HOST"):
            args.cvs_host = os.environ["CVS_HOST"]
        return args

@dataclass(kw_only=True)
class ObsClientArgs(ArgsInterface):
    obs_bucket_name: str = field(
        default=None,
        metadata={"help": "Object Storage bucket name (overrides OBS_BUCKET_NAME env var)"}
    )
    obs_access_key: str = field(
        default=None,
        metadata={"help": "Object Storage access key (overrides OBS_ACCESS_KEY env var)"}
    )
    obs_secret_key: str = field(
        default=None,
        metadata={"help": "Object Storage secret key (overrides OBS_SECRET_KEY env var)"}
    )
    obs_region: str = field(
        default=None,
        metadata={"help": "Object Storage region (overrides OBS_REGION env var)"}
    )
    obs_endpoint: str = field(
        default=None,
        metadata={"help": "Object Storage endpoint URL (overrides OBS_ENDPOINT env var)"}
    )
    obs_host: str = field(
        default=None,
        metadata={"help": "Object Storage host address (overrides OBS_HOST env var)"}
    )
    obs_min_image_size: int = field(
        default=2,
        metadata={"help": "minimum allowed image dimension in pixels for Object Storage (default: %(default)s)"}
    )
    obs_max_image_size: int = field(
        default=2240,
        metadata={"help": "maximum allowed image dimension in pixels for Object Storage (default: %(default)s)"}
    )
    obs_image_extension: str = field(
        default="JPEG",
        metadata={"help": "image format extension for Object Storage uploads (default: %(default)s)"}
    )

    @classmethod
    def validate(cls, args):
        # Priority: CLI > env > hardcoded fallback (None = no fallback, stays None).
        for arg_name, env_name, fallback in (
            ("obs_bucket_name", "OBS_BUCKET_NAME", None),
            ("obs_access_key", "OBS_ACCESS_KEY", None),
            ("obs_secret_key", "OBS_SECRET_KEY", None),
            ("obs_region", "OBS_REGION", None),
            ("obs_endpoint", "OBS_ENDPOINT", None),
            ("obs_host", "OBS_HOST", None),
        ):
            value = getattr(args, arg_name, None)
            if not (isinstance(value, str) and len(value) > 0):
                value = os.getenv(env_name) or fallback
                setattr(args, arg_name, value)
            if isinstance(value, str) and len(value) > 0:
                os.environ[env_name] = value
        return args

@dataclass(kw_only=True)
class S3ClientArgs(ArgsInterface):
    s3_bucket_name: str = field(
        default=None,
        metadata={"help": "S3 Storage bucket name (overrides S3_BUCKET_NAME env var)"}
    )
    s3_access_key: str = field(
        default=None,
        metadata={"help": "S3 Storage access key (overrides S3_ACCESS_KEY env var)"}
    )
    s3_secret_key: str = field(
        default=None,
        metadata={"help": "S3 Storage secret key (overrides S3_SECRET_KEY env var)"}
    )
    s3_endpoint_url: str = field(
        default=None,
        metadata={"help": "S3 Storage endpoint URL (overrides S3_ENDPOINT_URL env var)"}
    )
    s3_region: str = field(
        default=None,
        metadata={"help": "S3 Storage region (overrides S3_REGION env var; fallback: kr-standard)"}
    )

    @classmethod
    def validate(cls, args):
        # Priority: CLI > env > hardcoded fallback (None = no fallback, stays None).
        for arg_name, env_name, fallback in (
            ("s3_bucket_name", "S3_BUCKET_NAME", None),
            ("s3_access_key", "S3_ACCESS_KEY", None),
            ("s3_secret_key", "S3_SECRET_KEY", None),
            ("s3_endpoint_url", "S3_ENDPOINT_URL", None),
            ("s3_region", "S3_REGION", "kr-standard"),
        ):
            value = getattr(args, arg_name, None)
            if not (isinstance(value, str) and len(value) > 0):
                value = os.getenv(env_name) or fallback
                setattr(args, arg_name, value)
            if isinstance(value, str) and len(value) > 0:
                os.environ[env_name] = value
        return args
        return args