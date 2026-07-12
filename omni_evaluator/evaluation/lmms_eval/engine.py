# Reference from https://github.com/EvolvingLMMs-Lab/lmms-eval (Apache-2.0)
# This engine integrates and drives lmms-eval; the task/evaluation flow follows
# lmms-eval and it patches lmms-eval internals (see patches.py).

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

import argparse
import ast
from collections import defaultdict, OrderedDict
from functools import partial
import json
import logging
import numpy as np
from omegaconf import ListConfig, DictConfig
import os
import PIL
from tqdm import tqdm
import traceback
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Iterator

logger = logging.getLogger(__name__)

# Workaround for upstream lmms-eval bug in ``llm_judge/providers/openai.py``: its
# default ``OPENAI_API_URL`` is the malformed ``https://api.openai.com/v1/chat/completions/v1``,
# which the OpenAI SDK then re-suffixes with ``/chat/completions`` → 404. Set the
# correct base here so every downstream judge path (mathvista, etc.) resolves.
# ``setdefault`` respects an explicit user export (e.g. Azure / proxy endpoints).
os.environ.setdefault("OPENAI_API_URL", "https://api.openai.com/v1")

from omni_evaluator import DatasetSource, EvaluationEngine, EvaluationMethod, Modality, SubtaskType
from omni_evaluator.evaluation.common import get_system_prompt
from omni_evaluator.evaluation.lmms_eval.patches import ConfigurableTask__download
from omni_evaluator.evaluation.metrics.judge_evaluator import JudgeEvaluator
from omni_evaluator.inference import NUM_DEBUG_SAMPLES
from omni_evaluator.schemas.chat import (
    OcrToken, EntityToken,
    Message as ChatMessage, 
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
)
from omni_evaluator.schemas.inference import Record
from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from omni_evaluator.schemas.task import (
    TaskConfig, TaskMeta, 
    TaskPrompts, TaskDataset,
    TaskInference, TaskInferenceGenerationOptions,
    TaskEvaluation, TaskEvaluationPostprocess, TaskEvaluationJudge,
)
from omni_evaluator.utils.data import find_field, format_task_prompt, extract_options
from omni_evaluator.utils.string import is_integer
from omni_evaluator.utils.patches import ClassPatcher


DEFAULT_BENCHMARKS = [
    # "infovqa_test", # answer not included
    "ai2d",
    # "docvqa_test", # answer not included
    "chartqa",
    # "textvqa_test", # answer not included
    # "mmmu_test",  # set which shows high variance in usual before & after RLHF
    "llava_in_the_wild",
    "seedbench",
    "seedbench_2_plus",
    "mmstar",
    "scienceqa",
    "mme",
    "textvqa_val",
    "mmmu_val",
]

SYSTEM_PROMPT_MAP = {
} # empty


def _build_task_config(
    task_name: str,
    task: Any,
    num_records: int,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
) -> TaskConfig:
    # Build a TaskConfig from an lmms_eval ConfigurableTask object.
    # Args: task - lmms_eval ConfigurableTask instance, num_records - total sample count
    # Returns: TaskConfig populated with task metadata, prompts, and evaluation settings
    _subtask_type = None
    _evaluation_method = None
    if "generate_" in task.config["output_type"]:
        _subtask_type = SubtaskType.freeform
        _evaluation_method = EvaluationMethod.generation
    else:
        _subtask_type = SubtaskType.multiple_choice
        _evaluation_method = EvaluationMethod.perplexity

    _generation_kwargs = task.config["generation_kwargs"]
    if _generation_kwargs:
        _generation_kwargs = TaskInferenceGenerationOptions(
            max_new_tokens=_generation_kwargs.get("max_new_tokens", None),
            do_sample=_generation_kwargs.get("do_sample", None),
            temperature=_generation_kwargs.get("temperature", None),
            top_p=_generation_kwargs.get("top_p", None),
            top_k=_generation_kwargs.get("top_k", None),
            stop=_generation_kwargs.get("until", None),
        )

    if task_prompt is None:
        task_prompt = task.config.get("doc_to_text", None)

    task_config = TaskConfig(
        task_name=task_name,
        evaluation_engine=EvaluationEngine.lmms_eval,
        num_records=num_records,
        meta=TaskMeta(
            benchmark_name=task.config["task"],
            split=None,
            lang=None,
            input_modality=[Modality.text, ],
            output_modality=[Modality.text, ],
            task_type=None,
            subtask_type=_subtask_type,
            num_fewshot=task.config.get("num_fewshot", None),
        ),
        dataset=TaskDataset(
            source=DatasetSource.package,
            path=task.config.get("dataset_path", None),
            split=task.config.get("test_split", None),
        ),
        prompts=TaskPrompts(
            system_prompt=system_prompt,
            task_prompt=task_prompt,
        ),
        inference=TaskInference(
            generation_options=_generation_kwargs,
        ),
        evaluation=TaskEvaluation(
            method=_evaluation_method,
            target_metrics=[
                _metric["metric"].__name__
                if isinstance(_metric["metric"], Callable) else _metric["metric"]
                for _metric in task.config["metric_list"]
            ],
        ),
    )
    return task_config


def build_task(
    task_name: str,
    task_manager: Optional[Callable] = None,
    output_path: str = None,
    process_with_media: bool = False,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    # args: _adjust_config
    num_fewshot: Optional[int] = None,
    gen_kwargs: Union[str, dict, None] = None,
    predict_only: bool = False,
    fewshot_random_seed: int = 1234,
) -> Tuple[Dict[str, Any], List[Any]]:
    # Initialize lmms_eval task objects and build all inference requests.
    # Args: task_name - lmms_eval task identifier, num_fewshot - override for few-shot count
    # Returns: tuple of (task_dict mapping names to task objects, list of TaskOutput objects)
    from lmms_eval.tasks import TaskManager, get_task_dict
    from lmms_eval.api.task import ConfigurableTask
    from lmms_eval.evaluator_utils import get_task_list, get_sample_size
    
    # lmms_eval.evaluator.simple_evaluate
    if task_manager is None:
        task_manager = TaskManager(
            verbosity="INFO", # set logging-level DEBUG to detect anomaly
            model_name="omni_evaluator",
        )
    config = task_manager._get_config(task_name)
    output_type = config.get("output_type", "generate_until")
    task_type = "chat"
    if "generate_" not in output_type:
        task_type = "simple"
         
    patcher = ClassPatcher(ConfigurableTask)
    with patcher.patch_temporarily('download', ConfigurableTask__download):
        task_dict = get_task_dict(
            task_name, 
            task_manager,
            task_type=task_type,
        )
    
    # lmms_eval.evaluator.simple_evaluate #L305-354
    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    # Reference from https://github.com/EvolvingLMMs-Lab/lmms-eval (Apache-2.0) - lmms_eval/evaluator.py (simple_evaluate._adjust_config)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue

                if "generate_until" in task_obj.get_config("output_type"):
                    if gen_kwargs is not None:
                        task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

                if predict_only:
                    # eval_logger.info(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
                    print(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        # eval_logger.info(f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.")
                        print(f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.")
                    else:
                        # eval_logger.warning(f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}")
                        print(f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}")
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                # eval_logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict
    
    task_dict = _adjust_config(task_dict)
    
    # lmms_eval.evaluator.evaluate
    eval_tasks = get_task_list(task_dict)
    
    # omni_evaluator
    for task_idx, task_output in tqdm(
        enumerate(eval_tasks), initial=0, total=len(eval_tasks), desc=f'Buildng task requests: {task_name}'
    ):
        task_output.task.args = argparse.Namespace(**{
            "output_path": output_path,
            "process_with_media": process_with_media,
        }) # cli_args

        # lmms_eval.evaluator.evaluate
        task_output.task.build_all_requests(
            limit=None, # consider only 1 process
            rank=0, # consider only 1 process
            world_size=1, # consider only 1 process
            cache_requests=cache_requests, # default
            rewrite_requests_cache=rewrite_requests_cache, # default
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template, # default
            fewshot_as_multiturn=fewshot_as_multiturn, # default
            chat_template=None, # default # getattr(lm, "apply_chat_template") if apply_chat_template else None,
            tokenizer_name="", # default # getattr(lm, "tokenizer_name", "") if apply_chat_template else "",
        )
    return task_dict, eval_tasks
       

def get_data_iterator(
    evaluation_engine: str,
    task_name: str,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    task_manager: Optional[Callable] = None,
    output_path: str = None,
    process_with_media: bool = False,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    # args: build_task._adjust_config
    num_fewshot: Optional[int] = None,
    gen_kwargs: Union[str, dict, None] = None,
    predict_only: bool = False,
    fewshot_random_seed: int = 1234,
    run_index: Optional[int] = 0,
    override_generation_kwargs: Optional[dict] = None,
    debug: bool = False,
) -> Tuple[List[Record], TaskConfig]:
    # Build Record list and TaskConfig by decomposing lmms_eval task/dataset pipeline.
    # Args: task_name - lmms_eval benchmark identifier, num_ocr_tokens - OCR token limit per image
    # Returns: tuple of (list of Record objects ready for inference, TaskConfig)
    """
    # Decompose lmms modules to get benchmark dataset
        - lmms_eval.evaluator.simple_evaluate
        - lmms_eval.evaluator.evaluate
    # Even if it compromises immediate readability,
        the original code shoulde preserved by avoiding line-level modifications
        to facilitate future patches
    """ 
    from lmms_eval import utils
    
    # dummy variables to run lmms-eval.evaluate
    RANK = 0
    WORLD_SIZE = 1
    limit = NUM_DEBUG_SAMPLES if debug else None

    if not isinstance(system_prompt, str):
        system_prompt = get_system_prompt(task_name=task_name, system_prompt_map=SYSTEM_PROMPT_MAP)
    
    task_dict, task_outputs = build_task(
        task_name=task_name,
        task_manager=task_manager,
        output_path=output_path,
        process_with_media=process_with_media,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        system_instruction=system_prompt,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        num_fewshot=num_fewshot,
        gen_kwargs=gen_kwargs,
        predict_only=predict_only,
        fewshot_random_seed=fewshot_random_seed,
    )
    
    # pack as an record array 
    # Fields such as doc_iterator and eval_docs are redundantly defined to store similar information, 
    # but in lmm_eval, this instance object is actually used for inference.
    records = list()
    for task_idx, task_output in enumerate(task_outputs):
        task = task_output.task
        
        # lmms_eval.api.task #L400-407
        if task.has_test_docs():
            docs = task.test_docs()
            split = task.config.test_split
        elif task.has_validation_docs():
            docs = task.validation_docs()
            split = task.config.validation_split
        else:
            raise RuntimeError(f"Task dataset (path={task.DATASET_PATH}, name={task.DATASET_NAME}) must have valid or test docs!")
    
        # lmms_eval.api.task #L438-463
        # Reference from https://github.com/EvolvingLMMs-Lab/lmms-eval (Apache-2.0) - lmms_eval/api/task.py (build_all_requests)
        doc_id_docs = utils.create_iterator(
            enumerate(task.eval_docs),
            rank=RANK,
            limit=int(limit) if limit else None,
            world_size=WORLD_SIZE,
        )
        doc_iterator_for_counting = (
            utils.create_iterator(
                range(len(task.test_docs())),
                rank=RANK,
                limit=limit,
                world_size=WORLD_SIZE,
            )
            if task.has_test_docs()
            else utils.create_iterator(
                range(len(task.validation_docs())),
                rank=RANK,
                limit=limit,
                world_size=WORLD_SIZE,
            )
        )

        num_docs = sum(1 for _ in doc_iterator_for_counting)

        for doc_id, doc in tqdm(
            doc_id_docs,
            total=num_docs,
            desc=f'Collecting records: {evaluation_engine}/{task_name}',
        ):
            # sample fewshot context #TODO: need to offset doc_id by rank now!
            fewshot_ctx = task.fewshot_context(
                doc,
                0 if task.config.num_fewshot is None else task.config.num_fewshot,
                system_prompt,
                apply_chat_template,
                fewshot_as_multiturn,
                None, # chat_template,
            )

            # TODO: we should override self.config.repeats if doing greedy gen so users don't waste time+compute
            per_task_metadata = {"task": task.config["task"], "doc_id": doc_id, "repeats": task.config.repeats, "split": split}
            if task.config.metadata and type(task.config.metadata) == dict:  # TODO: temporary fix for metadata loading, ignore the list of dict type.
                per_task_metadata.update(task.config.metadata)

            instance = task.construct_requests(
                doc_id=doc_id, 
                ctx=fewshot_ctx, 
                metadata=per_task_metadata,
            )
            
            # collect records
            if isinstance(instance, (list, tuple)):
                # multichoice
                _record = sample_to_record(
                    task_name=task_name,
                    doc_id=doc_id,
                    doc=doc,
                    instance=instance[0],
                    task=task,
                    system_prompt=system_prompt,
                    task_prompt=task_prompt,
                    num_ocr_tokens=num_ocr_tokens,
                    num_subtitle_cues=num_subtitle_cues,
                    run_index=run_index,
                    override_generation_kwargs=override_generation_kwargs,
                )
                labels = list()
                _label_option, _label_content = extract_options(doc["answer"])
                labels.append(doc["answer"])
                if _label_option:
                    labels.append(_label_option[0])
                if _label_content:
                    labels.append(_label_content[0])
                _record.label = labels

                _options, _option_contents = list(), list()
                for _option in doc["option"]:
                    _options_, _option_contents_ = extract_options(_option)
                    _options += _options_
                    _option_contents += _option_contents_
                if _options:
                    _record.options = _options
                if len(_options) == len(_option_contents):
                    _record._option_contents = _option_contents
                records.append(_record)

            else:
                # generate_*
                _record = sample_to_record(
                    task_name=task_name,
                    doc_id=doc_id,
                    doc=doc,
                    instance=instance,
                    task=task,
                    system_prompt=system_prompt,
                    task_prompt=task_prompt,
                    num_ocr_tokens=num_ocr_tokens,
                    num_subtitle_cues=num_subtitle_cues,
                    run_index=run_index,
                    override_generation_kwargs=override_generation_kwargs,
                )
                records.append(_record)

    task_config = TaskConfig.from_lmms_eval(
        task_name=task_name, 
        task=task,
        num_records=len(records),
        system_prompt=system_prompt, 
        task_prompt=task_prompt,
    )
    return records, task_config


def evaluate_task(
    evaluation_engine: str,
    task_name: str,
    evaluation_method: str,
    task_config: Union[Dict[str, Any], TaskConfig],
    records: List[Dict[str, Any]],
    task_manager: Optional[Callable] = None,
    output_path: str = None,
    process_with_media: bool = False,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    # args: build_task._adjust_config
    num_fewshot: Optional[int] = None,
    gen_kwargs: Union[str, dict, None] = None,
    predict_only: bool = False,
    fewshot_random_seed: int = 1234,
    # args: evaluate_task
    log_samples: bool = False,
    bootstrap_iters: int = 100000,
    num_ocr_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    debug: bool = False,
) -> Tuple[EvaluationRunOutput, List[Dict[str, Any]]]:
    # Inject predictions into lmms_eval task instances and run full evaluation pipeline.
    # Args: records - inference result dicts with "prediction" keys, bootstrap_iters - CI bootstrap iterations
    # Returns: tuple of (EvaluationRunOutput with aggregated/group metrics, per-sample metric dicts)
    from lmms_eval.utils import (
        create_iterator, handle_non_serializable, hash_string,
    )
    from lmms_eval.evaluator_utils import (
        consolidate_group_results, consolidate_results, prepare_print_tasks,
    )
    
    # dummy variables to run lmms-eval.evaluate
    RANK = 0
    WORLD_SIZE = 1
    limit = None
    cli_args = argparse.Namespace(**{
        "output_path": output_path,
        "process_with_media": process_with_media,
    })
        
    task_dict, task_outputs = build_task(
        task_name=task_name,
        task_manager=task_manager,
        output_path=output_path,
        process_with_media=process_with_media,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        num_fewshot=num_fewshot,
        gen_kwargs=gen_kwargs,
        predict_only=predict_only,
        fewshot_random_seed=fewshot_random_seed,
    )
    
    num_samples = sum([len(task_output.task.eval_docs) for task_output in task_outputs])
    if not (debug or num_samples == len(records)):
        raise ValueError(f'Prediction length not match: {num_samples} vs. {len(records)}')
    
    # update task.instances.resps w/ inference_results
    _cur_instance_idx = 0
    _option_idx = 0
    for _task_idx, task_output in enumerate(task_outputs):
        for _instance_idx, _instance in enumerate(task_output.task.instances):
            _doc_id = getattr(_instance, "doc_id", None)
            _prediction = "" # empty prediction as a default
            _perplexity = 100
            if _cur_instance_idx >= len(records): # abnormal case 1: records not exist
                if not debug:
                    raise AssertionError(f'records or its prediction not exists: {task_output.task.task_name}/{_instance_idx}')
            elif not records[_cur_instance_idx]:
                _prediction = ""
                _option_idx += 1
            elif _doc_id == records[_cur_instance_idx]["index"]:
                _prediction = records[_cur_instance_idx]["prediction"]
                if records[_cur_instance_idx]["perplexities"]:
                    _perplexity = records[_cur_instance_idx]["perplexities"][_option_idx]
                if records[_cur_instance_idx]["prediction_postprocessed"]:
                    _prediction = records[_cur_instance_idx]["prediction_postprocessed"]
                if _prediction is None:
                    _prediction = ""
                _option_idx += 1
            else:
                _cur_instance_idx += 1
                _option_idx = 0
                if _cur_instance_idx >= len(records):
                    if not debug:
                        raise ValueError(f'num samples not match: {_cur_instance_idx} is greater than {len(records)}')
                else:
                    if not (_doc_id == records[_cur_instance_idx]["index"]):
                        if not debug:
                            raise ValueError(f'Doc_id not match: {_doc_id} vs. {records[_cur_instance_idx]["index"]}')
                    _prediction = records[_cur_instance_idx]["prediction"]
                    if records[_cur_instance_idx]["perplexities"]:
                        _perplexity = records[_cur_instance_idx]["perplexities"][_option_idx]
                    if records[_cur_instance_idx]["prediction_postprocessed"]:
                        _prediction = records[_cur_instance_idx]["prediction_postprocessed"]
                    if _prediction is None:
                        _prediction = ""
                    _option_idx += 1

            if _instance.request_type == "loglikelihood":
                task_outputs[_task_idx].task.instances[_instance_idx].resps = [(_perplexity, False), ]
            else:
                task_outputs[_task_idx].task.instances[_instance_idx].resps = [_prediction, ]

    # lmms_eval.evaluator.evaluate #L1129-L1280
    for _task_idx, task_output in enumerate(task_outputs):
        task = task_output.task
        task.apply_filters()
    
        # Collect values of metrics on all datapoints
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        # Reference from https://github.com/EvolvingLMMs-Lab/lmms-eval (Apache-2.0) - lmms_eval/evaluator.py (evaluate)
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps.keys():
            if not cli_args.process_with_media:
                doc_iterator = create_iterator(
                    enumerate(task.eval_docs_no_media),
                    rank=RANK,
                    limit=int(limit) if limit else None,
                    world_size=WORLD_SIZE,
                )
            else:
                doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE)
            doc_iterator_for_counting = (
                create_iterator(
                    range(len(task.test_docs())),
                    rank=RANK,
                    limit=limit,
                    world_size=WORLD_SIZE,
                )
                if task.has_test_docs()
                else create_iterator(
                    range(len(task.validation_docs())),
                    rank=RANK,
                    limit=limit,
                    world_size=WORLD_SIZE,
                )
            )
            total_docs = sum(1 for _ in doc_iterator_for_counting)
            if debug:
                total_docs = min(total_docs, NUM_DEBUG_SAMPLES)
            pbar = tqdm(total=total_docs, desc="Postprocessing", disable=(RANK != 0))
            for _doc_loop_idx, (doc_id, doc) in enumerate(doc_iterator):
                if debug and _doc_loop_idx >= NUM_DEBUG_SAMPLES:
                    break
                requests = instances_by_doc_id[doc_id]
                if "image" not in doc:
                    doc["image"] = None # to prevent lmms-eval error in hallusion_bench_image
                metrics = task.process_results(doc, [req.filtered_resps[filter_key] for req in requests])
                if log_samples:
                    target = task.doc_to_target(doc)
                    saved_doc = {}
                    for key, value in doc.items():
                        # If image is not in key
                        if "image" not in key:
                            # If audio is also not the value
                            if isinstance(value, dict) and "array" in value:
                                continue
                            else:
                                saved_doc[key] = value
                    filtered_arguments = []
                    for req in requests:
                        # check if req.args is a list of tuples, and each item in the list is a serializable object
                        for value in req.args:
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                filtered_arguments.append(value)
                            # else:
                            #     filtered_arguments.append(_handle_non_serializable(value))

                    example = {
                        "doc_id": doc_id,
                        "doc": saved_doc,
                        "target": target,
                        "arguments": filtered_arguments,
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [req.filtered_resps[filter_key] for req in requests],
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                        "prompt_hash": hash_string(requests[0].arguments[0]),
                        "target_hash": hash_string(str(target)),
                    }
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)
                pbar.update(1)

            pbar.close()

    # lmms_eval.evaluator.evaluate #L1384-L1401
    for task_output in task_outputs:
        try:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
        except Exception as e:
            logger.warning(
                f"calculate_aggregate_metric failed for task '{task_output.task_name}': {e}. "
                f"sample_metrics keys: {list(task_output.sample_metrics.keys())}"
            )
    (
        results,
        samples,
        configs,
        versions,
        num_fewshot,
        higher_is_better,
    ) = consolidate_results(task_outputs)
        
    # Calculate group metrics
    if bool(results):
        results, versions, show_group_table, *_ = consolidate_group_results(results, versions, task_dict)

    results_agg, group_agg = prepare_print_tasks(task_dict, results)

    # aggregate evaluation_output
    num_samples = task_config.num_records
    _num_valid_inferences = sum([
        isinstance(instance["prediction"], str) and len(instance["prediction"]) > 0
        for instance in records
    ])
    num_empty_predictions = len(records) - _num_valid_inferences
    coverage_inference = _num_valid_inferences / len(records) if len(records) > 0 else 0.0
    
    metrics = dict()
    metric_keys = list()
    group_metrics = dict()    
    # collect group_metrics
    for _metric_key, _metric_values in task_output.sample_metrics.items():
        _metric_name = _metric_key
        if isinstance(_metric_key, (list, tuple)):
            _metric_name = ",".join(_metric_key)
        for _record_idx, (_record, _metric_value) in enumerate(zip(records, _metric_values)):
            _group_name = _record["meta"]["category"]
            if not _group_name:
                continue
            if _group_name not in group_metrics:
                group_metrics[_group_name] = dict()
            if _metric_name not in group_metrics[_group_name]:
                group_metrics[_group_name][_metric_name] = list()
            if not isinstance(_metric_value, (int, float, np.floating)):
                continue
            group_metrics[_group_name][_metric_name].append(_metric_value)
    
    # aggregate group_metrics
    if group_metrics:
        for _group_name, _gropu_metrics in group_metrics.items():
            for _metric_name, _metric_values in _gropu_metrics.items():
                group_metrics[_group_name][_metric_name] = np.nanmean(_metric_values)
    
    # overwrite if group_metric has been computed in lm_eval_harness
    for _group_idx, (group_name, group_metric) in enumerate(results.items()):
        for _idx, (k, v) in enumerate(group_metric.items()):
            if k in ["alias", "samples", ]:
                continue
            elif not isinstance(v, (int, float, np.floating)):
                continue
            metric_name, filter_key = k.split(",")[:2]
            if isinstance(filter_key, str) and filter_key != "none":
                metric_name = f'{metric_name}/{filter_key}'
            if group_name == task_name:
                metrics[metric_name] = v
                if (
                    len(metric_keys) == 0
                    and "stderr" not in metric_name
                ):
                    metric_keys.append(metric_name)
            else:
                if group_name not in group_metrics:
                    group_metrics[group_name] = dict()
                group_metrics[group_name][metric_name] = v
    
    # average group_metrics if main task not exists
    if (
        len(metric_keys) < 1 
        and len(group_metrics) > 0
    ):
        _average_metrics = defaultdict(list)
        for _group_name, _group_metric in group_metrics.items():
            for _k, _v in _group_metric.items():
                _average_metrics[_k].append(_v)
        for _i, (_k, _v) in enumerate(_average_metrics.items()):
            metrics[_k] = np.mean(_v)
            if _i == 0:
                metric_keys.append(_k)

    # TODO: update sample-wise metrics
    sample_metrics = list()
    for _record_idx in range(0, len(records)):
        _metrics = dict()
        sample_metrics.append(_metrics)

    evaluation_run_output = EvaluationRunOutput(
        inference_engine=None,
        evaluation_engine=evaluation_engine,
        task_name=task_name,
        evaluation_method=evaluation_method,
        num_samples=num_samples,
        num_empty_predictions=num_empty_predictions,
        coverage_inference=coverage_inference,
        coverage_evaluation=len(records) / num_samples if num_samples > 0 else 0.0,
        runtime_inference=None,
        runtime_evaluation=None,
        metric_keys=metric_keys,
        metrics=metrics,
        group_metrics=group_metrics if len(group_metrics) > 0 else None,
        sample_metrics=sample_metrics,
    )
    return evaluation_run_output, sample_metrics


def sample_to_record(
    task_name: str,
    doc_id: Union[int, str],
    doc: Dict[str, Any],
    instance: Any,
    task: Any,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    run_index: Optional[int] = 0,
    override_generation_kwargs: Optional[dict] = None,
    **kwargs,
) -> Record:
    # Convert an lmms_eval doc/instance pair into a Record for inference.
    # Args: doc - dataset document dict, instance - lmms_eval Instance with query/visuals, task - ConfigurableTask
    # Returns: Record with messages, labels, options, and metadata extracted from the lmms_eval objects
    """
    doc_id: int or str
    doc: doc_id_docs[0]
    instance: lmms_eval.api.instance.Instance
    task: lmms_eval.api.task.ConfigurableTask
    """
    
    messages: List[ChatMessage] = list()
    label: str = None
    
    # extract query
    query = instance.arguments[0]
    question = find_field(
        obj=doc, 
        candidate_keys=["question", "query", "problem", "question_0", "question_1", "question_2", "sentence", ], 
        default="", 
        ignore_values=["none", "None", "없음", ],
    )
    
    if isinstance(getattr(task, "doc_to_messages", None), Callable):
        # ConfigurableMessagesTask
        _messages = task.doc_to_messages(doc)
        for _message_idx, _message in enumerate(_messages):
            _contents = list()
            if isinstance(_message["content"], str):
                _message["content"] = {
                    "type": "text", 
                    "text": _message["content"],
                }
            if isinstance(_message["content"], dict):
                _message["content"] = [_message["content"], ]
            for _content_idx, _content in enumerate(_message["content"]):
                if "audio" in _content["type"]:
                    _content_value = ChatAudioContent.get_value(_content)
                    _contents.append(ChatTextContent(
                        type="audio", 
                        value=_content_value,
                    ))
                elif "image" in _content["type"]:
                    _content_value = ChatImageContent.get_value(_content)
                    _contents.append(ChatTextContent(
                        type="image", 
                        value=_content_value,
                    ))
                elif "video" in _content["type"]:
                    _content_value = ChatVideoContent.get_value(_content)
                    _contents.append(ChatTextContent(
                        type="video", 
                        value=_content_value,
                    ))
                else:
                    _content_value = ChatTextContent.get_value(_content)
                    _contents.append(ChatTextContent(
                        type="text",
                        value=_content_value,
                    ))
                    query = _content_value
            
            if _message["role"] in [
                "system",
            ]:
                _message = ChatMessage(
                    role="system",
                    content=_contents,
                )
                messages.append(_message)
            elif _message["role"] in [
                "user",
                "human",
            ]:
                _message = ChatMessage(
                    role="user",
                    content=_contents,
                )
                messages.append(_message)
            elif _message["role"] in [
                "assistant",
                "agent",
            ]:
                _message = ChatMessage(
                    role="assistant",
                    content=_contents,
                )
                messages.append(_message)
    
    else:
        # ConfigurableTask
    
        # instance.arguments:
        #   - (query, generation_options, doc_to_visual, doc_id, task_name, task_split)

        if (
            not isinstance(query, str) 
            or len(query.strip()) < 1 
            or (query == "None" or query == "없음") # videochatgpt has "None" or "없음" as a query
        ):
            query = question
        query = format_task_prompt(
            task_prompt=task_prompt,
            query=query,
        )
        
        # extract multimodal_items
        images = None
        image_urls = find_field(
            obj=doc, 
            candidate_keys=["image_url", "img_url", "imgUrl", "flickr_original_url", "source", ], 
            default=None,
        )
        if isinstance(image_urls, str):
            image_urls = [image_urls, ]
        videos = None
        
        ocr_tokens = None
        _ocr_tokens = find_field(
            obj=doc, 
            candidate_keys=["ocr", "ocr_tokens"], 
            default=None,
        )
        if isinstance(_ocr_tokens, (tuple, list)):
            ocr_tokens = list()
            for _token_idx, _ocr_token in enumerate(_ocr_tokens):
                if isinstance(_ocr_token, str):
                    _ocr_token = OcrToken(
                        id=_token_idx,
                        text=_ocr_token,
                        bbox=None,
                        confidence=None,
                    ).to_dict()
                elif isinstance(_ocr_token, dict):
                    _ocr_token = OcrToken(**_ocr_token).to_dict()
                else: 
                    raise ValueError(f'invalid ocr_token: {_ocr_token}')
                ocr_tokens.append(_ocr_token)
        
        try:
            images: List[PIL.Image.Image] = list()
            videos: List[str] = list()
            _doc_to_visual = instance.arguments[2]
            visual = _doc_to_visual(doc)
            for _visual in visual:
                if isinstance(_visual, PIL.Image.Image):
                    images.append(_visual)
                elif isinstance(_visual, str): 
                    if (
                        _visual.endswith(".mp4")
                        or _visual.endswith(".mkv")
                        or _visual.endswith(".webm")
                        or _visual.endswith(".mov")
                        or _visual.endswith(".avi")
                    ): # video_path
                        videos.append(_visual)
                    else:
                        images.append(_visual) # image_path or image_url
                else:
                    raise ValueError(f'invalid visual type: {_visual}')
        except Exception as ex:
            pass # if task has no field related to image
        if images is None and image_urls is not None:
            images = image_urls

        # construct message
        messages = list()
        if isinstance(system_prompt, str):
            # add system message if not given
            messages.append(ChatMessage(
                role="system", 
                content=[
                    ChatTextContent(type="text", value=system_prompt),
                ],
            ))
        
        user_contents = list()
        if isinstance(images, (list, tuple)):
            for _image_idx, _image in enumerate(images):
                _content = {
                    "type": "image",
                    "value": _image,
                }
                if (
                    _image_idx == 0
                    and isinstance(ocr_tokens, (list, tuple, dict))
                    and (
                        num_ocr_tokens is None # None inidicates all
                        or num_ocr_tokens > 0
                    )
                ):
                    if isinstance(num_ocr_tokens, int):
                        ocr_tokens = ocr_tokens[:num_ocr_tokens]
                    _content["value"] = {
                        "image": _image,
                        "ocr": ocr_tokens,
                    }
                user_contents.append(ChatImageContent(**_content))
        if isinstance(videos, (list, tuple)):
            for _video_idx, _video in enumerate(videos):
                _content = {
                    "type": "video",
                    "value": _video,
                }
                user_contents.append(ChatVideoContent(**_content))

        user_contents.append(ChatTextContent(**{
            "type": "text",
            "value": query,
        }))
        messages.append(ChatMessage(
            role="user",
            content=user_contents,
        ))
        
        # extract answers
        if task.config.output_type == "multiple_choice":
            label = instance.arguments[1]
            
    # extract answers
    if label is None:
        label = find_field(
            obj=doc, 
            candidate_keys=[
                "answer", "answers", "answer_key", "answerKey", "answer_index", 
                "label", "target", "correct",  "correct_answers", "best_answer",
            ], 
            default=None,
        )
    if label is None:
        label = task.doc_to_target(doc)
    
    options = find_field(
        obj=doc, 
        candidate_keys=["options", "choices"], 
        default=None,
    )
    if isinstance(options, str):
        try: 
            options = ast.literal_eval(options)
        except Exception as ex:
            pass
    
    option_contents = None
    if (
        options is None 
        and isinstance(getattr(task, "doc_to_choice", None), Callable)
        and task.config.doc_to_choice is not None
    ):
        options = task.doc_to_choice(doc)
    if isinstance(options, (dict, DictConfig)):
        if isinstance(options.get("text", None), (list, tuple, ListConfig)):
            option_contents = options["text"]
        if isinstance(options.get("label", None), (list, tuple, ListConfig)):
            options = options["label"]

    # NOTE: We used to remap ``label = options[label]`` when label looked
    # like an int index, but that misfires on mixed MCQ/open-ended datasets
    # (MMMU): a numeric open-ended answer "0" is is_integer + int-castable
    # but is NOT an option index — remapping turns "0" into ``options[0]``
    # ("A"), which is wrong. Keep the dataset's raw ``label`` on the record;
    # downstream consumers (postprocess parsers, judge/verifier prompt
    # builders) receive ``options`` + ``option_contents`` separately and can
    # resolve any letter↔content mapping themselves.


    # extract generation_options
    if isinstance(override_generation_kwargs, dict):
        generation_options = override_generation_kwargs
    elif isinstance(getattr(task.config, "generation_kwargs", None), (dict, DictConfig)):
        generation_options = dict()
        for k, v in task.config.generation_kwargs.items():
            if k == "until":
                if isinstance(v, str): v = [v, ]
                generation_options["stop_words"] = v
            elif k in ["max_gen_toks", ]:
                generation_options["max_new_tokens"] = v
            else:
                generation_options[k] = v
    else:
        generation_options = None

    return Record(
        benchmark=task_name,
        index=doc_id,
        prompt=None,
        messages=messages,
        generation_options=generation_options,
        label=label,
        options=options,
        option_contents=option_contents,
        prediction=None,
        latency=None,
        metrics=None,
        meta={
            "subtask_name": getattr(instance, "task_name", None),
            "repeats": instance.repeats,
            "request_type": instance.request_type,
            "num_fewshot": getattr(task.config, "num_fewshot", None),
            "question": question,
            "category": doc.get("category", None),
            "subcategory": doc.get("subcategory", None),
            "task": doc.get("task", None),
            "subject": doc.get("subject", None),
            "topic": doc.get("topic", None),
            "grade": doc.get("grade", None),
            "explanation": doc.get("explanation", None),
            "subfield": doc.get("subfield", None),
            "type": doc.get("type", None),
            "question_id": find_field(
                obj=doc, 
                candidate_keys=["question_id", "questionId",],
                default=None,
            ),
            "question_type": find_field(
                obj=doc, 
                candidate_keys=["question_type", "question_types"], 
                default=None,
            ),
            "answer_type": find_field(
                obj=doc, 
                candidate_keys=["answer_type", "answer_types"], 
                default=None,
            ),
            "image_id": find_field(
                obj=doc, 
                candidate_keys=["image_id", "imageId",],
                default=None,
            ),
            "image_type": doc.get("img_type", None),
            "image_classes": doc.get("image_classes", None),
        },
    )