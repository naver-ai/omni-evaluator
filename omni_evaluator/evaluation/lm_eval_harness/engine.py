# Reference from https://github.com/EleutherAI/lm-evaluation-harness (MIT)

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
from collections import defaultdict, OrderedDict
from functools import partial
import json
import logging
import numpy as np
from omegaconf import ListConfig, DictConfig
import os
from tqdm import tqdm
import traceback
from typing import cast, List, Tuple, Dict, Any, Optional, Union, Callable, Iterable, Iterator

logger = logging.getLogger(__name__)

from omni_evaluator import DatasetSource, EvaluationEngine, EvaluationMethod, Modality, SubtaskType
from omni_evaluator.evaluation.common import get_system_prompt
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
from omni_evaluator.postprocess import BinaryProcessor
from omni_evaluator.postprocess.multichoice import MultichoiceProcessor
from omni_evaluator.utils.data import find_field, format_task_prompt, extract_options
from omni_evaluator.utils.string import is_integer


DEFAULT_BENCHMARKS = [
    # general
    "mmlu_generative",
    "mmlu_pro",
    # general_ko
    "kmmlu_direct",
    "kmmlu_direct_hard",
    "kmmlu_cot_hard", 
    # instruction_following
    "ifeval", 
    # math
    "gsm8k",
    "minerva_math",
    "hendrycks_math", 
    # code
    "humaneval",
    "humaneval_instruct",
    "humaneval_plus",
    "mbpp",
    "mbpp_plus", 
    # reasoning
    "gpqa_main_generative_n_shot",
    "gpqa_diamond_generative_n_shot",
    "gpqa_extended_generative_n_shot",
    "boolq",
    # # reasoning_hard
    # "bigbench_generate_until",
    # "bbh",
    # ethics
    "truthfulqa_gen",
    "toxigen",
    "hendrycks_ethics", 
    # multilingual
    "mgsm_direct",
]

SYSTEM_PROMPT_MAP = {
    "humaneval": "code/python", 
    "humaneval_instruct": "code/python", 
    "humaneval_64": "code/python", 
    "humaneval_64_instruct": "code/python",
    "humaneval_plus": "code/python",
    "humaneval_x": "code/general", # not supported yet
    "mbpp": "code/python", 
    "mbpp_plus": "code/python",
}


def _build_task_config(
    task_name: str,
    task: Any,
    num_records: int,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
) -> TaskConfig:
    # Build a TaskConfig from an lm_eval_harness task object.
    # Args: task - lm_eval ConfigurableTask instance, num_records - total sample count
    # Returns: TaskConfig populated with task metadata, generation kwargs, and evaluation settings
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
        task_prompt = getattr(task.config, "doc_to_text", None)

    task_config = TaskConfig(
        task_name=task_name,
        evaluation_engine=EvaluationEngine.lm_eval_harness,
        num_records=num_records,
        meta=TaskMeta(
            benchmark_name=task.config["task"],
            split=None,
            lang=None,
            input_modality=[Modality.text, ],
            output_modality=[Modality.text, ],
            task_type=None,
            subtask_type=_subtask_type,
            num_fewshot=getattr(task.config, "num_fewshot", None),
        ),
        dataset=TaskDataset(
            source=DatasetSource.huggingface_hub,
            path=getattr(task.config, "dataset_path", None),
            split=getattr(task.config, "test_split", None),
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
    cache_requests: bool = True,
    rewrite_requests_cache: bool = False,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    trust_remote_code: bool = False,
    # args: build_task._adjust_config
    num_fewshot: Optional[int] = None,
    gen_kwargs: Union[str, dict, None] = None,
    predict_only: bool = False,
    fewshot_random_seed: int = 1234,
) -> Tuple[Dict[str, Any], List[Any]]:
    # Initialize lm_eval_harness task objects and build all inference requests.
    # Args: task_name - lm_eval task identifier, trust_remote_code - allow HF remote code execution
    # Returns: tuple of (task_dict mapping names to task objects, list of TaskOutput objects)
    from lm_eval.api.group import Group
    from lm_eval.api.task import Task
    from lm_eval.evaluator_utils import get_sample_size, _handle_back_comp
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.tasks.manager import TaskDict
    
    _NestedDict = dict[Group, dict[str, Task] | Group] | dict[str, Task]

    if trust_remote_code:
        import datasets
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    # lm_eval.evaluator.simple_evaluate
    if task_manager is None:
        task_manager = TaskManager(
            verbosity="INFO", # set logging-level DEBUG to detect anomaly
            # metadata=None, # {"max_seq_lengths": [4096, 8192], } 
        )
    task_dict = task_manager.load([task_name, ])

    # lm_eval.evaluator.evaluate
    if "tasks" not in task_dict:
        groups, eval_tasks = _handle_back_comp(cast("_NestedDict", task_dict))
    else:
        task_dict = cast("TaskDict", task_dict)
        groups, eval_tasks = task_dict.get("groups", {}), task_dict.get("tasks", {})
    
    # omni_evaluator
    for task_idx, (_task_name, _task) in tqdm(
        enumerate(eval_tasks.items()),
        initial=0,
        total=len(eval_tasks),
        desc=f'Buildng task requests: {task_name}'
    ):
        # Apply num_fewshot override (mirrors lm_eval.evaluator.simple_evaluate _adjust_config logic)
        if num_fewshot is not None:
            if _task.get_config("num_fewshot") == 0:
                logger.warning(
                    "num_fewshot has been set to 0 for %s in its config. Manual configuration will be ignored.",
                    _task_name,
                )
            else:
                if _task.get_config("num_fewshot") != num_fewshot:
                    logger.info(
                        "Overwriting default num_fewshot of %s from %s to %s",
                        _task_name, _task.get_config("num_fewshot"), num_fewshot,
                    )
                _task.set_config(key="num_fewshot", value=num_fewshot)
        if _task.get_config("num_fewshot") is None:
            _task.set_config(key="num_fewshot", value=0)

        limit = get_sample_size(_task, None)
        _task.build_all_requests(
            limit=limit,
            samples=None,
            rank=0,
            world_size=1,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=bool(apply_chat_template),
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=apply_chat_template, # default
            tokenizer_name="", # default # getattr(lm, "tokenizer_name", "") if apply_chat_template else "",
        )
        
    return task_dict, eval_tasks
        

def get_data_iterator(
    evaluation_engine: str,
    task_name: str,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    task_manager: Optional[Callable] = None,
    output_path: str = None,
    process_with_media: bool = False,
    cache_requests: bool = True,
    rewrite_requests_cache: bool = False,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    trust_remote_code: bool = False,
    # args: build_task._adjust_config
    num_fewshot: Optional[int] = None,
    gen_kwargs: Union[str, dict, None] = None,
    predict_only: bool = False,
    fewshot_random_seed: int = 1234,
    run_index: Optional[int] = 0,
    debug: bool = False,
    override_generation_kwargs: Optional[dict] = None,
) -> Tuple[List[Record], TaskConfig]:
    # Build Record list and TaskConfig by decomposing lm_eval_harness task/dataset pipeline.
    # Args: task_name - lm_eval benchmark identifier, debug - limit to NUM_DEBUG_SAMPLES if True
    # Returns: tuple of (list of Record objects ready for inference, TaskConfig)
    """
    # Decompose lm-eval-harness modules to get benchmark dataset
        - lm_eval.evaluator.simple_evaluate
        - lm_eval.evaluator.evaluate
    # Even if it compromises immediate readability,
        the original code should be preserved by avoiding line-level modifications
        to facilitate future patches
    """
    if not isinstance(system_prompt, str):
        system_prompt = get_system_prompt(task_name=task_name, system_prompt_map=SYSTEM_PROMPT_MAP)

    task_dict, eval_tasks = build_task(
        task_name=task_name,
        task_manager=task_manager,
        output_path=output_path,
        process_with_media=process_with_media,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        system_instruction=system_prompt,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        trust_remote_code=trust_remote_code,
        num_fewshot=num_fewshot,
        gen_kwargs=gen_kwargs,
        predict_only=predict_only,
        fewshot_random_seed=fewshot_random_seed,
    )
    
    # pack as an record array 
    # Fields such as doc_iterator and eval_docs are redundantly defined to store similar information, 
    # but in lmm_eval, this instance object is actually used for inference.
    records = list()
    for task_idx, (_task_name, _task) in enumerate(eval_tasks.items()):
        # lm_eval.api.task #L324-L356
        limit = NUM_DEBUG_SAMPLES if debug else None
        doc_id_docs = list(_task.doc_iterator(rank=0, limit=limit, world_size=1))
        for doc_id, doc in tqdm(
            doc_id_docs,
            total=len(doc_id_docs),
            desc=f'Collecting records: {evaluation_engine}/{task_name}',
        ):
            # sample fewshot context #TODO: need to offset doc_id by rank now!
            # Reference from https://github.com/EleutherAI/lm-evaluation-harness (MIT) - lm_eval/api/task.py (build_all_requests)
            fewshot_ctx = _task.fewshot_context(
                doc,
                0 if _task.config.num_fewshot is None else _task.config.num_fewshot,
                system_prompt,
                apply_chat_template,
                fewshot_as_multiturn,
                None, # chat_template,
                gen_prefix=_task.doc_to_prefix(doc),
            )

            # TODO: we should override self.config.repeats if doing greedy gen so users don't waste time+compute
            instance = _task.construct_requests(
                doc=doc,
                ctx=fewshot_ctx,
                metadata=(_task.config["task"], doc_id, _task.config.repeats),
                apply_chat_template=apply_chat_template,
                # chat_template=None, # chat_template,
            )
            
            # collect records
            if isinstance(instance, (list, tuple)):
                # multichoice
                _record = sample_to_record(
                    task_name=task_name,
                    doc_id=doc_id,
                    doc=doc,
                    instance=instance[0],
                    task=_task,
                    system_prompt=system_prompt,
                    task_prompt=task_prompt,
                    run_index=run_index,
                    override_generation_kwargs=override_generation_kwargs,
                )
                _options, _option_contents = list(), list()
                if "option" in doc:
                    for _option in doc["option"]:
                        _options_, _option_contents_ = extract_options(_option)
                        if _options_:
                            _options += _options_
                        if _option_contents_:
                            _option_contents += _option_contents_
                if not _option_contents:
                    _option_contents = _task.doc_to_choice(doc)
                if _options:
                    _record.options = _options
                if not _options or len(_options) == len(_option_contents):
                    _record._option_contents = _option_contents

                labels = _task.doc_to_target(doc)
                if isinstance(labels, (int, float, str, bool)):
                    labels = [labels, ]
                if labels:
                    pass
                elif doc.get("label", None):
                    labels.append(doc["label"])
                elif doc.get("answer", None):
                    _label_option, _label_content = extract_options(doc["answer"])
                    labels.append(doc["answer"])
                    if _label_option:
                        labels.append(_label_option[0])
                    if _label_content:
                        labels.append(_label_content[0])
                if labels:
                    _record.label = labels

                records.append(_record)

            else:
                # generate_*
                _record = sample_to_record(
                    task_name=task_name,
                    doc_id=doc_id,
                    doc=doc,
                    instance=instance,
                    task=_task,
                    system_prompt=system_prompt,
                    task_prompt=task_prompt,
                    run_index=run_index,
                    override_generation_kwargs=override_generation_kwargs,
                )
                records.append(_record)
        
    task_config = TaskConfig.from_lm_eval_harness(
        task_name=task_name, 
        task=_task,
        num_records=len(records),
        system_prompt=system_prompt, 
        task_prompt=task_prompt,
    )
    return records, task_config


def evaluate_task(
    evaluation_engine: str,
    task_name: str,
    task_config: Union[Dict[str, Any], TaskConfig],
    evaluation_method: str,
    records: List[Dict[str, Any]],
    task_manager: Optional[Callable] = None,
    output_path: str = None,
    process_with_media: bool = False,
    cache_requests: bool = True,
    rewrite_requests_cache: bool = False,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    trust_remote_code: bool = False,
    # args: build_task._adjust_config
    num_fewshot: Optional[int] = None,
    gen_kwargs: Union[str, dict, None] = None,
    predict_only: bool = False,
    fewshot_random_seed: int = 1234,
    # args: evaluate_task
    log_samples: bool = False,
    bootstrap_iters: int = 100000,
    debug: bool = False,
) -> Tuple[EvaluationRunOutput, List[Dict[str, Any]]]:
    # Inject predictions into lm_eval_harness instances and run evaluation with metric aggregation.
    # Args: records - inference result dicts, bootstrap_iters - CI bootstrap iterations
    # Returns: tuple of (EvaluationRunOutput with aggregated/group metrics, per-sample metric dicts)
    from lm_eval.utils import (
        create_iterator, handle_non_serializable, hash_string,
    )

    # dummy variables to run lmms-eval.evaluate
    RANK = 0
    WORLD_SIZE = 1
    limit = len(records) if isinstance(records, list) and len(records) > 0 else None

    task_dict, eval_tasks = build_task(
        task_name=task_name,
        task_manager=task_manager,
        output_path=output_path,
        process_with_media=process_with_media,
        cache_requests=cache_requests,
        rewrite_requests_cache=rewrite_requests_cache,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        trust_remote_code=trust_remote_code,
        num_fewshot=num_fewshot,
        gen_kwargs=gen_kwargs,
        predict_only=predict_only,
        fewshot_random_seed=fewshot_random_seed,
    )

    # validate length of records
    num_samples = sum([len(_task.eval_docs) for _task_name, _task in eval_tasks.items()])
    if not (debug or num_samples == len(records)):
        raise ValueError(f'Prediction length not match: {num_samples} vs. {len(records)}')
    
    # update task.instances.resps w/ inference_results
    _cur_instance_idx = 0
    _option_idx = 0
    for _task_idx, (_task_name, _task) in enumerate(eval_tasks.items()):
        for _instance_idx, _instance in enumerate(_task.instances):
            _doc_id = getattr(_instance, "doc_id", None)
            _prediction = "" # empty prediction as a default
            _perplexity = 100
            if _cur_instance_idx >= len(records): # abnormal case 1: records not exist
                if not debug:
                    raise AssertionError(f'records or its prediction not exists: {_task.task_name}/{_instance_idx}')
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
                # multiply -1 to convert negative probs
                eval_tasks[_task_name].instances[_instance_idx].resps = [(-1 * _perplexity, False), ]
            else:
                eval_tasks[_task_name].instances[_instance_idx].resps = [_prediction, ]

    # lm_eval.evaluator.evaluate #L601-L655
    sample_metrics = list()
    for _task_idx, (_task_name, task) in enumerate(eval_tasks.items()):
        task.apply_filters()
    
        # Collect values of metrics on all datapoints
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        # Reference from https://github.com/EleutherAI/lm-evaluation-harness (MIT) - lm_eval/evaluator.py (evaluate)
        instances_by_doc_id = defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        for filter_key in task.instances[0].filtered_resps:
            doc_iterator = task.doc_iterator(
                rank=RANK, limit=limit, world_size=WORLD_SIZE
            )
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]
                metrics = task.process_results(
                    doc, [req.filtered_resps[filter_key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [
                            req.filtered_resps[filter_key] for req in requests
                        ],
                        "filter": filter_key,
                        "metrics": list(metrics.keys()),
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

                sample_metrics.append({
                    metric: value
                    for metric, value in metrics.items()
                    if isinstance(value, (bool, int, float, np.floating))
                })

    # omni_evaluator
    # aggregate evaluation_output
    num_samples = task_config.num_records
    _num_valid_inferences = sum([
        isinstance(instance["prediction"], str) and len(instance["prediction"]) > 0
        for instance in records
    ])
    num_empty_predictions = len(records) - _num_valid_inferences
    coverage_inference = _num_valid_inferences / len(records)
    
    metrics = dict()
    metric_keys = list()
    group_metrics = dict()    
    # collect metrics and group_metrics
    for _record_idx, _record in enumerate(records):
        _sample_metrics = sample_metrics[_record_idx]
        for _metric_key, _metric_value in _sample_metrics.items():
            _metric_name = _metric_key
            if isinstance(_metric_key, (list, tuple)):
                _metric_name = ",".join(_metric_key)
            if _metric_name not in metric_keys:
                metric_keys.append(_metric_name)
            
            # collect metrics
            if _metric_name not in metrics:
                metrics[_metric_name] = list()
            metrics[_metric_name].append(_metric_value)
            
            # collect group_metrics
            _group_name = _record["meta"]["category"]
            if not _group_name:
                continue
            if _group_name not in group_metrics:
                group_metrics[_group_name] = dict()
            if _metric_name not in group_metrics[_group_name]:
                group_metrics[_group_name][_metric_name] = list()
            group_metrics[_group_name][_metric_name].append(_metric_value)
    
    # aggregate metrics
    for _metric_name, _metric_values in metrics.items():
        metrics[_metric_name] = np.nanmean(_metric_values)
    # aggregate group_metrics
    if group_metrics:
        for _group_name, _group_metrics in group_metrics.items():
            for _metric_name, _metric_values in _group_metrics.items():
                group_metrics[_group_name][_metric_name] = np.nanmean(_metric_values)

    evaluation_run_output = EvaluationRunOutput(
        inference_engine=None,
        evaluation_engine=evaluation_engine,
        task_name=task_name,
        evaluation_method=evaluation_method,
        num_samples=num_samples,
        num_empty_predictions=num_empty_predictions,
        coverage_inference=coverage_inference,
        coverage_evaluation=len(records) / num_samples,
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
    run_index: Optional[int] = 0,
    override_generation_kwargs: Optional[dict] = None,
    **kwargs,
) -> Record:
    # Convert an lm_eval_harness doc/instance pair into a Record for inference.
    # Args: doc - dataset document dict, instance - lm_eval Instance, task - lm_eval ConfigurableTask
    # Returns: Record with messages, labels, options, generation_options, and metadata
    """
    doc_id: int or str
    doc: doc_id_docs[0]
    instance: lm_eval.api.instance.Instance
    task: lm_eval.api.task.ConfigurableTask
    """
    # instance.arguments differ according to task.OUTPUT_TYPE: 
    #   - generate_until: (query, generation_options) # but answer may not be correct
    #   - multiple_choice: (query, answers) # but answer may not be correct
    
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
    
    # extract query
    query = instance.arguments[0]
    question = find_field(
        obj=doc, 
        candidate_keys=["question", "query", "problem", "question_0", "question_1", "question_2", "sentence", ], 
        default="", 
        ignore_values=["none", "None", "없음", ],
    )
    if (
        not isinstance(query, str) 
        or len(query.strip()) < 1 
        or (query == "None" or query == "없음") # videochatgpt has "None" or "없음" as a query
    ):
        query = question
    
    user_contents = list()
    query = format_task_prompt(
        task_prompt=task_prompt,
        query=query,
    )
    user_contents.append(ChatTextContent(**{
        "type": "text",
        "value": query,
    }))
    messages.append(ChatMessage(
        role="user",
        content=user_contents,
    ))
    
    # extract answers
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
    
    # sample_meta
    sample_meta = {
        "subtask_name": getattr(instance, "task_name", None),
        "repeats": instance.repeats,
        "request_type": instance.request_type,
        "num_fewshot": getattr(task.config, "num_fewshot", None),
        "question": question,
        **doc,
    }
    sample_meta["question_id"] = find_field(
        obj=doc, 
        candidate_keys=["question_id", "questionId",],
        default=None,
    )
    sample_meta["question_type"] = find_field(
        obj=doc, 
        candidate_keys=["question_type", "question_types"], 
        default=None,
    )
    sample_meta["answer_type"] = find_field(
        obj=doc, 
        candidate_keys=["answer_type", "answer_types"], 
        default=None,
    )
    sample_meta["image_id"] = find_field(
        obj=doc, 
        candidate_keys=["image_id", "imageId",],
        default=None,
    )

    # update category if not exists
    if not sample_meta.get("category", None):
        sample_meta["category"] = find_field(
            obj=doc, 
            candidate_keys=[
                "task", "domain", "subject",
                "subcategory", "sub_category", "subtask",
            ],
            default=None,
        )
    
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
        meta=sample_meta,
    )


def process_results_multichoice(
    self, 
    doc: dict, 
    results: Iterable[Any],
    doc_to_text: Optional[Callable],
    doc_to_target: Optional[Callable],
    doc_to_choice: Optional[Callable],
    metric_name: str = "acc",
) -> dict:
    """
    A custom function to override process_results 
    when forcibly changing a task originally not of the generate_until type (e.g., multiple_choice, loglikelihood)
    to use the generate_until request type.
    - doc_to_any: usually, "ConfiguralbeTask.doc_to_choice"
    """
    if not isinstance(metric_name, str): 
        metric_name = "acc"
        
    gold = None
    if isinstance(doc_to_target, Callable):
        gold = doc_to_target(doc)
    choices = doc_to_choice(doc)
    if isinstance(choices, (list, tuple)):
        if is_integer(x=gold) is not None:
            gold = is_integer(x=gold)
        if isinstance(gold, (int, np.integer)):
            gold = choices[gold]
        else:
            gold = find_field(
                obj=doc, 
                candidate_keys=["answer", "answers", "answer_key", "answerKey", "label",], 
                default=None,
            )
            if is_integer(x=gold) is not None:
                gold = is_integer(x=gold)
            if isinstance(gold, (int, np.integer)):
                gold = choices[gold]
    query = ",".join(choices)
    if isinstance(doc_to_text, Callable):
        query = doc_to_text(doc)
            
    score = 0.0
    for _result in results: # if not isinstance(_result, str): continue
        _query = query
        _gold = gold
        _option = None
        
        if BinaryProcessor.is_binary(query=_gold):
            _option = BinaryProcessor.extract(prediction=_result)
            _gold = BinaryProcessor.extract(prediction=_gold)
        else:
            _option = MultichoiceProcessor.extract(
                prediction=_result,
                query=_query,
            )
        if isinstance(_option, str) and isinstance(gold, str):
            _option = _option.lower() 
            _gold = gold.lower()
            if isinstance(_result, str) and _result.lower() in _gold:
                score = 1.0
                break
        if _option == _gold:
            score = 1.0
            break
        # TODO: Add functionality to select the single option among the choices 
        # that is most similar to the prediction, based on word overlap or API-based evaluation.
        # else:
        # e.g. _result = select_most_similar(_result, choices, method="jaccard") 
    result_dict = {metric_name: score}
    return result_dict