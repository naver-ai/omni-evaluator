# Reference from https://github.com/open-compass/VLMEvalKit (Apache-2.0)

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

from collections import defaultdict, OrderedDict
import copy
from functools import partial
import inspect
import json
import logging
import numpy as np
from omegaconf import ListConfig, DictConfig
import os
import pandas as pd
import PIL
from tqdm import tqdm
import traceback
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Iterable

from omni_evaluator import DatasetSource, EvaluationEngine, EvaluationMethod, Modality, SubtaskType
from omni_evaluator.evaluation.common import get_system_prompt
from omni_evaluator.evaluation.metrics.judge_evaluator import JudgeEvaluator
from omni_evaluator.inference import NUM_DEBUG_SAMPLES
from omni_evaluator.schemas.inference import Record
from omni_evaluator.schemas.evaluation import EvaluationRunOutput
from omni_evaluator.schemas.task import (
    TaskConfig, TaskMeta, 
    TaskPrompts, TaskDataset,
    TaskInference, TaskInferenceGenerationOptions,
    TaskEvaluation, TaskEvaluationPostprocess, TaskEvaluationJudge,
)
from omni_evaluator.schemas.chat import (
    OcrToken, EntityToken,
    Message as ChatMessage,
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
    CONTENT_ACCESSOR_MAP,
)
from omni_evaluator.utils.data import find_field, format_task_prompt
from omni_evaluator.utils.string import is_url, is_integer, is_numeric, parse_string
from omni_evaluator.utils.io import read_file, write_file, get_temp_filepath

logger = logging.getLogger(__name__)


DEFAULT_BENCHMARKS = [
    # infovqa_test
    "AI2D_TEST",
    # docvqa_test
    # chartqa
    # mmmu_test
    # llavaw
    # seedbench
    # seedbench2_plus
    # mmstar
    # scienceqa
    # "mme"
    "TextVQA_VAL",
    "MMMU_DEV_VAL",
]

SYSTEM_PROMPT_MAP = {
} # empty


def _build_task_config(
    task_name: str,
    dataset: Any,
    num_records: int,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
) -> TaskConfig:
    # Build a TaskConfig from a VLMEvalKit dataset object.
    # Args: dataset - VLMEvalKit dataset with TYPE and MODALITY attributes, num_records - total sample count
    # Returns: TaskConfig populated with modality, evaluation method, and prompt settings
    _subtask_type = None
    _evaluation_method = None
    if dataset.TYPE in ["MCQA", "MCQ", ]:
        _subtask_type = SubtaskType.multiple_choice
        _evaluation_method = EvaluationMethod.perplexity
    else:
        _subtask_type = SubtaskType.freeform
        _evaluation_method = EvaluationMethod.generation
    _generation_kwargs = None
    if _generation_kwargs:
        _generation_kwargs = TaskInferenceGenerationOptions(
            max_new_tokens=_generation_kwargs.get("max_new_tokens", None),
            do_sample=_generation_kwargs.get("do_sample", None),
            temperature=_generation_kwargs.get("temperature", None),
            top_p=_generation_kwargs.get("top_p", None),
            top_k=_generation_kwargs.get("top_k", None),
            stop=_generation_kwargs.get("until", None),
        )

    input_modality = [Modality.text, ]
    if dataset.MODALITY.lower() == Modality.audio:
        input_modality = [Modality.audio, ]
    elif dataset.MODALITY.lower() == Modality.image:
        input_modality = [Modality.image, ]
    elif dataset.MODALITY.lower() == Modality.video:
        input_modality = [Modality.video, ]

    task_config = TaskConfig(
        task_name=task_name,
        evaluation_engine=EvaluationEngine.vlm_eval_kit,
        num_records=num_records,
        meta=TaskMeta(
            benchmark_name=dataset.dataset_name,
            split=None,
            lang=None,
            input_modality=input_modality,
            output_modality=[Modality.text, ],
            task_type=None,
            subtask_type=_subtask_type,
            num_fewshot=None,
        ),
        dataset=TaskDataset(
            source=DatasetSource.package,
            path=None,
            split=None,
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
            target_metrics=None,
        ),
    )
    return task_config


def build_dataset(
    dataset_name: str,
    model_name: Optional[str] = None,
    config: Optional[str] = None,
    fps: Optional[float] = None,
    nframe: Optional[int] = None,
):
    """
    # Decompose VLMEvalKit modules to get benchmark dataset
        - build_dataset
    # Even if it compromises immediate readability,
        the original code shoulde preserved by avoiding line-level modifications
        to facilitate future patches
    """
    import vlmeval.dataset
    from vlmeval.dataset import build_dataset as vlm_eval_kit_build_dataset
    
    cfg_data = None
    if isinstance(config, str):
        cfg = read_file(config)
        cfg_data = cfg['data']
        dataset_name = list(cfg_data.keys())[0]
    
    dataset = None
    if cfg_data is not None:
        dataset = build_dataset_from_config(cfg['data'], dataset_name)
    else:
        dataset_kwargs = dict()
        if dataset_name in [
            "MEGABench", 
        ]:
            dataset_kwargs["fps"] = fps
            dataset_kwargs["nframe"] = nframe
        if dataset_name in [
            'MMLongBench_DOC', 'DUDE', 'DUDE_MINI', 'SLIDEVQA', 'SLIDEVQA_MINI',
        ]:
            dataset_kwargs['model'] = model_name
        dataset = vlm_eval_kit_build_dataset(dataset_name, **dataset_kwargs)
    return dataset
        

def get_data_iterator(
    evaluation_engine: str,
    task_name: str,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    model_name: Optional[str] = None,
    config: Optional[str] = None,
    fps: Optional[float] = None,
    nframe: Optional[int] = None,
    run_index: Optional[int] = 0,
    debug: bool = False,
) -> Tuple[List[Record], TaskConfig]:
    # Build Record list and TaskConfig by decomposing VLMEvalKit dataset pipeline.
    # Args: task_name - VLMEvalKit dataset name, config - optional YAML config path for custom datasets
    # Returns: tuple of (list of Record objects ready for inference, TaskConfig)
    """
    # Decompose VLMEvalKit modules to get benchmark dataset
        - build_dataset
        - lm_eval.evaluator.evaluate
    # Even if it compromises immediate readability,
        the original code shoulde preserved by avoiding line-level modifications
        to facilitate future patches
    """ 
    
    if not isinstance(system_prompt, str):
        system_prompt = get_system_prompt(task_name=task_name, system_prompt_map=SYSTEM_PROMPT_MAP)
    
    dataset = build_dataset(
        dataset_name=task_name,
        model_name=model_name,
        config=config,
        fps=fps,
        nframe=nframe,
    )
    
    records = list()
    _function_kwargs = list()
    if hasattr(dataset, "build_prompt"):
        _function_kwargs = list(inspect.signature(dataset.build_prompt).parameters.keys())
    num_records = len(dataset.data)
    for _idx in tqdm(
        range(0, num_records),
        initial=0, 
        total=num_records,
        desc=f'Collecting records: {evaluation_engine}/{task_name}',
    ):
        if debug and _idx >= NUM_DEBUG_SAMPLES:
            break
        _row = dataset.data.iloc[_idx].to_dict()
        _kwargs = dict()
        if "video_llm" in _function_kwargs: # some dataset requires video_llm
            _kwargs["video_llm"] = True # if False, return image frames instead of video
        _user_content = dataset.build_prompt(
            dataset.data.iloc[_idx],
            **_kwargs,
        ) # list of dict, content of user turn (some dataset may return assistant turn)

        # collect records
        _record = sample_to_record(
            dataset_name=task_name,
            row=_row,
            user_content=_user_content,
            dataset=dataset,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            num_ocr_tokens=num_ocr_tokens,
            num_subtitle_cues=num_subtitle_cues,
            run_index=run_index,
        )
        records.append(_record)

    task_config = TaskConfig.from_vlm_eval_kit(
        task_name=task_name, dataset=dataset,
        num_records=num_records,
        system_prompt=system_prompt, task_prompt=task_prompt,
    )
    return records, task_config
    
    
def evaluate_task(
    evaluation_engine: str,
    dataset_name: str,
    evaluation_method: str,
    benchmark_config: Union[Dict[str, Any], TaskMeta],
    records: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    config: Optional[str] = None,
    fps: Optional[float] = None,
    nframe: Optional[int] = None,
    api_nproc: int = 4,
    retry: Optional[int] = None,
    judge: Optional[str] = None,
    judge_args: Optional[str] = None,
    use_verifier: bool = False,
    use_vllm: bool = False,
    verbose: bool = True,
) -> Tuple[EvaluationRunOutput, List[Dict[str, Any]]]:
    # Run VLMEvalKit evaluation by writing predictions to a temp xlsx and invoking dataset.evaluate().
    # Args: records - inference result dicts, judge - LLM judge model name override
    # Returns: tuple of (EvaluationRunOutput with aggregated/group metrics, per-sample metric dicts)
    from vlmeval.smp import listinstr, proxy_set, MMBenchOfficialServer
    from vlmeval.utils.result_transfer import MMMU_result_transfer, MMTBench_result_transfer
    
    dataset = build_dataset(
        dataset_name=dataset_name,
        model_name=model_name,
        config=config,
        fps=fps,
        nframe=nframe,
    )

    data_all = {_index: "" for _index in dataset.data['index']}
    for _record in records:
        _prediction = _record["prediction"]
        if (
            isinstance(_record.get("prediction_postprocessed", None), str)
            and len(_record["prediction_postprocessed"]) > 0
        ):
            _prediction = _record["prediction_postprocessed"]
        if _prediction is None:
            _prediction = ""
        data_all[_record["index"]] = _prediction

    # vlm_eval_kit.vlmeval.inference.py #L226-L265
    # vlm_eval_kit.vlmeval.inference_video.py #L245-L254
    # vlm_eval_kit.vlmeval.inference_mt.py #L188-L194
    data = dataset.data
    if dataset_name == 'MMBench-Video' and getattr(dataset, 'pack', False):
        data, vstats = dataset.load_pack_answers(data_all)
        logger.info(f'Statistics of Pack Video Inference: {vstats}')
    else:
        for x in data['index']:
            if x not in data_all:
                raise ValueError(f'Index {x} not found in data_all')
        data['prediction'] = [str(data_all[x]) for x in data['index']]
        if 'image' in data:
            data.pop('image')
    
    result_file = get_temp_filepath(
        suffix=".xlsx",
        dirpath="./temp",
    ) # temp_file to remove
    data.to_excel(result_file, index=False, engine='xlsxwriter')
    
    # vlm_eval_kit.run #L352-L464
    judge_kwargs = {
        'nproc': api_nproc,
        'verbose': verbose,
        'retry': retry if retry is not None else 3,
        **(json.loads(judge_args) if judge_args else {}),
    }

    if retry is not None:
        judge_kwargs['retry'] = retry
    if judge is not None:
        judge_kwargs['model'] = judge
    else:
        logger.debug(f'evaluate_task: dataset_name={dataset_name}')
        if dataset.TYPE in ['MCQ', 'Y/N', 'MCQ_MMMU_Pro'] or listinstr(
            ['moviechat1k', 'mme-reasoning'], dataset_name.lower()
        ):
            if listinstr(['WeMath', 'MME-Reasoning'], dataset_name):
                judge_kwargs['model'] = 'gpt-4o-mini'
            elif listinstr(['VisuLogic'], dataset_name):
                judge_kwargs['model'] = 'exact_matching'
            else:
                judge_kwargs['model'] = 'chatgpt-0125'
        elif listinstr(['MMVet', 'LLaVABench', 'MMBench_Video'], dataset_name):
            judge_kwargs['model'] = 'gpt-4-turbo'
        elif listinstr(['VGRPBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['MathVista', 'MathVerse', 'MathVision', 'DynaMath', 'VL-RewardBench', 'LogicVista', 'MOAT', 'OCR_Reasoning'], dataset_name):  # noqa: E501
            judge_kwargs['model'] = 'gpt-4o-mini'
        elif listinstr(['MMLongBench', 'MMDU', 'DUDE', 'SLIDEVQA', 'MIA-Bench', 'WildVision', 'MMAlignBench', 'MM-IFEval'], dataset_name):  # noqa: E501
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['ChartMimic'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'
        elif listinstr(['VDC'], dataset_name):
            judge_kwargs['model'] = 'llama31-8b'
        elif listinstr(['Video_MMLU_QA', 'Video_MMLU_CAP'], dataset_name):
            judge_kwargs['model'] = 'qwen-72b'
        elif listinstr(['MMVMBench'], dataset_name):
            judge_kwargs['model'] = 'gpt-4o'

    if use_verifier:
        judge_kwargs['use_verifier'] = True
    if use_vllm:
        judge_kwargs['use_vllm'] = True

    # skip evaluation
    eval_results = None
    if dataset_name in ['MMMU_TEST']:
        # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
        result_json = MMMU_result_transfer(result_file)
        logger.info(f'Transfer MMMU_TEST result to json for official evaluation, json file saved in {result_json}')

    elif 'MMT-Bench_ALL' in dataset_name:
        # Prepare Submission Files for MMMU_TEST AND MMT-Bench_ALL
        submission_file = MMTBench_result_transfer(result_file, **judge_kwargs)
        logger.info(
            f'Extract options from prediction of MMT-Bench FULL split for official evaluation '
            f'(https://eval.ai/web/challenges/challenge-page/2328/overview), '
            f'submission file saved in {submission_file}'
        )

    elif 'MLLMGuard_DS' in dataset_name:
        # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
        logger.warning('The evaluation of MLLMGuard_DS is not supported yet.')

    elif 'AesBench_TEST' == dataset_name:
        # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
        logger.info(f'The results are saved in {result_file}.')
    
    elif dataset_name in [
        'DocVQA_TEST',
        'InfoVQA_TEST',
        'Q-Bench1_TEST',
        'A-Bench_TEST',
    ]:
        # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
        logger.warning(f'{dataset_name} is a test split without ground-truth. Thus only the inference part is supported for those datasets.')
    
    elif dataset_name in [
        'MMBench_TEST_CN',
        'MMBench_TEST_EN', 
        'MMBench',
        'MMBench_CN',
        'MMBench_TEST_CN_V11',
        'MMBench_TEST_EN_V11',
        'MMBench_V11',
        'MMBench_CN_V11'
    ] and not MMBenchOfficialServer(dataset_name):
        # Skip the evaluation part if the dataset evaluation is not supported or annotations are missing
        logger.warning(f'Can not evaluate {dataset_name} on non-official servers, will skip the evaluation.')

    else:
        # Setup the proxy for the evaluation
        eval_proxy = os.environ.get('EVAL_PROXY', None)
        old_proxy = os.environ.get('HTTP_PROXY', '')
        if eval_proxy is not None:
            proxy_set(eval_proxy)
                            
        eval_results = dataset.evaluate(result_file, **judge_kwargs)
        if os.path.exists(result_file): # remove temp_file
            os.remove(result_file)
            
        if isinstance(eval_results, pd.DataFrame):
            if len(eval_results) <= len(eval_results.columns):
                eval_results = eval_results.T
        
        # Restore the proxy
        if eval_proxy is not None:
            proxy_set(old_proxy)
        
        # omni_evaluator  
        # aggregate evaluation_output
        if isinstance(eval_results, pd.DataFrame): 
            eval_results = eval_results.to_dict()
        
    num_samples = benchmark_config.num_records
    _num_valid_inferences = sum([
        isinstance(instance["prediction"], str) and len(instance["prediction"]) > 0
        for instance in records
    ])
    num_empty_predictions = len(records) - _num_valid_inferences
    coverage_inference = _num_valid_inferences / len(records)
    
    metrics = dict()
    metric_keys = list()
    group_metrics = dict()
    
    # collect overall metrics
    if eval_results:
        overall_field_names = ["overall", "all", "average", "avg", ]
        for _split_idx, (_split_key, _split) in enumerate(eval_results.items()):
            if isinstance(_split, (int, float)):
                _split = {
                    _split_key: _split,
                }
            elif isinstance(_split, (list, tuple)):
                _split_list = copy.deepcopy(_split)
                _split = {               
                    "split": _split_key,
                }
                for _e in _split_list: # e.g. [197, 200, "98.50%"]
                    if (
                        not is_integer(_e)
                        and is_numeric(_e)
                    ): # if not meta info but metric_value
                        _split["score"] = is_numeric(_e)

            # A category with no samples (num == 0) comes back as None from some
            # VLMEvalKit datasets — e.g. CRPE_EXIST returns None for the unused
            # subject/predicate/object splits. Such an entry has no metric to
            # aggregate, so skip it instead of crashing on _split.get(...) below.
            if not isinstance(_split, dict):
                continue

            _metric_name_suffix = ""
            if (
                _split.get("split", None)
                and _split["split"] != "none"
            ):
                _metric_name_suffix = f'__{_split["split"]}'

            for _overall_field_name in overall_field_names:
                if _overall_field_name not in _split:
                    continue
                metrics[_overall_field_name] = _split[_overall_field_name]
                metric_keys.append(_overall_field_name)

            _group_metrics = _split
            if  isinstance(_split.get("domain", None), dict):
                _group_metrics = _split["domain"]
            for _idx, (_group_name, _metric_value) in enumerate(_group_metrics.items()):
                if not isinstance(_group_name, str):
                    _group_name = str(_group_name)
                if (
                    _group_name in _overall_field_name
                    or _group_name in ["split", ]
                ):
                    continue
                
                _metric_name = "overall"
                if (
                    len(metric_keys) < 1
                    and not _metric_name_suffix
                ):
                    pass
                elif (
                    len(metric_keys) > 0
                    and _metric_name_suffix
                ):
                    _metric_name = f'{metric_keys[0]}{_metric_name_suffix}'
                elif len(metric_keys) > 0:
                    _metric_name = metric_keys[0]
                elif _metric_name_suffix:
                    _metric_name = _metric_name_suffix
                    if _metric_name.startswith("__"):
                        _metric_name = _metric_name[2:]
                    
                if _group_name not in group_metrics:
                    group_metrics[_group_name] = dict()
                group_metrics[_group_name][_metric_name] = _metric_value  
    
    # average group_metrics if main task not exists
    if (
        len(metric_keys) < 1 
        and len(group_metrics) > 0
    ):
        _average_metrics = defaultdict(list)
        for _group_name, _group_metric in group_metrics.items():
            for _k, _v in _group_metric.items():
                _average_metrics[_k].append(_v)
        _macro_average, _cnt = 0, 0
        for _i, (_k, _v) in enumerate(_average_metrics.items()):
            if not all([
                True if isinstance(_e, (int, float)) else False 
                for _e in _v
            ]):
                continue
            _macro_average += np.mean(_v)
            _cnt += 1
        if _cnt > 0:
            metrics["average"] = _macro_average / _cnt
            metric_keys.append("average")

    # TODO: update sample-wise metrics
    sample_metrics = list()
    for _record_idx in range(0, len(records)):
        _metrics = dict()
        sample_metrics.append(_metrics)

    evaluation_run_output = EvaluationRunOutput(
        inference_engine=None,
        evaluation_engine=evaluation_engine,
        task_name=dataset_name,
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
    dataset_name: str,
    row: Dict[str, Any],
    user_content: List[Dict[str, Any]],
    dataset: Any,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    run_index: Optional[int] = 0,
    **kwargs,
) -> Record:
    # Convert a VLMEvalKit dataset row and built prompt into a Record for inference.
    # Args: row - dataset.data.iloc[i].to_dict(), user_content - list of content dicts from build_prompt()
    # Returns: Record with messages, options, labels, and metadata extracted from the VLMEvalKit objects
    """
    record: dataset.data.iloc[_idx].to_dict()
    dataset:
    """
    img_root = getattr(dataset, "img_root", None)
    if (
        isinstance(img_root, str)
        and os.path.exists(img_root)
    ):
        image_path = row.get("image_path", None)
        if isinstance(image_path, str):
            image_path = os.path.join(img_root, image_path)
        elif isinstance(image_path, (list, tuple)):
            image_path = [os.path.join(img_root, _image_path) for _image_path in image_path]
        row["image_path"] = image_path
    
    ocr_tokens = None
    _ocr_tokens = row.get("words", None) # check if benchmark with ocr exists among VLMEvalKit datasets
    if isinstance(_ocr_tokens, (list, tuple)):
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
    
    messages = list()
    _system_content, _user_content, _assistant_content = list(), list(), list()
    for _content_idx, _content in enumerate(user_content):
        _content_cls = CONTENT_ACCESSOR_MAP.get(_content["type"])
        _value_key = _content_cls.get_key(_content) if _content_cls else None
        _chat_content = {
            "type": _content["type"],
            "value": _content[_value_key] if _value_key else _content.get("value"),
        }
        if _content["type"] == "audio":
            _chat_content = ChatAudioContent(**_chat_content)
        elif _content["type"] == "image":
            if isinstance(ocr_tokens, (list, tuple)):
                _chat_content["ocr"] = ocr_tokens
            _chat_content = ChatImageContent(**_chat_content)
        elif _content["type"] == "text":
            if (
                "role" not in _content
                or _content["role"] == "user"
            ):
                ChatTextContent.set_value(_content, format_task_prompt(
                    task_prompt=task_prompt,
                    query=ChatTextContent.get_value(_content),
                ))
            _chat_content = ChatTextContent(**_chat_content)
        elif _content["type"] == "video":
            _chat_content = ChatVideoContent(**_chat_content)
        
        if (
            "role" not in _content
            or _content["role"] == "user"
        ):
            _user_content.append(_chat_content)
        elif _content["role"] == "system":
            _system_content.append(_chat_content)
        elif _content["role"] == "assistant":
            _assistant_content.append(_chat_content)
    
    if len(_system_content) > 0:
        messages.append(ChatMessage(
            role="system", 
            content=_system_content,
        ))
    elif isinstance(system_prompt, str):
        # add system message if not given
        messages.append(ChatMessage(
            role="system", 
            content=[
                ChatTextContent(type="text", value=system_prompt),
            ],
        ))
        
    messages.append(ChatMessage(**{
        "role": "user",
        "content": _user_content,
    }))
    
    if len(_assistant_content) > 0:
        messages.append(ChatMessage(
            role="assistant",
            content=_assistant_content,
        ))
        
    dataset_type = dataset.TYPE
    options = None
    option_contents = None
    if dataset_type in ["MCQ", "Video-MCQ", ]:
        options = list()
        option_contents = list()
        for _option in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            if _option not in row:
                continue
            options.append(_option)
            option_contents.append(row[_option])
            
        if (
            len(options) > 0
            or len(option_contents) > 0
        ): # successfully extracted options or option_contents
            pass
        elif "candidates" in row:
            option_contents = parse_string(string=row["candidates"])

    meta = {
        "question": row["question"],
        "category": row.get("category", None),
        "task_type": row.get("task_type", None),
        "data_type": row.get("data_type", None),
    }
    meta["category"] = meta["category"] or meta["task_type"] or meta["data_type"]
    return Record(
        benchmark=dataset_name,
        index=row["index"],
        prompt=None,
        messages=messages,
        generation_options=None,
        label=row.get("answer", None),
        options=options,
        option_contents=option_contents,
        prediction=None,
        latency=None,
        metrics=None,
        meta=meta,
    )


def build_dataset_from_config(cfg, dataset_name):
    # copied from VLMEvalKit/run.py: L76-L95
    import vlmeval.dataset
    from vlmeval.dataset.video_dataset_config import supported_video_datasets

    config = copy.deepcopy(cfg[dataset_name])
    if config == {}:
        return supported_video_datasets[dataset_name]()
    if 'class' not in config:
        raise ValueError(f'Config for dataset {dataset_name} must contain a "class" key')
    cls_name = config.pop('class')
    if hasattr(vlmeval.dataset, cls_name):
        cls = getattr(vlmeval.dataset, cls_name)
        sig = inspect.signature(cls.__init__)
        valid_params = {k: v for k, v in config.items() if k in sig.parameters}
        if cls.MODALITY == 'VIDEO':
            if valid_params.get('fps', 0) > 0 and valid_params.get('nframe', 0) > 0:
                raise ValueError('fps and nframe should not be set at the same time')
            if valid_params.get('fps', 0) <= 0 and valid_params.get('nframe', 0) <= 0:
                raise ValueError('fps and nframe should be set at least one valid value')
        return cls(**valid_params)
    else:
        raise ValueError(f'Class {cls_name} is not supported in `vlmeval.dataset`')