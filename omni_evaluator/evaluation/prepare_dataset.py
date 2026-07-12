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

import copy
from datasets import (
    Audio as hf_Audio,
    load_dataset as hf_load_dataset,
    concatenate_datasets as hf_concatenate_datasets,
)
import importlib
import itertools
import json
import logging
import os
from pathlib import Path
import PIL
from PIL import Image
import shutil
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence, Callable, Iterable

logger = logging.getLogger(__name__)

DEFAULT_S3_URL_DURATION = 7 * 24 * 60 * 60  # 7 days

from omni_evaluator import CombineMethod, DatasetSource, EvaluationEngine, Modality
from omni_evaluator.clients.s3_client import S3Client
from omni_evaluator.schemas.chat import (
    OcrToken, EntityToken,
    Message as ChatMessage,
    AudioContent as ChatAudioContent,
    ImageContent as ChatImageContent,
    TextContent as ChatTextContent,
    VideoContent as ChatVideoContent,
    CONTENT_ACCESSOR_MAP,
)
from omni_evaluator.schemas.inference import Record
from omni_evaluator.schemas.task import TaskConfig, TaskInference, TaskInferenceGenerationOptions
from omni_evaluator.utils.common import get_custom_module
from omni_evaluator.utils.data import format_task_prompt, extract_options, shift_options
from omni_evaluator.utils.io import read_file, write_file, iter_file, count_lines
from omni_evaluator.utils.string import is_url
from omni_evaluator.utils.torch import is_torchcodec_loadable


def load_dataset(
    task_name: str,
    task_config: TaskConfig,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    local_dirpath: Optional[str] = None,
    cache_dirpath: Optional[str] = None,
    batch_size: Optional[int] = None,
    run_index: Optional[int] = 0,
) -> Tuple[Iterable, int]:
    # Load dataset and build a record iterator based on task_config.dataset.source.
    # Args: task_config - defines dataset source/path/prompts, run_index - circular evaluation shift index
    # Returns: tuple of (dataset iterator yielding Records, total dataset size)
    # dataset_iterator
    sample_to_record_func = None
    _custom_module = get_custom_module(
        evaluation_engine=EvaluationEngine.builtin,
        task_name=task_name,
    )
    if (
        _custom_module
        and hasattr(_custom_module, "sample_to_record")
    ):
        sample_to_record_func = getattr(_custom_module, "sample_to_record", None)

    if task_config.dataset.source in [
        DatasetSource.resources,
    ]:
        local_dirpath = importlib.resources.files(
            f'omni_evaluator.evaluation.builtin.tasks.{task_name}',
        ).joinpath("resources")
        local_dirpath = Path.as_posix(local_dirpath)

    dataset_iterator, dataset_size = None, None
    if not local_dirpath:
        local_dirpath = task_config.dataset.local_dirpath
    audio_dirpath = task_config.dataset.audio_dirpath
    image_dirpath = task_config.dataset.image_dirpath
    video_dirpath = task_config.dataset.video_dirpath
    
    if not local_dirpath:
        pass
    elif os.path.exists(local_dirpath):
        if audio_dirpath:
            audio_dirpath = os.path.join(local_dirpath, audio_dirpath)
        if image_dirpath:
            image_dirpath = os.path.join(local_dirpath, image_dirpath)
        if video_dirpath:
            video_dirpath = os.path.join(local_dirpath, video_dirpath)
    elif task_config.dataset.source != DatasetSource.local:
        local_dirpath = None
    else:
        raise ValueError(f'`local_dirpath` in task_config not exists: {local_dirpath}')
    
    if (
        local_dirpath
        and task_config.dataset.source in [
            DatasetSource.local,
            DatasetSource.s3,
            DatasetSource.resources,
        ]
    ):
        data_filepath = task_config.dataset.data_filepath
        data_filepath = os.path.join(local_dirpath, data_filepath)
        _subset = task_config.dataset.subset
        # Lazy path for row-oriented files (jsonl/gzip/csv/tsv): yield rows one
        # at a time so the first inference can start before the whole file is
        # parsed. dataset_size here is the file row count (pre-subset); the
        # accurate post-subset count is set as task_config.num_records once
        # inference finishes (see infer.py).
        if data_filepath.endswith((".jsonl", ".gzip", ".csv", ".tsv")):
            dataset_size = count_lines(data_filepath)
            data = iter_file(filepath=data_filepath)
            if _subset:
                data = (_s for _s in data if _match_subset(_s, _subset))
        else:
            data = read_file(filepath=data_filepath)
            if _subset:
                data = [_s for _s in data if _match_subset(_s, _subset)]
            dataset_size = len(data)
        if task_config.meta.num_sample_repetition:
            dataset_size = dataset_size * task_config.meta.num_sample_repetition
        dataset_iterator = to_generator_local(
            sequence=data,
            num_samples=dataset_size,
            task_name=task_name,
            task_config=task_config,
            sample_to_record_func=sample_to_record_func,
            audio_dirpath=audio_dirpath,
            image_dirpath=image_dirpath,
            video_dirpath=video_dirpath,
            cache_dirpath=cache_dirpath,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            num_ocr_tokens=num_ocr_tokens,
            num_entity_tokens=num_entity_tokens,
            num_subtitle_cues=num_subtitle_cues,
            run_index=run_index,
        )
        
    elif task_config.dataset.source == DatasetSource.s3:
        s3_client = S3Client(
            bucket_name=os.environ["S3_BUCKET_NAME"],
            access_key=os.environ["S3_ACCESS_KEY"],
            secret_key=os.environ["S3_SECRET_KEY"],
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            region=os.environ["S3_REGION"],
        )
        data_filepath = task_config.dataset.data_filepath
        data = s3_client.download_obj(remote_filepath=data_filepath)
        # Pre-filter by dataset.subset so dataset_size, num_samples, tqdm total and
        # downstream num_records all reflect the matched count.
        _subset = task_config.dataset.subset
        if _subset:
            data = [_s for _s in data if _match_subset(_s, _subset)]
        dataset_size = len(data)
        if task_config.meta.num_sample_repetition:
            dataset_size = dataset_size * task_config.meta.num_sample_repetition
        dataset_iterator = to_generator_s3(
            sequence=data,
            num_samples=dataset_size,
            task_name=task_name,
            task_config=task_config,
            sample_to_record_func=sample_to_record_func,
            audio_dirpath=audio_dirpath,
            image_dirpath=image_dirpath,
            video_dirpath=video_dirpath,
            cache_dirpath=None, # use presigned_url instead of local temp path
            batch_size=batch_size,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            num_ocr_tokens=num_ocr_tokens,
            num_entity_tokens=num_entity_tokens,
            num_subtitle_cues=num_subtitle_cues,
            run_index=run_index,
            s3_client=s3_client,
            url_duration=DEFAULT_S3_URL_DURATION,
        )
        
        del s3_client
    
    elif task_config.dataset.source == DatasetSource.huggingface_hub:
        names = list()
        if isinstance(task_config.dataset.name, str):
            names.append(task_config.dataset.name)
        elif isinstance(task_config.dataset.name, (list, tuple)):
            names += task_config.dataset.name
        if len(names) < 1:
            names = [None, ]

        revisions = list()
        if isinstance(task_config.dataset.revision, str):
            revisions.append(task_config.dataset.revision)
        elif isinstance(task_config.dataset.revision, (list, tuple)):
            revisions += task_config.dataset.revision
        if len(revisions) < 1:
            revisions = [None, ]
            
        splits = list()
        if isinstance(task_config.dataset.split, str):
            splits.append(task_config.dataset.split)
        elif isinstance(task_config.dataset.split, (list, tuple)):
            splits += task_config.dataset.split
        if len(splits) < 1:
            splits = [None, ]
        
        _length = max(len(names), len(revisions), len(splits))
        if len(names) == len(revisions) == len(splits):
            pass
        elif (
            (len(names) != _length and len(names) != 1)
            or (len(revisions) != _length and len(revisions) != 1)
            or (len(splits) != _length and len(splits) != 1)
        ):
            raise ValueError(f'Length not match between names, revisions, and splits: {names} vs. {revisions} vs. {splits}')
        else:
            if len(names) == 1:
                names = names * _length
            if len(revisions) == 1:
                revisions = revisions * _length
            if len(splits) == 1:
                splits = splits * _length

        dataset_iterator = None
        dataset_size = 0
        _combine = task_config.dataset.combine
        _combine_method = _combine.method if _combine is not None else CombineMethod.concatenate
        if _combine_method == CombineMethod.concatenate:
            dataset_iterator = list()
            for _name, _revision, _split in zip(names, revisions, splits):
                _dataset = hf_load_dataset(
                    path=task_config.dataset.path,
                    name=_name if _name else None,
                    revision=_revision if _revision else None,
                    split=_split if _split else None,
                    cache_dir=os.getenv("HF_HUB_CACHE", None),
                    trust_remote_code=task_config.dataset.trust_remote_code,
                    verification_mode=task_config.dataset.verification_mode,
                )
                _names = [_name, ] * len(_dataset)
                _revisions = [_revision, ] * len(_dataset)
                _splits = [_split, ] * len(_dataset)

                if isinstance(_dataset, dict):
                    _datasets = list()
                    _names, _revisions, _splits = list(), list(), list()
                    for _split_, _dataset_ in _dataset.items():
                        _datasets.append(_dataset_)
                        _names += [_name, ] * len(_dataset_)
                        _revisions += [_revision, ] * len(_dataset_)
                        _splits += [_split_, ] * len(_dataset_)
                    _dataset = hf_concatenate_datasets(_datasets)
                if "dataset_name" not in _dataset.column_names:
                    _dataset = _dataset.add_column("dataset_name", _names)
                if "dataset_revision" not in _dataset.column_names:
                    _dataset = _dataset.add_column("dataset_revision", _revisions)
                if "dataset_split" not in _dataset.column_names:
                    _dataset = _dataset.add_column("dataset_split", _splits)
                
                # Coerce Audio(decode=False) when the task asks for raw bytes
                # (yaml: ``audio_decode: false``) OR when torchcodec is unloadable
                # (env-level ABI fallback). Raw bytes/dicts then flow through
                # our librosa-based multimodal helpers.
                if (
                    not task_config.dataset.audio_decode
                    or not is_torchcodec_loadable()
                ):
                    _audio_columns = task_config.dataset.audio_column
                    if isinstance(_audio_columns, str):
                        _audio_columns = [_audio_columns, ]
                    for _audio_column in _audio_columns:
                        if _audio_column not in _dataset.column_names:
                            continue
                        _dataset = _dataset.cast_column(
                            _audio_column,
                            hf_Audio(decode=False),
                        )

                # Pre-filter by dataset.subset (stack branch). Use HF Dataset.filter so
                # downstream len(_dataset) reflects the matched count.
                _subset = task_config.dataset.subset
                if _subset:
                    _dataset = _dataset.filter(lambda _s: _match_subset(_s, _subset))

                _dataset_size = len(_dataset)
                if task_config.meta.num_sample_repetition:
                    _dataset_size = _dataset_size * task_config.meta.num_sample_repetition

                _dataset_iterator = to_generator_huggingface_hub(
                    sequence=_dataset,
                    num_samples=_dataset_size,
                    task_name=task_name,
                    task_config=task_config,
                    sample_to_record_func=sample_to_record_func,
                    audio_dirpath=audio_dirpath,
                    image_dirpath=image_dirpath,
                    video_dirpath=video_dirpath,
                    cache_dirpath=cache_dirpath,
                    batch_size=batch_size,
                    system_prompt=system_prompt,
                    task_prompt=task_prompt,
                    num_ocr_tokens=num_ocr_tokens,
                    num_entity_tokens=num_entity_tokens,
                    num_subtitle_cues=num_subtitle_cues,
                    run_index=run_index,
                )
                dataset_iterator.append(_dataset_iterator)
                dataset_size += _dataset_size
            
            dataset_iterator = itertools.chain.from_iterable(dataset_iterator)

        elif _combine_method == CombineMethod.join:
            _join_key = _combine.key
            dataset = None
            dataset_meta = dict()
            num_subdatasets = 0
            for _name, _revision, _split in zip(names, revisions, splits):
                _dataset = hf_load_dataset(
                    path=task_config.dataset.path,
                    name=_name if _name else None,
                    revision=_revision if _revision else None,
                    split=_split if _split else None,
                    cache_dir=os.getenv("HF_HUB_CACHE", None),
                    trust_remote_code=task_config.dataset.trust_remote_code,
                    verification_mode=task_config.dataset.verification_mode,
                )
                # Coerce Audio(decode=False) when the task asks for raw bytes
                # (yaml: ``audio_decode: false``) OR when torchcodec is unloadable
                # (env-level ABI fallback). Raw bytes/dicts then flow through
                # our librosa-based multimodal helpers.
                if (
                    not task_config.dataset.audio_decode
                    or not is_torchcodec_loadable()
                ):
                    _audio_columns = task_config.dataset.audio_column
                    if isinstance(_audio_columns, str):
                        _audio_columns = [_audio_columns, ]
                    for _audio_column in _audio_columns:
                        if _audio_column not in _dataset.column_names:
                            continue
                        _dataset = _dataset.cast_column(
                            _audio_column,
                            hf_Audio(decode=False),
                        )
                # Pre-filter by dataset.subset (join branch). Same as concatenate branch.
                _subset = task_config.dataset.subset
                if _subset:
                    _dataset = _dataset.filter(lambda _s: _match_subset(_s, _subset))
                _dataset_size = len(_dataset)
                if task_config.meta.num_sample_repetition:
                    _dataset_size = _dataset_size * task_config.meta.num_sample_repetition

                if _join_key not in _dataset.column_names:
                    raise ValueError(
                        f"dataset.combine.key='{_join_key}' not found in subset "
                        f"'{_name}' columns: {_dataset.column_names}"
                    )

                _meta = dict()
                _column_names = _dataset.column_names
                for _column_name in _column_names:
                    _column = _dataset[_column_name]
                    if not isinstance(_column, list):
                        _column = list(_column)
                    _meta[_column_name] = _column

                _meta_name = _name
                if not _meta_name:
                    _meta_name = _revision
                for _row in zip(*(_meta[_column_name] for _column_name in _column_names)):
                    _row = dict(zip(_column_names, _row))
                    _key_value = _row[_join_key]
                    if _key_value not in dataset_meta:
                        dataset_meta[_key_value] = dict()
                    dataset_meta[_key_value].update({_meta_name: _row})

                if dataset is None:
                    dataset = _dataset
                    dataset_size = _dataset_size
                num_subdatasets += 1

            dataset = dataset.add_column("dataset_name", len(dataset) * [names, ])
            dataset = dataset.add_column("dataset_revision", len(dataset) * [revisions, ])
            dataset = dataset.add_column("dataset_split", len(dataset) * [splits, ])

            indices = [
                _index
                for _index, _key_value
                in enumerate(dataset[_join_key])
                if (
                    _key_value in dataset_meta.keys()
                    and len(dataset_meta[_key_value]) == num_subdatasets
                )
            ]
            dataset = dataset.select(indices)
            dataset_meta = [
                copy.deepcopy(dataset_meta[_key_value])
                for _key_value in dataset[_join_key]
            ]
            dataset = dataset.add_column("meta", dataset_meta)
                
            dataset_iterator = to_generator_huggingface_hub(
                sequence=dataset,
                num_samples=dataset_size,
                task_name=task_name,
                task_config=task_config,
                sample_to_record_func=sample_to_record_func,
                audio_dirpath=audio_dirpath,
                image_dirpath=image_dirpath,
                video_dirpath=video_dirpath,
                cache_dirpath=cache_dirpath,
                batch_size=batch_size,
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                num_ocr_tokens=num_ocr_tokens,
                num_entity_tokens=num_entity_tokens,
                num_subtitle_cues=num_subtitle_cues,
                run_index=run_index,
            )
        
    else:
        raise ValueError(f'Invalid dataset type: {task_config.dataset.source}')
    
    return dataset_iterator, dataset_size


def to_generator_local(
    sequence: Union[Iterable, Sequence[Dict[str, Any]]],
    num_samples: int,
    task_name: str,
    task_config: TaskConfig,
    sample_to_record_func: Optional[Callable] = None,
    audio_dirpath: Optional[str] = None,
    image_dirpath: Optional[str] = None,
    video_dirpath: Optional[str] = None,
    *,
    cache_dirpath: Optional[str] = None,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    run_index: Optional[int] = 0,
) -> Iterable[Record]:
    # Yield Records from local-filesystem samples, resolving media file paths.
    # Args: sequence - raw data rows, sample_to_record_func - custom converter (falls back to default)
    # Returns: generator yielding Record objects with resolved local media paths
    if not sample_to_record_func:
        sample_to_record_func = sample_to_record

    for _sample_idx, _sample in enumerate(sequence):
        _records = sample_to_record_func(
            task_name=task_name,
            task_config=task_config,
            sample_idx=_sample_idx,
            sample=_sample,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            num_ocr_tokens=num_ocr_tokens,
            num_entity_tokens=num_entity_tokens,
            num_subtitle_cues=num_subtitle_cues,
            run_index=run_index,
        )
        if not isinstance(_records, (list, tuple)):
            _records = [_records, ]

        for _repetition_idx, _record in enumerate(_records):
            _record = copy.deepcopy(_record)
            for _message_idx, _message in enumerate(_record.messages):
                for _content_idx, _content in enumerate(_message.content):
                    # append local_dirpath to multimodal items
                    if (
                        not isinstance(_content.value, str)
                        or is_url(_content.value)
                        or os.path.exists(_content.value)
                    ):
                        pass
                    elif _content.type == Modality.text:
                        pass
                    elif _content.type == Modality.audio:
                        _audio_filepath = _content.value
                        if audio_dirpath:
                            _audio_filepath = os.path.join(audio_dirpath, _audio_filepath)
                            _message.content[_content_idx].value = _audio_filepath
                        if not os.path.exists(_audio_filepath):
                            raise FileNotFoundError(f'Audio content not exist: {_audio_filepath}')
                    elif _content.type == Modality.image:
                        _image_filepath = _content.value
                        if image_dirpath:
                            _image_filepath = os.path.join(image_dirpath, _content.value)
                            _message.content[_content_idx].value = _image_filepath
                        if not os.path.exists(_image_filepath):
                            raise FileNotFoundError(f'Image content not exist: {_image_filepath}')
                    elif _content.type == Modality.video:
                        _video_filepath = _content.value
                        if video_dirpath:
                            _video_filepath = os.path.join(video_dirpath, _content.value)
                            _message.content[_content_idx].value = _video_filepath
                        if not os.path.exists(_video_filepath):
                            raise FileNotFoundError(f'Video content not exist: {_video_filepath}')

            yield _record

        if _sample_idx >= num_samples:
            break


def to_generator_s3(
    sequence: Union[Iterable, Sequence[Dict[str, Any]]],
    num_samples: int,
    task_name: str,
    task_config: TaskConfig,
    sample_to_record_func: Optional[Callable] = None,
    audio_dirpath: Optional[str] = None,
    image_dirpath: Optional[str] = None,
    video_dirpath: Optional[str] = None,
    *,
    cache_dirpath: Optional[str] = None,
    batch_size: Optional[int] = None,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    run_index: Optional[int] = 0,
    s3_client: Optional[S3Client] = None,
    url_duration: int = DEFAULT_S3_URL_DURATION,
) -> Iterable[Record]:
    # Yield Records from S3-backed samples, downloading or generating presigned URLs for media.
    # Args: s3_client - reusable S3 client (created from env vars if None), url_duration - presigned URL TTL in seconds
    # Returns: generator yielding Record objects with S3 presigned URLs or cached local paths
    if not sample_to_record_func:
        sample_to_record_func = sample_to_record
    
    if not s3_client:
        s3_client = S3Client(
            bucket_name=os.environ["S3_BUCKET_NAME"],
            access_key=os.environ["S3_ACCESS_KEY"],
            secret_key=os.environ["S3_SECRET_KEY"],
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            region=os.environ["S3_REGION"],
        )
    
    filepaths_to_remove = set()
    try:
        for _sample_idx, _sample in enumerate(sequence):
            _records = sample_to_record_func(
                task_name=task_name,
                task_config=task_config,
                sample_idx=_sample_idx,
                sample=_sample,
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                num_ocr_tokens=num_ocr_tokens,
                num_entity_tokens=num_entity_tokens,
                num_subtitle_cues=num_subtitle_cues,
                run_index=run_index,
            )
            if not isinstance(_records, (list, tuple)):
                _records = [_records, ]

            for _repetition_idx, _record in enumerate(_records):
                _record = copy.deepcopy(_record)
                for _message_idx, _message in enumerate(_record.messages):
                    for _content_idx, _content in enumerate(_message.content):
                        # append local_dirpath to multimodal items
                        if (
                            not isinstance(_content.value, str)
                            or is_url(_content.value)
                            or os.path.exists(_content.value)
                        ):
                            pass
                        elif _content.type == Modality.text:
                            pass
                        elif _content.type == Modality.audio:
                            _audio_filepath = _content.value
                            if audio_dirpath:
                                _audio_filepath = os.path.join(audio_dirpath, _audio_filepath)

                            if cache_dirpath:
                                _local_audio_filepath = os.path.join(cache_dirpath, _audio_filepath)
                                if not Path(_local_audio_filepath).parent.exists():
                                    Path(_local_audio_filepath).parent.mkdir(exist_ok=True, parents=True)
                                s3_client.download_file(
                                    remote_filepath=_audio_filepath,
                                    filepath=_local_audio_filepath,
                                )
                                if not os.path.exists(_local_audio_filepath):
                                    raise FileNotFoundError(f'Audio content not exist: {_local_audio_filepath}')
                                _message.content[_content_idx].value = _local_audio_filepath
                                filepaths_to_remove.add(_local_audio_filepath)
                            else:
                                _audio_url = s3_client.get_presigned_url(
                                    remote_filepath=_audio_filepath,
                                    duration=url_duration,
                                )
                                _message.content[_content_idx].value = _audio_url
                                
                        elif _content.type == Modality.image:
                            _image_filepath = _content.value
                            if image_dirpath:
                                _image_filepath = os.path.join(image_dirpath, _content.value)

                            if cache_dirpath:
                                _local_image_filepath = os.path.join(cache_dirpath, _image_filepath)
                                if not Path(_local_image_filepath).parent.exists():
                                    Path(_local_image_filepath).parent.mkdir(exist_ok=True, parents=True)
                                s3_client.download_file(
                                    remote_filepath=_image_filepath,
                                    filepath=_local_image_filepath,
                                )
                                if not os.path.exists(_local_image_filepath):
                                    raise FileNotFoundError(f'Image content not exist: {_local_image_filepath}')
                                _message.content[_content_idx].value = _local_image_filepath
                                filepaths_to_remove.add(_local_image_filepath)
                            else:
                                _image_url = s3_client.get_presigned_url(
                                    remote_filepath=_image_filepath,
                                    duration=url_duration,
                                )
                                _message.content[_content_idx].value = _image_url
                                
                        elif _content.type == Modality.video:
                            _video_filepath = _content.value
                            if video_dirpath:
                                _video_filepath = os.path.join(video_dirpath, _content.value)

                            if cache_dirpath:
                                _local_video_filepath = os.path.join(cache_dirpath, _video_filepath)
                                if not Path(_local_video_filepath).parent.exists():
                                    Path(_local_video_filepath).parent.mkdir(exist_ok=True, parents=True)
                                s3_client.download_file(
                                    remote_filepath=_video_filepath,
                                    filepath=_local_video_filepath,
                                )
                                if not os.path.exists(_local_video_filepath):
                                    raise FileNotFoundError(f'Video content not exist: {_local_video_filepath}')
                                _message.content[_content_idx].value = _local_video_filepath
                                filepaths_to_remove.add(_local_video_filepath)
                            else:
                                _video_url = s3_client.get_presigned_url(
                                    remote_filepath=_video_filepath,
                                    duration=url_duration,
                                )
                                _message.content[_content_idx].value = _video_url
                
                yield _record
            
            if (
                (isinstance(batch_size, int) and ((_sample_idx + 1) * (_repetition_idx + 1)) % batch_size == 0)
                or _sample_idx >= num_samples
            ):
                for _filepath in filepaths_to_remove:
                    os.remove(_filepath)
            if _sample_idx >= num_samples:
                break
            
    finally:
        del s3_client
        for _filepath in filepaths_to_remove:
            os.remove(_filepath)

def to_generator_huggingface_hub(
    sequence: Union[Iterable, Sequence[Dict[str, Any]]],
    num_samples: int,
    task_name: str,
    task_config: TaskConfig,
    sample_to_record_func: Optional[Callable] = None,
    audio_dirpath: Optional[str] = None,
    image_dirpath: Optional[str] = None,
    video_dirpath: Optional[str] = None,
    *,
    cache_dirpath: Optional[str] = None,
    batch_size: Optional[int] = None,
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    run_index: Optional[int] = 0,
) -> Iterable[Record]:
    # Yield Records from a HuggingFace Hub dataset, resolving local media paths.
    # Args: sequence - HF dataset object, sample_to_record_func - custom converter (falls back to default)
    # Returns: generator yielding Record objects with resolved media paths
    if not sample_to_record_func:
        sample_to_record_func = sample_to_record

    for _sample_idx, _sample in enumerate(sequence):
        _records = sample_to_record_func(
            task_name=task_name,
            task_config=task_config,
            sample_idx=_sample_idx,
            sample=_sample,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            num_ocr_tokens=num_ocr_tokens,
            num_entity_tokens=num_entity_tokens,
            num_subtitle_cues=num_subtitle_cues,
            run_index=run_index,
        )
        if not isinstance(_records, (list, tuple)):
            _records = [_records, ]

        for _repetition_idx, _record in enumerate(_records):
            _record = copy.deepcopy(_record)
            for _message_idx, _message in enumerate(_record.messages):
                for _content_idx, _content in enumerate(_message.content):
                    # append local_dirpath to multimodal items
                    if (
                        not isinstance(_content.value, str)
                        or is_url(_content.value)
                        or os.path.exists(_content.value)
                    ):
                        pass
                    elif _content.type == Modality.text:
                        pass
                    elif _content.type == Modality.audio:
                        _audio_filepath = _content.value
                        if audio_dirpath:
                            _audio_filepath = os.path.join(audio_dirpath, _audio_filepath)
                            _message.content[_content_idx].value = _audio_filepath
                        if not os.path.exists(_audio_filepath):
                            raise FileNotFoundError(f'Audio content not exist: {_audio_filepath}')
                    elif _content.type == Modality.image:
                        _image_filepath = _content.value
                        if image_dirpath:
                            _image_filepath = os.path.join(image_dirpath, _content.value)
                            _message.content[_content_idx].value = _image_filepath
                        if not os.path.exists(_image_filepath):
                            raise FileNotFoundError(f'Image content not exist: {_image_filepath}')
                    elif _content.type == Modality.video:
                        _video_filepath = _content.value
                        if video_dirpath:
                            _video_filepath = os.path.join(video_dirpath, _content.value)
                            _message.content[_content_idx].value = _video_filepath
                        if not os.path.exists(_video_filepath):
                            raise FileNotFoundError(f'Video content not exist: {_video_filepath}')

            yield _record

        if _sample_idx >= num_samples:
            break

def _match_subset(
    sample: Dict[str, Any],
    subset: Optional[Dict[str, List[str]]],
) -> bool:
    # Filter predicate for dataset.subset. AND across keys, OR within each value list.
    # Missing key → reject. List-valued field (e.g. img_type) → intersection.
    # Lookup order: sample["meta"][key] first (s3/local convention), then sample[key]
    # (HF flat-column convention). This lets one subset spec work across sources.
    if not subset:
        return True
    if not isinstance(sample, dict):
        return False
    _meta = sample.get("meta") if isinstance(sample.get("meta"), dict) else None
    for _key, _allowed in subset.items():
        if _meta is not None and _key in _meta:
            _actual = _meta[_key]
        elif _key in sample:
            _actual = sample[_key]
        else:
            return False
        if _actual is None:
            return False
        if isinstance(_actual, (list, tuple)):
            if not any(_v in _allowed for _v in _actual):
                return False
        else:
            if _actual not in _allowed:
                return False
    return True


def _resolve_conditional_prompt(
    prompt: Union[str, Dict[str, Any]],
    sample: Dict[str, Any],
    conditional_on: Optional[str],
) -> Optional[str]:
    # Resolve a per-variant prompt dict to the string matching this sample's meta key.
    # Args: prompt - dict keyed by meta values (or str passthrough), conditional_on - meta field name
    # Returns: the matching string, or None when no match (silent fallback to "no prompt")
    if not isinstance(prompt, dict):
        return prompt
    if not conditional_on:
        return None
    _meta = sample.get("meta", None) or {}
    _variant_key = _meta.get(conditional_on, None)
    if _variant_key is None or _variant_key not in prompt:
        return None
    return prompt[_variant_key]


def sample_to_record(
    task_name: str,
    task_config: TaskConfig,
    sample_idx: int,
    sample: Dict[str, Any],
    system_prompt: Optional[str] = None,
    task_prompt: Optional[str] = None,
    task_prompt_kwargs: Optional[Dict[str, Any]] = None,
    num_ocr_tokens: Optional[int] = None,
    num_entity_tokens: Optional[int] = None,
    num_subtitle_cues: Optional[int] = None,
    run_index: Optional[int] = 0,
    **kwargs,
) -> Union[Record, List[Record]]:
    # Convert a raw dataset sample dict into one or more Record objects for inference.
    # Args: sample - raw data dict with "messages" or "query" key, run_index - circular evaluation shift index
    # Returns: single Record or list of Records (when num_sample_repetition is set)
    index = sample.get("index", sample_idx)
    if (
        not isinstance(num_ocr_tokens, int)
        and isinstance(task_config.inference, TaskInference)
        and task_config.inference.num_ocr_tokens
    ):
        num_ocr_tokens = task_config.inference.num_ocr_tokens
    if (
        not isinstance(num_entity_tokens, int)
        and isinstance(task_config.inference, TaskInference)
        and task_config.inference.num_entity_tokens
    ):
        num_entity_tokens = task_config.inference.num_entity_tokens
    if (
        not isinstance(num_subtitle_cues, int)
        and isinstance(task_config.inference, TaskInference)
        and task_config.inference.num_subtitle_cues
    ):
        num_subtitle_cues = task_config.inference.num_subtitle_cues

    _prompts = task_config.prompts
    _condition = getattr(_prompts, "conditional_on", None)
    if system_prompt is None:
        system_prompt = getattr(_prompts, "system_prompt", None)
    if isinstance(system_prompt, dict):
        system_prompt = _resolve_conditional_prompt(
            prompt=system_prompt,
            sample=sample,
            conditional_on=_condition,
        )

    _task_prompt_applied = True
    if task_prompt is None:
        task_prompt = getattr(_prompts, "task_prompt", None)
    if isinstance(task_prompt, dict):
        task_prompt = _resolve_conditional_prompt(
            prompt=task_prompt,
            sample=sample,
            conditional_on=_condition,
        )
    if task_prompt:
        _task_prompt_applied = False
    
    # label
    label = None
    if sample.get("label", None):
        label = sample["label"]
        if isinstance(label, str):
            label = [label, ]
    elif sample.get("labels", None):
        label = sample["labels"]
        if isinstance(label, str):
            label = [label, ]   
    
    # options & option_contents
    options = sample.get("options", None)
    option_contents = sample.get("option_contents", None)
    if (
        not options
        and task_config.dataset.options
    ):
        options = task_config.dataset.options
    
    # circular evaluation
    if (
        options
        and option_contents
        and run_index > 0
    ):
        options, option_contents, label = shift_options(
            options=options,
            option_contents=option_contents,
            label=label,
            run_index=run_index,
        )
 
    messages = list()
    # add system message if not given
    if (
        isinstance(system_prompt, str)
        and len(system_prompt.strip()) > 0
    ):
        messages.append(ChatMessage(
            role="system", 
            content=[
                ChatTextContent(type="text", value=system_prompt),
            ],
        ))
    
    if "messages" in sample:
        for _message_idx, _message in enumerate(sample["messages"]):
            if isinstance(_message["content"], str):
                _message["content"] = [{
                    "type": "text",
                    "value": _message["content"],
                }]
            elif isinstance(_message["content"], dict):
                _message["content"] = [
                    _message["content"], 
                ]

            _content_list = None
            if isinstance(_message["content"], (list, tuple)):
                _content_list = list()
                for _content in _message["content"]:
                    _content_cls = CONTENT_ACCESSOR_MAP.get(_content["type"])
                    _value_key = _content_cls.get_key(_content) if _content_cls else None
                    if not _value_key:
                        raise ValueError(f'`content_value` not specified: {_content}')
                    _content_value = _content[_value_key]

                    if _content["type"] == "text":
                        if (
                            _message["role"] == "user"
                            and _message_idx == len(sample["messages"]) - 1
                        ): # last user_message
                            _content_value = format_task_prompt(
                                task_prompt=task_prompt,
                                query=_content_value,
                                **task_prompt_kwargs if isinstance(task_prompt_kwargs, dict) else dict(),
                            ).rstrip()
                            _task_prompt_applied = True
                        _content_list.append(ChatTextContent(
                            type="text",
                            value=_content_value,
                        ))
                    elif "audio" in _content["type"]:
                        _content_list.append(ChatAudioContent(
                            type="audio",
                            value=_content_value,
                        ))
                    elif "image" in _content["type"]:
                        _content_list.append(ChatImageContent(
                            type="image",
                            value=_content_value,
                            ocr=_content.get("ocr"),
                            entity=_content.get("entity"),
                        ))
                    elif "video" in _content["type"]:
                        _subtitle_raw = _content.get("subtitle", None)
                        _subtitle_resolved = None
                        if (
                            isinstance(_subtitle_raw, (list, tuple))
                            and isinstance(num_subtitle_cues, int)
                            and num_subtitle_cues != 0
                        ):
                            # Gate semantics: None or 0 → off (falls through to None above).
                            # Positive N → first N; negative → keep all.
                            if num_subtitle_cues > 0:
                                _subtitle_resolved = list(_subtitle_raw)[:num_subtitle_cues]
                            else:
                                _subtitle_resolved = list(_subtitle_raw)
                        _content_list.append(ChatVideoContent(
                            type="video",
                            value=_content_value,
                            subtitle=_subtitle_resolved,
                        ))
                    else:
                        raise ValueError(f'invalid content type: {_content["type"]}')

                if not _task_prompt_applied:
                    query = format_task_prompt(
                        task_prompt=task_prompt, 
                        query=None,
                    ) # remote placeholders
                    _content_list.append(ChatTextContent(
                        type="text",
                        value=query,
                    ))

            messages.append(ChatMessage(
                role=_message["role"],
                name=_message.get("name", None),
                content=_content_list,
                tool_call_id=_message.get("tool_call_id", None),
                tool_calls=_message.get("tool_calls", None),
                function_call=_message.get("function_call", None),
                annotations=_message.get("annotations", None),
            ))
            
    elif "query" in sample:        
        user_contents = list()
        
        # audio_index
        audio_index = sample.get("audio_index", None)
        if isinstance(audio_index, str):
            audio_index = [audio_index, ]
        if isinstance(audio_index, (list, tuple)):
            for _audio_idx, _audio in enumerate(audio_index):
                _content = {
                    "type": "audio",
                    "value": _audio,
                }
                user_contents.append(ChatAudioContent(**_content))
                
        # image_index
        ocr_tokens = sample.get("ocr", None)
        image_index = sample.get("image_index", None)
        if isinstance(image_index, str):
            image_index = [image_index, ]
        if isinstance(image_index, (list, tuple)):
            for _image_idx, _image in enumerate(image_index):
                _content = {
                    "type": "image",
                    "value": _image,
                }
                if (
                    _image_idx == 0
                    and isinstance(ocr_tokens, (list, tuple))
                    and isinstance(num_ocr_tokens, int)
                    and num_ocr_tokens != 0
                ):
                    # Gate semantics: None or 0 → off (skipped above by isinstance/!=0).
                    # Positive N → first N; negative → keep all (no truncation).
                    if num_ocr_tokens > 0:
                        ocr_tokens = ocr_tokens[:num_ocr_tokens]
                    _content["value"] = {
                        "image": _image,
                        "ocr": ocr_tokens,
                    }
                user_contents.append(ChatImageContent(**_content))
                
        # video_index
        video_index = sample.get("video_index", None)
        if isinstance(video_index, str):
            video_index = [video_index, ]
        if isinstance(video_index, (list, tuple)):
            for _video_idx, _video in enumerate(video_index):
                _content = {
                    "type": "video",
                    "value": _video,
                }
                user_contents.append(ChatVideoContent(**_content))

        query = format_task_prompt(
            task_prompt=task_prompt,
            query=sample["query"],
        ).rstrip()
        user_contents.append(ChatTextContent(**{
            "type": "text",
            "value": query,
        }))
        messages.append(ChatMessage(
            role="user",
            content=user_contents,
        ))
    else:
        raise ValueError(f'`messages` or `query` should be included in dataset')
    
    # generation_options
    generation_options = dict() # generation_options is not given
    if (
        isinstance(task_config.inference, TaskInference)
        and isinstance(task_config.inference.generation_options, TaskInferenceGenerationOptions)
    ):
        generation_options.update(task_config.inference.generation_options.to_dict())

    # common
    sample_meta = {
        "task_name": task_name,
    }
    if task_config.meta:
        sample_meta.update(task_config.meta.to_dict())
    if sample.get("meta", None):
        sample_meta.update(sample["meta"])
    
    record = Record(
        benchmark=task_name,
        index=index,
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
    if task_config.meta.num_sample_repetition:
        record = [record, ] * task_config.meta.num_sample_repetition
    return record