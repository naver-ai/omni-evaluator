# Reference from https://github.com/EvolvingLMMs-Lab/lmms-eval (Apache-2.0)
  # ConfigurableTask__download is adapted from lmms-eval's ConfigurableTask.download()
  # in lmms_eval/api/task.py (video download / unzip / untar / From_YouTube handling),
  # with custom parquet & snapshot resolution added by omni_evaluator.

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

from accelerate import Accelerator
import datasets
from datasets import Audio, DownloadConfig, Image, Sequence
import importlib
from glob import glob
from huggingface_hub import snapshot_download
import json
import logging
from loguru import logger as eval_logger
import numpy as np
import os
from pathlib import Path, PosixPath
import shutil
import subprocess
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed
from tqdm import tqdm
import traceback
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Tuple, Union
import yaml

logger = logging.getLogger(__name__)


CUSTOM_TASKS = None
CUSTOM_SNAPSHOT_PATHS = None


def _load_custom_resources():
    global CUSTOM_TASKS, CUSTOM_SNAPSHOT_PATHS
    if CUSTOM_TASKS is not None:
        return
    CUSTOM_TASKS = dict()
    for _dirpath in importlib.resources.files(
        "omni_evaluator.evaluation.lmms_eval.resources.custom_parquets",
    ).iterdir(): # update custom task_names which requires custom_parquet_paths
        if not Path(_dirpath).is_dir():
            continue
        _task_name = Path(_dirpath).name
        if _task_name not in CUSTOM_TASKS:
            CUSTOM_TASKS[_task_name] = dict()
        CUSTOM_TASKS[_task_name]["custom_parquets"] = True

    with importlib.resources.files(
        "omni_evaluator.evaluation.lmms_eval.resources",
    ).joinpath("custom_snapshots.yaml").open("rb") as fp:
        CUSTOM_SNAPSHOT_PATHS = yaml.safe_load(fp)
    

@retry(stop=(stop_after_attempt(5) | stop_after_delay(60)), wait=wait_fixed(2))
def ConfigurableTask__download(self, dataset_kwargs=None) -> None:
    _load_custom_resources()
    # If the dataset is a video dataset,
    # Recursively search whether their is a zip and unzip it to the huggingface home
    download_config = DownloadConfig()
    download_config.max_retries = dataset_kwargs.get("max_retries", 10) if dataset_kwargs is not None else 10
    download_config.num_proc = dataset_kwargs.get("num_proc", 8) if dataset_kwargs is not None else 8
    download_config.local_files_only = dataset_kwargs.get("local_files_only", False) if dataset_kwargs is not None else False
    if dataset_kwargs is not None:
        if "From_YouTube" in dataset_kwargs:

            def _download_from_youtube(path):
                try:
                    for video in tqdm(self.all_dataset[split]):
                        video_id = video["videoID"]
                        target_path = os.path.join(path, f"{video_id}.mp4")
                        if shutil.which("yt-dlp") is None:
                            raise RuntimeError("yt-dlp must be installed and available in the system's PATH")
                        command = [
                            "yt-dlp", "-o", target_path, "-f", "mp4",
                            f"https://www.youtube.com/watch?v={video_id}",
                        ]
                        subprocess.run(command)
                    with open(os.path.join(cache_path, f"{task}_download_status.json"), "w") as f:
                        f.write(json.dumps({task: "downloaded"}))
                except Exception as e:
                    eval_logger.error(f"Error while downloading {task} data: {e}")
                    with open(os.path.join(cache_path, f"{task}_download_status.json"), "w") as f:
                        f.write(json.dumps({task: "not downloaded"}))

            hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
            accelerator = Accelerator()
            if accelerator.is_main_process:
                dataset_kwargs.pop("From_YouTube")
                if "load_from_disk" in dataset_kwargs:
                    raise ValueError("load_from_disk must not be True when From_YouTube is True")
                self.all_dataset = datasets.load_dataset(
                    path=self.DATASET_PATH,
                    name=self.DATASET_NAME,
                    download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
                    **dataset_kwargs if dataset_kwargs is not None else {},
                )
                dataset_kwargs["From_YouTube"] = True
                cache_path = snapshot_download(repo_id=self.DATASET_PATH, repo_type="dataset", cache_dir=os.environ["HF_HOME"])  # download_parquet
                split = vars(self.config)["test_split"]
                task = vars(self.config)["task"]

                video_path = os.path.join(hf_home, task)
                if os.path.exists(os.path.join(cache_path, f"{task}_download_status.json")):
                    with open(os.path.join(cache_path, f"{task}_download_status.json"), "r") as _f:
                        download_status = json.load(_f)
                    if download_status[task] == "downloaded":
                        eval_logger.info(f"Data for {task} already download!")
                    else:
                        eval_logger.info(f"Start downloading YouTube data to {video_path}...")
                        _download_from_youtube(video_path)
                else:
                    eval_logger.info(f"Start downloading YouTube data to {video_path}...")
                    _download_from_youtube(video_path)

            accelerator.wait_for_everyone()
            if "builder_script" in dataset_kwargs:
                builder_script = dataset_kwargs["builder_script"]
                self.DATASET_PATH = os.path.join(cache_path, builder_script)
                dataset_kwargs.pop("builder_script")

            downloaded_video_ids = [i.split(".mp4")[0] for i in os.listdir(os.path.expanduser(video_path)) if i.endswith(".mp4")]
            # Filtered the existing dataset with the downloaded video ids
            self.dataset = datasets.DatasetDict({split: self.all_dataset[split].filter(lambda x: x["videoID"] in downloaded_video_ids)})

            self.dataset_no_image = self.dataset
            dataset_kwargs.pop("From_YouTube")
            return

        if "video" in dataset_kwargs and dataset_kwargs["video"]:
            hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
            hf_home = os.path.expanduser(hf_home)
            cache_dir = dataset_kwargs["cache_dir"]
            cache_dir = os.path.join(hf_home, cache_dir)
            accelerator = Accelerator()
            if accelerator.is_main_process:
                force_download = dataset_kwargs.get("force_download", False)
                force_unzip = dataset_kwargs.get("force_unzip", False)
                revision = dataset_kwargs.get("revision", "main")
                create_link = dataset_kwargs.get("create_link", False)
                cache_path = snapshot_download(
                    repo_id=self.DATASET_PATH, 
                    revision=revision, 
                    repo_type="dataset", 
                    force_download=force_download, 
                    etag_timeout=60,
                    cache_dir=os.environ["HF_HOME"],
                )
                zip_files = glob(os.path.join(cache_path, "**/*.zip"), recursive=True)
                tar_files = glob(os.path.join(cache_path, "**/*.tar*"), recursive=True)

                def unzip_video_data(zip_file):
                    import os
                    import zipfile

                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        for file_info in zip_ref.infolist():
                            target_path = os.path.join(cache_dir, file_info.filename)
                            if not os.path.exists(target_path):
                                zip_ref.extract(file_info, cache_dir)
                            else:
                                eval_logger.info(f"Skipping existing file: {target_path}")

                    eval_logger.info(f"Extracted all files from {zip_file} to {cache_dir}")

                def untar_video_data(tar_file):
                    import tarfile

                    with tarfile.open(tar_file, "r") as tar_ref:
                        tar_ref.extractall(cache_dir)
                        eval_logger.info(f"Extracted all files from {tar_file} to {cache_dir}")

                def concat_tar_parts(tar_parts, output_tar):
                    with open(output_tar, "wb") as out_tar:
                        from tqdm import tqdm

                        for part in tqdm(sorted(tar_parts)):
                            with open(part, "rb") as part_file:
                                out_tar.write(part_file.read())
                    eval_logger.info(f"Concatenated parts {tar_parts} into {output_tar}")

                # Unzip zip files if needed
                if force_unzip or (not os.path.exists(cache_dir) and len(zip_files) > 0):
                    for zip_file in zip_files:
                        unzip_video_data(zip_file)

                # Concatenate and extract tar files if needed
                if force_unzip or (not os.path.exists(cache_dir) and len(tar_files) > 0):
                    tar_parts_dict = {}

                    # Group tar parts together
                    for tar_file in tar_files:
                        base_name = tar_file.split(".tar")[0]
                        if base_name not in tar_parts_dict:
                            tar_parts_dict[base_name] = []
                        tar_parts_dict[base_name].append(tar_file)

                    # Concatenate and untar split parts
                    for base_name, parts in tar_parts_dict.items():
                        eval_logger.info(f"Extracting following tar files: {parts}")
                        output_tar = base_name + ".tar"
                        if not os.path.exists(output_tar):
                            eval_logger.info(f"Start concatenating tar files")

                            concat_tar_parts(parts, output_tar)
                            eval_logger.info(f"Finish concatenating tar files")

                        if not os.path.exists(os.path.join(cache_dir, os.path.basename(base_name))):
                            untar_video_data(output_tar)

                # Link cache_path to cache_dir if needed.
                if create_link:
                    if not os.path.exists(cache_dir) or os.path.islink(cache_dir):
                        if os.path.islink(cache_dir):
                            os.remove(cache_dir)
                            eval_logger.info(f"Removed existing symbolic link: {cache_dir}")
                        # Create a new symbolic link
                        os.symlink(cache_path, cache_dir)
                        eval_logger.info(f"Symbolic link created successfully: {cache_path} -> {cache_dir}")

            accelerator.wait_for_everyone()
            dataset_kwargs.pop("cache_dir")
            dataset_kwargs.pop("video")

        if "builder_script" in dataset_kwargs:
            builder_script = dataset_kwargs["builder_script"]
            self.DATASET_PATH = os.path.join(cache_path, builder_script)
            dataset_kwargs.pop("builder_script")

        if "force_download" in dataset_kwargs:
            dataset_kwargs.pop("force_download")

        if "force_unzip" in dataset_kwargs:
            dataset_kwargs.pop("force_unzip")

        if "local_files_only" in dataset_kwargs:
            dataset_kwargs.pop("local_files_only")

        if "create_link" in dataset_kwargs:
            dataset_kwargs.pop("create_link")

    if (
        dataset_kwargs is not None 
        and dataset_kwargs.get("load_from_disk", None) is not None
    ):
        dataset_kwargs.pop("load_from_disk")
        # using local task in offline environment, need to process the online dataset into local format via
        # `ds = load_datasets("lmms-lab/MMMU")`
        self.dataset = datasets.load_from_disk(
            path=self.DATASET_PATH, 
            name=self.DATASET_NAME,
        )
    elif (
        dataset_kwargs is not None 
        and dataset_kwargs.get("custom_parquet_path", None) is not None
    ):
        try:
            custom_parquet_path = dataset_kwargs.get("custom_parquet_path", None)
            if not os.path.exists(custom_parquet_path):
                custom_parquet_path = importlib.resources.files(
                    "omni_evaluator.evaluation.lmms_eval.resources.custom_parquets",
                ).joinpath(custom_parquet_path)
                if isinstance(custom_parquet_path, PosixPath):
                    custom_parquet_path = Path.as_posix(custom_parquet_path)
            custom_dataset = datasets.Dataset.from_parquet(custom_parquet_path)
            self.dataset = datasets.DatasetDict({
                "test": custom_dataset
            })
        except Exception as ex:
            logger.error(f'Failed to download lmms-eval task: datasets.DatasetDict from custom_parquet_path: {custom_parquet_path}')
            traceback.print_exc()
            raise
    else:
        DATASET_PATH = self.DATASET_PATH
        if CUSTOM_SNAPSHOT_PATHS and self.DATASET_PATH in CUSTOM_SNAPSHOT_PATHS:
            # use custom snapshot if exists
            for _path in CUSTOM_SNAPSHOT_PATHS[self.DATASET_PATH]:
                if os.path.exists(_path):
                    DATASET_PATH = _path
                    break
            
        try:
            self.dataset = datasets.load_dataset(
                path=DATASET_PATH,
                name=self.DATASET_NAME,
                download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
                download_config=download_config,
                **dataset_kwargs if dataset_kwargs is not None else {},
            )
        except Exception as ex:
            logger.error(f'Failed to download lmms-eval task: datasets.load_dataset: {DATASET_PATH}')
            traceback.print_exc()
            raise

    if self.config.process_docs is not None:
        for split in self.dataset:
            if split in [self.config.training_split, self.config.validation_split, self.config.test_split, self.config.fewshot_split]:
                self.dataset[split] = self.config.process_docs(self.dataset[split])

    # copy dataset, remove image features
    self.dataset_no_image = self.dataset.copy()
    for doc_name in self.dataset_no_image:
        remove_cols = []
        features = self.dataset_no_image[doc_name].features
        # If it is an Image instance or a Sequence of Image instance. Remove it
        for feature in features:
            if isinstance(features[feature], Image):
                remove_cols.append(feature)
            elif isinstance(features[feature], Sequence) and isinstance(features[feature].feature, Image):
                remove_cols.append(feature)
            elif isinstance(features[feature], Audio):
                remove_cols.append(feature)
        for remove_col in remove_cols:
            self.dataset_no_image[doc_name] = self.dataset_no_image[doc_name].remove_columns(remove_col)