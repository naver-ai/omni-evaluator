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

import boto3
import botocore
import io
import json
import logging
import multiprocessing
import os
from pathlib import Path, PosixPath
import pickle
import pyarrow
import pyarrow.csv
import pyarrow.fs
import shutil
import time
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Union, Optional
import yaml

from omni_evaluator.utils.io import read_file, write_file

logger = logging.getLogger(__name__)

GLOBAL_BOTO_CLIENT = None

class S3Client:
    """Client for interacting with S3-compatible object storage (upload, download, list, delete)."""

    def __init__(
        self,
        bucket_name: str,
        access_key: str,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = "kr-standard",
        verbose: bool = True,
    ) -> None:
        os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
        os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
        self.bucket_name = bucket_name
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url
        self.region = region
        self.verbose = verbose
        logger.debug(f'S3Client config:')
        logger.debug(f'  bucket_name: {bucket_name}')
        logger.debug(f'  access_key: {access_key}')
        logger.debug(f'  secret_key: {secret_key}')
        logger.debug(f'  endpoint_url: {endpoint_url}')
        logger.debug(f'  region: {region}')
        
        self.pyarrow_client = pyarrow.fs.S3FileSystem(
            access_key=access_key,
            secret_key=secret_key,
            endpoint_override=endpoint_url,
            region=region,
        )
        logger.debug(f'Defined s3client.pyarrow_client')
        
        self.boto_client, self.boto_client_config = _init_boto_client(
            access_key=access_key, 
            secret_key=secret_key, 
            endpoint_url=endpoint_url,
            region=region,
            update_global_worker=False,
            max_attempts=5,
            mode="adaptive",
            connect_timeout=60,
            read_timeout=300,
        )
        logger.debug(f'Defined s3client.boto_client: {self.boto_client_config}')
    
    @staticmethod
    def _accumulate_result(output: dict, result: dict) -> None:
        """Merge a single file operation *result* into the running *output* summary dict."""
        if result is None:
            return
        _file_size = result.get("file_size", 0)
        if result.get("skipped", False):
            output["num_skipped"] += 1
        if result.get("status", False):
            output["num_passed"] += 1
        else:
            output["num_failed"] += 1
            if output["error_message"] is None:
                output["error_message"] = list()
            output["error_message"].append(result.get("error_message", None))
        output["total_bytes"] += _file_size

    def list(
        self,
        remote_dirpath: str,
    ) -> List[str]:
        """List immediate children (files and directories) under *remote_dirpath*."""
        if (
            len(remote_dirpath) > 0
            and not remote_dirpath.endswith("/")
        ):
            remote_dirpath += "/"

        output = list()
        paginator = self.boto_client.get_paginator("list_objects_v2")
        for _page in paginator.paginate(
            Bucket=self.bucket_name, 
            Prefix=remote_dirpath, 
            Delimiter="/",
        ):
            # child dirpath
            for _common_prefix in _page.get("CommonPrefixes", list()):
                _filename_or_dirname = _common_prefix["Prefix"]
                if _filename_or_dirname.startswith(remote_dirpath):
                    _filename_or_dirname = _filename_or_dirname[len(remote_dirpath):]
                if _filename_or_dirname.endswith("/"):
                    _filename_or_dirname = _filename_or_dirname[:-1]
                output.append(_filename_or_dirname)  # e.g. "data/2025-09-14/"
                
            # child filepath
            for _content in _page.get("Contents", list()):
                _filename_or_dirname = _content["Key"]
                if _filename_or_dirname.startswith(remote_dirpath):
                    _filename_or_dirname = _filename_or_dirname[len(remote_dirpath):]
                if _filename_or_dirname.endswith("/"):
                    _filename_or_dirname = _filename_or_dirname[:-1]
                output.append(_filename_or_dirname)  # e.g. "data/2025-09-14/info.json"
        return output

    def upload_obj(
        self,
        obj: Any,
        remote_filepath: str,
        encoding: str = "utf-8",
    ) -> None:
        """Serialize and upload *obj* to *remote_filepath* in the bucket."""
        if isinstance(remote_filepath, PosixPath):
            remote_filepath = Path.as_posix(remote_filepath)
        remote_filepath = os.path.join(
            self.bucket_name,
            remote_filepath,
        )
        with self.pyarrow_client.open_output_stream(remote_filepath) as fp:
            if remote_filepath.endswith(".txt"):
                obj = obj.encode(encoding)
                fp.write(obj)
            elif remote_filepath.endswith(".pickle"):
                obj = pickle.dumps(obj)
                fp.write(obj)
            elif remote_filepath.endswith(".json"):
                obj = json.dumps(obj, ensure_ascii=False, indent=2)
                obj = obj.encode(encoding)
                fp.write(obj)
            elif remote_filepath.endswith(".jsonl"):
                if not isinstance(obj, (list, tuple)):
                    raise TypeError(f'Check if obj is list or tuple: {type(obj)}')
                with io.TextIOWrapper(fp, encoding="utf-8", write_through=True) as _fp:
                    for _row in obj:
                        _fp.write(json.dumps(_row, ensure_ascii=False) + "\n")
            elif remote_filepath.endswith(".yaml"):
                obj = yaml.safe_dump(obj, allow_unicode=True, sort_keys=False)
                obj = obj.encode(encoding)
                fp.write(obj)
            else:
                fp.write(obj)
                # raise ValueError(f'# [error] unsupported file type: {Path(remote_filepath).suffix}')
                
    def upload_file(
        self,
        filepath: str,
        remote_dirpath: str,
        encoding: str = "utf-8",
    ) -> None:
        """Read a local file and upload it to *remote_dirpath*."""
        if isinstance(filepath, PosixPath):
            filepath = Path.as_posix(filepath)
        remote_filepath = os.path.join(
            remote_dirpath, 
            Path(filepath).name,
        )
        obj = read_file(filepath=filepath)
        return self.upload_obj(
            obj=obj,
            remote_filepath=remote_filepath,
            encoding=encoding,
        )
    
    def download_obj(
        self,
        remote_filepath: str,
        encoding: str = "utf-8",
    ) -> Any:
        """Download and deserialize an object from *remote_filepath*."""
        if isinstance(remote_filepath, PosixPath):
            remote_filepath = Path.as_posix(remote_filepath)
        remote_filepath = os.path.join(
            self.bucket_name,
            remote_filepath,
        )

        obj = None
        with self.pyarrow_client.open_input_stream(remote_filepath) as fp:
            if remote_filepath.endswith(".txt"):
                obj = fp.read().decode(encoding)
            elif remote_filepath.endswith(".pickle"):
                logger.warning(
                    f"Loading pickle from remote: {remote_filepath}. "
                    "Pickle deserialization can execute arbitrary code. "
                    "Only load pickle files from trusted sources."
                )
                obj = pickle.loads(fp.read())
            elif remote_filepath.endswith(".json"):
                obj = fp.read().decode(encoding)
                obj = json.loads(obj)
            elif remote_filepath.endswith(".jsonl"):
                obj = list()
                with io.TextIOWrapper(fp, encoding="utf-8") as _fp:
                    for _row in _fp:
                        _row = _row.strip()
                        if not _row:
                            continue
                        _row = json.loads(_row)
                        obj.append(_row)
            elif remote_filepath.endswith(".yaml"):
                obj = fp.read().decode(encoding)
                obj = yaml.safe_load(obj)
            else:
                obj = fp.read()
        return obj
        
    def download_file(
        self,
        filepath: str,
        remote_filepath: str,
        encoding: str = "utf-8",
    ) -> None:
        """Download a remote file to a local *filepath*."""
        if isinstance(filepath, PosixPath):
            filepath = Path.as_posix(filepath)
        if not Path(filepath).parent.exists():
            Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        remote_filepath = os.path.join(
            self.bucket_name,
            remote_filepath,
        )
        with self.pyarrow_client.open_input_stream(remote_filepath) as src_fp, open(filepath, "wb") as dst_fp:
            shutil.copyfileobj(src_fp, dst_fp)
    
    def upload_dir(
        self,
        dirpath: str,
        remote_dirpath: str,
        num_process: int = 8,
        resume: Optional[bool] = True,
        include_hidden: Optional[bool] = False,
        extra_args: Optional[Dict[str, Any]] = None, # e.g. {"ACL":"private"} 
    ):
        """
        Upload all files under dirpath to s3://bucket/remote_dirpath preserving relative paths.
        """
        dirpath = Path.as_posix(Path(dirpath).resolve())
        if not os.path.isdir(dirpath):
            raise ValueError(f"Dirpath is not a directory: {dirpath}")

        remote_dirpath = Path(remote_dirpath).as_posix()
        if (
            len(remote_dirpath) > 0 
            and not remote_dirpath.endswith("/")
        ):
            remote_dirpath += "/"

        keys = list()
        file_sizes = list()
        filepaths = list()
        for _path in Path(dirpath).rglob("*"):
            if not _path.is_file():
                continue
            if not include_hidden:
                _relative_parts = _path.relative_to(dirpath).parts
                if any([_part.startswith(".") for _part in _relative_parts]):
                    continue

            _relative_path = _path.relative_to(dirpath).as_posix()
            _key = os.path.join(remote_dirpath, _relative_path)
            _file_size = _path.stat().st_size
            keys.append(_key)
            file_sizes.append(_file_size)
            filepaths.append(Path(_path).as_posix())

        logger.info(f'Found {len(file_sizes)} files (Total: {sum(file_sizes) / (1024 ** 3):.2f}GiB)')
        output = {
            "success_rate": None,
            "execution_time": None,
            "num_files": len(keys),
            "num_passed": 0,
            "num_failed": 0,
            "num_skipped": 0,
            "total_bytes": 0.0,
            "error_message": None,
        }
        _start_time = time.time()
        if num_process > 1:
            with multiprocessing.Pool(
                processes=num_process,
                initializer=_init_boto_client,
                initargs=(
                    self.access_key,
                    self.secret_key,
                    self.endpoint_url,
                    self.region,
                    True, # update_global_worker
                ),
            ) as pool:
                tasks = []
                for _key, _filepath, _file_size in zip(keys, filepaths, file_sizes):
                    _task = pool.apply_async(
                        _upload_file_boto,
                        (
                            self.bucket_name,
                            _key,
                            _filepath,
                            _file_size,
                            resume,
                            extra_args,
                        ),
                    )
                    tasks.append(_task)

                for task in tqdm(
                    tasks, 
                    initial=0,
                    total=len(tasks), 
                    desc=f'Uploading to {self.bucket_name}: {dirpath}',
                ):
                    _result = task.get()
                    self._accumulate_result(output, _result)
        else:
            for _key, _filepath, _file_size in zip(keys, filepaths, file_sizes):
                _result = _upload_file_boto(
                    bucket_name=self.bucket_name,
                    key=_key,
                    filepath=_filepath,
                    file_size=_file_size,
                    resume=resume,
                    extra_args=extra_args,
                    client=self.boto_client,
                )
                self._accumulate_result(output, _result)

        output["success_rate"] = (
            output["num_passed"] / output["num_files"]
            if output["num_files"] > 0
            else 0.0
        )
        output["execution_time"] = time.time() - _start_time
        return output

    def download_dir(
        self,
        dirpath: str,
        remote_dirpath: str, 
        num_process: Optional[int] = 8,
        resume: Optional[bool] = True,
    ):        
        if isinstance(dirpath, PosixPath):
            dirpath = Path.as_posix(dirpath)
        dirpath = Path(dirpath).resolve() # to abs_path
        if not Path(dirpath).exists():
            Path(dirpath).mkdir(exist_ok=True, parents=True)
        if isinstance(remote_dirpath, PosixPath):
            remote_dirpath = Path.as_posix(remote_dirpath)
        remote_dirpath = Path.as_posix(Path(remote_dirpath))
        if len(remote_dirpath) > 0 and not remote_dirpath.endswith("/"):
            remote_dirpath += "/"

        keys = list()
        file_sizes = list()
        paginator = self.boto_client.get_paginator("list_objects_v2")
        for _page in paginator.paginate(
            Bucket=self.bucket_name, 
            Prefix=remote_dirpath,
        ):
            for _obj in _page.get("Contents", list()):
                _key = _obj["Key"]
                if _key.endswith("/"): # sub_directory
                    continue
                if not _key.startswith(remote_dirpath): # invalid key
                    continue
                _size = _obj.get("Size", 0)
                keys.append(_key)
                file_sizes.append(_size)
        
        logger.info(f'Found {len(file_sizes)} files (Total: {sum(file_sizes) / (1024 ** 3):.2f}GiB)')
        output = {
            "success_rate": None,
            "execution_time": None,
            "num_files": len(keys),
            "num_passed": 0,
            "num_failed": 0,
            "num_skipped": 0,
            "total_bytes": 0.0,
            "error_message": None,
        }
        _start_time = time.time()
        if num_process > 1:
            with multiprocessing.Pool(
                processes=num_process,
                initializer=_init_boto_client,
                initargs=(
                    self.access_key,
                    self.secret_key,
                    self.endpoint_url,
                    self.region,
                    True, # update_global_worker
                ),
            ) as pool:
                tasks = list()
                for _key, _file_size in zip(keys, file_sizes):
                    _relative_path = _key[len(remote_dirpath):]
                    _filepath = os.path.join(dirpath, _relative_path)
                    _task = pool.apply_async(
                        _download_file_boto,
                        (
                            self.bucket_name,
                            _key,
                            _filepath,
                            _file_size,
                            resume,
                        )
                    )
                    tasks.append(_task)
                    
                for task in tqdm(
                    tasks,
                    initial=0,
                    total=len(tasks),
                    desc=f'Downloading from {self.bucket_name}: {dirpath}',
                ):
                    _result = task.get()
                    self._accumulate_result(output, _result)

        else:
            for _key, _file_size in zip(keys, file_sizes):
                _relative_path = _key[len(remote_dirpath):]
                _filepath = os.path.join(dirpath, _relative_path)
                _result = _download_file_boto(
                    bucket_name=self.bucket_name,
                    key=_key,
                    filepath=_filepath,
                    file_size=_file_size,
                    resume=resume,
                    client=self.boto_client,
                )
                self._accumulate_result(output, _result)
        
        output["success_rate"] = (
            output["num_passed"] / output["num_files"]
            if output["num_files"] > 0
            else 0.0
        )
        output["execution_time"] = time.time() - _start_time
        return output

    def delete_file(
        self,
        remote_filepath: str,
    ) -> None:
        """Delete a single file from the bucket."""
        remote_filepath = os.path.join(
            self.bucket_name,
            remote_filepath,
        )
        self.pyarrow_client.delete_file(remote_filepath)
        
    def delete_dir(
        self,
        remote_dirpath: str,
    ) -> None:
        """Recursively delete all files under *remote_dirpath*."""
        if (
            len(remote_dirpath) > 0
            and not remote_dirpath.endswith("/")
        ):
            remote_dirpath += "/"

        keys = list()
        paginator = self.boto_client.get_paginator("list_objects_v2")
        for _page in paginator.paginate(
            Bucket=self.bucket_name, 
            Prefix=remote_dirpath,
        ):
            for _obj in _page.get("Contents", list()):
                _key = _obj["Key"]
                if _key.endswith("/"):
                    continue
                keys.append(_key)

        for _key in tqdm(
            keys,
            initial=0,
            total=len(keys),
            desc=f'Deleting {remote_dirpath} in {self.bucket_name}',
        ):
            self.boto_client.delete_object(Bucket=self.bucket_name, Key=_key)

    def get_presigned_url(
        self,
        remote_filepath: str,
        duration: int = 3600,
    ) -> str:
        """Generate a time-limited presigned URL for *remote_filepath*."""
        url = self.boto_client.generate_presigned_url(
            "get_object", 
            Params={
                "Bucket": self.bucket_name, 
                "Key": remote_filepath,
                "ResponseContentDisposition": f'attachment; filename="{remote_filepath}"',
            },
            ExpiresIn=duration,
        )
        return url

    def __del__(
        self,
    ):
        try:
            if self.pyarrow_client is not None:
                self.pyarrow_client.close()
        except Exception as ex:
            pass
        try:
            if self.boto_client is not None:
                self.boto_client.close()
        except Exception as ex:
            pass

def _init_boto_client( 
    access_key: str, 
    secret_key: str, 
    endpoint_url: str,
    region: str,
    update_global_worker: Optional[bool] = False,
    max_attempts: Optional[int] = 5,
    mode: Optional[str] = "adaptive",
    connect_timeout: Optional[int] = 60,
    read_timeout: Optional[int] = 300,
):
    client_config = botocore.config.Config(
        retries={
            "max_attempts": max_attempts, 
            "mode": mode,
        },
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )
    client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        region_name=region,
        config=client_config,
    )
    if update_global_worker:
        global GLOBAL_BOTO_CLIENT
        GLOBAL_BOTO_CLIENT = client
    return client, client_config

def _download_file_boto(
    bucket_name: str,
    key: str,
    filepath: str,
    file_size: Optional[int] = None,
    resume: Optional[bool] = False,
    client = None,
):
    global GLOBAL_BOTO_CLIENT
    if GLOBAL_BOTO_CLIENT is None:
        GLOBAL_BOTO_CLIENT = client

    output = {
        "status": True, 
        "skipped": True,
        "filepath": filepath, 
        "key": key,
        "file_size": file_size,
        "error_message": None,
    }
    
    if not Path(filepath).parent.exists():
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
    
    if (
        not resume 
        or not os.path.exists(filepath)
        or file_size != os.path.getsize(filepath)
    ):
        output["skipped"] = False
        try:
            GLOBAL_BOTO_CLIENT.download_file(
                Bucket=bucket_name, 
                Key=key, 
                Filename=filepath,
            )
        except Exception as ex:
            output["status"] = False
            output["error_message"] = str(ex)
    return output

def _upload_file_boto(
    bucket_name: str,
    key: str,
    filepath: str,
    file_size: Optional[int] = None,
    resume: bool = False,
    extra_args: Optional[Dict[str, Any]] = None,
    client = None,
):
    def _get_head_size(
        client,
        bucket_name: str, 
        key: str,
    ):
        file_size = None
        try:
            _response = client.head_object(Bucket=bucket_name, Key=key)
            file_size = int(_response.get("ContentLength", 0))
        except Exception as ex:
            pass
        return file_size
    
    global GLOBAL_BOTO_CLIENT
    if GLOBAL_BOTO_CLIENT is None:
        GLOBAL_BOTO_CLIENT = client

    output = {
        "status": True, 
        "skipped": True,
        "filepath": filepath, 
        "key": key,
        "file_size": file_size,
        "error_message": None,
    }

    if not os.path.exists(filepath):
        output["status"] = False
        output["skipped"] = False
        output["error_message"] = "local file not found"
        return output

    if file_size is None:
        try:
            file_size = os.path.getsize(filepath)
            output["file_size"] = file_size
        except Exception as ex:
            output["status"] = False
            output["skipped"] = False
            output["error_message"] = str(ex)
            return output

    if (
        not resume
        or file_size != _get_head_size(
            GLOBAL_BOTO_CLIENT,
            bucket_name=bucket_name, 
            key=key,
        )
    ):
        output["skipped"] = False
        try:
            GLOBAL_BOTO_CLIENT.upload_file(
                Filename=filepath,
                Bucket=bucket_name,
                Key=key,
                ExtraArgs=extra_args or dict(),
            )
        except Exception as ex:
            output["status"] = False
            output["error_message"] = str(ex)

    return output