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
import ast
import dotenv
import os
import sys
from omni_evaluator.args import get_parser, CustomArgumentParser
from omni_evaluator.evaluate import main as main__evaluate
from omni_evaluator.utils.io import read_file
import logging
logger = logging.getLogger(__name__)

# Load .env, skipping empty values so we never override library defaults
# (e.g. HF_ENDPOINT, TRANSFORMERS_CACHE) with "". Respects override=False semantics.
for _k, _v in dotenv.dotenv_values().items():
    if (
        not _v 
        or _k in os.environ
    ):
        continue
    os.environ[_k] = _v

# Mirror proxy env var case — some libs (requests/curl idiom) read lowercase, others uppercase.
for _upper, _lower in (("HTTP_PROXY", "http_proxy"), ("HTTPS_PROXY", "https_proxy")):
    if os.environ.get(_upper) and not os.environ.get(_lower):
        os.environ[_lower] = os.environ[_upper]
    elif os.environ.get(_lower) and not os.environ.get(_upper):
        os.environ[_upper] = os.environ[_lower]

if __name__ == "__main__":
    artifact_filepath = None
    for _idx, _arg in enumerate(sys.argv):
        if "artifact_filepath" not in _arg:
            continue
        if (
            _idx >= len(sys.argv) - 1
            or sys.argv[_idx+1].startswith("--")
        ): # arg value is included in current arg
            artifact_filepath = _arg.split("=")[-1].strip()
        else:
            # arg value is included in next arg
            artifact_filepath = sys.argv[_idx+1].strip()
    
    if artifact_filepath:
        CustomArgumentParser.restore_arguments(
            artifact_filepath=artifact_filepath,
        )
        logger.info(f'restored artifact: {artifact_filepath}')
    
    parser = CustomArgumentParser()
    parser, validations = get_parser(parser=parser)
    args = parser.parse_args()
    for _validation_func in validations:
        args = _validation_func(args=args)
    main__evaluate(args)