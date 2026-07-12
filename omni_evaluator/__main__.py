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
import logging
import os
import sys
from typing import List

import dotenv

from omni_evaluator import EvaluationEngine
from omni_evaluator.utils.common import list_inference_engines, list_evaluation_engines, list_tasks

logger = logging.getLogger(__name__)

# Load .env, skipping empty values so we never override library defaults
# (e.g. HF_ENDPOINT, TRANSFORMERS_CACHE) with "". Respects override=False semantics.
for _k, _v in dotenv.dotenv_values().items():
    if _v and _k not in os.environ:
        os.environ[_k] = _v

# Mirror proxy env var case — some libs (requests/curl idiom) read lowercase, others uppercase.
for _upper, _lower in (("HTTP_PROXY", "http_proxy"), ("HTTPS_PROXY", "https_proxy")):
    if os.environ.get(_upper) and not os.environ.get(_lower):
        os.environ[_lower] = os.environ[_upper]
    elif os.environ.get(_lower) and not os.environ.get(_upper):
        os.environ[_upper] = os.environ[_lower]


def main() -> None:
    """CLI entry point for ``python -m omni_evaluator``."""
    parser = argparse.ArgumentParser(
        prog="python -m omni_evaluator",
        description="HyperCLOVA-VLM-Evaluator CLI",
    )
    subparsers = parser.add_subparsers(
        dest="command",
    )

    subparser_list = subparsers.add_parser(
        "list",
        help="list inference_engines, evaluation_engines, or tasks",
    )
    subparser_list.add_argument(
        "--inference_engines",
        action="store_true",
        default=False,
        help="list available inference engines",
    )
    subparser_list.add_argument(
        "--evaluation_engines",
        action="store_true",
        default=False,
        help="list available evaluation engines",
    )
    subparser_list.add_argument(
        "--tasks",
        action="store_true",
        default=False,
        help="list available tasks for a given evaluation engine",
    )
    subparser_list.add_argument(
        "--evaluation_engine",
        type=str,
        required=False,
        choices=[engine.value for engine in EvaluationEngine],
        help="evaluation engine to list tasks for (used with --tasks)",
    )

    subparsers.add_parser(
        "evaluate",
        help="run evaluation",
        add_help=False,
    )

    # no arguments → print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args, remaining = parser.parse_known_args()

    if args.command == "list":
        _run_list(subparser_list, args)

    elif args.command == "evaluate":
        _run_evaluate(remaining)

    else:
        parser.print_help(sys.stderr)
        sys.exit(1)


def _run_list(subparser_list: argparse.ArgumentParser, args: argparse.Namespace) -> list:
    """Handle the ``list`` subcommand — print available engines or tasks."""
    if not (args.inference_engines or args.evaluation_engines or args.tasks):
        subparser_list.print_help(sys.stderr)
        sys.exit(1)

    if args.inference_engines:
        result = list_inference_engines()
    elif args.evaluation_engines:
        result = list_evaluation_engines()
    elif args.tasks:
        if not args.evaluation_engine:
            subparser_list.error("--evaluation_engine is required when using --tasks")
        result = list_tasks(evaluation_engine=args.evaluation_engine)

    print(result)
    return result


def _run_evaluate(remaining: List[str]) -> None:
    """Handle the ``evaluate`` subcommand — parse args and run evaluation."""
    from omni_evaluator.args import get_parser, CustomArgumentParser

    # restore artifact if specified
    artifact_filepath = None
    for _idx, _arg in enumerate(remaining):
        if "artifact_filepath" not in _arg:
            continue
        if (
            _idx >= len(remaining) - 1
            or remaining[_idx + 1].startswith("--")
        ):
            artifact_filepath = _arg.split("=")[-1].strip()
        else:
            artifact_filepath = remaining[_idx + 1].strip()

    if artifact_filepath:
        CustomArgumentParser.restore_arguments(
            artifact_filepath=artifact_filepath,
        )
        logger.info(f"restored artifact: {artifact_filepath}")

    # build full parser with evaluation arguments
    eval_parser = CustomArgumentParser(add_help=True)
    eval_parser, validations = get_parser(parser=eval_parser)
    args = eval_parser.parse_args(remaining)
    for _validation_func in validations:
        args = _validation_func(args=args)

    from omni_evaluator.evaluate import main as main__evaluate
    main__evaluate(args)


if __name__ == "__main__":
    main()
