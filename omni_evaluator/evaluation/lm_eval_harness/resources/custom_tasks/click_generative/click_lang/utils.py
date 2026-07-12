# Adapted from EleutherAI/lm-evaluation-harness (MIT License)
# Original: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/click/click_lang/utils.py
# Dataset: EunsuKim/CLIcK (CC BY 4.0) — https://arxiv.org/abs/2403.06412

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

from typing import List

from datasets import Dataset


def get_context(doc) -> str:
    ctx = doc["paragraph"]
    q = doc["question"]
    opt = doc["choices"]
    res = ""
    if ctx:
        res += f"주어진 맥락을 천천히 읽고, 질문에 대한 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n맥락: {ctx}\n질문: {q}\n보기:\nA:{opt[0]}, B: {opt[1]}, C: {opt[2]}, D: {opt[3]}\n"
    else:
        res += f"주어진 질문을 천천히 읽고, 적절한 정답을 A, B, C, D 중에 골라 알파벳 하나로 답하시오.\n\n질문: {q}\n보기:\nA:{opt[0]}, B: {opt[1]}, C: {opt[2]}, D: {opt[3]}\n"
    # res += "ASSISTANT: 정답:"

    return res


def get_target(doc) -> str:
    ans = doc["answer"]
    if "CSAT" in doc["id"]:
        return ["A", "B", "C", "D", "E"][doc["choices"].index(ans)]
    return ["A", "B", "C", "D"][doc["choices"].index(ans)]


def get_choices(doc) -> List[str]:
    if "CSAT" in doc["id"]:
        return ["A", "B", "C", "D", "E"]
    return ["A", "B", "C", "D"]


def extract_text(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: "CSAT_korean_22" in example["id"]
        or (
            "CSAT_korean_23" in example["id"] and int(example["id"].split("_")[-1]) < 35
        )
        or ("TK" in example["id"] and int(example["id"].split("_")[-1]) > 4)
    )


def extract_grammar(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: (
            "CSAT_korean" in example["id"]
            and (
                int(example["id"].split("_")[2]) < 21
                and int(example["id"].split("_")[3]) > 10
            )
        )
        or (
            "Kedu_1" in example["id"]
            and (
                example["id"].split("_")[1] != "16"
                or not (
                    "대화" in example["question"]
                    or "발화" in example["question"]
                    or "질의" in example["question"]
                )
            )
        )
        or ("TK" in example["id"] and int(example["id"].split("_")[-1]) < 5)
    )


def extract_function(dataset: Dataset) -> Dataset:
    return dataset.filter(
        lambda example: (
            "CSAT_korean" in example["id"]
            and (
                int(example["id"].split("_")[-1]) > 34
                or (
                    int(example["id"].split("_")[2]) < 21
                    and int(example["id"].split("_")[3]) < 11
                )
            )
        )
        or (
            "Kedu_16" in example["id"]
            and (
                "대화" in example["question"]
                or "발화" in example["question"]
                or "질의" in example["question"]
            )
        )
        or "PSE_korean" in example["id"]
    )
