# Reference from https://github.com/EvolvingLMMs-Lab/lmms-eval (Apache-2.0)

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

import re

import pandas as pd

from lmms_eval.tasks.wemath.wemath_utils import (
    calculate_metrics,
    compute_final_scores,
    process_steps_data,
    update_main_results_df,
)

from math_verify import parse as math_parse, verify as math_verify

SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two parts: "
    "first, the reasoning process enclosed in <think>...</think> tags, then the final answer enclosed in <answer>...</answer> tags."
    "Please provide a clear, concise response within <answer> </answer> tags that directly addresses the question."
)

SYSTEM_PROMPT_BOXED = (
    "You are an expert mathematics tutor. Solve each problem step by step and always put your final answer inside \\boxed{}."
)


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def _extract_boxed_answer(predict_str: str) -> str:
    boxed_start = "\\boxed{"
    start_indices = []
    pos = 0
    while True:
        pos = predict_str.find(boxed_start, pos)
        if pos == -1:
            break
        start_indices.append(pos)
        pos += 1
    if not start_indices:
        return ""
    results = []
    for start_pos in start_indices:
        brace_count = 0
        pos = start_pos + len(boxed_start) - 1
        while pos < len(predict_str):
            char = predict_str[pos]
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    content_start = start_pos + len(boxed_start)
                    results.append(predict_str[content_start:pos])
                    break
            pos += 1
    return results[-1] if results else ""


def _extract_answer_closed_tag(predict_str: str) -> str:
    """Extract from closed <answer>...</answer> tag, then boxed, then last number."""
    match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if match:
        return match.group(1)
    boxed = _extract_boxed_answer(predict_str)
    if boxed:
        return boxed
    for line in reversed(predict_str.strip().split("\n")):
        if line.strip():
            m = re.search(r"\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)", line)
            if m:
                return m.group(1)
    return ""


def _extract_answer_tag(predict_str: str) -> str:
    """Extract answer, also handling unclosed <answer> tags (e.g., truncated output)."""
    match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"<answer>(.*?)$", predict_str, re.DOTALL)
    if match:
        return match.group(1)
    boxed = _extract_boxed_answer(predict_str)
    if boxed:
        return boxed
    for line in reversed(predict_str.strip().split("\n")):
        if line.strip():
            m = re.search(r"\b(\d+(?:\.\d+)?)\b(?:\s*\.?\s*$)", line)
            if m:
                return m.group(1)
    return ""


def _simple_parse(predict_str: str) -> str:
    if predict_str.endswith("."):
        predict_str = predict_str[:-1]
    return predict_str.strip()


def _parse_mcq(predict_str: str) -> str:
    if not predict_str or not predict_str.strip():
        return ""
    response = " " + predict_str.strip() + " "
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)
    response = " " + response.strip() + " "
    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append((choice, response.rfind(f"({choice})")))
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append((choice, response.rfind(f" {choice} ")))
    if candidates:
        return max(candidates, key=lambda x: x[1])[0]
    return predict_str.strip()


def _parse_mcq_full(predict_str: str) -> str:
    """Full MCQ parser with 13 pattern types and priority ranking."""
    if not predict_str or not predict_str.strip():
        return ""
    response = predict_str.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        response = response.strip(char)
    response = " " + response + " "
    all_choices = ["A", "B", "C", "D", "E", "F", "G", "H"]
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append((choice, response.rfind(f"({choice})"), "parentheses"))
    for choice in all_choices:
        if f"{choice}." in response:
            candidates.append((choice, response.rfind(f"{choice}."), "period"))
    for choice in all_choices:
        if f"{choice}:" in response:
            candidates.append((choice, response.rfind(f"{choice}:"), "colon"))
    for choice in all_choices:
        if f"{choice})" in response:
            candidates.append((choice, response.rfind(f"{choice})"), "right_paren"))
    for choice in all_choices:
        if f"{choice} " in response:
            candidates.append((choice, response.rfind(f"{choice} "), "space"))
    for choice in all_choices:
        if f"{choice}-" in response:
            candidates.append((choice, response.rfind(f"{choice}-"), "dash"))
    for choice in all_choices:
        if f"{choice}_" in response:
            candidates.append((choice, response.rfind(f"{choice}_"), "underscore"))
    for choice in all_choices:
        if f"{choice}=" in response:
            candidates.append((choice, response.rfind(f"{choice}="), "equals"))
    answer_phrases = [
        "the answer is", "answer is", "the correct answer is", "correct answer is",
        "the answer", "answer", "correct answer", "the correct answer",
        "the best answer is", "best answer is", "the best answer", "best answer",
        "the option is", "option is", "the correct option is", "correct option is",
        "the choice is", "choice is", "the correct choice is", "correct choice is",
        "i choose", "i select", "i pick", "my answer is", "my choice is",
    ]
    for phrase in answer_phrases:
        if phrase in response.lower():
            phrase_start = response.lower().find(phrase)
            for choice in all_choices:
                choice_pos = response.find(choice, phrase_start)
                if choice_pos != -1:
                    candidates.append((choice, choice_pos, "phrase"))
    for choice in all_choices:
        if response.strip().startswith(choice):
            candidates.append((choice, 0, "start"))
    for choice in all_choices:
        if response.strip().endswith(choice):
            candidates.append((choice, len(response) - 1, "end"))
    for i, choice in enumerate(all_choices):
        if f"{i+1}. {choice}" in response:
            candidates.append((choice, response.rfind(f"{i+1}. {choice}"), "numbered"))
    if not candidates:
        for choice in all_choices:
            if choice in response:
                candidates.append((choice, response.rfind(choice), "fallback"))
    if candidates:
        format_priority = {
            "start": 10, "end": 9, "numbered": 8, "phrase": 7,
            "parentheses": 6, "period": 5, "colon": 4, "right_paren": 3,
            "space": 2, "dash": 1, "underscore": 1, "equals": 1, "fallback": 0,
        }
        candidates.sort(key=lambda x: (format_priority[x[2]], -x[1]), reverse=True)
        return candidates[0][0]
    return ""


def _relax_exact_match(predict_str: str, ground_truth: str, relax_portion: float = 0.9) -> float:
    if _parse_mcq(ground_truth) in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        predict_str = _parse_mcq(predict_str)
        return 1.0 if predict_str.lower().strip() == _parse_mcq(ground_truth).lower().strip() else 0.0
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0


def _relax_exact_match_full(predict_str: str, ground_truth: str, relax_portion: float = 0.9) -> float:
    """relax_exact_match using _parse_mcq_full for richer MCQ pattern matching."""
    if _parse_mcq_full(ground_truth) in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        predict_str = _parse_mcq_full(predict_str)
        return 1.0 if predict_str.lower().strip() == _parse_mcq_full(ground_truth).lower().strip() else 0.0
    if predict_str in ground_truth and len(predict_str) >= relax_portion * len(ground_truth):
        return 1.0
    if ground_truth in predict_str and len(ground_truth) >= relax_portion * len(predict_str):
        return 1.0
    return 1.0 if predict_str.strip() == ground_truth.strip() else 0.0


def _acc_reward(predict_str, ground_truth):
    predict_str = _simple_parse(predict_str)
    gt = _simple_parse(ground_truth)
    acc_score = _relax_exact_match(predict_str, gt)
    if acc_score == 0.0:
        try:
            gold = math_parse(gt)
            pred = math_parse(predict_str)
            acc_score = int(math_verify(gold, pred))
        except Exception:
            acc_score = 0.0
    return acc_score


def _acc_reward_tag(predict_str, ground_truth):
    """acc_reward using full MCQ parser (_parse_mcq_full)."""
    predict_str = _simple_parse(predict_str)
    gt = _simple_parse(ground_truth)
    acc_score = _relax_exact_match_full(predict_str, gt)
    if acc_score == 0.0:
        try:
            gold = math_parse(gt)
            pred = math_parse(predict_str)
            acc_score = int(math_verify(gold, pred))
        except Exception:
            acc_score = 0.0
    return acc_score


def _format_reward(predict_str: str) -> float:
    if re.search(r"<think>.*</think>.*<answer>.*</answer>", predict_str, re.DOTALL):
        return 1.0
    if re.search(r"<analysis>.*</analysis>.*<answer>.*</answer>", predict_str, re.DOTALL):
        return 1.0
    if _extract_boxed_answer(predict_str):
        return 1.0
    if len(predict_str.strip()) > 50:
        has_math = bool(re.search(r"[=\+\-\*/\(\)\[\]\\]", predict_str))
        has_answer = bool(_extract_answer_closed_tag(predict_str))
        if has_math and has_answer:
            return 0.8
    return 0.0


def _format_reward_tag(predict_str: str) -> float:
    """format_reward that also accepts unclosed <answer> tag."""
    if re.search(r"<think>.*</think>.*<answer>.*</answer>", predict_str, re.DOTALL):
        return 1.0
    if re.search(r"<analysis>.*</analysis>.*<answer>.*</answer>", predict_str, re.DOTALL):
        return 1.0
    if _extract_boxed_answer(predict_str):
        return 1.0
    if re.search(r"<answer>", predict_str):
        return 1.0
    if len(predict_str.strip()) > 50:
        has_math = bool(re.search(r"[=\+\-\*/\(\)\[\]\\]", predict_str))
        has_answer = bool(_extract_answer_tag(predict_str))
        if has_math and has_answer:
            return 0.8
    return 0.0


def _compute_score(solution_str, ground_truth):
    format_score_weight = 0.1
    format_reward_score = _format_reward(solution_str)
    extracted_answer = _extract_answer_closed_tag(solution_str).strip()
    acc_score = _acc_reward(extracted_answer, ground_truth)
    score = (1.0 - format_score_weight) * acc_score + format_score_weight * format_reward_score
    return {"score": score, "acc_score": acc_score, "format_reward_score": format_reward_score}


def _compute_score_tag(solution_str, ground_truth):
    """compute_score variant that also accepts unclosed <answer> tags and uses full MCQ parser."""
    format_score_weight = 0.1
    format_reward_score = _format_reward_tag(solution_str)
    extracted_answer = _extract_answer_tag(solution_str).strip()
    acc_score = _acc_reward_tag(extracted_answer, ground_truth)
    score = (1.0 - format_score_weight) * acc_score + format_score_weight * format_reward_score
    return {"score": score, "acc_score": acc_score, "format_reward_score": format_reward_score}


# ---------------------------------------------------------------------------
# lmms-eval task interface
# ---------------------------------------------------------------------------

def wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + "\n" + doc["option"]


def wemath_doc_to_visual(doc):
    return [doc["image_path"].convert("RGB")]


def wemath_doc_to_messages_cot(doc, lmms_eval_specific_kwargs=None):
    question = wemath_doc_to_text_cot(doc, lmms_eval_specific_kwargs)
    visuals = wemath_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def wemath_doc_to_text_boxed(doc, lmms_eval_specific_kwargs=None):
    return doc["question"] + "\n" + doc["option"]


def wemath_doc_to_messages_boxed(doc, lmms_eval_specific_kwargs=None):
    question = wemath_doc_to_text_boxed(doc, lmms_eval_specific_kwargs)
    visuals = wemath_doc_to_visual(doc)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_BOXED}]}]
    messages = [{"role": "user", "content": []}]
    messages[0]["content"].append({"type": "image", "url": visuals[0]})
    messages[0]["content"].append({"type": "text", "text": question.strip()})
    messages = system_messages + messages
    return messages


def _compute_score_boxed(solution_str, ground_truth):
    """compute_score with boxed extraction as primary, then fallback to tag/heuristics."""
    format_score_weight = 0.1
    format_reward_score = _format_reward(solution_str)
    extracted_answer = _extract_boxed_answer(solution_str).strip()
    extracted_answer = re.sub(r'\\text\{([A-Za-z])\}', r'\1', extracted_answer)
    if not extracted_answer:
        extracted_answer = _extract_answer_closed_tag(solution_str).strip()
    acc_score = _acc_reward(extracted_answer, ground_truth)
    score = (1.0 - format_score_weight) * acc_score + format_score_weight * format_reward_score
    return {"score": score, "acc_score": acc_score, "format_reward_score": format_reward_score}


def wemath_boxed_process_results(doc, results):
    acc_score = 0
    format_score = 0
    acc_score_tag = 0
    format_score_tag = 0
    for pred in results:
        score_dict = _compute_score_boxed(solution_str=pred.strip(), ground_truth=doc["answer"])
        acc_score += score_dict["acc_score"]
        format_score += score_dict["format_reward_score"]

        score_dict_tag = _compute_score_tag(solution_str=pred.strip(), ground_truth=doc["answer"])
        acc_score_tag += score_dict_tag["acc_score"]
        format_score_tag += score_dict_tag["format_reward_score"]

    data_dict = {
        "ID": doc["ID"],
        "split": doc["split"],
        "knowledge concept": doc["knowledge concept"],
        "question": doc["question"],
        "option": doc["option"],
        "answer": doc["answer"],
        "key": doc["key"],
        "question number": doc["question number"],
        "knowledge concept description": doc["knowledge concept description"],
        "acc_score": acc_score,
    }

    data_dict_tag = dict(data_dict)
    data_dict_tag["acc_score"] = acc_score_tag

    n = len(results) if results else 1
    return {
        "wemath_loose": data_dict,
        "wemath_strict": data_dict,
        "acc_score": acc_score / n,
        "format_score": format_score / n,
        "wemath_loose_tag": data_dict_tag,
        "wemath_strict_tag": data_dict_tag,
        "acc_score_tag": acc_score_tag / n,
        "format_score_tag": format_score_tag / n,
    }


def wemath_reasoning_process_results(doc, results):
    acc_score = 0
    format_score = 0
    acc_score_tag = 0
    format_score_tag = 0
    for pred in results:
        score_dict = _compute_score(solution_str=pred.strip(), ground_truth=doc["answer"])
        acc_score += score_dict["acc_score"]
        format_score += score_dict["format_reward_score"]

        score_dict_tag = _compute_score_tag(solution_str=pred.strip(), ground_truth=doc["answer"])
        acc_score_tag += score_dict_tag["acc_score"]
        format_score_tag += score_dict_tag["format_reward_score"]

    data_dict = {
        "ID": doc["ID"],
        "split": doc["split"],
        "knowledge concept": doc["knowledge concept"],
        "question": doc["question"],
        "option": doc["option"],
        "answer": doc["answer"],
        "key": doc["key"],
        "question number": doc["question number"],
        "knowledge concept description": doc["knowledge concept description"],
        "acc_score": acc_score,
    }

    data_dict_tag = dict(data_dict)
    data_dict_tag["acc_score"] = acc_score_tag

    n = len(results) if results else 1
    return {
        "wemath_loose": data_dict,
        "wemath_strict": data_dict,
        "acc_score": acc_score / n,
        "format_score": format_score / n,
        "wemath_loose_tag": data_dict_tag,
        "wemath_strict_tag": data_dict_tag,
        "acc_score_tag": acc_score_tag / n,
        "format_score_tag": format_score_tag / n,
    }


def wemath_aggregate_results(results, metric_name):
    data = pd.DataFrame(results)
    data["joker"] = data["acc_score"] == 1.0
    data_2steps = data[data["key"].str.contains("2steps")]
    data_3steps = data[data["key"].str.contains("3steps")]
    merged_2steps = process_steps_data(data_2steps, 2)
    merged_3steps = process_steps_data(data_3steps, 3)
    metrics = calculate_metrics(merged_2steps, merged_3steps)
    total_counts, rates = compute_final_scores(metrics, total_count=525)
    score_dict = update_main_results_df(total_counts, rates)
    if metric_name == "wemath_loose":
        score = score_dict["Score (Loose)"]
    elif metric_name == "wemath_strict":
        score = score_dict["Score (Strict)"]
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")
    # score may be a percentage string (e.g. "74.23%") — convert to float
    if isinstance(score, str):
        score = float(score.strip("%")) / 100
    return float(score)


def wemath_aggregate_results_loose(results):
    return wemath_aggregate_results(results, "wemath_loose")


def wemath_aggregate_results_strict(results):
    return wemath_aggregate_results(results, "wemath_strict")


def wemath_aggregate_results_loose_tag(results):
    return wemath_aggregate_results(results, "wemath_loose")


def wemath_aggregate_results_strict_tag(results):
    return wemath_aggregate_results(results, "wemath_strict")
