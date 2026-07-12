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

import random
import re

import datasets


# Matches the prompt format from custom make_prompt()
QUERY_TEMPLATE = (
    "Answer the following multiple choice question. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of {letters}. "
    "Think step by step before answering.\n\n"
    "{question}\n\n"
    "{choices}"
)


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    rng = random.Random(42)

    def _process_doc(doc):
        choices_list = [
            preprocess(doc["Correct Answer"]),
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
        ]
        rng.shuffle(choices_list)
        correct_letter = chr(65 + choices_list.index(preprocess(doc["Correct Answer"])))
        return {
            "question": doc["Question"],
            "choices": choices_list,
            "answer": correct_letter,
        }

    return dataset.map(_process_doc)


def doc_to_text(doc):
    letters = "ABCD"
    choices_str = "\n".join([f"{l}) {c}" for l, c in zip(letters, doc["choices"])])
    return QUERY_TEMPLATE.format(
        letters=letters,
        question=doc["question"],
        choices=choices_str,
    )


def _extract_letter(response: str) -> str:
    """Extract answer letter from model output.

    Thinking models often ignore the 'Answer: $LETTER' prompt and instead
    write \boxed{X}, Answer: $\boxed{X}$, or Answer: $X$. Handle all cases.

    Priority:
      1. Answer: $\boxed{X}$  — boxed inside Answer line
      2. Answer: $X$          — dollar-wrapped letter in Answer line
      3. Answer: X            — plain letter (original format)
      4. \boxed{X}            — boxed anywhere (last occurrence)
      5. \boxed{\text{X}}     — LaTeX text-wrapped boxed
    """
    if not response:
        return ""
    # 1. Answer: $\boxed{X}$
    m = re.search(r'(?i)Answer\s*:\s*\$\s*\\boxed\{([A-Da-d])\}\s*\$', response)
    if m:
        return m.group(1).upper()
    # 2. Answer: $X$
    m = re.search(r'(?i)Answer\s*:\s*\$\s*([A-Da-d])\s*\$', response)
    if m:
        return m.group(1).upper()
    # 3. Answer: X  (original format)
    m = re.search(r'(?i)Answer\s*:\s*([A-Z])', response)
    if m:
        return m.group(1).upper()
    # 4. \boxed{\text{X}} or \boxed{X}  — take last occurrence
    for pattern in (r'\\boxed\{\\text\{([A-Da-d])\}\}', r'\\boxed\{([A-Da-d])\}'):
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
    return ""


def process_results(doc, results):
    gt = doc["answer"]

    extracted = []
    exact_matches = []
    for response in results:
        pred = _extract_letter(response)
        extracted.append(pred)
        exact_matches.append(int(pred == gt))

    return {
        "exact_match": exact_matches[0] if exact_matches else 0,
        "exact_matches": exact_matches,
        "extracted_answers": extracted,
    }
