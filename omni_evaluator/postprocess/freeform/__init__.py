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

import logging
import re
from typing import List, Tuple, Optional, Union, Any, Dict

from omni_evaluator.api.chat_completions import chat_completion_sync
from omni_evaluator.postprocess._interface import ProcessorInterface
from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent as ChatTextContent,
)

logger = logging.getLogger(__name__)


class FreeformProcessor(ProcessorInterface):
    # Quote characters to support when answers are quoted
    QUOTE_CHARACTERS = '"“”\'‘’'
    # Whitespace class used for light cleanup at the end of the span
    WHITESPACE_PATTERN = r"[ \t\u00A0]"
    STOP_PATTERN = r'(?=\s*(?:because|since|which|that|;|—|–|-|,|입니다|이다|임|예요|에요|[!?\n]|$))'
    EXTRACT_PROMPT = """You will be given a **Question** and a **Model response**.  
Your job is to **extract only the final answer token** from the response and print **just that token** as the output (no extra words, no quotes, no punctuation).

---

## What counts as the “final answer token”
Follow these cue phrases (English & Korean). The answer appears **right after** them; it may be wrapped in quotes or brackets:

- **English cues** (case-insensitive):  
  `The correct/final answer is`, `Answer:`, `Ans is`, `Answer is`, `Option is`, `Choice is`
- **Korean cues**:  
  `(최종)? 정답`, `정답은`, `정답:`, `답은`, `선택지`, `보기` + (선택적으로) `:`, `은`, `는`, `이`

**Allowed answer token forms** (extract exactly as written after the cue):
- **Integer** (e.g., `14`)
- **Float** (e.g., `0.6`, `1.45`) — keep the original decimal precision in the response
- **Number with grouping** (e.g., `1,234`)
- **A math-like literal** (e.g., `24^2`)
- **Python list** (e.g., `[2007, 2008]`)
- **Multiple‐choice letter** (e.g., `A`, `B`, `C`, `D`) — **return only the letter**
- **Quoted or bracketed forms** — remove outer `"..."`, `'...'`, `(...)`, `[...]` when printing

**Stop reading** the answer token when you reach trailing connectors or punctuation that are **not part of the token**:  
`because`, `since`, `which`, `that`, `;`, `—`, `–`, `-`, a comma/period **that is not between digits** (so keep `1,234` and `5.12` intact), `!`, `?`, a newline, or the end of text.  
Korean stops: `입니다`, `이다`, `임`, `예요`, `에요` (these come **after** the token).

Tie-breaking:
- If multiple cue phrases appear, **use the last one** in the response.
- If both quoted/bracketed and plain forms appear, prefer the **innermost** answer immediately after the cue.
- If no cue phrase exists, output **N/A**.

**Output format**: print **only** the extracted token (no quotes, no trailing period/comma, no explanation).

---

## In-context examples

- *Question:* Which number is missing?  
  *Model response:* The number missing in the sequence is 14.  
  **Output:** `14`

- *Hint:* Float with one decimal  
  *Question:* What is the fraction of females facing the camera?  
  *Model response:* The fraction of females facing the camera is 0.6, which means that six out of ten females…  
  **Output:** `0.6`

- *Hint:* Float with two decimals  
  *Question:* How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)  
  *Model response:* Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.  
  **Output:** `1.45`

- *Hint:* Python list  
  *Question:* Between which two years does the line graph saw its maximum peak?  
  *Model response:* The line graph saw its maximum peak between 2007 and 2008.  
  **Output:** `[2007, 2008]`

- *Hint:* Multiple choice (A/B/C/D)  
  *Question:* What fraction of the shape is blue? Choices: (A) 3/11 (B) 8/11 (C) 6/11 (D) 3/5  
  *Model response:* The correct answer is (B) 8/11.  
  **Output:** `B`

---

## Your turn

**Question:**  
{query}

**Model response:**  
{prediction}

**Print only the extracted token below (no quotes, no extra text):**"""

    @classmethod
    def extract(
        cls,
        prediction: str,
        query: str,
        version_name: Optional[str] = None,
        api_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        # temperature / max_tokens / prompt_template default values mirror
        # _extract_freeform_api's signature so existing callers (no extra kwargs
        # in config.yaml) hit the same downstream behavior. Tasks needing a
        # paper-specific GPT extract (e.g. mathverse's demo_prompt_extract with
        # temperature=0.2, max_tokens=2048) declare those kwargs in
        # postprocess.<entry> of config.yaml and they propagate through here.
        if (
            not isinstance(prediction, str)
            or len(prediction) == 1
        ):
            return prediction

        output = None
        if (
            isinstance(api_name, str)
            and len(api_name) > 0
        ):
            output = cls._extract_freeform_api(
                prediction=prediction,
                query=query,
                api_name=api_name,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_template=prompt_template,
                verbose=verbose,
            )
        else:
            output = cls._extract_freeform(
                prediction=prediction,
                query=query,
                verbose=verbose,
            )

        return output
    

    @classmethod
    def _extract_freeform(
        cls,
        prediction: str,
        query: Optional[str] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ) -> Optional[str]:
        """
        Extract a free-form answer string from an LLM prediction.

        Supported cue phrases (English/Korean) include examples like:
        - "The correct/final answer is: <ANSWER>"
        - "Answer: <ANSWER>"
        - "정답은 <ANSWER> 입니다/이다"
        The answer may be a word, phrase, or short sentence.

        Args:
            prediction: Raw prediction text.
            query: Optional query text for context.
            verbose: If True, enable verbose logging.

        Returns:
            A cleaned answer string if found, otherwise None.
        """
        text = prediction.strip()

        # Build reusable subpatterns
        q = re.escape(cls.QUOTE_CHARACTERS)

        # Stop the greedy capture at common sentence/end markers or Korean verbal endings.

        # Ordered patterns: first match wins
        patterns = [
            # EN: (the) (final|correct)? (answer|ans|solution|option|choice) (is|:) "ANSWER"
            r"""(?i)\b(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|ans|solution|option|choice)\s*(?:is|:)\s*(?:(["“”'‘’])\s*(?P<ans1>[^"“”'‘’]+?)\s*\1|\(\s*(?P<ansp>[^)]+?)\s*\)|\[\s*(?P<ansb>[^\]]+?)\s*\]|(?P<ans2>.+?))(?=\s*(?:because|since|which|that|[;—–-]|,(?!\d)|\.(?!\d)|[!?]|\n|$))""",
            # KO: (final)? (answer|choice|option) [:particles]? "ANSWER"
            r"""(?:최종\s*)?(?:정답|답(?:안)?|선택(?:지)?|보기)\s*[:은는이]?\s*(?:(["“”'‘’])\s*(?P<ans1>[^"“”'‘’]+?)\s*\1|\(\s*(?P<ansp>[^)]+?)\s*\)|\[\s*(?P<ansb>[^\]]+?)\s*\]|(?P<ans2>.+?))(?=\s*(?:때문에|이므로|이며|이고|인데|하지만|그러나|그리고|;|—|–|-|,(?!\d)|\.(?!\d)|[!?]|입니다|이다|임|예요|에요|\n|$))""",
        ]
        patterns = [
            re.compile(p, re.IGNORECASE | re.MULTILINE | re.VERBOSE)
            for p in patterns
        ]

        output = None
        for pat in patterns:
            m = pat.search(text)
            if not m:
                continue
            ans = m.group("ans1") or m.group("ansp") or m.group("ansb") or m.group("ans2")
            ans = cls._clean_span(ans)
            if not ans:
                continue
            output = ans
            
        logger.debug(f'FreeformProcessor: {prediction} -> {output}')
        return output


    @classmethod
    def _extract_freeform_api(
        cls,
        prediction: str,
        query: Optional[str] = None,
        api_name: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        do_async: Optional[bool] = False,
        verbose: bool = False,
        prompt_template: Optional[str] = None,
    ) -> Optional[str]:

        # construct messages — caller may inject a task-specific template
        # (e.g. mathvista demo_prompt); default to the generic EXTRACT_PROMPT.
        template = prompt_template if isinstance(prompt_template, str) and len(prompt_template) > 0 else cls.EXTRACT_PROMPT
        query = template.format(
            query=query,
            prediction=prediction,
        )
        messages = list()
        messages.append(ChatMessage(
            role="user",
            content=[ChatTextContent(type="text", value=query), ],
        ))
        
        output = chat_completion_sync(
            api_name=api_name,
            messages=messages,
            generation_options={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        if (
            len(output.strip()) < 1
            or output.lower() in ["n/a"]
        ):
            output = None
        logger.debug(f'FreeformProcessor (API): {prediction} -> {output}')
        return output
    

    @classmethod
    def _clean_span(
        cls,
        text: str,
    ) -> str:
        """
        Normalize an extracted answer span:
        - Trim surrounding whitespace.
        - Remove a single pair of surrounding quotes/brackets if present.
        - Drop trailing punctuation like '.', ';', ':'.
        """
        text = text.strip()

        # Strip one outer layer of quotes/brackets if they match
        pairs = [
            ('"', '"'), ('“', '”'), ("'", "'"), ('‘', '’'),
            # ('(', ')'), ('[', ']'), ('<', '>')
        ]
        for left, right in pairs:
            if (
                text.startswith(left) 
                and text.endswith(right) 
                and len(text) >= 2
            ):
                text = text[1:-1].strip()

        # Remove trailing punctuation and surrounding spaces
        text = re.sub(rf"{cls.WHITESPACE_PATTERN}*[.;:,\u3002\uFF0E\uFF1A\uFF1B]+$", "", text).strip()
        return text