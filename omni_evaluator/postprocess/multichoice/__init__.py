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
import unicodedata
from enum import Enum
from typing import List, Tuple, Optional, Union, Any, Dict

from omni_evaluator.api.chat_completions import chat_completion_sync
from omni_evaluator.postprocess._interface import ProcessorInterface
from omni_evaluator.schemas.chat import (
    Message as ChatMessage,
    TextContent as ChatTextContent,
)

logger = logging.getLogger(__name__)


class MultichoiceVersion(str, Enum):
    """Explicit version selector for `MultichoiceProcessor.extract`.

    Each member's value matches the string that yaml task configs use under
    `postprocess.<step>.version_name`, so `MultichoiceVersion(value)` round-trips
    cleanly from yaml. A `None` version_name skips this enum entirely and routes
    to the auto-detect path.
    """
    NUMERIC = "numeric"
    ALPHA = "alpha"
    CIRCLED = "circled"
    KOREAN = "korean"


class MultichoiceProcessor(ProcessorInterface):
    LETTER_SET = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}
    # Per-token list pattern: each match captures ONE choice marker, so multi-line
    # enumerations like "A) foo\nB) bar\nC) baz" yield a separate match per line.
    # Use `finditer` and count distinct tokens to detect MC scheme.
    LETTER_PATTERN = r"(?m)(?:^[ \t]*[\(\[]?\s*(?P<tok>[A-J])\s*[\)\]\.,:]|[\(\[]\s*(?P<tok2>[A-Ja-j])\s*[\)\]])"
    LETTER_EXTRACT_PATTERNS = [
        # \boxed{X} from reasoning-style outputs (highest priority for last-occurrence semantics handled below).
        r"(?ix)\\boxed\{\s*[\*]{0,2}[\(\[]?\s*(?P<choice>[A-J])\s*[\)\]]?[\*]{0,2}\s*\}",
        r"(?ix)\b(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|ans|option|choice)\s*(?:is\s*[:,]?|[:,])\s*[\*]{0,2}[\(\[]?\s*(?P<choice>[A-J])\s*[\)\]]?[\*]{0,2}(?![A-Za-z])",
        r"(?ix)(?:정답(?:은)?|답(?:은)?)\s*[:,]?\s*[\*]{0,2}[\(\[]?\s*(?P<choice>[A-J])\s*[\)\]]?[\*]{0,2}\s*(?:입니다|이다)?",
        # "should choose X" / "Project X" / "Option X" / "select X"
        r"(?ix)\b(?:should\s+choose|choose|select|pick|go\s+with)\s+(?:option\s+|project\s+|choice\s+)?[\*]{0,2}[\(\[]?\s*(?P<choice>[A-J])\s*[\)\]]?[\*]{0,2}(?![A-Za-z])",
        r"(?ix)\b(?:option|project|choice)\s+[\*]{0,2}[\(\[]?\s*(?P<choice>[A-J])\s*[\)\]]?[\*]{0,2}(?:\s+is\s+(?:the\s+)?(?:correct|right|answer))?(?![A-Za-z])",
        r"(?ix)^[\(\[]?\s*(?P<choice>[A-J])\s*[\)\]\.,:]?$",
        r"(?ix)^[\(\[]?\s*(?P<choice>[A-J])\s*[\)\]\.,:]\s+\S",
    ]
    NUMBER_SET = {"1", "2", "3", "4", "5", "6", "7", "8", "9"}
    NUMBER_PATTERN = r"(?m)(?:^[ \t]*[\(\[]?\s*(?P<tok>[1-9])\s*[\)\]\.,:]|[\(\[]\s*(?P<tok2>[1-9])\s*[\)\]])"
    NUMBER_EXTRACT_PATTERNS = [
        # \boxed{N} from reasoning-style outputs (highest priority for last-occurrence semantics handled below).
        r"(?ix)\\boxed\{\s*[\*]{0,2}[\(\[]?\s*(?P<choice>[1-9])\s*[\)\]]?[\*]{0,2}\s*\}",
        r"(?ix)\b(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|ans|option|choice)\s*(?:is|:)\s*[\(\[]?\s*(?P<choice>[1-9])\s*[\)\]]? (?!\d)",
        r"(?ix)(?:정답(?:은)?|답(?:은)?)\s*[\(\[]?\s*(?P<choice>[1-9])\s*[\)\]]?\s*(?:입니다|이다)?",
        r"(?ix)^[\(\[]?\s*(?P<choice>[1-9])\s*[\)\]\.,:]?$",
        r"(?ix)^[\(\[]?\s*(?P<choice>[1-9])\s*[\)\]\.,:]\s+\S",
        # Korean ordinal suffix: "1번", "2번", ...
        r"(?ix)^[\(\[]?\s*(?P<choice>[1-9])\s*번\s*$",
    ]
    CIRCLED_NUMBER_SET = {"①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"}
    # NFKC dissolves ①-⑨ into ASCII "1"-"9". When the auto-detect path sees a
    # circled scheme but the prediction has been NFKC-folded at the entry, NUMBER
    # patterns may still pick up the digit — this map restores the canonical
    # circled form so valid-set membership works.
    _ASCII_TO_CIRCLED = dict(zip("123456789", "①②③④⑤⑥⑦⑧⑨"))
    # Circled numerals usually stand alone; trailing punctuation is optional but
    # still require line-start or inline-bracket context to avoid mid-sentence FPs.
    CIRCLED_NUMBER_PATTERN = r"(?m)(?:^[ \t]*[\(\[]?\s*(?P<tok>[①②③④⑤⑥⑦⑧⑨])\s*[\)\]\.,:]?|[\(\[]\s*(?P<tok2>[①②③④⑤⑥⑦⑧⑨])\s*[\)\]])"
    CIRCLED_NUMBER_EXTRACT_PATTERNS = [
        r"(?ix)\b(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|ans|option|choice)\s*(?:is|:)\s*[\(\[]?\s*(?P<choice>[①②③④⑤⑥⑦⑧⑨])\s*[\)\]]? (?!\d)",
        r"(?ix)(?:정답(?:은)?|답(?:은)?)\s*[\(\[]?\s*(?P<choice>[①②③④⑤⑥⑦⑧⑨])\s*[\)\]]?\s*(?:입니다|이다)?",
        r"(?ix)^[\(\[]?\s*(?P<choice>[①②③④⑤⑥⑦⑧⑨])\s*[\)\]\.,:]?$",
        r"(?ix)^[\(\[]?\s*(?P<choice>[①②③④⑤⑥⑦⑧⑨])\s*[\)\]\.,:]\s+\S",
        # "③ 8" / "③ something" — circled marker followed by whitespace + content (no closing punctuation).
        r"(?ix)^[\(\[]?\s*(?P<choice>[①②③④⑤⑥⑦⑧⑨])\s+\S",
    ]
    EXTRACT_PROMPT = """You are given a multiple-choice query. Your task is to extract the final selected answer choice from the model’s response, regardless of whether the response includes reasoning or not.

Answer choices may take various formats, including:
- English letters: A, B, C, D, E, ...
- Numbers: 1, 2, 3, 4, 5, ...
- circled number: ①, ②, ③, ④, ⑤, ...

Instructions:
- Return only the **final selected answer choice** as it appears in the response.
- If the answer is implied clearly in natural language (e.g., “Therefore, the correct one is B”), return the choice (e.g., "B").
- If the response contains a choice symbol followed by a number or explanation (e.g., “③ 8”), extract only the choice symbol (e.g., “③”).
- If and only if the response cannot be clearly parsed to match exactly one of the given answer choices, return "Z". Otherwise, return the selected answer choice.

Output format:
- Return only one character representing the final answer choice (e.g., "B", "3", "①").
- No explanation or extra text

question: {query}
response : {prediction}
choice: """

    @classmethod
    def extract(
        cls,
        prediction: str,
        query: Optional[str] = None,
        options: Optional[List[str]] = None,
        option_contents: Optional[List[str]] = None,
        version_name: Optional[str] = None,
        api_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ):
        if not isinstance(prediction, str):
            return prediction

        # Single-char early-return applies ONLY to the auto-detect path. Version-
        # explicit branches must run the full pipeline so that e.g. `"①"` with
        # version_name="numeric" gets NFKC-folded into `"1"` (KCSAT-normalize parity).
        if version_name is None and len(prediction) == 1:
            return prediction

        # Strip CoT trace once at the entry so every downstream branch sees the
        # post-think text. Reasoning-style outputs put their final answer after
        # `</think>`; anything before is scratch work.
        think_end = prediction.rfind("</think>")
        if think_end != -1:
            prediction = prediction[think_end + len("</think>"):].strip()

        # Explicit version_name forces a fixed scheme and skips auto-detection. Unknown
        # version_name raises so misconfigured task yaml fails loudly instead of silently
        # falling through to the auto-detect path.
        if version_name is not None:
            try:
                version_enum = (
                    version_name if isinstance(version_name, MultichoiceVersion)
                    else MultichoiceVersion(version_name)
                )
            except ValueError:
                raise ValueError(
                    f'Unknown version_name {version_name!r} for MultichoiceProcessor.extract; '
                    f'supported: {[v.value for v in MultichoiceVersion]} or None'
                )

            if version_enum is MultichoiceVersion.NUMERIC:
                output = cls._extract_numeric(prediction=prediction, verbose=verbose)
            elif version_enum is MultichoiceVersion.ALPHA:
                output = cls._extract_alpha(prediction=prediction, verbose=verbose)
            elif version_enum is MultichoiceVersion.CIRCLED:
                output = cls._extract_circled(prediction=prediction, verbose=verbose)
            elif version_enum is MultichoiceVersion.KOREAN:
                raise NotImplementedError(
                    'MultichoiceProcessor._extract_korean() is not implemented'
                )
            logger.debug(f'MultichoiceProcessor ({version_enum.value}): {prediction} -> {output}')
            return output

        # NFKC normalize at the common entry so both regex and API paths see
        # canonicalized characters (full-width letters, circled numerals, etc.).
        prediction = unicodedata.normalize("NFKC", prediction)

        # Try deterministic regex extraction first (cheap).
        output = cls._extract_multichoice(
            prediction=prediction,
            query=query,
            options=options,
            option_contents=option_contents,
            verbose=verbose,
        )

        # Fall back to API extraction only if regex returned nothing.
        if (
            output is None
            and isinstance(api_name, str)
            and len(api_name) > 0
        ):
            output = cls._extract_multichoice_api(
                prediction=prediction,
                query=query,
                options=options,
                option_contents=option_contents,
                api_name=api_name,
                verbose=verbose,
            )

        logger.debug(f'MultichoiceProcessor (default): {prediction} -> {output}')
        return output

    @classmethod
    def _scan_candidates(
        cls,
        scan_text: str,
        scheme: str,
        valid: set,
        *,
        upper: bool = False,
    ) -> Optional[str]:
        """Walk candidates from `_extract_candidate` for the given scheme and
        pick the first one that lands in `valid` (optionally upper-casing the
        candidate before the membership check).

        Falls back to the first raw candidate when no candidate landed in
        `valid`, mirroring the legacy behavior of all four branches.
        """
        output = None
        candidates = list()
        for cand in cls._extract_candidate(scan_text, scheme):
            c = cand.upper() if upper else cand
            if c in valid:
                output = c
                break
            candidates.append(cand)
        if output is None and candidates:
            output = candidates[0]
        return output

    @classmethod
    def _tail_line_fallback(
        cls,
        scan_text: str,
        valid: set,
        *,
        allowed_chars: str,
        upper: bool = False,
    ) -> Optional[str]:
        """Last-resort scan over the last 3 non-empty stripped lines.

        Recognizes either a single-character line in `valid`, or a "<token>." /
        "<token>) text" first-character line where `<token>` matches the
        `allowed_chars` regex class and lies in `valid`.
        """
        tail_lines = [
            ln.strip().strip("*_`'\"()[]{}.,:; ")
            for ln in scan_text.strip().splitlines()
            if ln.strip()
        ]
        for ln in reversed(tail_lines[-3:]):
            if not ln:
                continue
            cand = ln.upper() if upper else ln
            if len(ln) == 1 and cand in valid:
                return cand
            m = re.match(rf"^([{allowed_chars}])(?:[\.\):,]|\s|$)", ln)
            if m:
                tok = m.group(1).upper() if upper else m.group(1)
                if tok in valid:
                    return tok
        return None

    @classmethod
    def _extract_numeric(
        cls,
        prediction: str,
        verbose: bool = False,
    ) -> Optional[str]:
        """Force scheme="number" and extract a bare digit 1-9.

        Covers the full range of KCSAT-style normalization (circled ①-⑨ → 1-9 via
        NFKC; suffixed "1번", "1.", "1)", "(1)", "[1]" → "1"; standalone "1" → "1")
        plus the keyword patterns from NUMBER_EXTRACT_PATTERNS (e.g. "answer: 3",
        "정답은 2입니다").
        """
        # NFKC turns circled numerals ①-⑨ into ASCII "1"-"9" so the number patterns
        # match uniformly.
        scan_text = unicodedata.normalize("NFKC", prediction)
        output = cls._scan_candidates(scan_text, scheme="number", valid=cls.NUMBER_SET)
        if output is None:
            output = cls._tail_line_fallback(
                scan_text, cls.NUMBER_SET, allowed_chars="1-9",
            )
        return output

    @classmethod
    def _extract_alpha(
        cls,
        prediction: str,
        verbose: bool = False,
    ) -> Optional[str]:
        """Force scheme="letter" and extract a bare letter A-J.

        Covers "A", "A.", "A)", "(A)", "[A]", \\boxed{A}, "the answer is A",
        "정답은 A입니다", etc. Full-width letters are canonicalized via NFKC.
        """
        scan_text = unicodedata.normalize("NFKC", prediction)
        output = cls._scan_candidates(
            scan_text, scheme="letter", valid=cls.LETTER_SET, upper=True,
        )
        if output is None:
            output = cls._tail_line_fallback(
                scan_text, cls.LETTER_SET, allowed_chars="A-Ja-j", upper=True,
            )
        return output

    @classmethod
    def _extract_circled(
        cls,
        prediction: str,
        verbose: bool = False,
    ) -> Optional[str]:
        """Force scheme="circled_number" and extract a bare ①-⑨ token.

        NFKC is intentionally NOT applied here because it dissolves circled
        numerals into ASCII digits, which would defeat circled-only matching.
        Tail-line fallback is also omitted — circled markers are visually distinct
        enough that the explicit EXTRACT patterns cover the realistic cases.
        """
        return cls._scan_candidates(
            prediction, scheme="circled_number", valid=cls.CIRCLED_NUMBER_SET,
        )

    @classmethod
    def _extract_multichoice(
        cls,
        prediction: str,
        query: str,
        options: Optional[List[str]] = None,
        option_contents: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> Optional[str]:
        """
        If the query looks like a multiple-choice (A/B/C/D or 1/2/3/4),
        try to extract a valid choice token from the prediction; otherwise return None.

        Args:
            query: The original query text that may contain choices.
            prediction: The answer/prediction text (e.g., "Answer: B", "The answer is A", or just "C").

        Returns:
            The extracted choice token as a string (e.g., "B" or "2"), or None if:
            - the query does not appear to be multiple-choice, or
            - no valid token could be extracted from the prediction.
        """
        scheme, valid = cls.is_multichoice(
            query=query,
            options=options,
            option_contents=option_contents,
            return_choices=True,
        )
        if scheme is None:
            return None

        # Backstop `valid` with the scheme's full set so weak detection (empty or
        # partial hits) still recognises any in-range token from the prediction.
        effective_valid = set(valid) if valid else set()
        if scheme == "letter":
            effective_valid |= cls.LETTER_SET
        elif scheme == "number":
            effective_valid |= cls.NUMBER_SET
        elif scheme == "circled_number":
            effective_valid |= cls.CIRCLED_NUMBER_SET

        output = cls._scan_candidates(prediction, scheme, effective_valid)
        if output is None and scheme in ("letter", "number"):
            output = cls._tail_line_fallback(
                prediction, effective_valid, allowed_chars="A-Ja-j1-9", upper=True,
            )
        return output

    @classmethod
    def _extract_multichoice_api(
        cls,
        prediction: str,
        query: str,
        options: Optional[List[str]] = None,
        option_contents: Optional[List[str]] = None,
        api_name: str = "gpt-4o-mini-2024-07-18",
        temperature: float = 0.0,
        seed: int = 20251020,
        max_tokens: int = 4,
        do_async: Optional[bool] = False,
        verbose: bool = False,
    ) -> Optional[str]:
        # construct messages
        query = cls.EXTRACT_PROMPT.format(
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
            system_message=None,
            generation_options={
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

        logger.debug(f'MultichoiceProcessor (API): {prediction} -> {output}')
        return output

    @classmethod
    def is_multichoice(
        cls,
        query: str,
        options: Optional[List[str]] = None,
        option_contents: Optional[List[str]] = None,
        return_choices: bool = False,
    ) -> Tuple[Optional[str], set[str]]:
        """
        Detect the multiple-choice scheme used in the query.

        Returns:
            A tuple (scheme, valid_choices):
                - scheme: "letter" if choices look like A/B/C/D,
                        "number" if choices look like 1/2/3/4,
                        or None if no scheme is detected.
                - valid_choices: a set of detected choice symbols (e.g., {"A","B","C","D"} or {"1","2","3","4"}).
                If detection is weak, this may be empty and the caller should fallback to LETTER_SET/NUMBER_SET.
        """
        scheme, choices = None, set()

        if isinstance(options, (list, tuple)):
            if all([isinstance(e, str) for e in options]):
                normalized_options = [e.strip() for e in options]
                if all([e in cls.NUMBER_SET for e in normalized_options]):
                    scheme, choices = "number", normalized_options
                elif all([e in cls.CIRCLED_NUMBER_SET for e in normalized_options]):
                    scheme, choices = "circled_number", normalized_options
                else:
                    scheme, choices = "letter", normalized_options
            elif all([isinstance(e, int) for e in options]):
                scheme, choices = "number", [str(e) for e in options]

        elif isinstance(option_contents, (list, tuple)):
            if all([isinstance(e, str) for e in option_contents]):
                normalized_option_contents = [e.strip() for e in option_contents]
                if all([e in cls.NUMBER_SET for e in normalized_option_contents]):
                    scheme, choices = "number", normalized_option_contents
                elif all([e in cls.CIRCLED_NUMBER_SET for e in normalized_option_contents]):
                    scheme, choices = "circled_number", normalized_option_contents
                else:
                    scheme, choices = "letter", normalized_option_contents
            elif all([isinstance(e, int) for e in option_contents]):
                scheme, choices = "number", [str(e) for e in option_contents]

        else:
            q = query or ""

            # Per-token list scan: collect distinct choice markers across the
            # whole query. Handles both line-start enumerations ("A) foo\nB) bar")
            # and inline bracketed forms ("(A) foo (B) bar").
            def _collect(pattern: str) -> set:
                return {
                    (m.group("tok") or m.group("tok2"))
                    for m in re.finditer(pattern, q)
                    if (m.group("tok") or m.group("tok2"))
                }

            letter_hits = _collect(cls.LETTER_PATTERN)
            number_hits = _collect(cls.NUMBER_PATTERN)
            circled_number_hits = _collect(cls.CIRCLED_NUMBER_PATTERN)

            # Keep only valid tokens (LETTER pattern can capture lowercase from the
            # inline bracket alternative; normalize via the uppercase set check).
            letter_hits = {t.upper() for t in letter_hits} & cls.LETTER_SET
            number_hits &= cls.NUMBER_SET
            circled_number_hits &= cls.CIRCLED_NUMBER_SET

            # Heuristic: having at least 2 distinct labels is a sign of MC format
            has_letters = len(letter_hits) >= 2
            has_numbers = len(number_hits) >= 2
            has_circled_numbers = len(circled_number_hits) >= 2

            if not (has_letters or has_numbers or has_circled_numbers):
                scheme, choices = None, set()
            elif has_letters and not has_numbers and not has_circled_numbers:
                scheme, choices = "letter", (letter_hits or cls.LETTER_SET)
            elif has_numbers and not has_letters and not has_circled_numbers:
                scheme, choices = "number", (number_hits or cls.NUMBER_SET)
            elif has_circled_numbers and not has_numbers and not has_letters:
                scheme, choices = "circled_number", (circled_number_hits or cls.CIRCLED_NUMBER_SET)
            else:
                # If more than 1 detected, choose the largest set
                max_hits = max(len(letter_hits), len(number_hits), len(circled_number_hits))
                if len(letter_hits) == max_hits:
                    scheme, choices = ("letter", letter_hits or cls.LETTER_SET)
                elif len(number_hits) == max_hits:
                    scheme, choices = ("number", number_hits or cls.NUMBER_SET)
                else:
                    scheme, choices = ("circled_number", circled_number_hits or cls.CIRCLED_NUMBER_SET)

        if return_choices:
            return scheme, choices
        else:
            return scheme is not None

    @classmethod
    def _extract_candidate(
        cls,
        prediction: str,
        scheme: str,
    ) -> list:
        """
        Extract candidate choice tokens from the prediction text according to the detected scheme.

        Priority:
            1) After keywords like "answer/ans/정답/답/option/choice/선택"
            2) Inside brackets/parentheses
            3) As standalone tokens

        Args:
            prediction: The raw model/system prediction text.
            scheme: "letter" or "number" indicating which token family to extract.

        Returns:
            A de-duplicated list of candidate tokens in order of appearance (e.g., ["B", "A"] or ["2", "3"]).
        """
        p = prediction.strip()

        cands = list()
        # First pattern in each list is reserved for \boxed{X}; take the LAST occurrence
        # since reasoning-style outputs may produce intermediate \boxed{} during scratch work
        # and the final answer is the one near the tail.
        if scheme == "letter":
            for idx, pat in enumerate(cls.LETTER_EXTRACT_PATTERNS):
                if idx == 0:
                    matches = list(re.finditer(pat, p))
                    if matches:
                        cands.append(matches[-1].group("choice"))
                    continue
                m = re.search(pat, p)
                if m:
                    cands.append(m.group("choice"))

        elif scheme == "number":
            for pat in cls.NUMBER_EXTRACT_PATTERNS:
                m = re.search(pat, p)
                if m:
                    cands.append(m.group("choice"))

        else:  # circled_number
            for pat in cls.CIRCLED_NUMBER_EXTRACT_PATTERNS:
                m = re.search(pat, p)
                if m:
                    cands.append(m.group("choice"))
            # NFKC at the entry may have dissolved ①-⑨ into ASCII "1"-"9"; try
            # NUMBER patterns and map back so valid-set membership works in the
            # auto-detect (default) path. version_name="circled" branch skips NFKC so
            # this loop will silently find no NUMBER matches there.
            for pat in cls.NUMBER_EXTRACT_PATTERNS:
                m = re.search(pat, p)
                if m:
                    ascii_digit = m.group("choice")
                    if ascii_digit in cls._ASCII_TO_CIRCLED:
                        cands.append(cls._ASCII_TO_CIRCLED[ascii_digit])

        # stable-unique
        seen, ordered = set(), []
        for c in cands:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered
