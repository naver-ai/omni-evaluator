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
from typing import List, Tuple, Optional, Union, Any, Dict

from omni_evaluator.postprocess._interface import ProcessorInterface

logger = logging.getLogger(__name__)


class BinaryProcessor(ProcessorInterface):
    POSITIVE_REGEX = [
        # English positives
        r"\btrue\b", r"\bcorrect\b", r"\bright\b", r"\byes\b", r"\byep\b",
        r"\baffirmative\b", r"\bentail(?:s|ed|ment)?\b", r"\bconsistent\b", r"\bvalid\b",
        # Korean positives
        r"맞(?:다|음|아요|습니다|지요|죠)", r"네", r"예", r"정답", r"\b참\b",
        r"일치(?:함|한다|합니다|이다)?", r"그렇(?:다|습니다|아요)",
        r"옳(?:다|음|아요|습니다)?",
    ]
    NEGATIVE_REGEX = [
        # English negatives
        r"\bfalse\b", r"\bincorrect\b", r"\bnot\s+true\b", r"\bnot\s+correct\b",
        r"\bwrong\b", r"\bno\b", r"\bcontradiction\b", r"\brefuted?\b", r"\bnope\b",
        r"\bisn't\b", r"\baren't\b", r"\bdon't\b", r"\bdoesn't\b",
        # Korean negatives
        r"오답", r"거짓",
        r"맞지\s*않", r"그렇지\s*않",
        r"일치하지\s*않", r"불일치",
        r"아니(?:다|야|예|요|오)?", r"아님", r"않(?:다|아요|습니다)?",
        r"틀(?:리다|림|립니다|려요|렸다|렸음|렸어요|립니다|렸습니다)?",
    ]
    POSITIVE_PATTERN = re.compile("|".join(POSITIVE_REGEX), re.IGNORECASE)
    NEGATIVE_PATTERN = re.compile("|".join(NEGATIVE_REGEX), re.IGNORECASE)

    # Conservative startswith priority — lmms-eval `extract_pred` style. Used
    # only when the (post-think) response opens with an unambiguous token, so
    # answers like "yes, the man is dunking" are picked up cleanly instead of
    # falling through to the slower regex pass.
    _STARTSWITH_POSITIVE: Tuple[str, ...] = (
        "yes", "true", "correct", "right", "affirmative", "yep",
    )
    _STARTSWITH_NEGATIVE: Tuple[str, ...] = (
        "no", "false", "incorrect", "wrong", "nope",
    )

    # LLM fallback system message — used only when `api_name` is set + the
    # master `postprocess_allow_api` switch is on (matches MultichoiceProcessor).
    _API_SYSTEM_PROMPT = (
        "You are a yes/no classifier. Given a free-form response, output exactly "
        "one word: 'yes' (affirmative/true/correct), 'no' (negative/false/"
        "incorrect), or 'unknown' if the response cannot be classified."
    )
    _API_MAX_TOKENS = 10

    @classmethod
    def extract(
        cls,
        prediction: str,
        version_name: Optional[str] = None,
        api_name: Optional[str] = None,
        query: Optional[str] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ):
        """Infer a boolean from free-form model output.

        Pipeline stages (lmms-eval `extract_pred` parity + multichoice-style
        enhancements; existing bool/None return contract preserved):

          1) NFKC normalize (fullwidth / mixed Unicode → ASCII).
          2) Strip ``<think>...</think>`` CoT trace.
          3) startswith priority — match only when the response opens with an
             unambiguous yes/no/true/false token (conservative; resolves
             clauses like ``"yes, the man is dunking"`` here without falling
             through to the broader regex pass).
          4) Regex search anywhere in text — preserves existing coverage for
             embedded affirmations / negations. Negation wins over affirmation
             to avoid ``"not true"`` → True false-positives.
          5) LLM fallback when ``api_name`` is provided and stages 3-4 are
             undecidable. Returns the same bool/None contract.

        Args:
            prediction: Model's response text.
            version_name: Reserved for output-format selection. Currently
                ignored — extract always returns bool/None for backward
                compatibility with existing callers.
            api_name: Optional LLM identifier. When set AND the framework
                master switch (``postprocess_allow_api`` / ``allow_api``) is
                on, regex-undecidable inputs are forwarded to the model.
            query: Optional question context — currently passed through to
                the LLM fallback prompt only.
            verbose: If True, log decision steps.

        Returns:
            True if the response is affirmative, False if negative, None if
            undecidable (no startswith / regex / LLM cue found).
        """
        if not isinstance(prediction, str) or not prediction:
            return None

        # Stage 1 — NFKC normalize (parity with MultichoiceProcessor).
        _norm = unicodedata.normalize("NFKC", prediction).strip()

        # Stage 2 — strip <think>...</think> CoT trace (parity with
        # MultichoiceProcessor's internal strip). rfind handles nested /
        # repeated think blocks by taking the outermost end.
        _think_end = _norm.rfind("</think>")
        _scan_text = (
            _norm[_think_end + len("</think>"):].strip()
            if _think_end != -1 else _norm
        )

        _output: Optional[bool] = None

        # Stage 3 — startswith priority. lmms-eval `extract_pred` style.
        if _scan_text:
            _scan_lower = _scan_text.lower()
            for _pos in cls._STARTSWITH_POSITIVE:
                if _scan_lower.startswith(_pos):
                    _output = True
                    break
            if _output is None:
                for _neg in cls._STARTSWITH_NEGATIVE:
                    if _scan_lower.startswith(_neg):
                        _output = False
                        break

        # Stage 4 — regex search anywhere (existing behavior). Negation has
        # priority over affirmation.
        if _output is None:
            if cls.NEGATIVE_PATTERN.search(_scan_text):
                _output = False
            elif cls.POSITIVE_PATTERN.search(_scan_text):
                _output = True

        # Stage 5 — LLM fallback. Only fires when regex stages were
        # undecidable AND the master api switch left a non-None api_name.
        if _output is None and isinstance(api_name, str) and len(api_name) > 0:
            _output = cls._extract_binary_api(
                prediction=_scan_text, query=query,
                api_name=api_name, verbose=verbose,
            )

        if verbose:
            logger.debug(f'BinaryProcessor: {prediction!r} -> {_output}')

        # Return contract: bool / None (unchanged for backward compatibility).
        # The `version_name` kwarg is accepted for forward compatibility with
        # framework-wide conventions but currently does not switch the type.
        return _output

    @classmethod
    def _extract_binary_api(
        cls,
        prediction: str,
        api_name: str,
        query: Optional[str] = None,
        verbose: bool = False,
    ) -> Optional[bool]:
        """LLM fallback — asks a small classifier to map the response to
        yes/no/unknown. Returns bool/None to match `extract()`'s contract.
        Failures (network / parse) silently return None so the chain stays
        deterministic from the caller's perspective.
        """
        try:
            from omni_evaluator.api.chat_completions import chat_completion_sync
            from omni_evaluator.schemas.chat import (
                Message as ChatMessage,
                TextContent as ChatTextContent,
            )
        except ImportError as ex:
            logger.warning(f'BinaryProcessor LLM fallback unavailable: {ex}')
            return None

        _user_parts: List[str] = list()
        if isinstance(query, str) and query.strip():
            _user_parts.append(f"Question: {query.strip()}")
        _user_parts.append(f"Response: {prediction}")
        _user_parts.append("Classify (yes/no/unknown):")
        _user_text = "\n\n".join(_user_parts)

        try:
            _response = chat_completion_sync(
                api_name=api_name,
                messages=[
                    ChatMessage(
                        role="system",
                        content=[ChatTextContent(type="text", value=cls._API_SYSTEM_PROMPT)],
                    ),
                    ChatMessage(
                        role="user",
                        content=[ChatTextContent(type="text", value=_user_text)],
                    ),
                ],
                generation_options={
                    "temperature": 0.0,
                    "max_tokens": cls._API_MAX_TOKENS,
                },
            )
        except Exception as ex:
            if verbose:
                logger.warning(f'BinaryProcessor LLM fallback API error: {ex}')
            return None

        _raw = None
        if isinstance(_response, dict):
            _raw = _response.get("prediction")
        elif _response is not None:
            _raw = getattr(_response, "prediction", None)
        if not isinstance(_raw, str) or not _raw.strip():
            return None
        _raw_lower = _raw.strip().lower()
        if _raw_lower.startswith("yes") or _raw_lower == "y":
            return True
        if _raw_lower.startswith("no") or _raw_lower == "n":
            return False
        return None

    @classmethod
    def is_binary(
        cls,
        query: str,
    ):
        if not isinstance(query, str) or len(query) < 1:
            return False

        # Normalize (helps with fullwidth chars, mixed Unicode forms)
        query = unicodedata.normalize("NFKC", query).strip()

        if (
            cls.POSITIVE_PATTERN.search(query)
            or cls.NEGATIVE_PATTERN.search(query)
        ):
            return True
        else:
            return False
