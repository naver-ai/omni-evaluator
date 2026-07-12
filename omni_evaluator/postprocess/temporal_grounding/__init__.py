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
from typing import List, Optional

from omni_evaluator.postprocess._interface import ProcessorInterface

logger = logging.getLogger(__name__)


class TemporalGroundingProcessor(ProcessorInterface):
    """Extract a temporal ``[start, end]`` interval (seconds) from free-form text.

    Models answering temporal-grounding tasks (Charades-STA, ActivityNet-Captions,
    QVHighlights, ...) wrap the interval in prose: ``"The action occurs from
    24.3 to 30.4 seconds"``, ``"Answer: [24.3, 30.4]"``, ``"00:24.3 - 00:30.4"``,
    etc. The downstream ``compute_temporal_iou`` parses with a loose regex, but
    persisting the canonical ``[start, end]`` form on ``prediction_postprocessed``
    makes records auditable and lets evaluators bypass parsing altogether.

    Mirrors :class:`SpatialGroundingProcessor` — same interface, deterministic
    regex, last-match-wins for prose robustness. ``version_name`` is accepted
    for parity but currently has only one mode (``seconds``).

    Output: canonical ``"[start, end]"`` string with floats; ``None`` if no
    parseable interval is found. ``start < end`` and both within ``[0, 1e6]``
    are enforced (matches ``_parse_time_interval`` invariants).
    """

    _NUM = r"-?(?:\d+\.\d+|\.\d+|\d+)"
    # MM:SS(.ms) or HH:MM:SS(.ms) — converted to seconds.
    _TIMESTAMP = r"(?:\d{1,2}:)?\d{1,2}:\d{1,2}(?:\.\d+)?"
    _SEP = r"\s*(?:-|–|—|to|~|until|,)\s*"

    @classmethod
    def _to_seconds(cls, token: str) -> Optional[float]:
        # numeric seconds
        if re.fullmatch(cls._NUM, token):
            try:
                return float(token)
            except ValueError:
                return None
        # HH:MM:SS or MM:SS form
        parts = token.split(":")
        try:
            parts = [float(_p) for _p in parts]
        except ValueError:
            return None
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return None

    @classmethod
    def extract(
        cls,
        prediction: str,
        query: Optional[str] = None,
        version_name: Optional[str] = None,
        api_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ) -> Optional[str]:
        """Return canonical ``"[start, end]"`` second-pair string, or ``None``.

        Stages (parity with :class:`SpatialGroundingProcessor`):
          1) NFKC normalize + strip ``<think>...</think>`` CoT trace.
          2) Prefer the LAST ``start <sep> end`` pair anywhere in the text
             (model's final answer; robust to prose like "I think 12-15s ...
             but the right answer is 24.3 - 30.4 seconds").
          3) Fallback: first two bare numeric/timestamp tokens.

        Accepts numeric (``24.3``), MM:SS (``00:24.3``), HH:MM:SS, and any of
        the separators ``-``, ``–``, ``—``, ``to``, ``~``, ``until``, ``,``.

        ``api_name`` is accepted for ProcessorInterface signature parity but
        passing a non-None value raises ``NotImplementedError`` — no LLM
        fallback is implemented for temporal grounding extraction.
        """
        if isinstance(api_name, str) and len(api_name) > 0:
            raise NotImplementedError(
                'TemporalGroundingProcessor._extract_temporal_grounding_api() is not implemented'
            )

        if not isinstance(prediction, str) or not prediction.strip():
            return prediction if not isinstance(prediction, str) else None

        text = unicodedata.normalize("NFKC", prediction)
        _think_end = text.rfind("</think>")
        if _think_end != -1:
            text = text[_think_end + len("</think>"):]
        text = text.strip()

        token = rf"(?:{cls._TIMESTAMP}|{cls._NUM})"
        pair_re = re.compile(rf"({token}){cls._SEP}({token})", re.IGNORECASE)

        # Stage 2 — last separator-pair (model's final answer wins).
        pair: Optional[List[float]] = None
        for m in pair_re.finditer(text):
            s = cls._to_seconds(m.group(1))
            e = cls._to_seconds(m.group(2))
            if s is None or e is None:
                continue
            if 0.0 <= s < e <= 1e6:
                pair = [s, e]   # keep the last valid one

        # Stage 3 — fallback: first two timestamp/number tokens.
        if pair is None:
            tokens = re.findall(token, text)
            if len(tokens) >= 2:
                s = cls._to_seconds(tokens[0])
                e = cls._to_seconds(tokens[1])
                if s is not None and e is not None and 0.0 <= s < e <= 1e6:
                    pair = [s, e]
                    if verbose:
                        logger.debug(
                            "TemporalGroundingProcessor: no separator pair; using "
                            "first two tokens from %r", prediction,
                        )

        if pair is None:
            if verbose:
                logger.debug("TemporalGroundingProcessor: no interval in %r", prediction)
            return None

        output = "[" + ", ".join(repr(_v) for _v in pair) + "]"
        if verbose:
            logger.debug("TemporalGroundingProcessor: %r -> %s", prediction, output)
        return output
