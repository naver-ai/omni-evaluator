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

import json
import logging
import os
import re
import string
from typing import List, Tuple, Optional, Union, Any, Dict

from omni_evaluator.postprocess._interface import ProcessorInterface

logger = logging.getLogger(__name__)

_ENGLISH_SPELLING_MAPPING_PATH = os.path.join(
    os.path.dirname(__file__), "resources", "english.json",
)
_english_text_normalizer = None
_jiwer_default_pipeline = None


def _get_english_text_normalizer():
    global _english_text_normalizer
    if _english_text_normalizer is not None:
        return _english_text_normalizer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    with open(_ENGLISH_SPELLING_MAPPING_PATH, "r") as f:
        mapping = json.load(f)
    _english_text_normalizer = EnglishTextNormalizer(mapping)
    return _english_text_normalizer


def _get_jiwer_default_pipeline():
    global _jiwer_default_pipeline
    if _jiwer_default_pipeline is not None:
        return _jiwer_default_pipeline
    import jiwer
    _jiwer_default_pipeline = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemovePunctuation(),
    ])
    return _jiwer_default_pipeline


class AsrProcessor(ProcessorInterface):
    # PATTERN__ASR = r'((?:\\.|[^"\\])*)'
    PATTERN__ASR = r':\s*[\'"]?(.*?)[\'"]?\s*$'
    SPLIT_PATTERNS = [
        "다음은", "이것은", # ko
    ]
    NORMALIZE_DIGITS_TO_WORDS = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen',
        '17': 'seventeen', '18': 'eighteen', '19': 'nineteen',
        '20': 'twenty', '30': 'thirty', '40': 'forty', '50': 'fifty',
        '60': 'sixty', '70': 'seventy', '80': 'eighty', '90': 'ninety',
    }
    NORMALIZE_CONTRACTIONS = {
        "i'm": "i am",
        "you're": "you are",
        "he's": "he is",
        "she's": "she is",
        "it's": "it is",
        "we're": "we are",
        "they're": "they are",
        "i've": "i have",
        "you've": "you have",
        "we've": "we have",
        "they've": "they have",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "that's": "that is",
    }
    NORMALIZE_PARENTHESES_PATTERN = r'(\[|\(|\{|\<)[^\(\)\n\[\]]*(\]|\)|\}|\>)'
    NORMALIZE_NON_SPEECH_PATTERN = r'\b(uh|umm|um|er|ah)\b'

    @classmethod
    def extract(
        cls,
        prediction: str,
        query: Optional[str] = None,
        version_name: Optional[str] = None,
        api_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ):
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
            # output = cls._extract_asr_api(
            #     prediction=prediction,
            #     query=query,
            #     api_name=api_name,
            #     verbose=verbose,
            # )
            raise NotImplementedError('AsrProcessor._extract_asr_api() is not implemented')
        else:
            if version_name == "qwen":
                output = cls._extract_qwen(
                    prediction=prediction,
                    query=query,
                )
            elif version_name == "default_korean":
                output = cls._extract_default_korean(
                    prediction=prediction,
                    query=query,
                )
            elif version_name == "default_chinese":
                output = cls._extract_default_chinese(
                    prediction=prediction,
                    query=query,
                )
            else:
                output = cls._extract_default(
                    prediction=prediction,
                    query=query,
                )

        logger.debug(f'AsrProcessor ({version_name}): {prediction} -> {output}')
        return output

    @classmethod
    def normalize_default(
        cls,
        text: str,
        **kwargs,
    ) -> str:
        """Whisper-style English ASR normalization.

        Pipeline: lowercase -> Whisper EnglishTextNormalizer -> digit/contraction expansion ->
        bracketed-content removal -> jiwer (multi-spaces, contractions, kaldi non-words, punctuation) ->
        non-speech token removal -> strip.
        """
        if not isinstance(text, str):
            return text

        text = text.lower()

        # Whisper EnglishTextNormalizer (number normalization, contraction expansion, spelling)
        text = _get_english_text_normalizer()(text)

        # Digit-to-word substitutions
        for _digit, _word in cls.NORMALIZE_DIGITS_TO_WORDS.items():
            text = re.sub(r'\b' + _digit + r'\b', _word, text)

        # Common contraction expansion
        for _contraction, _expanded in cls.NORMALIZE_CONTRACTIONS.items():
            text = re.sub(r'\b' + _contraction + r'\b', _expanded, text)

        # Remove bracketed content: [], (), {}, <>
        text = re.sub(cls.NORMALIZE_PARENTHESES_PATTERN, "", text)

        # jiwer pipeline: collapse spaces, expand contractions, drop kaldi non-words, strip punctuation
        text = _get_jiwer_default_pipeline()(text)

        # Remove non-speech filler tokens
        text = re.sub(cls.NORMALIZE_NON_SPEECH_PATTERN, '', text).strip()

        return text

    NORMALIZE_KOREAN_MARKER_PATTERN = r'\b\S+/(?=\s|$)'
    NORMALIZE_KOREAN_PUNCT_PATTERN = r'[，。！？；：、""''（）【】《》…—\-]'

    @classmethod
    def normalize_korean(
        cls,
        text: str,
        **kwargs,
    ) -> str:
        """Korean ASR normalization (e.g., KsponSpeech-style transcripts).

        Pipeline: strip KsponSpeech word-level markers (`X/` such as `o/`, `b/`, `l/`,
        `n/`, `어/`, `아/`, `그/`) -> strip ASCII + Korean punctuation -> collapse
        whitespace -> strip.
        """
        if not isinstance(text, str):
            return text

        # Drop trailing-slash transcription markers like `o/`, `어/`, `b/`
        text = re.sub(cls.NORMALIZE_KOREAN_MARKER_PATTERN, ' ', text)

        # Remove bracketed content: [], (), {}, <>
        text = re.sub(cls.NORMALIZE_PARENTHESES_PATTERN, ' ', text)

        # Strip ASCII punctuation
        text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', text)

        # Strip common Korean / fullwidth punctuation
        text = re.sub(cls.NORMALIZE_KOREAN_PUNCT_PATTERN, ' ', text)

        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    NORMALIZE_CHINESE_CHAR_PATTERN = r'([一-鿿]+)'

    @classmethod
    def _separate_and_space_chinese(cls, text: str) -> str:
        """Insert space between adjacent CJK Unified Ideographs; non-CJK runs preserved as-is.

        Splits on contiguous CJK runs, char-joins each run with spaces, and
        concatenates with the non-CJK runs unchanged. Lets jiwer word-WER act
        as char-WER over hanzi while keeping any code-switch English/Latin
        tokens whole.
        """
        if not isinstance(text, str):
            return text
        parts = re.split(cls.NORMALIZE_CHINESE_CHAR_PATTERN, text)
        return ''.join(
            ' '.join(_part) if re.match(r'[一-鿿]+', _part) else _part
            for _part in parts
        )

    @classmethod
    def normalize_chinese(
        cls,
        text: str,
        **kwargs,
    ) -> str:
        """Chinese ASR normalization for code-switch (Chinese + Latin) transcripts.

        Pipeline: full English `normalize_default` (lowercase + Whisper
        EnglishTextNormalizer + digit/contraction expansion + bracket/non-speech
        removal + jiwer punctuation strip, which absorbs fullwidth CJK
        punctuation too via Unicode category P), then `_separate_and_space_chinese`
        so jiwer word-WER becomes char-WER over hanzi while preserving any
        code-switch Latin tokens.
        """
        if not isinstance(text, str):
            return text
        text = cls.normalize_default(text=text)
        text = cls._separate_and_space_chinese(text=text)
        return text

    @classmethod
    def _extract_default(
        cls,
        prediction: str,
        query: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        """
        BEFORE: The transcription of the speech is: 'Are you certain that this is the Mediterranean?'
        AFTER: Are you certain that this is the Mediterranean?
        """
        
        output = None
        _match = re.search(cls.PATTERN__ASR, prediction)
        if _match:
            output = _match.group(1)
        elif ':"' in prediction:
            output = '"' + prediction.split(':"')[1]
        elif re.search(r':\s*"', prediction):
            output = '"' + re.split(r':\s*"', prediction, maxsplit=1)[1]
        else:
            for _pattern in cls.SPLIT_PATTERNS:
                if not prediction.startswith(_pattern):
                    continue
                prediction = prediction[len(_pattern):].strip()
        return output
    
    # Shared markers and patterns for Korean/Chinese extraction
    _TRANSCRIPTION_MARKERS = ["Final Transcription:", "output:", "transcribed text is:"]
    _QUOTE_PATTERNS = [
        r"'([^']+)'",  # Single quotes
        r'"([^"]+)"',  # Double quotes
    ]
    _PREFIX_PATTERNS = [
        r"^the\s+(transcription|transcript|text)\s+(of\s+(the|your)\s+audio\s+is|is)[\s:]*",
        r"^the\s+original\s+content\s+of\s+this\s+audio\s+is[\s:]*",
        r"^here\s+is\s+the\s+(transcription|transcript)[\s:]*",
        r"^transcription[\s:]*",
        r"^transcript[\s:]*",
    ]

    @classmethod
    def _extract_transcription_common(
        cls,
        prediction: str,
        charset_pattern: str,
    ) -> str:
        """Common extraction logic for Korean/Chinese ASR:
        1. Extract text after markers.
        2. Extract quoted text matching charset.
        3. Remove English prefixes.
        """
        if prediction is None:
            return ""
        if not isinstance(prediction, str):
            prediction = str(prediction)

        # Step 1: extract text after markers
        for marker in cls._TRANSCRIPTION_MARKERS:
            if marker in prediction:
                prediction = prediction.split(marker, 1)[-1]

        # Step 2: extract quoted text matching charset
        matched_quotes = []
        for pattern in cls._QUOTE_PATTERNS:
            quotes = re.findall(pattern, prediction)
            matched_quotes.extend([q for q in quotes if re.search(charset_pattern, q)])
        if matched_quotes:
            prediction = matched_quotes[-1]

        # Step 3: remove English prefixes
        text_lower = prediction.lower()
        for prefix_pattern in cls._PREFIX_PATTERNS:
            text_lower = re.sub(prefix_pattern, "", text_lower, flags=re.IGNORECASE)
        if len(text_lower) < len(prediction.lower()):
            prediction = prediction[len(prediction) - len(text_lower):]

        return prediction

    @classmethod
    def _extract_default_korean(
        cls,
        prediction: str,
        query: Optional[str] = None,
        **kwargs,
    ):
        prediction = cls._extract_transcription_common(prediction, charset_pattern=r'[가-힣]')

        # Cleanup: remove brackets, newlines, punctuation
        prediction = prediction.replace("\n", " ").replace("\r", " ").replace(".", "").replace(",", "").replace("*", "")
        prediction = re.sub(r"\[.*?\]", "", prediction)
        prediction = re.sub(r"\(.*?\)", "", prediction)
        cleaned_text = re.sub(r"\s+", " ", prediction).strip()

        # Return "" if no Korean characters remain
        if not re.search(r'[가-힣]', cleaned_text):
            return ""
        return cleaned_text

    @classmethod
    def _extract_default_chinese(
        cls,
        prediction: str,
        query: Optional[str] = None,
        **kwargs,
    ):
        prediction = cls._extract_transcription_common(prediction, charset_pattern=r'[\u4e00-\u9fff]')

        # Cleanup: remove fullwidth/halfwidth punctuation
        cleaned_text = prediction.replace("\n", " ").replace("\r", " ").replace(".", "").replace(",", "").replace("*", "")
        cleaned_text = re.sub(r"[，。！？；：、""''（）【】《》\-\s]", "", cleaned_text)
        cleaned_text = re.sub(f"[{re.escape(string.punctuation)}]", "", cleaned_text)
        # Split by character
        cleaned_text = " ".join(list(cleaned_text))
        return cleaned_text
        
    @classmethod
    def _extract_qwen(
        cls,
        prediction: str,
        query: Optional[str] = None,
        **kwargs,
    ):
        raise NotImplementedError('AsrProcessor._extract_qwen() is not implemented')