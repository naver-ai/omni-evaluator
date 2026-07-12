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

from difflib import SequenceMatcher
import logging
import re
from typing import List, Tuple, Optional, Union, Any, Dict

from omni_evaluator.postprocess._interface import ProcessorInterface

logger = logging.getLogger(__name__)


class CodeProcessor(ProcessorInterface):
    PATTERN__CODE_BLOCK = r'```(?:python|java|javascript|sh|cpp|json)?\n((?:(?!```)[\s\S])+)```'
    PATTERN__CODE_START = {
        "python": r'^\s*(def|class|import|from|async|try)\s+'
    }

    @classmethod
    def extract(
        cls,
        prediction: str,
        language: str,
        query: Optional[str] = None,
        extract_continuation: bool = False,
        code_length_threshold: int = 10,
        version: Optional[str] = None,
        api_name: Optional[str] = None,
        verbose: Optional[bool] = False,
        **kwargs,
    ):
        code = cls._extract_code_block(
            prediction=prediction,
            remove_block=True,
            language=language,
        )
        if (
            not isinstance(code, str) 
            or len(code.strip()) < 1
        ): 
            output = None
            
        elif (
            not extract_continuation
            or not isinstance(query, str)
        ): # return code if not extract_continuation
            output = code
        
        else:
            if language not in cls.PATTERN__CODE_START:
                raise ValueError(f'Code_start_pattern has not been defined: {language}')
            
            _matcher = SequenceMatcher(None, query, code)
            _match = _matcher.find_longest_match(0, len(query), 0, len(code))
            _overlap = code[_match.b:_match.b+_match.size]
            if (
                len(_overlap) < code_length_threshold
                or re.search(cls.PATTERN__CODE_START[language], _overlap) is None
            ):
                # regard code as continuation itself if 
                #   - there is no common substring (overlap whose length is less than threshold)
                #   - there is not observation of starting pattern of language
                output = code
            else:
                # leave only continuation by removing overlap with query
                code = cls._extract_code_continuation(
                    generated=code,
                    query=query,
                    language=language,
                    code_length_threshold=code_length_threshold,
                )
                output = code
            
        logger.debug(f'CodeProcessor: {prediction} -> {output}')
        return output
            

    @classmethod
    def _has_code_block(
        cls,
        prediction: str,
    ):
        code_block_match = re.search(cls.PATTERN__CODE_BLOCK, prediction, re.DOTALL)
        if code_block_match is not None:
            return True
        else:
            return False
    
    @classmethod
    def _extract_code_block(
        cls,
        prediction: str,
        remove_block: Optional[bool] = True,
        language: Optional[str] = None,
    ):
        output = None
        output_idx = None
        # add markdown pattern considering truncated case
        left_prediction = f'```{language}\n{prediction}```' 
        pattern = re.compile(cls.PATTERN__CODE_BLOCK, re.DOTALL)
        _match = pattern.search(left_prediction)
        if _match is None:
            return prediction
        elif remove_block:
            output = _match.group(1)
            return output
        else:
            output = _match.group(0)
            return output

    @classmethod
    def _extract_code_continuation(
        cls,
        generated: str,
        query: str, 
        language: str = "python",
        code_length_threshold: int = 10,
    ) -> str:
        code_starts = re.compile(cls.PATTERN__CODE_START[language], re.MULTILINE)
        code_starts = list(code_starts.finditer(query))
        if not code_starts: 
            # if there is no python code pattern 
            return generated

        # the last code part in query
        _last_start = code_starts[-1].start()
        last_query_code = query[_last_start:].strip()
        
        # find the longest common substring
        matcher = SequenceMatcher(None, last_query_code, generated)
        match = matcher.find_longest_match(0, len(last_query_code), 0, len(generated))

        # remove overlapped part in generated
        overlap_index = 0
        if match.size > code_length_threshold:
            # if left length is greater than threshold
            overlap_index = match.b + match.size
        # exclude "\s+" to maintain function workable
        _match = re.search(r'\s+$', generated[:overlap_index])
        if _match is not None: 
            overlap_index -= (_match.end() - _match.start())
        
        continuation = generated[overlap_index:]
        if (
            query.rstrip().endswith('"""')
            and continuation.lstrip().startswith('"""')
        ):
            continuation = continuation.split('"""')[-1]
        return continuation