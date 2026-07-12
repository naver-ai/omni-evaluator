# Reference from https://github.com/jitsi/jiwer (Apache-2.0)
# Reference from https://github.com/enesyugan/PIER-CodeSwitching-Evaluation (Apache-2.0)

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

import inflect
from itertools import chain
import jiwer
from jiwer.transformations import wer_default, cer_default
import re
import regex
from typing import Any, Dict, List, Optional, Union, Literal, Tuple
import warnings

from omni_evaluator.utils.data import safe_percentage


# Reference from https://github.com/jitsi/jiwer (Apache-2.0) - jiwer/process.py
def _apply_transform(
    sentence: Union[str, List[str]],
    transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform],
    is_reference: bool,
):
    # Apply transforms. The transforms should collapse input to a
    # list with lists of words
    transformed_sentence = transform(sentence)

    # Validate the output is a list containing lists of strings
    if is_reference:
        if not _is_list_of_list_of_strings(
            transformed_sentence, require_non_empty_lists=True
        ):
            raise ValueError(
                "After applying the transformation, each reference should be a "
                "non-empty list of strings, with each string being a single word."
            )
    else:
        if not _is_list_of_list_of_strings(
            transformed_sentence, require_non_empty_lists=False
        ):
            raise ValueError(
                "After applying the transformation, each hypothesis should be a "
                "list of strings, with each string being a single word."
            )

    return transformed_sentence


# Reference from https://github.com/jitsi/jiwer (Apache-2.0) - jiwer/process.py
def _is_list_of_list_of_strings(x: Any, require_non_empty_lists: bool):
    if not isinstance(x, list):
        return False

    for e in x:
        if not isinstance(e, list):
            return False

        if require_non_empty_lists and len(e) == 0:
            return False

        if not all([isinstance(s, str) for s in e]):
            return False

    return True


# Reference from https://github.com/jitsi/jiwer (Apache-2.0) - jiwer/process.py
def _word2char(reference: List[List[str]], hypothesis: List[List[str]]):
    # tokenize each word into an integer
    vocabulary = set(chain(*reference, *hypothesis))

    if "" in vocabulary:
        raise ValueError(
            "Empty strings cannot be a word. "
            "Please ensure that the given transform removes empty strings."
        )

    word2char = dict(zip(vocabulary, range(len(vocabulary))))

    reference_chars = [
        "".join([chr(word2char[w]) for w in sentence]) for sentence in reference
    ]
    hypothesis_chars = [
        "".join([chr(word2char[w]) for w in sentence]) for sentence in hypothesis
    ]

    return reference_chars, hypothesis_chars

# Reference from https://github.com/enesyugan/PIER-CodeSwitching-Evaluation (Apache-2.0) - jiwer/process.py
def tokenize_for_mer(text):
    """
    split Hiragana, Katakana, Kanji/Han characters similar to Mixed-error-rate
    """
    #reg_range = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    reg_range = r"[\u4E00-\u9FFF]|[\u3040-\u309F]|[\u30A0-\u30FF]|[\uFF00-\uFFEF]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
    matches = re.findall(reg_range, text, re.UNICODE)
    p = inflect.engine()
    res = []
    for item in matches:
        try:
            temp = p.number_to_words(item) if (item.isnumeric() and len(regex.findall(r'\p{Han}+', item)) == 0) else item
        except Exception:
            temp = item
        res.append(temp)
    return res


# Reference from https://github.com/enesyugan/PIER-CodeSwitching-Evaluation (Apache-2.0) - jiwer/process.py
def tag_words(words, switch=False):
    latin_containing_pattern = r'\b\w*[a-zA-Z]+\w*\b'
    latin_pattern = r'\b[a-zA-Z]+(?:\'[a-zA-Z]+)?\b'

    num_words = len(words)
    eng_words = 0
    eng_chars = 0
    mixed_words = 0
    mixed_chars = 0
    rest = 0
    rest_chars = 0

    for i, word in enumerate(words):
        if re.match(latin_pattern, word):
            eng_words += 1
            eng_chars += len(word)
            if not re.search(r'<tag\s.*?>.*?</eng>', word) and not switch:
                words[i] = f'<tag {word}>'

        elif re.match(latin_containing_pattern, word):
            #eng_words += 1
            mixed_words += 1
            mixed_chars += len(word)
            if not re.search(r'<tag\s.*?>.*?</eng>', word):
                words[i] = f'<tag {word}>'
        else:
            rest += 1
            rest_chars += len(word)
            if switch and not re.search(r'<tag\s.*?>.*?</eng>', word):
                words[i] = f'<tag {word}>'
                

    res = {
        "words": words,
        "eng_words": eng_words,
        "mixed_words": mixed_words,
        "rest": rest,
        "eng_chars": eng_chars,
        "mixed_chars": mixed_chars,
        "rest_chars": rest_chars,
        }
    return res

# Reference from https://github.com/enesyugan/PIER-CodeSwitching-Evaluation (Apache-2.0) - jiwer/process.py
def tag_poi_words(text, scd_language, matrix_lang=None, fixedtags=False):
    if "<tag" in text.split(): raise ValueError(f"Your REF file contains tagged words '<tag', you also set scd_language to {scd_language}. Choose one")
    if scd_language == "cmn" or scd_language == "jap":
        text = " ".join(tokenize_for_mer(text))
    words = text.split()
    orig_words = words.copy()

    if matrix_lang == "eng":
        res = tag_words(words, switch=True)
    else:
        res = tag_words(words)

    if res["eng_words"] +res["mixed_words"] == len(words):
        return text
    if res["rest"] == len(words):
        return text

    if not fixedtags:
        if (res["eng_words"]+res["mixed_words"]+res["rest"]) != len(words): 
            raise ValueError("eng_words + mixed_words+ rest should equal total word")
        elif res["eng_words"] > res["rest"]:
            res = tag_words(orig_words, switch=True)

    tagged_text = ' '.join(res["words"])
    return tagged_text

# Reference from https://github.com/enesyugan/PIER-CodeSwitching-Evaluation (Apache-2.0) - jiwer/process.py
def extract_indices(text, split_hyphen, scd_language, matrix_lang, fixedtags):
    #print(text)
    if scd_language != None: 
        ##matrix_lang="cmn"
        text = tag_poi_words(text, scd_language, matrix_lang=matrix_lang, fixedtags=True)
    #else:
        #   text = add_space_before_punctuation(text)
    #print(text)
    # Correct the incorrect annotation pattern by removing the space before '>'
    corrected_text = re.sub(r'<tag\s+([^>]+)\s*>', r'<tag \1>', text)
    pattern = re.compile(r'<tag (.*?)>')
    poi_indices = []
    tags = 0
    if split_hyphen:
        all_words = text.replace("-", " ").split()
    else:
        all_words = text.split()

    for match in pattern.finditer(text):
        poi_text = match.group(1)
        if split_hyphen:
            start_index = len(text[:match.start()].replace("-", " ").split()) -tags
            words_in_poi = poi_text.replace("-", " ").split()
            end_index = start_index + len(words_in_poi)
        else:
            start_index = len(text[:match.start()].split()) - tags
            end_index = start_index + len(poi_text.split())
        poi_indices.extend(range(start_index, end_index))
        tags += 1
    
    all_indices = list(range(len(all_words)-tags))
    other_indices = [index for index in all_indices if index not in poi_indices]
   # print(text)
   # print(f"p: {poi_indices} o: {other_indices}")
   # if len(poi_indices) > len(other_indices):  
   #     print(text)
       # poi = poi_indices.copy()
       # poi_indices = other_indices
        #other_indices =poi

#    else:
 #      print("CMN")
    return poi_indices, other_indices

def process_pier(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    reference_transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform] = cer_default,
    hypothesis_transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform] = cer_default,
	scd_language: str=None,
	split_hyphen: bool=False,
):
    """
    # reference: https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/blob/e3436cd269a6c3e89495d16f0376877c936ebd1e/jiwer/process.py#L339
    Compute word-level levenstein disatnace and alignment between one or more reference and hypothesis sentences.
    Based on tagged words relevent alignements are extrected and PER is calculated.
    """

    # validate input type
    if isinstance(reference, str):
        reference = [reference]
    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]
    if any(len(t) == 0 for t in reference):
        raise ValueError("one or more references are empty strings")

    poi_indices, other_indices = [], []
    num_poi_words, num_other_words = [], []
    reference_notag = []

    #matrix_lang = determine_matrix_language(reference, split_hyphen, scd_language) if scd_language!= None else None
    matrix_lang = scd_language if scd_language!= None else None
    print(f"MATRIX LANG: {matrix_lang}")
    for ref in reference:
        poi_ind, o_ind = extract_indices(ref, split_hyphen, scd_language,  matrix_lang, fixedtags=True)
        #print(f"{ref} l: {len(ref.split())}\n{poi_ind}\n{o_ind}")
        poi_indices.append(poi_ind)
        other_indices.append(o_ind)
        num_poi_words.append(len(poi_indices))
        num_other_words.append(len(other_indices))
        ref_notag = re.sub(r'<tag (.*?)>', r'\1', ref)
        if split_hyphen:
            ref_notag = ref_notag.replace("-", " ")
        #reference_notag.append(re.sub(r'<tag (.*?)>', r'\1', ref))
        reference_notag.append(ref_notag)

    # pre-process reference and hypothesis by applying transforms
    ref_transformed = _apply_transform(
        reference_notag, reference_transform, is_reference=True
    )
    hyp_transformed = _apply_transform(
        hypothesis, hypothesis_transform, is_reference=False
    )
    
    if len(ref_transformed) != len(hyp_transformed):
        raise ValueError(
            "After applying the transforms on the reference and hypothesis sentences, "
            f"their lengths must match. "
            f"Instead got {len(ref_transformed)} reference and "
            f"{len(hyp_transformed)} hypothesis sentences."
        )

    # Change each word into a unique character in order to compute
    # word-level levenshtein distance
    ref_as_chars, hyp_as_chars = _word2char(ref_transformed, hyp_transformed)
    
    I, S, D = 0, 0, 0
    oI, oS, oD = 0, 0, 0
    poiWords, otherWords = 0, 0
    counter = 0
    H, oH = 0, 0

    debug= 0
    for reference_sentence, hypothesis_sentence, poi_idxs, other_idxs in zip(ref_as_chars, hyp_as_chars, poi_indices, other_indices):
        poiWords += len(poi_idxs)
        otherWords += len(other_idxs)
        if len(poi_idxs) <= 0 or len(other_idxs) <= 0:
            counter +=1
            continue
        #if len(poi_idxs) <= 0: counter += 1; continue
        #if len(other_idxs) <= 0: print(reference[counter]); print(counter); counter+= 1; continue
        #print(ref_transformed[debug])
        #print(poi_idxs)
        #print(other_idxs)

        debug+=1
        total_len = len(reference_notag[counter].split())

        edit_ops = rapidfuzz.distance.Levenshtein.editops(
                reference_sentence, hypothesis_sentence
                )
        #print(edit_ops)
        if len(poi_idxs) > 0:
            insertions, deletions, substitutions, hits = get_idsh(edit_ops, poi_idxs, total_len)

            S += substitutions
            I += insertions
            D += deletions
            H += hits

        if len(other_idxs) > 0:
            insertions, deletions, substitutions, hits = get_idsh(edit_ops, other_idxs, total_len)
            oS += substitutions
            oI += insertions
            oD += deletions
            oH += hits
        counter += 1
    #print(f"EEEEEE: {counter}")
    PER = safe_percentage(I+D+S, H+S+D)
    oPER = safe_percentage(oI+oD+oS, oH+oS+oD)
    #print(f"OTHER: {otherWords}")
    res = {
        "poi": {
            "PIER": PER,
            "insertions": I,
            "deletions": D,
            "substitutions":S,
            "hits": H,
            "poiWords": (H+S+D),
            },
        "rest": {
            "PIER": oPER,
            "insertions": oI,
            "deletions": oD,
            "substitutions": oS,
            "hits": oH,
            "otherWords": (oH+oS+oD),
            }
        }

    return res

def pier(
    reference: Union[str, List[str]] = None,
    hypothesis: Union[str, List[str]] = None,
    reference_transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform] = wer_default,
    hypothesis_transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform] = wer_default,
    truth: Union[str, List[str]] = None,
    truth_transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform] = None,
	scd_language: str=None,
	split_hyphen: bool=False,
) -> float:
    """
    # reference: https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/blob/e3436cd269a6c3e89495d16f0376877c936ebd1e/jiwer/measures.py#L408
    Calculate the point of interest error rate (WER) between one or more reference and
    hypothesis sentences.

    Args:
        reference: The reference sentence(s) tagged with PoI words <tag word>
        hypothesis: The hypothesis sentence(s)
        reference_transform: The transformation(s) to apply to the reference string(s)
        hypothesis_transform: The transformation(s) to apply to the hypothesis string(s)
        truth: Deprecated, renamed to `reference`
        truth_transform: Deprecated, renamed to `reference_transform`
        scd_language: Only works with English as embedded and languages not using latin script such as Arabic (ara), or Mandarin (cmn), Japanese (jap), as second language
        split_hyphen: If hyphen should be replaced by space

    Deprecated:
        Arguments `truth` and `truth_transform` have been renamed to respectively
        `reference` and `reference_transform`. Therefore, the keyword arguments
         `truth` and `truth_transform` will be removed in the next release.
         At the same time, `reference` and `reference_transform` will lose their
         default value.

    Returns:
        (float): The word error rate of the given reference and
                 hypothesis sentence(s).
    """
    (
        reference,
        hypothesis,
        reference_transform,
        hypothesis_transform,
    ) = _deprecate_truth(
        reference=reference,
        hypothesis=hypothesis,
        truth=truth,
        reference_transform=reference_transform,
        truth_transform=truth_transform,
        hypothesis_transform=hypothesis_transform,
    )
    output = process_pier(
        reference, hypothesis, reference_transform, hypothesis_transform, scd_language, split_hyphen
    )

    return output

# Reference from https://github.com/jitsi/jiwer (Apache-2.0) - jiwer/measures.py
def _deprecate_truth(
    reference: Union[str, List[str]],
    hypothesis: Union[str, List[str]],
    truth: Union[str, List[str]],
    reference_transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform],
    hypothesis_transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform],
    truth_transform: Union[jiwer.transforms.Compose, jiwer.transforms.AbstractTransform],
):
    """
    # reference: https://github.com/enesyugan/PIER-CodeSwitching-Evaluation/blob/e3436cd269a6c3e89495d16f0376877c936ebd1e/jiwer/measures.py#L462
    """
    if truth is not None:
        warnings.warn(
            DeprecationWarning(
                "keyword argument `truth` is deprecated, please use `reference`."
            )
        )
        if reference is not None:
            raise ValueError("cannot give `reference` and `truth`")
        reference = truth
    if truth_transform is not None:
        warnings.warn(
            DeprecationWarning(
                "keyword argument `truth_transform` is deprecated, "
                "please use `reference_transform`."
            )
        )
        reference_transform = truth_transform

    if reference is None or hypothesis is None:
        raise ValueError(
            "detected default values for reference or hypothesis arguments, "
            "please provide actual string or list of strings"
        )

    return reference, hypothesis, reference_transform, hypothesis_transform