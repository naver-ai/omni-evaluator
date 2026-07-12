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

"""Unit tests for utils/data.py — argument filtering, field fallback, distributed splitting, prompt formatting."""
import pytest

from omni_evaluator.utils.data import (
    align_tag,
    extract_options,
    filter_arguments,
    find_field,
    format_task_prompt,
    normalize_unit,
    rename_dict,
    shift_options,
    split_iterator,
)


# ─────────────────────────────────────────────────────────────
# align_tag — pad when tag count is short, trim when excess
# ─────────────────────────────────────────────────────────────

def test_align_tag_attach_when_short():
    """When tags are insufficient, fills the deficit according to head/tail option."""
    assert align_tag("abc", "<i>", 2, attach_head=True) == "<i><i>abc"
    assert align_tag("abc", "<i>", 2, attach_head=False) == "abc<i><i>"


def test_align_tag_remove_when_excess():
    """When tags are in excess, removes the surplus from the back leaving exactly num_attach tags."""
    assert align_tag("<i><i><i>", "<i>", 1) == "<i>"


def test_align_tag_equal_unchanged():
    """When tag count matches exactly, leaves it unchanged."""
    assert align_tag("<i>", "<i>", 1) == "<i>"


# ─────────────────────────────────────────────────────────────
# extract_options — split (letter, content) pairs from 'Options:' segment
# ─────────────────────────────────────────────────────────────

def test_extract_options():
    """Splits (letter, content) pairs from the 'Options:' segment."""
    options, contents = extract_options("Options: A: foo, B: bar, C: baz")
    assert options == ["A", "B", "C"]
    assert contents == ["foo", "bar", "baz"]


# ─────────────────────────────────────────────────────────────
# filter_arguments — drop kwargs not in function signature
# ─────────────────────────────────────────────────────────────

def test_filter_arguments():
    """Positional args are mapped by signature order; kwargs not in signature are dropped."""

    def func(a, b, c):
        ...

    assert filter_arguments(func, 1, b=2, unknown=9) == {"a": 1, "b": 2}


# ─────────────────────────────────────────────────────────────
# find_field — search candidate keys in order, ignoring empty values and case
# ─────────────────────────────────────────────────────────────

def test_find_field_returns_first_present():
    """Returns the first candidate key that exists and is non-empty."""
    assert find_field({"x": "v"}, ["missing", "x"]) == "v"


def test_find_field_case_insensitive():
    """Falls back to case-insensitive matching when no exact key is found."""
    assert find_field({"Question": "q"}, ["question"]) == "q"


def test_find_field_skips_empty_and_ignored():
    """Skips None/empty values and values in ignore_values, moving to the next candidate."""
    assert find_field({"a": None, "b": "", "c": "ok"}, ["a", "b", "c"]) == "ok"
    assert find_field({"a": "skip", "b": "keep"}, ["a", "b"], ignore_values=["skip"]) == "keep"


def test_find_field_default_when_all_miss():
    """Returns default when all candidates fail."""
    assert find_field({"x": 1}, ["y", "z"], default="fallback") == "fallback"


# ─────────────────────────────────────────────────────────────
# format_task_prompt — branching by placeholder type (empty/positional/named/none)
# ─────────────────────────────────────────────────────────────

def test_format_task_prompt_empty_returns_query():
    """Returns query as-is when task_prompt is an empty string or non-str."""
    assert format_task_prompt("", "my query") == "my query"


def test_format_task_prompt_query_none_strips_placeholder():
    """Returns task_prompt with leading placeholder removed when query is None."""
    assert format_task_prompt("{instruction} solve this", None) == "solve this"


def test_format_task_prompt_positional_brace():
    """Inserts query via positional format when '{}' is present."""
    assert format_task_prompt("Q: {}", "hi") == "Q: hi"


def test_format_task_prompt_named_placeholders():
    """Named placeholder 'query' is filled with query; remaining are filled from kwargs."""
    assert format_task_prompt("{query} in {lang}", "hi", lang="en") == "hi in en"


def test_format_task_prompt_no_placeholder_concat():
    """When no placeholder exists, appends task_prompt after query with a newline."""
    assert format_task_prompt("Answer concisely.", "hi") == "hi\nAnswer concisely."


# ─────────────────────────────────────────────────────────────
# normalize_unit — fractional approximation of x / unit
# ─────────────────────────────────────────────────────────────

def test_normalize_unit():
    """Returns the fractional approximation of x divided by unit."""
    assert normalize_unit(50, 100) == 0.5
    assert normalize_unit(1, 4) == 0.25


# ─────────────────────────────────────────────────────────────
# rename_dict — replace key names according to rename_map
# ─────────────────────────────────────────────────────────────

def test_rename_dict():
    """Only keys present in both rename_map and obj are moved to the new name."""
    assert rename_dict({"old": 1, "keep": 2}, {"old": "new", "absent": "x"}) == {
        "new": 1,
        "keep": 2,
    }


# ─────────────────────────────────────────────────────────────
# shift_options — circular rotation of multi-run labels
# ─────────────────────────────────────────────────────────────

def test_shift_options_no_shift():
    """Leaves options, contents, and label unchanged when run_index==0."""
    options, contents, label = shift_options(["A", "B"], ["x", "y"], ["x"], run_index=0)
    assert (options, contents, label) == (["A", "B"], ["x", "y"], ["x"])


def test_shift_options_circular():
    """Circularly rotates option_contents and moves the label along when run_index>0."""
    options, contents, label = shift_options(
        ["A", "B", "C"], ["x", "y", "z"], ["x"], run_index=1
    )
    assert contents == ["z", "x", "y"]
    assert label == ["x"]


def test_shift_options_invalid_label_raises():
    """Raises ValueError when label is not a list or tuple."""
    with pytest.raises(ValueError):
        shift_options(["A"], ["x"], "x", run_index=1)


# ─────────────────────────────────────────────────────────────
# split_iterator — distributed data splitting (returns all rank slices at once)
# ─────────────────────────────────────────────────────────────

def test_split_iterator_no_split():
    """The entire data is a single slice when world_size==1."""
    slices, sizes = split_iterator(range(10), total_size=10, world_size=1)
    assert sizes == [10]
    assert list(slices[0]) == list(range(10))


def test_split_iterator_even():
    """Each rank receives a contiguous block of equal size when evenly divisible."""
    slices, sizes = split_iterator(range(10), total_size=10, world_size=2)
    assert sizes == [5, 5]
    assert list(slices[0]) == [0, 1, 2, 3, 4]
    assert list(slices[1]) == [5, 6, 7, 8, 9]


def test_split_iterator_uneven():
    """Remainder is distributed +1 from the first rank; sum is preserved as total_size; excess ranks get 0."""
    slices, sizes = split_iterator(range(2), total_size=2, world_size=3)
    assert sizes == [1, 1, 0]
    assert sum(sizes) == 2
    assert [list(s) for s in slices] == [[0], [1], []]


def test_split_iterator_invalid_raises():
    """Raises ValueError when world_size<=0 or total_size<0."""
    with pytest.raises(ValueError):
        split_iterator(range(5), total_size=5, world_size=0)
    with pytest.raises(ValueError):
        split_iterator(range(5), total_size=-1, world_size=2)
