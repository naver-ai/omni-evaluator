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

"""Unit tests for utils/io.py — multi-format round-trip + path helpers."""
import numpy as np
import pytest

from omni_evaluator.utils.io import (
    ensure_per_run_format,
    get_output_dirpath,
    get_output_filename,
    get_temp_filepath,
    is_sub_path,
    read_file,
    write_file,
)


# ─────────────────────────────────────────────────────────────
# ensure_per_run_format — legacy run_outputs flattening / per-run passthrough
# ─────────────────────────────────────────────────────────────

def test_ensure_per_run_format_passthrough():
    """Already in List[List[Dict]] form (first element is a list) — returns as-is."""
    data = [[{"id": 0}], [{"id": 0}]]
    assert ensure_per_run_format(data) is data


def test_ensure_per_run_format_legacy():
    """Expands flat records with run_outputs into a per-run list."""
    legacy = [{"id": 0, "run_outputs": [{"prediction": "a"}, {"prediction": "b"}]}]
    out = ensure_per_run_format(legacy)
    assert len(out) == 2
    assert out[0][0]["prediction"] == "a"
    assert out[1][0]["prediction"] == "b"


# ─────────────────────────────────────────────────────────────
# get_output_dirpath — output directory path assembly (version_name branching)
# ─────────────────────────────────────────────────────────────

def test_get_output_dirpath():
    """Path includes the version or 'checkpoint-none' depending on whether version_name is provided."""
    assert get_output_dirpath("/out", "builtin", "exp", "v1").endswith(
        "/out/exp/v1/builtin"
    )
    assert get_output_dirpath("/out", "builtin", "exp").endswith(
        "/out/exp/checkpoint-none/builtin"
    )


# ─────────────────────────────────────────────────────────────
# get_output_filename — '{benchmark}__{method}.json' filename assembly
# ─────────────────────────────────────────────────────────────

def test_get_output_filename():
    """Produces the '{benchmark}__{method}.json' form."""
    assert get_output_filename("mmlu", "generation") == "mmlu__generation.json"


# ─────────────────────────────────────────────────────────────
# get_temp_filepath — create a temporary file in the specified directory
# ─────────────────────────────────────────────────────────────

def test_get_temp_filepath(tmp_path):
    """Creates an actual temporary file in the specified directory and returns its path."""
    import os

    path = get_temp_filepath(prefix="t_", suffix=".txt", dirpath=str(tmp_path))
    assert os.path.exists(path)
    assert path.endswith(".txt")


# ─────────────────────────────────────────────────────────────
# is_sub_path — determines whether child is a subpath of parent
# ─────────────────────────────────────────────────────────────

def test_is_sub_path(tmp_path):
    """Returns True if child is under parent, False if outside."""
    parent = str(tmp_path / "a")
    assert is_sub_path(parent, str(tmp_path / "a" / "b")) is True
    assert is_sub_path(parent, str(tmp_path / "c")) is False


# ─────────────────────────────────────────────────────────────
# read_file / write_file — multi-format round-trip by extension
# ─────────────────────────────────────────────────────────────

def test_json_roundtrip(tmp_path):
    """JSON write → read preserves the original object."""
    path = str(tmp_path / "d.json")
    obj = {"a": 1, "b": [2, 3]}
    write_file(path, obj)
    assert read_file(path) == obj


def test_jsonl_roundtrip(tmp_path):
    """JSONL write → read preserves the line-by-line list."""
    path = str(tmp_path / "d.jsonl")
    obj = [{"a": 1}, {"b": 2}]
    write_file(path, obj)
    assert read_file(path) == obj


def test_jsonl_requires_list(tmp_path):
    """Writing a non-list object to JSONL raises TypeError."""
    with pytest.raises(TypeError):
        write_file(str(tmp_path / "d.jsonl"), {"a": 1})


def test_csv_roundtrip(tmp_path):
    """CSV write → read preserves the row list (values as strings)."""
    path = str(tmp_path / "d.csv")
    write_file(path, [["a", "b"], ["1", "2"]])
    assert read_file(path) == [["a", "b"], ["1", "2"]]


def test_tsv_roundtrip(tmp_path):
    """TSV (tab-delimited) write → read round-trip."""
    path = str(tmp_path / "d.tsv")
    write_file(path, [["a", "b"], ["1", "2"]])
    assert read_file(path) == [["a", "b"], ["1", "2"]]


def test_xlsx_roundtrip(tmp_path):
    """XLSX write → read preserves the rows of the first sheet."""
    path = str(tmp_path / "d.xlsx")
    write_file(path, [["h1", "h2"], [1, 2]])
    assert read_file(path) == [["h1", "h2"], [1, 2]]


def test_pickle_roundtrip(tmp_path):
    """Pickle write → read round-trip."""
    path = str(tmp_path / "d.pickle")
    obj = {"nested": {"x": [1, 2]}}
    write_file(path, obj)
    assert read_file(path) == obj


def test_yaml_roundtrip(tmp_path):
    """YAML write is unsupported, so only read is verified on an externally written file."""
    path = tmp_path / "d.yaml"
    path.write_text("a: 1\nb: [2, 3]\n", encoding="utf-8")
    assert read_file(str(path)) == {"a": 1, "b": [2, 3]}


def test_bytes_roundtrip(tmp_path):
    """A path without a recognized extension is written/read as raw bytes."""
    path = str(tmp_path / "d.bin")
    write_file(path, b"\x00\x01rawbytes")
    assert read_file(path) == b"\x00\x01rawbytes"


def test_read_missing_raises():
    """Reading a non-existent path raises ValueError (silent None is forbidden)."""
    with pytest.raises(ValueError):
        read_file("/nonexistent/path/file.json")


def test_write_unsupported_raises(tmp_path):
    """Writing a non-bytes object with an unsupported extension raises ValueError."""
    with pytest.raises(ValueError):
        write_file(str(tmp_path / "d.weird"), {"a": 1})
