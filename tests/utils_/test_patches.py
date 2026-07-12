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

"""Unit tests for utils/patches.py — apply/restore of temporary patch context managers."""
import os

import pytest

from omni_evaluator.utils.patches import (
    ClassPatcher,
    inject_partial_local,
    is_function,
    is_variable,
    patch_envs,
    patch_function,
    patch_instance_method,
    patch_module,
    update_package_resources,
)


class _Greeter:
    """Dummy class for ClassPatcher / patch_instance_method tests."""

    def greet(self):
        return "orig"


# ─────────────────────────────────────────────────────────────
# ClassPatcher — batch patch class methods + undo
# ─────────────────────────────────────────────────────────────

def test_class_patcher():
    """Batch patches class methods and restores them on undo/context exit."""
    patcher = ClassPatcher(_Greeter)
    with patcher.patch_temporarily("greet", lambda self: "patched"):
        assert _Greeter().greet() == "patched"
    assert _Greeter().greet() == "orig"


# ─────────────────────────────────────────────────────────────
# inject_partial_local — temporarily replace name binding in caller scope
# ─────────────────────────────────────────────────────────────

def test_inject_partial_local():
    """Temporarily replaces a name binding in the given scope and restores it on exit."""
    scope = {"x": 1}
    with inject_partial_local("x", 99, scope):
        assert scope["x"] == 99
    assert scope["x"] == 1


# ─────────────────────────────────────────────────────────────
# is_function — distinguish functions/methods/builtins
# ─────────────────────────────────────────────────────────────

def test_is_function():
    """Functions/methods/builtins return True, plain values return False."""
    assert is_function(len) is True
    assert is_function(lambda: None) is True
    assert is_function(5) is False


# ─────────────────────────────────────────────────────────────
# is_variable — identify values that are not callable/module/class
# ─────────────────────────────────────────────────────────────

def test_is_variable():
    """Only values that are not callable/module/class are treated as variables."""
    assert is_variable(5) is True
    assert is_variable(len) is False
    assert is_variable(os) is False
    assert is_variable(_Greeter) is False


# ─────────────────────────────────────────────────────────────
# patch_envs — temporarily set env vars / restore backup on exit
# ─────────────────────────────────────────────────────────────

def test_patch_envs():
    """Backs up and restores existing env vars; newly added env vars are removed on exit."""
    os.environ["UTILS_TEST_EXIST"] = "old"
    try:
        with patch_envs({"UTILS_TEST_EXIST": "new", "UTILS_TEST_NEW": "v"}):
            assert os.environ["UTILS_TEST_EXIST"] == "new"
            assert os.environ["UTILS_TEST_NEW"] == "v"
        assert os.environ["UTILS_TEST_EXIST"] == "old"
        assert "UTILS_TEST_NEW" not in os.environ
    finally:
        os.environ.pop("UTILS_TEST_EXIST", None)


def test_patch_envs_restores_on_exception():
    """Env vars are restored even when an exception is raised inside the with block."""
    with pytest.raises(RuntimeError):
        with patch_envs({"UTILS_TEST_NEW": "v"}):
            raise RuntimeError("boom")
    assert "UTILS_TEST_NEW" not in os.environ


# ─────────────────────────────────────────────────────────────
# patch_function — temporarily replace a module function attribute
# ─────────────────────────────────────────────────────────────

def test_patch_function(fake_module_factory):
    """Replaces a module function attribute and restores the original function on exit."""
    pkg = fake_module_factory("fakepkg3", fn=lambda: "orig")
    with patch_function("fakepkg3", "fn", lambda: "patched"):
        assert pkg.fn() == "patched"
    assert pkg.fn() == "orig"


# ─────────────────────────────────────────────────────────────
# patch_instance_method — temporarily replace the method of a single instance
# ─────────────────────────────────────────────────────────────

def test_patch_instance_method():
    """Replaces the method only on the target instance; other instances are unaffected and the method is restored on exit."""
    target, other = _Greeter(), _Greeter()
    with patch_instance_method(target, "greet", lambda self: "patched"):
        assert target.greet() == "patched"
        assert other.greet() == "orig"
    assert target.greet() == "orig"


# ─────────────────────────────────────────────────────────────
# patch_module — temporarily replace submodule attribute / top-level module
# ─────────────────────────────────────────────────────────────

def test_patch_module_submodule(fake_module_factory):
    """Replaces a submodule attribute with a fake and restores the original on with exit."""
    pkg = fake_module_factory("fakepkg", sub="original")
    with patch_module("fakepkg.sub", "patched"):
        assert pkg.sub == "patched"
    assert pkg.sub == "original"


def test_patch_module_toplevel(fake_module_factory):
    """Replaces a top-level module in sys.modules and restores it on exit."""
    import sys

    original = fake_module_factory("fakepkg2")
    replacement = object()
    with patch_module("fakepkg2", replacement):
        assert sys.modules["fakepkg2"] is replacement
    assert sys.modules["fakepkg2"] is original


# ─────────────────────────────────────────────────────────────
# update_package_resources — copy source directory → installed package path
# ─────────────────────────────────────────────────────────────

def test_update_package_resources_missing_package(monkeypatch):
    """Raises ImportError when importlib.util.find_spec returns None."""
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ImportError):
        update_package_resources("no_such_pkg", "/tmp")


def test_update_package_resources_empty_source(monkeypatch, tmp_path):
    """Returns 0 without copying when source_dirpath is missing or empty."""
    import importlib.util
    import types as _types

    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: _types.SimpleNamespace(submodule_search_locations=[str(pkg_dir)]),
    )
    assert update_package_resources("fake_pkg", str(tmp_path / "no_source")) == 0


def test_update_package_resources_copies(monkeypatch, tmp_path):
    """Copies only subdirectories — skips example_ prefix and plain files, returns count."""
    import importlib.util
    import types as _types

    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "task_a").mkdir()
    (src_dir / "example_skip").mkdir()
    (src_dir / "task_b").mkdir()
    (src_dir / "notdir.txt").write_text("file")

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: _types.SimpleNamespace(submodule_search_locations=[str(pkg_dir)]),
    )
    assert update_package_resources("fake_pkg", str(src_dir)) == 2
    assert (pkg_dir / "task_a").is_dir()
    assert (pkg_dir / "task_b").is_dir()
    assert not (pkg_dir / "example_skip").exists()
