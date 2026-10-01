"""Regression test for the R0_TRANSFORMERS_RUNTIME collection gate.

L8 floor fix: ``tests/real_model/r0/conftest.py``'s ``pytest_collection_modifyitems``
originally matched items only by PATH (``str(item.path).startswith(this_dir)``), so a
sibling file elsewhere in ``tests/`` that imports ``tests.real_model.r0.families`` (and
transitively ``transformers``) at runtime -- e.g. ``tests/test_sem_resid_a02.py`` -- was
never marked skip and crashed with ``ModuleNotFoundError`` on a floor row with no
``transformers`` installed. This module lives outside ``tests/real_model/r0/`` and
carries no ``real_model`` marker itself, so the gate never skips IT (it must run on
every row, floor included, to prove the gate's own behavior).
"""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.smoke

r0_conftest = importlib.import_module("tests.real_model.r0.conftest")


class _FakeItem:
    """Minimal duck-typed stand-in for a ``pytest.Item`` the hook touches."""

    def __init__(self, path: str, marker: object | None = None) -> None:
        self.path = path
        self._marker = marker
        self.markers: list[object] = []

    def get_closest_marker(self, name: str) -> object | None:
        return self._marker if name == "real_model" else None

    def add_marker(self, marker: object) -> None:
        self.markers.append(marker)


def _run_gate(
    monkeypatch: pytest.MonkeyPatch, *, transformers_present: bool, items: list[_FakeItem]
) -> None:
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str) -> object | None:
        if name in ("transformers", "torchvision"):
            return object() if transformers_present else None
        return real_find_spec(name)

    monkeypatch.setattr(r0_conftest.importlib.util, "find_spec", _fake_find_spec)
    r0_conftest.pytest_collection_modifyitems(config=SimpleNamespace(), items=items)  # type: ignore[arg-type]


def test_real_model_marked_item_outside_r0_dir_is_skipped_when_transformers_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sibling file's ``real_model``-marked item is caught by the marker, not just path."""

    sibling_item = _FakeItem("/repo/tests/test_sem_resid_a02.py", marker=object())
    _run_gate(monkeypatch, transformers_present=False, items=[sibling_item])
    assert sibling_item.markers, "real_model-marked item outside r0/ must be skip-marked"


def test_unmarked_item_outside_r0_dir_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """An item with neither the r0 path nor the real_model marker is never touched."""

    unrelated_item = _FakeItem("/repo/tests/test_something_unrelated.py", marker=None)
    _run_gate(monkeypatch, transformers_present=False, items=[unrelated_item])
    assert not unrelated_item.markers


def test_item_under_r0_dir_is_still_skipped_by_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The original path-based match keeps working for items directly under r0/."""

    r0_dir_item = _FakeItem(f"{r0_conftest.os.path.dirname(r0_conftest.__file__)}/test_x.py")
    _run_gate(monkeypatch, transformers_present=False, items=[r0_dir_item])
    assert r0_dir_item.markers


def test_nothing_is_skipped_when_transformers_and_torchvision_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A row claiming the sweep (both deps importable) marks nothing skip."""

    item = _FakeItem("/repo/tests/test_sem_resid_a02.py", marker=object())
    _run_gate(monkeypatch, transformers_present=True, items=[item])
    assert not item.markers


def test_require_r0_env_fails_closed_instead_of_skipping(monkeypatch: pytest.MonkeyPatch) -> None:
    """TORCHLENS_REQUIRE_R0=1 on a broken install raises, never silently skips."""

    monkeypatch.setenv("TORCHLENS_REQUIRE_R0", "1")
    item = _FakeItem("/repo/tests/test_sem_resid_a02.py", marker=object())
    with pytest.raises(pytest.UsageError):
        _run_gate(monkeypatch, transformers_present=False, items=[item])
