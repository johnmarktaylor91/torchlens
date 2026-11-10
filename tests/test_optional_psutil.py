"""psutil is optional: a host without it must not search the import path on every capture."""

from __future__ import annotations

import sys
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens import _optional_psutil


class _PsutilLookups:
    """A ``sys.meta_path`` finder that records each search for ``psutil`` and finds nothing."""

    def __init__(self) -> None:
        self.count = 0

    def find_spec(self, name: str, path: Any = None, target: Any = None) -> None:
        if name == "psutil":
            self.count += 1
        return None


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> _PsutilLookups:
    finder = _PsutilLookups()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    monkeypatch.setattr(_optional_psutil, "_ABSENT", [])
    return finder


def test_psutil_is_searched_for_at_most_once(lookups: _PsutilLookups) -> None:
    first = _optional_psutil.optional_psutil()
    for _ in range(3):
        assert _optional_psutil.optional_psutil() is first
    assert lookups.count <= 1


def test_a_warmed_capture_does_not_search_for_psutil(lookups: _PsutilLookups) -> None:
    model, x = torch.nn.ReLU(), torch.randn(2, 3)
    tl.trace(model, x)
    before = lookups.count
    tl.trace(model, x)
    assert lookups.count == before


def test_a_hidden_psutil_reads_missing_and_is_not_remembered(
    lookups: _PsutilLookups, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert _optional_psutil.optional_psutil() is None
    assert _optional_psutil._ABSENT == []
