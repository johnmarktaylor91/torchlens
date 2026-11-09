"""The capture size guard: ``capture_memory_threshold_exceeded`` warns once per session.

A capture stays complete by default; past the saved-activation threshold it warns
once, naming the cheaper ``save=`` spelling, and later captures stay quiet.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.finalize import epilogue
from torchlens.errors import TorchLensWarning

CODE = "capture_memory_threshold_exceeded"


def _codes(caught: list[warnings.WarningMessage]) -> list[object]:
    return [
        item.message.fields.get("code")
        for item in caught
        if isinstance(item.message, TorchLensWarning)
    ]


def _capture(save: object = "all", **capture: object) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        options = tl.options.CaptureOptions(**capture) if capture else None
        tl.trace(nn.Linear(4, 4), torch.randn(2, 4), save=save, capture=options)
    return caught


def test_guard_warns_once_past_the_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the threshold the first capture warns with the code and remedy; the next is quiet."""

    monkeypatch.setattr(epilogue, "_CAPTURE_MEMORY_THRESHOLD", 0)
    first = _capture()
    assert _codes(first).count(CODE) == 1
    message = next(item.message for item in first if _codes([item]) == [CODE])
    assert "save=" in str(message)
    assert message.fields["remedy"]
    assert _codes(_capture()).count(CODE) == 0


def test_guard_is_quiet_under_the_threshold() -> None:
    """A small capture under the default threshold never warns."""

    assert CODE not in _codes(_capture())


def test_guard_is_quiet_when_nothing_is_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture that saved no activations keeps no bytes, so the guard stays quiet."""

    monkeypatch.setattr(epilogue, "_CAPTURE_MEMORY_THRESHOLD", 0)
    assert CODE not in _codes(_capture(save="none"))


def test_quiet_silences_the_guard_by_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tl.errors.quiet(code)`` hides the guard's warning."""

    monkeypatch.setattr(epilogue, "_CAPTURE_MEMORY_THRESHOLD", 0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with tl.errors.quiet(CODE):
            tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
    assert CODE not in _codes(caught)


@pytest.mark.parametrize("save_mode", ["view", "copy"])
def test_guard_is_quiet_when_the_budget_is_switched_off(
    monkeypatch: pytest.MonkeyPatch, save_mode: str
) -> None:
    """``save_budget=None`` turns budgeting off on purpose: no warning, and the once is not spent."""

    monkeypatch.setattr(epilogue, "_CAPTURE_MEMORY_THRESHOLD", 0)
    monkeypatch.setattr(epilogue, "_CAPTURE_MEMORY_WARNED", False)
    assert CODE not in _codes(_capture(save_mode=save_mode, save_budget=None))
    assert _codes(_capture(save_mode=save_mode)).count(CODE) == 1
