"""A06 capture-options truth: the ``module_filter`` third save gate is DISCLOSED.

WALKTHROUGH list-A row 10 (second clause): ``module_filter`` was a
doc-invisible third save gate whose argument is an op-record namespace, so
the natural nn.Module-shaped filter (``lambda m: isinstance(m, nn.Linear)``)
silently saved ZERO payloads. The gate now warns coded
(``module_filter_zero_saved``) when it suppressed every selected payload, and
the docstring names the gate order and the ctx contract.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import TorchLensWarning
from torchlens.options import CaptureOptions


class ThreeStep(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _warned_codes(caught: list[warnings.WarningMessage]) -> list[str]:
    return [
        getattr(entry.message, "fields", {}).get("code")
        for entry in caught
        if isinstance(entry.message, TorchLensWarning)
    ]


def test_module_filter_zero_saved_warns_coded() -> None:
    """The natural module-shaped filter saves nothing and now DISCLOSES it."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(
            ThreeStep(),
            torch.randn(2, 4),
            capture=CaptureOptions(module_filter=lambda mod: isinstance(mod, nn.Linear)),
        )
    assert sum(1 for op in log if getattr(op, "out", None) is not None) == 0
    assert "module_filter_zero_saved" in _warned_codes(caught)


def test_module_filter_partial_suppression_stays_silent() -> None:
    """A filter that keeps SOME payloads is a working gate, not a zero-save."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(
            ThreeStep(),
            torch.randn(2, 4),
            capture=CaptureOptions(
                module_filter=lambda op: getattr(op, "func_name", "") == "linear"
            ),
        )
    saved = [op.layer_label for op in log if getattr(op, "out", None) is not None]
    assert saved, "the op-record-shaped filter should keep the linear payloads"
    assert "module_filter_zero_saved" not in _warned_codes(caught)


def test_module_filter_none_never_warns() -> None:
    """The default path (no filter) is untouched."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(ThreeStep(), torch.randn(2, 4))
    assert sum(1 for op in log if getattr(op, "out", None) is not None) > 0
    assert "module_filter_zero_saved" not in _warned_codes(caught)


def test_module_filter_empty_save_selection_not_blamed() -> None:
    """A zero-save caused by the SAVE selection never blames module_filter."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tl.trace(
            ThreeStep(),
            torch.randn(2, 4),
            capture=CaptureOptions(
                layers_to_save="none",
                module_filter=lambda op: False,
            ),
        )
    assert "module_filter_zero_saved" not in _warned_codes(caught)


def test_module_filter_keeps_metadata_when_suppressing() -> None:
    """Suppression drops payloads only; op metadata stays complete."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        log = tl.trace(
            ThreeStep(),
            torch.randn(2, 4),
            capture=CaptureOptions(module_filter=lambda op: False),
        )
    del caught
    labels = [op.layer_label for op in log]
    assert any(label.startswith("linear_1") for label in labels)
    assert any(label.startswith("relu") for label in labels)
