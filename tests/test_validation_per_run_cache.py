"""Validation's per-run scratch never survives the run into saved state.

The orphan-argument sweep caches a candidate index on the trace while one
validation runs. Left in ``Trace.__dict__`` after a failed run, it reached
``save`` as live state with no portability policy and the save refused
(``_validation_orphan_candidate_index is missing from _PORTABLE_STATE_SPEC``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._state import pause_logging
from torchlens.options import CaptureOptions
from torchlens.validation import core as validation_core

_INDEX_KEY = "_validation_orphan_candidate_index"


class _Smuggles(nn.Module):
    """Feeds ``relu`` a tensor built mid-forward with logging paused (no provenance)."""

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        with pause_logging():
            hidden = v * 3.0
        return torch.relu(v + hidden)


def _failed_validation_trace(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, nn.Module, bool]:
    torch.manual_seed(0)
    model = _Smuggles()
    x = torch.randn(2, 4)
    reference = model(x)
    swept: list[bool] = []
    original = validation_core._orphan_candidate_index

    def _spy(trace: Any) -> Any:
        swept.append(True)
        return original(trace)

    monkeypatch.setattr(validation_core, "_orphan_candidate_index", _spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(model, x, capture=CaptureOptions(save_args=True, save_rng=True))
        passed = bool(trace.validate_forward_pass(reference))
    assert swept, "the model must reach the orphan-argument sweep"
    return trace, model, passed


# Kept by shrink-r4 (per-id subset rule): sole coverage of _io/scrub.py, validation/_trace_checks.py.
def test_failed_validation_leaves_no_scratch_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace, _model, passed = _failed_validation_trace(monkeypatch)
    assert passed is False
    assert _INDEX_KEY not in trace.__dict__
    trace.save(tmp_path / "trace.tlz")
    assert (tmp_path / "trace.tlz").exists()
