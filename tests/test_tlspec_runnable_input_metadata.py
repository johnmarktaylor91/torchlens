"""Model-input METADATA-PREDICATE witness contract for sparse runnable execution (r27-H2).

These tests lock the honesty rule that a sparse runnable ``run`` may never report
``VERIFIED``/``ATTESTED`` after silently replaying the recorded taken path for a
runtime input that differs on a metadata predicate the captured forward actually
READ (``is_contiguous`` / ``stride`` / ``requires_grad``). The input contract checks
only shape+dtype, so a byte-identical LAYOUT TWIN (same values, transposed strides)
or a ``requires_grad`` flip would otherwise take a different branch on a fresh model
while the replay follows the captured arm -- the exact false-VERIFIED money bug.
Captures that never read input metadata record no facts and must be completely
unaffected (zero over-trigger).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
)
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions


class ContiguityBranch(nn.Module):
    """Branch on the model input's contiguity -- a fact only a metadata read exposes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add or subtract according to the input's memory layout."""

        if x.is_contiguous():
            return x + 100
        return x - 100


class StrideBranch(nn.Module):
    """Branch on the model input's leading stride (row-major vs transposed layout)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a row-major layout than for a transposed one."""

        if x.stride(0) == x.size(1):
            return x * 2
        return x * 3


class RequiresGradBranch(nn.Module):
    """Branch on the model input's autograd flag -- erased by the executor's detach-clone."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a grad-tracking input than for a plain one."""

        if x.requires_grad:
            return x * 2
        return x * 3


class LayoutOblivious(nn.Module):
    """A model that never reads input metadata; it must record no facts at all."""

    def __init__(self) -> None:
        """Build a single linear layer."""

        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear layer with no metadata-predicate reads."""

        return self.lin(x)


def _save_runnable(model: nn.Module, x: torch.Tensor, path: Path) -> Path:
    """Capture and save a runnable artifact carrying activation attestation."""

    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            replayable=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return path


def _layout_twin(x: torch.Tensor) -> torch.Tensor:
    """Return a byte-identical, same-shape, NON-contiguous twin of a square tensor."""

    twin = x.t().contiguous().t()
    assert torch.equal(twin, x)
    assert not twin.is_contiguous()
    return twin


# Kept: the catch of its regression fault.
def test_h2_is_contiguous_layout_twin_diverges_and_is_not_attested(tmp_path: Path) -> None:
    """A byte-identical layout twin must diverge, never false VERIFIED+ATTESTED."""

    x = torch.randn(4, 4)
    path = _save_runnable(ContiguityBranch(), x, tmp_path / "contig.tlspec")

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=_layout_twin(x))

    diverged = tl.load(path).run(inputs=_layout_twin(x), on_divergence="return")
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert diverged.report.is_poisoned
    # A fresh model on the twin takes the OTHER arm; the replayed arm is wrong.
    fresh = ContiguityBranch()(_layout_twin(x))
    assert torch.equal(fresh, x - 100)


# Kept: sole branch coverage of _runnable/input_sites.py.
def test_h2_requires_grad_flip_diverges_and_is_not_attested(tmp_path: Path) -> None:
    """A runtime ``requires_grad`` flip on a grad-branching model must diverge."""

    x = torch.randn(3)
    path = _save_runnable(RequiresGradBranch(), x, tmp_path / "grad.tlspec")

    result = tl.load(path).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    grad_input = x.clone().requires_grad_(True)
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=grad_input)
    diverged = tl.load(path).run(inputs=grad_input, on_divergence="return")
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert diverged.report.is_poisoned
    # A fresh model on the grad input takes the OTHER arm; the replayed arm is wrong.
    fresh = RequiresGradBranch()(x.clone().requires_grad_(True))
    torch.testing.assert_close(fresh.detach(), x * 2)
