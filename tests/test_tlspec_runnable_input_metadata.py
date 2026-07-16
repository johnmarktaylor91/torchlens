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
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
)


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
            intervention_ready=True,
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


@pytest.mark.smoke
def test_h2_is_contiguous_layout_twin_diverges_and_is_not_attested(tmp_path: Path) -> None:
    """A byte-identical layout twin must diverge, never false VERIFIED+ATTESTED."""

    x = torch.randn(4, 4)
    path = _save_runnable(ContiguityBranch(), x, tmp_path / "contig.tlspec")

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=_layout_twin(x))

    diverged = tl.load(path).run(
        inputs=_layout_twin(x), on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert diverged.report.poisoned
    # A fresh model on the twin takes the OTHER arm; the replayed arm is wrong.
    fresh = ContiguityBranch()(_layout_twin(x))
    assert torch.equal(fresh, x - 100)


@pytest.mark.smoke
def test_h2_is_contiguous_original_input_still_verifies_and_attests(tmp_path: Path) -> None:
    """The original contiguous input must keep VERIFIED + ATTESTED (no over-trigger)."""

    x = torch.randn(4, 4)
    path = _save_runnable(ContiguityBranch(), x, tmp_path / "contig.tlspec")

    result = tl.load(path).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert all(check.passed for check in result.report.contract_checks)
    torch.testing.assert_close(result.output, x + 100)


def test_h2_stride_read_diverges_on_layout_change(tmp_path: Path) -> None:
    """A ``stride()`` read is witnessed: a same-shape stride change must diverge."""

    x = torch.randn(4, 4)
    path = _save_runnable(StrideBranch(), x, tmp_path / "stride.tlspec")

    result = tl.load(path).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=_layout_twin(x))
    diverged = tl.load(path).run(
        inputs=_layout_twin(x), on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_h2_requires_grad_flip_diverges_and_is_not_attested(tmp_path: Path) -> None:
    """A runtime ``requires_grad`` flip on a grad-branching model must diverge."""

    x = torch.randn(3)
    path = _save_runnable(RequiresGradBranch(), x, tmp_path / "grad.tlspec")

    result = tl.load(path).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    grad_input = x.clone().requires_grad_(True)
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=grad_input)
    diverged = tl.load(path).run(inputs=grad_input, on_divergence=DivergencePolicy.RETURN_DIVERGED)
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert diverged.report.poisoned
    # A fresh model on the grad input takes the OTHER arm; the replayed arm is wrong.
    fresh = RequiresGradBranch()(x.clone().requires_grad_(True))
    torch.testing.assert_close(fresh.detach(), x * 2)


@pytest.mark.smoke
def test_h2_layout_oblivious_model_does_not_over_trigger(tmp_path: Path) -> None:
    """A model that never reads input metadata records no facts and never diverges on them."""

    x = torch.randn(4, 4)
    trace = tl.trace(
        LayoutOblivious(),
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    # The recording is read-gated: no metadata-predicate read, no recorded fact.
    assert trace.__dict__.get("_runnable_input_metadata_reads") is None
    trace.save(tmp_path / "plain.tlspec", level="runnable", include_activations=True)

    result = tl.load(tmp_path / "plain.tlspec").run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    # Layout twin and requires_grad flip are legal changed inputs here: no witnessed
    # metadata read means neither may trip a divergence.
    twin_result = tl.load(tmp_path / "plain.tlspec").run(inputs=_layout_twin(x))
    assert twin_result.report.path_faithfulness is PathFaithfulness.VERIFIED
    grad_result = tl.load(tmp_path / "plain.tlspec").run(inputs=x.clone().requires_grad_(True))
    assert grad_result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_h2_scoped_patch_is_restored_after_capture() -> None:
    """The metadata observers must not leak past the capture window."""

    x = torch.randn(4, 4)
    options = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)
    # First capture forces TorchLens's own (permanent) torch wrapping, so the
    # snapshot below reflects the steady state the scoped patch must restore.
    tl.trace(ContiguityBranch(), x, capture=options)
    before_is_contiguous = torch.Tensor.is_contiguous
    before_stride = torch.Tensor.stride
    had_grad_shadow = "requires_grad" in torch.Tensor.__dict__

    tl.trace(ContiguityBranch(), x, capture=options)

    assert torch.Tensor.is_contiguous is before_is_contiguous
    assert torch.Tensor.stride is before_stride
    # ``requires_grad`` natively lives on the C base class; a leftover shadowing
    # property on ``torch.Tensor`` itself would mean the restore failed.
    assert ("requires_grad" in torch.Tensor.__dict__) == had_grad_shadow
    # And the unpatched surfaces behave normally outside a capture.
    assert x.is_contiguous() is True
    assert tuple(x.stride()) == (4, 1)
    assert x.requires_grad is False
