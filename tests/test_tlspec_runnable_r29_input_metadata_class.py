"""Closed host-input-metadata-read witness class for sparse runnable execution (r29-C1).

r27-H2 witnessed only ``is_contiguous`` / ``stride`` / ``requires_grad`` reads on a model
input. The confirming hunt found MANY more host-reading tensor accessors that steer control
flow on facts the shape+dtype input contract never pins, each a false VERIFIED+ATTESTED:

* ``storage_offset()`` (F1) -- a slice of a larger buffer has a non-zero offset a same-shape
  twin lacks; capture clones the leaf (offset -> 0), so it is recorded from the RAW input.
* ``grad_fn`` / ``is_leaf`` (F3) -- autograd-graph presence the executor's detach-clone erases.
* ``untyped_storage().nbytes()`` / ``.size()`` (F4) -- storage geometry that differs for a
  view of a larger base.
* a LAYOUT read on a DERIVED VIEW of an input (``x.t().is_contiguous()``, F5) -- the view is an
  orphan-pruned intermediate the replay never re-derives, so it fails closed to UNVERIFIABLE.

The over-trigger guard is equally load-bearing: a model that never reads these accessors, or
reads only the autograd family on internal/view tensors during ordinary escape bookkeeping,
must stay VERIFIED+ATTESTED on its original input.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import DivergencePolicy, NumericAttestationStatus, PathFaithfulness


def _save(model: nn.Module, x: torch.Tensor, path: Path) -> Path:
    """Capture and save a runnable artifact carrying activation attestation."""

    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return path


class StorageOffsetBranch(nn.Module):
    """Branch on the input's ``storage_offset`` -- non-zero only for a slice of a larger buffer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add or subtract by the input's storage offset."""

        return x + 100 if x.storage_offset() == 0 else x - 100


class IsLeafBranch(nn.Module):
    """Branch on the input's ``is_leaf`` autograd flag -- erased by the executor's detach-clone."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a leaf input than a non-leaf one."""

        return x + 5 if x.is_leaf else x - 5


class GradFnBranch(nn.Module):
    """Branch on the input's autograd-graph presence (``grad_fn is None``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a grad-graph input than a plain one."""

        return x + 7 if x.grad_fn is None else x - 7


class StorageBytesBranch(nn.Module):
    """Branch on the input's raw storage byte GEOMETRY (base storage size)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently when the storage is exactly ``numel * element_size``."""

        exact = x.untyped_storage().nbytes() == x.numel() * x.element_size()
        return x + 3 if exact else x - 3


class ViewContiguityBranch(nn.Module):
    """Read a LAYOUT predicate on a DERIVED VIEW of the input (``x.t().is_contiguous()``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the transpose view's contiguity."""

        return x + 1 if x.t().is_contiguous() else x - 1


@pytest.mark.smoke
def test_r29_storage_offset_twin_diverges(tmp_path: Path) -> None:
    """A same-shape input with a different storage_offset must diverge, never false VERIFIED."""

    base = torch.randn(8)
    x0 = base[0:4]  # storage_offset 0
    xn = base[2:6]  # storage_offset 2, identical shape/dtype
    path = _save(StorageOffsetBranch(), x0, tmp_path / "off.tlspec")

    result = tl.load(path).run(inputs=base.clone()[0:4])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=xn)
    diverged = tl.load(path).run(inputs=xn, on_divergence=DivergencePolicy.RETURN_DIVERGED)
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_r29_is_leaf_flip_diverges(tmp_path: Path) -> None:
    """A non-leaf runtime input on a ``is_leaf``-branching model must diverge."""

    x = torch.randn(4)
    path = _save(IsLeafBranch(), x, tmp_path / "leaf.tlspec")
    assert tl.load(path).run(inputs=x).report.path_faithfulness is PathFaithfulness.VERIFIED

    nonleaf = torch.randn(4, requires_grad=True) * 1.0  # grad_fn set, is_leaf False
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=nonleaf)


@pytest.mark.smoke
def test_r29_grad_fn_presence_diverges(tmp_path: Path) -> None:
    """A runtime input carrying a grad_fn on a grad_fn-branching model must diverge."""

    x = torch.randn(4)
    path = _save(GradFnBranch(), x, tmp_path / "gradfn.tlspec")
    assert tl.load(path).run(inputs=x).report.path_faithfulness is PathFaithfulness.VERIFIED

    with_grad_fn = torch.randn(4, requires_grad=True) + 0.0
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=with_grad_fn)


@pytest.mark.smoke
def test_r29_storage_geometry_view_diverges(tmp_path: Path) -> None:
    """A same-shape input that is a view of a larger storage must diverge on a geometry read."""

    base = torch.randn(16)
    x_full = base[0:4].clone()  # own storage, exactly numel * element_size
    x_view = base[0:4]  # base storage is larger
    path = _save(StorageBytesBranch(), x_full, tmp_path / "sb.tlspec")
    assert tl.load(path).run(inputs=x_full.clone()).report.path_faithfulness is (
        PathFaithfulness.VERIFIED
    )

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=x_view)


@pytest.mark.smoke
def test_r29_derived_view_layout_read_is_unverifiable(tmp_path: Path) -> None:
    """A layout read on an input-derived VIEW fails closed to UNVERIFIABLE (cannot re-verify)."""

    x = torch.randn(4, 4)
    path = _save(ViewContiguityBranch(), x, tmp_path / "vc.tlspec")

    report = tl.load(path).run(inputs=x.clone()).report
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_r29_metadata_oblivious_model_does_not_over_trigger(tmp_path: Path) -> None:
    """A model that never reads the new accessors stays VERIFIED even for exotic inputs."""

    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    path = _save(model, x, tmp_path / "plain.tlspec")

    # Original input verifies (a Linear's matmul activation is only ULP-reproducible, so its
    # attestation is legitimately NOT_APPLICABLE -- the point here is no metadata divergence).
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    # A same-shape slice-of-larger-buffer input (non-zero storage_offset, larger storage)
    # is a legal changed input here: no witnessed metadata read means it must NOT diverge.
    base = torch.randn(4, 4)
    exotic = base[1:3]  # storage_offset != 0, larger base storage, same shape as x
    twin = tl.load(path).run(inputs=exotic)
    assert twin.report.path_faithfulness is PathFaithfulness.VERIFIED


class LoopAccumScalarView(nn.Module):
    """Autograd/escape reads on an input VIEW must NOT fail closed (over-trigger guard)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Accumulate ``x[i].item()`` -- reads grad_fn on the ``x[i]`` view internally."""

        total = 0.0
        for i in range(x.shape[0]):
            total = total + x[i].item()
        return x * total


@pytest.mark.smoke
def test_r29_autograd_reads_on_input_view_do_not_over_trigger(tmp_path: Path) -> None:
    """Framework autograd reads on an input-derived view must stay VERIFIED+ATTESTED."""

    x = torch.tensor([2.0, 3.0])
    path = _save(LoopAccumScalarView(), x, tmp_path / "loop.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
