"""Storage-identity + capability-driven host-input-metadata witness (r31).

Round-30 (both labs) proved the r29 per-accessor input-metadata witness was STRUCTURALLY OPEN:

* Hole A (storage-identity attribution): ``x.data.storage_offset()`` /
  ``x.detach().storage_offset()`` / ``x.data.untyped_storage().nbytes()`` read the already
  enumerated layout/geometry facts through a ``.data`` / ``.detach()`` alias that SHARES the
  input leaf's storage but is neither the leaf object nor a ``_base``-linked view. The r29
  object-identity map recorded nothing -> false VERIFIED. r31 attributes such reads by the
  leaf's BASE-storage identity + geometry: an EQUIVALENT alias (same geometry) records the leaf
  fact, a DIFFERENT-geometry storage alias (a derived view) fails closed.
* Hole B (incomplete accessor table): the fixed r29 sets omitted many host-value accessors that
  steer control flow and are not shape+dtype-derived -- ``is_conj`` / ``is_neg`` /
  ``is_inference`` / ``is_shared`` / ``is_pinned`` / ``is_coalesced`` / ``_is_view`` /
  ``retains_grad`` / ``_base``. r31 replaces the ad-hoc sets with a capability-driven table.
* Hole C (autograd-on-view): r29 restricted the derived-view fail-closed to the LAYOUT family.
  ``retains_grad`` / ``is_leaf`` / ``_base`` / ``_is_view`` on an input-derived view dodged. r31
  extends the fail-closed to those four (verified: TorchLens never reads THEM internally on an
  input view). ``requires_grad`` / ``grad_fn`` stay LEAF-ONLY because TorchLens's own per-op
  bookkeeping reads them on input-derived views (a view attribution would over-trigger a normal
  model) -- a documented, empirically-grounded residual.

The over-trigger guard is equally load-bearing: a model that never reads these accessors stays
VERIFIED even for exotic inputs, and a ``requires_grad``-oblivious model must NOT diverge on a
``requires_grad``-changed input despite TorchLens's internal autograd reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness


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


# --- Hole A: storage-identity attribution of ``.data`` / ``.detach()`` aliases --------------


class DataStorageOffsetBranch(nn.Module):
    """Branch on ``x.data.storage_offset()`` -- a storage-alias the id map missed (r31 A)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add or subtract by the alias storage offset."""

        return x + 100 if x.data.storage_offset() == 0 else x - 100


class DetachStorageOffsetBranch(nn.Module):
    """Branch on ``x.detach().storage_offset()`` -- another storage-alias (r31 A)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add or subtract by the detached-alias storage offset."""

        return x + 100 if x.detach().storage_offset() == 0 else x - 100


class DataStorageBytesBranch(nn.Module):
    """Branch on ``x.data.untyped_storage().nbytes()`` -- storage geometry via a ``.data`` alias."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently when the aliased storage is exactly ``numel * element_size``."""

        exact = x.data.untyped_storage().nbytes() == x.numel() * x.element_size()
        return x + 3 if exact else x - 3


@pytest.mark.smoke
def test_r31_data_alias_storage_offset_diverges(tmp_path: Path) -> None:
    """A ``.data`` storage-offset branch must diverge on a differently-offset twin (was VERIFIED)."""

    base = torch.randn(8)
    x0 = base[0:4]  # storage_offset 0
    xn = base[2:6]  # storage_offset 2, identical shape/dtype
    path = _save(DataStorageOffsetBranch(), x0, tmp_path / "a1.tlspec")

    assert tl.load(path).run(inputs=base.clone()[0:4]).report.path_faithfulness is (
        PathFaithfulness.VERIFIED
    )
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=xn)


@pytest.mark.smoke
def test_r31_detach_alias_storage_offset_diverges(tmp_path: Path) -> None:
    """A ``.detach()`` storage-offset branch must diverge on a differently-offset twin."""

    base = torch.randn(8)
    x0 = base[0:4]
    xn = base[2:6]
    path = _save(DetachStorageOffsetBranch(), x0, tmp_path / "a2.tlspec")

    assert tl.load(path).run(inputs=base.clone()[0:4]).report.path_faithfulness is (
        PathFaithfulness.VERIFIED
    )
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=xn)


@pytest.mark.smoke
def test_r31_data_alias_storage_bytes_diverges(tmp_path: Path) -> None:
    """A ``.data`` storage-geometry branch must diverge on a slice-of-larger-buffer twin."""

    base = torch.randn(16)
    x_full = base[0:4].clone()  # own storage, exactly numel * element_size
    x_view = base[0:4]  # base storage is larger
    path = _save(DataStorageBytesBranch(), x_full, tmp_path / "a3.tlspec")

    assert tl.load(path).run(inputs=x_full.clone()).report.path_faithfulness is (
        PathFaithfulness.VERIFIED
    )
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=x_view)


class DataContiguityBranch(nn.Module):
    """Branch on ``x.data.is_contiguous()`` -- a LAYOUT read through an equivalence alias (r31 A)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a contiguous input than a non-contiguous one."""

        return x + 1 if x.data.is_contiguous() else x - 1


@pytest.mark.smoke
def test_r31_data_alias_contiguity_diverges(tmp_path: Path) -> None:
    """A ``.data`` contiguity branch verifies on the original and diverges on a transposed twin."""

    x = torch.randn(3, 4)  # contiguous
    path = _save(DataContiguityBranch(), x, tmp_path / "a4.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    noncontig = torch.randn(4, 3).t()  # same shape (3, 4), non-contiguous
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=noncontig)


# --- Hole B: capability-driven accessor surface ---------------------------------------------


class ConjBranch(nn.Module):
    """Branch on ``x.is_conj()`` -- the conjugate dispatch bit, not shape+dtype pinned (r31 B)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Real-part scale differently for a conj-view input."""

        return (x + 1).real if not x.is_conj() else (x - 1).real


@pytest.mark.smoke
def test_r31_is_conj_diverges(tmp_path: Path) -> None:
    """A conjugated same-shape/same-dtype input on an ``is_conj`` model must diverge."""

    x = torch.randn(4, dtype=torch.cfloat)
    path = _save(ConjBranch(), x, tmp_path / "b_conj.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=x.conj())


class NegBranch(nn.Module):
    """Branch on ``x.is_neg()`` -- the negative dispatch bit set by ``torch._neg_view`` (r31 B)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a neg-view input."""

        return x + 1 if not x.is_neg() else x - 1


@pytest.mark.smoke
def test_r31_is_neg_diverges(tmp_path: Path) -> None:
    """A ``torch._neg_view`` same-shape input on an ``is_neg`` model must diverge."""

    x = torch.randn(4)
    path = _save(NegBranch(), x, tmp_path / "b_neg.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=torch._neg_view(torch.randn(4)))


class InferenceBranch(nn.Module):
    """Branch on ``x.is_inference()`` -- inference-mode status, not shape+dtype pinned (r31 B)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for an inference-mode input."""

        return x + 1 if not x.is_inference() else x - 1


@pytest.mark.smoke
def test_r31_is_inference_diverges(tmp_path: Path) -> None:
    """An inference-mode runtime input on an ``is_inference`` model must diverge."""

    x = torch.randn(4)
    path = _save(InferenceBranch(), x, tmp_path / "b_inf.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    with torch.inference_mode():
        inference_input = torch.randn(4)
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=inference_input)


class IsSharedBranch(nn.Module):
    """Branch on ``x.is_shared()`` -- shared-memory storage placement, not pinned (r31 B)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a shared-memory input."""

        return x + 1 if not x.is_shared() else x - 1


@pytest.mark.smoke
def test_r31_is_shared_diverges(tmp_path: Path) -> None:
    """A shared-memory runtime input on an ``is_shared`` model must diverge."""

    x = torch.randn(4)
    path = _save(IsSharedBranch(), x, tmp_path / "b_shared.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    shared = torch.randn(4)
    shared.share_memory_()
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=shared)


class BasePresenceBranch(nn.Module):
    """Branch on ``x._base is None`` -- whether the input is itself a view (r31 B)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a view input."""

        return x + 1 if x._base is None else x - 1


@pytest.mark.smoke
def test_r31_base_presence_diverges(tmp_path: Path) -> None:
    """A view runtime input on a ``_base``-presence model must diverge."""

    x = torch.randn(4)  # own storage, _base is None
    path = _save(BasePresenceBranch(), x, tmp_path / "b_base.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    view_input = torch.randn(8)[0:4]  # _base is the parent
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=view_input)


class RetainsGradLeafBranch(nn.Module):
    """Branch on ``x.retains_grad`` read on the input LEAF (r31 B)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale differently for a retains_grad input."""

        return x + 1 if not x.retains_grad else x - 1


@pytest.mark.smoke
def test_r31_retains_grad_leaf_diverges(tmp_path: Path) -> None:
    """A runtime input with ``retains_grad`` set on a retains_grad model must diverge."""

    x = torch.randn(4)
    path = _save(RetainsGradLeafBranch(), x, tmp_path / "b_retains.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    retains = torch.randn(4, requires_grad=True) * 1.0  # non-leaf
    retains.retain_grad()  # retains_grad True only for a non-leaf
    assert bool(retains.retains_grad) is True
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=retains)


# --- Hole C: autograd/structural reads on a DERIVED VIEW fail closed -------------------------


class RetainsGradViewBranch(nn.Module):
    """Read ``retains_grad`` on a non-leaf VIEW of the input (r31 C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the reshape view's retains_grad flag."""

        return x + 1 if not x.view(-1).retains_grad else x - 1


class IsLeafViewBranch(nn.Module):
    """Read ``is_leaf`` on a VIEW of the input (r31 C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the reshape view's is_leaf flag."""

        return x + 1 if x.view(-1).is_leaf else x - 1


class IsViewViewBranch(nn.Module):
    """Read ``_is_view`` on a VIEW of the input (r31 C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on whether the reshape view is a view."""

        return x + 1 if not x.view(-1)._is_view() else x - 1


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_cls",
    [RetainsGradViewBranch, IsLeafViewBranch, IsViewViewBranch],
)
def test_r31_autograd_structural_view_read_is_unverifiable(
    tmp_path: Path, model_cls: type[nn.Module]
) -> None:
    """An autograd/structural read on an input-derived VIEW fails closed to UNVERIFIABLE (r31 C)."""

    x = torch.randn(4)
    path = _save(model_cls(), x, tmp_path / "c_view.tlspec")

    report = tl.load(path).run(inputs=x.clone()).report
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


# --- Over-trigger guards (the two-sided balance) --------------------------------------------


@pytest.mark.smoke
def test_r31_requires_grad_oblivious_model_does_not_over_trigger(tmp_path: Path) -> None:
    """A model that never reads autograd flags must NOT diverge on a requires_grad-changed input.

    TorchLens's own per-op bookkeeping reads ``grad_fn`` / ``requires_grad`` on input-derived
    views during capture; keeping those two accessors LEAF-ONLY (never view-attributed) is what
    prevents this normal model from spuriously diverging.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    x = torch.randn(2, 4)  # requires_grad False
    path = _save(model, x, tmp_path / "og_grad.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    grad_input = torch.randn(2, 4, requires_grad=True)
    twin = tl.load(path).run(inputs=grad_input)
    assert twin.report.path_faithfulness is PathFaithfulness.VERIFIED


class DisjointStorageViews(nn.Module):
    """Split the input into two disjoint sub-views -- a legitimate view-heavy model that VERIFIES."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum two disjoint column slices of the input."""

        left = x[:, :2]
        right = x[:, 2:]
        return left.sum() + right.sum()


@pytest.mark.smoke
def test_r31_disjoint_storage_view_model_verifies(tmp_path: Path) -> None:
    """A view-heavy model with no user metadata reads stays VERIFIED (no internal over-trigger)."""

    x = torch.randn(2, 4)
    path = _save(DisjointStorageViews(), x, tmp_path / "og_disjoint.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_r31_metadata_oblivious_model_verifies_exotic_input(tmp_path: Path) -> None:
    """A plain model stays VERIFIED for a slice-of-larger-buffer input (no witnessed reads)."""

    model = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    path = _save(model, x, tmp_path / "og_plain.tlspec")

    assert tl.load(path).run(inputs=x.clone()).report.path_faithfulness is PathFaithfulness.VERIFIED
    base = torch.randn(4, 4)
    exotic = base[1:3]  # storage_offset != 0, larger base storage, same shape
    assert tl.load(path).run(inputs=exotic).report.path_faithfulness is PathFaithfulness.VERIFIED
