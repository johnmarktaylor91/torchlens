"""Round-18 PARAM host-escape parity: bring PARAMETER host-escape handling to parity with the
(already-correct) registered-BUFFER handling on BOTH axes -- untracked host WRITE and read-only
host READ.

Params were inconsistent with buffers:

* Too LOOSE on writes (F2, CRITICAL). A zero-copy numpy alias of a PARAM acquired in ``__init__``
  (before tracing, so no escape observer ever sees the exposure) and written DURING the captured
  forward (``self.npw[0] += 1``) is an invisible host mutation of the param's storage. It bumps no
  torch version and emits no aten op, and the embedded param state is a pre-forward snapshot, so the
  invisible write is invisible on both ends: the DEFAULT sparse save falsely VERIFIED while the
  replay output disagreed with both the true live forward and the capture's own output. The
  registered-buffer twin correctly reported UNVERIFIABLE. The fix mirrors the buffer host-write-back
  tripwire onto params: ``buffer_writes.BufferWriteTracker`` now snapshots each named parameter's
  whole-storage bytes + version at forward START and, at forward END
  (``_reconcile_params``), flags a version-unchanged byte change as an opaque host write-back
  (``completeness_witness._HOST_ESCAPE_MUTABLE_WRITEBACK``) -> UNVERIFIABLE.

* Too TIGHT on reads (F3, MEDIUM over-trigger). A transient READ-ONLY param stat log
  (``self.w.detach().numpy().sum()``) reported UNVERIFIABLE while the IDENTICAL buffer twin
  (``self.b.detach().numpy().sum()``) stayed VERIFIED. A buffer keeps a graph source node, so its
  ``.detach()`` host read survives orphan-pruning and witnesses by its kept op; a parameter carries
  no source node, so the param-rooted ``.detach()`` op is orphan-pruned and its escape hit the
  fail-closed ``INCOMPLETE_SCALAR_ESCAPE`` gate. The fix resolves a read-only param host escape by
  its state slot (via the forward-start storage-pointer index): the escape is then witnessed by the
  param's capture-time digest -- value-correct (unchanged param re-digests identically -> VERIFIED;
  changed staged param -> UNVERIFIABLE), exactly the honest read/write distinction the buffer path
  draws.

This module proves the CRITICAL write is now honestly refused on BOTH save modes AND the hard
no-over-trigger contract holds: read-only param AND buffer escapes stay VERIFIED (parity), plain
Linear/Conv models stay VERIFIED, and read-only ``.item()`` / ``.tolist()`` param stats stay
VERIFIED.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


def _save(model: nn.Module, capture_input: torch.Tensor, path: Path, **save_kwargs) -> Path:
    trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", **save_kwargs)
    return path


def _run(path: Path, x: torch.Tensor):
    return tl.load(path).run(inputs=x.clone(), seed=0, on_divergence="return_diverged")


# --------------------------------------------------------------------------- #
# F2 (CRITICAL): an invisible within-forward PARAM write must be UNVERIFIABLE on BOTH save modes.
# --------------------------------------------------------------------------- #
class InitAcquiredParamAliasWrite(nn.Module):
    """Zero-copy numpy alias of a PARAM acquired in ``__init__``, written during the forward."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))
        # Acquired BEFORE tracing -> no escape observer ever sees the exposure.
        self.npw: np.ndarray = self.w.detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.npw[0] += 1.0  # invisible host write into the param's storage during the forward
        return x * self.w


class InitAcquiredBufferAliasWrite(nn.Module):
    """The correct BUFFER twin: the same pattern with a registered buffer -> UNVERIFIABLE."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 3.0]))
        self.npb: np.ndarray = self.b.detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.npb[0] += 1.0
        return x * self.b


@pytest.mark.smoke
def test_invisible_param_write_default_sparse_is_unverifiable(tmp_path: Path) -> None:
    model = InitAcquiredParamAliasWrite()
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(model, capture_x, tmp_path / "param_write.tlspec", include_weights=True)

    result = _run(path, capture_x)

    # The embedded pre-forward param snapshot does not reflect the invisible write, so the sparse
    # replay recomputes a value disagreeing with the true live forward: it must be refused.
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


@pytest.mark.smoke
def test_invisible_param_write_include_activations_is_unverifiable(tmp_path: Path) -> None:
    model = InitAcquiredParamAliasWrite()
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(
        model,
        capture_x,
        tmp_path / "param_write_acts.tlspec",
        include_weights=True,
        include_activations=True,
    )

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


@pytest.mark.smoke
def test_invisible_buffer_write_twin_is_unverifiable(tmp_path: Path) -> None:
    """The buffer twin was already correct; assert parity so a regression on either is caught."""
    model = InitAcquiredBufferAliasWrite()
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(model, capture_x, tmp_path / "buffer_write.tlspec", include_weights=True)

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE


# --------------------------------------------------------------------------- #
# F3 (MEDIUM over-trigger): a transient READ-ONLY host escape must be VERIFIED for PARAM AND BUFFER.
# --------------------------------------------------------------------------- #
class ReadOnlyParamEscape(nn.Module):
    """Transient read-only weight-stat log; the param is never mutated -> must stay VERIFIED."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.w.detach().numpy().sum()  # read-only host escape
        return x * self.w


class ReadOnlyBufferEscape(nn.Module):
    """The buffer twin of the read-only escape -> already VERIFIED; asserted for parity."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.b.detach().numpy().sum()
        return x * self.b


@pytest.mark.smoke
def test_readonly_param_escape_stays_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(
        ReadOnlyParamEscape(), capture_x, tmp_path / "ro_param.tlspec", include_weights=True
    )

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_readonly_buffer_escape_stays_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(
        ReadOnlyBufferEscape(), capture_x, tmp_path / "ro_buffer.tlspec", include_weights=True
    )

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_readonly_param_and_buffer_parity(tmp_path: Path) -> None:
    """The parity claim head-on: identical read-only escapes on a param and a buffer agree."""
    capture_x = torch.tensor([2.0, 4.0])
    param_result = _run(
        _save(ReadOnlyParamEscape(), capture_x, tmp_path / "p.tlspec", include_weights=True),
        capture_x,
    )
    buffer_result = _run(
        _save(ReadOnlyBufferEscape(), capture_x, tmp_path / "b.tlspec", include_weights=True),
        capture_x,
    )
    assert param_result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert param_result.report.path_faithfulness is buffer_result.report.path_faithfulness


# --------------------------------------------------------------------------- #
# HARD NO-OVER-TRIGGER: plain models with params used normally stay VERIFIED any input.
# --------------------------------------------------------------------------- #
class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = nn.Conv2d(3, 4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c(x))


@pytest.mark.smoke
def test_plain_mlp_stays_verified(tmp_path: Path) -> None:
    torch.manual_seed(0)
    capture_x = torch.randn(3, 4)
    path = _save(TinyMLP(), capture_x, tmp_path / "mlp.tlspec", include_weights=True)

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_plain_conv_stays_verified(tmp_path: Path) -> None:
    torch.manual_seed(0)
    capture_x = torch.randn(2, 3, 8, 8)
    path = _save(TinyConv(), capture_x, tmp_path / "conv.tlspec", include_weights=True)

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


class ReadOnlyParamScalarStat(nn.Module):
    """Read-only ``.item()`` / ``.tolist()`` param stat logging -> must stay VERIFIED."""

    def __init__(self) -> None:
        super().__init__()
        self.s = nn.Parameter(torch.tensor(2.0))
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.s.detach().item()
        _ = self.w.detach().tolist()
        return x * self.w * self.s


@pytest.mark.smoke
def test_readonly_param_scalar_stat_stays_verified(tmp_path: Path) -> None:
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(
        ReadOnlyParamScalarStat(), capture_x, tmp_path / "scalar_stat.tlspec", include_weights=True
    )

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
