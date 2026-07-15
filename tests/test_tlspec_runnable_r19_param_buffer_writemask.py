"""Round-19 PARAM/BUFFER within-forward write-masking + read-only over-trigger cluster.

The r18 param/buffer parity fix keyed its host-write tripwire on "bytes-changed-but-version-STATIC"
to flag an untracked host write. A version-BUMPING op (a tracked in-place op, OR a host write
followed by one) DEFEATS that gate, so an untracked mutation slips through masked by the version
bump -> false VERIFIED. Params and buffers must be handled DIFFERENTLY:

* FINDING A (CRITICAL, params fail closed). A within-forward in-place op on a PARAMETER
  (``with torch.no_grad(): self.w.add_(1.0)``) is NOT captured in the replayable DAG (params carry
  no graph source node and are excluded from wrapper output logging), so the embedded pre-forward
  param state cannot reproduce the mutated forward. The pre-r19 code EXEMPTED version-bumped params
  from the byte tripwire on the false premise "version bump => the write is in the DAG"; that holds
  for buffers, not params. The fix drops the version-static exemption for params: ANY within-forward
  whole-storage byte change fails closed to UNVERIFIABLE, version-bumped OR static. This also
  subsumes a host write masked by a later version-bumping param op (a net byte change is flagged).

* FINDING B (CRITICAL, buffers reconcile against the journal). Buffers DIFFER from params: a buffer
  in-place update IS journaled and IS replayed, so a PURE journaled update (a BatchNorm running-stat
  step, a plain ``self.b.add_(1.0)``) MUST stay VERIFIED (no r15-C3 regression). The hole is a HOST
  write masked because the buffer ALSO gets a journaled bump: the journaled op refreshed the buffer
  snapshot from the buffer's ACTUAL (host-contaminated) bytes, so the end-of-forward value-change
  check saw no discrepancy. The fix reconciles the buffer's PRE-op bytes against the JOURNAL-EXPECTED
  bytes (the pre-forward snapshot advanced only by tracked ops); a divergence before a journaled op
  reveals the masked host write -> UNVERIFIABLE, while a pure journaled update matches and stays
  VERIFIED + value-correct.

* FINDING C (MEDIUM over-trigger, read-only DERIVED param stat). r18's read-only param escape
  resolution only covered DIRECTLY-param-rooted reads (``self.w.detach().item()``, ``self.w[0]``). A
  DERIVED pruned read off the param (``self.w.sum().item()``, ``self.w.tolist()``,
  ``float(self.w.max())``) hit the fail-closed INCOMPLETE_SCALAR_ESCAPE gate -> spurious UNVERIFIABLE.
  The fix extends the resolution to values derived purely from registered params (a pruned read chain
  whose autograd ancestry bottoms out only at param leaves): they are witnessed by the param state
  digest and stay VERIFIED, exactly like the buffer twin.

This module proves each finding on the default sparse save AND (for the writes) include_activations,
plus the hard no-regression / no-over-trigger contract: a plain BatchNorm running-stat update and a
pure journaled buffer add_ stay VERIFIED + value-correct, plain CNN/MLP stay VERIFIED on any input,
and directly-param-rooted read-only stats stay VERIFIED.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness

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


def _fresh_output(model_factory, state_dict, x: torch.Tensor) -> torch.Tensor:
    """The oracle-1 reference: a FRESH instance loaded with the captured state, one forward."""

    fresh = model_factory()
    fresh.load_state_dict(state_dict)
    with torch.no_grad():
        return fresh(x.clone())


# --------------------------------------------------------------------------- #
# FINDING A (CRITICAL): a within-forward in-place PARAM op fails closed, version-agnostic.
# --------------------------------------------------------------------------- #
class ParamNoGradAdd(nn.Module):
    """A direct in-place aten op on a PARAM (bumps the version) consumed downstream."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * self.w
        with torch.no_grad():
            self.w.add_(1.0)
        return y + self.w


class ParamGradMul(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.w.mul_(2.0)
        return x * self.w


class ParamCopy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.w.copy_(torch.tensor([9.0, 9.0]))
        return x * self.w


class ParamSetItem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.w[0] = 99.0
        return x * self.w


class NestedInnerWeightWrite(nn.Module):
    """An in-place op on a NESTED submodule's weight param."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.lin.weight.add_(1.0)
        return self.lin(x)


class ParamHostWriteThenVersionBump(nn.Module):
    """An untracked host write MASKED by a subsequent version-bumping in-place param op."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))
        self.npw: np.ndarray = self.w.detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.npw[0] += 10.0  # invisible host write into the param storage
        with torch.no_grad():
            self.w.add_(1.0)  # version bump that would MASK the host write
        return x * self.w


_INPLACE_PARAM_MODULES = [
    ParamNoGradAdd,
    ParamGradMul,
    ParamCopy,
    ParamSetItem,
    NestedInnerWeightWrite,
    ParamHostWriteThenVersionBump,
]


@pytest.mark.smoke
@pytest.mark.parametrize("model_cls", _INPLACE_PARAM_MODULES)
@pytest.mark.parametrize("include_activations", [False, True])
def test_inplace_param_write_is_unverifiable(
    model_cls, include_activations: bool, tmp_path: Path
) -> None:
    model = model_cls()
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(
        model,
        capture_x,
        tmp_path / f"{model_cls.__name__}_{include_activations}.tlspec",
        include_weights=True,
        include_activations=include_activations,
    )

    result = _run(path, capture_x)

    # The param mutation is not replayable from the embedded pre-forward state; version-agnostic
    # byte-diff fail-closed refuses it. Never a false VERIFIED, never a false ATTESTED.
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


# --------------------------------------------------------------------------- #
# FINDING B (CRITICAL): a host write masked by a journaled BUFFER update is UNVERIFIABLE, but a
# pure journaled update stays VERIFIED + value-correct.
# --------------------------------------------------------------------------- #
class BufferHostWriteThenJournaledAdd(nn.Module):
    """An untracked host write MASKED by a subsequent journaled in-place buffer op."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 3.0]))
        self.npb: np.ndarray = self.b.detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.npb[0] += 10.0  # invisible host write into the buffer storage
        self.b.add_(1.0)  # journaled bump that would MASK the host write
        return x * self.b


class PureJournaledBufferAdd(nn.Module):
    """A pure journaled in-place buffer update: the replay reproduces it -> VERIFIED."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.b.add_(1.0)
        return x * self.b


class PlainBatchNormRunningStat(nn.Module):
    """A plain BatchNorm(affine=False) whose running-stat update is a journaled buffer write."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


@pytest.mark.smoke
@pytest.mark.parametrize("include_activations", [False, True])
def test_buffer_host_write_masked_by_journal_is_unverifiable(
    include_activations: bool, tmp_path: Path
) -> None:
    model = BufferHostWriteThenJournaledAdd()
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(
        model,
        capture_x,
        tmp_path / f"buffer_mask_{include_activations}.tlspec",
        include_weights=True,
        include_activations=include_activations,
    )

    result = _run(path, capture_x)

    # The journaled add_ masked the host write by refreshing the snapshot from contaminated bytes;
    # reconciling the pre-op bytes against the journal-expected bytes catches it.
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_pure_journaled_buffer_add_stays_verified_and_correct(tmp_path: Path) -> None:
    model = PureJournaledBufferAdd()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(model, capture_x, tmp_path / "pure_journal.tlspec", include_weights=True)

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    expected = _fresh_output(PureJournaledBufferAdd, state, capture_x)
    torch.testing.assert_close(result.output, expected)


@pytest.mark.smoke
def test_plain_batchnorm_running_stat_stays_verified_and_correct(tmp_path: Path) -> None:
    model = PlainBatchNormRunningStat()
    model.train()
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    capture_x = torch.randn(8, 4)
    path = _save(model, capture_x, tmp_path / "batchnorm.tlspec", include_weights=True)

    result = _run(path, capture_x)

    # The running-stat update is a TRACKED journaled buffer write; the replay reproduces it. No
    # r15-C3 regression: a plain BatchNorm stays VERIFIED and value-correct vs a fresh instance.
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    fresh = PlainBatchNormRunningStat()
    fresh.load_state_dict(state)
    fresh.train()
    with torch.no_grad():
        expected = fresh(capture_x.clone())
    torch.testing.assert_close(result.output, expected)


# --------------------------------------------------------------------------- #
# FINDING C (MEDIUM over-trigger): a read-only DERIVED param stat stays VERIFIED.
# --------------------------------------------------------------------------- #
class ParamSumItem(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.w.sum().item()
        return x * self.w


class ParamTolist(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.w.tolist()
        return x * self.w


class ParamFloatMax(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = float(self.w.max())
        return x * self.w


class MultiParamDerived(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([2.0, 3.0]))
        self.w2 = nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = (self.w1 + self.w2).mean().item()
        return x * self.w1 * self.w2


class ParamDirectDetachRead(nn.Module):
    """A DIRECTLY-param-rooted read-only read (r18); must stay VERIFIED after the r19 refactor."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.w[0].detach().item()
        return x * self.w


_READONLY_PARAM_MODULES = [
    ParamSumItem,
    ParamTolist,
    ParamFloatMax,
    MultiParamDerived,
    ParamDirectDetachRead,
]


# ``ParamFloatMax`` intentionally does float() on a requires_grad param reduction to exercise a
# derived read-only escape; torch's "converting to scalar" UserWarning is expected noise, not a
# torchlens signal, so it must not escalate under filterwarnings=error.
@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")
@pytest.mark.smoke
@pytest.mark.parametrize("model_cls", _READONLY_PARAM_MODULES)
def test_readonly_param_stat_stays_verified(model_cls, tmp_path: Path) -> None:
    model = model_cls()
    capture_x = torch.tensor([2.0, 4.0])
    path = _save(model, capture_x, tmp_path / f"{model_cls.__name__}.tlspec", include_weights=True)

    result = _run(path, capture_x)

    # A read-only derived param stat is a pure function of param state: it re-digests identically on
    # replay from the captured state, so it must stay VERIFIED (no INCOMPLETE_SCALAR_ESCAPE).
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------- #
# NO OVER-TRIGGER: plain models with no host write stay VERIFIED on any input.
# --------------------------------------------------------------------------- #
class PlainCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


@pytest.mark.smoke
def test_plain_cnn_stays_verified(tmp_path: Path) -> None:
    model = PlainCNN()
    capture_x = torch.randn(1, 3, 8, 8)
    path = _save(model, capture_x, tmp_path / "cnn.tlspec", include_weights=True)

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
