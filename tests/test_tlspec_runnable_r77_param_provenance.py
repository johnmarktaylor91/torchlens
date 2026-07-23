"""Fresh/foreign ``nn.Parameter`` provenance-launder closure (r77 F1).

Round-76 (hon1 + free-roam, cross-lab) confirmed the LAST layout-class vehicle: the first
rung of ``_tensor_has_known_provenance`` treated ANY ``torch.nn.Parameter`` -- including a
fresh in-forward ``nn.Parameter(y.detach())`` with no prep stamp, no barcode, no module
registration -- as attributed provenance. The consuming op therefore recorded an EMPTY
``unattributed_tensor_args`` marker, the r75 ancestry-integrity rung judged the chain CLEAN,
the empty ``input_ancestors`` rode the "positively state-rooted, record nothing" branch, and
a channels_last twin replayed the captured arm as a false VERIFIED (max-diff 20.0) with all
four r75 rungs evaded.

The r77 closure fixes the PREDICATE at the root: a Parameter counts as known provenance only
when it carries the prep-stamped ``ParamMeta`` address written by model preparation
(``_create_session_param_logs``). The lazy op-time barcode stamp writes ``param_address=""``
and does NOT count. Unprepped Parameters fall through to the tensor-meta rung like any other
tensor, so a fresh/foreign Parameter now leaves the break marker and the r75 ladder resolves
it honestly (ledger/storage rung, or the fail-closed floor -> UNVERIFIABLE).

Zero-collateral pins: registered/prepped parameters keep their empty markers, so state-rooted
layout reads stay residual (3) (VERIFIED both ways) and prepped-param mixed chains keep the
precise r73 witness (same-layout VERIFIED, twin ceiled). The save-time "no graph/source
provenance" warning now fires for ON-PATH fresh-Parameter consumption (it shares the marker);
branch-only chains are orphan-pruned before the warning step, so the honesty machinery -- not
the warning -- is their protection. The ``nn.Buffer`` control never had the exemption and
stays typed-honest.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import RunnablePreflightError
from torchlens.errors.runnable import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness


def _capture(model: nn.Module, x: torch.Tensor) -> "tl.Trace":
    """Capture one runnable-ready trace."""

    return tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )


def _save(model: nn.Module, x: torch.Tensor, path: Path) -> Path:
    """Capture and save a runnable artifact with embedded weights."""

    _capture(model, x).save(path, level="runnable", include_weights=True)
    return path


def _nchw() -> torch.Tensor:
    """Return a fixed contiguous NCHW probe input."""

    torch.manual_seed(0)
    return torch.randn(2, 4, 8, 8)


def _twin(x: torch.Tensor) -> torch.Tensor:
    """Return the same-shape+dtype+values channels_last twin."""

    return x.clone().to(memory_format=torch.channels_last)


class ParamLaunderBranch(nn.Module):
    """The r76 F1 vehicle: layout read laundered through a fresh in-forward Parameter."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the laundered intermediate's channels_last contiguity."""

        y = x * 2.0
        p = torch.nn.Parameter(y.detach(), requires_grad=False)
        z = p * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class RegisteredParamWrapBranch(nn.Module):
    """Zero-collateral control: layout read through a REGISTERED/PREPPED parameter."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a prepped-param-derived intermediate's layout; never the input's."""

        q = self.w * 1.0
        if q.is_contiguous(memory_format=torch.channels_last):
            return x + 1.0
        return x - 1.0


class MixedParamChainBranch(nn.Module):
    """Zero-collateral control: prepped param MIXED into an input-rooted layout chain."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(2)
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the layout of ``x + w`` (labeled clean chain through the param)."""

        z = x + self.w
        if z.is_contiguous(memory_format=torch.channels_last):
            return z + 10.0
        return z - 10.0


class OnPathFreshParam(nn.Module):
    """Save-door control: fresh Parameter consumed ON the value path (not branch-only)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route the laundered Parameter's value into the output."""

        y = x * 2.0
        p = torch.nn.Parameter(y.detach(), requires_grad=False)
        return p * 1.0 + y


class BufferLaunderBranch(nn.Module):
    """Control: ``nn.Buffer`` never had the isinstance exemption and must stay honest."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a fresh in-forward Buffer's contiguity."""

        y = x * 2.0
        b = torch.nn.Buffer(y.detach())
        z = b * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


@pytest.mark.smoke
def test_r77_fresh_parameter_launder_ceils_on_layout_twin(tmp_path: Path) -> None:
    """RED-now-fixed: the r76 Parameter-launder twin must ceiling, never false-VERIFY.

    Pre-fix this replayed the captured arm as VERIFIED with replay-vs-live max-diff 20.0;
    the fresh Parameter now leaves the ``unattributed_tensor_args`` break marker, the
    ancestry-integrity rung taints the chain, and the fail-closed floor ceilings the run.
    """

    x = _nchw()
    path = _save(ParamLaunderBranch().eval(), x, tmp_path / "launder.tlspec")

    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


@pytest.mark.smoke
def test_r77_fresh_parameter_launder_fails_closed_on_same_layout(tmp_path: Path) -> None:
    """Honest-ceiling posture pin: the launder model fails CLOSED even on the original input.

    The dispatch-origin ledger never saw ``Parameter.__new__`` (not a logged op), so the
    tainted chain resolves through no positive rung and hits the r75 fail-closed floor on
    EVERY run -- an honest completeness ceiling (like the hidden-global floor), never a
    silent no-record.
    """

    x = _nchw()
    path = _save(ParamLaunderBranch().eval(), x, tmp_path / "launder_same.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r77_registered_param_wrap_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: a REGISTERED/PREPPED param keeps its empty marker.

    The wrap's layout read is positively state-rooted (residual (3)'s STATE side): the
    prep-stamped address satisfies the fixed predicate, the chain stays clean, and both the
    same-layout run and the input-layout twin stay VERIFIED (the read never keyed on the
    input's layout).
    """

    x = _nchw()
    path = _save(RegisteredParamWrapBranch().eval(), x, tmp_path / "regwrap.tlspec")

    same = tl.load(path).run(inputs=x.clone())
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not same.report.poisoned

    twin = tl.load(path).run(inputs=_twin(x))
    assert twin.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not twin.report.poisoned


@pytest.mark.smoke
def test_r77_prepped_param_mixed_chain_keeps_precise_witness(tmp_path: Path) -> None:
    """Zero collateral: a prepped param mixed into an input-rooted chain stays PRECISE.

    ``x + w`` is a labeled clean chain (the prepped param leaves no marker), so the layout
    read keeps the r73 per-input witness: same-layout VERIFIED, twin ceiled UNVERIFIABLE --
    not a blanket fail-close.
    """

    x = _nchw()
    path = _save(MixedParamChainBranch().eval(), x, tmp_path / "mixed.tlspec")

    same = tl.load(path).run(inputs=x.clone())
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not same.report.poisoned

    twin = tl.load(path).run(inputs=_twin(x))
    assert twin.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert twin.report.poisoned


@pytest.mark.smoke
def test_r77_on_path_fresh_parameter_warns_and_refuses_at_save(tmp_path: Path) -> None:
    """Save-door posture: ON-PATH fresh-Parameter consumption now WARNS and still refuses.

    Pre-fix the bare-isinstance exemption suppressed the "no graph/source provenance"
    warning for Parameters; the marker now drives it (hon1's sibling consumer). The sparse
    recipe still cannot rebuild the unattributable value, so the producer preflight refusal
    is unchanged. (Branch-only chains are orphan-pruned before the warning step, so only
    the honesty ceiling -- not the warning -- covers them.)
    """

    x = _nchw()
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = _capture(OnPathFreshParam().eval(), x)
    with pytest.raises(RunnablePreflightError):
        trace.save(tmp_path / "onpath.tlspec", level="runnable", include_weights=True)


@pytest.mark.smoke
def test_r77_buffer_control_stays_honest(tmp_path: Path) -> None:
    """Control: ``nn.Buffer`` (never exempted) keeps its typed-honest behavior.

    A fresh in-forward Buffer registration violates its in-place alias/version expectation
    on replay, so the twin raises typed ``PathDivergenceError`` (raise-and-rollback default)
    -- never a false VERIFIED.
    """

    if not hasattr(torch.nn, "Buffer"):
        pytest.skip("torch.nn.Buffer requires torch >= 2.5")

    x = _nchw()
    path = _save(BufferLaunderBranch().eval(), x, tmp_path / "buffer.tlspec")

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=_twin(x))
