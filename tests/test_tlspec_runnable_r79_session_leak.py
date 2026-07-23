"""Session-provenance-stamp leak closure (r79, r78 free HIGH).

Round-78 (free, Fable) demonstrated ONE root bug outside the r77 predicate's
session-scoped world: session cleanup removed provenance stamps by RE-TRAVERSING
the live model tree (``_restore_session_param_state`` iterated
``model.parameters()``; ``_undecorate_model_tensors`` iterated
``model.modules()`` / ``_buffers``), so a parameter or buffer POPPED from
``_parameters`` / ``_buffers`` DURING the donor capture's forward escaped both
walks and its prep stamp (non-empty ``ParamMeta.param_address``, or buffer
``TensorMeta`` address) SURVIVED the session. A LATER capture that consumed the
same object then had ``_tensor_has_known_provenance`` accept the stale stamp, no
``unattributed_tensor_args`` break marker was left, the r75 ladder judged the
chain clean, and the r76-F1 launder resurrected: layout twin VERIFIED (max-diff
20.0) on the param and buffer rungs, plus a WRONG-BIND on the SAME input --
``_build_param_fields`` resolved the foreign leaked object BY STALE ADDRESS into
this trace's ``param_logs`` with no object-identity check, so the save bound the
slot to the model's own different-valued ``w`` and replay staged the wrong
values as VERIFIED (max-diff 4.69 re-derived; free measured 4.55).

The r79 closure is two-layered:

1. ROOT (cleanup completeness): prep records every stamped param/buffer in
   ``trace._session_param_inventory`` / ``trace._session_buffer_inventory`` and
   cleanup clears stamps from that RECORDED inventory (the live-tree walks stay
   as belt-and-suspenders). A popped object can escape the tree, never the list.
2. DEFENSE IN DEPTH (consumers): a non-empty prep address counts as provenance
   ONLY when it resolves in THIS capture's session -- the address maps in
   ``trace.param_logs`` AND the recorded log's live object IS the value (exact
   identity). ``_build_param_fields`` applies the same identity gate before
   resolving, closing the wrong-bind even if a stamp ever leaks again.

Zero-collateral pins: registered/prepped param layout reads stay VERIFIED both
ways, honest conv+BN stays VERIFIED, hon1's non-popped cross-session reuse stays
blocked, and sequential captures double-clean without KeyError.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import (
    get_buffer_address,
    get_param_meta,
    get_tensor_meta,
    set_param_meta,
)
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness
from torchlens.utils import make_random_barcode

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _capture(model: nn.Module, x: torch.Tensor) -> "tl.Trace":
    """Capture one runnable-ready trace."""

    return tl.trace(model, x, capture=_CAPTURE)


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


class PopParamDonor(nn.Module):
    """Pathological donor: pops its own registered param mid-forward."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the param, then pop it from ``_parameters`` before returning."""

        out = x + self.w
        self._parameters.pop("w")
        return out


class NoPopParamDonor(nn.Module):
    """hon1's control donor: the param stays registered through cleanup."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Plain state-consuming forward."""

        return x + self.w


class PopBufferDonor(nn.Module):
    """Pathological donor: pops its own registered buffer mid-forward."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.register_buffer("b", torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the buffer, then pop it from ``_buffers`` before returning."""

        out = x + self.b
        self._buffers.pop("b")
        return out


class StashLaunderBranch(nn.Module):
    """The r76-F1 vehicle fed a foreign object held in a plain list.

    The plain-list stash dodges registration, so neither prep nor the old
    tree-walk cleanup of the SECOND capture ever sees the object; only a stale
    stamp from the DONOR capture can make it look state-rooted.
    """

    def __init__(self, stash: object) -> None:
        super().__init__()
        self._stash = [stash]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind the stashed object to input data, branch on its layout."""

        y = x * 2.0
        p = self._stash[0]
        p.data = y.detach()
        z = p * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class WrongBind(nn.Module):
    """(c) vehicle: consumes a foreign param on-path while registering OWN ``w``.

    Pre-fix the foreign object's stale ``'w'`` stamp resolved into THIS trace's
    ``param_logs['w']`` (the model's own, different-valued param) with no
    object-identity check, so replay staged the wrong values as VERIFIED.
    """

    def __init__(self, foreign: nn.Parameter) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8), requires_grad=False)
        self._stash = [foreign]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the foreign param on the value path alongside own state."""

        return x + self._stash[0] + 0.0 * self.w


class RegisteredParamWrapBranch(nn.Module):
    """Zero-collateral control: layout read through a REGISTERED/PREPPED param."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(2)
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a prepped-param-derived intermediate's layout."""

        q = self.w * 1.0
        if q.is_contiguous(memory_format=torch.channels_last):
            return x + 1.0
        return x - 1.0


class ConvBN(nn.Module):
    """Zero-collateral control: honest conv+BN (params AND buffers on-path)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(4)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard conv -> BN chain."""

        return self.bn(self.conv(x))


# ---------------------------------------------------------------------------
# RED-now-fixed: the leak scope (stamps must not survive the donor session)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_r79_popped_param_stamp_cleared_after_capture() -> None:
    """RED-now-fixed: a param popped mid-forward keeps NO session stamp.

    Pre-fix ``get_param_meta(donor_w).param_address`` survived as ``'w'`` (the
    popped param escaped the ``model.parameters()`` cleanup walk) and its forced
    ``requires_grad=True`` was never restored. The inventory-driven cleanup
    clears both; the non-popped control was always cleared.
    """

    x = _nchw()
    donor = PopParamDonor().eval()
    donor_w = donor.w
    assert donor_w.requires_grad is False
    tl.trace(donor, x.clone())
    meta = get_param_meta(donor_w)
    assert meta is None or not meta.param_address
    assert donor_w.requires_grad is False

    control = NoPopParamDonor().eval()
    control_w = control.w
    tl.trace(control, x.clone())
    cmeta = get_param_meta(control_w)
    assert cmeta is None or not cmeta.param_address


@pytest.mark.smoke
def test_r79_popped_buffer_stamp_cleared_after_capture() -> None:
    """RED-now-fixed: a buffer popped mid-forward keeps NO session tensor-meta.

    Pre-fix the leaked buffer kept ``TensorMeta(label_raw='buffer_1_raw',
    address='b')`` because it escaped the ``model.modules()`` / ``_buffers``
    cleanup traversal.
    """

    x = _nchw()
    donor = PopBufferDonor().eval()
    donor_b = donor.b
    tl.trace(donor, x.clone())
    assert get_buffer_address(donor_b) is None
    meta = get_tensor_meta(donor_b)
    assert meta is None or (
        meta.label_raw is None and meta.address is None and meta.buffer_source is None
    )


# ---------------------------------------------------------------------------
# RED-now-fixed: the three demonstrated false-VERIFIED consequences
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_r79_leaked_param_launder_ceils_on_layout_twin(tmp_path: Path) -> None:
    """RED-now-fixed (a): the popped-donor param launder twin must ceiling.

    Pre-fix the stale ``'w'`` stamp suppressed the break marker and the
    channels_last twin replayed the captured arm as VERIFIED (replay-vs-fresh
    max-diff 20.0). With the stamp cleared at donor cleanup the object is an
    ordinary unattributed tensor and the run fails closed.
    """

    x = _nchw()
    donor = PopParamDonor().eval()
    donor_w = donor.w
    tl.trace(donor, x.clone())

    path = _save(StashLaunderBranch(donor_w).eval(), x.clone(), tmp_path / "a.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r79_leaked_buffer_launder_ceils_on_layout_twin(tmp_path: Path) -> None:
    """RED-now-fixed (b): the popped-donor BUFFER launder twin must ceiling.

    Pre-fix the leaked buffer's surviving tensor-meta address rode the
    tensor-meta rung to the same false VERIFIED.
    """

    x = _nchw()
    donor = PopBufferDonor().eval()
    donor_b = donor.b
    tl.trace(donor, x.clone())

    path = _save(StashLaunderBranch(donor_b).eval(), x.clone(), tmp_path / "b.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r79_wrong_bind_refuses_at_save(tmp_path: Path) -> None:
    """RED-now-fixed (c): the address-collision wrong-bind must never save clean.

    Pre-fix the foreign leaked param (stale stamp ``'w'``) resolved into THIS
    trace's ``param_logs['w']``, the save bound the slot to the model's OWN
    different-valued ``w`` with no warning, and the ORIGINAL-input replay staged
    the wrong values as VERIFIED (re-derived max-diff 4.69). Post-fix the
    foreign object is unattributed: capture warns and the producer preflight
    refuses, so no artifact that would stage wrong values can exist.
    """

    x = _nchw()
    donor = PopParamDonor().eval()
    donor_w = donor.w
    tl.trace(donor, x.clone())

    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = _capture(WrongBind(donor_w).eval(), x.clone())
    with pytest.raises(RunnablePreflightError):
        trace.save(tmp_path / "c.tlspec", level="runnable", include_weights=True)


# ---------------------------------------------------------------------------
# Defense in depth: a FORGED stale stamp (simulated future cleanup escape)
# must be rejected by the current-session identity gate alone
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_r79_forged_stale_param_stamp_launder_ceils(tmp_path: Path) -> None:
    """White-box: even a stamp that DID survive (forged) never resolves.

    Simulates a hypothetical future cleanup escape by stamping a foreign param
    directly via ``set_param_meta``. The consumer-side session gate (address
    must map in THIS session AND ``param_logs[addr]._param_ref is value``) must
    reject it: the launder twin ceilings instead of resurrecting the r76-F1
    false VERIFIED.
    """

    x = _nchw()
    forged = nn.Parameter(torch.zeros(2, 4, 8, 8), requires_grad=False)
    set_param_meta(forged, barcode=make_random_barcode(), address="w", requires_grad_before=False)

    path = _save(StashLaunderBranch(forged).eval(), x.clone(), tmp_path / "forged_a.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r79_forged_stale_param_stamp_wrong_bind_refuses(tmp_path: Path) -> None:
    """White-box: the forged-stamp address collision cannot wrong-bind.

    The identity gate in ``_build_param_fields`` refuses to resolve the foreign
    object into ``param_logs['w']`` (the log's ``_param_ref`` is the model's own
    param, not this value), so the on-path consumption is unattributed: capture
    warns and the save refuses -- replay can never stage the wrong values.
    """

    x = _nchw()
    forged = nn.Parameter(torch.full((2, 4, 8, 8), 7.0), requires_grad=False)
    set_param_meta(forged, barcode=make_random_barcode(), address="w", requires_grad_before=False)

    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = _capture(WrongBind(forged).eval(), x.clone())
    with pytest.raises(RunnablePreflightError):
        trace.save(tmp_path / "forged_c.tlspec", level="runnable", include_weights=True)


# ---------------------------------------------------------------------------
# Zero-collateral GREENs
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_r79_registered_param_layout_read_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: a REGISTERED/PREPPED param layout read stays VERIFIED.

    The identity gate must not disturb honest state-rooted reads: the prepped
    param resolves (same session, same object), the chain stays clean, and both
    the same-layout run and the input-layout twin stay VERIFIED.
    """

    x = _nchw()
    path = _save(RegisteredParamWrapBranch().eval(), x.clone(), tmp_path / "reg.tlspec")

    same = tl.load(path).run(inputs=x.clone())
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not same.report.poisoned

    twin = tl.load(path).run(inputs=_twin(x))
    assert twin.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not twin.report.poisoned


@pytest.mark.smoke
def test_r79_honest_conv_bn_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: honest conv+BN (params AND buffers on-path) verifies."""

    x = _nchw()
    path = _save(ConvBN().eval(), x.clone(), tmp_path / "convbn.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r79_non_popped_cross_session_reuse_stays_blocked(tmp_path: Path) -> None:
    """Zero collateral (hon1's case): non-popped cross-session reuse stays blocked.

    A donor that keeps its param registered has the stamp cleared by cleanup
    (both pre- and post-fix), so reusing the object in a later launder capture
    must keep failing closed.
    """

    x = _nchw()
    donor = NoPopParamDonor().eval()
    donor_w = donor.w
    tl.trace(donor, x.clone())

    path = _save(StashLaunderBranch(donor_w).eval(), x.clone(), tmp_path / "nonpop.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r79_sequential_captures_cleanup_idempotent(tmp_path: Path) -> None:
    """Zero collateral: normal sequential captures double-clean without errors.

    The inventory pass runs BEFORE the belt-and-suspenders tree walk; both are
    idempotent, so two back-to-back captures of the same model must clean up
    perfectly (no KeyError/double-clear), restore ``requires_grad``, and keep
    producing VERIFIED runnable artifacts.
    """

    x = _nchw()
    model = ConvBN().eval()
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}

    tl.trace(model, x.clone())
    for name, p in model.named_parameters():
        assert p.requires_grad == original_requires_grad[name]
        meta = get_param_meta(p)
        assert meta is None or not meta.param_address
    for _, b in model.named_buffers():
        assert get_buffer_address(b) is None

    path = _save(model, x.clone(), tmp_path / "second.tlspec")
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    for name, p in model.named_parameters():
        assert p.requires_grad == original_requires_grad[name]
