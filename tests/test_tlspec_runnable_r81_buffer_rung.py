"""Buffer-rung full parity with the param rung (r81, r80 free/hon1/Sol HIGHs).

Round 80 demonstrated TWO HIGH false-VERIFIEDs sharing ONE root: the
buffer/tensor-meta provenance rung trusted a STATIC buffer stamp
(``TensorMeta.address``, and the sibling ``label_raw``/``buffer_source``
components) with NO verification that the receiver's CURRENT storage is still
current-session captured state. r79 gave the PARAM rung a complete defense
(prep inventory + the ``_param_ref is value`` identity belt); the BUFFER rung
had only partial inventory coverage and no belt.

Finding 1 (cross-capture): a buffer stamped by ``_record_write`` (reached from
BOTH ``record_reassignment`` and ``record_op_writes``) was never appended to
``trace._session_buffer_inventory``, and an object stashed on a
``types.ModuleType`` attribute escaped the cleanup tree-walk entirely
(``_clear_session_tensor_metadata`` returned immediately for modules). The
surviving stamp -- e.g. ``('buffer_2_raw', 'b', None)`` on a reassigned
external -- rode the un-belted rung in a LATER capture: the layout twin of the
r76-F1 launder replayed VERIFIED/unpoisoned with replay-vs-fresh max-diff 20.0.

Finding 2 (single capture, more realistic): a plain-tensor attribute or
list-element buffer gets a legitimate prep stamp but is NOT host-write-tracked
(``refresh_index`` walks ``named_buffers()`` only), so ``q.data =
<input-derived>`` mid-forward rebinds its storage UNDETECTED. The raw stamp
then resolved every downstream product to a pure ``state:`` leaf basis in the
dispatch-origin ledger (``_operand_leaf_origins``), the layout ladder recorded
nothing (residual (3)), and an INPUT-derived layout branch -- the exact class
r73/r75 closed -- replayed the twin as false VERIFIED (max-diff 20.0).
Registered buffers correctly ceiled (M4); the asymmetry was the bug.

The r81 closure (parity with the param rung, in three layers):

1. ROOT completeness: EVERY buffer-stamp path routes through
   ``register_session_buffer_stamp`` -- prep (registered, plain-attr, and
   list-element), ``_tag_untagged_buffers``, ``refresh_index``, and BOTH
   ``_record_write`` entry points -- so the inventory cleanup always clears
   the stamp; the ModuleType cleanup blind spot is closed with a shallow
   module-namespace sweep.
2. STORAGE-IDENTITY BELT (``session_validated_buffer_address``): a static
   stamp is trusted ONLY when the exact object was stamped THIS session
   (``trace._session_buffer_identity``) AND its live storage is still the
   pinned stamp-time storage. Enforced at every consumer: the provenance rung,
   the three buffer-source logging gates (wrappers / ops / module entry), the
   tracker's direct-stamp fast path, and the witness state-attribution rungs
   (``_state_derived_addresses`` / ``_state_direct_address`` /
   ``_operand_origins`` / ``_operand_leaf_origins`` /
   ``_is_buffer_state_view_dispatch``).
3. SESSION-SCOPED rung components: ``label_raw`` and ``buffer_source`` count
   as provenance only when they resolve in THIS capture's live event index.

Zero-collateral pins: a plain-attr STATE-derived layout branch (no rebind)
stays VERIFIED both ways (the belt passes for unrebound stamps), registered
buffer layout reads and residual-(3) ``self.w.data`` reads stay VERIFIED,
ConvBN stays VERIFIED, plain-attr value-path reads keep their PRE-EXISTING
typed save refusal, and sequential captures double-clean the registry.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch._tl import (
    get_buffer_address,
    get_tensor_meta,
    set_buffer_address,
    set_tensor_label,
)
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

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


def _meta_clear(t: torch.Tensor) -> bool:
    """Return whether a tensor carries NO session tensor metadata components."""

    meta = get_tensor_meta(t)
    return meta is None or (
        meta.label_raw is None and meta.address is None and meta.buffer_source is None
    )


class ReassignPopDonor(nn.Module):
    """F1a donor: reassigns its buffer to a ModuleType-stashed external, then pops.

    ``self.b = <external>`` routes through ``record_reassignment`` ->
    ``_record_write``, which stamps the external with address ``'b'``; the pop
    removes it from ``_buffers`` so no tree re-traversal can reach it, and the
    ``types.ModuleType`` holder was the r80 cleanup-walk blind spot.
    """

    def __init__(self, stash_mod: types.ModuleType) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.register_buffer("b", torch.randn(2, 4, 8, 8))
        self._stash_mod = stash_mod

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the buffer, reassign it to the stashed external, pop it."""

        out = x + self.b
        self.b = self._stash_mod.payload
        self._buffers.pop("b")
        return out


class OpWriteReassignPopDonor(nn.Module):
    """F1b donor: reassign to an external, in-place-write it, then pop.

    The ``add_`` exercises the ``record_op_writes`` -> ``_record_write`` entry
    point on the reassigned external; nulling ``self._ext`` leaves the object
    reachable only from test scope (hon1's unreachable configuration).
    """

    def __init__(self, ext: torch.Tensor) -> None:
        super().__init__()
        torch.manual_seed(2)
        self.register_buffer("b", torch.randn(2, 4, 8, 8))
        self._ext: torch.Tensor | None = ext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the buffer, reassign, in-place-write, pop, drop the model ref."""

        out = x + self.b
        self.b = self._ext
        self.b.add_(1.0)
        self._buffers.pop("b")
        self._ext = None
        return out


class StashLaunderBranch(nn.Module):
    """The r76-F1/r80 launder vehicle: rebind a stashed object, branch on layout.

    The underscore-prefixed plain-list stash dodges prep stamping, so ONLY a
    provenance stamp already carried by the stashed object can suppress the
    unattributed break marker for ``z = q * 1.0``.
    """

    def __init__(self, stash: object) -> None:
        super().__init__()
        self._stash = [stash]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind the stashed object to input data, branch on its layout."""

        y = x * 2.0
        q = self._stash[0]
        q.data = y.detach()
        z = q * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class CollidingBufferLaunder(StashLaunderBranch):
    """Launder variant registering its OWN buffer at the forged/stale address.

    Pins the tracker's direct-stamp fast path: a foreign object whose stamp
    address merely COLLIDES with a current-session registered name must not
    resolve (identity is required), so the launder still ceilings.
    """

    def __init__(self, stash: object) -> None:
        super().__init__(stash)
        torch.manual_seed(13)
        self.register_buffer("b", torch.randn(2, 4, 8, 8))


class PlainAttrRebindBranch(nn.Module):
    """F2/M3: direct plain-tensor attribute, prep-stamped, ``.data``-rebound."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.q = torch.randn(2, 4, 8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind the stamped plain-attr buffer to input data, branch on layout."""

        y = x * 2.0
        q = self.q
        q.data = y.detach()
        z = q * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class ListAttrRebindBranch(nn.Module):
    """F2/M1: list-element plain tensor, prep-stamped ``holder.0``, rebound."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(4)
        self.holder = [torch.randn(2, 4, 8, 8)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind the stamped list-element buffer to input data, branch on layout."""

        y = x * 2.0
        q = self.holder[0]
        q.data = y.detach()
        z = q * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class UnderscoreAttrRebindBranch(nn.Module):
    """F2/M2 control: underscore attr is never prep-stamped -> already ceiled."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(5)
        self._holder = torch.randn(2, 4, 8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind the UNSTAMPED plain tensor, branch on layout."""

        y = x * 2.0
        q = self._holder
        q.data = y.detach()
        z = q * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class RegisteredBufferRebindBranch(nn.Module):
    """F2/M4 control: registered buffer is host-write-tracked -> already ceiled."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(6)
        self.register_buffer("q", torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rebind the registered buffer, branch on layout."""

        y = x * 2.0
        q = self.q
        q.data = y.detach()
        z = q * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class PlainAttrHonestLayoutBranch(nn.Module):
    """Zero-collateral: STATE-derived layout branch off an UNREBOUND plain attr.

    The belt must PASS here (same object, same storage as stamped), so the
    branch stays attributed to state and both runs stay VERIFIED (residual (3)).
    """

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.q = torch.randn(2, 4, 8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a layout derived purely from plain-attr state."""

        y = x * 2.0
        z = self.q * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return y + 10.0
        return y - 10.0


class RegisteredBufferLayoutBranch(nn.Module):
    """Zero-collateral: layout branch through a REGISTERED buffer."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.register_buffer("b", torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a registered-buffer-derived intermediate's layout."""

        z = self.b * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return x + 1.0
        return x - 1.0


class ParamDataReadBranch(nn.Module):
    """Zero-collateral: residual-(3) layout branch through ``self.w.data``."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(8)
        self.w = nn.Parameter(torch.randn(2, 4, 8, 8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on a state ``.data``-READ-derived intermediate's layout."""

        z = self.w.data * 1.0
        if z.is_contiguous(memory_format=torch.channels_last):
            return x + 1.0
        return x - 1.0


class HonestPlainAttrValueRead(nn.Module):
    """Plain-attr buffer consumed on the VALUE path (pre-existing save refusal)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(11)
        self.scale = torch.randn(2, 4, 8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Consume the plain-attr buffer's values."""

        return x * 2.0 + self.scale


class ConvBN(nn.Module):
    """Zero-collateral control: honest conv+BN (params AND buffers on-path)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(9)
        self.conv = nn.Conv2d(4, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard conv -> BN chain."""

        return self.bn(self.conv(x))


def _fresh_module_stash() -> types.ModuleType:
    """Return a ModuleType holding a fresh external payload tensor."""

    stash = types.ModuleType("r81_stash")
    torch.manual_seed(17)
    stash.payload = torch.randn(2, 4, 8, 8)
    return stash


# ---------------------------------------------------------------------------
# F1 RED-now-fixed: _record_write stamps must not survive the donor session
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_r81_record_write_reassign_stamp_cleared() -> None:
    """RED-now-fixed: the reassignment-stamped ModuleType-stashed external is cleaned.

    Pre-fix the external kept ``TensorMeta('buffer_2_raw', 'b', None)``: the
    ``_record_write`` stamp was never inventoried, the pop removed it from the
    ``_buffers`` walk, and the ModuleType holder was skipped by the cleanup
    tree-walk. Post-fix both the inventory and the shallow module-namespace
    sweep clear it.
    """

    x = _nchw()
    stash = _fresh_module_stash()
    donor = ReassignPopDonor(stash).eval()
    tl.trace(donor, x.clone())
    assert get_buffer_address(stash.payload) is None
    assert _meta_clear(stash.payload)


@pytest.mark.smoke
def test_r81_record_write_opwrite_stamp_cleared() -> None:
    """RED-now-fixed: the in-place (``record_op_writes``) vehicle is cleaned too.

    Both ``_record_write`` entry points leaked pre-fix (hon1's second vehicle:
    reassign-to-external, ``add_``, pop, null the model ref).
    """

    x = _nchw()
    torch.manual_seed(18)
    ext = torch.randn(2, 4, 8, 8)
    donor = OpWriteReassignPopDonor(ext).eval()
    tl.trace(donor, x.clone())
    assert get_buffer_address(ext) is None
    assert _meta_clear(ext)


@pytest.mark.smoke
def test_r81_stale_reassign_stamp_launder_ceils(tmp_path: Path) -> None:
    """RED-now-fixed (F1a): the reuse launder twin must ceiling, never VERIFY.

    Pre-fix the surviving stamp rode the un-belted buffer rung: the
    channels_last twin replayed the captured arm as VERIFIED/unpoisoned with
    replay-vs-fresh max-diff 20.0.
    """

    x = _nchw()
    stash = _fresh_module_stash()
    donor = ReassignPopDonor(stash).eval()
    tl.trace(donor, x.clone())

    path = _save(StashLaunderBranch(stash.payload).eval(), x.clone(), tmp_path / "f1a.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r81_stale_opwrite_stamp_launder_ceils(tmp_path: Path) -> None:
    """RED-now-fixed (F1b): the op-write-stamped external's launder twin ceilings."""

    x = _nchw()
    torch.manual_seed(18)
    ext = torch.randn(2, 4, 8, 8)
    donor = OpWriteReassignPopDonor(ext).eval()
    tl.trace(donor, x.clone())

    path = _save(StashLaunderBranch(ext).eval(), x.clone(), tmp_path / "f1b.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r81_no_donor_control_ceils(tmp_path: Path) -> None:
    """Control B: a fresh foreign tensor in the SAME holder always ceilinged.

    The stale stamp is the sole delta between this control and the F1 REDs.
    """

    x = _nchw()
    torch.manual_seed(19)
    fresh_foreign = torch.randn(2, 4, 8, 8)
    path = _save(StashLaunderBranch(fresh_foreign).eval(), x.clone(), tmp_path / "f1ctrl.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


# ---------------------------------------------------------------------------
# Defense in depth: FORGED stamps (simulated future tagging-path escape) must
# be rejected by the session belt alone
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_r81_forged_stale_buffer_stamp_launder_ceils(tmp_path: Path) -> None:
    """White-box: a raw ``set_buffer_address`` stamp that DID survive never resolves.

    Simulates a hypothetical future tagging path that escapes both the
    inventory and the cleanup walks. The consumer-side belt (current-session
    identity registry + live storage identity) must reject it on its own.
    """

    x = _nchw()
    torch.manual_seed(20)
    forged = torch.zeros(2, 4, 8, 8)
    set_buffer_address(forged, "q")

    path = _save(StashLaunderBranch(forged).eval(), x.clone(), tmp_path / "forged.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r81_forged_stamp_address_collision_ceils(tmp_path: Path) -> None:
    """White-box: a forged stamp COLLIDING with a registered name cannot resolve.

    The launder model registers its OWN buffer ``'b'``; the foreign object's
    forged ``'b'`` stamp must not resolve through the tracker's direct-stamp
    fast path (identity required) nor the belt (not session-registered), so
    the twin still ceilings instead of binding to the model's own state.
    """

    x = _nchw()
    torch.manual_seed(21)
    forged = torch.full((2, 4, 8, 8), 7.0)
    set_buffer_address(forged, "b")

    path = _save(CollidingBufferLaunder(forged).eval(), x.clone(), tmp_path / "collide.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r81_forged_stale_label_launder_ceils(tmp_path: Path) -> None:
    """White-box: a bare stale ``label_raw`` no longer satisfies the rung.

    Pre-fix ANY non-None ``label_raw`` counted as provenance; post-fix the
    label must resolve in THIS capture's live event index. A stale label that
    resolves nowhere leaves the break marker and the twin ceilings.
    """

    x = _nchw()
    torch.manual_seed(22)
    forged = torch.zeros(2, 4, 8, 8)
    set_tensor_label(forged, "buffer_9_raw")

    path = _save(StashLaunderBranch(forged).eval(), x.clone(), tmp_path / "label.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


# ---------------------------------------------------------------------------
# F2 RED-now-fixed: plain-attr buffer ``.data=`` input-layout launder
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_r81_plain_attr_data_rebind_launder_ceils(tmp_path: Path) -> None:
    """RED-now-fixed (F2/M3): the direct plain-attr rebind twin must ceiling.

    Pre-fix the legit prep stamp resolved the rebound receiver -- and every
    downstream product -- to a pure ``state:`` leaf basis, the layout ladder
    recorded nothing (residual (3)), and the channels_last twin replayed the
    captured arm as VERIFIED with max-diff 20.0 in a SINGLE fresh capture.
    """

    x = _nchw()
    path = _save(PlainAttrRebindBranch().eval(), x.clone(), tmp_path / "m3.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r81_list_attr_data_rebind_launder_ceils(tmp_path: Path) -> None:
    """RED-now-fixed (F2/M1): the list-element spelling must ceiling too."""

    x = _nchw()
    path = _save(ListAttrRebindBranch().eval(), x.clone(), tmp_path / "m1.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r81_underscore_attr_rebind_stays_ceiled(tmp_path: Path) -> None:
    """Control (F2/M2): the unstamped underscore-attr spelling keeps ceiling."""

    x = _nchw()
    path = _save(UnderscoreAttrRebindBranch().eval(), x.clone(), tmp_path / "m2.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_r81_registered_buffer_rebind_stays_ceiled(tmp_path: Path) -> None:
    """Control (F2/M4): the registered-buffer spelling keeps ceiling.

    Registered buffers were always host-write-tracked; plain-attr spellings
    now ceiling exactly like this control (the asymmetry was the bug).
    """

    x = _nchw()
    path = _save(RegisteredBufferRebindBranch().eval(), x.clone(), tmp_path / "m4.tlspec")
    result = tl.load(path).run(inputs=_twin(x))
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.poisoned


# ---------------------------------------------------------------------------
# Zero-collateral GREENs
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_r81_plain_attr_state_layout_branch_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: an UNREBOUND plain-attr state layout branch verifies.

    The belt passes for a stamp whose object and storage are unchanged, so the
    branch stays state-attributed (residual (3)) on both the same-layout run
    and the channels_last input twin.
    """

    x = _nchw()
    path = _save(PlainAttrHonestLayoutBranch().eval(), x.clone(), tmp_path / "green_pa.tlspec")

    same = tl.load(path).run(inputs=x.clone())
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not same.report.poisoned

    twin = tl.load(path).run(inputs=_twin(x))
    assert twin.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not twin.report.poisoned


@pytest.mark.smoke
def test_r81_registered_buffer_layout_branch_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: a registered-buffer layout read verifies both ways."""

    x = _nchw()
    path = _save(RegisteredBufferLayoutBranch().eval(), x.clone(), tmp_path / "green_rb.tlspec")

    same = tl.load(path).run(inputs=x.clone())
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not same.report.poisoned

    twin = tl.load(path).run(inputs=_twin(x))
    assert twin.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not twin.report.poisoned


@pytest.mark.smoke
def test_r81_param_data_read_branch_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: residual-(3) ``self.w.data`` READ-sourced layout verifies."""

    x = _nchw()
    path = _save(ParamDataReadBranch().eval(), x.clone(), tmp_path / "green_pd.tlspec")

    same = tl.load(path).run(inputs=x.clone())
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not same.report.poisoned

    twin = tl.load(path).run(inputs=_twin(x))
    assert twin.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not twin.report.poisoned


@pytest.mark.smoke
def test_r81_honest_conv_bn_stays_verified(tmp_path: Path) -> None:
    """Zero collateral: honest conv+BN (params AND buffers on-path) verifies."""

    x = _nchw()
    path = _save(ConvBN().eval(), x.clone(), tmp_path / "green_bn.tlspec")
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_r81_plain_attr_value_read_keeps_typed_refusal(tmp_path: Path) -> None:
    """Boundary pin: plain-attr VALUE-path reads keep the pre-existing refusal.

    A plain-tensor attribute is not part of the declared runnable state model,
    so consuming its VALUES on the output path refused at save BEFORE r81 and
    must keep refusing typed (never silently succeed or crash) after.
    """

    x = _nchw()
    trace = _capture(HonestPlainAttrValueRead().eval(), x.clone())
    with pytest.raises(RunnablePreflightError):
        trace.save(tmp_path / "value_read.tlspec", level="runnable", include_weights=True)


@pytest.mark.smoke
def test_r81_sequential_captures_registry_idempotent(tmp_path: Path) -> None:
    """Zero collateral: back-to-back captures reset and re-clean the registry.

    The identity registry is rebuilt at prep and emptied at cleanup; two
    sequential captures of the same model must leave no stamp behind and keep
    producing VERIFIED artifacts.
    """

    x = _nchw()
    model = PlainAttrHonestLayoutBranch().eval()

    tl.trace(model, x.clone())
    assert _meta_clear(model.q)

    path = _save(model, x.clone(), tmp_path / "second.tlspec")
    assert _meta_clear(model.q)
    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
