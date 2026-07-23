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
from torchlens.backends.torch._tl import get_buffer_address, get_tensor_meta
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
