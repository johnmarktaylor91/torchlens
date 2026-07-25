"""Registered buffer write-side node keeps its backend_address (r87, free FINDING-1 residual).

r85's Option-a ``backend_address`` cleanup established the rule "a REGISTERED
buffer keeps its registered address; a plain-attribute / list-element buffer is
decoupled to ``None``" and restored op / activation / input / output nodes to the
coupled ``backend_address == address`` default. It closed the flat read-side case
and the plain-attribute case, but left one residual open: a registered buffer that
appears as MORE THAN ONE graph node.

The near-universal read-side + write-side split -- any in-place ``add_`` /
``copy_`` / ``mul_``, and every BatchNorm / InstanceNorm running-stat -- produces
two buffer nodes for the SAME registered buffer. In ``_buffer_addresses_by_label``
the READ node claims the registered address from a one-claim pool; the WRITE
node's recorded address is then no longer in the pool, so it fell through
unassigned and ``_fields_from_event`` stamped ``backend_address = None`` on it.
The result contradicted r85's own rule and broke intra-trace consistency::

    buffer_1  addr='b' backend_address='b'    <- READ side
    buffer_2  addr='b' backend_address=None   <- WRITE side (SAME registered 'b')

r87 CLOSURE. The write-side node reports the SAME registered ``backend_address``
as the read-side node. The discriminator is REGISTERED-ness -- the node's OWN
recorded address must genuinely name a buffer in the model's declared-state
universe (``trace._buffer_initial_values``, the same set the runnable preflight
refuses an address against) -- NOT one-claim-pool membership. This is never a
heuristic borrow: only a node whose own recorded address is a registered buffer
name is given that address.

THE HARD CONSTRAINT (r84 C2 must NOT reopen). The one-claim pool exists to stop a
DIFFERENT, non-registered tensor (a plain ``__dict__`` attribute or list element)
from stealing a registered buffer's address -- a silent wrong bind that once
replayed ``VERIFIED`` against the wrong state. Because the r87 discriminator keys
on registered-ness, a non-registered tensor -- even one that collides on a
registered buffer's SHAPE -- never receives a registered ``backend_address``; it
stays ``None`` exactly as C2 requires. And because ``backend_address`` is
verdict-neutral (no runnable-load / param_source / resolver consumer reads it --
only the string-when-present metadata invariant does), the fix moves no verdict
and no replay fingerprint; the display ``address`` and the save binding that DRIVE
the C2 refusal are untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


def _buffer_ops(trace: Any) -> list[Any]:
    """Return every materialized buffer op of ``trace``."""

    return [op for op in trace if getattr(op, "is_buffer", False)]


def _capture(model: nn.Module, x: torch.Tensor) -> Any:
    """Capture one runnable-ready trace."""

    return tl.trace(model, x, capture=_CAPTURE)


# --------------------------------------------------------------------------- #
# Models.
# --------------------------------------------------------------------------- #


class _InPlaceBuffer(nn.Module):
    """A registered buffer read, written in place, then read again.

    ``b.add_(x)`` produces a write-side buffer node distinct from the read-side
    node, both for the SAME registered buffer ``b``.
    """

    def __init__(self) -> None:
        """Register the single mutated buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into the buffer in place, then consume it."""

        self.b.add_(x)
        return self.b * 2


class _BatchNorm(nn.Module):
    """A single BatchNorm2d -- running stats are read + write registered buffers."""

    def __init__(self) -> None:
        """Build the BatchNorm layer."""

        super().__init__()
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BatchNorm so its running stats appear as read + write nodes."""

        return self.bn(x)


class _PlainVsRegisteredSameShape(nn.Module):
    """A non-registered constant colliding on a registered buffer's SHAPE.

    The r84 C2 theft target: ``q`` is a plain attribute, ``top`` is registered,
    and they share a shape, so the old value/shape heuristic gave ``q`` ``top``'s
    address. ``q`` must keep a display ``address`` but NO ``backend_address``.
    """

    def __init__(self) -> None:
        """Hold one plain and one registered constant of identical shape."""

        super().__init__()
        self.q = torch.full((2, 2), 3.0)
        self.register_buffer("top", torch.full((2, 2), 5.0))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Consume the plain attribute first, then the registered buffer."""

        return (z - self.q) + self.top


# --------------------------------------------------------------------------- #
# THE FIX -- registered read + write nodes agree on backend_address.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_inplace_buffer_read_and_write_nodes_agree_on_backend_address() -> None:
    """Both nodes of an in-place-mutated registered buffer report its address.

    Pre-r87 the write-side node reported ``backend_address=None`` while the
    read-side node reported ``'b'``. Both must now report ``'b'`` -- and a read
    AND a write node must genuinely be present (parents empty vs non-empty), so
    the test cannot be satisfied by a trace that lost the split.
    """

    trace = _capture(_InPlaceBuffer(), torch.ones(3))
    b_ops = [op for op in _buffer_ops(trace) if op.address == "b"]

    assert len(b_ops) == 2, f"expected a read + write node for 'b', got {b_ops}"
    read_nodes = [op for op in b_ops if not op.parents]
    write_nodes = [op for op in b_ops if op.parents]
    assert len(read_nodes) == 1 and len(write_nodes) == 1, "need one read and one write node"

    for op in b_ops:
        assert op.address == "b", "display address unchanged on both nodes"
        assert op.backend_address == "b", (
            f"{op.layer_label} (parents={op.parents}) reports "
            f"backend_address={op.backend_address!r}, expected 'b'"
        )


@pytest.mark.smoke
@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
def test_batchnorm_running_stat_read_and_write_pairs_agree(train: bool) -> None:
    """Every BatchNorm running-stat node -- read AND write -- keeps its address.

    ``bn.running_mean`` / ``bn.running_var`` each appear as a read node and a
    write node. No node bearing one of those display addresses may report
    ``backend_address=None``.
    """

    model = _BatchNorm()
    model.train(train)
    trace = _capture(model, torch.randn(4, 2, 4, 4))

    stat_ops = [
        op for op in _buffer_ops(trace) if op.address in {"bn.running_mean", "bn.running_var"}
    ]
    assert stat_ops, "BatchNorm running stats did not materialize as buffer nodes"

    # At least one read + one write node exist for the running stats overall.
    assert any(not op.parents for op in stat_ops), "no running-stat read node"
    assert any(op.parents for op in stat_ops), "no running-stat write node"

    for op in stat_ops:
        assert op.backend_address == op.address, (
            f"{op.layer_label} (addr={op.address!r}, parents={op.parents}) reports "
            f"backend_address={op.backend_address!r}"
        )


# --------------------------------------------------------------------------- #
# THE HARD CONSTRAINT -- a non-registered collision stays None (C2 closed).
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_plain_attribute_colliding_on_registered_shape_keeps_no_backend_address() -> None:
    """The r84 C2 guard: registered-ness, not the name/shape, gates the address.

    ``q`` is a plain attribute sharing ``top``'s shape. It must keep its display
    ``address`` ('q') but report ``backend_address=None`` -- the registered-only
    discriminator must not hand it ``top``'s (or any) registered address, even
    though the r87 fix now assigns a registered address to a same-named
    registered second node. ``top`` keeps its own address, coupled.
    """

    trace = _capture(_PlainVsRegisteredSameShape(), torch.zeros(2, 2))
    by_address = {op.address: op for op in _buffer_ops(trace) if op.address is not None}

    assert by_address["q"].address == "q", "plain attribute keeps its display address"
    assert by_address["q"].backend_address is None, (
        "a non-registered plain attribute must have no backend_address (C2)"
    )
    assert by_address["top"].backend_address == "top", "registered buffer keeps its address"


# --------------------------------------------------------------------------- #
# VERDICT NEUTRALITY -- the fix moves no runnable verdict.
# --------------------------------------------------------------------------- #


def test_read_write_buffer_model_still_verifies(tmp_path: Path) -> None:
    """A read+write buffer model stays VERIFIED -- ``backend_address`` is neutral.

    The fix only changes the verdict-neutral ``backend_address`` public field, so
    the runnable verdict, poison state, and replay output for an in-place buffer
    model must be unchanged: VERIFIED, not poisoned, byte-exact against oracle 1,
    and robust to a DIFFERENT input (proving the replay models ``b -> add -> *2``
    rather than a hardcoded value).
    """

    model = _InPlaceBuffer()
    x = torch.ones(3)
    trace = _capture(model, x)
    path = tmp_path / "inplace_buffer.tlspec"
    trace.save(path, level="runnable", include_weights=True)

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    with torch.no_grad():
        oracle = _InPlaceBuffer()(x.clone())
    assert (result.output - oracle).abs().max().item() < 1e-6

    changed = torch.full((3,), 7.0)
    changed_result = tl.load(path).run(inputs=changed.clone())
    assert changed_result.report.path_faithfulness is PathFaithfulness.VERIFIED
    with torch.no_grad():
        changed_oracle = _InPlaceBuffer()(changed.clone())
    assert (changed_result.output - changed_oracle).abs().max().item() < 1e-6
    assert torch.allclose(changed_result.output, torch.full((3,), 14.0))
