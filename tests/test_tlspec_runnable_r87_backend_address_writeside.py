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

from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._utils._record_group_table import record_member
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(replayable=True, capture_container_structure=True, cache=False)


def _buffer_ops(trace: Any) -> list[Any]:
    """Return every materialized buffer op of ``trace``."""

    return [op for op in trace if record_member(op, "is_buffer", False)]


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


# Kept: the catch of its regression fault.
def test_inplace_buffer_read_and_write_nodes_agree_on_backend_address() -> None:
    """Both nodes of an in-place-mutated registered buffer report its address.

    Pre-r87 the write-side node reported ``backend_address=None`` while the
    read-side node reported ``'b'``. Both must now report ``'b'`` -- and a read
    AND a write node must genuinely be present (parents empty vs non-empty), so
    the test cannot be satisfied by a trace that lost the split.
    """

    trace = _capture(_InPlaceBuffer(), torch.ones(3))
    b_ops = [op for op in _buffer_ops(trace) if op.buffer_address == "b"]

    assert len(b_ops) == 2, f"expected a read + write node for 'b', got {b_ops}"
    read_nodes = [op for op in b_ops if not op.parents]
    write_nodes = [op for op in b_ops if op.parents]
    assert len(read_nodes) == 1 and len(write_nodes) == 1, "need one read and one write node"

    for op in b_ops:
        assert op.buffer_address == "b", "display address unchanged on both nodes"
        assert op.backend_address == "b", (
            f"{op.layer_label} (parents={op.parents}) reports "
            f"backend_address={op.backend_address!r}, expected 'b'"
        )


# --------------------------------------------------------------------------- #
# THE HARD CONSTRAINT -- a non-registered collision stays None (C2 closed).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# VERDICT NEUTRALITY -- the fix moves no runnable verdict.
# --------------------------------------------------------------------------- #
