"""The record of one plain module attribute a traced forward wrote."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

__all__ = ["AttributeWrite", "AttributeWriteKind"]

#: How the forward changed the attribute: ``"added"`` (absent before the
#: forward), ``"replaced"`` (rebound to another object, or the same container
#: or tensor changed in place) or ``"removed"`` (deleted by the forward).
AttributeWriteKind = Literal["added", "replaced", "removed"]


@dataclass(frozen=True, slots=True)
class AttributeWrite:
    """One plain instance attribute a traced forward wrote on a module.

    A default ``tl.trace`` is a pure observer: it puts every plain attribute
    back as it was before the forward, so a value the forward stashes on a
    module (``self.last_weights = w``) is read from the trace instead, at
    ``trace.modules[address].attribute_writes[name].value``.
    ``CaptureOptions(carry_state=True)`` keeps the writes on the module and
    records them the same way.

    Parameters
    ----------
    kind:
        ``"added"``, ``"replaced"`` or ``"removed"``.
    value:
        What the forward left in the attribute: the object itself when the
        forward bound a new object (the record keeps it alive as long as the
        trace, so a large stashed tensor costs its full memory), a copy when
        the forward changed the original object in place (builtin
        ``list``/``tuple``/``dict``/``set`` containers are copied recursively,
        a tensor that existed before the forward is cloned whatever its size,
        a tensor the forward created is kept by reference), and ``None`` when
        the forward removed the attribute.

    Notes
    -----
    Session-time only: a saved and reloaded trace has no record
    (``Module.attribute_writes`` is ``None``). Objects the capture tracks by
    identity (custom objects, containers over 128 items or nested deeper than
    64 levels, containers holding unsupported values) are recorded only when
    the forward rebinds or removes them; a change inside such an object is
    neither recorded nor undone. Rebinding to an equal immutable scalar is not
    a write. A capture that re-ran the forward (escape rescue) records the net
    effect of both runs.
    """

    kind: AttributeWriteKind
    value: Any
