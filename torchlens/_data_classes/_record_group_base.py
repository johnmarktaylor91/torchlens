"""Base classes of the Op and Layer concern groups (``op.cost``, ``layer.call``, ...).

A leaf module: it imports only the constants and the member table, so every
group module can subclass these without joining the data-class import cycle.
"""

from __future__ import annotations

from typing import Any

from .._constants import LAYER_PASS_LOG_FIELD_ORDER
from .._utils._record_group_table import OP_MEMBER_MOVES

#: Per-pass fields a multi-pass Layer refuses as ambiguous instead of delegating.
_LAYER_DELEGATED_PASS_FIELDS = frozenset((*LAYER_PASS_LOG_FIELD_ORDER, "out_ref", "grad_ref"))


class _RecordGroup:
    """Base for the read-only concern groups of one ``Op`` or ``Layer``."""

    __slots__ = ("_record",)

    def __init__(self, record: Any) -> None:
        """Bind the group to ``record``."""

        self._record = record

    def __repr__(self) -> str:
        """Name the group and its record."""

        record = self._record
        label = getattr(record, "label", None)
        return f"<{type(self).__name__} of {type(record).__name__} {label!r}>"

    def __dir__(self) -> list[str]:
        """List the group's public members."""

        return sorted(n for n in dir(type(self)) if not n.startswith("_"))


class _LayerGroup(_RecordGroup):
    """Base for a Layer group: Op members of the same group read through the layer's pass.

    A Layer used to delegate an unknown member to its single pass
    (``layer.conditional_branch_stack`` read ``ops[0]``). The group keeps that
    contract: ``layer.control_flow.branch_stack`` reads the single pass's
    ``op.control_flow.branch_stack``, and a multi-pass layer refuses a per-pass
    member with the same typed ambiguity error the flat read raised.
    """

    __slots__ = ()
    _GROUP = ""

    def __getattr__(self, name: str) -> Any:
        """Read an Op member of this group through the layer's single pass."""

        if name.startswith("_"):
            raise AttributeError(name)
        layer = self._record
        ops = layer.__dict__.get("ops")
        old = _op_member_by_leaf(self._GROUP).get(name)
        if old is not None and ops and len(ops) == 1:
            return getattr(getattr(ops[0], self._GROUP), name)
        if old is not None and ops and len(ops) > 1 and old in _LAYER_DELEGATED_PASS_FIELDS:
            return layer._single_pass_or_error(old)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def _op_member_by_leaf(group: str) -> dict[str, str]:
    """Return ``leaf -> former Op name`` for the Op members of ``group``."""

    return {leaf: old for old, row_group, leaf in OP_MEMBER_MOVES if row_group == group}
