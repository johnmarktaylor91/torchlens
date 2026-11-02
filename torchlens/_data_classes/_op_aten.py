"""Per-op ATen-operation accessor for ``Op`` (``CaptureOptions(aten_ops=True)``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._trace_aten import _require_aten_profile

if TYPE_CHECKING:
    from .aten_op import AtenOp


class OpAtenMixin:
    """Adds ``aten_ops`` to :class:`Op`."""

    @property
    def aten_ops(self) -> tuple[AtenOp, ...]:
        """The ATen operations this op's torch call dispatched to, in dispatch order.

        Rows are matched by the op's final label in their resolved owner
        references, so an op whose call dispatched nothing observable (or whose
        interior fell inside a disclosed gap, see ``Trace.aten_gaps``) yields an
        empty tuple while the trace still ``has_aten_ops``.

        Raises
        ------
        AtenOpsUnavailableError
            If the owning trace recorded no ATen operations.
        """
        record: Any = self
        profile = _require_aten_profile(record.source_trace, what="Op.aten_ops")
        label = record.layer_label
        return tuple(
            row
            for row in profile.primitive_ops
            if any(ref.op_label == label for ref in row.parent_op_refs)
        )
