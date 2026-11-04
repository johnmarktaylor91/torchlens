"""Per-op ATen-operation accessors for ``Op`` (``aten_ops`` and the ``gpu_kernels`` view)."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from ._trace_aten import _require_aten_profile

if TYPE_CHECKING:
    from .._kernel_telemetry import KernelLaunch
    from .aten_op import AtenOp

#: The detachable kernel-telemetry lane. ``gpu_kernels`` resolves it by name on
#: read, so ``Op`` always carries the view while neither this layer nor the
#: package imports the lane (``torchlens/_kernel_telemetry.py``).
_KERNEL_TELEMETRY_MODULE = "torchlens._kernel_telemetry"


class OpAtenMixin:
    """Adds ``aten_ops`` and ``gpu_kernels`` to :class:`Op`."""

    # Empty: Op is slotted (``_core``/``_row``); a slotless base would give every op a __dict__.
    __slots__ = ()

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

    @property
    def gpu_kernels(self) -> tuple[KernelLaunch, ...]:
        """The GPU kernel launches this op's ATen calls made, deduplicated across those calls.

        The union of ``AtenOp.gpu_kernels`` over the rows in ``aten_ops``: kernels
        matched to each ATen call by the profiler's correlation id, each launch
        listed once. When kernel telemetry was off for the trace, the value is one
        row with ``attribution_status="unavailable"``, never an empty tuple.
        Reading it needs the optional kernel-telemetry adapter.

        Returns
        -------
        tuple[KernelLaunch, ...]
            The matched kernel launches, or one ``unavailable`` row.
        """

        telemetry = importlib.import_module(_KERNEL_TELEMETRY_MODULE)
        return telemetry._op_gpu_kernels(self)
