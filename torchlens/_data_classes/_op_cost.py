"""Cost derivations for ``Op``: compute record, MACs, FLOPs, arithmetic intensity.

Logic bodies live here as functions taking the record.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._quantities import Bytes, Flops, Macs, as_macs
from . import _doors
from ._op_out_read import _refuse_read_without_trace

if TYPE_CHECKING:
    from .op import Op


__all__ = [
    "compute_record",
    "macs_forward",
    "macs_backward",
    "flops_total",
    "macs_total",
    "arithmetic_intensity",
]


def compute_record(op: Op) -> Any | None:
    """Two-term compute record (fma_macs / other_flops), derived on read.

    DOCUMENTED-UNSTABLE spelling (A07 numbers-truth lane; the C02 FactCore
    substrate consumes this). ``None`` for boundary pseudo-rows and ops
    with unknown FLOPs. See ``torchlens._capture.flops.ComputeRecord``.
    """

    from .._capture.compute_record import derive_compute_record_for_op

    return derive_compute_record_for_op(op)


def macs_forward(op: Op) -> Macs | None:
    """TRUE forward multiply-accumulate count, never ``flops // 2``.

    A ReLU has ZERO MACs; a biased Linear's bias adds are not MACs. Ops
    whose exact MAC split cannot be derived (user-registered rules,
    unresolvable attention argument shapes) return ``None`` and are named
    in the summary's MAC coverage disclosure.
    """

    record = op.compute_record
    if record is None or record.fma_macs is None:
        return None
    return as_macs(record.fma_macs)


def macs_backward(op: Op) -> Macs | None:
    """Backward MACs are not derivable and always ``None``.

    Backward FLOPs are multiplier ESTIMATES on forward FLOPs; no analytic
    backward op record exists to split them, and a relabeled
    ``flops // 2`` is exactly the fake-MACs disease this surface removed.
    """

    return None


def flops_total(op: Op) -> Flops:
    """Approximate total FLOPs for this Op.

    Returns
    -------
    Flops
        Forward plus backward FLOPs, treating unknown halves as zero.
    """

    return Flops((op.flops_forward or 0) + (op.flops_backward or 0))


def macs_total(op: Op) -> Macs | None:
    """Total MACs are not derivable and always ``None``.

    The backward component is a multiplier estimate with no MAC split
    (see ``macs_backward``); a forward-only figure under a "total" name
    would silently under-claim instead.
    """

    return None


def theoretical_bytes(op: Op, side: int) -> Bytes | None:
    """Theoretical ideal read-once (``side`` 0) or write-once (1) traffic for this Op.

    Refuses typed on an Op whose Trace was garbage-collected: the traffic model
    reads the Op's graph parents through the Trace, so ``None`` there would
    read as "insufficient metadata".
    """

    _refuse_read_without_trace(op)
    return _doors.debug()._cost.theoretical_op_bytes(op)[side]


def arithmetic_intensity(op: Op) -> float | None:
    """Return theoretical forward FLOPs per ideal traffic byte.

    The value is ``flops_forward / (bytes_read + bytes_written)`` using the
    theoretical ideal read-once/write-once model. It is not measured roofline
    intensity and can diverge under kernel fusion, caching, or real memory
    transactions. Missing FLOPs/traffic and zero-traffic logical views return
    ``None``.

    Returns
    -------
    float | None
        Theoretical forward arithmetic intensity in FLOPs per byte.
    """

    bytes_read = op.bytes_read
    bytes_written = op.bytes_written
    if op.flops_forward is None or bytes_read is None or bytes_written is None:
        return None
    total_bytes = int(bytes_read) + int(bytes_written)
    if total_bytes == 0:
        return None
    return float(op.flops_forward) / total_bytes
