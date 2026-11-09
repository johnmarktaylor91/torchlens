"""Admission guard of the label join behind ``tl.debug.compare`` and ``first_divergence``.

Op pairing (``align="label"``, or ``"auto"`` on one backend) matches the ops of two traces by label
or graph structure. That pairing only means
something when the two runs were captured by the same backend: labels and op
order do not correspond across backends (a PyTorch trace and an MLX trace of the
same model number and spell their ops differently), so a label join would report
a divergence that is not there, or value-compare two unrelated ops whose labels
coincide by accident. The guard refuses such a pair before any value is read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchlens._runnable import refuse_poisoned_trace

from ..backends._errors import BackendUnsupportedError

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace


def refuse_uncomparable_pair(trace_a: Trace, trace_b: Trace) -> None:
    """Refuse a pair of traces that op-by-op comparison cannot answer for.

    A trace refused by validation (poisoned) refuses first, as before. Then two
    traces whose ``trace.provenance.backend`` differ refuse with
    :class:`torchlens.errors.BackendUnsupportedError` (code
    ``backend_unsupported``), before any payload is read.

    Parameters
    ----------
    trace_a:
        First completed TorchLens trace.
    trace_b:
        Second completed TorchLens trace.

    Raises
    ------
    BackendUnsupportedError
        When the two traces were captured by different backends.
    """

    refuse_poisoned_trace(trace_a, "faithful comparison")
    refuse_poisoned_trace(trace_b, "faithful comparison")
    backend_a = str(trace_a._provenance_backend)
    backend_b = str(trace_b._provenance_backend)
    if backend_a == backend_b:
        return
    raise BackendUnsupportedError(
        f"cannot compare a {backend_a!r} trace with a {backend_b!r} trace op by op: "
        "operation labels and operation order do not correspond across backends, so "
        "joining the two runs by label would report a divergence that is not there. "
        "align='label' compares two runs captured on the same backend; across backends "
        "tl.debug.compare and tl.debug.first_divergence pair module calls instead",
        code="backend_unsupported",
        remedy=(
            "pass align='auto' (the default) or align='structure' to pair module calls, "
            f"or compare two traces captured on the same backend, for example both with "
            f"backend={backend_a!r}"
        ),
        backend_a=backend_a,
        backend_b=backend_b,
    )
