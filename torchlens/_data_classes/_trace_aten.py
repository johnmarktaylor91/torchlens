"""Public ATen-operation accessors for ``Trace`` (``CaptureOptions(aten_ops=True)``).

Function bodies behind the ``Trace.has_aten_ops``, ``Trace.aten_ops`` and
``Trace.aten_gaps`` properties. They read the trace's primitive-op profile
(``_primitive_op_profile``, materialized at postprocess step 0 when the switch is
on and restored from v8+ artifacts) and refuse typed when the profile is absent,
so an empty answer never masquerades as "nothing dispatched".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._errors import AtenOpsUnavailableError

if TYPE_CHECKING:
    from .aten_op import AtenOp, _ModePausedInteriorGap, _PrimitiveOpProfile


def _require_aten_profile(record: Any, *, what: str) -> _PrimitiveOpProfile:
    """Return the trace's primitive-op profile or refuse typed.

    Parameters
    ----------
    record
        Trace-like object owning ``_primitive_op_profile``.
    what
        Accessor name for the refusal message.

    Returns
    -------
    _PrimitiveOpProfile
        The materialized profile (present and possibly empty when recording was on).

    Raises
    ------
    AtenOpsUnavailableError
        If the capture recorded no ATen operations or the loaded artifact carries none.
    """

    profile = getattr(record, "_primitive_op_profile", None)
    if profile is None:
        raise AtenOpsUnavailableError(
            f"{what} needs ATen operation rows, but this capture recorded none "
            "(the aten_ops option was off, or the artifact was saved without it)",
            code="aten_ops_unavailable",
        )
    return profile


def has_aten_ops(record: Any) -> bool:
    """Return whether ``record`` carries a primitive-op profile (possibly empty)."""

    return getattr(record, "_primitive_op_profile", None) is not None


def aten_ops(record: Any) -> tuple[AtenOp, ...]:
    """Return every recorded ATen operation of ``record`` in dispatch order, or refuse typed."""

    return tuple(_require_aten_profile(record, what="Trace.aten_ops").primitive_ops)


def aten_gaps(record: Any) -> tuple[_ModePausedInteriorGap, ...]:
    """Return the disclosed ATen-row gaps of ``record``, or refuse typed."""

    return tuple(_require_aten_profile(record, what="Trace.aten_gaps").mode_paused_interior)
