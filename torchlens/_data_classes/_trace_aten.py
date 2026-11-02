"""Public ATen-operation accessors for ``Trace`` (``CaptureOptions(aten_ops=True)``).

A leaf mixin like ``_trace_narrate``: it reads the trace's primitive-op
profile (``_primitive_op_profile``, materialized at postprocess step 0 when the
switch is on and restored from v8+ artifacts) and refuses typed when the
profile is absent, so an empty answer never masquerades as "nothing dispatched".
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
            "(the aten_ops option was off, the artifact was saved without it, "
            "or the trace is a fast re-run)"
        )
    return profile


class TraceAtenMixin:
    """Adds ``has_aten_ops``, ``aten_ops`` and ``aten_gaps`` to :class:`Trace`."""

    @property
    def has_aten_ops(self) -> bool:
        """Whether this trace carries recorded ATen operations.

        ``True`` after a capture with ``CaptureOptions(aten_ops=True)`` and after
        loading an artifact saved from one (a present-but-empty profile counts);
        ``False`` otherwise, including ``run(fast=True)`` results.
        """
        return getattr(self, "_primitive_op_profile", None) is not None

    @property
    def aten_ops(self) -> tuple[AtenOp, ...]:
        """All recorded ATen operations in dispatch order (forward, then backward).

        Raises
        ------
        AtenOpsUnavailableError
            If the capture recorded none; re-capture with ``aten_ops=True``.
        """
        profile = _require_aten_profile(self, what="Trace.aten_ops")
        return tuple(profile.primitive_ops)

    @property
    def aten_gaps(self) -> tuple[_ModePausedInteriorGap, ...]:
        """Disclosed stretches where ATen rows are missing by construction.

        Each gap names the capture phase, the dispatch sequence it spans and the
        owning op references of an interior where the dispatch mode was paused
        (for example inside a user-injected intervention). An empty tuple means
        the recorded rows are complete.

        Raises
        ------
        AtenOpsUnavailableError
            If the capture recorded no ATen operations.
        """
        profile = _require_aten_profile(self, what="Trace.aten_gaps")
        return tuple(profile.mode_paused_interior)
