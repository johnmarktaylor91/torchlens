"""Trace receptive-field functions reached from the records through ``_data_classes._doors.receptive_field``.

Bodies of ``Trace.receptive_fields`` and ``Trace.projective_fields`` live here
as ``f(trace, ...)`` functions; the record keeps one thin method per name with
its public docstring and forwards to these through the door.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ._table import build_rf_profile

if TYPE_CHECKING:
    from collections.abc import Collection

    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from ._types import ReceptiveFieldDirection, ReceptiveFieldProfile, ReceptiveFieldStatus

__all__: list[str] = []


def receptive_fields(  # noqa: PLR0913 - forwards the public Trace.receptive_fields knob surface one-to-one
    trace: Trace,
    level: Literal["op", "layer", "call", "module"] = "op",
    *,
    input_op: Op | None = None,
    statuses: Collection[ReceptiveFieldStatus] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    direction: ReceptiveFieldDirection | str = "receptive",
) -> ReceptiveFieldProfile:
    """Body of ``Trace.receptive_fields``; the record keeps the thin spelling and docstring."""

    return build_rf_profile(
        trace,
        level=level,
        input_op=input_op,
        statuses=statuses,
        sort_by=sort_by,
        ascending=ascending,
        direction=direction,
    )


def projective_fields(  # noqa: PLR0913 - forwards the public Trace.projective_fields knob surface one-to-one
    trace: Trace,
    level: Literal["op", "layer", "call", "module"] = "op",
    *,
    input_op: Op | None = None,
    statuses: Collection[ReceptiveFieldStatus] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
) -> ReceptiveFieldProfile:
    """Body of ``Trace.projective_fields``; the record keeps the thin spelling and docstring."""

    return trace.receptive_fields(
        level=level,
        input_op=input_op,
        statuses=statuses,
        sort_by=sort_by,
        ascending=ascending,
        direction="projective",
    )
