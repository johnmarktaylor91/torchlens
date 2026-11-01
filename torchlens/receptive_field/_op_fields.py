"""Receptive and projective field views for ``Op``.

Reached from ``Op`` through ``_data_classes._doors.receptive_field()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _engine
from ._view import ReceptiveFieldView

if TYPE_CHECKING:
    from .._data_classes.op import Op


__all__ = [
    "receptive_field",
    "projective_field",
]


def receptive_field(op: Op) -> ReceptiveFieldView:
    """Return the lazy receptive-field query view for this Op."""

    trace = op.source_trace
    solution = _engine.solve(trace)
    cache = op._slot("_receptive_field_cache")
    if cache is not None and cache._solution is not solution:
        for member in trace._layer_list:
            if member._slot("_receptive_field_cache") is not None:
                object.__setattr__(
                    member,
                    "_receptive_field_cache",
                    ReceptiveFieldView(member, solution),
                )
        cache = op._slot("_receptive_field_cache")
    if cache is None:
        cache = ReceptiveFieldView(op, solution)
        object.__setattr__(op, "_receptive_field_cache", cache)
    return cache


def projective_field(op: Op) -> ReceptiveFieldView:
    """Return the lazy source-anchored projective-field query view."""

    cache = op._slot("_projective_field_cache")
    current = ReceptiveFieldView.projective(op)
    if cache is None or cache._solution is not current._solution:
        cache = current
        object.__setattr__(op, "_projective_field_cache", cache)
    return cache
