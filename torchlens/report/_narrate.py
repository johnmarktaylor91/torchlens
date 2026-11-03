"""Trace post-hoc narration function (record rebuild destination).

The body of ``Trace.narrate`` lives here; the method in ``_data_classes/trace.py``
reaches it through the ``_doors.report()`` door so the record package stays a leaf
with respect to the narration renderer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..debug._snoop._narrate import narrate_trace

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def narrate(trace: Trace, last: int | None = None, *, select: Any = None) -> str:
    """Body of ``Trace.narrate``; the method docstring is the contract."""
    return narrate_trace(trace, last=last, select=select)


__all__ = ["narrate"]
