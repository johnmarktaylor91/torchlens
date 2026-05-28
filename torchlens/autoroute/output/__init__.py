"""Reserved output auto-routing namespace."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Raise for all reserved output registry access.

    Parameters
    ----------
    name:
        Requested output namespace attribute.

    Raises
    ------
    NotImplementedError
        Always raised until output auto-routing is implemented.
    """

    raise NotImplementedError("tl.autoroute.output is reserved for a future sprint")


__all__: list[str] = []
