"""Private argument forwarding for the Trace visualization surface."""

from collections.abc import Callable
from typing import Any


def _forward_draw(draw: Callable[..., Any], arguments: dict[str, Any]) -> Any:
    """Forward the public draw arguments to the drawing implementation.

    Parameters
    ----------
    draw : callable
        Drawing implementation selected by the record's visualization door.
    arguments : dict
        Fresh local argument mapping from the public method.
    """
    trace = arguments.pop("self")
    path = arguments.pop("path")
    return draw(trace, path, **arguments)


def _forward_summary(summary: Callable[..., str], arguments: dict[str, Any]) -> str:
    """Forward summary arguments, retaining duplicate-keyword refusal.

    Parameters
    ----------
    summary : callable
        Summary implementation selected by the record's report door.
    arguments : dict
        Fresh local argument mapping, including unsupported keywords.
    """
    trace = arguments.pop("self")
    unsupported = arguments.pop("unsupported")
    return summary(trace, **arguments, **unsupported)
