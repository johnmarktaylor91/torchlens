"""Min/max order failure message for the distance invariants."""

from __future__ import annotations

from typing import Any


def _distance_order_message(label: str, graph: Any, *, toward_output: bool) -> str | None:
    """Return the ``min > max`` failure message for one stored distance pair.

    Parameters
    ----------
    label:
        Layer label reported in the message.
    graph:
        The layer's graph group, holding the stored min/max distances.
    toward_output:
        ``True`` for the distance to the outputs, ``False`` for the distance
        from the inputs.

    Returns
    -------
    str | None
        The failure message when both distances are set and the minimum
        exceeds the maximum, otherwise ``None``.
    """

    if toward_output:
        field = "distance_to_output"
        low, high = graph.min_distance_to_output, graph.max_distance_to_output
    else:
        field = "distance_from_input"
        low, high = graph.min_distance_from_input, graph.max_distance_from_input
    if low is None or high is None or low <= high:
        return None
    return f"Layer '{label}': min_{field}={low} > max={high}"
