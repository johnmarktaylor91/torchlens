"""Distance-bound invariants for one layer: min <= max, and zero at inputs and outputs."""

from __future__ import annotations

from typing import Any


def _check_distance_order(name: str, lpl: Any, label: str) -> None:
    """Check one layer's min distances do not exceed its max distances.

    Parameters
    ----------
    name:
        Invariant check name for raised errors.
    lpl:
        Op record being checked.
    label:
        The op's layer label.
    """

    # min <= max for input distances
    if (
        lpl.graph.min_distance_from_input is not None
        and lpl.graph.max_distance_from_input is not None
    ):
        if lpl.graph.min_distance_from_input > lpl.graph.max_distance_from_input:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': min_distance_from_input="
                f"{lpl.graph.min_distance_from_input} > max={lpl.graph.max_distance_from_input}",
            )

    # min <= max for output distances
    if (
        lpl.graph.min_distance_to_output is not None
        and lpl.graph.max_distance_to_output is not None
    ):
        if lpl.graph.min_distance_to_output > lpl.graph.max_distance_to_output:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': min_distance_to_output="
                f"{lpl.graph.min_distance_to_output} > max={lpl.graph.max_distance_to_output}",
            )


def _check_terminal_distances(
    name: str,
    lpl: Any,
    label: str,
    input_set: set[str],
    output_set: set[str],
) -> None:
    """Check input layers sit at input distance 0 and output layers at output distance 0.

    Parameters
    ----------
    name:
        Invariant check name for raised errors.
    lpl:
        Op record being checked.
    label:
        The op's layer label.
    input_set:
        The trace's input layer labels.
    output_set:
        The trace's output layer labels.
    """

    # Input layers: distance from input == 0
    if label in input_set:
        if lpl.graph.min_distance_from_input != 0 or lpl.graph.max_distance_from_input != 0:
            raise MetadataInvariantError(
                name,
                f"Input layer '{label}': distance_from_input should be 0, got "
                f"min={lpl.graph.min_distance_from_input}, max={lpl.graph.max_distance_from_input}",
            )

    # Output layers: distance from output == 0
    if label in output_set:
        if lpl.graph.min_distance_to_output != 0 or lpl.graph.max_distance_to_output != 0:
            raise MetadataInvariantError(
                name,
                f"Output layer '{label}': distance_from_output should be 0, got "
                f"min={lpl.graph.min_distance_to_output}, max={lpl.graph.max_distance_to_output}",
            )


from ._invariants import MetadataInvariantError  # noqa: E402
