"""Trace-graph helpers for the JAX backend's validation oracles.

Saved-output lookup on the materialized trace graph (single, multi-output and
region-boundary ops), the parent-perturbation oracle that tells a genuine data
parent from a label-only edge, the data/control parent-label readers and the
raw-to-final relabeling of parent arg positions, plus the small pytree and
payload-closeness utilities they share. Pure functions over captured records;
the backend module imports them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..._ir.container import DictKey, TupleIndex
from ..._ir.events import is_control_edge_use
from ..._utils._record_group_table import record_member
from .._validation_shared import float_replay_tolerances
from ._perturbation_probes import _all_bool_outputs, _candidate_stream
from .jaxpr import JaxEquationCapture, replay_equation


def _saved_op_outputs(
    op: Any,
    expected_count: int,
    *,
    hidden_outputs_by_label: Mapping[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Return saved output payloads from a materialized JAX operation.

    Parameters
    ----------
    op
        Materialized TorchLens operation.
    expected_count
        Number of primitive outputs expected by replay.
    hidden_outputs_by_label
        Runtime-only replay outputs keyed by raw and final op labels.

    Returns
    -------
    tuple[Any, ...]
        Saved output payloads.
    """

    output = _saved_single_output(op, hidden_outputs_by_label)
    if expected_count == 1:
        return (output,)
    if not isinstance(output, tuple) or len(output) != expected_count:
        raise ValueError("JAX multi-output payload does not match primitive replay.")
    return output


def _saved_region_boundary_outputs(
    op: Any,
    expected_count: int,
    *,
    hidden_outputs_by_label: Mapping[str, tuple[Any, ...]],
) -> tuple[Any, ...]:
    """Return the saved output tuple from a JAX region boundary op.

    Parameters
    ----------
    op
        Materialized region boundary op.
    expected_count
        Number of boundary outputs expected by the runtime capture.
    hidden_outputs_by_label
        Runtime-only replay outputs keyed by raw and final op labels.

    Returns
    -------
    tuple[Any, ...]
        Saved boundary output tuple.
    """

    output = _saved_single_output(op, hidden_outputs_by_label)
    if not isinstance(output, tuple) or len(output) != expected_count:
        raise ValueError("JAX region boundary payload does not match output arity.")
    return output


def _saved_single_output(
    op: Any,
    hidden_outputs_by_label: Mapping[str, tuple[Any, ...]] | None = None,
) -> Any:
    """Return one operation's saved payload, failing when it was dropped.

    Parameters
    ----------
    op
        Materialized TorchLens operation.
    hidden_outputs_by_label
        Runtime-only replay outputs keyed by raw and final op labels.

    Returns
    -------
    Any
        Saved output payload.
    """

    # A dropped public payload (filtered ``save=``) or a cleared ``out`` (transform
    # without ``save_raw_activations``) replays from the hidden raw copy.
    if not getattr(op, "has_saved_activation", False) or op.out is None:
        for label in (getattr(op, "_label_raw", None), getattr(op, "label", None)):
            if (
                isinstance(label, str)
                and hidden_outputs_by_label
                and label in hidden_outputs_by_label
            ):
                outputs = hidden_outputs_by_label[label]
                return outputs[0] if len(outputs) == 1 else outputs
        raise ValueError("JAX validation requires every equation payload to be saved.")
    return op.out


_NO_RECORDED_INPUT: Any = object()


def _saved_parent_input(
    parent_op: Any,
    hidden_outputs_by_label: Mapping[str, tuple[Any, ...]] | None,
    recorded_input: Any = _NO_RECORDED_INPUT,
) -> Any:
    """Return the saved parent payload a child consumed at one argument position.

    A multi-result equation (``split``) materializes as one op whose payload is
    the tuple of its outputs, and capture wired each child to the one output it
    read, by value identity. Replay must hand the child that element, never the
    whole tuple: the element identical to the child's recorded input, else the
    single saved element equal to it. A saved tuple with no matching element
    (a dropped or corrupted payload) raises, so the replay fails closed.

    Parameters
    ----------
    parent_op
        Materialized parent operation.
    hidden_outputs_by_label
        Runtime-only replay outputs keyed by raw and final op labels.
    recorded_input
        The child's captured input value at this position; omitted for a
        single-result op read whole (a region projection).

    Returns
    -------
    Any
        The saved payload (one element of a multi-result parent).
    """

    saved = _saved_single_output(parent_op, hidden_outputs_by_label)
    if recorded_input is _NO_RECORDED_INPUT or not isinstance(saved, tuple | list):
        return saved
    for element in saved:
        if element is recorded_input:
            return element
    for element in saved:
        if _same_array(element, recorded_input):
            return element
    raise ValueError("JAX multi-output parent saved no output matching the child's input.")


def _same_array(left: Any, right: Any) -> bool:
    """Return whether two arrays agree exactly in shape, dtype and values (NaN-equal)."""

    import jax.numpy as jnp

    if getattr(left, "shape", None) != getattr(right, "shape", None):
        return False
    if getattr(left, "dtype", None) != getattr(right, "dtype", None):
        return False
    try:
        return bool(jnp.array_equal(left, right, equal_nan=True))
    except TypeError:  # key arrays and other non-numeric payloads have no value equality
        return False


def _parent_perturbations_change_output(
    capture: JaxEquationCapture,
    op: Any,
    inputs: tuple[Any, ...],
    saved_outputs: tuple[Any, ...],
) -> bool:
    """Return whether a recorded parent perturbation affects child replay output.

    Parameters
    ----------
    capture
        Captured JAX equation metadata.
    op
        Materialized TorchLens operation for the equation.
    inputs
        Replay inputs derived from trace graph parents.
    saved_outputs
        Saved child output payloads.

    Returns
    -------
    bool
        True when at least one graph parent perturbation changes the child output.
    """

    import jax

    if capture.kind in {"cond_decision", "while_decision"}:
        return True
    graph_positions = _data_parent_arg_positions(op)
    if not graph_positions:
        return True
    positions_by_parent: dict[str, list[int]] = {}
    for position, parent_label in graph_positions.items():
        positions_by_parent.setdefault(parent_label, []).append(position)
    bool_child = _all_bool_outputs(saved_outputs)
    for positions in positions_by_parent.values():
        for candidate in _candidate_stream(capture, inputs, positions, bool_child=bool_child):
            perturbed_inputs = list(inputs)
            for position in positions:
                perturbed_inputs[position] = candidate
            try:
                perturbed_outputs = replay_equation(capture, perturbed_inputs)
            except ValueError:
                if capture.kind in {"cond_decision", "while_decision"}:
                    return True
                raise
            if not jax.tree.all(jax.tree.map(_values_close, perturbed_outputs, saved_outputs)):
                return True
    return False


def _control_parent_labels(op: Any) -> set[str]:
    """Return graph parents that express control dependencies.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    set[str]
        Parent labels whose edge-use metadata is marked ``"control"``.
    """

    return _parent_labels_by_control_class(op)[0]


def _data_parent_labels(op: Any) -> set[str]:
    """Return graph parents that participate in value replay.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    set[str]
        Parent labels excluding control-only dependencies.
    """

    return _parent_labels_by_control_class(op)[1]


def _data_parent_arg_positions(op: Any) -> dict[int, str]:
    """Return positional parent-argument mappings for data parents only.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    dict[int, str]
        Positional replay inputs keyed by primitive argument position.
    """

    data_labels = _data_parent_labels(op)
    graph_positions = record_member(op, "parent_arg_positions", {}).get("args", {})
    return {
        position: parent_label
        for position, parent_label in graph_positions.items()
        if isinstance(position, int)
        and isinstance(parent_label, str)
        and parent_label in data_labels
    }


def _ordered_jax_data_parent_labels(op: Any, raw_label_set: set[str]) -> tuple[str, ...]:
    """Return raw data-parent labels in recorded parent order.

    Parameters
    ----------
    op
        Materialized JAX operation log.
    raw_label_set
        Raw labels present in the materialized trace.

    Returns
    -------
    tuple[str, ...]
        Deduplicated data-parent labels that belong to the raw graph.
    """

    data_parent_set = _data_parent_labels(op)
    ordered: list[str] = []
    seen: set[str] = set()
    for parent in getattr(op, "parents", ()):
        if parent in data_parent_set and parent in raw_label_set and parent not in seen:
            ordered.append(parent)
            seen.add(parent)
    return tuple(ordered)


def _relabel_jax_parent_arg_positions(
    parent_arg_positions: Mapping[str, Mapping[Any, Any]],
    raw_to_final: Mapping[str, str],
) -> dict[str, dict[Any, Any]]:
    """Return parent-argument positions with raw labels replaced.

    Parameters
    ----------
    parent_arg_positions
        Existing parent-position metadata.
    raw_to_final
        Mapping from raw op labels to final op labels.

    Returns
    -------
    dict[str, dict[Any, Any]]
        Relabeled parent-position metadata.
    """

    return {
        arg_kind: {
            position: raw_to_final.get(parent_label, parent_label)
            if isinstance(parent_label, str)
            else parent_label
            for position, parent_label in positions.items()
        }
        for arg_kind, positions in parent_arg_positions.items()
    }


def _relabel_jax_edge_use(edge: Any, raw_to_final: Mapping[str, str]) -> Any:
    """Return an edge-use record with raw endpoint labels replaced.

    Parameters
    ----------
    edge
        Edge-use record or legacy tuple.
    raw_to_final
        Mapping from raw op labels to final op labels.

    Returns
    -------
    Any
        Relabeled edge-use record.
    """

    parent_label = getattr(edge, "parent_label", None)
    child_label = getattr(edge, "child_label", None)
    if isinstance(parent_label, str) and isinstance(child_label, str):
        return replace(
            edge,
            parent_label=raw_to_final.get(parent_label, parent_label),
            child_label=raw_to_final.get(child_label, child_label),
        )
    if isinstance(edge, tuple) and len(edge) >= 3 and isinstance(edge[0], str):
        return (raw_to_final.get(edge[0], edge[0]), *edge[1:])
    return edge


def _parent_labels_by_control_class(op: Any) -> tuple[set[str], set[str]]:
    """Split operation parents into control and value-replay sets.

    Parameters
    ----------
    op
        Materialized TorchLens operation.

    Returns
    -------
    tuple[set[str], set[str]]
        ``(control_parent_labels, data_parent_labels)``.
    """

    parents = {label for label in getattr(op, "parents", ()) if isinstance(label, str)}
    control = {
        parent_label
        for edge in getattr(op, "_edge_uses", ())
        if (parent_label := _edge_parent_label(edge)) is not None and is_control_edge_use(edge)
    }
    return control & parents, parents - control


def _edge_parent_label(edge: Any) -> str | None:
    """Return the parent label from an edge-use record or legacy tuple.

    Parameters
    ----------
    edge
        Edge-use record or tuple.

    Returns
    -------
    str | None
        Parent label when present.
    """

    parent_label = getattr(edge, "parent_label", None)
    if isinstance(parent_label, str):
        return parent_label
    if isinstance(edge, tuple) and edge and isinstance(edge[0], str):
        return edge[0]
    return None


def _values_close(left: Any, right: Any) -> bool:
    """Return whether two JAX values are numerically close.

    Parameters
    ----------
    left
        Left value.
    right
        Right value.

    Returns
    -------
    bool
        True when values are close.
    """

    import jax.numpy as jnp

    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return False
    if left_array.dtype == jnp.bool_ or jnp.issubdtype(left_array.dtype, jnp.integer):
        return bool(jnp.array_equal(left_array, right_array, equal_nan=True))
    if jnp.issubdtype(left_array.dtype, jnp.floating) or jnp.issubdtype(
        left_array.dtype, jnp.complexfloating
    ):
        # Per-dtype ULP-derived bands (ported paddle/mlx validation-oracle
        # derivation); ``jnp.finfo`` reports component precision for complex
        # and covers the extended ml_dtypes floats (bfloat16, fp8).
        rtol, atol = float_replay_tolerances(jnp.finfo(left_array.dtype))
        return bool(jnp.allclose(left_array, right_array, rtol=rtol, atol=atol, equal_nan=True))
    return bool(jnp.array_equal(left_array, right_array))


def _path_to_string(path: Sequence[Any]) -> str:
    """Convert a JAX pytree path to a stable dotted string.

    Parameters
    ----------
    path
        JAX pytree path entries.

    Returns
    -------
    str
        Dotted path string.
    """

    if not path:
        return "root"
    parts: list[str] = []
    for entry in path:
        name = getattr(entry, "name", None)
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        if name is not None:
            parts.append(str(name))
        elif key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        else:
            parts.append(str(entry).strip("[]'"))
    return ".".join(parts)


def _jax_dict_keys(value: Mapping[Any, Any]) -> tuple[Any, ...]:
    """Return dict keys in JAX builtin pytree traversal order.

    Parameters
    ----------
    value
        Builtin dict output.

    Returns
    -------
    tuple[Any, ...]
        Keys sorted when possible, matching JAX's dict pytree order.
    """

    try:
        return tuple(sorted(value.keys()))
    except TypeError:
        return tuple(value.keys())


def _path_to_components(path: Sequence[Any]) -> tuple[object, ...]:
    """Convert a JAX pytree path to TorchLens output-path components.

    Parameters
    ----------
    path
        JAX pytree path entries.

    Returns
    -------
    tuple[object, ...]
        Backend-neutral container path components.
    """

    components: list[object] = []
    for entry in path:
        name = getattr(entry, "name", None)
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        if name is not None:
            components.append(str(name))
        elif key is not None:
            components.append(DictKey(key))
        elif idx is not None:
            components.append(TupleIndex(int(idx)))
        else:
            components.append(str(entry).strip("[]'"))
    return tuple(components)
