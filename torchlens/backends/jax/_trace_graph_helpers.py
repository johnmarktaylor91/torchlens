"""Trace-graph helpers for the JAX preview backend's validation oracles.

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
from .._validation_shared import float_replay_tolerances
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

    if not getattr(op, "has_saved_activation", False):
        for label in (getattr(op, "_label_raw", None), getattr(op, "label", None)):
            if (
                isinstance(label, str)
                and hidden_outputs_by_label
                and label in hidden_outputs_by_label
            ):
                outputs = hidden_outputs_by_label[label]
                return outputs[0] if len(outputs) == 1 else outputs
        raise ValueError("JAX validation requires every equation payload to be saved.")
    output = op.out
    if output is None:
        raise ValueError("JAX validation found a missing saved payload.")
    return output


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
        first_position = positions[0]
        candidates = _perturb_candidates(inputs[first_position])
        if bool_child:
            candidates = (*candidates, *_bool_child_substitutes(capture, inputs, positions))
        for candidate in candidates:
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


#: Elementwise comparison primitives whose bool output admits the comparand
#: straddle (torch parity: ``validation/_influence_probes.py``).
_COMPARISON_PRIMITIVES = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})


def _all_bool_outputs(saved_outputs: tuple[Any, ...]) -> bool:
    """Return whether every saved output of an equation is a bool array."""

    import jax.numpy as jnp

    return bool(saved_outputs) and all(
        getattr(output, "dtype", None) == jnp.bool_ for output in saved_outputs
    )


def _bool_child_substitutes(
    capture: JaxEquationCapture, inputs: Sequence[Any], positions: Sequence[int]
) -> tuple[Any, ...]:
    """Return threshold-crossing substitutes for one parent of a bool-output equation.

    A bool child is a threshold op, so the additive steps of
    :func:`_perturb_candidates` routinely fail to cross it (``mask != 0`` on an
    all-nonzero mask, the ``x != x`` NaN test). As the torch validator does
    for bool children, the parent is also replaced by a dtype battery (zero,
    one, sign, non-finite and extreme values) and, for a two-operand
    comparison where the parent fills one operand, by the comparand and its
    two adjacent representable values. A substitute that moves the output
    proves the edge transmits value; when every substitute leaves the output
    at its saved value the check still fails.

    Parameters
    ----------
    capture
        Captured JAX equation metadata.
    inputs
        Replay inputs derived from trace graph parents.
    positions
        Primitive argument positions the perturbed parent fills.

    Returns
    -------
    tuple[Any, ...]
        Substitute values with the parent's shape and dtype.
    """

    import jax.numpy as jnp

    parent = jnp.asarray(inputs[positions[0]])
    substitutes = list(_bool_probe_battery(parent))
    if (
        str(capture.primitive) in _COMPARISON_PRIMITIVES
        and len(inputs) == 2
        and len(positions) == 1
    ):
        other = jnp.asarray(inputs[1 - positions[0]])
        substitutes.extend(_straddle_substitutes(parent, other))
    return tuple(substitutes)


def _bool_probe_battery(parent: Any) -> tuple[Any, ...]:
    """Return dtype-directed substitutes that flip a value-transmitting predicate."""

    import jax.numpy as jnp

    dtype = parent.dtype
    if dtype == jnp.bool_:
        return (jnp.zeros_like(parent), jnp.ones_like(parent), jnp.logical_not(parent))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return (
            jnp.zeros_like(parent),
            jnp.ones_like(parent),
            jnp.full_like(parent, 1j),
            jnp.full_like(parent, complex(float("nan"), 0.0)),
        )
    if jnp.issubdtype(dtype, jnp.floating):
        return (
            jnp.zeros_like(parent),
            jnp.ones_like(parent),
            -jnp.ones_like(parent),
            -parent,
            jnp.full_like(parent, jnp.nan),
            jnp.full_like(parent, jnp.inf),
            jnp.full_like(parent, -jnp.inf),
            jnp.full_like(parent, jnp.finfo(dtype).max / 2),
        )
    if jnp.issubdtype(dtype, jnp.integer):
        info = jnp.iinfo(dtype)
        battery = [
            jnp.zeros_like(parent),
            jnp.ones_like(parent),
            jnp.full_like(parent, info.max),
            jnp.full_like(parent, info.min),
        ]
        if info.min < 0:
            battery.append(-jnp.ones_like(parent))
        return tuple(battery)
    return ()


def _straddle_substitutes(parent: Any, other: Any) -> tuple[Any, ...]:
    """Return the comparand and its two adjacent representable values, shaped as ``parent``."""

    import jax.numpy as jnp

    dtype = parent.dtype
    numeric = jnp.issubdtype(dtype, jnp.floating) or jnp.issubdtype(dtype, jnp.integer)
    if not numeric or not (
        jnp.issubdtype(other.dtype, jnp.floating) or jnp.issubdtype(other.dtype, jnp.integer)
    ):
        return ()
    try:
        base = jnp.broadcast_to(other.astype(dtype), parent.shape)
    except ValueError:  # the comparand broadcasts wider than the parent: no straddle
        return ()
    if jnp.issubdtype(dtype, jnp.floating):
        if not bool(jnp.all(jnp.isfinite(base))):
            return ()
        try:
            return (
                base,
                jnp.nextafter(base, jnp.full_like(base, -jnp.inf)),
                jnp.nextafter(base, jnp.full_like(base, jnp.inf)),
            )
        except (TypeError, NotImplementedError):  # no nextafter for this float dtype
            return (base,)
    one = jnp.ones_like(base)
    return (base, base - one, base + one)


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
    graph_positions = getattr(op, "parent_arg_positions", {}).get("args", {})
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


def _perturb_candidates(value: Any) -> tuple[Any, ...]:
    """Return deterministic perturbation candidates for a JAX value.

    Parameters
    ----------
    value
        Parent payload to perturb.

    Returns
    -------
    tuple[Any, ...]
        Perturbed values with the same shape and dtype family.
    """

    import jax
    import jax.numpy as jnp

    array = jnp.asarray(value)
    if str(array.dtype).startswith("key<"):
        if not array.shape:
            return (jax.random.fold_in(value, 1),)
        flat = jnp.reshape(array, (-1,))
        folded = jax.vmap(lambda key: jax.random.fold_in(key, 1))(flat)
        return (jnp.reshape(folded, array.shape),)
    if array.dtype == jnp.bool_:
        return (jnp.logical_not(array),)
    if jnp.issubdtype(array.dtype, jnp.integer):
        info = jnp.iinfo(array.dtype)
        base_candidates = [
            array + jnp.asarray(1, dtype=array.dtype),
            array - jnp.asarray(1, dtype=array.dtype),
            jnp.zeros_like(array),
            jnp.full_like(array, info.max),
        ]
        if info.min < 0:
            base_candidates.append(jnp.full_like(array, info.min))
        return (
            *base_candidates,
            jnp.bitwise_xor(array, jnp.asarray(info.max, dtype=array.dtype)),
        )
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        magnitude = jnp.max(jnp.abs(array)) + jnp.asarray(1.0, dtype=array.real.dtype)
        return (
            array + jnp.asarray(magnitude + 0.125j, dtype=array.dtype),
            array - jnp.asarray(magnitude + 0.125j, dtype=array.dtype),
        )
    if jnp.issubdtype(array.dtype, jnp.floating):
        magnitude = jnp.max(jnp.abs(array)) + jnp.asarray(1.0, dtype=array.dtype)
        candidates: tuple[Any, ...] = (array + magnitude, array - magnitude)
        if not bool(jnp.all(jnp.isfinite(array))):
            # A NaN or inf parent absorbs any additive perturbation; finite
            # replacements are the only candidates that can move the child.
            candidates = (*candidates, jnp.zeros_like(array), jnp.ones_like(array))
        return candidates
    return (array,)


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
