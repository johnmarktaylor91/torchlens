"""Executed-DAG endpoint resolution and directed path slicing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ._errors import (
    AmbiguousCallError,
    AmbiguousPassError,
    NoInfluencePathError,
    ReceptiveFieldError,
)
from ._types import ReceptiveFieldDirection


if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace


def resolve_graph_point(trace: Trace, handle: object) -> Op:
    """Resolve one supported graph-point handle to a pass-qualified operation.

    Parameters
    ----------
    trace:
        Trace against which the handle is resolved.
    handle:
        Exact pass-qualified label, ``Op``, ``Layer``, ``ModuleCall``, or ``Module``.

    Returns
    -------
    Op
        The uniquely resolved operation.

    Raises
    ------
    ReceptiveFieldError
        If the handle belongs to another trace, is unsupported, or does not resolve.
    AmbiguousPassError
        If a logical layer has multiple executed passes.
    AmbiguousCallError
        If a module has multiple executed calls.
    MultiOutputModuleError
        If a module call has zero or multiple output operations.
    """

    from ..data_classes.layer import Layer
    from ..data_classes.module import Module, ModuleCall
    from ..data_classes.op import Op
    from ..intervention.errors import MultiOutputModuleError

    if isinstance(handle, str):
        matches = [op for op in trace.layer_list if op.label == handle]
        if len(matches) != 1:
            raise ReceptiveFieldError(
                f"No operation with exact pass-qualified label {handle!r} exists in this trace."
            )
        return matches[0]

    if isinstance(handle, Op):
        _require_trace_owner(trace, handle.source_trace, handle.label)
        return handle

    if isinstance(handle, Layer):
        _require_trace_owner(trace, handle.source_trace, handle.layer_label)
        if handle.num_passes != 1 or len(handle.ops) != 1:
            passes = ", ".join(f"layer.ops[{index}]" for index in handle.ops.keys())
            raise AmbiguousPassError(
                f"Layer {handle.layer_label!r} has {handle.num_passes} passes: {passes}."
            )
        return handle.ops[0]

    if isinstance(handle, ModuleCall):
        _require_trace_owner(trace, handle._source_trace, handle.call_label)
        if len(handle.output_ops) != 1:
            candidates = ", ".join(
                f"module_call.output_ops[{index}]={label!r}"
                for index, label in enumerate(handle.output_ops)
            )
            raise MultiOutputModuleError(
                f"ModuleCall {handle.call_label!r} has {len(handle.output_ops)} output ops; "
                f"select one candidate explicitly: {candidates or 'none'}."
            )
        return resolve_graph_point(trace, handle.output_ops[0])

    if isinstance(handle, Module):
        _require_trace_owner(trace, handle._source_trace, handle.address)
        if handle.num_calls != 1 or len(handle.calls) != 1:
            calls = ", ".join(f"module.calls[{index}]" for index in handle.calls.keys())
            raise AmbiguousCallError(
                f"Module {handle.address!r} has {handle.num_calls} calls: {calls}."
            )
        return resolve_graph_point(trace, handle.calls[0])

    raise ReceptiveFieldError(
        "Graph point must be an exact pass-qualified label, Op, Layer, ModuleCall, or Module."
    )


def ancestor_labels(trace: Trace, target: Op | str) -> frozenset[str]:
    """Return the target and all of its executed-DAG ancestors.

    Parameters
    ----------
    trace:
        Trace containing the target.
    target:
        Target operation or exact pass-qualified label.

    Returns
    -------
    frozenset[str]
        Pass-qualified labels reachable by following parent edges, including the target.
    """

    target_op = resolve_graph_point(trace, target)
    _, parents, _ = _graph_indexes(trace)
    return _reachable_labels(target_op.label, parents)


def descendant_labels(trace: Trace, source: Op | str) -> frozenset[str]:
    """Return the source and all of its executed-DAG descendants.

    Parameters
    ----------
    trace:
        Trace containing the source.
    source:
        Source operation or exact pass-qualified label.

    Returns
    -------
    frozenset[str]
        Pass-qualified labels reachable by following child edges, including the source.
    """

    source_op = resolve_graph_point(trace, source)
    _, _, children = _graph_indexes(trace)
    return _reachable_labels(source_op.label, children)


def between_labels(trace: Trace, source: Op | str, target: Op | str) -> frozenset[str]:
    """Return the executed sub-DAG lying on at least one source-to-target path.

    Parameters
    ----------
    trace:
        Trace containing both endpoints.
    source:
        Source operation or exact pass-qualified label.
    target:
        Target operation or exact pass-qualified label.

    Returns
    -------
    frozenset[str]
        Intersection of the source descendants and target ancestors. The set is empty when
        no directed path exists.
    """

    source_op = resolve_graph_point(trace, source)
    target_op = resolve_graph_point(trace, target)
    return descendant_labels(trace, source_op) & ancestor_labels(trace, target_op)


def require_path(
    source: Op,
    target: Op,
    direction: ReceptiveFieldDirection | str,
) -> frozenset[str]:
    """Require and return the executed sub-DAG from ``source`` to ``target``.

    Parameters
    ----------
    source:
        Resolved source operation.
    target:
        Resolved target operation.
    direction:
        Public query direction used to contextualize any failure.

    Returns
    -------
    frozenset[str]
        Labels on at least one directed source-to-target path, including both endpoints.

    Raises
    ------
    ReceptiveFieldError
        If the endpoints belong to different traces.
    NoInfluencePathError
        If no directed source-to-target path exists.
    """

    trace = source.source_trace
    if trace is None or target.source_trace is not trace:
        raise ReceptiveFieldError(
            "Influence-geometry endpoints must belong to the same captured trace."
        )
    normalized_direction = ReceptiveFieldDirection(direction)
    path = between_labels(trace, source, target)
    if target.label in path:
        return path

    message = (
        "Influence geometry requires at least one directed A -> B path in the captured DAG; "
        f"source={source.label!r}, target={target.label!r}, "
        f"direction={normalized_direction.value!r}."
    )
    if source.label in descendant_labels(trace, target):
        message += " A reverse path exists; swap the source and target endpoints."
    raise NoInfluencePathError(message)


def _require_trace_owner(trace: Trace, owner: object, label: str) -> None:
    """Raise when an entity handle is detached or owned by another trace.

    Parameters
    ----------
    trace:
        Expected owning trace.
    owner:
        Actual trace reference exposed by the entity.
    label:
        Human-readable entity identity for the error.
    """

    if owner is not trace:
        raise ReceptiveFieldError(
            f"Graph point {label!r} does not belong to the supplied captured trace."
        )


def _graph_indexes(
    trace: Trace,
) -> tuple[Mapping[str, Op], Mapping[str, frozenset[str]], Mapping[str, frozenset[str]]]:
    """Build canonical operation, parent, and child indexes for one executed DAG.

    Parameters
    ----------
    trace:
        Trace whose live executed operations are indexed.

    Returns
    -------
    tuple[Mapping[str, Op], Mapping[str, frozenset[str]], Mapping[str, frozenset[str]]]
        Canonical operation lookup followed by parent and child adjacency mappings.
    """

    operations = tuple(op for op in trace.layer_list if isinstance(getattr(op, "label", None), str))
    by_reference = {
        reference: op
        for op in operations
        for reference in (op.label, op.layer_label, op._layer_label_raw)
    }
    by_label = {op.label: op for op in operations}
    parents: dict[str, set[str]] = {op.label: set() for op in operations}
    children: dict[str, set[str]] = defaultdict(set)
    for op in operations:
        for parent_reference in op.parents:
            parent = by_reference.get(parent_reference)
            if parent is None:
                continue
            parents[op.label].add(parent.label)
            children[parent.label].add(op.label)
    return (
        by_label,
        {label: frozenset(values) for label, values in parents.items()},
        {label: frozenset(children.get(label, set())) for label in by_label},
    )


def _reachable_labels(start: str, adjacency: Mapping[str, frozenset[str]]) -> frozenset[str]:
    """Return labels reachable from one start label through an adjacency mapping.

    Parameters
    ----------
    start:
        Canonical pass-qualified start label.
    adjacency:
        Directed adjacency mapping.

    Returns
    -------
    frozenset[str]
        Reachable labels including ``start``.
    """

    reachable = {start}
    stack = [start]
    while stack:
        label = stack.pop()
        unseen = adjacency.get(label, frozenset()) - reachable
        reachable.update(unseen)
        stack.extend(unseen)
    return frozenset(reachable)


__all__: list[str] = []
