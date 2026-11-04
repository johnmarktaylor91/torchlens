"""Executed-DAG endpoint resolution and directed path slicing."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

from ._errors import (
    AmbiguousCallError,
    AmbiguousPassError,
    AmbiguousTargetError,
    NoInfluencePathError,
    ReceptiveFieldError,
)
from ._types import ReceptiveFieldDirection

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from ._query import _IndexSet
    from ._types import ReceptiveField

# Output path segments that name a cache field; see ``_is_cache_role``.
_CACHE_ROLE_SEGMENTS = frozenset({"past_key_values", "presents", "mems"})


def _op_by_name(trace: Trace, handle: str) -> Op:
    """Resolve a label, or the IO role that field status reports, to one operation.

    Parameters
    ----------
    trace:
        Trace against which the name is resolved.
    handle:
        Exact pass-qualified label, or an IO role such as ``"output.logits"``.

    Returns
    -------
    Op
        The uniquely resolved operation.

    Raises
    ------
    ReceptiveFieldError
        If the name matches no operation, or an IO role matches several.
    """

    matches = [op for op in trace._layer_list if op.label == handle]
    if len(matches) == 1:
        return matches[0]
    detail = f"No operation with exact pass-qualified label {handle!r} exists in this trace."
    if not matches:
        by_role = [op for op in trace._layer_list if str(op.io_role or "") == handle]
        if len(by_role) == 1:
            return by_role[0]
        if by_role:
            spellings = ", ".join(op.label for op in by_role)
            detail = (
                f"IO role {handle!r} names several operations ({spellings}); "
                "select one by its pass-qualified label."
            )
    raise ReceptiveFieldError(detail)


def resolve_graph_point(trace: Trace, handle: object) -> Op:
    """Resolve one supported graph-point handle to a pass-qualified operation.

    Parameters
    ----------
    trace:
        Trace against which the handle is resolved.
    handle:
        Exact pass-qualified label, IO role as field status reports it, ``Op``,
        ``Layer``, ``ModuleCall``, or ``Module``.

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

    from .._data_classes.layer import Layer
    from .._data_classes.module import Module, ModuleCall
    from .._data_classes.op import Op
    from ..intervention.errors import MultiOutputModuleError

    if isinstance(handle, str):
        return _op_by_name(trace, handle)

    if isinstance(handle, Op):
        _require_trace_owner(trace, handle.source_trace, handle.label)
        return handle

    if isinstance(handle, Layer):
        _require_trace_owner(trace, handle.source_trace, handle.layer_label)
        if handle.num_passes != 1 or len(handle.ops) != 1:
            passes = ", ".join(f"layer.ops[{index}]" for index in handle.ops)
            raise AmbiguousPassError(
                f"Layer {handle.layer_label!r} has {handle.num_passes} passes: {passes}."
            )
        return handle.ops[0]

    if isinstance(handle, ModuleCall):
        _require_trace_owner(trace, handle._source_trace, handle.label)
        if len(handle.output_ops) != 1:
            candidates = ", ".join(
                f"module_call.output_ops[{index}]={label!r}"
                for index, label in enumerate(handle.output_ops)
            )
            raise MultiOutputModuleError(
                f"ModuleCall {handle.label!r} has {len(handle.output_ops)} output ops; "
                f"select one candidate explicitly: {candidates or 'none'}."
            )
        return resolve_graph_point(trace, handle.output_ops[0])

    if isinstance(handle, Module):
        _require_trace_owner(trace, handle._source_trace, handle.address)
        if handle.num_calls != 1 or len(handle.calls) != 1:
            calls = ", ".join(f"module.calls[{index}]" for index in handle.calls)
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


def forward_index_image(
    source_set: _IndexSet,
    candidates: _IndexSet,
    backward_map: Callable[[_IndexSet], tuple[_IndexSet, bool]],
) -> tuple[_IndexSet, bool]:
    """Transpose an integer relation by querying its backward membership oracle.

    Parameters
    ----------
    source_set:
        Parent indices whose forward image is requested.
    candidates:
        Sound child-index envelope to test.
    backward_map:
        Sealed backward set callback specialized to one operation axis.

    Returns
    -------
    tuple[_IndexSet, bool]
        Candidate child indices whose backward image intersects ``source_set``, followed by
        whether every oracle answer and both supplied sets were exact.
    """

    from ._query import _IndexSet

    source_values = frozenset(source_set.values())
    reached: list[int] = []
    exact = source_set.exact and candidates.exact
    for candidate in candidates.values():
        backward_set, callback_exact = backward_map(_IndexSet.singleton(candidate))
        exact = exact and callback_exact and backward_set.exact
        if source_values.intersection(backward_set.values()):
            reached.append(candidate)
    return _IndexSet.from_values(reached, exact=exact), exact


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

    operations = tuple(
        op for op in trace._layer_list if isinstance(getattr(op, "label", None), str)
    )
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


def _is_cache_role(role: str) -> bool:
    """Return whether an output IO role names a cache tensor rather than a real output.

    A role is a cache role when one of its dotted path segments is a known
    cache field (``past_key_values``, ``presents``, ``mems``) or contains
    ``cache`` (``cache_params``, ``kv_cache``), compared case-insensitively.
    """

    return any(
        segment in _CACHE_ROLE_SEGMENTS or "cache" in segment for segment in role.lower().split(".")
    )


def _primary_target_role(
    descriptors: Mapping[str, ReceptiveField], order: Sequence[str]
) -> str | None:
    """Return the primary output role when every other reachable target is a cache.

    Parameters
    ----------
    descriptors:
        Reachable target descriptors keyed by output IO role.
    order:
        Target operation labels in model output order.

    Returns
    -------
    str | None
        The role whose last segment is ``logits``, else the first non-cache
        role in output order; ``None`` when no cache role is reachable (two
        real outputs stay ambiguous) or every reachable role is a cache.
    """

    real = [role for role in descriptors if not _is_cache_role(role)]
    if not real or len(real) == len(descriptors):
        return None
    logits = [role for role in real if role.rsplit(".", 1)[-1] == "logits"]
    if logits:
        return logits[0]
    rank = {label: index for index, label in enumerate(order)}
    return min(real, key=lambda role: rank.get(descriptors[role].input_op_label, len(rank)))


def _select_target_descriptor(
    descriptors: Mapping[str, ReceptiveField],
    target: Op | str | None,
    order: Sequence[str] = (),
    source_label: str | None = None,
) -> ReceptiveField:
    """Resolve one projective result descriptor without choosing silently.

    With no ``target`` and several reachable targets, the primary output is
    chosen only when the others are cache tensors (see ``_primary_target_role``;
    ``order`` is the solution's target labels in model output order). Any other
    several-target case raises ``AmbiguousTargetError``.
    """

    if target is None:
        if len(descriptors) == 1:
            return next(iter(descriptors.values()))
        primary = _primary_target_role(descriptors, order)
        if primary is not None:
            return descriptors[primary]
        roles = ", ".join(descriptors)
        prefix = "This source" if source_label is None else f"Source {source_label!r}"
        raise AmbiguousTargetError(
            f"{prefix} has multiple reachable targets: {roles}. "
            "Select one with view[target_op] or target=<io_role>."
        )
    identity = target if isinstance(target, str) else target.label
    for role, descriptor in descriptors.items():
        if identity in {role, descriptor.input_op_label}:
            return descriptor
    raise ReceptiveFieldError(f"Target {identity!r} is not reachable from this source.")
