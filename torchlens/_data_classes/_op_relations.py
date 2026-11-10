"""Relation views for ``Op``: parents, children, siblings, containers, modules.

Logic bodies live here as functions taking the record; ``Op`` keeps one thin
property per view.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from .op import Op


__all__ = [
    "multi_output_type",
    "siblings",
    "co_parents",
    "memoized_relation_lookup",
    "relation_lookup",
    "input_ops_by_index",
    "buffer_source_ops_by_index",
    "buffer_sink_ops_by_index",
    "input_activations",
    "get_children",
    "get_parents",
    "kept",
]


def kept(op: Op, accessor: Any) -> Any:
    """Return ``accessor`` holding the trace that ``op`` holds.

    An op accessor names its ops by store row and builds each view when it is
    read. In ``tl.trace(model, x)["relu_1"].input_ops[1]`` the op is a
    temporary, gone with its trace before the parent is looked up, so the
    accessor an op hands out keeps the trace the op kept. An op that holds no
    trace (copied or restored on its own) hands out an accessor that holds
    none.
    """

    try:
        accessor._owner = object.__getattribute__(op, "_owner")
    except AttributeError:
        pass
    return accessor


def multi_output_type(op: Op) -> type | str | None:
    """Python class of the multi-output container this Op came from.

    Resolves ``container_spec.type_module``/``type_qualname`` to a runtime class
    through the single default-deny container resolver in
    ``torchlens._ir.container`` -- the SAME resolver used by output-container
    reconstruction. That resolver NEVER imports an attacker-named module (the
    ``ContainerSpec`` fields are portable, bundle-controlled strings that ride
    through the safe unpickler untouched, so importing one would execute its
    top-level code = arbitrary-code-execution); it resolves only from a module
    already present in ``sys.modules`` and only to a type that structurally
    matches the recorded container kind. When the type cannot be resolved
    WITHOUT importing (its defining module is not loaded), fall back to the
    qualified name string exactly as before, rather than importing it. ``None``
    when this Op is not from a multi-output container.

    Raises
    ------
    ContainerReconstructionError
        If the reference resolves to an already-loaded type that is not
        admissible for the recorded container kind (a tampered spec). This is
        the security tripwire firing, not a legit-capture path.
    """

    from .._ir.container import resolve_container_type

    if not op.in_multi_output:
        return None
    spec = op.container_spec
    if spec is None:
        return None
    resolved = resolve_container_type(spec)
    if resolved is not None:
        return resolved
    # Unresolvable without importing an untrusted, bundle-controlled module
    # name. Preserve the historical graceful string fallback WITHOUT importing.
    type_module = getattr(spec, "type_module", None)
    type_qualname = getattr(spec, "type_qualname", None)
    if type_qualname is None:
        return None
    if type_module:
        return f"{type_module}.{type_qualname}"
    return type_qualname


#: What a relation lookup returns for a label neither the trace nor its orphans hold.
_NOT_FOUND: Any = object()


def relation_lookup(trace: Any) -> Callable[[str], Any]:
    """Return ``label -> record`` the sibling and co-parent views resolve labels with.

    A label reads through ``trace[label]``, then the graph's orphans; one
    neither holds answers ``_NOT_FOUND`` (the view skips it).
    """

    def lookup(label: str) -> Any:
        """Return the record a relation list's ``label`` names, or ``_NOT_FOUND``."""

        try:
            return trace[label]
        except (KeyError, ValueError):
            try:
                return trace.graph.orphans[label]
            except KeyError:
                return _NOT_FOUND

    return lookup


def memoized_relation_lookup(trace: Any) -> Callable[[str], Any]:
    """Return :func:`relation_lookup` with each label resolved once.

    For a bulk reader that resolves the relation views of every op of one
    finished trace (``to_pandas``): label lookups are pure there, so a label
    met again answers the record found the first time.
    """

    plain = relation_lookup(trace)
    found: dict[str, Any] = {}

    def lookup(label: str) -> Any:
        """Return the record ``label`` names, resolving it on first use."""

        try:
            return found[label]
        except KeyError:
            record = found[label] = plain(label)
            return record

    return lookup


def _graph_neighbours(
    op: Op, outward: str, inward: str, lookup: Callable[[str], Any] | None
) -> list[str]:
    """Labels reached ``outward`` then back ``inward``, excluding outputs and this op."""

    if lookup is None:
        ml = op._source_trace
        if ml is None:
            return []
        lookup = relation_lookup(ml)
    neighbours = []
    seen = op._own_label_spellings()
    for via_label in getattr(op, outward):
        via = lookup(via_label)
        if via is _NOT_FOUND:
            continue
        for label in getattr(via, inward):
            if label not in seen:
                seen.add(label)
                record = lookup(label)
                if record is not _NOT_FOUND and not record.is_output:
                    neighbours.append(label)
    return neighbours


def siblings(op: Op, lookup: Callable[[str], Any] | None = None) -> list[str]:
    """Layers sharing at least one parent (excluding output layers and this op).

    ``lookup`` resolves labels (:func:`memoized_relation_lookup` for a bulk
    reader); by default the op's source trace, and ``[]`` without one.
    """

    return _graph_neighbours(op, "parents", "children", lookup)


def co_parents(op: Op, lookup: Callable[[str], Any] | None = None) -> list[str]:
    """Layers sharing at least one child (excluding output layers and this op).

    ``lookup`` as for :func:`siblings`.
    """

    return _graph_neighbours(op, "children", "parents", lookup)


def input_ops_by_index(op: Op) -> dict[int, Op]:
    """Accessor over graph-parent Op records in ``parents`` order."""

    trace = op._source_trace_or_error()
    parent_ops = {}
    for parent_index, parent_label in enumerate(op.parents, start=1):
        try:
            parent_ops[parent_index] = trace.ops[parent_label]
        except KeyError:
            if parent_label in trace._layer_dict_all_keys:
                parent_ops[parent_index] = trace._layer_dict_all_keys[parent_label]
    return parent_ops


def buffer_source_ops_by_index(op: Op) -> dict[int, Op]:
    """Accessor over buffer source Ops that feed this Op."""

    trace = op._source_trace_or_error()
    source_ops = {}
    for parent_index, parent_label in enumerate(op.parents, start=1):
        try:
            parent = trace.ops[parent_label]
        except KeyError:
            if parent_label not in trace._layer_dict_all_keys:
                continue
            parent = trace._layer_dict_all_keys[parent_label]
        if parent.graph.is_buffer and parent.buffer_write_kind is None:
            source_ops[parent_index] = parent
    return source_ops


def buffer_sink_ops_by_index(op: Op) -> dict[int, Op]:
    """Accessor over buffer sink Ops directly consuming this Op."""

    trace = op._source_trace_or_error()
    sink_ops = {}
    for child_index, child_label in enumerate(op.children, start=1):
        try:
            child = trace.ops[child_label]
        except KeyError:
            if child_label not in trace._layer_dict_all_keys:
                continue
            child = trace._layer_dict_all_keys[child_label]
        if child.graph.is_buffer and child.buffer_write_kind is not None:
            sink_ops[child_index] = child
    return sink_ops


def input_activations(op: Op) -> tuple[torch.Tensor | None, ...]:
    """Saved parent activations consumed by this Op, as references.

    Returned tensors are not copied. Mutating them mutates TorchLens saved
    state. For in-place-modified parents, the per-child version consumed by
    this Op is returned when available.
    """

    trace = op._source_trace_or_error()
    activations: list[torch.Tensor | None] = []
    child_labels = (
        op.layer_label,
        op.label,
        op._label_raw,
        op._layer_label_raw,
    )
    # ``Trace.ops`` is a cache-checking property, so resolving it per parent
    # made a wide fan-in operation (concatenation) pay that check once per
    # parent; the accessor cannot change while this loop runs.
    ops = trace.ops
    for parent_label in op.parents:
        parent: Op | None
        try:
            parent = ops[parent_label]
        except KeyError:
            parent = cast("Op | None", trace._layer_dict_all_keys.get(parent_label))
        if parent is None:
            activations.append(None)
            continue
        if not parent.has_saved_activation:
            activations.append(None)
            continue
        child_versions = getattr(parent, "out_versions_by_child", {}) or {}
        consumed = next(
            (child_versions[label] for label in child_labels if label in child_versions),
            parent.out,
        )
        activations.append(consumed if isinstance(consumed, torch.Tensor) else None)
    return tuple(activations)


def _resolve_relation_record(op: Op, label: str) -> Op | None:
    """Resolve one relation label through the mainline, then orphans.

    The 62aba742 orphan tolerance stopped at the label aggregates: on a
    ``keep_orphans=True`` trace the aggregates include an orphan's label
    while the OBJECT-resolving surfaces crashed on the bare mainline
    lookup. Mirror the per-op ``siblings`` behavior: fold
    the ``orphans`` fallback, skip what neither surface resolves.
    """

    trace = op.source_trace
    try:
        return cast("Op", trace.ops[label])
    except (KeyError, ValueError):
        try:
            return cast("Op", trace.graph.orphans[label])
        except KeyError:
            return None


def get_children(op: Op) -> list[Op]:
    """Return child Op objects for this pass.

    Returns
    -------
    list[Op]
        Child ops resolved through the owning model log; orphan-relation
        labels resolve through ``trace.graph.orphans`` (unresolvable skipped).
    """
    resolved = (op._resolve_relation_record(label) for label in op.children)
    return [record for record in resolved if record is not None]


def get_parents(op: Op) -> list[Op]:
    """Return parent Op objects for this pass.

    Returns
    -------
    list[Op]
        Parent ops resolved through the owning model log; orphan-relation
        labels resolve through ``trace.graph.orphans`` (unresolvable skipped).
    """
    resolved = (op._resolve_relation_record(label) for label in op.parents)
    return [record for record in resolved if record is not None]
