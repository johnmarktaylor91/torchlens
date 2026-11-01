"""Relation views for ``Op``: parents, children, siblings, containers, modules.

Logic bodies live here as functions taking the record; ``Op`` keeps one thin
property per view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from .module import Module
    from .op import Op


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


def atomic_module(op: Op) -> Module | None:
    """Module record resolved from ``atomic_module_address``, or ``None``.

    Returns ``None`` when this Op is not an atomic-module output or when the
    owning Trace is unavailable.
    """

    address = op.atomic_module_address
    if address is None:
        return None
    trace = op._source_trace
    if trace is None:
        return None
    try:
        return cast("Module", trace.modules[address])
    except (KeyError, TypeError):
        return None


def siblings(op: Op) -> list[str]:
    """Layers sharing at least one parent (excluding output layers and this op)."""
    ml = op._source_trace
    if ml is None:
        return []
    siblings = []
    seen = op._own_label_spellings()
    for parent_label in op.parents:
        try:
            parent = ml[parent_label]
        except (KeyError, ValueError):
            try:
                parent = ml.graph.orphans[parent_label]
            except KeyError:
                continue
        for child_label in parent.children:
            if child_label not in seen:
                seen.add(child_label)
                try:
                    child = ml[child_label]
                except (KeyError, ValueError):
                    try:
                        child = ml.graph.orphans[child_label]
                    except KeyError:
                        continue
                if not child.is_output:
                    siblings.append(child_label)
    return siblings


def co_parents(op: Op) -> list[str]:
    """Layers sharing at least one child (excluding output layers and this op)."""
    ml = op._source_trace
    if ml is None:
        return []
    spouses = []
    seen = op._own_label_spellings()
    for child_label in op.children:
        try:
            child = ml[child_label]
        except (KeyError, ValueError):
            try:
                child = ml.graph.orphans[child_label]
            except KeyError:
                continue
        for parent_label in child.parents:
            if parent_label not in seen:
                seen.add(parent_label)
                try:
                    parent = ml[parent_label]
                except (KeyError, ValueError):
                    try:
                        parent = ml.graph.orphans[parent_label]
                    except KeyError:
                        continue
                if not parent.is_output:
                    spouses.append(parent_label)
    return spouses


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
        if parent.is_buffer and parent.buffer_write_kind is None:
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
        if child.is_buffer and child.buffer_write_kind is not None:
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


def input_to_modules(op: Op) -> list[str]:
    """Module addresses (no ``:N``) whose input this Op's output fed.

    Derived from ``input_to_module_calls`` by stripping the ``:N`` pass suffix
    and de-duplicating, mirroring the ``output_of_modules`` /
    ``output_of_module_calls`` split. Use ``input_to_module_calls`` for the
    ModuleCall-label list.
    """

    seen: dict[str, None] = {}
    for call_label in op.input_to_module_calls:
        address = str(call_label).rsplit(":", 1)[0]
        seen.setdefault(address, None)
    return list(seen)


def _resolve_relation_record(op: Op, label: str) -> Op | None:
    """Resolve one relation label through the mainline, then orphans.

    The 62aba742 orphan tolerance stopped at the label aggregates: on a
    ``keep_orphans=True`` trace the aggregates include an orphan's label
    while the OBJECT-resolving surfaces crashed on the bare mainline
    lookup (b3 R05-N2). Mirror the per-op ``siblings`` behavior: fold
    the ``orphans`` fallback, skip what neither surface resolves.
    """

    trace = op.source_trace
    try:
        return cast("Op", trace[label])
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
