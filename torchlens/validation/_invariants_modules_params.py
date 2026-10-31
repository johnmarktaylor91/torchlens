"""Module-call and parameter cross-reference invariants."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from .invariants import (
        MetadataInvariantError,
        _param_address_index,
        _param_log_list_contains_param,
        _resolve_trace_label,
        _strip_pass_suffix,
    )

__all__ = (
    "_check_module_call_boundary_and_tree",
    "_check_module_call_tree_links",
    "_check_module_call_output_structure_paths",
    "_container_tensor_leaf_paths",
    "_container_tensor_components",
    "_check_param_xrefs",
    "_check_param_usage_reciprocal_links",
    "_check_param_co_parent_links",
    "_check_layers_with_params_matches_param_usage",
    "_deduped_layers_with_params",
    "_check_layer_param_aggregate_dedup",
)


def _check_module_call_boundary_and_tree(
    ml: Trace,
    module_call: object,
    name: str,
) -> None:
    """Check one materialized ModuleCall boundary and dynamic call-tree links.

    Parameters
    ----------
    ml:
        Trace containing module-call metadata.
    module_call:
        ModuleCall record to validate.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If the ModuleCall's label, output boundary, call-tree links, stack, or
        output structure references are inconsistent.
    """

    call_label = getattr(module_call, "call_label", None)
    address = getattr(module_call, "address", None)
    call_index = getattr(module_call, "call_index", None)
    expected_call_label = f"{address}:{call_index}"
    if call_label != expected_call_label:
        raise MetadataInvariantError(
            name,
            f"ModuleCall {call_label!r} has address/call_index label {expected_call_label!r}",
        )

    call_ops = set(getattr(module_call, "ops", ()) or ())
    call_ops_no_pass = {_strip_pass_suffix(label) for label in call_ops}
    for output_op_label in getattr(module_call, "output_ops", ()) or ():
        resolved_label = _resolve_trace_label(ml, output_op_label)
        if resolved_label is None:
            raise MetadataInvariantError(
                name,
                f"ModuleCall '{call_label}' output_ops contains unresolved {output_op_label!r}",
            )
        if output_op_label not in call_ops and _strip_pass_suffix(output_op_label) not in call_ops:
            resolved_no_pass = _strip_pass_suffix(resolved_label)
            if resolved_label not in call_ops and resolved_no_pass not in call_ops_no_pass:
                raise MetadataInvariantError(
                    name,
                    f"ModuleCall '{call_label}' output_ops contains {output_op_label!r} "
                    "outside its ops",
                )

    for input_op_label in getattr(module_call, "input_ops", ()) or ():
        if _resolve_trace_label(ml, input_op_label) is None:
            raise MetadataInvariantError(
                name,
                f"ModuleCall '{call_label}' input_ops contains unresolved {input_op_label!r}",
            )

    _check_module_call_tree_links(ml, module_call, name)
    _check_module_call_output_structure_paths(ml, module_call, name)


def _check_module_call_tree_links(ml: Trace, module_call: object, name: str) -> None:
    """Check ModuleCall parent/child links and stack prefixes.

    Parameters
    ----------
    ml:
        Trace containing module-call metadata.
    module_call:
        ModuleCall record to validate.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a call-tree relation is unresolved, non-bidirectional, or has an
        inconsistent child stack prefix.
    """

    call_label = getattr(module_call, "call_label", "")
    call_accessor = getattr(ml, "module_calls", {})
    call_parent = getattr(module_call, "call_parent", None)
    if call_parent is not None:
        try:
            parent_call = call_accessor[call_parent]
        except (KeyError, IndexError):
            raise MetadataInvariantError(
                name,
                f"ModuleCall '{call_label}' call_parent={call_parent!r} is not a ModuleCall",
            )
        if call_label not in getattr(parent_call, "call_children", ()):
            raise MetadataInvariantError(
                name,
                f"ModuleCall '{call_label}' parent {call_parent!r} does not list it as a child",
            )

    for child_label in getattr(module_call, "call_children", ()) or ():
        try:
            child_call = call_accessor[child_label]
        except (KeyError, IndexError):
            raise MetadataInvariantError(
                name,
                f"ModuleCall '{call_label}' call_children contains unresolved {child_label!r}",
            )
        if getattr(child_call, "call_parent", None) != call_label:
            raise MetadataInvariantError(
                name,
                f"ModuleCall '{call_label}' child {child_label!r} has call_parent="
                f"{getattr(child_call, 'call_parent', None)!r}",
            )
        expected_prefix = list(getattr(module_call, "module_call_stack", ()) or ())
        if call_label != "self:1":
            expected_prefix.append(call_label)
        child_stack = list(getattr(child_call, "module_call_stack", ()) or ())
        if child_stack[: len(expected_prefix)] != expected_prefix:
            raise MetadataInvariantError(
                name,
                f"ModuleCall '{child_label}' module_call_stack={child_stack!r} does not "
                f"start with {expected_prefix!r}",
            )


def _check_module_call_output_structure_paths(
    ml: Trace,
    module_call: object,
    name: str,
) -> None:
    """Check complete output structures agree with retained output-op paths.

    The precondition contract is intentionally narrow: retained outputs and
    ``ContainerSpec`` leaves are only compared when both sides expose the same
    number of non-root paths. Partial structures are legitimate for captures
    that retain or project only part of a module output, and they are skipped
    rather than treated as corruption.

    Parameters
    ----------
    ml:
        Trace containing module-call metadata.
    module_call:
        ModuleCall record to validate.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a complete retained output path set disagrees with a complete output
        structure path set.
    """

    output_structure = getattr(module_call, "output_structure", None)
    if output_structure is None:
        return
    structure_paths = set(_container_tensor_leaf_paths(output_structure))
    output_paths: set[tuple[object, ...]] = set()
    captured_output_paths = tuple(getattr(module_call, "output_paths", ()) or ())
    if captured_output_paths:
        output_paths.update(tuple(path) for path in captured_output_paths if tuple(path))
    for output_op_label in getattr(module_call, "output_ops", ()) or ():
        if captured_output_paths:
            break
        resolved_label = _resolve_trace_label(ml, output_op_label)
        if resolved_label is None:
            continue
        output_op = ml[resolved_label]
        output_path = tuple(getattr(output_op, "container_path", ()) or ())
        if output_path:
            output_paths.add(output_path)
    if structure_paths and output_paths and len(structure_paths) == len(output_paths):
        mismatched_paths = sorted(output_paths ^ structure_paths, key=repr)
    else:
        mismatched_paths = []
    if mismatched_paths:
        raise MetadataInvariantError(
            name,
            f"ModuleCall '{getattr(module_call, 'call_label', '')}' output_structure "
            f"paths disagree with retained output paths {mismatched_paths!r}",
        )


def _container_tensor_leaf_paths(
    spec: object, prefix: tuple[object, ...] = ()
) -> list[tuple[object, ...]]:
    """Return tensor leaf paths from a ``ContainerSpec``-like object.

    Parameters
    ----------
    spec:
        ContainerSpec-like object with ``child_specs``.
    prefix:
        Path prefix accumulated during recursion.

    Returns
    -------
    list[tuple[object, ...]]
        Tensor leaf paths in traversal order.
    """

    kind = getattr(spec, "kind", None)
    if kind in {"literal", "opaque"}:
        return []
    components = _container_tensor_components(spec)
    if not components:
        return []
    child_specs = tuple(getattr(spec, "child_specs", ()) or ())
    child_by_component = dict(child_specs)
    paths: list[tuple[object, ...]] = []
    for component in components:
        child_spec = child_by_component.get(component)
        if child_spec is None:
            paths.append((*prefix, component))
        else:
            paths.extend(_container_tensor_leaf_paths(child_spec, (*prefix, component)))
    return paths


def _container_tensor_components(spec: object) -> tuple[object, ...]:
    """Return child components that may represent tensor leaves.

    Parameters
    ----------
    spec:
        ContainerSpec-like object.

    Returns
    -------
    tuple[object, ...]
        Components in traversal order.
    """

    kind = getattr(spec, "kind", None)
    if kind in {"tuple", "list", "registered"}:
        return tuple(TupleIndex(index) for index in range(int(getattr(spec, "length", 0) or 0)))
    if kind == "dict":
        return tuple(DictKey(key) for key in getattr(spec, "keys", ()) or ())
    if kind == "hf_model_output":
        return tuple(HFKey(key) for key in getattr(spec, "keys", ()) or ())
    if kind == "namedtuple":
        return tuple(NamedField(field) for field in getattr(spec, "fields", ()) or ())
    if kind == "dataclass":
        return tuple(DataclassField(field) for field in getattr(spec, "fields", ()) or ())
    return ()


def _check_param_xrefs(ml: Trace) -> None:
    """Check J: Param <-> Layer <-> Module cross-references.

    Precondition contract: this torch-native check asserts deep reciprocal
    references only for parameters that are actually used by at least one
    operation in the captured graph. Unused or skipped-module parameters may
    legitimately have no usage lists. ``Param.num_uses_by_ops`` is the
    pass-qualified usage source of truth for reciprocal usage checks; the
    stored ``num_calls`` field is compatibility metadata and is not used as the
    invariant oracle. Layer-level aggregate checks deduplicate by no-pass
    ``layer_label`` to match the trace aggregate semantics for recurrent and
    weight-shared parameters.

    Validates:
    - Param.used_by_ops labels are valid op labels.
    - Param.used_by_layers labels are valid layer labels.
    - Used Param usage lists reciprocate through Op/Layer ``_param_logs``.
    - ``layers_with_params`` layer membership matches Param.used_by_layers.
    - Co-parent params are symmetric and resolve.
    - uses_params == True implies _param_logs is non-empty.
    - layers_with_params values are valid layer labels.
    """
    name = "param_xrefs"
    label_set = set(ml.layer_labels)
    op_label_set = set(ml.op_labels)

    # Built on first co-parent link, not up front: most traces declare no
    # co-parents at all (weight tying and weight/bias siblings are the sources),
    # and those pay nothing.
    param_address_index: dict[object, object] | None = None

    for param in ml.param_logs:
        for lbl in param.used_by_ops:
            if lbl not in op_label_set:
                raise MetadataInvariantError(
                    name,
                    f"Param '{param.address}' used_by_ops contains '{lbl}' not in op_labels",
                )
        # used_by_layers exist
        for lbl in param.used_by_layers:
            if lbl not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"Param '{param.address}' used_by_layers contains '{lbl}' not in layer_labels",
                )

        # The documented "address exists" step used to be ``mod_accessor[param.address]``
        # inside ``try/except/pass`` with the result discarded -- a literal no-op that
        # accepted ``param.address = "nonexistent.module.weight"``. What IS provable
        # here is self-consistency: ``all_addresses`` is seeded with ``address`` at
        # construction and only ever GROWS (weight tying appends aliases), so every
        # correct capture satisfies this and a rewritten address is caught.
        #
        # Deliberately NOT asserted: that the param's owning MODULE resolves in
        # ``ml.modules``. A parameter can legitimately be USED while its owning module
        # is never entered -- ``F.linear(x, self.lin.weight, self.lin.bias)``, and in
        # stock torch ``nn.MultiheadAttention`` / ``nn.TransformerEncoderLayer`` (whose
        # ``out_proj`` submodule is bypassed by the fused attention kernel) -- so an
        # owner-resolution requirement false-fails those correct captures. Measured on
        # both at base commit e7f036fe.
        if param.address not in param.all_addresses:
            raise MetadataInvariantError(
                name,
                f"Param address {param.address!r} is absent from its canonical address set "
                f"{param.all_addresses!r}",
            )

        if param.num_uses_by_ops == 0 and not param.used_by_layers:
            continue
        _check_param_usage_reciprocal_links(ml, param, name)
        # Guarding on a non-empty co-parent list is behavior-neutral: with no
        # co-parents the callee's only loop body never runs.
        if getattr(param, "co_parent_params", None):
            if param_address_index is None:
                param_address_index = _param_address_index(ml)
            _check_param_co_parent_links(param_address_index, param, name)

    # uses_params forward check
    for lpl in ml.layer_list:
        if lpl.uses_params:
            if not lpl._param_logs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' has uses_params=True but _param_logs is empty",
                )

    # layers_with_params labels exist
    for param_addr, layer_labels in ml.layers_with_params.items():
        for lbl in layer_labels:
            if lbl not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"layers_with_params['{param_addr}'] contains '{lbl}' not in layer_labels",
                )

    _check_layers_with_params_matches_param_usage(ml, name)
    _check_layer_param_aggregate_dedup(ml, name)


def _check_param_usage_reciprocal_links(ml: Trace, param: object, name: str) -> None:
    """Check Param usage lists have reciprocal Op and Layer references.

    Parameters
    ----------
    ml:
        Trace containing parameter metadata.
    param:
        Param record whose usage references should be checked.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a used operation or layer does not point back to ``param``.
    """

    address = getattr(param, "address", "<unknown>")
    for op_label in getattr(param, "used_by_ops", ()):
        op = ml[op_label]
        if not _param_log_list_contains_param(getattr(op, "_param_logs", ()), param):
            raise MetadataInvariantError(
                name,
                f"Param '{address}' used_by_ops contains '{op_label}' but that Op does "
                "not list the Param in _param_logs",
            )
    for layer_label in getattr(param, "used_by_layers", ()):
        layer = ml.layer_logs[layer_label]
        if not _param_log_list_contains_param(getattr(layer, "_param_logs", ()), param):
            raise MetadataInvariantError(
                name,
                f"Param '{address}' used_by_layers contains '{layer_label}' but that "
                "Layer does not list the Param in _param_logs",
            )


def _check_param_co_parent_links(
    param_address_index: dict[object, object], param: object, name: str
) -> None:
    """Check co-parent parameter links resolve and are symmetric.

    Parameters
    ----------
    param_address_index:
        Primary/alias address -> Param index from ``_param_address_index``.
    param:
        Param record whose co-parent links should be checked.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a co-parent address is unresolved or not reciprocal.
    """

    address = getattr(param, "address", "<unknown>")
    for co_parent_address in getattr(param, "co_parent_params", ()) or ():
        co_parent = param_address_index.get(co_parent_address)
        if co_parent is None:
            raise MetadataInvariantError(
                name,
                f"Param '{address}' co_parent_params contains unresolved {co_parent_address!r}",
            )
        if address not in getattr(co_parent, "co_parent_params", ()):
            raise MetadataInvariantError(
                name,
                f"Param '{address}' links co-parent '{co_parent_address}' but the "
                "reverse link is missing",
            )


def _check_layers_with_params_matches_param_usage(ml: Trace, name: str) -> None:
    """Check ``layers_with_params`` matches Param usage at the layer boundary.

    Parameters
    ----------
    ml:
        Trace containing parameter metadata.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If trace-level layer-with-param groups drift from Param usage lists.
    """

    expected_layer_union: set[str] = set()
    for param in ml.param_logs:
        if getattr(param, "num_uses_by_ops", 0) == 0 and not getattr(param, "used_by_layers", ()):
            continue
        expected_layer_union.update(getattr(param, "used_by_layers", ()) or ())
    actual_layer_union: set[str] = set()
    for group_key, layer_labels in getattr(ml, "layers_with_params", {}).items():
        for layer_label in layer_labels:
            actual_layer_union.add(layer_label)
            layer = ml.layer_logs[layer_label]
            if not getattr(layer, "_param_logs", ()):
                raise MetadataInvariantError(
                    name,
                    f"layers_with_params[{group_key!r}] contains '{layer_label}' but the "
                    "Layer has no _param_logs",
                )
    if actual_layer_union != expected_layer_union:
        raise MetadataInvariantError(
            name,
            f"layers_with_params layer union {actual_layer_union!r} does not match "
            f"Param.used_by_layers union {expected_layer_union!r}",
        )


def _deduped_layers_with_params(ml: Trace) -> int:
    """Count distinct layer labels whose Layer carries any _param_logs.

    Parameters
    ----------
    ml:
        Trace containing layer metadata.
    """

    seen_layer_labels: set[str] = set()
    layers_with_params = 0
    for layer in ml.layer_list:
        if layer.layer_label in seen_layer_labels:
            continue
        seen_layer_labels.add(layer.layer_label)
        if getattr(layer, "_param_logs", ()):
            layers_with_params += 1
    return layers_with_params


def _check_layer_param_aggregate_dedup(ml: Trace, name: str) -> None:
    """Check trace param aggregate counts with layer-label deduplication.

    Parameters
    ----------
    ml:
        Trace containing layer and parameter metadata.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If deduplicated layer parameter aggregates drift from trace totals.
    """

    # A07 numbers truth: trace parameter totals follow PARAMETER OBJECT
    # IDENTITY (param_logs, object-deduplicated at the pre-forward scan) --
    # never a per-layer sum, which double-counts a parameter consumed by more
    # than one layer (tied embeddings, weight-reused layers) and drops
    # declared-but-never-executed parameters. The tripwire recomputes the
    # identity sums independently from param_logs.
    layers_with_params = _deduped_layers_with_params(ml)

    param_logs = list(getattr(ml, "param_logs", []) or [])
    total_params = sum(int(pl.num_params) for pl in param_logs)
    trainable_params = sum(int(pl.num_params) for pl in param_logs if pl.is_trainable)
    frozen_params = sum(int(pl.num_params) for pl in param_logs if not pl.is_trainable)

    # Cross-path tripwire: the DISTINCT parameter identities consumed by
    # layers must sum to no more than the declared identity inventory (layers
    # can never consume a parameter the inventory does not declare).
    consumed_by_identity: dict[int, int] = {}
    for layer in ml.layer_list:
        for pl in getattr(layer, "_param_logs", ()) or ():
            consumed_by_identity[id(pl)] = int(pl.num_params)
    consumed_total = sum(consumed_by_identity.values())
    if consumed_total > total_params:
        raise MetadataInvariantError(
            name,
            f"layer-consumed distinct-identity param total {consumed_total} exceeds "
            f"the declared identity inventory {total_params}",
        )

    if total_params != getattr(ml, "num_params", 0):
        raise MetadataInvariantError(
            name,
            f"identity param total {total_params} != trace.num_params={ml.num_params}",
        )
    if trainable_params != getattr(ml, "num_params_trainable", 0):
        raise MetadataInvariantError(
            name,
            "identity trainable param total "
            f"{trainable_params} != trace.num_params_trainable={ml.num_params_trainable}",
        )
    if frozen_params != getattr(ml, "num_params_frozen", 0):
        raise MetadataInvariantError(
            name,
            f"identity frozen param total {frozen_params} != "
            f"trace.num_params_frozen={ml.num_params_frozen}",
        )
    if layers_with_params != getattr(ml, "num_layers_with_params", 0):
        raise MetadataInvariantError(
            name,
            f"deduped layers_with_params count {layers_with_params} != "
            f"trace.num_layers_with_params={ml.num_layers_with_params}",
        )
