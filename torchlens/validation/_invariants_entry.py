"""Invariant dispatch and backend-neutral identity checks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING

from .status import has_importer_region_provenance, is_region_replay_annotation

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace
    from .invariants import (
        InvariantResult,
        MetadataInvariantError,
        _check_module_containment_logic,
        _check_module_hierarchy,
        _check_module_layer_containment,
        _check_param_xrefs,
        _is_func_call_id_exempt,
        _metadata_invariant_contracts_for_trace,
        _plain_func_call_group_signature,
    )

__all__ = (
    "check_metadata_invariants",
    "_check_receptive_field_metadata_invariants",
    "_check_backend_neutral_module_mode_invariants",
    "_check_region_replay_provenance",
    "_check_function_root_module_invariants",
    "_check_compute_op_module_attribution",
    "_compute_ops",
    "_module_claims",
    "_module_claim_address",
    "_check_backend_identity_invariants",
    "_check_non_torch_backward_inert",
    "_check_backend_neutral_accessor_refs",
    "_record_has_backend_neutral_accessor_metadata",
    "check_func_call_id_invariant",
)


def check_metadata_invariants(trace: Trace) -> bool:
    """Run all metadata invariant checks on a completed ``Trace``.

    Checks run in dependency order: Phase 1 structural checks first, then
    Phase 2 semantic checks. Raises ``MetadataInvariantError`` on the first
    failure, so later checks can assume earlier ones passed.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.

    Returns
    -------
    bool
        ``True`` if all invariants pass.
    """
    for contract in _metadata_invariant_contracts_for_trace(trace):
        contract.check(trace)
    return True


def _check_receptive_field_metadata_invariants(trace: Trace) -> None:
    """Check autograd-free receptive-field descriptor and box invariants.

    Parameters
    ----------
    trace:
        Postprocessed trace whose influence geometry should be checked.

    Raises
    ------
    MetadataInvariantError
        If receptive-field metadata violates a geometric contract.
    """

    from ..receptive_field._errors import ReceptiveFieldError
    from ..receptive_field._validation import check_geometric_metadata_invariants

    try:
        check_geometric_metadata_invariants(trace)
    except ReceptiveFieldError as exc:
        raise MetadataInvariantError("receptive_field_metadata", str(exc)) from exc


def _check_backend_neutral_module_mode_invariants(trace: Trace) -> None:
    """Run module invariants appropriate to the trace's module identity mode.

    Parameters
    ----------
    trace:
        Postprocessed non-torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If mode-specific module metadata is internally inconsistent.
    """

    if getattr(trace, "module_identity_mode", None) == "function_root":
        _check_function_root_module_invariants(trace)
        return

    _check_compute_op_module_attribution(trace)
    _check_module_layer_containment(trace)  # H
    _check_module_hierarchy(trace)  # I
    _check_param_xrefs(trace)  # J
    _check_module_containment_logic(trace)  # Q


def _check_region_replay_provenance(trace: Trace) -> None:
    """Check region replay annotations have importer-owned provenance.

    Parameters
    ----------
    trace:
        Trace whose operation annotations should be checked.

    Raises
    ------
    MetadataInvariantError
        If an op is marked as a replay region without importer provenance on
        both the trace and the op.
    """

    name = "region_replay_provenance"
    trace_annotations = getattr(trace, "annotations", None)
    if trace_annotations is not None and not isinstance(trace_annotations, Mapping):
        trace_annotations = None
    for layer in getattr(trace, "layer_list", ()):
        op_annotations = getattr(layer, "annotations", None)
        if op_annotations is not None and not isinstance(op_annotations, Mapping):
            op_annotations = None
        if not is_region_replay_annotation(op_annotations):
            continue
        if has_importer_region_provenance(trace_annotations, op_annotations):
            continue
        label = getattr(layer, "layer_label", getattr(layer, "label", type(layer).__name__))
        raise MetadataInvariantError(
            name,
            f"Region replay annotation on '{label}' requires importer-owned provenance",
        )


def _check_function_root_module_invariants(trace: Trace) -> None:
    """Check minimal module metadata required for ``function_root`` traces.

    Parameters
    ----------
    trace:
        Postprocessed function-root trace to validate.

    Raises
    ------
    MetadataInvariantError
        If the root module is not the sole module, does not mirror trace
        layers, has invalid root boundary lists, or compute ops claim non-root
        module attribution.
    """

    name = "function_root_module_invariants"
    modules = list(trace.modules)
    module_addresses = [module.address for module in modules]
    if module_addresses != ["self"]:
        raise MetadataInvariantError(
            name,
            f"function_root traces must contain exactly ['self'] modules, got {module_addresses}",
        )

    root = modules[0]
    trace_layer_labels = list(trace.layer_labels)
    if list(root.layer_labels) != trace_layer_labels:
        raise MetadataInvariantError(
            name,
            "root module layer_labels must exactly match trace.layer_labels",
        )

    root_call = root.ops.get(1)
    if root_call is None:
        raise MetadataInvariantError(name, "root module must have exactly one self:1 call")
    # The call lists pass-qualified Op labels in execution order (exactly
    # trace.op_labels), and its distinct Layers in order are exactly
    # trace.layer_labels. One raise site for the check.
    root_call_ops = list(root_call.ops)
    root_call_problem = (
        "root module self:1 ops must exactly match trace.op_labels"
        if root_call_ops != list(trace.op_labels)
        else "root module self:1 ops must project onto exactly trace.layer_labels"
        if list(dict.fromkeys(trace.ops[label].layer_label for label in root_call_ops))
        != list(trace_layer_labels)
        else None
    )
    if root_call_problem is not None:
        raise MetadataInvariantError(name, root_call_problem)

    trace_layer_set = set(trace_layer_labels)
    for owner_label, owner in (("root module", root), ("root module call", root_call)):
        for attr_name in ("input_layers", "output_layers"):
            labels = set(getattr(owner, attr_name, ()) or ())
            extra = labels - trace_layer_set
            if extra:
                raise MetadataInvariantError(
                    name,
                    f"{owner_label} {attr_name} contains labels outside trace.layer_labels: "
                    f"{extra}",
                )

    for layer in _compute_ops(trace):
        non_root_claims = [
            claim
            for claim in _module_claims(layer)
            if _module_claim_address(claim) not in {None, "self"}
        ]
        if non_root_claims:
            raise MetadataInvariantError(
                name,
                f"Compute op '{layer.layer_label}' claims non-root module attribution: "
                f"{non_root_claims}",
            )


def _check_compute_op_module_attribution(trace: Trace) -> None:
    """Check compute ops resolve to a containing module in non-root modes.

    Parameters
    ----------
    trace:
        Postprocessed non-function-root trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a compute op has no module/module-call attribution, or the
        attribution does not resolve to a module that lists the op.
    """

    name = "module_attribution"
    for layer in _compute_ops(trace):
        claims = _module_claims(layer)
        if not claims:
            raise MetadataInvariantError(
                name,
                f"Compute op '{layer.layer_label}' has no module/module-call attribution",
            )

        resolved = False
        for claim in claims:
            address = _module_claim_address(claim)
            if address is None:
                continue
            try:
                module = trace.modules[address]
            except (KeyError, IndexError):
                continue
            if layer.layer_label in module.layer_labels:
                resolved = True
                break

        if not resolved:
            raise MetadataInvariantError(
                name,
                f"Compute op '{layer.layer_label}' attribution {claims} does not resolve "
                "to a Module that lists it",
            )


def _compute_ops(trace: Trace) -> list[Op]:
    """Return non-bookkeeping compute ops from ``trace``.

    Parameters
    ----------
    trace:
        Trace whose layer list should be filtered.

    Returns
    -------
    list[Op]
        Layers that are not synthetic input, output, or buffer entries.
    """

    return [
        layer
        for layer in trace.layer_list
        if not (layer.is_input or layer.is_output or layer.is_buffer)
    ]


def _module_claims(layer: Op) -> list[str]:
    """Return module/module-call attribution claims from an op.

    Parameters
    ----------
    layer:
        Op to inspect.

    Returns
    -------
    list[str]
        Non-empty string module claims in stable field order.
    """

    claims: list[str] = []
    for attr_name in (
        "module",
        "modules",
        "module_call_stack",
        "input_to_module_calls",
        "output_of_modules",
        "output_of_module_calls",
        "atomic_module_call",
    ):
        value = getattr(layer, attr_name, None)
        if isinstance(value, str):
            if value:
                claims.append(value)
        elif value:
            claims.extend(str(item) for item in value if item)
    return claims


def _module_claim_address(claim: str) -> str | None:
    """Return a module address from a module or module-call claim.

    Parameters
    ----------
    claim:
        Module address or call-qualified module label.

    Returns
    -------
    str | None
        Address with a trailing call suffix removed, or ``None`` for empty
        claims.
    """

    if not claim:
        return None
    return claim.rsplit(":", 1)[0]


def _check_backend_identity_invariants(trace: Trace) -> None:
    """Check backend identity and declared mode fields.

    Precondition contract: every completed trace, including torch traces, must
    declare a registered backend, a backend-supported module identity mode, and
    a param-source domain value. ``param_source='none'`` is legitimate for
    parameterless captures, but it is corruption if parameter tensors are
    reported elsewhere on the trace. Backend-specific address or resolver
    coupling is intentionally outside this identity contract.

    Parameters
    ----------
    trace:
        Postprocessed trace to validate.

    Raises
    ------
    MetadataInvariantError
        If backend identity fields are invalid or unsupported by the registry.
    """

    from torchlens.backends.registry import UnknownBackendError

    from ..backends.registry import get_backend_spec

    name = "backend_identity_invariants"
    backend = getattr(trace, "backend", None)
    if not isinstance(backend, str) or backend == "":
        raise MetadataInvariantError(name, "Trace.backend must be a non-empty string")
    try:
        spec = get_backend_spec(backend)
    except UnknownBackendError as exc:
        raise MetadataInvariantError(name, f"Trace.backend {backend!r} is not registered") from exc

    module_identity_mode = getattr(trace, "module_identity_mode", None)
    if module_identity_mode not in spec.capabilities.module_identity_modes:
        raise MetadataInvariantError(
            name,
            f"module_identity_mode={module_identity_mode!r} is not supported by backend "
            f"{backend!r}",
        )

    param_source = getattr(trace, "param_source", None)
    valid_param_sources = {"native-module", "pytree-derived", "none"}
    if param_source not in valid_param_sources:
        raise MetadataInvariantError(name, f"param_source={param_source!r} is invalid")
    if param_source == "none" and getattr(trace, "num_param_tensors", 0) != 0:
        raise MetadataInvariantError(
            name,
            "param_source='none' requires num_param_tensors=0",
        )


def _check_non_torch_backward_inert(trace: Trace) -> None:
    """Check that non-torch traces do not fake true backward graph metadata.

    Parameters
    ----------
    trace:
        Postprocessed non-torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If true-backward metadata is populated on a non-torch trace.
    """

    name = "non_torch_backward_inert"
    if getattr(trace, "has_backward_pass", False):
        raise MetadataInvariantError(name, "non-torch traces must not set has_backward_pass")
    if getattr(trace, "grad_fn_logs", None):
        raise MetadataInvariantError(name, "non-torch traces must not populate grad_fn_logs")
    if getattr(trace, "grad_fn_order", None):
        raise MetadataInvariantError(name, "non-torch traces must not populate grad_fn_order")
    if getattr(trace, "backward_pass_logs", None):
        raise MetadataInvariantError(name, "non-torch traces must not populate backward_pass_logs")
    if getattr(trace, "backward_root_grad_fn_object_ids", None):
        raise MetadataInvariantError(
            name,
            "non-torch traces must not populate backward_root_grad_fn_object_ids",
        )
    if getattr(trace, "num_backward_passes", 0) != 0:
        raise MetadataInvariantError(name, "non-torch traces must have num_backward_passes=0")


def _check_backend_neutral_accessor_refs(trace: Trace) -> None:
    """Check structural backend-neutral dtype/device/address resolver fields.

    Precondition contract: Op, Layer, and Param records may carry neutral mirror
    fields independent of retained payloads. Missing or ``None`` dtype/device
    refs and backend addresses are legitimate. When a neutral field is
    populated, the structural contract is backend-neutral: ``resolver_status``
    must be in the public status domain when present, dtype/device refs must
    expose non-empty ``backend`` and ``name`` strings when present, and
    ``backend_address`` must be a string when present. This check intentionally
    does not compare ref names to legacy dtype/device payload values, and it
    does not assert any ``backend_address`` <-> ``resolver_status`` semantic
    coupling.

    Parameters
    ----------
    trace:
        Postprocessed trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a layer or param has malformed neutral accessor metadata.
    """

    name = "backend_neutral_accessor_refs"
    valid_statuses = {"resolved", "unresolved", "audit_only", "metadata_only"}
    records = [
        *getattr(trace, "layer_list", ()),
        *list(getattr(trace, "layer_logs", {}).values()),
        *list(getattr(trace, "param_logs", {}).values()),
    ]
    for record in records:
        if not _record_has_backend_neutral_accessor_metadata(record):
            continue
        label = getattr(record, "layer_label", getattr(record, "address", type(record).__name__))
        resolver_status = getattr(record, "resolver_status", None)
        if resolver_status is not None and resolver_status not in valid_statuses:
            raise MetadataInvariantError(
                name,
                f"{label} has invalid resolver_status={resolver_status!r}",
            )
        for field_name in ("dtype_ref", "device_ref"):
            ref = getattr(record, field_name, None)
            if ref is not None and (
                not isinstance(getattr(ref, "backend", None), str)
                or getattr(ref, "backend", "") == ""
                or not isinstance(getattr(ref, "name", None), str)
                or getattr(ref, "name", "") == ""
            ):
                raise MetadataInvariantError(name, f"{label} has malformed {field_name}")
        backend_address = getattr(record, "backend_address", None)
        if backend_address is not None and not isinstance(backend_address, str):
            raise MetadataInvariantError(name, f"{label} has non-string backend_address")


def _record_has_backend_neutral_accessor_metadata(record: object) -> bool:
    """Return whether ``record`` has any populated neutral accessor field.

    Parameters
    ----------
    record:
        Op-, Layer-, or Param-like object to inspect.

    Returns
    -------
    bool
        ``True`` when a backend-neutral mirror field is present and non-``None``.
    """

    return any(
        getattr(record, field_name, None) is not None
        for field_name in ("resolver_status", "dtype_ref", "device_ref", "backend_address")
    )


def check_func_call_id_invariant(trace: Trace) -> InvariantResult:
    """Invariant S: func_call_id consistency.

    Precondition contract: torch exhaustive and predicate captures populate
    ``func_call_id`` for non-synthetic compute outputs. Synthetic input,
    output, buffer, and internal placeholder nodes are exempt. Sparse recording
    projections may have empty templates, ``_edge_use='unknown'`` records, and
    ``container_spec=None``; those fields are not required for this invariant.
    When a ``func_call_id`` group is populated, members must agree only on the
    plain-capture-stable function name and container spec, and populated
    container paths must be unique within the group. Intervention signature
    fields (argument templates and ``code_context`` reprs) are
    intentionally outside this plain-capture contract.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.

    Returns
    -------
    InvariantResult
        Passing result when no inconsistency is found.
    """

    name = "func_call_id_consistency"
    groups: dict[int, list[Op]] = defaultdict(list)
    for layer in trace.layer_list:
        if _is_func_call_id_exempt(layer):
            continue
        func_call_id = getattr(layer, "func_call_id", None)
        if func_call_id is None:
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label} has no func_call_id",
            )
        if not isinstance(func_call_id, int):
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label} has non-integer func_call_id {func_call_id!r}",
            )
        groups[func_call_id].append(layer)

    for func_call_id, group in groups.items():
        reference = group[0]
        expected_signature = _plain_func_call_group_signature(reference)
        container_paths: list[tuple[object, ...]] = []
        for layer in group:
            if _plain_func_call_group_signature(layer) != expected_signature:
                raise MetadataInvariantError(
                    name,
                    f"func_call_id {func_call_id} has incompatible call metadata",
                )
            container_path = tuple(getattr(layer, "container_path", ()) or ())
            if container_path and container_path in container_paths:
                raise MetadataInvariantError(
                    name,
                    f"func_call_id {func_call_id} has duplicate container_path {container_path!r}",
                )
            if container_path:
                container_paths.append(container_path)
    return InvariantResult(name=name, passed=True)
