"""Metadata invariant checks for ``Trace`` and its sub-objects.

Single entry point: ``check_metadata_invariants(trace)`` runs all checks
and raises ``MetadataInvariantError`` on the first failure.

**Phase 1 -- Structural invariants:**
  A. Trace self-consistency (counts, timing, label uniqueness)
  B. Special layer lists match per-layer boolean flags
  C. Graph topology (parent-child bidirectionality, boolean flag consistency)
  D. Op field consistency (shape, dtype, pass numbering, nesting)
  E. Recurrence / loop invariants (is_recurrent, pass dicts)
  F. Branching invariants (is_branching)
  F2. Conditional metadata invariants (15 conditional consistency checks)
  G. Op <-> Layer cross-references (pass numbering, back-pointers)
  H. Module <-> Layer containment (layers, module pass layers, reverse check)
  I. Module hierarchy (address parent-child bidirectionality, pass consistency)
  J. Param cross-references (Param -> layer, uses_params flag)
  K. Buffer cross-references (buffer_layers list, Buffer module references)
  L. Equivalence group symmetry (op_equivalence_classes labels are valid)

**Phase 2 -- Semantic invariants:**
  M. Graph ordering (raw_index uniqueness/monotonicity, topological order, no raw labels)
  N. Loop detection invariants (recurrent_ops symmetry, func identity, param sharing, pass numbering)
  O. Distance / reachability (min <= max, input/output layer distances == 0, ancestor/descendent consistency)
  P. Graph connectivity (non-input non-buffer layers have parents, orphans removed)
  Q. Module containment logic (address acyclicity, depth consistency, nested path ordering)
  R. Lookup key bidirectionality (forward/reverse dicts, raw/final label maps)
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, cast

from ..ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from ..errors._base import ValidationError
from .status import has_importer_region_provenance, is_region_replay_annotation

if TYPE_CHECKING:
    from ..data_classes.layer import Layer
    from ..backends import BackendSpec
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace
    from ..data_classes.module import Module

InvariantApplicability = Literal["torch", "non_torch", "all"]
MetadataInvariantFunc = Callable[["Trace"], object]


class MetadataInvariantError(ValidationError, ValueError):
    """Raised when a metadata invariant check fails.

    Embeds the check name (e.g., ``"graph_topology"``) in the message prefix
    and stores it as an attribute for programmatic inspection in tests.
    """

    def __init__(self, check_name: str, message: str) -> None:
        """Initialize a metadata invariant failure.

        Parameters
        ----------
        check_name:
            Invariant check group name.
        message:
            Human-readable failure detail.
        """

        super().__init__(f"[{check_name}] {message}")
        self.check_name = check_name


@dataclass(frozen=True)
class InvariantResult:
    """Structured result for an individual metadata invariant.

    Attributes
    ----------
    name:
        Invariant check name.
    passed:
        Whether the invariant passed.
    message:
        Optional diagnostic message.
    """

    name: str
    passed: bool
    message: str = ""


@dataclass(frozen=True)
class MetadataInvariantContract:
    """Backend applicability contract for one metadata invariant check.

    Attributes
    ----------
    name:
        Stable check name used for dispatch introspection.
    check:
        Callable that runs the invariant.
    applies_to:
        Backend family this check currently runs on.
    requires_capability:
        Optional backend capability flag that must be truthy for the check to run.
    """

    name: str
    check: MetadataInvariantFunc
    applies_to: InvariantApplicability
    requires_capability: str | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_metadata_invariants(trace: "Trace") -> bool:
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


def _check_torch_metadata_invariants(trace: "Trace") -> bool:
    """Run the unchanged torch metadata invariant sequence.

    Parameters
    ----------
    trace:
        Postprocessed torch trace to validate.

    Returns
    -------
    bool
        ``True`` if all torch invariants pass.
    """

    for contract in _metadata_invariant_contracts_for_backend("torch"):
        contract.check(trace)
    return True


def _check_backend_neutral_metadata_invariants(trace: "Trace") -> bool:
    """Run backend-neutral metadata invariants for non-torch traces.

    Parameters
    ----------
    trace:
        Postprocessed non-torch trace to validate.

    Returns
    -------
    bool
        ``True`` if all backend-neutral invariants pass.
    """

    for contract in _metadata_invariant_contracts_for_backend("non_torch"):
        contract.check(trace)
    return True


def _check_backend_neutral_module_mode_invariants(trace: "Trace") -> None:
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


def _check_region_replay_provenance(trace: "Trace") -> None:
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


def _check_function_root_module_invariants(trace: "Trace") -> None:
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
    if list(root_call.ops) != trace_layer_labels:
        raise MetadataInvariantError(
            name,
            "root module self:1 ops must exactly match trace.layer_labels",
        )

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


def _check_compute_op_module_attribution(trace: "Trace") -> None:
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


def _compute_ops(trace: "Trace") -> list["Op"]:
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


def _module_claims(layer: "Op") -> list[str]:
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


def _check_backend_identity_invariants(trace: "Trace") -> None:
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

    from ..backends import UnknownBackendError, get_backend_spec

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


def _check_non_torch_backward_inert(trace: "Trace") -> None:
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


def _check_backend_neutral_accessor_refs(trace: "Trace") -> None:
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


def check_func_call_id_invariant(trace: "Trace") -> InvariantResult:
    """Invariant S: func_call_id consistency.

    Precondition contract: torch exhaustive and predicate captures populate
    ``func_call_id`` for non-synthetic compute outputs. Synthetic input,
    output, buffer, and internal placeholder nodes are exempt. Sparse recording
    projections may have empty templates, ``edge_use='unknown'`` records, and
    ``container_spec=None``; those fields are not required for this invariant.
    When a ``func_call_id`` group is populated, members must agree only on the
    plain-capture-stable function name and container spec, and populated
    container paths must be unique within the group. The old intervention
    signature fields (argument templates and ``code_context`` reprs) are
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
    groups: dict[int, list["Op"]] = defaultdict(list)
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


def _check_backward_graph_invariants(trace: "Trace") -> None:
    """Check T: backward grad-fn metadata consistency.

    A forward layer with a recorded ``grad_fn_object_id`` must retain a
    layer-to-GradFn backpointer unless it is structurally outside the backward
    pass contract: its final ``step_index`` must be greater than every recorded
    backward trigger's structural forward boundary. The boundary starts from the
    trigger-time forward op count and is tightened by paired root GradFns and
    observed op-gradient events when those identify the walked forward prefix.
    This admits legitimate mid-forward ``autograd.grad`` cases where later
    forward layers did not exist when backward graph walking ran, while still
    failing pre-trigger layers whose backpointer was accidentally severed.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.

    Raises
    ------
    MetadataInvariantError
        If backward metadata is internally inconsistent.
    """

    name = "backward_graph_invariants"
    sync_projection = getattr(trace, "_sync_backward_projection_if_needed", None)
    if callable(sync_projection):
        sync_projection()
    _check_backward_event_flow_invariants(trace, name)
    if not trace.grad_fn_logs:
        return

    valid_pass_indices = _check_backward_grad_fn_registry(trace, name)
    _check_backward_grad_fn_handle_records(trace, name, valid_pass_indices)
    _check_backward_layer_backpointers(trace, name)
    _check_backward_saved_grad_records(trace, name)
    _check_backward_pass_index_density(trace, name)
    _check_grad_fn_topology_invariants(trace, name)
    _check_backward_pass_domain_invariants(trace, name, valid_pass_indices)
    _check_backward_pass_record_consistency(trace, name, valid_pass_indices)


def _check_backward_grad_fn_registry(trace: "Trace", name: str) -> set[int]:
    """Check backward GradFn registry and root references.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.

    Returns
    -------
    set[int]
        Valid backward pass indices.

    Raises
    ------
    MetadataInvariantError
        If registry or root references are inconsistent.
    """

    grad_fn_ids = set(trace.grad_fn_logs)
    order_ids = set(trace.grad_fn_order)
    if not order_ids <= grad_fn_ids:
        missing = sorted(order_ids - grad_fn_ids)
        raise MetadataInvariantError(name, f"grad_fn_order contains unknown ids {missing!r}")

    root_ids = trace.backward_root_grad_fn_object_ids
    if not isinstance(root_ids, list):
        raise MetadataInvariantError(
            name,
            f"backward_root_grad_fn_object_ids must be a list, got {type(root_ids).__name__}",
        )
    missing_root_ids = [root_id for root_id in root_ids if root_id not in trace.grad_fn_logs]
    if missing_root_ids:
        raise MetadataInvariantError(
            name,
            f"backward_root_grad_fn_object_ids {missing_root_ids!r} are not present in grad_fn_logs",
        )

    return set(getattr(trace, "backward_pass_logs", {}).keys())


def _check_backward_grad_fn_handle_records(
    trace: "Trace",
    name: str,
    valid_pass_indices: set[int],
) -> None:
    """Check per-GradFn handle metadata consistency.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.
    valid_pass_indices:
        Backward pass indices known on the trace.

    Raises
    ------
    MetadataInvariantError
        If a GradFn handle has inconsistent fields or call records.
    """

    layer_labels = set(trace.layer_labels)
    for grad_fn_object_id, grad_fn_handle in trace.grad_fn_logs.items():
        if not re.fullmatch(r"[a-z0-9_]+_back_[1-9]\d*_[1-9]\d*", grad_fn_handle.label):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label!r} does not match backward-native label grammar",
            )
        if grad_fn_handle.has_op and grad_fn_handle.op_label not in layer_labels:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} points to missing layer {grad_fn_handle.op_label!r}",
            )
        membership_source = getattr(grad_fn_handle, "module_membership_source", None)
        if membership_source not in {None, "paired", "inferred"}:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has invalid module_membership_source "
                f"{membership_source!r}",
            )
        if membership_source is None:
            if grad_fn_handle.module_address is not None or grad_fn_handle.modules:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} has module containment without a source",
                )
        elif grad_fn_handle.module_address is None or not grad_fn_handle.modules:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has incomplete {membership_source!r} module containment",
            )
        op = grad_fn_handle.op
        if grad_fn_handle.has_op != (op is not None):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has inconsistent has_op/op fields",
            )
        if grad_fn_handle.grad_fn_object_id != grad_fn_object_id:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} stored id {grad_fn_handle.grad_fn_object_id!r} under {grad_fn_object_id!r}",
            )
        creator_object_id = getattr(grad_fn_handle, "creator_object_id", None)
        if creator_object_id is not None:
            creator = trace.grad_fn_logs.get(creator_object_id)
            if creator is None:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} points to missing creator id {creator_object_id!r}",
                )
            if grad_fn_handle.origin_backward_pass not in valid_pass_indices:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} has invalid origin backward pass "
                    f"{grad_fn_handle.origin_backward_pass!r}",
                )
            if creator.order is not None and grad_fn_handle.order is not None:
                expected_order = creator.order + 1
                if grad_fn_handle.order != expected_order:
                    raise MetadataInvariantError(
                        name,
                        f"{grad_fn_handle.label} order {grad_fn_handle.order!r} does not "
                        f"match creator order + 1 ({expected_order!r})",
                    )
        call_ordinals = sorted(grad_fn_handle.calls.keys())
        if call_ordinals != list(range(1, len(call_ordinals) + 1)):
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has non-dense local call ordinals {call_ordinals!r}",
            )
        for ordinal, call in grad_fn_handle.calls.items():
            if call.ordinal != ordinal or call.call_index != ordinal:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label}:{ordinal} has inconsistent local ordinal fields",
                )
            if call.backward_pass_index is None:
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label}:{ordinal} is missing backward_pass_index",
                )


def _check_backward_layer_backpointers(trace: "Trace", name: str) -> None:
    """Check forward-layer backpointers into backward GradFn handles.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.

    Raises
    ------
    MetadataInvariantError
        If a layer points to a missing or severed GradFn handle.
    """

    for layer in trace.layer_list:
        grad_fn_object_id = layer.grad_fn_object_id
        if grad_fn_object_id is None:
            continue
        if layer.grad_fn is None:
            if _layer_postdates_all_backward_triggers(trace, layer):
                continue
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label} with grad_fn_handle id {grad_fn_object_id!r} "
                "is missing its GradFn backpointer",
            )
        if grad_fn_object_id not in trace.grad_fn_logs:
            raise MetadataInvariantError(
                name,
                f"Layer {layer.layer_label} points to missing grad_fn_handle id "
                f"{grad_fn_object_id!r}",
            )


def _check_backward_saved_grad_records(trace: "Trace", name: str) -> None:
    """Check saved gradient-op records match layers that have gradients.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.

    Raises
    ------
    MetadataInvariantError
        If saved-grad op labels and layer ``has_grad`` flags disagree.
    """

    expected_saved_grad_labels = {layer.label for layer in trace.layer_list if layer.has_grad}
    saved_grad_labels = {op.label for op in trace.saved_grad_ops}
    if saved_grad_labels != expected_saved_grad_labels:
        raise MetadataInvariantError(
            name,
            "saved_grad_ops does not match layers with saved grad tensors",
        )


def _check_backward_pass_record_consistency(
    trace: "Trace",
    name: str,
    valid_pass_indices: set[int],
) -> None:
    """Check backward pass logs and call references are internally consistent.

    Parameters
    ----------
    trace:
        Trace with populated backward GradFn metadata.
    name:
        Invariant check name for raised errors.
    valid_pass_indices:
        Backward pass indices known on the trace.

    Raises
    ------
    MetadataInvariantError
        If backward pass logs are not dense or calls reference missing passes.
    """

    for pass_index, backward_pass in getattr(trace, "backward_pass_logs", {}).items():
        if backward_pass.pass_index != pass_index:
            raise MetadataInvariantError(
                name,
                f"BackwardPass stored index {backward_pass.pass_index!r} under {pass_index!r}",
            )
        for call in backward_pass.grad_fn_calls:
            if call.backward_pass_index != pass_index:
                raise MetadataInvariantError(
                    name,
                    f"{call.call_label} is attached to pass {pass_index} but records "
                    f"pass {call.backward_pass_index}",
                )
    for grad_fn_handle in trace.grad_fn_logs.values():
        for call in grad_fn_handle.calls.values():
            if call.backward_pass_index not in valid_pass_indices:
                raise MetadataInvariantError(
                    name,
                    f"{call.call_label} references missing backward pass "
                    f"{call.backward_pass_index}",
                )


def _check_backward_pass_index_density(trace: "Trace", name: str) -> None:
    """Check backward pass log keys are dense.

    Parameters
    ----------
    trace:
        Trace with populated backward pass metadata.
    name:
        Invariant check name for raised errors.

    Raises
    ------
    MetadataInvariantError
        If backward pass log keys are not exactly ``1..num_backward_passes``.
    """

    expected_pass_indices = list(range(1, trace.num_backward_passes + 1))
    actual_pass_indices = sorted(getattr(trace, "backward_pass_logs", {}).keys())
    if actual_pass_indices != expected_pass_indices:
        raise MetadataInvariantError(
            name,
            f"backward_pass_logs keys {actual_pass_indices!r} are not dense "
            f"1..{trace.num_backward_passes}",
        )


def _check_grad_fn_topology_invariants(trace: "Trace", name: str) -> None:
    """Check backward GradFn relation lists for reciprocal, resolvable links.

    The precondition contract is backward-capture only: callers invoke this
    helper after proving ``trace.grad_fn_logs`` is populated. Forward-only
    traces legitimately have no GradFn topology and are skipped by
    ``_check_backward_graph_invariants`` before this helper is reached.

    Parameters
    ----------
    trace:
        Trace with materialized backward GradFn projections.
    name:
        Invariant check name to use in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a GradFn relation references a missing node or lacks its reciprocal
        back-reference.
    """

    grad_fns_by_label = {
        grad_fn_handle.label: grad_fn_handle for grad_fn_handle in trace.grad_fn_logs.values()
    }
    grad_fn_ids = set(trace.grad_fn_logs)
    for grad_fn_handle in trace.grad_fn_logs.values():
        missing_next_ids = [
            next_id for next_id in grad_fn_handle.next_grad_fn_ids if next_id not in grad_fn_ids
        ]
        if missing_next_ids:
            raise MetadataInvariantError(
                name,
                f"{grad_fn_handle.label} has next_grad_fn_ids missing from grad_fn_logs: "
                f"{missing_next_ids!r}",
            )

        _check_grad_fn_relation_list(
            grad_fn_handle,
            "parents",
            "children",
            grad_fns_by_label,
            name,
        )
        _check_grad_fn_relation_list(
            grad_fn_handle,
            "children",
            "parents",
            grad_fns_by_label,
            name,
        )
        _check_grad_fn_relation_list(
            grad_fn_handle,
            "siblings",
            "siblings",
            grad_fns_by_label,
            name,
        )
        _check_grad_fn_relation_list(
            grad_fn_handle,
            "co_parents",
            "co_parents",
            grad_fns_by_label,
            name,
        )

        for flag_name, relation_name in (
            ("has_parents", "parents"),
            ("has_children", "children"),
            ("has_siblings", "siblings"),
            ("has_co_parents", "co_parents"),
        ):
            if getattr(grad_fn_handle, flag_name) != bool(getattr(grad_fn_handle, relation_name)):
                raise MetadataInvariantError(
                    name,
                    f"{grad_fn_handle.label} has inconsistent {flag_name}/{relation_name}",
                )


def _check_grad_fn_relation_list(
    grad_fn_handle: object,
    relation_name: str,
    reciprocal_name: str,
    grad_fns_by_label: Mapping[str, object],
    name: str,
) -> None:
    """Check that one GradFn relation list resolves and reciprocates.

    Parameters
    ----------
    grad_fn_handle:
        GradFn object whose relation list is being validated.
    relation_name:
        Name of the outbound relation list.
    reciprocal_name:
        Name of the relation list expected on each target.
    grad_fns_by_label:
        Mapping from GradFn label to GradFn object.
    name:
        Invariant check name to use in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a relation label is missing or lacks the reciprocal label.
    """

    source_label = str(getattr(grad_fn_handle, "label"))
    related_labels = getattr(grad_fn_handle, relation_name)
    if not isinstance(related_labels, list):
        raise MetadataInvariantError(
            name,
            f"{source_label} has non-list {relation_name}: {related_labels!r}",
        )
    for related_label in related_labels:
        related = grad_fns_by_label.get(related_label)
        if related is None:
            raise MetadataInvariantError(
                name,
                f"{source_label} {relation_name} references missing GradFn {related_label!r}",
            )
        reciprocal_labels = getattr(related, reciprocal_name)
        if source_label not in reciprocal_labels:
            raise MetadataInvariantError(
                name,
                f"{source_label} {relation_name} references {related_label!r}, but "
                f"{related_label!r} does not list {source_label!r} in {reciprocal_name}",
            )


def _check_backward_pass_domain_invariants(
    trace: "Trace",
    name: str,
    valid_pass_indices: set[int],
) -> None:
    """Check backward-pass domain fields and root coverage.

    The precondition contract is backward-capture only: this helper runs only
    after ``grad_fn_logs`` and dense ``backward_pass_logs`` have been proven
    present. It validates projected pass records against the event-domain
    literals used by the torch backward capture path.

    Parameters
    ----------
    trace:
        Trace with materialized backward-pass projections.
    name:
        Invariant check name to use in raised errors.
    valid_pass_indices:
        Dense set of known backward pass indices.

    Raises
    ------
    MetadataInvariantError
        If a BackwardPass field is outside its recorded domain or references a
        missing pass/GradFn.
    """

    valid_triggers = {
        "autograd_backward",
        "autograd_grad",
        "backward",
        "implicit",
        "recording_backward",
        "replay",
    }
    valid_statuses = {"error", "ok"}
    global_root_ids = set(trace.backward_root_grad_fn_object_ids)
    roots_seen_by_pass: set[int] = set()
    backward_pass_logs = getattr(trace, "backward_pass_logs", {})

    for pass_index, backward_pass in backward_pass_logs.items():
        if backward_pass.trigger not in valid_triggers:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has invalid trigger {backward_pass.trigger!r}",
            )
        if backward_pass.status not in valid_statuses:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has invalid status {backward_pass.status!r}",
            )
        if backward_pass.save_grads_policy is not None and not isinstance(
            backward_pass.save_grads_policy,
            str,
        ):
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has invalid save_grads_policy "
                f"{backward_pass.save_grads_policy!r}",
            )
        if backward_pass.duration is not None and float(backward_pass.duration) < 0:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has negative duration {backward_pass.duration!r}",
            )
        if backward_pass.peak_memory is not None and backward_pass.peak_memory < 0:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has negative peak_memory {backward_pass.peak_memory!r}",
            )
        coverage = backward_pass.order_attribution_coverage
        if coverage is not None and not 0.0 <= coverage <= 1.0:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} has invalid order_attribution_coverage {coverage!r}",
            )
        origin_pass = backward_pass.origin_backward_pass
        if origin_pass is not None and origin_pass not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} references missing origin_backward_pass "
                f"{origin_pass!r}",
            )

        missing_root_ids = [
            root_id
            for root_id in backward_pass.root_grad_fn_ids
            if root_id not in trace.grad_fn_logs
        ]
        if missing_root_ids:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} root_grad_fn_ids are missing from grad_fn_logs: "
                f"{missing_root_ids!r}",
            )
        roots_seen_by_pass.update(backward_pass.root_grad_fn_ids)

    if len(backward_pass_logs) == 1:
        pass_index, backward_pass = next(iter(backward_pass_logs.items()))
        if set(backward_pass.root_grad_fn_ids) != global_root_ids:
            raise MetadataInvariantError(
                name,
                f"BackwardPass {pass_index} root_grad_fn_ids do not match trace roots",
            )
    elif roots_seen_by_pass != global_root_ids:
        raise MetadataInvariantError(
            name,
            "BackwardPass root_grad_fn_ids union does not match trace roots",
        )


def _layer_postdates_all_backward_triggers(trace: "Trace", layer: "Layer | Op") -> bool:
    """Return whether a layer was created after every recorded backward trigger.

    Parameters
    ----------
    trace:
        Trace containing backward sidecar events.
    layer:
        Layer whose forward position is being checked.

    Returns
    -------
    bool
        True only when every recorded ``BackwardPassStart`` has a structural
        forward-op boundary and ``layer.step_index`` is beyond all of them.
    """

    layer_step_index = getattr(layer, "step_index", None)
    if not isinstance(layer_step_index, int):
        return False
    trigger_positions = _backward_trigger_forward_positions(trace)
    return bool(trigger_positions) and layer_step_index > max(trigger_positions)


def _backward_trigger_forward_positions(trace: "Trace") -> list[int]:
    """Return strict forward boundaries for recorded backward triggers.

    Parameters
    ----------
    trace:
        Trace containing backward sidecar events and projections.

    Returns
    -------
    list[int]
        One forward boundary per trigger with enough structural metadata. Root
        GradFn pairings refine active mid-forward markers because the root's
        forward op is the last layer guaranteed to exist when graph walking ran.
    """

    from ..ir.events import BackwardPassStart

    event_positions = {
        event.pass_index: event.forward_op_count_at_trigger
        for event in getattr(getattr(trace, "_capture_events", None), "backward_events", ())
        if isinstance(event, BackwardPassStart)
        and isinstance(event.forward_op_count_at_trigger, int)
    }
    positions: list[int] = []
    for pass_index, event_position in event_positions.items():
        structural_positions = [
            position
            for position in (
                _backward_pass_root_forward_position(trace, pass_index),
                _backward_pass_observed_forward_position(trace, pass_index),
            )
            if position is not None
        ]
        if not structural_positions:
            positions.append(event_position)
        else:
            positions.append(min(event_position, *structural_positions))
    return positions


def _backward_pass_root_forward_position(trace: "Trace", pass_index: int) -> int | None:
    """Return the highest paired forward position among a backward pass's roots.

    Parameters
    ----------
    trace:
        Trace with materialized backward projections.
    pass_index:
        One-based backward pass index.

    Returns
    -------
    int | None
        Highest root-paired forward ``step_index`` for the pass, when available.
    """

    backward_pass = getattr(trace, "backward_pass_logs", {}).get(pass_index)
    if backward_pass is None:
        return None
    root_steps = [
        step_index
        for root_id in getattr(backward_pass, "root_grad_fn_ids", ())
        if isinstance(
            (
                step_index := getattr(
                    getattr(getattr(trace, "grad_fn_logs", {}).get(root_id), "op", None),
                    "step_index",
                    None,
                )
            ),
            int,
        )
    ]
    return max(root_steps) if root_steps else None


def _backward_pass_observed_forward_position(trace: "Trace", pass_index: int) -> int | None:
    """Return the highest forward position with an observed gradient in a pass.

    Parameters
    ----------
    trace:
        Trace with backward sidecar events.
    pass_index:
        One-based backward pass index.

    Returns
    -------
    int | None
        Highest observed forward ``step_index`` for the pass, when available.
    """

    from ..ir.events import OpGradObserved

    layer_lookup = getattr(trace, "layer_dict_all_keys", {})
    observed_steps = []
    for event in getattr(getattr(trace, "_capture_events", None), "backward_events", ()):
        if not isinstance(event, OpGradObserved) or event.pass_index != pass_index:
            continue
        final_label = _resolve_op_grad_event_label(trace, event.op_label)
        layer = layer_lookup.get(final_label)
        step_index = getattr(layer, "step_index", None)
        if isinstance(step_index, int):
            observed_steps.append(step_index)
    return max(observed_steps) if observed_steps else None


def _resolve_op_grad_event_label(trace: "Trace", op_label: str) -> str:
    """Return the final lookup label for an ``OpGradObserved`` label.

    Parameters
    ----------
    trace:
        Trace containing postprocessed label maps.
    op_label:
        Raw or final label recorded by the tensor hook.

    Returns
    -------
    str
        Final lookup label when available, otherwise the original label.
    """

    raw_to_final_layer = getattr(trace, "_raw_to_final_layer_labels", {})
    if isinstance(raw_to_final_layer, dict) and op_label in raw_to_final_layer:
        return str(raw_to_final_layer[op_label])
    raw_to_final_op = getattr(trace, "_raw_to_final_op_labels", {})
    if isinstance(raw_to_final_op, dict) and op_label in raw_to_final_op:
        return str(raw_to_final_op[op_label])
    return op_label


def _check_backward_event_flow_invariants(trace: "Trace", name: str) -> None:
    """Check runtime backward event stream consistency against projections.

    Parameters
    ----------
    trace:
        Postprocessed model log to validate.
    name:
        Invariant check name to use in raised errors.

    Raises
    ------
    MetadataInvariantError
        If runtime backward sidecar events are internally inconsistent or no
        longer match projected records.
    """

    from ..ir.events import BackwardPassEnd, BackwardPassStart, GradFnFired, OpGradObserved

    capture_events = getattr(trace, "_capture_events", None)
    events = list(getattr(capture_events, "backward_events", ()) or ())
    if not events:
        return

    starts = [event for event in events if isinstance(event, BackwardPassStart)]
    ends = [event for event in events if isinstance(event, BackwardPassEnd)]
    op_grad_events = [event for event in events if isinstance(event, OpGradObserved)]
    fired_events = [event for event in events if isinstance(event, GradFnFired)]
    start_indices = [event.pass_index for event in starts]
    end_indices = [event.pass_index for event in ends]
    if len(start_indices) != len(set(start_indices)):
        raise MetadataInvariantError(name, "backward events contain duplicate pass starts")
    if len(end_indices) != len(set(end_indices)):
        raise MetadataInvariantError(name, "backward events contain duplicate pass ends")
    bracket_indices = sorted(set(start_indices) | set(end_indices))
    if bracket_indices != list(range(1, len(bracket_indices) + 1)):
        raise MetadataInvariantError(
            name,
            f"backward event pass indices {bracket_indices!r} are not dense from 1",
        )
    if set(start_indices) != set(end_indices):
        raise MetadataInvariantError(
            name,
            "backward events must contain exactly one start and end per pass",
        )

    valid_pass_indices = set(bracket_indices)
    for op_grad_event in op_grad_events:
        if op_grad_event.pass_index not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"backward event references missing pass {op_grad_event.pass_index!r}",
            )
    for fired_event in fired_events:
        if fired_event.pass_index not in valid_pass_indices:
            raise MetadataInvariantError(
                name,
                f"backward event references missing pass {fired_event.pass_index!r}",
            )

    seq_values = [event.seq for event in events if isinstance(event, OpGradObserved | GradFnFired)]
    if seq_values != sorted(seq_values) or len(seq_values) != len(set(seq_values)):
        raise MetadataInvariantError(name, "backward event seq values must be unique and monotonic")

    layer_labels = set(getattr(trace, "layer_dict_all_keys", {}))
    projected_grad_records: set[tuple[str, int]] = set()
    for layer in getattr(trace, "layer_list", []):
        for record in getattr(layer, "_grad_records", ()):
            projected_grad_records.add((layer.layer_label, record.backward_pass_index))

    event_grad_records: set[tuple[str, int]] = set()
    for op_grad_event in op_grad_events:
        event_label = _resolve_op_grad_event_label(trace, op_grad_event.op_label)
        if event_label not in layer_labels:
            raise MetadataInvariantError(
                name,
                f"OpGradObserved points to missing op label {op_grad_event.op_label!r}",
            )
        event_op = trace[event_label]
        event_grad_records.add((event_op.layer_label, op_grad_event.pass_index))
    if projected_grad_records != event_grad_records:
        raise MetadataInvariantError(
            name,
            "projected op gradient records do not match OpGradObserved events",
        )

    projected_calls: dict[tuple[int, int], int] = defaultdict(int)
    for grad_fn_handle in getattr(trace, "grad_fn_logs", {}).values():
        for call in grad_fn_handle.calls.values():
            projected_calls[(grad_fn_handle.grad_fn_object_id, call.backward_pass_index)] += 1
    event_calls: dict[tuple[int, int], int] = defaultdict(int)
    grad_fn_ids = set(getattr(trace, "grad_fn_logs", {}))
    for fired_event in fired_events:
        if fired_event.object_id not in grad_fn_ids:
            raise MetadataInvariantError(
                name,
                f"GradFnFired points to missing grad_fn id {fired_event.object_id!r}",
            )
        event_calls[(fired_event.object_id, fired_event.pass_index)] += 1
    if projected_calls != event_calls:
        raise MetadataInvariantError(
            name,
            "projected grad_fn calls do not match GradFnFired events",
        )


def _is_func_call_id_exempt(layer: "Op") -> bool:
    """Return whether a layer is exempt from Invariant S.

    Parameters
    ----------
    layer:
        Layer pass to classify.

    Returns
    -------
    bool
        Whether the layer is synthetic input/output/buffer metadata.
    """

    if layer.is_input or layer.is_output or layer.is_buffer:
        return True
    func = getattr(layer, "func", None)
    func_name = str(getattr(layer, "func_name", "")).lower()
    return func in {"input", "output", "buffer"} or func_name in {
        "input",
        "output",
        "buffer",
        "none",
    }


def _plain_func_call_group_signature(layer: "Op") -> tuple[object, ...]:
    """Return plain-capture-stable same-call metadata.

    Parameters
    ----------
    layer:
        Layer pass to summarize.

    Returns
    -------
    tuple[object, ...]
        Function name and container spec representation for same-call grouping.
    """

    return (
        getattr(layer, "func_name", None),
        repr(getattr(layer, "container_spec", None)),
    )


def _func_call_group_signature(layer: "Op") -> tuple[object, ...]:
    """Return comparable same-call metadata for Invariant S.

    Parameters
    ----------
    layer:
        Layer pass to summarize.

    Returns
    -------
    tuple[object, ...]
        Stable comparison tuple.
    """

    return (
        layer.func_name,
        tuple(repr(location) for location in (layer.code_context or ())),
        repr(layer.args_template),
        repr(layer.kwargs_template),
        repr(layer.container_spec),
    )


# ---------------------------------------------------------------------------
# A. Trace self-consistency
# ---------------------------------------------------------------------------


def _check_trace_self_consistency(ml: "Trace") -> None:
    """Check A: Trace aggregate counts and metadata are internally consistent.

    Validates:
    - layer_labels length matches layer_list length, no duplicates.
    - num_ops == count of computational (non-input, non-output,
      non-buffer) layers.
    - Param counts (total, trainable, frozen) are consistent and sum correctly.
      Uses deduplication by layer_label to match labeling.py logic.
    - At least one output layer exists.
    - Timing values are non-negative and ordered.
    - Tensor counts: total >= saved.
    """
    name = "trace_self_consistency"

    # op_labels vs layer_list length
    if len(ml.op_labels) != len(ml.layer_list):
        raise MetadataInvariantError(
            name,
            f"len(op_labels)={len(ml.op_labels)} != len(layer_list)={len(ml.layer_list)}",
        )

    # No duplicate labels
    if len(ml.op_labels) != len(set(ml.op_labels)):
        dupes = [lbl for lbl in ml.op_labels if ml.op_labels.count(lbl) > 1]
        raise MetadataInvariantError(name, f"Duplicate op_labels: {set(dupes)}")

    # num_ops counts computational layers only (excludes input, output,
    # buffer).  We check per-layer flags instead of comparing against label
    # sets because buffer_layers stores pass-qualified labels while
    # layer_labels strips the pass suffix -- they use different formats.
    expected_ops = sum(
        1 for lpl in ml.layer_list if not (lpl.is_input or lpl.is_output or lpl.is_buffer)
    )
    if ml.num_ops != expected_ops:
        raise MetadataInvariantError(
            name,
            f"num_ops={ml.num_ops} != expected computational layers={expected_ops}",
        )

    # Param counts must be deduplicated by layer_label because
    # multi-pass layers share the same params -- counting each pass would
    # double-count.  This matches the summation logic in labeling.py:116-122.
    seen_no_pass: set[str] = set()
    expected_param_sum = 0
    expected_num_params = 0
    for lpl in ml.layer_list:
        if lpl.layer_label not in seen_no_pass:
            expected_param_sum += lpl.num_param_tensors
            expected_num_params += lpl.num_params
            seen_no_pass.add(lpl.layer_label)
    if ml.num_param_tensors != expected_param_sum:
        raise MetadataInvariantError(
            name,
            f"num_param_tensors={ml.num_param_tensors} != "
            f"sum(unique num_param_tensors)={expected_param_sum}",
        )
    if ml.num_params != expected_num_params:
        raise MetadataInvariantError(
            name,
            f"num_params={ml.num_params} != sum(unique num_params)={expected_num_params}",
        )

    if ml.num_params_trainable + ml.num_params_frozen != ml.num_params:
        raise MetadataInvariantError(
            name,
            f"trainable({ml.num_params_trainable}) + frozen({ml.num_params_frozen}) "
            f"!= total({ml.num_params})",
        )

    # At least one output layer
    if len(ml.output_layers) == 0:
        raise MetadataInvariantError(name, "No output layers found")

    # Timing
    if ml.capture_duration < 0:
        raise MetadataInvariantError(name, f"capture_duration={ml.capture_duration} < 0")
    if ml.capture_start_time > ml.capture_end_time:
        raise MetadataInvariantError(
            name,
            f"capture_start_time={ml.capture_start_time} > capture_end_time={ml.capture_end_time}",
        )

    # Tensor counts
    if ml.num_tensors < ml.num_saved_ops:
        raise MetadataInvariantError(
            name,
            f"num_tensors={ml.num_tensors} < num_saved_ops={ml.num_saved_ops}",
        )


# ---------------------------------------------------------------------------
# B. Special layer lists ↔ Op flags
# ---------------------------------------------------------------------------

_SPECIAL_LIST_FLAG_PAIRS = [
    ("input_layers", "is_input", "layer"),
    ("output_layers", "is_output", "layer"),
    ("buffer_layers", "is_buffer", "layer"),
    ("internal_source_ops", "is_internal_source", "op"),
    ("internal_sink_ops", "is_internal_sink", "op"),
]


def _check_special_layer_lists(ml: "Trace") -> None:
    """Check B: special layer lists (input, output, buffer, etc.) match per-layer boolean flags.

    For each (list_attr, flag_attr) pair, verifies bidirectional consistency:
    - Forward: every label in the list has the flag set on its Op.
    - Reverse: every Op with the flag set appears in the list.
    """
    name = "special_layer_lists"
    for list_attr, flag_attr, label_kind in _SPECIAL_LIST_FLAG_PAIRS:
        special_list = getattr(ml, list_attr)
        special_set = set(special_list)
        label_set = set(ml.op_labels if label_kind == "op" else ml.layer_labels)
        label_field = "op_labels" if label_kind == "op" else "layer_labels"

        # All entries must be valid labels
        missing = special_set - label_set
        if missing:
            raise MetadataInvariantError(
                name, f"{list_attr} contains labels not in {label_field}: {missing}"
            )

        # Forward: every label in the list has the flag set
        for label in special_list:
            lpl = ml[label]
            if not getattr(lpl, flag_attr):
                raise MetadataInvariantError(
                    name,
                    f"{label_kind.title()} {label} is in {list_attr} but {flag_attr}=False",
                )

        # Reverse: every layer/op with the flag is in the list.
        for lpl in ml.layer_list:
            label = lpl.label if label_kind == "op" else lpl.layer_label
            if getattr(lpl, flag_attr) and label not in special_set:
                raise MetadataInvariantError(
                    name,
                    f"{label_kind.title()} {label} has {flag_attr}=True but is not in {list_attr}",
                )


# ---------------------------------------------------------------------------
# C. Graph topology
# ---------------------------------------------------------------------------


def _check_graph_topology(ml: "Trace") -> None:
    """Check C: parent-child edge bidirectionality and boolean flag consistency.

    Validates:
    - Every parent edge has a corresponding child edge (and vice versa).
    - has_children/has_parents/has_siblings/has_co_parents flags match actual counts.
      Note: has_children excludes output layers (added during postprocessing,
      not during capture when the flag was set).
    - Input layers have no parents.
    - out_versions_by_child keys are a subset of children.
    """
    name = "graph_topology"
    label_set = set(ml.layer_labels) | set(ml.op_labels)
    output_set = set(ml.output_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Parent-child bidirectionality
        for p in lpl.parents:
            parent = ml[p]
            if p not in label_set and parent.layer_label not in label_set:
                raise MetadataInvariantError(
                    name, f"Layer {label} has parent {p} not in layer_labels"
                )
            if label not in parent.children and lpl.label not in parent.children:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {p} as parent, but {p} does not list {label} as child",
                )

        for c in lpl.children:
            child = ml[c]
            if c not in label_set and child.layer_label not in label_set:
                raise MetadataInvariantError(
                    name, f"Layer {label} has child {c} not in layer_labels"
                )
            if label not in child.parents and lpl.label not in child.parents:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {c} as child, but {c} does not list {label} as parent",
                )

        # Boolean flag consistency
        # Note: has_children/has_parents/has_siblings/has_co_parents are set during
        # capture and don't account for output layers added during postprocessing.
        # So we exclude output layers from the child count for this check.
        non_output_children = [
            c for c in lpl.children if c not in output_set and ml[c].layer_label not in output_set
        ]
        if lpl.has_children != (len(non_output_children) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_children={lpl.has_children} but "
                f"non-output children={non_output_children}",
            )
        if not lpl.is_output and lpl.has_parents != (len(lpl.parents) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_parents={lpl.has_parents} but len(parents)={len(lpl.parents)}",
            )
        if lpl.has_siblings != (len(lpl.siblings) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_siblings={lpl.has_siblings} but "
                f"len(siblings)={len(lpl.siblings)}",
            )
        if lpl.has_co_parents != (len(lpl.co_parents) > 0):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: has_co_parents={lpl.has_co_parents} but "
                f"len(co_parents)={len(lpl.co_parents)}",
            )

        # Input layers have no parents
        if lpl.is_input and len(lpl.parents) > 0:
            raise MetadataInvariantError(
                name,
                f"Input layer {label} has parents={lpl.parents}",
            )

        # out_versions_by_child keys subset of children
        ctv_keys = set(lpl.out_versions_by_child.keys())
        child_set = set(lpl.children)
        extra = ctv_keys - child_set
        if extra:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: out_versions_by_child has keys not in children: {extra}",
            )


def _check_backend_neutral_graph_topology(ml: "Trace") -> None:
    """Check parent/child symmetry for non-torch traces where fields exist.

    Parameters
    ----------
    ml:
        Postprocessed non-torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a populated parent or child list references a missing layer or lacks
        the reciprocal edge.
    """

    name = "backend_neutral_graph_topology"
    labels = {
        getattr(layer, "label", getattr(layer, "layer_label", ""))
        for layer in getattr(ml, "layer_list", ())
    } | {
        getattr(layer, "layer_label", getattr(layer, "label", ""))
        for layer in getattr(ml, "layer_list", ())
    }
    for layer in getattr(ml, "layer_list", ()):
        label = getattr(layer, "layer_label", getattr(layer, "label", type(layer).__name__))
        parents = list(getattr(layer, "parents", ()) or ())
        children = list(getattr(layer, "children", ()) or ())
        for parent_label in parents:
            if parent_label not in labels:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} has parent {parent_label!r} outside trace labels",
                )
            parent = ml[parent_label]
            parent_children = set(getattr(parent, "children", ()) or ())
            if (
                label not in parent_children
                and getattr(layer, "label", label) not in parent_children
            ):
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {parent_label!r} as parent, but reciprocal child is missing",
                )
        for child_label in children:
            if child_label not in labels:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} has child {child_label!r} outside trace labels",
                )
            child = ml[child_label]
            child_parents = set(getattr(child, "parents", ()) or ())
            if label not in child_parents and getattr(layer, "label", label) not in child_parents:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label} lists {child_label!r} as child, but reciprocal parent is missing",
                )


def _check_edge_use_parent_arg_invariants(ml: "Trace") -> None:
    """Check existing edge-use records and parent-arg references.

    Precondition contract: edge-use metadata is optional on torch graph edges.
    The torch eager builder emits ``_edge_uses`` only for args/kwargs-derived
    parent entries. Buffer-source, output, control, module, and
    intervention-injected edges may legitimately have no edge-use record. When
    an ``_edge_uses`` record exists, its kind must be in ``EdgeUseKind`` and
    its parent/child labels must resolve. When a ``parent_arg_positions`` entry
    exists, its referenced parent label must resolve. This invariant never
    asserts that every parent edge has a corresponding edge-use record.

    Parameters
    ----------
    ml:
        Postprocessed torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If populated edge-use or parent-arg-position metadata is malformed or
        references labels that do not resolve.
    """

    name = "edge_use_parent_arg_consistency"
    valid_edge_uses = {"arg", "kwarg", "container", "module", "buffer", "output", "control"}
    valid_arg_kinds = {"positional", "keyword"}
    for layer in ml.layer_list:
        layer_label = getattr(layer, "layer_label", type(layer).__name__)
        for record in getattr(layer, "_edge_uses", ()) or ():
            edge_use = getattr(record, "edge_use", None)
            if edge_use not in valid_edge_uses:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has invalid edge_use kind {edge_use!r}",
                )
            arg_kind = getattr(record, "arg_kind", None)
            if arg_kind not in valid_arg_kinds:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has invalid edge arg_kind {arg_kind!r}",
                )
            parent_label = getattr(record, "parent_label", None)
            if not isinstance(parent_label, str) or _resolve_trace_label(ml, parent_label) is None:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has edge-use record with unresolved parent "
                    f"{parent_label!r}",
                )
            child_label = getattr(record, "child_label", None)
            if not isinstance(child_label, str) or _resolve_trace_label(ml, child_label) is None:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' has edge-use record with unresolved child "
                    f"{child_label!r}",
                )

        parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
        if not isinstance(parent_arg_positions, Mapping):
            raise MetadataInvariantError(
                name,
                f"Layer '{layer_label}' has non-mapping parent_arg_positions",
            )
        for arg_domain in ("args", "kwargs"):
            entries = parent_arg_positions.get(arg_domain, {}) or {}
            if not isinstance(entries, Mapping):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}] is not a mapping",
                )
            for position, parent_label in entries.items():
                if not isinstance(parent_label, str):
                    raise MetadataInvariantError(
                        name,
                        f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}]"
                        f"[{position!r}] is not a label string",
                    )
                if _resolve_trace_label(ml, parent_label) is None:
                    raise MetadataInvariantError(
                        name,
                        f"Layer '{layer_label}' parent_arg_positions[{arg_domain!r}]"
                        f"[{position!r}] references missing parent {parent_label!r}",
                    )


# ---------------------------------------------------------------------------
# D. Op field consistency
# ---------------------------------------------------------------------------


def _check_op_log_fields(ml: "Trace") -> None:
    """Check D: per-layer field consistency (shape, dtype, pass numbering, func, nesting).

    Validates:
    - Saved tensor shape/dtype match actual out (when saved).
    - Pass numbering: pass_index >= 1, num_passes >= pass_index.
    - Computational layers have callable func and non-empty func_name.
    - step_index >= 1 for non-input/non-buffer layers.
    - module_call_depth matches len(modules).
    - Label format: pass-qualified label has ':' iff multi-pass; no-pass label never has ':'.
    """
    name = "op_log_fields"

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Tensor shape/dtype consistency when outs are saved
        if lpl.has_saved_activation and lpl.out is not None:
            actual_shape = tuple(lpl.out.shape)
            if lpl.shape != actual_shape:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: shape={lpl.shape} != actual shape={actual_shape}",
                )
            if lpl.dtype != lpl.out.dtype:
                raise MetadataInvariantError(
                    name,
                    f"Layer {label}: dtype={lpl.dtype} != actual dtype={lpl.out.dtype}",
                )

        # Pass numbering
        if lpl.pass_index < 1:
            raise MetadataInvariantError(name, f"Layer {label}: pass_index={lpl.pass_index} < 1")
        if lpl.num_passes < lpl.pass_index:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: num_passes={lpl.num_passes} < pass_index={lpl.pass_index}",
            )

        # A GENUINE raw-forward-hook output replacement is legitimately
        # functionless: the user substituted an opaque tensor for a module's
        # output, so there is no torch function to validate. This exemption is
        # deliberately narrow -- it must NOT cover auto-synthesized placeholders
        # during plain capture (a previous band-aid widened it to silence the
        # vmap-built attention mask, disarming this tripwire).
        is_functionless_replacement = (
            lpl.func_name == "intervention_replacement"
            and getattr(lpl, "intervention_replaced", False)
            and not getattr(lpl, "is_internal_source", False)
        )

        # An internally generated *source* tensor whose construction TorchLens
        # could not trace (e.g. an attention mask built inside torch.vmap) is a
        # genuine functionless graph source, exactly like a buffer: func is None
        # and func_name is "none". Traced ops that merely have an internal-source
        # ancestor still carry a real callable func and are NOT exempted here.
        is_functionless_internal_source = (
            getattr(lpl, "is_internal_source", False) and lpl.func is None
        )

        # Function applied (non-input, non-buffer, non-output, non-source,
        # non-hook-replacement layers).
        if not (
            lpl.is_input
            or lpl.is_buffer
            or lpl.is_output
            or is_functionless_internal_source
            or is_functionless_replacement
        ):
            if not callable(lpl.func):
                raise MetadataInvariantError(name, f"Layer {label}: func is not callable")
            if not lpl.func_name:
                raise MetadataInvariantError(name, f"Layer {label}: func_name is empty")

        # Operation numbering (input/buffer/output bookkeeping layers have step_index=0)
        if not (lpl.is_input or lpl.is_buffer or lpl.is_output):
            if lpl.step_index is not None and lpl.step_index < 1:
                raise MetadataInvariantError(
                    name, f"Layer {label}: step_index={lpl.step_index} < 1"
                )
        if lpl.raw_index < 1:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: raw_index={lpl.raw_index} < 1",
            )

        # Module nesting depth
        if lpl.module_call_depth != len(lpl.modules):
            raise MetadataInvariantError(
                name,
                f"Layer {label}: module_call_depth={lpl.module_call_depth} != "
                f"len(modules)="
                f"{len(lpl.modules)}",
            )

        # Label format: pass-qualified label has ":" iff multi-pass
        if lpl.num_passes > 1 and ":" not in lpl.label:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: multi-pass but label='{lpl.label}' has no ':'",
            )
        if ":" in lpl.layer_label:
            raise MetadataInvariantError(
                name,
                f"Layer {label}: layer_label='{lpl.layer_label}' contains ':'",
            )


def _check_payload_metadata_invariants(ml: "Trace") -> None:
    """Check saved and transformed live payload metadata.

    Precondition contract: tensor payload fields may be legitimately absent
    because of selective save, loaded traces, detached/audit-only metadata,
    disk-only storage, streaming finalization, or gradient eviction. This check
    compares shape, dtype, and memory only when a live payload object is
    present. Presence of a live raw or transformed activation requires
    ``has_saved_activation=True``; presence of a live raw or transformed
    gradient requires ``has_grad=True``. Missing payloads never imply
    corruption by themselves.

    Parameters
    ----------
    ml:
        Postprocessed torch trace to validate.

    Raises
    ------
    MetadataInvariantError
        If a present payload disagrees with its recorded metadata.
    """

    name = "payload_metadata_invariants"
    for op in ml.layer_list:
        label = getattr(op, "label", getattr(op, "layer_label", type(op).__name__))
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "out"),
            shape=getattr(op, "shape", None),
            dtype=getattr(op, "dtype", None),
            memory=getattr(op, "activation_memory", None),
            presence_flag=getattr(op, "has_saved_activation", False),
            presence_flag_name="has_saved_activation",
            payload_name="out",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "transformed_out"),
            shape=getattr(op, "transformed_out_shape", None),
            dtype=getattr(op, "transformed_out_dtype", None),
            memory=getattr(op, "transformed_activation_memory", None),
            presence_flag=getattr(op, "has_saved_activation", False),
            presence_flag_name="has_saved_activation",
            payload_name="transformed_out",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "grad"),
            shape=getattr(op, "grad_shape", None),
            dtype=getattr(op, "grad_dtype", None),
            memory=getattr(op, "gradient_memory", None),
            presence_flag=getattr(op, "has_grad", False),
            presence_flag_name="has_grad",
            payload_name="grad",
        )
        _check_live_payload_metadata(
            name,
            label,
            payload=_live_payload_value(op, "transformed_grad"),
            shape=getattr(op, "transformed_grad_shape", None),
            dtype=getattr(op, "transformed_grad_dtype", None),
            memory=getattr(op, "transformed_gradient_memory", None),
            presence_flag=getattr(op, "has_grad", False),
            presence_flag_name="has_grad",
            payload_name="transformed_grad",
        )
        for record in getattr(op, "_grad_records", ()) or ():
            record_label = f"{label}.grad_record[{getattr(record, 'backward_pass_index', '?')}]"
            _check_live_payload_metadata(
                name,
                record_label,
                payload=_live_payload_value(record, "grad"),
                shape=getattr(record, "shape", None),
                dtype=getattr(record, "dtype", None),
                memory=getattr(record, "memory", None),
                presence_flag=getattr(record, "is_saved", False),
                presence_flag_name="is_saved",
                payload_name="grad",
            )
            _check_live_payload_metadata(
                name,
                record_label,
                payload=_live_payload_value(record, "transformed_grad"),
                shape=getattr(record, "transformed_grad_shape", None),
                dtype=getattr(record, "transformed_grad_dtype", None),
                memory=getattr(record, "transformed_gradient_memory", None),
                presence_flag=getattr(record, "is_saved", False),
                presence_flag_name="is_saved",
                payload_name="transformed_grad",
            )


def _live_payload_value(owner: object, payload_name: str) -> object | None:
    """Return an already-live payload without invoking guarded payload accessors.

    Parameters
    ----------
    owner:
        Object that owns the payload field.
    payload_name:
        Name of the payload field to inspect.

    Returns
    -------
    object or None
        The live payload object, or ``None`` when no payload is currently attached.
    """

    slot_getter = getattr(owner, "_slot", None)
    if callable(slot_getter):
        return slot_getter(payload_name, None)
    return getattr(owner, payload_name, None)


def _check_live_payload_metadata(
    name: str,
    label: str,
    *,
    payload: object | None,
    shape: object,
    dtype: object,
    memory: object,
    presence_flag: bool,
    presence_flag_name: str,
    payload_name: str,
) -> None:
    """Check metadata for one live tensor-like payload.

    Parameters
    ----------
    name:
        Invariant name used in raised errors.
    label:
        Owner label for diagnostics.
    payload:
        Live payload object, or ``None`` when absent.
    shape:
        Recorded shape metadata.
    dtype:
        Recorded dtype metadata.
    memory:
        Recorded memory metadata.
    presence_flag:
        Boolean metadata that should be true when payload is present.
    presence_flag_name:
        Name of ``presence_flag`` for diagnostics.
    payload_name:
        Payload field name for diagnostics.

    Raises
    ------
    MetadataInvariantError
        If present payload metadata disagrees with the payload.
    """

    if payload is None:
        return
    if not presence_flag:
        raise MetadataInvariantError(
            name,
            f"{label} has live {payload_name} payload but {presence_flag_name}=False",
        )
    actual_shape = _payload_shape(payload)
    if actual_shape is not None and shape != actual_shape:
        raise MetadataInvariantError(
            name,
            f"{label} {payload_name} shape metadata {shape!r} != payload shape {actual_shape!r}",
        )
    actual_dtype = _payload_dtype(payload)
    if (
        actual_dtype is not None
        and dtype is not None
        and not _dtype_values_match(dtype, actual_dtype)
    ):
        raise MetadataInvariantError(
            name,
            f"{label} {payload_name} dtype metadata {dtype!r} != payload dtype {actual_dtype!r}",
        )
    actual_memory = _payload_memory(payload)
    if actual_memory is not None and memory is not None:
        if not isinstance(memory, int):
            raise MetadataInvariantError(
                name,
                f"{label} {payload_name} memory metadata {memory!r} is not an integer",
            )
        if memory != actual_memory:
            raise MetadataInvariantError(
                name,
                f"{label} {payload_name} memory metadata {memory!r} != payload memory "
                f"{actual_memory!r}",
            )


def _payload_shape(payload: object) -> tuple[int, ...] | None:
    """Return a tuple shape for a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    tuple[int, ...] | None
        Shape tuple when available.
    """

    shape = getattr(payload, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _payload_dtype(payload: object) -> object | None:
    """Return dtype metadata from a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    object | None
        Payload dtype when available.
    """

    return getattr(payload, "dtype", None)


def _payload_memory(payload: object) -> int | None:
    """Return byte memory for a tensor-like payload.

    Parameters
    ----------
    payload:
        Candidate tensor-like payload.

    Returns
    -------
    int | None
        Number of bytes when ``nelement`` and ``element_size`` are available.
    """

    nelement = getattr(payload, "nelement", None)
    element_size = getattr(payload, "element_size", None)
    if not callable(nelement) or not callable(element_size):
        return None
    return int(nelement() * element_size())


def _dtype_values_match(left: object, right: object) -> bool:
    """Return whether two dtype representations are equivalent.

    Parameters
    ----------
    left:
        Recorded dtype value.
    right:
        Payload dtype value.

    Returns
    -------
    bool
        ``True`` when exact or normalized string forms agree.
    """

    if left == right:
        return True
    left_str = str(left).replace("torch.", "")
    right_str = str(right).replace("torch.", "")
    return left_str == right_str


# ---------------------------------------------------------------------------
# E. Recurrence / loop invariants
# ---------------------------------------------------------------------------


def _check_recurrence_invariants(ml: "Trace") -> None:
    """Check E: recurrence / loop invariants.

    Validates:
    - is_recurrent == True iff any layer has >1 pass.
    - max_layer_op_count matches the maximum pass count.
    - layer_num_calls keys are valid no-pass labels.
    - Layer.ops dict keys are contiguous {1..N}.
    """
    name = "recurrence_invariants"

    any_recurrent = any(v > 1 for v in ml.layer_num_calls.values())
    if ml.is_recurrent != any_recurrent:
        raise MetadataInvariantError(
            name,
            f"is_recurrent={ml.is_recurrent} but any layer has >1 pass = {any_recurrent}",
        )

    if ml.is_recurrent:
        expected_max = max(ml.layer_num_calls.values())
        if ml.max_layer_op_count != expected_max:
            raise MetadataInvariantError(
                name,
                f"max_layer_op_count={ml.max_layer_op_count} != "
                f"max(layer_num_calls)={expected_max}",
            )

    # Per-layer pass consistency: layer_num_calls is keyed by no-pass labels.
    # Validate that each key exists in layer_labels, and that the
    # recorded count matches the actual Layer.num_passes.
    no_call_labels = set(ml.layer_labels)
    for label_key, num_calls in ml.layer_num_calls.items():
        if label_key not in no_call_labels:
            raise MetadataInvariantError(
                name,
                f"layer_num_calls key '{label_key}' not in layer_labels",
            )
        if label_key in ml.layer_logs:
            actual = ml.layer_logs[label_key].num_passes
            if num_calls != actual:
                raise MetadataInvariantError(
                    name,
                    f"layer_num_calls['{label_key}']={num_calls} != Layer.num_passes={actual}",
                )

    # For top-level (no-pass) layer_logs, verify pass dict consistency
    for no_call_label, ll in ml.layer_logs.items():
        expected_keys = set(range(1, ll.num_passes + 1))
        actual_keys = set(ll.ops.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Layer '{no_call_label}' ops keys={actual_keys} != expected {expected_keys}",
            )


# ---------------------------------------------------------------------------
# F. Branching invariants
# ---------------------------------------------------------------------------


def _check_branching_invariants(ml: "Trace") -> None:
    """Check F: is_branching matches whether any layer has >1 child."""
    name = "branching_invariants"
    any_branching = any(len(lpl.children) > 1 for lpl in ml.layer_list)
    if ml.is_branching != any_branching:
        raise MetadataInvariantError(
            name,
            f"is_branching={ml.is_branching} but any layer has >1 child = {any_branching}",
        )


# ---------------------------------------------------------------------------
# F2. Conditional metadata invariants
# ---------------------------------------------------------------------------


def _fail_conditional_invariant(check_name: str, number: int, message: str) -> None:
    """Raise a numbered conditional metadata invariant failure.

    Parameters
    ----------
    check_name:
        ``MetadataInvariantError.check_name`` value for this check family.
    number:
        Conditional invariant number from the Phase 6 plan.
    message:
        Human-readable failure details.
    """

    raise MetadataInvariantError(check_name, f"Invariant {number}: {message}")


def _strip_pass_suffix(layer_label: str) -> str:
    """Return a layer label without any trailing ``:call_index`` suffix.

    Parameters
    ----------
    layer_label:
        Layer label that may include a pass suffix.

    Returns
    -------
    str
        Pass-stripped label.
    """

    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return label_parts[0]
    return layer_label


def _get_label_call_index(layer_label: str) -> int:
    """Extract the pass number encoded in a layer label.

    Parameters
    ----------
    layer_label:
        Layer label that may include a ``:call_index`` suffix.

    Returns
    -------
    int
        Parsed pass number, or ``1`` when no suffix is present.
    """

    label_parts = layer_label.rsplit(":", 1)
    if len(label_parts) == 2 and label_parts[1].isdigit():
        return int(label_parts[1])
    return 1


def _append_unique(values: list[str], value: str) -> None:
    """Append ``value`` to ``values`` only if not already present.

    Parameters
    ----------
    values:
        Ordered list being built.
    value:
        Candidate value to append.
    """

    if value not in values:
        values.append(value)


def _is_prefix_stack(prefix: list[tuple[int, str]], full: list[tuple[int, str]]) -> bool:
    """Return whether one conditional branch stack prefixes another.

    Parameters
    ----------
    prefix:
        Candidate prefix stack.
    full:
        Candidate full stack.

    Returns
    -------
    bool
        ``True`` if ``prefix`` matches the first ``len(prefix)`` entries of
        ``full``.
    """

    return len(prefix) <= len(full) and full[: len(prefix)] == prefix


def _expected_layer_pass_child_views(
    conditional_arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project pass-level child views from the primary conditional structure.

    Parameters
    ----------
    conditional_arm_children:
        Primary ``cond_id -> branch_kind -> child labels`` mapping on a
        ``Op``.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        Expected THEN, ELIF, and ELSE child views.
    """

    then_children = sorted(
        {
            child_label
            for branch_children in conditional_arm_children.values()
            for child_label in branch_children.get("then", [])
        }
    )
    elif_children: dict[int, list[str]] = {}
    grouped_elif_children: dict[int, set[str]] = defaultdict(set)
    for branch_children in conditional_arm_children.values():
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            grouped_elif_children[elif_index].update(child_labels)
    for elif_index, elif_label_set in sorted(grouped_elif_children.items()):
        elif_children[elif_index] = sorted(elif_label_set)

    else_children = sorted(
        {
            child_label
            for branch_children in conditional_arm_children.values()
            for child_label in branch_children.get("else", [])
        }
    )
    return then_children, elif_children, else_children


def _expected_layer_log_child_views(
    conditional_arm_children: dict[int, dict[str, list[str]]],
) -> tuple[list[str], dict[int, list[str]], list[str]]:
    """Project aggregate child views from a ``Layer`` primary structure.

    Parameters
    ----------
    conditional_arm_children:
        Aggregate ``cond_id -> branch_kind -> child labels`` mapping on a
        ``Layer``.

    Returns
    -------
    tuple[list[str], dict[int, list[str]], list[str]]
        Expected THEN, ELIF, and ELSE child views preserving first-seen order.
    """

    then_children: list[str] = []
    elif_children: dict[int, list[str]] = {}
    else_children: list[str] = []
    for branch_children in conditional_arm_children.values():
        for child_label in branch_children.get("then", []):
            _append_unique(then_children, child_label)
        for branch_kind, child_labels in branch_children.items():
            if not branch_kind.startswith("elif_"):
                continue
            elif_index = int(branch_kind.split("_", 1)[1])
            expected_children = elif_children.setdefault(elif_index, [])
            for child_label in child_labels:
                _append_unique(expected_children, child_label)
        for child_label in branch_children.get("else", []):
            _append_unique(else_children, child_label)
    return then_children, elif_children, else_children


def _expected_layer_log_child_union(
    layer_log: "Layer",
) -> dict[int, dict[str, list[str]]]:
    """Build the expected aggregate ``conditional_arm_children`` for a ``Layer``.

    Parameters
    ----------
    layer_log:
        Aggregate layer entry being validated.

    Returns
    -------
    dict[int, dict[str, list[str]]]
        Pass-stripped union of every pass-level child list.
    """

    expected_children_by_cond: dict[int, dict[str, list[str]]] = {}
    for call_index, pass_log in sorted(layer_log.ops.items()):
        for conditional_id, branch_children in pass_log.conditional_arm_children.items():
            merged_branch_children = expected_children_by_cond.setdefault(conditional_id, {})
            for branch_kind, child_labels in branch_children.items():
                merged_child_labels = merged_branch_children.setdefault(branch_kind, [])
                for child_label in child_labels:
                    _append_unique(merged_child_labels, _strip_pass_suffix(child_label))
    return expected_children_by_cond


def _valid_conditional_child_labels(ml: "Trace") -> set[str]:
    """Return the set of valid labels for conditional child references.

    Parameters
    ----------
    ml:
        Model log being validated.

    Returns
    -------
    set[str]
        Union of pass-level labels and aggregate ``Layer`` keys.
    """

    return set(ml.layer_labels) | set(ml.layer_logs)


def _check_conditional_invariants(ml: "Trace") -> None:
    """Check F2: conditional metadata invariants added in Phase 6.

    Parameters
    ----------
    ml:
        Model log being validated.
    """

    name = "conditional_invariants"
    layer_label_set = set(ml.layer_labels)
    valid_child_labels = _valid_conditional_child_labels(ml)
    event_id_set = {event.id for event in ml.conditional_records}
    branch_context_kinds = {"if_test", "elif_test", "ifexp"}
    wrapped_context_kinds = branch_context_kinds | {"bool_cast"}

    _check_conditional_arm_entry_child_symmetry(ml, name, layer_label_set)
    _check_conditional_derived_child_views(ml, name)
    _check_conditional_child_labels_resolve(ml, name, valid_child_labels)
    _check_conditional_bool_classification(
        ml,
        name,
        branch_context_kinds,
        wrapped_context_kinds,
    )
    _check_conditional_event_references(ml, name, event_id_set)
    _check_conditional_branch_stack_monotonicity(ml, name)
    _check_conditional_elif_key_contiguity(ml, name)
    _check_conditional_bool_event_backrefs(ml, name, layer_label_set)
    _check_conditional_layer_aggregate_views(ml, name)
    _check_conditional_rolled_edge_call_indices(ml, name)
    _check_conditional_transient_bool_keys_removed(ml, name)
    _check_conditional_arm_child_pass_union(ml, name)
    _check_conditional_branch_entry_edges(ml, name, layer_label_set)
    _check_conditional_arm_edges_match_graph(ml, name)
    _check_conditional_branch_membership_records(ml, name)


def _check_conditional_arm_entry_child_symmetry(
    ml: "Trace",
    name: str,
    layer_label_set: set[str],
) -> None:
    """Check conditional arm-entry edge and child-map symmetry.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 1: conditional_arm_entry_edges ↔ conditional_arm_children.
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            if parent_label not in layer_label_set:
                _fail_conditional_invariant(
                    name,
                    1,
                    f"conditional_arm_entry_edges[{(conditional_id, branch_kind)}] references "
                    f"missing parent layer {parent_label!r}",
                )
            parent_layer = ml.layer_logs[parent_label]
            branch_children = parent_layer.conditional_arm_children.get(conditional_id, {}).get(
                branch_kind, []
            )
            if child_label not in branch_children:
                _fail_conditional_invariant(
                    name,
                    1,
                    f"conditional_arm_entry_edges[{(conditional_id, branch_kind)}] includes edge "
                    f"({parent_label!r}, {child_label!r}) but "
                    f"{parent_label}.conditional_arm_children[{conditional_id}][{branch_kind!r}]="
                    f"{branch_children}",
                )

    for parent_layer in ml.layer_logs.values():
        for conditional_id, branch_map in parent_layer.conditional_arm_children.items():
            for branch_kind, child_labels in branch_map.items():
                model_edges = ml.conditional_arm_entry_edges.get((conditional_id, branch_kind), [])
                for child_label in child_labels:
                    if (parent_layer.layer_label, child_label) not in model_edges:
                        _fail_conditional_invariant(
                            name,
                            1,
                            f"{parent_layer.layer_label}.conditional_arm_children"
                            f"[{conditional_id}][{branch_kind!r}] includes {child_label!r} "
                            f"but conditional_arm_entry_edges[{(conditional_id, branch_kind)}]={model_edges}",
                        )


def _check_conditional_derived_child_views(
    ml: "Trace",
    name: str,
) -> None:
    """Check derived conditional child views match primary structures.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 2: per-layer derived views are exact projections of the primary structures.
    for layer in ml.layer_list:
        expected_then_children, expected_elif_children, expected_else_children = (
            _expected_layer_pass_child_views(layer.conditional_arm_children)
        )
        if layer.conditional_then_children != expected_then_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_then_children={layer.conditional_then_children} != "
                f"expected projection {expected_then_children}",
            )
        if layer.conditional_elif_children != expected_elif_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_elif_children={layer.conditional_elif_children} != "
                f"expected projection {expected_elif_children}",
            )
        if layer.conditional_else_children != expected_else_children:
            _fail_conditional_invariant(
                name,
                2,
                f"{layer.layer_label}.conditional_else_children={layer.conditional_else_children} != "
                f"expected projection {expected_else_children}",
            )

    for layer_log in ml.layer_logs.values():
        expected_then_children, expected_elif_children, expected_else_children = (
            _expected_layer_log_child_views(layer_log.conditional_arm_children)
        )
        if layer_log.conditional_then_children != expected_then_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_then_children="
                f"{layer_log.conditional_then_children} != expected projection "
                f"{expected_then_children}",
            )
        if layer_log.conditional_elif_children != expected_elif_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_elif_children="
                f"{layer_log.conditional_elif_children} != expected projection "
                f"{expected_elif_children}",
            )
        if layer_log.conditional_else_children != expected_else_children:
            _fail_conditional_invariant(
                name,
                2,
                f"Layer {layer_log.layer_label}.conditional_else_children="
                f"{layer_log.conditional_else_children} != expected projection "
                f"{expected_else_children}",
            )


def _check_conditional_child_labels_resolve(
    ml: "Trace",
    name: str,
    valid_child_labels: set[str],
) -> None:
    """Check conditional child labels resolve to known layers.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 3: every child label in conditional child views exists in the log.
    for layer in ml.layer_list:
        for field_name, child_labels in (
            ("conditional_entry_children", layer.conditional_entry_children),
            ("conditional_then_children", layer.conditional_then_children),
            ("conditional_else_children", layer.conditional_else_children),
        ):
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"{layer.layer_label}.{field_name} contains missing child label "
                        f"{child_label!r}",
                    )
        for elif_index, child_labels in layer.conditional_elif_children.items():
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"{layer.layer_label}.conditional_elif_children[{elif_index}] "
                        f"contains missing child label {child_label!r}",
                    )

    for layer_log in ml.layer_logs.values():
        for field_name, child_labels in (
            ("conditional_entry_children", layer_log.conditional_entry_children),
            ("conditional_then_children", layer_log.conditional_then_children),
            ("conditional_else_children", layer_log.conditional_else_children),
        ):
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"Layer {layer_log.layer_label}.{field_name} contains missing child "
                        f"label {child_label!r}",
                    )
        for elif_index, child_labels in layer_log.conditional_elif_children.items():
            for child_label in child_labels:
                if child_label not in valid_child_labels:
                    _fail_conditional_invariant(
                        name,
                        3,
                        f"Layer {layer_log.layer_label}.conditional_elif_children[{elif_index}] "
                        f"contains missing child label {child_label!r}",
                    )

    for parent_label, child_label in ml.conditional_branch_edges:
        if child_label not in valid_child_labels:
            _fail_conditional_invariant(
                name,
                3,
                f"Trace.conditional_branch_edges contains missing child label {child_label!r} "
                f"for parent {parent_label!r}",
            )
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            if child_label not in valid_child_labels:
                _fail_conditional_invariant(
                    name,
                    3,
                    f"Trace.conditional_arm_entry_edges contains missing child label "
                    f"{child_label!r} for edge {(conditional_id, branch_kind, parent_label)}",
                )


def _check_conditional_bool_classification(
    ml: "Trace",
    name: str,
    branch_context_kinds: set[str],
    wrapped_context_kinds: set[str],
) -> None:
    """Check conditional bool classification fields are mutually consistent.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 4: bool classification fields are mutually consistent.
    for layer in ml.layer_list:
        expected_is_branch = layer.conditional_context_kind in branch_context_kinds
        if layer.is_terminal_conditional_bool != expected_is_branch:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has is_terminal_conditional_bool={layer.is_terminal_conditional_bool} but "
                f"conditional_context_kind={layer.conditional_context_kind!r}",
            )
        if layer.is_terminal_conditional_bool and layer.terminal_conditional_id is None:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has is_terminal_conditional_bool=True but terminal_conditional_id is None",
            )
        if layer.conditional_context_kind is not None and not layer.is_terminal_bool:
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has conditional_context_kind={layer.conditional_context_kind!r} but "
                f"is_terminal_bool=False",
            )
        if (
            layer.conditional_wrapper_kind is not None
            and layer.conditional_context_kind not in wrapped_context_kinds
        ):
            _fail_conditional_invariant(
                name,
                4,
                f"{layer.layer_label} has conditional_wrapper_kind={layer.conditional_wrapper_kind!r} but "
                f"conditional_context_kind={layer.conditional_context_kind!r}",
            )


def _check_conditional_event_references(
    ml: "Trace",
    name: str,
    event_id_set: set[int],
) -> None:
    """Check referenced conditional ids resolve to events.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 5: every referenced cond_id corresponds to a ConditionalEvent.
    referenced_cond_ids: set[int] = set()
    for layer in ml.layer_list:
        referenced_cond_ids.update(
            conditional_id for conditional_id, _ in layer.conditional_branch_stack
        )
        if layer.terminal_conditional_id is not None:
            referenced_cond_ids.add(layer.terminal_conditional_id)
        referenced_cond_ids.update(layer.conditional_arm_children)
    for layer_log in ml.layer_logs.values():
        referenced_cond_ids.update(
            conditional_id
            for branch_stack in layer_log.conditional_role_stacks
            for conditional_id, _ in branch_stack
        )
        referenced_cond_ids.update(layer_log.conditional_arm_children)
    referenced_cond_ids.update(
        conditional_id for conditional_id, _ in ml.conditional_arm_entry_edges
    )

    for conditional_id in sorted(referenced_cond_ids):
        if conditional_id not in event_id_set:
            _fail_conditional_invariant(
                name,
                5,
                f"Referenced cond_id {conditional_id} has no matching ConditionalEvent.id "
                f"in Trace.conditional_records",
            )


def _check_conditional_branch_stack_monotonicity(
    ml: "Trace",
    name: str,
) -> None:
    """Check parent-child conditional stacks are monotone by prefix.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 6: parent->child stacks are monotone by prefix relation.
    for parent_op in ml.layer_list:
        for child_label in parent_op.children:
            child_layer = ml[child_label]
            if parent_op.pass_index != child_layer.pass_index:
                continue
            if parent_op.conditional_branch_stack == child_layer.conditional_branch_stack:
                continue
            if _is_prefix_stack(
                parent_op.conditional_branch_stack, child_layer.conditional_branch_stack
            ):
                continue
            if _is_prefix_stack(
                child_layer.conditional_branch_stack, parent_op.conditional_branch_stack
            ):
                continue
            _fail_conditional_invariant(
                name,
                6,
                f"Edge ({parent_op.layer_label!r}, {child_label!r}) has non-prefix "
                f"conditional stacks parent={parent_op.conditional_branch_stack} "
                f"child={child_layer.conditional_branch_stack}",
            )


def _check_conditional_elif_key_contiguity(
    ml: "Trace",
    name: str,
) -> None:
    """Check elif branch keys are contiguous on conditional events.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 7: elif keys are contiguous on ConditionalEvent.
    for event in ml.conditional_records:
        for field_name, mapping in (
            ("branch_ranges", event.branch_ranges),
            ("branch_test_spans", event.branch_test_spans),
        ):
            elif_indices = sorted(
                int(key.split("_", 1)[1]) for key in mapping if key.startswith("elif_")
            )
            if elif_indices != list(range(1, len(elif_indices) + 1)):
                _fail_conditional_invariant(
                    name,
                    7,
                    f"ConditionalEvent id={event.id} {field_name} has non-contiguous elif keys "
                    f"{elif_indices}",
                )


def _check_conditional_bool_event_backrefs(
    ml: "Trace",
    name: str,
    layer_label_set: set[str],
) -> None:
    """Check conditional event bool-layer backreferences.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 8: ConditionalEvent.bool_layers back-reference to the event id.
    for event in ml.conditional_records:
        for bool_label in event.bool_layers:
            if bool_label not in layer_label_set:
                _fail_conditional_invariant(
                    name,
                    8,
                    f"ConditionalEvent id={event.id} bool_layers contains missing label "
                    f"{bool_label!r}",
                )
            try:
                bool_layer = ml.ops[bool_label]
            except (KeyError, ValueError, TypeError):
                bool_layer = ml[bool_label]
            bool_ops = (
                list(cast("Layer", bool_layer).ops.values())
                if hasattr(bool_layer, "ops") and not hasattr(bool_layer, "terminal_conditional_id")
                else [bool_layer]
            )
            mismatched_bool_ops = [
                op for op in bool_ops if getattr(op, "terminal_conditional_id", None) != event.id
            ]
            if mismatched_bool_ops:
                _fail_conditional_invariant(
                    name,
                    8,
                    f"ConditionalEvent id={event.id} bool_layers includes {bool_label!r} but "
                    f"{bool_label}.terminal_conditional_id="
                    f"{getattr(mismatched_bool_ops[0], 'terminal_conditional_id', None)}",
                )


def _check_conditional_layer_aggregate_views(
    ml: "Trace",
    name: str,
) -> None:
    """Check layer conditional aggregate views match pass-level data.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 9: Layer conditional aggregate views match pass-level data.
    for layer_log in ml.layer_logs.values():
        expected_stack_order: list[list[tuple[int, str]]] = []
        expected_stack_ops: dict[tuple[tuple[int, str], ...], list[int]] = {}
        for call_index, pass_log in sorted(layer_log.ops.items()):
            stack_signature = tuple(pass_log.conditional_branch_stack)
            if stack_signature not in expected_stack_ops:
                expected_stack_order.append(list(pass_log.conditional_branch_stack))
                expected_stack_ops[stack_signature] = []
            expected_stack_ops[stack_signature].append(call_index)

        if layer_log.conditional_role_stacks != expected_stack_order:
            _fail_conditional_invariant(
                name,
                9,
                f"Layer {layer_log.layer_label}.conditional_role_stacks="
                f"{layer_log.conditional_role_stacks} != expected {expected_stack_order}",
            )
        if layer_log.conditional_branch_stack_ops != expected_stack_ops:
            _fail_conditional_invariant(
                name,
                9,
                f"Layer {layer_log.layer_label}.conditional_branch_stack_ops="
                f"{layer_log.conditional_branch_stack_ops} != expected "
                f"{expected_stack_ops}",
            )


def _check_conditional_rolled_edge_call_indices(
    ml: "Trace",
    name: str,
) -> None:
    """Check rolled conditional edge call indices.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 10: rolled conditional_edge_call_indices reference known
    # layer-level arm-entry edges. Exact pass lists live only in
    # conditional_edge_call_indices after the label remap.
    actual_arm_edges: set[tuple[str, str, int, str]] = set()
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            actual_arm_edges.add(
                (
                    _strip_pass_suffix(parent_label),
                    _strip_pass_suffix(child_label),
                    conditional_id,
                    branch_kind,
                )
            )

    for edge_key, call_indexs in ml.conditional_edge_call_indices.items():
        parent_no_pass, child_no_pass, conditional_id, branch_kind = edge_key
        if call_indexs != sorted(call_indexs) or len(call_indexs) != len(set(call_indexs)):
            _fail_conditional_invariant(
                name,
                10,
                f"conditional_edge_call_indices[{edge_key}] has unsorted or duplicate pass list "
                f"{call_indexs}",
            )
        for call_index in call_indexs:
            if call_index < 1:
                _fail_conditional_invariant(
                    name,
                    10,
                    f"conditional_edge_call_indices[{edge_key}] includes invalid pass {call_index}",
                )
            actual_edge = (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
            )
            if actual_edge not in actual_arm_edges:
                _fail_conditional_invariant(
                    name,
                    10,
                    f"conditional_edge_call_indices[{edge_key}] includes pass metadata but "
                    f"conditional_arm_entry_edges has no matching layer edge",
                )

    for actual_edge in sorted(actual_arm_edges):
        parent_no_pass, child_no_pass, conditional_id, branch_kind = actual_edge
        edge_key = (parent_no_pass, child_no_pass, conditional_id, branch_kind)
        if edge_key not in ml.conditional_edge_call_indices:
            _fail_conditional_invariant(
                name,
                10,
                f"conditional_arm_entry_edges implies rolled edge {actual_edge} but "
                f"conditional_edge_call_indices[{edge_key}]={ml.conditional_edge_call_indices.get(edge_key)}",
            )


def _check_conditional_transient_bool_keys_removed(
    ml: "Trace",
    name: str,
) -> None:
    """Check transient bool conditional keys were removed.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 11: no transient _bool_conditional_key remains after step 5c.
    for layer in ml.layer_list:
        if hasattr(layer, "_bool_conditional_key"):
            _fail_conditional_invariant(
                name,
                11,
                f"{layer.layer_label} still has transient _bool_conditional_key attribute",
            )


def _check_conditional_arm_child_pass_union(
    ml: "Trace",
    name: str,
) -> None:
    """Check layer conditional arm children are exact pass unions.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 12: Layer conditional_arm_children is the exact pass union.
    for layer_log in ml.layer_logs.values():
        expected_children_by_cond = _expected_layer_log_child_union(layer_log)
        if layer_log.conditional_arm_children != expected_children_by_cond:
            _fail_conditional_invariant(
                name,
                12,
                f"Layer {layer_log.layer_label}.conditional_arm_children="
                f"{layer_log.conditional_arm_children} != expected pass union "
                f"{expected_children_by_cond}",
            )


def _check_conditional_branch_entry_edges(
    ml: "Trace",
    name: str,
    layer_label_set: set[str],
) -> None:
    """Check legacy branch edges match conditional entry children.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 13: legacy IF-view conditional_branch_edges ↔ start-children.
    for parent_label, bool_label in ml.conditional_branch_edges:
        if parent_label not in layer_label_set:
            _fail_conditional_invariant(
                name,
                13,
                f"conditional_branch_edges references missing parent layer {parent_label!r}",
            )
        parent_layer = ml.layer_logs[parent_label]
        if bool_label not in parent_layer.conditional_entry_children:
            _fail_conditional_invariant(
                name,
                13,
                f"conditional_branch_edges includes ({parent_label!r}, {bool_label!r}) but "
                f"{parent_label}.conditional_entry_children={parent_layer.conditional_entry_children}",
            )

    for parent_layer in ml.layer_logs.values():
        for bool_label in parent_layer.conditional_entry_children:
            if (parent_layer.layer_label, bool_label) not in ml.conditional_branch_edges:
                _fail_conditional_invariant(
                    name,
                    13,
                    f"{parent_layer.layer_label}.conditional_entry_children includes "
                    f"{bool_label!r} but conditional_branch_edges={ml.conditional_branch_edges}",
                )


def _check_conditional_arm_edges_match_graph(
    ml: "Trace",
    name: str,
) -> None:
    """Check conditional arm-entry edges correspond to graph edges.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 14: conditional arm-entry edges correspond to real graph edges.
    # Invariants 1-2 tie the THEN/ELIF/ELSE child views to the arm-entry edges;
    # this check closes the loop by tying those edges to the actual rolled
    # parent->child topology, so conditional edge metadata can never reference
    # an edge that does not exist in the graph.
    rolled_graph_edges: set[tuple[str, str]] = set()
    for layer in ml.layer_list:
        for child_label in layer.children:
            rolled_graph_edges.add((layer.layer_label, _strip_pass_suffix(child_label)))
    for (conditional_id, branch_kind), edge_list in ml.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            rolled_edge = (
                _strip_pass_suffix(parent_label),
                _strip_pass_suffix(child_label),
            )
            if rolled_edge not in rolled_graph_edges:
                _fail_conditional_invariant(
                    name,
                    14,
                    f"conditional_arm_entry_edges[{(conditional_id, branch_kind)}] includes "
                    f"({parent_label!r}, {child_label!r}) but the graph has no "
                    f"{rolled_edge[0]} -> {rolled_edge[1]} edge",
                )


def _check_conditional_branch_membership_records(
    ml: "Trace",
    name: str,
) -> None:
    """Check per-op conditional branch membership records agree.

    Parameters
    ----------
    ml:
        Trace containing conditional metadata.
    name:
        Invariant check name for raised errors.
    """

    # Invariant 15: per-op conditional-branch membership records agree.
    # ``conditional_branch_stack`` is the canonical per-op record of arm
    # membership; the depth counter and any 'body' roles in
    # ``in_conditionals`` must be consistent with it.
    event_id_by_bool_label: dict[str, int] = {}
    for event in ml.conditional_records:
        for bool_label in event.bool_layers:
            event_id_by_bool_label[bool_label] = event.id
    event_id_by_conditional_id: dict[str, int] = {}
    branch_kind_by_conditional_arm: dict[tuple[str, int], str] = {}
    for conditional in getattr(ml, "conditionals", []) or []:
        for arm in conditional.arms:
            if arm.terminal_bool_op_label is not None:
                event_id = event_id_by_bool_label.get(arm.terminal_bool_op_label)
                if event_id is not None:
                    event_id_by_conditional_id[conditional.id] = event_id
                    break
        for arm_index, arm in enumerate(conditional.arms):
            if arm.kind == "elif":
                branch_kind_by_conditional_arm[(conditional.id, arm_index)] = f"elif_{arm_index}"
            else:
                branch_kind_by_conditional_arm[(conditional.id, arm_index)] = arm.kind
    stack_entries_by_layer_label: dict[str, set[tuple[int, str]]] = {}
    for layer in ml.layer_list:
        stack_entries_by_layer_label.setdefault(layer.layer_label, set()).update(
            layer.conditional_branch_stack
        )

    for layer in ml.layer_list:
        if layer.conditional_branch_depth != len(layer.conditional_branch_stack):
            _fail_conditional_invariant(
                name,
                15,
                f"{layer.label} has conditional_branch_depth="
                f"{layer.conditional_branch_depth} but len(conditional_branch_stack)="
                f"{len(layer.conditional_branch_stack)}",
            )
        has_body_role = any(role.role == "body" for role in (layer.in_conditionals or []))
        if has_body_role and not layer.conditional_branch_stack:
            _fail_conditional_invariant(
                name,
                15,
                f"{layer.label} has a 'body' conditional role in in_conditionals but an "
                f"empty conditional_branch_stack",
            )
        stack_entries = set(layer.conditional_branch_stack)
        for role in layer.in_conditionals or []:
            if role.role != "body":
                continue
            expected_event_id = event_id_by_conditional_id.get(role.conditional_id)
            if expected_event_id is None:
                _fail_conditional_invariant(
                    name,
                    15,
                    f"{layer.label} has body role conditional_id={role.conditional_id!r} "
                    f"but no matching conditional event was found",
                )
            expected_branch_kind = branch_kind_by_conditional_arm.get(
                (role.conditional_id, role.arm_index), role.arm_kind
            )
            expected_stack_entry = (expected_event_id, expected_branch_kind)
            layer_stack_entries = stack_entries_by_layer_label.get(layer.layer_label, set())
            if (
                expected_stack_entry not in stack_entries
                and expected_stack_entry not in layer_stack_entries
            ):
                _fail_conditional_invariant(
                    name,
                    15,
                    f"{layer.label} has body role conditional_id={role.conditional_id!r} "
                    f"arm_kind={role.arm_kind!r} but conditional_branch_stack="
                    f"{layer.conditional_branch_stack}; expected entry {expected_stack_entry}",
                )


# ---------------------------------------------------------------------------
# G. Op ↔ Layer cross-references
# ---------------------------------------------------------------------------


def _check_layer_pass_to_layer_log_xrefs(ml: "Trace") -> None:
    """Check G: Op <-> Layer cross-references.

    Validates:
    - Layer key matches its layer_label.
    - ops dict keys are contiguous {1..N}.
    - Each Op's call_index matches its dict key.
    - Each Op's layer_label matches the parent Layer's label.
    """
    name = "layer_pass_layer_log_xrefs"

    for ll_label, ll in ml.layer_logs.items():
        if ll.layer_label != ll_label:
            raise MetadataInvariantError(
                name,
                f"Layer key '{ll_label}' != Layer.layer_label='{ll.layer_label}'",
            )

        expected_keys = set(range(1, ll.num_passes + 1))
        actual_keys = set(ll.ops.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Layer '{ll_label}' ops keys={actual_keys} != expected {expected_keys}",
            )

        for call_index, lpl in ll.ops.items():
            if lpl.pass_index != call_index:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{ll_label}' pass key={call_index} but Op.pass_index={lpl.pass_index}",
                )
            if lpl.layer_label != ll.layer_label:
                raise MetadataInvariantError(
                    name,
                    f"Op '{lpl.layer_label}' layer_label="
                    f"'{lpl.layer_label}' != "
                    f"parent Layer.layer_label='{ll.layer_label}'",
                )


def _check_pass_count_consistency(ml: "Trace") -> None:
    """Check pass-count consistency across multi-pass layer records.

    Parameters
    ----------
    ml:
        Trace whose per-layer pass maps should be checked.

    Raises
    ------
    MetadataInvariantError
        If layer call counts, op maps, and op pass metadata disagree.
    """

    name = "pass_count_consistency"
    layer_num_calls = getattr(ml, "layer_num_calls", {}) or {}
    for layer_label, layer_log in ml.layer_logs.items():
        ops = getattr(layer_log, "ops", {}) or {}
        expected_keys = set(range(1, getattr(layer_log, "num_passes", 0) + 1))
        actual_keys = set(ops)
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Layer '{layer_label}' ops keys {sorted(actual_keys)!r} do not match "
                f"num_passes={getattr(layer_log, 'num_passes', None)!r}",
            )
        for pass_index, op in ops.items():
            if getattr(op, "pass_index", None) != pass_index:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' stores op with pass_index="
                    f"{getattr(op, 'pass_index', None)!r} under key {pass_index!r}",
                )
            if getattr(op, "num_passes", None) != getattr(layer_log, "num_passes", None):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{layer_label}' op {getattr(op, 'label', pass_index)!r} has "
                    f"num_passes={getattr(op, 'num_passes', None)!r}, expected "
                    f"{getattr(layer_log, 'num_passes', None)!r}",
                )
        if layer_label in layer_num_calls and layer_num_calls[layer_label] != layer_log.num_passes:
            raise MetadataInvariantError(
                name,
                f"layer_num_calls[{layer_label!r}]={layer_num_calls[layer_label]!r} "
                f"!= Layer.num_passes={layer_log.num_passes!r}",
            )


# ---------------------------------------------------------------------------
# H. Module ↔ Layer containment
# ---------------------------------------------------------------------------


def _check_module_layer_containment(ml: "Trace") -> None:
    """Check H: Module <-> Layer containment consistency.

    Validates forward and reverse directions:
    - Forward: Module.layer_labels exist in layer_logs; num_layers matches.
      ModuleCall.ops labels exist; input/output_layers subset of ops.
    - Reverse: each layer's module points to a valid module
      that lists the layer in its layers.
    """
    name = "module_layer_containment"
    mod_accessor = ml.modules
    label_set = set(ml.op_labels)
    no_pass_set = set(ml.layer_labels)
    all_layer_label_set = label_set | no_pass_set
    module_layer_label_sets = {
        mod_log.address: set(mod_log.layer_labels) for mod_log in mod_accessor
    }

    for mod_log in mod_accessor:
        addr = mod_log.address

        # Module.layer_labels exist in layer_logs
        for lbl in mod_log.layer_labels:
            if lbl not in ml.layer_logs:
                raise MetadataInvariantError(
                    name,
                    f"Module '{addr}' layers contains '{lbl}' not in trace.layer_logs",
                )

        if mod_log.num_layers != len(mod_log.layer_labels):
            raise MetadataInvariantError(
                name,
                f"Module '{addr}': num_layers={mod_log.num_layers} != "
                f"len(layer_labels)={len(mod_log.layer_labels)}",
            )

        # ModuleCall checks
        # mpl.ops may contain pass-qualified labels OR no-pass labels
        # (e.g., root module in recurrent models uses no-pass labels).
        for call_index, mpl in mod_log.ops.items():
            for lbl in mpl.ops:
                if lbl not in label_set and lbl not in no_pass_set:
                    raise MetadataInvariantError(
                        name,
                        f"ModuleCall '{addr}:{call_index}' ops contains "
                        f"'{lbl}' not in op_labels or layer_labels",
                    )

            if mpl.num_layers != len(mpl.ops):
                raise MetadataInvariantError(
                    name,
                    f"ModuleCall '{addr}:{call_index}': "
                    f"num_layers={mpl.num_layers} != len(ops)={len(mpl.ops)}",
                )

            # input/output layers subset of ops (using both pass-qualified
            # and no-pass labels to handle recurrent models)
            for sub_attr in ("input_layers", "output_layers"):
                sub_list = getattr(mpl, sub_attr)
                sub_set = set(sub_list)
                extra = sub_set - all_layer_label_set
                if extra:
                    raise MetadataInvariantError(
                        name,
                        f"ModuleCall '{addr}:{call_index}' "
                        f"{sub_attr} has labels not in layers: {extra}",
                    )

    # Reverse check: layer's module exists in modules
    for lpl in ml.layer_list:
        cmo = lpl.module
        if cmo:
            # module may include pass suffix (e.g. 'fc:1')
            cmo_addr = cmo.split(":")[0] if ":" in cmo else cmo
            try:
                mod_accessor[cmo_addr]
            except (KeyError, IndexError):
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' module='{cmo}' "
                    f"(addr='{cmo_addr}') not found in module accessor",
                )
            module_layer_labels = module_layer_label_sets.get(cmo_addr, set())
            if lpl.layer_label not in module_layer_labels:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' (no_pass='{lpl.layer_label}') "
                    f"not in Module '{cmo_addr}'.layers",
                )


# ---------------------------------------------------------------------------
# I. Module hierarchy consistency
# ---------------------------------------------------------------------------


def _check_module_hierarchy(ml: "Trace") -> None:
    """Check I: module address tree consistency and pass structure.

    Precondition contract: rich ModuleCall checks run only for materialized
    real ModuleCall records in called modules. Static uncalled child modules
    are still handled by the existing address-level exceptions. ModuleCall
    boundary inputs are allowed to be external to the call because they are
    commonly produced in the parent scope; only output ops are required to be
    produced inside the call's own ``ops`` list.

    Validates:
    - Root module 'self' exists.
    - Address parent-child bidirectionality (with exemptions for shared
      modules, where aliases may diverge from the primary path).
    - Container modules (ModuleList) that were never called may not have
      ModuleLogs -- skip rather than error.
    - Pass dict keys are contiguous {1..N} and match num_passes.
    - call_parent and call_children reference valid modules.
    - ModuleCall labels, output boundaries, call-tree links, stacks, and
      output structures are internally consistent.
    """
    name = "module_hierarchy"
    mod_accessor = ml.modules

    # Root module exists
    try:
        mod_accessor["self"]
    except (KeyError, IndexError):
        raise MetadataInvariantError(name, "'self' module not found in module accessor")

    for mod_log in mod_accessor:
        addr = mod_log.address

        # Address hierarchy bidirectional
        if mod_log.address_parent is not None:
            try:
                parent: Module = mod_accessor[mod_log.address_parent]  # type: ignore[assignment]
            except (KeyError, IndexError):
                # Parent module may be a container (ModuleList, ModuleDict)
                # that is never called during the forward pass, so no
                # Module exists.  Skip rather than error.
                parent = None  # type: ignore[assignment]
            if parent is not None and addr not in parent.address_children:
                # For shared modules, addr may be an alias that the parent
                # lists under a different address prefix.  Check if any of the
                # parent's address_children resolve to the same Module.
                if not mod_log.has_multiple_addresses:
                    raise MetadataInvariantError(
                        name,
                        f"Module '{addr}' has address_parent='{mod_log.address_parent}' "
                        f"but parent doesn't list it in address_children",
                    )

        for child_addr in mod_log.address_children:
            try:
                child: Module = mod_accessor[child_addr]  # type: ignore[assignment]
            except (KeyError, IndexError):
                # Static children may not have been invoked during the forward
                # pass, so no Module exists.  Skip rather than error.
                continue
            if child.address_parent != addr:
                # For shared modules (same nn.Module registered under multiple
                # addresses), the child's address_parent refers to its primary
                # alias's parent, which may differ from the current parent addr.
                # This is expected — there is only one Module per module
                # instance, so address_parent always reflects the primary path.
                if not child.has_multiple_addresses:
                    raise MetadataInvariantError(
                        name,
                        f"Module '{addr}' lists '{child_addr}' as address_child, "
                        f"but child's address_parent='{child.address_parent}'",
                    )

        # Module pass consistency
        if len(mod_log.ops) != mod_log.num_calls:
            raise MetadataInvariantError(
                name,
                f"Module '{addr}': len(ops)={len(mod_log.ops)} != num_calls={mod_log.num_calls}",
            )

        expected_keys = set(range(1, mod_log.num_calls + 1))
        actual_keys = set(mod_log.ops.keys())
        if actual_keys != expected_keys:
            raise MetadataInvariantError(
                name,
                f"Module '{addr}' pass keys={actual_keys} != expected {expected_keys}",
            )

        # Call hierarchy: parent exists
        for call_index, mpl in mod_log.ops.items():
            if mpl.call_parent is not None:
                try:
                    mod_accessor[mpl.call_parent]
                except (KeyError, IndexError):
                    raise MetadataInvariantError(
                        name,
                        f"ModuleCall '{addr}:{call_index}' call_parent="
                        f"'{mpl.call_parent}' not in module accessor",
                    )
            for cc in mpl.call_children:
                try:
                    mod_accessor[cc]
                except (KeyError, IndexError):
                    raise MetadataInvariantError(
                        name,
                        f"ModuleCall '{addr}:{call_index}' call_children "
                        f"contains '{cc}' not in module accessor",
                    )

            _check_module_call_boundary_and_tree(ml, mpl, name)


def _check_module_call_boundary_and_tree(
    ml: "Trace",
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


def _check_module_call_tree_links(ml: "Trace", module_call: object, name: str) -> None:
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
    ml: "Trace",
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


# ---------------------------------------------------------------------------
# J. Param ↔ Layer ↔ Module cross-references
# ---------------------------------------------------------------------------


def _check_param_xrefs(ml: "Trace") -> None:
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
    mod_accessor = ml.modules

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

        # address exists (skip for conditional models where the module
        # was never called, e.g. MoE routing that skips some experts)
        try:
            mod_accessor[param.address]
        except (KeyError, IndexError):
            pass  # Module was never invoked during forward pass

        if param.num_uses_by_ops == 0 and not param.used_by_layers:
            continue
        _check_param_usage_reciprocal_links(ml, param, name)
        _check_param_co_parent_links(ml, param, name)

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


def _check_param_usage_reciprocal_links(ml: "Trace", param: object, name: str) -> None:
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


def _check_param_co_parent_links(ml: "Trace", param: object, name: str) -> None:
    """Check co-parent parameter links resolve and are symmetric.

    Parameters
    ----------
    ml:
        Trace containing parameter metadata.
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
        co_parent = _param_by_address(ml, co_parent_address)
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


def _check_layers_with_params_matches_param_usage(ml: "Trace", name: str) -> None:
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


def _check_layer_param_aggregate_dedup(ml: "Trace", name: str) -> None:
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

    seen_layer_labels: set[str] = set()
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    layers_with_params = 0
    for layer in ml.layer_list:
        if layer.layer_label in seen_layer_labels:
            continue
        seen_layer_labels.add(layer.layer_label)
        if not getattr(layer, "_param_logs", ()):
            continue
        layers_with_params += 1
        total_params += getattr(layer, "num_params", 0)
        trainable_params += getattr(layer, "num_params_trainable", 0)
        frozen_params += getattr(layer, "num_params_frozen", 0)

    if total_params != getattr(ml, "num_params", 0):
        raise MetadataInvariantError(
            name,
            f"deduped layer param total {total_params} != trace.num_params={ml.num_params}",
        )
    if trainable_params != getattr(ml, "num_params_trainable", 0):
        raise MetadataInvariantError(
            name,
            "deduped trainable param total "
            f"{trainable_params} != trace.num_params_trainable={ml.num_params_trainable}",
        )
    if frozen_params != getattr(ml, "num_params_frozen", 0):
        raise MetadataInvariantError(
            name,
            f"deduped frozen param total {frozen_params} != "
            f"trace.num_params_frozen={ml.num_params_frozen}",
        )
    if layers_with_params != getattr(ml, "num_layers_with_params", 0):
        raise MetadataInvariantError(
            name,
            f"deduped layers_with_params count {layers_with_params} != "
            f"trace.num_layers_with_params={ml.num_layers_with_params}",
        )


def _param_log_list_contains_param(param_logs: object, param: object) -> bool:
    """Return whether a parameter log collection contains ``param`` by identity.

    Parameters
    ----------
    param_logs:
        Iterable of Param records.
    param:
        Param record to look for.

    Returns
    -------
    bool
        ``True`` when any log has the same address or barcode as ``param``.
    """

    address = getattr(param, "address", None)
    barcode = getattr(param, "barcode", None)
    if not isinstance(param_logs, Iterable):
        return False
    return any(
        candidate is param
        or (
            address is not None
            and getattr(candidate, "address", None) == address
            and getattr(candidate, "barcode", None) == barcode
        )
        for candidate in param_logs or ()
    )


def _param_by_address(ml: "Trace", address: str) -> object | None:
    """Return a Param by primary or alias address.

    Parameters
    ----------
    ml:
        Trace containing parameter metadata.
    address:
        Primary or alias parameter address.

    Returns
    -------
    object | None
        Matching Param record when present.
    """

    for param in ml.param_logs:
        if getattr(param, "address", None) == address:
            return param
        if address in (getattr(param, "all_addresses", None) or ()):
            return param
    return None


# ---------------------------------------------------------------------------
# K. Buffer cross-references
# ---------------------------------------------------------------------------


def _check_buffer_xrefs(ml: "Trace") -> None:
    """Check K: buffer layer and Buffer cross-references.

    Precondition contract: torch registered buffers are represented by
    ``Buffer`` entities whose versions are buffer Op nodes. Static read
    versions and write versions share the same ``Buffer.versions`` list, so
    write-only fields are only required for versions with
    ``buffer_write_kind is not None``. Buffer addresses may live under an
    uncalled child module, a container, or a top-level attribute; resolving any
    ancestor module is sufficient. ``buffer_source`` is required to resolve
    only when it is populated, because selective materialization may null it
    when the producer raw label did not survive. ``buffer_replay_validated`` is
    asserted as backed only for write versions that explicitly set it to
    ``True``; static init-only buffers and write versions with ``None``/``False``
    do not claim successful identity replay.

    Validates:
    - buffer_layers list entries are valid layer labels.
    - static Buffer version nodes resolve to buffer Ops at the same address.
    - write versions have valid write-kind domains and dense pass sets per address.
    - populated source and replay-validation metadata are backed by resolvable evidence.
    """
    name = "buffer_xrefs"
    label_set = set(ml.layer_labels)

    for lbl in ml.buffer_layers:
        if lbl not in label_set:
            raise MetadataInvariantError(
                name, f"buffer_layers contains '{lbl}' not in layer_labels"
            )

    # Check Buffer objects via buffer accessor
    if hasattr(ml, "_buffer_accessor") and ml._buffer_accessor is not None:
        for buf in ml.buffers:
            _check_buffer_static_versions(ml, buf, name)
            _check_buffer_semantic_ownership(ml, buf, name)
            _check_buffer_write_versions(ml, buf, name)
            _check_buffer_replay_validated_versions(ml, buf, name)


def _check_buffer_static_versions(ml: "Trace", buf: object, name: str) -> None:
    """Check static Buffer entity/version structure.

    Parameters
    ----------
    ml:
        Trace containing buffer metadata.
    buf:
        Buffer entity from the trace buffer accessor.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If the Buffer entity has no address, no versions, non-buffer versions,
        mismatched version addresses, or no acceptable ancestor module.
    """

    address = getattr(buf, "address", None)
    layer_label = getattr(buf, "layer_label", None)
    if not address:
        raise MetadataInvariantError(
            name,
            f"Buffer '{layer_label}' has empty address",
        )
    versions = list(getattr(buf, "versions", ()) or ())
    if not versions:
        raise MetadataInvariantError(
            name,
            f"Buffer '{address}' has no version nodes",
        )
    for version in versions:
        label = getattr(version, "layer_label", type(version).__name__)
        if not getattr(version, "is_buffer", False):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' is not a buffer Op",
            )
        if getattr(version, "address", None) != address:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' has address "
                f"{getattr(version, 'address', None)!r}",
            )
        if label not in set(getattr(ml, "layer_dict_all_keys", {})):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' does not resolve in trace",
            )

    if not _buffer_address_has_module_ancestor(ml, address):
        raise MetadataInvariantError(
            name,
            f"Buffer '{layer_label}' address='{address}' — no ancestor found in module accessor",
        )


def _buffer_address_has_module_ancestor(ml: "Trace", address: str) -> bool:
    """Return whether ``address`` or an ancestor is in the module accessor.

    Parameters
    ----------
    ml:
        Trace containing module metadata.
    address:
        Registered buffer address.

    Returns
    -------
    bool
        ``True`` when the buffer address satisfies the loose ancestry rule.
    """

    addr = address
    found_ancestor = addr in ml.modules
    while not found_ancestor and "." in addr:
        addr = addr.rsplit(".", 1)[0]
        found_ancestor = addr in ml.modules
    return found_ancestor or "" in ml.modules


def _check_buffer_semantic_ownership(ml: "Trace", buf: object, name: str) -> None:
    """Check buffer source versions are owned by their module or a consumer.

    Parameters
    ----------
    ml:
        Trace containing module and buffer metadata.
    buf:
        Buffer entity from the trace buffer accessor.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a buffer source version claims a module stack unrelated to the
        registered buffer owner and unrelated to any active consumer op.
    """

    address = getattr(buf, "address", None)
    if not isinstance(address, str) or not address:
        return
    owner_address = _owner_module_address_for_buffer(address)
    for version in list(getattr(buf, "versions", ()) or ()):
        version_label = getattr(version, "layer_label", type(version).__name__)
        module_claims = _module_addresses_for_buffer_version(version)
        if not module_claims:
            continue
        valid_addresses = {owner_address}
        valid_addresses.update(_active_buffer_consumer_module_addresses(ml, version))
        if module_claims.isdisjoint(valid_addresses):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{version_label}' has module stack "
                f"{sorted(module_claims)!r}, expected owner/consumer in "
                f"{sorted(valid_addresses)!r}",
            )


def _owner_module_address_for_buffer(address: str) -> str:
    """Return the owning module address for a registered buffer address.

    Parameters
    ----------
    address:
        Dotted registered buffer address.

    Returns
    -------
    str
        Owning module address, or ``"self"`` for top-level buffers.
    """

    if "." not in address:
        return "self"
    return address.rsplit(".", 1)[0]


def _module_addresses_for_buffer_version(version: object) -> set[str]:
    """Return module addresses claimed by a buffer version node.

    Parameters
    ----------
    version:
        Buffer op version.

    Returns
    -------
    set[str]
        Pass-stripped module addresses.
    """

    claims: set[str] = set()
    for claim in _module_claims(cast("Op", version)):
        address = _module_claim_address(claim)
        if address:
            claims.add(address)
    return claims


def _active_buffer_consumer_module_addresses(ml: "Trace", version: object) -> set[str]:
    """Return module addresses for real active consumers of a buffer version.

    Parameters
    ----------
    ml:
        Trace containing the graph.
    version:
        Buffer op version whose children should be inspected.

    Returns
    -------
    set[str]
        Pass-stripped module addresses claimed by child ops consuming the
        buffer version.
    """

    addresses: set[str] = set()
    for child_label in getattr(version, "children", ()) or ():
        try:
            child = ml[child_label]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        if getattr(child, "is_output", False):
            continue
        for claim in _module_claims(child):
            address = _module_claim_address(claim)
            if address:
                addresses.add(address)
    return addresses


def _check_buffer_write_versions(ml: "Trace", buf: object, name: str) -> None:
    """Check write-version buffer metadata domains and resolvable populated fields.

    Parameters
    ----------
    ml:
        Trace containing buffer metadata.
    buf:
        Buffer entity from the trace buffer accessor.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If write-kind domains are invalid, buffer passes are not dense as a
        set, or populated source labels fail to resolve.
    """

    valid_write_kinds = {"reassign", "inplace", "fused", "data_reassign"}
    address = getattr(buf, "address", None)
    versions = list(getattr(buf, "versions", ()) or ())
    write_versions = [
        version for version in versions if getattr(version, "buffer_write_kind", None) is not None
    ]
    if not write_versions:
        return

    passes: list[int] = [
        buffer_pass
        for version in versions
        if isinstance(buffer_pass := getattr(version, "buffer_pass", None), int)
    ]
    for version in write_versions:
        label = getattr(version, "layer_label", type(version).__name__)
        write_kind = getattr(version, "buffer_write_kind", None)
        if write_kind not in valid_write_kinds:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' has invalid buffer_write_kind "
                f"{write_kind!r}",
            )
        buffer_pass = getattr(version, "buffer_pass", None)
        if not isinstance(buffer_pass, int) or buffer_pass < 1:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' has invalid buffer_pass {buffer_pass!r}",
            )

        source = getattr(version, "buffer_source", None)
        if source is not None and _resolve_trace_label(ml, source) is None:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' has unresolved buffer_source {source!r}",
            )

    expected_passes = set(range(1, max(passes) + 1))
    if set(passes) != expected_passes:
        raise MetadataInvariantError(
            name,
            f"Buffer '{address}' write buffer_pass values must be dense as a set, got "
            f"{sorted(passes)!r}",
        )


def _resolve_trace_label(ml: "Trace", label: str) -> str | None:
    """Resolve a final or raw layer label to a known trace label.

    Parameters
    ----------
    ml:
        Trace containing raw/final lookup maps.
    label:
        Raw or final label to resolve.

    Returns
    -------
    str | None
        Resolved label when present in the trace, otherwise ``None``.
    """

    all_keys = set(getattr(ml, "layer_dict_all_keys", {}))
    final_label = getattr(ml, "_raw_to_final_layer_labels", {}).get(label)
    if final_label in all_keys:
        return final_label
    if label in all_keys:
        return label
    return None


def _check_buffer_replay_validated_versions(ml: "Trace", buf: object, name: str) -> None:
    """Check explicit successful buffer replay claims have identity-replay evidence.

    Parameters
    ----------
    ml:
        Trace containing buffer metadata.
    buf:
        Buffer entity from the trace buffer accessor.
    name:
        Invariant name used in raised errors.

    Raises
    ------
    MetadataInvariantError
        If a write version asserts ``buffer_replay_validated=True`` without the
        source-parent and saved-argument evidence created by buffer replay
        postprocessing.
    """

    address = getattr(buf, "address", None)
    for version in getattr(buf, "versions", ()) or ():
        if getattr(version, "buffer_write_kind", None) is None:
            continue
        if getattr(version, "buffer_replay_validated", None) is not True:
            continue
        label = getattr(version, "layer_label", type(version).__name__)
        source = getattr(version, "buffer_source", None)
        if source is None:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' claims replay validation without "
                "buffer_source",
            )
        resolved_source = _resolve_trace_label(ml, source)
        if resolved_source is None:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' replay source {source!r} does not resolve",
            )
        parents = set(getattr(version, "parents", ()) or ())
        if resolved_source not in parents:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' replay source is not a parent",
            )
        parent_args = getattr(version, "parent_arg_positions", {}) or {}
        arg_positions = parent_args.get("args", {}) if isinstance(parent_args, Mapping) else {}
        resolved_arg0 = _resolve_trace_label(ml, arg_positions.get(0, ""))
        if resolved_arg0 != resolved_source:
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' replay source is not args[0]",
            )
        if not getattr(version, "saved_args", None):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' claims replay validation without "
                "saved source argument",
            )
        if (
            getattr(version, "out", None) is None
            or getattr(ml[resolved_source], "out", None) is None
        ):
            raise MetadataInvariantError(
                name,
                f"Buffer '{address}' version '{label}' claims replay validation without "
                "comparable payloads",
            )


# ---------------------------------------------------------------------------
# L. Equivalence group symmetry
# ---------------------------------------------------------------------------


def _check_equivalence_symmetry(ml: "Trace") -> None:
    """Check L: op_equivalence_classes groups reference valid Op labels.

    Validates:
    - Each equivalence set value is actually a set.
    - All labels in equivalence sets exist in op_labels.
    """
    name = "equivalence_symmetry"
    label_set = set(ml.op_labels)

    # op_equivalence_classes is keyed by equivalence type descriptors (not Op labels),
    # with values being sets of Op labels in that equivalence group.
    for eq_type, equiv_set in ml.op_equivalence_classes.items():
        if not isinstance(equiv_set, set):
            raise MetadataInvariantError(
                name,
                f"op_equivalence_classes['{eq_type}'] is not a set",
            )
        for label in equiv_set:
            if label not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"op_equivalence_classes['{eq_type}'] contains '{label}' not in op_labels",
                )

    # Each Op that appears in any equivalence group should exist.
    all_equiv_labels = set()
    for equiv_set in ml.op_equivalence_classes.values():
        all_equiv_labels.update(equiv_set)
    extra = all_equiv_labels - label_set
    if extra:
        raise MetadataInvariantError(
            name,
            f"op_equivalence_classes contains labels not in op_labels: {extra}",
        )


# ---------------------------------------------------------------------------
# M. Graph ordering invariants
# ---------------------------------------------------------------------------

# Raw labels (e.g., "l_42") are internal identifiers assigned during capture
# and must be replaced by human-readable labels during postprocessing.
_RAW_LABEL_PATTERN = re.compile(r"^l_\d+$")


def _check_graph_ordering(ml: "Trace") -> None:
    """Check M: graph ordering invariants.

    Validates:
    - raw_index is unique across all layers and monotonically
      increasing in layer_list order.
    - step_index is unique among computational layers (non-input, non-buffer,
      non-output).
    - Topological order: every parent's raw_index < child's.
    - No raw labels (``l_\\d+``) survive postprocessing.
    """
    name = "graph_ordering"

    # raw_index uniqueness and monotonicity
    seen_rt_nums: dict[int, str] = {}
    prev_rt = -1
    for lpl in ml.layer_list:
        rt = lpl.raw_index
        if rt in seen_rt_nums:
            raise MetadataInvariantError(
                name,
                f"Duplicate raw_index={rt}: '{seen_rt_nums[rt]}' and '{lpl.layer_label}'",
            )
        seen_rt_nums[rt] = lpl.layer_label
        if rt <= prev_rt:
            raise MetadataInvariantError(
                name,
                f"raw_index not monotonically increasing: "
                f"{prev_rt} then {rt} at '{lpl.layer_label}'",
            )
        prev_rt = rt

    # step_index uniqueness among computational layers
    input_set = set(ml.input_layers)
    buffer_set = set(ml.buffer_layers)
    output_set = set(ml.output_layers)
    seen_op_nums: dict[int, str] = {}
    for lpl in ml.layer_list:
        label = lpl.layer_label
        if label in input_set or label in buffer_set or label in output_set:
            continue
        op = lpl.step_index
        if op is not None:
            if op in seen_op_nums:
                raise MetadataInvariantError(
                    name,
                    f"Duplicate step_index={op}: '{seen_op_nums[op]}' and '{label}'",
                )
            seen_op_nums[op] = label

    # Topological order: parent.raw_index < child.raw_index
    rt_map = {lpl.layer_label: lpl.raw_index for lpl in ml.layer_list}
    for lpl in ml.layer_list:
        for p in lpl.parents:
            if rt_map.get(p, -1) >= lpl.raw_index:
                raise MetadataInvariantError(
                    name,
                    f"Topological violation: parent '{p}' (rt={rt_map.get(p)}) "
                    f">= child '{lpl.layer_label}' (rt={lpl.raw_index})",
                )

    # No raw labels survive postprocessing
    for label in ml.layer_labels:
        if _RAW_LABEL_PATTERN.match(label):
            raise MetadataInvariantError(name, f"Raw label '{label}' survived postprocessing")


# ---------------------------------------------------------------------------
# N. Layer equivalence / loop detection invariants
# ---------------------------------------------------------------------------


def _check_loop_detection_invariants(ml: "Trace") -> None:
    """Check N: loop detection / recurrent_ops invariants.

    Validates per-layer:
    - recurrent_ops is non-empty and includes self.
    - Symmetry: all members agree on the same group.
    - All members share: layer_label, equivalence_class,
      func_name (for computational layers).
    - num_passes == len(recurrent_ops).
    - Pass numbering within group is contiguous {1..N}.

    Validates cross-layer:
    - Parameter sharing rule: layers with same (func_name,
      sorted(_param_barcodes)) must share layer_label.
    - Equivalence group consistency: all members of a recurrent_ops
      group belong to the same Trace.op_equivalence_classes set.

    Note: subgraph-level adjacency (Rule 3 from loop_detection.py) cannot
    be verified post-hoc from metadata alone.
    """
    name = "loop_detection"
    label_set = set(ml.op_labels)

    # Build same-layer groups from the authoritative recurrent_ops lists
    # Key: frozenset of labels, Value: list of OpLogs in the group
    groups_seen: dict[frozenset[str], list[str]] = {}

    for lpl in ml.layer_list:
        slo = lpl.recurrent_ops
        if not slo:
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}' has empty recurrent_ops",
            )

        # All members in recurrent_ops must exist
        for member in slo:
            if member not in label_set:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}' recurrent_ops contains '{member}' not in op_labels",
                )

        # Self-inclusion
        if lpl.label not in slo:
            raise MetadataInvariantError(
                name,
                f"Op '{lpl.label}' not in its own recurrent_ops",
            )

        # Symmetry: all members agree on the group
        for member_label in slo:
            member = ml[member_label]
            if set(member.recurrent_ops) != set(slo):
                raise MetadataInvariantError(
                    name,
                    f"Asymmetric recurrent_ops: '{lpl.layer_label}' has "
                    f"{sorted(slo)} but '{member_label}' has "
                    f"{sorted(member.recurrent_ops)}",
                )

        # All members share layer_label
        for member_label in slo:
            member = ml[member_label]
            if member.layer_label != lpl.layer_label:
                raise MetadataInvariantError(
                    name,
                    f"recurrent_ops inconsistency: '{lpl.layer_label}' "
                    f"(no_pass='{lpl.layer_label}') and '{member_label}' "
                    f"(no_pass='{member.layer_label}') differ",
                )

        # All members share equivalence_class
        for member_label in slo:
            member = ml[member_label]
            if member.equivalence_class != lpl.equivalence_class:
                raise MetadataInvariantError(
                    name,
                    f"recurrent_ops type mismatch: '{lpl.layer_label}' "
                    f"type='{lpl.equivalence_class}' vs '{member_label}' "
                    f"type='{member.equivalence_class}'",
                )

        # All members share func_name (for computational layers)
        if not (lpl.is_input or lpl.is_buffer or lpl.is_output):
            for member_label in slo:
                member = ml[member_label]
                if member.func_name != lpl.func_name:
                    raise MetadataInvariantError(
                        name,
                        f"recurrent_ops func mismatch: '{lpl.layer_label}' "
                        f"func='{lpl.func_name}' vs '{member_label}' "
                        f"func='{member.func_name}'",
                    )

        # num_passes == len(recurrent_ops)
        if lpl.num_passes != len(slo):
            raise MetadataInvariantError(
                name,
                f"Layer '{lpl.layer_label}': num_passes={lpl.num_passes} "
                f"!= len(recurrent_ops)={len(slo)}",
            )

        # Pass numbering: unique {1..N}
        group_key = frozenset(slo)
        if group_key not in groups_seen:
            pass_indices = []
            for member_label in slo:
                member = ml[member_label]
                pass_indices.append(member.pass_index)
            expected = set(range(1, len(slo) + 1))
            actual = set(pass_indices)
            if actual != expected:
                raise MetadataInvariantError(
                    name,
                    f"Pass numbering for group {sorted(slo)}: expected {expected}, got {actual}",
                )
            groups_seen[group_key] = slo

    # Rule 1: Parameter sharing invariant.
    # Layers with the same func_name, identical sorted(_param_barcodes), and the
    # same output-specific equivalence class must share layer_label. The
    # equivalence class keeps distinct outputs of a multi-output parameterized op
    # from being collapsed into one logical layer.
    param_groups: dict[tuple[str, tuple[str, ...], str], list[Op]] = defaultdict(list)
    for lpl in ml.layer_list:
        if lpl.uses_params and lpl._param_barcodes:
            key = (lpl.func_name, tuple(sorted(lpl._param_barcodes)), lpl.equivalence_class)
            param_groups[key].append(lpl)

    for param_key, layers in param_groups.items():
        if len(layers) > 1:
            no_call_labels = {lpl.layer_label for lpl in layers}
            if len(no_call_labels) > 1:
                raise MetadataInvariantError(
                    name,
                    f"Param sharing violation: layers with same param barcodes "
                    f"{param_key} have different layer_label: {no_call_labels}",
                )

    # Equivalence group ↔ same_layer consistency: all members of a
    # recurrent_ops group must belong to the same equivalence set.
    # Note: Trace.op_equivalence_classes keys use the pre-module-suffix type
    # (from loop_detection), while per-layer equivalence_class has
    # a module suffix appended by control_flow.py. So we check group membership
    # consistency, not exact key matching.
    op_label_to_equiv_key: dict[str, str] = {}
    for eq_type, equiv_set in ml.op_equivalence_classes.items():
        for label in equiv_set:
            op_label_to_equiv_key[label] = eq_type

    for group_key in groups_seen:
        slo = list(group_key)
        if len(slo) <= 1:
            continue
        # All members of a same-layer group should be in the same equivalence set
        equiv_keys = set()
        for member_label in slo:
            member = ml[member_label]
            if member.label in op_label_to_equiv_key:
                equiv_keys.add(op_label_to_equiv_key[member.label])
        equiv_stems = {re.sub(r"_outindex\d+$", "", equiv_key) for equiv_key in equiv_keys}
        if len(equiv_keys) > 1 and len(equiv_stems) > 1:
            raise MetadataInvariantError(
                name,
                f"recurrent_ops group {sorted(slo)} spans multiple equivalence types: {equiv_keys}",
            )

    # Note: subgraph-level adjacency (Rule 3) is verified during BFS in
    # loop_detection.py and cannot be reconstructed from post-hoc data.
    # Non-param multi-pass groups may be the ONLY multi-pass group in a model
    # (e.g., param-free loops like repeated addition), so we cannot require
    # connection to other multi-pass layers.  The checks above (func identity,
    # equiv type, pass numbering, symmetry, param sharing) are sufficient.


# ---------------------------------------------------------------------------
# O. Distance / reachability invariants
# ---------------------------------------------------------------------------


def _check_distance_invariants(ml: "Trace") -> None:
    """Check O: distance and reachability invariants.

    Only runs when ``mark_layer_depths`` was enabled during logging.

    Validates:
    - min_distance <= max_distance for both input and output distances.
    - Input layers have distance_from_input == 0.
    - Output layers have distance_from_output == 0.
    - has_input_ancestor <-> input_ancestors is non-empty.
    - has_output_descendant <-> output_descendants is non-empty.
    - input_ancestors subset of input_layers; output_descendants subset of
      output_layers.
    """
    if not ml.mark_layer_depths:
        return

    name = "distance_invariants"
    input_set = set(ml.input_layers)
    output_set = set(ml.output_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # min <= max for input distances
        if lpl.min_distance_from_input is not None and lpl.max_distance_from_input is not None:
            if lpl.min_distance_from_input > lpl.max_distance_from_input:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{label}': min_distance_from_input="
                    f"{lpl.min_distance_from_input} > max={lpl.max_distance_from_input}",
                )

        # min <= max for output distances
        if lpl.min_distance_to_output is not None and lpl.max_distance_to_output is not None:
            if lpl.min_distance_to_output > lpl.max_distance_to_output:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{label}': min_distance_to_output="
                    f"{lpl.min_distance_to_output} > max={lpl.max_distance_to_output}",
                )

        # Input layers: distance from input == 0
        if label in input_set:
            if lpl.min_distance_from_input != 0 or lpl.max_distance_from_input != 0:
                raise MetadataInvariantError(
                    name,
                    f"Input layer '{label}': distance_from_input should be 0, got "
                    f"min={lpl.min_distance_from_input}, max={lpl.max_distance_from_input}",
                )

        # Output layers: distance from output == 0
        if label in output_set:
            if lpl.min_distance_to_output != 0 or lpl.max_distance_to_output != 0:
                raise MetadataInvariantError(
                    name,
                    f"Output layer '{label}': distance_from_output should be 0, got "
                    f"min={lpl.min_distance_to_output}, max={lpl.max_distance_to_output}",
                )

        # has_input_ancestor ↔ input_ancestors non-empty
        has_ancestors = len(lpl.input_ancestors) > 0
        if lpl.has_input_ancestor != has_ancestors:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': has_input_ancestor={lpl.has_input_ancestor} but "
                f"len(input_ancestors)={len(lpl.input_ancestors)}",
            )

        # has_output_descendant ↔ output_descendants non-empty
        has_descendents = len(lpl.output_descendants) > 0
        if lpl.has_output_descendant != has_descendents:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': has_output_descendant={lpl.has_output_descendant} but "
                f"len(output_descendants)={len(lpl.output_descendants)}",
            )

        # input_ancestors subset of input_layers
        extra_ancestors = lpl.input_ancestors - input_set
        if extra_ancestors:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': input_ancestors contains labels not in "
                f"input_layers: {extra_ancestors}",
            )

        # output_descendants subset of output_layers
        extra_desc = lpl.output_descendants - output_set
        if extra_desc:
            raise MetadataInvariantError(
                name,
                f"Layer '{label}': output_descendants contains labels not in "
                f"output_layers: {extra_desc}",
            )


# ---------------------------------------------------------------------------
# P. Graph connectivity invariants
# ---------------------------------------------------------------------------


def _check_graph_connectivity(ml: "Trace") -> None:
    """Check P: graph connectivity invariants.

    Validates:
    - Every non-input, non-buffer, non-internally-initialized, non-output
      layer has at least one parent (no dangling computational nodes).
    - _orphan_labels (removed during postprocessing) do NOT appear in the
      active layer_labels (they were pruned from the graph).
    """
    name = "graph_connectivity"
    label_set = set(ml.layer_labels)
    input_set = set(ml.input_layers)
    buffer_set = set(ml.buffer_layers)

    for lpl in ml.layer_list:
        label = lpl.layer_label

        # Non-input, non-buffer, non-internally-initialized layers must have parents
        if (
            label not in input_set
            and label not in buffer_set
            and not lpl.is_internal_source
            and not lpl.is_output
            and len(lpl.parents) == 0
        ):
            raise MetadataInvariantError(
                name,
                f"Layer '{label}' has no parents but is not input, buffer, "
                f"internally initialized, or output",
            )

    # _orphan_labels is a subset of all known labels (pre-removal)
    orphan_set = set(ml._orphan_labels)
    # Orphans should NOT appear in the active layer_list (they were removed)
    orphan_in_list = orphan_set & label_set
    if orphan_in_list:
        raise MetadataInvariantError(
            name,
            f"_orphan_labels contains labels still in layer_labels: {orphan_in_list}",
        )


# ---------------------------------------------------------------------------
# Q. Module containment logical consistency
# ---------------------------------------------------------------------------


def _check_module_containment_logic(ml: "Trace") -> None:
    """Check Q: module containment logical consistency.

    Validates:
    - Address tree is acyclic (walking address_parent chain reaches None
      without revisiting a node).
    - Root module 'self' has address_depth == 0; others have
      address_depth == addr.count('.') + 1.
    - Per-layer modules (call nesting stack):
      - Last element matches module.
      - Every element is a known module address.
      - No duplicate addresses (can't be inside the same module twice).
    """
    name = "module_containment_logic"
    mod_accessor = ml.modules

    # Build set of known module addresses
    known_addrs = set()
    for mod_log in mod_accessor:
        known_addrs.add(mod_log.address)

    for mod_log in mod_accessor:
        addr = mod_log.address

        # Address tree acyclicity: walk address_parent to root
        visited: set[str] = set()
        current: str | None = addr
        while current is not None:
            if current in visited:
                raise MetadataInvariantError(
                    name,
                    f"Cycle in address_parent chain starting from '{addr}': revisited '{current}'",
                )
            visited.add(current)
            try:
                parent_mod: Module = mod_accessor[current]  # type: ignore[assignment]
            except (KeyError, IndexError):
                break
            current = parent_mod.address_parent

        # Address depth consistency
        if addr == "self":
            if mod_log.address_depth != 0:
                raise MetadataInvariantError(
                    name,
                    f"Root module 'self' has address_depth={mod_log.address_depth}, expected 0",
                )
        else:
            expected_depth = addr.count(".") + 1
            if mod_log.address_depth != expected_depth:
                raise MetadataInvariantError(
                    name,
                    f"Module '{addr}': address_depth={mod_log.address_depth} "
                    f"!= expected {expected_depth} (addr.count('.')+1)",
                )

    # Per-layer: modules path validity
    # Format after postprocessing: list of "addr:pass" strings, ordered from
    # outermost enclosing submodule to innermost. Does NOT include "self".
    for lpl in ml.layer_list:
        nested = lpl.modules
        if not nested:
            continue

        # Leaf consistency: last element matches module
        if lpl.module is not None:
            if nested[-1] != lpl.module:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': last nested module '{nested[-1]}' "
                    f"!= module '{lpl.module}'",
                )

        # Path validity: each element must be a known module, and no
        # duplicate addresses (a module can't appear twice in the same call
        # stack).  We do NOT check address depth ordering — address depth
        # (position in the nn.Module tree) is independent of call nesting
        # depth (position on the forward() call stack).  A module at a
        # shallow address can be called from inside a deeply-addressed
        # module's forward(), e.g., encoder.blocks.1.1.attention calling
        # encoder.attention_structure.sin_dropout.
        seen_addrs = set()
        for entry in nested:
            addr = entry.split(":")[0] if ":" in entry else entry
            if addr not in known_addrs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': nested path contains unknown "
                    f"module address '{addr}'",
                )
            if addr in seen_addrs:
                raise MetadataInvariantError(
                    name,
                    f"Layer '{lpl.layer_label}': duplicate module address "
                    f"'{addr}' in nested path {nested}",
                )
            seen_addrs.add(addr)


# ---------------------------------------------------------------------------
# R. Lookup key bidirectionality
# ---------------------------------------------------------------------------


def _check_lookup_key_consistency(ml: "Trace") -> None:
    """Check R: lookup key bidirectional consistency.

    Validates:
    - _lookup_keys_to_layer_num_dict (forward: key->num) and
      _layer_num_to_lookup_keys_dict (reverse: num->[keys]) are consistent:
      forward[key]=num implies key in reverse[num], and vice versa.
    - _raw_to_final_layer_labels and _final_to_raw_layer_labels are inverse
      bijections.
    - All final labels in the raw->final map exist in layer_labels.
    """
    name = "lookup_key_consistency"

    # _lookup_keys_to_layer_num_dict maps key→num (last assigned wins).
    # _layer_num_to_lookup_keys_dict maps num→[keys] (accumulates all assignments).
    # Forward → reverse: for every (key, num) in forward, key must be in reverse[num].
    fwd = ml._lookup_keys_to_layer_num_dict
    rev = ml._layer_num_to_lookup_keys_dict

    for key, num in fwd.items():
        if num not in rev:
            raise MetadataInvariantError(
                name,
                f"_lookup_keys_to_layer_num_dict['{key}']={num} but "
                f"{num} not in _layer_num_to_lookup_keys_dict",
            )
        if key not in rev[num]:
            raise MetadataInvariantError(
                name,
                f"_lookup_keys_to_layer_num_dict['{key}']={num} but "
                f"'{key}' not in _layer_num_to_lookup_keys_dict[{num}]",
            )

    # Reverse → forward: every key in reverse must exist in forward (but may
    # point to a different num if the key was reassigned to a later layer).
    for num, keys in rev.items():
        for key in keys:
            if key not in fwd:
                raise MetadataInvariantError(
                    name,
                    f"_layer_num_to_lookup_keys_dict[{num}] has '{key}' but "
                    f"'{key}' not in _lookup_keys_to_layer_num_dict",
                )

    # _raw_to_final_layer_labels ↔ _final_to_raw_layer_labels
    raw_fwd = ml._raw_to_final_layer_labels
    raw_rev = ml._final_to_raw_layer_labels

    for raw, final in raw_fwd.items():
        if final not in raw_rev:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels['{raw}']='{final}' but "
                f"'{final}' not in _final_to_raw_layer_labels",
            )
        if raw_rev[final] != raw:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels['{raw}']='{final}' but "
                f"_final_to_raw_layer_labels['{final}']='{raw_rev[final]}'",
            )

    for final, raw in raw_rev.items():
        if raw not in raw_fwd:
            raise MetadataInvariantError(
                name,
                f"_final_to_raw_layer_labels['{final}']='{raw}' but "
                f"'{raw}' not in _raw_to_final_layer_labels",
            )

    # All final labels are valid lookup labels. Multi-pass raw labels map to
    # pass-qualified Op labels, while single-pass raw labels may map to Layer
    # labels for compatibility lookup.
    label_set = set(ml.layer_labels) | set(ml.op_labels)
    for final in raw_fwd.values():
        if final not in label_set:
            raise MetadataInvariantError(
                name,
                f"_raw_to_final_layer_labels maps to '{final}' which is not a valid label",
            )


METADATA_INVARIANT_CONTRACTS: tuple[MetadataInvariantContract, ...] = (
    # Setup checks.
    MetadataInvariantContract(
        "backend_identity_invariants",
        _check_backend_identity_invariants,
        "all",
    ),
    # --- Phase 1: structural invariants (A-L) ---
    MetadataInvariantContract("trace_self_consistency", _check_trace_self_consistency, "all"),
    MetadataInvariantContract(
        "region_replay_provenance",
        _check_region_replay_provenance,
        "all",
    ),
    MetadataInvariantContract(
        "backward_graph_invariants",
        _check_backward_graph_invariants,
        "torch",
    ),
    MetadataInvariantContract(
        "non_torch_backward_inert",
        _check_non_torch_backward_inert,
        "non_torch",
    ),
    MetadataInvariantContract(
        "backend_neutral_accessor_refs",
        _check_backend_neutral_accessor_refs,
        "all",
    ),
    MetadataInvariantContract(
        "backend_neutral_module_mode_invariants",
        _check_backend_neutral_module_mode_invariants,
        "non_torch",
    ),
    MetadataInvariantContract("special_layer_lists", _check_special_layer_lists, "torch"),
    MetadataInvariantContract("graph_topology", _check_graph_topology, "torch"),
    MetadataInvariantContract(
        "backend_neutral_graph_topology",
        _check_backend_neutral_graph_topology,
        "non_torch",
    ),
    MetadataInvariantContract(
        "edge_use_parent_arg_consistency",
        _check_edge_use_parent_arg_invariants,
        "torch",
    ),
    MetadataInvariantContract("op_log_fields", _check_op_log_fields, "torch"),
    MetadataInvariantContract(
        "payload_metadata_invariants",
        _check_payload_metadata_invariants,
        "torch",
    ),
    MetadataInvariantContract("recurrence_invariants", _check_recurrence_invariants, "torch"),
    MetadataInvariantContract("branching_invariants", _check_branching_invariants, "torch"),
    MetadataInvariantContract("conditional_invariants", _check_conditional_invariants, "torch"),
    MetadataInvariantContract(
        "layer_pass_layer_log_xrefs",
        _check_layer_pass_to_layer_log_xrefs,
        "torch",
    ),
    MetadataInvariantContract(
        "module_layer_containment",
        _check_module_layer_containment,
        "torch",
    ),
    MetadataInvariantContract("module_hierarchy", _check_module_hierarchy, "torch"),
    MetadataInvariantContract("param_xrefs", _check_param_xrefs, "torch"),
    MetadataInvariantContract("buffer_xrefs", _check_buffer_xrefs, "torch"),
    MetadataInvariantContract(
        "equivalence_symmetry",
        _check_equivalence_symmetry,
        "torch",
    ),
    # --- Phase 2: semantic invariants (M-R) ---
    MetadataInvariantContract("graph_ordering", _check_graph_ordering, "all"),
    MetadataInvariantContract(
        "loop_detection_invariants",
        _check_loop_detection_invariants,
        "torch",
    ),
    MetadataInvariantContract(
        "pass_count_consistency",
        _check_pass_count_consistency,
        "torch",
    ),
    MetadataInvariantContract("distance_invariants", _check_distance_invariants, "torch"),
    MetadataInvariantContract("graph_connectivity", _check_graph_connectivity, "torch"),
    MetadataInvariantContract(
        "module_containment_logic",
        _check_module_containment_logic,
        "torch",
    ),
    MetadataInvariantContract(
        "lookup_key_consistency",
        _check_lookup_key_consistency,
        "all",
    ),
    MetadataInvariantContract(
        "func_call_id_consistency",
        check_func_call_id_invariant,
        "torch",
    ),
)


def _metadata_invariant_contracts_for_trace(
    trace: "Trace",
) -> tuple[MetadataInvariantContract, ...]:
    """Return metadata invariant contracts applicable to ``trace``.

    Parameters
    ----------
    trace:
        Trace whose backend selects the contract subset.

    Returns
    -------
    tuple[MetadataInvariantContract, ...]
        Ordered invariant contracts matching the trace backend and capability
        requirements.
    """

    from ..backends import get_backend_spec

    spec = get_backend_spec(getattr(trace, "backend", "torch"))
    torch_spec = get_backend_spec("torch")
    backend_family: Literal["torch", "non_torch"] = "torch" if spec is torch_spec else "non_torch"
    return _metadata_invariant_contracts_for_backend(backend_family, spec=spec)


def _metadata_invariant_contracts_for_backend(
    backend_family: Literal["torch", "non_torch"],
    *,
    spec: "BackendSpec | None" = None,
) -> tuple[MetadataInvariantContract, ...]:
    """Return ordered metadata invariant contracts for a backend family.

    Parameters
    ----------
    backend_family:
        ``"torch"`` for torch traces, otherwise ``"non_torch"``.
    spec:
        Optional backend registry spec used for capability-gated contracts.

    Returns
    -------
    tuple[MetadataInvariantContract, ...]
        Ordered invariant contracts whose backend and capability contracts match.
    """

    return tuple(
        contract
        for contract in METADATA_INVARIANT_CONTRACTS
        if _metadata_invariant_applies(contract, backend_family, spec)
    )


def _metadata_invariant_applies(
    contract: MetadataInvariantContract,
    backend_family: Literal["torch", "non_torch"],
    spec: "BackendSpec | None",
) -> bool:
    """Return whether ``contract`` applies to a backend family and spec.

    Parameters
    ----------
    contract:
        Metadata invariant contract to evaluate.
    backend_family:
        Backend family selected from the trace backend.
    spec:
        Backend registry spec, when available.

    Returns
    -------
    bool
        True when backend applicability and optional capability gates match.
    """

    if contract.applies_to not in {"all", backend_family}:
        return False
    if contract.requires_capability is None:
        return True
    if spec is None:
        return False
    return bool(getattr(spec.capabilities, contract.requires_capability, False))
