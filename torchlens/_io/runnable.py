"""Sparse runnable descriptor production from a cooked :class:`Trace` projection."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from hashlib import sha256
import json
import platform
from typing import Any, cast

import torch

from .. import __version__ as TORCHLENS_VERSION
from ..data_classes._state_adapter import state_items
from ..errors import RunnablePreflightError
from ..intervention.types import (
    CapturedArgTemplate,
    FunctionRegistryKey,
    LiteralTensor,
    LiteralValue,
    ParentRef,
    Unsupported,
)
from ..ir.container import DataclassField, DictKey, HFKey, NamedField, TupleIndex
from ..ir.container_registry import ModelSite, Role
from ..runnable import (
    RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
    RUNNABLE_CALL_RECIPE_VERSION,
    RUNNABLE_INITIALIZER_POLICY_VERSION,
    RUNNABLE_TLSPEC_SCHEMA_VERSION,
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    CallableRegistryEntry,
    ControlWitness,
    ControlWitnessKind,
    InputSlotBinding,
    LiteralArgumentRef,
    LiteralAtom,
    LiteralAtomKind,
    LiteralMapping,
    LiteralMappingEntry,
    LiteralSequence,
    LiteralSequenceKind,
    LiteralTorchSymbol,
    LiteralTupleKey,
    NonTensorLiteral,
    PayloadLayerDescriptor,
    PayloadLayersDescriptor,
    ProducerPreflight,
    RunnableCallDescriptor,
    RunnableCompatibility,
    RunnableDiagnostic,
    RunnableErrorCode,
    RunnableRngProfile,
    SparseRunDescriptor,
    SlotByteDigest,
    StateByteDigest,
    StateSlotBinding,
    StateSlotRole,
    TensorArgumentRef,
    TensorSlotDescriptor,
    TensorSlotRole,
    TensorUseSite,
    WitnessCompleteness,
)


@dataclass(slots=True)
class _SlotDraft:
    """Mutable producer-only accumulator for a frozen tensor-slot descriptor."""

    slot_id: str
    role: TensorSlotRole
    shape: tuple[int, ...]
    dtype: str
    device_type: str
    device_index: int | None
    mutable: bool = False
    version_of: str | None = None
    producer_slot_id: str | None = None
    output_path: tuple[str | int, ...] | None = None
    input_binding: InputSlotBinding | None = None
    state_binding: StateSlotBinding | None = None
    use_sites: list[TensorUseSite] | None = None

    def freeze(self) -> TensorSlotDescriptor:
        """Freeze this draft into the Stage-0 descriptor type.

        Returns
        -------
        TensorSlotDescriptor
            Immutable value-free slot descriptor.
        """

        return TensorSlotDescriptor(
            slot_id=self.slot_id,
            role=self.role,
            use_sites=tuple(self.use_sites or ()),
            shape=self.shape,
            dtype=self.dtype,
            rank=len(self.shape),
            device_type=self.device_type,
            device_index=self.device_index,
            mutable=self.mutable,
            version_of=self.version_of,
            producer_slot_id=self.producer_slot_id,
            output_path=self.output_path,
            input_binding=self.input_binding,
            state_binding=self.state_binding,
        )


class _UnsupportedLiteralError(ValueError):
    """Internal signal for a value outside the frozen literal grammar."""


def build_sparse_run_descriptor(trace: Any) -> SparseRunDescriptor:
    """Project a cooked Trace into the frozen sparse runnable descriptor.

    Parameters
    ----------
    trace:
        Fully postprocessed Trace produced from a sealed capture core.

    Returns
    -------
    SparseRunDescriptor
        Descriptor with a complete producer preflight report. A failed report
        is returned for diagnostics and must not be written as runnable.
    """

    diagnostics: list[RunnableDiagnostic] = []
    backend = str(getattr(trace, "backend", "torch"))
    if backend != "torch":
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY,
                f"Sparse runnable rung 1 supports only the torch backend, not {backend!r}.",
                detection_stage="producer_backend",
            )
        )

    ops = list(getattr(trace, "layer_list", ()))
    op_by_alias = _op_alias_index(trace, ops)
    slot_drafts, slot_for_op = _build_op_slot_drafts(trace, ops, diagnostics)
    _build_child_version_slot_drafts(trace, ops, slot_drafts, slot_for_op)
    state_slots = _build_parameter_slot_drafts(trace)
    slot_drafts.update(state_slots)
    _add_persistent_buffer_slot_drafts(trace, slot_drafts)

    registry_entries: list[CallableRegistryEntry] = []
    registry_id_by_key: dict[FunctionRegistryKey, str] = {}
    calls: list[RunnableCallDescriptor] = []
    producer_call_by_slot: dict[str, str] = {}
    grouped_ops = _group_computational_ops(ops)
    for call_number, call_ops in grouped_ops:
        representative = call_ops[0]
        call_id = f"call:{call_number}"
        func_id = getattr(representative, "func_id", None)
        if not isinstance(func_id, FunctionRegistryKey):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_CALLABLE_REF,
                    "Computational call has no cooked FunctionRegistryKey; capture with "
                    "intervention_ready=True before saving level='runnable'.",
                    affected_ops=tuple(str(op.label) for op in call_ops),
                    detection_stage="producer_callable",
                )
            )
            continue
        registry_id = registry_id_by_key.get(func_id)
        if registry_id is None:
            registry_id = f"callable:{len(registry_entries) + 1}"
            registry_id_by_key[func_id] = registry_id
            registry_entries.append(CallableRegistryEntry(registry_id=registry_id, key=func_id))

        template = getattr(representative, "args_template", None)
        if not isinstance(template, CapturedArgTemplate):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                    "Computational call has no cooked argument template.",
                    registry_id=registry_id,
                    affected_ops=tuple(str(op.label) for op in call_ops),
                    detection_stage="producer_call_recipe",
                )
            )
            continue

        tensor_args, literal_args = _build_call_arguments(
            representative,
            template,
            call_id=call_id,
            registry_id=registry_id,
            op_by_alias=op_by_alias,
            slot_for_op=slot_for_op,
            slot_drafts=slot_drafts,
            diagnostics=diagnostics,
        )
        output_slot_ids = tuple(slot_for_op[id(op)] for op in call_ops)
        for output_slot_id in output_slot_ids:
            producer_call_by_slot[output_slot_id] = call_id
            for version_slot_id, draft in slot_drafts.items():
                if draft.version_of == output_slot_id:
                    producer_call_by_slot[version_slot_id] = call_id
        parent_call_ids = tuple(
            dict.fromkeys(
                producer_call_by_slot[argument.slot_id]
                for argument in tensor_args
                if argument.slot_id in producer_call_by_slot
            )
        )
        calls.append(
            RunnableCallDescriptor(
                call_id=call_id,
                op_labels=tuple(str(op.label) for op in call_ops),
                registry_id=registry_id,
                dispatch_kind=func_id.dispatch_kind,
                argument_names=tuple(
                    str(name) for name in getattr(representative, "arg_names", ())
                ),
                num_positional_args=int(getattr(representative, "num_pos_args", 0)),
                num_keyword_args=int(getattr(representative, "num_kwargs", 0)),
                tensor_arguments=tuple(tensor_args),
                literal_arguments=tuple(literal_args),
                output_slot_ids=output_slot_ids,
                parent_call_ids=parent_call_ids,
                is_inplace=bool(getattr(representative, "is_inplace", False)),
                runtime_fingerprint=_runtime_fingerprint(representative, func_id, call_ops),
            )
        )

    _mark_inplace_versions(calls, slot_drafts)
    witnesses, completeness = _build_control_witnesses(trace, ops, diagnostics)
    literal_witnesses, saw_opaque_leaf = _input_literal_witnesses(trace, start_order=len(witnesses))
    witnesses.extend(literal_witnesses)
    if saw_opaque_leaf and completeness is WitnessCompleteness.COMPLETE:
        # An opaque non-tensor input leaf cannot be re-verified, so its control
        # dependency is unobserved: downgrade to keep the run honest
        # (UNVERIFIABLE + NOT_APPLICABLE), never a false VERIFIED/ATTESTED.
        completeness = WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
    diagnostics.extend(_preflight_output_contracts(trace, ops))
    diagnostics = _deduplicate_diagnostics(diagnostics)
    preflight = ProducerPreflight(passed=not diagnostics, diagnostics=tuple(diagnostics))
    descriptor = SparseRunDescriptor(
        capability=RUNNABLE_TLSPEC_SCHEMA_VERSION,
        backend=backend,
        call_recipe=RUNNABLE_CALL_RECIPE_VERSION,
        callable_ref_schema=RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        state_binding="module_path_role_v1",
        input_binding="model_site_io_role_v1",
        control_witness="scalar_bool_and_arm_entry_v1",
        initializer_policy_version=RUNNABLE_INITIALIZER_POLICY_VERSION,
        payload_layers=PayloadLayersDescriptor(
            weights=PayloadLayerDescriptor(present=False, schema="state_dict_v1"),
            activations=PayloadLayerDescriptor(
                present=False,
                schema="selected_activation_v1",
            ),
        ),
        callable_registry=tuple(registry_entries),
        calls=tuple(calls),
        tensor_slots=tuple(draft.freeze() for draft in slot_drafts.values()),
        control_witnesses=tuple(witnesses),
        witness_completeness=completeness,
        rng_profile=_build_rng_profile(trace),
        compatibility=RunnableCompatibility(
            torchlens_version=TORCHLENS_VERSION,
            python_version=platform.python_version(),
            backend_version=str(torch.__version__),
            descriptor_version=RUNNABLE_TLSPEC_SCHEMA_VERSION,
            call_recipe_version=RUNNABLE_CALL_RECIPE_VERSION,
            callable_ref_schema_version=RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
            initializer_policy_version=RUNNABLE_INITIALIZER_POLICY_VERSION,
        ),
        preflight=preflight,
        unsupported_sites=tuple(diagnostics),
    )
    assert_sparse_core_has_no_tensor_payload(descriptor)
    return descriptor


def _build_rng_profile(trace: Any) -> RunnableRngProfile:
    """Record host-RNG reproducibility metadata from the capture-time trace.

    ``_runnable_host_rng_consumed`` is stamped during exhaustive capture by
    bracketing the user forward with side-effect-free host-RNG snapshots;
    ``random_seed`` is the concrete effective seed every capture is seeded with.
    """

    consumed = bool(getattr(trace, "_runnable_host_rng_consumed", False))
    seed = getattr(trace, "random_seed", None)
    capture_seed = int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else None
    return RunnableRngProfile(host_rng_consumed=consumed, capture_seed=capture_seed)


def require_sparse_run_descriptor(trace: Any) -> SparseRunDescriptor:
    """Build a descriptor and reject any failed whole-graph preflight.

    Parameters
    ----------
    trace:
        Cooked Trace to project.

    Returns
    -------
    SparseRunDescriptor
        Passed sparse descriptor.

    Raises
    ------
    RunnablePreflightError
        If any call, slot, input, output, or control site is unsupported.
    """

    descriptor = build_sparse_run_descriptor(trace)
    if not descriptor.preflight.passed:
        raise RunnablePreflightError(
            "Sparse runnable producer preflight failed.",
            code=RunnableErrorCode.SPARSE_PREFLIGHT_FAILED.value,
            diagnostics=descriptor.preflight.diagnostics,
        )
    return descriptor


def with_weight_payload(descriptor: SparseRunDescriptor) -> SparseRunDescriptor:
    """Return a descriptor declaring the optional state-dict payload present.

    Parameters
    ----------
    descriptor:
        Value-free sparse descriptor produced for the runnable artifact.

    Returns
    -------
    SparseRunDescriptor
        Descriptor with only the separately stored weight-layer presence flag
        changed; the sparse tensor-slot recipe remains value-free.
    """

    return replace(
        descriptor,
        payload_layers=replace(
            descriptor.payload_layers,
            weights=replace(descriptor.payload_layers.weights, present=True),
        ),
    )


def with_activation_payload(
    descriptor: SparseRunDescriptor,
    *,
    members: tuple[ActivationPayloadMember, ...],
    original_input_digests: tuple[SlotByteDigest, ...],
    capture_state_digests: tuple[StateByteDigest, ...],
) -> SparseRunDescriptor:
    """Declare one separately stored selected-activation payload family.

    Parameters
    ----------
    descriptor:
        Value-free sparse descriptor produced for the runnable artifact.
    members:
        Exact capture-selected payload membership and logical byte digests.
    original_input_digests:
        Available capture-input slot digests used only for attestation eligibility.
    capture_state_digests:
        Capture-time state digests used to recognize equivalent real state.

    Returns
    -------
    SparseRunDescriptor
        Descriptor with activation-layer metadata; the sparse recipe remains value-free.
    """

    return replace(
        descriptor,
        payload_layers=replace(
            descriptor.payload_layers,
            activations=ActivationPayloadLayerDescriptor(
                present=True,
                schema="selected_activation_v1",
                members=members,
                original_input_digests=original_input_digests,
                capture_state_digests=capture_state_digests,
            ),
        ),
    )


def sparse_descriptor_to_json(descriptor: SparseRunDescriptor) -> dict[str, Any]:
    """Convert a frozen sparse descriptor to deterministic JSON values.

    Parameters
    ----------
    descriptor:
        Passed Stage-0 sparse descriptor.

    Returns
    -------
    dict[str, Any]
        JSON-ready descriptor preserving dataclass field names and enum values.
    """

    value = _json_value(descriptor)
    if not isinstance(value, dict):
        raise TypeError("Sparse descriptor JSON projection must be an object.")
    assert_sparse_core_has_no_tensor_payload(value)
    return value


def assert_sparse_core_has_no_tensor_payload(value: Any) -> None:
    """Assert that a sparse core contains no tensor or tensor-blob value.

    Parameters
    ----------
    value:
        Descriptor, JSON projection, or scrubbed sparse metadata tree.

    Raises
    ------
    AssertionError
        If a tensor, parameter, or portable tensor blob reference is present.
    """

    from . import BlobRef

    seen: set[int] = set()

    def visit(node: Any, path: tuple[str, ...]) -> None:
        """Visit one node in the sparse-core invariant walk."""

        if isinstance(node, (torch.Tensor, BlobRef)):
            dotted_path = ".".join(path) or "<root>"
            raise AssertionError(f"Sparse core tensor payload at {dotted_path}.")
        if node is None or isinstance(node, (str, bytes, bool, int, float, Enum)):
            return
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)
        if is_dataclass(node) and not isinstance(node, type):
            for field in fields(node):
                try:
                    field_value = getattr(node, field.name)
                except AttributeError:
                    continue
                visit(field_value, (*path, field.name))
            return
        if isinstance(node, Mapping):
            for key, item in node.items():
                visit(key, (*path, "<key>"))
                visit(item, (*path, str(key)))
            return
        if isinstance(node, (list, tuple, set, frozenset)):
            for index, item in enumerate(node):
                visit(item, (*path, str(index)))
            return
        for field_name, field_value in state_items(node):
            visit(field_value, (*path, str(field_name)))

    visit(value, ())


def _build_op_slot_drafts(
    trace: Any,
    ops: Sequence[Any],
    diagnostics: list[RunnableDiagnostic],
) -> tuple[dict[str, _SlotDraft], dict[int, str]]:
    """Build source, intermediate, and output slot drafts for cooked ops."""

    drafts: dict[str, _SlotDraft] = {}
    slot_for_op: dict[int, str] = {}
    for op in ops:
        slot_id = f"slot:{op.label}"
        slot_for_op[id(op)] = slot_id
        shape = _shape_tuple(getattr(op, "shape", None))
        dtype = _dtype_name(op)
        if shape is None or dtype is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_TENSOR_SLOT,
                    "Cooked tensor op is missing shape or dtype metadata.",
                    affected_ops=(str(op.label),),
                    detection_stage="producer_tensor_slot",
                )
            )
            shape = shape or ()
            dtype = dtype or "unknown"
        role = _op_slot_role(op)
        if role is TensorSlotRole.BUFFER:
            buffer_names = set((getattr(trace, "_buffer_initial_values", {}) or {}).keys())
            address = getattr(op, "address", None)
            if address not in buffer_names:
                role = TensorSlotRole.CONSTANT_LIKE_TENSOR
                diagnostics.append(
                    _diagnostic(
                        RunnableErrorCode.UNSUPPORTED_TENSOR_CONSTANT,
                        "Internal tensor source is not present in captured named buffer state "
                        "and has no reproducible initializer recipe.",
                        affected_ops=(str(op.label),),
                        detection_stage="producer_tensor_constant",
                        details=(("address", str(address)),),
                    )
                )
        try:
            output_path = _normalize_container_path(getattr(op, "container_path", ()))
        except ValueError:
            # An output/container path with a non-str/int key (tuple, float, ...)
            # cannot be represented in the frozen slot-path vocabulary. Reject
            # honestly with a typed diagnostic instead of a raw ValueError crash;
            # the run is refused rather than advertised-then-broken.
            output_path = ()
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT,
                    "Output container path uses a key that cannot be represented in the "
                    "runnable slot-path vocabulary (only str/int keys are supported).",
                    affected_ops=(str(op.label),),
                    detection_stage="producer_output_binding",
                )
            )
        input_binding = None
        if role is TensorSlotRole.MODEL_INPUT:
            input_binding = _input_binding_for_op(trace, op, diagnostics)
        producer_slot_id = None
        if role is TensorSlotRole.OUTPUT:
            parents = list(getattr(op, "parents", ()))
            if parents:
                parent_op = _resolve_op(trace, parents[0])
                if parent_op is not None:
                    producer_slot_id = f"slot:{parent_op.label}"
        device_type, device_index = _device_parts(getattr(op, "device_ref", None))
        drafts[slot_id] = _SlotDraft(
            slot_id=slot_id,
            role=role,
            shape=shape,
            dtype=dtype,
            device_type=device_type,
            device_index=device_index,
            mutable=bool(getattr(op, "is_inplace", False)),
            producer_slot_id=producer_slot_id,
            output_path=output_path
            if output_path
            else (() if role is TensorSlotRole.OUTPUT else None),
            input_binding=input_binding,
            state_binding=_buffer_binding(op) if role is TensorSlotRole.BUFFER else None,
            use_sites=[],
        )
    return drafts, slot_for_op


def _build_parameter_slot_drafts(trace: Any) -> dict[str, _SlotDraft]:
    """Build value-free parameter slots from cooked Param metadata."""

    drafts: dict[str, _SlotDraft] = {}
    param_logs = getattr(trace, "param_logs", {})
    values = getattr(param_logs, "values", lambda: ())()
    for param in values:
        device_type, device_index = _device_parts(getattr(param, "device_ref", None))
        addresses = tuple(str(address) for address in getattr(param, "all_addresses", ()))
        if not addresses:
            addresses = (str(param.address),)
        alias_group = f"alias:{param.barcode}" if len(addresses) > 1 else None
        for address in addresses:
            module_path, separator, _name = address.rpartition(".")
            slot_id = f"state:{address}"
            drafts[slot_id] = _SlotDraft(
                slot_id=slot_id,
                role=TensorSlotRole.PARAMETER,
                shape=tuple(int(dim) for dim in param.shape),
                dtype=str(param.dtype),
                device_type=device_type,
                device_index=device_index,
                state_binding=StateSlotBinding(
                    module_path=module_path if separator else "self",
                    state_dict_name=address,
                    semantic_role=_parameter_role(param),
                    trainable=bool(param.is_trainable),
                    persistent=True,
                    alias_group=alias_group,
                ),
                use_sites=[],
            )
    return drafts


def _add_persistent_buffer_slot_drafts(
    trace: Any,
    drafts: dict[str, _SlotDraft],
) -> None:
    """Add every persistent source-model buffer to the value-free state map.

    Parameters
    ----------
    trace:
        Live cooked Trace whose weak source-model reference is inspection-only.
    drafts:
        Mutable descriptor slots, including any buffers already represented by
        graph source ops.
    """

    source_ref = getattr(trace, "_source_model_ref", None)
    model = source_ref() if callable(source_ref) else None
    state_dict_method = getattr(model, "state_dict", None)
    named_parameters_method = getattr(model, "named_parameters", None)
    named_buffers_method = getattr(model, "named_buffers", None)
    if (
        not callable(state_dict_method)
        or not callable(named_parameters_method)
        or not callable(named_buffers_method)
    ):
        return
    state = state_dict_method()
    if not isinstance(state, Mapping):
        return
    parameter_names = {
        str(name) for name, _value in named_parameters_method(remove_duplicate=False)
    }
    buffers = {str(name): value for name, value in named_buffers_method(remove_duplicate=False)}
    buffer_names = tuple(
        name
        for name, value in state.items()
        if name not in parameter_names and name in buffers and isinstance(value, torch.Tensor)
    )
    names_by_object: dict[int, list[str]] = defaultdict(list)
    for name in buffer_names:
        names_by_object[id(buffers[name])].append(name)
    alias_by_name = {
        name: (f"buffer_alias:{min(names)}" if len(names) > 1 else None)
        for names in names_by_object.values()
        for name in names
    }
    existing_by_name = {
        draft.state_binding.state_dict_name: draft
        for draft in drafts.values()
        if draft.state_binding is not None
    }
    for name in buffer_names:
        alias_group = alias_by_name[name]
        existing = existing_by_name.get(name)
        if existing is not None:
            binding = existing.state_binding
            assert binding is not None
            existing.state_binding = replace(binding, alias_group=alias_group)
            continue
        value = cast(torch.Tensor, state[name])
        module_path, separator, leaf_name = name.rpartition(".")
        device_type, device_index = _device_parts(value.device)
        slot_id = f"state:{name}"
        drafts[slot_id] = _SlotDraft(
            slot_id=slot_id,
            role=TensorSlotRole.BUFFER,
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
            device_type=device_type,
            device_index=device_index,
            state_binding=StateSlotBinding(
                module_path=module_path if separator else "self",
                state_dict_name=name,
                semantic_role=_buffer_role(leaf_name or name),
                trainable=False,
                persistent=True,
                alias_group=alias_group,
            ),
            use_sites=[],
        )


def _build_child_version_slot_drafts(
    trace: Any,
    ops: Sequence[Any],
    drafts: dict[str, _SlotDraft],
    slot_for_op: Mapping[int, str],
) -> None:
    """Add value-free per-child tensor-version identities from cooked mapping keys."""

    aliases = _op_alias_index(trace, ops)
    for parent in ops:
        versions = getattr(parent, "out_versions_by_child", {}) or {}
        if not isinstance(versions, Mapping):
            continue
        base_slot_id = slot_for_op[id(parent)]
        base = drafts[base_slot_id]
        for child_label in versions:
            child = aliases.get(str(child_label))
            stable_child_label = str(getattr(child, "label", child_label))
            version_slot_id = f"{base_slot_id}:use:{stable_child_label}"
            drafts[version_slot_id] = _SlotDraft(
                slot_id=version_slot_id,
                role=base.role,
                shape=base.shape,
                dtype=base.dtype,
                device_type=base.device_type,
                device_index=base.device_index,
                mutable=True,
                version_of=base_slot_id,
                producer_slot_id=base_slot_id,
                output_path=base.output_path,
                input_binding=base.input_binding,
                state_binding=base.state_binding,
                use_sites=[],
            )


def _child_version_slot_id(
    parent: Any,
    child: Any,
    base_slot_id: str,
    drafts: Mapping[str, _SlotDraft],
) -> str:
    """Return the child-specific version slot when the cooked core records one."""

    exact_slot_id = f"{base_slot_id}:use:{child.label}"
    if exact_slot_id in drafts:
        return exact_slot_id
    versions = getattr(parent, "out_versions_by_child", {}) or {}
    child_aliases = {
        str(getattr(child, "label", "")),
        str(getattr(child, "layer_label", "")),
    }
    matching_label = next((label for label in versions if str(label) in child_aliases), None)
    if matching_label is None:
        return base_slot_id
    fallback_slot_id = f"{base_slot_id}:use:{matching_label}"
    return fallback_slot_id if fallback_slot_id in drafts else base_slot_id


def _build_call_arguments(
    op: Any,
    template: CapturedArgTemplate,
    *,
    call_id: str,
    registry_id: str,
    op_by_alias: Mapping[str, Any],
    slot_for_op: Mapping[int, str],
    slot_drafts: dict[str, _SlotDraft],
    diagnostics: list[RunnableDiagnostic],
) -> tuple[list[TensorArgumentRef], list[LiteralArgumentRef]]:
    """Build tensor and literal call leaves from one cooked argument template."""

    tensor_args: list[TensorArgumentRef] = []
    literal_args: list[LiteralArgumentRef] = []
    parameter_candidates = list(getattr(op, "_param_logs", ()) or ())
    non_tensor_positional = iter(getattr(op, "non_tensor_pos_args", ()) or ())

    for index, component in enumerate(template.args):
        path: tuple[str | int, ...] = ("args", index)
        override = None
        if not _component_contains_tensor(component):
            override = next(non_tensor_positional, _NO_OVERRIDE)
        _append_argument_component(
            component,
            path=path,
            literal_override=override,
            op=op,
            call_id=call_id,
            registry_id=registry_id,
            op_by_alias=op_by_alias,
            slot_for_op=slot_for_op,
            slot_drafts=slot_drafts,
            parameter_candidates=parameter_candidates,
            tensor_args=tensor_args,
            literal_args=literal_args,
            diagnostics=diagnostics,
        )
    non_tensor_kwargs = getattr(op, "non_tensor_kwargs", {}) or {}
    for key, component in template.kwargs:
        path = ("kwargs", str(key))
        override = non_tensor_kwargs.get(key, _NO_OVERRIDE)
        _append_argument_component(
            component,
            path=path,
            literal_override=override,
            op=op,
            call_id=call_id,
            registry_id=registry_id,
            op_by_alias=op_by_alias,
            slot_for_op=slot_for_op,
            slot_drafts=slot_drafts,
            parameter_candidates=parameter_candidates,
            tensor_args=tensor_args,
            literal_args=literal_args,
            diagnostics=diagnostics,
        )
    return tensor_args, literal_args


_NO_OVERRIDE = object()


def _append_argument_component(
    component: Any,
    *,
    path: tuple[str | int, ...],
    literal_override: Any,
    op: Any,
    call_id: str,
    registry_id: str,
    op_by_alias: Mapping[str, Any],
    slot_for_op: Mapping[int, str],
    slot_drafts: dict[str, _SlotDraft],
    parameter_candidates: list[Any],
    tensor_args: list[TensorArgumentRef],
    literal_args: list[LiteralArgumentRef],
    diagnostics: list[RunnableDiagnostic],
) -> None:
    """Append one captured argument component to a sparse call recipe."""

    if not _component_contains_tensor(component):
        value = component.value if isinstance(component, LiteralValue) else component
        if literal_override is not _NO_OVERRIDE:
            value = literal_override
        try:
            literal_args.append(LiteralArgumentRef(path, _encode_literal(value)))
        except _UnsupportedLiteralError as exc:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.UNSUPPORTED_LITERAL,
                    str(exc),
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_literal",
                    details=(("argument_path", repr(path)),),
                )
            )
        return
    if isinstance(component, (list, tuple)):
        try:
            literal_args.append(LiteralArgumentRef(path, _tensor_container_skeleton(component)))
        except _UnsupportedLiteralError as exc:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                    str(exc),
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_call_recipe",
                    details=(("argument_path", repr(path)),),
                )
            )
            return
        for index, item in enumerate(component):
            _append_argument_component(
                item,
                path=(*path, index),
                literal_override=_NO_OVERRIDE,
                op=op,
                call_id=call_id,
                registry_id=registry_id,
                op_by_alias=op_by_alias,
                slot_for_op=slot_for_op,
                slot_drafts=slot_drafts,
                parameter_candidates=parameter_candidates,
                tensor_args=tensor_args,
                literal_args=literal_args,
                diagnostics=diagnostics,
            )
        return
    if isinstance(component, Mapping):
        try:
            literal_args.append(LiteralArgumentRef(path, _tensor_container_skeleton(component)))
        except _UnsupportedLiteralError as exc:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                    str(exc),
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_call_recipe",
                    details=(("argument_path", repr(path)),),
                )
            )
            return
        for key, item in component.items():
            if not isinstance(key, (str, int)):
                diagnostics.append(
                    _diagnostic(
                        RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
                        "A tensor-containing mapping argument has a key outside the sparse "
                        "argument-path grammar.",
                        registry_id=registry_id,
                        affected_ops=(str(op.label),),
                        detection_stage="producer_call_recipe",
                        details=(("argument_path", repr(path)),),
                    )
                )
                continue
            _append_argument_component(
                item,
                path=(*path, key),
                literal_override=_NO_OVERRIDE,
                op=op,
                call_id=call_id,
                registry_id=registry_id,
                op_by_alias=op_by_alias,
                slot_for_op=slot_for_op,
                slot_drafts=slot_drafts,
                parameter_candidates=parameter_candidates,
                tensor_args=tensor_args,
                literal_args=literal_args,
                diagnostics=diagnostics,
            )
        return
    if isinstance(component, ParentRef):
        parent = op_by_alias.get(component.parent_label)
        if parent is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_TENSOR_SLOT,
                    f"No cooked parent slot matches {component.parent_label!r}.",
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_tensor_argument",
                )
            )
            return
        base_slot_id = slot_for_op[id(parent)]
        slot_id = _child_version_slot_id(parent, op, base_slot_id, slot_drafts)
        _append_tensor_argument(
            tensor_args,
            path,
            slot_id,
            call_id=call_id,
            slot_drafts=slot_drafts,
        )
        return
    if isinstance(component, LiteralTensor):
        param = _match_parameter(component.value, path, op, parameter_candidates)
        if param is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.UNSUPPORTED_TENSOR_CONSTANT,
                    "Tensor literal is not a named parameter, registered buffer, model input, "
                    "or reproducible source call.",
                    registry_id=registry_id,
                    affected_ops=(str(op.label),),
                    detection_stage="producer_tensor_constant",
                    details=(("argument_path", repr(path)),),
                )
            )
            return
        parameter_candidates.remove(param)
        slot_id = f"state:{param.address}"
        draft = slot_drafts.get(slot_id)
        if draft is not None:
            draft.device_type, draft.device_index = _device_parts(
                getattr(component.value, "device", None)
            )
        _append_tensor_argument(
            tensor_args,
            path,
            slot_id,
            call_id=call_id,
            slot_drafts=slot_drafts,
        )
        return
    if isinstance(component, Unsupported):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.UNSUPPORTED_LITERAL,
                component.reason,
                registry_id=registry_id,
                affected_ops=(str(op.label),),
                detection_stage="producer_literal",
                details=(("argument_path", repr(path)), ("value_type", component.value_type)),
            )
        )
        return

    diagnostics.append(
        _diagnostic(
            RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
            "A tensor-containing argument container lacks a preserved list/tuple/mapping "
            "contract in the cooked projection.",
            registry_id=registry_id,
            affected_ops=(str(op.label),),
            detection_stage="producer_call_recipe",
            details=(("argument_path", repr(path)),),
        )
    )


def _append_tensor_argument(
    tensor_args: list[TensorArgumentRef],
    path: tuple[str | int, ...],
    slot_id: str,
    *,
    call_id: str,
    slot_drafts: dict[str, _SlotDraft],
) -> None:
    """Append a tensor reference and its reverse use-site metadata."""

    tensor_args.append(TensorArgumentRef(argument_path=path, slot_id=slot_id))
    draft = slot_drafts.get(slot_id)
    if draft is not None:
        if draft.use_sites is None:
            draft.use_sites = []
        draft.use_sites.append(TensorUseSite(call_id=call_id, argument_path=path))


def _match_parameter(
    tensor: Any,
    path: tuple[str | int, ...],
    op: Any,
    candidates: Sequence[Any],
) -> Any | None:
    """Match a template tensor literal to one cooked named parameter."""

    shape = _shape_tuple(getattr(tensor, "shape", None))
    dtype = str(getattr(tensor, "dtype", ""))
    matches = [
        param for param in candidates if tuple(param.shape) == shape and str(param.dtype) == dtype
    ]
    argument_names = tuple(getattr(op, "arg_names", ()) or ())
    top_index = path[1] if len(path) > 1 and isinstance(path[1], int) else None
    argument_name = (
        str(argument_names[top_index])
        if top_index is not None and top_index < len(argument_names)
        else None
    )
    named_matches = [param for param in matches if str(param.name) == argument_name]
    if len(named_matches) == 1:
        return named_matches[0]
    return matches[0] if len(matches) == 1 else None


def _group_computational_ops(ops: Sequence[Any]) -> list[tuple[int, list[Any]]]:
    """Group cooked computational output tensors by capture call ID."""

    groups: dict[int, list[Any]] = defaultdict(list)
    first_index: dict[int, int] = {}
    for index, op in enumerate(ops):
        call_number = getattr(op, "func_call_id", None)
        if call_number is None or bool(getattr(op, "is_input", False)):
            continue
        if bool(getattr(op, "is_buffer", False)) or bool(getattr(op, "is_output", False)):
            continue
        call_number = int(call_number)
        groups[call_number].append(op)
        first_index.setdefault(call_number, index)
    return sorted(groups.items(), key=lambda item: first_index[item[0]])


def _op_alias_index(trace: Any, ops: Sequence[Any]) -> dict[str, Any]:
    """Index cooked ops under raw, layer, pass-qualified, and short labels."""

    aliases: dict[str, Any] = {}
    raw_to_final = getattr(trace, "_raw_to_final_op_labels", {}) or {}
    for op in ops:
        for value in (
            getattr(op, "label", None),
            getattr(op, "layer_label", None),
            getattr(op, "label_short", None),
            getattr(op, "layer_label_short", None),
        ):
            if isinstance(value, str):
                aliases[value] = op
        label = str(getattr(op, "label", ""))
        if ":" in label:
            aliases.setdefault(label.rsplit(":", 1)[0], op)
    for raw_label, final_label in raw_to_final.items():
        final_op = aliases.get(str(final_label))
        if final_op is not None:
            aliases[str(raw_label)] = final_op
    return aliases


def _resolve_op(trace: Any, label: str) -> Any | None:
    """Resolve one cooked op label through the Trace lookup tables."""

    layer_dict = getattr(trace, "layer_dict_all_keys", {}) or {}
    if label in layer_dict:
        return layer_dict[label]
    try:
        layer = trace[label]
    except (KeyError, TypeError, ValueError):
        return None
    ops = getattr(layer, "ops", None)
    if ops:
        return ops[0]
    return layer


def _input_binding_for_op(
    trace: Any,
    op: Any,
    diagnostics: list[RunnableDiagnostic],
) -> InputSlotBinding | None:
    """Resolve an input source op to a captured model-boundary container site."""

    containers = getattr(trace, "__dict__", {}).get("_containers", {}) or {}
    raw_candidates = {
        str(getattr(op, "label", "")),
        str(getattr(op, "layer_label", "")),
    }
    final_to_raw = getattr(trace, "_final_to_raw_layer_labels", {}) or {}
    raw_label = final_to_raw.get(getattr(op, "layer_label", None))
    if isinstance(raw_label, str):
        raw_candidates.add(raw_label)
    for record in containers.values():
        for snapshot in getattr(record, "snapshots", ()):
            if getattr(snapshot, "role", None) is not Role.MODEL_INPUT:
                continue
            site = getattr(snapshot, "site", None)
            if not isinstance(site, ModelSite):
                continue
            for occurrence in getattr(snapshot, "leaf_occurrences", ()):
                producer = getattr(occurrence, "producer_op_label", None)
                if not isinstance(producer, str):
                    continue
                cooked = (getattr(trace, "_raw_to_final_op_labels", {}) or {}).get(producer)
                if producer not in raw_candidates and cooked not in raw_candidates:
                    continue
                position = _normalize_model_site_position(site.position)
                if position is None:
                    break
                return InputSlotBinding(
                    io_role="model_input",
                    model_ref=site.model_ref,
                    model_site_position=position,
                    container_record_id=int(record.ordinal),
                    container_path=_normalize_container_path(occurrence.path),
                )

    io_role = str(getattr(op, "io_role", ""))
    if io_role.count(".") > 1:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.MISSING_INPUT_CONTAINER_CONTRACT,
                "Nested model input has no captured ContainerRecord/ModelSite contract; "
                "capture with capture_container_structure=True.",
                affected_ops=(str(op.label),),
                detection_stage="producer_input_binding",
            )
        )
        return None
    input_ops = [item for item in getattr(trace, "layer_list", ()) if item.is_input]
    input_index = next((index for index, item in enumerate(input_ops) if item is op), 0)
    return InputSlotBinding(
        io_role="model_input",
        model_ref="self:1",
        model_site_position=("arg", input_index),
        container_record_id=-1,
        container_path=(),
    )


def _build_control_witnesses(
    trace: Any,
    ops: Sequence[Any],
    diagnostics: list[RunnableDiagnostic],
) -> tuple[list[ControlWitness], WitnessCompleteness]:
    """Build ordered scalar, loop, arm-entry, and structure witnesses."""

    witnesses: list[ControlWitness] = []
    completeness = WitnessCompleteness.COMPLETE
    for op in ops:
        if not bool(getattr(op, "is_scalar_bool", False)):
            continue
        bool_value = getattr(op, "bool_value", None)
        context_kind = getattr(op, "conditional_context_kind", None)
        if bool_value is None:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_CONTROL_CLASSIFICATION,
                    "Scalar-bool op has no recorded bool_value.",
                    affected_ops=(str(op.label),),
                    detection_stage="producer_control_witness",
                )
            )
            completeness = WitnessCompleteness.INCOMPLETE_UNOBSERVED_PREDICATE
            continue
        if context_kind in {None, "unknown"} and bool(getattr(op, "is_terminal_bool", False)):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.MISSING_CONTROL_CLASSIFICATION,
                    "Terminal scalar-bool escaped without a classified consumer.",
                    affected_ops=(str(op.label),),
                    detection_stage="producer_control_witness",
                )
            )
            completeness = WitnessCompleteness.INCOMPLETE_SCALAR_ESCAPE
        kind = (
            ControlWitnessKind.LOOP_PREDICATE
            if context_kind == "while"
            else ControlWitnessKind.SCALAR_BOOL
        )
        call_number = getattr(op, "func_call_id", None)
        witnesses.append(
            ControlWitness(
                witness_id=f"witness:{len(witnesses) + 1}",
                kind=kind,
                order=len(witnesses),
                call_id=None if call_number is None else f"call:{call_number}",
                site_label=str(op.label),
                observed_value=_encode_literal(bool(bool_value)),
            )
        )

    arm_edges = getattr(trace, "conditional_arm_entry_edges", {}) or {}
    for (conditional_id, arm_kind), edges in sorted(
        arm_edges.items(), key=lambda item: (int(item[0][0]), str(item[0][1]))
    ):
        for parent, child in edges:
            witnesses.append(
                ControlWitness(
                    witness_id=f"witness:{len(witnesses) + 1}",
                    kind=ControlWitnessKind.CONDITIONAL_ARM_ENTRY,
                    order=len(witnesses),
                    call_id=None,
                    site_label=f"conditional:{conditional_id}:{arm_kind}:{parent}->{child}",
                    observed_value=_encode_literal(True),
                )
            )

    witnesses.extend(_container_structure_witnesses(trace, start_order=len(witnesses)))
    return witnesses, completeness


def _container_structure_witnesses(trace: Any, *, start_order: int) -> list[ControlWitness]:
    """Encode captured model-boundary container facts as non-tensor witnesses."""

    witnesses: list[ControlWitness] = []
    containers = getattr(trace, "__dict__", {}).get("_containers", {}) or {}
    for record_id, record in sorted(containers.items()):
        for snapshot_index, snapshot in enumerate(getattr(record, "snapshots", ())):
            if getattr(snapshot, "role", None) not in {Role.MODEL_INPUT, Role.MODEL_OUTPUT}:
                continue
            spec = getattr(snapshot, "spec", None)
            fact = {
                "record_id": int(record_id),
                "snapshot": snapshot_index,
                "role": snapshot.role.value,
                "kind": getattr(spec, "kind", "unknown"),
                "reconstructable": bool(getattr(snapshot, "reconstructable", False)),
                "leaf_paths": [
                    list(normalized)
                    for occurrence in getattr(snapshot, "leaf_occurrences", ())
                    if (normalized := _safe_normalize_container_path(occurrence.path)) is not None
                ],
            }
            try:
                observed = _encode_literal(fact)
            except _UnsupportedLiteralError:
                continue
            order = start_order + len(witnesses)
            witnesses.append(
                ControlWitness(
                    witness_id=f"witness:{order + 1}",
                    kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
                    order=order,
                    call_id=None,
                    site_label=f"container:{record_id}:{snapshot_index}",
                    observed_value=observed,
                )
            )
    return witnesses


MODEL_INPUT_LITERAL_SITE_PREFIX = "model_input_literal:"
"""``site_label`` prefix marking a witnessed non-tensor model-input leaf."""

MODEL_INPUT_LITERAL_FACT_KEY = "model_input_literal"
"""Discriminator key present in every non-tensor model-input leaf fact."""


def _input_literal_witnesses(
    trace: Any,
    *,
    start_order: int,
) -> tuple[list[ControlWitness], bool]:
    """Witness capture-time non-tensor model-input leaves as structure facts.

    The runnable executor binds only tensor input leaves; a changed non-tensor
    Python input (bool/int/float/str/None) can silently steer control flow that
    was never captured as an op, making the recorded taken path wrong. Each
    grammar-encodable non-tensor leaf is recorded as a ``SHAPE_STRUCTURE_FACT``
    witness carrying its site position, container path, and literal value so the
    executor can diverge on a changed non-tensor input rather than falsely
    reporting a verified, attested result.

    A leaf *outside* the frozen literal grammar (enum, dataclass, set, bytes,
    complex, numpy scalar, non-finite ``inf``/``nan`` float, ...) cannot be
    compared across save/load, so its value is recorded ``None`` -- a value-free
    fact. Such a leaf can still steer unobserved Python control flow, and because
    the executor cannot re-verify it, the run's witness coverage is genuinely
    INCOMPLETE: the caller must downgrade ``witness_completeness`` so the run
    reports ``UNVERIFIABLE`` + ``NOT_APPLICABLE`` instead of a false
    ``VERIFIED``/``ATTESTED`` over a possibly-wrong replayed path. This function
    signals that condition by returning ``saw_opaque_leaf=True`` so the caller can
    downgrade ``witness_completeness``. It does *not* fail producer preflight: an
    opaque-leaf artifact still saves and runs, but honestly reports
    ``UNVERIFIABLE`` rather than a false ``VERIFIED``. No tensors are recorded.

    Parameters
    ----------
    trace:
        Cooked Trace carrying the capture-time non-tensor input leaf stash.
    start_order:
        First dense witness order to assign, continuing the existing sequence.

    Returns
    -------
    tuple[list[ControlWitness], bool]
        Ordered non-tensor model-input leaf witnesses, and whether any leaf was
        value-free (opaque), signalling incomplete witness coverage.
    """

    leaves = getattr(trace, "__dict__", {}).get("_runnable_input_nontensor_leaves", ())
    witnesses: list[ControlWitness] = []
    saw_opaque_leaf = False
    for position, path, value in leaves:
        try:
            _encode_literal(value)
            encodable = True
        except _UnsupportedLiteralError:
            encodable = False
        if not encodable:
            saw_opaque_leaf = True
        fact = {
            MODEL_INPUT_LITERAL_FACT_KEY: True,
            "position": list(position) if isinstance(position, tuple) else position,
            "path": list(path),
            "encodable": encodable,
            "value": value if encodable else None,
        }
        try:
            observed = _encode_literal(fact)
        except _UnsupportedLiteralError:
            # Defensive: an exotic path/position component that cannot be encoded
            # cannot be witnessed. Tensor-slot binding still constrains arity.
            continue
        order = start_order + len(witnesses)
        witnesses.append(
            ControlWitness(
                witness_id=f"witness:{order + 1}",
                kind=ControlWitnessKind.SHAPE_STRUCTURE_FACT,
                order=order,
                call_id=None,
                site_label=f"{MODEL_INPUT_LITERAL_SITE_PREFIX}{position!r}:{list(path)!r}",
                observed_value=observed,
            )
        )
    return witnesses, saw_opaque_leaf


def _preflight_output_contracts(trace: Any, ops: Sequence[Any]) -> list[RunnableDiagnostic]:
    """Report structured model outputs whose container contract is unavailable."""

    diagnostics: list[RunnableDiagnostic] = []
    output_ops = [op for op in ops if bool(getattr(op, "is_output", False))]
    containers = getattr(trace, "__dict__", {}).get("_containers", {}) or {}
    model_output_snapshots = [
        snapshot
        for record in containers.values()
        for snapshot in getattr(record, "snapshots", ())
        if getattr(snapshot, "role", None) is Role.MODEL_OUTPUT
    ]
    # A recorded model-output container that is non-reconstructable (opaque custom
    # Mapping, unsafe defaultdict, unknown dict subclass, or an unrepresentable
    # non-tensor leaf) must NOT advertise runnable regardless of how many tensors it
    # holds: otherwise the loaded run silently returns a bare tensor / plain dict and
    # reports VERIFIED. Honest-reject at save closes that class.
    if any(
        not getattr(snapshot, "reconstructable", True)
        or getattr(getattr(snapshot, "spec", None), "kind", None) == "opaque"
        for snapshot in model_output_snapshots
    ):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT,
                "Model output is a non-reconstructable container; runnable replay cannot "
                "restore its exact type and non-tensor leaves.",
                affected_ops=tuple(str(op.label) for op in output_ops),
                detection_stage="producer_output_binding",
            )
        )
        return diagnostics
    if len(output_ops) <= 1:
        return diagnostics
    if any(getattr(op, "container_path", None) for op in output_ops):
        return diagnostics
    has_model_output = bool(model_output_snapshots)
    if not has_model_output:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT,
                "Multi-tensor model output has no cooked container contract.",
                affected_ops=tuple(str(op.label) for op in output_ops),
                detection_stage="producer_output_binding",
            )
        )
    return diagnostics


def _mark_inplace_versions(
    calls: Sequence[RunnableCallDescriptor], slot_drafts: Mapping[str, _SlotDraft]
) -> None:
    """Attach version relations for in-place call outputs."""

    for call in calls:
        if not call.is_inplace or not call.tensor_arguments:
            continue
        version_of = call.tensor_arguments[0].slot_id
        for output_slot_id in call.output_slot_ids:
            draft = slot_drafts.get(output_slot_id)
            if draft is not None:
                draft.mutable = True
                draft.version_of = version_of


def _buffer_binding(op: Any) -> StateSlotBinding | None:
    """Build a named buffer binding from a cooked source op."""

    address = getattr(op, "address", None)
    if not isinstance(address, str) or not address:
        return None
    module_path, _, name = address.rpartition(".")
    return StateSlotBinding(
        module_path=module_path or "self",
        state_dict_name=address,
        semantic_role=_buffer_role(name or address),
        trainable=False,
        persistent=True,
        alias_group=None,
    )


def _parameter_role(param: Any) -> StateSlotRole:
    """Classify one cooked parameter into the frozen initializer role table."""

    name = str(getattr(param, "name", ""))
    module_address = str(getattr(param, "module_address", ""))
    trace = getattr(param, "source_trace", None)
    modules = getattr(trace, "modules", {}) if trace is not None else {}
    module = modules.get(module_address) if isinstance(modules, Mapping) else None
    class_name = str(getattr(module, "class_name", module_address)).lower()
    is_norm = "norm" in class_name
    if name == "weight":
        return StateSlotRole.NORM_SCALE if is_norm else StateSlotRole.WEIGHT
    if name == "bias":
        return StateSlotRole.NORM_OFFSET if is_norm else StateSlotRole.BIAS
    return StateSlotRole.WEIGHT


def _buffer_role(name: str) -> StateSlotRole:
    """Classify a named buffer into the frozen state role vocabulary."""

    if name == "running_mean":
        return StateSlotRole.RUNNING_MEAN
    if name in {"running_var", "running_variance"}:
        return StateSlotRole.RUNNING_VAR
    if name in {"num_batches_tracked", "counter", "step"}:
        return StateSlotRole.COUNTER
    return StateSlotRole.GENERIC_BUFFER


def _op_slot_role(op: Any) -> TensorSlotRole:
    """Classify one cooked op tensor into the frozen slot role vocabulary."""

    if bool(getattr(op, "is_input", False)):
        return TensorSlotRole.MODEL_INPUT
    if bool(getattr(op, "is_buffer", False)):
        return TensorSlotRole.BUFFER
    if bool(getattr(op, "is_output", False)):
        return TensorSlotRole.OUTPUT
    if str(getattr(op, "func_name", "")) in {
        "rand",
        "randn",
        "randint",
        "rand_like",
        "randn_like",
        "bernoulli",
    }:
        return TensorSlotRole.RNG_SOURCE
    return TensorSlotRole.INTERMEDIATE


def _component_contains_tensor(component: Any) -> bool:
    """Return whether a captured component contains a tensor-valued leaf."""

    if isinstance(component, (ParentRef, LiteralTensor)):
        return True
    if isinstance(component, tuple):
        return any(_component_contains_tensor(item) for item in component)
    if isinstance(component, list):
        return any(_component_contains_tensor(item) for item in component)
    if isinstance(component, Mapping):
        return any(_component_contains_tensor(item) for item in component.values())
    return False


def _tensor_container_skeleton(component: Any) -> NonTensorLiteral:
    """Encode a mutable sparse-call container shell without tensor values.

    Tensor leaves are represented by ``None`` placeholders and overwritten
    with ``ParentRef`` slots during sparse execution. Captured tensor
    containers are projected as tuples by the eager capture layer, so their
    runnable shell intentionally uses a list, which is accepted by the
    relevant variadic torch operators and supports leaf replacement.

    Parameters
    ----------
    component:
        Captured argument component containing at least one tensor leaf.

    Returns
    -------
    NonTensorLiteral
        Value-free literal shell used to reconstruct the argument tree.
    """

    if isinstance(component, (ParentRef, LiteralTensor)):
        return LiteralAtom(LiteralAtomKind.NONE, None)
    if isinstance(component, LiteralValue):
        return _encode_literal(component.value)
    if isinstance(component, Unsupported):
        raise _UnsupportedLiteralError(component.reason)
    if isinstance(component, (list, tuple)):
        return LiteralSequence(
            LiteralSequenceKind.LIST,
            tuple(_tensor_container_skeleton(item) for item in component),
        )
    if isinstance(component, Mapping):
        return LiteralMapping(
            tuple(
                LiteralMappingEntry(
                    _encode_literal_key(key),
                    _tensor_container_skeleton(item),
                )
                for key, item in component.items()
            )
        )
    return _encode_literal(component)


def _encode_literal(value: Any) -> NonTensorLiteral:
    """Encode a Python value using only the frozen safe literal grammar."""

    if value is None:
        return LiteralAtom(LiteralAtomKind.NONE, None)
    if isinstance(value, bool):
        return LiteralAtom(LiteralAtomKind.BOOL, value)
    if isinstance(value, int):
        # Normalize integer subclasses (e.g. ``IntEnum``) to a plain ``int`` so
        # the stored literal round-trips through JSON / the safe unpickler.
        return LiteralAtom(LiteralAtomKind.INT, int(value))
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)).item():
            raise _UnsupportedLiteralError("Non-finite floating-point literals are unsupported.")
        # Normalize float subclasses (e.g. ``numpy.float64``) to a plain
        # ``float`` so the recorded literal round-trips to a grammar-native value
        # the safe metadata unpickler admits and value-equality can verify.
        return LiteralAtom(LiteralAtomKind.FLOAT, float(value))
    if isinstance(value, str):
        return LiteralAtom(LiteralAtomKind.STR, value)
    torch_symbol = _torch_symbol_qualname(value)
    if torch_symbol is not None:
        return LiteralTorchSymbol(torch_symbol)
    if isinstance(value, list):
        return LiteralSequence(
            LiteralSequenceKind.LIST,
            tuple(_encode_literal(item) for item in value),
        )
    if isinstance(value, tuple):
        return LiteralSequence(
            LiteralSequenceKind.TUPLE,
            tuple(_encode_literal(item) for item in value),
        )
    if isinstance(value, Mapping):
        return LiteralMapping(
            tuple(
                LiteralMappingEntry(_encode_literal_key(key), _encode_literal(item))
                for key, item in value.items()
            )
        )
    value_type = f"{type(value).__module__}.{type(value).__qualname__}"
    raise _UnsupportedLiteralError(
        f"Value of type {value_type} is outside the frozen non-tensor literal grammar."
    )


def _encode_literal_key(value: Any) -> LiteralAtom | LiteralTupleKey:
    """Encode a mapping key using the frozen safe key subset."""

    encoded = _encode_literal(value)
    if isinstance(encoded, LiteralAtom):
        return encoded
    if isinstance(encoded, LiteralSequence) and encoded.kind is LiteralSequenceKind.TUPLE:
        items: list[LiteralAtom | LiteralTupleKey] = []
        for item in encoded.items:
            if isinstance(item, LiteralAtom):
                items.append(item)
            elif isinstance(item, LiteralSequence) and item.kind is LiteralSequenceKind.TUPLE:
                items.append(_encode_literal_key(_literal_sequence_to_python(item)))
            else:
                raise _UnsupportedLiteralError("Mapping tuple keys may contain only scalar atoms.")
        return LiteralTupleKey(tuple(items))
    raise _UnsupportedLiteralError("Mapping keys must be scalar atoms or safe tuples.")


def _literal_sequence_to_python(value: LiteralSequence) -> tuple[Any, ...]:
    """Convert an encoded scalar tuple key back to a Python tuple for recursion."""

    result: list[Any] = []
    for item in value.items:
        if isinstance(item, LiteralAtom):
            result.append(item.value)
        elif isinstance(item, LiteralSequence):
            result.append(_literal_sequence_to_python(item))
        else:
            raise _UnsupportedLiteralError("Mapping tuple keys may contain only scalar atoms.")
    return tuple(result)


def _torch_symbol_qualname(value: Any) -> str | None:
    """Return an allowlisted torch symbolic name for a non-callable value."""

    if isinstance(value, torch.device):
        return f"torch.device({value})"
    for name, candidate in vars(torch).items():
        if callable(candidate):
            continue
        if candidate is value and isinstance(
            value, (torch.dtype, torch.layout, torch.memory_format)
        ):
            return f"torch.{name}"
    return None


def _runtime_fingerprint(op: Any, func_id: FunctionRegistryKey, call_ops: Sequence[Any]) -> str:
    """Hash the recorded runtime call signature without tensor values."""

    payload = {
        "callable": {
            "namespace": func_id.namespace,
            "qualname": func_id.qualname,
            "dispatch_kind": func_id.dispatch_kind,
            "version": func_id.version,
            "import_path": func_id.import_path,
        },
        "argument_names": list(getattr(op, "arg_names", ()) or ()),
        "num_positional_args": int(getattr(op, "num_pos_args", 0)),
        "num_keyword_args": int(getattr(op, "num_kwargs", 0)),
        "outputs": [
            {"shape": list(_shape_tuple(item.shape) or ()), "dtype": _dtype_name(item)}
            for item in call_ops
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest()


def _diagnostic(
    code: RunnableErrorCode,
    message: str,
    *,
    registry_id: str | None = None,
    affected_ops: tuple[str, ...] = (),
    detection_stage: str,
    details: tuple[tuple[str, str], ...] = (),
) -> RunnableDiagnostic:
    """Construct one structured producer diagnostic."""

    return RunnableDiagnostic(
        code=code,
        message=message,
        registry_id=registry_id,
        affected_op_labels=affected_ops,
        recorded_runtime=str(torch.__version__),
        current_runtime=str(torch.__version__),
        detection_stage=detection_stage,
        resolver_provenance=None,
        analysis_load_available=True,
        details=details,
    )


def _deduplicate_diagnostics(
    diagnostics: Iterable[RunnableDiagnostic],
) -> list[RunnableDiagnostic]:
    """Deduplicate diagnostics while retaining deterministic first occurrence."""

    result: list[RunnableDiagnostic] = []
    seen: set[tuple[Any, ...]] = set()
    for diagnostic in diagnostics:
        key = (
            diagnostic.code,
            diagnostic.registry_id,
            diagnostic.affected_op_labels,
            diagnostic.detection_stage,
            diagnostic.details,
        )
        if key not in seen:
            seen.add(key)
            result.append(diagnostic)
    return result


def _shape_tuple(value: Any) -> tuple[int, ...] | None:
    """Normalize shape metadata to a tuple of integers."""

    if value is None:
        return None
    try:
        return tuple(int(dim) for dim in value)
    except (TypeError, ValueError):
        return None


def _dtype_name(op: Any) -> str | None:
    """Return canonical dtype metadata for one cooked op."""

    dtype_ref = getattr(op, "dtype_ref", None)
    if dtype_ref is not None:
        return str(dtype_ref)
    dtype = getattr(op, "dtype", None)
    return None if dtype is None else str(dtype)


def _device_parts(value: Any) -> tuple[str, int | None]:
    """Split cooked device metadata into type and optional index."""

    text = "cpu" if value is None else str(value)
    device = torch.device(text)
    return device.type, device.index


def _safe_normalize_container_path(path: Iterable[Any]) -> tuple[str | int, ...] | None:
    """Return the frozen container path, or ``None`` when a key is unrepresentable.

    Used where an unrepresentable output-container path (non-str/int key) must
    degrade gracefully -- the enclosing container is recorded opaque and rejected
    at preflight, so its leaf paths are advisory and must never crash descriptor
    build with a raw ``ValueError``.
    """

    try:
        return _normalize_container_path(path)
    except ValueError:
        return None


def _normalize_container_path(path: Iterable[Any]) -> tuple[str | int, ...]:
    """Convert cooked container path components to frozen string/int paths."""

    normalized: list[str | int] = []
    for component in path:
        if isinstance(component, TupleIndex):
            normalized.append(component.index)
        elif isinstance(component, (DictKey, HFKey)):
            if not isinstance(component.key, (str, int)):
                raise ValueError("Runnable container paths require string or integer keys.")
            normalized.append(component.key)
        elif isinstance(component, (NamedField, DataclassField)):
            normalized.append(component.name)
        elif isinstance(component, (str, int)):
            normalized.append(component)
        else:
            raise ValueError(f"Unsupported runnable container path component {component!r}.")
    return tuple(normalized)


def _normalize_model_site_position(value: Any) -> str | int | tuple[str | int, ...] | None:
    """Normalize a captured ModelSite position to the frozen path vocabulary."""

    if isinstance(value, (str, int)):
        return value
    if isinstance(value, tuple) and all(isinstance(item, (str, int)) for item in value):
        return value
    return None


def _json_value(value: Any) -> Any:
    """Recursively convert dataclasses, enums, tuples, and mappings to JSON values."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Sparse descriptor contains non-JSON value {type(value).__qualname__}.")


from .runnable_load import (  # noqa: E402 - keep producer helpers grouped above
    attach_sparse_run_readiness,
    parse_sparse_run_descriptor,
    preflight_sparse_run_descriptor,
)


__all__ = [
    "assert_sparse_core_has_no_tensor_payload",
    "attach_sparse_run_readiness",
    "build_sparse_run_descriptor",
    "parse_sparse_run_descriptor",
    "preflight_sparse_run_descriptor",
    "require_sparse_run_descriptor",
    "sparse_descriptor_to_json",
    "with_activation_payload",
    "with_weight_payload",
]
