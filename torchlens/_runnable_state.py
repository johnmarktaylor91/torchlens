"""Non-executing state binding and allocation for sparse runnable traces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from typing import Any

import torch

from .errors import StateBindingError
from .runnable import (
    CANONICAL_INITIALIZER_BY_ROLE,
    RUNNABLE_INITIALIZER_POLICY_VERSION,
    InitializerPolicy,
    RunnableDiagnostic,
    RunnableErrorCode,
    SparseRunDescriptor,
    StateSlotRole,
    StateSource,
    TensorSlotDescriptor,
    TensorSlotRole,
)


@dataclass(frozen=True, slots=True)
class PreparedRunnableState:
    """Run-preflight state values prepared without executing graph operations."""

    slot_values: Mapping[str, torch.Tensor]
    state_source: StateSource
    initializer_policy_version: str | None
    seed: int | None
    random_filled_slot_ids: tuple[str, ...]


def load_trace_state_dict(trace: Any, sd: Mapping[str, Any]) -> None:
    """Validate and atomically stage a user state mapping on a sparse Trace.

    Parameters
    ----------
    trace:
        Trace receiving transient run state.
    sd:
        Canonically named parameter and persistent-buffer tensors.

    Raises
    ------
    StateBindingError
        If the Trace is not sparse-runnable or any strict slot contract fails.
    """

    staged = _validate_state_mapping(trace, sd)
    readiness = trace.__dict__.get("_runnable_readiness")
    updated_readiness = readiness
    if readiness is not None and hasattr(readiness, "state_sources_available"):
        sources = tuple(
            source
            for source in readiness.state_sources_available
            if source is not StateSource.USER_STATE_DICT
        )
        updated_readiness = replace(
            readiness,
            state_sources_available=(StateSource.USER_STATE_DICT, *sources),
        )
    trace.__dict__["_runnable_staged_user_state"] = staged
    if updated_readiness is not readiness:
        trace.__dict__["_runnable_readiness"] = updated_readiness


def prepare_runnable_state(trace: Any, seed: int | None = None) -> PreparedRunnableState:
    """Resolve and allocate all parameter/buffer slots without executing the DAG.

    Parameters
    ----------
    trace:
        Loaded sparse Trace whose descriptor supplies state-slot contracts.
    seed:
        Optional isolated initializer seed. ``None`` uses normal runtime RNG.

    Returns
    -------
    PreparedRunnableState
        Run-local slot values and honest source/initializer reporting.

    Raises
    ------
    StateBindingError
        If the descriptor or selected state source violates a slot contract.
    """

    descriptor = _require_descriptor(trace)
    user_state = trace.__dict__.get("_runnable_staged_user_state")
    if isinstance(user_state, Mapping):
        return _prepared_bound_state(user_state, StateSource.USER_STATE_DICT, seed)

    embedded_state = trace.__dict__.get("_runnable_embedded_state")
    if embedded_state is not None:
        if not isinstance(embedded_state, Mapping):
            raise _binding_error(
                (
                    _diagnostic(
                        RunnableErrorCode.STATE_ROLE_MISMATCH,
                        "Embedded state hook does not contain a state mapping.",
                        detection_stage="state_embedded_hook",
                    ),
                )
            )
        validated = _validate_state_mapping(trace, embedded_state)
        return _prepared_bound_state(validated, StateSource.EMBEDDED_CAPTURE_STATE, seed)
    if descriptor.payload_layers.weights.present:
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.STATE_MISSING_KEY,
                    "Descriptor declares embedded capture state, but its Stage 7 hook is empty.",
                    detection_stage="state_embedded_hook",
                ),
            )
        )

    slot_values, random_slot_ids = _initialize_state_slots(descriptor, seed)
    return PreparedRunnableState(
        slot_values=MappingProxyType(slot_values),
        state_source=StateSource.RANDOM_INITIALIZATION,
        initializer_policy_version=RUNNABLE_INITIALIZER_POLICY_VERSION,
        seed=seed,
        random_filled_slot_ids=random_slot_ids,
    )


def _prepared_bound_state(
    state: Mapping[str, torch.Tensor],
    source: StateSource,
    seed: int | None,
) -> PreparedRunnableState:
    """Build a preparation record for a previously validated state source."""

    return PreparedRunnableState(
        slot_values=MappingProxyType(dict(state)),
        state_source=source,
        initializer_policy_version=None,
        seed=seed,
        random_filled_slot_ids=(),
    )


def _validate_state_mapping(trace: Any, sd: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Validate one strict mapping and return detached slot-keyed values."""

    descriptor = _require_descriptor(trace)
    if not isinstance(sd, Mapping):
        raise TypeError("sd must be a mapping of canonical state_dict names to tensors.")

    state_slots = _state_slots(descriptor)
    slots_by_name: dict[str, list[TensorSlotDescriptor]] = defaultdict(list)
    for slot in state_slots:
        assert slot.state_binding is not None
        slots_by_name[slot.state_binding.state_dict_name].append(slot)

    diagnostics: list[RunnableDiagnostic] = []
    supplied_names = {name for name in sd if isinstance(name, str)}
    expected_names = set(slots_by_name)
    for name in sorted(expected_names - supplied_names):
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_MISSING_KEY,
                f"State mapping is missing canonical key {name!r}.",
                detection_stage="state_name_binding",
                details=(("state_dict_name", name),),
            )
        )
    for key in sd:
        if not isinstance(key, str) or key not in expected_names:
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.STATE_UNEXPECTED_KEY,
                    f"State mapping contains unexpected key {key!r}.",
                    detection_stage="state_name_binding",
                    details=(("state_dict_name", repr(key)),),
                )
            )

    values_by_name: dict[str, torch.Tensor] = {}
    for name in sorted(expected_names & supplied_names):
        value = sd[name]
        if not isinstance(value, torch.Tensor):
            diagnostics.append(
                _diagnostic(
                    RunnableErrorCode.STATE_DTYPE_MISMATCH,
                    f"State value for {name!r} is not a tensor.",
                    detection_stage="state_tensor_contract",
                    details=(("state_dict_name", name), ("actual_type", type(value).__name__)),
                )
            )
            continue
        values_by_name[name] = value
        for slot in slots_by_name[name]:
            diagnostics.extend(_slot_contract_diagnostics(slot, value))

    diagnostics.extend(_alias_value_diagnostics(state_slots, values_by_name))
    if diagnostics:
        raise _binding_error(tuple(diagnostics))

    staged: dict[str, torch.Tensor] = {}
    shared_by_alias: dict[str, torch.Tensor] = {}
    shared_by_name: dict[str, torch.Tensor] = {}
    for slot in sorted(state_slots, key=lambda item: item.slot_id):
        binding = slot.state_binding
        assert binding is not None
        group_key = binding.alias_group
        if group_key is not None and group_key in shared_by_alias:
            staged[slot.slot_id] = shared_by_alias[group_key]
            continue
        if binding.state_dict_name in shared_by_name:
            value = shared_by_name[binding.state_dict_name]
        else:
            value = values_by_name[binding.state_dict_name].detach().clone()
            shared_by_name[binding.state_dict_name] = value
        if group_key is not None:
            shared_by_alias[group_key] = value
        staged[slot.slot_id] = value
    return MappingProxyType(staged)


def _slot_contract_diagnostics(
    slot: TensorSlotDescriptor,
    value: torch.Tensor,
) -> list[RunnableDiagnostic]:
    """Return strict name-derived and tensor-contract diagnostics for one slot."""

    binding = slot.state_binding
    assert binding is not None
    diagnostics: list[RunnableDiagnostic] = []
    name = binding.state_dict_name
    inferred_module, inferred_role = _name_contract(name, slot.role)
    if binding.module_path != inferred_module:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_MODULE_PATH_MISMATCH,
                f"Recorded module path for {name!r} disagrees with its canonical name.",
                detection_stage="state_module_path_validation",
                details=(
                    ("state_dict_name", name),
                    ("recorded_module_path", binding.module_path),
                    ("canonical_module_path", inferred_module),
                ),
            )
        )
    if binding.semantic_role not in inferred_role:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_ROLE_MISMATCH,
                f"Recorded semantic role for {name!r} disagrees with its canonical name.",
                detection_stage="state_role_validation",
                details=(
                    ("state_dict_name", name),
                    ("recorded_role", binding.semantic_role.value),
                    ("allowed_roles", ",".join(sorted(role.value for role in inferred_role))),
                ),
            )
        )
    if tuple(value.shape) != slot.shape:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_SHAPE_MISMATCH,
                f"State tensor {name!r} has shape {tuple(value.shape)}, expected {slot.shape}.",
                detection_stage="state_tensor_contract",
                details=(
                    ("state_dict_name", name),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(tuple(value.shape))),
                ),
            )
        )
    if str(value.dtype) != slot.dtype:
        diagnostics.append(
            _diagnostic(
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"State tensor {name!r} has dtype {value.dtype}, expected {slot.dtype}.",
                detection_stage="state_tensor_contract",
                details=(
                    ("state_dict_name", name),
                    ("expected_dtype", slot.dtype),
                    ("actual_dtype", str(value.dtype)),
                ),
            )
        )
    return diagnostics


def _name_contract(
    state_dict_name: str,
    slot_role: TensorSlotRole,
) -> tuple[str, frozenset[StateSlotRole]]:
    """Infer the module-path and allowed semantic roles from a canonical state name."""

    module_path, separator, leaf_name = state_dict_name.rpartition(".")
    canonical_module = module_path if separator else "self"
    if leaf_name == "weight":
        roles = frozenset({StateSlotRole.WEIGHT, StateSlotRole.NORM_SCALE})
    elif leaf_name == "bias":
        roles = frozenset({StateSlotRole.BIAS, StateSlotRole.NORM_OFFSET})
    elif leaf_name == "running_mean":
        roles = frozenset({StateSlotRole.RUNNING_MEAN})
    elif leaf_name == "running_var":
        roles = frozenset({StateSlotRole.RUNNING_VAR})
    elif leaf_name in {"num_batches_tracked", "counter"}:
        roles = frozenset({StateSlotRole.COUNTER})
    elif slot_role is TensorSlotRole.BUFFER:
        roles = frozenset({StateSlotRole.GENERIC_BUFFER})
    else:
        roles = frozenset({StateSlotRole.WEIGHT})
    return canonical_module, roles


def _alias_value_diagnostics(
    slots: tuple[TensorSlotDescriptor, ...],
    values_by_name: Mapping[str, torch.Tensor],
) -> list[RunnableDiagnostic]:
    """Return diagnostics when named entries in an alias group are not coherent."""

    slots_by_alias: dict[str, list[TensorSlotDescriptor]] = defaultdict(list)
    for slot in slots:
        binding = slot.state_binding
        assert binding is not None
        if binding.alias_group is not None:
            slots_by_alias[binding.alias_group].append(slot)
    diagnostics: list[RunnableDiagnostic] = []
    for alias_group, members in sorted(slots_by_alias.items()):
        named_values = [
            (slot.state_binding.state_dict_name, values_by_name[slot.state_binding.state_dict_name])
            for slot in members
            if slot.state_binding is not None
            and slot.state_binding.state_dict_name in values_by_name
        ]
        if len(named_values) < 2:
            continue
        first_name, first_value = named_values[0]
        for name, value in named_values[1:]:
            if (
                first_value.shape != value.shape
                or first_value.dtype != value.dtype
                or not torch.equal(first_value, value)
            ):
                diagnostics.append(
                    _diagnostic(
                        RunnableErrorCode.STATE_ALIAS_CONFLICT,
                        f"Alias group {alias_group!r} has conflicting named state values.",
                        detection_stage="state_alias_validation",
                        details=(
                            ("alias_group", alias_group),
                            ("first_state_dict_name", first_name),
                            ("conflicting_state_dict_name", name),
                        ),
                    )
                )
                break
    return diagnostics


def _initialize_state_slots(
    descriptor: SparseRunDescriptor,
    seed: int | None,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Allocate every state slot using the frozen role initializer table."""

    state_slots = _state_slots(descriptor)
    groups: dict[str, list[TensorSlotDescriptor]] = defaultdict(list)
    for slot in state_slots:
        binding = slot.state_binding
        assert binding is not None
        group = binding.alias_group or f"name:{binding.state_dict_name}"
        groups[group].append(slot)

    values: dict[str, torch.Tensor] = {}
    generator_by_device: dict[str, torch.Generator] = {}
    for members in sorted(groups.values(), key=lambda items: min(item.slot_id for item in items)):
        ordered = sorted(members, key=lambda item: item.slot_id)
        _validate_alias_allocation_contract(ordered)
        representative = ordered[0]
        generator = _generator_for_slot(representative, seed, generator_by_device)
        value = _initialize_slot(representative, generator)
        for member in ordered:
            values[member.slot_id] = value
    random_slot_ids = tuple(sorted(slot.slot_id for slot in state_slots))
    return values, random_slot_ids


def _validate_alias_allocation_contract(members: list[TensorSlotDescriptor]) -> None:
    """Require all members of one allocation group to share an initializer contract."""

    first = members[0]
    first_binding = first.state_binding
    assert first_binding is not None
    first_policy = CANONICAL_INITIALIZER_BY_ROLE[first_binding.semantic_role]
    for member in members[1:]:
        binding = member.state_binding
        assert binding is not None
        policy = CANONICAL_INITIALIZER_BY_ROLE[binding.semantic_role]
        if (
            member.shape != first.shape
            or member.dtype != first.dtype
            or member.device_type != first.device_type
            or member.device_index != first.device_index
            or policy is not first_policy
        ):
            raise _binding_error(
                (
                    _diagnostic(
                        RunnableErrorCode.STATE_ALIAS_CONFLICT,
                        "Alias group has incompatible allocation contracts.",
                        detection_stage="state_random_alias_preflight",
                        details=(("slot_ids", ",".join(item.slot_id for item in members)),),
                    ),
                )
            )


def _generator_for_slot(
    slot: TensorSlotDescriptor,
    seed: int | None,
    generators: dict[str, torch.Generator],
) -> torch.Generator | None:
    """Return one isolated per-device generator when an explicit seed is supplied."""

    if seed is None:
        return None
    device = _slot_device(slot)
    key = str(device)
    if key not in generators:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        generators[key] = generator
    return generators[key]


def _initialize_slot(
    slot: TensorSlotDescriptor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Allocate and fill one representative slot under N1-a."""

    binding = slot.state_binding
    assert binding is not None
    dtype = _torch_dtype(slot.dtype)
    device = _slot_device(slot)
    policy = CANONICAL_INITIALIZER_BY_ROLE[binding.semantic_role]
    _validate_initializer_dtype(slot, dtype)
    value = torch.empty(slot.shape, dtype=dtype, device=device)
    if policy is InitializerPolicy.ZEROS:
        return value.zero_()
    if policy is InitializerPolicy.ONES:
        return value.fill_(1)
    if not dtype.is_floating_point or not slot.shape:
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.STATE_DTYPE_MISMATCH,
                    f"Kaiming initialization is unsupported for slot {slot.slot_id!r}.",
                    detection_stage="state_random_initializer",
                    details=(("slot_id", slot.slot_id), ("dtype", slot.dtype)),
                ),
            )
        )
    fan_in = math.prod(slot.shape[1:]) if len(slot.shape) >= 2 else max(1, slot.shape[0])
    return value.normal_(mean=0.0, std=math.sqrt(2.0 / fan_in), generator=generator)


def _validate_initializer_dtype(slot: TensorSlotDescriptor, dtype: torch.dtype) -> None:
    """Reject dtype/semantic-role combinations outside frozen N1-a."""

    binding = slot.state_binding
    assert binding is not None
    floating_roles = {
        StateSlotRole.WEIGHT,
        StateSlotRole.BIAS,
        StateSlotRole.NORM_SCALE,
        StateSlotRole.NORM_OFFSET,
        StateSlotRole.RUNNING_MEAN,
        StateSlotRole.RUNNING_VAR,
    }
    integral_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    compatible = (
        dtype.is_floating_point
        if binding.semantic_role in floating_roles
        else binding.semantic_role is not StateSlotRole.COUNTER or dtype in integral_dtypes
    )
    if compatible:
        return
    raise _binding_error(
        (
            _diagnostic(
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"State role {binding.semantic_role.value!r} is incompatible with {slot.dtype!r}.",
                detection_stage="state_random_initializer",
                details=(
                    ("slot_id", slot.slot_id),
                    ("semantic_role", binding.semantic_role.value),
                    ("dtype", slot.dtype),
                ),
            ),
        )
    )


def _torch_dtype(dtype_name: str) -> torch.dtype:
    """Resolve one recorded public torch dtype without evaluating artifact text."""

    name = dtype_name.removeprefix("torch.")
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.STATE_DTYPE_MISMATCH,
                    f"Recorded state dtype {dtype_name!r} is unsupported.",
                    detection_stage="state_random_dtype",
                    details=(("dtype", dtype_name),),
                ),
            )
        )
    return value


def _slot_device(slot: TensorSlotDescriptor) -> torch.device:
    """Build the recorded allocation device for one state slot."""

    if slot.device_index is None:
        return torch.device(slot.device_type)
    return torch.device(slot.device_type, slot.device_index)


def _state_slots(descriptor: SparseRunDescriptor) -> tuple[TensorSlotDescriptor, ...]:
    """Return all state slots and reject parameter/buffer slots without bindings."""

    state_roles = {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}
    missing = tuple(
        slot.slot_id
        for slot in descriptor.tensor_slots
        if slot.role in state_roles and slot.state_binding is None
    )
    if missing:
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.MISSING_TENSOR_SLOT,
                    "Parameter or buffer slot is missing its state binding contract.",
                    detection_stage="state_slot_preflight",
                    details=(("slot_ids", ",".join(missing)),),
                ),
            )
        )
    return tuple(
        slot
        for slot in descriptor.tensor_slots
        if slot.role in state_roles and slot.state_binding is not None
    )


def _require_descriptor(trace: Any) -> SparseRunDescriptor:
    """Return a sparse descriptor or raise a structured binding error."""

    descriptor = trace.__dict__.get("_runnable_descriptor")
    if not isinstance(descriptor, SparseRunDescriptor):
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE,
                    "State binding requires a loaded sparse runnable Trace.",
                    detection_stage="state_descriptor_presence",
                ),
            )
        )
    return descriptor


def _binding_error(diagnostics: tuple[RunnableDiagnostic, ...]) -> StateBindingError:
    """Build one structured strict state-binding exception."""

    codes = tuple(diagnostic.code.value for diagnostic in diagnostics)
    return StateBindingError(
        f"Strict state binding failed with {len(diagnostics)} diagnostic(s): {', '.join(codes)}.",
        diagnostics=diagnostics,
        codes=codes,
    )


def _diagnostic(
    code: RunnableErrorCode,
    message: str,
    *,
    detection_stage: str,
    details: tuple[tuple[str, str], ...] = (),
) -> RunnableDiagnostic:
    """Build one state-binding diagnostic in the frozen shared shape."""

    return RunnableDiagnostic(
        code=code,
        message=message,
        registry_id=None,
        affected_op_labels=(),
        recorded_runtime=None,
        current_runtime=str(torch.__version__),
        detection_stage=detection_stage,
        resolver_provenance=None,
        analysis_load_available=True,
        details=details,
    )


__all__ = ["PreparedRunnableState", "load_trace_state_dict", "prepare_runnable_state"]
