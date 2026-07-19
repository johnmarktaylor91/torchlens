"""Non-executing state binding and allocation for sparse runnable traces."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
import math
from types import MappingProxyType
from typing import Any

import torch

from . import _state
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


def snapshot_capture_state(model: object) -> Mapping[str, torch.Tensor] | None:
    """Clone a model's persistent state at the capture execution boundary.

    Parameters
    ----------
    model:
        Live model about to execute the captured forward pass.

    Returns
    -------
    Mapping[str, torch.Tensor] | None
        Detached tensor clones keyed by canonical ``state_dict`` name, or
        ``None`` when the model cannot provide a tensor-only state mapping.

    Notes
    -----
    The clones deliberately retain their original device. Runnable descriptor
    validation records the capture device contract, while serialization later
    performs its ordinary device-neutral transport conversion.
    """

    state_dict_method = getattr(model, "state_dict", None)
    if not callable(state_dict_method):
        return None
    try:
        state = state_dict_method()
    except Exception:
        return None
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        return None
    with _state.pause_logging():
        return MappingProxyType({name: value.detach().clone() for name, value in state.items()})


def snapshot_state_alias_topology(model: object) -> Mapping[str, Any] | None:
    """Capture the live bound-state alias topology BEFORE cloning erases it (r37 corr2-4).

    Walks every named parameter and buffer (``remove_duplicate=False``) of the LIVE
    model and classifies each pair through the shared absolute-byte relation engine:

    * repeated live Python object identity -> a shared ``alias_group`` (one staged
      allocation reproduces ``a is b`` semantics -- tied weights, double-registered
      buffers);
    * DISTINCT objects whose touched bytes ``overlap`` (or are ``unknown``, including
      an unprovable footprint) -> a save-time REFUSAL record: the v2 schema has no
      backing-storage/view recipe, so serializing the pair as independent values
      would silently change in-place propagation semantics on replay (the corr2-4
      false-VERIFIED class);
    * proved-``disjoint`` pairs (including disjoint views of one storage) serialize
      independently.

    An interval sweep per device address space bounds the pairwise work. Runs under
    ``pause_logging`` so geometry reads are never captured. Returns ``None`` when the
    model exposes no named state accessors.
    """

    from .utils.tensor_utils import tensor_byte_footprint, touched_bytes_relation

    named_parameters = getattr(model, "named_parameters", None)
    named_buffers = getattr(model, "named_buffers", None)
    if not callable(named_parameters) or not callable(named_buffers):
        return None
    try:
        entries: list[tuple[str, torch.Tensor]] = [
            (str(name), value)
            for name, value in named_parameters(remove_duplicate=False)
            if isinstance(value, torch.Tensor)
        ]
        entries.extend(
            (str(name), value)
            for name, value in named_buffers(remove_duplicate=False)
            if isinstance(value, torch.Tensor)
        )
    except Exception:
        return None
    groups: dict[str, str] = {}
    names_by_object: dict[int, list[str]] = {}
    tensor_by_name: dict[str, torch.Tensor] = {}
    for name, value in entries:
        names_by_object.setdefault(id(value), []).append(name)
        tensor_by_name.setdefault(name, value)
    for names in names_by_object.values():
        if len(names) > 1:
            group_id = f"state_alias:{min(names)}"
            for name in names:
                groups[name] = group_id
    refusals: list[tuple[str, str, str]] = []
    with _state.pause_logging():
        footprints: list[tuple[str, Any]] = []
        for name, value in tensor_by_name.items():
            footprint = tensor_byte_footprint(value)
            if footprint is None:
                # An unprovable footprint cannot participate in any disjointness
                # proof: fail closed against every other state name (INV-2).
                refusals.append((name, "<unprovable footprint>", "unknown"))
                continue
            if footprint.numel == 0:
                continue
            footprints.append((name, footprint))
        # Interval sweep per device address space: only interval-intersecting pairs
        # need the exact relation (identity pairs were already grouped above).
        footprints.sort(key=lambda item: (item[1].device_key, item[1].start_byte))
        active: list[tuple[str, Any]] = []
        for name, footprint in footprints:
            still_active: list[tuple[str, Any]] = []
            for other_name, other in active:
                if (
                    other.device_key == footprint.device_key
                    and other.end_byte > footprint.start_byte
                ):
                    still_active.append((other_name, other))
            active = still_active
            for other_name, _other in active:
                left_tensor = tensor_by_name[name]
                right_tensor = tensor_by_name[other_name]
                if left_tensor is right_tensor:
                    continue  # identity: covered by an alias group
                relation = touched_bytes_relation(left_tensor, right_tensor)
                if relation != "disjoint":
                    refusals.append((other_name, name, relation))
            active.append((name, footprint))
    # Plain containers (not MappingProxyType): the snapshot rides on the Trace
    # ``__dict__`` and must survive pickle/deepcopy like its sibling capture-state.
    return {"groups": groups, "refusals": tuple(refusals)}


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


def bind_embedded_trace_state(trace: Any, sd: Mapping[str, Any]) -> None:
    """Validate and atomically bind an embedded capture-time state mapping.

    Parameters
    ----------
    trace:
        Loaded sparse Trace receiving the optional weight payload.
    sd:
        Canonically named parameter and persistent-buffer tensors decoded from
        the runnable artifact.

    Raises
    ------
    StateBindingError
        If any embedded entry violates the ordinary strict state contract.
    """

    _validate_state_mapping(trace, sd)
    trace.__dict__["_runnable_embedded_state"] = MappingProxyType(
        {
            name: value.detach().clone()
            for name, value in sd.items()
            if isinstance(name, str) and isinstance(value, torch.Tensor)
        }
    )


def bind_embedded_nonpersistent_buffers(trace: Any, buffers: Mapping[str, Any]) -> None:
    """Validate and bind captured values for used non-persistent buffers.

    Parameters
    ----------
    trace:
        Loaded sparse Trace receiving the mandatory buffer payload.
    buffers:
        Registered buffer names mapped to capture-time tensor values.

    Raises
    ------
    StateBindingError
        If names, shapes, dtypes, or roles violate the non-persistent slots.
    """

    descriptor = _require_descriptor(trace)
    validate_nonpersistent_buffer_mapping_for_descriptor(descriptor, buffers)
    trace.__dict__["_runnable_embedded_nonpersistent_buffers"] = {
        name: value.detach().clone()
        for name, value in buffers.items()
        if isinstance(name, str) and isinstance(value, torch.Tensor)
    }


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
    nonpersistent_buffers = _prepared_nonpersistent_buffers(trace, descriptor)

    def _staged(prepared: PreparedRunnableState) -> PreparedRunnableState:
        return replace(
            prepared,
            slot_values=stage_state_to_slot_devices(descriptor, prepared.slot_values),
        )

    user_state = trace.__dict__.get("_runnable_staged_user_state")
    if isinstance(user_state, Mapping):
        return _staged(
            _with_nonpersistent_buffers(
                _prepared_bound_state(user_state, StateSource.USER_STATE_DICT, seed),
                nonpersistent_buffers,
            )
        )

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
        return _staged(
            _with_nonpersistent_buffers(
                _prepared_bound_state(validated, StateSource.EMBEDDED_CAPTURE_STATE, seed),
                nonpersistent_buffers,
            )
        )
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
    return _staged(
        _with_nonpersistent_buffers(
            PreparedRunnableState(
                slot_values=MappingProxyType(slot_values),
                state_source=StateSource.RANDOM_INITIALIZATION,
                initializer_policy_version=RUNNABLE_INITIALIZER_POLICY_VERSION,
                seed=seed,
                random_filled_slot_ids=random_slot_ids,
            ),
            nonpersistent_buffers,
        )
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


def _with_nonpersistent_buffers(
    prepared: PreparedRunnableState,
    buffers: Mapping[str, torch.Tensor],
) -> PreparedRunnableState:
    """Merge capture-embedded non-persistent buffers into prepared run state."""

    values = dict(prepared.slot_values)
    values.update(buffers)
    return replace(prepared, slot_values=MappingProxyType(values))


def _prepared_nonpersistent_buffers(
    trace: Any,
    descriptor: SparseRunDescriptor,
) -> Mapping[str, torch.Tensor]:
    """Return validated slot-keyed capture values for non-persistent buffers."""

    slots = _nonpersistent_buffer_slots(descriptor)
    declared = descriptor.payload_layers.nonpersistent_buffers
    embedded = trace.__dict__.get("_runnable_embedded_nonpersistent_buffers")
    if not slots:
        return MappingProxyType({})
    if (
        not declared.present
        or declared.schema != "runnable_nonpersistent_buffer_v1"
        or not isinstance(embedded, Mapping)
    ):
        raise _binding_error(
            (
                _diagnostic(
                    RunnableErrorCode.STATE_MISSING_KEY,
                    "Used non-persistent buffers require their capture-embedded payload.",
                    detection_stage="state_nonpersistent_buffer_hook",
                ),
            )
        )
    return validate_nonpersistent_buffer_mapping_for_descriptor(descriptor, embedded)


def _validate_state_mapping(trace: Any, sd: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Validate one strict mapping and return detached slot-keyed values."""

    descriptor = _require_descriptor(trace)
    return validate_state_mapping_for_descriptor(descriptor, sd)


def validate_state_mapping_for_descriptor(
    descriptor: SparseRunDescriptor,
    sd: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Validate a strict state mapping against an explicit sparse descriptor.

    Parameters
    ----------
    descriptor:
        Runnable descriptor supplying canonical state slot contracts.
    sd:
        Mapping of canonical state names to tensor values.

    Returns
    -------
    Mapping[str, torch.Tensor]
        Detached, slot-keyed values suitable for one runnable execution.

    Raises
    ------
    StateBindingError
        If names, roles, aliases, shapes, or dtypes violate the contract.
    """

    return _validate_named_slot_mapping(_persistent_state_slots(descriptor), sd)


def validate_nonpersistent_buffer_mapping_for_descriptor(
    descriptor: SparseRunDescriptor,
    buffers: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Validate captured non-persistent buffers against their runnable slots.

    Parameters
    ----------
    descriptor:
        Runnable descriptor supplying non-persistent buffer contracts.
    buffers:
        Mapping of registered buffer names to captured tensor values.

    Returns
    -------
    Mapping[str, torch.Tensor]
        Detached, slot-keyed buffer values suitable for execution.

    Raises
    ------
    StateBindingError
        If names, roles, aliases, shapes, or dtypes violate the contract.
    """

    return _validate_named_slot_mapping(_nonpersistent_buffer_slots(descriptor), buffers)


def _validate_named_slot_mapping(
    state_slots: tuple[TensorSlotDescriptor, ...],
    values: Mapping[str, Any],
) -> Mapping[str, torch.Tensor]:
    """Validate one exact canonical-name mapping for a selected slot set."""

    if not isinstance(values, Mapping):
        raise TypeError("state values must be a mapping of canonical names to tensors.")

    slots_by_name: dict[str, list[TensorSlotDescriptor]] = defaultdict(list)
    for slot in state_slots:
        assert slot.state_binding is not None
        slots_by_name[slot.state_binding.state_dict_name].append(slot)

    diagnostics: list[RunnableDiagnostic] = []
    supplied_names = {name for name in values if isinstance(name, str)}
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
    for key in values:
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
        value = values[name]
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


def _alias_values_coherent(first: torch.Tensor, second: torch.Tensor) -> bool:
    """Return whether two same-shape/dtype alias-group values are byte-coherent.

    r35 corr2_2: coherence is the STORAGE-IDENTITY fast path (the tied-parameter
    case -- one allocation exposed under two canonical names) or frozen
    LOGICAL-BYTE equality via the same representation as
    ``runnable_tensor_byte_digest``. Never float ``torch.equal`` (NaN payloads
    falsely conflict), never ``allclose``, never ``equal_nan=True`` (a
    different-NaN-payload pair must still conflict): NaN payloads, complex
    components, signed zero, infinities, and int/bool bytes are all exact.
    """

    try:
        if (
            first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
            and first.storage_offset() == second.storage_offset()
            and tuple(first.stride()) == tuple(second.stride())
            and tuple(first.shape) == tuple(second.shape)
        ):
            return True
    except (RuntimeError, AttributeError, TypeError, NotImplementedError):
        pass
    try:
        return runnable_tensor_byte_digest(first) == runnable_tensor_byte_digest(second)
    except Exception:
        # An undigestable exotic value cannot be proven coherent: conflict
        # (fail closed, atomic typed rejection at the caller).
        return False


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
                or not _alias_values_coherent(first_value, value)
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

    state_slots = _persistent_state_slots(descriptor)
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


def _kaiming_fan_in(shape: tuple[int, ...]) -> int:
    """Return the frozen N1-a fan-in for a nonempty Kaiming-initialized shape."""

    return math.prod(shape[1:]) if len(shape) >= 2 else max(1, shape[0])


def initializer_contract_diagnostics(
    slot: TensorSlotDescriptor,
) -> tuple[RunnableDiagnostic, ...]:
    """Validate the ``torchlens_role_init_v2`` totality contract for one slot.

    Shared by producer preflight AND the runtime initializer (defense in depth,
    corr2_3): a legal ``numel() == 0`` slot is total for every role (allocated
    and returned with ZERO generator consumption), and every nonempty Kaiming
    slot must have finite positive ``fan_in``. An unsupported contract fails
    typed here, never by division or backend sampling at run time.
    """

    binding = slot.state_binding
    if binding is None:
        return ()
    policy = CANONICAL_INITIALIZER_BY_ROLE[binding.semantic_role]
    if policy is not InitializerPolicy.KAIMING_NORMAL:
        return ()
    if math.prod(slot.shape) == 0:
        # Degenerate-total: nothing to sample.
        return ()
    if not slot.shape:
        return (
            _diagnostic(
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"Kaiming initialization is unsupported for scalar slot {slot.slot_id!r}.",
                detection_stage="state_random_initializer",
                details=(("slot_id", slot.slot_id), ("dtype", slot.dtype)),
            ),
        )
    fan_in = _kaiming_fan_in(slot.shape)
    if fan_in <= 0:
        return (
            _diagnostic(
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"Kaiming initialization requires finite positive fan_in for slot "
                f"{slot.slot_id!r}; got {fan_in}.",
                detection_stage="state_random_initializer",
                details=(("slot_id", slot.slot_id), ("shape", repr(slot.shape))),
            ),
        )
    return ()


def _initialize_slot(
    slot: TensorSlotDescriptor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Allocate and fill one representative slot under ``torchlens_role_init_v2``."""

    binding = slot.state_binding
    assert binding is not None
    dtype = _torch_dtype(slot.dtype)
    device = _slot_device(slot)
    policy = CANONICAL_INITIALIZER_BY_ROLE[binding.semantic_role]
    _validate_initializer_dtype(slot, dtype)
    value = torch.empty(slot.shape, dtype=dtype, device=device)
    if value.numel() == 0:
        # v2 degenerate totality: a legal empty slot returns its allocation
        # immediately -- no sampling, PROVABLY zero generator consumption.
        return value
    if policy is InitializerPolicy.ZEROS:
        return value.zero_()
    if policy is InitializerPolicy.ONES:
        return value.fill_(1)
    contract = initializer_contract_diagnostics(slot)
    if contract:
        raise _binding_error(contract)
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
    fan_in = _kaiming_fan_in(slot.shape)
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


def _slot_device_compatible(value: torch.Tensor, slot: TensorSlotDescriptor) -> bool:
    """Return whether a tensor already satisfies a slot's recorded device contract."""

    if value.device.type != slot.device_type:
        return False
    return (
        slot.device_index is None
        or value.device.index is None
        or value.device.index == slot.device_index
    )


def stage_state_to_slot_devices(
    descriptor: SparseRunDescriptor,
    values: Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    """Stage every bound state value to its recorded slot device, atomically (r37 R5).

    LAZY placement: blobs stay transport-placed (``map_location`` semantics) through
    load and analysis; THIS single helper -- the sole execution-placement authority
    -- runs once at run preparation for embedded capture state, staged user state,
    and required non-persistent buffers alike. Transfers happen once per shared
    value (alias groups keep one allocation and their shared identity), preserve
    dtype (already validated) and ``requires_grad``, and publish NOTHING on failure:
    a device this runtime cannot allocate raises the typed
    ``run_capability_unavailable`` refusal before any callable resolves an input.
    """

    from .errors import RunCapabilityUnavailableError

    slots_by_id = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    staged: dict[str, torch.Tensor] = {}
    moved_by_identity: dict[int, torch.Tensor] = {}
    for slot_id, value in values.items():
        slot = slots_by_id.get(slot_id)
        if slot is None or _slot_device_compatible(value, slot):
            staged[slot_id] = value
            continue
        cached = moved_by_identity.get(id(value))
        if cached is None:
            try:
                with _state.pause_logging():
                    cached = value.to(_slot_device(slot))
            except (RuntimeError, AssertionError) as exc:
                raise RunCapabilityUnavailableError(
                    f"State slot {slot_id!r} requires device "
                    f"{slot.device_type}"
                    + (f":{slot.device_index}" if slot.device_index is not None else "")
                    + f", which this runtime cannot stage: {exc}",
                    code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                ) from exc
            if value.requires_grad and not cached.requires_grad:
                cached.requires_grad_(True)
            moved_by_identity[id(value)] = cached
        staged[slot_id] = cached
    return MappingProxyType(staged)


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


def _persistent_state_slots(
    descriptor: SparseRunDescriptor,
) -> tuple[TensorSlotDescriptor, ...]:
    """Return only slots belonging to the canonical persistent state mapping."""

    return tuple(
        slot
        for slot in _state_slots(descriptor)
        if slot.state_binding is not None and slot.state_binding.persistent
    )


def _nonpersistent_buffer_slots(
    descriptor: SparseRunDescriptor,
) -> tuple[TensorSlotDescriptor, ...]:
    """Return registered buffer slots intentionally excluded from ``state_dict``."""

    return tuple(
        slot
        for slot in _state_slots(descriptor)
        if slot.role is TensorSlotRole.BUFFER
        and slot.state_binding is not None
        and not slot.state_binding.persistent
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


def runnable_tensor_byte_digest(value: torch.Tensor) -> str:
    """Return a SHA-256 digest of one tensor's exact logical value.

    Parameters
    ----------
    value:
        Dense tensor whose capture/runtime bytes should be attested.

    Returns
    -------
    str
        Lowercase SHA-256 hexadecimal digest of dtype, shape, and contiguous
        CPU bytes.

    Raises
    ------
    RunPreconditionError
        If the tensor is not a plain strided dense tensor (r35 I5 defense in
        depth: the digest helper itself refuses exotic layouts typed instead of
        surfacing a raw backend error mid-materialization).
    """

    if (
        value.layout is not torch.strided
        or bool(getattr(value, "is_meta", False))
        or value.is_nested
        or bool(getattr(value, "is_quantized", False))
        or any(name is not None for name in (value.names or ()))
    ):
        from .errors import RunPreconditionError

        raise RunPreconditionError(
            "Byte digesting requires a plain strided dense tensor; got layout "
            f"{value.layout} (meta={bool(getattr(value, 'is_meta', False))}, "
            f"nested={bool(value.is_nested)}, "
            f"quantized={bool(getattr(value, 'is_quantized', False))}).",
            code=RunnableErrorCode.INPUT_TREE_MISMATCH.value,
        )
    with _state.pause_logging():
        cpu_value = value.detach().cpu().contiguous()
        payload = cpu_value.reshape(-1).view(torch.uint8).numpy().tobytes()
        logical_prefix = f"{cpu_value.dtype}|{tuple(cpu_value.shape)}|".encode("utf-8")
    return sha256(logical_prefix + payload).hexdigest()


__all__ = [
    "PreparedRunnableState",
    "load_trace_state_dict",
    "prepare_runnable_state",
    "runnable_tensor_byte_digest",
]
