"""Transactional execution providers for the unified :meth:`Trace.run` surface."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from itertools import count
from typing import Any, cast

import torch

from . import _state
from ._runnable_state import (
    PreparedRunnableState,
    prepare_runnable_state,
    runnable_tensor_byte_digest,
)
from .errors import (
    NumericAttestationError,
    PathDivergenceError,
    ReattachError,
    RunCapabilityUnavailableError,
    RunPreconditionError,
    RuntimeSignatureDriftError,
)
from .intervention.replay import _CallConeNode, _walk_call_cone
from .ir.container import ContainerSpec, rebuild_container_from_spec
from .runnable import (
    ActivationPayloadLayerDescriptor,
    CallableRegistryEntry,
    ContractCheck,
    ControlWitness,
    ControlWitnessKind,
    DivergencePolicy,
    LiteralAtom,
    LiteralMapping,
    LiteralSequence,
    LiteralSequenceKind,
    LiteralTorchSymbol,
    LiteralTupleKey,
    NonTensorLiteral,
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessReport,
    ReadinessStatus,
    RunProvider,
    RunReport,
    RunResult,
    RunnableCallDescriptor,
    RunnableDiagnostic,
    RunnableErrorCode,
    SparseRunDescriptor,
    StateSlotRole,
    StateSource,
    TensorSlotDescriptor,
    TensorSlotRole,
    WitnessCompleteness,
    mark_trace_path_status,
)

_RUN_FORK_COUNTER = count(1)


def run_loaded_sparse_trace(
    trace: Any,
    inputs: Any,
    *,
    seed: int | None,
    on_divergence: DivergencePolicy,
) -> RunResult:
    """Execute a loaded sparse recipe on a transactional Trace fork.

    Parameters
    ----------
    trace:
        Loaded Trace that owns the sparse descriptor and resolved callables.
    inputs:
        Runtime model-input tree.
    seed:
        Optional isolated state-initialization and runtime RNG seed.
    on_divergence:
        Frozen Stage-6 policy argument. Stage 5 reports faithfulness but does
        not yet enforce divergence behavior.

    Returns
    -------
    RunResult
        Structured output, run fork, and execution report.
    """

    try:
        divergence_policy = DivergencePolicy(on_divergence)
    except ValueError as exc:
        raise ValueError(
            "on_divergence must be DivergencePolicy.RAISE or DivergencePolicy.RETURN_DIVERGED."
        ) from exc
    descriptor, readiness, callables = _require_loaded_sparse_provider(trace)
    slot_values, input_checks = _bind_runtime_inputs(descriptor, inputs)
    input_byte_digests = _snapshot_input_byte_digests(descriptor, slot_values)
    _raise_first_divergence(input_checks, divergence_policy, fork=None)
    prepared_state = prepare_runnable_state(trace, seed=seed)
    slot_values.update(_clone_state_values(prepared_state.slot_values))
    fork = trace._fork_trace(name=_run_fork_name(trace))
    try:
        return _execute_loaded_sparse_transaction(
            trace,
            inputs,
            seed=seed,
            divergence_policy=divergence_policy,
            descriptor=descriptor,
            readiness=readiness,
            callables=callables,
            slot_values=slot_values,
            input_byte_digests=input_byte_digests,
            input_checks=input_checks,
            prepared_state=prepared_state,
            fork=fork,
        )
    except BaseException:
        _state._unregister_log(fork)
        raise


def _execute_loaded_sparse_transaction(
    trace: Any,
    inputs: Any,
    *,
    seed: int | None,
    divergence_policy: DivergencePolicy,
    descriptor: SparseRunDescriptor,
    readiness: ReadinessReport,
    callables: Mapping[str, Callable[..., Any]],
    slot_values: dict[str, torch.Tensor],
    input_byte_digests: Mapping[str, str],
    input_checks: tuple[ContractCheck, ...],
    prepared_state: PreparedRunnableState,
    fork: Any,
) -> RunResult:
    """Execute one sparse transaction whose caller owns rollback on escape."""

    _populate_source_slots(fork, descriptor, slot_values)
    contract_checks: list[ContractCheck] = [
        *input_checks,
        *_state_contract_checks(descriptor, slot_values),
    ]
    _raise_first_divergence(contract_checks, divergence_policy, fork=fork)
    call_outputs: dict[str, Any] = {}
    attestation_slot_ids = _raw_activation_slot_ids(descriptor)
    attestation_slot_values: dict[str, torch.Tensor] = {}

    devices = sorted(
        {
            value.device.index
            for value in slot_values.values()
            if value.device.type == "cuda" and value.device.index is not None
        }
    )
    rng_context = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
    with rng_context, _state.pause_logging():
        if seed is not None:
            torch.manual_seed(seed)

        def execute_call(call_node: _CallConeNode) -> None:
            """Execute and stage one dependency-ready sparse call."""

            call = cast(RunnableCallDescriptor, call_node)
            call_checks, before_versions = _pre_call_contract_checks(
                descriptor,
                call,
                slot_values,
            )
            contract_checks.extend(call_checks)
            _raise_first_divergence(contract_checks, divergence_policy, fork=fork)
            output = _execute_sparse_call(call, callables[call.call_id], slot_values)
            call_outputs[call.call_id] = output
            contract_checks.extend(
                _bind_call_outputs(
                    descriptor,
                    call,
                    output,
                    slot_values,
                    fork,
                    before_versions=before_versions,
                    attestation_slot_ids=attestation_slot_ids,
                    attestation_slot_values=attestation_slot_values,
                )
            )
            contract_checks.extend(_call_witness_checks(descriptor, call, slot_values))
            _raise_first_divergence(contract_checks, divergence_policy, fork=fork)

        _walk_call_cone(descriptor.calls, execute_call)

    output = _reconstruct_output(descriptor, slot_values, fork)
    contract_checks.extend(
        _post_execution_contract_checks(
            descriptor,
            inputs=inputs,
            output=output,
            call_outputs=call_outputs,
            slot_values=slot_values,
            fork=fork,
        )
    )
    _raise_first_divergence(contract_checks, divergence_policy, fork=fork)
    numeric_attestation, attestation_check = _numeric_attestation_check(
        descriptor,
        prepared_state,
        slot_values=slot_values,
        attestation_slot_values=attestation_slot_values,
        input_byte_digests=input_byte_digests,
        trace=trace,
    )
    if attestation_check is not None:
        contract_checks.append(attestation_check)
        if not attestation_check.passed:
            _raise_numeric_attestation_failure(fork, attestation_check)
    path_faithfulness, mismatch = _path_faithfulness(descriptor, contract_checks)
    path_faithfulness, mismatch = mark_trace_path_status(
        fork,
        path_faithfulness,
        mismatch,
    )
    _raise_monotonic_divergence(
        fork,
        path_faithfulness,
        mismatch,
        divergence_policy,
    )
    report = _run_report(
        readiness,
        prepared_state,
        contract_checks=tuple(contract_checks),
        path_faithfulness=path_faithfulness,
        first_mismatch=mismatch,
        numeric_attestation=numeric_attestation,
    )
    return RunResult(output=output, trace=fork, report=report)


def run_live_trace(
    trace: Any,
    inputs: Any,
    *,
    seed: int | None,
) -> RunResult:
    """Run the live-model refresh provider on a transactional fork.

    Parameters
    ----------
    trace:
        Live Trace retaining its source-model weak reference.
    inputs:
        New forward input accepted by the existing ``save_new_outs`` path.
    seed:
        Optional refresh seed.

    Returns
    -------
    RunResult
        Structured output, refreshed fork, and live-provider report.

    Raises
    ------
    RunCapabilityUnavailableError
        If the live source model is no longer available.
    """

    source_ref = getattr(trace, "_source_model_ref", None)
    model = source_ref() if source_ref is not None else None
    if model is None:
        raise RunCapabilityUnavailableError(
            "The live Trace no longer retains its source model.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            provider=RunProvider.LIVE,
        )
    fork = trace._fork_trace(name=_run_fork_name(trace))
    fork.save_new_outs(model, inputs, random_seed=seed)
    output = _reconstruct_live_output(fork)
    readiness = ReadinessReport(
        status=ReadinessStatus.READY,
        provider=RunProvider.LIVE,
        backend=str(getattr(trace, "backend", "torch")),
        capability="live_model_fast_capture",
        resolver_records=(),
        state_sources_available=(StateSource.LIVE_MODEL_STATE,),
        witness_completeness=None,
        diagnostics=(),
    )
    report = RunReport(
        readiness=readiness,
        state_source=StateSource.LIVE_MODEL_STATE,
        initializer_policy_version=None,
        seed=seed,
        random_filled_slot_ids=(),
        contract_checks=(ContractCheck("live_graph_alignment", True, None),),
        path_faithfulness=PathFaithfulness.VERIFIED,
        first_mismatch=None,
        numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
        poisoned=False,
    )
    return RunResult(output=output, trace=fork, report=report)


def raise_analysis_run_unavailable(trace: Any) -> None:
    """Raise the typed capability error for an analysis-only loaded Trace.

    Parameters
    ----------
    trace:
        Analysis-only loaded Trace.

    Raises
    ------
    RunCapabilityUnavailableError
        Always, with the load-time readiness report attached.
    """

    readiness = trace.__dict__.get("_runnable_readiness")
    diagnostics = () if readiness is None else readiness.diagnostics
    raise RunCapabilityUnavailableError(
        "This loaded Trace is analysis-only and has no sparse run descriptor.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        readiness=readiness,
        diagnostics=diagnostics,
    )


def _require_loaded_sparse_provider(
    trace: Any,
) -> tuple[SparseRunDescriptor, ReadinessReport, Mapping[str, Callable[..., Any]]]:
    """Return ready descriptor state or raise one aggregate typed error."""

    readiness = trace.__dict__.get("_runnable_readiness")
    descriptor = trace.__dict__.get("_runnable_descriptor")
    callables = trace.__dict__.get("_runnable_callables_by_call_id")
    if not isinstance(readiness, ReadinessReport) or not isinstance(
        descriptor, SparseRunDescriptor
    ):
        raise_analysis_run_unavailable(trace)
    if readiness.status is not ReadinessStatus.READY or not isinstance(callables, Mapping):
        raise ReattachError(
            "Sparse callable reattachment did not produce a ready atomic attachment.",
            readiness=readiness,
            diagnostics=readiness.diagnostics,
        )
    return descriptor, readiness, cast(Mapping[str, Callable[..., Any]], callables)


def _bind_runtime_inputs(
    descriptor: SparseRunDescriptor,
    inputs: Any,
) -> tuple[dict[str, torch.Tensor], tuple[ContractCheck, ...]]:
    """Bind and defensively clone public input leaves by persisted model sites."""

    input_slots = tuple(
        slot for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    )
    values: dict[str, torch.Tensor] = {}
    checks: list[ContractCheck] = []
    positions = {
        slot.input_binding.model_site_position
        for slot in input_slots
        if slot.input_binding is not None
    }
    for slot in input_slots:
        binding = slot.input_binding
        if binding is None:
            raise _input_error(
                RunnableErrorCode.MISSING_INPUT_CONTAINER_CONTRACT,
                slot,
                "Input slot has no persisted model-site binding.",
            )
        try:
            root = _input_site_value(inputs, binding.model_site_position, positions)
            value = _value_at_path(root, binding.container_path)
        except (KeyError, IndexError, TypeError) as exc:
            raise _input_error(
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                slot,
                f"Runtime input tree does not contain the recorded input path: {exc}",
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise _input_error(
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                slot,
                f"Runtime input leaf is {type(value).__name__}, expected torch.Tensor.",
            )
        values[slot.slot_id] = value.detach().clone()
        shape_ok = tuple(value.shape) == slot.shape
        dtype_ok = str(value.dtype) == slot.dtype
        checks.append(
            _contract_check(
                f"input_shape:{slot.slot_id}",
                shape_ok,
                RunnableErrorCode.INPUT_SHAPE_MISMATCH,
                f"Runtime input shape {tuple(value.shape)} does not match {slot.shape}.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(tuple(value.shape))),
                ),
            )
        )
        checks.append(
            _contract_check(
                f"input_dtype:{slot.slot_id}",
                dtype_ok,
                RunnableErrorCode.INPUT_DTYPE_MISMATCH,
                f"Runtime input dtype {value.dtype} does not match {slot.dtype}.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_dtype", slot.dtype),
                    ("actual_dtype", str(value.dtype)),
                ),
            )
        )
    checks.extend(_input_tree_contract_checks(descriptor, inputs))
    return values, tuple(checks)


def _clone_state_values(
    values: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Clone run-local state while preserving recorded alias groups."""

    clones_by_identity: dict[int, torch.Tensor] = {}
    cloned: dict[str, torch.Tensor] = {}
    for slot_id, value in values.items():
        clone = clones_by_identity.get(id(value))
        if clone is None:
            clone = value.detach().clone()
            clones_by_identity[id(value)] = clone
        cloned[slot_id] = clone
    return cloned


def _snapshot_input_byte_digests(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> dict[str, str]:
    """Digest cloned model inputs before sparse calls can mutate them in place."""

    return {
        slot.slot_id: runnable_tensor_byte_digest(slot_values[slot.slot_id])
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.MODEL_INPUT and slot.slot_id in slot_values
    }


def _input_site_value(inputs: Any, position: Any, positions: set[Any]) -> Any:
    """Select one top-level argument or keyword site from the public input tree."""

    if isinstance(position, tuple) and len(position) == 2:
        kind, key = position
        if kind == "arg":
            if len(positions) == 1 and key == 0:
                return inputs
            if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
                raise TypeError("multiple positional model sites require a sequence input")
            return inputs[cast(int, key)]
        if kind == "kwarg":
            if not isinstance(inputs, Mapping):
                raise TypeError("keyword model sites require a mapping input")
            return inputs[key]
    return _value_at_path(inputs, position if isinstance(position, tuple) else (position,))


def _input_tree_contract_checks(
    descriptor: SparseRunDescriptor,
    inputs: Any,
) -> tuple[ContractCheck, ...]:
    """Compare the runtime tensor-leaf tree with every recorded input site."""

    slots = tuple(
        slot
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.MODEL_INPUT and slot.input_binding is not None
    )
    positions = {
        slot.input_binding.model_site_position for slot in slots if slot.input_binding is not None
    }
    expected_by_position: dict[Any, set[tuple[str | int, ...]]] = {}
    for slot in slots:
        assert slot.input_binding is not None
        expected_by_position.setdefault(slot.input_binding.model_site_position, set()).add(
            slot.input_binding.container_path
        )
    checks: list[ContractCheck] = []
    for position, expected in expected_by_position.items():
        try:
            root = _input_site_value(inputs, position, positions)
            actual = set(_tensor_leaf_paths(root))
        except (KeyError, IndexError, TypeError):
            actual = set()
        checks.append(
            _contract_check(
                f"input_tree:{position!r}",
                actual == expected,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime input tensor-leaf paths disagree with the recorded input tree.",
                details=(
                    ("model_site_position", repr(position)),
                    ("expected_paths", repr(sorted(expected, key=repr))),
                    ("actual_paths", repr(sorted(actual, key=repr))),
                ),
            )
        )
    return tuple(checks)


def _state_contract_checks(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> tuple[ContractCheck, ...]:
    """Recheck state tensor and alias contracts inside the DAG transaction."""

    checks: list[ContractCheck] = []
    aliases: dict[str, list[TensorSlotDescriptor]] = {}
    for slot in descriptor.tensor_slots:
        if slot.role not in {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}:
            continue
        binding = slot.state_binding
        value = slot_values.get(slot.slot_id)
        present = isinstance(value, torch.Tensor)
        checks.append(
            _contract_check(
                f"state_slot:{slot.slot_id}",
                present,
                RunnableErrorCode.MISSING_TENSOR_SLOT,
                f"State slot {slot.slot_id!r} was not bound for execution.",
                details=(("slot_id", slot.slot_id),),
            )
        )
        if not present:
            continue
        assert value is not None
        checks.append(
            _contract_check(
                f"state_shape:{slot.slot_id}",
                tuple(value.shape) == slot.shape,
                RunnableErrorCode.STATE_SHAPE_MISMATCH,
                f"State slot {slot.slot_id!r} has a runtime shape mismatch.",
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(tuple(value.shape))),
                ),
            )
        )
        checks.append(
            _contract_check(
                f"state_dtype:{slot.slot_id}",
                str(value.dtype) == slot.dtype,
                RunnableErrorCode.STATE_DTYPE_MISMATCH,
                f"State slot {slot.slot_id!r} has a runtime dtype mismatch.",
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_dtype", slot.dtype),
                    ("actual_dtype", str(value.dtype)),
                ),
            )
        )
        if binding is not None and binding.alias_group is not None:
            aliases.setdefault(binding.alias_group, []).append(slot)
        if binding is not None:
            module_path, separator, leaf_name = binding.state_dict_name.rpartition(".")
            canonical_module = module_path if separator else "self"
            allowed_roles = _allowed_state_roles(leaf_name, slot.role)
            checks.append(
                _contract_check(
                    f"state_name_role:{slot.slot_id}",
                    bool(binding.state_dict_name)
                    and binding.module_path == canonical_module
                    and binding.semantic_role in allowed_roles,
                    RunnableErrorCode.STATE_ROLE_MISMATCH,
                    f"State slot {slot.slot_id!r} has an inconsistent name/role contract.",
                    details=(
                        ("slot_id", slot.slot_id),
                        ("state_dict_name", binding.state_dict_name),
                        ("recorded_module_path", binding.module_path),
                        ("canonical_module_path", canonical_module),
                        ("semantic_role", binding.semantic_role.value),
                    ),
                )
            )
    for alias_group, members in sorted(aliases.items()):
        values = [slot_values[slot.slot_id] for slot in members if slot.slot_id in slot_values]
        checks.append(
            _contract_check(
                f"state_alias:{alias_group}",
                bool(values) and all(value is values[0] for value in values[1:]),
                RunnableErrorCode.STATE_ALIAS_CONFLICT,
                f"State alias group {alias_group!r} did not retain one shared tensor.",
                details=(
                    ("alias_group", alias_group),
                    ("slot_ids", repr(tuple(slot.slot_id for slot in members))),
                ),
            )
        )
    return tuple(checks)


def _allowed_state_roles(
    leaf_name: str,
    slot_role: TensorSlotRole,
) -> frozenset[StateSlotRole]:
    """Return canonical semantic roles for a state-dict leaf name."""

    if leaf_name == "weight":
        return frozenset({StateSlotRole.WEIGHT, StateSlotRole.NORM_SCALE})
    if leaf_name == "bias":
        return frozenset({StateSlotRole.BIAS, StateSlotRole.NORM_OFFSET})
    if leaf_name == "running_mean":
        return frozenset({StateSlotRole.RUNNING_MEAN})
    if leaf_name == "running_var":
        return frozenset({StateSlotRole.RUNNING_VAR})
    if leaf_name in {"num_batches_tracked", "counter"}:
        return frozenset({StateSlotRole.COUNTER})
    if slot_role is TensorSlotRole.BUFFER:
        return frozenset({StateSlotRole.GENERIC_BUFFER})
    return frozenset({StateSlotRole.WEIGHT})


def _pre_call_contract_checks(
    descriptor: SparseRunDescriptor,
    call: RunnableCallDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> tuple[tuple[ContractCheck, ...], dict[str, int]]:
    """Validate callable dispatch/arity metadata and snapshot input versions."""

    registry_entry = next(
        (entry for entry in descriptor.callable_registry if entry.registry_id == call.registry_id),
        None,
    )
    valid_dispatch = (
        call.dispatch_kind in {"function", "method", "dunder", "namespace_alias"}
        and registry_entry is not None
        and registry_entry.key.dispatch_kind == call.dispatch_kind
    )
    referenced_paths = [argument.argument_path for argument in call.tensor_arguments] + [
        argument.argument_path for argument in call.literal_arguments
    ]
    positional_indices = {
        cast(int, path[1])
        for path in referenced_paths
        if len(path) >= 2 and path[0] == "args" and isinstance(path[1], int)
    }
    keyword_names = {
        cast(str, path[1])
        for path in referenced_paths
        if len(path) >= 2 and path[0] == "kwargs" and isinstance(path[1], str)
    }
    arity_ok = positional_indices == set(range(call.num_positional_args)) and (
        len(keyword_names) == call.num_keyword_args
    )
    checks = (
        _contract_check(
            f"call_dispatch:{call.call_id}",
            valid_dispatch,
            RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
            f"Call {call.call_id!r} has an unsupported dispatch contract.",
            affected_op_labels=call.op_labels,
            details=(("dispatch_kind", call.dispatch_kind),),
        ),
        _contract_check(
            f"call_arity:{call.call_id}",
            arity_ok,
            RunnableErrorCode.CALL_ARITY_MISMATCH,
            f"Call {call.call_id!r} argument leaves do not satisfy its recorded arity.",
            affected_op_labels=call.op_labels,
            details=(
                ("expected_positional", str(call.num_positional_args)),
                ("actual_positional_sites", repr(sorted(positional_indices))),
                ("expected_keyword", str(call.num_keyword_args)),
                ("actual_keyword_names", repr(sorted(keyword_names))),
            ),
        ),
    )
    versions = {
        argument.slot_id: slot_values[argument.slot_id]._version
        for argument in call.tensor_arguments
        if argument.slot_id in slot_values
    }
    return checks, versions


def _execute_sparse_call(
    call: RunnableCallDescriptor,
    func: Callable[..., Any],
    slot_values: Mapping[str, torch.Tensor],
) -> Any:
    """Construct and execute one sparse call from literal and tensor leaves."""

    args: list[Any] = [None] * call.num_positional_args
    kwargs: dict[str, Any] = {}
    for literal_argument in call.literal_arguments:
        _write_argument(
            args,
            kwargs,
            literal_argument.argument_path,
            _decode_literal(literal_argument.value),
        )
    for tensor_argument in call.tensor_arguments:
        try:
            value = slot_values[tensor_argument.slot_id]
        except KeyError as exc:
            raise RunPreconditionError(
                f"Sparse call {call.call_id!r} references unavailable slot "
                f"{tensor_argument.slot_id!r}.",
                code=RunnableErrorCode.MISSING_TENSOR_SLOT.value,
                call_id=call.call_id,
                slot_id=tensor_argument.slot_id,
            ) from exc
        _write_argument(args, kwargs, tensor_argument.argument_path, value)
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        raise RuntimeSignatureDriftError(
            f"Resolved callable rejected sparse recipe for {call.call_id!r}: {exc}",
            code=RunnableErrorCode.RUNTIME_SIGNATURE_DRIFT.value,
            call_id=call.call_id,
            affected_op_labels=call.op_labels,
        ) from exc


def _populate_source_slots(
    fork: Any,
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> None:
    """Populate input and buffer source Ops on the transactional run fork."""

    for slot in descriptor.tensor_slots:
        if slot.role not in {TensorSlotRole.MODEL_INPUT, TensorSlotRole.BUFFER}:
            continue
        value = slot_values.get(slot.slot_id)
        op = _op_for_slot(fork, slot.slot_id)
        if value is not None and op is not None:
            op._internal_set("out", value.detach().clone())


def _write_argument(
    args: list[Any], kwargs: dict[str, Any], path: tuple[str | int, ...], value: Any
) -> None:
    """Write one reconstructed value at an args/kwargs argument path."""

    if len(path) < 2 or path[0] not in {"args", "kwargs"}:
        raise RunPreconditionError(
            f"Invalid sparse argument path {path!r}.",
            code=RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value,
        )
    root: Any = args if path[0] == "args" else kwargs
    _write_path(root, path[1:], value)


def _write_path(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    """Write a value into a dynamically reconstructed list/dict tree."""

    current = root
    for index, component in enumerate(path):
        last = index == len(path) - 1
        if last:
            current[component] = value
            return
        next_component = path[index + 1]
        if isinstance(current, list):
            child = current[cast(int, component)]
            if child is None:
                child = [] if isinstance(next_component, int) else {}
                current[cast(int, component)] = child
        else:
            child = current.get(component)
            if child is None:
                child = [] if isinstance(next_component, int) else {}
                current[component] = child
        current = child


def _bind_call_outputs(
    descriptor: SparseRunDescriptor,
    call: RunnableCallDescriptor,
    output: Any,
    slot_values: dict[str, torch.Tensor],
    fork: Any,
    *,
    before_versions: Mapping[str, int],
    attestation_slot_ids: frozenset[str],
    attestation_slot_values: dict[str, torch.Tensor],
) -> tuple[ContractCheck, ...]:
    """Slice, validate, and stage one grouped call's tensor outputs."""

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    checks: list[ContractCheck] = []
    expected_paths = tuple(slots[slot_id].output_path or () for slot_id in call.output_slot_ids)
    actual_paths = _tensor_leaf_paths(output)
    checks.append(
        _contract_check(
            f"output_structure:{call.call_id}",
            len(call.output_slot_ids) == len(call.op_labels)
            and tuple(actual_paths) == expected_paths,
            RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
            f"Call {call.call_id!r} output tensor paths disagree with the recorded container.",
            affected_op_labels=call.op_labels,
            details=(
                ("expected_paths", repr(expected_paths)),
                ("actual_paths", repr(tuple(actual_paths))),
            ),
        )
    )
    for slot_id, op_label in zip(call.output_slot_ids, call.op_labels):
        slot = slots[slot_id]
        try:
            value = _value_at_path(output, slot.output_path or ())
        except (KeyError, IndexError, TypeError) as exc:
            checks.append(
                _contract_check(
                    f"slot_production:{slot_id}",
                    False,
                    RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                    f"Call {call.call_id!r} output lacks path {slot.output_path!r}: {exc}",
                    affected_op_labels=call.op_labels,
                    details=(("slot_id", slot_id),),
                )
            )
            continue
        if not isinstance(value, torch.Tensor):
            checks.append(
                _contract_check(
                    f"slot_production:{slot_id}",
                    False,
                    RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                    f"Call {call.call_id!r} output is not a tensor at {slot.output_path!r}.",
                    affected_op_labels=call.op_labels,
                    details=(("slot_id", slot_id),),
                )
            )
            continue
        slot_values[slot_id] = value
        produced_slot_ids = {slot_id}
        for version in descriptor.tensor_slots:
            if version.version_of == slot_id and version.producer_slot_id == slot_id:
                slot_values[version.slot_id] = value
                produced_slot_ids.add(version.slot_id)
        for produced_slot_id in produced_slot_ids & attestation_slot_ids:
            attestation_slot_values[produced_slot_id] = value.detach().clone()
        op = _op_for_label(fork, op_label)
        if op is not None:
            op._internal_set("out", value.detach().clone())
        shape_ok = tuple(value.shape) == slot.shape
        dtype_ok = str(value.dtype) == slot.dtype
        checks.append(
            _contract_check(
                f"slot_production:{slot_id}",
                True,
                RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                f"Call {call.call_id!r} did not produce slot {slot_id!r}.",
                affected_op_labels=(op_label,),
            )
        )
        checks.append(
            _contract_check(
                f"output_shape:{slot_id}",
                shape_ok,
                RunnableErrorCode.OUTPUT_SHAPE_MISMATCH,
                f"Output slot {slot_id!r} has shape {tuple(value.shape)}, expected {slot.shape}.",
                affected_op_labels=(op_label,),
                details=(
                    ("slot_id", slot_id),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(tuple(value.shape))),
                ),
            )
        )
        checks.append(
            _contract_check(
                f"output_dtype:{slot_id}",
                dtype_ok,
                RunnableErrorCode.OUTPUT_DTYPE_MISMATCH,
                f"Output slot {slot_id!r} has dtype {value.dtype}, expected {slot.dtype}.",
                affected_op_labels=(op_label,),
                details=(
                    ("slot_id", slot_id),
                    ("expected_dtype", slot.dtype),
                    ("actual_dtype", str(value.dtype)),
                ),
            )
        )
    checks.extend(_mutation_contract_checks(call, output, slot_values, before_versions))
    return tuple(checks)


def _mutation_contract_checks(
    call: RunnableCallDescriptor,
    output: Any,
    slot_values: Mapping[str, torch.Tensor],
    before_versions: Mapping[str, int],
) -> tuple[ContractCheck, ...]:
    """Validate recorded in-place aliasing and tensor-version expectations."""

    changed = {
        slot_id
        for slot_id, before in before_versions.items()
        if slot_id in slot_values and slot_values[slot_id]._version != before
    }
    if not call.is_inplace:
        return (
            _contract_check(
                f"mutation:{call.call_id}",
                not changed,
                RunnableErrorCode.MUTATION_VERSION_MISMATCH,
                f"Non-mutating call {call.call_id!r} changed an input tensor version.",
                affected_op_labels=call.op_labels,
                details=(("changed_slot_ids", repr(tuple(sorted(changed)))),),
            ),
        )
    if not call.tensor_arguments or not call.output_slot_ids:
        return (
            _contract_check(
                f"mutation:{call.call_id}",
                False,
                RunnableErrorCode.MUTATION_VERSION_MISMATCH,
                f"In-place call {call.call_id!r} lacks an input/output version relation.",
                affected_op_labels=call.op_labels,
            ),
        )
    input_slot_id = call.tensor_arguments[0].slot_id
    input_value = slot_values.get(input_slot_id)
    try:
        output_value = _value_at_path(output, ()) if len(call.output_slot_ids) == 1 else output
        if len(call.output_slot_ids) > 1:
            output_value = _value_at_path(output, (0,))
    except (KeyError, IndexError, TypeError):
        output_value = None
    aliases = (
        isinstance(input_value, torch.Tensor)
        and isinstance(output_value, torch.Tensor)
        and (input_value is output_value or input_value.data_ptr() == output_value.data_ptr())
    )
    return (
        _contract_check(
            f"mutation:{call.call_id}",
            input_slot_id in changed and aliases,
            RunnableErrorCode.MUTATION_VERSION_MISMATCH,
            f"In-place call {call.call_id!r} violated its alias/version expectation.",
            affected_op_labels=call.op_labels,
            details=(
                ("input_slot_id", input_slot_id),
                ("version_changed", repr(input_slot_id in changed)),
                ("output_aliases_input", repr(aliases)),
            ),
        ),
    )


def _reconstruct_output(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    fork: Any,
) -> Any:
    """Reconstruct the model-output container and populate synthetic output Ops."""

    output_slots = tuple(
        slot for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.OUTPUT
    )
    values: list[tuple[tuple[str | int, ...], torch.Tensor]] = []
    for slot in output_slots:
        source_id = slot.producer_slot_id or slot.version_of
        if source_id is None or source_id not in slot_values:
            raise RunPreconditionError(
                f"Output slot {slot.slot_id!r} has no produced source slot.",
                code=RunnableErrorCode.SLOT_PRODUCTION_MISMATCH.value,
                slot_id=slot.slot_id,
            )
        value = slot_values[source_id]
        slot_values_dict = cast(dict[str, torch.Tensor], slot_values)
        slot_values_dict[slot.slot_id] = value
        op = _op_for_slot(fork, slot.slot_id)
        if op is not None:
            op._internal_set("out", value.detach().clone())
        values.append((slot.output_path or (), value))
    container_spec = _output_container_spec(fork)
    if container_spec is not None:
        try:
            return rebuild_container_from_spec(container_spec, [value for _, value in values])
        except ValueError as exc:
            raise RunPreconditionError(
                f"Recorded output container could not be reconstructed: {exc}",
                code=RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH.value,
            ) from exc
    return _container_from_paths(values)


def _output_container_spec(trace: Any) -> ContainerSpec | None:
    """Return the shared recorded model-output container specification."""

    for label in getattr(trace, "output_layers", ()):
        op = _op_for_label(trace, label)
        spec = getattr(op, "container_spec", None)
        if isinstance(spec, ContainerSpec):
            return spec
    return None


def _reconstruct_live_output(trace: Any) -> Any:
    """Reconstruct refreshed live output from synthetic output-node payloads."""

    values = [
        (tuple(getattr(trace[label], "container_path", ()) or ()), trace[label].out)
        for label in trace.output_layers
    ]
    return _container_from_paths(values)


def _container_from_paths(values: Sequence[tuple[tuple[str | int, ...], Any]]) -> Any:
    """Build a conservative tuple/dict output container from leaf paths."""

    if len(values) == 1 and not values[0][0]:
        return values[0][1]
    if not values:
        return None
    paths = [path for path, _ in values]
    root: Any = [] if all(path and isinstance(path[0], int) for path in paths) else {}
    if isinstance(root, list):
        root.extend([None] * (max(cast(int, path[0]) for path in paths) + 1))
    for path, value in values:
        _write_output_path(root, path, value)
    return tuple(root) if isinstance(root, list) else root


def _write_output_path(root: Any, path: tuple[str | int, ...], value: Any) -> None:
    """Write one output leaf, growing positional containers as needed."""

    if not path:
        raise RunPreconditionError(
            "Multiple output leaves cannot share an empty container path.",
            code=RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH.value,
        )
    current = root
    for index, component in enumerate(path):
        last = index == len(path) - 1
        if isinstance(current, list):
            position = cast(int, component)
            while len(current) <= position:
                current.append(None)
            if last:
                current[position] = value
                return
            if current[position] is None:
                current[position] = [] if isinstance(path[index + 1], int) else {}
            current = current[position]
        else:
            if last:
                current[component] = value
                return
            current = current.setdefault(component, [] if isinstance(path[index + 1], int) else {})


def _call_witness_checks(
    descriptor: SparseRunDescriptor,
    call: RunnableCallDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> tuple[ContractCheck, ...]:
    """Compare scalar-bool and loop witnesses immediately after their call."""

    checks: list[ContractCheck] = []
    for witness in sorted(descriptor.control_witnesses, key=lambda item: item.order):
        if witness.call_id != call.call_id or witness.kind not in {
            ControlWitnessKind.SCALAR_BOOL,
            ControlWitnessKind.LOOP_PREDICATE,
        }:
            continue
        values = [
            slot_values[slot_id] for slot_id in call.output_slot_ids if slot_id in slot_values
        ]
        expected = bool(_decode_literal(witness.observed_value))
        scalar = values[0] if values else None
        actual: bool | None = None
        if isinstance(scalar, torch.Tensor) and scalar.numel() == 1:
            actual = bool(scalar.item())
        code = (
            RunnableErrorCode.LOOP_PREDICATE_DIVERGENCE
            if witness.kind is ControlWitnessKind.LOOP_PREDICATE
            else RunnableErrorCode.SCALAR_BOOL_DIVERGENCE
        )
        checks.append(
            _contract_check(
                f"control_witness:{witness.witness_id}",
                actual is not None and actual == expected,
                code,
                f"Control witness {witness.witness_id!r} disagreed with the recorded path.",
                affected_op_labels=(witness.site_label,),
                details=(
                    ("witness_id", witness.witness_id),
                    ("expected", repr(expected)),
                    ("actual", repr(actual)),
                    ("order", str(witness.order)),
                ),
            )
        )
    return tuple(checks)


def _post_execution_contract_checks(
    descriptor: SparseRunDescriptor,
    *,
    inputs: Any,
    output: Any,
    call_outputs: Mapping[str, Any],
    slot_values: Mapping[str, torch.Tensor],
    fork: Any,
) -> tuple[ContractCheck, ...]:
    """Validate final slot production, arm identity, and structure witnesses."""

    checks: list[ContractCheck] = []
    for call in descriptor.calls:
        missing = tuple(slot_id for slot_id in call.output_slot_ids if slot_id not in slot_values)
        checks.append(
            _contract_check(
                f"call_slot_production:{call.call_id}",
                not missing,
                RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                f"Call {call.call_id!r} did not produce every recorded output slot.",
                affected_op_labels=call.op_labels,
                details=(("missing_slot_ids", repr(missing)),),
            )
        )
    for witness in sorted(descriptor.control_witnesses, key=lambda item: item.order):
        if witness.kind is ControlWitnessKind.CONDITIONAL_ARM_ENTRY:
            checks.append(_conditional_arm_check(witness, fork))
        elif witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            checks.append(
                _structure_witness_check(
                    witness,
                    descriptor,
                    inputs=inputs,
                    output=output,
                    call_outputs=call_outputs,
                )
            )
        elif (
            witness.kind
            in {
                ControlWitnessKind.SCALAR_BOOL,
                ControlWitnessKind.LOOP_PREDICATE,
            }
            and witness.call_id is None
        ):
            checks.append(
                _contract_check(
                    f"control_witness:{witness.witness_id}",
                    False,
                    RunnableErrorCode.SLOT_PRODUCTION_MISMATCH,
                    f"Control witness {witness.witness_id!r} has no recomputable call.",
                    affected_op_labels=(witness.site_label,),
                )
            )
    return tuple(checks)


def _conditional_arm_check(witness: ControlWitness, fork: Any) -> ContractCheck:
    """Validate that one recorded conditional arm-entry edge was produced."""

    edge_text = witness.site_label.rsplit(":", 1)[-1]
    parent, separator, child = edge_text.partition("->")
    parent_op = _op_for_label(fork, parent) if separator else None
    child_op = _op_for_label(fork, child) if separator else None
    passed = (
        parent_op is not None
        and child_op is not None
        and isinstance(getattr(parent_op, "out", None), torch.Tensor)
        and isinstance(getattr(child_op, "out", None), torch.Tensor)
    )
    affected = tuple(label for label in (parent, child) if label)
    return _contract_check(
        f"control_witness:{witness.witness_id}",
        passed,
        RunnableErrorCode.CONDITIONAL_ARM_DIVERGENCE,
        f"Conditional arm witness {witness.witness_id!r} did not enter its recorded edge.",
        affected_op_labels=affected or (witness.site_label,),
        details=(("recorded_edge", edge_text), ("order", str(witness.order))),
    )


def _structure_witness_check(
    witness: ControlWitness,
    descriptor: SparseRunDescriptor,
    *,
    inputs: Any,
    output: Any,
    call_outputs: Mapping[str, Any],
) -> ContractCheck:
    """Compare a model-boundary container witness with runtime structure."""

    expected = _decode_literal(witness.observed_value)
    role = expected.get("role")
    if role == "model_input":
        runtime_value = _runtime_input_for_structure_witness(descriptor, expected, inputs)
        expected_paths = tuple(tuple(path) for path in expected.get("leaf_paths", ()))
        expected_kind = str(expected.get("kind", "unknown"))
    else:
        runtime_value = _raw_runtime_output(descriptor, output, call_outputs)
        expected_paths = tuple(_container_leaf_paths(output))
        expected_kind = _container_kind(output)
    actual_paths = tuple(_container_leaf_paths(runtime_value))
    actual_kind = _container_kind(runtime_value)
    kind_matches = expected_kind in {"unknown", actual_kind}
    passed = expected_paths == actual_paths and kind_matches
    return _contract_check(
        f"control_witness:{witness.witness_id}",
        passed,
        RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
        f"Structure witness {witness.witness_id!r} disagreed with the runtime container.",
        affected_op_labels=(witness.site_label,),
        details=(
            ("role", repr(role)),
            ("expected_kind", expected_kind),
            ("actual_kind", actual_kind),
            ("expected_paths", repr(expected_paths)),
            ("actual_paths", repr(actual_paths)),
        ),
    )


def _runtime_input_for_structure_witness(
    descriptor: SparseRunDescriptor,
    expected: Mapping[str, Any],
    inputs: Any,
) -> Any:
    """Select the runtime input site named by a container structure witness."""

    record_id = expected.get("record_id")
    bindings = [
        slot.input_binding
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.MODEL_INPUT
        and slot.input_binding is not None
        and slot.input_binding.container_record_id == record_id
    ]
    positions = {binding.model_site_position for binding in bindings}
    if len(positions) == 1:
        return _input_site_value(inputs, next(iter(positions)), positions)
    return inputs


def _raw_runtime_output(
    descriptor: SparseRunDescriptor,
    reconstructed_output: Any,
    call_outputs: Mapping[str, Any],
) -> Any:
    """Return the raw final call container when one call owns every model output."""

    output_sources = {
        slot.producer_slot_id or slot.version_of
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.OUTPUT
    }
    owner_ids = {
        call.call_id
        for call in descriptor.calls
        if output_sources and output_sources.issubset(set(call.output_slot_ids))
    }
    if len(owner_ids) == 1:
        return call_outputs.get(next(iter(owner_ids)), reconstructed_output)
    return reconstructed_output


def _tensor_leaf_paths(
    value: Any, path: tuple[str | int, ...] = ()
) -> tuple[tuple[str | int, ...], ...]:
    """Return ordered tensor-leaf paths for a runtime container."""

    if isinstance(value, torch.Tensor):
        return (path,)
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        paths: list[tuple[str | int, ...]] = []
        for name in value._fields:
            paths.extend(_tensor_leaf_paths(getattr(value, name), (*path, str(name))))
        return tuple(paths)
    if isinstance(value, Mapping):
        paths = []
        for key, child in value.items():
            if isinstance(key, (str, int)):
                paths.extend(_tensor_leaf_paths(child, (*path, key)))
        return tuple(paths)
    if isinstance(value, (list, tuple)):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_tensor_leaf_paths(child, (*path, index)))
        return tuple(paths)
    return ()


def _container_leaf_paths(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str | int, ...], ...]:
    """Return ordered paths for every leaf in a runtime boundary container."""

    if isinstance(value, tuple) and hasattr(value, "_fields"):
        paths: list[tuple[str | int, ...]] = []
        for name in value._fields:
            paths.extend(_container_leaf_paths(getattr(value, name), (*path, str(name))))
        return tuple(paths)
    if isinstance(value, Mapping):
        paths = []
        for key, child in value.items():
            if isinstance(key, (str, int)):
                paths.extend(_container_leaf_paths(child, (*path, key)))
        return tuple(paths)
    if isinstance(value, (list, tuple)):
        paths = []
        for index, child in enumerate(value):
            paths.extend(_container_leaf_paths(child, (*path, index)))
        return tuple(paths)
    return (path,)


def _container_kind(value: Any) -> str:
    """Return the sparse witness vocabulary name for a runtime container."""

    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return "namedtuple"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    return type(value).__name__


def _path_faithfulness(
    descriptor: SparseRunDescriptor,
    checks: Sequence[ContractCheck],
) -> tuple[PathFaithfulness, RunnableDiagnostic | None]:
    """Classify exact three-state path faithfulness after all honesty checks."""

    failed = next((check for check in checks if not check.passed), None)
    if failed is not None:
        return PathFaithfulness.DIVERGED, failed.diagnostic
    if descriptor.witness_completeness is not WitnessCompleteness.COMPLETE:
        return PathFaithfulness.UNVERIFIABLE, None
    return PathFaithfulness.VERIFIED, None


def _run_report(
    readiness: ReadinessReport,
    state: PreparedRunnableState,
    *,
    contract_checks: tuple[ContractCheck, ...],
    path_faithfulness: PathFaithfulness,
    first_mismatch: RunnableDiagnostic | None,
    numeric_attestation: NumericAttestationStatus,
) -> RunReport:
    """Build the settled sparse run-report surface."""

    return RunReport(
        readiness=readiness,
        state_source=state.state_source,
        initializer_policy_version=state.initializer_policy_version,
        seed=state.seed,
        random_filled_slot_ids=state.random_filled_slot_ids,
        contract_checks=contract_checks,
        path_faithfulness=path_faithfulness,
        first_mismatch=first_mismatch,
        numeric_attestation=numeric_attestation,
        poisoned=path_faithfulness is not PathFaithfulness.VERIFIED,
    )


def _numeric_attestation_check(
    descriptor: SparseRunDescriptor,
    state: PreparedRunnableState,
    *,
    slot_values: Mapping[str, torch.Tensor],
    attestation_slot_values: Mapping[str, torch.Tensor],
    input_byte_digests: Mapping[str, str],
    trace: Any,
) -> tuple[NumericAttestationStatus, ContractCheck | None]:
    """Compare recomputed saved slots with the independent activation archive.

    Parameters
    ----------
    descriptor:
        Runnable descriptor declaring activation membership and eligibility digests.
    state:
        Bound state used by this run.
    slot_values:
        Fresh scheduler outputs and source slots from this transaction.
    attestation_slot_values:
        Tensor snapshots taken when selected internal slots were produced,
        before any later in-place call can mutate their storage.
    input_byte_digests:
        Model-input byte digests captured before any in-place sparse call.
    trace:
        Loaded source Trace retaining inspection-only archived activations.

    Returns
    -------
    tuple[NumericAttestationStatus, ContractCheck | None]
        Applicability/result status and the aggregate byte-exact tripwire check.
    """

    layer = descriptor.payload_layers.activations
    if not layer.present:
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if not isinstance(layer, ActivationPayloadLayerDescriptor):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if _descriptor_has_nondeterministic_rng(descriptor):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    # Sparse execution recomputes raw output slots, never activation transforms.
    # An archive containing transformed outputs is therefore outside the scope
    # of the all-selected-activations byte-exact claim.
    if any(member.field != "out" for member in layer.members):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    raw_members = layer.members
    if not raw_members or not _attestation_inputs_match(descriptor, layer, input_byte_digests):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if not _attestation_state_matches(descriptor, layer, state, slot_values):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    archived = trace.__dict__.get("_runnable_archived_activations")
    if not isinstance(archived, Mapping):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    for member in raw_members:
        archive_key = f"{member.slot_id}:{member.field}"
        archived_record = archived.get(archive_key)
        recomputed = attestation_slot_values.get(member.slot_id, slot_values.get(member.slot_id))
        archived_value = getattr(archived_record, "value", None)
        if member.slot_id in input_byte_digests:
            recomputed_digest = input_byte_digests[member.slot_id]
        else:
            recomputed_digest = (
                runnable_tensor_byte_digest(recomputed)
                if isinstance(recomputed, torch.Tensor)
                else "missing"
            )
        archived_digest = (
            runnable_tensor_byte_digest(archived_value)
            if isinstance(archived_value, torch.Tensor)
            else "missing"
        )
        passed = recomputed_digest == member.byte_digest and archived_digest == member.byte_digest
        if not passed:
            return (
                NumericAttestationStatus.NUMERIC_ATTESTATION_FAILED,
                _contract_check(
                    f"numeric_attestation:{member.slot_id}",
                    False,
                    RunnableErrorCode.NUMERIC_ATTESTATION_FAILED,
                    f"Byte-exact numeric attestation failed for {member.slot_id!r}.",
                    affected_op_labels=(member.op_label,),
                    details=(
                        ("slot_id", member.slot_id),
                        ("call_id", repr(member.call_id)),
                        ("field", member.field),
                        ("expected_digest", member.byte_digest),
                        ("archived_digest", archived_digest),
                        ("recomputed_digest", recomputed_digest),
                    ),
                ),
            )
    return (
        NumericAttestationStatus.ATTESTED,
        ContractCheck(name="numeric_attestation:selected_slots", passed=True, diagnostic=None),
    )


def _raw_activation_slot_ids(descriptor: SparseRunDescriptor) -> frozenset[str]:
    """Return raw selected-activation slots that require production snapshots.

    Parameters
    ----------
    descriptor:
        Sparse runnable descriptor carrying optional activation archive metadata.

    Returns
    -------
    frozenset[str]
        Slot IDs whose raw output values must be copied at production time.
    """

    layer = descriptor.payload_layers.activations
    if not isinstance(layer, ActivationPayloadLayerDescriptor):
        return frozenset()
    return frozenset(member.slot_id for member in layer.members if member.field == "out")


def _descriptor_has_nondeterministic_rng(descriptor: SparseRunDescriptor) -> bool:
    """Return whether replay contains a PyTorch seeded-RNG operation.

    Parameters
    ----------
    descriptor:
        Sparse runnable descriptor whose registry entries identify replayed
        PyTorch operations.

    Returns
    -------
    bool
        Whether replay includes a captured RNG source or an ATen operation
        marked by PyTorch as ``nondeterministic_seeded``.
    """

    if any(slot.role is TensorSlotRole.RNG_SOURCE for slot in descriptor.tensor_slots):
        return True
    return any(_registry_entry_has_seeded_rng_tag(entry) for entry in descriptor.callable_registry)


def _registry_entry_has_seeded_rng_tag(entry: CallableRegistryEntry) -> bool:
    """Return whether one portable callable maps to a seeded ATen RNG op.

    Parameters
    ----------
    entry:
        Portable callable identity recorded in a runnable descriptor.

    Returns
    -------
    bool
        Whether any matching ATen overload has PyTorch's maintained
        ``nondeterministic_seeded`` tag.
    """

    if entry.key.namespace not in {"torch", "torch.Tensor", "torch.nn.functional"}:
        return False
    name = entry.key.qualname.rsplit(".", 1)[-1]
    candidate_names = (name, name[:-1]) if name.endswith("_") else (name,)
    for candidate_name in candidate_names:
        packet = getattr(torch.ops.aten, candidate_name, None)
        overloads = getattr(packet, "overloads", None)
        if not callable(overloads):
            continue
        for overload_name in overloads():
            overload = getattr(packet, overload_name)
            if torch.Tag.nondeterministic_seeded in getattr(overload, "tags", ()):
                return True
    return False


def _attestation_inputs_match(
    descriptor: SparseRunDescriptor,
    layer: Any,
    input_byte_digests: Mapping[str, str],
) -> bool:
    """Return whether every available capture-input digest matches this run."""

    expected_slot_ids = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    }
    observed_slot_ids = {digest.slot_id for digest in layer.original_input_digests}
    if not expected_slot_ids or observed_slot_ids != expected_slot_ids:
        return False
    return all(
        input_byte_digests.get(digest.slot_id) == digest.byte_digest
        for digest in layer.original_input_digests
    )


def _attestation_state_matches(
    descriptor: SparseRunDescriptor,
    layer: Any,
    state: PreparedRunnableState,
    slot_values: Mapping[str, torch.Tensor],
) -> bool:
    """Return whether runtime state is capture-equivalent rather than random."""

    state_slots = {
        slot.state_binding.state_dict_name: slot
        for slot in descriptor.tensor_slots
        if slot.state_binding is not None
    }
    if not state_slots:
        return True
    if state.state_source not in {
        StateSource.EMBEDDED_CAPTURE_STATE,
        StateSource.USER_STATE_DICT,
    }:
        return False
    expected = {item.state_dict_name: item.byte_digest for item in layer.capture_state_digests}
    if set(expected) != set(state_slots):
        return False
    return all(
        slot.slot_id in slot_values
        and runnable_tensor_byte_digest(slot_values[slot.slot_id]) == expected[name]
        for name, slot in state_slots.items()
    )


def _raise_numeric_attestation_failure(fork: Any, check: ContractCheck) -> None:
    """Rollback and raise the mandatory saved-activation mismatch tripwire."""

    _state._unregister_log(fork)
    diagnostic = check.diagnostic
    raise NumericAttestationError(
        diagnostic.message if diagnostic is not None else "Numeric attestation failed.",
        code=RunnableErrorCode.NUMERIC_ATTESTATION_FAILED.value,
        path_faithfulness=PathFaithfulness.DIVERGED,
        numeric_attestation=NumericAttestationStatus.NUMERIC_ATTESTATION_FAILED,
        first_mismatch=diagnostic,
        contract_check=check,
    )


def _contract_check(
    name: str,
    passed: bool,
    code: RunnableErrorCode,
    message: str,
    *,
    affected_op_labels: tuple[str, ...] = (),
    details: tuple[tuple[str, str], ...] = (),
) -> ContractCheck:
    """Build one honesty check with a diagnostic only on contradiction."""

    diagnostic = None
    if not passed:
        diagnostic = RunnableDiagnostic(
            code=code,
            message=message,
            registry_id=None,
            affected_op_labels=affected_op_labels,
            recorded_runtime=None,
            current_runtime=str(torch.__version__),
            detection_stage="run_honesty_contract",
            resolver_provenance=None,
            analysis_load_available=True,
            details=details,
        )
    return ContractCheck(name=name, passed=passed, diagnostic=diagnostic)


def _raise_first_divergence(
    checks: Sequence[ContractCheck],
    policy: DivergencePolicy,
    *,
    fork: Any | None,
) -> None:
    """Raise and discard transactional state at the first observed contradiction."""

    failed = next((check for check in checks if not check.passed), None)
    if failed is None or policy is DivergencePolicy.RETURN_DIVERGED:
        return
    if fork is not None:
        _state._unregister_log(fork)
    diagnostic = failed.diagnostic
    raise PathDivergenceError(
        diagnostic.message if diagnostic is not None else "Sparse run path diverged.",
        code=(
            diagnostic.code.value
            if diagnostic is not None
            else RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value
        ),
        path_faithfulness=PathFaithfulness.DIVERGED,
        first_mismatch=diagnostic,
        contract_check=failed,
    )


def _raise_monotonic_divergence(
    fork: Any,
    status: PathFaithfulness,
    mismatch: RunnableDiagnostic | None,
    policy: DivergencePolicy,
) -> None:
    """Enforce strict policy for an inherited monotonic divergence mark."""

    if status is not PathFaithfulness.DIVERGED or policy is DivergencePolicy.RETURN_DIVERGED:
        return
    _state._unregister_log(fork)
    raise PathDivergenceError(
        "Sparse run Trace retains a prior path divergence and cannot become faithful.",
        code=(
            mismatch.code.value
            if mismatch is not None
            else RunnableErrorCode.POISONED_RUN_REFUSED.value
        ),
        path_faithfulness=status,
        first_mismatch=mismatch,
    )


def _decode_literal(value: NonTensorLiteral | LiteralTupleKey) -> Any:
    """Decode one safe sparse literal without importing artifact-selected code."""

    if isinstance(value, LiteralAtom):
        return value.value
    if isinstance(value, LiteralTupleKey):
        return tuple(_decode_literal(item) for item in value.items)
    if isinstance(value, LiteralSequence):
        items = [_decode_literal(item) for item in value.items]
        return tuple(items) if value.kind is LiteralSequenceKind.TUPLE else items
    if isinstance(value, LiteralMapping):
        return {_decode_literal(entry.key): _decode_literal(entry.value) for entry in value.entries}
    if isinstance(value, LiteralTorchSymbol):
        if value.qualname.startswith("torch.device(") and value.qualname.endswith(")"):
            return torch.device(value.qualname[13:-1])
        name = value.qualname.removeprefix("torch.")
        symbol = getattr(torch, name, None)
        if symbol is None or callable(symbol):
            raise RunPreconditionError(
                f"Unsupported torch literal symbol {value.qualname!r}.",
                code=RunnableErrorCode.UNSUPPORTED_LITERAL.value,
            )
        return symbol
    raise TypeError(f"Unknown sparse literal type {type(value).__name__}.")


def _value_at_path(value: Any, path: Sequence[str | int]) -> Any:
    """Read one list/tuple/mapping/object path from a runtime value."""

    current = value
    for component in path:
        if isinstance(current, Mapping):
            current = current[component]
        elif isinstance(component, int):
            current = current[component]
        else:
            current = getattr(current, component)
    return current


def _input_error(
    code: RunnableErrorCode,
    slot: TensorSlotDescriptor,
    message: str,
) -> RunPreconditionError:
    """Build a typed input-binding precondition error."""

    return RunPreconditionError(message, code=code.value, slot_id=slot.slot_id)


def _op_for_label(trace: Any, label: str) -> Any | None:
    """Resolve a descriptor op label against a fork's lookup aliases."""

    layer_dict = getattr(trace, "layer_dict_all_keys", {}) or {}
    if label in layer_dict:
        return layer_dict[label]
    return next(
        (op for op in getattr(trace, "layer_list", ()) if str(getattr(op, "label", "")) == label),
        None,
    )


def _op_for_slot(trace: Any, slot_id: str) -> Any | None:
    """Resolve the cooked Op named by a ``slot:<label>`` descriptor ID."""

    return _op_for_label(trace, slot_id.removeprefix("slot:"))


def _run_fork_name(trace: Any) -> str:
    """Return a process-unique monotonic Trace label for a run transaction."""

    base_name = trace.trace_label or "trace"
    return f"{base_name}_fork_{next(_RUN_FORK_COUNTER)}"


__all__ = ["raise_analysis_run_unavailable", "run_live_trace", "run_loaded_sparse_trace"]
