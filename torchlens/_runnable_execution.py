"""Transactional execution providers for the unified :meth:`Trace.run` surface."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from typing import Any, cast

import torch

from . import _state
from ._runnable_state import PreparedRunnableState, prepare_runnable_state
from .errors import (
    ReattachError,
    RunCapabilityUnavailableError,
    RunPreconditionError,
    RuntimeSignatureDriftError,
)
from .runnable import (
    ContractCheck,
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
    StateSource,
    TensorSlotDescriptor,
    TensorSlotRole,
    WitnessCompleteness,
)


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

    del on_divergence
    descriptor, readiness, callables = _require_loaded_sparse_provider(trace)
    prepared_state = prepare_runnable_state(trace, seed=seed)
    slot_values, input_checks = _bind_runtime_inputs(descriptor, inputs)
    slot_values.update(_clone_state_values(prepared_state.slot_values))
    fork = trace._fork_trace(name=_run_fork_name(trace))
    _populate_source_slots(fork, descriptor, slot_values)
    contract_checks: list[ContractCheck] = list(input_checks)

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
        for call in descriptor.calls:
            output = _execute_sparse_call(call, callables[call.call_id], slot_values)
            contract_checks.extend(_bind_call_outputs(descriptor, call, output, slot_values, fork))

    output = _reconstruct_output(descriptor, slot_values, fork)
    path_faithfulness, mismatch = _path_faithfulness(
        descriptor,
        slot_values,
        contract_checks,
    )
    report = _run_report(
        readiness,
        prepared_state,
        contract_checks=tuple(contract_checks),
        path_faithfulness=path_faithfulness,
        first_mismatch=mismatch,
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
        if tuple(value.shape) != slot.shape:
            raise _input_error(
                RunnableErrorCode.INPUT_SHAPE_MISMATCH,
                slot,
                f"Runtime input shape {tuple(value.shape)} does not match {slot.shape}.",
            )
        if str(value.dtype) != slot.dtype:
            raise _input_error(
                RunnableErrorCode.INPUT_DTYPE_MISMATCH,
                slot,
                f"Runtime input dtype {value.dtype} does not match {slot.dtype}.",
            )
        values[slot.slot_id] = value.detach().clone()
        checks.append(ContractCheck(f"input:{slot.slot_id}", True, None))
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


def _execute_sparse_call(
    call: RunnableCallDescriptor,
    func: Callable[..., Any],
    slot_values: Mapping[str, torch.Tensor],
) -> Any:
    """Construct and execute one sparse call from literal and tensor leaves."""

    args: list[Any] = [None] * call.num_positional_args
    kwargs: dict[str, Any] = {}
    for argument in call.literal_arguments:
        _write_argument(args, kwargs, argument.argument_path, _decode_literal(argument.value))
    for argument in call.tensor_arguments:
        try:
            value = slot_values[argument.slot_id]
        except KeyError as exc:
            raise RunPreconditionError(
                f"Sparse call {call.call_id!r} references unavailable slot {argument.slot_id!r}.",
                code=RunnableErrorCode.MISSING_TENSOR_SLOT.value,
                call_id=call.call_id,
                slot_id=argument.slot_id,
            ) from exc
        _write_argument(args, kwargs, argument.argument_path, value)
    try:
        return func(*args, **kwargs)
    except (TypeError, AttributeError) as exc:
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
            op._internal_set("out", value)


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
) -> tuple[ContractCheck, ...]:
    """Slice, validate, and stage one grouped call's tensor outputs."""

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    checks: list[ContractCheck] = []
    for slot_id, op_label in zip(call.output_slot_ids, call.op_labels):
        slot = slots[slot_id]
        try:
            value = _value_at_path(output, slot.output_path or ())
        except (KeyError, IndexError, TypeError) as exc:
            raise RunPreconditionError(
                f"Call {call.call_id!r} output lacks path {slot.output_path!r}: {exc}",
                code=RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH.value,
                call_id=call.call_id,
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise RunPreconditionError(
                f"Call {call.call_id!r} output is not a tensor at {slot.output_path!r}.",
                code=RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH.value,
                call_id=call.call_id,
            )
        slot_values[slot_id] = value
        for version in descriptor.tensor_slots:
            if version.version_of == slot_id and version.producer_slot_id == slot_id:
                slot_values[version.slot_id] = value
        op = _op_for_label(fork, op_label)
        if op is not None:
            op._internal_set("out", value)
        shape_ok = tuple(value.shape) == slot.shape
        dtype_ok = str(value.dtype) == slot.dtype
        checks.append(ContractCheck(f"output_shape:{slot_id}", shape_ok, None))
        checks.append(ContractCheck(f"output_dtype:{slot_id}", dtype_ok, None))
    return tuple(checks)


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
    return _container_from_paths(values)


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


def _path_faithfulness(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    checks: Sequence[ContractCheck],
) -> tuple[PathFaithfulness, RunnableDiagnostic | None]:
    """Classify witness and contract agreement without Stage-6 enforcement."""

    failed = next((check for check in checks if not check.passed), None)
    if failed is not None:
        return PathFaithfulness.DIVERGED, failed.diagnostic
    call_slots = {call.call_id: call.output_slot_ids for call in descriptor.calls}
    for witness in sorted(descriptor.control_witnesses, key=lambda item: item.order):
        if (
            witness.kind
            not in {
                ControlWitnessKind.SCALAR_BOOL,
                ControlWitnessKind.LOOP_PREDICATE,
            }
            or witness.call_id is None
        ):
            continue
        expected = _decode_literal(witness.observed_value)
        actual_values = [
            slot_values[slot_id]
            for slot_id in call_slots.get(witness.call_id, ())
            if slot_id in slot_values
        ]
        if actual_values and bool(actual_values[0].item()) != bool(expected):
            diagnostic = RunnableDiagnostic(
                code=(
                    RunnableErrorCode.LOOP_PREDICATE_DIVERGENCE
                    if witness.kind is ControlWitnessKind.LOOP_PREDICATE
                    else RunnableErrorCode.SCALAR_BOOL_DIVERGENCE
                ),
                message=f"Control witness {witness.witness_id!r} disagreed with the recipe.",
                registry_id=None,
                affected_op_labels=(witness.site_label,),
                recorded_runtime=descriptor.compatibility.backend_version,
                current_runtime=str(torch.__version__),
                detection_stage="run_control_witness",
                resolver_provenance=None,
                analysis_load_available=True,
                details=(
                    ("expected", repr(expected)),
                    ("actual", repr(bool(actual_values[0].item()))),
                ),
            )
            return PathFaithfulness.DIVERGED, diagnostic
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
) -> RunReport:
    """Build the settled Stage-5 run-report surface."""

    return RunReport(
        readiness=readiness,
        state_source=state.state_source,
        initializer_policy_version=state.initializer_policy_version,
        seed=state.seed,
        random_filled_slot_ids=state.random_filled_slot_ids,
        contract_checks=contract_checks,
        path_faithfulness=path_faithfulness,
        first_mismatch=first_mismatch,
        numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
        poisoned=False,
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
    """Return the ordinary deterministic Trace fork label for a run transaction."""

    return trace._next_fork_name()


__all__ = ["raise_analysis_run_unavailable", "run_live_trace", "run_loaded_sparse_trace"]
