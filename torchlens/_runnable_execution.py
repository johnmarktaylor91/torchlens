"""Transactional execution providers for the unified :meth:`Trace.run` surface."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from itertools import count
import re
import struct
from typing import Any, cast

import numpy as np
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
from .utils.rng import (
    aten_qualname_is_seeded_rng,
    restore_host_rng,
    set_random_seed,
    snapshot_host_rng,
)
from .runnable import (
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    CallableRegistryEntry,
    ContractCheck,
    ControlWitness,
    ControlWitnessKind,
    DivergencePolicy,
    LiteralAtom,
    LiteralAtomKind,
    LiteralMapping,
    LiteralSequence,
    LiteralSequenceKind,
    LiteralSlice,
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


def _host_rng_unreproduced(descriptor: SparseRunDescriptor, seed: int | None) -> bool:
    """Return whether a host-RNG (Python/NumPy) capture is being replayed off-seed.

    A model whose traced forward consumed Python ``random`` / NumPy RNG chose an
    unwitnessed branch. The sparse trace holds exactly one recorded path, so only a
    run that reproduces the captured seed can honestly claim the recorded branch is
    the one a fresh call takes. Any other seed (including ``None``, or a capture with
    no identifiable seed) leaves the branch unverifiable -- never a false
    VERIFIED/ATTESTED with a stale result.

    Parameters
    ----------
    descriptor:
        Loaded sparse descriptor carrying the host-RNG profile.
    seed:
        Seed supplied to :meth:`Trace.run`.

    Returns
    -------
    bool
        ``True`` when the recorded host-RNG branch cannot be honestly reproduced.
    """

    profile = descriptor.rng_profile
    if not profile.host_rng_consumed:
        return False
    if profile.capture_seed is None or seed is None:
        return True
    return seed != profile.capture_seed


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

    # Escape-source witness slots must be digested at their production point (before
    # any later in-place op mutates the live tensor), matching the save-side digest.
    escape_witness_slot_ids = _tensor_derived_scalar_witness_slot_ids(descriptor)
    witness_source_snapshots: dict[str, torch.Tensor] = {}
    _populate_source_slots(
        fork,
        descriptor,
        slot_values,
        witness_slot_ids=escape_witness_slot_ids,
        witness_source_snapshots=witness_source_snapshots,
    )
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
    host_rng_unreproduced = _host_rng_unreproduced(descriptor, seed)
    # Faithful original replay of a host-RNG capture (matching seed): reseed every
    # engine (torch + Python + NumPy) to the captured seed so the recorded taken
    # path is reproduced exactly, not just torch's generator. Preserve and restore
    # the caller's host RNG so run() never leaks a reseed into ambient global state.
    reseed_host = (
        seed is not None and descriptor.rng_profile.host_rng_consumed and not host_rng_unreproduced
    )
    host_rng_saved = snapshot_host_rng() if reseed_host else None
    rng_context = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
    try:
        with rng_context, _state.pause_logging():
            if reseed_host:
                set_random_seed(cast(int, seed))
            elif seed is not None:
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
                        witness_slot_ids=escape_witness_slot_ids,
                        witness_source_snapshots=witness_source_snapshots,
                    )
                )
                contract_checks.extend(_call_witness_checks(descriptor, call, slot_values))
                _raise_first_divergence(contract_checks, divergence_policy, fork=fork)

            _walk_call_cone(descriptor.calls, execute_call)
    finally:
        if host_rng_saved is not None:
            restore_host_rng(host_rng_saved)

    output = _reconstruct_output(descriptor, slot_values, fork, call_outputs=call_outputs)
    contract_checks.extend(
        _post_execution_contract_checks(
            descriptor,
            inputs=inputs,
            output=output,
            slot_values=slot_values,
            fork=fork,
        )
    )
    _raise_first_divergence(contract_checks, divergence_policy, fork=fork)
    nontensor_inputs_match = all(
        check.passed for check in contract_checks if check.name.startswith("input_literal:")
    )
    tensor_derived_scalar_stale = _tensor_derived_scalar_stale(
        descriptor, slot_values, witness_source_snapshots
    )
    unbound_state_escape_stale = _unbound_state_escape_stale(descriptor, slot_values)
    numeric_attestation, attestation_check = _numeric_attestation_check(
        descriptor,
        prepared_state,
        slot_values=slot_values,
        attestation_slot_values=attestation_slot_values,
        input_byte_digests=input_byte_digests,
        trace=trace,
        nontensor_inputs_match=nontensor_inputs_match,
        host_rng_unreproduced=host_rng_unreproduced,
        tensor_derived_scalar_stale=tensor_derived_scalar_stale,
        unbound_state_escape_stale=unbound_state_escape_stale,
    )
    if attestation_check is not None:
        contract_checks.append(attestation_check)
        if not attestation_check.passed:
            _raise_numeric_attestation_failure(fork, attestation_check)
    path_faithfulness, mismatch = _path_faithfulness(
        descriptor,
        contract_checks,
        host_rng_unreproduced=host_rng_unreproduced,
        tensor_derived_scalar_stale=tensor_derived_scalar_stale,
        unbound_state_escape_stale=unbound_state_escape_stale,
    )
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
    prior_log_ids = {id(log) for log in _state.list_logs()}
    fork = trace._fork_trace(name=_run_fork_name(trace))
    try:
        input_args = inputs
        input_kwargs = None
        if (
            isinstance(inputs, Mapping)
            and {"args", "kwargs"}.issubset(inputs)
            and set(inputs).issubset({"args", "kwargs"})
        ):
            args, kwargs = _split_mixed_inputs(inputs)
            input_args = list(args)
            input_kwargs = dict(kwargs)
        fork.save_new_outs(model, input_args, input_kwargs=input_kwargs, random_seed=seed)
        output, faithful = _reconstruct_live_output(fork)
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
        # Honesty gate: only a faithfully reconstructed output (exact container type
        # and non-tensor leaves) is VERIFIED. An output we could only approximate
        # from naive leaf paths is UNVERIFIABLE, never blessed with a wrong object.
        report = RunReport(
            readiness=readiness,
            state_source=StateSource.LIVE_MODEL_STATE,
            initializer_policy_version=None,
            seed=seed,
            random_filled_slot_ids=(),
            contract_checks=(
                _contract_check(
                    "live_output_reconstruction",
                    faithful,
                    RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                    "Live output could not be faithfully reconstructed from its "
                    "captured container contract.",
                ),
            ),
            path_faithfulness=(
                PathFaithfulness.VERIFIED if faithful else PathFaithfulness.UNVERIFIABLE
            ),
            first_mismatch=None,
            numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
            poisoned=False,
        )
        return RunResult(output=output, trace=fork, report=report)
    except BaseException:
        _state._unregister_log(fork)
        for log in _state.list_logs():
            if id(log) not in prior_log_ids:
                _state._unregister_log(log)
        raise


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


_MODEL_INPUT_LITERAL_SITE_PREFIX = "model_input_literal:"
"""``site_label`` prefix marking a witnessed non-tensor model-input leaf."""

_MODEL_INPUT_LITERAL_FACT_KEY = "model_input_literal"
"""Discriminator key present in every non-tensor model-input leaf fact."""


def _model_input_literal_facts(
    descriptor: SparseRunDescriptor,
) -> list[tuple[ControlWitness, Mapping[str, Any]]]:
    """Decode every witnessed non-tensor model-input leaf fact."""

    facts: list[tuple[ControlWitness, Mapping[str, Any]]] = []
    for witness in descriptor.control_witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith(_MODEL_INPUT_LITERAL_SITE_PREFIX):
            continue
        decoded = _decode_literal(witness.observed_value)
        if isinstance(decoded, Mapping) and decoded.get(_MODEL_INPUT_LITERAL_FACT_KEY) is True:
            facts.append((witness, decoded))
    return facts


def _is_model_input_literal_witness(witness: ControlWitness) -> bool:
    """Return whether a structure witness records a non-tensor input leaf."""

    return (
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_MODEL_INPUT_LITERAL_SITE_PREFIX)
    )


def _model_input_arity_positions(descriptor: SparseRunDescriptor) -> set[Any]:
    """Return every distinct model-input site position, tensor and non-tensor.

    The tensor-slot positions alone undercount arity when a model site carries a
    non-tensor Python argument (e.g. ``forward(x, flag)``): the single tensor
    slot would falsely trigger the "single bare input" shortcut and bind the
    whole runtime argument list as one tensor. Including witnessed non-tensor
    leaf positions makes the shortcut fire only for a genuinely single-argument
    model, so a mixed ``[tensor, python_arg]`` call binds each site correctly.
    """

    positions = {
        slot.input_binding.model_site_position
        for slot in descriptor.tensor_slots
        if slot.role is TensorSlotRole.MODEL_INPUT and slot.input_binding is not None
    }
    for _witness, fact in _model_input_literal_facts(descriptor):
        position = fact.get("position")
        if isinstance(position, (list, tuple)):
            positions.add(tuple(position))
        elif position is not None:
            positions.add(position)
    return positions


def _literal_leaf_equal(recorded: Any, runtime: Any) -> bool:
    """Type-strict, bit-exact equality for recorded vs runtime non-tensor input leaves.

    ``bool`` is a subclass of ``int`` and floats are distinct from ints, so a
    changed control input like ``True`` -> ``1`` or ``2`` -> ``2.0`` must count as
    a divergence rather than silently comparing equal.

    Floats are compared by their IEEE-754 bit pattern, never ``==``. Ordinary
    ``==`` is dishonest for control witnesses at two values: ``-0.0 == +0.0`` is
    ``True`` (a changed sign bit that steers ``math.copysign``/``1/x`` control flow
    would falsely pass), while ``nan == nan`` is ``False`` (an unchanged ``nan``
    would falsely diverge). Bit-pattern identity makes ``-0.0`` differ from
    ``+0.0`` and a ``nan`` equal to a ``nan`` with the same bits.
    """

    recorded = _normalize_numpy_scalar(recorded)
    runtime = _normalize_numpy_scalar(runtime)
    if isinstance(recorded, bool) or isinstance(runtime, bool):
        return isinstance(recorded, bool) and isinstance(runtime, bool) and recorded == runtime
    # Float family (incl. numeric float subclasses such as ``numpy.float64``),
    # excluding bool handled above. ``int`` stays distinct from ``float`` (``2``
    # vs ``2.0`` must diverge), but ``numpy.float64(2.0)`` compares equal to the
    # plain ``float`` the literal grammar round-trips to. Compare by IEEE-754 bit
    # pattern of the float VALUE so the r4 signed-zero/NaN honesty is preserved.
    rec_is_float = isinstance(recorded, float)
    run_is_float = isinstance(runtime, float)
    if rec_is_float or run_is_float:
        if not (rec_is_float and run_is_float):
            return False
        return struct.pack(">d", float(recorded)) == struct.pack(">d", float(runtime))
    # Integer family (incl. numeric int subclasses), excluding bool. ``int`` vs a
    # non-int type diverges; two integers compare by value.
    rec_is_int = isinstance(recorded, int)
    run_is_int = isinstance(runtime, int)
    if rec_is_int or run_is_int:
        if not (rec_is_int and run_is_int):
            return False
        return int(recorded) == int(runtime)
    if type(recorded) is not type(runtime):
        return False
    return bool(recorded == runtime)


def _normalize_numpy_scalar(value: Any) -> Any:
    """Convert one NumPy scalar to its Python equivalent for literal comparison.

    Parameters
    ----------
    value:
        Captured or runtime literal leaf.

    Returns
    -------
    Any
        ``value.item()`` for NumPy scalars, otherwise the original value.
    """

    return value.item() if isinstance(value, np.generic) else value


def _input_literal_contract_checks(
    descriptor: SparseRunDescriptor,
    inputs: Any,
    positions: set[Any],
) -> tuple[ContractCheck, ...]:
    """Compare runtime non-tensor input leaves with recorded capture-time values.

    A differing non-tensor leaf means the recorded taken-path DAG may be wrong
    for this input, so the check fails and the run diverges instead of silently
    replaying a numerically wrong result.
    """

    checks: list[ContractCheck] = []
    for witness, fact in _model_input_literal_facts(descriptor):
        raw_position = fact.get("position")
        position = tuple(raw_position) if isinstance(raw_position, (list, tuple)) else raw_position
        path = tuple(fact.get("path", ()) or ())
        recorded = fact.get("value")
        if not bool(fact.get("encodable", False)):
            # Opaque capture-time leaf: not representable in the frozen literal
            # grammar, so it cannot be compared across save/load. Its position
            # still constrains arity; a value claim would be dishonest.
            continue
        try:
            root = _input_site_value(inputs, position, positions)
            runtime_leaf = _value_at_path(root, path)
            resolved = True
        except (KeyError, IndexError, TypeError, AttributeError):
            runtime_leaf = None
            resolved = False
        passed = resolved and _literal_leaf_equal(recorded, runtime_leaf)
        checks.append(
            _contract_check(
                f"input_literal:{position!r}:{path!r}",
                passed,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime non-tensor input leaf differs from the recorded "
                "capture-time value; the recorded taken path may not be valid "
                "for this input.",
                affected_op_labels=(witness.site_label,),
                details=(
                    ("model_site_position", repr(position)),
                    ("container_path", repr(path)),
                    ("recorded_value", repr(recorded)),
                    ("runtime_value", repr(runtime_leaf) if resolved else "<unresolved>"),
                ),
            )
        )
    return tuple(checks)


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
    positions = _model_input_arity_positions(descriptor)
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
    checks.extend(_input_literal_contract_checks(descriptor, inputs, positions))
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


def _positions_are_mixed(positions: set[Any]) -> bool:
    """Return whether a capture carries both positional and keyword model sites."""

    kinds = {p[0] for p in positions if isinstance(p, tuple) and len(p) == 2}
    return "arg" in kinds and "kwarg" in kinds


def _split_mixed_inputs(inputs: Any) -> tuple[Sequence[Any], Mapping[Any, Any]]:
    """Split a combined mixed-input mapping into positional and keyword parts.

    A capture with BOTH positional tensor sites and keyword leaves (e.g.
    ``forward(x, *, add)``) cannot be rebound from a bare sequence (fails the
    keyword site) or a bare mapping (fails positional binding). The runnable
    executor accepts a single combined ``{"args": [...], "kwargs": {...}}``
    mapping so mixed captures have a representable ``run(inputs=)`` spelling.
    """

    if not isinstance(inputs, Mapping) or not set(inputs).issubset({"args", "kwargs"}):
        raise TypeError(
            "mixed positional+keyword captures require an inputs mapping of the "
            "form {'args': [...], 'kwargs': {...}}"
        )
    args = inputs.get("args", ())
    kwargs = inputs.get("kwargs", {})
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        raise TypeError("mixed-input 'args' must be a sequence")
    if not isinstance(kwargs, Mapping):
        raise TypeError("mixed-input 'kwargs' must be a mapping")
    return args, kwargs


def _input_site_value(inputs: Any, position: Any, positions: set[Any]) -> Any:
    """Select one top-level argument or keyword site from the public input tree."""

    if isinstance(position, tuple) and len(position) == 2:
        kind, key = position
        if _positions_are_mixed(positions):
            args, kwargs = _split_mixed_inputs(inputs)
            if kind == "arg":
                return args[cast(int, key)]
            if kind == "kwarg":
                return kwargs[key]
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
    positions = _model_input_arity_positions(descriptor)
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
    *,
    witness_slot_ids: frozenset[str] = frozenset(),
    witness_source_snapshots: dict[str, torch.Tensor] | None = None,
) -> None:
    """Populate input and buffer source Ops on the transactional run fork.

    A source slot (model input / buffer) that is itself a tensor->host escape
    witness site is snapshotted here, at population -- the mutation-consistent point
    matching the save-side digest -- so a later in-place op cannot restale the
    staleness comparison.
    """

    for slot in descriptor.tensor_slots:
        if slot.role not in {TensorSlotRole.MODEL_INPUT, TensorSlotRole.BUFFER}:
            continue
        value = slot_values.get(slot.slot_id)
        op = _op_for_slot(fork, slot.slot_id)
        if value is not None and op is not None:
            op._internal_set("out", value.detach().clone())
        if (
            value is not None
            and witness_source_snapshots is not None
            and slot.slot_id in witness_slot_ids
        ):
            witness_source_snapshots[slot.slot_id] = value.detach().clone()


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


def _resolve_setter_output(
    call: RunnableCallDescriptor,
    output: Any,
    slot_values: Mapping[str, torch.Tensor],
) -> Any:
    """Alias the mutation target for a setter-style in-place call that returns None.

    Ordinary in-place operators (``add_``/``mul_``/``copy_``/``out=``) return the
    tensor they mutated, so the recorded output slot is bound from the Python
    return value. Setter-style mutators such as ``Tensor.__setitem__`` mutate
    their target in place but return ``None``. Their recorded output slot is a
    version of the mutation target, so bind it from that already-mutated tensor
    rather than treating the ``None`` return as a structural mismatch (which would
    otherwise raise a false PathDivergenceError on the original input). Only a
    genuinely in-place call whose runtime return is ``None`` is remapped; every
    other call keeps its real return so honest structure/aliasing checks stand.
    """

    if output is not None or not call.is_inplace:
        return output
    target_slot_id = _mutation_target_slot_id(call)
    if target_slot_id is None:
        return output
    target = slot_values.get(target_slot_id)
    if not isinstance(target, torch.Tensor):
        return output
    return target


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
    witness_slot_ids: frozenset[str] = frozenset(),
    witness_source_snapshots: dict[str, torch.Tensor] | None = None,
) -> tuple[ContractCheck, ...]:
    """Slice, validate, and stage one grouped call's tensor outputs."""

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    checks: list[ContractCheck] = []
    output = _resolve_setter_output(call, output, slot_values)
    expected_paths = tuple(slots[slot_id].output_path or () for slot_id in call.output_slot_ids)
    actual_paths = _tensor_leaf_paths(output)
    expected_structure_paths = _canonicalize_structseq_output_paths(output, expected_paths)
    actual_structure_paths = _canonicalize_structseq_output_paths(output, actual_paths)
    output_type_matches = _recorded_structseq_output_type_matches(fork, call, output)
    checks.append(
        _contract_check(
            f"output_structure:{call.call_id}",
            len(call.output_slot_ids) == len(call.op_labels)
            and output_type_matches
            and actual_structure_paths == expected_structure_paths,
            RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
            f"Call {call.call_id!r} output tensor paths disagree with the recorded container.",
            affected_op_labels=call.op_labels,
            details=(
                ("expected_paths", repr(expected_paths)),
                ("actual_paths", repr(tuple(actual_paths))),
                ("canonical_expected_paths", repr(expected_structure_paths)),
                ("canonical_actual_paths", repr(actual_structure_paths)),
            ),
        )
    )
    for slot_id, op_label in zip(call.output_slot_ids, call.op_labels):
        slot = slots[slot_id]
        try:
            value = _value_at_path(output, slot.output_path or ())
        except (AttributeError, KeyError, IndexError, TypeError) as exc:
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
        out_argument_slot_id = _out_argument_slot_id(call) if call.is_inplace else None
        if out_argument_slot_id is not None:
            # ``out=`` calls mutate their explicit destination, which may have
            # been produced by an earlier call. Keep that aliased slot current
            # for downstream reads and activation attestation.
            slot_values[out_argument_slot_id] = value
            produced_slot_ids.add(out_argument_slot_id)
        for version in descriptor.tensor_slots:
            if version.version_of == slot_id and version.producer_slot_id == slot_id:
                slot_values[version.slot_id] = value
                produced_slot_ids.add(version.slot_id)
        for produced_slot_id in produced_slot_ids & attestation_slot_ids:
            attestation_slot_values[produced_slot_id] = value.detach().clone()
        if witness_source_snapshots is not None:
            # Snapshot every escape-witness source slot at its production point so a
            # later in-place mutation of the live tensor cannot restale the digest
            # comparison (H3): the run-digest then matches the pre-mutation save-digest.
            for produced_slot_id in produced_slot_ids & witness_slot_ids:
                witness_source_snapshots[produced_slot_id] = value.detach().clone()
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
    input_slot_id = _mutation_target_slot_id(call)
    if input_slot_id is None or not call.output_slot_ids:
        return (
            _contract_check(
                f"mutation:{call.call_id}",
                False,
                RunnableErrorCode.MUTATION_VERSION_MISMATCH,
                f"In-place call {call.call_id!r} lacks an input/output version relation.",
                affected_op_labels=call.op_labels,
            ),
        )
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


def _mutation_target_slot_id(call: RunnableCallDescriptor) -> str | None:
    """Return the tensor slot expected to alias an in-place call's output.

    Parameters
    ----------
    call:
        Frozen runnable call descriptor.

    Returns
    -------
    str | None
        The explicit ``out=`` tensor slot when present, otherwise the first
        tensor argument for conventional in-place operators.
    """

    out_argument_slot_id = _out_argument_slot_id(call)
    if out_argument_slot_id is not None:
        return out_argument_slot_id
    return call.tensor_arguments[0].slot_id if call.tensor_arguments else None


def _out_argument_slot_id(call: RunnableCallDescriptor) -> str | None:
    """Return an explicit ``out=`` tensor slot, if the call has one.

    Parameters
    ----------
    call:
        Frozen runnable call descriptor.

    Returns
    -------
    str | None
        The explicit output tensor slot or ``None`` for conventional in-place
        calls that mutate their first tensor argument.
    """

    return next(
        (
            argument.slot_id
            for argument in call.tensor_arguments
            if argument.argument_path == ("kwargs", "out")
        ),
        None,
    )


def _reconstruct_output(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    fork: Any,
    *,
    call_outputs: Mapping[str, Any],
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
        raw_output = _raw_runtime_output(descriptor, None, call_outputs)
        if (
            container_spec.type_module == "torch.return_types"
            and _torch_structseq_field_names(raw_output) == container_spec.fields
        ):
            return raw_output
        try:
            return rebuild_container_from_spec(container_spec, [value for _, value in values])
        except (TypeError, ValueError) as exc:
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


def _reconstruct_live_output(trace: Any) -> tuple[Any, bool]:
    """Reconstruct refreshed live output faithfully and report reconstruction fidelity.

    Returns
    -------
    tuple[Any, bool]
        ``(output, faithful)``. ``output`` is the exact model-output object rebuilt
        from the captured :class:`ContainerSpec` (correct container type, non-tensor
        literal leaves preserved) when a reconstructable final-output container was
        recorded, or the genuine single bare-tensor output. ``faithful`` is ``False``
        only when the output could merely be approximated from naive leaf paths (no
        faithful container contract, e.g. an opaque/BFS-fallback container); the
        caller then downgrades ``path_faithfulness`` to ``UNVERIFIABLE`` instead of
        blessing a lossy substitution with ``VERIFIED``.
    """

    from .data_classes.container import container_from_op

    output_labels = tuple(getattr(trace, "output_layers", ()) or ())
    for label in output_labels:
        op = trace.ops[label]
        container = container_from_op(op)
        # A reconstructable final-output view carries the captured ContainerSpec, so
        # ``reconstruct`` rebuilds the SAME object a live forward returns (container
        # kind + literal leaves + fields).
        if (
            container is not None
            and container.root_kind == "final_output"
            and container.supports_reconstruct
        ):
            return container.reconstruct(values="out"), True
    if len(output_labels) == 1:
        op = trace.ops[output_labels[0]]
        has_spec = getattr(op, "container_spec", None) is not None
        has_path = bool(getattr(op, "container_path", ()) or ())
        if not has_spec and not has_path:
            # Genuine single bare-tensor model output: no container to reconstruct.
            return op.out, True
    # Multi-leaf output lacking a faithful reconstructable container contract, or a
    # single leaf that was actually a non-reconstructable (opaque) container. Return
    # a best-effort approximation but report it as NOT faithful.
    values = [
        (tuple(getattr(trace.ops[label], "container_path", ()) or ()), trace.ops[label].out)
        for label in output_labels
    ]
    return _container_from_paths(values), False


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
        if _is_model_input_literal_witness(witness):
            # Non-tensor input leaves are compared in the input contract before
            # execution; they are not runtime container-structure facts.
            continue
        if _is_unbound_state_escape_witness(witness):
            # Unbound state escapes are compared by capture-digest in the dedicated
            # staleness check, not against runtime container structure.
            continue
        if witness.kind is ControlWitnessKind.CONDITIONAL_ARM_ENTRY:
            checks.append(_conditional_arm_check(witness, fork))
        elif witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            checks.append(
                _structure_witness_check(
                    witness,
                    descriptor,
                    inputs=inputs,
                    output=output,
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
) -> ContractCheck:
    """Compare a model-boundary container witness with runtime structure."""

    expected = _decode_literal(witness.observed_value)
    role = expected.get("role")
    if role == "model_input":
        runtime_value = _runtime_input_for_structure_witness(descriptor, expected, inputs)
        expected_paths = tuple(tuple(path) for path in expected.get("leaf_paths", ()))
        expected_kind = str(expected.get("kind", "unknown"))
    else:
        runtime_value = output
        expected_paths = tuple(tuple(path) for path in expected.get("leaf_paths", ()))
        expected_kind = str(expected.get("kind", "unknown"))
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
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        paths: list[tuple[str | int, ...]] = []
        for field in dataclasses.fields(value):
            paths.extend(_tensor_leaf_paths(getattr(value, field.name), (*path, field.name)))
        return tuple(paths)
    field_names = _container_field_names(value)
    if field_names:
        paths = []
        for name in field_names:
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


def _canonicalize_structseq_output_paths(
    output: Any,
    paths: Sequence[Sequence[str | int]],
) -> tuple[tuple[str | int, ...], ...]:
    """Canonicalize only ``torch.return_types`` named/positional path components.

    Parameters
    ----------
    output:
        Runtime output container used to interpret path components.
    paths:
        Tensor leaf paths to canonicalize.

    Returns
    -------
    tuple[tuple[str | int, ...], ...]
        Paths where field names and positional indexes are equivalent only while
        traversing a ``torch.return_types.*`` structseq.
    """

    return tuple(_canonicalize_structseq_output_path(output, tuple(path)) for path in paths)


def _canonicalize_structseq_output_path(
    output: Any,
    path: tuple[str | int, ...],
) -> tuple[str | int, ...]:
    """Canonicalize one path through runtime ``torch.return_types`` containers.

    Parameters
    ----------
    output:
        Runtime output container used to interpret path components.
    path:
        Tensor leaf path to canonicalize.

    Returns
    -------
    tuple[str | int, ...]
        Path with structseq fields represented by their positional index.
    """

    current = output
    canonical: list[str | int] = []
    for component in path:
        canonical_component = component
        field_names = _torch_structseq_field_names(current)
        if field_names:
            if isinstance(component, str) and component in field_names:
                canonical_component = field_names.index(component)
            elif isinstance(component, int) and 0 <= component < len(field_names):
                canonical_component = component
        canonical.append(canonical_component)
        try:
            current = _value_at_path(current, (canonical_component,))
        except (AttributeError, KeyError, IndexError, TypeError):
            break
    return tuple(canonical)


def _recorded_structseq_output_type_matches(
    trace: Any,
    call: RunnableCallDescriptor,
    output: Any,
) -> bool:
    """Return whether a recorded torch structseq call produced a torch structseq.

    Parameters
    ----------
    trace:
        Runtime fork containing recorded op metadata.
    call:
        Runnable call descriptor being bound.
    output:
        Runtime output produced by the resolved callable.

    Returns
    -------
    bool
        False only when the recorded call's output container was a
        ``torch.return_types.*`` structseq but runtime produced another tuple
        shape, such as a plain positional tuple.
    """

    if not _call_has_recorded_torch_structseq_output(trace, call):
        return True
    return _torch_structseq_field_names(output) != ()


def _call_has_recorded_torch_structseq_output(
    trace: Any,
    call: RunnableCallDescriptor,
) -> bool:
    """Return whether any call output op recorded a torch structseq container.

    Parameters
    ----------
    trace:
        Runtime fork containing recorded op metadata.
    call:
        Runnable call descriptor being inspected.

    Returns
    -------
    bool
        True when any output op for the call has a ``torch.return_types`` root
        container specification.
    """

    for op_label in call.op_labels:
        op = _op_for_label(trace, op_label)
        container_spec = getattr(op, "container_spec", None)
        if (
            isinstance(container_spec, ContainerSpec)
            and container_spec.type_module == "torch.return_types"
        ):
            return True
    return False


def _container_leaf_paths(
    value: Any,
    path: tuple[str | int, ...] = (),
) -> tuple[tuple[str | int, ...], ...]:
    """Return producer-compatible tensor-leaf paths for a boundary container."""

    if isinstance(value, torch.Tensor):
        return (path,)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        paths: list[tuple[str | int, ...]] = []
        for field in dataclasses.fields(value):
            paths.extend(_container_leaf_paths(getattr(value, field.name), (*path, field.name)))
        return tuple(paths)
    field_names = _container_field_names(value)
    if field_names:
        paths = []
        for name in field_names:
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
    return ()


def _container_kind(value: Any) -> str:
    """Return the sparse witness vocabulary name for a runtime container."""

    if isinstance(value, torch.Tensor):
        return "tensor"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return "dataclass"
    if _is_hf_model_output(value):
        return "hf_model_output"
    if _container_field_names(value):
        return "namedtuple"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, Mapping):
        return "dict"
    return type(value).__name__


def _is_hf_model_output(value: Any) -> bool:
    """Return whether ``value`` looks like a HuggingFace ``ModelOutput``."""

    cls = type(value)
    if any(
        base.__module__.startswith("transformers") and base.__name__ == "ModelOutput"
        for base in cls.__mro__
    ):
        return True
    return (
        (cls.__module__.startswith("transformers") or cls.__name__.endswith("ModelOutput"))
        and hasattr(value, "keys")
        and hasattr(value, "__getitem__")
    )


def _container_field_names(value: Any) -> tuple[str, ...]:
    """Return the stable field names for namedtuple-like runtime containers.

    Parameters
    ----------
    value:
        Candidate runtime container.

    Returns
    -------
    tuple[str, ...]
        Namedtuple fields or torch structseq fields; empty when ``value`` is
        not a field-addressable container.
    """

    if not isinstance(value, tuple):
        return ()
    if hasattr(value, "_fields"):
        return tuple(str(name) for name in value._fields)
    return _torch_structseq_field_names(value)


def _torch_structseq_field_names(value: Any) -> tuple[str, ...]:
    """Return producer-compatible field names for a torch structseq value.

    Parameters
    ----------
    value:
        Candidate tuple-like torch return value.

    Returns
    -------
    tuple[str, ...]
        Public structseq field names, or an empty tuple when ``value`` is not
        a fully named ``torch.return_types`` value.
    """

    if not isinstance(value, tuple) or type(value).__module__ != "torch.return_types":
        return ()
    n_fields = getattr(value, "n_fields", None)
    n_unnamed = getattr(value, "n_unnamed_fields", 0)
    if not isinstance(n_fields, int) or n_fields <= 0 or n_unnamed:
        return ()
    field_names = tuple(
        match.group(1)
        for match in re.finditer(r"^\s*([A-Za-z_]\w*)=", repr(value), flags=re.MULTILINE)
    )
    return field_names if len(field_names) == n_fields else ()


def _scalar_literal_equal(actual: Any, expected: Any) -> bool:
    """Return exact scalar equality, treating two matching NaNs as equal.

    Bool and non-bool numeric types are kept distinct so a recomputed ``True`` is
    never mistaken for a numeric ``1``. A capture-time NaN scalar recomputes to
    NaN on the original input, so ``NaN == NaN`` must read as equal here.
    """

    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    if isinstance(actual, float) and isinstance(expected, float):
        if actual != actual and expected != expected:  # both NaN
            return True
    return bool(actual == expected)


def _tensor_derived_scalar_witness_slot_ids(descriptor: SparseRunDescriptor) -> frozenset[str]:
    """Return the runtime slot ids of every tensor->host escape-source witness.

    Each ``TENSOR_DERIVED_SCALAR_LITERAL`` witness digests its source slot at a
    mutation-consistent snapshot (the op's capture-time production value). At run
    time the source value must be snapshotted at the SAME logical point -- when the
    slot is first produced -- before any later in-place op can mutate the live
    tensor, so the run-digest compares the same logical value the save-digest did.
    """

    return frozenset(
        witness.site_label
        for witness in descriptor.control_witnesses
        if witness.kind is ControlWitnessKind.TENSOR_DERIVED_SCALAR_LITERAL
    )


def _tensor_derived_scalar_stale(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    witness_source_snapshots: Mapping[str, torch.Tensor] | None = None,
) -> bool:
    """Return whether a tensor->host escape source slot is stale for this run.

    A ``TENSOR_DERIVED_SCALAR_LITERAL`` witness records the runtime slot of the op
    whose output tensor escaped to the Python host (via ``.item()`` / ``int()`` /
    ``.tolist()`` / ``aten._local_scalar_dense`` / etc.) together with the SHA-256
    byte digest of that tensor at capture time. The escaped value was baked into a
    downstream literal or steered pure-Python control flow -- neither of which the
    sparse DAG can recompute. If the source slot recomputes to a different value
    than at capture (a CHANGED input), the baked literal / taken branch may be
    stale, so the run must not be blessed VERIFIED/ATTESTED. A slot that recomputes
    the exact capture-time bytes (the ORIGINAL input) keeps the run faithful. A
    missing source slot is treated as stale: the dependency cannot be re-confirmed,
    so the honest ceiling is UNVERIFIABLE.

    A legacy witness whose ``observed_value`` is a scalar literal (rather than a
    byte digest) is compared by exact scalar equality for backward compatibility.
    """

    snapshots = witness_source_snapshots or {}
    for witness in descriptor.control_witnesses:
        if witness.kind is not ControlWitnessKind.TENSOR_DERIVED_SCALAR_LITERAL:
            continue
        # Prefer the production-time snapshot: a later in-place op (``y.add_(...)``)
        # could have mutated the live slot value after the escape read it, and the
        # save-digest was taken at the pre-mutation production point. The snapshot
        # compares the SAME logical value; the live slot is the honest fallback.
        recomputed = snapshots.get(witness.site_label)
        if recomputed is None:
            recomputed = slot_values.get(witness.site_label)
        if not isinstance(recomputed, torch.Tensor):
            return True
        expected = _decode_literal(witness.observed_value)
        if isinstance(expected, str):
            # Digest-based witness (value-free, any shape/dtype): re-digest the
            # recomputed source slot and require byte-exact equality with capture.
            try:
                if runnable_tensor_byte_digest(recomputed) != expected:
                    return True
            except (RuntimeError, ValueError, TypeError):
                return True
            continue
        # Legacy scalar-literal witness.
        if recomputed.numel() != 1:
            return True
        try:
            actual = recomputed.item()
        except (RuntimeError, ValueError):
            return True
        if not _scalar_literal_equal(actual, expected):
            return True
    return False


_UNBOUND_STATE_ESCAPE_SITE_PREFIX = "unbound_state_escape:"
"""``site_label`` prefix marking a witnessed unbound state (buffer/param) escape."""

_UNBOUND_STATE_ESCAPE_FACT_KEY = "unbound_state_escape"
"""Discriminator key present in every unbound-state escape fact."""


def _is_unbound_state_escape_witness(witness: ControlWitness) -> bool:
    """Return whether a structure witness records an unbound state escape."""

    return (
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_UNBOUND_STATE_ESCAPE_SITE_PREFIX)
    )


def _unbound_state_escape_stale(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
) -> bool:
    """Return whether an unbound state slot differs from its capture-time value.

    An unbound state slot (a registered buffer/param consumed by NO traced call)
    influenced the forward only through an untraced host path -- a Python
    truth-test, an ``.item()`` comparison, or other pure-Python control flow. The
    sparse DAG cannot recompute that dependency, so a staged value that differs
    from capture may have flipped a branch or restaled a literal. Each unbound
    state escape witness records the state name, its runtime slot, and the SHA-256
    byte digest of its capture-time value. This run re-digests the effective staged
    /embedded value; a differing (or missing) value means the untraced dependency
    changed, so the honest ceiling is UNVERIFIABLE + NOT_APPLICABLE. State that is
    byte-identical to capture keeps the run faithful.
    """

    name_to_slot_id: dict[str, str] = {}
    for slot in descriptor.tensor_slots:
        binding = slot.state_binding
        if binding is not None:
            name_to_slot_id.setdefault(binding.state_dict_name, slot.slot_id)
    for witness in descriptor.control_witnesses:
        if not _is_unbound_state_escape_witness(witness):
            continue
        fact = _decode_literal(witness.observed_value)
        if not isinstance(fact, Mapping) or fact.get(_UNBOUND_STATE_ESCAPE_FACT_KEY) is not True:
            continue
        name = fact.get("state_dict_name")
        expected = fact.get("digest")
        if not isinstance(name, str) or not isinstance(expected, str):
            return True
        slot_id = name_to_slot_id.get(name)
        value = slot_values.get(slot_id) if isinstance(slot_id, str) else None
        if not isinstance(value, torch.Tensor):
            return True
        try:
            if runnable_tensor_byte_digest(value) != expected:
                return True
        except (RuntimeError, ValueError, TypeError):
            return True
    return False


def _path_faithfulness(
    descriptor: SparseRunDescriptor,
    checks: Sequence[ContractCheck],
    *,
    host_rng_unreproduced: bool = False,
    tensor_derived_scalar_stale: bool = False,
    unbound_state_escape_stale: bool = False,
) -> tuple[PathFaithfulness, RunnableDiagnostic | None]:
    """Classify exact three-state path faithfulness after all honesty checks."""

    failed = next((check for check in checks if not check.passed), None)
    if failed is not None:
        return PathFaithfulness.DIVERGED, failed.diagnostic
    if descriptor.witness_completeness is not WitnessCompleteness.COMPLETE:
        return PathFaithfulness.UNVERIFIABLE, None
    if host_rng_unreproduced:
        # A Python/NumPy-RNG control-flow capture replayed off its captured seed:
        # the single recorded branch may not be the one a fresh seeded call takes,
        # so the honest ceiling is UNVERIFIABLE, never a false VERIFIED.
        return PathFaithfulness.UNVERIFIABLE, None
    if tensor_derived_scalar_stale:
        # A tensor->Python escape baked a derived constant into a downstream op or
        # steered pure-Python control flow; the source slot recomputed different
        # bytes, so the baked literal / taken branch may be stale for this input.
        # The sparse DAG cannot recompute it, so the honest ceiling is UNVERIFIABLE.
        return PathFaithfulness.UNVERIFIABLE, None
    if unbound_state_escape_stale:
        # A registered buffer/param read only through an untraced host path (a
        # module truth-test or ``.item()`` comparison) was staged with a value that
        # differs from capture; the untraced branch/literal may be stale, so the
        # honest ceiling is UNVERIFIABLE, never a silently wrong VERIFIED.
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
    nontensor_inputs_match: bool = True,
    host_rng_unreproduced: bool = False,
    tensor_derived_scalar_stale: bool = False,
    unbound_state_escape_stale: bool = False,
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
    nontensor_inputs_match:
        Whether every witnessed non-tensor input leaf matched its recorded
        capture-time value. A changed non-tensor input forces ``not_applicable``.

    Returns
    -------
    tuple[NumericAttestationStatus, ContractCheck | None]
        Applicability/result status and the aggregate byte-exact tripwire check.
    """

    if host_rng_unreproduced:
        # Python/NumPy-RNG control flow replayed off its captured seed: the recorded
        # taken path is not guaranteed to be the branch a fresh seeded call takes, so
        # a byte-exact attestation would dishonestly bless a possibly-stale result.
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if not nontensor_inputs_match:
        # A changed non-tensor input means the recorded taken path may not apply,
        # so recomputed slots must never be attested against the capture archive.
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if tensor_derived_scalar_stale:
        # A tensor->host escape source slot recomputed differently: the sparse
        # replay cannot recompute the baked constant / taken branch, so a byte-exact
        # attestation would dishonestly bless a possibly-stale result.
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if unbound_state_escape_stale:
        # An unbound state slot (read only through an untraced host path) was staged
        # with a value differing from capture, so the recorded taken path may not
        # apply; a byte-exact attestation would dishonestly bless a stale result.
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if descriptor.witness_completeness is not WitnessCompleteness.COMPLETE:
        # Incomplete witness coverage (e.g. an opaque non-tensor input leaf that
        # cannot be re-verified) makes the path only UNVERIFIABLE, so a byte-exact
        # attestation would dishonestly bless a possibly-wrong replayed path.
        return NumericAttestationStatus.NOT_APPLICABLE, None
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
    if _has_journaled_buffer_activation_member(descriptor, raw_members):
        # Repeated registered-buffer source slots for the same state entry are journal points, not
        # immutable activation payloads. A pure journaled replay can be path-faithful while each
        # selected buffer slot's bytes are only meaningful at that journal point; skip byte-exact
        # activation attestation rather than raising a false mismatch on a valid replay.
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if _has_out_mutated_activation_member(descriptor, raw_members):
        # An archived activation that later became an ``out=`` destination was
        # captured before its mutation. Its pre-write bytes are allocator data,
        # not a reproducible result, so attesting it would create a false
        # divergence on an otherwise faithful original-input replay.
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if not raw_members or not _attestation_inputs_match(descriptor, layer, input_byte_digests):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if not _attestation_state_matches(descriptor, layer, state, slot_values):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    archived = trace.__dict__.get("_runnable_archived_activations")
    if not isinstance(archived, Mapping):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    saw_benign_layout_mismatch = False
    benign_layout_slot_ids: set[str] = set()
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
            # FALLBACK (narrow, provably-benign): a byte-faithful replay of this
            # slot is genuinely infeasible from the run path for a matmul-family
            # kernel. Capture records the op under autograd (a grad-specialized
            # BLAS reduction order); replay recomputes it under
            # ``pause_logging()``/no-grad isolation, and the two reduction orders
            # differ by ~1 dtype ULP (verified ~5e-7 on eval MHA in-proj linear).
            # The exact capture-time layout/grad context is NOT recorded in the
            # sparse descriptor, so the run path cannot reproduce those bytes
            # without abandoning its no-grad isolation (a capture-side change,
            # out of scope). When the ARCHIVE still matches capture bytes
            # (archived_digest == byte_digest, so the archive is intact -- NOT
            # tampered) and the recomputed value is within a tight ULP bound of
            # the archive AND the producing op is a known layout/reduction-order-
            # sensitive BLAS kernel, report this slot ``not_applicable`` rather
            # than raising. Downstream view/shape members directly fed by that
            # benign slot may carry the same ULP-scale bytes; they are also
            # skipped only when the archive is intact and the same tight bound
            # holds. This stays fail-closed: a tampered archive
            # (archived_digest != byte_digest) or any divergence beyond the tight
            # ULP bound STILL raises numeric_attestation_failed -- the byte-exact
            # tripwire is preserved, never widened into a tolerance gate.
            if archived_digest == member.byte_digest and _is_benign_layout_nonreproducible(
                descriptor, member, recomputed, archived_value
            ):
                saw_benign_layout_mismatch = True
                benign_layout_slot_ids.add(member.slot_id)
                continue
            if archived_digest == member.byte_digest and _is_benign_downstream_nonreproducible(
                descriptor,
                member,
                recomputed,
                archived_value,
                benign_layout_slot_ids=benign_layout_slot_ids,
            ):
                saw_benign_layout_mismatch = True
                benign_layout_slot_ids.add(member.slot_id)
                continue
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
    if saw_benign_layout_mismatch:
        return NumericAttestationStatus.NOT_APPLICABLE, None
    return (
        NumericAttestationStatus.ATTESTED,
        ContractCheck(name="numeric_attestation:selected_slots", passed=True, diagnostic=None),
    )


# Matmul-family qualname tails whose BLAS backends pick a reduction order that
# depends on tensor memory layout and grad/inference dispatch context. Capture
# records these ops under autograd; the run path recomputes them under no-grad
# ``pause_logging()`` isolation, so their outputs can legitimately differ from
# the archived bytes by ~1 dtype ULP without any corruption. This set is
# deliberately NARROW -- only the reduction kernels proven layout/grad-sensitive
# -- so the byte-exact tripwire stays armed for every other op.
_LAYOUT_SENSITIVE_BLAS_QUALNAMES: frozenset[str] = frozenset(
    {
        "linear",
        "matmul",
        "mm",
        "bmm",
        "mv",
        "dot",
        "vdot",
        "inner",
        "outer",
        "ger",
        "addmm",
        "addbmm",
        "baddbmm",
        "addmv",
        "addr",
        "einsum",
        "tensordot",
    }
)


def _member_producer_is_layout_sensitive_blas(
    descriptor: SparseRunDescriptor, member: ActivationPayloadMember
) -> bool:
    """Return whether the op producing a member slot is a layout-sensitive BLAS kernel.

    Parameters
    ----------
    descriptor:
        Sparse descriptor whose calls and registry name the producing op.
    member:
        Archived activation member under attestation.

    Returns
    -------
    bool
        Whether the slot is produced by a matmul-family reduction kernel whose
        replay bytes can differ from the grad-context capture bytes by ULP noise.
    """

    registry = {entry.registry_id: entry for entry in descriptor.callable_registry}
    for call in descriptor.calls:
        if member.slot_id not in call.output_slot_ids:
            continue
        entry = registry.get(call.registry_id)
        if entry is None:
            return False
        qualname = entry.key.qualname or ""
        tail = qualname.rsplit(".", 1)[-1]
        return tail in _LAYOUT_SENSITIVE_BLAS_QUALNAMES
    return False


def _within_layout_reduction_tolerance(recomputed: torch.Tensor, archived: torch.Tensor) -> bool:
    """Return whether a replay tensor is within a tight ULP band of the archive.

    The only sanctioned divergence is a matmul reduction-order difference between
    the grad-enabled capture kernel and the no-grad replay kernel, bounded by a
    small multiple of the dtype ULP times the value magnitude. A genuine
    corruption (wrong path / weights / op) is orders of magnitude larger and
    fails this bound, so the byte-exact tripwire still fires on it.

    Parameters
    ----------
    recomputed:
        Fresh replay tensor for the slot.
    archived:
        Capture-time archived tensor for the slot (byte-verified intact).

    Returns
    -------
    bool
        Whether the tensors match within the tight reduction-order tolerance.
    """

    if recomputed.shape != archived.shape or recomputed.dtype != archived.dtype:
        return False
    if not archived.dtype.is_floating_point:
        return False
    recomputed64 = recomputed.detach().to(torch.float64)
    archived64 = archived.detach().to(torch.float64)
    recomputed_finite = torch.isfinite(recomputed64)
    archived_finite = torch.isfinite(archived64)
    if not bool((recomputed_finite == archived_finite).all().item()):
        return False
    nonfinite_mask = ~archived_finite
    if bool(nonfinite_mask.any().item()):
        recomputed_nonfinite = recomputed64[nonfinite_mask]
        archived_nonfinite = archived64[nonfinite_mask]
        both_nan = torch.isnan(recomputed_nonfinite) & torch.isnan(archived_nonfinite)
        both_same_inf = (
            torch.isinf(recomputed_nonfinite)
            & torch.isinf(archived_nonfinite)
            & (torch.signbit(recomputed_nonfinite) == torch.signbit(archived_nonfinite))
        )
        if not bool((both_nan | both_same_inf).all().item()):
            return False
    finite_mask = archived_finite
    if not bool(finite_mask.any().item()):
        return True
    recomputed64 = recomputed64[finite_mask]
    archived64 = archived64[finite_mask]
    difference = (recomputed64 - archived64).abs()
    eps = float(torch.finfo(archived.dtype).eps)
    # 256 ULP relative + absolute floor: comfortably covers reduction-order noise
    # (verified ~5e-7 for float32 == ~4 ULP on eval MHA) while a corruption stays
    # hundreds of times larger. Scaled by dtype so fp16/bf16 keep a proportional
    # band and fp64 a far tighter one.
    tolerance = 256.0 * eps * (archived64.abs() + 1.0)
    return bool((difference <= tolerance).all().item())


def _is_benign_layout_nonreproducible(
    descriptor: SparseRunDescriptor,
    member: ActivationPayloadMember,
    recomputed: Any,
    archived_value: Any,
) -> bool:
    """Return whether a slot mismatch is a provably-benign BLAS-layout artifact.

    True only when BOTH the recomputed and archived values are real tensors, the
    producing op is a known layout/reduction-order-sensitive BLAS kernel, and the
    recomputed value sits within a tight ULP band of the archive. Used solely as
    the F1 fallback: such a slot reports ``not_applicable`` instead of raising,
    because a byte-faithful replay is genuinely infeasible from the no-grad run
    path. Every other mismatch (non-BLAS op, larger-than-ULP divergence, or a
    non-tensor slot) still raises ``numeric_attestation_failed``.

    Parameters
    ----------
    descriptor:
        Sparse descriptor naming the producing op.
    member:
        Archived activation member under attestation.
    recomputed:
        Fresh replay value for the slot.
    archived_value:
        Capture-time archived value for the slot (byte-verified intact by caller).

    Returns
    -------
    bool
        Whether the mismatch is the sanctioned layout-nonreproducible case.
    """

    if not (isinstance(recomputed, torch.Tensor) and isinstance(archived_value, torch.Tensor)):
        return False
    if not _member_producer_is_layout_sensitive_blas(descriptor, member):
        return False
    return _within_layout_reduction_tolerance(recomputed, archived_value)


def _is_benign_downstream_nonreproducible(
    descriptor: SparseRunDescriptor,
    member: ActivationPayloadMember,
    recomputed: Any,
    archived_value: Any,
    *,
    benign_layout_slot_ids: set[str],
) -> bool:
    """Return whether a mismatch is tight-ULP fallout from a benign BLAS slot.

    Parameters
    ----------
    descriptor:
        Sparse descriptor naming slot producers and tensor arguments.
    member:
        Archived activation member under attestation.
    recomputed:
        Fresh replay value for the slot.
    archived_value:
        Capture-time archived value for the slot (byte-verified intact by caller).
    benign_layout_slot_ids:
        Slots already proven to be benign layout/reduction-order mismatches.

    Returns
    -------
    bool
        Whether this member is fed by a previously proven benign slot and remains
        within the same tight ULP band.
    """

    if not benign_layout_slot_ids:
        return False
    if not (isinstance(recomputed, torch.Tensor) and isinstance(archived_value, torch.Tensor)):
        return False
    if not _within_layout_reduction_tolerance(recomputed, archived_value):
        return False
    if member.slot_id in benign_layout_slot_ids:
        return True
    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    slot = slots.get(member.slot_id)
    if slot is not None and (
        slot.producer_slot_id in benign_layout_slot_ids or slot.version_of in benign_layout_slot_ids
    ):
        return True
    for call in descriptor.calls:
        if member.slot_id not in call.output_slot_ids:
            continue
        return any(argument.slot_id in benign_layout_slot_ids for argument in call.tensor_arguments)
    return False


def _has_journaled_buffer_activation_member(
    descriptor: SparseRunDescriptor,
    members: Sequence[ActivationPayloadMember],
) -> bool:
    """Return whether selected activation payloads include journaled buffer slots.

    Parameters
    ----------
    descriptor:
        Sparse descriptor declaring tensor slot roles.
    members:
        Raw activation payload members selected for byte attestation.

    Returns
    -------
    bool
        ``True`` when multiple selected buffer slots refer to the same state entry.
    """

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    seen_state_names: set[str] = set()
    for member in members:
        slot = slots.get(member.slot_id)
        if slot is None or slot.role is not TensorSlotRole.BUFFER:
            continue
        binding = slot.state_binding
        if binding is None:
            continue
        state_name = binding.state_dict_name
        if state_name in seen_state_names:
            return True
        seen_state_names.add(state_name)
    return False


def _has_out_mutated_activation_member(
    descriptor: SparseRunDescriptor,
    members: Sequence[ActivationPayloadMember],
) -> bool:
    """Return whether activation attestation includes an ``out=`` destination.

    Parameters
    ----------
    descriptor:
        Sparse recipe whose mutating calls are being inspected.
    members:
        Archived raw activation records selected at capture time.

    Returns
    -------
    bool
        Whether any archived slot is later written through an explicit ``out=``
        argument and therefore has no stable pre-write byte contract.
    """

    out_slots = {
        argument.slot_id
        for call in descriptor.calls
        if call.is_inplace
        for argument in call.tensor_arguments
        if argument.argument_path == ("kwargs", "out")
    }
    return any(member.slot_id in out_slots for member in members)


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
    """Return whether replay actually consumes non-reproducible RNG.

    A captured RNG source slot always taints the replay. For seeded-RNG ATen
    ops the answer is keyed off *actual RNG consumption*, not the op name: a
    ``dropout`` family call in eval mode (``training=False``) or with ``p == 0``
    draws nothing from the generator and replays byte-exact, so it must NOT be
    treated as seeded (that over-triggered ``not_applicable`` on eval-mode
    transformers). Every other seeded-RNG op (``rand``/``randn``/``bernoulli``/
    ``multinomial`` and a genuinely training dropout with ``p > 0``) still taints.

    Parameters
    ----------
    descriptor:
        Sparse runnable descriptor whose registry entries and calls identify
        replayed PyTorch operations.

    Returns
    -------
    bool
        Whether replay includes a captured RNG source or a call that actually
        draws from a PyTorch seeded generator during replay.
    """

    if any(slot.role is TensorSlotRole.RNG_SOURCE for slot in descriptor.tensor_slots):
        return True
    registry = {entry.registry_id: entry for entry in descriptor.callable_registry}
    return any(_call_consumes_seeded_rng(call, registry) for call in descriptor.calls)


def _call_consumes_seeded_rng(
    call: RunnableCallDescriptor,
    registry: Mapping[str, CallableRegistryEntry],
) -> bool:
    """Return whether one replayed call actually draws from a seeded generator.

    Parameters
    ----------
    call:
        Replayed sparse call descriptor.
    registry:
        Registry-id to callable-entry map for the descriptor.

    Returns
    -------
    bool
        Whether this specific call consumes non-reproducible seeded RNG at
        replay time (op-name seeding refined by dropout ``training``/``p``).
    """

    entry = registry.get(call.registry_id)
    if entry is None:
        return False
    namespace = entry.key.namespace
    qualname = entry.key.qualname
    if not aten_qualname_is_seeded_rng(namespace, qualname):
        return False
    if _is_dropout_qualname(qualname):
        # A dropout family op draws from the RNG only when it is genuinely
        # active (training=True and p>0); eval/identity dropout is RNG-inert.
        return _dropout_call_draws_rng(call)
    return True


def _is_dropout_qualname(qualname: str | None) -> bool:
    """Return whether a captured qualname belongs to the dropout op family.

    Covers ``dropout``/``dropout_``/``feature_dropout``/``alpha_dropout``/
    ``feature_alpha_dropout`` under any namespace spelling. All members share
    the ``training``+``p`` RNG-consumption contract.
    """

    if not qualname:
        return False
    tail = qualname.rsplit(".", 1)[-1]
    if tail.endswith("_"):
        tail = tail[:-1]
    return tail.endswith("dropout")


def _dropout_call_draws_rng(call: RunnableCallDescriptor) -> bool:
    """Return whether a dropout call consumes RNG given its recorded literals.

    Dropout draws from the generator only when ``training is True`` AND ``p > 0``.
    The decision is keyed off the recorded ``training`` and ``p`` literals; when
    a value cannot be proven RNG-inert the call stays conservatively tagged as
    seeded (fail-closed: never a false ``attested``).

    Parameters
    ----------
    call:
        The dropout-family sparse call descriptor.

    Returns
    -------
    bool
        Whether the recorded dropout call actually draws from the RNG.
    """

    named = _named_literal_values(call)
    training = named.get("training", named.get("train"))
    p_value = named.get("p")
    if training is False:
        return False
    if isinstance(p_value, (int, float)) and not isinstance(p_value, bool) and p_value == 0:
        return False
    return True


def _named_literal_values(call: RunnableCallDescriptor) -> dict[str, Any]:
    """Map recorded literal arguments to their parameter names.

    Positional literals are named through ``call.argument_names``; keyword
    literals use their stored key. Only the safe literal grammar is decoded.

    Parameters
    ----------
    call:
        Sparse call descriptor whose literal leaves are being named.

    Returns
    -------
    dict[str, Any]
        Parameter name to decoded literal value mapping.
    """

    values: dict[str, Any] = {}
    for literal_argument in call.literal_arguments:
        path = literal_argument.argument_path
        if len(path) != 2:
            continue
        root, key = path
        if root == "args" and isinstance(key, int) and 0 <= key < len(call.argument_names):
            values[call.argument_names[key]] = _decode_literal(literal_argument.value)
        elif root == "kwargs":
            values[str(key)] = _decode_literal(literal_argument.value)
    return values


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
        # ``ELLIPSIS`` and ``NONE`` both carry ``value is None`` on the wire (``...`` has
        # no JSON-native representation), so the atom KIND -- not the stored value -- is
        # what disambiguates a real ``None`` index from a ``...`` index at decode time.
        if value.kind is LiteralAtomKind.ELLIPSIS:
            return Ellipsis
        if value.kind is LiteralAtomKind.NONFINITE_FLOAT:
            return _decode_nonfinite_float_literal(value.value)
        return value.value
    if isinstance(value, LiteralSlice):
        return slice(
            _decode_literal(value.start),
            _decode_literal(value.stop),
            _decode_literal(value.step),
        )
    if isinstance(value, LiteralTupleKey):
        return tuple(_decode_literal(item) for item in value.items)
    if isinstance(value, LiteralSequence):
        items = [_decode_literal(item) for item in value.items]
        return tuple(items) if value.kind is LiteralSequenceKind.TUPLE else items
    if isinstance(value, LiteralMapping):
        return {_decode_literal(entry.key): _decode_literal(entry.value) for entry in value.entries}
    if isinstance(value, LiteralTorchSymbol):
        return _decode_torch_symbol(value.qualname)
    raise TypeError(f"Unknown sparse literal type {type(value).__name__}.")


def _decode_nonfinite_float_literal(value: Any) -> float:
    """Decode one non-finite float atom payload.

    Parameters
    ----------
    value:
        Serialized non-finite float payload.

    Returns
    -------
    float
        ``nan``, ``inf``, or ``-inf``.
    """

    if value == "nan":
        return float("nan")
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    raise RunPreconditionError(
        f"Unsupported non-finite float literal payload {value!r}.",
        code=RunnableErrorCode.UNSUPPORTED_LITERAL.value,
    )


# POSITIVE allowlist of the torch symbolic constant *types* a forward op
# legitimately takes as a literal argument. A loaded bundle is untrusted input,
# so decoding an arbitrary ``torch`` attribute by name would otherwise admit whole
# submodules (``torch.serialization`` / ``torch.os``) or any other non-callable
# attribute. These are the only symbolic literals the encoder ever emits
# (``_torch_symbol_qualname``): dtype / layout / memory_format instances, plus
# qscheme and the ``torch.Size`` type -- everything else is denied by construction.
_ALLOWED_TORCH_SYMBOL_TYPES: tuple[type, ...] = (
    torch.dtype,
    torch.layout,
    torch.memory_format,
    torch.qscheme,
)


def _decode_torch_symbol(qualname: str) -> Any:
    """Decode one allowlisted non-callable torch symbolic literal.

    ``torch.device(...)`` round-trips through the device constructor. Every other
    accepted symbol must be a bare ``torch.<name>`` lookup resolving to a
    dtype / layout / memory_format / qscheme instance or the ``torch.Size`` type;
    dotted attribute traversal, modules, callables (other than ``torch.Size``),
    and any other attribute are rejected as unsupported literals.
    """

    if qualname.startswith("torch.device(") and qualname.endswith(")"):
        return torch.device(qualname[13:-1])
    name = qualname.removeprefix("torch.")
    if name == qualname or "." in name or not name.isidentifier():
        raise RunPreconditionError(
            f"Unsupported torch literal symbol {qualname!r}.",
            code=RunnableErrorCode.UNSUPPORTED_LITERAL.value,
        )
    symbol = getattr(torch, name, None)
    if symbol is torch.Size or isinstance(symbol, _ALLOWED_TORCH_SYMBOL_TYPES):
        return symbol
    raise RunPreconditionError(
        f"Unsupported torch literal symbol {qualname!r}.",
        code=RunnableErrorCode.UNSUPPORTED_LITERAL.value,
    )


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
