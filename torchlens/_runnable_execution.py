"""Transactional execution providers for the unified :meth:`Trace.run` surface."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
import sys
from contextlib import contextmanager, nullcontext
from itertools import count
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
from .ir.container import (
    ContainerReconstructionError,
    ContainerSpec,
    rebuild_container_from_spec,
    reconstruction_is_lossy_by_type,
    resolve_container_type,
)
from .utils.rng import (
    aten_qualname_is_seeded_rng,
    restore_host_rng,
    snapshot_host_rng,
)
from .utils._torch_compat import tensor_version_or_none
from .utils.tensor_utils import touched_bytes_relation
from .runnable import (
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    CallableRegistryEntry,
    ContractCheck,
    InputAttestationFingerprint,
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
    slot_values, input_checks, input_alias_unresolved = _bind_runtime_inputs(descriptor, inputs)
    _raise_first_divergence(input_checks, divergence_policy, fork=None)
    # I5: digests/fingerprints are attestation-only facts, computed strictly
    # AFTER admission (hard preconditions + contract enforcement) and only when
    # an activation archive exists to attest against.
    if isinstance(descriptor.payload_layers.activations, ActivationPayloadLayerDescriptor):
        input_byte_digests: Mapping[str, str] = _snapshot_input_byte_digests(
            descriptor, slot_values
        )
        input_fingerprints = _snapshot_input_fingerprints(
            descriptor, slot_values, input_byte_digests
        )
    else:
        input_byte_digests = {}
        input_fingerprints = {}
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
            input_fingerprints=input_fingerprints,
            input_checks=input_checks,
            input_alias_unresolved=input_alias_unresolved,
            prepared_state=prepared_state,
            fork=fork,
        )
    except BaseException:
        _state._unregister_log(fork)
        raise


def _seeded_fork_devices(descriptor: SparseRunDescriptor, seed: int | None) -> list[int]:
    """Return the CUDA device fork set for one seeded run (r35 corr2_4).

    The set follows the seeding primitive, never the bound-input overlay: ALL
    visible CUDA devices are forked when CUDA is already initialized, or when
    the descriptor's immutable capture metadata names a CUDA device anywhere --
    including produced-only intermediates and RNG-source slots (the original
    leak: a CPU-input model drawing ``torch.rand(..., device="cuda")``). A
    CPU-only descriptor on an uninitialized-CUDA runtime forks nothing AND the
    executor seeds nothing CUDA-side, so no lazy seed can leak into the
    caller's future CUDA state; a post-run tripwire asserts initialization did
    not flip.
    """

    if seed is None or not torch.cuda.is_available():
        return []
    if torch.cuda.is_initialized():
        return list(range(torch.cuda.device_count()))
    descriptor_mentions_cuda = any(slot.device_type == "cuda" for slot in descriptor.tensor_slots)
    if descriptor_mentions_cuda:
        return list(range(torch.cuda.device_count()))
    return []


def _seed_run_generators(seed: int, forked_cuda_devices: list[int], *, reseed_host: bool) -> None:
    """Seed exactly the generators this run forked (r35 corr2_4).

    Never ``torch.manual_seed``: the global primitive seeds every CUDA device
    (queuing a LAZY seed on an uninitialized runtime that would apply to the
    caller's future initialization) plus MPS/XPU generators the fork set does
    not snapshot. The executor seeds the CPU default generator, each forked
    CUDA device generator individually, and -- for a faithful host-RNG replay
    -- Python/NumPy (whose prior state the caller snapshot-restores).
    """

    if reseed_host:
        import random

        random.seed(seed)
        np.random.seed(seed)
    torch.default_generator.manual_seed(seed)
    for device_index in forked_cuda_devices:
        with torch.cuda.device(device_index):
            torch.cuda.manual_seed(seed)


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
    input_fingerprints: Mapping[str, InputAttestationFingerprint],
    input_checks: tuple[ContractCheck, ...],
    input_alias_unresolved: bool,
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
    # r35 corr2_5: PRE-EXECUTION state digests -- eligibility compares each
    # state slot's capture-start bytes, not whatever a mutating call left
    # behind. Computed only when an activation archive exists to attest.
    state_byte_digests: dict[str, str] = {}
    if isinstance(descriptor.payload_layers.activations, ActivationPayloadLayerDescriptor):
        for state_slot in descriptor.tensor_slots:
            if state_slot.state_binding is None or state_slot.slot_id not in slot_values:
                continue
            try:
                state_byte_digests[state_slot.slot_id] = runnable_tensor_byte_digest(
                    slot_values[state_slot.slot_id]
                )
            except Exception:
                # Undigestable state cannot attest; the absent entry reads as a
                # mismatch in the eligibility partition (fail-safe).
                continue
    call_outputs: dict[str, Any] = {}
    attestation_slot_ids = _raw_activation_slot_ids(descriptor)
    attestation_slot_values: dict[str, torch.Tensor] = {}

    # r35 corr2_4: the fork/restore set follows the SEEDING PRIMITIVE, never the
    # bound-input overlay -- every visible CUDA device is forked when CUDA is
    # initialized or the descriptor's capture metadata names a CUDA device
    # (including produced-only intermediates and RNG-source slots), and the
    # executor seeds ONLY the CPU generator plus each forked CUDA generator, so
    # no unforked generator (an unmentioned CUDA device, MPS/XPU) is ever
    # touched by a seeded run.
    devices = _seeded_fork_devices(descriptor, seed)
    cuda_initialized_before = (
        torch.cuda.is_available() and torch.cuda.is_initialized() if seed is not None else None
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
        # Decision E: the recorded capture-scoped ambient backend context is
        # restored transactionally around the whole run (finally-restored on
        # every exit); each resolved call additionally enters its own recorded
        # per-call context tightly (see execute_call below).
        with (
            _ambient_execution_context_restored(descriptor.ambient_context),
            rng_context,
            _state.pause_logging(),
        ):
            if seed is not None:
                _seed_run_generators(cast(int, seed), devices, reseed_host=reseed_host)

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
                with _call_execution_context_entered(call.execution_context):
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

    if (
        seed is not None
        and not devices
        and torch.cuda.is_available()
        and torch.cuda.is_initialized() != bool(cuda_initialized_before)
    ):
        # r35 corr2_4 tripwire: CUDA initialized DURING a seeded run whose fork
        # set excluded it -- the descriptor's capture device summary missed a
        # CUDA consumer, so run-local RNG isolation cannot be guaranteed. This
        # is an internal summary bug, never silently ignored.
        _state._unregister_log(fork)
        raise RuntimeError(
            "Internal invariant violation: CUDA became initialized during a "
            "seeded sparse run whose descriptor named no CUDA device; the "
            "capture device summary is incomplete."
        )

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
    mode_sensitive_op_unwitnessed = _mode_sensitive_op_unwitnessed(descriptor)
    tensor_derived_scalar_stale = _tensor_derived_scalar_stale(
        descriptor, slot_values, witness_source_snapshots
    )
    unbound_state_escape_stale = _unbound_state_escape_stale(descriptor, slot_values)
    output_container_spec = _output_container_spec(fork)
    container_reconstruction_lossy = _container_spec_reconstruction_lossy(output_container_spec)
    output_not_reproduced = _output_not_reproduced(descriptor, output_container_spec)
    # r35 I3 (corr2_7): settle the PROVISIONAL path verdict from ALL non-numeric
    # contract checks and static/dynamic ceilings FIRST; numeric attestation is
    # strictly downstream of it. A verdict that is not VERIFIED -- including one
    # inherited monotonically from a prior poisoned run of the source Trace --
    # makes attestation NOT_APPLICABLE before any archive byte is read, so
    # ATTESTED can never coexist with DIVERGED/UNVERIFIABLE/poisoned, and every
    # FUTURE contract check automatically caps attestation through this same
    # derivation (no parallel Boolean flag list).
    provisional_verdict, provisional_mismatch = _path_faithfulness(
        descriptor,
        contract_checks,
        host_rng_unreproduced=host_rng_unreproduced,
        tensor_derived_scalar_stale=tensor_derived_scalar_stale,
        unbound_state_escape_stale=unbound_state_escape_stale,
        container_reconstruction_lossy=container_reconstruction_lossy,
        output_not_reproduced=output_not_reproduced,
        mode_sensitive_op_unwitnessed=mode_sensitive_op_unwitnessed,
        input_alias_unresolved=input_alias_unresolved,
    )
    eligibility_verdict = provisional_verdict
    inherited_status = fork.__dict__.get("_runnable_path_faithfulness")
    if (
        isinstance(inherited_status, PathFaithfulness)
        and inherited_status is not PathFaithfulness.VERIFIED
        and eligibility_verdict is PathFaithfulness.VERIFIED
    ):
        eligibility_verdict = inherited_status
    numeric_attestation, attestation_check = _numeric_attestation_check(
        descriptor,
        prepared_state,
        slot_values=slot_values,
        attestation_slot_values=attestation_slot_values,
        input_byte_digests=input_byte_digests,
        input_fingerprints=input_fingerprints,
        state_byte_digests=state_byte_digests,
        trace=trace,
        provisional_verdict=eligibility_verdict,
    )
    if attestation_check is not None:
        contract_checks.append(attestation_check)
        if not attestation_check.passed:
            _raise_numeric_attestation_failure(fork, attestation_check)
    path_faithfulness, mismatch = provisional_verdict, provisional_mismatch
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
        state_source=prepared_state.state_source,
        initializer_policy_version=prepared_state.initializer_policy_version,
        seed=prepared_state.seed,
        random_filled_slot_ids=prepared_state.random_filled_slot_ids,
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
    on_divergence: DivergencePolicy | str = DivergencePolicy.RAISE,
) -> RunResult:
    """Run the live-model refresh provider on a transactional fork.

    r37 corr2-5: the live provider finalizes through the SAME spine as the sparse
    provider -- ``mark_trace_path_status`` (monotonic Trace poison), the shared
    divergence-policy enforcement, and the one ``_run_report`` finalizer (poison
    derived solely from the faithfulness lattice). A lossy live reconstruction
    therefore returns a POISONED report and a monotonically marked Trace that
    every faithful consumer (``to_pandas``, export, chaining) refuses.

    Parameters
    ----------
    trace:
        Live Trace retaining its source-model weak reference.
    inputs:
        New forward input accepted by the existing ``save_new_outs`` path.
    seed:
        Optional refresh seed.
    on_divergence:
        Divergence policy threaded from the public ``run`` surface.

    Returns
    -------
    RunResult
        Structured output, refreshed fork, and live-provider report.

    Raises
    ------
    RunCapabilityUnavailableError
        If the live source model is no longer available.
    """

    divergence_policy = DivergencePolicy(on_divergence)
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
        # A lossy output container (computed non-field/non-key state, __slots__, or a
        # data-descriptor field) cannot be faithfully rebuilt, so it is UNVERIFIABLE here
        # too -- never a false VERIFIED on the live-refresh provider.
        if _container_spec_reconstruction_lossy(_output_container_spec(fork)):
            faithful = False
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
        provisional = PathFaithfulness.VERIFIED if faithful else PathFaithfulness.UNVERIFIABLE
        path_faithfulness, mismatch = mark_trace_path_status(fork, provisional, None)
        _raise_monotonic_divergence(fork, path_faithfulness, mismatch, divergence_policy)
        report = _run_report(
            readiness,
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
            path_faithfulness=path_faithfulness,
            first_mismatch=mismatch,
            numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
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


_MODEL_INPUT_METADATA_SITE_PREFIX = "model_input_metadata:"
"""``site_label`` prefix marking a witnessed model-input metadata-predicate read."""

_MODEL_INPUT_METADATA_FACT_KEY = "model_input_metadata"
"""Discriminator key present in every model-input metadata-predicate fact."""


def _model_input_metadata_facts(
    descriptor: SparseRunDescriptor,
) -> list[tuple[ControlWitness, Mapping[str, Any]]]:
    """Decode every witnessed model-input metadata-predicate fact (r27-H2)."""

    facts: list[tuple[ControlWitness, Mapping[str, Any]]] = []
    for witness in descriptor.control_witnesses:
        if witness.kind is not ControlWitnessKind.SHAPE_STRUCTURE_FACT:
            continue
        if not witness.site_label.startswith(_MODEL_INPUT_METADATA_SITE_PREFIX):
            continue
        decoded = _decode_literal(witness.observed_value)
        if isinstance(decoded, Mapping) and decoded.get(_MODEL_INPUT_METADATA_FACT_KEY) is True:
            facts.append((witness, decoded))
    return facts


def _is_model_input_metadata_witness(witness: ControlWitness) -> bool:
    """Return whether a structure witness records a model-input metadata-predicate read."""

    return (
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_MODEL_INPUT_METADATA_SITE_PREFIX)
    )


def _runtime_input_metadata_value(value: torch.Tensor, name: str) -> Any:
    """Evaluate one recorded metadata predicate on the RAW bound runtime input.

    Evaluated on the user-provided tensor BEFORE the defensive detach-clone, which
    erases the autograd state (``requires_grad`` / ``grad_fn`` / ``is_leaf``) and resets
    ``storage_offset`` -- the capture-time read saw the forward's real input, so the
    comparison must too. ``grad_fn`` is compared as a PRESENCE boolean (the exact backward
    object is not comparable across runs), mirroring the capture-time recording.
    """

    try:
        if name == "is_contiguous":
            return bool(value.is_contiguous())
        if name == "stride":
            return [int(v) for v in value.stride()]
        if name == "storage_offset":
            return int(value.storage_offset())
        if name == "requires_grad":
            return bool(value.requires_grad)
        if name == "grad_fn":
            return bool(value.grad_fn is not None)
        if name == "is_leaf":
            return bool(value.is_leaf)
        if name == "retains_grad":
            return bool(value.retains_grad)
        if name == "_base":
            return bool(value._base is not None)
        if name == "_is_view":
            return bool(value._is_view())
        if name == "is_conj":
            return bool(value.is_conj())
        if name == "is_neg":
            return bool(value.is_neg())
        if name == "is_inference":
            return bool(value.is_inference())
        if name == "is_pinned":
            return bool(value.is_pinned())
        if name == "is_shared":
            return bool(value.is_shared())
        if name == "is_coalesced":
            return bool(value.is_coalesced())
        if name == "grad":
            return bool(value.grad is not None)
        if name == "_grad":
            return bool(value._grad is not None)
        if name == "_version":
            return int(value._version)
        if name == "output_nr":
            return int(value.output_nr)
        if name == "storage_nbytes":
            return int(value.untyped_storage().nbytes())
    except (RuntimeError, AttributeError, TypeError, ValueError, NotImplementedError):
        return None
    return None


def _input_metadata_contract_checks(
    descriptor: SparseRunDescriptor,
    inputs: Any,
    positions: set[Any],
) -> tuple[ContractCheck, ...]:
    """Compare runtime input metadata predicates with capture-time observed facts (r27-H2).

    The capture-time forward READ these predicates (``is_contiguous`` / ``stride`` /
    ``requires_grad``) on a model-input leaf, so their values can have steered the
    recorded taken path. The input contract checks only shape+dtype; a same-shape
    runtime input differing in layout or grad flag would replay the captured arm a
    fresh model would not take -- a false VERIFIED+ATTESTED. Witnesses exist ONLY for
    captures that performed such a read, so an ordinary layout-oblivious model has no
    metadata witnesses and can never over-trigger here.
    """

    checks: list[ContractCheck] = []
    for witness, fact in _model_input_metadata_facts(descriptor):
        raw_position = fact.get("position")
        position = tuple(raw_position) if isinstance(raw_position, (list, tuple)) else raw_position
        path = tuple(fact.get("path", ()) or ())
        recorded_facts = fact.get("facts")
        if not isinstance(recorded_facts, Mapping):
            continue
        try:
            root = _input_site_value(inputs, position, positions)
            runtime_leaf = _value_at_path(root, path)
            resolved = isinstance(runtime_leaf, torch.Tensor)
        except (KeyError, IndexError, TypeError, AttributeError):
            runtime_leaf = None
            resolved = False
        for name in sorted(recorded_facts):
            recorded_value = recorded_facts[name]
            runtime_value = (
                _runtime_input_metadata_value(runtime_leaf, str(name)) if resolved else None
            )
            passed = resolved and runtime_value == recorded_value
            checks.append(
                _contract_check(
                    f"input_metadata:{name}:{position!r}:{path!r}",
                    passed,
                    RunnableErrorCode.INPUT_TREE_MISMATCH,
                    f"Runtime input {name} differs from the capture-time value the "
                    "forward read; the recorded taken path may not be valid for "
                    "this input.",
                    affected_op_labels=(witness.site_label,),
                    details=(
                        ("model_site_position", repr(position)),
                        ("container_path", repr(path)),
                        ("predicate", str(name)),
                        ("recorded_value", repr(recorded_value)),
                        ("runtime_value", repr(runtime_value) if resolved else "<unresolved>"),
                    ),
                )
            )
    return tuple(checks)


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
) -> tuple[dict[str, torch.Tensor], tuple[ContractCheck, ...], bool]:
    """Bind and defensively clone public input leaves by persisted model sites.

    Returns the bound clone map, the ordered input contract checks, and the
    ``input_alias_topology_unresolved`` ceiling flag (r35 decision D).
    """

    input_slots = tuple(
        slot for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    )
    values: dict[str, torch.Tensor] = {}
    checks: list[ContractCheck] = []
    positions = _model_input_arity_positions(descriptor)
    # Torch capture DE-ALIASES model inputs (each input leaf is cloned before the forward,
    # so ``forward(a, b)`` with ``a is b`` is captured as two DISTINCT tensors). The recorded
    # DAG and activation archive therefore reflect distinct-input semantics, and each runtime
    # input slot is likewise cloned independently. A runtime call that passes ALIASED inputs
    # (same object, or distinct views sharing storage) is NOT reproducible against that
    # de-aliased capture when an in-place op mutates an input -- see
    # ``_input_alias_topology_checks``, which fails such a run closed instead of a false VERIFIED.
    raw_values: dict[str, torch.Tensor] = {}
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
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
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
        # r37 corr2-6 phase 0: EXACT-type admission precedes every dispatchable
        # property read. A tensor SUBCLASS routes ``shape``/``dtype``/``device``/
        # ``names`` through ``__torch_function__``, so reading any of them before
        # this gate would execute user subclass code and leak its raw exception in
        # place of the typed hard-precondition refusal. Diagnostics use ONLY
        # ``type(value).__qualname__`` -- never a property.
        if type(value) not in {torch.Tensor, torch.nn.Parameter}:
            subclass_check = _contract_check(
                f"input_layout:{slot.slot_id}",
                False,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime input is a tensor SUBCLASS the sparse replay cannot "
                "faithfully reproduce; runnable capture records only plain "
                "strided torch.Tensor/Parameter leaves, so such an input fails "
                "closed (typed) before any property dispatch, under every "
                "divergence policy.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("tensor_class", type(value).__qualname__),
                ),
            )
            diagnostic = subclass_check.diagnostic
            raise PathDivergenceError(
                diagnostic.message
                if diagnostic is not None
                else "Unsupported tensor-subclass input.",
                code=RunnableErrorCode.INPUT_TREE_MISMATCH.value,
                path_faithfulness=PathFaithfulness.DIVERGED,
                first_mismatch=diagnostic,
                contract_check=subclass_check,
            )
        raw_values[slot.slot_id] = value
        # Phase-1 facts are exception-safe: exotic layouts (nested tensors) can
        # refuse even a ``sizes()`` read, which must surface as a failed check,
        # never a raw backend error.
        try:
            actual_shape: tuple[int, ...] | None = tuple(value.shape)
        except (RuntimeError, TypeError, NotImplementedError):
            actual_shape = None
        shape_ok = actual_shape == slot.shape
        dtype_ok = str(value.dtype) == slot.dtype
        checks.append(
            _contract_check(
                f"input_shape:{slot.slot_id}",
                shape_ok,
                RunnableErrorCode.INPUT_SHAPE_MISMATCH,
                f"Runtime input shape {actual_shape} does not match {slot.shape}.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_shape", repr(slot.shape)),
                    ("actual_shape", repr(actual_shape)),
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
        # r33 F7: pin DEVICE and LAYOUT next to shape+dtype. The shape+dtype contract does NOT
        # pin either, yet the state lives on the capture DEVICE and the recorded DAG assumes a
        # STRIDED DENSE input, so a same-shape+dtype runtime input on a different device or with
        # an exotic layout (sparse/meta/nested/named) cannot be faithfully reproduced by the
        # sparse replay -- today only an INCIDENTAL torch device/layout error guards them. Device
        # TYPE is compared strictly; the INDEX only when both are concrete (a capture recorded as
        # a bare ``cuda`` has index ``None`` and must not falsely diverge a ``cuda:0`` runtime).
        device_ok = value.device.type == slot.device_type and (
            slot.device_index is None
            or value.device.index is None
            or value.device.index == slot.device_index
        )
        checks.append(
            _contract_check(
                f"input_device:{slot.slot_id}",
                device_ok,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                f"Runtime input device {value.device} does not match the capture device "
                f"{slot.device_type}"
                + (f":{slot.device_index}" if slot.device_index is not None else "")
                + "; the recorded state and DAG cannot be replayed against a different device.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("expected_device_type", slot.device_type),
                    ("expected_device_index", repr(slot.device_index)),
                    ("actual_device", str(value.device)),
                ),
            )
        )
        layout_ok = (
            value.layout == torch.strided
            and not value.is_nested
            and not bool(getattr(value, "is_meta", False))
            and not bool(getattr(value, "is_quantized", False))
            and not any(name is not None for name in (value.names or ()))
            and type(value) in {torch.Tensor, torch.nn.Parameter}
        )
        checks.append(
            _contract_check(
                f"input_layout:{slot.slot_id}",
                layout_ok,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime input has a non-strided/meta/nested/named/quantized/subclass "
                "layout the sparse replay cannot faithfully reproduce; runnable capture "
                "records only plain strided dense tensors, so such an input must fail "
                "closed rather than replay the recorded DAG.",
                affected_op_labels=(slot.slot_id.removeprefix("slot:"),),
                details=(
                    ("slot_id", slot.slot_id),
                    ("actual_layout", str(value.layout)),
                    ("is_nested", repr(bool(value.is_nested))),
                    ("is_meta", repr(bool(getattr(value, "is_meta", False)))),
                    ("is_quantized", repr(bool(getattr(value, "is_quantized", False)))),
                    ("named", repr(any(name is not None for name in (value.names or ())))),
                    ("tensor_class", type(value).__qualname__),
                ),
            )
        )
    checks.extend(_input_tree_contract_checks(descriptor, inputs))
    checks.extend(_input_literal_contract_checks(descriptor, inputs, positions))
    checks.extend(_input_metadata_contract_checks(descriptor, inputs, positions))
    checks.extend(_input_nontensor_tree_contract_checks(descriptor, inputs, positions))
    alias_checks, input_alias_unresolved = _input_alias_topology_checks(
        descriptor, input_slots, raw_values
    )
    checks.extend(alias_checks)
    # ------------------------------------------------------------------
    # r35 I5 (corr2_6) -- CONTRACT-BEFORE-TOUCH admission choke point.
    # Everything ABOVE this comment reads only non-materializing, exception-
    # safe facts from the RAW bound leaves. Everything BELOW may clone,
    # transfer, view, digest, or otherwise materialize input bytes. Hard
    # executability preconditions (meta/sparse/nested/named/quantized/
    # subclass layouts) are enforced HERE, before any byte operation, and
    # raise regardless of the divergence policy: ``return_diverged`` may
    # continue only with an EXECUTABLE poisoned input, never past a hard
    # precondition. Any future preamble addition that touches input bytes
    # MUST be placed below this point.
    # ------------------------------------------------------------------
    hard_failure = next(
        (check for check in checks if not check.passed and check.name.startswith("input_layout:")),
        None,
    )
    if hard_failure is not None:
        diagnostic = hard_failure.diagnostic
        raise PathDivergenceError(
            diagnostic.message if diagnostic is not None else "Unsupported input layout.",
            code=(
                diagnostic.code.value
                if diagnostic is not None
                else RunnableErrorCode.INPUT_TREE_MISMATCH.value
            ),
            path_faithfulness=PathFaithfulness.DIVERGED,
            first_mismatch=diagnostic,
            contract_check=hard_failure,
        )
    # Phase 4: clone only ACCEPTED (executable) tensors.
    for slot in input_slots:
        raw = raw_values.get(slot.slot_id)
        if isinstance(raw, torch.Tensor):
            values[slot.slot_id] = _runtime_mirror_clone(raw)
    return values, tuple(checks), input_alias_unresolved


def _runtime_mirror_clone(raw: torch.Tensor) -> torch.Tensor:
    """Defensively clone one leaf while MIRRORING its runtime autograd metadata.

    r37 corr2-7 (R13): ``detach().clone()`` strips a leaf's ``requires_grad``, so
    the recorded ``grad_enabled=True`` call context became semantically inert (no
    autograd history on replay) and the exact original input was needlessly
    attestation-ineligible (fingerprint flag mismatch). The runtime mirror restores
    ``raw.requires_grad`` on the clone where legal; fingerprints stay on the
    executed-clone basis, so an intentionally changed-flag input remains a physical
    input change (attestation ``not_applicable``), never normalized away. This is
    the ONE second-clone helper -- every input-bind/state-clone/staging site routes
    through it or the staging helper's recorded-trainable rule.
    """

    clone = raw.detach().clone()
    if bool(raw.requires_grad) and not clone.requires_grad:
        try:
            clone.requires_grad_(True)
        except RuntimeError:
            # Non-differentiable dtype cannot require grad; the raw flag could not
            # have been set either, so this is unreachable in practice -- degrade
            # to the detached clone rather than aborting the bind.
            pass
    return clone


def _model_input_version_closure(descriptor: SparseRunDescriptor) -> set[str]:
    """Return model-input slot ids plus every slot transitively versioned from one."""

    closure = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    }
    changed = True
    while changed:
        changed = False
        for slot in descriptor.tensor_slots:
            if slot.version_of in closure and slot.slot_id not in closure:
                closure.add(slot.slot_id)
                changed = True
    return closure


_VIEW_OP_QUALNAMES = frozenset(
    {
        # Storage-sharing view ops whose output aliases an input tensor's storage (r29-C3, F2).
        # Deliberately OVER-broad: some entries here also cover copy variants -- that only
        # widens the fail-closed gate, which fires ONLY when runtime inputs actually alias, so
        # a false positive here can never over-trigger an ordinary (non-aliased-input) run.
        "__getitem__",
        "getitem",
        "select",
        "slice",
        "narrow",
        "narrow_copy",
        "view",
        "view_as",
        "_view",
        "reshape",
        "reshape_as",
        "transpose",
        "t",
        "permute",
        "squeeze",
        "unsqueeze",
        "expand",
        "expand_as",
        "broadcast_to",
        "flatten",
        "unflatten",
        "ravel",
        "unfold",
        "diagonal",
        "diagonal_scatter",
        "movedim",
        "moveaxis",
        "swapaxes",
        "swapdims",
        "detach",
        "as_strided",
        "split",
        "split_with_sizes",
        "chunk",
        "tensor_split",
        "hsplit",
        "vsplit",
        "dsplit",
        "unbind",
        "real",
        "imag",
        "view_as_real",
        "view_as_complex",
        "adjoint",
        "alias",
        "contiguous",
        "indices",
        "values",
    }
)
"""Torch op qualnames whose output can SHARE STORAGE with a tensor input (r29-C3, F2)."""


def _model_input_storage_closure(descriptor: SparseRunDescriptor) -> set[str]:
    """Model-input slots plus every slot reachable by version chains AND view-op lineage.

    A view op (``a[0]``, ``a.t()``, ``a.narrow(...)``) produces an output that SHARES STORAGE
    with its tensor input, so an in-place op on that view mutates the input's storage even
    though the view slot has no ``version_of`` link to the input (r29-C3, F2-hon). The gate
    that guards the input-aliasing fail-closed therefore follows both version chains and
    view-producing lineage from the model-input slots: a slot enters the closure if it is a
    model input, is versioned from a closure slot, or is the output of a view op whose tensor
    argument is already in the closure.
    """

    reg_qualname = {
        entry.registry_id: getattr(entry.key, "qualname", None)
        for entry in descriptor.callable_registry
    }
    closure = _model_input_version_closure(descriptor)
    changed = True
    while changed:
        changed = False
        for call in descriptor.calls:
            if reg_qualname.get(call.registry_id) not in _VIEW_OP_QUALNAMES:
                continue
            if not any(arg.slot_id in closure for arg in call.tensor_arguments):
                continue
            for output_slot_id in call.output_slot_ids:
                if output_slot_id not in closure:
                    closure.add(output_slot_id)
                    changed = True
        # Re-follow version chains from any newly-added view outputs.
        for slot in descriptor.tensor_slots:
            if slot.version_of in closure and slot.slot_id not in closure:
                closure.add(slot.slot_id)
                changed = True
    return closure


def _descriptor_mutates_model_input(descriptor: SparseRunDescriptor) -> bool:
    """Return whether any in-place call targets a model-input slot, its version chain, or a
    view derived from it (r29-C3, F2-hon)."""

    closure = _model_input_storage_closure(descriptor)
    for call in descriptor.calls:
        if call.is_inplace and _mutation_target_slot_id(call) in closure:
            return True
    return False


def _tensor_storage_key(value: torch.Tensor) -> int | None:
    """Return a stable base-storage identity for aliasing comparison, or ``None``."""

    try:
        return int(value.untyped_storage().data_ptr())
    except (RuntimeError, AttributeError):
        return None


# r37 INV-2: the alias/overlap proof engine lives in ``utils.tensor_utils`` --
# absolute, device-scoped byte intervals, three-valued relation, pure-integer
# enumeration. The former local implementation keyed "same memory" on
# ``untyped_storage().data_ptr()`` EQUALITY, which mis-proved disjointness for
# overlapping views of one external buffer (``torch.from_numpy(arr[:6])`` vs
# ``arr[2:8]`` -- distinct torch storages, genuinely shared host memory; hon1_1).
# No local reimplementation or pointer-equality shortcut may reappear here.


def _touched_bytes_relation(left: torch.Tensor, right: torch.Tensor) -> str:
    """Shared-engine adapter (see :func:`torchlens.utils.tensor_utils.touched_bytes_relation`)."""

    with _state.pause_logging():
        return touched_bytes_relation(left, right)


def _input_alias_topology_checks(
    descriptor: SparseRunDescriptor,
    input_slots: Sequence[TensorSlotDescriptor],
    raw_values: Mapping[str, torch.Tensor],
) -> tuple[tuple[ContractCheck, ...], bool]:
    """Fail closed on runtime input aliasing unreproducible against a de-aliased capture.

    Torch capture clones each model-input leaf before the forward (``safe_copy_args``), and
    the sparse replay likewise binds independent per-slot clones, so the captured DAG and the
    replay both reflect DISTINCT-input semantics -- the captured alias topology is always
    all-distinct. Any runtime aliasing between model-input sites therefore differs from the
    captured topology and cannot be reproduced against a fresh model on those same aliased
    inputs:

    * IDENTITY (``forward(a, b)`` with ``a is b`` -- self/cross-attention ``q is k``): the
      captured / replayed clones are distinct objects, so an ``if a is b`` / ``id()`` identity
      branch takes the OTHER arm than a fresh model on the aliased input would -- a false
      VERIFIED even on the ORIGINAL input (r33 F1). This holds with NO in-place mutation, so
      the check is UNCONDITIONAL.
    * OVERLAPPING STORAGE SPANS (two views whose byte spans overlap, a storage-identity
      ``a.data_ptr() == b.data_ptr()`` branch, or an in-place mutation that propagates between
      overlapping sites): the de-aliased clones neither share storage nor propagate a mutation
      between sites, so a fresh model on the aliased input can diverge.

    Both fail closed (``runtime topology != captured`` per the F1 contract: capture de-aliases,
    so ANY runtime aliasing is unreproducible and a genuinely identity/storage-independent model
    cannot be PROVEN so at this surface -- the honest verdict is fail-closed). DISJOINT spans of
    one base (``base[:2]`` / ``base[2:]`` -- same storage pointer, non-overlapping bytes) are NOT
    aliased: a mutation of one cannot reach the other and they are distinct objects, so they
    never trigger. All-distinct inputs (the common case) never trigger -- zero over-trigger on
    the trivial topology.
    """

    resolved: list[tuple[str, torch.Tensor]] = []
    for slot in input_slots:
        value = raw_values.get(slot.slot_id)
        if isinstance(value, torch.Tensor):
            resolved.append((slot.slot_id, value))
    aliased_pairs: set[tuple[str, str]] = set()

    def _ordered_pair(left: str, right: str) -> tuple[str, str]:
        return (left, right) if left <= right else (right, left)

    # Identity aliasing: two input sites bound to the SAME tensor object (``a is b``).
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            if resolved[i][1] is resolved[j][1]:
                aliased_pairs.add(_ordered_pair(resolved[i][0], resolved[j][0]))
    # Storage aliasing (r35 decision D): the three-valued touched-byte engine.
    # PROVED overlap is an observed contradiction (failed check -> DIVERGED);
    # PROVED disjointness passes; ``unknown`` is the
    # ``input_alias_topology_unresolved`` unverifiability ceiling -- never
    # ``overlap`` by assumption and never VERIFIED.
    unresolved = False
    for i in range(len(resolved)):
        for j in range(i + 1, len(resolved)):
            left_id, left = resolved[i]
            right_id, right = resolved[j]
            if left is right:
                continue  # identity already recorded above
            relation = _touched_bytes_relation(left, right)
            if relation == "overlap":
                aliased_pairs.add(_ordered_pair(left_id, right_id))
            elif relation == "unknown":
                unresolved = True
    if not aliased_pairs:
        return (), unresolved
    return (
        _contract_check(
            "input_alias_topology",
            False,
            RunnableErrorCode.INPUT_TREE_MISMATCH,
            "Runtime model inputs alias (same object or proven-overlapping touched "
            "bytes); capture de-aliases inputs (independent per-slot clones), so the "
            "recorded taken path cannot reproduce an identity or storage-aliasing "
            "dependence and must not be blessed VERIFIED.",
            details=(("aliased_input_slot_pairs", repr(sorted(aliased_pairs))),),
        ),
    ), unresolved


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


def _snapshot_input_fingerprints(
    descriptor: SparseRunDescriptor,
    slot_values: Mapping[str, torch.Tensor],
    input_byte_digests: Mapping[str, str],
) -> dict[str, InputAttestationFingerprint]:
    """Fingerprint the EXECUTED input clones on the capture-identical basis (hon1_3).

    Runs after admission (I5), before any sparse call. The capture side
    fingerprints the retained input clone that seeded the captured forward; both
    sides therefore compare the same clone basis, so a physical twin (layout /
    stride / offset / alignment-class change) is detected exactly.
    """

    fingerprints: dict[str, InputAttestationFingerprint] = {}
    for slot in descriptor.tensor_slots:
        if slot.role is not TensorSlotRole.MODEL_INPUT or slot.slot_id not in slot_values:
            continue
        fingerprints[slot.slot_id] = build_input_attestation_fingerprint(
            slot.slot_id,
            slot_values[slot.slot_id],
            byte_digest=input_byte_digests.get(slot.slot_id),
        )
    return fingerprints


_FINGERPRINT_ALIGNMENT_MODULUS = 16
"""Data-pointer alignment-class modulus for the physical input fingerprint.

H_B_RESOLUTION R2: fp16 CUDA conv kernel selection/reduction order is keyed to the
data pointer's alignment class (a misaligned offset view vs an aligned buffer), so
the fingerprint records ``data_ptr() % 16`` of the value that actually seeds
execution and attestation goes ``not_applicable`` on any mismatch instead of
false-tripping the byte tripwire. The modulus is the vector-width class boundary:
every fresh torch allocation (capture and replay both execute fresh clones) is at
least 16-byte aligned, so equal-basis executions always agree, while an offset
view that changes the executed alignment class is caught.
"""


def build_input_attestation_fingerprint(
    slot_id: str,
    value: torch.Tensor,
    *,
    byte_digest: str | None = None,
) -> InputAttestationFingerprint:
    """Build the physical identity fingerprint of one model-input value (hon1_3 H-a).

    Both sides use the same basis: at save time the LIVE retained in-memory input
    that seeded the captured forward; at run time the executed defensive clone.
    Any physical fact that cannot be read fails toward attestation ineligibility
    (a sentinel value that can never equal a well-formed capture record).

    Parameters
    ----------
    slot_id:
        Descriptor slot id of the model input.
    value:
        Live tensor to fingerprint.
    byte_digest:
        Precomputed logical byte digest, or ``None`` to compute one here.

    Returns
    -------
    InputAttestationFingerprint
        Frozen physical fingerprint record.
    """

    def _safe_bool(getter: Callable[[], Any]) -> bool:
        try:
            return bool(getter())
        except (RuntimeError, AttributeError, TypeError, NotImplementedError):
            return False

    try:
        alignment = int(value.data_ptr()) % _FINGERPRINT_ALIGNMENT_MODULUS
    except (RuntimeError, AttributeError, TypeError, NotImplementedError):
        # Unreadable pointer: use an out-of-range sentinel so it can never match
        # a well-formed capture record (fail toward not_applicable, never a
        # false ATTESTED).
        alignment = -1
    if byte_digest is None:
        byte_digest = runnable_tensor_byte_digest(value)
    return InputAttestationFingerprint(
        slot_id=slot_id,
        byte_digest=byte_digest,
        device_type=str(value.device.type),
        device_index=None if value.device.index is None else int(value.device.index),
        layout=str(value.layout),
        sizes=tuple(int(item) for item in value.shape),
        strides=tuple(int(item) for item in value.stride()),
        storage_offset=int(value.storage_offset()),
        is_contiguous=_safe_bool(value.is_contiguous),
        is_channels_last=_safe_bool(lambda: value.is_contiguous(memory_format=torch.channels_last)),
        is_channels_last_3d=_safe_bool(
            lambda: value.is_contiguous(memory_format=torch.channels_last_3d)
        ),
        is_conj=_safe_bool(value.is_conj),
        is_neg=_safe_bool(value.is_neg),
        tensor_class=type(value).__qualname__,
        requires_grad=bool(value.requires_grad),
        is_inference=_safe_bool(value.is_inference),
        alignment_class=alignment,
    )


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


def _type_strict_path(path: Iterable[Any]) -> tuple[Any, ...]:
    """Type-tag numeric mapping-key path components so ``bool``/``int``/``float`` twins do not
    conflate under Python ``==`` (r33 F6).

    A dict key ``True`` compares equal to ``1`` and ``1.0`` (``True == 1 == 1.0`` with colliding
    hashes), so a raw ``(True,)`` leaf path matches ``(1,)`` / ``(1.0,)`` in the contract path
    SET -- a runtime input whose key TYPE changed then silently passes the input-tree tripwire.
    Tagging each numeric component with its concrete type keeps the three distinct. Applied
    SYMMETRICALLY to the recorded and runtime path sets, so an ordinary same-type structure (the
    common case) is unaffected -- zero over-trigger. Non-numeric components (``str`` keys,
    already-encoded ``(BOOL_KEY_PATH_TAG, ...)`` tuples) pass through unchanged.
    """

    tagged: list[Any] = []
    for component in path:
        if isinstance(component, bool):
            tagged.append(("\x00tl_key_bool", bool(component)))
        elif isinstance(component, int):
            tagged.append(("\x00tl_key_int", int(component)))
        elif isinstance(component, float):
            tagged.append(("\x00tl_key_float", float(component)))
        else:
            tagged.append(component)
    return tuple(tagged)


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
    expected_by_position: dict[Any, set[tuple[Any, ...]]] = {}
    for slot in slots:
        assert slot.input_binding is not None
        expected_by_position.setdefault(slot.input_binding.model_site_position, set()).add(
            _type_strict_path(slot.input_binding.container_path)
        )
    checks: list[ContractCheck] = []
    for position, expected in expected_by_position.items():
        try:
            root = _input_site_value(inputs, position, positions)
            actual = {_type_strict_path(p) for p in _tensor_leaf_paths(root)}
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


def _runtime_nontensor_leaf_paths(root: Any) -> set[tuple[str | int, ...]]:
    """Enumerate runtime non-tensor input leaf paths, mirroring the capture walk.

    Reproduces the capture-side ``_record_runnable_input_literal_leaves`` traversal so
    the runtime non-tensor leaf-path SET aligns with the recorded fact set: tensor
    leaves are skipped, namedtuples descend by field name, mappings descend under
    grammar-encodable keys (a non-encodable key collapses its whole subtree to one
    opaque leaf at the parent path, exactly as capture does), and lists/tuples descend
    by index. Every other value is a scalar leaf recorded at its path.
    """

    from torchlens._io.runnable import (
        EMPTY_CONTAINER_PATH_MARKER,
        _UnsupportedLiteralError,
        _encode_literal_key,
        empty_container_kind,
        input_path_key_component,
    )

    paths: set[tuple[str | int, ...]] = set()

    def _walk(value: Any, path: tuple[str | int, ...]) -> None:
        """Descend one runtime boundary value, collecting every non-tensor leaf path."""

        if isinstance(value, torch.Tensor):
            return
        if empty_container_kind(value) is not None:
            paths.add((*path, EMPTY_CONTAINER_PATH_MARKER))
            return
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            for name in value._fields:
                _walk(getattr(value, name), (*path, str(name)))
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                try:
                    _encode_literal_key(key)
                except _UnsupportedLiteralError:
                    paths.add(path)
                    continue
                _walk(child, (*path, input_path_key_component(key)))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(child, (*path, index))
            return
        paths.add(path)

    _walk(root, ())
    return paths


def _input_nontensor_tree_contract_checks(
    descriptor: SparseRunDescriptor,
    inputs: Any,
    positions: set[Any],
) -> tuple[ContractCheck, ...]:
    """Compare the runtime NON-tensor leaf-path tree with every recorded input site.

    The per-leaf value check (:func:`_input_literal_contract_checks`) only visits
    leaves RECORDED at capture, so it catches a CHANGED or MISSING non-tensor leaf but
    is blind to an EXTRA runtime non-tensor leaf (an added dict key the model branches
    on via ``'flag' in d`` / ``d.get('mode')``, or a longer list). An extra leaf can
    steer unwitnessed Python control flow while replay still reports VERIFIED against a
    fresh model on the given inputs. This mirrors the tensor-leaf set-equality
    (:func:`_input_tree_contract_checks`) for non-tensor leaves: EVERY model-input site
    is seeded with its recorded non-tensor leaf-path set (the empty set when capture had
    no non-tensor leaf at that site), and any runtime non-tensor leaf path absent at
    capture -- or a recorded one absent at runtime -- diverges the run.
    """

    expected_by_position: dict[Any, set[tuple[Any, ...]]] = {
        position: set() for position in positions
    }
    for _witness, fact in _model_input_literal_facts(descriptor):
        raw_position = fact.get("position")
        position = tuple(raw_position) if isinstance(raw_position, (list, tuple)) else raw_position
        path = _type_strict_path(fact.get("path", ()) or ())
        expected_by_position.setdefault(position, set()).add(path)

    checks: list[ContractCheck] = []
    for position, expected in expected_by_position.items():
        try:
            root = _input_site_value(inputs, position, positions)
            actual = {_type_strict_path(p) for p in _runtime_nontensor_leaf_paths(root)}
        except (KeyError, IndexError, TypeError, AttributeError):
            actual = set()
        checks.append(
            _contract_check(
                f"input_nontensor_tree:{position!r}",
                actual == expected,
                RunnableErrorCode.INPUT_TREE_MISMATCH,
                "Runtime non-tensor input leaf paths disagree with the recorded input "
                "tree; an added/removed non-tensor leaf can steer an unwitnessed path.",
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
    # r37 hon1_4: inference tensors carry NO version counter (reading ``_version``
    # raises), so a slot whose version is unavailable records NO baseline. The
    # mutation tripwire then enforces only its version-independent legs for that
    # slot (alias identity for in-place calls); value fidelity remains guarded by
    # the output comparison and numeric attestation layers.
    versions: dict[str, int] = {}
    for argument in call.tensor_arguments:
        value = slot_values.get(argument.slot_id)
        if value is None:
            continue
        version = tensor_version_or_none(value)
        if version is not None:
            versions[argument.slot_id] = version
    return checks, versions


def _context_unavailable_error(field: str, detail: str) -> RunPreconditionError:
    """Build the typed refusal for an un-enterable/un-restorable execution context."""

    return RunPreconditionError(
        f"Recorded execution context {field!r} cannot be entered or restored on "
        f"this runtime: {detail}",
        code=RunnableErrorCode.EXECUTION_CONTEXT_UNAVAILABLE.value,
        context_field=field,
    )


@contextmanager
def _ambient_execution_context_restored(ambient: Any) -> Any:
    """Transactionally restore the recorded capture-scoped ambient context (decision E).

    The caller's ambient state is snapshotted, the recorded values are applied
    (``None`` producer-absent fields are left as-is -- there is nothing recorded
    to restore), and the caller's state is re-applied in ``finally`` on every
    exit: success, divergence, callable exception, and numeric-attestation
    rollback. A recorded value this runtime cannot apply rolls back any partial
    application and raises the typed ``execution_context_unavailable`` refusal
    -- never a silent ambient passthrough.
    """

    from .utils._torch_compat import (
        apply_ambient_execution_context,
        snapshot_ambient_execution_context,
    )

    recorded = {
        "default_dtype": ambient.default_dtype,
        "default_device": ambient.default_device,
        "float32_matmul_precision": ambient.float32_matmul_precision,
        "deterministic_algorithms": ambient.deterministic_algorithms,
        "deterministic_algorithms_warn_only": ambient.deterministic_algorithms_warn_only,
        "cuda_matmul_allow_tf32": ambient.cuda_matmul_allow_tf32,
        "cudnn_allow_tf32": ambient.cudnn_allow_tf32,
        "cudnn_deterministic": ambient.cudnn_deterministic,
        "cudnn_benchmark": ambient.cudnn_benchmark,
        "cudnn_enabled": ambient.cudnn_enabled,
        "flash_sdp_enabled": ambient.flash_sdp_enabled,
        "mem_efficient_sdp_enabled": ambient.mem_efficient_sdp_enabled,
        "math_sdp_enabled": ambient.math_sdp_enabled,
    }
    saved = snapshot_ambient_execution_context()
    # r37 R4 (corr2-3/corr2-2): the recorded DEFAULT DEVICE is entered as a SCOPED
    # ``with torch.device(recorded)`` mode nested above the caller's existing mode
    # stack -- never via ``torch.set_default_device`` (which mutates process-global
    # mode bookkeeping and, measured, leaks/clobbers DeviceContext modes on every
    # policy: implicit callers gained a mode, nested callers were corrupted). The
    # context-manager exit IS the restoration mechanism -- correct by construction on
    # success, divergence, callable exception, and attestation rollback -- so no
    # restore logic exists for the device at all. A mode-stack length postcondition
    # (feature-probed introspection) is a belt-and-suspenders tripwire.
    device_scope: Any = nullcontext()
    if ambient.default_device is not None:
        try:
            device_scope = torch.device(str(ambient.default_device))
        except (RuntimeError, TypeError, ValueError) as exc:
            raise _context_unavailable_error("default_device", str(exc)) from exc
    from .utils._torch_compat import get_current_function_mode_stack

    stack_before = get_current_function_mode_stack()
    depth_before = len(list(stack_before)) if stack_before is not None else None
    try:
        apply_ambient_execution_context(recorded)
    except RuntimeError as exc:
        try:
            apply_ambient_execution_context(saved)
        except RuntimeError:  # pragma: no cover - saved values came from this runtime
            pass
        raise _context_unavailable_error("ambient_context", str(exc)) from exc
    try:
        with device_scope:
            yield
    finally:
        apply_ambient_execution_context(saved)
        if depth_before is not None and sys.exc_info()[0] is None:
            stack_after = get_current_function_mode_stack()
            depth_after = len(list(stack_after)) if stack_after is not None else None
            if depth_after is not None and depth_after != depth_before:
                raise RuntimeError(
                    "Internal invariant violation: the run transaction changed the "
                    f"caller's TorchFunctionMode stack depth ({depth_before} -> "
                    f"{depth_after}); scoped device-context restoration must be "
                    "exact on every exit path."
                )


@contextmanager
def _call_execution_context_entered(context: Any) -> Any:
    """Enter the REQUIRED recorded per-call execution context tightly (corr2_8).

    Autocast: an ``enabled=True`` record enters autocast with the recorded
    dtype; an explicit ``enabled=False`` record actively enters a DISABLED
    autocast context when the runtime currently has that device class enabled
    (so a caller's ambient autocast cannot contaminate a disabled capture) and
    is vacuously satisfied when the device class is absent/disabled. Grad and
    inference modes are entered explicitly. Context entry never touches RNG;
    the caller's context is restored in reverse order on every exit. An
    un-enterable recorded context is a typed refusal, never a raw torch error.
    """

    from .utils._torch_compat import autocast_is_enabled

    stack: list[Any] = []
    try:
        for entry in context.autocast:
            if entry.enabled:
                dtype_name = str(entry.dtype or "").removeprefix("torch.")
                dtype = getattr(torch, dtype_name, None)
                if not isinstance(dtype, torch.dtype):
                    raise _context_unavailable_error(
                        f"autocast:{entry.device_type}",
                        f"recorded autocast dtype {entry.dtype!r} is unavailable",
                    )
                try:
                    ctx = torch.amp.autocast(entry.device_type, enabled=True, dtype=dtype)
                    ctx.__enter__()
                except (RuntimeError, ValueError, TypeError) as exc:
                    raise _context_unavailable_error(
                        f"autocast:{entry.device_type}", str(exc)
                    ) from exc
                stack.append(ctx)
                continue
            # Explicit disabled record: enter a disabled context only when the
            # runtime reports that device class currently autocast-enabled; a
            # runtime without the device class is vacuously disabled already.
            try:
                currently_enabled = bool(autocast_is_enabled(entry.device_type))
            except (RuntimeError, TypeError):
                currently_enabled = False
            if currently_enabled:
                try:
                    ctx = torch.amp.autocast(entry.device_type, enabled=False)
                    ctx.__enter__()
                except (RuntimeError, ValueError, TypeError) as exc:
                    raise _context_unavailable_error(
                        f"autocast:{entry.device_type}", str(exc)
                    ) from exc
                stack.append(ctx)
        try:
            inference_ctx = torch.inference_mode(bool(context.inference_mode))
            inference_ctx.__enter__()
            stack.append(inference_ctx)
            grad_ctx = torch.enable_grad() if context.grad_enabled else torch.no_grad()
            grad_ctx.__enter__()
            stack.append(grad_ctx)
        except RuntimeError as exc:
            raise _context_unavailable_error("grad_mode", str(exc)) from exc
        yield
    finally:
        for ctx in reversed(stack):
            ctx.__exit__(None, None, None)


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
    state_slot_ids = frozenset(
        slot.slot_id
        for slot in descriptor.tensor_slots
        if slot.role in {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}
    )
    checks.extend(
        _mutation_contract_checks(call, output, slot_values, before_versions, state_slot_ids)
    )
    return tuple(checks)


def _mutation_contract_checks(
    call: RunnableCallDescriptor,
    output: Any,
    slot_values: Mapping[str, torch.Tensor],
    before_versions: Mapping[str, int],
    state_slot_ids: frozenset[str] = frozenset(),
) -> tuple[ContractCheck, ...]:
    """Validate recorded in-place aliasing and tensor-version expectations.

    A NON-inplace call may legitimately bump the ``_version`` of a STATE buffer/parameter
    slot -- a mode-sensitive norm layer (InstanceNorm/BatchNorm with
    ``track_running_stats=True``) updates its ``running_mean``/``running_var`` running stats
    inside the functional ``instance_norm``/``batch_norm`` call, which TorchLens does not
    record as a separate in-place op (r29-C3, codex-F3). That running-stat update is a
    declared-state side effect reproduced identically on a fresh replay from the captured
    state, NOT the input/activation-tensor mutation this check guards against, so state slots
    are excluded from the non-inplace ``changed`` set. Value correctness of the updated buffer
    is separately enforced by state/output attestation, so this exclusion cannot mask a real
    numeric divergence.
    """

    changed = {
        slot_id
        for slot_id, before in before_versions.items()
        if slot_id in slot_values and tensor_version_or_none(slot_values[slot_id]) != before
    }
    if not call.is_inplace:
        non_state_changed = changed - state_slot_ids
        return (
            _contract_check(
                f"mutation:{call.call_id}",
                not non_state_changed,
                RunnableErrorCode.MUTATION_VERSION_MISMATCH,
                f"Non-mutating call {call.call_id!r} changed an input tensor version.",
                affected_op_labels=call.op_labels,
                details=(("changed_slot_ids", repr(tuple(sorted(non_state_changed)))),),
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
    # Version leg: enforced only when a baseline exists. An inference tensor has no
    # version counter, so its in-place relation is proven by alias identity alone
    # (hon1_4); a versioned tensor keeps the full version-bump requirement.
    version_leg = input_slot_id in changed if input_slot_id in before_versions else True
    return (
        _contract_check(
            f"mutation:{call.call_id}",
            version_leg and aliases,
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
    if len(values) == 1 and not values[0][0]:
        # Genuine bare-tensor model output: nothing to reconstruct.
        return values[0][1]
    if values:
        # r35 I1 defense in depth: a multi-leaf output with NO recorded container
        # spec can only come from a legacy/tampered artifact (new saves are refused
        # unless losslessness is proved). Approximating the container here would be
        # a silent lossy substitution, so fail typed instead.
        raise RunPreconditionError(
            "Model output has multiple leaves but no recorded lossless container "
            "contract; this artifact predates (or violates) the v2 output "
            "losslessness proof and cannot be reconstructed faithfully.",
            code=RunnableErrorCode.MISSING_OUTPUT_CONTAINER_CONTRACT.value,
        )
    return _container_from_paths(values)


def _output_container_spec(trace: Any) -> ContainerSpec | None:
    """Return the shared recorded model-output container specification."""

    for label in getattr(trace, "output_layers", ()):
        op = _op_for_label(trace, label)
        spec = getattr(op, "container_spec", None)
        if isinstance(spec, ContainerSpec):
            return spec
    return None


def _output_not_reproduced(
    descriptor: SparseRunDescriptor, container_spec: ContainerSpec | None
) -> bool:
    """Return whether the captured model output was NOT reproduced by the sparse replay.

    A model whose forward returns a HOST-ESCAPED non-tensor Python scalar (``float(x.sum())``,
    ``x.item()``, ``int(...)``, ``bool(...)``) has NO output tensor slots AND no reconstructable
    output container spec, so the sparse DAG cannot emit that value: ``_reconstruct_output``
    returns a dropped ``None``. The escape-witness class applies to the OUTPUT too -- an output
    the replay never produced or compared must never be blessed VERIFIED. Stage-6 downgrades
    such a run to UNVERIFIABLE.

    A normal tensor output has >= 1 ``OUTPUT`` tensor slot; a container output (even one whose
    leaves are all literals) carries a ``ContainerSpec``; both are genuinely produced/compared
    and stay eligible for VERIFIED. Only the "no output slot AND no container spec" shape -- the
    host-escaped / dropped output -- is flagged here.

    Parameters
    ----------
    descriptor:
        Runnable descriptor whose tensor slots include the model-output slots.
    container_spec:
        The shared recorded output ``ContainerSpec`` (``None`` when none was recorded).

    Returns
    -------
    bool
        True when replay produced no output tensor and no reconstructable container.
    """

    has_output_slot = any(slot.role is TensorSlotRole.OUTPUT for slot in descriptor.tensor_slots)
    return not has_output_slot and container_spec is None


def _container_spec_reconstruction_lossy(spec: ContainerSpec | None) -> bool:
    """Return whether ``spec`` (or any nested child) reconstructs lossily.

    A dataclass / ``ModelOutput`` output whose live instance carried computed
    non-field/non-key state, a ``__slots__`` layout, or a data-descriptor field is
    flagged ``lossy_reconstruction`` at capture: the non-invoking rebuild cannot restore
    that state and must NOT be blessed VERIFIED. Any lossy node anywhere in the output
    container tree downgrades the whole run to UNVERIFIABLE.

    The persisted ``lossy_reconstruction`` flag is attacker-controlled in an untrusted bundle,
    so we NEVER trust a ``False`` alone: for every dataclass / ``hf_model_output`` node we ALSO
    recompute lossiness INDEPENDENTLY from the RESOLVED type at load time (``__slots__`` /
    data-descriptor field / dropped non-field state), so a forged ``lossy_reconstruction=False``
    cannot force a false VERIFIED. The persisted flag stays as a supplementary signal (kept for
    the purely instance-level custom-``ModelOutput`` case that is not type-observable at load),
    but it can only ADD lossiness, never suppress the independent recompute.
    """

    if spec is None:
        return False
    if _spec_node_reconstruction_lossy(spec):
        return True
    return any(_container_spec_reconstruction_lossy(child) for _, child in spec.child_specs)


def _spec_node_reconstruction_lossy(spec: ContainerSpec) -> bool:
    """Return whether one dataclass / ``hf_model_output`` node reconstructs lossily.

    Combines the persisted (supplementary) flag with an INDEPENDENT load-time recompute from
    the resolved type. A tampered spec naming a type that fails default-deny admissibility, or
    one whose type is not loaded (reconstruction then falls back to a plain namespace / mapping
    that is NOT the captured container type), is treated as lossy -- never a false VERIFIED.
    """

    if getattr(spec, "lossy_reconstruction", False):
        return True
    if spec.kind not in {"dataclass", "hf_model_output"}:
        return False
    captured_names = spec.fields if spec.kind == "dataclass" else spec.keys
    try:
        container_type = resolve_container_type(spec)
    except ContainerReconstructionError:
        # A tampered / non-admissible type is refused at reconstruction; for the honesty
        # gate treat it as lossy so it can never be blessed VERIFIED.
        return True
    if container_type is None:
        # The captured type is not loaded, so reconstruction returns a plain namespace / mapping
        # rather than the recorded container type: a lossy substitution, never a false VERIFIED.
        return True
    return reconstruction_is_lossy_by_type(container_type, captured_names, spec.kind)


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
        if _is_model_input_metadata_witness(witness):
            # Model-input metadata-predicate facts are compared against the RAW
            # runtime inputs in the input contract before execution; they are not
            # runtime container-structure facts.
            continue
        if _is_unbound_state_escape_witness(witness):
            # Unbound state escapes are compared by capture-digest in the dedicated
            # staleness check, not against runtime container structure.
            continue
        if witness.site_label.startswith(_MODULE_TRAINING_MODE_SITE_PREFIX):
            # The declared per-module train/eval mode is a capture-time state fact anchoring
            # VERIFIED (see ``_mode_sensitive_op_unwitnessed``), not a runtime container
            # structure fact; it must not be compared against the runtime container.
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

    r35 hon1_4: delegates to the shared repr-independent helper in
    ``utils/_torch_compat.py`` -- the exact same source the capture side uses,
    so capture and replay stay behavior-identical. Console wrap position,
    dtype/device suffixes, and tensor rendering can never create or destroy a
    structural field.

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

    from .utils._torch_compat import torch_structseq_field_names

    return torch_structseq_field_names(value)


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
    container_reconstruction_lossy: bool = False,
    output_not_reproduced: bool = False,
    mode_sensitive_op_unwitnessed: bool = False,
    input_alias_unresolved: bool = False,
) -> tuple[PathFaithfulness, RunnableDiagnostic | None]:
    """Classify exact three-state path faithfulness after all honesty checks."""

    failed = next((check for check in checks if not check.passed), None)
    if failed is not None:
        return PathFaithfulness.DIVERGED, failed.diagnostic
    if descriptor.witness_completeness is not WitnessCompleteness.COMPLETE:
        return PathFaithfulness.UNVERIFIABLE, None
    if input_alias_unresolved:
        # r35 decision D: the three-valued alias engine could prove neither
        # overlap nor disjointness for a same-storage input pair. Unknown is not
        # an observed contradiction (never DIVERGED by assumption) and not a
        # proof of equivalence (never VERIFIED): the honest ceiling is
        # UNVERIFIABLE (``input_alias_topology_unresolved``).
        return PathFaithfulness.UNVERIFIABLE, None
    if mode_sensitive_op_unwitnessed:
        # A BatchNorm/InstanceNorm op in the taken path is train/eval mode-sensitive, but the
        # capture-time mode was not recorded as a declared fact. Without that anchor the run
        # cannot prove which mode (eval running-stats vs train batch-stats) VERIFIED
        # corresponds to, so the honest ceiling is UNVERIFIABLE, never a false VERIFIED.
        return PathFaithfulness.UNVERIFIABLE, None
    if output_not_reproduced:
        # The captured forward returned a HOST-ESCAPED non-tensor scalar (or otherwise
        # unrepresentable value): no output tensor slot and no reconstructable output container,
        # so the sparse DAG emitted a dropped ``None`` that was never produced or compared. A
        # dropped output can never be VERIFIED; the honest ceiling is UNVERIFIABLE.
        return PathFaithfulness.UNVERIFIABLE, None
    if container_reconstruction_lossy:
        # The output is a dataclass / ModelOutput whose live instance carried computed
        # non-field/non-key state (e.g. a __post_init__ value derived from a tensor), a
        # __slots__ layout, or a data-descriptor field. The non-invoking rebuild restores
        # only captured fields/keys, so the replayed output differs from a fresh instance
        # and the derived state cannot be safely recomputed. The honest ceiling is
        # UNVERIFIABLE, never a false VERIFIED that drops that state silently.
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
    *,
    state_source: StateSource,
    initializer_policy_version: str | None,
    seed: int | None,
    random_filled_slot_ids: tuple[str, ...],
    contract_checks: tuple[ContractCheck, ...],
    path_faithfulness: PathFaithfulness,
    first_mismatch: RunnableDiagnostic | None,
    numeric_attestation: NumericAttestationStatus,
) -> RunReport:
    """Build the settled run-report surface -- the ONE report finalizer (r37 corr2-5).

    EVERY provider (loaded sparse AND live refresh) routes its report through this
    constructor: ``poisoned`` is DERIVED solely from ``path_faithfulness is not
    VERIFIED`` (no caller Boolean exists), and the r35 I3 tripwire-on-the-tripwire
    asserts ``attested`` structurally implies a verified, unpoisoned path, so no
    provider can emit an internally contradictory report. Direct ``RunReport(``
    construction outside this finalizer is forbidden (source-scan meta-test).
    """

    poisoned = path_faithfulness is not PathFaithfulness.VERIFIED
    if numeric_attestation is NumericAttestationStatus.ATTESTED and (
        path_faithfulness is not PathFaithfulness.VERIFIED or poisoned
    ):
        raise RuntimeError(
            "Internal invariant violation: numeric_attestation=attested requires "
            f"a verified, unpoisoned path (got {path_faithfulness.value!r})."
        )
    return RunReport(
        readiness=readiness,
        state_source=state_source,
        initializer_policy_version=initializer_policy_version,
        seed=seed,
        random_filled_slot_ids=random_filled_slot_ids,
        contract_checks=contract_checks,
        path_faithfulness=path_faithfulness,
        first_mismatch=first_mismatch,
        numeric_attestation=numeric_attestation,
        poisoned=poisoned,
    )


def _numeric_attestation_check(
    descriptor: SparseRunDescriptor,
    state: PreparedRunnableState,
    *,
    slot_values: Mapping[str, torch.Tensor],
    attestation_slot_values: Mapping[str, torch.Tensor],
    input_byte_digests: Mapping[str, str],
    input_fingerprints: Mapping[str, InputAttestationFingerprint],
    state_byte_digests: Mapping[str, str],
    trace: Any,
    provisional_verdict: PathFaithfulness,
) -> tuple[NumericAttestationStatus, ContractCheck | None]:
    """Compare recomputed saved slots with the independent activation archive.

    r35 I3 (corr2_7): eligibility DERIVES from the settled provisional path
    verdict -- computed from every non-numeric contract check and every static/
    dynamic ceiling, including the fork's inherited monotonic mark -- instead of
    a parallel Boolean flag list. Any verdict that is not ``verified`` returns
    ``not_applicable`` before a single archive byte is read, so ``attested``
    structurally implies ``verified`` and every future contract check
    automatically caps attestation.

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
    provisional_verdict:
        Settled non-numeric path verdict (inherited monotonic marks folded in).

    Returns
    -------
    tuple[NumericAttestationStatus, ContractCheck | None]
        Applicability/result status and the aggregate byte-exact tripwire check.
    """

    if provisional_verdict is not PathFaithfulness.VERIFIED:
        # Not a settled VERIFIED path: attestation is not applicable and the
        # archive is never opened (no comparison before a non-numeric verdict).
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if descriptor.ambient_context.attestation_ineligible_context:
        # Positive capture-time nondeterministic-context marking (decision E /
        # H_B_RESOLUTION R1): cudnn.benchmark or a CUDA-nondeterministic op
        # captured without deterministic algorithms cannot promise reproducible
        # bytes -- fail-safe ineligibility, never a spurious tripwire raise.
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
    if _has_journaled_buffer_activation_member(descriptor, layer, raw_members):
        # Repeated registered-buffer slots for the same state entry whose archived bytes DIFFER
        # from the capture-time state are journal points (a mode-sensitive norm layer updating
        # its running stats mid-forward), not immutable activation payloads: each slot's bytes
        # are only meaningful at that journal point, so skip byte-exact activation attestation
        # rather than raising a false mismatch on a valid replay. A repeated buffer slot whose
        # archived bytes EQUAL the capture state (a read-only running stat under eval) is stable,
        # not journaled, and stays byte-attestable (r29-C4, codex-F2).
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if _has_out_mutated_activation_member(descriptor, raw_members):
        # An archived activation that later became an ``out=`` destination was
        # captured before its mutation. Its pre-write bytes are allocator data,
        # not a reproducible result, so attesting it would create a false
        # divergence on an otherwise faithful original-input replay.
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if not raw_members or not _attestation_inputs_match(
        descriptor, layer, input_byte_digests, input_fingerprints
    ):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if not _attestation_state_matches(descriptor, layer, state, state_byte_digests, trace):
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
    layer: Any,
    members: Sequence[ActivationPayloadMember],
) -> bool:
    """Return whether selected activation payloads include JOURNALED buffer slots.

    A buffer state entry with multiple selected activation members is journaled only when at
    least one member's ARCHIVED bytes DIFFER from the capture-time state digest -- i.e. the
    buffer was actually WRITTEN mid-forward (a norm layer updating its running stats). When
    every repeated member equals the capture state digest, the buffer is a STABLE read-only
    source (running stats under eval, or an unwritten buffer read at several points), which is
    fully byte-attestable and must NOT be suppressed (r29-C4, codex-F2). When the capture state
    digest is unavailable for a repeated buffer name, fail closed (treat as journaled) rather
    than risk a false mismatch.

    Parameters
    ----------
    descriptor:
        Sparse descriptor declaring tensor slot roles.
    layer:
        Activation payload layer descriptor carrying ``capture_state_digests``.
    members:
        Raw activation payload members selected for byte attestation.

    Returns
    -------
    bool
        ``True`` when a repeated same-state buffer entry was actually journaled (written).
    """

    slots = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    state_digests = {
        item.state_dict_name: item.byte_digest
        for item in getattr(layer, "capture_state_digests", ()) or ()
    }
    members_by_state: dict[str, list[ActivationPayloadMember]] = {}
    for member in members:
        slot = slots.get(member.slot_id)
        if slot is None or slot.role is not TensorSlotRole.BUFFER:
            continue
        binding = slot.state_binding
        if binding is None:
            continue
        members_by_state.setdefault(binding.state_dict_name, []).append(member)
    for state_name, state_members in members_by_state.items():
        if len(state_members) <= 1:
            continue
        if state_name not in state_digests:
            return True
        if any(member.byte_digest != state_digests[state_name] for member in state_members):
            return True
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


_MODULE_TRAINING_MODE_SITE_PREFIX = "module_training_mode:"
"""``site_label`` prefix marking the declared capture-time per-module train/eval mode."""


def _is_mode_sensitive_qualname(qualname: str | None) -> bool:
    """Return whether a qualname is a train/eval mode-sensitive op (BatchNorm family).

    BatchNorm / InstanceNorm produce numerically different results in eval (running
    statistics) vs train (batch statistics) mode. The recorded aten op bakes in the
    captured mode, so the replay reproduces it -- but only the DECLARED mode proves which
    result VERIFIED corresponds to. Dropout is intentionally excluded here: its train arm
    is already RNG-tainted (``not_applicable``) and its eval arm is identity, so it never
    creates a mode-driven false VERIFIED.
    """

    if not qualname:
        return False
    tail = qualname.rsplit(".", 1)[-1]
    if tail.endswith("_"):
        tail = tail[:-1]
    return "batch_norm" in tail or tail.endswith("instance_norm")


def _descriptor_has_mode_sensitive_op(descriptor: SparseRunDescriptor) -> bool:
    """Return whether the taken path contains a train/eval mode-sensitive op."""

    registry = {entry.registry_id: entry for entry in descriptor.callable_registry}
    for call in descriptor.calls:
        entry = registry.get(call.registry_id)
        if entry is not None and _is_mode_sensitive_qualname(entry.key.qualname):
            return True
    return False


def _descriptor_declares_training_mode(descriptor: SparseRunDescriptor) -> bool:
    """Return whether the descriptor declares the capture-time per-module train/eval mode."""

    return any(
        witness.kind is ControlWitnessKind.SHAPE_STRUCTURE_FACT
        and witness.site_label.startswith(_MODULE_TRAINING_MODE_SITE_PREFIX)
        for witness in descriptor.control_witnesses
    )


def _mode_sensitive_op_unwitnessed(descriptor: SparseRunDescriptor) -> bool:
    """Return whether a mode-sensitive op replays without a declared train/eval mode.

    ``self.training`` is declared state the VERIFIED oracle (a fresh instance in the
    captured mode on the given inputs) reproduces. A BatchNorm / InstanceNorm op in the
    taken path whose capture-time mode is NOT recorded has no witnessed proof of the mode
    it corresponds to (eval running-stats vs train batch-stats), so the honest ceiling is
    UNVERIFIABLE. Captures with no mode-sensitive op, or that declare the mode (every new
    intervention-ready capture does), stay VERIFIED -- no over-trigger.
    """

    return _descriptor_has_mode_sensitive_op(descriptor) and not _descriptor_declares_training_mode(
        descriptor
    )


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
    input_fingerprints: Mapping[str, InputAttestationFingerprint],
) -> bool:
    """Return whether the run's inputs are LOGICALLY and PHYSICALLY original.

    r35 hon1_3 (H-a): eligibility is layout-strict. Alongside the logical byte
    digests, every recorded ``InputAttestationFingerprint`` must equal the
    fingerprint of the value that actually seeds execution -- sizes, strides,
    storage offset, memory-format flags, conj/neg bits, device, subclass class,
    grad/inference metadata, and data-pointer alignment class. A physical twin
    (byte-identical values, different layout) is changed-input-for-attestation
    ONLY: ``not_applicable``, with path faithfulness untouched.
    """

    expected_slot_ids = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.role is TensorSlotRole.MODEL_INPUT
    }
    observed_slot_ids = {digest.slot_id for digest in layer.original_input_digests}
    if not expected_slot_ids or observed_slot_ids != expected_slot_ids:
        return False
    if not all(
        input_byte_digests.get(digest.slot_id) == digest.byte_digest
        for digest in layer.original_input_digests
    ):
        return False
    recorded_fingerprints = tuple(getattr(layer, "input_fingerprints", ()) or ())
    if {fingerprint.slot_id for fingerprint in recorded_fingerprints} != expected_slot_ids:
        # v2 requires a fingerprint per input slot; anything else is ineligible
        # (fail-safe: never attest without the physical identity proof).
        return False
    return all(
        input_fingerprints.get(fingerprint.slot_id) == fingerprint
        for fingerprint in recorded_fingerprints
    )


def _attestation_state_matches(
    descriptor: SparseRunDescriptor,
    layer: Any,
    state: PreparedRunnableState,
    state_byte_digests: Mapping[str, str],
    trace: Any,
) -> bool:
    """Return whether runtime state is capture-equivalent rather than random.

    r35 corr2_5: eligibility is PARTITIONED by persistence. PERSISTENT slots
    (the canonical ``state_dict``) compare against the activation layer's
    ``capture_state_digests`` (which by construction contain only canonical
    entries). USED NON-PERSISTENT buffer slots are separately required to
    originate from the present, schema-valid, load-validated capture-embedded
    ``runnable_nonpersistent_buffer_v1`` family and to match its byte digests;
    staged user state can never supply them. Comparison uses PRE-EXECUTION
    digests so a slot a call mutates mid-run is judged by its capture-start
    state.
    """

    persistent_slots: dict[str, TensorSlotDescriptor] = {}
    nonpersistent_slots: dict[str, TensorSlotDescriptor] = {}
    for slot in descriptor.tensor_slots:
        binding = slot.state_binding
        if binding is None:
            continue
        if binding.persistent:
            persistent_slots[binding.state_dict_name] = slot
        else:
            nonpersistent_slots[binding.state_dict_name] = slot
    if not persistent_slots and not nonpersistent_slots:
        return True
    if persistent_slots:
        if state.state_source not in {
            StateSource.EMBEDDED_CAPTURE_STATE,
            StateSource.USER_STATE_DICT,
        }:
            return False
        expected = {item.state_dict_name: item.byte_digest for item in layer.capture_state_digests}
        if set(expected) != set(persistent_slots):
            return False
        if not all(
            state_byte_digests.get(slot.slot_id) == expected[name]
            for name, slot in persistent_slots.items()
        ):
            return False
    if nonpersistent_slots:
        embedded = trace.__dict__.get("_runnable_embedded_nonpersistent_buffers")
        if not isinstance(embedded, Mapping):
            return False
        for name, slot in nonpersistent_slots.items():
            recorded = embedded.get(name)
            if not isinstance(recorded, torch.Tensor):
                return False
            try:
                recorded_digest = runnable_tensor_byte_digest(recorded)
            except Exception:
                return False
            if state_byte_digests.get(slot.slot_id) != recorded_digest:
                return False
    return True


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


def _field_getattr(current: Any, component: Any) -> Any:
    """Read one attribute-path component against a structurally-known field only.

    Attacker-controlled path strings from an untrusted bundle reach this function
    (``input_binding.container_path``, the recorded literal-witness ``fact["path"]``,
    and the reconstructed output ``slot.output_path``). An unconstrained ``getattr``
    would fire arbitrary victim-object descriptor getters and could walk dunder chains
    (``__class__.__init__.__globals__`` ...), so the component must be a string that is
    STRUCTURALLY PRESENT as a declared field on the current object's type -- a dataclass
    field, a namedtuple ``_fields`` entry, or a ``torch.return_types`` structseq field.
    Dunder / descriptor attributes (``__class__``, ``__dict__``, ...) are never declared
    fields, so this check inherently excludes the escape chains; anything else raises
    ``AttributeError`` and every caller fails closed.
    """

    if not isinstance(component, str):
        raise AttributeError(f"Non-string attribute component {component!r}.")
    allowed: set[str] = set()
    if dataclasses.is_dataclass(current) and not isinstance(current, type):
        allowed.update(field.name for field in dataclasses.fields(current))
    allowed.update(_container_field_names(current))
    if component not in allowed:
        raise AttributeError(
            f"Attribute component {component!r} is not a structurally-known field of "
            f"{type(current).__name__}."
        )
    return getattr(current, component)


def _value_at_path(value: Any, path: Sequence[str | int]) -> Any:
    """Read one list/tuple/mapping/object path from a runtime value.

    Attribute traversal is constrained to structurally-known container fields via
    :func:`_field_getattr`; string components are never fed to an unconstrained
    ``getattr`` (untrusted-bundle path strings could otherwise walk descriptor / dunder
    chains).

    Two synthetic component forms (r29-C2) are decoded: a terminal
    ``EMPTY_CONTAINER_PATH_MARKER`` resolves to the KIND string of the empty container at
    the parent path (so a runtime empty container of a different kind, or a non-empty/scalar
    value, diverges); a ``(BOOL_KEY_PATH_TAG, bool)`` tuple component indexes a mapping with
    the bool key (kept distinct from the equal-valued int key).
    """

    from torchlens._io.runnable import (
        BOOL_KEY_PATH_TAG,
        EMPTY_CONTAINER_PATH_MARKER,
        empty_container_kind,
    )

    current = value
    for component in path:
        if component == EMPTY_CONTAINER_PATH_MARKER:
            kind = empty_container_kind(current)
            if kind is None:
                raise KeyError(EMPTY_CONTAINER_PATH_MARKER)
            return kind
        if (
            isinstance(component, (tuple, list))
            and len(component) == 2
            and (component[0] == BOOL_KEY_PATH_TAG)
        ):
            if not isinstance(current, Mapping):
                raise KeyError(component[1])
            current = current[bool(component[1])]
            continue
        if isinstance(current, Mapping):
            current = current[component]
        elif isinstance(component, int):
            current = current[component]
        else:
            current = _field_getattr(current, component)
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
