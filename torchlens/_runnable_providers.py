"""Public provider entry points and run finalization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from . import _state
from ._runnable_state import (
    RunResourceCeiling,
    prepare_runnable_state,
    validate_run_seed,
)
from .errors import (
    RunPreconditionError,
)
from .runnable import (
    ActivationPayloadLayerDescriptor,
    ContractCheck,
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessReport,
    RunnableDiagnostic,
    RunnableErrorCode,
    RunResult,
    SparseRunDescriptor,
    StateSource,
    derived_witness_completeness,
    mark_trace_path_status,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _bind_runtime_inputs,
        _clone_state_values,
        _execute_loaded_sparse_transaction,
        _raise_first_divergence,
        _raise_monotonic_divergence,
        _require_loaded_sparse_provider,
        _resolve_run_until_plan,
        _run_fork_name,
        _run_report,
        _snapshot_input_byte_digests,
        _snapshot_input_fingerprints,
    )

__all__ = (
    "run_loaded_sparse_trace",
    "_seeded_fork_devices",
    "_seed_run_generators",
    "_host_rng_unreproduced",
    "_finalize_provider_run",
)


def run_loaded_sparse_trace(
    trace: Any,
    inputs: Any,
    *,
    seed: int | None,
    on_divergence: DivergencePolicy,
    until: Any = None,
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
    until:
        Optional static site selection (layer labels / module addresses /
        ``"saved"``); resolved against the loaded trace's settled labels
        before anything binds or executes, truncating the scheduled cut.

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
    # L4 2.1: resolve the static until= selection against the loaded trace's
    # settled labels BEFORE anything binds or executes; the scheduler cut is
    # computed pre-transaction from descriptor facts only.
    until_plan = None if until is None else _resolve_run_until_plan(trace, until)
    # r71 A4 defense-in-depth: run preparation RE-ASSERTS the parser-derived
    # completeness floor before anything executes. Parse already refused a summary
    # differing from the floor; a descriptor that somehow reaches execution with the
    # contradiction fails typed here (never a silently trusted summary).
    if descriptor.witness_completeness is not derived_witness_completeness(
        descriptor.coverage_gaps
    ):
        raise RunPreconditionError(
            "Persisted witness_completeness does not equal the parser-derived "
            "completeness floor; the descriptor is internally contradictory.",
            code=RunnableErrorCode.CONTEXT_FIELD_INVALID.value,
            remedy=(
                "re-save the artifact with the current torchlens producer instead of "
                "hand-editing witness_completeness or coverage_gaps in the descriptor"
            ),
        )
    # r61 corr_2: ONE aggregate resource ceiling per transaction, constructed BEFORE
    # any TorchLens-owned re-materialization clone runs (input mirror, run-time state
    # clone, transaction snapshots) so every clone site shares one byte-guarded
    # boundary. Construction only sums recorded counts, so building it pre-bind
    # changes no accounting.
    ceiling = RunResourceCeiling(descriptor)
    slot_values, input_checks, input_alias_unresolved = _bind_runtime_inputs(
        descriptor, inputs, ceiling
    )
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
    slot_values.update(_clone_state_values(descriptor, prepared_state.slot_values, ceiling))
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
            ceiling=ceiling,
            input_byte_digests=input_byte_digests,
            input_fingerprints=input_fingerprints,
            input_checks=input_checks,
            input_alias_unresolved=input_alias_unresolved,
            prepared_state=prepared_state,
            fork=fork,
            until_plan=until_plan,
        )
    except BaseException:
        _state.unregister_log(fork)
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

    # r79 seed-door mirror: no path may reach raw ``manual_seed`` with a
    # bool/out-of-range seed even if it bypassed the run door.
    validate_run_seed(seed)
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


def _finalize_provider_run(
    *,
    fork: Any,
    output: Any,
    readiness: ReadinessReport,
    state_source: StateSource,
    initializer_policy_version: str | None,
    seed: int | None,
    random_filled_slot_ids: tuple[str, ...],
    contract_checks: tuple[ContractCheck, ...],
    provisional_path_faithfulness: PathFaithfulness,
    provisional_mismatch: RunnableDiagnostic | None,
    numeric_attestation: NumericAttestationStatus,
    divergence_policy: DivergencePolicy,
    nondeterministic_sources: Iterable[str] = (),
    unregister_fork_on_divergence: bool = True,
    state_carried: bool = False,
    truncation: Any = None,
) -> RunResult:
    """THE single provider settlement finalizer (r39 CLASS B sparse<->live parity immunizer).

    Both providers -- loaded sparse and live refresh -- settle EXCLUSIVELY here: the monotonic
    Trace-poison mark (``mark_trace_path_status``), divergence-policy enforcement
    (``_raise_monotonic_divergence``), the one report constructor (``_run_report``, which derives
    ``poisoned`` solely from the faithfulness lattice), and ``RunResult`` construction. Providers
    differ in the EVIDENCE they gather (sparse computes numeric attestation before entering; live
    passes ``NOT_PRESENT``) but never in the settlement DECISION. A source-scan meta-test forbids
    constructing ``RunResult`` or calling ``_run_report`` anywhere else, so a future provider or
    payload family cannot fork the settlement path and reintroduce a parity drift.
    """

    path_faithfulness, mismatch = mark_trace_path_status(
        fork,
        provisional_path_faithfulness,
        provisional_mismatch,
    )
    _raise_monotonic_divergence(
        fork,
        path_faithfulness,
        mismatch,
        divergence_policy,
        unregister_fork=unregister_fork_on_divergence,
    )
    # Annotations TRAVEL POLICY (foldA D6, lane F40a): every settled product
    # presents a re-executed forward, so capture-evidence sub-keys (episode
    # ledger, observer values) are dropped here -- VERIFIED must never coexist
    # with foreign step evidence. Runs after divergence enforcement so a
    # raised-and-rolled-back run never scrubs the caller's live trace.
    # Function-local import: _runnable_execution rebinds this function into
    # its own globals (_rebind_function), so a module-level name is invisible
    # at call time.
    from .capture._annotations_travel import scrub_fresh_execution_annotations

    scrub_fresh_execution_annotations(fork)
    report = _run_report(
        readiness,
        state_source=state_source,
        initializer_policy_version=initializer_policy_version,
        seed=seed,
        random_filled_slot_ids=random_filled_slot_ids,
        contract_checks=contract_checks,
        path_faithfulness=path_faithfulness,
        first_mismatch=mismatch,
        numeric_attestation=numeric_attestation,
        nondeterministic_sources=nondeterministic_sources,
        state_carried=state_carried,
        truncation=truncation,
    )
    return RunResult(output=output, trace=fork, report=report)
