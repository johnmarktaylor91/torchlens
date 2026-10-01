"""Loaded-sparse transaction execution and allocation checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from . import _state
from ._runnable_state import (
    PreparedRunnableState,
    RunResourceCeiling,
    runnable_tensor_byte_digest,
)
from .errors import (
    PathDivergenceError,
    RunCapabilityUnavailableError,
    RuntimeSignatureDriftError,
)
from .intervention.replay import _CallConeNode, _walk_call_cone
from .runnable import (
    ActivationPayloadLayerDescriptor,
    ContractCheck,
    DivergencePolicy,
    InputAttestationFingerprint,
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessReport,
    ReadinessStatus,
    RunnableCallDescriptor,
    RunnableErrorCode,
    RunProvider,
    RunResult,
    SparseRunDescriptor,
    StateSource,
    TensorSlotRole,
    _RunUntilPlan,
)
from .utils.rng import (
    restore_host_rng,
    snapshot_host_rng,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _HOST_RNG_SOURCE_KIND,
        _INPUT_CHECK_UNAVAILABLE,
        _ambient_execution_context_restored,
        _bind_call_outputs,
        _call_execution_context_entered,
        _call_witness_checks,
        _container_spec_reconstruction_lossy,
        _contract_check,
        _control_witness_source_slot_ids,
        _declared_nondeterministic_sources,
        _diagnose_live_output,
        _execute_sparse_call,
        _finalize_provider_run,
        _first_failed_live_input_check,
        _host_rng_unreproduced,
        _input_derived_layout_stale,
        _mode_sensitive_op_unwitnessed,
        _nondeterministic_value_sources,
        _numeric_attestation_check,
        _output_container_spec,
        _output_not_reproduced,
        _path_faithfulness,
        _populate_source_slots,
        _post_execution_contract_checks,
        _pre_call_contract_checks,
        _raise_failed_contract_as_divergence,
        _raise_numeric_attestation_failure,
        _raw_activation_slot_ids,
        _reconstruct_output,
        _run_fork_name,
        _seed_run_generators,
        _seeded_fork_devices,
        _split_mixed_inputs,
        _state_contract_checks,
        _tensor_derived_scalar_stale,
        _tensor_derived_scalar_witness_slot_ids,
        _unbound_state_escape_stale,
        _uninit_taint_reaches,
    )

__all__ = (
    "_execute_loaded_sparse_transaction",
    "run_live_trace",
    "_live_runtime_input_leaves",
)


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
    ceiling: RunResourceCeiling,
    input_byte_digests: Mapping[str, str],
    input_fingerprints: Mapping[str, InputAttestationFingerprint],
    input_checks: tuple[ContractCheck, ...],
    input_alias_unresolved: bool,
    prepared_state: PreparedRunnableState,
    fork: Any,
    until_plan: _RunUntilPlan | None = None,
) -> RunResult:
    """Execute one sparse transaction whose caller owns rollback on escape."""

    # L4 2.1: the until= cut is computed BEFORE the transaction opens, applied
    # where the transaction walks descriptor.calls, never by filtering results
    # afterward. The executed set is the sequential prefix through the last
    # requested call; the regime label is "closure" only when that prefix IS
    # the dependency closure (no candidate skip exists), else the disclosed
    # sequential_prefix fallback (a strict superset, always honest).
    executed_calls: tuple[RunnableCallDescriptor, ...] = tuple(descriptor.calls)
    until_regime: str | None = None
    until_cause: str | None = None
    if until_plan is not None:
        executed_calls, until_regime, until_cause = _loaded_until_cut(descriptor, until_plan)

    # r59/r61: the ONE aggregate resource ceiling for this transaction -- constructed
    # by ``run_loaded_sparse_trace`` before input binding, threaded here -- bounds
    # realized output count (per-call projection + bind backstop) and re-materialization
    # clone bytes at every snapshot site. The run-prep retention floor
    # (prepare_runnable_state) already refused a guaranteed-OOM retained population
    # before we got here.
    # Escape-source witness slots must be digested at their production point (before
    # any later in-place op mutates the live tensor), matching the save-side digest.
    escape_witness_slot_ids = _tensor_derived_scalar_witness_slot_ids(descriptor)
    witness_source_snapshots: dict[str, torch.Tensor] = {}
    _populate_source_slots(
        fork,
        descriptor,
        slot_values,
        ceiling=ceiling,
        witness_slot_ids=escape_witness_slot_ids,
        witness_source_snapshots=witness_source_snapshots,
    )
    contract_checks: list[ContractCheck] = [
        *input_checks,
        *_state_contract_checks(descriptor, slot_values),
    ]

    # ``contract_checks`` only ever grows by appends and ``ContractCheck`` is frozen,
    # so the earliest failed check is found by visiting each check exactly once across
    # the whole transaction instead of rescanning the cumulative list at every call
    # boundary (O(calls^2) ``passed`` reads on long replays). Same earliest check,
    # same raise, same policy handling as scanning the full list front-to-back.
    scan_cursor = 0
    first_failed_check: ContractCheck | None = None

    def first_failed_contract_so_far() -> ContractCheck | None:
        """Earliest failed check appended so far, or ``None`` while all pass.

        Each check is visited at most once across the whole transaction, which
        is sound only because ``contract_checks`` grows by append and
        ``ContractCheck`` is frozen: no already-scanned check can later fail.
        """

        nonlocal scan_cursor, first_failed_check
        while first_failed_check is None and scan_cursor < len(contract_checks):
            check = contract_checks[scan_cursor]
            scan_cursor += 1
            if not check.passed:
                first_failed_check = check
        return first_failed_check

    def raise_first_divergence_incremental() -> None:
        """Raise the earliest failed contract check as a divergence, if any failed.

        A no-op under :attr:`DivergencePolicy.RETURN_DIVERGED`, where the
        failure is carried in the poisoned result instead of raised.
        """

        failed = first_failed_contract_so_far()
        if failed is None or divergence_policy is DivergencePolicy.RETURN_DIVERGED:
            return
        _raise_failed_contract_as_divergence(failed, fork=fork)

    raise_first_divergence_incremental()
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
    # r55 C3: registry lookup for the per-call allocation preflight (namespace /
    # qualname drive the size-driving classification).
    registry_by_id = {entry.registry_id: entry for entry in descriptor.callable_registry}
    # Run-invariant ``descriptor.tensor_slots`` indexes for the per-call output bind.
    # The descriptor is frozen for the whole replay, so these are built ONCE here
    # instead of once per call (they were O(calls x slots) rebuilds).
    slots_by_id = {slot.slot_id: slot for slot in descriptor.tensor_slots}
    state_slot_ids = frozenset(
        slot.slot_id
        for slot in descriptor.tensor_slots
        if slot.role in {TensorSlotRole.PARAMETER, TensorSlotRole.BUFFER}
    )
    # Producer slot id -> its version-alias slot ids (``version_of`` AND
    # ``producer_slot_id`` both naming the producer), in ``tensor_slots`` order, so
    # the per-output bind stages aliases by O(1) lookup instead of scanning every
    # slot for every produced output (O(outputs x slots)).
    version_alias_ids: dict[str, tuple[str, ...]] = {}
    for slot in descriptor.tensor_slots:
        if slot.version_of is not None and slot.version_of == slot.producer_slot_id:
            version_alias_ids[slot.version_of] = (
                *version_alias_ids.get(slot.version_of, ()),
                slot.slot_id,
            )

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
                raise_first_divergence_incremental()
                try:
                    with _call_execution_context_entered(call.execution_context):
                        output = _execute_sparse_call(
                            call,
                            callables[call.call_id],
                            slot_values,
                            registry_entry=registry_by_id.get(call.registry_id),
                            ceiling=ceiling,
                        )
                except RuntimeSignatureDriftError:
                    # r39 corr2_4: under ``return_diverged`` an admitted-but-INEXECUTABLE
                    # divergent input (wrong feature shape/dtype/device) reaches the resolved
                    # callable and throws. If an input contract check ALREADY failed, this is
                    # input DIVERGENCE, not resolved-callable signature drift -- roll back and
                    # raise the typed ``PathDivergenceError`` carrying the first failed check.
                    # Genuine drift (no prior failed check) keeps ``RuntimeSignatureDriftError``.
                    failed = first_failed_contract_so_far()
                    if failed is not None:
                        _raise_failed_contract_as_divergence(failed, fork=fork)
                    raise
                call_outputs[call.call_id] = output
                contract_checks.extend(
                    _bind_call_outputs(
                        call,
                        output,
                        slot_values,
                        fork,
                        ceiling=ceiling,
                        before_versions=before_versions,
                        attestation_slot_ids=attestation_slot_ids,
                        attestation_slot_values=attestation_slot_values,
                        slots=slots_by_id,
                        state_slot_ids=state_slot_ids,
                        version_alias_ids=version_alias_ids,
                        witness_slot_ids=escape_witness_slot_ids,
                        witness_source_snapshots=witness_source_snapshots,
                    )
                )
                contract_checks.extend(_call_witness_checks(descriptor, call, slot_values))
                raise_first_divergence_incremental()

            _walk_call_cone(executed_calls, execute_call)
    except BaseException:
        # R36-7: an escaping call loop (typed divergence raise, signature
        # drift, native failure) pins this frame inside the exception
        # traceback, so a caller retaining the exception would pin every
        # staged device copy indefinitely. Clear the staging containers IN
        # PLACE before propagating; the caller's rollback owns fork
        # unregistration.
        slot_values.clear()
        call_outputs.clear()
        attestation_slot_values.clear()
        witness_source_snapshots.clear()
        raise
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
        _state.unregister_log(fork)
        raise RuntimeError(
            "Internal invariant violation: CUDA became initialized during a "
            "seeded sparse run whose descriptor named no CUDA device; the "
            "capture device summary is incomplete."
        )

    if until_plan is not None:
        # L4 2.2/2.4 truncated loaded-sparse settlement: the skipped region is a
        # DISCLOSED set, semantically "not-run". There is no full-forward output
        # to reconstruct (output is None; callers read executed values off the
        # result trace), the output/post-execution checks concern the skipped
        # region and are replaced by the truncation ceiling, and the
        # slot-consuming stale classifiers are skipped (they derive UNVERIFIABLE
        # ceilings the run_truncated cap already imposes; executed-region
        # CONTRADICTIONS still settle DIVERGED through the in-loop checks).
        output = None
        # Mark the fork as a truncated run product: the run door refuses a
        # re-run of it typed (3.3.2), and the disclosure names the skipped set.
        fork.__dict__["_run_truncation_skipped_raw_labels"] = tuple(until_plan.skipped_raw_labels)
        tensor_derived_scalar_stale = False
        unbound_state_escape_stale = False
        input_derived_layout_stale = False
        container_reconstruction_lossy = False
        output_not_reproduced = False
    else:
        output = _reconstruct_output(
            descriptor, slot_values, fork, ceiling=ceiling, call_outputs=call_outputs
        )
        contract_checks.extend(
            _post_execution_contract_checks(
                descriptor,
                inputs=inputs,
                output=output,
                slot_values=slot_values,
                fork=fork,
            )
        )
        raise_first_divergence_incremental()
        tensor_derived_scalar_stale = _tensor_derived_scalar_stale(
            descriptor, slot_values, witness_source_snapshots
        )
        unbound_state_escape_stale = _unbound_state_escape_stale(descriptor, slot_values)
        # r73 F1: compared against the RAW user input tree (pre-clone leaves; the run
        # executed on defensive clones, so runtime strides are unchanged here).
        input_derived_layout_stale = _input_derived_layout_stale(descriptor, inputs)
    mode_sensitive_op_unwitnessed = _mode_sensitive_op_unwitnessed(descriptor)
    # r53 hon_2: ONE load-side classifier settles declared nondeterministic value
    # sources; the branch ceiling, the attestation gate, and the report signal
    # all consult it (the r52 raise-vs-not_applicable inconsistency is
    # structurally unrepresentable).
    value_source_taint = _nondeterministic_value_sources(descriptor)
    nondeterministic_control_source = _uninit_taint_reaches(
        value_source_taint, _control_witness_source_slot_ids(descriptor)
    )
    declared_nondeterministic_sources = _declared_nondeterministic_sources(
        descriptor, value_source_taint
    )
    if until_plan is None:
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
        nondeterministic_control_source=nondeterministic_control_source,
        input_derived_layout_stale=input_derived_layout_stale,
        run_truncated=until_plan is not None,
    )
    eligibility_verdict = provisional_verdict
    inherited_status = fork._runnable.path_faithfulness
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
    return _finalize_provider_run(
        fork=fork,
        output=output,
        readiness=readiness,
        state_source=prepared_state.state_source,
        initializer_policy_version=prepared_state.initializer_policy_version,
        seed=prepared_state.seed,
        random_filled_slot_ids=prepared_state.random_filled_slot_ids,
        contract_checks=tuple(contract_checks),
        provisional_path_faithfulness=provisional_verdict,
        provisional_mismatch=provisional_mismatch,
        numeric_attestation=numeric_attestation,
        divergence_policy=divergence_policy,
        nondeterministic_sources=declared_nondeterministic_sources,
        truncation=(
            None
            if until_plan is None
            else _run_truncation_record(
                until_plan, until_regime or "sequential_prefix", until_cause
            )
        ),
    )


def _resolve_run_until_plan(trace: Any, until: Any) -> _RunUntilPlan:
    """Resolve one static-form ``until=`` selection against the source trace.

    Accepts the static site-selection forms (final layer labels, module
    addresses, the literal ``"saved"``, or a list of those); the selection is
    INCLUSIVE (``until=x`` computes ``x`` and stops). Predicate/selector forms
    refuse typed until the S4 contract merge; other non-string forms refuse
    typed. Unknown labels surface the standard typed lookup refusal.
    """

    tokens = _validated_until_tokens(until)
    resolved: list[Any] = []
    requested: list[str] = []
    requested_layer_labels: list[str] = []
    for token in tokens:
        site_layers, site_names, site_layer_labels = _resolve_until_site(trace, token)
        resolved.extend(site_layers)
        requested.extend(site_names)
        requested_layer_labels.extend(site_layer_labels)
    stop_raw_index = _until_stop_raw_index(resolved)
    executed, skipped, stopped_at = _until_execution_partition(trace, stop_raw_index)
    return _RunUntilPlan(
        requested_sites=tuple(dict.fromkeys(requested)),
        stop_raw_index=stop_raw_index,
        stopped_at=stopped_at,
        executed_raw_labels=tuple(executed),
        skipped_raw_labels=tuple(skipped),
        requested_layer_labels=tuple(dict.fromkeys(requested_layer_labels)),
    )


def _validated_until_tokens(until: Any) -> list[Any]:
    """Normalize the until= selection to tokens; refuse empty/non-static forms."""

    from ._errors import InvalidArgumentError

    tokens = list(until) if isinstance(until, (list, tuple, set, frozenset)) else [until]
    if not tokens:
        raise InvalidArgumentError(
            "until= received an empty selection; pass at least one layer label, "
            "module address, or the literal 'saved'",
            code="run_until_form_invalid",
            remedy="pass a non-empty static site selection",
            argument="until",
        )
    for token in tokens:
        if callable(token) or hasattr(token, "__torchlens_predicate__"):
            raise RunCapabilityUnavailableError(
                "Predicate/selector forms of until= are gated on the S4 predicate "
                "contract merge and refuse typed until it lands; the static forms "
                "(layer labels, module addresses, 'saved') are available now.",
                code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
                detection_stage="predicate_surface_pending",
                remedy="pass static layer labels, module addresses, or 'saved'",
            )
        if not isinstance(token, str):
            raise InvalidArgumentError(
                f"until= accepts layer-label/module-address strings or 'saved'; "
                f"got {type(token).__name__!r}",
                code="run_until_form_invalid",
                remedy="pass static layer labels, module addresses, or 'saved'",
                argument="until",
            )
    return tokens


def _resolve_until_site(trace: Any, token: str) -> tuple[list[Any], list[str], list[str]]:
    """Resolve ONE until= token to (layers, requested names, layer labels)."""

    from ._errors import InvalidArgumentError

    if token == "saved":
        saved_layers = [
            layer
            for layer in trace.layer_list
            if bool(getattr(layer, "has_saved_activation", False))
            and layer.layer_type not in ("input", "output")
        ]
        if not saved_layers:
            raise InvalidArgumentError(
                "until='saved' resolved to an empty site set: this capture "
                "retained no saved activations",
                code="run_until_form_invalid",
                remedy="capture with save= selection, or pass explicit labels",
                argument="until",
            )
        labels = [layer.layer_label for layer in saved_layers]
        return saved_layers, labels, list(labels)
    keys = trace.layer_dict_all_keys
    if token in keys:
        layer = keys[token]
        return [layer], [layer.layer_label], [layer.layer_label]
    module_accessor = getattr(trace, "modules", None)
    module = None
    if module_accessor is not None and token in module_accessor:
        module = module_accessor[token]
    if module is not None:
        member_labels = tuple(getattr(module, "layer_labels", ()) or ())
        members = [keys[label] for label in member_labels if label in keys]
        if members:
            return members, [token], [member.layer_label for member in members]
    # Unknown site: surface the standard typed lookup refusal with fuzzy
    # feedback (never a bespoke vocabulary row for a plain lookup miss).
    trace[token]
    raise InvalidArgumentError(
        f"until= site {token!r} did not resolve to layers",
        code="run_until_form_invalid",
        remedy="pass a resolvable layer label or module address",
        argument="until",
    )


def _until_stop_raw_index(resolved: list[Any]) -> int:
    """The stop frontier: the max recorded raw index across resolved sites."""

    from ._errors import InvalidArgumentError

    stop_raw_index = -1
    for layer in resolved:
        for op in getattr(layer, "ops", None) or (layer,):
            raw_index = getattr(op, "raw_index", None)
            if raw_index is not None:
                stop_raw_index = max(stop_raw_index, int(raw_index))
    if stop_raw_index < 0:
        raise InvalidArgumentError(
            "until= selection resolved to sites without recorded operation "
            "indexes; the stop frontier is undecidable",
            code="run_until_form_invalid",
            remedy="pass sites that resolve to captured operations",
            argument="until",
        )
    return stop_raw_index


def _until_execution_partition(
    trace: Any, stop_raw_index: int
) -> tuple[list[str], list[str], str | None]:
    """Partition the source layers into executed/skipped at the stop frontier."""

    executed: list[str] = []
    skipped: list[str] = []
    stopped_at: str | None = None
    for layer in trace.layer_list:
        raw_index = getattr(layer, "raw_index", None)
        raw_label = layer._layer_label_raw
        if layer.layer_type == "output" or raw_index is None or int(raw_index) > stop_raw_index:
            skipped.append(raw_label)
            continue
        executed.append(raw_label)
        if int(raw_index) == stop_raw_index:
            stopped_at = layer.layer_label
    return executed, skipped, stopped_at


def _loaded_until_cut(
    descriptor: SparseRunDescriptor, plan: _RunUntilPlan
) -> tuple[tuple[RunnableCallDescriptor, ...], str, str | None]:
    """Compute the until= scheduler cut over one recorded call schedule (L4 2.1).

    The EXECUTED set is always the sequential prefix of the recorded schedule
    through the last call producing a requested site -- a strict superset of
    any dependency closure, parent-closed by the schedule's topological order,
    so the cut is sound by construction. The REGIME label is ``"closure"`` only
    when that prefix IS the widened dependency closure (C1 tensor deps union C3
    declared-state deps union C5 control-witness deps leave no candidate skip);
    any candidate skip is unproven independent this release (the C4
    certified-fresh vocabulary is not yet shipped) and the run discloses the
    ``sequential_prefix`` regime with the ``unprovable_independence`` cause tag.
    """

    from ._errors import InvalidArgumentError

    calls = tuple(descriptor.calls)
    requested = set(plan.requested_layer_labels)

    def _bases(call: RunnableCallDescriptor) -> set[str]:
        """Return the pass-stripped base labels of one recorded call."""

        return {op_label.rsplit(":", 1)[0] for op_label in call.op_labels}

    target_indexes = [index for index, call in enumerate(calls) if _bases(call) & requested]
    if not target_indexes:
        raise InvalidArgumentError(
            "until= selection resolved to sites with no producing recorded call "
            "(input-only or synthetic sites cannot anchor a stop frontier)",
            code="run_until_form_invalid",
            remedy="pass sites produced by recorded operations",
            argument="until",
        )
    stop_index = max(target_indexes)
    prefix = calls[: stop_index + 1]
    by_id = {call.call_id: call for call in prefix}
    op_label_to_call = {op_label: call.call_id for call in prefix for op_label in call.op_labels}
    state_slot_ids = {
        slot.slot_id for slot in descriptor.tensor_slots if slot.state_binding is not None
    }
    closure = _until_dependency_closure(
        prefix=prefix,
        by_id=by_id,
        op_label_to_call=op_label_to_call,
        seed_call_ids=[calls[index].call_id for index in target_indexes],
        state_slot_ids=state_slot_ids,
    )
    candidate_skips = {call.call_id for call in prefix} - closure
    if candidate_skips:
        return prefix, "sequential_prefix", "unprovable_independence"
    return prefix, "closure", None


def _until_dependency_closure(
    *,
    prefix: tuple[RunnableCallDescriptor, ...],
    by_id: dict[str, RunnableCallDescriptor],
    op_label_to_call: dict[str, str],
    seed_call_ids: list[str],
    state_slot_ids: set[str],
) -> set[str]:
    """Widened dependency closure: C1 tensor deps, C5 control-witness deps,
    then the C3 declared-state widening over shared state slots."""

    closure: set[str] = set()
    frontier = list(seed_call_ids)
    while frontier:
        call_id = frontier.pop()
        if call_id in closure or call_id not in by_id:
            continue
        closure.add(call_id)
        call = by_id[call_id]
        frontier.extend(call.parent_call_ids)
        for edge in call.control_dependencies:
            parent_call_id = op_label_to_call.get(edge.parent_op_label)
            if parent_call_id is not None:
                frontier.append(parent_call_id)
    closure_state_slots = {
        argument.slot_id
        for call_id in closure
        for argument in by_id[call_id].tensor_arguments
        if argument.slot_id in state_slot_ids
    }
    if closure_state_slots:
        for call in prefix:
            if call.call_id in closure:
                continue
            if any(argument.slot_id in closure_state_slots for argument in call.tensor_arguments):
                closure.add(call.call_id)
    return closure


def _run_truncation_record(plan: _RunUntilPlan, regime: str, cause: str | None = None) -> Any:
    """Build the RunTruncation disclosure for one truncated run."""

    from hashlib import sha256

    from .runnable import RunTruncation

    digest = sha256("\n".join(plan.skipped_raw_labels).encode("utf-8")).hexdigest()[:16]
    return RunTruncation(
        regime=regime,
        requested_sites=plan.requested_sites,
        stopped_at=plan.stopped_at,
        executed_count=len(plan.executed_raw_labels),
        skipped_count=len(plan.skipped_raw_labels),
        skipped_digest=digest,
        cause=cause,
    )


def _require_live_source_model(trace: Any) -> Any:
    """Resolve the weakly-held live source model or refuse typed."""

    source_ref = getattr(trace, "_source_model_ref", None)
    model = source_ref() if source_ref is not None else None
    if model is None:
        raise RunCapabilityUnavailableError(
            "The live Trace no longer retains its source model: the trace "
            "holds it only weakly, so live-run availability depends on the "
            "caller keeping a strong reference (an inline-constructed model "
            "is collected at the first gc pass after capture). Keep the "
            "model alive, or save/load a runnable artifact instead.",
            code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
            provider=RunProvider.LIVE,
        )
    return model


def _attempt_live_forward(
    fork: Any,
    model: Any,
    inputs_pair: tuple[Any, Any],
    until_plan: _RunUntilPlan | None,
    ctx: _LiveFinalizeContext,
) -> tuple[ContractCheck, BaseException] | None:
    """Run the refresh forward with the soft input-contract classifier.

    r41 hon1_3 (corr2_4 parity): the soft input-contract checks are
    PRE-computed BEFORE the forward -- a failing forward may in-place-mutate
    an input leaf (``resize_``) before the failing op, so a post-hoc metadata
    read could misclassify. The precompute has ZERO admission power: a
    divergent-but-executable input (changed batch / seq-len) still runs and
    may honestly settle VERIFIED (fresh-refresh semantics); classification
    happens only at native-failure time.

    Returns
    -------
    tuple[ContractCheck, BaseException] | None
        ``None`` when the refresh projected cleanly. When the refresh projector's
        generic graph-signature arm refused (the executed graph no longer matches
        the captured one -- value-dependent control flow took another branch), the
        failed ``call_structure_mismatch`` contract check paired with the projector
        error, for :func:`_settle_live_graph_change` to enforce the caller's
        ``on_divergence`` policy through the shared spine (AUD-HONESTY H3). The
        typed D18 buffer-sink refusal and native model failures re-raise raw.
    """

    input_args, input_kwargs = inputs_pair
    first_failed = _first_failed_live_input_check(ctx.trace, input_args, input_kwargs)
    if first_failed is _INPUT_CHECK_UNAVAILABLE:
        # This consumer is SOFT (consulted only at native-failure time): an
        # unavailable classifier means the failure cannot be classified as
        # input divergence, so the native error re-raises raw -- the old
        # ``None`` behavior, now explicit (R22-2).
        first_failed = None
    try:
        fork.save_new_outs(
            model,
            input_args,
            input_kwargs=input_kwargs,
            random_seed=ctx.seed,
            _run_until_plan=until_plan,
        )
    except Exception as exc:  # not BaseException: KeyboardInterrupt/SystemExit stay raw
        if first_failed is not None:
            # An admitted-but-inexecutable DIVERGENT input surfaces as the typed
            # PathDivergenceError carrying the first failed input check, with the
            # native error chained as ``__cause__`` (corr2_4 on both providers).
            # A native failure on a NON-divergent input re-raises raw below -- a
            # genuinely failing model is not a divergence.
            _raise_failed_contract_as_divergence(first_failed, fork=None, cause=exc)
        graph_change_detail = getattr(exc, "refresh_graph_change_detail", None)
        if isinstance(exc, ValueError) and graph_change_detail is not None:
            # H3: the projector's generic graph-signature arm IS a path divergence
            # (the recorded schedule was not re-executed); hand it to the policy
            # spine instead of leaking the projector's bare ``ValueError``.
            return _live_graph_change_check(exc, str(graph_change_detail)), exc
        raise
    return None


def _live_graph_change_check(error: BaseException, detail: str) -> ContractCheck:
    """Build the failed live-provider contract check for a refresh graph change (H3).

    Code ``call_structure_mismatch`` (the executed op sequence differs from the
    recorded one), stage ``refresh_graph_signature``; the message keeps the pinned
    "computational graph changed" phrase for historical callers and ends with the
    remedy. Consumers branch on the code, never the text.
    """

    return _contract_check(
        "live_refresh_graph_signature",
        False,
        RunnableErrorCode.CALL_STRUCTURE_MISMATCH,
        f"{error} Remedy: re-capture with the new inputs (tl.trace(model, inputs)); the "
        "live provider refreshes only a forward whose executed graph matches the "
        "captured one",
        details=(
            ("reason", "refresh_graph_changed"),
            ("detail", detail),
            ("detection_stage", "refresh_graph_signature"),
            (
                "remedy",
                "re-capture with the new inputs; the live provider refreshes only a "
                "forward whose executed graph matches the captured one",
            ),
        ),
    )


def _settle_live_graph_change(
    fork: Any,
    check: ContractCheck,
    error: BaseException,
    ctx: _LiveFinalizeContext,
) -> RunResult:
    """Enforce ``on_divergence`` for a live refresh whose graph changed (H3).

    ``RAISE`` (default) raises the typed ``PathDivergenceError`` carrying the
    failed check with the projector error chained, exactly like the sparse
    provider's first-contradiction raise. ``RETURN_DIVERGED`` settles through the
    ONE provider finalizer: ``path_faithfulness=DIVERGED``, poisoned, ``output``
    ``None`` (no faithful refreshed output exists -- the fork retains the SOURCE
    capture's values, which the poison mark keeps every faithful consumer from
    reading as fresh).

    Interim typing bridge: ``PathDivergenceError`` does not (yet) carry
    ``ValueError`` in its lineage while the historical ``save_new_outs`` /
    ``run()`` callers pin ``except ValueError`` on this exact refusal. Until the
    one-line lineage change lands in ``torchlens.errors.runnable`` (recorded
    OUT-OF-FENCE by lane W051-HONESTY; this bridge then dissolves on its own), the
    RAISE arm re-raises the projector's ``ValueError`` decorated with the SAME
    structured facts the typed error carries (``fields["code"]`` /
    ``["path_faithfulness"]`` / ``["first_mismatch"]`` / ``["contract_check"]`` /
    ``["remedy"]``), so code branching on the structured fields already works.
    """

    diagnostic = check.diagnostic
    if ctx.divergence_policy is DivergencePolicy.RAISE:
        if issubclass(PathDivergenceError, ValueError):
            _raise_failed_contract_as_divergence(check, fork=None, cause=error)
        remedy = dict(diagnostic.details).get("remedy", "") if diagnostic is not None else ""
        error.fields = {  # type: ignore[attr-defined]
            "code": RunnableErrorCode.CALL_STRUCTURE_MISMATCH.value,
            "path_faithfulness": PathFaithfulness.DIVERGED,
            "first_mismatch": diagnostic,
            "contract_check": check,
            "remedy": remedy,
        }
        raise error
    return _finalize_provider_run(
        fork=fork,
        output=None,
        readiness=_live_readiness_report(ctx.trace),
        state_source=StateSource.LIVE_MODEL_STATE,
        initializer_policy_version=None,
        seed=ctx.seed,
        random_filled_slot_ids=(),
        contract_checks=(check,),
        provisional_path_faithfulness=PathFaithfulness.DIVERGED,
        provisional_mismatch=diagnostic,
        numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
        divergence_policy=ctx.divergence_policy,
        nondeterministic_sources=_live_nondeterministic_sources(ctx.trace),
        state_carried=ctx.carry_state,
    )


def _split_live_inputs(inputs: Any) -> tuple[Any, Any]:
    """Split the unified inputs= tree into (args, kwargs) when mixed-form."""

    if (
        isinstance(inputs, Mapping)
        and {"args", "kwargs"}.issubset(inputs)
        and set(inputs).issubset({"args", "kwargs"})
    ):
        args, kwargs = _split_mixed_inputs(inputs)
        return list(args), dict(kwargs)
    return inputs, None


def _install_live_until_latch(until_plan: _RunUntilPlan) -> None:
    """Install the run-scoped latch-once halt predicate on the plan."""

    def _until_latch(ctx: Any, _plan: _RunUntilPlan = until_plan) -> bool:
        """Fire once at the first boundary past the last requested site."""

        raw_index = getattr(ctx, "raw_index", None)
        if raw_index is not None and int(raw_index) >= _plan.stop_raw_index:
            _plan.fired = True
            return True
        return False

    until_plan.halt_predicate = _until_latch


@dataclass(frozen=True)
class _LiveFinalizeContext:
    """Shared facts both live finalize paths need."""

    trace: Any
    seed: int | None
    divergence_policy: DivergencePolicy
    carry_state: bool
    prior_log_ids: frozenset[int]


def _live_nondeterministic_sources(trace: Any) -> tuple[str, ...]:
    """Capture-side host-RNG disclosure shared by both live finalize paths."""

    runnable_seam = getattr(trace, "_runnable", None)
    if runnable_seam is not None and bool(runnable_seam.host_rng_consumed):
        return (_HOST_RNG_SOURCE_KIND,)
    return ()


def _live_readiness_report(trace: Any) -> ReadinessReport:
    """The constant live-provider readiness report."""

    return ReadinessReport(
        status=ReadinessStatus.READY,
        provider=RunProvider.LIVE,
        backend=str(getattr(trace, "backend", "torch")),
        capability="live_model_fast_capture",
        resolver_records=(),
        state_sources_available=(StateSource.LIVE_MODEL_STATE,),
        witness_completeness=None,
        diagnostics=(),
    )


def _finalize_truncated_live_run(
    fork: Any,
    until_plan: _RunUntilPlan,
    ctx: _LiveFinalizeContext,
) -> RunResult:
    """Finalize the truncated-success live path (L4 3.2).

    The internal refresh capture settled HALTED -- honestly, on the
    throwaway -- and must never be enumerable through a public surface:
    every log minted by this run is unregistered EXCEPT the result fork
    (the fork carve-out: a verbatim exception-bracket reuse would drop the
    result the caller is handed).
    """

    if not until_plan.fired:
        raise RuntimeError(
            "Internal invariant violation: the live until= latch never "
            "fired although the stop index derives from the target's own "
            "recorded operation indexes."
        )
    for log in _state.list_logs():
        if id(log) not in ctx.prior_log_ids and log is not fork:
            _state.unregister_log(log)
    # RunResult.output is None under live truncation ([PROV]): there IS
    # no full-forward return value; callers read executed-prefix values
    # off the result trace. The live output-reconstruction contract
    # check is REPLACED by the truncation ceiling (an
    # OUTPUT_STRUCTURE_MISMATCH would misattribute a deliberate stop).
    return _finalize_provider_run(
        fork=fork,
        output=None,
        readiness=_live_readiness_report(ctx.trace),
        state_source=StateSource.LIVE_MODEL_STATE,
        initializer_policy_version=None,
        seed=ctx.seed,
        random_filled_slot_ids=(),
        contract_checks=(),
        provisional_path_faithfulness=PathFaithfulness.UNVERIFIABLE,
        provisional_mismatch=None,
        numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
        divergence_policy=ctx.divergence_policy,
        nondeterministic_sources=_live_nondeterministic_sources(ctx.trace),
        state_carried=ctx.carry_state,
        truncation=_run_truncation_record(until_plan, "live_stop_after"),
    )


def _finalize_full_live_run(fork: Any, ctx: _LiveFinalizeContext) -> RunResult:
    """Finalize the full (untruncated) live path with the output honesty gate."""

    diagnosis = _diagnose_live_output(fork)
    output, faithful = diagnosis.output, diagnosis.faithful
    # Honesty gate: only a faithfully reconstructed output (exact container type
    # and non-tensor leaves) is VERIFIED. An output we could only approximate
    # from naive leaf paths -- or a lossy container type (computed non-field/non-key
    # state, __slots__, a data-descriptor field) -- is UNVERIFIABLE, never blessed
    # with a wrong object. The diagnosis names WHY and the remedy (AUD-HONESTY
    # H1/H2): a declared container whose contract the default capture did not
    # persist, an opaque tensor-holding leaf (HuggingFace DynamicCache), or an
    # opaque root.
    provisional = PathFaithfulness.VERIFIED if faithful else PathFaithfulness.UNVERIFIABLE
    check_message = (
        "Live output could not be faithfully reconstructed from its captured container contract."
    )
    check_details: tuple[tuple[str, str], ...] = ()
    if not faithful:
        reason = diagnosis.reason or "unknown"
        remedy = diagnosis.remedy or ""
        check_message = (
            f"{check_message} Reason: {reason}"
            f"{' (' + diagnosis.detail + ')' if diagnosis.detail else ''}. Remedy: {remedy}"
        )
        check_details = (
            ("reason", reason),
            ("detail", diagnosis.detail),
            ("remedy", remedy),
        )
    # Deephunt F2: the live report must declare the same capture-side host-RNG
    # evidence the sparse producer derives ``host_rng`` from. VERIFIED stays
    # correct for this provider (the fresh refresh is its own oracle-1 run),
    # but two successive live runs of a host-RNG model can legitimately differ,
    # so an empty tuple would misread as a deterministic verified.
    return _finalize_provider_run(
        fork=fork,
        output=output,
        readiness=_live_readiness_report(ctx.trace),
        state_source=StateSource.LIVE_MODEL_STATE,
        initializer_policy_version=None,
        seed=ctx.seed,
        random_filled_slot_ids=(),
        contract_checks=(
            _contract_check(
                "live_output_reconstruction",
                faithful,
                RunnableErrorCode.OUTPUT_STRUCTURE_MISMATCH,
                check_message,
                details=check_details,
            ),
        ),
        provisional_path_faithfulness=provisional,
        provisional_mismatch=None,
        numeric_attestation=NumericAttestationStatus.NOT_PRESENT,
        divergence_policy=ctx.divergence_policy,
        nondeterministic_sources=_live_nondeterministic_sources(ctx.trace),
        state_carried=ctx.carry_state,
    )


@dataclass(frozen=True)
class _LiveRunOptions:
    """The L4 live-run shaping knobs threaded from the public ``run`` surface."""

    carry_state: bool = False
    until: Any = None


def run_live_trace(
    trace: Any,
    inputs: Any,
    *,
    seed: int | None,
    on_divergence: DivergencePolicy | str = DivergencePolicy.RAISE,
    options: _LiveRunOptions | None = None,
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
    options:
        The L4 run-shaping knobs: ``carry_state=True`` lets the run's
        declared-state mutations survive on the live model instead of being
        restored by the snapshot-restore bracket, and ``until=`` is the
        optional static site selection (layer labels / module addresses /
        ``"saved"``), resolved against the source trace's settled final
        labels and installed as a run-scoped halt latch on the refresh
        capture.

    Returns
    -------
    RunResult
        Structured output, refreshed fork, and live-provider report.

    Raises
    ------
    RunCapabilityUnavailableError
        If the live source model is no longer available.
    """

    if options is None:
        options = _LiveRunOptions()
    carry_state = options.carry_state
    until = options.until
    divergence_policy = DivergencePolicy(on_divergence)
    model = _require_live_source_model(trace)
    # L4 2.3 live until=: resolve the static site selection against the SOURCE
    # trace's settled final labels and install a run-scoped halt latch on the
    # internal refresh capture (latch-once, fired at the first boundary after
    # the last requested site is produced -- save-then-halt keeps it INCLUSIVE).
    until_plan = None if until is None else _resolve_run_until_plan(trace, until)
    if until_plan is not None:
        _install_live_until_latch(until_plan)
    # L4 5.2 snapshot-restore bracket: the default live run leaves the model
    # bit-identical. The snapshot is taken and VALIDATED before the fork and
    # before any forward (fail-before-execute, typed run_state_snapshot_unsupported);
    # carry_state=True is the sole opt-in that skips the whole bracket.
    state_snapshot = None
    if not carry_state:
        from ._runnable_state import snapshot_live_declared_state

        state_snapshot = snapshot_live_declared_state(model)
    prior_log_ids = frozenset(id(log) for log in _state.list_logs())
    fork = trace._fork_trace(name=_run_fork_name(trace))
    finalize_ctx = _LiveFinalizeContext(
        trace=trace,
        seed=seed,
        divergence_policy=divergence_policy,
        carry_state=carry_state,
        prior_log_ids=prior_log_ids,
    )
    try:
        graph_change = _attempt_live_forward(
            fork, model, _split_live_inputs(inputs), until_plan, finalize_ctx
        )
        if graph_change is not None:
            return _settle_live_graph_change(fork, graph_change[0], graph_change[1], finalize_ctx)
        if until_plan is not None:
            return _finalize_truncated_live_run(fork, until_plan, finalize_ctx)
        return _finalize_full_live_run(fork, finalize_ctx)
    except BaseException:
        _state.unregister_log(fork)
        for log in _state.list_logs():
            if id(log) not in prior_log_ids:
                _state.unregister_log(log)
        raise
    finally:
        # L4 5.2: RESTORE runs in finally on EVERY path (success, divergence,
        # callable exception, rollback), symmetric with the RNG fork/restore
        # discipline. A SECONDARY restore failure marks and raises typed (5.4).
        if state_snapshot is not None:
            _restore_declared_state_or_mark(trace, fork, state_snapshot)


def _restore_declared_state_or_mark(trace: Any, fork: Any, snapshot: Any) -> None:
    """Restore the declared-state snapshot; on failure, mark both traces and raise.

    L4 5.4 restore-failure policy: the transactional FORK is poisoned (its run
    genuinely failed) and unregistered; the SOURCE Trace gets the session-scoped
    STATE-COMPROMISED latch (deliberately NOT the poison bit -- the live MODEL's
    state is unknown, not the trace's recorded path facts), which refuses the
    live and fast run doors typed while loaded-sparse runs of a saved artifact
    stay legal. The typed exception carries the failed slot name, the count of
    alias groups restored before the failure, and chains the restore exception.
    """

    from ._runnable_state import LiveStateRestoreFailure, restore_live_declared_state
    from .errors import StateBindingError
    from .runnable import mark_trace_path_status

    try:
        restore_live_declared_state(snapshot)
    except LiveStateRestoreFailure as exc:
        mark_trace_path_status(fork, PathFaithfulness.UNVERIFIABLE, None)
        _state.unregister_log(fork)
        runnable_state = getattr(trace, "_runnable", None)
        if runnable_state is not None:
            runnable_state.state_compromised = {
                "state_dict_name": exc.state_dict_name,
                "groups_restored": exc.groups_restored,
            }
        raise StateBindingError(
            "Restoring the live model's declared state failed AFTER execution: "
            f"restore stopped at state entry {exc.state_dict_name!r} with "
            f"{exc.groups_restored} alias group(s) already restored, so the live "
            "model's declared state is now UNKNOWN. Later live/fast run() calls on "
            "this trace refuse until the state is re-established. Remedy: reload "
            "known-good weights onto the model (or re-capture), then run again",
            code="run_state_restore_failed",
            detection_stage="state_restore",
            state_dict_name=exc.state_dict_name,
            groups_restored=exc.groups_restored,
        ) from exc


def _live_runtime_input_leaves(input_args: Any, input_kwargs: Any) -> list[torch.Tensor] | None:
    """Flatten runtime live-refresh inputs to ordered tensor leaves, or ``None``.

    Mirrors the capture-side flatten (``backend.fetch_label_move_input_tensors``:
    ``get_vars_of_type_from_obj`` per positional arg, then per kwarg value, at
    cycle-safe finite input walk) so leaf ORDER pairs 1:1 with the capture's recorded
    ``input_layers`` ordering. Returns ``None`` on any traversal failure -- the
    caller then skips classification entirely (never masks the native error).
    """

    from .utils.introspection import INPUT_SEARCH_DEPTH_LIMIT, get_vars_of_type_from_obj

    try:
        if isinstance(input_args, (list, tuple)):
            args_list = list(input_args)
        else:
            args_list = [input_args]
        leaves: list[torch.Tensor] = []
        for arg in args_list:
            unresolved: list[str] = []
            leaves.extend(
                get_vars_of_type_from_obj(
                    arg,
                    torch.Tensor,
                    search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                    depth_exceeded_paths=unresolved,
                )
            )
            if unresolved:
                return None
        if input_kwargs:
            for value in input_kwargs.values():
                unresolved = []
                leaves.extend(
                    get_vars_of_type_from_obj(
                        value,
                        torch.Tensor,
                        search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                        depth_exceeded_paths=unresolved,
                    )
                )
                if unresolved:
                    return None
        return leaves
    except Exception:
        return None
