"""Path-faithfulness and state comparison helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from ._runnable_state import (
    PreparedRunnableState,
    runnable_tensor_byte_digest,
)
from .runnable import (
    NONDETERMINISTIC_SOURCE_VOCABULARY,
    RUN_TRUNCATION_CAUSE_VOCABULARY,
    RUN_TRUNCATION_REGIME_VOCABULARY,
    ActivationPayloadLayerDescriptor,
    ActivationPayloadMember,
    ContractCheck,
    InputAttestationFingerprint,
    NumericAttestationStatus,
    PathFaithfulness,
    ReadinessReport,
    RunnableDiagnostic,
    RunnableErrorCode,
    RunReport,
    RunTruncation,
    SparseRunDescriptor,
    StateSource,
    WitnessCompleteness,
    derived_witness_completeness,
)

if TYPE_CHECKING:
    from ._runnable_execution import (
        _LAYOUT_SENSITIVE_BLAS_QUALNAMES,
        _attestation_inputs_match,
        _attestation_state_matches,
        _contract_check,
        _descriptor_has_nondeterministic_rng,
        _has_journaled_buffer_activation_member,
        _has_out_mutated_activation_member,
        _is_benign_downstream_nonreproducible,
        _lacks_recorded_original_input_eligibility,
    )

__all__ = (
    "_path_faithfulness",
    "_run_report",
    "_numeric_attestation_check",
    "_member_producer_is_layout_sensitive_blas",
    "_within_layout_reduction_tolerance",
    "_is_benign_layout_nonreproducible",
)


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
    nondeterministic_control_source: bool = False,
    input_derived_layout_stale: bool = False,
    run_truncated: bool = False,
) -> tuple[PathFaithfulness, RunnableDiagnostic | None]:
    """Classify exact three-state path faithfulness after all honesty checks."""

    failed = next((check for check in checks if not check.passed), None)
    if failed is not None:
        return PathFaithfulness.DIVERGED, failed.diagnostic
    if run_truncated:
        # L4 2.4 positive-claim bar: a truncated (until=) run may never settle a
        # positive claim. The ceiling is an input to the verdict derivation
        # UPSTREAM of attestation (the r35-I3 extension point), so VERIFIED is
        # unreachable and attestation derives NOT_APPLICABLE via the existing
        # lattice rule before a single archive byte is read. Contradictions the
        # executed region DOES evidence still degrade normally: the failed-check
        # arm above settles DIVERGED first (the cap is a ceiling, never a floor).
        return PathFaithfulness.UNVERIFIABLE, None
    # r71 A3: the VERIFIED gate consults ONLY the parser-derived completeness FLOOR
    # (re-derived here from the typed gap ledger by the ONE derivation function).
    # The persisted summary is a redundant assertion checked equal at parse; reading
    # it for a verdict is forbidden (source-scan tripwire).
    if derived_witness_completeness(descriptor.coverage_gaps) is not WitnessCompleteness.COMPLETE:
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
    if nondeterministic_control_source:
        # r53 hon_2: a recorded control fact (scalar-bool, loop predicate, or a
        # tensor->host escape source) derives from surviving uninitialized-memory
        # taint -- the taken branch is a function of allocator garbage no seed
        # governs, so NO replay is reproducible. Exact parity with the seeded
        # RNG-driven branch ceiling: UNVERIFIABLE, never a false VERIFIED.
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
    if input_derived_layout_stale:
        # r73 F1: the capture read a layout predicate on an activation ROOTED at a
        # model input, and this run's input carries different strides than capture
        # (channels_last twin, transposed-then-copied layout, ...). Memory format
        # propagates through elementwise ops, so the recorded taken path may not be
        # the one a fresh model takes on this input -- but a different input layout
        # does not PROVE the derived read flipped (an intervening reshape can
        # canonicalize it), so the honest ceiling is UNVERIFIABLE, never DIVERGED
        # by assumption and never a false VERIFIED. Same-stride inputs never reach
        # here: honest channels_last-on-channels_last models stay VERIFIED.
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
    nondeterministic_sources: Iterable[str] = (),
    state_carried: bool = False,
    truncation: RunTruncation | None = None,
) -> RunReport:
    """Build the settled run-report surface -- the ONE report finalizer (r37 corr2-5).

    EVERY provider (loaded sparse AND live refresh) routes its report through this
    constructor: ``poisoned`` is DERIVED solely from ``path_faithfulness is not
    VERIFIED`` (no caller Boolean exists), and the r35 I3 tripwire-on-the-tripwire
    asserts ``attested`` structurally implies a verified, unpoisoned path, so no
    provider can emit an internally contradictory report. Direct ``RunReport(``
    construction outside this finalizer is forbidden (source-scan meta-test).
    ``nondeterministic_sources`` (r53 F4) is normalized (sorted, deduplicated)
    and closed-vocabulary-checked HERE and only here.
    """

    poisoned = path_faithfulness is not PathFaithfulness.VERIFIED
    if numeric_attestation is NumericAttestationStatus.ATTESTED and (
        path_faithfulness is not PathFaithfulness.VERIFIED or poisoned
    ):
        raise RuntimeError(
            "Internal invariant violation: numeric_attestation=attested requires "
            f"a verified, unpoisoned path (got {path_faithfulness.value!r})."
        )
    declared_sources = tuple(sorted(set(nondeterministic_sources)))
    unknown_sources = set(declared_sources) - NONDETERMINISTIC_SOURCE_VOCABULARY
    if unknown_sources:
        raise RuntimeError(
            "Internal invariant violation: nondeterministic sources outside the "
            f"closed vocabulary: {sorted(unknown_sources)!r}."
        )
    if truncation is not None:
        # L4 2.4 redundant tamper assert (tripwire, not the mechanism): the
        # run_truncated ceiling is threaded into the verdict derivation UPSTREAM
        # of attestation, so a truncated run reaching this spine with a positive
        # claim is an internal invariant violation, never a reachable state.
        if path_faithfulness is PathFaithfulness.VERIFIED:
            raise RuntimeError(
                "Internal invariant violation: a truncated run may never settle "
                "VERIFIED path_faithfulness (the run_truncated ceiling was bypassed)."
            )
        if numeric_attestation is NumericAttestationStatus.ATTESTED:
            raise RuntimeError(
                "Internal invariant violation: a truncated run may never settle "
                "ATTESTED numeric attestation."
            )
        if truncation.regime not in RUN_TRUNCATION_REGIME_VOCABULARY:
            raise RuntimeError(
                f"Internal invariant violation: truncation regime {truncation.regime!r} "
                "is outside the closed vocabulary."
            )
        if truncation.cause is not None and truncation.cause not in RUN_TRUNCATION_CAUSE_VOCABULARY:
            raise RuntimeError(
                f"Internal invariant violation: truncation cause {truncation.cause!r} "
                "is outside the closed vocabulary."
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
        is_poisoned=poisoned,
        nondeterministic_sources=declared_sources,
        has_carried_state=bool(state_carried),
        truncation=truncation,
        truncated=truncation is not None,
        stopped_at=None if truncation is None else truncation.stopped_at,
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
    if raw_members and _lacks_recorded_original_input_eligibility(descriptor, layer):
        # r6 H2 (disclosure half only -- the status stays NOT_APPLICABLE by design).
        # A ``save=`` selective capture that did not select the model input records ZERO
        # ``original_input_digests`` / ``input_fingerprints``, so attestation can never
        # apply no matter what the caller passes to ``.run()``. Left unnamed that is
        # INDISTINGUISHABLE from "you changed the input". Surface the real reason -- a
        # SAVE-time eligibility gap -- as a named, non-verdict-changing report entry. The
        # save side emits the matching one-time disclosure warning.
        return NumericAttestationStatus.NOT_APPLICABLE, ContractCheck(
            name="numeric_attestation:not_applicable:no_recorded_original_input_eligibility",
            passed=True,
            diagnostic=None,
        )
    if not raw_members or not _attestation_inputs_match(
        descriptor, layer, input_byte_digests, input_fingerprints
    ):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    if not _attestation_state_matches(descriptor, layer, state, state_byte_digests, trace):
        return NumericAttestationStatus.NOT_APPLICABLE, None
    archived = trace._runnable.archived_activations
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
    # 64 ULP relative + absolute floor. Observed basis: reduction-order noise
    # on eval MHA measures ~5e-7 for float32 == ~4 ULP, so 64 gives 16x
    # headroom over the worst observation while a real corruption (sign flip,
    # stale buffer, zeroed value) stays thousands of ULPs outside the band.
    # The former 256 (64x headroom) was asserted from that single observation
    # with no derivation and classified 4x more divergence as "benign"; this
    # band gates a not_applicable-instead-of-raise downgrade, so looseness
    # here silently launders corruption. Scaled by dtype eps so fp16/bf16
    # keep a proportional band and fp64 a far tighter one.
    tolerance = 64.0 * eps * (archived64.abs() + 1.0)
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
