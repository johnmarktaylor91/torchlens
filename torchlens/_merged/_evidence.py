"""Rank-core evidence extraction for the merge engine.

One extraction path serves live traces, loaded traces, and rank-core bundle
paths, so merge time and load rederivation consume IDENTICAL evidence
(design-merge-ranks-c v5, 4.3: one derivation function called twice). The
extractor validates the ``collective_boundary_v1`` payloads against their
closed vocabularies at parse time; a malformed core refuses typed
(``merged_schema_invalid``) instead of degrading silently.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._data_classes._trace_missing_attrs import is_live_trace
from .._distributed._ledger import GroupLifecycleLedger

# The boundary vocabularies and parse rules live in ``_boundary_rules`` (the
# group-membership and role rules in ``_boundary_membership``); the
# reader vocabularies are re-exported here (redundant aliases) beside the
# extractor that applies them, where the writer-lockstep tests read them.
from ._boundary_membership import check_group_coherence, check_group_membership, check_roles
from ._boundary_rules import (
    _COMPLETION_BINDINGS as _COMPLETION_BINDINGS,
    _DISCLOSURE_TOKENS as _DISCLOSURE_TOKENS,
    _INSTALL_EPOCHS,
    _NOT_PRESENT_REASONS as _NOT_PRESENT_REASONS,
    _WITNESS_POLICIES as _WITNESS_POLICIES,
    BOUNDARY_SCHEMA,
    KNOWN_KINDS,
    P2P_KINDS,
    _refuse,
    check_boundary_identity,
    check_completion_events,
    check_correlation_key,
    check_disclosures,
    check_kind_class_records,
    check_lifetime_evidence,
    check_op_label_backrefs,
    check_scalar_records,
    check_witness_digests,
    check_witness_policy,
)
from ._enums import REDUCE_OP_KINDS, TENSORLESS_KINDS, MergedErrorCode
from ._errors import MergeInputError
from ._rules import (
    SEQ_READ_FAILED_TOKEN,
    CorrelationKey,
    GroupRankPresence,
    GroupSeqLatch,
    correlation_key,
    group_uid,
)

__all__ = [
    "BOUNDARY_SCHEMA",
    "KNOWN_KINDS",
    "P2P_KINDS",
    "REDUCE_OP_KINDS",
    "TENSORLESS_KINDS",
    "RankEvidence",
    "extract_rank_evidence",
    "resolve_rank_inputs",
]


@dataclass(frozen=True)
class RankEvidence:
    """Everything the merge engine reads from one rank core.

    Parameters
    ----------
    rank:
        The core's global rank, proven by its own boundary records.
    boundaries:
        The ordered boundary entries from the trace-level distributed
        journal (issue order; each carries the ``collective_boundary_v1``
        payload plus ``op_labels_raw`` back-references).
    ledger:
        The rank's group-lifecycle ledger, rebuilt from its portable payload.
    install_epoch:
        The rank's arming install-epoch record.
    source:
        Where the evidence came from (diagnostic): ``"live"``, ``"loaded"``,
        or the bundle path string.
    shard_local:
        Whether the member trace carries the shard-local capture marker
        (``distributed_scope == "rank_local_shard"``; L8 census plan 3.2c(2)).
        REQUIRED with no default -- every constructor, direct construction
        included, must consciously supply it, so the ``_guard_scope`` marker
        key can never be silently omitted. Populated here in
        :func:`extract_rank_evidence`, the single common ancestor of all
        ``derive_merge`` entry paths (public input resolution, save-time
        reverify, load rederivation).
    """

    rank: int
    boundaries: tuple[dict[str, Any], ...]
    ledger: GroupLifecycleLedger
    install_epoch: str
    source: str
    shard_local: bool


def _validate_boundary(entry: dict[str, Any], index: int, source: str) -> None:
    """Validate one journal entry against the closed C0 vocabularies.

    Runs the rule groups of ``_boundary_rules`` in their contract order; when
    an entry carries several defects, the earliest group's refusal wins.

    Parameters
    ----------
    entry:
        One boundary journal entry (already known to be a mapping).
    index:
        The entry's position in the rank's journal (diagnostic).
    source:
        Merge input origin carried on every refusal payload.

    Raises
    ------
    MergeInputError
        ``merged_schema_invalid`` for the first defect found.
    """

    where = f"boundary {index} of {source}"
    kind = check_boundary_identity(entry, where, source)
    correlation = check_correlation_key(entry, where, source)
    global_ranks = check_group_membership(entry, where, source)
    check_group_coherence(entry["group"], global_ranks, correlation, where, source)
    check_roles(entry, kind, where, source)
    c10d_group_seq = check_scalar_records(entry, kind, where, source)
    # Event/disclosure/witness COHERENCE (R18 fixwave-6): every field checked
    # from here on is redundant with the others on an honest record, and each
    # redundancy is a forgery axis when left unchecked. A tampered core that
    # flips ONE of them (forged destination digests on an async boundary,
    # digests under witness policy "none", a stripped in-flight-read
    # disclosure) used to sail through parse and could only be caught -- or
    # worse, ATTESTED -- by the cross-rank derivation. The only forgery that
    # survives these checks is a fully coherent reauthoring of every field on
    # every core, which is the documented out-of-scope boundary (contract
    # section 11 analog), not an open residual.
    completion_binding = check_completion_events(entry, where, source)
    check_disclosures(entry, completion_binding, c10d_group_seq, where, source)
    check_kind_class_records(entry, kind, where, source)
    witness = check_witness_policy(entry, completion_binding, where, source)
    check_witness_digests(witness, completion_binding, kind, where, source)
    check_op_label_backrefs(entry, kind, where, source)
    check_lifetime_evidence(entry, where, source)


def _distributed_record(trace: Any, source: str) -> dict[str, Any]:
    """Return the trace's distributed record, refusing a trace that carries none.

    Raises
    ------
    MergeInputError
        ``merge_input_invalid`` when ``trace.annotations['distributed']`` is
        absent, not a mapping, or has no boundaries.
    """

    record = getattr(trace, "annotations", {}).get("distributed")
    if not isinstance(record, dict) or not record.get("boundaries"):
        raise MergeInputError(
            f"Merge input {source} carries no distributed boundary evidence "
            "(trace.annotations['distributed'] is absent or empty). Only rank "
            "captures taken under the distributed opt-in "
            "(torchlens.distributed.arm() or SPMD lazy arming) can be merged.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="not_a_rank_capture",
            remedy="capture each rank under torchlens.distributed.arm() and merge those",
            source=source,
        )
    return record


def _refuse_funcol_boundary(entry: dict[str, Any], index: int, source: str) -> None:
    """Refuse a core carrying a functional-collective (funcol) boundary.

    Merge-ranks C2 recording (fail-closed): funcol boundaries carry the
    documented-unstable v0 payload, which the frozen C1 derivation cannot
    join. Refusing the CORE typed is strictly honest; the shipped alternative
    was merging with the funcol traffic invisibly absent. The merged-side
    funcol join is C2 merged-side work behind its own ruling (L8 plan 3.2c).

    Raises
    ------
    MergeInputError
        ``merge_scope_unsupported`` for a ``functional_collective_boundary_v0`` entry.
    """

    if entry.get("schema") != "functional_collective_boundary_v0":
        return
    raise MergeInputError(
        f"Merge input {source} records a functional-collective "
        f"(funcol) boundary (index {index}, kind "
        f"{entry.get('kind')!r}). C1 joins the frozen "
        "collective_boundary_v1 payload only; funcol boundary joining "
        "is rung-C2 merged-side scope and stays refused until its "
        "capture-fidelity census and authorizing ruling land.",
        code=MergedErrorCode.MERGE_SCOPE_UNSUPPORTED,
        reason="functional_collective_boundary_unsupported",
        source=source,
        boundary_index=index,
    )


class _CoreCoherence:
    """The per-core coherence rules every validated boundary must keep with its predecessors.

    Each rule reads a group-level or core-level fact that the recorder writes
    uniformly; a boundary that contradicts an earlier one of the same core is
    tamper, never honest evidence. Applied in order after ``_validate_boundary``:
    group-rank presence (identity), the group-seq latch (provenance), the
    install epoch (provenance), then the rank-local correlation key (identity).
    """

    def __init__(self, install_epoch: Any, source: str) -> None:
        self._install_epoch = install_epoch
        self._source = source
        self._group_rank_presence = GroupRankPresence()
        self._seq_latch = GroupSeqLatch()
        self._seen_keys: set[CorrelationKey] = set()

    def check(self, entry: dict[str, Any], index: int) -> None:
        """Apply every per-core rule to one validated boundary.

        Raises
        ------
        MergeInputError
            ``merged_schema_invalid`` for the first rule the boundary breaks.
        """

        self._check_group_rank_presence(entry, index)
        self._check_group_seq_latch(entry, index)
        self._check_install_epoch(entry, index)
        self._check_unique_correlation_key(entry, index)

    def _check_group_rank_presence(self, entry: dict[str, Any], index: int) -> None:
        """Refuse a group whose boundaries mix ``my_group_rank`` presence and absence.

        Presence is uniform per group within one rank core. Without this,
        stripping ``my_group_rank`` from one boundary of a slice-witnessed join
        silently deleted its value_divergence finding (R18-2; findings must
        never disappear by record deletion).
        """

        uid = group_uid(entry["correlation"])
        is_absent = entry["group"].get("my_group_rank") is None
        if self._group_rank_presence.mixed(uid, is_absent):
            raise _refuse(
                f"boundary {index} of {self._source} mixes my_group_rank presence and "
                f"absence within one group {uid}; group-rank readability is a "
                "group-level fact, so selective absence is tamper",
                source=self._source,
            )

    def _check_group_seq_latch(self, entry: dict[str, Any], index: int) -> None:
        """Refuse a ``c10d_group_seq`` sequence that breaks the recorder's latch shape."""

        disclosed = SEQ_READ_FAILED_TOKEN in entry["disclosures"]
        defect = self._seq_latch.observe(entry["c10d_group_seq"], disclosed)
        if defect is None:
            return
        where = f"boundary {index} of {self._source}"
        detail = {
            GroupSeqLatch.VALUE_AFTER_DROP: (
                f"{where} carries a c10d_group_seq value after an earlier boundary "
                "of this core recorded none; the probe never recovers within one capture"
            ),
            GroupSeqLatch.UNDISCLOSED_DROP: (
                f"{where} drops c10d_group_seq without the c10d_group_seq_read_failed "
                "disclosure while earlier boundaries of this core carry values; an "
                "undisclosed drop is tamper, not honest probe absence"
            ),
            GroupSeqLatch.REPEATED_DISCLOSURE: (
                f"{where} repeats the c10d_group_seq_read_failed disclosure after the "
                "probe already latched off; the recorder discloses the failed read "
                "exactly once"
            ),
        }[defect]
        raise _refuse(detail, source=self._source)

    def _check_install_epoch(self, entry: dict[str, Any], index: int) -> None:
        """Refuse a boundary whose install epoch differs from the rank record's.

        A rank has exactly ONE install epoch; a boundary claiming a different
        one is a forged record trying to promote (or demote) its own lifetime
        completeness independently of the rank's arming record.
        """

        claimed = entry["lifetime_evidence"]["install_epoch"]
        if claimed != self._install_epoch:
            raise _refuse(
                f"boundary {index} of {self._source} claims install_epoch "
                f"{claimed!r} but the rank record's install_epoch is "
                f"{self._install_epoch!r}",
                source=self._source,
            )

    def _check_unique_correlation_key(self, entry: dict[str, Any], index: int) -> None:
        """Refuse a correlation key already seen in this rank core."""

        key = correlation_key(entry["correlation"])
        if key in self._seen_keys:
            raise _refuse(
                f"boundary {index} of {self._source} duplicates rank-local correlation key {key}",
                source=self._source,
            )
        self._seen_keys.add(key)


def _walk_boundaries(boundaries: Sequence[Any], install_epoch: Any, source: str) -> set[int]:
    """Validate every boundary of one core in journal order; return the ranks it claims.

    Raises
    ------
    MergeInputError
        For the first malformed, out-of-scope or incoherent boundary.
    """

    ranks: set[int] = set()
    coherence = _CoreCoherence(install_epoch, source)
    for index, entry in enumerate(boundaries):
        if not isinstance(entry, dict):
            raise _refuse(f"boundary {index} of {source} is not a mapping", source=source)
        _refuse_funcol_boundary(entry, index, source)
        _validate_boundary(entry, index, source)
        coherence.check(entry, index)
        ranks.add(int(entry["group"]["my_global_rank"]))
    return ranks


def _single_rank(ranks: set[int], source: str) -> int:
    """Return the one rank a core claims.

    Raises
    ------
    MergeInputError
        ``merge_input_invalid`` when the core's boundaries name several ranks.
    """

    if len(ranks) != 1:
        raise MergeInputError(
            f"Merge input {source} claims multiple global ranks {sorted(ranks)}; "
            "a rank core is a single-rank capture.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="multiple_ranks_in_one_core",
            remedy="re-capture the rank; one core must come from exactly one rank",
            source=source,
        )
    return next(iter(ranks))


def _parse_ledger(record: dict[str, Any], install_epoch: Any, source: str) -> GroupLifecycleLedger:
    """Rebuild the core's group-lifecycle ledger and tie its events to the rank's epoch.

    Raises
    ------
    MergeInputError
        ``merged_schema_invalid`` when the ledger payload is absent, empty or
        unparseable, or its events' install epochs disagree with the record's.
    """

    ledger_payload = record.get("group_lifecycle_ledger")
    if not isinstance(ledger_payload, list) or not ledger_payload:
        raise _refuse("group_lifecycle_ledger is absent or empty", source=source)
    try:
        ledger = GroupLifecycleLedger.from_payload(ledger_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise _refuse(f"group_lifecycle_ledger does not parse ({exc})", source=source) from exc
    # ``lineage_vectors()`` stamps each vector's install epoch from the EVENTS, while
    # the audit's completeness reasoning also reads the record-level epoch. A rank
    # has exactly one epoch, so a disagreement is a forged/corrupt sidecar trying to
    # promote a ``seeded`` rank to a complete witness; refuse rather than let the two
    # readings diverge.
    event_epochs = {event.install_epoch for event in ledger.events}
    if event_epochs != {install_epoch}:
        raise _refuse(
            f"group_lifecycle_ledger event install_epochs {sorted(event_epochs)} disagree "
            f"with the record install_epoch {install_epoch!r}",
            source=source,
        )
    return ledger


def _shard_local_scope(trace: Any) -> bool:
    """Return whether a live trace declares the rank-local shard distributed scope."""

    scope = trace._provenance_distributed_scope if is_live_trace(trace) else None
    return bool(scope == "rank_local_shard")


def extract_rank_evidence(trace: Any, source: str) -> RankEvidence:
    """Extract and validate one rank core's merge evidence.

    Parameters
    ----------
    trace:
        A finished (live or loaded) ``Trace`` captured under the distributed
        opt-in.
    source:
        Diagnostic origin string recorded on the evidence.

    Returns
    -------
    RankEvidence
        Validated evidence for the engine.

    Raises
    ------
    MergeInputError
        If the trace carries no distributed record, its journal is malformed,
        or its boundary records disagree about the rank's own identity.
    """

    record = _distributed_record(trace, source)
    boundaries = record["boundaries"]
    if not isinstance(boundaries, (list, tuple)):
        # W051-CAPT3 (IO remainder): the per-rank journal the merge JOINS is
        # validated here at merge entry, live and loaded cores alike. A
        # non-sequence ``boundaries`` (an int, a bool, a mapping, a string)
        # previously escaped as a bare TypeError from the row walk or was
        # mis-read row-by-row; it refuses typed with the container named.
        raise _refuse(
            f"boundaries of {source} is not a list of boundary records "
            f"(got {type(boundaries).__name__})",
            code=MergedErrorCode.MERGED_SCHEMA_INVALID,
            source=source,
        )
    install_epoch = record.get("install_epoch")
    if install_epoch not in _INSTALL_EPOCHS:
        raise _refuse(
            f"install_epoch {install_epoch!r} outside the closed vocabulary",
            source=source,
        )
    rank = _single_rank(_walk_boundaries(boundaries, install_epoch, source), source)
    ledger = _parse_ledger(record, install_epoch, source)
    return RankEvidence(
        rank=rank,
        boundaries=tuple(boundaries),
        ledger=ledger,
        install_epoch=str(install_epoch),
        source=source,
        # The marker key of _guard_scope (L8 3.2c(2)): the geometry key alone
        # does not fire on TP/FSDP2 boundaries (funcol/c10d traffic on
        # to_local()-ed plain tensors carries no role geometry), so the
        # trace-level marker travels on the evidence itself. Hostile inputs:
        # anything other than the exact marker string reads False -- absence
        # of the marker never blocks, presence of ANY other value never
        # blocks; only the one documented value engages the scope guard.
        shard_local=_shard_local_scope(trace),
    )


def _require_mergeable_outcome(trace: Any, source: str) -> None:
    """Refuse typed when a member core's settled capture outcome cannot merge.

    R06c: ``merge_ranks`` never consulted member capture outcomes, so a FAILED
    or UNKNOWN member core merged into ``aligned``/``attested_complete`` with
    zero findings. FAILED / ABORTED_NONFINITE / UNKNOWN members refuse here at
    the one input chokepoint (live traces and path-loaded bundles alike);
    HALTED and legacy UNATTESTED members merge and are DISCLOSED through
    ``MergedTrace.member_outcomes`` and ``summary()`` (demote-only: the
    disclosure is presenter-side and never edits the derivation). A trace
    object carrying no settled outcome sidecar (hand-built evidence carriers)
    makes no outcome claim and is not refused here -- the boundary-journal
    parse remains its gate.
    """

    from .._capture.outcome import CaptureStatus, outcome_for

    outcome = outcome_for(trace)
    if outcome is None:
        return
    if outcome.status in (
        CaptureStatus.FAILED,
        CaptureStatus.ABORTED_NONFINITE,
        CaptureStatus.UNKNOWN,
    ):
        raise MergeInputError(
            f"Merge input {source} carries a settled capture outcome "
            f"{outcome.status.value!r}; a failed, aborted, or unprovable rank "
            "capture cannot join a cross-rank merge.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="member_outcome_not_mergeable",
            member_status=outcome.status.value,
            remedy=(
                "re-capture the rank to a settled complete (or halted) outcome and merge that core"
            ),
            source=source,
        )


def resolve_rank_inputs(inputs: Sequence[Any]) -> dict[int, tuple[RankEvidence, Any]]:
    """Resolve merge inputs (traces or bundle paths) into per-rank evidence.

    Parameters
    ----------
    inputs:
        Live/loaded ``Trace`` objects or ``.tlspec`` rank-core paths, in any
        mix and order. Paths are loaded for analysis.

    Returns
    -------
    dict[int, tuple[RankEvidence, Any]]
        Mapping from global rank to ``(evidence, trace)``, ordered by rank.

    Raises
    ------
    MergeInputError
        On empty input, unloadable paths, duplicate ranks, or invalid cores.
    """

    if not inputs:
        raise MergeInputError(
            "merge_ranks requires at least one rank capture or rank-core path.",
            code=MergedErrorCode.MERGE_INPUT_INVALID,
            reason="empty_inputs",
            remedy="pass at least one rank capture or rank-core path",
        )
    resolved: dict[int, tuple[RankEvidence, Any]] = {}
    for position, item in enumerate(inputs):
        if isinstance(item, (str, Path)):
            from .._io.bundle import load as load_bundle

            source = str(item)
            try:
                trace = load_bundle(source)
            except Exception as exc:
                raise MergeInputError(
                    f"Merge input {source} failed to load as a rank core: {exc}",
                    code=MergedErrorCode.MERGE_INPUT_INVALID,
                    reason="member_load_failed",
                    remedy="inspect the chained cause; pass a loadable rank-core bundle",
                    source=source,
                ) from exc
        else:
            trace = item
            source = f"live[{position}]"
        evidence = extract_rank_evidence(trace, source)
        _require_mergeable_outcome(trace, source)
        if evidence.rank in resolved:
            raise MergeInputError(
                f"Merge inputs contain global rank {evidence.rank} twice "
                f"({resolved[evidence.rank][0].source} and {source}); every rank "
                "core must come from a distinct rank of one run.",
                code=MergedErrorCode.MERGE_INPUT_INVALID,
                reason="duplicate_rank",
                remedy="pass one core per rank; drop the duplicate input",
            )
        resolved[evidence.rank] = (evidence, trace)
    return dict(sorted(resolved.items()))
