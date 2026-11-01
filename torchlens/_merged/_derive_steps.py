"""Evidence stages of the merge derivation that need no join records.

:func:`torchlens._merged._engine.derive_merge` stays the ONE derivation entry
point; it calls these steps in its normative order. Each step reads the
per-rank evidence (or a slice of one join's per-rank facts) and either
refuses typed or returns findings. None of them builds a join record, so this
module never imports the engine and the import graph stays acyclic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from ..distributed._audit import MembershipLineageVerdict, verify_membership_lineages
from ..distributed._ledger import InstallEpoch
from ._boundary_rules import P2P_KINDS
from ._enums import BoundaryConsistency, MergeAlignment, MergedErrorCode, MergeValueStatus
from ._errors import MergedFinding, MergeInputError
from ._rules import (
    COLLECTIVE_CHANNEL,
    ChannelKey,
    CorrelationKey,
    GroupUid,
    armed_before_any_group,
    channel_key,
    correlation_key,
    group_uid,
    membership_digest_bound,
    ranks_outside_membership,
    recorded_members,
)

if TYPE_CHECKING:
    from ._evidence import RankEvidence

ChannelRecords = dict[int, list[tuple[int, int]]]
"""Per-rank ``(seq_abs, boundary_index)`` records on one channel."""

DeltaTable = dict[int, dict[int, tuple[int, int]]]
"""``seq_delta -> rank -> (seq_abs, boundary_index)`` for one channel."""

_STRUCTURAL_KINDS = frozenset(
    {
        "group_lifetime_evidence_conflict",
        "relation_violation",
        "order_contradiction",
        "correlation_delta_mismatch",
    }
)


def _guard_scope(evidence: Mapping[int, RankEvidence]) -> None:
    """Refuse out-of-C1-scope boundaries typed (p2p/pipeline: C3; DTensor: C2)."""

    for rank in sorted(evidence):
        if evidence[rank].shard_local:
            # L8 3.2c(2), the SECOND scope key (marker-keyed; refusal-widening
            # beside the geometry key below, which STAYS): a member whose
            # trace carries the shard-local capture marker refuses regardless
            # of boundary geometry. Runs inside derive_merge, so directly
            # constructed RankEvidence refuses too (the membership-authority
            # precedent). Narrowing this refusal is D-L8-MRG, never this code.
            raise MergeInputError(
                f"Rank {rank} is a shard-local capture (distributed_scope == "
                "'rank_local_shard'). Sharded-topology merging stays refused "
                "pending its own authorizing ruling plus the merged-side "
                "tiling-proof deliverable; the geometry-independent marker "
                "key refuses it at merge scope.",
                code=MergedErrorCode.MERGE_SCOPE_UNSUPPORTED,
                rank=rank,
                reason="shard_local_member_unsupported",
            )
        for index, entry in enumerate(evidence[rank].boundaries):
            kind = entry["kind"]
            channel = entry["correlation"]["channel"]
            if kind in P2P_KINDS or channel != COLLECTIVE_CHANNEL:
                raise MergeInputError(
                    f"Rank {rank} boundary {index} is a point-to-point boundary "
                    f"(kind={kind!r}, channel={channel!r}). C1 merges symmetric "
                    "collectives only; p2p/pipeline pairing is rung C3.",
                    code=MergedErrorCode.MERGE_SCOPE_UNSUPPORTED,
                    rank=rank,
                    kind=kind,
                    channel=channel,
                )
            for role in entry.get("roles", ()):
                if role.get("logical_shape") is not None or role.get("placements") is not None:
                    raise MergeInputError(
                        f"Rank {rank} boundary {index} carries DTensor dual "
                        "geometry; sharded-topology merging is rung C2 and stays "
                        "refused until its capture-fidelity census is green.",
                        code=MergedErrorCode.MERGE_SCOPE_UNSUPPORTED,
                        rank=rank,
                        kind=kind,
                    )


def check_membership_authority(
    evidence: Mapping[int, RankEvidence], input_ranks: tuple[int, ...]
) -> None:
    """Refuse boundaries whose presence or digest contradicts their membership.

    Evidence extraction enforces this for public merge inputs. The check is
    repeated here because load rederivation and internal callers can
    construct ``RankEvidence`` directly: presence outside membership must
    never participate in a join or complete a witness verdict.

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.
    input_ranks:
        The presented ranks, sorted.

    Raises
    ------
    MergeInputError
        ``merged_schema_invalid`` when a rank presents a boundary for a
        membership it is not part of, when a boundary's membership digest is
        not the digest of its own recorded membership, or when a rank repeats
        a correlation key.
    """

    for rank in input_ranks:
        seen_correlation_keys: set[CorrelationKey] = set()
        for entry in evidence[rank].boundaries:
            members = recorded_members(entry)
            if ranks_outside_membership((rank,), members):
                raise MergeInputError(
                    f"Rank {rank} presents a collective boundary for membership {members}, "
                    "but presence must be a subset of the recorded membership.",
                    code=MergedErrorCode.MERGED_SCHEMA_INVALID,
                    rank=rank,
                )
            correlation = entry["correlation"]
            if not membership_digest_bound(correlation, members):
                raise MergeInputError(
                    f"Rank {rank} presents a boundary whose membership_digest does "
                    f"not equal the digest of its own recorded membership {members}.",
                    code=MergedErrorCode.MERGED_SCHEMA_INVALID,
                    rank=rank,
                )
            key = correlation_key(correlation)
            if key in seen_correlation_keys:
                raise MergeInputError(
                    f"Rank {rank} presents duplicate correlation key {key}; "
                    "rank-local issue sequence values must be unique.",
                    code=MergedErrorCode.MERGED_SCHEMA_INVALID,
                    rank=rank,
                )
            seen_correlation_keys.add(key)


def audit_membership_lineages(
    evidence: Mapping[int, RankEvidence],
) -> tuple[dict[str, MembershipLineageVerdict], tuple[str, ...], list[MergedFinding]]:
    """Run the pre-join membership-lineage audit.

    It runs BEFORE any joining or presence-gap derivation: a conflicted
    membership refuses structurally, so none of its uids join and none of
    its unmatched keys become presence gaps.

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.

    Returns
    -------
    tuple[dict[str, MembershipLineageVerdict], tuple[str, ...], list[MergedFinding]]
        The per-digest audit verdicts, the sorted conflicted membership
        digests, and one ``group_lifetime_evidence_conflict`` finding per
        conflicted digest.
    """

    epochs = cast(
        "Mapping[int, InstallEpoch]",
        {rank: ev.install_epoch for rank, ev in evidence.items()},
    )
    audit = verify_membership_lineages({rank: ev.ledger for rank, ev in evidence.items()}, epochs)
    conflicted_memberships = tuple(
        sorted(digest for digest, verdict in audit.items() if verdict.is_conflict)
    )
    findings: list[MergedFinding] = []
    for digest in conflicted_memberships:
        verdict = audit[digest]
        findings.append(
            MergedFinding(
                kind="group_lifetime_evidence_conflict",
                detail=verdict.detail,
                membership_digest=digest,
                ranks=verdict.presenting_ranks,
            )
        )
    return audit, conflicted_memberships, findings


def _member_order_finding(
    rank: int,
    uid: GroupUid,
    members: tuple[int, ...],
    recorded: tuple[int, ...],
    input_ranks: tuple[int, ...],
) -> MergedFinding:
    """Build the ``relation_violation`` for a rank whose member order disagrees.

    Parameters
    ----------
    rank:
        The rank whose record disagrees.
    uid:
        The group uid both records name.
    members:
        This rank's recorded member order.
    recorded:
        The member order another rank recorded first.
    input_ranks:
        The presented ranks, sorted.

    Returns
    -------
    MergedFinding
        The member-order dispute finding.
    """

    return MergedFinding(
        kind="relation_violation",
        detail=(
            f"rank {rank} records group {uid} with member order "
            f"{members} but another rank recorded "
            f"{recorded}; c10d group rank lists must be "
            "identical on every member rank."
        ),
        membership_digest=uid[0],
        ranks=input_ranks,
    )


def _backend_dispute_finding(
    rank: int,
    uid: GroupUid,
    backend: str | None,
    recorded: str | None,
    input_ranks: tuple[int, ...],
) -> MergedFinding:
    """Build the ``relation_violation`` for a rank whose group backend disagrees.

    Parameters
    ----------
    rank:
        The rank whose record disagrees.
    uid:
        The group uid both records name.
    backend:
        This rank's recorded backend.
    recorded:
        The backend another rank recorded first.
    input_ranks:
        The presented ranks, sorted.

    Returns
    -------
    MergedFinding
        The backend dispute finding.
    """

    return MergedFinding(
        kind="relation_violation",
        detail=(
            f"rank {rank} records group {uid} with backend "
            f"{backend!r} but another rank recorded "
            f"{recorded!r}; a disputed backend is "
            "never verdict-grade."
        ),
        membership_digest=uid[0],
        ranks=input_ranks,
    )


def build_group_table(
    evidence: Mapping[int, RankEvidence], input_ranks: tuple[int, ...]
) -> tuple[dict[GroupUid, tuple[int, ...]], dict[GroupUid, str | None], list[MergedFinding]]:
    """Build the group table from the recorded memberships.

    The rank cores are the authority: the first recorded member order and
    backend per group uid win, and any later disagreement is a
    ``relation_violation``. A disputed backend demotes to ``None`` so every
    witness verdict on the group is not applicable (demote-only).

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.
    input_ranks:
        The presented ranks, sorted.

    Returns
    -------
    tuple[dict[GroupUid, tuple[int, ...]], dict[GroupUid, str | None], list[MergedFinding]]
        Members per group uid, backend per group uid, and the member-order and
        backend dispute findings in discovery order.
    """

    group_members: dict[GroupUid, tuple[int, ...]] = {}
    group_backend: dict[GroupUid, str | None] = {}
    backend_disputed: set[GroupUid] = set()
    findings: list[MergedFinding] = []
    for rank in input_ranks:
        for entry in evidence[rank].boundaries:
            uid = group_uid(entry["correlation"])
            members = recorded_members(entry)
            backend = entry["group"].get("backend")
            if uid in group_members and group_members[uid] != members:
                findings.append(
                    _member_order_finding(rank, uid, members, group_members[uid], input_ranks)
                )
            # Backend disagreement was silent first-writer-wins: one rank
            # claiming "gloo" flipped an unknown-backend group into the
            # WITNESS_VERDICT_BACKENDS set, rendering verdict-grade
            # attestations under semantics the other ranks never recorded.
            # Disagreement is a relation violation like the member list, and
            # the disputed backend demotes to None so every witness verdict
            # on the group is NOT_APPLICABLE (demote-only, P3).
            if (
                uid in group_backend
                and uid not in backend_disputed
                and group_backend[uid] != backend
            ):
                backend_disputed.add(uid)
                findings.append(
                    _backend_dispute_finding(rank, uid, backend, group_backend[uid], input_ranks)
                )
                group_backend[uid] = None
            group_members.setdefault(uid, members)
            group_backend.setdefault(uid, backend)
    return group_members, group_backend, findings


def collect_channel_records(
    evidence: Mapping[int, RankEvidence],
    input_ranks: tuple[int, ...],
    conflicted_memberships: tuple[str, ...],
) -> dict[ChannelKey, ChannelRecords]:
    """Group every joinable boundary by its correlation channel.

    Boundaries of a conflicted membership are skipped entirely: they neither
    join nor become presence gaps.

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.
    input_ranks:
        The presented ranks, sorted.
    conflicted_memberships:
        Membership digests the lineage audit refused.

    Returns
    -------
    dict[ChannelKey, ChannelRecords]
        ``(digest, ordinal, channel) -> rank -> [(seq_abs, boundary_index)]``
        in rank and journal order.
    """

    per_channel: dict[ChannelKey, ChannelRecords] = {}
    for rank in input_ranks:
        for index, entry in enumerate(evidence[rank].boundaries):
            correlation = entry["correlation"]
            channel = channel_key(correlation)
            if channel[0] in conflicted_memberships:
                continue  # no join AND no gap for a conflicted membership (1.3)
            per_channel.setdefault(channel, {}).setdefault(rank, []).append(
                (int(correlation["seq"]), index)
            )
    return per_channel


def align_seq_deltas(by_rank: Mapping[int, list[tuple[int, int]]]) -> DeltaTable:
    """Align one channel's records as seq DELTAS from each rank's first key.

    A seeded rank's absolute seq base is a rank-local fact and is never
    compared; only the offset from that rank's first recorded key aligns.

    Parameters
    ----------
    by_rank:
        One channel's per-rank ``(seq_abs, boundary_index)`` records.

    Returns
    -------
    DeltaTable
        ``seq_delta -> rank -> (seq_abs, boundary_index)``.
    """

    deltas: DeltaTable = {}
    for rank, records in by_rank.items():
        ordered = sorted(records)
        base = ordered[0][0]
        for seq_abs, index in ordered:
            deltas.setdefault(seq_abs - base, {})[rank] = (seq_abs, index)
    return deltas


def c10d_delta_mismatch(
    key: tuple[str, int, str, int],
    presence: tuple[int, ...],
    group_seqs: Mapping[int, int | None],
    group_seq_base: dict[int, int],
) -> MergedFinding | None:
    """Cross-check one join against the backend's own group-seq counter.

    Parameters
    ----------
    key:
        The join key ``(digest, ordinal, channel, seq_delta)``.
    presence:
        The ranks presenting this key, sorted.
    group_seqs:
        Each presenting rank's recorded backend group seq, or ``None``.
    group_seq_base:
        The channel's per-rank group-seq base, taken at each rank's first
        joined key and updated in place there; the caller keeps one table per
        channel.

    Returns
    -------
    MergedFinding | None
        A ``correlation_delta_mismatch`` finding, or ``None`` when the
        recorded group-seq deltas agree.
    """

    # Redundant c10d group-seq cross-check, as deltas from the first joined
    # key (1.6): a disagreement means c10d itself orders these issues
    # differently than the join claims.
    delta = key[3]
    for rank in presence:
        value = group_seqs[rank]
        if value is not None and rank not in group_seq_base and delta == 0:
            group_seq_base[rank] = value
    c10d_deltas: dict[int, int] = {}
    for rank in presence:
        observed = group_seqs[rank]
        if observed is not None and rank in group_seq_base:
            c10d_deltas[rank] = observed - group_seq_base[rank]
    if len(set(c10d_deltas.values())) <= 1:
        return None
    return MergedFinding(
        kind="correlation_delta_mismatch",
        detail=(
            f"c10d group-seq deltas disagree at key {key}: "
            f"{dict(sorted(c10d_deltas.items()))}; the backend's own "
            "sequence counter contradicts the join alignment."
        ),
        membership_digest=key[0],
        key=key,
        ranks=tuple(sorted(c10d_deltas)),
    )


def absolute_seq_mismatch(
    evidence: Mapping[int, RankEvidence],
    key: tuple[str, int, str, int],
    seq_abs_by_rank: Mapping[int, int],
) -> MergedFinding | None:
    """Cross-check one join's absolute seqs across ranks armed before any group.

    Delta alignment anchors at each rank's first RECORDED key; for ranks
    armed before any group the absolute counters also agree on every
    collective, so a disagreement proves the anchors paired different
    collectives. Seeded ranks stay out: their absolute bases are rank-local.

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.
    key:
        The join key ``(digest, ordinal, channel, seq_delta)``.
    seq_abs_by_rank:
        Each presenting rank's absolute seq at this key, in presence order.

    Returns
    -------
    MergedFinding | None
        A ``correlation_delta_mismatch`` finding, or ``None`` when every
        armed-before-any-group rank agrees.
    """

    # Absolute-seq cross-check for ranks armed BEFORE any group: their
    # per-(uid, channel) counters tick on EVERY issue from group creation
    # on (captured or not), so the same collective carries the SAME absolute
    # seq on every such rank. Delta alignment anchors at each rank's first
    # RECORDED key; when capture windows differ, the anchors name different
    # collectives and the join is fabricated. The c10d group-seq cross-check
    # is structurally blind to this class (each rank's base is taken at its
    # own delta 0, so a constant offset cancels); the absolute counters are
    # not. Seeded ranks stay out: their arm-time histories differ, so
    # absolute bases are legitimately rank-local facts (P3).
    armed_seq_abs = {
        rank: seq_abs
        for rank, seq_abs in seq_abs_by_rank.items()
        if armed_before_any_group(evidence[rank].install_epoch)
    }
    if len(set(armed_seq_abs.values())) <= 1:
        return None
    return MergedFinding(
        kind="correlation_delta_mismatch",
        detail=(
            f"absolute issue sequences disagree at key {key}: "
            f"{dict(sorted(armed_seq_abs.items()))}; every rank armed "
            "before any group ticks the same counter on the same "
            "collective, so the joined boundaries are not the same "
            "collective (the delta anchors name different first "
            "captures)."
        ),
        membership_digest=key[0],
        key=key,
        ranks=tuple(sorted(armed_seq_abs)),
    )


def presence_gap(key: tuple[str, int, str, int], missing: tuple[int, ...]) -> MergedFinding | None:
    """Return the presence-gap finding for member ranks that did not present a key.

    Parameters
    ----------
    key:
        The join key ``(digest, ordinal, channel, seq_delta)``.
    missing:
        Member ranks absent from this key, sorted.

    Returns
    -------
    MergedFinding | None
        A ``presence_gap`` finding, or ``None`` when no member is missing.
    """

    if not missing:
        return None
    digest, ordinal = key[0], key[1]
    return MergedFinding(
        kind="presence_gap",
        detail=(
            f"member rank(s) {list(missing)} of group {digest[:12]}… "
            f"(ordinal {ordinal}) did not present key {key}."
        ),
        membership_digest=digest,
        key=key,
        ranks=missing,
    )


def value_divergence(
    key: tuple[str, int, str, int], kind: str, presence: tuple[int, ...]
) -> MergedFinding:
    """Return the value-divergence finding for a mismatched witness verdict.

    Parameters
    ----------
    key:
        The join key ``(digest, ordinal, channel, seq_delta)``.
    kind:
        The collective kind at this key.
    presence:
        The ranks presenting this key, sorted.

    Returns
    -------
    MergedFinding
        A ``value_divergence`` finding; it is recorded, never repaired.
    """

    return MergedFinding(
        kind="value_divergence",
        detail=(
            f"byte-exact witness digests disagree at key {key} "
            f"({kind}); rank values are NOT interchangeable. The "
            "divergence is recorded, never repaired (P3)."
        ),
        membership_digest=key[0],
        key=key,
        ranks=presence,
    )


def declared_rank_gap(
    declared: tuple[int, ...] | None, input_ranks: tuple[int, ...]
) -> MergedFinding | None:
    """Return the presence gap for declared expected ranks that presented no core.

    Declared expected ranks can only WIDEN expectations: declaring fewer
    ranks than were presented never narrows anything.

    Parameters
    ----------
    declared:
        The sorted, de-duplicated declared world, or ``None``.
    input_ranks:
        The presented ranks, sorted.

    Returns
    -------
    MergedFinding | None
        A ``presence_gap`` finding, or ``None`` when nothing declared is absent.
    """

    if declared is None:
        return None
    absent = tuple(sorted(set(declared) - set(input_ranks)))
    if not absent:
        return None
    return MergedFinding(
        kind="presence_gap",
        detail=(f"declared expected rank(s) {list(absent)} presented no rank core."),
        ranks=absent,
    )


def stored_alignment(findings: Sequence[MergedFinding]) -> MergeAlignment:
    """Derive the stored alignment verdict from the findings ledger.

    Parameters
    ----------
    findings:
        Every finding of the derivation.

    Returns
    -------
    MergeAlignment
        ``conflicted`` on any structural finding, else ``partial`` on any
        presence gap, else ``aligned``.
    """

    structural = [f for f in findings if f.kind in _STRUCTURAL_KINDS]
    gaps = [f for f in findings if f.kind == "presence_gap"]
    if structural:
        return MergeAlignment.CONFLICTED
    if gaps:
        return MergeAlignment.PARTIAL
    return MergeAlignment.ALIGNED


def stored_value_status(consistencies: Sequence[BoundaryConsistency]) -> MergeValueStatus:
    """Derive the stored value status from every join's witness verdict.

    Parameters
    ----------
    consistencies:
        Every join's witness verdict, in join order.

    Returns
    -------
    MergeValueStatus
        ``divergent`` on any mismatch; ``attested_complete`` when every
        applicable verdict attests; ``attested_partial`` when some do;
        otherwise ``unwitnessed``.
    """

    applicable = [c for c in consistencies if c is not BoundaryConsistency.NOT_APPLICABLE]
    attested = [c for c in applicable if c is BoundaryConsistency.ATTESTED]
    if BoundaryConsistency.MISMATCHED in applicable:
        return MergeValueStatus.DIVERGENT
    if applicable and len(attested) == len(applicable):
        return MergeValueStatus.ATTESTED_COMPLETE
    if attested:
        return MergeValueStatus.ATTESTED_PARTIAL
    return MergeValueStatus.UNWITNESSED
