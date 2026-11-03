"""PRE-JOIN membership-lineage audit over per-rank lifecycle ledgers.

This is the cross-rank fail-closed layer above restricted local seeding
(design-merge-ranks-c v5, section 1.3). Before ANY correlation-key joining or
presence-gap derivation, the merge collects each member rank's lineage vector
for every membership appearing in two or more rank cores and audits:

1. Complete witnesses (install epoch ``armed_before_any_group``) must present
   element-wise identical lineage vectors.
2. Every ``seeded`` entry must be DISCHARGED against a one-generation complete
   witness; if any complete witness evidences two or more generations, every
   seeded entry from any other rank is unprovable and the membership refuses.
3. With no complete witness, all presenting vectors must be element-wise
   identical.
4. A conflicted membership refuses STRUCTURALLY
   (``group_lifetime_evidence_conflict``): no uid of that membership may join
   and no unmatched uid of it may become a presence gap. Enforcing that is the
   caller's (the C1 merge engine's) obligation; this module only renders the
   verdicts.

The audit is a pure function of the presented evidence so the merge engine can
call the SAME function at merge time and at load rederivation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ._ledger import GroupLifecycleLedger, InstallEpoch, LineageVector

__all__ = [
    "GROUP_LIFETIME_EVIDENCE_CONFLICT",
    "MembershipLineageVerdict",
    "verify_membership_lineages",
]

GROUP_LIFETIME_EVIDENCE_CONFLICT = "group_lifetime_evidence_conflict"
"""Finding kind rendered when presented lineage evidence cannot be reconciled."""

AuditStatus = Literal["compatible", "conflict"]


@dataclass(frozen=True)
class MembershipLineageVerdict:
    """Audit outcome for one membership digest.

    Parameters
    ----------
    membership_digest:
        The audited membership.
    status:
        ``"compatible"`` when every presenting rank's evidence provably
        describes the same generation sequence, else ``"conflict"``.
    kind:
        ``"group_lifetime_evidence_conflict"`` on conflict, ``None`` otherwise.
    detail:
        Human-readable explanation of the verdict. Diagnostics may sharpen a
        conflict report but never rescue a join.
    presenting_ranks:
        Ranks that presented lineage evidence for this membership.
    complete_witness_ranks:
        The subset whose install epoch makes them complete witnesses.
    """

    membership_digest: str
    status: AuditStatus
    kind: str | None
    detail: str
    presenting_ranks: tuple[int, ...]
    complete_witness_ranks: tuple[int, ...]

    @property
    def is_conflict(self) -> bool:
        """Whether this membership refuses structurally."""

        return self.status == "conflict"


def _vectors_identical(left: LineageVector, right: LineageVector) -> bool:
    """Element-wise lineage equality: ordinals, sources, and destroy marks."""

    return left.entries == right.entries


def _describe(vector: LineageVector) -> str:
    """Render a vector compactly for conflict details."""

    body = ", ".join(
        f"{entry.ordinal}: {entry.source}{' (destroyed)' if entry.destroyed else ''}"
        for entry in vector.entries
    )
    return "{" + body + "}"


def _mixed_case_compatible(agreed: LineageVector, candidate: LineageVector) -> tuple[bool, str]:
    """Check a seeded-epoch rank's vector against the agreed witness lineage.

    Returns
    -------
    tuple[bool, str]
        ``(compatible, reason)``. Compatible when the candidate is element-wise
        identical to the agreed lineage, or when its sole deviation is a
        ``seeded`` ordinal-0 entry discharged against a ONE-generation agreed
        lineage (destroy marks still required to agree).
    """

    if _vectors_identical(agreed, candidate):
        return True, "identical to complete-witness lineage"
    candidate_ordinals = tuple(entry.ordinal for entry in candidate.entries)
    agreed_ordinals = tuple(entry.ordinal for entry in agreed.entries)
    if candidate_ordinals != agreed_ordinals:
        return False, (
            f"ordinal set {candidate_ordinals} does not match the complete "
            f"witnesses' agreed ordinal set {agreed_ordinals}"
        )
    has_seed = any(entry.source == "seeded" for entry in candidate.entries)
    if has_seed and agreed.generations_created != 1:
        return False, (
            "a seeded entry is unprovable: the complete witnesses evidence "
            f"{agreed.generations_created} generations of this membership"
        )
    for agreed_entry, candidate_entry in zip(agreed.entries, candidate.entries):
        if candidate_entry.destroyed != agreed_entry.destroyed:
            return False, (f"destroy marks disagree at ordinal {agreed_entry.ordinal}")
        if candidate_entry.source == agreed_entry.source:
            continue
        if candidate_entry.source == "seeded" and candidate_entry.ordinal == 0:
            # Discharged above against the one-generation agreed lineage.
            continue
        return False, (
            f"sources disagree at ordinal {agreed_entry.ordinal}: "
            f"{candidate_entry.source} vs {agreed_entry.source}"
        )
    return True, "seed discharged against one-generation complete witness"


def verify_membership_lineages(
    rank_ledgers: Mapping[int, GroupLifecycleLedger],
    rank_install_epochs: Mapping[int, InstallEpoch] | None = None,
) -> dict[str, MembershipLineageVerdict]:
    """Audit per-membership lineage evidence across rank cores.

    Parameters
    ----------
    rank_ledgers:
        Mapping from global rank to that rank's group-lifecycle ledger.
    rank_install_epochs:
        Optional explicit install epochs per rank. When omitted, each
        membership uses the epoch stamped on the rank's ledger events for
        that membership (the normal, self-describing case).

    Returns
    -------
    dict[str, MembershipLineageVerdict]
        One verdict per membership digest presented by two or more ranks.
        Memberships presented by a single rank have no cross-rank evidence to
        reconcile and are not audited here; presence expectations for them are
        derived downstream from the recorded group memberships.

    Notes
    -----
    Conflicts refuse structurally: the caller must ensure no correlation key
    of a conflicted membership joins and no unmatched key of it becomes a
    presence gap. A gap is a claim about missing ranks; a conflict is a claim
    that the ranks PRESENT cannot be proven to describe the same communicator
    lineage. The remedy a conflict names is arming before any group creation
    (``torchlens.distributed.arm``).
    """

    per_membership: dict[str, dict[int, LineageVector]] = {}
    for rank, ledger in rank_ledgers.items():
        for digest, vector in ledger.lineage_vectors().items():
            if rank_install_epochs is not None and rank in rank_install_epochs:
                vector = LineageVector(
                    membership_digest=vector.membership_digest,
                    entries=vector.entries,
                    install_epoch=rank_install_epochs[rank],
                )
            per_membership.setdefault(digest, {})[int(rank)] = vector

    verdicts: dict[str, MembershipLineageVerdict] = {}
    for digest, by_rank in per_membership.items():
        if len(by_rank) < 2:
            continue
        verdicts[digest] = _audit_one_membership(digest, by_rank)
    return verdicts


def _audit_one_membership(
    digest: str, by_rank: dict[int, LineageVector]
) -> MembershipLineageVerdict:
    """Render the verdict for one membership's presented vectors."""

    ranks = tuple(sorted(by_rank))
    witness_ranks = tuple(
        rank for rank in ranks if by_rank[rank].install_epoch == "armed_before_any_group"
    )

    def conflict(detail: str) -> MembershipLineageVerdict:
        """Build a ``conflict`` verdict for this membership, with the arming remedy appended."""

        remedy = (
            " Remedy: call torchlens.distributed.arm() at process start, "
            "before any process group is created, on every rank."
        )
        return MembershipLineageVerdict(
            membership_digest=digest,
            status="conflict",
            kind=GROUP_LIFETIME_EVIDENCE_CONFLICT,
            detail=detail + remedy,
            presenting_ranks=ranks,
            complete_witness_ranks=witness_ranks,
        )

    def compatible(detail: str) -> MembershipLineageVerdict:
        """Build a ``compatible`` verdict for this membership."""

        return MembershipLineageVerdict(
            membership_digest=digest,
            status="compatible",
            kind=None,
            detail=detail,
            presenting_ranks=ranks,
            complete_witness_ranks=witness_ranks,
        )

    if witness_ranks:
        agreed = by_rank[witness_ranks[0]]
        for rank in witness_ranks[1:]:
            if not _vectors_identical(agreed, by_rank[rank]):
                return conflict(
                    "complete witnesses disagree: rank "
                    f"{witness_ranks[0]} presents {_describe(agreed)} but rank "
                    f"{rank} presents {_describe(by_rank[rank])}; c10d requires "
                    "identical creation/destruction program order across member "
                    "ranks, so this is evidence corruption."
                )
        for rank in ranks:
            if rank in witness_ranks:
                continue
            ok, reason = _mixed_case_compatible(agreed, by_rank[rank])
            if not ok:
                return conflict(
                    f"rank {rank} presents {_describe(by_rank[rank])} against the "
                    f"complete witnesses' agreed {_describe(agreed)}: {reason}."
                )
        return compatible(
            "complete witnesses agree and every non-witness vector is identical or seed-discharged"
        )

    # No complete witness: every presenting vector must be element-wise
    # identical. Soundness: an armed member rank cannot miss a later
    # generation, so identical vectors pin the same generation sequence.
    first_rank = ranks[0]
    first = by_rank[first_rank]
    for rank in ranks[1:]:
        if not _vectors_identical(first, by_rank[rank]):
            return conflict(
                "no complete witness for this membership and the presented "
                f"vectors are not identical: rank {first_rank} presents "
                f"{_describe(first)} but rank {rank} presents "
                f"{_describe(by_rank[rank])}; the seeds cannot be proven to "
                "denote the same generation."
            )
    return compatible("no complete witness; all presenting vectors identical")
