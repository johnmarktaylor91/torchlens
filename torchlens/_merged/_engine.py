"""The C1 merge derivation: audit-first joining, delta alignment, witnesses.

ONE pure function (:func:`derive_merge`) produces the complete merge
derivation from per-rank evidence. It runs at merge time and again, verbatim,
at load rederivation (design-merge-ranks-c v5, 4.3) -- the saved descriptor is
only a cache of its output.

Order of operations is normative (v5 1.3): the PRE-JOIN membership-lineage
audit runs BEFORE any correlation-key joining or presence-gap derivation; a
conflicted membership refuses structurally -- no uid of it joins and no
unmatched key of it becomes a presence gap.

Correlation alignment is by counting (P3): per ``(group_uid, channel)`` each
rank's recorded boundaries align as seq DELTAS from that rank's first recorded
key -- a SEEDED rank's absolute seq base is a rank-local fact (arm-time
histories differ) and is never compared. Ranks armed BEFORE any group are the
exception: their counters tick on every issue from group creation on, so the
same collective carries the same absolute seq on every such rank, and a
disagreement at a joined key proves the delta anchors paired non-corresponding
collectives (a structural ``correlation_delta_mismatch``). Witness digests are
redundant byte-exact evidence that can only DEMOTE a verdict, never rescue or
repair one.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..distributed._audit import MembershipLineageVerdict
from ._derive_steps import (
    _STRUCTURAL_KINDS,
    _guard_scope,
    absolute_seq_mismatch,
    align_seq_deltas,
    audit_membership_lineages,
    build_group_table,
    c10d_delta_mismatch,
    check_membership_authority,
    collect_channel_records,
    declared_rank_gap,
    presence_gap,
    stored_alignment,
    stored_value_status,
    value_divergence,
)
from ._enums import (
    BoundaryConsistency,
    MergeAlignment,
    MergedErrorCode,
    MergeValueStatus,
)
from ._errors import MergedFinding, MergeInputError
from ._evidence import RankEvidence
from ._relation_table import _relation_findings
from ._rules import (
    CONTRIBUTION_ROLES,
    DESTINATION_ROLES,
    WITNESS_DIGEST_FIELDS,
    is_digest_sequence,
    roles_of,
)
from ._witness import witness_consistency

__all__ = [
    "JoinKey",
    "JoinRecord",
    "MergeDerivation",
    "PerRankRef",
    "derive_merge",
]

JoinKey = tuple[str, int, str, int]
"""``(membership_digest, lifetime_ordinal, channel, seq_delta)``."""


@dataclass(frozen=True)
class PerRankRef:
    """One rank's contribution to a join: references, never copies (P1)."""

    rank: int
    boundary_index: int
    seq_abs: int
    op_labels_raw: tuple[str, ...]
    group_rank: int | None
    witness_policy: str
    n_contribution_roles: int
    n_destination_roles: int
    contribution_digests: tuple[str, ...] | None
    destination_digests: tuple[str, ...] | None
    c10d_group_seq: int | None

    def to_payload(self) -> dict[str, Any]:
        """Canonical JSON projection (descriptor cache row)."""

        return {
            "rank": self.rank,
            "boundary_index": self.boundary_index,
            "seq_abs": self.seq_abs,
            "op_labels_raw": list(self.op_labels_raw),
            "group_rank": self.group_rank,
            "witness_policy": self.witness_policy,
            "n_contribution_roles": self.n_contribution_roles,
            "n_destination_roles": self.n_destination_roles,
            "contribution_digests": (
                None if self.contribution_digests is None else list(self.contribution_digests)
            ),
            "destination_digests": (
                None if self.destination_digests is None else list(self.destination_digests)
            ),
            "c10d_group_seq": self.c10d_group_seq,
        }


@dataclass(frozen=True)
class JoinRecord:
    """One cross-rank collective join at component level."""

    key: JoinKey
    kind: str
    reduce_op: str | None
    membership: tuple[int, ...]
    backend: str | None
    presence: tuple[int, ...]
    missing: tuple[int, ...]
    per_rank: Mapping[int, PerRankRef]
    consistency: BoundaryConsistency

    def to_payload(self) -> dict[str, Any]:
        """Canonical JSON projection (descriptor cache row)."""

        return {
            "key": list(self.key),
            "kind": self.kind,
            "reduce_op": self.reduce_op,
            "membership": list(self.membership),
            "backend": self.backend,
            "presence": list(self.presence),
            "missing": list(self.missing),
            "per_rank": {
                str(rank): ref.to_payload() for rank, ref in sorted(self.per_rank.items())
            },
            "consistency": self.consistency.value,
        }


@dataclass(frozen=True)
class MergeDerivation:
    """Complete, deterministic output of the one merge derivation."""

    ranks: tuple[int, ...]
    expected_ranks: tuple[int, ...] | None
    joins: tuple[JoinRecord, ...]
    findings: tuple[MergedFinding, ...]
    audit: Mapping[str, MembershipLineageVerdict]
    stored_alignment: MergeAlignment
    stored_value_status: MergeValueStatus
    conflicted_memberships: tuple[str, ...] = field(default_factory=tuple)

    @property
    def structural_findings(self) -> tuple[MergedFinding, ...]:
        """Findings that make the merge ``conflicted``."""

        return tuple(f for f in self.findings if f.kind in _STRUCTURAL_KINDS)

    @property
    def gap_findings(self) -> tuple[MergedFinding, ...]:
        """The typed presence-gap ledger."""

        return tuple(f for f in self.findings if f.kind == "presence_gap")

    @property
    def divergence_findings(self) -> tuple[MergedFinding, ...]:
        """Value-divergence findings (never structural)."""

        return tuple(f for f in self.findings if f.kind == "value_divergence")

    def to_payload(self) -> dict[str, Any]:
        """Canonical JSON projection: the descriptor cache body.

        Pure function of the input rank cores -- contains no timestamps, no
        environment strings, and no absolute paths, so byte-equality of this
        payload is the load-time tamper oracle.
        """

        return {
            "ranks": list(self.ranks),
            "expected_ranks": (None if self.expected_ranks is None else list(self.expected_ranks)),
            "joins": [join.to_payload() for join in self.joins],
            "findings": [finding.to_dict() for finding in self.findings],
            "audit": {
                digest: {
                    "status": verdict.status,
                    "kind": verdict.kind,
                    "presenting_ranks": list(verdict.presenting_ranks),
                    "complete_witness_ranks": list(verdict.complete_witness_ranks),
                }
                for digest, verdict in sorted(self.audit.items())
            },
            "conflicted_memberships": list(self.conflicted_memberships),
            "stored_alignment": self.stored_alignment.value,
            "stored_value_status": self.stored_value_status.value,
        }


def _role_digests(entry: dict[str, Any]) -> tuple[tuple[str, ...] | None, tuple[str, ...] | None]:
    """Return ``(contribution_digests, destination_digests)`` or ``None`` each."""

    witness = entry["witness"]
    values: list[tuple[str, ...] | None] = []
    for digest_field in WITNESS_DIGEST_FIELDS:
        digests = witness.get(digest_field)
        if digests is None:
            values.append(None)
            continue
        # Belt for internal RankEvidence constructors that bypass the parse
        # boundary (parse additionally requires a JSON list, non-empty for
        # tensor-carrying kinds, of lowercase hex digests).
        if not is_digest_sequence(digests):
            raise MergeInputError(
                f"Boundary witness {digest_field} is not a list of digests "
                f"(got {type(digests).__name__}).",
                code=MergedErrorCode.MERGED_SCHEMA_INVALID,
            )
        values.append(tuple(str(item) for item in digests))
    return values[0], values[1]


class _JoinClasses:
    """Union-find over boundary nodes ``(rank, index)`` and join classes ``("join", key)``."""

    def __init__(self) -> None:
        self._parent: dict[Any, Any] = {}

    def find(self, node: Any) -> Any:
        """Return the representative of ``node``, with path compression."""

        parent = self._parent
        root = node
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(node, node) != node:
            parent[node], node = root, parent[node]
        return root

    def union(self, a: Any, b: Any) -> None:
        """Merge the classes of ``a`` and ``b``."""

        self._parent[self.find(a)] = self.find(b)


OrderGraph = tuple[set[Any], dict[Any, set[Any]], dict[Any, int]]
"""``(nodes, successors, indegree)`` of the joined order graph."""


def _order_graph(
    evidence: Mapping[int, RankEvidence],
    joined_nodes: Mapping[tuple[int, int], JoinKey],
) -> OrderGraph:
    """Build the order graph: each rank's issue order over the join-identified nodes.

    Every joined boundary collapses into its join class; consecutive
    boundaries of one rank become one edge each (self-edges and repeated
    edges dropped).
    """

    classes = _JoinClasses()
    for (rank, index), key in joined_nodes.items():
        classes.union((rank, index), ("join", key))
    edges: dict[Any, set[Any]] = {}
    indegree: dict[Any, int] = {}
    nodes: set[Any] = set()
    for rank in sorted(evidence):
        boundary_nodes = [
            classes.find((rank, index)) if (rank, index) in joined_nodes else (rank, index)
            for index in range(len(evidence[rank].boundaries))
        ]
        nodes.update(boundary_nodes)
        for left, right in zip(boundary_nodes, boundary_nodes[1:]):
            if left == right:
                continue
            if right not in edges.setdefault(left, set()):
                edges[left].add(right)
                indegree[right] = indegree.get(right, 0) + 1
    return nodes, edges, indegree


def _has_cycle(graph: OrderGraph) -> bool:
    """Return whether the order graph has a cycle (Kahn's algorithm)."""

    nodes, edges, indegree = graph
    remaining = dict(indegree)
    queue = [node for node in nodes if remaining.get(node, 0) == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for successor in edges.get(node, ()):
            remaining[successor] -= 1
            if remaining[successor] == 0:
                queue.append(successor)
    return visited != len(nodes)


def _check_order(
    evidence: Mapping[int, RankEvidence],
    joined_nodes: Mapping[tuple[int, int], JoinKey],
) -> MergedFinding | None:
    """Cycle check over rank-local order + join identifications (2.2)."""

    if not _has_cycle(_order_graph(evidence, joined_nodes)):
        return None
    return MergedFinding(
        kind="order_contradiction",
        detail=(
            "the joined cross-rank order contradicts at least one rank's "
            "local issue order (a cycle exists through the join graph); "
            "the presented cores cannot describe one consistent execution."
        ),
        ranks=tuple(sorted(evidence)),
    )


def _per_rank_refs(
    evidence: Mapping[int, RankEvidence], entries_at: Mapping[int, tuple[int, int]]
) -> tuple[dict[int, dict[str, Any]], dict[int, PerRankRef]]:
    """Reference each presenting rank's boundary at one key, never copying it.

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.
    entries_at:
        ``rank -> (seq_abs, boundary_index)`` for the ranks presenting the key.

    Returns
    -------
    tuple[dict[int, dict[str, Any]], dict[int, PerRankRef]]
        The boundary records by rank (in ``entries_at`` order) and the
        per-rank join references (in rank order).

    Raises
    ------
    MergeInputError
        When a witness digest field is not a list of digests.
    """

    entries = {rank: evidence[rank].boundaries[index] for rank, (_seq, index) in entries_at.items()}
    per_rank: dict[int, PerRankRef] = {}
    for rank, (seq_abs, index) in sorted(entries_at.items()):
        entry = entries[rank]
        contribution, destination = _role_digests(entry)
        per_rank[rank] = PerRankRef(
            rank=rank,
            boundary_index=index,
            seq_abs=seq_abs,
            op_labels_raw=tuple(entry["op_labels_raw"]),
            group_rank=entry["group"].get("my_group_rank"),
            witness_policy=entry["witness"]["policy_resolved"],
            n_contribution_roles=len(roles_of(entry, CONTRIBUTION_ROLES)),
            n_destination_roles=len(roles_of(entry, DESTINATION_ROLES)),
            contribution_digests=contribution,
            destination_digests=destination,
            c10d_group_seq=entry.get("c10d_group_seq"),
        )
    return entries, per_rank


def _presence_and_missing(
    key: JoinKey, per_rank: Mapping[int, PerRankRef], membership: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return a key's ``(presence, missing)`` ranks, refusing presence outside membership.

    Raises
    ------
    MergeInputError
        When a presenting rank is not a member of the key's group.
    """

    presence = tuple(sorted(per_rank))
    if not set(presence).issubset(membership):
        raise MergeInputError(
            f"Join key {key} has presence {presence} outside membership {membership}.",
            code=MergedErrorCode.MERGED_SCHEMA_INVALID,
        )
    return presence, tuple(sorted(set(membership) - set(presence)))


def _reduce_op(entries: Mapping[int, dict[str, Any]]) -> str | None:
    """Return the least distinct ``reduce_op`` the presenting entries record, or ``None``."""

    reduce_ops = sorted(
        {
            str(entry.get("reduce_op"))
            for entry in entries.values()
            if entry.get("reduce_op") is not None
        }
    )
    return reduce_ops[0] if reduce_ops else None


def _key_cross_checks(
    evidence: Mapping[int, RankEvidence],
    key: JoinKey,
    per_rank: Mapping[int, PerRankRef],
    group_seq_base: dict[int, int],
    missing: tuple[int, ...],
) -> list[MergedFinding]:
    """Run one key's two correlation cross-checks (both structural), then the gap.

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.
    key:
        The join key ``(digest, ordinal, channel, seq_delta)``.
    per_rank:
        The per-rank join references, keyed and ordered by rank.
    group_seq_base:
        The channel's per-rank backend group-seq base, shared across its keys.
    missing:
        The member ranks that do not present the key.

    Returns
    -------
    list[MergedFinding]
        The backend group-seq, absolute-seq and presence-gap findings, in
        that order, omitting the checks that pass.
    """

    presence = tuple(per_rank)
    checks = (
        c10d_delta_mismatch(
            key,
            presence,
            {rank: per_rank[rank].c10d_group_seq for rank in presence},
            group_seq_base,
        ),
        absolute_seq_mismatch(evidence, key, {rank: per_rank[rank].seq_abs for rank in presence}),
        presence_gap(key, missing),
    )
    return [finding for finding in checks if finding is not None]


def _join_key(
    evidence: Mapping[int, RankEvidence],
    key: JoinKey,
    entries_at: Mapping[int, tuple[int, int]],
    group: tuple[tuple[int, ...], str | None],
    group_seq_base: dict[int, int],
) -> tuple[JoinRecord, list[MergedFinding]]:
    """Join one correlation key across the ranks presenting it.

    Builds the per-rank references, then runs the key's checks in their
    normative order: relation table, backend group-seq cross-check,
    absolute-seq cross-check, presence gap, and the demote-only witness
    verdict.

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.
    key:
        The join key ``(digest, ordinal, channel, seq_delta)``.
    entries_at:
        ``rank -> (seq_abs, boundary_index)`` for the ranks presenting ``key``.
    group:
        The group-table row ``(membership, backend)``; the backend is
        ``None`` when unknown or disputed.
    group_seq_base:
        The channel's per-rank backend group-seq base, shared across its keys.

    Returns
    -------
    tuple[JoinRecord, list[MergedFinding]]
        The join record and this key's findings, in order.

    Raises
    ------
    MergeInputError
        On a malformed witness digest field, presence outside the
        membership, or an out-of-range group rank.
    """

    membership, backend = group
    entries, per_rank = _per_rank_refs(evidence, entries_at)
    presence, missing = _presence_and_missing(key, per_rank, membership)
    kind = entries[presence[0]]["kind"]
    findings = _relation_findings(key, kind, membership, entries)
    findings.extend(_key_cross_checks(evidence, key, per_rank, group_seq_base, missing))
    consistency = witness_consistency(kind, backend, membership, presence, per_rank)
    if consistency is BoundaryConsistency.MISMATCHED:
        findings.append(value_divergence(key, kind, presence))
    return JoinRecord(
        key=key,
        kind=kind,
        reduce_op=_reduce_op(entries),
        membership=membership,
        backend=backend,
        presence=presence,
        missing=missing,
        per_rank=per_rank,
        consistency=consistency,
    ), findings


def _join_channel(
    evidence: Mapping[int, RankEvidence],
    channel_key: tuple[str, int, str],
    by_rank: Mapping[int, list[tuple[int, int]]],
    group: tuple[tuple[int, ...], str | None],
) -> tuple[list[JoinRecord], dict[tuple[int, int], JoinKey], list[MergedFinding]]:
    """Join every key of one correlation channel, in seq-delta order.

    Parameters
    ----------
    evidence:
        Mapping from global rank to rank evidence.
    channel_key:
        ``(membership_digest, lifetime_ordinal, channel)``.
    by_rank:
        The channel's per-rank ``(seq_abs, boundary_index)`` records.
    group:
        The group-table row ``(membership, backend)`` of the channel's uid.

    Returns
    -------
    tuple[list[JoinRecord], dict[tuple[int, int], JoinKey], list[MergedFinding]]
        The channel's joins, the ``(rank, boundary_index) -> key`` node map
        for the order check, and the channel's findings, all in key order.
    """

    digest, ordinal, channel = channel_key
    deltas = align_seq_deltas(by_rank)
    joins: list[JoinRecord] = []
    joined_nodes: dict[tuple[int, int], JoinKey] = {}
    findings: list[MergedFinding] = []
    c10d_base: dict[int, int] = {}
    for delta in sorted(deltas):
        key: JoinKey = (digest, ordinal, channel, delta)
        record, key_findings = _join_key(evidence, key, deltas[delta], group, c10d_base)
        for rank, ref in record.per_rank.items():
            joined_nodes[(rank, ref.boundary_index)] = key
        joins.append(record)
        findings.extend(key_findings)
    return joins, joined_nodes, findings


def derive_merge(
    evidence: Mapping[int, RankEvidence],
    expected_ranks: Iterable[int] | None = None,
) -> MergeDerivation:
    """Derive the complete merge from per-rank evidence (the ONE function).

    Parameters
    ----------
    evidence:
        Mapping from global rank to validated rank evidence.
    expected_ranks:
        Optional declared world. Can only WIDEN presence expectations beyond
        the group memberships recorded inside the rank cores (4.3) -- a
        declared rank with no core is a presence gap; declaring fewer ranks
        than the recorded memberships never narrows anything.

    Returns
    -------
    MergeDerivation
        Joins, findings ledgers, audit verdicts, and stored verdicts.

    Raises
    ------
    MergeInputError
        On out-of-scope (p2p/pipeline/DTensor) boundaries.
    """

    _guard_scope(evidence)
    input_ranks = tuple(sorted(evidence))
    declared = None if expected_ranks is None else tuple(sorted({int(r) for r in expected_ranks}))
    check_membership_authority(evidence, input_ranks)

    # 1. PRE-JOIN membership-lineage audit, before ANY joining or gaps (1.3).
    audit, conflicted_memberships, findings = audit_membership_lineages(evidence)

    # 2. Group table from the recorded memberships (rank cores are authority).
    group_members, group_backend, group_findings = build_group_table(evidence, input_ranks)
    findings.extend(group_findings)

    # 3. Correlation joining: seq DELTAS from each rank's first recorded key.
    per_channel = collect_channel_records(evidence, input_ranks, conflicted_memberships)
    joins: list[JoinRecord] = []
    joined_nodes: dict[tuple[int, int], JoinKey] = {}
    for channel_key in sorted(per_channel):
        uid = (channel_key[0], channel_key[1])
        channel_joins, channel_nodes, channel_findings = _join_channel(
            evidence,
            channel_key,
            per_channel[channel_key],
            (group_members[uid], group_backend[uid]),
        )
        joins.extend(channel_joins)
        joined_nodes.update(channel_nodes)
        findings.extend(channel_findings)

    # 4. Declared expected ranks can only WIDEN expectations (4.3).
    declared_gap = declared_rank_gap(declared, input_ranks)
    if declared_gap is not None:
        findings.append(declared_gap)

    # 5. Partial-order consistency.
    order_finding = _check_order(evidence, joined_nodes)
    if order_finding is not None:
        findings.append(order_finding)

    # 6. Verdicts.
    return MergeDerivation(
        ranks=input_ranks,
        expected_ranks=declared,
        joins=tuple(joins),
        findings=tuple(findings),
        audit=dict(sorted(audit.items())),
        stored_alignment=stored_alignment(findings),
        stored_value_status=stored_value_status([join.consistency for join in joins]),
        conflicted_memberships=conflicted_memberships,
    )
