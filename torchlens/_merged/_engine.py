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
    TENSORLESS_KINDS,
    WITNESS_IDENTITY_KINDS,
    WITNESS_NOT_APPLICABLE_KINDS,
    WITNESS_VERDICT_BACKENDS,
    BoundaryConsistency,
    MergeAlignment,
    MergedErrorCode,
    MergeValueStatus,
)
from ._errors import MergedFinding, MergeInputError
from ._evidence import RankEvidence

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
    for digest_field in ("contribution_digests", "destination_digests"):
        digests = witness.get(digest_field)
        if digests is None:
            values.append(None)
            continue
        # Belt for internal RankEvidence constructors that bypass the parse
        # boundary: a bare string here would char-split through ``tuple(...)``
        # and its characters would then be compared as digests, so two cores
        # carrying the same garbage string fabricated an ATTESTED verdict.
        if isinstance(digests, str) or not isinstance(digests, (list, tuple)):
            raise MergeInputError(
                f"Boundary witness {digest_field} is not a list of digests "
                f"(got {type(digests).__name__}).",
                code=MergedErrorCode.MERGED_SCHEMA_INVALID,
            )
        values.append(tuple(str(item) for item in digests))
    return values[0], values[1]


def _roles_of(entry: dict[str, Any], role_names: Iterable[str]) -> list[dict[str, Any]]:
    """Return the ordered role entries matching ``role_names``."""

    wanted = set(role_names)
    return [role for role in entry.get("roles", ()) if role.get("role") in wanted]


def _relation_findings(
    key: JoinKey,
    kind: str,
    membership: tuple[int, ...],
    entries: dict[int, dict[str, Any]],
) -> list[MergedFinding]:
    """Evaluate the per-kind relation table over the presenting entries (1.5)."""

    findings: list[MergedFinding] = []
    ranks = tuple(sorted(entries))

    def violation(detail: str) -> None:
        """Append one ``relation_violation`` finding for this key."""

        findings.append(
            MergedFinding(
                kind="relation_violation",
                detail=f"{kind} at key {key}: {detail}",
                membership_digest=key[0],
                key=key,
                ranks=ranks,
            )
        )

    kinds_seen = {entry["kind"] for entry in entries.values()}
    if len(kinds_seen) > 1:
        violation(f"ranks disagree on the collective kind: {sorted(kinds_seen)}")
        return findings

    # Role cardinality vs kind (b6-opus-R18-1 belt): the set-of-shapes
    # agreement below is vacuously satisfied when EVERY presenting rank holds
    # zero roles, so uniform roles deletion rendered the TOP verdict. Evidence
    # parse refuses that shape outright; this belt keeps direct engine callers
    # honest and names each offending rank.
    if kind not in TENSORLESS_KINDS:
        all_roles = ("contribution", "destination", "contribution_destination")
        for rank, entry in sorted(entries.items()):
            if not _roles_of(entry, all_roles):
                violation(f"rank {rank} presents zero tensor roles for a tensor-carrying boundary")

    reduce_ops = {entry.get("reduce_op") for entry in entries.values()}
    if len(reduce_ops) > 1:
        violation(f"ranks disagree on the reduce op: {sorted(map(str, reduce_ops))}")

    def shapes(entry: dict[str, Any], names: Iterable[str]) -> tuple[tuple[int, ...], ...]:
        """Collect the shapes of ``entry``'s roles matching any of ``names``, in order."""

        return tuple(tuple(role["shape"]) for role in _roles_of(entry, names))

    group_size = len(membership)
    contribution_roles = ("contribution", "contribution_destination")
    destination_roles = ("destination", "contribution_destination")
    if kind in ("all_reduce", "all_to_all_single", "all_gather_into_tensor", "broadcast"):
        shape_sets = {
            shapes(entry, contribution_roles) or shapes(entry, destination_roles)
            for entry in entries.values()
        }
        if len(shape_sets) > 1:
            violation(f"tensor shapes disagree across ranks: {sorted(shape_sets)}")
    # Distinct-shape-set agreement belt for the remaining symmetric kinds
    # (R18-1 iii fallback): every role of these kinds shares one element
    # geometry across the group (all_gather/gather/scatter slots equal the
    # contribution shape; reduce inputs equal outputs; reduce_scatter feeds
    # group_size output-shaped slots; reduce_scatter_tensor's two shapes
    # repeat on every rank), so the SET of distinct role shapes must agree
    # across ranks even though per-rank role COUNTS legitimately differ
    # (root list vs leaf tensor). all_to_all stays out: uneven splits make
    # per-rank shape sets legitimately rank-local. A UNIFORM cross-rank shape
    # rewrite still passes this belt -- coherent reauthoring, the documented
    # out-of-scope boundary (only a per-core cross-check against the member's
    # own op records could catch it, and parse never dereferences the trace).
    if kind in (
        "all_gather",
        "gather",
        "scatter",
        "reduce",
        "reduce_scatter",
        "reduce_scatter_tensor",
    ):
        distinct_shape_sets = {
            frozenset(shapes(entry, ("contribution", "destination", "contribution_destination")))
            for entry in entries.values()
            if entry.get("roles")
        }
        if len(distinct_shape_sets) > 1:
            violation(
                "distinct role shapes disagree across ranks: "
                f"{sorted(sorted(s) for s in distinct_shape_sets)}"
            )
    if kind == "all_gather":
        for rank, entry in sorted(entries.items()):
            n_destinations = len(_roles_of(entry, destination_roles))
            if n_destinations != group_size:
                violation(
                    f"rank {rank} records {n_destinations} destination(s) for a "
                    f"group of {group_size}"
                )
    if kind == "all_to_all":
        for rank, entry in sorted(entries.items()):
            n_in = len(_roles_of(entry, contribution_roles))
            n_out = len(_roles_of(entry, destination_roles))
            if n_in != group_size or n_out != group_size:
                violation(
                    f"rank {rank} records {n_in} contribution(s) and {n_out} "
                    f"destination(s) for a group of {group_size}"
                )
    if kind in ("broadcast", "reduce", "gather", "scatter"):
        # Root-only aggregate presence (1.5): at most one rank may present
        # the root's role signature.
        if kind == "broadcast":
            roots = [
                rank
                for rank, entry in sorted(entries.items())
                if not _roles_of(entry, ("destination",))
            ]
        elif kind == "reduce":
            roots = [
                rank
                for rank, entry in sorted(entries.items())
                if _roles_of(entry, destination_roles)
            ]
        else:
            # gather roots hold a destination LIST; scatter roots a
            # contribution LIST (group_size entries; leaves hold one tensor).
            list_roles = destination_roles if kind == "gather" else contribution_roles
            roots = [
                rank
                for rank, entry in sorted(entries.items())
                if len(_roles_of(entry, list_roles)) == group_size and group_size > 1
            ]
        if len(roots) > 1:
            violation(f"multiple ranks {roots} present root-only aggregate roles")
    return findings


def _witness_consistency(
    kind: str,
    backend: str | None,
    membership: tuple[int, ...],
    presence: tuple[int, ...],
    per_rank: Mapping[int, PerRankRef],
) -> BoundaryConsistency:
    """Totalized per-join witness derivation (v5 3.3; demote-only)."""

    if kind in WITNESS_NOT_APPLICABLE_KINDS:
        return BoundaryConsistency.NOT_APPLICABLE
    if backend is None or backend not in WITNESS_VERDICT_BACKENDS:
        return BoundaryConsistency.NOT_APPLICABLE
    group_size = len(membership)

    if kind in WITNESS_IDENTITY_KINDS:
        # Every member rank ends holding identical bytes. The comparable
        # digest is the rank's DESTINATION digest; a broadcast root has no
        # destination role and its contribution bytes ARE the final bytes.
        # A rank with destination roles but absent destination digests
        # (witness level "none", async completion unobserved) contributes
        # nothing -- its pre-collective contribution digest is never a
        # substitute (that comparison could only fabricate a mismatch).
        values: list[tuple[str, ...]] = []
        for rank in presence:
            ref = per_rank[rank]
            if ref.n_destination_roles:
                value = ref.destination_digests
            else:
                value = ref.contribution_digests
            if value:
                values.append(tuple(value))
        if len(set(values)) > 1:
            return BoundaryConsistency.MISMATCHED
        if len(values) < group_size:
            return BoundaryConsistency.NOT_PRESENT
        return BoundaryConsistency.ATTESTED

    # Slice kinds: gather / scatter / all_to_all, group-rank indexed pairs.
    pairs: list[tuple[str, str]] = []
    incomplete = len(presence) < group_size
    by_group_rank: dict[int, int] = {}
    for rank in presence:
        group_rank = per_rank[rank].group_rank
        if group_rank is not None:
            # Belt for internal RankEvidence constructors that bypass the
            # parse boundary (matching the _role_digests belt above): a
            # permuted/out-of-range group rank used to reach the group-rank
            # list indexing below as a raw IndexError -- or silently rebind a
            # slice pairing -- instead of the promised typed refusal. Parse
            # already pins my_group_rank to the membership position (R18-2).
            if (
                isinstance(group_rank, bool)
                or not isinstance(group_rank, int)
                or not 0 <= group_rank < group_size
            ):
                raise MergeInputError(
                    f"Boundary group rank {group_rank!r} for rank {rank} is "
                    f"outside its recorded group of {group_size} member(s).",
                    code=MergedErrorCode.MERGED_SCHEMA_INVALID,
                )
            by_group_rank[group_rank] = rank
    if len(by_group_rank) < len(presence):
        incomplete = True
    if kind in ("gather", "scatter"):
        # gather: root's destination list slice[i] == member i's contribution.
        # scatter: member i's destination == root's contribution list slice[i].
        root_list_attr = "destination_digests" if kind == "gather" else "contribution_digests"
        leaf_attr = "contribution_digests" if kind == "gather" else "destination_digests"
        n_list_attr = "n_destination_roles" if kind == "gather" else "n_contribution_roles"
        roots = [r for r in presence if getattr(per_rank[r], n_list_attr) == group_size]
        expected_pairs = group_size
        if len(roots) != 1:
            incomplete = True
        else:
            root_list = getattr(per_rank[roots[0]], root_list_attr)
            if not root_list or len(root_list) != group_size:
                incomplete = True
            else:
                for group_rank in sorted(by_group_rank):
                    rank = by_group_rank[group_rank]
                    leaf = getattr(per_rank[rank], leaf_attr)
                    if not leaf:
                        incomplete = True
                        continue
                    pairs.append((root_list[group_rank], leaf[0]))
    else:  # all_to_all: rank j's destination[i] == rank i's contribution[j]
        expected_pairs = group_size * group_size
        for gr_j in sorted(by_group_rank):
            dest_j = per_rank[by_group_rank[gr_j]].destination_digests
            if not dest_j:
                incomplete = True
                continue
            for gr_i in sorted(by_group_rank):
                contrib_i = per_rank[by_group_rank[gr_i]].contribution_digests
                if not contrib_i or gr_i >= len(dest_j) or gr_j >= len(contrib_i):
                    incomplete = True
                    continue
                pairs.append((dest_j[gr_i], contrib_i[gr_j]))
    if any(left != right for left, right in pairs):
        return BoundaryConsistency.MISMATCHED
    if incomplete or len(pairs) < expected_pairs:
        return BoundaryConsistency.NOT_PRESENT
    return BoundaryConsistency.ATTESTED


def _check_order(
    evidence: Mapping[int, RankEvidence],
    joined_nodes: Mapping[tuple[int, int], JoinKey],
) -> MergedFinding | None:
    """Cycle check over rank-local order + join identifications (2.2)."""

    parent: dict[Any, Any] = {}

    def find(node: Any) -> Any:
        """Union-find representative of ``node``, with path compression."""

        root = node
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(node, node) != node:
            parent[node], node = root, parent[node]
        return root

    def union(a: Any, b: Any) -> None:
        """Merge the classes of ``a`` and ``b``."""

        parent[find(a)] = find(b)

    for (rank, index), key in joined_nodes.items():
        union((rank, index), ("join", key))

    edges: dict[Any, set[Any]] = {}
    indegree: dict[Any, int] = {}
    nodes: set[Any] = set()
    for rank in sorted(evidence):
        boundary_nodes = [
            find((rank, index)) if (rank, index) in joined_nodes else (rank, index)
            for index in range(len(evidence[rank].boundaries))
        ]
        for node in boundary_nodes:
            nodes.add(node)
        for left, right in zip(boundary_nodes, boundary_nodes[1:]):
            if left == right:
                continue
            if right not in edges.setdefault(left, set()):
                edges[left].add(right)
                indegree[right] = indegree.get(right, 0) + 1
                nodes.add(right)
    queue = [node for node in nodes if indegree.get(node, 0) == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for successor in edges.get(node, ()):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if visited != len(nodes):
        return MergedFinding(
            kind="order_contradiction",
            detail=(
                "the joined cross-rank order contradicts at least one rank's "
                "local issue order (a cycle exists through the join graph); "
                "the presented cores cannot describe one consistent execution."
            ),
            ranks=tuple(sorted(evidence)),
        )
    return None


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
            n_contribution_roles=len(
                _roles_of(entry, ("contribution", "contribution_destination"))
            ),
            n_destination_roles=len(_roles_of(entry, ("destination", "contribution_destination"))),
            contribution_digests=contribution,
            destination_digests=destination,
            c10d_group_seq=entry.get("c10d_group_seq"),
        )
    return entries, per_rank


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
    presence = tuple(sorted(per_rank))
    if not set(presence).issubset(membership):
        raise MergeInputError(
            f"Join key {key} has presence {presence} outside membership {membership}.",
            code=MergedErrorCode.MERGED_SCHEMA_INVALID,
        )
    missing = tuple(sorted(set(membership) - set(presence)))
    kind = entries[presence[0]]["kind"]
    findings = _relation_findings(key, kind, membership, entries)
    # The two correlation cross-checks (both structural), then the gap.
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
    findings.extend(finding for finding in checks if finding is not None)
    reduce_ops = sorted(
        {
            str(entry.get("reduce_op"))
            for entry in entries.values()
            if entry.get("reduce_op") is not None
        }
    )
    consistency = _witness_consistency(kind, backend, membership, presence, per_rank)
    if consistency is BoundaryConsistency.MISMATCHED:
        findings.append(value_divergence(key, kind, presence))
    record = JoinRecord(
        key=key,
        kind=kind,
        reduce_op=reduce_ops[0] if reduce_ops else None,
        membership=membership,
        backend=backend,
        presence=presence,
        missing=missing,
        per_rank=per_rank,
        consistency=consistency,
    )
    return record, findings


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
