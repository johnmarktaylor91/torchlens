"""The per-kind relation table the merge derivation checks at each join key.

:func:`torchlens._merged._engine._join_key` runs :func:`_relation_findings`
first among a key's checks (design-merge-ranks-c v5, 1.5). It reads only the presenting
ranks' boundary records, so this module, like ``_derive_steps``, never
imports the engine. The role vocabulary and selectors it shares with the
evidence validator live in ``_rules``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from ._errors import MergedFinding
from ._rules import (
    ALL_ROLES,
    CONTRIBUTION_ROLES,
    DESTINATION_ROLES,
    empty_for_tensor_kind,
    role_shapes,
    roles_of,
)

Entries = dict[int, dict[str, Any]]
"""The presenting ranks' boundary records at one key, by rank."""

_SYMMETRIC_SHAPE_KINDS: Final = frozenset(
    {"all_reduce", "all_to_all_single", "all_gather_into_tensor", "broadcast"}
)
"""Kinds whose contribution (else destination) shapes agree on every rank."""

_DISTINCT_SHAPE_KINDS: Final = frozenset(
    {"all_gather", "gather", "scatter", "reduce", "reduce_scatter", "reduce_scatter_tensor"}
)
"""Kinds whose SET of distinct role shapes agrees on every rank (R18-1 iii).

Every role of these kinds shares one element geometry across the group
(all_gather/gather/scatter slots equal the contribution shape; reduce inputs
equal outputs; reduce_scatter feeds group_size output-shaped slots;
reduce_scatter_tensor's two shapes repeat on every rank), so the set of
distinct role shapes must agree across ranks even though per-rank role COUNTS
legitimately differ (root list vs leaf tensor). all_to_all stays out: uneven
splits make per-rank shape sets legitimately rank-local. A UNIFORM cross-rank
shape rewrite still passes this belt: coherent reauthoring, the documented
out-of-scope boundary (only a per-core cross-check against the member's own
op records could catch it, and parse never dereferences the trace).
"""

_ROOTED_KINDS: Final = frozenset({"broadcast", "reduce", "gather", "scatter"})
"""Kinds where at most one rank may present the root's role signature (1.5)."""


def _relation_findings(
    key: tuple[str, int, str, int],
    kind: str,
    membership: tuple[int, ...],
    entries: Entries,
) -> list[MergedFinding]:
    """Evaluate the per-kind relation table over the presenting entries (1.5)."""

    ranks = tuple(sorted(entries))
    return [
        MergedFinding(
            kind="relation_violation",
            detail=f"{kind} at key {key}: {detail}",
            membership_digest=key[0],
            key=key,
            ranks=ranks,
        )
        for detail in _relation_violations(kind, len(membership), entries)
    ]


def _relation_violations(kind: str, group_size: int, entries: Entries) -> list[str]:
    """Return the relation violations at one key, in the table's order.

    A kind disagreement is reported alone: no other row of the table is
    defined across kinds.
    """

    kinds_seen = {entry["kind"] for entry in entries.values()}
    if len(kinds_seen) > 1:
        return [f"ranks disagree on the collective kind: {sorted(kinds_seen)}"]
    details = _role_presence_violations(kind, entries)
    details.extend(_reduce_op_violations(entries))
    details.extend(_shape_violations(kind, entries))
    details.extend(_cardinality_violations(kind, group_size, entries))
    details.extend(_root_violations(kind, group_size, entries))
    return details


def _role_presence_violations(kind: str, entries: Entries) -> list[str]:
    """Name each rank presenting zero tensor roles for a tensor-carrying kind.

    Role cardinality vs kind (b6-opus-R18-1 belt): the set-of-shapes agreement
    is vacuously satisfied when EVERY presenting rank holds zero roles, so
    uniform roles deletion rendered the TOP verdict. Evidence parse refuses
    that shape outright through the same rule; this belt keeps direct engine
    callers honest.
    """

    return [
        f"rank {rank} presents zero tensor roles for a tensor-carrying boundary"
        for rank, entry in sorted(entries.items())
        if empty_for_tensor_kind(kind, roles_of(entry, ALL_ROLES))
    ]


def _reduce_op_violations(entries: Entries) -> list[str]:
    """Report ranks that disagree on the reduce op."""

    reduce_ops = {entry.get("reduce_op") for entry in entries.values()}
    if len(reduce_ops) > 1:
        return [f"ranks disagree on the reduce op: {sorted(map(str, reduce_ops))}"]
    return []


def _shape_violations(kind: str, entries: Entries) -> list[str]:
    """Report cross-rank shape disagreement for the symmetric and distinct-shape kinds."""

    details: list[str] = []
    if kind in _SYMMETRIC_SHAPE_KINDS:
        shape_sets = {
            role_shapes(entry, CONTRIBUTION_ROLES) or role_shapes(entry, DESTINATION_ROLES)
            for entry in entries.values()
        }
        if len(shape_sets) > 1:
            details.append(f"tensor shapes disagree across ranks: {sorted(shape_sets)}")
    if kind in _DISTINCT_SHAPE_KINDS:
        distinct_shape_sets = {
            frozenset(role_shapes(entry, ALL_ROLES))
            for entry in entries.values()
            if entry.get("roles")
        }
        if len(distinct_shape_sets) > 1:
            details.append(
                "distinct role shapes disagree across ranks: "
                f"{sorted(sorted(s) for s in distinct_shape_sets)}"
            )
    return details


def _cardinality_violations(kind: str, group_size: int, entries: Entries) -> list[str]:
    """Report per-rank role counts that do not match the group size."""

    if kind == "all_gather":
        return _all_gather_cardinality(group_size, entries)
    if kind == "all_to_all":
        return _all_to_all_cardinality(group_size, entries)
    return []


def _all_gather_cardinality(group_size: int, entries: Entries) -> list[str]:
    """all_gather: every rank records one destination per member."""

    details: list[str] = []
    for rank, entry in sorted(entries.items()):
        n_destinations = len(roles_of(entry, DESTINATION_ROLES))
        if n_destinations != group_size:
            details.append(
                f"rank {rank} records {n_destinations} destination(s) for a group of {group_size}"
            )
    return details


def _all_to_all_cardinality(group_size: int, entries: Entries) -> list[str]:
    """all_to_all: every rank records one contribution and one destination per member."""

    details: list[str] = []
    for rank, entry in sorted(entries.items()):
        n_in = len(roles_of(entry, CONTRIBUTION_ROLES))
        n_out = len(roles_of(entry, DESTINATION_ROLES))
        if n_in != group_size or n_out != group_size:
            details.append(
                f"rank {rank} records {n_in} contribution(s) and {n_out} "
                f"destination(s) for a group of {group_size}"
            )
    return details


def _root_violations(kind: str, group_size: int, entries: Entries) -> list[str]:
    """Root-only aggregate presence (1.5): at most one rank presents the root's roles."""

    if kind not in _ROOTED_KINDS:
        return []
    roots = _root_ranks(kind, group_size, entries)
    if len(roots) > 1:
        return [f"multiple ranks {roots} present root-only aggregate roles"]
    return []


def _root_ranks(kind: str, group_size: int, entries: Entries) -> list[int]:
    """Return the ranks presenting ``kind``'s root role signature, in rank order."""

    is_root = _root_signature(kind, group_size)
    return [rank for rank, entry in sorted(entries.items()) if is_root(entry)]


def _root_signature(kind: str, group_size: int) -> Callable[[dict[str, Any]], bool]:
    """Return the predicate that recognises ``kind``'s root from one rank's roles.

    A broadcast root has no destination role; a reduce root has one. gather
    roots hold a destination LIST and scatter roots a contribution LIST
    (group_size entries; leaves hold one tensor).
    """

    if kind == "broadcast":
        return lambda entry: not roles_of(entry, ("destination",))
    if kind == "reduce":
        return lambda entry: bool(roles_of(entry, DESTINATION_ROLES))
    list_roles = DESTINATION_ROLES if kind == "gather" else CONTRIBUTION_ROLES
    return lambda entry: group_size > 1 and len(roles_of(entry, list_roles)) == group_size
