"""The per-kind relation table the merge derivation checks at each join key.

:func:`torchlens._merged._engine._join_key` runs :func:`_relation_findings`
first among a key's checks (design-merge-ranks-c v5, 1.5). It reads only the presenting
ranks' boundary records, so this module, like ``_derive_steps``, never
imports the engine.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._enums import TENSORLESS_KINDS
from ._errors import MergedFinding


def _roles_of(entry: dict[str, Any], role_names: Iterable[str]) -> list[dict[str, Any]]:
    """Return the ordered role entries matching ``role_names``."""

    wanted = set(role_names)
    return [role for role in entry.get("roles", ()) if role.get("role") in wanted]


def _relation_findings(
    key: tuple[str, int, str, int],
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
