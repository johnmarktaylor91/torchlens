"""The totalized per-join witness verdict (design-merge-ranks-c v5, 3.3; demote-only).

:func:`torchlens._merged._engine._join_key` calls :func:`witness_consistency`
last among a key's checks. Witness digests are redundant byte-exact evidence:
they can only DEMOTE a verdict to ``mismatched`` or leave it ``not_present``,
never rescue or repair one. This module reads the per-rank join references
only, so it never imports the engine (the engine imports it).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from ._enums import (
    WITNESS_IDENTITY_KINDS,
    WITNESS_NOT_APPLICABLE_KINDS,
    WITNESS_VERDICT_BACKENDS,
    BoundaryConsistency,
    MergedErrorCode,
)
from ._errors import MergeInputError
from ._rules import group_rank_in_range

if TYPE_CHECKING:
    from ._engine import PerRankRef

DigestPairs = list[tuple[str, str]]
"""``(expected, observed)`` digest pairs; any unequal pair is a mismatch."""


def witness_consistency(
    kind: str,
    backend: str | None,
    membership: tuple[int, ...],
    presence: tuple[int, ...],
    per_rank: Mapping[int, PerRankRef],
) -> BoundaryConsistency:
    """Derive one join's witness verdict from the presenting ranks' digests.

    Parameters
    ----------
    kind:
        The collective kind at the key.
    backend:
        The group's recorded backend, or ``None`` when unknown or disputed.
    membership:
        The group's recorded members.
    presence:
        The ranks presenting the key, sorted.
    per_rank:
        The per-rank join references.

    Returns
    -------
    BoundaryConsistency
        ``not_applicable`` for kinds and backends without witness semantics;
        otherwise ``mismatched``, ``not_present`` or ``attested``.

    Raises
    ------
    MergeInputError
        On a group rank outside its recorded group (a belt for internal
        ``RankEvidence`` constructors that bypass the parse boundary).
    """

    if kind in WITNESS_NOT_APPLICABLE_KINDS or backend not in WITNESS_VERDICT_BACKENDS:
        return BoundaryConsistency.NOT_APPLICABLE
    group_size = len(membership)
    if kind in WITNESS_IDENTITY_KINDS:
        return _identity_verdict(presence, per_rank, group_size)
    by_group_rank, indexed_all = _index_by_group_rank(presence, per_rank, group_size)
    incomplete = len(presence) < group_size or not indexed_all
    if kind in ("gather", "scatter"):
        pairs, paired_all = _root_leaf_pairs(kind, presence, per_rank, by_group_rank, group_size)
        expected_pairs = group_size
    else:  # all_to_all
        pairs, paired_all = _all_to_all_pairs(per_rank, by_group_rank)
        expected_pairs = group_size * group_size
    return _pairs_verdict(pairs, expected_pairs, incomplete or not paired_all)


def _identity_verdict(
    presence: tuple[int, ...], per_rank: Mapping[int, PerRankRef], group_size: int
) -> BoundaryConsistency:
    """Verdict for the identity kinds: every member rank ends holding identical bytes.

    The comparable digest is the rank's DESTINATION digest; a broadcast root
    has no destination role and its contribution bytes ARE the final bytes. A
    rank with destination roles but absent destination digests (witness level
    "none", async completion unobserved) contributes nothing: its
    pre-collective contribution digest is never a substitute (that comparison
    could only fabricate a mismatch).
    """

    values: list[tuple[str, ...]] = []
    for rank in presence:
        ref = per_rank[rank]
        value = ref.destination_digests if ref.n_destination_roles else ref.contribution_digests
        if value:
            values.append(tuple(value))
    if len(set(values)) > 1:
        return BoundaryConsistency.MISMATCHED
    if len(values) < group_size:
        return BoundaryConsistency.NOT_PRESENT
    return BoundaryConsistency.ATTESTED


def _index_by_group_rank(
    presence: tuple[int, ...], per_rank: Mapping[int, PerRankRef], group_size: int
) -> tuple[dict[int, int], bool]:
    """Map each presenting rank's group rank to the rank, for the slice kinds.

    Returns
    -------
    tuple[dict[int, int], bool]
        ``group_rank -> rank`` for the ranks that recorded a group rank, and
        whether every presenting rank did.

    Raises
    ------
    MergeInputError
        On a group rank outside ``0 <= group_rank < group_size``. Parse
        already pins ``my_group_rank`` to the membership position (R18-2);
        this belt keeps a permuted or out-of-range value from reaching the
        slice pairing below as a raw ``IndexError`` or a silently rebound
        pairing.
    """

    by_group_rank: dict[int, int] = {}
    for rank in presence:
        group_rank = per_rank[rank].group_rank
        if group_rank is None:
            continue
        if not group_rank_in_range(group_rank, group_size):
            raise MergeInputError(
                f"Boundary group rank {group_rank!r} for rank {rank} is "
                f"outside its recorded group of {group_size} member(s).",
                code=MergedErrorCode.MERGED_SCHEMA_INVALID,
            )
        by_group_rank[group_rank] = rank
    return by_group_rank, len(by_group_rank) == len(presence)


def _root_leaf_pairs(
    kind: str,
    presence: tuple[int, ...],
    per_rank: Mapping[int, PerRankRef],
    by_group_rank: Mapping[int, int],
    group_size: int,
) -> tuple[DigestPairs, bool]:
    """Pair the root's list slices with the leaves' single digests.

    gather: the root's destination list slice ``i`` equals member ``i``'s
    contribution. scatter: member ``i``'s destination equals the root's
    contribution list slice ``i``.

    Returns
    -------
    tuple[DigestPairs, bool]
        The digest pairs and whether every slice could be paired (exactly one
        root holding a full-length list, every leaf holding a digest).
    """

    is_gather = kind == "gather"
    leaf_attr = "contribution_digests" if is_gather else "destination_digests"
    root_list = _root_list(presence, per_rank, group_size, is_gather)
    if root_list is None:
        return [], False
    pairs: DigestPairs = []
    paired_all = True
    for group_rank in sorted(by_group_rank):
        leaf = getattr(per_rank[by_group_rank[group_rank]], leaf_attr)
        if not leaf:
            paired_all = False
            continue
        pairs.append((root_list[group_rank], leaf[0]))
    return pairs, paired_all


def _root_list(
    presence: tuple[int, ...],
    per_rank: Mapping[int, PerRankRef],
    group_size: int,
    is_gather: bool,
) -> tuple[str, ...] | None:
    """Return the unique root's full-length digest list, or ``None`` when there is none."""

    n_list_attr = "n_destination_roles" if is_gather else "n_contribution_roles"
    root_list_attr = "destination_digests" if is_gather else "contribution_digests"
    roots = [rank for rank in presence if getattr(per_rank[rank], n_list_attr) == group_size]
    if len(roots) != 1:
        return None
    root_list = getattr(per_rank[roots[0]], root_list_attr)
    if not root_list or len(root_list) != group_size:
        return None
    return tuple(root_list)


def _all_to_all_pairs(
    per_rank: Mapping[int, PerRankRef], by_group_rank: Mapping[int, int]
) -> tuple[DigestPairs, bool]:
    """Pair ``rank j``'s destination ``[i]`` with ``rank i``'s contribution ``[j]``.

    Returns
    -------
    tuple[DigestPairs, bool]
        The digest pairs and whether every ``(j, i)`` slot could be paired.
    """

    pairs: DigestPairs = []
    paired_all = True
    for gr_j in sorted(by_group_rank):
        dest_j = per_rank[by_group_rank[gr_j]].destination_digests
        if not dest_j:
            paired_all = False
            continue
        for gr_i in sorted(by_group_rank):
            contrib_i = per_rank[by_group_rank[gr_i]].contribution_digests
            if not contrib_i or gr_i >= len(dest_j) or gr_j >= len(contrib_i):
                paired_all = False
                continue
            pairs.append((dest_j[gr_i], contrib_i[gr_j]))
    return pairs, paired_all


def _pairs_verdict(
    pairs: DigestPairs, expected_pairs: int, incomplete: bool
) -> BoundaryConsistency:
    """Totalize the slice pairs: any unequal pair demotes, any gap leaves it unwitnessed."""

    if any(left != right for left, right in pairs):
        return BoundaryConsistency.MISMATCHED
    if incomplete or len(pairs) < expected_pairs:
        return BoundaryConsistency.NOT_PRESENT
    return BoundaryConsistency.ATTESTED
