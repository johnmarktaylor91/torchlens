"""Characterization pins for ``derive_merge`` across its derivation stages.

Each case runs the one merge derivation over synthetic rank evidence and pins
either a hash digest of the whole returned ``MergeDerivation`` (its full
``repr``, which includes every join, per-rank reference, finding, audit
verdict and dict insertion order, plus the canonical payload) or the refusal
it raises (class, message and every payload field, code included). The
expected values were recorded on the single-function implementation, so a
stage split that changes any result, error, message, code or the order in
which refusals fire fails here.

The cases cover every stage of the derivation: scope guard refusals, the
membership-authority belt, the pre-join lineage audit, the group table
(member-order and backend disputes), delta joining, relation findings, the
c10d and absolute-seq cross-checks, presence gaps, witness verdicts and their
typed refusals, declared expected ranks, the order check, and each stored
alignment and value-status verdict.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import pytest
from test_merged_engine import (
    WORLD,
    armed_ledger,
    boundary,
    digest_kwargs,
    evidence,
    seeded_ledger,
)

from torchlens._merged import derive_merge
from torchlens._merged._engine import MergeDerivation
from torchlens._merged._errors import MergedTraceError
from torchlens._merged._evidence import RankEvidence
from torchlens.distributed._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    membership_digest_for_ranks,
)

WORLD3 = membership_digest_for_ranks([0, 1, 2])

Case = tuple[dict[int, RankEvidence], Any]


def _role(name: str, index: int, shape: list[int] | None = None) -> dict[str, Any]:
    """Build one plain (non-DTensor) role entry.

    Parameters
    ----------
    name:
        Role name.
    index:
        Role index.
    shape:
        Tensor shape; defaults to ``[2]``.

    Returns
    -------
    dict[str, Any]
        The role entry.
    """

    return {
        "role": name,
        "index": index,
        "shape": [2] if shape is None else shape,
        "logical_shape": None,
        "placements": None,
    }


def _hex64(char: str) -> str:
    """Return a syntactically valid hex witness digest made of ``char``.

    Parameters
    ----------
    char:
        One hex character.

    Returns
    -------
    str
        The full-length digest.
    """

    return char * 64


def _armed(rank: int, boundaries: list[dict[str, Any]], generations: int = 1) -> RankEvidence:
    """Build evidence for a rank armed before any group.

    Parameters
    ----------
    rank:
        Global rank.
    boundaries:
        Boundary journal.
    generations:
        Group generations the complete witness evidences.

    Returns
    -------
    RankEvidence
        The rank evidence.
    """

    return evidence(
        rank,
        boundaries,
        ledger=armed_ledger(generations=generations),
        epoch="armed_before_any_group",
    )


def _two_generation_ledger() -> GroupLifecycleLedger:
    """Return a complete-witness ledger evidencing two live generations.

    Returns
    -------
    GroupLifecycleLedger
        The ledger.
    """

    ledger = GroupLifecycleLedger()
    ledger.append(
        GroupLifecycleEvent(0, "create", WORLD, 0, "wrapped", "armed_before_any_group", 0)
    )
    ledger.append(
        GroupLifecycleEvent(1, "create", WORLD, 1, "wrapped", "armed_before_any_group", 1)
    )
    return ledger


def _recreated_world_ledger() -> GroupLifecycleLedger:
    """Return a seeded ledger whose world group was destroyed and recreated.

    Returns
    -------
    GroupLifecycleLedger
        The ledger.
    """

    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent(0, "seed", WORLD, 0, "seeded", "seeded", 0))
    ledger.append(GroupLifecycleEvent(1, "destroy", WORLD, 0, "seeded", "seeded"))
    ledger.append(GroupLifecycleEvent(2, "create", WORLD, 1, "wrapped", "seeded", 1))
    return ledger


def _world3_ledger() -> GroupLifecycleLedger:
    """Return a seeded ledger for both the three-rank world and the 0/1 subgroup.

    Returns
    -------
    GroupLifecycleLedger
        The ledger.
    """

    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent(0, "seed", WORLD3, 0, "seeded", "seeded", 0))
    ledger.append(GroupLifecycleEvent(1, "seed", WORLD, 0, "seeded", "seeded", 0))
    return ledger


def _gather_pair(leaf_group_rank: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a root/leaf gather boundary pair with digest witnesses.

    Parameters
    ----------
    leaf_group_rank:
        When given, overrides the leaf's recorded group rank.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Root (rank 0) and leaf (rank 1) boundaries.
    """

    root = boundary(
        0,
        0,
        kind="gather",
        reduce_op=None,
        roles=[_role("contribution", 0), _role("destination", 0), _role("destination", 1)],
        witness_policy="digest",
        contribution_digests=[_hex64("a")],
        destination_digests=[_hex64("a"), _hex64("b")],
    )
    leaf = boundary(
        1,
        0,
        kind="gather",
        reduce_op=None,
        roles=[_role("contribution", 0)],
        witness_policy="digest",
        contribution_digests=[_hex64("b")],
    )
    if leaf_group_rank is not None:
        leaf["group"]["my_group_rank"] = leaf_group_rank
    return root, leaf


# --- success cases -----------------------------------------------------------


def _case_agree_delta_bases_attested() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 5, **digest_kwargs()), boundary(0, 6, **digest_kwargs())]),
            1: evidence(1, [boundary(1, 0, **digest_kwargs()), boundary(1, 1, **digest_kwargs())]),
        },
        None,
    )


def _case_agree_unwitnessed() -> Case:
    return ({0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])}, None)


def _case_world3_and_subgroup_channels() -> Case:
    def core(rank: int) -> RankEvidence:
        entries = [boundary(rank, 0, digest=WORLD3, members=(0, 1, 2), **digest_kwargs())]
        if rank in (0, 1):
            entries.append(boundary(rank, 0, **digest_kwargs()))
        entries.append(boundary(rank, 1, digest=WORLD3, members=(0, 1, 2), **digest_kwargs()))
        return evidence(rank, entries, ledger=_world3_ledger())

    return ({rank: core(rank) for rank in (0, 1, 2)}, [2, 1, 0, 0])


def _case_missing_key_presence_gap() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0), boundary(0, 1)]),
            1: evidence(1, [boundary(1, 0)]),
        },
        None,
    )


def _case_member_without_core() -> Case:
    return ({0: evidence(0, [boundary(0, 0, **digest_kwargs())])}, None)


def _case_expected_ranks_widen() -> Case:
    return (
        {0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])},
        (3, 1, 2, 0),
    )


def _case_expected_ranks_never_narrow() -> Case:
    return ({0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [boundary(1, 0)])}, [0])


def _case_expected_ranks_without_cores() -> Case:
    return ({}, [0, 1])


def _case_empty_evidence() -> Case:
    return ({}, None)


def _case_rank_without_boundaries() -> Case:
    return ({0: evidence(0, []), 1: evidence(1, [boundary(1, 0)])}, None)


def _case_conflicted_membership_audit() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, ordinal=1)], ledger=_recreated_world_ledger()),
            1: evidence(1, [boundary(1, 0, ordinal=0)], ledger=seeded_ledger()),
        },
        [0, 1, 2],
    )


def _case_conflicted_witness_disagreement() -> Case:
    return (
        {
            0: _armed(0, [boundary(0, 0)], generations=1),
            1: _armed(1, [boundary(1, 0, ordinal=1)], generations=2),
        },
        None,
    )


def _case_seed_discharge_mixed() -> Case:
    return (
        {
            0: _armed(0, [boundary(0, 0, **digest_kwargs())]),
            1: evidence(1, [boundary(1, 0, **digest_kwargs())], ledger=seeded_ledger()),
        },
        None,
    )


def _case_member_order_mismatch() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, members=(0, 1)), boundary(0, 1, members=(0, 1))]),
            1: evidence(1, [boundary(1, 0, members=(1, 0)), boundary(1, 1, members=(1, 0))]),
        },
        None,
    )


def _case_backend_disputed_three_ways() -> Case:
    def core(rank: int, backend: str) -> RankEvidence:
        return evidence(
            rank,
            [
                boundary(rank, 0, digest=WORLD3, members=(0, 1, 2), backend=backend),
                boundary(rank, 1, digest=WORLD3, members=(0, 1, 2), backend=backend),
            ],
            ledger=seeded_ledger(WORLD3),
        )

    return ({0: core(0, "gloo"), 1: core(1, "nccl"), 2: core(2, "mpi")}, None)


def _case_backend_unknown_not_applicable() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, backend="fancy_tpu", **digest_kwargs())]),
            1: evidence(1, [boundary(1, 0, backend="fancy_tpu", **digest_kwargs())]),
        },
        None,
    )


def _case_backend_none_not_applicable() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, backend=None, **digest_kwargs())]),
            1: evidence(1, [boundary(1, 0, backend=None, **digest_kwargs())]),
        },
        None,
    )


def _case_relation_kind_disagreement() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, kind="all_reduce")]),
            1: evidence(1, [boundary(1, 0, kind="all_gather_into_tensor", reduce_op=None)]),
        },
        None,
    )


def _case_relation_reduce_op_and_shape() -> Case:
    roles = [_role("contribution_destination", 0, [3, 3])]
    return (
        {
            0: evidence(0, [boundary(0, 0, reduce_op="RedOpType.MAX", **digest_kwargs())]),
            1: evidence(1, [boundary(1, 0, roles=roles, **digest_kwargs())]),
        },
        None,
    )


def _case_tensorless_barrier() -> Case:
    def core(rank: int) -> RankEvidence:
        entry = boundary(rank, 0, kind="barrier", reduce_op=None, roles=[])
        return evidence(rank, [entry])

    return ({0: core(0), 1: core(1)}, None)


def _case_c10d_delta_mismatch() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, c10d_group_seq=10), boundary(0, 1, c10d_group_seq=11)]),
            1: evidence(1, [boundary(1, 0, c10d_group_seq=20), boundary(1, 1, c10d_group_seq=25)]),
        },
        None,
    )


def _case_c10d_agreeing_deltas() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, c10d_group_seq=10), boundary(0, 1, c10d_group_seq=11)]),
            1: evidence(1, [boundary(1, 0, c10d_group_seq=20), boundary(1, 1, c10d_group_seq=21)]),
        },
        None,
    )


def _case_c10d_base_only_at_delta_zero() -> Case:
    # Rank 1 carries no c10d value at delta 0, so its later value has no base
    # and never enters the cross-check.
    return (
        {
            0: evidence(0, [boundary(0, 0, c10d_group_seq=10), boundary(0, 1, c10d_group_seq=11)]),
            1: evidence(
                1, [boundary(1, 0, c10d_group_seq=None), boundary(1, 1, c10d_group_seq=99)]
            ),
        },
        None,
    )


def _case_armed_absolute_seq_mismatch() -> Case:
    return (
        {
            0: _armed(0, [boundary(0, 0), boundary(0, 1)]),
            1: _armed(1, [boundary(1, 1)]),
        },
        None,
    )


def _case_armed_absolute_seq_equal() -> Case:
    return (
        {
            0: _armed(0, [boundary(0, 3), boundary(0, 4)]),
            1: _armed(1, [boundary(1, 3), boundary(1, 4)]),
        },
        None,
    )


def _case_seeded_base_offsets_never_compared() -> Case:
    return (
        {
            0: _armed(0, [boundary(0, 0)]),
            1: evidence(1, [boundary(1, 7)], ledger=seeded_ledger()),
        },
        None,
    )


def _case_witness_mismatch_divergent() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, **digest_kwargs("aa"))]),
            1: evidence(1, [boundary(1, 0, **digest_kwargs("bb"))]),
        },
        None,
    )


def _case_witness_partial() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, **digest_kwargs()), boundary(0, 1)]),
            1: evidence(1, [boundary(1, 0, **digest_kwargs()), boundary(1, 1)]),
        },
        None,
    )


def _case_gather_slice_attested() -> Case:
    root, leaf = _gather_pair()
    return ({0: evidence(0, [root]), 1: evidence(1, [leaf])}, None)


def _case_order_contradiction() -> Case:
    return (
        {
            0: evidence(
                0,
                [boundary(0, 0, ordinal=0), boundary(0, 0, ordinal=1)],
                ledger=_two_generation_ledger(),
                epoch="armed_before_any_group",
            ),
            1: evidence(
                1,
                [boundary(1, 0, ordinal=1), boundary(1, 0, ordinal=0)],
                ledger=_two_generation_ledger(),
                epoch="armed_before_any_group",
            ),
        },
        [0, 1, 3],
    )


def _case_every_finding_stage_at_once() -> Case:
    # Audit conflict on the world group, member-order and backend disputes on
    # the three-rank group, a c10d mismatch, a presence gap, a value
    # divergence and a declared absent rank in one derivation, so the
    # cross-stage finding ORDER is pinned.
    def core(rank: int) -> RankEvidence:
        members = (0, 1, 2) if rank != 2 else (2, 1, 0)
        backend = "gloo" if rank != 1 else "nccl"
        entries = [
            boundary(
                rank,
                0,
                digest=WORLD3,
                members=members,
                backend=backend,
                c10d_group_seq=rank * 10,
                **digest_kwargs("aa" if rank != 2 else "bb"),
            ),
            boundary(
                rank,
                1,
                digest=WORLD3,
                members=members,
                backend=backend,
                c10d_group_seq=rank * 10 + (1 if rank != 1 else 5),
            ),
        ]
        if rank != 2:
            entries.append(boundary(rank, 2, digest=WORLD3, members=members, backend=backend))
        if rank in (0, 1):
            entries.append(boundary(rank, 0, ordinal=1 if rank == 0 else 0))
        return evidence(rank, entries, ledger=_world3_ledger())

    cores = {rank: core(rank) for rank in (0, 1, 2)}
    world3_and_recreated = GroupLifecycleLedger()
    world3_and_recreated.append(GroupLifecycleEvent(0, "seed", WORLD3, 0, "seeded", "seeded", 0))
    world3_and_recreated.append(GroupLifecycleEvent(1, "seed", WORLD, 0, "seeded", "seeded", 0))
    world3_and_recreated.append(GroupLifecycleEvent(2, "destroy", WORLD, 0, "seeded", "seeded"))
    world3_and_recreated.append(GroupLifecycleEvent(3, "create", WORLD, 1, "wrapped", "seeded", 1))
    cores[0] = RankEvidence(
        shard_local=False,
        rank=0,
        boundaries=cores[0].boundaries,
        ledger=world3_and_recreated,
        install_epoch="seeded",
        source="synthetic[0]",
    )
    return (cores, [0, 1, 2, 4])


def _case_every_key_check_on_one_key() -> Case:
    # Ranks 0 and 1 of a three-rank group are armed before any group; rank 2
    # presents no core. Key delta 1 then carries a relation violation (reduce
    # op), a c10d mismatch, an absolute-seq mismatch, a presence gap and a
    # value divergence, so the per-key finding ORDER is pinned.
    def core(rank: int, seqs: tuple[int, int], c10d: tuple[int, int], op: str) -> RankEvidence:
        entries = [
            boundary(
                rank,
                seqs[0],
                digest=WORLD3,
                members=(0, 1, 2),
                c10d_group_seq=c10d[0],
                **digest_kwargs(),
            ),
            boundary(
                rank,
                seqs[1],
                digest=WORLD3,
                members=(0, 1, 2),
                c10d_group_seq=c10d[1],
                reduce_op=op,
                **digest_kwargs("aa" if rank == 0 else "bb"),
            ),
        ]
        return evidence(rank, entries, ledger=armed_ledger(WORLD3), epoch="armed_before_any_group")

    return (
        {
            0: core(0, (0, 1), (10, 11), "RedOpType.SUM"),
            1: core(1, (5, 6), (20, 25), "RedOpType.MAX"),
        },
        None,
    )


# --- refusal cases -----------------------------------------------------------


def _case_refuse_shard_local() -> Case:
    core = evidence(0, [boundary(0, 0)])
    shard = RankEvidence(
        shard_local=True,
        rank=0,
        boundaries=core.boundaries,
        ledger=core.ledger,
        install_epoch=core.install_epoch,
        source=core.source,
    )
    return ({0: shard, 1: evidence(1, [boundary(1, 0)])}, None)


def _case_refuse_shard_local_before_p2p_same_rank() -> Case:
    core = evidence(0, [boundary(0, 0, kind="send", channel="p2p/0->1", reduce_op=None)])
    shard = RankEvidence(
        shard_local=True,
        rank=0,
        boundaries=core.boundaries,
        ledger=core.ledger,
        install_epoch=core.install_epoch,
        source=core.source,
    )
    return ({0: shard}, None)


def _case_refuse_p2p_lower_rank_before_shard_local() -> Case:
    core = evidence(1, [boundary(1, 0)])
    shard = RankEvidence(
        shard_local=True,
        rank=1,
        boundaries=core.boundaries,
        ledger=core.ledger,
        install_epoch=core.install_epoch,
        source=core.source,
    )
    return (
        {
            1: shard,
            0: evidence(0, [boundary(0, 0, kind="send", channel="p2p/0->1", reduce_op=None)]),
        },
        None,
    )


def _case_refuse_p2p_kind() -> Case:
    return (
        {0: evidence(0, [boundary(0, 0, kind="send", channel="p2p/0->1", reduce_op=None)])},
        None,
    )


def _case_refuse_non_coll_channel() -> Case:
    return ({0: evidence(0, [boundary(0, 0), boundary(0, 1, channel="p2p/0->1")])}, None)


def _case_refuse_dtensor_geometry() -> Case:
    roles = [
        {
            "role": "contribution_destination",
            "index": 0,
            "shape": [2, 4],
            "logical_shape": [4, 4],
            "placements": ["Shard(dim=0)"],
        }
    ]
    return ({0: evidence(0, [boundary(0, 0, roles=roles)])}, None)


def _case_refuse_scope_before_authority() -> Case:
    roles = [
        {
            "role": "contribution_destination",
            "index": 0,
            "shape": [2, 4],
            "logical_shape": [4, 4],
            "placements": None,
        }
    ]
    return (
        {
            0: evidence(0, [boundary(0, 7), boundary(0, 7)]),
            1: evidence(1, [boundary(1, 0, roles=roles)]),
        },
        None,
    )


def _case_refuse_presence_outside_membership() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 0, **digest_kwargs())]),
            5: evidence(5, [boundary(5, 0, **digest_kwargs())]),
        },
        None,
    )


def _case_refuse_membership_digest_mismatch() -> Case:
    entry = boundary(1, 0)
    entry["correlation"]["membership_digest"] = WORLD3
    return ({0: evidence(0, [boundary(0, 0)]), 1: evidence(1, [entry])}, None)


def _case_refuse_duplicate_correlation_key() -> Case:
    return (
        {
            0: evidence(0, [boundary(0, 7), boundary(0, 7)]),
            1: evidence(1, [boundary(1, 3)]),
        },
        None,
    )


def _case_refuse_authority_before_audit_conflict() -> Case:
    return (
        {
            0: evidence(
                0,
                [boundary(0, 0, ordinal=1), boundary(0, 0, ordinal=1)],
                ledger=_recreated_world_ledger(),
            ),
            1: evidence(1, [boundary(1, 0, ordinal=0)], ledger=seeded_ledger()),
        },
        None,
    )


def _case_refuse_string_witness_digests() -> Case:
    entry = boundary(1, 0, witness_policy="digest")
    entry["witness"]["destination_digests"] = "aa"
    return ({0: evidence(0, [boundary(0, 0, **digest_kwargs())]), 1: evidence(1, [entry])}, None)


def _case_refuse_group_rank_out_of_range() -> Case:
    root, leaf = _gather_pair(leaf_group_rank=7)
    return ({0: evidence(0, [root]), 1: evidence(1, [leaf])}, None)


def _case_refuse_digest_belt_before_later_group_rank() -> Case:
    # The string-digest refusal sits on the sorted-first channel (ordinal 0);
    # the group-rank refusal on ordinal 1 must never be reached first.
    root, leaf = _gather_pair(leaf_group_rank=7)
    for entry in (root, leaf):
        entry["correlation"]["lifetime_ordinal"] = 1
    bad = boundary(1, 0, witness_policy="digest")
    bad["witness"]["contribution_digests"] = 3
    return (
        {
            0: evidence(0, [boundary(0, 0), root], ledger=_two_generation_ledger()),
            1: evidence(1, [leaf, bad], ledger=_two_generation_ledger()),
        },
        None,
    )


CASES: dict[str, Callable[[], Case]] = {
    "agree_delta_bases_attested": _case_agree_delta_bases_attested,
    "agree_unwitnessed": _case_agree_unwitnessed,
    "world3_and_subgroup_channels": _case_world3_and_subgroup_channels,
    "missing_key_presence_gap": _case_missing_key_presence_gap,
    "member_without_core": _case_member_without_core,
    "expected_ranks_widen": _case_expected_ranks_widen,
    "expected_ranks_never_narrow": _case_expected_ranks_never_narrow,
    "expected_ranks_without_cores": _case_expected_ranks_without_cores,
    "empty_evidence": _case_empty_evidence,
    "rank_without_boundaries": _case_rank_without_boundaries,
    "conflicted_membership_audit": _case_conflicted_membership_audit,
    "conflicted_witness_disagreement": _case_conflicted_witness_disagreement,
    "seed_discharge_mixed": _case_seed_discharge_mixed,
    "member_order_mismatch": _case_member_order_mismatch,
    "backend_disputed_three_ways": _case_backend_disputed_three_ways,
    "backend_unknown_not_applicable": _case_backend_unknown_not_applicable,
    "backend_none_not_applicable": _case_backend_none_not_applicable,
    "relation_kind_disagreement": _case_relation_kind_disagreement,
    "relation_reduce_op_and_shape": _case_relation_reduce_op_and_shape,
    "tensorless_barrier": _case_tensorless_barrier,
    "c10d_delta_mismatch": _case_c10d_delta_mismatch,
    "c10d_agreeing_deltas": _case_c10d_agreeing_deltas,
    "c10d_base_only_at_delta_zero": _case_c10d_base_only_at_delta_zero,
    "armed_absolute_seq_mismatch": _case_armed_absolute_seq_mismatch,
    "armed_absolute_seq_equal": _case_armed_absolute_seq_equal,
    "seeded_base_offsets_never_compared": _case_seeded_base_offsets_never_compared,
    "witness_mismatch_divergent": _case_witness_mismatch_divergent,
    "witness_partial": _case_witness_partial,
    "gather_slice_attested": _case_gather_slice_attested,
    "order_contradiction": _case_order_contradiction,
    "every_finding_stage_at_once": _case_every_finding_stage_at_once,
    "every_key_check_on_one_key": _case_every_key_check_on_one_key,
    "refuse_shard_local": _case_refuse_shard_local,
    "refuse_shard_local_before_p2p_same_rank": _case_refuse_shard_local_before_p2p_same_rank,
    "refuse_p2p_lower_rank_before_shard_local": _case_refuse_p2p_lower_rank_before_shard_local,
    "refuse_p2p_kind": _case_refuse_p2p_kind,
    "refuse_non_coll_channel": _case_refuse_non_coll_channel,
    "refuse_dtensor_geometry": _case_refuse_dtensor_geometry,
    "refuse_scope_before_authority": _case_refuse_scope_before_authority,
    "refuse_presence_outside_membership": _case_refuse_presence_outside_membership,
    "refuse_membership_digest_mismatch": _case_refuse_membership_digest_mismatch,
    "refuse_duplicate_correlation_key": _case_refuse_duplicate_correlation_key,
    "refuse_authority_before_audit_conflict": _case_refuse_authority_before_audit_conflict,
    "refuse_string_witness_digests": _case_refuse_string_witness_digests,
    "refuse_group_rank_out_of_range": _case_refuse_group_rank_out_of_range,
    "refuse_digest_belt_before_later_group_rank": (
        _case_refuse_digest_belt_before_later_group_rank
    ),
}


def _digest(derivation: MergeDerivation) -> str:
    """Return a stable hash digest of a whole derivation.

    Parameters
    ----------
    derivation:
        The derivation to fingerprint.

    Returns
    -------
    str
        Hex digest over the full ``repr`` and the canonical payload.
    """

    payload = json.dumps(derivation.to_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(f"{derivation!r}\n{payload}".encode()).hexdigest()


def observe(name: str) -> dict[str, Any]:
    """Run one case and return its observable outcome.

    Parameters
    ----------
    name:
        Case name in :data:`CASES`.

    Returns
    -------
    dict[str, Any]
        Either the derivation's verdicts, finding kinds and digest, or the
        refusal's class, message and payload fields.
    """

    cores, expected_ranks = CASES[name]()
    try:
        derivation = derive_merge(cores, expected_ranks)
    except MergedTraceError as exc:
        return {
            "error": type(exc).__name__,
            "message": str(exc),
            "fields": json.loads(json.dumps(dict(exc.fields), sort_keys=True, default=repr)),
        }
    return {
        "alignment": derivation.stored_alignment.value,
        "value_status": derivation.stored_value_status.value,
        "finding_kinds": [finding.kind for finding in derivation.findings],
        "num_joins": len(derivation.joins),
        "digest": _digest(derivation),
    }


EXPECTED: dict[str, dict[str, Any]] = {
    "agree_delta_bases_attested": {
        "alignment": "aligned",
        "digest": "686bc241f93012776713507e5b11ab553c5787f913ed84b07cdbd51de8c8c729",
        "finding_kinds": [],
        "num_joins": 2,
        "value_status": "attested_complete",
    },
    "agree_unwitnessed": {
        "alignment": "aligned",
        "digest": "6895be693300c3742bf43f1c0261670213e9d8860abe1e2aa589aa94e0efdd81",
        "finding_kinds": [],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "armed_absolute_seq_equal": {
        "alignment": "aligned",
        "digest": "095e671439395f705c06f5b171ed6bb6f796f7767772db784f3ba7ac9cbd22df",
        "finding_kinds": [],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "armed_absolute_seq_mismatch": {
        "alignment": "conflicted",
        "digest": "19db340bd318e3cc21df82bfe36be750884523ccde82525840eccc86a4f2a7b3",
        "finding_kinds": ["correlation_delta_mismatch", "presence_gap"],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "backend_disputed_three_ways": {
        "alignment": "conflicted",
        "digest": "3de3747163acf3676cd415a07a9d1dc0424305e0cff4034a787cbce1063c073f",
        "finding_kinds": ["relation_violation"],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "backend_none_not_applicable": {
        "alignment": "aligned",
        "digest": "143d30ea566868c60988a3ff8dd08600f9dd7c95f3534bd917aae63e4791ab68",
        "finding_kinds": [],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "backend_unknown_not_applicable": {
        "alignment": "aligned",
        "digest": "95f460a701114c6609682c4c37b9079c2a69e16b649c5704225ba5fb8712047a",
        "finding_kinds": [],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "c10d_agreeing_deltas": {
        "alignment": "aligned",
        "digest": "1fa018aceca518ceb17af3852b61057e2bf7d045e90dd790474e6931cb075db3",
        "finding_kinds": [],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "c10d_base_only_at_delta_zero": {
        "alignment": "aligned",
        "digest": "036993d66ab389ea57b2e43eb49d6b7ceb4e8ae7167316d95de9bccfdc4d66eb",
        "finding_kinds": [],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "c10d_delta_mismatch": {
        "alignment": "conflicted",
        "digest": "ea299a09256502d3fdf8bcc77ad75d28bc7bda71689626511355884f91c6573a",
        "finding_kinds": ["correlation_delta_mismatch"],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "conflicted_membership_audit": {
        "alignment": "conflicted",
        "digest": "3c2f86a0876664c05ad9a41026bd707e854884e1947a3c56a39f06280ed1606e",
        "finding_kinds": ["group_lifetime_evidence_conflict", "presence_gap"],
        "num_joins": 0,
        "value_status": "unwitnessed",
    },
    "conflicted_witness_disagreement": {
        "alignment": "conflicted",
        "digest": "96b1a4c001e0b994af619439b5a41386172cc9f35a76819ea8db592acc0f71e7",
        "finding_kinds": ["group_lifetime_evidence_conflict"],
        "num_joins": 0,
        "value_status": "unwitnessed",
    },
    "empty_evidence": {
        "alignment": "aligned",
        "digest": "1773786db19135384c535ac1270a9f45dad1450390154f1d464ed0ac569472e5",
        "finding_kinds": [],
        "num_joins": 0,
        "value_status": "unwitnessed",
    },
    "every_finding_stage_at_once": {
        "alignment": "conflicted",
        "digest": "ff591dd0b5e15e656afbdb73c9f776363a6427764409985ae3e2f9387506d1c2",
        "finding_kinds": [
            "group_lifetime_evidence_conflict",
            "relation_violation",
            "relation_violation",
            "relation_violation",
            "correlation_delta_mismatch",
            "presence_gap",
            "presence_gap",
        ],
        "num_joins": 3,
        "value_status": "unwitnessed",
    },
    "every_key_check_on_one_key": {
        "alignment": "conflicted",
        "digest": "6e58ddd33c6e5403bd1638accd91441b8877d23a6cdf40019b847061d7eb473b",
        "finding_kinds": [
            "correlation_delta_mismatch",
            "presence_gap",
            "relation_violation",
            "correlation_delta_mismatch",
            "correlation_delta_mismatch",
            "presence_gap",
            "value_divergence",
        ],
        "num_joins": 2,
        "value_status": "divergent",
    },
    "expected_ranks_never_narrow": {
        "alignment": "aligned",
        "digest": "ab86fb88c332199984a8f424988675b8a780805abaa821bf11ddd2d822507326",
        "finding_kinds": [],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "expected_ranks_widen": {
        "alignment": "partial",
        "digest": "3ee15c7e77ee7fa8561a5db586b003fd376f62704fbdd8b44f2aad2c45125d37",
        "finding_kinds": ["presence_gap"],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "expected_ranks_without_cores": {
        "alignment": "partial",
        "digest": "98bc09223fa278dc95c1add29701575423892e138ce829c725524ef920927f4b",
        "finding_kinds": ["presence_gap"],
        "num_joins": 0,
        "value_status": "unwitnessed",
    },
    "gather_slice_attested": {
        "alignment": "aligned",
        "digest": "5a7c3fabec403e62b20cab4a96272df12d1a25957307ca43fcf75684a7eda221",
        "finding_kinds": [],
        "num_joins": 1,
        "value_status": "attested_complete",
    },
    "member_order_mismatch": {
        "alignment": "conflicted",
        "digest": "4b9a73db6781e82475d04bb748d1086412cecf511714280db30f98bd390baddb",
        "finding_kinds": ["relation_violation", "relation_violation"],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "member_without_core": {
        "alignment": "partial",
        "digest": "a917b532ca25da5d76e25b771071c979c9443695e2505faa3a741a7d5be30dd2",
        "finding_kinds": ["presence_gap"],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "missing_key_presence_gap": {
        "alignment": "partial",
        "digest": "7857e9794945bf441edf8e64e5cc13064d56c029c27943774f334a6874d3771d",
        "finding_kinds": ["presence_gap"],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "order_contradiction": {
        "alignment": "conflicted",
        "digest": "1ea201c73b3cd4494f484ef2db458eb0a08c6dc2838e386f7d4021d3f3c5b3a6",
        "finding_kinds": ["presence_gap", "order_contradiction"],
        "num_joins": 2,
        "value_status": "unwitnessed",
    },
    "rank_without_boundaries": {
        "alignment": "partial",
        "digest": "fae102d3a81ae0bb8ab1eeb232f746970e17273ce79c8b9a5d1fd94b3f53acf2",
        "finding_kinds": ["presence_gap"],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "refuse_authority_before_audit_conflict": {
        "error": "MergeInputError",
        "fields": {"code": "merged_schema_invalid", "findings": [], "rank": 0},
        "message": "Rank 0 presents duplicate correlation key ('83b97b859aa5f81b2f0f86ba2a675efaf515ad2d5e2b8652cf2de7e1c2267350', 1, 'coll', 0); rank-local issue sequence values must be unique.",
    },
    "refuse_digest_belt_before_later_group_rank": {
        "error": "MergeInputError",
        "fields": {"code": "merged_schema_invalid", "findings": []},
        "message": "Boundary witness contribution_digests is not a list of digests (got int).",
    },
    "refuse_dtensor_geometry": {
        "error": "MergeInputError",
        "fields": {
            "code": "merge_scope_unsupported",
            "findings": [],
            "kind": "all_reduce",
            "rank": 0,
        },
        "message": "Rank 0 boundary 0 carries DTensor dual geometry; sharded-topology merging is rung C2 and stays refused until its capture-fidelity census is green.",
    },
    "refuse_duplicate_correlation_key": {
        "error": "MergeInputError",
        "fields": {"code": "merged_schema_invalid", "findings": [], "rank": 0},
        "message": "Rank 0 presents duplicate correlation key ('83b97b859aa5f81b2f0f86ba2a675efaf515ad2d5e2b8652cf2de7e1c2267350', 0, 'coll', 7); rank-local issue sequence values must be unique.",
    },
    "refuse_group_rank_out_of_range": {
        "error": "MergeInputError",
        "fields": {"code": "merged_schema_invalid", "findings": []},
        "message": "Boundary group rank 7 for rank 1 is outside its recorded group of 2 member(s).",
    },
    "refuse_membership_digest_mismatch": {
        "error": "MergeInputError",
        "fields": {"code": "merged_schema_invalid", "findings": [], "rank": 1},
        "message": "Rank 1 presents a boundary whose membership_digest does not equal the digest of its own recorded membership (0, 1).",
    },
    "refuse_non_coll_channel": {
        "error": "MergeInputError",
        "fields": {
            "channel": "p2p/0->1",
            "code": "merge_scope_unsupported",
            "findings": [],
            "kind": "all_reduce",
            "rank": 0,
        },
        "message": "Rank 0 boundary 1 is a point-to-point boundary (kind='all_reduce', channel='p2p/0->1'). C1 merges symmetric collectives only; p2p/pipeline pairing is rung C3.",
    },
    "refuse_p2p_kind": {
        "error": "MergeInputError",
        "fields": {
            "channel": "p2p/0->1",
            "code": "merge_scope_unsupported",
            "findings": [],
            "kind": "send",
            "rank": 0,
        },
        "message": "Rank 0 boundary 0 is a point-to-point boundary (kind='send', channel='p2p/0->1'). C1 merges symmetric collectives only; p2p/pipeline pairing is rung C3.",
    },
    "refuse_p2p_lower_rank_before_shard_local": {
        "error": "MergeInputError",
        "fields": {
            "channel": "p2p/0->1",
            "code": "merge_scope_unsupported",
            "findings": [],
            "kind": "send",
            "rank": 0,
        },
        "message": "Rank 0 boundary 0 is a point-to-point boundary (kind='send', channel='p2p/0->1'). C1 merges symmetric collectives only; p2p/pipeline pairing is rung C3.",
    },
    "refuse_presence_outside_membership": {
        "error": "MergeInputError",
        "fields": {"code": "merged_schema_invalid", "findings": [], "rank": 5},
        "message": "Rank 5 presents a collective boundary for membership (0, 1), but presence must be a subset of the recorded membership.",
    },
    "refuse_scope_before_authority": {
        "error": "MergeInputError",
        "fields": {
            "code": "merge_scope_unsupported",
            "findings": [],
            "kind": "all_reduce",
            "rank": 1,
        },
        "message": "Rank 1 boundary 0 carries DTensor dual geometry; sharded-topology merging is rung C2 and stays refused until its capture-fidelity census is green.",
    },
    "refuse_shard_local": {
        "error": "MergeInputError",
        "fields": {
            "code": "merge_scope_unsupported",
            "findings": [],
            "rank": 0,
            "reason": "shard_local_member_unsupported",
        },
        "message": "Rank 0 is a shard-local capture (distributed_scope == 'rank_local_shard'). Sharded-topology merging stays refused pending its own authorizing ruling plus the merged-side tiling-proof deliverable; the geometry-independent marker key refuses it at merge scope.",
    },
    "refuse_shard_local_before_p2p_same_rank": {
        "error": "MergeInputError",
        "fields": {
            "code": "merge_scope_unsupported",
            "findings": [],
            "rank": 0,
            "reason": "shard_local_member_unsupported",
        },
        "message": "Rank 0 is a shard-local capture (distributed_scope == 'rank_local_shard'). Sharded-topology merging stays refused pending its own authorizing ruling plus the merged-side tiling-proof deliverable; the geometry-independent marker key refuses it at merge scope.",
    },
    "refuse_string_witness_digests": {
        "error": "MergeInputError",
        "fields": {"code": "merged_schema_invalid", "findings": []},
        "message": "Boundary witness destination_digests is not a list of digests (got str).",
    },
    "relation_kind_disagreement": {
        "alignment": "conflicted",
        "digest": "a4f29346ad29331d09efae174f59c5f61e8fc631ec0ad5faeb028a38d412b6d1",
        "finding_kinds": ["relation_violation"],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "relation_reduce_op_and_shape": {
        "alignment": "conflicted",
        "digest": "bd6ce8a1f214b55e6ee78e4a0c887b8873a3a0e7b9a96237aa167521ca6944c7",
        "finding_kinds": ["relation_violation", "relation_violation"],
        "num_joins": 1,
        "value_status": "attested_complete",
    },
    "seed_discharge_mixed": {
        "alignment": "aligned",
        "digest": "1811569de1fba0e5d74e2b5677c51fbb0d0b038211c48fea1034a3b6ba1be628",
        "finding_kinds": [],
        "num_joins": 1,
        "value_status": "attested_complete",
    },
    "seeded_base_offsets_never_compared": {
        "alignment": "aligned",
        "digest": "32ab7501a334a3ccb21438c92c3de89ae6527aaa5b452fa1aac8ff3a713f923c",
        "finding_kinds": [],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "tensorless_barrier": {
        "alignment": "aligned",
        "digest": "94b1b854a2329814453033e974f01b269568e624b6f6e53db0bd12d243b36738",
        "finding_kinds": [],
        "num_joins": 1,
        "value_status": "unwitnessed",
    },
    "witness_mismatch_divergent": {
        "alignment": "aligned",
        "digest": "674369f3b483c5a6acb08280a7a9f8bec483dd00eb7402fcd03500724041991b",
        "finding_kinds": ["value_divergence"],
        "num_joins": 1,
        "value_status": "divergent",
    },
    "witness_partial": {
        "alignment": "aligned",
        "digest": "622942e182512c0ebe2648e34357a340e41b754c3529a1330cb5b948c8d3c008",
        "finding_kinds": [],
        "num_joins": 2,
        "value_status": "attested_partial",
    },
    "world3_and_subgroup_channels": {
        "alignment": "aligned",
        "digest": "77cf83c9fc1b97a84c91396076d3f2942b4c82f90b7274adce57a2ebc830211c",
        "finding_kinds": [],
        "num_joins": 3,
        "value_status": "attested_complete",
    },
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_derive_merge_characterization(name: str) -> None:
    """Each case's result or refusal equals the recorded pre-split value."""

    assert observe(name) == EXPECTED[name]


def test_characterization_table_is_complete() -> None:
    """Every case has a recorded expectation and no expectation is orphaned."""

    assert sorted(EXPECTED) == sorted(CASES)


def test_derive_merge_is_deterministic_across_runs() -> None:
    """Two runs over freshly built inputs return equal whole objects."""

    for name, build in CASES.items():
        if "error" in EXPECTED.get(name, {}):
            continue
        first = derive_merge(*build())
        second = derive_merge(*build())
        assert first == second, name
        assert repr(first) == repr(second), name
