"""Behavioral tests for slice-kind (gather/scatter/all_to_all) witness derivation.

``derive_merge``'s identity-kind witness comparison is covered by
``test_merged_engine.py``; these pin the group-rank-indexed slice pairings:
gather roots against member contributions, scatter roots against member
destinations, and the full all_to_all matrix, including mismatch detection
and honest NOT_PRESENT demotion when the root is missing.
"""

from __future__ import annotations

import pytest

from torchlens.distributed._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    membership_digest_for_ranks,
)
from torchlens.merged import (
    BoundaryConsistency,
    MergeAlignment,
    MergeValueStatus,
    derive_merge,
)
from torchlens.merged._errors import MergedFinding
from torchlens.merged._evidence import RankEvidence

WORLD = membership_digest_for_ranks([0, 1])


def _seeded_ledger() -> GroupLifecycleLedger:
    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent(0, "seed", WORLD, 0, "seeded", "seeded", 0))
    return ledger


def _role(role: str, index: int) -> dict:
    return {
        "role": role,
        "index": index,
        "shape": [2],
        "logical_shape": None,
        "placements": None,
    }


def _boundary(
    rank: int,
    *,
    kind: str,
    roles: list[dict],
    contribution_digests: list[str] | None,
    destination_digests: list[str] | None,
) -> dict:
    members = (0, 1)
    return {
        "schema": "collective_boundary_v1",
        "kind": kind,
        "func": f"torch.distributed.{kind}",
        "correlation": {
            "membership_digest": WORLD,
            "lifetime_ordinal": 0,
            "channel": "coll",
            "seq": 0,
        },
        "group": {
            "global_ranks": list(members),
            "size": len(members),
            "backend": "gloo",
            "my_global_rank": rank,
            "my_group_rank": members.index(rank),
            "coord_provenance": "test",
        },
        "reduce_op": None,
        "peer": None,
        "events": {"async_op": False, "completion_binding": "issue_sync"},
        "roles": roles,
        "witness": {
            "policy_resolved": "digest",
            "contribution_digests": contribution_digests,
            "destination_digests": destination_digests,
            "not_present_reason": None,
        },
        "lifetime_evidence": {
            "ordinal_source": "seeded",
            "install_epoch": "seeded",
            "arming_source": "explicit",
        },
        "c10d_group_seq": None,
        "disclosures": [],
        "op_labels_raw": [f"{kind}_0_raw_r{rank}"],
        "op_node": True,
    }


def _evidence(rank: int, entry: dict) -> RankEvidence:
    return RankEvidence(
        shard_local=False,
        rank=rank,
        boundaries=(entry,),
        ledger=_seeded_ledger(),
        install_epoch="seeded",
        source=f"synthetic[{rank}]",
    )


def _gather_pair(leaf_contribution: str) -> dict[int, RankEvidence]:
    root = _boundary(
        0,
        kind="gather",
        roles=[_role("contribution", 0), _role("destination", 0), _role("destination", 1)],
        contribution_digests=["c0"],
        destination_digests=["c0", "c1"],
    )
    leaf = _boundary(
        1,
        kind="gather",
        roles=[_role("contribution", 0)],
        contribution_digests=[leaf_contribution],
        destination_digests=None,
    )
    return {0: _evidence(0, root), 1: _evidence(1, leaf)}


def test_gather_root_slices_attest_against_member_contributions() -> None:
    """Each root destination slice must equal that member's contribution digest."""

    derivation = derive_merge(_gather_pair("c1"))
    assert derivation.stored_alignment is MergeAlignment.ALIGNED
    assert derivation.joins[0].consistency is BoundaryConsistency.ATTESTED


def test_gather_slice_disagreement_is_mismatched() -> None:
    """A leaf whose bytes differ from the root's slice demotes to MISMATCHED."""

    derivation = derive_merge(_gather_pair("f" * 2))
    assert derivation.joins[0].consistency is BoundaryConsistency.MISMATCHED
    # Witnesses are demote-only on the VALUE status; membership alignment
    # itself stays aligned.
    assert derivation.stored_alignment is MergeAlignment.ALIGNED
    assert derivation.stored_value_status is MergeValueStatus.DIVERGENT


@pytest.mark.smoke
def test_gather_without_a_root_demotes_to_not_present() -> None:
    """No rank holding the full destination list means the pairing cannot attest."""

    leaf_0 = _boundary(
        0,
        kind="gather",
        roles=[_role("contribution", 0)],
        contribution_digests=["c0"],
        destination_digests=None,
    )
    leaf_1 = _boundary(
        1,
        kind="gather",
        roles=[_role("contribution", 0)],
        contribution_digests=["c1"],
        destination_digests=None,
    )
    derivation = derive_merge({0: _evidence(0, leaf_0), 1: _evidence(1, leaf_1)})
    assert derivation.joins[0].consistency is BoundaryConsistency.NOT_PRESENT


@pytest.mark.smoke
def test_scatter_root_slices_attest_against_member_destinations() -> None:
    """Each member's received slice must equal the root's contribution slice."""

    root = _boundary(
        0,
        kind="scatter",
        roles=[_role("contribution", 0), _role("contribution", 1), _role("destination", 0)],
        contribution_digests=["s0", "s1"],
        destination_digests=["s0"],
    )
    leaf = _boundary(
        1,
        kind="scatter",
        roles=[_role("destination", 0)],
        contribution_digests=None,
        destination_digests=["s1"],
    )
    derivation = derive_merge({0: _evidence(0, root), 1: _evidence(1, leaf)})
    assert derivation.joins[0].consistency is BoundaryConsistency.ATTESTED

    wrong_leaf = _boundary(
        1,
        kind="scatter",
        roles=[_role("destination", 0)],
        contribution_digests=None,
        destination_digests=["e" * 2],
    )
    mismatched = derive_merge({0: _evidence(0, root), 1: _evidence(1, wrong_leaf)})
    assert mismatched.joins[0].consistency is BoundaryConsistency.MISMATCHED


def _all_to_all_entry(rank: int, contribution: list[str], destination: list[str]) -> dict:
    return _boundary(
        rank,
        kind="all_to_all",
        roles=[
            _role("contribution", 0),
            _role("contribution", 1),
            _role("destination", 0),
            _role("destination", 1),
        ],
        contribution_digests=contribution,
        destination_digests=destination,
    )


@pytest.mark.smoke
def test_all_to_all_full_matrix_attests_and_detects_one_wrong_cell() -> None:
    """dest_j[i] == contrib_i[j] over the full rank matrix; one bad cell demotes."""

    good = derive_merge(
        {
            0: _evidence(0, _all_to_all_entry(0, ["a00", "a01"], ["a00", "a10"])),
            1: _evidence(1, _all_to_all_entry(1, ["a10", "a11"], ["a01", "a11"])),
        }
    )
    assert good.joins[0].consistency is BoundaryConsistency.ATTESTED

    bad = derive_merge(
        {
            0: _evidence(0, _all_to_all_entry(0, ["a00", "a01"], ["a00", "a10"])),
            1: _evidence(1, _all_to_all_entry(1, ["a10", "a11"], ["XX", "a11"])),
        }
    )
    assert bad.joins[0].consistency is BoundaryConsistency.MISMATCHED


@pytest.mark.smoke
def test_merged_finding_payload_round_trip_preserves_key_and_ranks() -> None:
    """MergedFinding serializes to canonical payload and rebuilds identically."""

    finding = MergedFinding(
        kind="relation_violation",
        detail="example",
        membership_digest=WORLD,
        key=(WORLD, 0, "coll", 3),
        ranks=(0, 1),
    )
    payload = finding.to_payload()
    rebuilt = MergedFinding.from_payload(payload)
    assert rebuilt == finding

    keyless = MergedFinding(kind="presence_gap", detail="no key")
    assert MergedFinding.from_payload(keyless.to_payload()) == keyless
