"""r-b4 R29-2: input alias validation pre-filters pairs with an interval sweep.

``_record_unpreserved_tensor_aliases`` ran the exact ``touched_bytes_relation``
ladder on EVERY pair -- O(T^2) in tensor-leaf count (measured 0.72 s at 4k
disjoint leaves). The sweep (``_alias_candidate_pairs``) drops only pairs the
ladder provably answers ``disjoint`` from bounding intervals alone; every
surviving pair still runs the unchanged per-pair ladder, so these tests pin
BYTE-IDENTICAL diagnostics against a brute-force reference implementation.
"""

from __future__ import annotations

import pytest
import torch

from torchlens.utils.alias_footprint import tensor_byte_footprint, touched_bytes_relation
from torchlens.utils.arg_handling import (
    _alias_candidate_pairs,
    _record_unpreserved_tensor_aliases,
)


def _reference_scan(
    tensor_records: list[tuple[str, torch.Tensor, bool]],
    *,
    require_distinct_tensor_sites: bool,
) -> list[str]:
    """The historical exhaustive pairwise scan, kept verbatim as the oracle."""

    semantic_gaps: list[str] = []
    for left_index, (left_path, left, left_transparent) in enumerate(tensor_records):
        for right_path, right, right_transparent in tensor_records[left_index + 1 :]:
            if left is right:
                if require_distinct_tensor_sites:
                    semantic_gaps.append(
                        f"{left_path} <-> {right_path}: runnable input sites share "
                        "one tensor identity, which the sparse descriptor cannot encode"
                    )
                continue
            if left_transparent and right_transparent and not require_distinct_tensor_sites:
                continue
            try:
                relation = touched_bytes_relation(left, right)
            except (RuntimeError, TypeError, NotImplementedError):
                relation = "unknown"
            if relation == "disjoint":
                continue
            if require_distinct_tensor_sites:
                semantic_gaps.append(
                    f"{left_path} <-> {right_path}: runnable input sites have "
                    f"{relation} storage, which the sparse descriptor cannot encode"
                )
            else:
                semantic_gaps.append(
                    f"{left_path} <-> {right_path}: grad-preserving clones cannot prove "
                    f"the original tensor alias topology ({relation})"
                )
    return semantic_gaps


def _torture_records() -> list[tuple[str, torch.Tensor, bool]]:
    """Identity repeats, overlapping views, adjacent slices, empties, transparents."""

    base = torch.arange(32, dtype=torch.float32)
    shared = torch.ones(3)
    return [
        ("input.args.0", base[0:8], True),
        ("input.args.1", base[4:12], False),  # overlaps 0
        ("input.args.2", base[12:20], True),  # adjacent, disjoint
        ("input.args.3", base[::2], False),  # strided over everything
        ("input.args.4", shared, False),
        ("input.args.5", shared, True),  # identity repeat of 4
        ("input.args.6", torch.empty(0), False),  # empty view
        ("input.args.7", torch.ones(4), True),  # disjoint fresh storage
        ("input.args.8", torch.ones(4), True),  # transparent pair with 7
        ("input.args.9", base[1:3], False),  # inside 0's span
    ]


@pytest.mark.smoke_cells("test_alias_scan_matches_exhaustive_reference[True]")
@pytest.mark.parametrize("require_distinct", [False, True])
def test_alias_scan_matches_exhaustive_reference(require_distinct: bool) -> None:
    """The swept scan reproduces the exhaustive scan's diagnostics byte for byte."""

    records = _torture_records()
    expected = _reference_scan(records, require_distinct_tensor_sites=require_distinct)
    actual: list[str] = []
    _record_unpreserved_tensor_aliases(
        records, actual, require_distinct_tensor_sites=require_distinct
    )
    assert actual == expected
    assert expected  # the torture set must actually exercise diagnostic lanes


def test_disjoint_leaves_produce_zero_candidate_pairs() -> None:
    """Distinct-storage leaves generate NO candidate pairs: the scan is O(T log T)."""

    records = [(f"input.args.{i}", torch.ones(1), False) for i in range(512)]
    footprints = [tensor_byte_footprint(tensor) for _, tensor, _ in records]
    assert _alias_candidate_pairs(records, footprints) == set()

    gaps: list[str] = []
    _record_unpreserved_tensor_aliases(records, gaps, require_distinct_tensor_sites=True)
    assert gaps == []


@pytest.mark.smoke
def test_candidate_pairs_cover_identity_overlap_and_unknown() -> None:
    """Identity, interval-overlap, and unprovable-footprint pairs all survive."""

    base = torch.arange(16, dtype=torch.float32)
    shared = torch.ones(2)
    records = [
        ("a", base[0:8], False),
        ("b", base[4:12], False),
        ("c", shared, False),
        ("d", shared, False),
        ("e", torch.ones(3), False),
    ]
    footprints = [tensor_byte_footprint(tensor) for _, tensor, _ in records]
    pairs = _alias_candidate_pairs(records, footprints)
    assert (0, 1) in pairs  # byte-interval overlap
    assert (2, 3) in pairs  # identity repeat
    assert (0, 4) not in pairs and (1, 4) not in pairs  # fresh storage stays excluded

    # An unprovable footprint is a candidate against every partner.
    footprints[4] = None
    pairs = _alias_candidate_pairs(records, footprints)
    assert {(0, 4), (1, 4), (2, 4), (3, 4)} <= pairs
