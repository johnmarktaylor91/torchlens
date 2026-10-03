"""Provocations for documented refusal codes with no prior test coverage (R25-6).

Two families were documented in the public vocabulary but never provoked by any
test, so a code swap or a dead raise site would have failed zero tests:

- ``RecordBindingError`` doors (``record_not_bound``,
  ``trace_reference_collected``): record views read after their owning Trace is
  unreachable.
- ``MergedErrorCode.MERGE_CONFLICT``: the non-lifetime structural-conflict
  branch of ``tl.merge_ranks`` (its lifetime sibling
  ``GROUP_LIFETIME_EVIDENCE_CONFLICT`` was already exercised elsewhere).

Every provocation goes through the production raise site; nothing monkeypatches
the raising code.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import RecordBindingError
from torchlens.distributed._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    membership_digest_for_ranks,
)
from torchlens.merged import MergedErrorCode
from torchlens.merged._errors import MergeConflictError


class _TwoLayer(nn.Module):
    """Two-submodule model so module-call records have real children."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


def test_layer_read_after_trace_collection_refuses_trace_reference_collected() -> None:
    """A Layer read whose owning Trace was garbage-collected refuses typed."""

    trace = tl.trace(_TwoLayer(), torch.randn(2, 4))
    layer = trace["linear_1_1"]

    del trace
    gc.collect()

    with pytest.raises(RecordBindingError) as exc_info:
        _ = layer.source_trace

    assert exc_info.value.fields["code"] == "trace_reference_collected"
    assert exc_info.value.fields["remedy"]


def test_module_call_read_without_trace_refuses_record_not_bound() -> None:
    """A ModuleCall accessor read with no bound Trace refuses ``record_not_bound``."""

    trace = tl.trace(_TwoLayer(), torch.randn(2, 4))
    call = next(iter(trace.module_calls.values()))
    call._source_trace = None

    with pytest.raises(RecordBindingError) as exc_info:
        _ = call.module

    assert exc_info.value.fields["code"] == "record_not_bound"
    assert exc_info.value.fields["remedy"]


# --- MERGE_CONFLICT (non-lifetime structural branch) -------------------------

_WORLD = membership_digest_for_ranks([0, 1])


def _seeded_ledger() -> GroupLifecycleLedger:
    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent(0, "seed", _WORLD, 0, "seeded", "seeded", 0))
    return ledger


def _boundary(rank: int, *, kind: str, reduce_op: str | None) -> dict:
    return {
        "schema": "collective_boundary_v1",
        "kind": kind,
        "func": f"torch.distributed.{kind}",
        "correlation": {
            "membership_digest": _WORLD,
            "lifetime_ordinal": 0,
            "channel": "coll",
            "seq": 0,
        },
        "group": {
            "global_ranks": [0, 1],
            "size": 2,
            "backend": "gloo",
            "my_global_rank": rank,
            "my_group_rank": rank,
            "coord_provenance": "test",
        },
        "reduce_op": reduce_op,
        "peer": None,
        "events": {"async_op": False, "completion_binding": "issue_sync"},
        "roles": [
            {
                "role": "contribution_destination",
                "index": 0,
                "shape": [2, 4],
                "logical_shape": None,
                "placements": None,
            }
        ],
        "witness": {
            "policy_resolved": "none",
            "contribution_digests": None,
            "destination_digests": None,
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


def _rank_trace(rank: int, *, kind: str, reduce_op: str | None) -> SimpleNamespace:
    """Minimal trace surface consumed by rank-evidence extraction."""

    return SimpleNamespace(
        annotations={
            "distributed": {
                "boundaries": [_boundary(rank, kind=kind, reduce_op=reduce_op)],
                "group_lifecycle_ledger": _seeded_ledger().to_payload(),
                "install_epoch": "seeded",
            }
        }
    )


@pytest.mark.smoke
def test_structural_kind_conflict_raises_merge_conflict_code() -> None:
    """Contradicting collective kinds at one joined key refuse ``merge_conflict``.

    This is the NON-lifetime branch of the ``merge_ranks`` conflict raise: the
    structural findings are relation violations, not group-lifetime evidence
    conflicts, so the code must be ``MERGE_CONFLICT`` (never the lifetime
    sibling) and the findings must ride on ``fields['findings']``.
    """

    with pytest.raises(MergeConflictError) as exc_info:
        tl.merge_ranks(
            [
                _rank_trace(0, kind="all_reduce", reduce_op="RedOpType.SUM"),
                _rank_trace(1, kind="broadcast", reduce_op=None),
            ]
        )

    assert exc_info.value.fields["code"] == MergedErrorCode.MERGE_CONFLICT.value
    findings = exc_info.value.fields["findings"]
    assert findings
    assert any(f.kind != "group_lifetime_evidence_conflict" for f in findings)
