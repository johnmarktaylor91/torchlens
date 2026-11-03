"""Shard-local capture marker substrate (L8/F6, merge-ranks C2; lands DARK).

Pins the uncontended F6 substrate slice: the ``distributed_scope`` marker
field (S3-registered ``FieldPolicy.DROP``), the ONE sharded predicate over the
placements limb (census plan 3.3(c): setter and future load validator share
it), the 3.2b PERMANENT erasure-prevention invariant at the bundle-save
chokepoint, and the 3.2c(2) merge-scope marker key on ``RankEvidence`` /
``_guard_scope`` asserted on BOTH derive_merge entry paths (census row C5r).

DARK by construction: no capture can set the marker until the D-L8-CAP
capture relaxation (every sharded topology still refuses at entry -- Group
B/C census rows pin that), so positives here construct the marker
synthetically, exactly like the plan's other pre-relaxation reds.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed gloo unavailable",
)

import torchlens as tl  # noqa: E402
from torchlens._distributed import _lifecycle as lifecycle  # noqa: E402
from torchlens._distributed._dtensor import (  # noqa: E402
    RANK_LOCAL_SHARD,
    shard_local_placements,
    value_marks_shard_local,
)
from torchlens.distributed import (  # noqa: E402
    is_vetted_build,
)

# F1 ruling (Lead, 2026-10-01): full collective arming only runs where a
# census-vetted torch build exists (torchlens.distributed.is_vetted_build());
# on an unvetted torch, arm() fails closed with UncapturedCollectiveOpError.
# Only the two TestMergeScopeMarkerKey tests that build a real armed rank
# core need this; the rest of the module never arms.
requires_vetted_snapshot = pytest.mark.skipif(
    not is_vetted_build(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.is_vetted_build() is False here)",
)


@pytest.fixture()
def gloo_world(tmp_path):
    """Single-process gloo world, armed state guaranteed clean around a test."""

    import torch.distributed as dist

    lifecycle.disarm()
    if dist.is_initialized():
        dist.destroy_process_group()
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield dist
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()


class TestShardedPredicate:
    """The Shard / Replicate / mixed matrix, pinned explicitly (plan 3.3)."""

    def test_live_placement_objects_matrix(self, gloo_world):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard, distribute_tensor

        mesh = init_device_mesh("cpu", (1,))
        sharded = distribute_tensor(torch.randn(4, 4), mesh, [Shard(0)])
        replicated = distribute_tensor(torch.randn(4, 4), mesh, [Replicate()])
        assert shard_local_placements(sharded.placements)
        assert not shard_local_placements(replicated.placements)
        # Value-level classification is by the predicate over the value's OWN
        # placements, never by dual-geometry record presence (3.2(4b)).
        assert value_marks_shard_local(sharded)
        assert not value_marks_shard_local(replicated)  # N5's over-labeling red
        assert not value_marks_shard_local(torch.randn(3))

    def test_one_predicate_two_evaluation_times(self, gloo_world):
        """The capture-time (live objects) and load-time (repr strings) limbs
        of THE one predicate agree on every matrix cell."""

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Replicate, Shard, distribute_tensor

        mesh = init_device_mesh("cpu", (1,))
        for placements in ([Shard(0)], [Replicate()]):
            value = distribute_tensor(torch.randn(4, 4), mesh, list(placements))
            live = shard_local_placements(value.placements)
            persisted = shard_local_placements([repr(p) for p in value.placements])
            assert live == persisted


class TestMarkerFieldSubstrate:
    def test_plain_round_trip_preserves_the_marker(self, tmp_path):
        """Tamper row (a) counterpart: at tlspec v8 the marker persists on a
        PLAIN save and survives load intact (validated closed-vocabulary)."""

        lifecycle.disarm()
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        log._provenance_distributed_scope = RANK_LOCAL_SHARD
        path = tmp_path / "shard-local.tlspec"
        log.save(str(path))
        loaded = tl.load(str(path))
        assert loaded.provenance.distributed_scope == RANK_LOCAL_SHARD


class TestErasurePreventionInvariant:
    """Plan 3.2b tamper rows (b) and (e): the chokepoint refusal is permanent."""

    def _marked_trace(self):
        lifecycle.disarm()
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        log._provenance_distributed_scope = RANK_LOCAL_SHARD
        return log

    def test_row_e_forced_drop_refires_post_bump(self, tmp_path, monkeypatch):
        """The predicate keys on the ACTUAL field-policy state: forcing the
        policy back to DROP (the schema-regression simulation) re-fires the
        invariant -- the refusal was never deleted at the bump."""

        from dataclasses import replace

        from torchlens._data_classes.trace import Trace
        from torchlens._errors import InvalidArgumentError
        from torchlens._io import FieldPolicy, prerelease

        log = self._marked_trace()
        entry = Trace._FIELD_POLICY["_provenance_distributed_scope"]
        monkeypatch.setitem(
            Trace._FIELD_POLICY,
            "_provenance_distributed_scope",
            replace(entry, portable_policy=FieldPolicy.DROP),
        )
        monkeypatch.setattr(prerelease, "_ACTIVE", False)
        with pytest.raises(InvalidArgumentError) as excinfo:
            log.save(str(tmp_path / "regression.tlspec"))
        assert excinfo.value.fields["code"] == "shard_local_persistence_unsupported"


class TestMergeScopeMarkerKey:
    """Plan 3.2c(2) + census row C5r: BOTH derive_merge entry paths refuse."""

    def _rank_core_with_marker(self, dist):
        class CollectiveModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                hidden = self.fc(x)
                dist.all_reduce(hidden)
                return hidden

        lifecycle.arm()
        log = tl.trace(CollectiveModel(), torch.randn(2, 4))
        log._provenance_distributed_scope = RANK_LOCAL_SHARD
        return log

    @requires_vetted_snapshot
    def test_c5r_public_merge_entry_refuses(self, gloo_world):
        from torchlens._merged._errors import MergeInputError

        log = self._rank_core_with_marker(gloo_world)
        with pytest.raises(MergeInputError) as excinfo:
            tl.merge_ranks([log])
        assert excinfo.value.fields["code"] == "merge_scope_unsupported"
        assert excinfo.value.fields["reason"] == "shard_local_member_unsupported"
