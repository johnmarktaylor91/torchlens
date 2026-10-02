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

import dataclasses

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed gloo unavailable",
)

import torchlens as tl  # noqa: E402
from torchlens.distributed import (  # noqa: E402
    _lifecycle as lifecycle,
    has_vetted_snapshot,
)
from torchlens.distributed._dtensor import (  # noqa: E402
    RANK_LOCAL_SHARD,
    shard_local_placements,
    value_marks_shard_local,
)

# F1 ruling (Lead, 2026-10-01): full collective arming only runs where a
# census-vetted torch build exists (torchlens.distributed.has_vetted_snapshot());
# on an unvetted torch, arm() fails closed with UncapturedCollectiveOpError.
# Only the two TestMergeScopeMarkerKey tests that build a real armed rank
# core need this; the rest of the module never arms.
requires_vetted_snapshot = pytest.mark.skipif(
    not has_vetted_snapshot(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.has_vetted_snapshot() is False here)",
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

    def test_string_placements_matrix(self):
        assert shard_local_placements(["Shard(dim=0)"])
        assert shard_local_placements(["Replicate()", "Shard(dim=1)"])
        assert not shard_local_placements(["Replicate()"])
        assert not shard_local_placements(["Replicate()", "Replicate()"])
        assert not shard_local_placements([])
        assert not shard_local_placements(None)

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
    def test_field_declared_ordered_and_persisting(self):
        from torchlens._io import FieldPolicy
        from torchlens.constants import MODEL_LOG_FIELD_ORDER
        from torchlens.data_classes.trace import Trace

        assert "distributed_scope" in MODEL_LOG_FIELD_ORDER
        entry = Trace.FIELD_POLICY["distributed_scope"]
        # tlspec v8: the marker persists (the erasure guard's second conjunct
        # goes false by construction; see TestErasurePreventionInvariant).
        assert getattr(entry, "portable_policy", entry) is FieldPolicy.KEEP

    def test_plain_capture_never_carries_the_marker(self):
        lifecycle.disarm()
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        assert log.distributed_scope is None

    def test_plain_round_trip_preserves_the_marker(self, tmp_path):
        """Tamper row (a) counterpart: at tlspec v8 the marker persists on a
        PLAIN save and survives load intact (validated closed-vocabulary)."""

        lifecycle.disarm()
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        log.distributed_scope = RANK_LOCAL_SHARD
        path = tmp_path / "shard-local.tlspec"
        tl.save(log, str(path))
        loaded = tl.load(str(path))
        assert loaded.distributed_scope == RANK_LOCAL_SHARD


class TestErasurePreventionInvariant:
    """Plan 3.2b tamper rows (b) and (e): the chokepoint refusal is permanent."""

    def _marked_trace(self):
        lifecycle.disarm()
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        log.distributed_scope = RANK_LOCAL_SHARD
        return log

    def test_row_b_marked_save_proceeds_and_round_trips_at_v8(self, tmp_path):
        """tlspec v8 persists the marker, so the guard's second conjunct is
        false by construction and ordinary saves proceed with the disclosure
        intact across the round trip -- the marker-free-artifact class stays
        EMPTY through persistence rather than through refusal."""

        log = self._marked_trace()
        for level in ("portable", "audit"):
            path = tmp_path / f"marked-{level}.tlspec"
            tl.save(log, str(path), level=level)
            assert tl.load(str(path)).distributed_scope == RANK_LOCAL_SHARD

    def test_row_e_forced_drop_refires_post_bump(self, tmp_path, monkeypatch):
        """The predicate keys on the ACTUAL field-policy state: forcing the
        policy back to DROP (the schema-regression simulation) re-fires the
        invariant -- the refusal was never deleted at the bump."""

        from dataclasses import replace

        from torchlens._errors import InvalidArgumentError
        from torchlens._io import FieldPolicy, prerelease
        from torchlens.data_classes.trace import Trace

        log = self._marked_trace()
        entry = Trace.FIELD_POLICY["distributed_scope"]
        monkeypatch.setitem(
            Trace.FIELD_POLICY,
            "distributed_scope",
            replace(entry, portable_policy=FieldPolicy.DROP),
        )
        monkeypatch.setattr(prerelease, "_ACTIVE", False)
        with pytest.raises(InvalidArgumentError) as excinfo:
            tl.save(log, str(tmp_path / "regression.tlspec"))
        assert excinfo.value.fields["code"] == "shard_local_persistence_unsupported"

    def test_unmarked_trace_saves_normally(self, tmp_path):
        lifecycle.disarm()
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        tl.save(log, str(tmp_path / "plain.tlspec"))
        assert tl.load(str(tmp_path / "plain.tlspec")).distributed_scope is None


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
        log.distributed_scope = RANK_LOCAL_SHARD
        return log

    @requires_vetted_snapshot
    def test_c5r_public_merge_entry_refuses(self, gloo_world):
        from torchlens.merged._errors import MergeInputError

        log = self._rank_core_with_marker(gloo_world)
        with pytest.raises(MergeInputError) as excinfo:
            tl.merge_ranks([log])
        assert excinfo.value.fields["code"] == "merge_scope_unsupported"
        assert excinfo.value.fields["reason"] == "shard_local_member_unsupported"

    @requires_vetted_snapshot
    def test_c5r_direct_rank_evidence_construction_refuses(self, gloo_world):
        """The membership-authority precedent: the marker key runs INSIDE
        derive_merge, so directly constructed RankEvidence refuses too."""

        from torchlens.merged._engine import derive_merge
        from torchlens.merged._errors import MergeInputError
        from torchlens.merged._evidence import extract_rank_evidence

        log = self._rank_core_with_marker(gloo_world)
        evidence = extract_rank_evidence(log, "live[0]")
        assert evidence.shard_local is True
        with pytest.raises(MergeInputError) as excinfo:
            derive_merge({0: evidence})
        assert excinfo.value.fields["reason"] == "shard_local_member_unsupported"
        # The geometry key STAYS beside the marker key: an evidence carrier
        # without the marker keeps merging (shipped behavior untouched).
        clean = dataclasses.replace(evidence, shard_local=False)
        derivation = derive_merge({0: clean})
        assert derivation is not None

    def test_marker_field_is_required_not_defaulted(self):
        """Direct construction must consciously supply shard_local."""

        from torchlens.merged._evidence import RankEvidence

        with pytest.raises(TypeError):
            RankEvidence(  # type: ignore[call-arg]
                rank=0,
                boundaries=(),
                ledger=None,
                install_epoch="seeded",
                source="live",
            )
