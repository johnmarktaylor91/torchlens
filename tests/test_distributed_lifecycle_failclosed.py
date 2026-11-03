"""Fail-closed group-lifecycle bookkeeping (round-3 R18 / SF-09 class).

Two fail-open holes are pinned shut here:

1. The arm-time group-history probe stamped ``armed_before_any_group`` -- the
   COMPLETE-WITNESS epoch whose lineage vector overrides other ranks' evidence
   in the merge-time audit -- whenever the probe itself failed or the private
   registry was unreadable. Unprovable must demote to ``seeded``, never
   promote to the strongest claim.

2. A wrapped creation/destroy whose membership could not be enumerated was
   SILENTLY dropped from the ledger, so restricted seeding still blessed
   generation-0 claims the rank cannot prove, and later same-membership
   ordinals could collide with the dropped generation. An enumeration gap now
   discloses loudly, mints no identities, and refuses restricted seeding
   typed.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed gloo unavailable",
)

from torchlens._distributed import _lifecycle as lifecycle  # noqa: E402
from torchlens._distributed._lifecycle import AmbiguousGroupLifetimeError  # noqa: E402
from torchlens.distributed import is_vetted_build  # noqa: E402

# F1 ruling (Lead, 2026-10-01): full collective arming only runs where a
# census-vetted torch build exists (torchlens.distributed.is_vetted_build());
# on an unvetted torch, arm() fails closed with UncapturedCollectiveOpError.
# Only the two classes below call a real arm() expecting success;
# TestAutoArmProbeDisclosure exercises the lazy auto-arm probe-failure path,
# which is unrelated to vetting and must keep running everywhere.
requires_vetted_snapshot = pytest.mark.skipif(
    not is_vetted_build(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.is_vetted_build() is False here)",
)


@pytest.fixture()
def clean_lifecycle():
    """Guarantee unarmed lifecycle state around a test, no world required."""

    lifecycle.disarm()
    try:
        yield
    finally:
        lifecycle.disarm()


@pytest.fixture()
def gloo_world(tmp_path, clean_lifecycle):
    """Single-process gloo world with guaranteed teardown."""

    import torch.distributed as dist

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


@requires_vetted_snapshot
class TestArmEpochProbeFailClosed:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of distributed/_lifecycle.py.
    def test_probe_exception_stamps_seeded_not_complete_witness(self, clean_lifecycle, monkeypatch):
        def raising_is_initialized() -> bool:
            raise RuntimeError("torch probe blew up")

        monkeypatch.setattr(torch.distributed, "is_initialized", raising_is_initialized)
        record = lifecycle.arm()
        # Unprovable history must DEMOTE the witness claim: "seeded" says the
        # ordinals need the merge-time audit, "armed_before_any_group" would
        # forge complete-witness authority out of a probe failure.
        assert record.install_epoch == "seeded"


@requires_vetted_snapshot
class TestEnumerationGapsFailClosed:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of distributed/_lifecycle.py.
    def test_unenumerable_creation_poisons_identities_and_seeding(self, gloo_world, monkeypatch):
        dist = gloo_world
        lifecycle.arm()

        def raising_ranks(group):
            raise RuntimeError("membership unreadable")

        monkeypatch.setattr(torch.distributed, "get_process_group_ranks", raising_ranks)
        with pytest.warns(UserWarning, match="could not be enumerated"):
            extra = dist.new_group([0])
        state = lifecycle.armed_state()
        assert state is not None
        assert state.ledger_gaps, "the dropped creation must be recorded as a gap"
        assert id(extra) not in state.identities, "no identity may be minted on a gap"
        # The gapped group itself refuses typed (previously the raw torch
        # error leaked out of the seeding path mid-capture).
        with pytest.raises(AmbiguousGroupLifetimeError):
            lifecycle.resolve_group_identity(extra)
        monkeypatch.undo()
        # Restricted seeding is disabled rank-wide while a gap is open: the
        # dropped event may have been a generation of ANY membership, so even
        # the world group's generation-0 claim is unprovable.
        with pytest.raises(AmbiguousGroupLifetimeError, match="could not be enumerated"):
            lifecycle.resolve_group_identity(None)

    # Kept by shrink-r4 (per-id subset rule): sole coverage of distributed/_lifecycle.py.
    def test_unenumerable_destroy_records_a_gap(self, gloo_world, monkeypatch):
        dist = gloo_world
        # Created BEFORE arming: the group is never identified, so its destroy
        # takes the churn-recording branch that needs membership enumeration.
        pre_arm_group = dist.new_group([0])
        lifecycle.arm()

        def raising_ranks(group):
            raise RuntimeError("membership unreadable")

        monkeypatch.setattr(torch.distributed, "get_process_group_ranks", raising_ranks)
        with pytest.warns(UserWarning, match="could not be enumerated"):
            dist.destroy_process_group(pre_arm_group)
        state = lifecycle.armed_state()
        assert state is not None
        assert state.ledger_gaps, "the dropped destroy must be recorded as a gap"

    # Kept by shrink-r4 (per-id subset rule): sole coverage of distributed/_lifecycle.py.
    def test_unreadable_registry_refuses_seeding(self, gloo_world, monkeypatch):
        lifecycle.arm()
        real_world = torch.distributed.distributed_c10d._world

        class WorldWithoutPgMap:
            """Registry proxy whose pg_map is unreadable (private-API drift)."""

            def __getattr__(self, name):
                if name == "pg_map":
                    raise AttributeError(name)
                return getattr(real_world, name)

        monkeypatch.setattr(torch.distributed.distributed_c10d, "_world", WorldWithoutPgMap())
        # Seeding needs the same-membership ALIVE count; an unreadable
        # registry is not a provable zero.
        with pytest.raises(AmbiguousGroupLifetimeError, match="cannot be read"):
            lifecycle.resolve_group_identity(None)


class TestAutoArmProbeDisclosure:
    # Kept by shrink-r4 (per-id subset rule): sole coverage of distributed/_lifecycle.py.
    def test_probe_exception_warns_instead_of_silent_skip(self, clean_lifecycle, monkeypatch):
        def raising_is_initialized() -> bool:
            raise RuntimeError("torch probe blew up")

        monkeypatch.setattr(torch.distributed, "is_initialized", raising_is_initialized)
        monkeypatch.setattr(lifecycle, "_AUTO_ARM_DEGRADATION", None)
        with pytest.warns(UserWarning, match="could not probe torch.distributed"):
            assert lifecycle.maybe_auto_arm() is None
