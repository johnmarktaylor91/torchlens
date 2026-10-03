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

from torchlens.distributed import _lifecycle as lifecycle, has_vetted_snapshot  # noqa: E402
from torchlens.distributed._lifecycle import AmbiguousGroupLifetimeError  # noqa: E402

# F1 ruling (Lead, 2026-10-01): full collective arming only runs where a
# census-vetted torch build exists (torchlens.distributed.has_vetted_snapshot());
# on an unvetted torch, arm() fails closed with UncapturedCollectiveOpError.
# Only the two classes below call a real arm() expecting success;
# TestAutoArmProbeDisclosure exercises the lazy auto-arm probe-failure path,
# which is unrelated to vetting and must keep running everywhere.
requires_vetted_snapshot = pytest.mark.skipif(
    not has_vetted_snapshot(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.has_vetted_snapshot() is False here)",
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
    def test_probe_exception_stamps_seeded_not_complete_witness(self, clean_lifecycle, monkeypatch):
        def raising_is_initialized() -> bool:
            raise RuntimeError("torch probe blew up")

        monkeypatch.setattr(torch.distributed, "is_initialized", raising_is_initialized)
        record = lifecycle.arm()
        # Unprovable history must DEMOTE the witness claim: "seeded" says the
        # ordinals need the merge-time audit, "armed_before_any_group" would
        # forge complete-witness authority out of a probe failure.
        assert record.install_epoch == "seeded"

    def test_unreadable_registry_stamps_seeded(self, clean_lifecycle, monkeypatch):
        class NoPgMap:
            """Registry holder without a readable pg_map (private-API drift)."""

        monkeypatch.setattr(torch.distributed.distributed_c10d, "_world", NoPgMap())
        record = lifecycle.arm()
        assert record.install_epoch == "seeded"

    def test_clean_empty_registry_still_proves_the_negative(self, clean_lifecycle):
        record = lifecycle.arm()
        assert record.install_epoch == "armed_before_any_group"


@requires_vetted_snapshot
class TestEnumerationGapsFailClosed:
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
    def test_probe_exception_warns_instead_of_silent_skip(self, clean_lifecycle, monkeypatch):
        def raising_is_initialized() -> bool:
            raise RuntimeError("torch probe blew up")

        monkeypatch.setattr(torch.distributed, "is_initialized", raising_is_initialized)
        monkeypatch.setattr(lifecycle, "_AUTO_ARM_DEGRADATION", None)
        with pytest.warns(UserWarning, match="could not probe torch.distributed"):
            assert lifecycle.maybe_auto_arm() is None

    def test_probe_exception_warns_on_every_capture_entry(self, clean_lifecycle, monkeypatch):
        """p5 T-DISTRIBUTED: the disclosure must not latch away after capture 1.

        Fail-before: ``_AUTO_ARM_WARNED`` latched once per process, so every
        capture after the first silently skipped arming -- collectives
        silently omitted with ZERO disclosure for the rest of the process.
        """

        def raising_is_initialized() -> bool:
            raise RuntimeError("torch probe blew up")

        monkeypatch.setattr(torch.distributed, "is_initialized", raising_is_initialized)
        monkeypatch.setattr(lifecycle, "_AUTO_ARM_DEGRADATION", None)
        with pytest.warns(UserWarning, match="could not probe torch.distributed"):
            assert lifecycle.maybe_auto_arm() is None
        # The SECOND degraded capture entry warns again.
        with pytest.warns(UserWarning, match="could not probe torch.distributed"):
            assert lifecycle.maybe_auto_arm() is None

    def test_degradation_reason_is_introspectable_and_clears(self, clean_lifecycle, monkeypatch):
        """The reason survives as in-band state, not just a stderr line."""

        def raising_is_initialized() -> bool:
            raise RuntimeError("torch probe blew up")

        monkeypatch.setattr(torch.distributed, "is_initialized", raising_is_initialized)
        monkeypatch.setattr(lifecycle, "_AUTO_ARM_DEGRADATION", None)
        with pytest.warns(UserWarning):
            lifecycle.maybe_auto_arm()
        reason = lifecycle.auto_arm_degradation()
        assert reason is not None and "probe_failed" in reason
        # A later clean probe (distributed provably not in play) clears it.
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        assert lifecycle.maybe_auto_arm() is None
        assert lifecycle.auto_arm_degradation() is None
