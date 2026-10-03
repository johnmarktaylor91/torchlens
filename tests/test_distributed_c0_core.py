"""C0 core pins: lifecycle ledger, pre-join lineage audit, recognizer, arming.

These are the single-process pins for the merge-ranks C0 evidence layer
(design-merge-ranks-c v5, sections 1.3 and 5.0/5.2). The multiprocess gloo
sims for boundary capture live in ``test_distributed_boundary_gloo.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not torch.distributed.is_available(), reason="torch.distributed unavailable"
    ),
]

from torchlens.distributed import (  # noqa: E402
    AmbiguousGroupLifetimeError,
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    UncapturedCollectiveOpError,
    _lifecycle as lifecycle,  # noqa: E402
    _recognizer as recognizer_mod,  # noqa: E402
    audit_membership_lineages,
    derive_collective_recognizer,
    has_vetted_snapshot,
    membership_digest_for_ranks,
)

# ---------------------------------------------------------------------------
# ledger helpers
# ---------------------------------------------------------------------------


def _event(index, kind, digest, ordinal, source, epoch, **kwargs):
    return GroupLifecycleEvent(
        event_index=index,
        kind=kind,
        membership_digest=digest,
        ordinal=ordinal,
        ordinal_source=source,
        install_epoch=epoch,
        **kwargs,
    )


def _ledger(entries):
    """entries: list of (kind, digest, ordinal, source, epoch)."""

    ledger = GroupLifecycleLedger()
    for index, (kind, digest, ordinal, source, epoch) in enumerate(entries):
        ledger.append(_event(index, kind, digest, ordinal, source, epoch))
    return ledger


M = membership_digest_for_ranks([0, 1])
M2 = membership_digest_for_ranks([0, 1, 2, 3])

# F1 ruling (Lead, 2026-10-01): full arming only runs where a census-vetted
# torch build exists; on an unvetted torch the typed fail-closed refusal is
# the correct behavior and is asserted directly instead (see
# TestCollectiveRecognizer.test_derivation_refuses_typed_when_not_vetted and
# tests/test_distributed_boundary_gloo.py::TestUnvettedTorchRefusesArming).
requires_vetted_snapshot = pytest.mark.skipif(
    not has_vetted_snapshot(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.has_vetted_snapshot() is False here)",
)


class TestMembershipDigest:
    def test_order_insensitive_and_deterministic(self):
        assert membership_digest_for_ranks([1, 0]) == membership_digest_for_ranks([0, 1])
        assert membership_digest_for_ranks([0, 1]) != membership_digest_for_ranks([0, 2])
        assert len(M) == 64


class TestGroupLifecycleLedger:
    def test_ordinals_are_ever_created_never_reused(self):
        ledger = _ledger(
            [
                ("create", M, 0, "wrapped", "armed_before_any_group"),
                ("destroy", M, 0, "wrapped", "armed_before_any_group"),
            ]
        )
        # A destroyed generation retires its ordinal; the next create advances.
        assert ledger.next_ordinal(M) == 1

    def test_lineage_vectors_carry_sources_and_destroy_marks(self):
        ledger = _ledger(
            [
                ("seed", M, 0, "seeded", "seeded"),
                ("destroy", M, 0, "seeded", "seeded"),
                ("create", M, 1, "wrapped", "seeded"),
                ("create", M2, 0, "wrapped", "seeded"),
            ]
        )
        vectors = ledger.lineage_vectors()
        vector = vectors[M]
        assert [(e.ordinal, e.source, e.destroyed) for e in vector.entries] == [
            (0, "seeded", True),
            (1, "wrapped", False),
        ]
        assert vector.install_epoch == "seeded"
        assert vectors[M2].generations_created == 1

    def test_payload_round_trip(self):
        ledger = _ledger(
            [
                ("create", M, 0, "wrapped", "armed_before_any_group"),
                ("destroy", M, 0, "wrapped", "armed_before_any_group"),
            ]
        )
        rebuilt = GroupLifecycleLedger.from_payload(ledger.to_payload())
        assert rebuilt.events == ledger.events
        assert rebuilt.lineage_vectors() == ledger.lineage_vectors()

    def test_event_index_must_increase(self):
        ledger = _ledger([("create", M, 0, "wrapped", "seeded")])
        with pytest.raises(ValueError):
            ledger.append(_event(0, "create", M2, 0, "wrapped", "seeded"))


class TestLifecycleFailureAtomicity:
    """Lifecycle uncertainty and teardown failures remain visible and repairable."""

    def test_alive_membership_scan_failure_refuses_seeding(self, monkeypatch) -> None:
        """An unreadable live group cannot lower the ambiguity count."""

        target = object()
        unreadable = object()
        fake_world = SimpleNamespace(pg_map={unreadable: object()})
        fake_dist = SimpleNamespace(distributed_c10d=SimpleNamespace(_world=fake_world))
        monkeypatch.setattr(torch, "distributed", fake_dist)

        def ranks_for(group: object) -> tuple[int, ...]:
            """Resolve only the target group's membership."""

            if group is target:
                return (0, 1)
            raise RuntimeError("registry membership unavailable")

        monkeypatch.setattr(lifecycle, "_group_global_ranks", ranks_for)
        state = lifecycle._ArmedState(
            arming=lifecycle.ArmingRecord("seeded", "test", "explicit"),
            recognizer=object(),
        )
        with pytest.raises(AmbiguousGroupLifetimeError, match="registry"):
            lifecycle._seed_group_locked(state, target)
        assert state.ledger.events == ()

    def test_disarm_retains_state_when_restore_fails(self, monkeypatch) -> None:
        """A failed restore keeps the original-function ledger available for retry."""

        original = object()

        def installed_wrap() -> object:
            """Live TorchLens wrap over ``original`` (so restore is attempted)."""

        installed_wrap.__tl_distributed_wrap__ = True
        installed_wrap.__wrapped__ = original

        class RefusingModule:
            """Module-like object that rejects restoration of one attribute."""

            def __setattr__(self, name: str, value: object) -> None:
                """Reject the pristine function while allowing setup values."""

                if name == "all_reduce" and value is original:
                    raise RuntimeError("restore refused")
                object.__setattr__(self, name, value)

        module = RefusingModule()
        module.all_reduce = installed_wrap
        state = lifecycle._ArmedState(
            arming=lifecycle.ArmingRecord("seeded", "test", "explicit"),
            recognizer=object(),
            originals={(module, "all_reduce"): original},
        )
        monkeypatch.setattr(lifecycle, "_STATE", state)
        with pytest.raises(RuntimeError, match="restore refused"):
            lifecycle.disarm()
        assert lifecycle.armed_state() is state
        assert state.originals[(module, "all_reduce")] is original

    def test_destroy_is_recorded_only_after_delegate_succeeds(self, monkeypatch) -> None:
        """A failed destroy call must leave the live-group ledger unchanged."""

        group = object()

        def failing_destroy(group_arg: object = None) -> None:
            """Simulate a backend destroy failure."""

            _ = group_arg
            raise RuntimeError("destroy failed")

        class PatchModule:
            """Hashable module-like holder for the destroy function."""

        module = PatchModule()
        module.destroy_process_group = failing_destroy
        monkeypatch.setattr(lifecycle, "_patch_modules", lambda: [module])
        state = lifecycle._ArmedState(
            arming=lifecycle.ArmingRecord("seeded", "test", "explicit"),
            recognizer=object(),
            identities={id(group): lifecycle.GroupIdentity(M, 0, "seeded", (0, 1), "gloo")},
        )
        lifecycle._install_lifecycle_wraps(state)
        with pytest.raises(RuntimeError, match="destroy failed"):
            module.destroy_process_group(group)
        assert state.ledger.events == ()
        assert id(group) in state.identities


class TestPreJoinLineageAudit:
    """The v5 1.3 audit matrix, including sol's round-4 repro shape."""

    def test_asymmetric_arming_conflicts_structurally(self):
        # The review's repro: rank 0 armed early (epoch seeded) -- seeds g0, observes
        # destroy, wraps g1. Rank 1 armed after g0's destruction -- registry
        # shows one live group, restricted seeding legitimately fires at 0.
        rank0 = _ledger(
            [
                ("seed", M, 0, "seeded", "seeded"),
                ("destroy", M, 0, "seeded", "seeded"),
                ("create", M, 1, "wrapped", "seeded"),
            ]
        )
        rank1 = _ledger([("seed", M, 0, "seeded", "seeded")])
        verdicts = audit_membership_lineages({0: rank0, 1: rank1})
        verdict = verdicts[M]
        assert verdict.is_conflict
        assert verdict.kind == "group_lifetime_evidence_conflict"
        assert verdict.complete_witness_ranks == ()
        # Structural: the audit names the conflict; it never renders gaps.
        assert "presence" not in verdict.detail.lower()

    def test_seed_discharges_against_one_generation_complete_witness(self):
        rank0 = _ledger([("create", M, 0, "wrapped", "armed_before_any_group")])
        rank1 = _ledger([("seed", M, 0, "seeded", "seeded")])
        verdicts = audit_membership_lineages({0: rank0, 1: rank1})
        assert verdicts[M].status == "compatible"
        assert verdicts[M].complete_witness_ranks == (0,)

    def test_seed_refused_when_witness_shows_two_generations(self):
        rank0 = _ledger(
            [
                ("create", M, 0, "wrapped", "armed_before_any_group"),
                ("destroy", M, 0, "wrapped", "armed_before_any_group"),
                ("create", M, 1, "wrapped", "armed_before_any_group"),
            ]
        )
        rank1 = _ledger([("seed", M, 0, "seeded", "seeded")])
        verdicts = audit_membership_lineages({0: rank0, 1: rank1})
        assert verdicts[M].is_conflict
        assert verdicts[M].kind == "group_lifetime_evidence_conflict"

    def test_armed_churn_with_identical_vectors_joins(self):
        entries = [
            ("create", M, 0, "wrapped", "armed_before_any_group"),
            ("destroy", M, 0, "wrapped", "armed_before_any_group"),
            ("create", M, 1, "wrapped", "armed_before_any_group"),
        ]
        verdicts = audit_membership_lineages({0: _ledger(entries), 1: _ledger(entries)})
        assert verdicts[M].status == "compatible"

    def test_spmd_lazy_symmetric_seeding_joins(self):
        entries = [("seed", M, 0, "seeded", "seeded")]
        verdicts = audit_membership_lineages({0: _ledger(entries), 1: _ledger(entries)})
        assert verdicts[M].status == "compatible"

    def test_complete_witness_disagreement_is_evidence_corruption(self):
        rank0 = _ledger([("create", M, 0, "wrapped", "armed_before_any_group")])
        rank1 = _ledger(
            [
                ("create", M, 0, "wrapped", "armed_before_any_group"),
                ("destroy", M, 0, "wrapped", "armed_before_any_group"),
                ("create", M, 1, "wrapped", "armed_before_any_group"),
            ]
        )
        verdicts = audit_membership_lineages({0: rank0, 1: rank1})
        assert verdicts[M].is_conflict
        assert "corruption" in verdicts[M].detail

    def test_no_witness_asymmetric_vectors_conflict(self):
        rank0 = _ledger(
            [
                ("seed", M, 0, "seeded", "seeded"),
                ("create", M, 1, "wrapped", "seeded"),
            ]
        )
        rank1 = _ledger([("seed", M, 0, "seeded", "seeded")])
        verdicts = audit_membership_lineages({0: rank0, 1: rank1})
        assert verdicts[M].is_conflict

    def test_single_rank_membership_not_audited(self):
        verdicts = audit_membership_lineages(
            {0: _ledger([("create", M, 0, "wrapped", "armed_before_any_group")])}
        )
        assert M not in verdicts

    def test_destroy_mark_disagreement_conflicts(self):
        rank0 = _ledger(
            [
                ("create", M, 0, "wrapped", "armed_before_any_group"),
                ("destroy", M, 0, "wrapped", "armed_before_any_group"),
            ]
        )
        rank1 = _ledger([("seed", M, 0, "seeded", "seeded")])
        verdicts = audit_membership_lineages({0: rank0, 1: rank1})
        assert verdicts[M].is_conflict


class TestCollectiveRecognizer:
    @requires_vetted_snapshot
    def test_derivation_matches_vetted_snapshot_on_pinned_torch(self):
        recognizer = derive_collective_recognizer()
        assert recognizer.snapshot_name
        assert recognizer.classify("c10d::allreduce_") == "collective"
        assert recognizer.classify("_c10d_functional::wait_tensor") == "collective"
        assert recognizer.classify("_dtensor::shard_dim_alltoall") == "collective"
        assert recognizer.classify("c10d::not_a_real_op") == "unknown_collective"
        assert recognizer.classify("aten::mm") is None

    @pytest.mark.skipif(
        has_vetted_snapshot(),
        reason="the unvetted-torch refusal is only observable without a census match",
    )
    def test_derivation_refuses_typed_when_not_vetted(self):
        # F1 (Lead ruling, 2026-10-01): on a torch build with no censused
        # snapshot, derive_collective_recognizer() -- and therefore
        # torchlens.distributed.arm() -- must fail closed rather than silently
        # arming an uncensused dispatcher. This is the correct product
        # behavior, proven here against the REAL (untampered) runtime census,
        # not a synthetic mismatch.
        assert not has_vetted_snapshot()
        with pytest.raises(UncapturedCollectiveOpError) as excinfo:
            derive_collective_recognizer()
        assert excinfo.value.fields["kind"] == "uncaptured_collective_op"
        assert excinfo.value.fields["layer"] in (0, 1)

    def test_layer1_set_inequality_refuses_typed(self, monkeypatch):
        vetted = dict(recognizer_mod.VETTED_NAMESPACE_SNAPSHOTS[0][1])
        vetted["c10d"] = vetted["c10d"] - {"allreduce_"}
        monkeypatch.setattr(
            recognizer_mod,
            "VETTED_NAMESPACE_SNAPSHOTS",
            (("tampered", vetted),),
        )
        with pytest.raises(UncapturedCollectiveOpError) as excinfo:
            derive_collective_recognizer()
        assert excinfo.value.fields["kind"] == "uncaptured_collective_op"
        assert excinfo.value.fields["layer"] == 1
        mismatches = excinfo.value.fields["mismatches"]["tampered"]
        assert "allreduce_" in mismatches["c10d"]["added"]

    def test_layer2_c10d_typed_schema_outside_five_refuses_typed(self, monkeypatch):
        # This test is about the layer-2 scan mechanism, not the layer-1
        # census or the real torch dispatcher's actual SymmetricMemory ops
        # (which this torch build may or may not even have): synthesize a
        # tiny closed dispatcher schema list so the test is independent of
        # both VETTED_NAMESPACE_SNAPSHOTS and the running torch version.
        fake_schemas = [
            SimpleNamespace(name="c10d::allreduce_", arguments=[], returns=[]),
            SimpleNamespace(
                name="symm_mem::fake_op",
                arguments=[SimpleNamespace(type="__torch__.torch.classes.c10d.SymmetricMemory")],
                returns=[],
            ),
        ]
        monkeypatch.setattr(recognizer_mod, "_all_dispatcher_schemas", lambda: fake_schemas)
        monkeypatch.setattr(
            recognizer_mod,
            "VETTED_NAMESPACE_SNAPSHOTS",
            (
                (
                    "fake",
                    {
                        "c10d": frozenset({"allreduce_"}),
                        "_c10d_functional": frozenset(),
                        "_c10d_functional_autograd": frozenset(),
                        "c10d_functional": frozenset(),
                        "_dtensor": frozenset(),
                    },
                ),
            ),
        )
        # SymmetricMemory is deliberately OUTSIDE the three-type rule; widening
        # the marker list to include it proves the layer-2 scan fires on
        # dispatcher contents outside the five namespaces.
        monkeypatch.setattr(
            recognizer_mod,
            "_LAYER2_TYPE_MARKERS",
            (".c10d.SymmetricMemory",),
        )
        with pytest.raises(UncapturedCollectiveOpError) as excinfo:
            derive_collective_recognizer()
        assert excinfo.value.fields["kind"] == "uncaptured_collective_op"
        assert excinfo.value.fields["layer"] == 2
        assert any(name.startswith("symm_mem::") for name in excinfo.value.fields["offending_ops"])


@pytest.fixture()
def gloo_world(tmp_path):
    """Single-process gloo world with guaranteed teardown."""

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    lifecycle.disarm()
    store = dist.FileStore(str(tmp_path / "store"), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)
    try:
        yield dist
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture()
def unarmed(tmp_path):
    """Guaranteed-unarmed, uninitialized state around a test."""

    import torch.distributed as dist

    lifecycle.disarm()
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        yield dist
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()


@requires_vetted_snapshot
class TestArmingAndSeeding:
    def test_arm_before_any_group_is_complete_witness(self, unarmed, tmp_path):
        dist = unarmed
        record = lifecycle.arm()
        assert record.install_epoch == "armed_before_any_group"
        assert lifecycle.is_armed()
        # arm() is idempotent.
        assert lifecycle.arm() == record
        store = dist.FileStore(str(tmp_path / "store"), 1)
        dist.init_process_group("gloo", store=store, rank=0, world_size=1)
        state = lifecycle.armed_state()
        vectors = state.ledger.lineage_vectors()
        digest = membership_digest_for_ranks([0])
        assert [(e.ordinal, e.source) for e in vectors[digest].entries] == [(0, "wrapped")]
        identity = lifecycle.resolve_group_identity(None)
        assert identity.group_uid == (digest, 0)
        assert identity.ordinal_source == "wrapped"

    def test_arm_after_init_seeds_world_at_ordinal_zero(self, gloo_world):
        record = lifecycle.arm()
        assert record.install_epoch == "seeded"
        identity = lifecycle.resolve_group_identity(None)
        digest = membership_digest_for_ranks([0])
        assert identity.group_uid == (digest, 0)
        assert identity.ordinal_source == "seeded"
        state = lifecycle.armed_state()
        events = [e for e in state.ledger.events if e.membership_digest == digest]
        assert [e.kind for e in events] == ["seed"]

    def test_two_alive_same_membership_groups_refuse_seeding(self, gloo_world):
        dist = gloo_world
        # A subgroup [0] has the same membership as the world in a 1-proc
        # world; created BEFORE arming, so neither is wrap-observed.
        dist.new_group(ranks=[0])
        lifecycle.arm()
        with pytest.raises(AmbiguousGroupLifetimeError) as excinfo:
            lifecycle.resolve_group_identity(None)
        assert excinfo.value.fields["kind"] == "ambiguous_group_lifetime"

    def test_observed_churn_refuses_late_seeding(self, gloo_world):
        dist = gloo_world
        lifecycle.arm()
        # Wrapped create + destroy of membership {0} (a subgroup), then a
        # NEW pre-arm-style group of the same membership cannot seed. Emulate
        # by resolving the WORLD group (same membership {0}) after churn.
        subgroup = dist.new_group(ranks=[0])
        dist.destroy_process_group(subgroup)
        with pytest.raises(AmbiguousGroupLifetimeError):
            lifecycle.resolve_group_identity(None)

    def test_wrapped_recreation_advances_ordinal(self, gloo_world):
        dist = gloo_world
        lifecycle.arm()
        digest = membership_digest_for_ranks([0])
        first = dist.new_group(ranks=[0])
        first_identity = lifecycle.resolve_group_identity(first)
        dist.destroy_process_group(first)
        second = dist.new_group(ranks=[0])
        second_identity = lifecycle.resolve_group_identity(second)
        assert first_identity.group_uid == (digest, 0)
        assert second_identity.group_uid == (digest, 1)

    def test_seq_counters_key_on_full_group_uid(self, gloo_world):
        dist = gloo_world
        lifecycle.arm()
        first = dist.new_group(ranks=[0])
        first_identity = lifecycle.resolve_group_identity(first)
        assert lifecycle.next_seq(first_identity, "coll") == 0
        assert lifecycle.next_seq(first_identity, "coll") == 1
        dist.destroy_process_group(first)
        second = dist.new_group(ranks=[0])
        second_identity = lifecycle.resolve_group_identity(second)
        # A recreated communicator is a new uid: counters start fresh.
        assert lifecycle.next_seq(second_identity, "coll") == 0
        # Channels are independent.
        assert lifecycle.next_seq(second_identity, "p2p/0->0/0") == 0

    def test_disarm_restores_lifecycle_functions(self, unarmed):
        dist = unarmed
        original = dist.new_group
        lifecycle.arm()
        assert dist.new_group is not original
        lifecycle.disarm()
        assert dist.new_group is original
        assert not lifecycle.is_armed()


class TestTeardownClobberSafety:
    """b8 KNOWN-held (p5 3.15 rollup): teardown must not clobber foreign patches.

    Fail-before: ``disarm()`` / ``remove_collective_wraps()`` / the arm
    rollback ``setattr``'d the pristine original blindly, so a third-party
    library that patched the same c10d attribute AFTER TorchLens wrapped it
    had its patch silently destroyed at teardown.
    """

    def test_restore_helper_skips_foreign_patch_and_warns(self):
        original = object()

        def our_wrap():
            """Stand-in for an installed TorchLens wrap."""

        our_wrap.__tl_distributed_wrap__ = True
        our_wrap.__wrapped__ = original

        class Module:
            __name__ = "fake_module"

        module = Module()
        module.f = our_wrap
        lifecycle.restore_wrapped_attr(module, "f", original)
        assert module.f is original

        def foreign():
            """A third-party patch layered over (or replacing) our wrap."""

        module.f = foreign
        with pytest.warns(UserWarning, match="re-patched by a third party"):
            lifecycle.restore_wrapped_attr(module, "f", original)
        assert module.f is foreign

    @requires_vetted_snapshot
    def test_disarm_leaves_foreign_patches_intact(self, unarmed):
        dist = unarmed
        pristine_all_reduce = dist.all_reduce
        pristine_new_group = dist.new_group
        pristine_broadcast = dist.broadcast
        lifecycle.arm()
        try:
            shim_all_reduce = dist.all_reduce
            assert getattr(shim_all_reduce, "__tl_distributed_wrap__", False)

            def third_party_all_reduce(*args, **kwargs):
                return shim_all_reduce(*args, **kwargs)

            shim_new_group = dist.new_group

            def third_party_new_group(*args, **kwargs):
                return shim_new_group(*args, **kwargs)

            dist.all_reduce = third_party_all_reduce
            dist.new_group = third_party_new_group
            with pytest.warns(UserWarning, match="re-patched by a third party"):
                lifecycle.disarm()
            # Both wrap families' foreign patches survive teardown...
            assert dist.all_reduce is third_party_all_reduce
            assert dist.new_group is third_party_new_group
            # ...the un-patched site restored pristine, and the shims under
            # the foreign patches are inert passthroughs (state retired).
            assert dist.broadcast is pristine_broadcast
            assert lifecycle.armed_state() is None
        finally:
            lifecycle.disarm()
            dist.all_reduce = pristine_all_reduce
            dist.new_group = pristine_new_group
            dist.broadcast = pristine_broadcast


class TestC10dGroupSeqCompatRouting:
    """Deep-hunt F8: the private group-seq probe routes through _torch_compat.

    Fail-before: ``_c10d_group_seq`` called the private
    ``_get_sequence_number_for_group`` under a bare ``except Exception ->
    None``, so a private-API rename silently and permanently disabled the
    redundant correlation cross-check -- the only in-band detector in the
    base-misalignment neighborhood -- with zero visibility in ``doctor()`` /
    ``compat.report()``.
    """

    def test_missing_capability_short_circuits_to_none(self, monkeypatch):
        from torchlens.backends.torch import collectives
        from torchlens.utils import _torch_compat as tc

        monkeypatch.setattr(
            tc, "probe_c10d_capabilities", lambda **_: {"HAS_C10D_GROUP_SEQ": False}
        )

        class Recording:
            called = False

            def _get_sequence_number_for_group(self):
                Recording.called = True
                return 41

        assert collectives._c10d_group_seq(Recording()) == (None, None)
        assert Recording.called is False

    def test_present_capability_reads_the_private_counter(self, monkeypatch):
        from torchlens.backends.torch import collectives
        from torchlens.utils import _torch_compat as tc

        monkeypatch.setattr(tc, "probe_c10d_capabilities", lambda **_: {"HAS_C10D_GROUP_SEQ": True})

        class Fake:
            def _get_sequence_number_for_group(self):
                return 41

        assert collectives._c10d_group_seq(Fake()) == (41, None)

    def test_getter_raise_demotes_capability_and_discloses(self, monkeypatch):
        """b7-sol-R22-1: a raising getter must not silently vanish the witness.

        Fail-before: with ``HAS_C10D_GROUP_SEQ`` probed True, every getter
        exception was swallowed to a bare ``None`` while the flag stayed True
        -- the only in-band base-misalignment cross-check silently vanished,
        invisible to ``doctor()`` / ``compat.report()`` and indistinguishable
        on the boundary record from honest capability absence.
        """

        import warnings as warnings_module

        from torchlens.backends.torch import collectives
        from torchlens.utils import _torch_compat as tc

        monkeypatch.setattr(tc, "HAS_C10D_GROUP_SEQ", True)
        monkeypatch.setattr(tc, "_C10D_GROUP_SEQ_PROBED", True)
        monkeypatch.setattr(tc, "_warned_missing_capabilities", set())

        class Raising:
            def _get_sequence_number_for_group(self):
                raise RuntimeError("private API drifted at read time")

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            value, disclosure = collectives._c10d_group_seq(Raising())
        assert value is None
        assert disclosure == "c10d_group_seq_read_failed"
        # The degradation is now VISIBLE: the capability flag flipped through
        # the standard channel, so doctor()/compat.report() report it.
        assert tc.HAS_C10D_GROUP_SEQ is False
        assert tc.probe_c10d_capabilities()["HAS_C10D_GROUP_SEQ"] is False
        assert any(
            issubclass(item.category, tc.TorchCapabilityWarning)
            and "raised at read time" in str(item.message)
            for item in caught
        )

    def test_payload_carries_the_read_failure_disclosure(self, monkeypatch):
        """The boundary record where the witness vanished names the failure."""

        from torchlens.backends.torch import collectives

        monkeypatch.setattr(
            collectives,
            "_c10d_group_seq",
            lambda group: (None, "c10d_group_seq_read_failed"),
        )
        site = next(s for s in collectives.COLLECTIVE_SITES if s.attr == "barrier")

        class Identity:
            membership_digest = "d" * 64
            lifetime_ordinal = 0
            ordinal_source = "wrapped"
            global_ranks = (0,)
            backend = "gloo"

        class Arming:
            install_epoch = "armed_before_any_group"
            source = "explicit"

        monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 0, raising=False)
        payload = collectives._build_payload(
            site,
            {},
            Identity(),
            "coll",
            0,
            Arming(),
            None,
            [],
            [],
            None,
            False,
            "none",
            None,
        )
        assert payload["c10d_group_seq"] is None
        assert "c10d_group_seq_read_failed" in payload["disclosures"]


class TestBrokenArmPoisoning:
    """Deep-hunt F9: a failed arm with a failed rollback must not present as armed.

    Fail-before: the error path published the half-armed ``_STATE`` (so
    ``disarm()`` could retry restoration) with no poison marker, so a
    subsequent ``arm()`` / ``maybe_auto_arm()`` returned the arming record and
    capture proceeded "armed" while unwrapped collective sites were silently
    omitted -- precisely the fail-open arming exists to prevent.
    """

    @requires_vetted_snapshot
    def test_failed_arm_with_failed_restore_poisons_state(self, unarmed, monkeypatch):
        from torchlens.errors._base import CompatibilityError

        _ = unarmed
        original = object()

        class RefusingModule:
            allow = False

            def __setattr__(self, name: str, value: object) -> None:
                if name == "f" and value is original and not RefusingModule.allow:
                    raise RuntimeError("restore refused")
                object.__setattr__(self, name, value)

        module = RefusingModule()

        def installed_wrap() -> object:
            """Live TorchLens wrap over ``original`` (so rollback restores)."""

        installed_wrap.__tl_distributed_wrap__ = True
        installed_wrap.__wrapped__ = original

        def failing_install(state):
            state.originals[(module, "f")] = original
            module.f = installed_wrap
            raise RuntimeError("install failed")

        monkeypatch.setattr(lifecycle, "_install_lifecycle_wraps", failing_install)
        with pytest.raises(RuntimeError, match="restore refused"):
            lifecycle.arm()
        state = lifecycle.armed_state()
        assert state is not None and state.broken is True

        # Every arming entry point refuses typed on the poisoned state.
        with pytest.raises(CompatibilityError, match="half-armed"):
            lifecycle.arm()
        with pytest.raises(CompatibilityError, match="half-armed"):
            lifecycle.maybe_auto_arm()

        # disarm() keeps retrying restoration; success clears the poison.
        with pytest.raises(RuntimeError, match="restore refused"):
            lifecycle.disarm()
        assert lifecycle.armed_state() is state
        RefusingModule.allow = True
        lifecycle.disarm()
        assert not lifecycle.is_armed()
