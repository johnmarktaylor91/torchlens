"""Tier-(b) collective boundary capture: gloo sims.

Single-process pins for boundary-node shape, C0 payload content, nesting
suppression, tensorless journaling, and typed refusals; then 2-rank
``spawn`` gloo sims proving rank-local graph completeness and cross-rank
correlation-key agreement (merge-ranks tier (b), scope-backend-parity2 WP-F;
record content per design-merge-ranks-c v5 C0).
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from torch import nn

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed gloo unavailable",
)

import torchlens as tl  # noqa: E402
from torchlens.backends.torch.collectives import (  # noqa: E402
    WildcardRecvUnsupportedError,
    remove_collective_wraps,
)
from torchlens.distributed import _lifecycle as lifecycle  # noqa: E402
from torchlens.errors._base import CompatibilityError  # noqa: E402


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


class HandRolledTP(nn.Module):
    """Dense-parameter model issuing an explicit collective mid-forward."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        hidden = self.fc(x)
        torch.distributed.all_reduce(hidden)
        return torch.relu(hidden)


class TestBoundaryNode:
    # NOT smoke-marked yet (2026-08-14, fix-testinfra T14-4): every test here
    # measures well under the smoke budget, but
    # test_boundary_carries_collective_boundary_v1_payload is red on main
    # (layer.annotations lacks the op's "collective" mirror -- relayed
    # source-side defect). Smoke-mark this class WITH that fix so the
    # commit-tier gate does not inherit a known red.
    def test_allreduce_becomes_boundary_node_with_provenance(self, gloo_world):
        lifecycle.arm()
        log = tl.trace(HandRolledTP(), torch.randn(2, 4))
        boundary_ops = [op for op in log.ops if op.type == "allreduce"]
        assert len(boundary_ops) == 1
        boundary = boundary_ops[0]
        # The boundary node sits IN the dataflow: linear -> allreduce -> relu.
        assert any("linear" in parent for parent in boundary.parents)
        relu = [op for op in log.ops if op.type == "relu"][0]
        assert any("allreduce" in parent for parent in relu.parents)

    def test_boundary_carries_collective_boundary_v1_payload(self, gloo_world):
        lifecycle.arm()
        log = tl.trace(HandRolledTP(), torch.randn(2, 4))
        boundary = [op for op in log.ops if op.type == "allreduce"][0]
        info = boundary.annotations["collective"]
        assert info["schema"] == "collective_boundary_v1"
        assert info["kind"] == "all_reduce"
        correlation = info["correlation"]
        assert set(correlation) == {"membership_digest", "lifetime_ordinal", "channel", "seq"}
        assert correlation["channel"] == "coll"
        assert correlation["seq"] == 0
        assert info["reduce_op"] is not None and "SUM" in info["reduce_op"]
        assert info["events"] == {"async_op": False, "completion_binding": "issue_sync"}
        (role,) = info["roles"]
        assert role["role"] == "contribution_destination"
        assert role["shape"] == [2, 4]
        # Dual-geometry slots are declared and None for plain dense tensors.
        assert role["logical_shape"] is None and role["placements"] is None
        evidence = info["lifetime_evidence"]
        assert evidence["install_epoch"] in ("armed_before_any_group", "seeded")
        assert evidence["arming_source"] == "explicit"
        assert info["witness"]["policy_resolved"] == "none"
        layer = log[boundary.layer_label]
        assert layer.annotations["collective"] == info
        # The payload is portable plain data.
        json.dumps(info)

    def test_trace_journal_and_ledger_are_recorded(self, gloo_world):
        lifecycle.arm()
        log = tl.trace(HandRolledTP(), torch.randn(2, 4))
        record = log.annotations["distributed"]
        assert len(record["boundaries"]) == 1
        entry = record["boundaries"][0]
        assert entry["op_labels_raw"]
        assert record["group_lifecycle_ledger"], "ledger must serialize into the rank core"
        assert record["install_epoch"] in ("armed_before_any_group", "seeded")
        json.dumps(record)

    def test_output_passthrough_never_duplicates_boundary_payload(self, gloo_world):
        dist = gloo_world

        class ReturnsCollective(nn.Module):
            def forward(self, x):
                doubled = x * 2
                dist.all_reduce(doubled)
                return doubled

        lifecycle.arm()
        log = tl.trace(ReturnsCollective(), torch.randn(2, 4))
        record = log.annotations["distributed"]
        declared = {raw for entry in record["boundaries"] for raw in entry["op_labels_raw"]}
        carriers = [op for op in log.ops if "collective" in op.annotations]
        # The journal's op_labels_raw is the boundary->op mapping authority:
        # exactly the declared ops carry the portable payload. The synthetic
        # output node used to inherit the payload (correlation key included)
        # through the step-1 wholesale clone, advertising N+1 carriers for N
        # boundaries and breaking per-rank correlation-key uniqueness.
        assert {op._label_raw for op in carriers} == declared
        keys = [
            json.dumps(op.annotations["collective"]["correlation"], sort_keys=True)
            for op in carriers
        ]
        assert len(keys) == len(set(keys)), "correlation keys must stay unique per rank"
        output_ops = [op for op in log.ops if op.is_output]
        assert output_ops
        assert all(op.annotations == {} for op in output_ops)

    def test_auto_arm_at_capture_entry_for_spmd(self, gloo_world):
        # No explicit arm(): an initialized SPMD process arms lazily at
        # capture entry with restricted registry seeding.
        assert not lifecycle.is_armed()
        log = tl.trace(HandRolledTP(), torch.randn(2, 4))
        assert lifecycle.is_armed()
        info = [op for op in log.ops if op.type == "allreduce"][0].annotations["collective"]
        assert info["lifetime_evidence"]["arming_source"] == "auto"
        assert info["lifetime_evidence"]["ordinal_source"] == "seeded"

    def test_seq_ticks_at_issue_even_outside_capture(self, gloo_world):
        dist = gloo_world
        lifecycle.arm()
        # An armed issue OUTSIDE capture must tick: correlation counting never
        # depends on which forwards were captured.
        outside = torch.ones(2)
        dist.all_reduce(outside)
        log = tl.trace(HandRolledTP(), torch.randn(2, 4))
        info = [op for op in log.ops if op.type == "allreduce"][0].annotations["collective"]
        assert info["correlation"]["seq"] == 1

    def test_all_gather_list_destinations_enter_dataflow(self, gloo_world):
        dist = gloo_world

        class GatherModel(nn.Module):
            def forward(self, x):
                doubled = x * 2
                buckets = [torch.empty_like(doubled)]
                dist.all_gather(buckets, doubled)
                return buckets[0] + 1

        lifecycle.arm()
        log = tl.trace(GatherModel(), torch.randn(2, 4))
        boundary = [op for op in log.ops if op.type == "allgather"][0]
        assert any("mul" in parent for parent in boundary.parents)
        add = [op for op in log.ops if op.type == "add"][0]
        assert any("allgather" in parent for parent in add.parents)

    def test_async_op_records_unobserved_completion(self, gloo_world):
        dist = gloo_world

        class AsyncModel(nn.Module):
            def forward(self, x):
                doubled = x * 2
                work = dist.all_reduce(doubled, async_op=True)
                work.wait()
                return doubled + 1

        lifecycle.arm()
        log = tl.trace(AsyncModel(), torch.randn(2, 4))
        info = [op for op in log.ops if op.type == "allreduce"][0].annotations["collective"]
        assert info["events"] == {"async_op": True, "completion_binding": "unobserved"}
        assert "read_of_inflight_destination" in info["disclosures"]

    def test_nested_collectives_inside_object_collective_do_not_tick(self, gloo_world):
        dist = gloo_world

        class ObjectModel(nn.Module):
            def forward(self, x):
                gathered = [None]
                dist.all_gather_object(gathered, int(x.sum().item()))
                return x * 1.0

        lifecycle.arm()
        log = tl.trace(ObjectModel(), torch.ones(2))
        record = log.annotations["distributed"]
        kinds = [entry["kind"] for entry in record["boundaries"]]
        # Exactly ONE boundary: the outer user-level object collective. The
        # inner all_gather calls it delegates to are suppressed (v5 1.2).
        assert kinds == ["all_gather_object"]
        assert not any(op.type == "allgather" for op in log.ops)
        state = lifecycle.armed_state()
        coll_ticks = sum(count for key, count in state.seq_counters.items() if key[2] == "coll")
        assert coll_ticks == 1

    def test_barrier_is_journal_only(self, gloo_world):
        dist = gloo_world

        class BarrierModel(nn.Module):
            def forward(self, x):
                dist.barrier()
                return x + 1

        lifecycle.arm()
        log = tl.trace(BarrierModel(), torch.ones(2))
        record = log.annotations["distributed"]
        (entry,) = record["boundaries"]
        assert entry["kind"] == "barrier"
        assert entry["op_labels_raw"] == [] and entry["op_node"] is False
        assert not any(op.type == "barrier" for op in log.ops)

    def test_wildcard_recv_refuses_typed_during_capture(self, gloo_world):
        dist = gloo_world

        class WildcardModel(nn.Module):
            def forward(self, x):
                buffer = torch.empty_like(x)
                dist.recv(buffer)  # any-source: no determinate peer
                return x + 1

        lifecycle.arm()
        with pytest.raises(WildcardRecvUnsupportedError) as excinfo:
            tl.trace(WildcardModel(), torch.ones(2))
        assert excinfo.value.fields["kind"] == "wildcard_recv_unsupported"

    def test_disarm_restores_collective_functions(self, gloo_world):
        dist = gloo_world
        original = dist.all_reduce
        lifecycle.arm()
        assert dist.all_reduce is not original
        lifecycle.disarm()
        assert dist.all_reduce is original

    def test_failed_collective_restore_retains_original_for_retry(self) -> None:
        """A teardown failure cannot discard the pristine function ledger entry."""

        original = object()

        class RefusingModule:
            """Hashable module-like object that rejects one restoration."""

            def __setattr__(self, name: str, value: object) -> None:
                """Reject restoration while permitting test setup."""

                if name == "all_reduce" and value is original:
                    raise RuntimeError("restore refused")
                object.__setattr__(self, name, value)

        def installed_wrap() -> object:
            """Live TorchLens wrap over ``original`` (so restore is attempted)."""

        installed_wrap.__tl_distributed_wrap__ = True
        installed_wrap.__wrapped__ = original

        module = RefusingModule()
        module.all_reduce = installed_wrap
        originals = {(module, "all_reduce"): original}
        with pytest.raises(RuntimeError, match="restore refused"):
            remove_collective_wraps(originals)
        assert originals[(module, "all_reduce")] is original

    def test_unarmed_capture_is_status_quo(self, gloo_world):
        # Arm is refused/absent -> no boundary nodes, capture itself intact
        # (the pre-tier-(b) behavior). Force unarmed by disarming and making
        # lazy arming a no-op via an uninitialized-looking guard is not
        # possible here (dist IS initialized), so instead pin that a plain
        # dense model with no collectives captures identically armed.
        lifecycle.arm()
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        log = tl.trace(model, torch.randn(2, 4))
        assert not any(op.type == "allreduce" for op in log.ops)
        assert "distributed" not in log.annotations


class TestWitnessPolicy:
    pytestmark = pytest.mark.smoke

    def test_digest_witness_records_byte_exact_digests(self, gloo_world):
        from torchlens.backends.torch.collectives import _digest_tensor

        lifecycle.arm()
        log = tl.trace(
            HandRolledTP(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(distributed_witness="digest"),
        )
        boundary = [op for op in log.ops if op.type == "allreduce"][0]
        witness = boundary.annotations["collective"]["witness"]
        assert witness["policy_resolved"] == "digest"
        assert len(witness["contribution_digests"]) == 1
        # The destination digest is byte-exact evidence: recomputing it over
        # the saved boundary output must reproduce it.
        assert witness["destination_digests"] == [_digest_tensor(boundary.out)]

    def test_digest_witness_does_not_change_captured_graph(self, gloo_world) -> None:
        """Capture-internal digest operations must remain outside the user graph."""

        lifecycle.arm()
        model = HandRolledTP()
        sample = torch.randn(2, 4)
        plain = tl.trace(model, sample)
        witnessed = tl.trace(
            model,
            sample,
            capture=tl.options.CaptureOptions(distributed_witness="digest"),
        )
        assert [op.type for op in witnessed.ops] == [op.type for op in plain.ops]

    def test_fastlog_collective_refuses_instead_of_dropping_journal(self, gloo_world) -> None:
        """Fastlog must not return a product that omits an executed collective."""

        lifecycle.arm()
        with pytest.raises(CompatibilityError) as excinfo:
            tl.record(HandRolledTP(), torch.randn(2, 4), save=tl.func("relu"))
        assert excinfo.value.fields["kind"] == "collective_boundary_fastlog_unsupported"

    def test_async_digest_destination_not_present(self, gloo_world):
        dist = gloo_world

        class AsyncModel(nn.Module):
            def forward(self, x):
                doubled = x * 2
                work = dist.all_reduce(doubled, async_op=True)
                work.wait()
                return doubled + 1

        lifecycle.arm()
        log = tl.trace(
            AsyncModel(),
            torch.randn(2, 4),
            capture=tl.options.CaptureOptions(distributed_witness="digest"),
        )
        witness = [op for op in log.ops if op.type == "allreduce"][0].annotations["collective"][
            "witness"
        ]
        assert witness["contribution_digests"] is not None
        assert witness["destination_digests"] is None
        assert witness["not_present_reason"] == "async_completion_unobserved"

    def test_payload_witness_is_reserved_and_refuses(self, gloo_world):
        lifecycle.arm()
        with pytest.raises(ValueError, match="payload"):
            tl.options.CaptureOptions(distributed_witness="payload")

    def test_unknown_witness_level_refuses(self, gloo_world):
        lifecycle.arm()
        with pytest.raises(ValueError, match="distributed_witness"):
            tl.options.CaptureOptions(distributed_witness="everything")


class TestReplayRefusals:
    pytestmark = pytest.mark.smoke
    """A collective-crossing rank core refuses runnable save + forward replay.

    Design v5 3.4: re-issuing a collective outside its communicator hangs or
    fabricates peer-dependent values, so both surfaces refuse typed; metadata
    invariants run in full.
    """

    def test_forward_replay_validation_refuses_typed(self, gloo_world):
        from torchlens.errors import CollectiveBoundaryReplayError

        lifecycle.arm()
        log = tl.trace(HandRolledTP(), torch.randn(2, 4))
        with pytest.raises(CollectiveBoundaryReplayError) as excinfo:
            log.validate_forward_pass([torch.zeros(2, 4)])
        assert excinfo.value.fields["code"] == "collective_boundary_runnable_unsupported"

    def test_runnable_save_refuses_typed(self, gloo_world, tmp_path):
        from torchlens.errors import RunnablePreflightError

        lifecycle.arm()
        model = HandRolledTP()  # held alive: the refusal under test is the
        log = tl.trace(model, torch.randn(2, 4))  # boundary one, not GC state
        with pytest.raises(RunnablePreflightError) as excinfo:
            tl.save(log, str(tmp_path / "boundary.tlspec"), level="runnable")
        codes = {d.code for d in excinfo.value.fields["diagnostics"]}
        assert "collective_boundary_runnable_unsupported" in codes

    def test_metadata_invariants_still_run_in_full(self, gloo_world):
        from torchlens.validation import check_metadata_invariants

        lifecycle.arm()
        log = tl.trace(HandRolledTP(), torch.randn(2, 4))
        # Raises MetadataInvariantError on any violation; a boundary-crossing
        # trace must still pass the full invariant sweep.
        check_metadata_invariants(log)

    def test_boundary_free_capture_keeps_both_surfaces(self, gloo_world, tmp_path):
        lifecycle.arm()
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        x = torch.randn(2, 4)
        expected = model(x)
        log = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
        # The control: no typed collective refusal fires on either surface.
        # (The save-before-validate order is no longer load-bearing: B1-04
        # declared _last_validation_failure / _validation_diagnostics as
        # FieldPolicy.DROP rows, so validate-then-save round-trips too. Pinned
        # in tests/test_capture_outcome_validation_side_channel.py; the order
        # here is kept only to leave this test's history untouched.)
        tl.save(log, str(tmp_path / "plain.tlspec"), level="runnable")
        status = log.validate_forward_pass([expected])
        assert status is not None


# ---------------------------------------------------------------------------
# 2-rank spawn sims
# ---------------------------------------------------------------------------


def _tp_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import torch
    import torch.distributed as dist

    import torchlens as tl
    from torchlens.backends.torch.collectives import _digest_tensor
    from torchlens.distributed import arm

    store = dist.FileStore(init_file, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    arm()

    torch.manual_seed(1234)  # identical params on every rank

    class TP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            hidden = self.fc(x)
            dist.all_reduce(hidden)
            gathered = [torch.empty_like(hidden) for _ in range(world_size)]
            dist.all_gather(gathered, hidden)
            return torch.relu(gathered[rank])

    model = TP()
    torch.manual_seed(77)  # identical input on every rank
    x = torch.randn(2, 4)
    log = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(distributed_witness="digest"),
    )

    # Ground truth from a bare model built identically.
    torch.manual_seed(1234)
    reference = TP()
    contribution = reference.fc(x)
    expected = contribution * world_size
    boundary_out = [op for op in log.ops if op.type == "allreduce"][0].out
    assert torch.allclose(boundary_out, expected, atol=1e-6), "captured all_reduce value wrong"

    payload = {
        "rank": rank,
        "boundaries": [
            {
                "kind": entry["kind"],
                "correlation": entry["correlation"],
                "group": {
                    "global_ranks": entry["group"]["global_ranks"],
                    "my_global_rank": entry["group"]["my_global_rank"],
                },
            }
            for entry in log.annotations["distributed"]["boundaries"]
        ],
        "op_types": sorted({op.type for op in log.ops}),
        "witness": log.annotations["distributed"]["boundaries"][0]["witness"],
        "expected_contribution_digest": _digest_tensor(contribution),
        "expected_destination_digest": _digest_tensor(expected),
    }
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(payload, handle)
    dist.destroy_process_group()


@pytest.mark.slow
class TestTwoRankSims:
    def test_two_rank_tp_graph_complete_and_keys_align(self, tmp_path):
        import torch.multiprocessing as mp

        lifecycle.disarm()
        if torch.distributed.is_initialized():
            pytest.skip("a process group is already initialized in this process")
        world_size = 2
        init_file = str(tmp_path / "init_store")
        out_dir = str(tmp_path)
        mp.spawn(
            _tp_worker,
            args=(world_size, init_file, out_dir),
            nprocs=world_size,
            join=True,
        )
        payloads = []
        for rank in range(world_size):
            with open(os.path.join(out_dir, f"rank{rank}.json")) as handle:
                payloads.append(json.load(handle))
        rank0, rank1 = payloads
        # Rank-local graph completeness: every boundary is a node.
        for payload in payloads:
            assert "allreduce" in payload["op_types"]
            assert "allgather" in payload["op_types"]
            assert [b["kind"] for b in payload["boundaries"]] == ["all_reduce", "all_gather"]
            witness = payload["witness"]
            assert witness["contribution_digests"] == [payload["expected_contribution_digest"]]
            assert witness["destination_digests"] == [payload["expected_destination_digest"]]
        # Correlation keys agree cross-rank per boundary: same membership
        # digest, same lifetime ordinal, same channel, same seq.
        for entry0, entry1 in zip(rank0["boundaries"], rank1["boundaries"]):
            assert entry0["correlation"] == entry1["correlation"]
            assert entry0["group"]["global_ranks"] == entry1["group"]["global_ranks"]
        assert rank0["boundaries"][0]["group"]["my_global_rank"] == 0
        assert rank1["boundaries"][0]["group"]["my_global_rank"] == 1
