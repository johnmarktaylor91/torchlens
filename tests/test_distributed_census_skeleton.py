"""Census harness skeleton self-tests + a 2-rank p2p channel sim.

The census skeleton is a named C0 gate item ("census harness skeleton
landed"); criterion 1 must be runnable and falsifiable NOW so C2 inherits a
harness that cannot pass vacuously. The p2p spawn sim pins the 1.4 channel
model cross-rank: sender and receiver derive the SAME canonical channel
string and aligned seqs from opposite ends of the wire.
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

from tests.support.census_harness import (  # noqa: E402
    run_census_criterion_1,
    run_census_criterion_2,
)
from torchlens.distributed import (  # noqa: E402
    _lifecycle as lifecycle,
    has_vetted_snapshot,
)

# F1 ruling (Lead, 2026-10-01; test_distributed_census_topologies.py): full
# collective arming only runs where a census-vetted torch build exists; on an
# unvetted torch, arm() fails closed with UncapturedCollectiveOpError inside
# the spawned worker, which mp.spawn re-raises as ProcessRaisedException.
# _p2p_worker (below) arms unconditionally, so the same gate applies here.
requires_vetted_snapshot = pytest.mark.skipif(
    not has_vetted_snapshot(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.has_vetted_snapshot() is False here)",
)


class TestCensusSkeleton:
    def test_criterion_1_green_on_dense_model(self):
        result = run_census_criterion_1(
            lambda: nn.Sequential(nn.Linear(4, 4), nn.ReLU()),
            lambda: torch.randn(2, 4),
        )
        assert result.green and result.outputs_bit_identical
        assert result.criteria_run == (1,)
        # The ground-truth stream is real, not empty (aten ops recorded).
        assert any("addmm" in op or "mm" in op for op in result.ground_truth_ops)

    def test_criterion_1_is_falsifiable(self):
        # A model whose factory differs between legs must go RED: the census
        # can never be vacuous about non-perturbation.
        calls = {"n": 0}

        def factory():
            calls["n"] += 1
            model = nn.Linear(4, 4)
            if calls["n"] > 1:
                with torch.no_grad():
                    model.weight.add_(1.0)
            return model

        result = run_census_criterion_1(factory, lambda: torch.randn(2, 4))
        assert not result.green
        assert any("not bit-identical" in failure for failure in result.failures)

    def test_criteria_2_through_4_refuse_without_plane_p(self):
        """Wave-1 form of the skeleton honesty pin: the criteria BODIES exist
        now (C2 recording lane), but an UNARMED capture carries no plane-P
        journal, so requesting them still raises -- a green can never be
        vacuous about which criteria actually ran."""

        import torchlens as tl

        lifecycle.disarm()
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        with pytest.raises(NotImplementedError, match="C2"):
            run_census_criterion_2([], log)


def _p2p_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    import torch.distributed as dist

    import torchlens as tl
    from torchlens.distributed import arm

    store = dist.FileStore(init_file, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    arm()

    if rank == 0:

        class Sender(nn.Module):
            def forward(self, x):
                doubled = x * 2
                dist.send(doubled, dst=1, tag=7)
                return doubled + 0.0

        log = tl.trace(Sender(), torch.ones(3))
        kind = "send"
    else:

        class Receiver(nn.Module):
            def forward(self, x):
                buffer = torch.empty(3)
                dist.recv(buffer, src=0, tag=7)
                return buffer + x

        log = tl.trace(Receiver(), torch.zeros(3))
        kind = "recv"
        received = [op for op in log.ops if op.type == "recv"][0]
        assert torch.equal(received.out, torch.full((3,), 2.0)), "recv value wrong"

    (entry,) = log.annotations["distributed"]["boundaries"]
    assert entry["kind"] == kind
    payload = {"rank": rank, "correlation": entry["correlation"], "peer": entry["peer"]}
    with open(os.path.join(out_dir, f"p2p_rank{rank}.json"), "w") as handle:
        json.dump(payload, handle)
    dist.destroy_process_group()


@pytest.mark.slow
class TestP2PChannelSim:
    @requires_vetted_snapshot
    def test_sender_and_receiver_derive_the_same_channel(self, tmp_path):
        import torch.multiprocessing as mp

        lifecycle.disarm()
        if torch.distributed.is_initialized():
            pytest.skip("a process group is already initialized in this process")
        mp.spawn(
            _p2p_worker,
            args=(2, str(tmp_path / "p2p_store"), str(tmp_path)),
            nprocs=2,
            join=True,
        )
        payloads = []
        for rank in range(2):
            with open(os.path.join(str(tmp_path), f"p2p_rank{rank}.json")) as handle:
                payloads.append(json.load(handle))
        sender, receiver = payloads
        # Opposite ends of the wire derive the SAME channel: canonical global
        # src->dst plus the tag (gloo honors tags), and aligned seqs.
        assert sender["correlation"]["channel"] == "p2p/0->1/7"
        assert sender["correlation"] == receiver["correlation"]
        assert sender["peer"]["canonical"] == {"src": 0, "dst": 1}
        assert receiver["peer"]["canonical"] == {"src": 0, "dst": 1}
        assert sender["peer"]["tag_in_channel"] is True
