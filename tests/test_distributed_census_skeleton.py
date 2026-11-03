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

from torchlens._distributed import _lifecycle as lifecycle  # noqa: E402
from torchlens.distributed import (  # noqa: E402
    is_vetted_build,
)

# F1 ruling (Lead, 2026-10-01; test_distributed_census_topologies.py): full
# collective arming only runs where a census-vetted torch build exists; on an
# unvetted torch, arm() fails closed with UncapturedCollectiveOpError inside
# the spawned worker, which mp.spawn re-raises as ProcessRaisedException.
# _p2p_worker (below) arms unconditionally, so the same gate applies here.
requires_vetted_snapshot = pytest.mark.skipif(
    not is_vetted_build(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.is_vetted_build() is False here)",
)


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
