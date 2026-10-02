"""Tier-(b) DTensor param identity + refusal parity.

Two pins. (1) DTensor dual-geometry extraction: every surface that talks
about sharded state declares logical AND local geometry explicitly, so a
refused TP/FSDP2 model is refused with its parameters precisely identified
(the tier-(a) report said only "zero parameters"). (2) Refusal parity: arming
collective-boundary capture relaxes NOTHING -- dtensor / tensor_parallel /
pipeline_parallel captures still refuse typed, because their capture fidelity
is unproven until the C2 census (LOCKED: relaxation follows fidelity, never
precedes it).
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
from torchlens._distributed import (  # noqa: E402
    REFUSING_KINDS,
    DistributedCaptureUnsupportedError,
    detect_distributed_state,
)
from torchlens.distributed import (  # noqa: E402
    _lifecycle as lifecycle,
    has_vetted_snapshot,
)
from torchlens.distributed._dtensor import dtensor_dual_geometry  # noqa: E402

# F1 ruling (Lead, 2026-10-01): full collective arming only runs where a
# census-vetted torch build exists (torchlens.distributed.has_vetted_snapshot());
# on an unvetted torch, arm() fails closed with UncapturedCollectiveOpError.
# Only the three TestRefusalParity tests below call a real arm(); the rest of
# the module (dual-geometry extraction, test_refusing_kinds_unchanged) is
# unrelated to vetting and must keep running everywhere.
requires_vetted_snapshot = pytest.mark.skipif(
    not has_vetted_snapshot(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.has_vetted_snapshot() is False here)",
)


@pytest.fixture()
def single_rank_mesh():
    """Single-rank CPU gloo mesh with guaranteed teardown."""

    import torch.distributed as dist

    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this process")
    lifecycle.disarm()
    import os

    saved_env = {
        key: os.environ.get(key) for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    # OS-assigned ephemeral port: a hardcoded port (29581 historically)
    # collides across parallel lanes/worktrees on one box, and the collision
    # lands in the except-skip below -- distributed coverage silently degrades
    # to SKIP instead of failing loudly (T14-3).
    import socket

    with socket.socket() as _probe:
        _probe.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(_probe.getsockname()[1])
    try:
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    except Exception as error:  # pragma: no cover - environment dependent
        pytest.skip(f"gloo init failed: {error}")
    from torch.distributed.device_mesh import init_device_mesh

    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.heavy
class TestDualGeometry:
    def test_sharded_dtensor_dual_geometry(self, single_rank_mesh):
        from torch.distributed.tensor import Shard, distribute_tensor

        logical = torch.arange(16.0).reshape(4, 4)
        sharded = distribute_tensor(logical, single_rank_mesh, [Shard(0)])
        geometry = dtensor_dual_geometry(sharded)
        assert geometry is not None
        assert geometry["logical_shape"] == [4, 4]
        assert geometry["local_shape"] == [4, 4]  # world 1: the shard is whole
        assert geometry["placements"] == ["Shard(dim=0)"]
        assert geometry["shard_offset"] == [0, 0]
        assert geometry["mesh_coords"] == [0]
        assert geometry["logical_numel"] == 16 and geometry["local_numel"] == 16

    def test_replicated_dtensor_dual_geometry(self, single_rank_mesh):
        from torch.distributed.tensor import Replicate, distribute_tensor

        logical = torch.ones(2, 3)
        replicated = distribute_tensor(logical, single_rank_mesh, [Replicate()])
        geometry = dtensor_dual_geometry(replicated)
        assert geometry is not None
        assert geometry["logical_shape"] == [2, 3]
        assert geometry["placements"] == ["Replicate()"]

    def test_plain_tensor_has_no_geometry(self):
        assert dtensor_dual_geometry(torch.ones(2)) is None

    def test_dtensor_finding_identifies_parameters(self, single_rank_mesh):
        from torch.distributed.tensor import Shard, distribute_tensor

        model = nn.Linear(4, 4)
        model.weight = nn.Parameter(
            distribute_tensor(torch.randn(4, 4), single_rank_mesh, [Shard(0)])
        )
        findings = detect_distributed_state(model, torch.randn(2, 4))
        dtensor_finding = next(f for f in findings if f.kind == "dtensor")
        assert len(dtensor_finding.geometry) == len(dtensor_finding.sites)
        weight_index = dtensor_finding.sites.index("weight")
        geometry = dtensor_finding.geometry[weight_index]
        assert geometry is not None and geometry["logical_shape"] == [4, 4]
        # The refusal now IDENTIFIES the logical state instead of just
        # reporting zero parameters.
        assert "Identified logical state: 16 logical element(s)" in dtensor_finding.detail


@pytest.mark.heavy
class TestRefusalParity:
    """Arming relaxes nothing: every tier-(a) refusal still fires armed."""

    def test_refusing_kinds_unchanged(self):
        assert (
            frozenset({"dtensor", "tensor_parallel", "pipeline_parallel", "scan_incomplete"})
            == REFUSING_KINDS
        )

    @requires_vetted_snapshot
    def test_dtensor_capture_still_refuses_when_armed(self, single_rank_mesh):
        from torch.distributed.tensor import Shard, distribute_tensor

        lifecycle.arm()
        model = nn.Linear(4, 4)
        model.weight = nn.Parameter(
            distribute_tensor(torch.randn(4, 4), single_rank_mesh, [Shard(0)])
        )
        with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
            tl.trace(model, torch.randn(2, 4))
        kinds = [finding.kind for finding in excinfo.value.fields["findings"]]
        assert "dtensor" in kinds

    @requires_vetted_snapshot
    def test_dense_tp_hooks_still_refuse_when_armed(self, single_rank_mesh):
        pytest.importorskip("torch.distributed.tensor.parallel")
        from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

        lifecycle.arm()
        model = nn.Sequential(nn.Linear(4, 4))
        parallelize_module(
            model,
            single_rank_mesh,
            {"0": PrepareModuleInput(desired_input_layouts=None)},
        )
        with pytest.raises(DistributedCaptureUnsupportedError) as excinfo:
            tl.trace(model, torch.randn(2, 4))
        kinds = [finding.kind for finding in excinfo.value.fields["findings"]]
        assert "tensor_parallel" in kinds

    @requires_vetted_snapshot
    def test_dense_explicit_collectives_capture_when_armed(self, single_rank_mesh):
        # The parity counterpoint: what tier (b) actually makes honest.
        import torch.distributed as dist

        lifecycle.arm()

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                hidden = self.fc(x)
                dist.all_reduce(hidden)
                return hidden

        log = tl.trace(Model(), torch.randn(2, 4))
        assert any(op.type == "allreduce" for op in log.ops)
