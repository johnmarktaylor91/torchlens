"""DDP unwrap smoke tests for fastlog."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl


class DdpModel(nn.Module):
    """Simple model for DDP unwrap tests."""

    def __init__(self) -> None:
        """Initialize the layer."""

        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer."""

        return self.linear(x)


def _init_process_group(tmp_path: Path) -> bool:
    """Initialize a local single-rank process group if needed."""

    if not torch.distributed.is_available():
        return False
    if torch.distributed.is_initialized():
        return True
    init_file = tmp_path / "ddp_init"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=0,
        world_size=1,
    )
    return True


def test_ddp_wrapped_model_records_unwrapped_module(tmp_path: Path) -> None:
    """This exercises only the unwrapped .module, NOT DDP forward semantics."""

    if not _init_process_group(tmp_path):
        pytest.skip("torch.distributed is unavailable")
    ddp_model = torch.nn.parallel.DistributedDataParallel(DdpModel())

    recording = tl.fastlog.record(ddp_model, torch.ones(1, 2), default_op=True)

    assert len(recording) > 0


def test_ddp_bundle_path_gets_rank_prefix(tmp_path: Path) -> None:
    """DDP disk bundles are written under a rank_NN prefix."""

    if not _init_process_group(tmp_path):
        pytest.skip("torch.distributed is unavailable")
    ddp_model = torch.nn.parallel.DistributedDataParallel(DdpModel())
    requested = tmp_path / "bundle.tlfast"

    recording = tl.fastlog.record(
        ddp_model,
        torch.ones(1, 2),
        default_op=True,
        streaming=tl.StreamingOptions(bundle_path=requested, retain_in_memory=False),
    )

    assert recording.bundle_path == tmp_path / "rank_00" / "bundle.tlfast"
    assert (tmp_path / "rank_00" / "bundle.tlfast" / "manifest.json").exists()
