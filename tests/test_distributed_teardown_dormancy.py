"""Armed distributed state never steers a capture that has no process group.

Arming is process-lifetime by design: it survives ``destroy_process_group`` so
the group-lifecycle ledger outlives every group (a disarm-and-rearm after
teardown would read the emptied group registry as "no history" and reuse
lifetime ordinals). With no process group initialized no collective can run,
so the armed state is DORMANT: plane-P's dispatch witness and the funcol
completion session stay off and the capture takes the unarmed path. Every test
here arms inside its own body, so the conftest autouse disarm (which runs only
after a test) cannot mask the contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from test_weightsfree_fixtures import ConvBnPool, meta_like, weightsfree_trace
from torch import nn

import torchlens as tl
from torchlens.backends.torch.completeness_witness import _HOST_ESCAPE_MUTABLE_WRITEBACK
from torchlens.distributed import _lifecycle as lifecycle, has_vetted_snapshot

pytestmark = pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
    reason="torch.distributed gloo unavailable",
)

requires_vetted_snapshot = pytest.mark.skipif(
    not has_vetted_snapshot(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.has_vetted_snapshot() is False here)",
)


def _init_group(tmp_path: Path, name: str) -> None:
    dist = torch.distributed
    store = dist.FileStore(str(tmp_path / name), 1)
    dist.init_process_group("gloo", store=store, rank=0, world_size=1)


@pytest.fixture()
def clean_distributed(tmp_path: Path) -> Iterator[Path]:
    """Start and end disarmed with no process group."""

    dist = torch.distributed
    lifecycle.disarm()
    if dist.is_initialized():
        dist.destroy_process_group()
    try:
        yield tmp_path
    finally:
        lifecycle.disarm()
        if dist.is_initialized():
            dist.destroy_process_group()


def _train_mode_meta_capture() -> Any:
    with torch.device("meta"):
        meta = ConvBnPool()
    meta.train()
    return weightsfree_trace(meta, meta_like(torch.randn(2, 3, 8, 8)))


def _dense_trace() -> Any:
    return tl.trace(nn.Sequential(nn.Linear(4, 4), nn.ReLU()), torch.randn(2, 4))


@requires_vetted_snapshot
def test_capture_after_group_teardown_is_unarmed_path(clean_distributed: Path) -> None:
    """After destroy, the armed process captures like an unarmed one."""

    _init_group(clean_distributed, "store")
    armed = _dense_trace()
    state = lifecycle.armed_state()
    assert state is not None
    assert armed._distributed_plane_p is not None
    torch.distributed.destroy_process_group()

    # The armed state and its ledger survive teardown (lifetime ordinals).
    assert lifecycle.armed_state() is state
    assert lifecycle.capture_armed_state() is None

    dense = _dense_trace()
    assert dense._distributed_plane_p is None
    meta_trace = _train_mode_meta_capture()
    assert meta_trace not in _HOST_ESCAPE_MUTABLE_WRITEBACK


@requires_vetted_snapshot
def test_groupless_armed_capture_does_not_run_plane_p(clean_distributed: Path) -> None:
    """The dormancy contract: armed with no group, plane-P never runs.

    Also the documented arm-at-process-start pattern: dormant until init, then
    observed.
    """

    tl.distributed.arm()
    assert lifecycle.is_armed()
    assert not torch.distributed.is_initialized()
    assert lifecycle.capture_armed_state() is None
    assert _dense_trace()._distributed_plane_p is None
    assert _train_mode_meta_capture()._distributed_plane_p is None
    assert lifecycle.is_armed(), "dormancy must not disarm"

    _init_group(clean_distributed, "store")
    assert lifecycle.capture_armed_state() is lifecycle.armed_state()
    assert _dense_trace()._distributed_plane_p is not None


@requires_vetted_snapshot
def test_reinitialized_group_wakes_the_armed_state(clean_distributed: Path) -> None:
    """A group created after a teardown is observed again."""

    _init_group(clean_distributed, "first")
    _dense_trace()
    torch.distributed.destroy_process_group()
    assert _dense_trace()._distributed_plane_p is None
    _init_group(clean_distributed, "second")
    assert _dense_trace()._distributed_plane_p is not None


@requires_vetted_snapshot
def test_live_group_meta_capture_raises_no_fabricated_flag(clean_distributed: Path) -> None:
    """With plane-P active, storage-less meta state is never flagged as written."""

    _init_group(clean_distributed, "store")
    _dense_trace()
    assert lifecycle.capture_armed_state() is not None
    meta_trace = _train_mode_meta_capture()
    assert meta_trace._distributed_plane_p is not None
    assert meta_trace not in _HOST_ESCAPE_MUTABLE_WRITEBACK
