"""DDP unwrap smoke tests for fastlog."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.distributed import has_vetted_snapshot


class DdpModel(nn.Module):
    """Simple model for DDP unwrap tests."""

    def __init__(self) -> None:
        """Initialize the layer."""

        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the layer."""

        return self.linear(x)


def _teardown_process_group(created: bool) -> None:
    """Disarm distributed capture, then destroy the group this module created.

    Capturing under an initialized process group lazily ARMS distributed
    capture (``maybe_auto_arm`` at capture entry), and arming is process-
    lifetime by design: it survives ``destroy_process_group``. Left armed,
    ``_plane_p_requested()`` stays True and the completeness dispatch witness
    rides EVERY later capture in the session -- flipping ``FailureOrigin``
    classifications (test_capture_outcome_authority) and raising host-write
    witness flags on innocent value-free traces (test_weightsfree_persistence).
    Disarm through the public spelling BEFORE the group goes away so the
    wrapped ``destroy_process_group`` is restored to the original first.
    """

    tl.distributed.disarm()
    if created and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.fixture()
def process_group(tmp_path: Path) -> Iterator[None]:
    """Single-rank gloo group, disarmed and destroyed on teardown.

    The historical helper left the default group INITIALIZED for the rest of
    the session, which flipped ``warn_parallel``'s distributed-rank exception
    on for every later test in the process (the round-3 child-process refusal
    reds: a fake child read as a rank because ``dist.is_initialized()`` was
    still True). Only a group THIS module created is destroyed. Teardown also
    disarms the lazily-armed distributed capture (see
    :func:`_teardown_process_group`).
    """

    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    created = not torch.distributed.is_initialized()
    if created:
        init_file = tmp_path / "ddp_init"
        torch.distributed.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )
    try:
        yield
    finally:
        _teardown_process_group(created)


def _record_under_group(*args: object, **kwargs: object):
    """Call tl.fastlog.record, expecting the unvetted-torch disclosure.

    Capturing under a genuinely initialized process group lazily arms
    collective capture; on a census-vetted torch build that is silent, and on
    an unvetted build (F1, Lead ruling 2026-10-01) it degrades and warns on
    every capture entry -- the correct, honest product behavior (see
    test_distributed_boundary_gloo.py::TestUnvettedTorchRefusesArming), not a
    gap to route around.
    """

    if has_vetted_snapshot():
        return tl.fastlog.record(*args, **kwargs)
    with pytest.warns(UserWarning, match="uncaptured_collective_op"):
        return tl.fastlog.record(*args, **kwargs)


def test_ddp_wrapped_model_records_unwrapped_module(process_group: None) -> None:
    """This exercises only the unwrapped .module, NOT DDP forward semantics."""

    ddp_model = torch.nn.parallel.DistributedDataParallel(DdpModel())

    recording = _record_under_group(ddp_model, torch.ones(1, 2), default_op=True)

    assert len(recording) > 0


def test_ddp_bundle_path_gets_rank_prefix(process_group: None, tmp_path: Path) -> None:
    """DDP disk bundles are written under a rank_NN prefix."""

    ddp_model = torch.nn.parallel.DistributedDataParallel(DdpModel())
    requested = tmp_path / "bundle.tlfast"

    recording = _record_under_group(
        ddp_model,
        torch.ones(1, 2),
        default_op=True,
        streaming=tl.options.StreamingOptions(bundle_path=requested, retain_in_memory=False),
    )

    assert recording.bundle_path == tmp_path / "rank_00" / "bundle.tlfast"
    assert (tmp_path / "rank_00" / "bundle.tlfast" / "manifest.json").exists()


@pytest.mark.skipif(
    not has_vetted_snapshot(),
    reason="full collective arming requires a census-vetted torch build "
    "(torchlens.distributed.has_vetted_snapshot() is False here)",
)
def test_process_group_teardown_disarms_distributed_capture(process_group: None) -> None:
    """A capture under the group arms distributed capture; teardown must disarm it.

    Regression pin for the session-pollution leak: the arming survived the
    fixture, so every later capture in the process ran under the completeness
    dispatch witness. The teardown helper is exercised directly so the
    contract is checked inside the test body rather than trusted to a
    finalizer nobody asserts on.
    """

    ddp_model = torch.nn.parallel.DistributedDataParallel(DdpModel())
    tl.fastlog.record(ddp_model, torch.ones(1, 2), default_op=True)
    assert tl.distributed.is_armed(), "capture under an initialized group should auto-arm"

    _teardown_process_group(created=False)

    assert not tl.distributed.is_armed()
    assert torch.distributed.is_initialized(), "created=False must not destroy a foreign group"
