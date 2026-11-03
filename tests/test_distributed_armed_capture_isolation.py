"""Captures in an ARMED process behave like unarmed ones outside collectives.

Distributed arming is process-lifetime by design and survives
``destroy_process_group``; while armed, plane-P installs the completeness
witness dispatch mode around every capture. Two defects surfaced through that
dispatch mode (found as test-order pollution on shared xdist workers):

* the per-consumption state-write sampler matched meta operands on
  ``data_ptr() == 0`` and fabricated the opaque host-write flag, so every
  weights-free capture with buffers refused settlement;
* a user op failing inside the dispatch mode's redispatch was classified as a
  TorchLens failure instead of the user's.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._capture.outcome import classify_failure_origin
from torchlens._distributed import _lifecycle
from torchlens.errors import TraceCaptureError
from torchlens.types import FailureOrigin


@pytest.fixture
def armed_process(tmp_path: Path) -> Iterator[None]:
    """Arm distributed capture lazily under a LIVE one-rank gloo group.

    The group stays initialized for the whole test: armed state is dormant
    without one (plane-P only observes captures that could issue collectives),
    so only a live group puts the dispatch mode around the captures under test.
    """

    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        pytest.skip("needs torch.distributed and no pre-existing process group")
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'pg_init'}", rank=0, world_size=1
    )
    try:
        # The capture's only purpose is the lazy arm; an unvetted torch build
        # warns and degrades to unarmed capture (skipped below).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl.trace(nn.Linear(2, 2), torch.randn(1, 2))
        if not _lifecycle.is_armed():
            pytest.skip("lazy arming refused on this torch build (unvetted recognizer)")
        assert _lifecycle.capture_armed_state() is not None
        yield
    finally:
        _lifecycle.disarm()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


# Kept by shrink-r4 (per-id subset rule): sole coverage of _capture/outcome.py, backends/torch/completeness_witness.py, campaign mutant kills.
@pytest.mark.usefixtures("armed_process")
def test_user_op_failure_classifies_user_op_while_armed() -> None:
    class ShapeMismatch(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.matmul(x, torch.ones(7, 7))

    with pytest.raises(TraceCaptureError) as excinfo:
        tl.trace(ShapeMismatch(), torch.ones(1, 3))
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert classify_failure_origin(excinfo.value) is FailureOrigin.USER_OP
