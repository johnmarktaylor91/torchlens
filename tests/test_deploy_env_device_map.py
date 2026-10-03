"""Deployment envelope (lane F37): device_map dispatch capture + compat row.

Single-execution-device dispatch (the pure-CPU map, cpu/disk offload maps) is
capture-supported and its compat row reads pass/info; multi-real-device maps
keep the honest known_broken until the GPU campaign (C-DEPLOY) verifies
cross-device moves.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.compat import report


def test_device_map_row_absent_reads_pass_ok() -> None:
    """No hf_device_map: undetected pass/ok."""

    row = report(nn.Linear(4, 2), torch.randn(1, 4)).row("accelerate_device_map_auto")
    assert (row.detected, row.status, row.severity) == (False, "pass", "ok")


def test_device_map_row_single_execution_device_reads_pass_info() -> None:
    """A pure-CPU map (and cpu/disk offload maps) is the supported envelope."""

    model = nn.Linear(4, 2)
    model.hf_device_map = {"": "cpu"}
    row = report(model, torch.randn(1, 4)).row("accelerate_device_map_auto")
    assert (row.detected, row.status, row.severity) == (True, "pass", "info")

    offload_model = nn.Linear(4, 2)
    offload_model.hf_device_map = {"encoder": 0, "decoder": "cpu", "head": "disk"}
    offload_row = report(offload_model, torch.randn(1, 4)).row("accelerate_device_map_auto")
    assert (offload_row.detected, offload_row.status) == (True, "pass")


def test_device_map_row_multi_device_stays_known_broken() -> None:
    """Two real execution devices: cross-device capture is not yet verified."""

    model = nn.Linear(4, 2)
    model.hf_device_map = {"encoder": 0, "decoder": 1}
    row = report(model, torch.randn(1, 4)).row("accelerate_device_map_auto")
    assert (row.detected, row.status, row.severity) == (True, "known_broken", "error")


@pytest.mark.heavy
def test_cpu_device_map_dispatch_captures_and_validates() -> None:
    """dispatch_model on a pure-CPU map traces and passes replay validation."""

    transformers = pytest.importorskip("transformers")
    accelerate = pytest.importorskip("accelerate")
    config = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=128, n_positions=64)
    torch.manual_seed(0)
    model = accelerate.dispatch_model(
        transformers.GPT2LMHeadModel(config).eval(), device_map={"": "cpu"}
    )
    input_ids = torch.randint(0, 128, (1, 8))
    trace = tl.trace(model, input_ids)
    assert trace.capture_verified is None
    assert len(trace.ops) > 0
    assert tl.validate(model, input_ids, scope="forward") is True
