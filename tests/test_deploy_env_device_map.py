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

from torchlens.report import doctor as report


def test_device_map_row_absent_reads_pass_ok() -> None:
    """No hf_device_map: undetected pass/ok."""

    row = report(nn.Linear(4, 2), torch.randn(1, 4)).check("accelerate_device_map_auto")
    assert (row.is_detected, row.status, row.severity) == (False, "pass", "ok")


@pytest.mark.smoke
def test_device_map_row_single_execution_device_reads_pass_info() -> None:
    """A pure-CPU map (and cpu/disk offload maps) is the supported envelope."""

    model = nn.Linear(4, 2)
    model.hf_device_map = {"": "cpu"}
    row = report(model, torch.randn(1, 4)).check("accelerate_device_map_auto")
    assert (row.is_detected, row.status, row.severity) == (True, "pass", "info")

    offload_model = nn.Linear(4, 2)
    offload_model.hf_device_map = {"encoder": 0, "decoder": "cpu", "head": "disk"}
    offload_row = report(offload_model, torch.randn(1, 4)).check("accelerate_device_map_auto")
    assert (offload_row.is_detected, offload_row.status) == (True, "pass")


def test_device_map_row_multi_device_stays_known_broken() -> None:
    """Two real execution devices: cross-device capture is not yet verified."""

    model = nn.Linear(4, 2)
    model.hf_device_map = {"encoder": 0, "decoder": 1}
    row = report(model, torch.randn(1, 4)).check("accelerate_device_map_auto")
    assert (row.is_detected, row.status, row.severity) == (True, "known_broken", "error")
