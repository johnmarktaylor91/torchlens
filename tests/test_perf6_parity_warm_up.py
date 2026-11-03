"""The parity harness warms every intra-op thread with one tanh before any model work.

The first float ``torch.tanh`` of a process can compute one intra-op thread's
chunk by a different kernel path, so whichever of the baseline capture and the
plain forward runs first may differ from the other in a fresh process. The
parity subcommand runs one warm-up tanh first and records it in its
environment block; these tests pin the order and the record, not the drift.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest
import torch

from benchmarks import head_sweep_engine_cost as harness


def test_warm_up_covers_every_intra_op_thread_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
    seen: list[torch.Size] = []
    real_tanh = torch.tanh

    def spy(tensor: torch.Tensor) -> torch.Tensor:
        seen.append(tensor.shape)
        return real_tanh(tensor)

    monkeypatch.setattr(torch, "tanh", spy)
    record = harness.warm_up_intra_op_threads()
    threads = torch.get_num_threads()
    assert record == {
        "kind": "torch.tanh",
        "shape": [threads, 32768],
        "torch_threads": threads,
        "MKL_CBWR": "COMPATIBLE",
    }
    assert seen == [torch.Size([threads, 32768])]
    assert harness._environment(record)["warm_up"] == record
    assert harness._environment(record)["thread_env"]["MKL_CBWR"] == "COMPATIBLE"


def test_parity_warms_up_before_the_model_and_keeps_its_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    record = {"kind": "torch.tanh", "shape": [1, 32768], "torch_threads": 1, "MKL_CBWR": None}

    def fake_warm_up() -> dict[str, Any]:
        order.append("warm_up")
        return record

    class _Stop(Exception):
        pass

    def fake_build() -> None:
        order.append("build_gpt2")
        raise _Stop

    monkeypatch.setattr(harness, "warm_up_intra_op_threads", fake_warm_up)
    monkeypatch.setattr(harness, "build_gpt2", fake_build)
    with pytest.raises(_Stop):
        harness.cmd_parity(argparse.Namespace(limit=1, stop_on_baseline_drift=True))
    assert order == ["warm_up", "build_gpt2"]
    # The warm-up is measurement setup: the bars stay where they were declared.
    assert harness.PARITY_ATOL == 1e-5
    assert harness.CONTROL_MIN_DIFF == 1e-4
