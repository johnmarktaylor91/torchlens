"""A fast-refresh rerun releases the capture it discarded before it returns.

The rerun's own capture is garbage once its values are copied into the fork. It sits
in reference cycles, so without an explicit release the last-owner payload eviction of
its op store (a walk over every cell) ran whenever the cyclic collector freed it,
inside whatever executed then: on a 12-layer GPT-2 head sweep, the next ``fork()``
made about 1.25 million Python calls instead of about 419,000. The rerun now pays that
cost itself, and nothing about what a fork or the source trace exposes changes.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core import op_store
from torchlens.options import CaptureOptions, InterventionOptions


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _capture() -> tuple[_Net, torch.Tensor, Any]:
    torch.manual_seed(0)
    model = _Net()
    x = torch.randn(2, 4)
    return model, x, tl.trace(model, x, capture=CaptureOptions(replayable=True))


def _rerun_fork(trace: Any, model: nn.Module, x: torch.Tensor) -> Any:
    return trace.fork().do(
        tl.func("relu"),
        tl.scale(2.0),
        model=model,
        x=x,
        intervention=InterventionOptions(engine="rerun"),
    )


def _eviction_calls(action: Callable[[], Any]) -> int:
    """Calls into the payload eviction walk made while ``action`` runs."""

    target = op_store._evict_payloads_in_cell.__code__
    calls = 0

    def count(frame: Any, event: str, _arg: Any) -> None:
        nonlocal calls
        if event == "call" and frame.f_code is target:
            calls += 1

    sys.setprofile(count)
    try:
        action()
    finally:
        sys.setprofile(None)
    return calls


def test_nothing_the_rerun_discarded_is_left_for_the_collector_to_evict() -> None:
    model, x, trace = _capture()
    gc.collect()
    forks: list[Any] = []
    assert _eviction_calls(lambda: forks.append(_rerun_fork(trace, model, x))) > 0, (
        "the rerun evicts its discarded capture's payloads itself"
    )
    assert forks[0].last_run["fast_refresh"] is True
    assert _eviction_calls(gc.collect) == 0, "no eviction walk is left for a later collection"


def test_values_after_the_release_are_unchanged() -> None:
    model, x, trace = _capture()
    recorded = {op.label: op.out.detach().clone() for op in trace._layer_list}
    fork = _rerun_fork(trace, model, x)
    gc.collect()
    pushed = trace.fork().do(
        tl.func("relu"), tl.scale(2.0), intervention=InterventionOptions(engine="replay")
    )
    for fork_op, pushed_op in zip(fork._layer_list, pushed._layer_list, strict=True):
        assert fork_op.out is not None, fork_op.label
        assert torch.equal(fork_op.out, pushed_op.out), fork_op.label
    for op in trace._layer_list:
        assert torch.equal(op.out, recorded[op.label]), "the source keeps its payloads"
    later = fork.fork()
    for fork_op, later_op in zip(fork._layer_list, later._layer_list, strict=True):
        assert torch.equal(fork_op.out, later_op.out), "a fork of the rerun fork reads the same"
