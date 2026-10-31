"""A fast-refreshed rerun fork's host-RNG verdict covers the rerun itself.

A same-shape rerun refreshes the fork's ops in place and keeps the fork's own
runnable record. The host-RNG verdict in that record describes the source
capture, so a rerun whose edit draws host RNG must add its draw to it:
otherwise the fork reports that no host RNG was read when its values came
from a forward pass that read it. The merged verdict is the union of the
source capture's and the rerun's. A rerun that draws nothing leaves the
verdict exactly as it was, and the source trace is never changed.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import Trace
from torchlens.intervention.rerun import _merge_rerun_host_rng_verdict
from torchlens.options import CaptureOptions, InterventionOptions

_FIELDS = (
    "host_rng_consumed",
    "host_rng_unreplayable",
    "host_rng_channels",
    "host_rng_replayable_reads",
    "rng_monitor_uncertain",
    "rng_monitor_uncertain_detail",
)


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


class _DrawingNet(_Net):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)) * (1.0 + 0.0 * random.random()))


def _verdict(trace: Trace) -> dict[str, Any]:
    return {name: getattr(trace._runnable, name) for name in _FIELDS}


def _capture(model: nn.Module) -> tuple[torch.Tensor, Trace]:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    return x, tl.trace(model, x, capture=CaptureOptions(replayable=True))


def _rerun_with(trace: Trace, model: nn.Module, x: torch.Tensor, edit: Any) -> Trace:
    return trace.fork().do(
        tl.func("relu"),
        edit,
        model=model,
        x=x,
        intervention=InterventionOptions(engine="rerun"),
    )


def _drawing_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
    return out * (2.0 + 0.0 * random.random())


def test_a_rerun_hook_that_draws_host_rng_marks_the_fork() -> None:
    torch.manual_seed(0)
    model = _Net().eval()
    x, trace = _capture(model)
    before = _verdict(trace)
    assert before["host_rng_consumed"] is False
    fork = _rerun_with(trace, model, x, _drawing_hook)
    assert fork.last_run["fast_refresh"] is True
    assert fork._runnable.host_rng_consumed is True
    assert _verdict(trace) == before


def test_a_rerun_that_draws_nothing_keeps_the_verdict() -> None:
    for model_type in (_Net, _DrawingNet):
        torch.manual_seed(0)
        model = model_type().eval()
        x, trace = _capture(model)
        fork = _rerun_with(trace, model, x, tl.scale(2.0))
        assert fork.last_run["fast_refresh"] is True
        assert _verdict(fork) == _verdict(trace)


def test_the_merge_is_a_union_with_source_entries_first() -> None:
    torch.manual_seed(0)
    model = _Net().eval()
    _, source = _capture(model)
    _, rerun = _capture(model)
    source._runnable.host_rng_consumed = False
    source._runnable.host_rng_channels = ("a", "b")
    source._runnable.rng_monitor_uncertain = None
    source._runnable.rng_monitor_uncertain_detail = None
    rerun._runnable.host_rng_consumed = True
    rerun._runnable.host_rng_channels = ("c", "a")
    rerun._runnable.rng_monitor_uncertain = None
    rerun._runnable.rng_monitor_uncertain_detail = None
    _merge_rerun_host_rng_verdict(source, rerun)
    assert source._runnable.host_rng_consumed is True
    assert source._runnable.host_rng_channels == ("a", "b", "c")
    assert source._runnable.rng_monitor_uncertain is None
    assert source._runnable.rng_monitor_uncertain_detail is None
