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
        inputs=x,
        intervention=InterventionOptions(engine="rerun"),
    )


class _EvictionProbe:
    """Profile hook counting payload eviction walks, attributed to the store they walk.

    Records the ids of the op stores constructed while it is installed (``made``) and,
    per store id, the calls into the cell eviction walk made by that store's
    last-owner eviction (``walks``). Only ids are kept, so the probe pins nothing it
    observes and cannot keep a discarded store from reaching the collector.
    """

    def __init__(self) -> None:
        self.made: set[int] = set()
        self.walks: dict[int, int] = {}
        self._evicting: list[int] = []

    def __call__(self, frame: Any, event: str, _arg: Any) -> None:
        code = frame.f_code
        if event == "call":
            if code is _STORE_INIT:
                self.made.add(id(frame.f_locals["self"]))
            elif code is _STORE_EVICT:
                self._evicting.append(id(frame.f_locals["self"]))
            elif code is _CELL_EVICT and self._evicting:
                store = self._evicting[-1]
                self.walks[store] = self.walks.get(store, 0) + 1
        elif event == "return" and code is _STORE_EVICT:
            self._evicting.pop()

    def calls(self, stores: set[int] | None = None) -> int:
        """Eviction walk calls, over every store or over the stores in ``stores``."""

        return sum(n for key, n in self.walks.items() if stores is None or key in stores)


_STORE_INIT = op_store.OpRowStore.__init__.__code__
_STORE_EVICT = op_store.OpRowStore._evict_tensor_payloads.__code__
_CELL_EVICT = op_store._evict_payloads_in_cell.__code__


def _profiled(action: Callable[[], Any]) -> _EvictionProbe:
    """Run ``action`` under a fresh eviction probe and return the probe."""

    probe = _EvictionProbe()
    sys.setprofile(probe)
    try:
        action()
    finally:
        sys.setprofile(None)
    return probe


def test_nothing_the_rerun_discarded_is_left_for_the_collector_to_evict() -> None:
    model, x, trace = _capture()
    gc.collect()
    forks: list[Any] = []
    rerun = _profiled(lambda: forks.append(_rerun_fork(trace, model, x)))
    assert rerun.made, "the rerun builds its own capture's op store"
    assert rerun.calls(rerun.made) > 0, "the rerun evicts its discarded capture's payloads itself"
    assert forks[0].last_run["fast_refresh"] is True
    # Only the stores this rerun built are its garbage: a store an earlier test on the
    # same worker left reachable (and released while the rerun ran) is not, and its
    # eviction says nothing about the rerun.
    later = _profiled(gc.collect)
    assert later.calls(rerun.made) == 0, "no eviction walk is left for a later collection"


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
