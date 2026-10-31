"""Independent-clock and event-recount oracles for the legacy durations and witness counters.

A duration has no closed form, so each is bounded by a clock TorchLens does not
read: ``time.perf_counter`` around the capture (an upper bound) and a planted
``time.sleep`` inside the forward or a backward hook (a lower bound, and a gap
the function-call totals must exclude). Thread counts are recounted with
``threading.active_count()``; the completeness witness's ATen events with a
``TorchDispatchMode`` and the escape detector's torch-function events with a
``TorchFunctionMode`` on plain torch. These are the I1 oracles the census
(``tests/oracles/data/legacy_quantities.tsv``) names for those doors.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

import torchlens as tl
from tests.oracles.test_legacy_graph_counts import Grad, Graph
from tests.oracles.test_legacy_module_counts import Cond
from torchlens import _state
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.options import CaptureOptions

SLEEP = 0.05  # seconds planted in Sleeper.forward
BACKWARD_SLEEP = 0.03  # seconds planted in a backward hook


class Sleeper(nn.Module):
    """Sleeps, then runs a Linear: the sleep is inside this call, outside every op."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time.sleep(SLEEP)
        return self.fc(x)


class Sleepy(nn.Module):
    """Root -> Sleeper -> Linear, then a ReLU at the root."""

    def __init__(self) -> None:
        super().__init__()
        self.s = Sleeper()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.s(x).relu()


@pytest.fixture(scope="module")
def timed_trace():
    torch.manual_seed(0)
    model = Sleepy()
    started = time.perf_counter()
    trace = tl.trace(model, torch.randn(2, 4))
    wall = time.perf_counter() - started
    yield trace, wall
    trace.cleanup()


def test_capture_phase_durations_are_bounded_by_an_outside_clock(timed_trace) -> None:
    trace, wall = timed_trace
    capture = float(trace.capture_duration)
    setup, forward, cleanup = (
        float(trace.setup_duration),
        float(trace.forward_duration),
        float(trace.cleanup_duration),
    )
    assert SLEEP <= forward <= capture <= wall
    assert 0 < setup <= wall and 0 < cleanup <= wall
    assert setup + forward + cleanup <= wall
    func_calls = float(trace.func_calls_duration)
    # The planted sleep is not a function call: the op total excludes it.
    assert 0 < func_calls <= forward - SLEEP
    assert SLEEP <= float(trace.cost.overhead_duration) <= wall


def test_module_and_call_durations_are_bounded(timed_trace) -> None:
    trace, wall = timed_trace
    root, sleeper, fc = (trace.module_calls[label] for label in ("self:1", "s:1", "s.fc:1"))
    assert SLEEP <= float(sleeper.forward_duration) <= float(root.forward_duration) <= wall
    # fc runs after the sleep, inside Sleeper's call.
    assert 0 < float(fc.forward_duration) <= float(sleeper.forward_duration) - SLEEP
    for call in (root, sleeper, fc):
        assert 0 < float(call.func_calls_duration) <= float(call.forward_duration)
    assert float(sleeper.func_calls_duration) <= float(sleeper.forward_duration) - SLEEP
    for address, call in (("self", root), ("s", sleeper), ("s.fc", fc)):
        module = trace.modules[address]
        assert float(module.forward_duration) == pytest.approx(float(call.forward_duration))
        assert float(module.total_forward_duration) <= wall
        assert float(module.total_forward_duration) >= SLEEP or address == "s.fc"
        assert 0 < float(module.func_calls_duration) <= float(module.total_forward_duration)
        assert float(module.total_func_calls_duration) <= float(module.total_forward_duration)


def test_op_and_layer_func_durations_are_bounded(timed_trace) -> None:
    trace, wall = timed_trace
    budget = float(trace.forward_duration) - SLEEP  # ops run outside the sleep
    total = 0.0
    for op in trace.ops:
        duration = float(op.func_duration)
        if op.is_input or op.is_output:
            assert duration == 0.0, op.label  # boundary nodes run no function
            continue
        assert 0 < duration <= budget, op.label
        total += duration
        layer = trace.layers[op.layer_label]
        assert 0 < float(layer.func_duration) <= budget
        assert 0 < float(layer.total_func_duration) <= budget
    assert total <= budget <= wall


def test_backward_durations_are_bounded_by_an_outside_clock() -> None:
    torch.manual_seed(0)
    trace = tl.trace(Grad(), torch.randn(2, 4), capture=CaptureOptions(save_grads=True))
    try:
        out = trace[trace.output_layers[0]].out
        out.register_hook(lambda grad: (time.sleep(BACKWARD_SLEEP), grad)[1])
        walls = []
        for retain in (True, False):
            started = time.perf_counter()
            trace.backward(out.sum(), retain_graph=retain)
            walls.append(time.perf_counter() - started)
        last = float(trace.cost.last_backward_duration)
        total = float(trace.cost.total_backward_duration)
        assert BACKWARD_SLEEP <= last <= walls[1]
        assert 2 * BACKWARD_SLEEP <= total <= sum(walls)
    finally:
        trace.cleanup()


SPAWNED_THREAD = "legacy-timing-spawned"


class Spawner(nn.Module):
    """Starts one thread that outlives the forward."""

    def __init__(self, stop: threading.Event) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.stop = stop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        threading.Thread(target=self.stop.wait, name=SPAWNED_THREAD, daemon=True).start()
        return self.fc(x)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_capture_thread_counts_match_a_recount() -> None:
    torch.manual_seed(0)
    before = threading.active_count()
    trace = tl.trace(Grad(), torch.randn(2, 4))
    try:
        assert trace.capture_thread_count_start == before
        assert trace.capture_thread_count_end == before  # nothing started a thread
    finally:
        trace.cleanup()
    stop = threading.Event()
    try:
        before = threading.active_count()
        trace = tl.trace(Spawner(stop), torch.randn(2, 4))
        try:
            assert trace.capture_thread_count_start == before
            assert (
                trace.capture_thread_count_end == before + 1
            )  # the one thread the forward started
        finally:
            trace.cleanup()
    finally:
        stop.set()
        for thread in threading.enumerate():
            if thread.name == SPAWNED_THREAD:
                thread.join(timeout=30)


class _DispatchCount(TorchDispatchMode):
    """Counts every ATen dispatch event, the completeness witness's unit."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0
        self.scalar_reads = 0  # tensor-to-Python reads (``if t:``, ``.item()``)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.count += 1
        self.scalar_reads += str(func) == "aten._local_scalar_dense.default"
        return func(*args, **(kwargs or {}))


class _FunctionCount(TorchFunctionMode):
    """Counts every torch-function call, the escape detector's unit."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.count += 1
        return func(*args, **(kwargs or {}))


@pytest.fixture
def _detectors_armed():
    """Arm the escape detector and completeness witness, then restore the prior modes."""

    saved = (_state._escape_detector_mode, _state._completeness_witness_mode)
    unwrap_torch()
    wrap_torch(escape_detector="shadow", completeness_witness=True)
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved[0], completeness_witness=saved[1])


@pytest.mark.usefixtures("_detectors_armed")
@pytest.mark.parametrize("model_name", ["grad", "graph", "cond"])
def test_witness_and_detector_event_counts_match_a_recount(model_name: str) -> None:
    torch.manual_seed(0)
    model = {"grad": Grad, "graph": Graph, "cond": Cond}[model_name]().eval()
    inputs = {
        "grad": (torch.randn(2, 4),),
        "graph": (torch.randn(2, 4), torch.randn(2, 4)),
        "cond": (torch.ones(2, 3),),  # both ifs taken: two bool reads
    }[model_name]
    unwrap_torch()  # recount on plain torch
    with _DispatchCount() as dispatches:
        model(*inputs)
    with _FunctionCount() as functions:
        model(*inputs)
    wrap_torch(escape_detector="shadow", completeness_witness=True)
    started = time.perf_counter()
    trace = tl.trace(model, inputs if len(inputs) > 1 else inputs[0])
    wall_ns = (time.perf_counter() - started) * 1e9
    try:
        assert trace.completeness_witness_event_count == dispatches.count
        # A tensor bool read is an audited opaque boundary; every other
        # dispatch lies under a captured wrapped call and is accounted.
        assert trace.completeness_witness_expected_opaque_count == dispatches.scalar_reads
        assert dispatches.scalar_reads == (2 if model_name == "cond" else 0)
        accounted = dispatches.count - dispatches.scalar_reads
        assert trace.completeness_witness_accounted_count == accounted
        assert trace.escape_detector_event_count == functions.count
        assert 0 < trace.completeness_witness_callback_ns <= wall_ns
        assert 0 < trace.escape_detector_callback_ns <= wall_ns
    finally:
        trace.cleanup()
