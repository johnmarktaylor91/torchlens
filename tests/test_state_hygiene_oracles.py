"""Standing state-hygiene oracles for every intervention entry point.

Two oracles, applied to each entry point that stores or applies a spec:

1. REPEAT EQUALS FRESH: calling the same door again with the same inputs yields
   the same numbers as a fresh capture, and the staged spec does not grow.
2. PRISTINE MODEL: after the call, the user's model carries no extra hooks,
   no changed flags and no grads; the capture globals are released.

These are oracles over observable results, never a tolerance change in validation.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state

pytestmark = pytest.mark.smoke


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _hook_counts(model: nn.Module) -> dict[str, int]:
    return {
        name: len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
        for name, m in model.named_modules()
    }


def _model_snapshot(model: nn.Module) -> dict:
    return {
        "hooks": _hook_counts(model),
        "training": {n: m.training for n, m in model.named_modules()},
        "requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
        "grads": {n: p.grad is not None for n, p in model.named_parameters()},
    }


def _capture_globals_released() -> bool:
    return (
        _state._active_trace is None
        and _state._active_hook_plan is None
        and _state._active_intervention_spec is None
        and _state._capture_reserved_by is None
    )


def _spec_size(trace: tl.Trace) -> tuple[int, int, int]:
    spec = trace._intervention_spec
    return (len(spec.targets), len(spec.target_value_specs), len(spec.hook_specs))


@pytest.fixture
def net() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = model.fc2(2.0 * torch.relu(model.fc1(x)))
    return model, x, expected


SPEC = tl.when(tl.module("fc1"), tl.scale(2.0))


def _out(trace: tl.Trace) -> torch.Tensor:
    return trace.raw_output


@pytest.mark.parametrize("repeats", [3])
def test_legacy_rerun_repeat_equals_fresh(net, repeats: int) -> None:
    model, x, expected = net
    trace = tl.trace(model, x, intervene=SPEC)
    size = _spec_size(trace)
    for _ in range(repeats):
        trace.run(model, x)
        torch.testing.assert_close(_out(trace), expected)
        assert _spec_size(trace) == size, "staged spec grew across reruns"


def test_fork_do_rerun_repeat_equals_fresh(net) -> None:
    model, x, expected = net
    base = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    fork = base.fork()
    fork.do(tl.module("fc1"), tl.scale(2.0))
    size = _spec_size(fork)
    for _ in range(3):
        fork.run(model, x)
        torch.testing.assert_close(_out(fork), expected)
        assert _spec_size(fork) == size


def test_attach_rerun_then_remove_restores_baseline(net) -> None:
    model, x, _ = net
    trace = tl.trace(model, x)
    baseline = _out(trace).clone()
    with trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True):
        trace.run(model, x)
        trace.run(model, x)
    assert _spec_size(trace)[2] == 0, "handle exit left sticky hook copies behind"
    trace.run(model, x)
    torch.testing.assert_close(_out(trace), baseline)


def test_failed_rerun_leaves_spec_unchanged(net) -> None:
    model, x, expected = net
    trace = tl.trace(model, x, intervene=SPEC)
    size = _spec_size(trace)

    class _Boom(Exception):
        pass

    def raising_forward(*_a, **_k):
        raise _Boom("injected")

    original = model.fc2.forward
    model.fc2.forward = raising_forward
    try:
        with pytest.raises(_Boom):
            trace.run(model, x)
    finally:
        model.fc2.forward = original
    assert _spec_size(trace) == size, "a failed rerun mutated the staged spec"
    trace.run(model, x)
    torch.testing.assert_close(_out(trace), expected)


def test_capture_time_func_selector_is_rerunnable(net) -> None:
    model, x, expected = net
    trace = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.scale(2.0)))
    trace.run(model, x)  # must not refuse with LiveModeLabelError
    torch.testing.assert_close(_out(trace), expected)


def test_saved_intervened_trace_does_not_rerun_silently_unintervened(net, tmp_path) -> None:
    model, x, expected = net
    trace = tl.trace(model, x, intervene=SPEC)
    trace.run(model, x)
    path = tmp_path / "t.tlspec"
    trace.save(path)
    loaded = tl.load(path)
    if _spec_size(loaded)[2] == 0:
        # The spec was dropped on save: a rerun must then refuse or warn,
        # never return un-intervened numbers for an intervened artifact.
        with pytest.raises(Exception, match=r"."):
            loaded.run(model, x)
    else:
        loaded.run(model, x)
        torch.testing.assert_close(_out(loaded), expected)


@pytest.mark.parametrize(
    "door",
    ["trace", "run", "record", "bind", "fork_do", "attach_detach", "failing_capture"],
)
def test_entry_point_leaves_model_and_globals_pristine(net, door: str) -> None:
    model, x, _ = net
    tl.trace(model, x)  # one-time lazy wrap and persistent model prep
    before = _model_snapshot(model)
    if door == "trace":
        tl.trace(model, x, intervene=SPEC)
    elif door == "run":
        tl.trace(model, x, intervene=SPEC).run(model, x)
    elif door == "record":
        tl.record(model, x, save=tl.func("relu"), intervene=SPEC)
    elif door == "bind":
        SPEC.bind(model)(x)
    elif door == "fork_do":
        fork = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
        fork.do(tl.module("fc1"), tl.scale(2.0))
        fork.run(model, x)
    elif door == "attach_detach":
        trace = tl.trace(model, x)
        handle = trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True)
        trace.run(model, x)
        handle.remove()
    elif door == "failing_capture":

        def boom(t, *, hook):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            tl.trace(model, x, intervene=tl.when(tl.module("fc1"), boom))
    assert _model_snapshot(model) == before
    assert _capture_globals_released()
