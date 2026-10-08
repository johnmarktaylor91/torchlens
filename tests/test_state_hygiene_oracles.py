"""State-hygiene oracles for every door that stores or applies an intervention spec.

Two oracles, applied per entry point:

1. REPEAT EQUALS FRESH: calling the same door again with the same inputs yields
   the numbers an independent expectation gives (here the analytic steered
   forward), and the trace's staged spec does not change.
2. PRISTINE MODEL: after the call the user's model carries no extra hooks, no
   changed flags and no grads, and the capture globals are released.

Validation cannot catch this class: each rerun is self-consistent with the hook
plan it actually ran, so these oracles compare against an expectation computed
outside TorchLens. They are observable-result checks, never a validation change.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state


class _Net(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        """Build the two linears."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.fc2(torch.relu(self.fc1(x)))


def _hook_counts(model: nn.Module) -> dict[str, int]:
    """Count the torch module hooks on every submodule."""

    return {
        name: len(m._forward_hooks) + len(m._forward_pre_hooks) + len(m._backward_hooks)
        for name, m in model.named_modules()
    }


def _model_snapshot(model: nn.Module) -> dict[str, Any]:
    """Return the model state a TorchLens call must leave untouched."""

    return {
        "hooks": _hook_counts(model),
        "training": {n: m.training for n, m in model.named_modules()},
        "requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
        "grads": {n: p.grad is not None for n, p in model.named_parameters()},
    }


def _capture_globals_released() -> bool:
    """Return whether every process-global capture slot is released."""

    return (
        _state._active_trace is None
        and _state._active_hook_plan is None
        and _state._active_intervention_spec is None
        and _state._capture_reserved_by is None
    )


def _staged(trace: Any) -> tuple[Any, ...]:
    """Return the trace's staged spec content (fire records excluded)."""

    spec = trace._intervention_spec
    if spec is None:
        return ()
    frozen = spec.freeze()
    return (
        frozen.targets,
        frozen.helper,
        frozen.value,
        frozen.hook,
        frozen.target_value_specs,
        frozen.hook_specs,
        frozen.metadata,
    )


def _readout(trace: Any) -> torch.Tensor:
    """Return the saved model readout (the fc2 output)."""

    return trace.find_sites(tl.module("fc2")).first().out


def _scale_fc1() -> Any:
    """Return the one-clause spec that doubles the fc1 output."""

    return tl.when(tl.module("fc1"), tl.scale(2.0))


@pytest.fixture
def net() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """A seeded model, an input, and the analytic doubled-fc1 readout."""

    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(4, 8)
    with torch.no_grad():
        expected = model.fc2(2.0 * torch.relu(model.fc1(x)))
        plain = model(x)
    # The edit must move the readout, or exactness below proves nothing.
    assert not torch.allclose(expected, plain)
    return model, x, expected


@pytest.mark.smoke
def test_chunked_rerun_steers_every_row_once(net) -> None:
    """F1: a chunked rerun's rows all match one fresh steered forward."""

    model, x, expected = net
    trace = tl.trace(model, x, intervene=_scale_fc1())
    staged = _staged(trace)
    for _ in range(2):
        trace.run(model, x, replay=tl.options.ReplayOptions(chunk_size=2))
        torch.testing.assert_close(_readout(trace), expected)
        assert _staged(trace) == staged, "a chunked rerun changed the staged spec"


@pytest.mark.smoke
@pytest.mark.parametrize("exit_door", ["context", "remove"])
def test_attach_rerun_then_detach_restores_baseline(net, exit_door: str) -> None:
    """F2: after reruns, removing an attached edit leaves nothing staged."""

    model, x, expected = net
    trace = tl.trace(model, x)
    baseline = _readout(trace).clone()
    if exit_door == "context":
        with trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True):
            trace.run(model, x)
            trace.run(model, x)
            torch.testing.assert_close(_readout(trace), expected)
    else:
        handle = trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True)
        trace.run(model, x)
        trace.run(model, x)
        torch.testing.assert_close(_readout(trace), expected)
        handle.remove()
    assert len(trace._intervention_spec.hook_specs) == 0, "detach left hook copies behind"
    trace.run(model, x)
    torch.testing.assert_close(_readout(trace), baseline)


class _InjectedFailure(RuntimeError):
    """Raised by a patched submodule forward."""


@pytest.mark.smoke
@pytest.mark.parametrize("failure", [_InjectedFailure, KeyboardInterrupt])
def test_failed_rerun_leaves_staged_spec_unchanged(net, failure: type[BaseException]) -> None:
    """F3: a rerun that raises or is interrupted mid-forward stages nothing."""

    model, x, expected = net
    trace = tl.trace(model, x, save=tl.module("fc2"), intervene=_scale_fc1())
    staged = _staged(trace)
    before = _model_snapshot(model)

    def raising_forward(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise failure("injected mid-forward")

    original = model.fc2.forward
    model.fc2.forward = raising_forward
    try:
        with pytest.raises(failure):
            trace.run(model, x)
    finally:
        model.fc2.forward = original
    assert _staged(trace) == staged, "a failed rerun mutated the staged spec"
    assert _model_snapshot(model) == before
    assert _capture_globals_released()
    trace.run(model, x)
    torch.testing.assert_close(_readout(trace), expected)
    assert _staged(trace) == staged


@pytest.mark.smoke
def test_portable_spec_saved_after_reruns_carries_no_duplicates(net, tmp_path) -> None:
    """F4: a spec saved after reruns holds exactly the staged entries."""

    from torchlens.io import load_intervention_spec

    model, x, _ = net
    trace = tl.trace(model, x, intervene=_scale_fc1())
    staged_hooks = len(trace._intervention_spec.hook_specs)
    trace.run(model, x)
    trace.run(model, x)
    path = tmp_path / "spec.tlspec"
    trace.save_intervention(path, level="portable")
    loaded = load_intervention_spec(path)
    assert len(loaded.hook_specs) == staged_hooks
    assert len(loaded.target_value_specs) == len(trace._intervention_spec.target_value_specs)


@pytest.mark.smoke
def test_callable_set_rerun_repeat_equals_fresh(net) -> None:
    """F5: a callable ``set`` replacement applies once per rerun, never compounds."""

    model, x, expected = net
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    # set() only stages the replacement; each run applies it.
    trace.set(tl.module("fc1"), lambda out: out * 2.0, confirm_mutation=True)
    staged = _staged(trace)
    for _ in range(3):
        trace.run(model, x)
        torch.testing.assert_close(_readout(trace), expected)
        assert _staged(trace) == staged


@pytest.mark.smoke
def test_loaded_intervened_trace_refuses_unintervened_rerun(net, tmp_path) -> None:
    """F6: a reloaded intervened trace never reruns silently un-intervened."""

    from torchlens.intervention.errors import EngineDispatchError

    model, x, _ = net
    trace = tl.trace(model, x, intervene=_scale_fc1())
    path = tmp_path / "t.tlspec"
    with pytest.warns(tl.errors.TorchLensWarning, match="save_intervention"):
        trace.save(path)
    loaded = tl.load(path)
    with pytest.raises(EngineDispatchError) as excinfo:
        loaded.run(model, x)
    assert excinfo.value.fields["code"] == "run_intervention_spec_not_persisted"


@pytest.mark.smoke
def test_capture_time_func_selector_is_rerunnable(net) -> None:
    """F7: a ``tl.when(tl.func(...))`` capture reruns with the same edit."""

    model, x, expected = net
    trace = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.scale(2.0)))
    torch.testing.assert_close(_readout(trace), expected)
    staged = _staged(trace)
    for _ in range(2):
        trace.run(model, x)
        torch.testing.assert_close(_readout(trace), expected)
        assert _staged(trace) == staged


@pytest.mark.smoke
def test_fork_leaves_parent_spec_untouched(net) -> None:
    """F8: ``fork()`` never creates or changes the parent's staged spec."""

    model, x, expected = net
    parent = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    parent_spec = parent._intervention_spec
    staged = _staged(parent)
    fork = parent.fork()
    fork.do(tl.module("fc1"), tl.scale(2.0))
    torch.testing.assert_close(_readout(fork), expected)
    assert parent._intervention_spec is parent_spec
    assert _staged(parent) == staged
    assert fork._intervention_spec is not parent_spec
    # A parent with no spec object at all stays without one.
    parent._intervention_spec = None
    parent.fork()
    assert parent._intervention_spec is None


@pytest.mark.parametrize(
    "door",
    ["trace", "run", "chunked", "record", "bind", "fork_do", "attach_detach", "failing_capture"],
)
def test_entry_point_leaves_model_and_globals_pristine(net, door: str) -> None:
    """Every door leaves the user's model and the capture globals as it found them."""

    model, x, _ = net
    tl.trace(model, x)  # one-time lazy wrap and persistent model prep
    before = _model_snapshot(model)
    if door == "trace":
        tl.trace(model, x, intervene=_scale_fc1())
    elif door == "run":
        tl.trace(model, x, intervene=_scale_fc1()).run(model, x)
    elif door == "chunked":
        tl.trace(model, x, intervene=_scale_fc1()).run(
            model, x, replay=tl.options.ReplayOptions(chunk_size=2)
        )
    elif door == "record":
        tl.record(model, x, save=tl.func("relu"), intervene=_scale_fc1())
    elif door == "bind":
        with torch.no_grad():
            _scale_fc1().bind(model)(x)
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

        def boom(_out: torch.Tensor, *, hook: Any) -> torch.Tensor:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            tl.trace(model, x, intervene=tl.when(tl.module("fc1"), boom))
    assert _model_snapshot(model) == before
    assert _capture_globals_released()
