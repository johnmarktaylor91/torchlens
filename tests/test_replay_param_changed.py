"""Replay refuses a parameter changed since capture; ``allow_changed_params`` opts in.

A replay template holds the live ``nn.Parameter``, so an edit after capture would make every
later replay of the trace, and of each fork, compute with the new weight. The capture records
each parameter's version counter, storage pointer and shape; replay refuses typed
(``replay_param_changed_since_capture``) on a change, on every replay door and fork kind.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import ReplayPreconditionError
from torchlens.options import InterventionOptions, ReplayOptions

CODE = "replay_param_changed_since_capture"
RUN_KEY = "replayed_with_changed_params"


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


def _capture() -> tuple[nn.Module, torch.Tensor, tl.Trace]:
    torch.manual_seed(0)
    model = _Mlp()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    return model, x, trace


def _in_place_no_grad(model: nn.Module) -> None:
    with torch.no_grad():
        model.fc2.weight.add_(1.0)


def _load_state_dict(model: nn.Module) -> None:
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state["fc2.weight"] += 1.0
    model.load_state_dict(state)


def _optimizer_step(model: nn.Module) -> None:
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    model.fc2.weight.grad = -torch.ones_like(model.fc2.weight)
    optimizer.step()


def _data_assignment(model: nn.Module) -> None:
    model.fc2.weight.data = model.fc2.weight.detach().clone() + 1.0


def _detach_alias(model: nn.Module) -> None:
    model.fc2.weight.detach().add_(1.0)


EDITS: dict[str, Callable[[nn.Module], None]] = {
    "in_place_no_grad": _in_place_no_grad,
    "load_state_dict": _load_state_dict,
    "optimizer_step": _optimizer_step,
    "data_assignment": _data_assignment,
    "detach_alias": _detach_alias,
}


def _relu_label(trace: tl.Trace) -> str:
    return next(str(op.label) for op in trace.layer_list if op.func_name == "relu")


def _double(out: torch.Tensor, *, hook) -> torch.Tensor:
    return out * 2.0


def _replay_fork_do(trace: tl.Trace, **options: bool) -> tl.Trace:
    return trace.fork().do(
        tl.label(_relu_label(trace)),
        _double,
        intervention=InterventionOptions(engine="replay", **options),
    )


def _push(trace: tl.Trace, **options: bool) -> tl.Trace:
    fork = trace.fork()
    return fork.push(replay=ReplayOptions(hooks={tl.label(_relu_label(trace)): _double}, **options))


def _push_from(trace: tl.Trace, **options: bool) -> tl.Trace:
    fork = trace.fork()
    origin = fork.layer_dict_all_keys[_relu_label(trace)]
    origin._internal_set("out", origin.out * 2.0)
    return fork.push_from(origin, replay=ReplayOptions(**options))


def _differentiable_push(trace: tl.Trace, **options: bool) -> tl.Trace:
    return trace.fork().push(
        replay=ReplayOptions(
            differentiable=True, hooks={tl.label(_relu_label(trace)): _double}, **options
        )
    )


DOORS: dict[str, Callable[..., tl.Trace]] = {
    "do_replay": _replay_fork_do,
    "push": _push,
    "push_from": _push_from,
    "differentiable_push": _differentiable_push,
}


def _expected_output(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model.fc2(2.0 * torch.relu(model.fc1(x)))


@pytest.mark.parametrize("edit", sorted(EDITS))
def test_each_caught_edit_refuses_replay(edit: str) -> None:
    model, _x, trace = _capture()
    EDITS[edit](model)
    with pytest.raises(ReplayPreconditionError) as excinfo:
        _replay_fork_do(trace)
    assert excinfo.value.fields["code"] == CODE
    assert excinfo.value.fields["param"] == "fc2.weight"


@pytest.mark.parametrize("door", sorted(DOORS))
def test_every_replay_door_refuses_after_an_edit(door: str) -> None:
    model, _x, trace = _capture()
    _in_place_no_grad(model)
    with pytest.raises(ReplayPreconditionError) as excinfo:
        DOORS[door](trace)
    assert excinfo.value.fields["code"] == CODE


@pytest.mark.parametrize("door", sorted(DOORS))
def test_opt_in_replays_current_weights_and_records_it(door: str) -> None:
    model, x, trace = _capture()
    _in_place_no_grad(model)
    result = DOORS[door](trace, allow_changed_params=True)
    out = str(trace.output_layers[0])
    assert torch.allclose(result.layer_dict_all_keys[out].out.detach(), _expected_output(model, x))
    assert result.last_run[RUN_KEY] == ("fc2.weight",)


@pytest.mark.parametrize("door", sorted(DOORS))
def test_unedited_replay_is_unchanged(door: str) -> None:
    model, x, trace = _capture()
    result = DOORS[door](trace)
    out = str(trace.output_layers[0])
    assert torch.allclose(result.layer_dict_all_keys[out].out.detach(), _expected_output(model, x))
    assert RUN_KEY not in result.last_run


def test_a_later_push_on_a_replay_fork_refuses() -> None:
    """Replay forks share the templates, so an edit after the fork still refuses."""

    model, _x, trace = _capture()
    fork = _replay_fork_do(trace)
    _in_place_no_grad(model)
    with pytest.raises(ReplayPreconditionError, match=CODE):
        _replay_fork_do(fork)


def test_a_rerun_fork_refuses_a_later_edit_but_not_its_own_weights() -> None:
    """A rerun runs the live model; its fork refuses only edits made after the rerun."""

    model, x, trace = _capture()
    _in_place_no_grad(model)
    rerun = trace.fork().do(
        tl.func("relu"),
        _double,
        model=model,
        x=x,
        intervention=InterventionOptions(engine="rerun"),
    )
    out = str(trace.output_layers[0])
    assert torch.allclose(rerun.layer_dict_all_keys[out].out, _expected_output(model, x))
    pushed = _replay_fork_do(rerun)
    assert RUN_KEY not in pushed.last_run
    _in_place_no_grad(model)
    with pytest.raises(ReplayPreconditionError, match=CODE):
        _replay_fork_do(rerun)


def test_an_edit_through_param_data_is_not_caught() -> None:
    """Documented gap (known gotchas): ``.data`` has its own version counter."""

    model, x, trace = _capture()
    model.fc2.weight.data.add_(1.0)
    result = _replay_fork_do(trace)
    out = str(trace.output_layers[0])
    assert torch.allclose(result.layer_dict_all_keys[out].out, _expected_output(model, x))


def test_a_replaced_parameter_object_keeps_the_captured_values() -> None:
    """The template holds the captured object, so replay is still faithful to capture."""

    model, x, trace = _capture()
    expected = _expected_output(model, x)
    model.fc2.weight = nn.Parameter(model.fc2.weight.detach() + 1.0)
    result = _replay_fork_do(trace)
    out = str(trace.output_layers[0])
    assert torch.allclose(result.layer_dict_all_keys[out].out, expected)
