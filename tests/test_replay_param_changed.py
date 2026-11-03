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
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
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
    return next(str(op.label) for op in trace._layer_list if op.func_name == "relu")


def _double(out: torch.Tensor, *, ctx) -> torch.Tensor:
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
    origin = fork._layer_dict_all_keys[_relu_label(trace)]
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


@pytest.mark.smoke_cells("test_opt_in_replays_current_weights_and_records_it[differentiable_push]")
@pytest.mark.parametrize("door", sorted(DOORS))
def test_opt_in_replays_current_weights_and_records_it(door: str) -> None:
    model, x, trace = _capture()
    _in_place_no_grad(model)
    result = DOORS[door](trace, allow_changed_params=True)
    out = str(trace.output_layers[0])
    assert torch.allclose(result._layer_dict_all_keys[out].out.detach(), _expected_output(model, x))
    assert result.last_run[RUN_KEY] == ("fc2.weight",)


@pytest.mark.parametrize("engine", ["replay", "rerun"])
def test_allow_changed_params_never_changes_the_capture(engine: str) -> None:
    """Cache-key dont-care witness: the opt-in only gates replay's parameter check.

    ``tl.trace`` takes neither options group, and a ``do()`` that captures (rerun) or
    replays an unedited model produces the same trace under both values.
    """

    model, x, trace = _capture()
    results = [
        trace.fork().do(
            tl.func("relu"),
            _double,
            model=model,
            inputs=x,
            intervention=InterventionOptions(engine=engine, allow_changed_params=allow),
        )
        for allow in (False, True)
    ]
    labels = [[str(op.label) for op in result._layer_list] for result in results]
    assert labels[0] == labels[1]
    for left, right in zip(results[0]._layer_list, results[1]._layer_list, strict=True):
        if isinstance(left.out, torch.Tensor):
            assert torch.equal(left.out, right.out), left.label
    assert all(RUN_KEY not in result.last_run for result in results)
