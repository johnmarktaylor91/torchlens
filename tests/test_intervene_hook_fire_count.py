"""A capture-time hook fires exactly once per selected op per pass, one clause or many.

``tl.trace(intervene=..., on_zero_fire="error")`` with a callable hook used to
refuse the capture with a false "fired at zero sites while other rules fired"
(the zero-fire settlement compared action reprs, and a callable never renders
like its fire record). These tests pin that the capture completes and that the
hook fires exactly once per selected op in the capture pass and again in a
``trace.run`` replay pass, for a one-clause spec and for a multi-clause spec.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))


def _counting_hook() -> tuple[Callable[..., Any], list[Any]]:
    run_states: list[Any] = []

    def hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        run_states.append(ctx.run_state)
        return out + 1

    return hook, run_states


def _one_clause(hook: Callable[..., Any]) -> Any:
    return tl.when(tl.func("relu"), hook)


def _two_clauses(hook: Callable[..., Any]) -> Any:
    return tl.when(tl.func("linear"), tl.scale(0.5)) & tl.when(tl.func("relu"), hook)


@pytest.mark.parametrize("build", [_one_clause, _two_clauses], ids=["one_clause", "two_clauses"])
def test_hook_fires_once_per_selected_op_per_pass(build: Callable[[Any], Any]) -> None:
    hook, run_states = _counting_hook()
    x = torch.randn(3, 4)
    trace = tl.trace(_model(), x, intervene=build(hook), on_zero_fire="error")
    selected = [op for op in trace.ops if op.func_name == "relu"]
    assert len(selected) == 2
    assert len(run_states) == len(selected)
    assert all(state is run_states[0] for state in run_states)
    assert all(op.is_edited for op in selected)

    capture_fires = len(run_states)
    trace.run(_model(), x)
    replay = run_states[capture_fires:]
    assert len(replay) == len(selected)
    assert all(state is replay[0] for state in replay)
    assert replay[0] is not run_states[0]
