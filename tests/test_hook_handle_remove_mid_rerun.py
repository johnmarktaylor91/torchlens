"""A hook removed through its handle during a run stops firing at once.

``HookHandle.remove()`` matches torch's own hook handles: the next hook site
the run reaches after ``remove()`` returns skips the removed hook, inside the
same rerun or backward pass, not from the next one. Hooks still queued at the
site that is firing when ``remove()`` is called finish that site (torch
snapshots the hook list per call the same way). After the run the trace's
recipe and the handle are the same as if the hook had been removed between
runs: no entry for it, no zero-fire disclosure for it.
"""

from __future__ import annotations

import warnings
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning
from torchlens.options import CaptureOptions

pytestmark = pytest.mark.filterwarnings("ignore::torchlens._selection.errors.MultiMatchWarning")


def _model() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU())


def _ready_trace(**capture: Any) -> tuple[nn.Sequential, torch.Tensor, Any]:
    model = _model()
    x = torch.randn(3, 8)
    trace = tl.trace(model, [x], capture=CaptureOptions(replayable=True, **capture))
    return model, x, trace


def _recipe_handles(trace: Any) -> set[str | None]:
    return {spec.handle for spec in trace._intervention_spec.hook_specs}


def test_forward_hook_removed_before_its_site_never_fires_on_that_run() -> None:
    """The probe: a prepend hook on linear removes the relu hook on its first fire."""

    model, x, trace = _ready_trace()
    events: list[str] = []

    def victim(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        events.append("victim")
        return out * 0.5

    relu_handle = trace.attach_hooks(tl.func("relu"), victim, confirm_mutation=True)

    removed: list[bool] = []

    def remover(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        if not removed:
            relu_handle.remove()
            removed.append(True)
            events.append("removed")
        events.append("linear")
        return out

    trace.attach_hooks(tl.func("linear"), remover, prepend=True, confirm_mutation=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", MultiMatchWarning)  # tl.func fans out to both sites
        trace.run(model, x)

    assert events == ["removed", "linear", "linear"], events
    assert trace.last_run["hooks"] == 1
    assert trace.last_run["hooks_fired"] == 2
    assert trace.last_run["hooks_unfired"] == 0
    assert relu_handle.handle_ids[0] not in _recipe_handles(trace)
    # The unhooked output: the relu hook never scaled anything on this run.
    expected = model(x)
    assert torch.allclose(trace[trace.output_layers[0]].out, expected)

    events.clear()
    trace.run(model, x)
    assert events == ["linear", "linear"], events


def test_forward_hook_removing_itself_fires_once() -> None:
    model, x, trace = _ready_trace()
    fired: list[int] = []
    handle_box: list[Any] = []

    def once(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        fired.append(1)
        handle_box[0].remove()
        return out

    handle_box.append(trace.attach_hooks(tl.func("relu"), once, confirm_mutation=True))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", MultiMatchWarning)  # tl.func fans out to both sites
        trace.run(model, x)

    assert len(fired) == 1, "a hook that removed itself fired again at a later site"
    assert trace.last_run["hooks"] == 0
    assert trace.last_run["hooks_fired"] == 1
    assert _recipe_handles(trace) == set()


def test_forward_hook_on_a_passed_site_is_gone_at_its_next_site() -> None:
    """The relu hook removes the linear hook after the first linear already fired."""

    model, x, trace = _ready_trace()
    events: list[str] = []

    def linear_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        events.append("linear")
        return out

    linear_handle = trace.attach_hooks(tl.func("linear"), linear_hook, confirm_mutation=True)

    def relu_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        if "removed" not in events:
            linear_handle.remove()
            events.append("removed")
        events.append("relu")
        return out

    trace.attach_hooks(tl.func("relu"), relu_hook, confirm_mutation=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", MultiMatchWarning)  # tl.func fans out to both sites
        trace.run(model, x)

    assert events == ["linear", "removed", "relu", "relu"], events
    assert trace.last_run["hooks"] == 1
    assert trace.last_run["hooks_fired"] == 3


def test_hooks_later_at_the_firing_site_still_finish_that_site() -> None:
    """torch semantics: the site in progress fires its snapshot; the next site skips."""

    model, x, trace = _ready_trace()
    events: list[str] = []

    def second(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        events.append("second")
        return out

    second_handle = trace.attach_hooks(tl.func("relu"), second, confirm_mutation=True)

    def first(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        if "removed" not in events:
            second_handle.remove()
            events.append("removed")
        events.append("first")
        return out

    trace.attach_hooks(tl.func("relu"), first, prepend=True, confirm_mutation=True)
    trace.run(model, x)

    assert events == ["removed", "first", "second", "first"], events


def test_nothing_removed_keeps_the_rerun_unchanged() -> None:
    model, x, trace = _ready_trace()
    fired: list[int] = []

    def count(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        fired.append(1)
        return out

    trace.attach_hooks(tl.func("relu"), count, confirm_mutation=True)
    trace.run(model, x)
    assert len(fired) == 2
    assert trace.last_run["hooks"] == 1
    assert trace.last_run["hooks_fired"] == 2
    assert trace.last_run["hooks_unfired"] == 0


def test_backward_hook_removed_mid_pass_skips_the_remaining_grad_fns() -> None:
    """The backward door builds its own plan: the relu grad hook removes the addmm one."""

    model = _model()
    x = torch.randn(3, 8, requires_grad=True)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(replayable=True, backward_ready=True, save_grads="all"),
    )
    events: list[str] = []

    def victim(
        grad_input: tuple[torch.Tensor | None, ...],
        *,
        grad_output: tuple[torch.Tensor | None, ...] | None,
        grad_fn_handle: Any,
        call_index: int,
        run_ctx: dict[str, Any],
    ) -> None:
        events.append("addmm")

    addmm_handle = trace.attach_hooks(tl.grad_fn(name="addmm"), victim, confirm_mutation=True)
    removed: list[bool] = []

    def remover(
        grad_input: tuple[torch.Tensor | None, ...],
        *,
        grad_output: tuple[torch.Tensor | None, ...] | None,
        grad_fn_handle: Any,
        call_index: int,
        run_ctx: dict[str, Any],
    ) -> None:
        if not removed:
            addmm_handle.remove()
            removed.append(True)
            events.append("removed")
        events.append("relu")

    trace.attach_hooks(tl.grad_fn(name="relu"), remover, confirm_mutation=True)
    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)

    # Backward order: relu2, addmm2, relu1, addmm1. The addmm hook is removed at relu2.
    assert events == ["removed", "relu", "relu"], events
    assert addmm_handle.handle_ids[0] not in _recipe_handles(trace)

    events.clear()
    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)
    assert events == ["relu", "relu"], events
