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

import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import MultiMatchWarning
from torchlens.options import CaptureOptions


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


def _attach(trace: Any, selector: Any, hook: Any, **kwargs: Any) -> Any:
    """Attach a hook to every matching site; the fan-out disclosure is expected here."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MultiMatchWarning)
        return trace.attach_hooks(selector, hook, confirm_mutation=True, **kwargs)


def test_forward_hook_removed_before_its_site_never_fires_on_that_run() -> None:
    """The probe: a prepend hook on linear removes the relu hook on its first fire."""

    model, x, trace = _ready_trace()
    events: list[str] = []

    def victim(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        events.append("victim")
        return out * 0.5

    relu_handle = _attach(trace, tl.func("relu"), victim)
    removed_once: list[bool] = []

    def remover(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        if not removed_once:
            relu_handle.remove()
            removed_once.append(True)
            events.append("removed")
        events.append("linear")
        return out

    _attach(trace, tl.func("linear"), remover, prepend=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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

    handle_box.append(_attach(trace, tl.func("relu"), once))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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

    linear_handle = _attach(trace, tl.func("linear"), linear_hook)

    def relu_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        if "removed" not in events:
            linear_handle.remove()
            events.append("removed")
        events.append("relu")
        return out

    _attach(trace, tl.func("relu"), relu_hook)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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

    second_handle = _attach(trace, tl.func("relu"), second)

    def first(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        if "removed" not in events:
            second_handle.remove()
            events.append("removed")
        events.append("first")
        return out

    _attach(trace, tl.func("relu"), first, prepend=True)
    trace.run(model, x)

    assert events == ["removed", "first", "second", "first"], events


def test_nothing_removed_keeps_the_rerun_unchanged() -> None:
    model, x, trace = _ready_trace()
    fired: list[int] = []

    def count(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        fired.append(1)
        return out

    _attach(trace, tl.func("relu"), count)
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

    victim.signal = "backward"  # type: ignore[attr-defined]
    addmm_handle = _attach(trace, tl.grad_fn(name="addmm"), victim)
    removed_once: list[bool] = []

    def remover(
        grad_input: tuple[torch.Tensor | None, ...],
        *,
        grad_output: tuple[torch.Tensor | None, ...] | None,
        grad_fn_handle: Any,
        call_index: int,
        run_ctx: dict[str, Any],
    ) -> None:
        if not removed_once:
            addmm_handle.remove()
            removed_once.append(True)
            events.append("removed")
        events.append("relu")

    remover.signal = "backward"  # type: ignore[attr-defined]
    _attach(trace, tl.grad_fn(name="relu"), remover)
    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)

    # Backward order: relu2, addmm2, relu1, addmm1. The addmm hook is removed at relu2.
    assert events == ["removed", "relu", "relu"], events
    assert addmm_handle.handle_ids[0] not in _recipe_handles(trace)

    events.clear()
    trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)
    assert events == ["relu", "relu"], events
