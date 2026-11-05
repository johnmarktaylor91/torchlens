"""Optimizer step hooks keep firing when the class step lost torch's hook wrapper.

On torch 2.1 the first ``torch.compile`` in a process replaces every torch
optimizer class's ``step`` with the bare update, still marked ``hooked``, so
hooks registered through ``register_step_pre_hook`` / ``register_step_post_hook``
never fire again and a watch session saw no optimizer boundary. These tests
emulate that state on a private optimizer class (so they hold on every torch)
and check the instance binding TorchLens adds only in that state.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections.abc import Callable
from typing import Any

import pytest
import torch

import torchlens.trackers as trk
from torchlens._utils._optimizer_hooks import _BINDINGS, hook_around, register_step_hooks
from torchlens._utils._torch_compat import stripped_step_hook_wrapper


def _bare_step(events: list[str], *, stripped: bool) -> Callable[..., Any]:
    """Build a plain SGD update that records ``"step"`` in ``events``.

    Parameters
    ----------
    events:
        List each call appends ``"step"`` to.
    stripped:
        Mark the function ``hooked`` the way torch 2.1's compile leaves a
        stripped class step, so torch never wraps it again.

    Returns
    -------
    Callable[..., Any]
        A function to install as an optimizer class's ``step``.
    """

    def step(self: torch.optim.Optimizer, closure: Any = None) -> None:
        """Apply ``p -= lr * grad`` and record the call."""

        events.append("step")
        with torch.no_grad():
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.add_(param.grad, alpha=-group["lr"])

    if stripped:
        step.hooked = True  # type: ignore[attr-defined]
    return step


def _optimizer(*, stripped: bool) -> tuple[torch.nn.Module, torch.optim.Optimizer, list[str]]:
    """Return a model, an optimizer of a fresh class and the class's event list.

    Parameters
    ----------
    stripped:
        Give the class a step without torch's hook wrapper (torch 2.1 after
        ``torch.compile``); otherwise torch wraps the class step as usual.

    Returns
    -------
    tuple[torch.nn.Module, torch.optim.Optimizer, list[str]]
        The model, its optimizer and the list the class step records into.
    """

    events: list[str] = []
    optimizer_type = type(
        "_EmulatedSGD", (torch.optim.SGD,), {"step": _bare_step(events, stripped=stripped)}
    )
    model = torch.nn.Linear(4, 2)
    return model, optimizer_type(model.parameters(), lr=0.1), events


def _recorder(events: list[str], name: str) -> Callable[..., None]:
    """Return a step hook that appends ``name`` to ``events``.

    Parameters
    ----------
    events:
        Shared event list.
    name:
        Label to append on each call.

    Returns
    -------
    Callable[..., None]
        Hook usable as a step pre- or post-hook.
    """

    def hook(*_args: Any) -> None:
        """Record one call."""

        events.append(name)

    return hook


def _train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Run one forward, backward and optimizer step."""

    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()


def _hook_counts(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    """Return the number of step pre- and post-hooks on ``optimizer``."""

    return len(optimizer._optimizer_step_pre_hooks), len(optimizer._optimizer_step_post_hooks)


@pytest.mark.parametrize("stripped", [False, True], ids=["class_keeps_wrapper", "class_stripped"])
def test_each_hook_fires_once_per_step_in_registration_order(stripped: bool) -> None:
    """User and TorchLens hooks fire once per step, in torch's order, either way."""

    model, optimizer, events = _optimizer(stripped=stripped)
    optimizer.register_step_pre_hook(_recorder(events, "user_pre"))
    handles = register_step_hooks(
        optimizer, _recorder(events, "session_pre"), _recorder(events, "session_post")
    )
    optimizer.register_step_post_hook(_recorder(events, "user_post"))
    _train_step(model, optimizer)
    _train_step(model, optimizer)
    once = ["user_pre", "session_pre", "step", "session_post", "user_post"]
    assert events == once + once
    assert ("step" in vars(optimizer)) is stripped
    for handle in handles:
        handle.remove()


def test_binding_is_added_only_when_the_class_step_lost_the_wrapper() -> None:
    """A class that kept torch's wrapper gets no instance binding and no hold."""

    _, kept, _ = _optimizer(stripped=False)
    _, stripped, _ = _optimizer(stripped=True)
    assert stripped_step_hook_wrapper(kept) is None
    assert stripped_step_hook_wrapper(stripped) is torch.optim.Optimizer.profile_hook_step
    assert len(register_step_hooks(kept, _recorder([], "pre"))) == 1
    assert kept not in _BINDINGS and "step" not in vars(kept)


def test_close_leaves_the_optimizer_instance_as_it_was_found() -> None:
    """After every handle is removed, ``vars(optimizer)`` and the hooks are as before."""

    model, optimizer, events = _optimizer(stripped=True)
    before_keys, before_counts = set(vars(optimizer)), _hook_counts(optimizer)
    handles = register_step_hooks(optimizer, _recorder(events, "pre"), _recorder(events, "post"))
    _train_step(model, optimizer)
    for handle in handles:
        handle.remove()
    assert set(vars(optimizer)) == before_keys
    assert _hook_counts(optimizer) == before_counts
    assert optimizer not in _BINDINGS
    events.clear()
    _train_step(model, optimizer)
    assert events == ["step"]


def test_nested_holds_share_one_binding_until_the_last_release() -> None:
    """Two sessions on one optimizer bind once; the binding goes with the last hold."""

    model, optimizer, events = _optimizer(stripped=True)
    outer = register_step_hooks(optimizer, _recorder(events, "outer"))
    bound = vars(optimizer)["step"]
    inner = register_step_hooks(optimizer, _recorder(events, "inner"))
    assert vars(optimizer)["step"] is bound and _BINDINGS[optimizer][0] == 2
    _train_step(model, optimizer)
    assert events == ["outer", "inner", "step"]
    for handle in outer:
        handle.remove()
    events.clear()
    _train_step(model, optimizer)
    assert events == ["inner", "step"] and vars(optimizer)["step"] is bound
    for handle in inner + inner:
        handle.remove()
    assert "step" not in vars(optimizer) and optimizer not in _BINDINGS


def test_a_failing_attach_removes_the_pre_hook_and_the_binding() -> None:
    """``hook_around`` removes what it registered when ``attach`` raises."""

    model, optimizer, events = _optimizer(stripped=True)
    before_keys, before_counts = set(vars(optimizer)), _hook_counts(optimizer)

    def attach() -> None:
        """Fail the way a collector attach can."""

        raise RuntimeError("attach failed")

    with pytest.raises(RuntimeError, match="attach failed"):
        hook_around(optimizer, _recorder(events, "pre"), attach, _recorder(events, "post"))
    assert set(vars(optimizer)) == before_keys and _hook_counts(optimizer) == before_counts
    _train_step(model, optimizer)
    assert events == ["step"]


def test_an_instance_step_found_at_bind_time_is_restored() -> None:
    """A ``step`` the user set on the instance runs under the binding and comes back."""

    model, optimizer, events = _optimizer(stripped=True)
    original = optimizer.step

    def user_step(*args: Any, **kwargs: Any) -> Any:
        """Record, then run the class step."""

        events.append("user_step")
        return original(*args, **kwargs)

    optimizer.step = user_step  # type: ignore[method-assign]
    handles = register_step_hooks(optimizer, _recorder(events, "pre"))
    _train_step(model, optimizer)
    assert events == ["pre", "user_step", "step"]
    for handle in handles:
        handle.remove()
    assert vars(optimizer)["step"] is user_step


def test_watch_session_sees_the_boundary_when_the_class_step_was_stripped() -> None:
    """A tracker session over a stripped optimizer closes every step scope."""

    model, optimizer, _ = _optimizer(stripped=True)
    before_keys = set(vars(optimizer))
    sink = trk.MemorySink()
    session = trk.watch(model, to=sink, signals=("gradients",), optimizer=optimizer, every=1)
    for step in range(3):
        with session.step(step):
            _train_step(model, optimizer)
    session.close()
    assert set(vars(optimizer)) == before_keys


@pytest.mark.heavy
def test_watch_after_torch_compile_in_a_fresh_process() -> None:
    """The real trigger: ``torch.compile`` before a watch session, in a fresh process."""

    script = textwrap.dedent(
        """
        import torch
        import torchlens.trackers as trk

        torch.compile(lambda x: x + 1, backend="eager")
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        fired = []
        optimizer.register_step_post_hook(lambda *_: fired.append("user_post"))
        before = set(vars(optimizer))
        sink = trk.MemorySink()
        session = trk.watch(model, to=sink, signals=("gradients",), optimizer=optimizer, every=1)
        for step in range(2):
            with session.step(step):
                optimizer.zero_grad(set_to_none=True)
                model(torch.randn(4, 8)).sum().backward()
                optimizer.step()
        session.close()
        assert fired == ["user_post", "user_post"], fired
        assert set(vars(optimizer)) == before, set(vars(optimizer)) ^ before
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=240, check=False
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip().endswith("ok")
