"""Count the passes a hooks cell runs with its hook armed, independently of the hook's fires.

The hooks oracle promises one fire per selected op per pass. A cell runs more
than one pass with the capture-time spec armed: the capture itself, the
``forward_match`` replay (``trace.run``), and whatever reruns or second
captures its partner feature makes. Counting fires over the cell's lifetime
against one pass's selected ops is wrong for every such cell, and inferring the
passes from the fires would hide a pass that fired nowhere. This module counts
the passes from the capture door instead: every capture pass (``tl.trace``,
``trace.run``, a second capture) goes through
``torchlens._capture.run_entry._run_resolved_request``, so the recorder wraps
that function once and keeps each returned trace whose request references the
cell's hook. Each pass's fires are then the fires whose ``ctx.run_state`` is
that trace's run context (``trace.last_run``).
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["HookPass", "PassRecorder", "hook_pass_problems"]


@dataclass
class PassRecorder:
    """The armed passes one hooks cell ran, appended by the capture-door wrapper."""

    hook: Callable[..., Any]
    traces: list[Any] = field(default_factory=list)


_ACTIVE: list[PassRecorder | None] = [None]


def start_recording(hook: Callable[..., Any]) -> PassRecorder:
    """Install the capture-door wrapper (once per process) and record passes arming ``hook``.

    A later call replaces the recorder, so a cell whose after-checks never run
    leaves nothing behind that a later cell would count.
    """

    from torchlens._capture import run_entry

    if not getattr(run_entry._run_resolved_request, "_behaviour_pass_counter", False):
        original = run_entry._run_resolved_request

        @functools.wraps(original)
        def counted(r: Any) -> Any:
            trace = original(r)
            recorder = _ACTIVE[0]
            if recorder is not None and _references(r, recorder.hook):
                recorder.traces.append(trace)
            return trace

        counted._behaviour_pass_counter = True  # type: ignore[attr-defined]
        run_entry._run_resolved_request = counted
    recorder = PassRecorder(hook)
    _ACTIVE[0] = recorder
    return recorder


def stop_recording(recorder: PassRecorder) -> None:
    """Stop recording into ``recorder`` (the wrapper stays and records nothing)."""

    if _ACTIVE[0] is recorder:
        _ACTIVE[0] = None


def _references(root: Any, target: Any, depth: int = 7) -> bool:
    """Return whether ``target`` is reachable from ``root`` by identity.

    Walks containers, instance attributes, dataclass slots, ``__wrapped__``,
    ``functools.partial`` and closure cells; modules and tensors are leaves.
    """

    seen: set[int] = set()
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        obj, level = stack.pop()
        if obj is target:
            return True
        if level >= depth or id(obj) in seen or isinstance(obj, (torch.nn.Module, torch.Tensor)):
            continue
        seen.add(id(obj))
        stack.extend((child, level + 1) for child in _children(obj))
    return False


def _children(obj: Any) -> list[Any]:
    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return []
    if isinstance(obj, dict):
        return [*obj.keys(), *obj.values()]
    if isinstance(obj, (list, tuple, set, frozenset)):
        return list(obj)
    children: list[Any] = []
    if isinstance(obj, functools.partial):
        children += [obj.func, *obj.args, *obj.keywords.values()]
    for name in ("__wrapped__", "__func__"):
        if hasattr(obj, name):
            children.append(getattr(obj, name))
    closure = getattr(obj, "__closure__", None)
    if closure:
        children += [cell.cell_contents for cell in closure if _filled(cell)]
    if callable(obj) and not hasattr(obj, "__dict__"):
        return children
    try:
        children += list(vars(obj).values())
    except TypeError:
        pass
    for name in getattr(type(obj), "__slots__", ()):
        if hasattr(obj, name):
            children.append(getattr(obj, name))
    return children


def _filled(cell: Any) -> bool:
    try:
        cell.cell_contents  # noqa: B018 - an empty cell raises ValueError
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class HookPass:
    """One armed pass: its run context and how many ops the hook selects in it."""

    run_state: Any
    selected: int


def hook_pass_problems(passes: Sequence[HookPass], fire_states: Sequence[Any]) -> list[str]:
    """Return what breaks "one fire per selected op per pass" (empty when it holds).

    Parameters
    ----------
    passes:
        The armed passes the harness ran, counted from the capture door.
    fire_states:
        The ``ctx.run_state`` object of each fire, in fire order.
    """

    problems: list[str] = []
    if not passes:
        problems.append("no armed pass was counted")
    for index, armed in enumerate(passes):
        fired = sum(
            1 for state in fire_states if armed.run_state is not None and state is armed.run_state
        )
        if fired != armed.selected:
            problems.append(f"pass {index} fired {fired} times for {armed.selected} selected ops")
    stray = sum(1 for state in fire_states if not any(state is armed.run_state for armed in passes))
    if stray:
        problems.append(f"{stray} fires in no counted pass")
    return problems
