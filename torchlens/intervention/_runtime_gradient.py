"""Backward-pass and hook-identity helpers for live intervention hooks."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from .. import _state

if TYPE_CHECKING:
    from .hooks import NormalizedHookEntry


class _HookReentrancyGuard:
    """Track recursive TorchLens hook execution depth in global state."""

    def __init__(self) -> None:
        """Initialise an inactive re-entrancy guard."""

        self.depth = 0
        self.active_log_id: int | None = None

    @property
    def active(self) -> bool:
        """Return whether hook execution is currently active.

        Returns
        -------
        bool
            Whether at least one hook is on the call stack.
        """

        return self.depth > 0

    def __enter__(self) -> _HookReentrancyGuard:
        """Enter hook execution.

        Returns
        -------
        _HookReentrancyGuard
            This guard.
        """

        self.depth += 1
        _state._hook_reentrancy_depth = self.depth
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Leave hook execution.

        Parameters
        ----------
        exc_type:
            Exception type, if any.
        exc:
            Exception value, if any.
        traceback:
            Exception traceback, if any.
        """

        self.depth = max(0, self.depth - 1)
        _state._hook_reentrancy_depth = self.depth
        if self.depth == 0:
            self.active_log_id = None


@contextmanager
def _current_injection_rule(state: dict[str, Any] | None, rule_id: Any) -> Iterator[None]:
    """Stamp ``rule_id`` as the armed injection state's ``current_rule`` for the body.

    ``state`` is ``None`` when ``log_injections`` is not armed (or when the
    caller has nothing to stamp); the body then runs with the recorder's own
    ``current_rule`` untouched. Otherwise the stamp is cleared on exit, even
    when the hook raises, so no later firing inherits a stale rule id.
    """

    if state is None:
        yield
        return
    state["current_rule"] = rule_id
    try:
        yield
    finally:
        state["current_rule"] = None


def _next_gradient_call_index(
    run_state: dict[str, Any], entry: NormalizedHookEntry, site_label: str
) -> int:
    """Return the one-based fire count of one gradient hook at one site.

    Parameters
    ----------
    run_state:
        Run-scoped dictionary of the backwarded trace (session-only).
    entry:
        Gradient-universe hook entry.
    site_label:
        Forward site label.

    Returns
    -------
    int
        Number of times this hook has now fired at this site on this trace.
    """

    counts: dict[tuple[str, int], int] = run_state.setdefault("_tl_gradient_fire_counts", {})
    key = (site_label, _hook_identity(entry))
    counts[key] = counts.get(key, 0) + 1
    return counts[key]


def _hook_state(
    run_state: dict[str, Any], site_key: Any, entry: NormalizedHookEntry
) -> dict[str, Any]:
    """Return the per-(site, hook) scratch dict exposed as ``ctx.state``.

    The dict lives in the run state so it persists across fires of the same
    hook at the same site (every backward pass of a sticky gradient hook,
    every pass of a multi-pass layer) and ends with the run.

    Parameters
    ----------
    run_state:
        Run-scoped dictionary (``ctx.run_state``).
    site_key:
        Site identity of the fire (label).
    entry:
        Hook entry that fires.

    Returns
    -------
    dict[str, Any]
        Mutable scratch dictionary for this (site, hook) pair.
    """

    store: dict[tuple[str, str, int], dict[str, Any]] = run_state.setdefault("_tl_hook_state", {})
    key = (entry.signal, str(site_key), _hook_identity(entry))
    return store.setdefault(key, {})


def _hook_identity(entry: NormalizedHookEntry) -> int:
    """Return an identity for "the same hook" that survives plan re-normalization.

    A sticky plan is re-normalized at every rerun and backward pass, which
    rebuilds helper closures and observer bound methods; the stored helper
    spec or the observer object is what stays the same.

    Parameters
    ----------
    entry:
        Hook entry.

    Returns
    -------
    int
        ``id`` of the helper spec, the bound method's owner, or the callable.
    """

    if entry.helper_spec is not None:
        return id(entry.helper_spec)
    hook = entry.normalized_callable
    return id(getattr(hook, "__self__", hook))


def _record_backward_selector_fire(entry: NormalizedHookEntry) -> None:
    """Increment the deferred zero-match ledger for one backward selector fire.

    Parameters
    ----------
    entry:
        Normalized backward hook entry that matched the live GradFn site.
    """

    if entry.metadata.get("created_by") != "intervene_backward_selector":
        return
    trace, _enabled = _state.active_capture()
    if trace is None:
        return
    trace._tl_intervene_selector_fire_count = (
        int(getattr(trace, "_tl_intervene_selector_fire_count", 0)) + 1
    )


def _live_run_ctx() -> dict[str, Any]:
    """Return the shared run context for the active model log.

    Returns
    -------
    dict[str, Any]
        Mutable context shared by hooks in this live run.
    """

    trace = _state._active_trace
    if trace is None:
        return {}
    run_ctx = getattr(trace, "last_run", None)
    if run_ctx is None:
        run_ctx = {"engine": "live", "timestamp": time.monotonic()}
        trace.last_run = run_ctx
    else:
        run_ctx.setdefault("engine", "live")
        run_ctx.setdefault("timestamp", time.monotonic())
    # Seed law D5 (F02): the capture's recorded seed rides the run context so
    # seed='auto' stochastic edits can canonicalize their base seed without
    # ever touching ambient RNG.
    trace_seed = getattr(trace, "random_seed", None)
    if trace_seed is not None:
        run_ctx.setdefault("trace_random_seed", trace_seed)
    return run_ctx


def _hook_display_name(entry: NormalizedHookEntry) -> str:
    """Return a stable display name for a hook entry.

    Parameters
    ----------
    entry:
        Normalized hook entry.

    Returns
    -------
    str
        Helper name or callable qualname.
    """

    if entry.helper_spec is not None:
        return entry.helper_spec.name
    return getattr(entry.normalized_callable, "__qualname__", "user_hook")


def _first_replaced_tuple_index(
    previous: tuple[torch.Tensor | None, ...],
    current: tuple[torch.Tensor | None, ...],
) -> int | None:
    """Return the first tuple slot whose object identity changed.

    Parameters
    ----------
    previous:
        Tuple before helper execution.
    current:
        Tuple after helper execution.

    Returns
    -------
    int | None
        First changed slot index, or ``None`` when object identities match.
    """

    for index, (before, after) in enumerate(zip(previous, current, strict=False)):
        if before is not after:
            return index
    return None


def _active_backward_pass_index(grad_fn_handle: Any) -> int | None:
    """Return the active trace backward pass index for a grad_fn handle.

    Parameters
    ----------
    grad_fn_handle:
        Runtime GradFn record or compatible test stub.

    Returns
    -------
    int | None
        Active one-based backward pass index, when available.
    """

    trace = getattr(grad_fn_handle, "trace", None)
    if trace is None:
        return None
    return _trace_backward_pass_index(trace)


def _trace_backward_pass_index(trace: Any) -> int | None:
    """Return a trace's active one-based backward pass index (``None`` outside a pass).

    The backward door sets the index for the length of one pass and drops it
    afterwards, so its absence is the "no pass open" reading.

    Parameters
    ----------
    trace:
        Trace being backwarded.

    Returns
    -------
    int | None
        Active one-based backward pass index, when a pass is open.
    """

    value = getattr(trace, "_active_backward_pass_index", None)
    return int(value) if isinstance(value, int) else None
