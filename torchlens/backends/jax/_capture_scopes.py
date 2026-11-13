"""Capture-time hook calls into the optional random-state and backward modules.

The JAX capture wraps its forward in the interpreter hooks (retention keyed on
the capture's session) and the hooks the backend parity contracts name: ``rng_scope(session)`` from ``backends/_rng_capture/jax.py`` and, for a
backward-ready capture, ``backward_ready_scope(session)`` plus
``retain_backward_state(trace, session)`` from ``backends/_backward_native/jax.py``.
Each module is called only when it is installed; an absent module leaves the
capture exactly as it was. A module that is present but fails to import
raises, so a real defect is never hidden.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from ._interp_hooks import InterpreterHooks
from ._retention import Release, pending_release

_RNG_MODULE = "torchlens.backends._rng_capture.jax"
_BACKWARD_MODULE = "torchlens.backends._backward_native.jax"


def _optional_module(name: str) -> ModuleType | None:
    """Import ``name`` when it exists, else return ``None``.

    Parameters
    ----------
    name
        Dotted module name.

    Returns
    -------
    ModuleType | None
        The imported module, or ``None`` when it or its package is absent.
    """

    package = name.rpartition(".")[0]
    if importlib.util.find_spec(package) is None:
        return None
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


@dataclass(frozen=True)
class CaptureScope:
    """The interpreter hooks of one forward and its optional value release."""

    hooks: InterpreterHooks
    release: Release | None = None

    def equation_hook(self, hook: Callable[..., Any]) -> Callable[..., Any]:
        """Return the equation hook to pass the interpreter (wrapped when releasing)."""

        return hook if self.release is None else self.release.wrap(hook)

    def finish(self, result: Any) -> Any:
        """Return the interpreter result with every unkept capture released."""

        return result if self.release is None else self.release.finish(result)


@contextmanager
def forward_scopes(session: Any, *, backward_ready: bool) -> Iterator[CaptureScope]:
    """Enter the installed capture-time scopes around one JAX forward.

    Parameters
    ----------
    session
        The capture's open equation session.
    backward_ready
        Whether the capture keeps state for a later ``trace.backward``.

    Yields
    ------
    CaptureScope
        The interpreter hooks for this forward, plus the release a pending
        retention asked for (``_retention.pending_retention``); a plain capture
        has no release and keeps every value.
    """

    with ExitStack() as stack:
        rng = _optional_module(_RNG_MODULE)
        if rng is not None:
            stack.enter_context(rng.rng_scope(session))
        backward = _optional_module(_BACKWARD_MODULE) if backward_ready else None
        if backward is not None:
            stack.enter_context(backward.backward_ready_scope(session))
        yield CaptureScope(InterpreterHooks(session=session), pending_release())


def retain_backward_state(trace: Any, session: Any, events: Any, *, backward_ready: bool) -> None:
    """Hand a backward-ready trace's captures and event stream to the backward module.

    Parameters
    ----------
    trace
        The finished trace; ``jax_equation_captures`` is already stamped.
    session
        The capture's equation session.
    events
        The capture's ``CaptureEvents`` stream, released from the trace already.
    backward_ready
        Whether the capture keeps state for a later ``trace.backward``.

    Returns
    -------
    None
        The backward module stores what it needs on ``trace``.
    """

    backward = _optional_module(_BACKWARD_MODULE) if backward_ready else None
    if backward is None:
        return
    trace.capture_events = events
    try:
        backward.retain_backward_state(trace, session)
    finally:
        delattr(trace, "capture_events")
