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
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import ModuleType
from typing import Any

from ._interp_hooks import InterpreterHooks

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


@contextmanager
def forward_scopes(session: Any, *, backward_ready: bool) -> Iterator[InterpreterHooks]:
    """Enter the installed capture-time scopes around one JAX forward.

    Parameters
    ----------
    session
        The capture's open equation session.
    backward_ready
        Whether the capture keeps state for a later ``trace.backward``.

    Yields
    ------
    InterpreterHooks
        The interpreter hooks for this forward. Retention is keyed on ``session``
        (``_retention.retention_of``); a session with no policy keeps every value.
    """

    with ExitStack() as stack:
        rng = _optional_module(_RNG_MODULE)
        if rng is not None:
            stack.enter_context(rng.rng_scope(session))
        backward = _optional_module(_BACKWARD_MODULE) if backward_ready else None
        if backward is not None:
            stack.enter_context(backward.backward_ready_scope(session))
        yield InterpreterHooks(session=session)


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
