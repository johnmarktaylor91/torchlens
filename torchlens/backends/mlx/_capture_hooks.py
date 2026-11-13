"""Optional capture hooks owned by other modules (private).

The MLX capture path calls three hooks around each forward: the random-state
scope (``_rng_capture.mlx.rng_scope``), the backward-ready scope
(``_backward_native.mlx.backward_ready_scope``) and the backward-state handoff
(``_backward_native.mlx.retain_backward_state``). Each owner module is called
only when it can be imported, so a capture without the owner behaves exactly as
before. A module that exists but fails while importing is a real error and is
raised, never treated as absent.

The hooks receive :class:`MLXHookSession`, not the trace, so whatever state an
owner keeps during the forward never lands as an undeclared trace attribute.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from types import ModuleType
from typing import Any

RNG_HOOK_MODULE = "torchlens.backends._rng_capture.mlx"
BACKWARD_HOOK_MODULE = "torchlens.backends._backward_native.mlx"


class MLXHookSession:
    """Per-capture holder the optional hooks may attach their own state to.

    Parameters
    ----------
    trace:
        The trace being captured. Owners read from it; they keep their own
        state as attributes of this session.
    """

    def __init__(self, trace: Any) -> None:
        self.trace = trace


def optional_hook_module(module_name: str) -> ModuleType | None:
    """Import ``module_name`` or return ``None`` when it does not exist.

    Parameters
    ----------
    module_name:
        Dotted module path of a hook owner.

    Returns
    -------
    ModuleType or None
        The module, or ``None`` when it (or a parent package) is missing.

    Raises
    ------
    ModuleNotFoundError
        When the module exists but one of its own imports is missing.
    """

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (module_name == missing or module_name.startswith(missing + ".")):
            return None
        raise


@contextmanager
def forward_hooks(trace: Any) -> Iterator[MLXHookSession]:
    """Enter the random-state and backward-ready scopes that are importable.

    Parameters
    ----------
    trace:
        The trace being captured.

    Yields
    ------
    MLXHookSession
        The capture's hook session, while the forward runs inside every
        available scope; pass it to :func:`retain_backward_state` afterwards.
    """

    session = MLXHookSession(trace)
    with ExitStack() as stack:
        rng = optional_hook_module(RNG_HOOK_MODULE)
        if rng is not None:
            stack.enter_context(rng.rng_scope(session))
        backward = optional_hook_module(BACKWARD_HOOK_MODULE)
        if backward is not None:
            stack.enter_context(backward.backward_ready_scope(session))
        yield session


def retain_backward_state(trace: Any, session: MLXHookSession) -> None:
    """Hand the finished capture to the backward owner when it is importable.

    Parameters
    ----------
    trace:
        The finished trace.
    session:
        The capture's hook session.

    Returns
    -------
    None
        No effect when the owner module does not exist.
    """

    backward = optional_hook_module(BACKWARD_HOOK_MODULE)
    if backward is not None:
        backward.retain_backward_state(trace, session)
