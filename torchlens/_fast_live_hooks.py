"""Module-hook handles of the guarded fast live run.

The fast live session collects each planned module site through one torch
forward hook that hands the call's output to the session, held only by weak
reference so the hooks never keep a session alive. The handles live in the
session's list; the removal runs from the session's finalizer.
"""

from __future__ import annotations

import weakref
from collections.abc import Mapping
from typing import Any

from torch import nn

__all__ = [
    "_remove_fast_live_hooks",
    "install_collection_hooks",
]


def install_collection_hooks(
    handles: list[Any],
    modules: Mapping[str, nn.Module],
    plan_addresses: tuple[str, ...],
    session_ref: weakref.ReferenceType[Any],
) -> None:
    """Register one collection forward hook per planned address, appending to ``handles``."""

    for address in plan_addresses:
        module = modules["" if address == "self" else address]

        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            output: Any,
            *,
            module_address: str = address,
            ref: weakref.ReferenceType[Any] = session_ref,
        ) -> None:
            """Forward one module boundary to the active session."""

            session = ref()
            if session is not None:
                session.capture_module(module_address, output)

        handles.append(module.register_forward_hook(hook))


def _remove_fast_live_hooks(handles: list[Any]) -> None:
    """Remove and discard module-hook handles without retaining their session.

    Every handle gets its own removal attempt: ``weakref.finalize`` pops its
    registry entry BEFORE invoking the callback, so this is the one chance to
    remove these hooks -- a single raising ``remove()`` must never strand the
    remaining handles on the user's modules forever. The first failure
    re-raises only after every handle was attempted and the list cleared.
    """

    first_error: BaseException | None = None
    for handle in handles:
        try:
            handle.remove()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    handles.clear()
    if first_error is not None:
        raise first_error
