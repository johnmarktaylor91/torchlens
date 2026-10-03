"""Shared ownership marker and pause bracket for TorchLens dispatch modes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch.utils._python_dispatch import TorchDispatchMode

from ...errors._base import CompatibilityError
from ...utils._torch_compat import get_current_dispatch_mode_stack


class _TorchLensDispatchMode(TorchDispatchMode):
    """Marker base for Python dispatch modes owned by TorchLens."""


class SubclassConstructionUnderDispatchModeError(CompatibilityError, RuntimeError):
    """A strict Tensor subclass could not be constructed while a mode was active here.

    Torch 2.1 and 2.2 CAN raise "Creating a new Tensor subclass X but the raw
    Tensor object is already associated to a python object of type Tensor"
    from ``__new__``/``_make_subclass``/``as_subclass`` while a python
    ``TorchDispatchMode`` is active -- reproduced on stock torch with a no-op
    mode, not something TorchLens's own wrapping causes (see
    ``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE`` in ``torchlens.utils._torch_compat``).
    Whether a GIVEN construction actually hits it is not cheaply predictable
    (e.g. ``torch.nn.Parameter`` construction via ``_make_subclass`` routinely
    succeeds there), so TorchLens translates the torch error into this typed
    one on actual failure rather than preemptively refusing every strict-
    subclass construction under an active mode. A single non-reentrant pause
    bracket safely covers TorchLens's OWN nested reconstruction calls
    (``safe_copy``'s subclass-preserving clone); pausing a DIRECT, top-level
    construction call made while a TorchLens dispatch mode (the completeness
    witness; an intervention-ready capture's mode) is armed was tried and
    reverted after it corrupted interpreter state when exercised across
    several capture paths in one process (segfault on torch 2.1.2). This
    typed, disclosed refusal is the legitimate degradation instead: upgrade
    to a torch release where ``HAS_SUBCLASS_CTOR_IN_DISPATCH_MODE`` is
    ``True``, or avoid constructing or converting into a custom Tensor
    subclass (``__new__``, ``_make_subclass``, ``as_subclass`` into a
    non-``torch.Tensor`` cls) inside a model forward, an intervention hook,
    or a ``validate_forward_pass`` replay while capturing on this torch
    build.
    """


def _exit_own_dispatch_modes() -> tuple[_TorchLensDispatchMode, ...] | None:
    """Exit top-contiguous TorchLens dispatch modes from the active stack.

    Returns
    -------
    tuple[_TorchLensDispatchMode, ...] | None
        Exited modes in inner-to-outer order, or ``None`` when the stack is
        unreadable or no TorchLens-owned mode is at its top.
    """

    stack = get_current_dispatch_mode_stack()
    if not stack:
        return None
    exited: list[_TorchLensDispatchMode] = []
    while stack and isinstance(stack[-1], _TorchLensDispatchMode):
        mode = stack.pop()
        mode.__exit__(None, None, None)
        exited.append(mode)
    return tuple(exited) or None


def _reenter_own_dispatch_modes(exited: tuple[_TorchLensDispatchMode, ...]) -> None:
    """Re-enter TorchLens modes exited by :func:`_exit_own_dispatch_modes`.

    Parameters
    ----------
    exited:
        Modes in inner-to-outer exit order.
    """

    for mode in reversed(exited):
        mode.__enter__()


@contextmanager
def pause_own_dispatch_modes() -> Iterator[tuple[_TorchLensDispatchMode, ...]]:
    """Pause removable TorchLens modes and restore the exact stack on exit.

    Yields
    ------
    tuple[_TorchLensDispatchMode, ...]
        Exact mode instances paused in inner-to-outer order. An empty tuple
        means stack introspection was unavailable or a foreign mode blocked
        access to every owned mode.
    """

    stack_before = get_current_dispatch_mode_stack()
    exited = _exit_own_dispatch_modes() or ()
    try:
        yield exited
    finally:
        _reenter_own_dispatch_modes(exited)
        stack_after = get_current_dispatch_mode_stack()
        if stack_before is not None and stack_after is not None:
            if len(stack_after) != len(stack_before):
                raise RuntimeError("TorchLens dispatch-mode pause changed the active stack depth")
            if any(
                after is not before for before, after in zip(stack_before, stack_after, strict=True)
            ):
                raise RuntimeError(
                    "TorchLens dispatch-mode pause changed active stack identity or order"
                )
