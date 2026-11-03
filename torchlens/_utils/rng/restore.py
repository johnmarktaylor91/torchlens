"""The one door for preserving the user's global RNG streams across TorchLens work.

:class:`RngStreamsRestore` snapshots every stream at construction and restores it exactly
once, on request or at context exit, on every settlement path. ``tl.trace`` (capture
seeding), intervention and validation replay (:func:`execute_with_restored_rng_autocast`)
and ``tl.validate`` (:func:`restores_rng_streams`) all go through it, so code after a
TorchLens call continues from the user's own stream, never from the capture's.
"""

from __future__ import annotations

import functools as _functools_module
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from .autocast import AutocastRestore
from .streams import log_current_rng_states, set_rng_from_saved_states

_T = TypeVar("_T")


class RngStreamsRestore:
    """Snapshot of every global RNG stream, restored exactly once.

    Construction snapshots the streams (:func:`~.streams.log_current_rng_states`);
    :meth:`restore` writes them back and is idempotent, so a caller with several
    settlement paths (a ``finally`` plus pre-``try`` failure windows) can register it on
    each. As a context manager it restores at exit on every path, including exceptions.
    """

    __slots__ = ("_pending", "_states")

    def __init__(self) -> None:
        self._states = log_current_rng_states()
        self._pending = True

    @property
    def states(self) -> dict[str, Any]:
        """The snapshot taken at construction (the per-op record shape)."""
        return self._states

    def restore(self) -> None:
        """Restore the snapshot once; later calls are no-ops."""
        if self._pending:
            self._pending = False
            set_rng_from_saved_states(self._states)

    def __enter__(self) -> RngStreamsRestore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.restore()


_P = ParamSpec("_P")
_R = TypeVar("_R")


def restores_rng_streams(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Run ``func`` under :class:`RngStreamsRestore` so it leaves the user's streams as found.

    The bracket is around the whole call, never inside it: a validation entry decorated
    with it still seeds and replays exactly as before, and the caller's ``random``, NumPy,
    torch CPU and CUDA generator positions are restored when it returns or raises.
    """

    @_functools_module.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Call ``func`` inside the restore bracket."""
        with RngStreamsRestore():
            return func(*args, **kwargs)

    return wrapper


def execute_with_restored_rng_autocast(
    func: Callable[..., _T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    rng_states: dict[str, Any] | None,
    autocast_state: dict[str, Any] | None,
) -> _T:
    """Execute a callable with saved RNG and autocast state in a tight scope.

    Parameters
    ----------
    func:
        Callable to execute.
    args:
        Positional arguments for ``func``.
    kwargs:
        Keyword arguments for ``func``.
    rng_states:
        RNG states captured before the original operation. ``None`` or an empty
        dict leaves the current RNG state untouched until final restoration.
    autocast_state:
        Autocast state captured before the original operation.

    Returns
    -------
    _T
        Return value from ``func``.

    Raises
    ------
    Exception
        Re-raises any exception from ``func`` after restoring caller RNG state.
    """

    # The target RNG state is applied INSIDE the bracket so the caller's state is
    # restored even when the restore itself partially applies and then raises (a
    # malformed rng_states dict sets the Python/NumPy engines then KeyErrors on the
    # torch key).
    with RngStreamsRestore():
        if rng_states:
            set_rng_from_saved_states(rng_states)
        with AutocastRestore(autocast_state or {}):
            return func(*args, **kwargs)
