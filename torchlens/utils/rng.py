"""RNG and autocast state capture/restore for reproducible forward-pass replay.

During the exhaustive logging pass, RNG states are captured *before* each
logged operation so that the validation replay can restore the exact same
random state and reproduce the operation's output.  This is critical for
ops like ``dropout`` or ``torch.randn`` that consume RNG.

**Ordering invariant**: RNG states must be captured *before*
``active_logging()`` is entered, because entering the logging context
itself may call decorated functions (e.g. tensor allocations for internal
bookkeeping) that would advance the RNG.

Three independent RNG engines are captured:
  - Python's ``random`` module
  - NumPy's ``np.random``
  - PyTorch's CPU generator (``torch.random``)
  - PyTorch's CUDA generator (if CUDA is available)

Autocast state (``torch.amp.autocast``) is captured similarly so that
mixed-precision ops can be replayed under the same dtype context.
"""

import random
from collections.abc import Callable
from typing import Any, Dict, List, TypeVar

import numpy as np
import torch

from .tensor_utils import _is_cuda_available

_AUTOCAST_DEVICES = ("cpu", "cuda")
_T = TypeVar("_T")


def set_random_seed(seed: int):
    """Set the random seed for all RNG engines simultaneously.

    Ensures deterministic behavior across Python, NumPy, and PyTorch
    (CPU + all CUDA devices).

    Args:
        seed: Seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def execute_with_restored_rng_autocast(
    func: Callable[..., _T],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    rng_states: Dict[str, Any] | None,
    autocast_state: Dict[str, Any] | None,
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

    current_rng_states = log_current_rng_states()
    if rng_states:
        set_rng_from_saved_states(rng_states)
    try:
        with AutocastRestore(autocast_state or {}):
            return func(*args, **kwargs)
    finally:
        set_rng_from_saved_states(current_rng_states)


def log_current_rng_states(torch_only: bool = False) -> Dict:
    """Snapshot the current state of all RNG engines.

    The returned dict can be passed to :func:`set_rng_from_saved_states`
    to restore the exact same RNG position later (e.g. during validation
    replay).

    Args:
        torch_only: If True, only capture PyTorch RNG state (skip Python
            ``random`` and NumPy). This is faster and sufficient for most
            torch operations (dropout, randn, etc.).

    Returns:
        Dict with keys ``"random"``, ``"np"``, ``"torch"``, and optionally
        ``"torch_cuda"``, each holding the opaque state object for that engine.
    """
    rng_dict: Dict[str, object] = {"torch": torch.random.get_rng_state()}
    if not torch_only:
        rng_dict["random"] = random.getstate()
        rng_dict["np"] = np.random.get_state()
    if _is_cuda_available():
        rng_dict["torch_cuda"] = torch.cuda.get_rng_state("cuda")
    return rng_dict


def set_rng_from_saved_states(rng_states: Dict):
    """Restore RNG engines to a previously captured state.

    Args:
        rng_states: Dict produced by :func:`log_current_rng_states`.
            If empty (RNG capture was disabled), this is a no-op.
    """
    if not rng_states:
        return
    if "random" in rng_states:
        random.setstate(rng_states["random"])
    if "np" in rng_states:
        np.random.set_state(rng_states["np"])
    torch.random.set_rng_state(rng_states["torch"])
    if _is_cuda_available() and "torch_cuda" in rng_states:
        torch.cuda.set_rng_state(rng_states["torch_cuda"], "cuda")


def log_current_autocast_state() -> Dict:
    """Capture the current ``torch.amp.autocast`` enabled/dtype state.

    Checked for each device in :data:`_AUTOCAST_DEVICES`.  If a device
    doesn't support autocast queries, it is silently skipped.

    Returns:
        Dict mapping device name to ``{"enabled": bool, "dtype": torch.dtype}``.
    """
    state = {}
    for device in _AUTOCAST_DEVICES:
        try:
            state[device] = {
                "enabled": torch.is_autocast_enabled(device),
                "dtype": torch.get_autocast_dtype(device),
            }
        except (RuntimeError, TypeError):
            # Device doesn't support autocast queries (e.g. no CUDA).
            pass
    return state


class AutocastRestore:
    """Context manager that re-enters saved autocast contexts during replay.

    Only devices that were *enabled* at capture time get an autocast
    context opened.  Contexts are exited in reverse order on ``__exit__``.

    Usage::

        with AutocastRestore(saved_state):
            result = func(*args, **kwargs)
    """

    __slots__ = ("_autocast_state", "_contexts")

    def __init__(self, autocast_state: Dict):
        self._autocast_state = autocast_state
        self._contexts: List[Any] = []

    def __enter__(self):
        for device, state in self._autocast_state.items():
            if state["enabled"]:
                ctx = torch.amp.autocast(device, dtype=state["dtype"])
                ctx.__enter__()
                self._contexts.append(ctx)
        return self

    def __exit__(self, *exc_info):
        # Exit in reverse order to mirror the nesting order of __enter__.
        for ctx in reversed(self._contexts):
            ctx.__exit__(*exc_info)
