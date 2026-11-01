"""Global RNG stream state: snapshot and restore of every engine TorchLens seeds.

Streams covered: Python ``random``, legacy ``numpy.random``, the torch CPU generator and
every CUDA device generator that is already live (a CPU capture never initializes CUDA to
read a generator it cannot have consumed; :func:`_snapshot_cuda_rng_states`). The dict
shape returned by :func:`log_current_rng_states` is the one per-op record shape the
capture stores and replay restores through :func:`set_rng_from_saved_states`.

:func:`snapshot_host_rng` / :func:`host_rng_advanced` / :func:`restore_host_rng` are the
cheap host-only pair the sparse runnable path uses to detect host RNG consumption.
"""

from __future__ import annotations

import random
import warnings as _warnings_module
from typing import Any

import numpy as np
import torch

from ..tensor_utils import _is_cuda_available, _is_cuda_initialized
from .monitor_state import _suppress_active_monitor_marks


def snapshot_host_rng() -> tuple[Any, Any]:
    """Snapshot the Python ``random`` and NumPy RNG states without advancing them.

    Reading ``random.getstate()`` / ``np.random.get_state()`` is side-effect free,
    so this can bracket a forward pass to detect whether user code consumed host
    (non-torch) RNG -- the signal a sparse runnable path uses to stay honest about
    Python/NumPy control-flow branches it cannot re-observe.

    Returns
    -------
    tuple[Any, Any]
        ``(python_random_state, numpy_random_state)`` snapshot pair.
    """
    return (random.getstate(), np.random.get_state())


def host_rng_advanced(before: tuple[Any, Any], after: tuple[Any, Any]) -> bool:
    """Return whether a host (Python/NumPy) RNG engine advanced between snapshots.

    Parameters
    ----------
    before:
        Snapshot from :func:`snapshot_host_rng` taken before the observed region.
    after:
        Snapshot from :func:`snapshot_host_rng` taken after the observed region.

    Returns
    -------
    bool
        ``True`` when either the Python ``random`` or NumPy engine state changed.
    """
    py_before, np_before = before
    py_after, np_after = after
    if py_before != py_after:
        return True
    return not _numpy_states_equal(np_before, np_after)


def restore_host_rng(snapshot: tuple[Any, Any]) -> None:
    """Restore Python ``random`` and NumPy engines from a :func:`snapshot_host_rng` pair."""
    py_state, np_state = snapshot
    random.setstate(py_state)
    np.random.set_state(np_state)


def _numpy_states_equal(a: Any, b: Any) -> bool:
    """Compare two ``np.random.get_state()`` tuples (array-aware) for equality."""
    try:
        if a[0] != b[0] or not np.array_equal(a[1], b[1]):
            return False
        return tuple(a[2:]) == tuple(b[2:])
    except (TypeError, IndexError, ValueError):
        return a is b


_cuda_rng_unusable: bool = False
"""Capture-scoped retry latch: a CUDA RNG snapshot raised, stop retrying for now.

Set only by :func:`_snapshot_cuda_rng_states`; RE-ARMED by
:func:`set_random_seed` (which every capture runs at entry). The snapshot is
called per logged op, so within one capture the first failure latches — the
failure cost and the warning are paid once, not per op. But the failure itself
can be TRANSIENT (a busy device, a momentary OOM, a fork-context error), so a
process-lifetime latch silently downgraded EVERY later capture's replay
fidelity because of one bad moment (grind p5, B2P3-16). Re-arming at capture
entry bounds the damage to the capture that actually hit the failure.
"""


def rearm_cuda_rng_latch() -> None:
    """Re-arm the CUDA RNG snapshot latch at capture entry (:func:`~.seeding.set_random_seed`)."""
    global _cuda_rng_unusable
    _cuda_rng_unusable = False


def _snapshot_cuda_rng_states() -> list[Any]:
    """Return per-device CUDA RNG states, or ``[]`` when no CUDA state is live.

    ``torch.cuda.get_rng_state_all()`` calls ``torch.cuda._lazy_init()`` and
    reads a generator for EVERY visible device.  Calling it unconditionally
    (the pre-fix behavior, gated only on ``torch.cuda.is_available()``) meant a
    pure-CPU capture on a host with visible CUDA devices paid full CUDA
    initialization -- and, on a host whose CUDA stack is visible but unusable
    (stale driver, mismatched build, one bad device in a multi-GPU box), the
    initialization raised and aborted the CPU capture outright.

    Two guards, in order (both short-circuited while ``_cuda_rng_unusable`` is
    latched; the latch is re-armed at every capture entry by
    :func:`set_random_seed`, so a transient failure degrades only the capture
    that hit it, never the whole process):

    1. If this process has never initialized CUDA, no CUDA generator can have
       produced a number that any captured op consumed, so there is no state to
       snapshot -- and reading one would create the very CUDA context this
       capture does not need.  Returning ``[]`` here is exactly equivalent for
       replay purposes and never touches the driver.
    2. If CUDA *is* initialized, snapshot every device exactly as before (byte
       identical for real CUDA captures: a capture that touches CUDA has
       initialized it by definition).  Should that read still fail, degrade to
       ``[]`` with a warning instead of aborting the capture, and latch the
       failure so the warning is not repeated per op.

    Returns
    -------
    list[Any]
        Opaque per-device CUDA RNG state objects, in device order; empty when
        no live CUDA RNG state exists or it could not be read.
    """
    global _cuda_rng_unusable
    if _cuda_rng_unusable:
        return []
    if not _is_cuda_initialized():
        return []
    # Defense in depth: an initialized CUDA runtime implies availability, unless the
    # availability probe itself already failed earlier in this process.
    if not _is_cuda_available():
        return []
    try:
        return torch.cuda.get_rng_state_all()
    except Exception as exc:  # noqa: BLE001 - any broken-CUDA failure mode
        _cuda_rng_unusable = True
        _warnings_module.warn(
            "Could not read CUDA RNG state "
            f"({type(exc).__name__}: {exc}); continuing without CUDA RNG "
            "snapshots. Replay of operations that consume CUDA randomness "
            "cannot be reproduced exactly for this capture.",
            stacklevel=3,
        )
        return []


def log_current_rng_states(torch_only: bool = False) -> dict[str, Any]:
    """Snapshot the current state of all RNG engines.

    The returned dict can be passed to :func:`set_rng_from_saved_states`
    to restore the exact same RNG position later (e.g. during validation
    replay).

    Parameters
    ----------
    torch_only:
        If True, only capture PyTorch RNG state (skip Python ``random`` and
        NumPy). This is faster and sufficient for most torch operations
        (dropout, randn, etc.).

    Returns
    -------
    dict[str, Any]
        Dict with keys ``"random"``, ``"np"``, ``"torch"``, and optionally
        ``"torch_cuda_all"``, each holding the opaque state object for that
        engine. ``"torch_cuda"`` is also populated for backward compatibility
        with older single-device snapshots. The two CUDA keys are present only
        when this process has live CUDA RNG state that could be read; a CPU-only
        capture omits them rather than initializing CUDA (see
        :func:`_snapshot_cuda_rng_states`).
    """
    # r65 CLUSTER Z: per-op state logging runs INSIDE the capture window; a
    # TorchLens-initiated snapshot is never model host nondeterminism (the
    # ``get_rng_state`` family carries no registry row TODAY -- this bracket keeps the
    # invariant structural rather than dependent on that vocabulary staying read-free).
    with _suppress_active_monitor_marks():
        rng_dict: dict[str, Any] = {"torch": torch.random.get_rng_state()}
        if not torch_only:
            rng_dict["random"] = random.getstate()
            rng_dict["np"] = np.random.get_state()
        cuda_states = _snapshot_cuda_rng_states()
        if cuda_states:
            rng_dict["torch_cuda_all"] = cuda_states
            rng_dict["torch_cuda"] = cuda_states[0]
        return rng_dict


def set_rng_from_saved_states(rng_states: dict[str, Any]) -> None:
    """Restore RNG engines to a previously captured state.

    Parameters
    ----------
    rng_states:
        Dict produced by :func:`log_current_rng_states`. If empty (RNG capture
        was disabled), this is a no-op.
    """
    if not rng_states:
        return
    # r65 CLUSTER Z: in-forward intervention replay restores engine state THROUGH the
    # monitored ``set_rng_state`` mutation channels; a TorchLens-owned restore is never
    # model host nondeterminism, so the whole restore brackets under the active
    # monitor's mark suppression (the replayed USER op itself runs outside any bracket
    # in ``execute_with_restored_rng_autocast``).
    with _suppress_active_monitor_marks():
        if "random" in rng_states:
            random.setstate(rng_states["random"])
        if "np" in rng_states:
            np.random.set_state(rng_states["np"])
        torch.random.set_rng_state(rng_states["torch"])
        # ``_cuda_rng_unusable`` latches when a CUDA generator read already failed
        # in this process; re-entering the CUDA RNG API to *write* it would raise
        # from inside the ``finally`` restore of
        # ``execute_with_restored_rng_autocast`` and mask the caller's real
        # exception. The failure was already surfaced (warned) at snapshot time.
        if _cuda_rng_unusable or not _is_cuda_available():
            return
        if "torch_cuda_all" in rng_states:
            torch.cuda.set_rng_state_all(rng_states["torch_cuda_all"])
        elif "torch_cuda" in rng_states:
            torch.cuda.set_rng_state(rng_states["torch_cuda"], "cuda")
