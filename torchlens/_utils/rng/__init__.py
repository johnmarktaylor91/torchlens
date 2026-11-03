"""RNG state for reproducible capture and replay, and the host-nondeterminism monitor.

Seam. Capture, replay and validation reach every RNG stream through this package:

* :mod:`.streams`: snapshot and restore of the Python ``random``, NumPy, torch CPU and
  live CUDA generator states (the per-op record shape), plus the host-only pair the
  sparse runnable path uses to detect host RNG consumption.
* :mod:`.seeding`: :func:`set_random_seed` seeds every engine from one capture seed;
  :func:`aten_qualname_is_seeded_rng` names the seeded ATen operators.
* :mod:`.restore`: the one door for preserving the user's streams across TorchLens work
  (:class:`RngStreamsRestore`, :func:`restores_rng_streams`) and the replay bracket
  :func:`execute_with_restored_rng_autocast`.
* :mod:`.autocast`: autocast and execution-mode state captured per op and re-entered at
  replay.
* :mod:`.monitor` with :mod:`.registry`, :mod:`.channels`, :mod:`.engine_state`,
  :mod:`.inventory_types`, :mod:`.call_sites` and :mod:`.monitor_state`: the
  capture-window monitor that witnesses host entropy, clock and RNG-instance
  consumption the replay cannot reproduce.

Ordering invariant: per-op RNG states are captured *before* ``active_logging()`` is
entered, because entering the logging context may itself call decorated functions that
advance the RNG.

Nothing here is public surface; the names below are the package's import seam.
"""

from __future__ import annotations

from .autocast import _AUTOCAST_DEVICES, AutocastRestore, log_current_autocast_state
from .monitor import host_nondeterminism_monitor
from .registry import HOST_NONDETERMINISM_REGISTRY
from .restore import RngStreamsRestore, execute_with_restored_rng_autocast, restores_rng_streams
from .seeding import aten_qualname_is_seeded_rng, set_random_seed
from .streams import (
    _snapshot_cuda_rng_states,
    host_rng_advanced,
    log_current_rng_states,
    restore_host_rng,
    set_rng_from_saved_states,
    snapshot_host_rng,
)

__all__ = [
    "HOST_NONDETERMINISM_REGISTRY",
    "AutocastRestore",
    "RngStreamsRestore",
    "_AUTOCAST_DEVICES",
    "_snapshot_cuda_rng_states",
    "aten_qualname_is_seeded_rng",
    "execute_with_restored_rng_autocast",
    "host_nondeterminism_monitor",
    "host_rng_advanced",
    "log_current_autocast_state",
    "log_current_rng_states",
    "restore_host_rng",
    "restores_rng_streams",
    "set_random_seed",
    "set_rng_from_saved_states",
    "snapshot_host_rng",
]
