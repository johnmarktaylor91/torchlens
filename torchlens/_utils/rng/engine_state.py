"""Python and NumPy global-engine STATE rows (``getstate``/``setstate``/``seed`` surface).

An in-forward ``getstate -> draw -> setstate`` leaves the boundary snapshot equal while the
forward consumed a host scalar; these ``replayable_read`` rows make the replay require the
capture seed instead of reading as no consumption.
"""

from __future__ import annotations

import random as _random_module
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .monitor import host_nondeterminism_monitor


# ---- Python / NumPy global-engine state surface (W051 / AUD-CODE 2.17) -----------------
#
# The seed-replayable global engines are witnessed by a before/after state snapshot at
# the capture boundary. An owner-thread ``getstate -> draw -> setstate`` (or a
# ``seed``/``set_state`` from a pre-window value) leaves the snapshot compare EQUAL while
# the forward consumed a host scalar -- a false ``verified`` when the replay's ambient
# engine state differs. These rows carry the ``replayable_read`` disposition (the torch
# ``initial_seed`` analog): they set ``host_rng_consumed`` so the replay REQUIRES the
# capture seed (at which the fresh oracle reproduces the same draw) and never ceiling --
# the contract's own asymmetry note: a python/numpy in-forward reseed is self-reproducing
# on-seed, unlike a torch-engine mutation, which desyncs replay-re-executed DAG RNG ops.

GLOBAL_ENGINE_STATE_ROWS: tuple[tuple[str, str, str], ...] = (
    # (public module spelling, method, channel)
    ("random", "getstate", "random.getstate"),
    ("random", "setstate", "random.setstate"),
    ("random", "seed", "random.seed"),
    ("numpy.random", "get_state", "numpy.random.get_state"),
    ("numpy.random", "set_state", "numpy.random.set_state"),
    ("numpy.random", "seed", "numpy.random.seed"),
)
"""Module-attr rows of the Python / legacy-NumPy global-engine STATE surface."""

GLOBAL_ENGINE_STATE_DISPOSITION = "replayable_read"
"""Every global-engine state row is consumed-only: replay at the capture seed reproduces."""


def registry_rows() -> tuple[tuple[str, str, str, str, str], ...]:
    """``HostNondeterminismRow`` tuples for the global-engine state surface.

    One ``module_patch`` row per spelling (thread-independent -- a foreign thread's
    ``random.seed()`` desyncs the shared engine exactly like the owner's) plus one
    ``c_call_identity`` row per Python-``random`` spelling for the held-reference layer
    (``from random import getstate`` calls the singleton's bound Python method, whose
    body enters the C base ``_random.Random.getstate`` -- classified by receiver identity
    on the owner and every in-window hooked thread). The legacy NumPy singleton's bound
    Cython methods emit no ``c_call`` on numpy>=2, so their held-reference spelling is a
    disclosed residual of the same class as the numpy>=2 instance-draw shape.
    """

    rows: list[tuple[str, str, str, str, str]] = []
    for _module_name, _method, channel in GLOBAL_ENGINE_STATE_ROWS:
        rows.append(
            ("rng_global_state", channel, "module_patch", "any", GLOBAL_ENGINE_STATE_DISPOSITION)
        )
        if channel.startswith("random."):
            rows.append(
                (
                    "rng_global_state",
                    channel,
                    "c_call_identity",
                    "hooked",
                    GLOBAL_ENGINE_STATE_DISPOSITION,
                )
            )
    return tuple(rows)


def global_engine_state_prefixes() -> dict[int, str]:
    """``id(global engine singleton) -> channel prefix`` for the held-ref ``c_call`` layer."""

    prefixes: dict[int, str] = {}
    inst = getattr(_random_module, "_inst", None)
    if inst is not None:
        prefixes[id(inst)] = "random"
    np_singleton = getattr(getattr(np.random, "mtrand", None), "_rand", None)
    if np_singleton is not None:
        prefixes[id(np_singleton)] = "numpy.random"
    return prefixes


_GLOBAL_ENGINE_STATE_METHODS: frozenset[str] = frozenset(
    {"getstate", "setstate", "seed", "get_state", "set_state"}
)


def global_engine_state_channel(prefix: str, name: Any) -> str | None:
    """Channel for a state method ``c_call`` on a global-engine singleton, or ``None``."""

    if isinstance(name, str) and name in _GLOBAL_ENGINE_STATE_METHODS:
        return f"{prefix}.{name}"
    return None


def install_global_engine_state_surfaces(monitor: host_nondeterminism_monitor) -> None:
    """Patch the module-attr spellings of the Python / legacy-NumPy state surface.

    ``numpy.random.mtrand`` re-exports the same bound methods as ``numpy.random``; both
    module attributes are patched so either spelling marks. TorchLens's own in-window
    snapshot/restore/seed helpers (``log_current_rng_states``,
    ``set_rng_from_saved_states``, ``set_random_seed``) run under the active monitor's
    mark suppression and never self-mark through these wrappers.
    """

    holders: dict[str, tuple[Any, ...]] = {
        "random": (_random_module,),
        "numpy.random": tuple(
            module
            for module in (np.random, getattr(np.random, "mtrand", None))
            if module is not None
        ),
    }
    for module_name, method, channel in GLOBAL_ENGINE_STATE_ROWS:
        for holder in holders[module_name]:
            original = getattr(holder, method, None)
            if original is None or not callable(original):
                continue
            monitor._patch_attr(holder, method, _replayable_wrapper(monitor, original, channel))


def _replayable_wrapper(monitor: host_nondeterminism_monitor, original: Any, channel: str) -> Any:
    """Build a consumed-only marking passthrough for one global-engine state method."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Mark the replayable-read channel at entry, then delegate to the original."""

        monitor._mark_replayable(channel)
        return original(*args, **kwargs)

    return wrapper
