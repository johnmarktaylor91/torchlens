"""RNG and autocast state capture/restore for reproducible forward-pass replay."""

import random
from typing import Any, Dict, List

import numpy as np
import torch

from .tensor_utils import _is_cuda_available

_AUTOCAST_DEVICES = ("cpu", "cuda")


def set_random_seed(seed: int):
    """Sets the random seed for all random number generators.

    Args:
        seed: Seed to set.

    Returns:
        Nothing.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_current_rng_states() -> Dict:
    """Utility function to fetch sufficient information from all RNG states to recover the same state later.

    Returns:
        Dict with sufficient information to recover all RNG states.
    """
    rng_dict = {
        "random": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if _is_cuda_available():
        rng_dict["torch_cuda"] = torch.cuda.get_rng_state("cuda")
    return rng_dict


def set_rng_from_saved_states(rng_states: Dict):
    """Utility function to set the state of random seeds to a cached value.

    Args:
        rng_states: Dict of rng_states saved by get_random_seed_states

    Returns:
        Nothing, but correctly sets all random seed states.
    """
    random.setstate(rng_states["random"])
    np.random.set_state(rng_states["np"])
    torch.random.set_rng_state(rng_states["torch"])
    if _is_cuda_available() and "torch_cuda" in rng_states:
        torch.cuda.set_rng_state(rng_states["torch_cuda"], "cuda")


def log_current_autocast_state() -> Dict:
    """Capture the current autocast enabled/dtype state for all supported devices.

    Returns:
        Dict mapping device name to {enabled: bool, dtype: torch.dtype}.
    """
    state = {}
    for device in _AUTOCAST_DEVICES:
        try:
            state[device] = {
                "enabled": torch.is_autocast_enabled(device),
                "dtype": torch.get_autocast_dtype(device),
            }
        except (RuntimeError, TypeError):
            pass
    return state


class AutocastRestore:
    """Context manager that restores saved autocast states.

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
        for ctx in reversed(self._contexts):
            ctx.__exit__(*exc_info)
