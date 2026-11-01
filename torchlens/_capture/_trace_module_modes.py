"""Capture-time per-module training-mode recorder for runnable honesty."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _record_runnable_module_training_modes(trace: Trace, model: Any) -> None:
    """Stash the capture-time per-module ``training`` mode for runnable honesty.

    ``self.training`` is module state that is NOT part of the ``state_dict`` and is not a
    model input, yet it steers mode-sensitive ops (BatchNorm running-stats vs batch-stats,
    Dropout on/off). The runnable VERIFIED oracle is a *fresh instance in the captured mode*
    on the given inputs, so the captured mode is DECLARED state the replay reproduces.
    Recording it (per submodule -- submodules can differ) lets the producer declare the mode
    as a witness fact; a mode-sensitive op replayed without a recorded mode fact is downgraded
    to UNVERIFIABLE (fail closed). It runs for every capture (D18 widened the former
    intervention-ready gate so the live refresh projector's mode-claim belt holds on the
    default path; the widening is verdict-inert on the sparse side because the runnable
    producer refuses non-intervention-ready captures outright), touches no tensors, and
    stores an in-memory map consumed by the producer at save time and by the projector belt.

    Parameters
    ----------
    trace:
        Active trace.
    model:
        The prepared source model whose per-module ``training`` flags are recorded.
    """
    from .._runnable_seam import (
        runnable_trace_state,
    )  # deferred: keeps the capture tier eager-import free

    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        return
    modes: dict[str, bool] = {}
    try:
        for name, module in named_modules():
            address = name or "self"
            modes[address] = bool(getattr(module, "training", False))
    except (AttributeError, TypeError):
        return
    if modes:
        runnable_trace_state(trace).module_training_modes = modes
