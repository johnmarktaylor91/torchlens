"""Public I/O and administrative helpers for TorchLens."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._io import rehydrate_nested
from .._io.bundle import cleanup_tmp, load, save
from .._run_state import RunState
from ..intervention.save import save_intervention
from ..intervention.types import InterventionSpec
from ..options import suppress_mutate_warnings
from ..user_funcs import get_model_metadata, list_logs, log_model_metadata, reset_naming_counter


def load_intervention_spec(path: str | Path) -> InterventionSpec:
    """Load an intervention spec through the canonical polymorphic loader.

    Parameters
    ----------
    path:
        Directory containing an intervention ``.tlspec``.

    Returns
    -------
    InterventionSpec
        Loaded intervention spec.

    Raises
    ------
    TypeError
        If ``path`` does not load as an intervention spec.
    """

    loaded = load(path)
    if not isinstance(loaded, InterventionSpec):
        raise TypeError("torchlens.io.load_intervention_spec expected an intervention spec.")
    return loaded


__all__ = [
    "RunState",
    "cleanup_tmp",
    "get_model_metadata",
    "list_logs",
    "load",
    "load_intervention_spec",
    "log_model_metadata",
    "rehydrate_nested",
    "reset_naming_counter",
    "save",
    "save_intervention",
    "suppress_mutate_warnings",
]
