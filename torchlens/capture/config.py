"""Internal grouped capture configuration for spine orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .stop import StopDirective


@dataclass(frozen=True, slots=True)
class InternalCaptureConfig:
    """Grouped internal capture configuration for one forward run.

    Parameters
    ----------
    capture_mode
        Active producer mode: exhaustive, fast, or predicate.
    layers_to_save
        Public layer-save selector after entry-boundary normalization.
    grad_layers_to_save
        Gradient layer selector after entry-boundary normalization.
    random_seed
        Forward RNG seed.
    postprocess
        Whether the run should materialize the final ``Trace`` immediately.
    stop
        Compiled stop directive for halt/non-finite/forward-error behavior.
    """

    capture_mode: str
    layers_to_save: Any
    grad_layers_to_save: Any
    random_seed: int | None
    postprocess: bool
    stop: StopDirective


def capture_config_for_trace(trace: object) -> InternalCaptureConfig | None:
    """Return the grouped capture config attached to a trace-like object.

    Parameters
    ----------
    trace
        Trace-like capture session.

    Returns
    -------
    InternalCaptureConfig | None
        Attached config, if present.
    """

    config = getattr(trace, "_capture_config", None)
    if isinstance(config, InternalCaptureConfig):
        return config
    return None
