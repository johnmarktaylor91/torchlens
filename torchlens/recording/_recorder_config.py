"""Resolved stop and capture configuration for one predicate recording pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._capture.stages import InternalCaptureConfig

if TYPE_CHECKING:
    from .._data_classes.trace import Trace
    from ._options import _RecordingConfig


def _configure_predicate_capture(trace: Trace, options: _RecordingConfig) -> None:
    """Install the capture configuration using the trace's resolved stop directive.

    Parameters
    ----------
    trace
        Predicate-mode trace prepared for the next recorded pass.
    options
        Resolved recorder settings supplying the capture seed.
    """

    trace._capture_config = InternalCaptureConfig(
        capture_mode="predicate",
        layers_to_save=[],
        grad_layers_to_save=[],
        random_seed=options.seed,
        postprocess=False,
        stop=trace._stop_directive,
    )
