"""Resolved stop, capture and identity configuration for one predicate recording pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._capture.stages import InternalCaptureConfig

if TYPE_CHECKING:
    from torch import nn

    from .._data_classes.trace import Trace
    from .options import _RecordingConfig


def _configure_predicate_capture(
    trace: Trace, options: _RecordingConfig, *, model: nn.Module
) -> None:
    """Install the capture configuration and the root identity facts for one pass.

    The exhaustive path stamps the identity facts in ``_run_setup``; the Recorder
    builds its own trace shell, so the same facts are written here. Every recorded
    pass is the Recorder calling the model's ``forward``, which makes the root entry
    point a ``module_call`` honestly. The weight fingerprint is skipped on purpose:
    hashing every parameter per pass is a fixed cost the fast path refuses, and the
    rerun identity gate treats an absent fingerprint as no evidence, not a match.

    Parameters
    ----------
    trace
        Predicate-mode trace prepared for the next recorded pass.
    options
        Resolved recorder settings supplying the capture seed.
    model
        The module the Recorder runs; its class and object identity are stamped.
    """

    trace._capture_config = InternalCaptureConfig(
        capture_mode="predicate",
        layers_to_save=[],
        grad_layers_to_save=[],
        random_seed=options.seed,
        postprocess=False,
        stop=trace._stop_directive,
    )
    # The same ``module.qualname`` spelling ``_qualname_for_model`` writes on the
    # exhaustive path, spelled here so this leaf stays out of the capture-state cycle.
    model_type = type(model)
    qualname = f"{model_type.__module__}.{model_type.__qualname__}"
    trace._source_class_qualname = qualname
    trace._provenance_root_entry_point = f"module_call:{qualname}.forward"
    trace._provenance_model_object_id = id(model)
