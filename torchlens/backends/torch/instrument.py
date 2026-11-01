"""Model instrumentation entry for the WRAP stage.

Seam entry: ``instrumented()`` composes today's preparation calls in today's
order (compiled-capture preparation, one-time model preparation, per-session
preparation) and runs today's session cleanup on exit. RUN still makes those
calls inline in ``torchlens._capture.run.run_and_log_inputs_through_model``;
the WRAP lane fills this module and RUN adopts it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from ..._capture.stages import InstrumentedModel
from ..._capture_state_helpers import prepare_compiled_capture

if TYPE_CHECKING:
    from torch import nn

    from ..._capture.stages import CaptureRequest
    from .backend import TorchBackend


@contextmanager
def instrumented(
    backend: TorchBackend,
    trace: Any,
    model: nn.Module,
    cleanup_payload: Any,
    request: CaptureRequest | None = None,
) -> Iterator[InstrumentedModel]:
    """Prepare ``model`` for one capture and clean the session up on exit.

    Parameters
    ----------
    backend:
        The torch capture backend.
    trace:
        The trace being populated.
    model:
        The user's model.
    cleanup_payload:
        The payload ``TorchBackend.cleanup_model_session`` takes on exit.
    request:
        The capture request, when known.

    Yields
    ------
    InstrumentedModel
        The prepared model and its compiled-capture preparation.
    """
    del request
    with prepare_compiled_capture(model) as compiled_prep:
        backend.prepare_model_once(model)
        backend.prepare_model_session(trace, model)
        try:
            yield InstrumentedModel(model=model, compiled_prep=compiled_prep)
        finally:
            backend.cleanup_model_session(trace, cleanup_payload)
