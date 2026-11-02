"""Trace backward-capture functions reached from the records through ``_data_classes._doors.backends_torch``.

``Trace.log_backward`` and ``Trace.capture_backward`` are the thin methods; the two refusals here
(``_refuse_loaded_backward_capture``, ``_refuse_stats_capture``) are shared with the validation and run
destinations, which import them downward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch

from ..._capture.outcome import require_capture_capability
from ..._capture.structure_only import require_structure_only_capability
from ..._errors import PayloadUnavailableError
from ...errors import RunCapabilityUnavailableError
from .._backend_protocol import refuse_backward_capture_unsupported
from ..registry import get_backend_spec
from .backward import (
    log_backward as _log_backward_impl,
    recording_backward as _recording_backward_impl,
)

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = [
    "capture_backward",
    "log_backward",
]


def _refuse_loaded_backward_capture(trace: Trace, entry_point: str) -> None:
    """Refuse live backward capture on a bundle-loaded analysis trace.

    WT1 A-IV item 21 (lane A08): a loaded trace carries no live autograd
    handles tied to its recorded forward, so ``log_backward`` used to hook
    whatever FOREIGN graph the caller's loss came from, half-mutate the trace
    (backward passes and grad_fns from an unrelated forward), and die with an
    untyped ``AttributeError`` -- after which ``draw_backward`` silently
    rendered that wrong graph. Refuse typed BEFORE any mutation.
    """
    try:
        loaded_from_bundle = trace._loaded_from_bundle
    except AttributeError:
        loaded_from_bundle = False
    if not loaded_from_bundle:
        return
    from ..._runnable import RunnableErrorCode

    raise RunCapabilityUnavailableError(
        f"{entry_point}() is unavailable on a bundle-loaded Trace: the loaded "
        "artifact carries no live autograd graph tied to its recorded "
        "forward, so a backward here would capture whatever unrelated graph "
        "the loss came from and record a wrong backward pass. Re-capture "
        "live (tl.trace(model, x, capture=tl.options.CaptureOptions("
        "backward_ready=True))) and call log_backward on that Trace, or inspect the "
        "backward metadata this artifact already persisted.",
        code=RunnableErrorCode.RUN_CAPABILITY_UNAVAILABLE.value,
        detection_stage="loaded_backward_capture",
    )


def _refuse_stats_capture(trace: Any, purpose: str) -> None:
    """Refuse replay and validation on a ``save="stats"`` trace.

    A reduce-and-discard trace kept no payloads, so a replay has nothing to
    start from and validation nothing to compare; refusing is the honest
    outcome (a missing payload must never read as a pass).
    """

    from ...stats._capture_stats import is_stats_capture

    if not is_stats_capture(trace):
        return

    action = "validation" if purpose == "validation_entry" else "replay"
    raise PayloadUnavailableError(
        f"this trace was captured with save='stats', which keeps no activation payloads, "
        f"so {action} has nothing to check against",
        code="stats_capture_not_replayable",
        remedy="capture again keeping payloads, tl.trace(model, x, save='all'), and run "
        f"{action} on that trace; tl.validate(model, x, scope='forward') runs its own full capture",
        purpose=purpose,
    )


def log_backward(trace: Trace, loss: torch.Tensor, **backward_kwargs: Any) -> Trace:
    """Body of ``Trace.log_backward``; the method carries the documented contract."""
    # N3: the backward projection assumes a structurally complete captured
    # forward graph; HALTED stays allowed (the autograd graph of a halted
    # capture IS the captured prefix).
    require_capture_capability(trace, "backward")
    require_structure_only_capability(trace, "backward_grads")
    _refuse_stats_capture(trace, "backward_grads")
    _refuse_loaded_backward_capture(trace, "log_backward")
    spec = get_backend_spec(trace._provenance_backend)
    refuse_backward_capture_unsupported(spec)
    return cast("Trace", _log_backward_impl(trace, loss, **backward_kwargs))


def capture_backward(trace: Trace) -> Any:
    """Body of ``Trace.capture_backward``; the method carries the documented contract."""
    require_capture_capability(trace, "backward")
    require_structure_only_capability(trace, "backward_grads")
    _refuse_stats_capture(trace, "backward_grads")
    _refuse_loaded_backward_capture(trace, "recording_backward")
    spec = get_backend_spec(trace._provenance_backend)
    refuse_backward_capture_unsupported(spec)
    return _recording_backward_impl(trace)
