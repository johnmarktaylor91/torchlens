"""Trace backward-capture functions reached from the records through ``_data_classes._doors.backends_torch``.

``Trace.backward`` and ``Trace.capture_backward`` are the thin methods; the two refusals here
(``_refuse_loaded_backward_capture``, ``_refuse_stats_capture``) are shared with the validation and run
destinations, which import them downward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ..._capture.outcome import require_capture_capability
from ..._capture.structure_only import require_structure_only_capability
from ..._errors import PayloadUnavailableError
from ..._trace_state import is_stats_capture
from ...errors import RunCapabilityUnavailableError
from .._backend_protocol import refuse_backward_capture_unsupported
from ..registry import get_backend_spec, require_capability_implementation

if TYPE_CHECKING:
    from ..._data_classes.trace import Trace

__all__ = [
    "capture_backward",
    "backward",
]


def _refuse_loaded_backward_capture(trace: Trace, entry_point: str) -> None:
    """Refuse live backward capture on a bundle-loaded analysis trace.

    A loaded trace carries no live autograd
    handles tied to its recorded forward, so ``backward`` used to hook
    whatever FOREIGN graph the caller's loss came from, half-mutate the trace
    (backward passes and grad_fns from an unrelated forward), and die with an
    untyped ``AttributeError`` -- after which the backward drawing silently
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
        "backward_ready=True))) and call backward on that Trace, or inspect the "
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


def _require_live_replay(trace: Any) -> None:
    """Refuse a live replay that the trace's capture cannot drive.

    The gates every propagation runs before touching the graph, shared by the
    value ``do()`` and the hook ``do()`` that starts only from its own hooks: a halted or failed
    capture, a structure-only capture, and a ``save="stats"`` trace.
    """

    require_capture_capability(trace, "live_replay")
    require_structure_only_capability(trace, "live_replay")
    _refuse_stats_capture(trace, "live_replay")


def backward(trace: Trace, loss: Any, **backward_kwargs: Any) -> Trace:
    """Body of ``Trace.backward``; the method carries the documented contract.

    ``loss`` is the scalar to differentiate, or a callable mapping the model's
    return value (``trace.output``) to it, with ``loss_fn``'s meaning. PyTorch
    evaluates the callable here on the live output; a native backend receives
    the callable itself, because its framework differentiates the function.
    """
    # N3: the backward projection assumes a structurally complete captured
    # forward graph; HALTED stays allowed (the autograd graph of a halted
    # capture IS the captured prefix).
    require_capture_capability(trace, "backward")
    require_structure_only_capability(trace, "backward_grads")
    _refuse_stats_capture(trace, "backward_grads")
    _refuse_loaded_backward_capture(trace, "backward")
    spec = get_backend_spec(trace._provenance_backend)
    refuse_backward_capture_unsupported(spec)
    surface = cast(Any, require_capability_implementation(spec, "backward_capture"))
    if callable(loss) and spec.name == "torch":
        loss = loss(trace.output)
    return cast("Trace", surface.trace_backward(trace, loss, **backward_kwargs))


def capture_backward(trace: Trace) -> Any:
    """Body of ``Trace.capture_backward``; the method carries the documented contract."""
    require_capture_capability(trace, "backward")
    require_structure_only_capability(trace, "backward_grads")
    _refuse_stats_capture(trace, "backward_grads")
    _refuse_loaded_backward_capture(trace, "recording_backward")
    spec = get_backend_spec(trace._provenance_backend)
    refuse_backward_capture_unsupported(spec)
    surface = cast(Any, require_capability_implementation(spec, "backward_capture"))
    return surface.recording_backward(trace)


def sync_backward_projection_if_needed(trace: Trace) -> None:
    """Synchronize lazy backward projections from runtime sidecar events."""

    if trace.__dict__.get("_tl_active_backward_bracket", False):
        return
    if not getattr(trace, "backward_events", ()):
        return
    backend = getattr(trace, "_provenance_backend", None) or "torch"
    if backend != "torch":
        # Another backend's events project through its own backward surface.
        spec = get_backend_spec(backend)
        cast(Any, require_capability_implementation(spec, "backward_capture")).sync_projection(
            trace
        )
        return
    from .backward import (
        _backward_finalize_pending,
        _close_implicit_backward_pass_if_open,
        _materialize_backward_projections,
    )

    _close_implicit_backward_pass_if_open(trace)
    # A read from INSIDE an engine invocation journals but must not
    # materialize while the close's FINALIZE step is still pending:
    # materializing there would publish records ahead of the D2H
    # fence. The first post-pass read finalizes fully.
    if not _backward_finalize_pending(trace):
        _materialize_backward_projections(trace)
