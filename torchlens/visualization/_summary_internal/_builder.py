"""The compact ``repr`` of a ``Trace``.

The historical text-summary builder that once lived here was removed with the
legacy ``summary()`` spellings; ``trace.summary()`` renders through
``torchlens.report`` and ``trace.provenance()`` through ``_discoverability``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...data_classes.trace import Trace


def format_model_repr(trace: Trace) -> str:
    """Return a short ``repr`` string for a ``Trace``.

    Parameters
    ----------
    trace:
        Logged model metadata to summarize.

    Returns
    -------
    str
        Short two-line representation.
    """
    state = getattr(getattr(trace, "state", None), "name", "UNKNOWN")
    model_class_name = getattr(trace, "model_class_name", None)
    tracing_finished = getattr(trace, "_tracing_finished", True)
    if not tracing_finished:
        return (
            f"Trace(name={getattr(trace, 'trace_label', None)!r}, "
            f"model_class_qualname={model_class_name!r}, layers={_live_op_count(trace)}, "
            f"state={state})"
        )

    layer_logs = getattr(trace, "layer_logs", {}) or {}
    # Weightsfree memo L7: the repr was an unmarked channel — a slice of a
    # hypothesis is a hypothesis, and so is the identity card. The claim
    # ladder rides the session discharge state (HYPOTHESIS / CORROBORATED /
    # REFUTED), never a bare flag.
    structure_note = ""
    if bool(getattr(trace, "structure_only", False)):
        from ...capture.structure_only import claim_status_for

        structure_note = f", structure_only={claim_status_for(trace).value.upper()}"
    return (
        f"Trace(name={getattr(trace, 'trace_label', None)!r}, "
        f"model_class_qualname={model_class_name!r}, layers={len(layer_logs)}, "
        f"state={state}{structure_note})"
    )


def _live_op_count(trace: Trace) -> int:
    """Return live op-event count when capture events are present.

    Parameters
    ----------
    trace:
        Trace to inspect.

    Returns
    -------
    int
        Number of live operation events or raw logs.
    """

    events = getattr(trace, "capture_events", None)
    if events is not None and getattr(events, "op_events", None) is not None:
        return len(events.op_events)
    return len(trace._raw_graph_ws.raw_layer_dict)
