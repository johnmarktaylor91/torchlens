"""Streaming-append detection and its refusal message for append reruns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._live_hook_plan import (
    _assign_unique_plan_ids as _assign_unique_plan_ids,
    _hook_plan_identifier as _hook_plan_identifier,
)

if TYPE_CHECKING:
    from .._data_classes.trace import Trace


def _is_streaming_append_active(log: Trace) -> bool:
    """Return whether append would need to update active streaming state.

    Parameters
    ----------
    log:
        Trace inspected before append capture.

    Returns
    -------
    bool
        True when a bundle writer or out sink is still attached.
    """

    return (
        getattr(log, "_out_writer", None) is not None or getattr(log, "_out_sink", None) is not None
    )


def _streaming_append_error_message(log: Trace) -> str:
    """Build a descriptive streaming append rejection message.

    Parameters
    ----------
    log:
        Trace whose active streaming handles block append.

    Returns
    -------
    str
        User-facing exception message.
    """

    details = []
    if getattr(log, "_out_writer", None) is not None:
        details.append("path streaming")
    if getattr(log, "_out_sink", None) is not None:
        details.append("on_save streaming")
    details_text = ", ".join(details) if details else "streaming"
    return (
        f"Trace {log._source_class_name!r} has active {details_text}; "
        "rerun(append=True) cannot update streamed activation storage. Save and reload "
        "the trace before appending, or disable streaming for this capture."
    )
