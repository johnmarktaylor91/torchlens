"""Compatibility exports for capture event accumulators.

``CaptureEvents`` moved to :mod:`torchlens.ir.capture_events`; this module remains
for internal imports that still use the previous module path.
"""

from __future__ import annotations

from .capture_events import (
    CaptureEvents,
    LiveOpRecord,
    live_record_for_label,
    register_live_event,
    replace_op_event,
)

__all__ = [
    "CaptureEvents",
    "LiveOpRecord",
    "live_record_for_label",
    "register_live_event",
    "replace_op_event",
]
