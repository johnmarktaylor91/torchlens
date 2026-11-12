"""Which traces carry the intervention evidence a Trace's pristine reading uses.

``Trace.is_pristine`` is derived from records an intervention leaves on the trace:
the intervention audit, the operation rows in ``state_history``, the trace state
and the op intervention marks. The torch intervention engine writes those
records for every intervention. Finished-trace edits on every backend go through
the shared replay engine, which writes the same records
(``intervention/_backend_replay_run.py``). A non-torch capture writes its
part with :func:`stamp_capture_evidence`: a capture that ran with ``intervene=``
marks the trace and the ops its edits fired on, and every stamped capture,
edited or not, is known to carry complete evidence. An unstamped non-torch
trace proves nothing, so its pristine reading is ``None`` (not recorded),
never ``True``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .._trace_state import TraceState
from ._backend_names import TORCH_BACKEND_NAME

#: Trace attribute a non-torch capture sets once its intervention evidence is written.
CAPTURE_EVIDENCE_STAMP = "_tl_capture_intervention_evidence"


def records_intervention_evidence(backend_name: str) -> bool:
    """Return whether every trace of ``backend_name`` carries intervention evidence.

    Parameters
    ----------
    backend_name:
        The trace's ``backend`` value.

    Returns
    -------
    bool
        ``True`` only for the torch backend, whose capture and intervention
        engine always write the audit rows, operation rows and op marks. Other
        backends carry it per trace (:func:`trace_records_intervention_evidence`).
    """

    return backend_name == TORCH_BACKEND_NAME


def trace_records_intervention_evidence(trace: Any) -> bool:
    """Return whether this trace carries complete intervention evidence.

    Parameters
    ----------
    trace:
        Live ``Trace``.

    Returns
    -------
    bool
        ``True`` for a torch trace, and for a non-torch trace whose capture
        stamped its evidence with :func:`stamp_capture_evidence`.
    """

    if records_intervention_evidence(str(getattr(trace, "_provenance_backend", None))):
        return True
    return trace.__dict__.get(CAPTURE_EVIDENCE_STAMP) is True


def stamp_capture_evidence(
    trace: Any, *, intervened: bool, edited_labels: Iterable[str] = ()
) -> None:
    """Write a non-torch capture's intervention evidence and stamp it complete.

    Called by a backend's capture once its forward pass ends. A capture that
    ran with ``intervene=`` sets the live-captured state (touch evidence, as
    the torch capture sets it for a hook plan) and marks each op its edits
    fired on as edited; a capture without ``intervene=`` writes nothing but the
    stamp, so its trace can read pristine.

    Parameters
    ----------
    trace:
        The finished non-torch ``Trace``.
    intervened:
        Whether the capture ran with ``intervene=`` (whether or not an edit fired).
    edited_labels:
        Labels of the ops whose recorded value an edit replaced.
    """

    for label in edited_labels:
        trace._layer_dict_all_keys[label]._internal_set("is_edited", True)
    if intervened:
        trace.state = TraceState.LIVE_CAPTURED
    trace.__dict__[CAPTURE_EVIDENCE_STAMP] = True
