"""Which backends write the intervention evidence a Trace's pristine reading uses.

``Trace.pristine`` is derived from records an intervention leaves on the trace:
the intervention audit, the operation rows in ``state_history`` and the op
intervention marks. Only the torch intervention engine writes those records.
The other backends that accept interventions apply them without writing any of
it, so an untouched-looking trace from them proves nothing: its pristine
reading is ``None`` (not recorded), never ``True``.
"""

from __future__ import annotations

from .registry import TORCH_BACKEND_NAME


def records_intervention_evidence(backend_name: str) -> bool:
    """Return whether ``backend_name``'s interventions write trace evidence.

    Parameters
    ----------
    backend_name:
        The trace's ``backend`` value.

    Returns
    -------
    bool
        ``True`` only for the torch backend, the one backend whose intervention
        engine writes the audit rows, operation rows and op marks.
    """

    return backend_name == TORCH_BACKEND_NAME
