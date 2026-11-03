"""The guarded PEP 678 exception-note helper, shared by every crash-note site.

``BaseException.add_note`` exists from Python 3.11. On 3.10 (a supported
version) a bare ``exc.add_note(...)`` raises ``AttributeError``; the crash-note
sites wrapped it in a blanket suppress, so the note telling the user a partial
capture exists silently never appeared there. This helper adds the note where
notes exist and falls back to a ``RuntimeWarning`` carrying the same text where
they do not.
"""

from __future__ import annotations

import warnings

__tl_layer__ = "L1"


def annotate_guarded(exc: BaseException, note: str, *, stacklevel: int = 3) -> None:
    """Attach ``note`` to ``exc``; on Python 3.10 surface it as a warning.

    Parameters
    ----------
    exc:
        The in-flight exception to annotate.
    note:
        Note text.
    stacklevel:
        ``warnings.warn`` stack level for the 3.10 fallback.
    """

    add_note = getattr(exc, "add_note", None)
    if add_note is not None:
        add_note(note)
    else:  # Python 3.10: no PEP 678 notes -- surface via warning.
        warnings.warn(note, RuntimeWarning, stacklevel=stacklevel)


_PENDING_FAILURE_NOTES_KEY = "_pending_failure_notes"
"""Capture-session slot holding notes for the error the capture door raises."""


def defer_failure_note(session: object, exc: BaseException, note: str) -> None:
    """Queue ``note`` for the error a failing capture door raises.

    A note written while the forward's exception unwinds must not mutate that
    exception: the capture door wraps it in a ``TraceCaptureError`` (or lets a
    TorchLens error propagate) and moves queued notes onto the raised error.
    Without a capture session to hold the queue, ``exc`` is annotated directly.

    Parameters
    ----------
    session:
        Active capture session (the ``Trace`` being captured), or ``None``.
    exc:
        The in-flight exception the note describes.
    note:
        Note text.
    """

    slots = getattr(session, "__dict__", None)
    if not isinstance(slots, dict):
        annotate_guarded(exc, note)
        return
    slots.setdefault(_PENDING_FAILURE_NOTES_KEY, []).append(note)


def pop_failure_notes(session: object) -> tuple[str, ...]:
    """Remove and return the notes queued on ``session`` by :func:`defer_failure_note`.

    Parameters
    ----------
    session:
        Capture session whose queue is drained.

    Returns
    -------
    tuple[str, ...]
        Queued notes in arrival order (empty when none).
    """

    slots = getattr(session, "__dict__", None)
    if not isinstance(slots, dict):
        return ()
    return tuple(slots.pop(_PENDING_FAILURE_NOTES_KEY, ()))
