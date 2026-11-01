"""Build the exception a capture door raises when the forward fails.

A non-TorchLens exception raised inside the forward (the model's code, a user
hook, or a torch op the forward calls) escapes every capture door as one typed
:class:`~torchlens.errors.TraceCaptureError`, raised ``from`` the original so
its type, text and traceback stay intact. The original exception object is
never mutated: the partial capture, the origin and every TorchLens note ride
the ``TraceCaptureError``. TorchLens's own errors (a :class:`TorchLensError`,
or any exception class defined inside the ``torchlens`` package) propagate as
themselves and carry the partial capture as ``partial_trace``.
"""

from __future__ import annotations

import contextlib
from types import MappingProxyType
from typing import Any

from .._errors import TraceCaptureError
from .._utils._exception_notes import annotate_guarded, pop_failure_notes
from ..errors import TorchLensError
from .outcome import FailureOrigin, classify_failure_origin, safe_exception_str

_ORIGIN_SENTENCES = MappingProxyType(
    {
        FailureOrigin.USER_OP: "The model's own code raised this error, not TorchLens.",
        FailureOrigin.TORCHLENS: (
            "The innermost frame that raised is inside TorchLens, so this may be a "
            "TorchLens bug; please report it with the chained traceback."
        ),
        FailureOrigin.UNKNOWN: (
            "TorchLens could not attribute the raising frame to the model or to TorchLens."
        ),
    }
)


def is_torchlens_exception(exc: BaseException) -> bool:
    """Return whether ``exc`` is one of TorchLens's own exception types.

    Parameters
    ----------
    exc:
        Exception that escaped the forward.

    Returns
    -------
    bool
        True for a :class:`TorchLensError` or any exception class defined in
        the ``torchlens`` package; such errors propagate unwrapped.
    """

    if isinstance(exc, TorchLensError):
        return True
    module = getattr(type(exc), "__module__", "") or ""
    return module == "torchlens" or module.startswith("torchlens.")


def failed_forward_error(trace: object, exc: Exception, num_ops: int | None) -> Exception:
    """Return the exception the capture door raises for a failed forward.

    Parameters
    ----------
    trace:
        Capture session the forward failed in, read before cleanup.
    exc:
        Exception that escaped the forward.
    num_ops:
        Committed op count at the time of failure, when known.

    Returns
    -------
    Exception
        ``exc`` itself when it is a TorchLens exception, else a new
        :class:`TraceCaptureError` whose ``__cause__`` is ``exc``.
    """

    notes = pop_failure_notes(trace)
    if is_torchlens_exception(exc):
        _add_notes(exc, notes)
        return exc
    origin = classify_failure_origin(exc)
    last_op = _last_captured_op_label(trace)
    error = TraceCaptureError(
        _message(exc, origin, num_ops, last_op),
        origin=origin,
        cause_type=type(exc).__name__,
        last_captured_op=last_op,
        num_captured_ops=num_ops,
    )
    error.__cause__ = exc
    _add_notes(error, notes)
    return error


def _add_notes(error: BaseException, notes: tuple[str, ...]) -> None:
    """Attach queued in-forward notes to the error the door raises.

    Parameters
    ----------
    error:
        Exception the capture door is about to raise.
    notes:
        Notes queued while the forward's exception unwound.
    """

    for note in notes:
        with contextlib.suppress(Exception):
            annotate_guarded(error, note)


def attach_partial_trace(error: BaseException, partial: Any) -> None:
    """Attach the partial capture to the escaping error.

    Parameters
    ----------
    error:
        Exception the capture door is about to raise.
    partial:
        Partial capture built from the failed session.

    Raises
    ------
    Exception
        Whatever the exception's attribute assignment raises; the caller
        reports it without masking ``error``.
    """

    if isinstance(error, TraceCaptureError):
        error.partial_trace = partial
        return
    if getattr(error, "partial_trace", None) is None:
        error.partial_trace = partial  # type: ignore[attr-defined]


def original_exception(error: BaseException) -> BaseException:
    """Return the exception the forward raised, unwrapping a TraceCaptureError.

    Parameters
    ----------
    error:
        Exception a capture door raised.

    Returns
    -------
    BaseException
        The ``__cause__`` of a :class:`TraceCaptureError`, else ``error``.
    """

    if isinstance(error, TraceCaptureError) and error.__cause__ is not None:
        return error.__cause__
    return error


def _message(
    exc: BaseException,
    origin: FailureOrigin,
    num_ops: int | None,
    last_op: str | None,
) -> str:
    """Compose the TraceCaptureError message.

    Parameters
    ----------
    exc:
        Exception the forward raised.
    origin:
        Classified failure origin.
    num_ops:
        Committed op count, when known.
    last_op:
        Label of the last captured op, when known.

    Returns
    -------
    str
        One message naming the original exception, the capture frontier and
        where the partial capture lives.
    """

    text = safe_exception_str(exc)
    head = f"The forward raised {type(exc).__name__}" + (f": {text}" if text else "")
    if num_ops is None:
        frontier = "TorchLens could not count the ops captured before the failure"
    else:
        noun = "op" if num_ops == 1 else "ops"
        frontier = f"TorchLens captured {num_ops} {noun} before the failure"
    if last_op is not None:
        frontier += f"; the last captured op was `{last_op}` and the raising call came after it"
    origin_sentence = _ORIGIN_SENTENCES.get(origin, _ORIGIN_SENTENCES[FailureOrigin.UNKNOWN])
    return (
        f"{head}. {frontier}. {origin_sentence} The original exception and its "
        "traceback are chained as __cause__; the partial capture is on "
        ".partial_trace."
    )


def _last_captured_op_label(trace: object) -> str | None:
    """Return the display label of the last op captured before the failure.

    Parameters
    ----------
    trace:
        Capture session, read before cleanup clears its raw workspace.

    Returns
    -------
    str | None
        The last raw layer's display label, or ``None`` when unavailable.
    """

    try:
        raw_graph_ws = getattr(trace, "__dict__", {}).get("_raw_graph_ws")
        entries = getattr(raw_graph_ws, "raw_layer_dict", None)
        if not entries:
            return None
        last = next(reversed(entries.values()))
        from .._partial import _display_label

        return _display_label(last)
    except Exception:  # noqa: BLE001 - the message is diagnostic and best effort
        return None


__all__ = [
    "attach_partial_trace",
    "failed_forward_error",
    "is_torchlens_exception",
    "original_exception",
]
