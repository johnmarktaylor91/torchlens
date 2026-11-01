"""The model-output readback shared by ``Trace.output`` and ``Recording.output``.

A capture holds the forward's return value exactly as the model returned it
(tensor, tuple, dict, a Hugging Face output object, a distribution; for
generation, what ``generate()`` returned) in a private :class:`OutputSlot`.
The slot lives in memory only: it is never written to a bundle and never
pickled, so an object restored from disk has no slot. :func:`read_output`
returns the value or refuses with a typed :class:`PayloadUnavailableError`
naming why there is none; it never returns ``None`` for a missing output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .._errors import PayloadUnavailableError

OutputStatus = Literal["returned", "forward_failed", "stopped"]


@dataclass(frozen=True, slots=True)
class OutputSlot:
    """Private holder for one capture's model output, or the reason it has none.

    Attributes
    ----------
    status:
        ``"returned"`` (``value`` is the model's return value),
        ``"forward_failed"`` (the forward raised; ``detail`` names the failure)
        or ``"stopped"`` (``stop_after=`` ended the capture before the model
        returned; ``detail`` names the stop).
    value:
        The return value when ``status == "returned"``.
    detail:
        The failure or stop description otherwise.
    """

    status: OutputStatus
    value: Any = None
    detail: str | None = None


def returned(value: Any) -> OutputSlot:
    """Return the slot for a forward that returned ``value``."""

    return OutputSlot(status="returned", value=value)


def forward_failed(detail: str | None) -> OutputSlot:
    """Return the slot for a forward that raised."""

    return OutputSlot(status="forward_failed", detail=detail)


def stopped(detail: str | None) -> OutputSlot:
    """Return the slot for a capture that ``stop_after=`` ended early."""

    return OutputSlot(status="stopped", detail=detail)


def read_output(slot: OutputSlot | None, *, owner: str) -> Any:
    """Return the model output held by ``slot`` or refuse with the reason.

    Parameters
    ----------
    slot:
        The capture's private output slot; ``None`` when the object was
        restored from disk (or a pickle) or built without a live forward.
    owner:
        ``"Trace"`` or ``"Recording"``, used in the refusal text.

    Returns
    -------
    Any
        The model's return value, exactly as returned.

    Raises
    ------
    PayloadUnavailableError
        ``output_not_stored`` for an object without a live output (loaded from
        disk), ``output_forward_failed`` when the forward raised, and
        ``output_capture_stopped`` when ``stop_after=`` ended the capture
        before the model returned.
    """

    if slot is None:
        raise PayloadUnavailableError(
            f"this {owner} has no model output: the output is held in memory only and is "
            f"not stored with saved {owner}s, so a {owner} loaded from disk (or unpickled) "
            "never carries it",
            code="output_not_stored",
            remedy=f"re-run the capture and read .output on the live {owner}",
        )
    if slot.status == "forward_failed":
        raise PayloadUnavailableError(
            f"this {owner} has no model output: the forward raised before the model "
            f"returned ({slot.detail or 'no failure text'})",
            code="output_forward_failed",
            remedy="fix the failing forward and re-run; the partial result keeps the failure",
        )
    if slot.status == "stopped":
        raise PayloadUnavailableError(
            f"this {owner} has no model output: stop_after= ended the capture before the "
            f"model returned ({slot.detail or 'stop site fired'})",
            code="output_capture_stopped",
            remedy="drop stop_after= (or move it past the last op) to let the model return",
        )
    return slot.value
