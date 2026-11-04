"""The model-output slot behind ``Trace.output`` and ``Recording.output``.

The forward's return value is held in a private :class:`OutputSlot` that lives
in memory only: never written to a bundle, never pickled. :func:`read_output`
returns it or refuses typed with the reason; it never returns ``None`` for a
missing output. This module imports only the error basis so every capture
path (the eager spine, the recorder and the non-torch backends) can import it
at module top without joining the core import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from ._errors import PayloadUnavailableError

__tl_layer__ = "L1"
__tl_vocabulary__ = True

__all__ = (
    "OutputSlot",
    "OutputStatus",
    "forward_failed",
    "fork_output_refusal",
    "read_output",
    "mark_transformed_output_stale",
    "read_transformed_output",
    "saved_transformed_output",
    "returned",
    "stopped",
)

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


def read_output(
    slot: OutputSlot | None, *, owner: str, missing: tuple[str, str, str] | None = None
) -> Any:
    """Return the model output held by ``slot`` or refuse with the reason.

    Parameters
    ----------
    slot:
        The capture's private output slot; ``None`` when the object was
        restored from disk (or a pickle) or built without a live forward.
    owner:
        ``"Trace"`` or ``"Recording"``, used in the refusal text.
    missing:
        ``(what, reason, remedy)`` for the ``slot is None`` refusal when the
        missing value is not the plain model output (the stale
        ``transformed_output``); ``None`` uses the loaded-from-disk text.

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
        what, reason, remedy = missing or (
            "model output",
            f"the output is held in memory only and is not stored with saved {owner}s, so a "
            f"{owner} loaded from disk (or unpickled) never carries it",
            f"re-run the capture and read .output on the live {owner}",
        )
        raise PayloadUnavailableError(
            f"this {owner} has no {what}: {reason}", code="output_not_stored", remedy=remedy
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


class _TransformedOutputStale:
    """Private marker stored in place of a transform result a replay made stale.

    ``output_transform`` runs on the model's return value. A replay recomputes
    ops from captured values and has no model return to re-apply it to, so
    once it rewrites an output op the inherited result describes the parent's
    forward. The marker pickles by reference to the module singleton, so a
    deep-copied or unpickled trace keeps refusing; a portable save writes
    ``None`` for it.
    """

    __slots__ = ()

    def __reduce__(self) -> str:
        return "_TRANSFORMED_OUTPUT_STALE"

    def __repr__(self) -> str:
        return "<transformed output stale: a replay rewrote the model output>"


_TRANSFORMED_OUTPUT_STALE = _TransformedOutputStale()
_STALE_TRANSFORM_REFUSAL = (
    "transformed output",
    "a replay rewrote the model output after capture and has no model return to re-apply "
    "output_transform to",
    "apply the edit with intervention=tl.options.InterventionOptions(engine='rerun'), which "
    "re-runs the model and re-applies output_transform",
)


_FORK_OUTPUT_REFUSAL = (
    "model output",
    "it is a fork, built from its parent's recorded values, and a replay recomputes ops "
    "without a model return",
    "read .output on the parent, or apply the edit with "
    "intervention=tl.options.InterventionOptions(engine='rerun'), which re-runs the model",
)


def fork_output_refusal(is_fork: bool) -> tuple[str, str, str] | None:
    """Return the ``read_output`` ``missing=`` reason for a fork, ``None`` otherwise.

    A fork is built without the parent's output slot; a loaded or unpickled
    trace (no parent link) keeps the loaded-from-disk text.
    """

    return _FORK_OUTPUT_REFUSAL if is_fork else None


def read_transformed_output(value: Any, *, owner: str) -> Any:
    """Return a stored transform result or refuse when a replay made it stale.

    Parameters
    ----------
    value:
        The stored ``transformed_output`` value (``None`` when no
        ``output_transform`` was given).
    owner:
        ``"Trace"``, used in the refusal text.

    Raises
    ------
    PayloadUnavailableError
        ``output_not_stored`` when a replay rewrote an output op after capture.
    """

    if value is _TRANSFORMED_OUTPUT_STALE:
        # The same refusal site as a missing model output: one code, one contract row.
        read_output(None, owner=owner, missing=_STALE_TRANSFORM_REFUSAL)
    return value


def mark_transformed_output_stale(trace: Any) -> None:
    """Mark ``trace``'s transform result stale after a replay rewrote an output op.

    Writes the stored key directly (never the refusing property). A trace with
    no transform result (``None``) keeps ``None``: it still means no
    ``output_transform`` was given.
    """

    if trace.__dict__.get("transformed_output") is not None:
        trace.__dict__["transformed_output"] = _TRANSFORMED_OUTPUT_STALE


def saved_transformed_output(value: Any) -> Any:
    """Return the value a portable save writes for a stored transform result.

    A stale result saves as ``None``, exactly as a trace with no transform
    result saves.
    """

    return None if value is _TRANSFORMED_OUTPUT_STALE else value
