"""Private initialization and verification-state blocks for Trace persistence."""

from typing import Any


def _initialize_output_metadata(trace: Any) -> None:
    """Initialize the optional output identity metadata.

    Parameters
    ----------
    trace : Trace
        Trace being initialized.
    """
    trace.output_id2label = None
    trace.output_num_classes = None


def _sanitize_pickled_verification(state: dict[str, Any]) -> None:
    """Preserve only the negative verification disclosure in pickle state.

    Parameters
    ----------
    state : dict
        Pickle state being normalized in place.
    """
    # grind-r5 P7 (b3 opus R10-1 / b6 opus R16-1): the producer's NEGATIVE
    # verification claim persists -- a capture TorchLens itself refused to
    # bless must not round-trip into "no claim". A POSITIVE claim never
    # persists (a tampered artifact could otherwise forge verified=True;
    # verdicts may only degrade across persistence). ``rescue_rerun``
    # stays session-time per the migration doc.
    if state.get("_provenance_verified") is False:
        reason = state.get("_provenance_verification_reason")
        state["_provenance_verification_reason"] = reason if isinstance(reason, str) else None
    else:
        state["_provenance_verified"] = None
        state["_provenance_verification_reason"] = None


def _restore_pickled_verification(state: dict[str, Any]) -> None:
    """Restore a well-formed negative verification disclosure from pickle state.

    Parameters
    ----------
    state : dict
        Pickle state being normalized in place.
    """
    # P7/R10: restore the persisted NEGATIVE verification disclosure. The
    # row is string-only and can only ever WORSEN a verdict: anything but
    # the exact {"verified": False} shape is ignored (stays "no claim"), so
    # a forged row cannot bless a capture and a stripped row merely reverts
    # to the historical launder this closes for honest artifacts.
    verification_row = state.pop("_capture_verification", None)
    if (
        isinstance(verification_row, dict)
        and verification_row.get("verified") is False
        and isinstance(verification_row.get("reason"), (str, type(None)))
    ):
        state["_provenance_verified"] = False
        state["_provenance_verification_reason"] = verification_row.get("reason")
