"""The non-gold capability gate and the warn-once raw-read machinery (B10).

Memo D7: on any non-gold trace (synthesized values), structural topology,
observed shapes/dtypes, parameter facts, and measured costs are legitimate
claims; DERIVED-SEMANTICS claims (decoded labels / top-k, attention
interpretation, nonfinite conclusions, value-stat tables, value-driven render
channels) hard-refuse with the real-input teach. Raw payloads stay readable
-- hiding measured tensors would make the Trace less truthful -- but the
FIRST raw value read on a non-gold trace warns once per Trace object, with a
registered suppressible code and a stacklevel resolving to the user's frame.
Internal readers (save, projections, validation) pass an ack through
:func:`internal_read` so the warning is never misattributed to library code.

The warning exists for the HANDOFF case: a loaded synthesized trace read by
someone who never saw any disclosure. A persisted record is passive; the
warning is active.
"""

from __future__ import annotations

import contextlib
import contextvars
import warnings
from collections.abc import Iterator
from typing import Any

from .._errors import CaptureContextError
from ..errors._base import TorchLensWarning
from ._provenance import trace_input_provenance

__tl_layer__ = "L3"

#: Ack contextvar: True while an internal (library-side) reader is active.
_INTERNAL_READ: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "torchlens_quickstart_internal_read", default=False
)

#: Traces that already warned (session-time, id-keyed with a weak finalizer
#: so ids are never reused while stale).
_WARNED_TRACE_IDS: set[int] = set()


class SynthesizedValueReadWarning(TorchLensWarning):
    """First raw value read on a trace whose input values were synthesized."""


def is_gold(trace: Any) -> bool:
    """Return whether a trace's values are caller-authoritative real data.

    Absent provenance (legacy traces captured before the resolver existed,
    or foreign preprocessing records) reads as gold: an absent record means
    caller-authoritative history, never suspicion.
    """

    provenance = trace_input_provenance(trace)
    if provenance is None:
        return True
    return bool(provenance.values_semantic)


def require_gold(trace: Any, claim: str) -> None:
    """Hard-refuse a derived-semantics claim on a synthesized-value trace.

    Parameters
    ----------
    trace:
        The trace the claim is being made about.
    claim:
        Human-readable name of the refused claim (e.g. ``"decoded top-k
        labels"``), spliced into the teach.

    Raises
    ------
    torchlens._errors.CaptureContextError
        Code ``nongold_semantics_unavailable``: the capture's values are
        synthesized, so the claim would be a statement about random numbers
        presented as a statement about data.
    """

    if is_gold(trace):
        return
    provenance = trace_input_provenance(trace)
    origin = provenance.origin if provenance is not None else "unknown"
    raise CaptureContextError(
        f"{claim} requires real input values, but this trace's input values "
        f"were synthesized (origin: {origin}). Structure, shapes, parameter "
        f"facts, and costs are all still legitimate reads.",
        code="nongold_semantics_unavailable",
        remedy=(
            "re-capture with a real input -- tl.trace(model, x) with your "
            "data, or a prompt string for HF models -- and make the claim on "
            "that trace"
        ),
        claim=claim,
        origin=origin,
    )


@contextlib.contextmanager
def internal_read() -> Iterator[None]:
    """Ack context for library-side payload readers (save, projections).

    Reads inside this context never trigger the first-raw-read warning, so
    the warning always names a USER read site.
    """

    token = _INTERNAL_READ.set(True)
    try:
        yield
    finally:
        _INTERNAL_READ.reset(token)


def _forget_trace(trace_id: int) -> None:
    """Drop a dead trace's id from the warned set (weakref finalizer)."""

    _WARNED_TRACE_IDS.discard(trace_id)


def warn_on_raw_read(trace: Any, *, site: str) -> None:
    """Fire the once-per-Trace synthesized-value read warning when due.

    Parameters
    ----------
    trace:
        The owning trace of the payload being read.
    site:
        The reading accessor's name (e.g. ``"Layer.out"``), disclosed in the
        message.
    """

    if _INTERNAL_READ.get():
        return
    if trace is None or id(trace) in _WARNED_TRACE_IDS:
        return
    provenance = trace_input_provenance(trace)
    if provenance is None or provenance.values_semantic:
        return
    _WARNED_TRACE_IDS.add(id(trace))
    import weakref

    with contextlib.suppress(TypeError):
        weakref.finalize(trace, _forget_trace, id(trace))
    warnings.warn(
        SynthesizedValueReadWarning(
            f"Reading raw values ({site}) from a trace whose input values were "
            f"synthesized (origin: {provenance.origin}): these numbers are real "
            f"measurements of a forward pass over RANDOM data, not of your "
            f"data. Fires once per trace. Remedy: re-capture with a real "
            f"input for value-level claims, or filter this warning by its "
            f"code to acknowledge.",
            code="nongold_raw_value_read",
            origin=provenance.origin,
            site=site,
        ),
        # warnings.warn frame ladder: 1 = this function, 2 = the accessor's
        # _warn_synthesized_read helper, 3 = the accessor property
        # (Layer.out), 4 = the user's read site.
        stacklevel=4,
    )
