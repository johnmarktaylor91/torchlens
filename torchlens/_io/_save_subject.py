"""Resolve the object handed to ``save`` into the Trace it stands for.

``trace.save()`` accepts one subject kind, a Trace. Presenters
(MergedTrace, TraceSlice) refuse typed here, and a sparse ``Recording`` cooks
into its Trace through ``Recording.to_trace()`` so the Trace save contract is
the one contract. Every branch dispatches on a disjoint type and uses the
``sys.modules`` gate: a presenter or recording can only exist after its module
was imported, so ordinary saves import none of them.
"""

from __future__ import annotations

import sys
from typing import Any

from .._errors import SelectionError


def resolve_save_subject(trace: Any) -> Any:
    """Return the Trace ``save`` should write for ``trace``.

    Parameters
    ----------
    trace:
        The object handed to ``save``.

    Returns
    -------
    Any
        ``trace`` itself, or the cooked Trace of a ``Recording``.

    Raises
    ------
    MergedSurfaceUnsupportedError
        For a ``MergedTrace`` (``merged_surface_unsupported``).
    SelectionError
        For a ``TraceSlice`` (``slice_save_unsupported``).
    """

    # A MergedTrace is a presenter, never a Trace: trace.save() refuses it typed
    # (contract 2.5 "merged runnable export ... refused typed") instead of the
    # historical bare AttributeError from the poison gate. The sys.modules
    # gate keeps ordinary saves free of any merged import: a MergedTrace can
    # only exist after its module was imported.
    merged_presenter = sys.modules.get("torchlens._merged._presenter")
    if merged_presenter is not None and isinstance(trace, merged_presenter.MergedTrace):
        from .._merged._enums import MergedErrorCode
        from .._merged._errors import MergedSurfaceUnsupportedError

        raise MergedSurfaceUnsupportedError(
            "trace.save() does not support MergedTrace. Write the merged-directory "
            "artifact with merged.save(path), or save a single rank core with "
            "merged.ranks[r].save(path).",
            code=MergedErrorCode.MERGED_SURFACE_UNSUPPORTED,
        )
    # A TraceSlice is likewise a presenter, never a Trace: a slice has
    # dangling boundary edges whose other ends live outside it, so a
    # persisted slice would imply replay/validation capabilities it cannot
    # honour. Same sys.modules gate shape as the MergedTrace refusal above
    # (owners disjoint; a slice can only exist after its module imported).
    slice_presenter = sys.modules.get("torchlens._trace_slice")
    if slice_presenter is not None and isinstance(trace, slice_presenter.TraceSlice):
        raise SelectionError(
            "trace.save() does not support TraceSlice: a slice is a VIEW with "
            "declared dangling boundary edges, not a self-contained capture. "
            "Save the trace the slice was taken from (trace.save(path)) "
            "and re-derive the view after loading.",
            code="slice_save_unsupported",
        )
    recording_types = sys.modules.get("torchlens.recording.types")
    if recording_types is not None and isinstance(trace, recording_types.Recording):
        # A failed recording has no trace to cook; hand it on unchanged so the
        # N1 capture-outcome gate below refuses it under its own settled
        # outcome instead of a cook-time CaptureNotActiveError.
        if trace.is_failed:
            return trace
        return trace.to_trace()
    return trace
