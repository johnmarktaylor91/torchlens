"""Copy a ``CaptureOptions`` with some fields forced, keeping its explicit-field record.

The debug helpers that run their own capture (``find_nan``, ``isolated_capture``)
force one field on the caller's ``capture=`` object. A plain rebuild would mark
every field explicit, and explicit backend-only fields refuse on the torch
backend, so the copy keeps which fields the caller set and adds only the forced
ones. Both helpers import ``_with_capture_fields``; it builds the copy through
the instance's own class, so it needs no runtime import of ``options``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..options import CaptureOptions


def _with_capture_fields(capture: CaptureOptions, **fields: Any) -> CaptureOptions:
    """Return ``capture`` with ``fields`` set and marked explicit.

    Parameters
    ----------
    capture
        The caller's capture options.
    **fields
        Field values to force on the copy.

    Returns
    -------
    CaptureOptions
        A copy whose explicit-field record is the caller's plus ``fields``,
        with the private layers-to-save selection carried over unchanged.
    """

    values = capture.as_dict()
    values.update(fields)
    return type(capture)._from_values(
        values,
        capture._specified_fields | frozenset(fields),
        _layers_to_save=capture._layers_to_save,
        _layers_to_save_explicit=capture._layers_to_save_explicit,
    )
