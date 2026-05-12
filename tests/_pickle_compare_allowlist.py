"""Committed allow-list for TorchLens parity pickle comparisons."""

from __future__ import annotations

ALLOWED_PICKLE_DIFF_FIELDS: dict[str, frozenset[str]] = {
    "Trace": frozenset(
        {
            "backward_peak_memory",
            "cleanup_duration",
            "end_time",
            "forward_duration",
            "forward_peak_memory",
            "func_calls_duration",
            "setup_duration",
            "start_time",
        }
    ),
    "OpLog": frozenset(
        {
            "bytes_peak_at_call",
            "capture_index",
            "func_call_id",
            "func_duration",
        }
    ),
}


def allowed_pickle_diff_fields() -> dict[str, frozenset[str]]:
    """Return the committed pickle-diff allow-list.

    Returns
    -------
    dict[str, frozenset[str]]
        Mapping from class name to fields ignored by canonical pickle diff.
    """

    return ALLOWED_PICKLE_DIFF_FIELDS
