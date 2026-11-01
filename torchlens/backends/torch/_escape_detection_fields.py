"""Field table copying escape/completeness diagnostics from a trace onto a fastlog recording."""

from __future__ import annotations

_RECORDING_DIAGNOSTIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("escape_detector_mode", "_health_escape_detector_mode"),
    ("escape_detector_verified", "_health_escape_detector_verified"),
    ("completeness_witness_mode", "_health_completeness_mode"),
    ("completeness_witness_verified", "_health_completeness_verified"),
    ("capture_owner_thread_id", "_provenance_owner_thread_id"),
    ("capture_owner_thread_qualified", "_provenance_owner_thread_qualified"),
    ("capture_thread_count_start", "_provenance_thread_count_start"),
    ("capture_thread_count_end", "_provenance_thread_count_end"),
    ("capture_thread_activity_detected", "_provenance_thread_activity_detected"),
    ("capture_verified", "_provenance_verified"),
    ("capture_verification_reason", "_provenance_verification_reason"),
    ("escape_detector_backward_coverage", "_autograd_escape_detector_backward_coverage"),
    ("escape_diagnostics", "escape_diagnostics"),
    ("escape_detector_event_count", "_health_num_escape_detector_events"),
    ("escape_detector_callback_ns", "_health_escape_detector_callback_ns"),
    ("completeness_diagnostics", "completeness_diagnostics"),
    ("completeness_decompositions", "_health_completeness_decompositions"),
    ("completeness_witness_event_count", "_health_num_witness_events"),
    ("completeness_witness_accounted_count", "_health_num_accounted_witness_events"),
    (
        "completeness_witness_expected_opaque_count",
        "_health_num_expected_opaque_witness_events",
    ),
    ("completeness_witness_unaccounted_count", "_health_num_unaccounted_witness_events"),
    ("completeness_witness_callback_ns", "_health_completeness_callback_ns"),
    ("capture_guard_passes", "capture_guard_passes"),
)
"""``(recording field, trace store)`` pairs mirrored onto a recording at guard exit."""
