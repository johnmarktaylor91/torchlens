"""Tests for menagerie skip-reason taxonomy discipline."""

from __future__ import annotations

from menagerie.validate_menagerie import MANIFEST_STATUS_VALUES

KNOWN_JUSTIFIED_SKIP_REASONS = {
    # Scheduler could not dispatch because cluster execution was unavailable or
    # explicitly disabled for a cluster-required model.
    "skipped:cluster_unavailable",
    # A dependency environment could not be made available in the selected
    # validation mode; this is an honest terminal, not a validation pass.
    "skipped:dependency_unavailable",
    # Operator-requested dry runs must record a non-validating terminal status.
    "skipped:dry_run",
    # Rows with unsupported input recipes cannot execute a faithful forward pass.
    "skipped:unsupported_input_recipe",
}


def test_skip_reason_taxonomy_is_exhaustive_and_justified() -> None:
    """A new skipped:* manifest status must update the justified taxonomy."""

    production_skip_reasons = {
        status for status in MANIFEST_STATUS_VALUES if status.startswith("skipped:")
    }

    assert production_skip_reasons == KNOWN_JUSTIFIED_SKIP_REASONS
