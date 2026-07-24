"""Terminal partition and completeness/funnel report tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from menagerie.crawler.constants import StatusKind
from menagerie.crawler.status import (
    StatusCompletenessError,
    _STATUS_TRANSITIONS,
    _StatusTransition,
    assert_status_completeness,
    completeness_report,
    partition_report,
)
from menagerie.crawler.tests.conftest import make_model


def _complete_record(stable_id: str, status_code: str) -> dict[str, Any]:
    """Return a synthetic record whose per-record completion flags pass.

    Parameters
    ----------
    stable_id:
        Model stable ID.
    status_code:
        Closed terminal status code.

    Returns
    -------
    dict[str, Any]
        Synthetic current record.
    """

    record = make_model(stable_id, accepted=True, status_code="runs")
    kind = status_code.split(":", 1)[0]
    record["status"]["kind"] = kind
    record["status"]["code"] = status_code
    return record


def test_partition_counts_each_terminal_bucket() -> None:
    """Small synthetic current state forms the exact ruled partition."""

    records = [
        _complete_record("m_run", "runs"),
        _complete_record("m_cuda", "deferred:needs-cuda"),
        _complete_record("m_skip", "skipped:not-a-real-NN"),
    ]
    report = partition_report(["m_run", "m_cuda", "m_skip"], records)
    assert report.valid
    assert report.buckets["runs"] == frozenset({"m_run"})
    assert report.buckets["deferred:needs-cuda"] == frozenset({"m_cuda"})


def test_partition_keeps_all_skip_reasons_as_distinct_terminal_buckets() -> None:
    """Every ruled R5 skip reason remains a distinct valid partition member.

    Returns
    -------
    None
        The assertion validates all terminal skip buckets.
    """

    skip_codes = (
        "skipped:insufficient-description",
        "skipped:no-description",
        "skipped:not-a-real-NN",
    )
    records = [
        _complete_record(f"m_skip_{index}", code) for index, code in enumerate(skip_codes, start=1)
    ]
    report = partition_report([record["stable_id"] for record in records], records)
    assert report.valid
    for index, code in enumerate(skip_codes, start=1):
        assert report.buckets[code] == frozenset({f"m_skip_{index}"})


def test_completeness_reports_pending_metadata_and_workflow() -> None:
    """Partition can pass while completion remains false and queryable."""

    complete = _complete_record("m_complete", "runs")
    pending = make_model("m_pending", accepted=False, status_code="runs")
    report = completeness_report(
        ["m_complete", "m_pending"],
        [complete, pending],
        workflow_states=["awaiting-gate"],
    )
    assert report.partition.valid
    assert not report.complete
    assert report.workflow_counts == {"awaiting-gate": 1}
    assert report.incomplete_by_issue["source_read_fields_complete"] == ("m_pending",)
    assert report.funnel_counts["status:runs"] == 2
    assert report.funnel_counts["runs:metadata-pending"] == 1


def test_completeness_is_true_only_with_exact_complete_partition() -> None:
    """All-zero issues and no workflow rows yield a complete report."""

    record = _complete_record("m_complete", "runs")
    record = deepcopy(record)
    report = completeness_report(["m_complete"], [record])
    assert report.complete


def test_status_transition_table_is_exhaustive_and_preserves_invalid_diagnostic() -> None:
    """Every status kind is ruled and malformed code/kind pairs fail first."""

    assert set(_STATUS_TRANSITIONS) == set(StatusKind)
    assert all(isinstance(value, _StatusTransition) for value in _STATUS_TRANSITIONS.values())
    record = _complete_record("m_invalid", "runs")
    record["status"]["kind"] = "failed"
    with pytest.raises(
        StatusCompletenessError,
        match="terminal status completeness invalid: m_invalid:invalid-code-kind",
    ):
        assert_status_completeness([record])


def test_status_transition_table_preserves_nonfailure_field_diagnostic() -> None:
    """A nonfailure transition still rejects failure-only evidence verbatim."""

    record = _complete_record("m_runs", "runs")
    record["status"]["stage"] = "forward"
    with pytest.raises(
        StatusCompletenessError,
        match="terminal status completeness invalid: m_runs:nonfailure-has-failure-fields",
    ):
        assert_status_completeness([record])
