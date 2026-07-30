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


# ---------------------------------------------------------------------------
# The access deferral and its barrier evidence are ONE fact, bound both ways.
# ---------------------------------------------------------------------------


def _barrier_probe(claimed: str = "access-barrier") -> dict[str, Any]:
    """Return one machine candidate-probe row with a controllable authored class."""

    return {
        "identifier_kind": "doi",
        "identifier": "10.1109/5.726791",
        "locator": "https://doi.org/10.1109/5.726791",
        "attempted_at": "2026-07-30T00:00:00Z",
        "probe_outcome": "unreachable",
        "http_status": 403,
        "author_claimed_class": claimed,
    }


def _access_record(status_code: str, probes: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a synthetic terminal record carrying exactly these candidate probes."""

    record = _complete_record("m_access", status_code)
    record["discovery_probes"] = probes
    return record


def test_access_deferral_without_barrier_evidence_is_refused() -> None:
    """Omission fails: the terminal names a capability but nothing it applies to.

    An access deferral whose record names no unreachable locator is useless to the
    later access pass the code exists to enable -- there is nothing to go and fetch.
    """

    with pytest.raises(StatusCompletenessError, match="missing-access-barrier-evidence"):
        assert_status_completeness([_access_record("deferred:needs-source-access", [])])


def test_barrier_evidence_under_an_absence_claim_is_refused() -> None:
    """Over-claim fails, and this is the dangerous half.

    A record that carries a locator its own author says it could not read, while
    terminalising as "no descriptive text exists after bounded search", is exactly the
    false permanent statement this whole change exists to remove. It may not pass just
    because the author reached for the cheaper code.
    """

    with pytest.raises(
        StatusCompletenessError, match="access-barrier-evidence-under-skipped:no-description"
    ):
        assert_status_completeness(
            [_access_record("skipped:no-description", [_barrier_probe()])]
        )


def test_access_deferral_with_barrier_evidence_passes() -> None:
    """The honest shape: the terminal and the locator it rests on, together."""

    assert_status_completeness(
        [_access_record("deferred:needs-source-access", [_barrier_probe()])]
    )


def test_non_barrier_probes_do_not_trip_an_absence_claim() -> None:
    """Scoping: probes only bind the status when the AUTHOR claims a barrier.

    The machine's own HTTP status deliberately does NOT decide this. Bot-walls return
    403 to automated probes constantly, and letting an incidental 403 hard-refuse an
    otherwise honest skip would convert a routine observation into a lost model. The
    disagreement stays visible in the row for a checker to act on.
    """

    assert_status_completeness(
        [_access_record("skipped:no-description", [_barrier_probe("not-this-model")])]
    )


def test_a_failure_record_is_outside_the_barrier_contract() -> None:
    """A `failed:*` record asserts nothing about the world, so it is not bound here.

    Binding it would also make the failure fallback itself unrecordable, turning one
    bad model into a second-order failure.
    """

    record = _access_record("failed:runner", [_barrier_probe()])
    record["status"]["stage"] = "runner"
    record["status"]["reason_code"] = "protocol-violation"
    record["status"]["traceback"] = None
    record["status"]["no_traceback_reason"] = "synthetic record"
    record["status"]["root_cause_fingerprint"] = "sha256:" + "0" * 64
    assert_status_completeness([record])
