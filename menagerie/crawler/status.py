"""Terminal partition enforcement and queryable crawler funnel reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from menagerie.crawler.constants import (
    SKIPPED_STATUS_CODES,
    TERMINAL_STATUS_CODES,
    WORKFLOW_STATES,
    StatusKind,
)
from menagerie.crawler.family_templates import family_variant_currency_error
from menagerie.crawler.models import (
    CompletenessReport,
    FunnelQuery,
    JsonObject,
    PartitionReport,
)
from menagerie.crawler.wakeup import reduce_wake_episodes, render_wakeup_status


class PartitionError(ValueError):
    """Raised when current terminal records do not exactly partition intake."""


class StatusCompletenessError(ValueError):
    """Raised when a terminal status omits its closed required evidence."""


@dataclass(frozen=True)
class _StatusTransition:
    """Completeness policy applied after a code/kind pair is validated."""

    requires_failure_evidence: bool
    forbids_failure_fields: bool
    expects_terminal_issue: bool
    requires_run_completeness: bool


_STATUS_TRANSITIONS: dict[StatusKind, _StatusTransition] = {
    StatusKind.RUNS: _StatusTransition(
        requires_failure_evidence=False,
        forbids_failure_fields=True,
        expects_terminal_issue=False,
        requires_run_completeness=True,
    ),
    StatusKind.DEFERRED: _StatusTransition(
        requires_failure_evidence=False,
        forbids_failure_fields=True,
        expects_terminal_issue=True,
        requires_run_completeness=False,
    ),
    StatusKind.SKIPPED: _StatusTransition(
        requires_failure_evidence=False,
        forbids_failure_fields=True,
        expects_terminal_issue=True,
        requires_run_completeness=False,
    ),
    StatusKind.FAILED: _StatusTransition(
        requires_failure_evidence=True,
        forbids_failure_fields=False,
        expects_terminal_issue=True,
        requires_run_completeness=False,
    ),
}
if set(_STATUS_TRANSITIONS) != set(StatusKind):
    raise RuntimeError("status transition table is not exhaustive")


def wakeup_status(events: Sequence[Mapping[str, Any]]) -> JsonObject:
    """Render usage-pause episode state from canonical operational events.

    Parameters
    ----------
    events:
        Canonical operational events in ledger order.

    Returns
    -------
    dict[str, Any]
        Active/history wake status from the shared episode reducer.
    """

    return render_wakeup_status(reduce_wake_episodes(events))


def _records(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Normalize current-record mappings and iterables.

    Parameters
    ----------
    current_records:
        Stable-ID mapping or record iterable.

    Returns
    -------
    list[Mapping[str, Any]]
        Current record values.
    """

    if isinstance(current_records, Mapping):
        return list(current_records.values())
    return list(current_records)


def partition_report(
    intake_ids: Iterable[str],
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> PartitionReport:
    """Compute exact, pairwise-disjoint current terminal status buckets.

    Parameters
    ----------
    intake_ids:
        Stable IDs in trusted intake.
    current_records:
        Materialized current model revisions. Duplicate stable IDs remain visible.

    Returns
    -------
    PartitionReport
        Coverage, extras, duplicates, and terminal buckets.
    """

    intake = frozenset(intake_ids)
    buckets_mutable: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    extra: set[str] = set()
    for record in _records(current_records):
        stable_id = str(record.get("stable_id"))
        code = record.get("status", {}).get("code")
        occurrences[stable_id] += 1
        if stable_id not in intake:
            extra.add(stable_id)
        if code in SKIPPED_STATUS_CODES:
            buckets_mutable[str(code)].add(stable_id)
        elif code not in TERMINAL_STATUS_CODES:
            buckets_mutable[f"invalid:{code}"].add(stable_id)
        else:
            buckets_mutable[str(code)].add(stable_id)
    present = frozenset(occurrences)
    duplicates = frozenset(stable_id for stable_id, count in occurrences.items() if count != 1)
    buckets = {code: frozenset(stable_ids) for code, stable_ids in sorted(buckets_mutable.items())}
    return PartitionReport(
        intake_ids=intake,
        buckets=buckets,
        missing_ids=intake - present,
        extra_ids=frozenset(extra),
        duplicate_ids=duplicates,
    )


def assert_partition(
    intake_ids: Iterable[str],
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> PartitionReport:
    """Assert exact terminal partition coverage.

    Parameters
    ----------
    intake_ids:
        Trusted intake stable IDs.
    current_records:
        Materialized current model revisions.

    Returns
    -------
    PartitionReport
        Valid partition report.

    Raises
    ------
    PartitionError
        If coverage is missing/extra/duplicated or a status is not terminal.
    """

    records = _records(current_records)
    report = partition_report(intake_ids, records)
    invalid = sorted(code for code in report.buckets if code.startswith("invalid:"))
    if not report.valid or invalid:
        raise PartitionError(
            "terminal partition invalid: "
            f"missing={sorted(report.missing_ids)}, extra={sorted(report.extra_ids)}, "
            f"duplicates={sorted(report.duplicate_ids)}, invalid={invalid}"
        )
    return report


def assert_status_completeness(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> None:
    """Assert that every represented terminal status is internally complete.

    Parameters
    ----------
    current_records:
        Current terminal model revisions.

    Raises
    ------
    StatusCompletenessError
        If a status is not closed or omits required failure evidence.
    """

    failures = _status_completeness_failures(_records(current_records))
    if failures:
        raise StatusCompletenessError(
            f"terminal status completeness invalid: {', '.join(sorted(failures))}"
        )


def _status_completeness_failures(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return closed-status evidence failures without raising.

    Parameters
    ----------
    records:
        Current terminal model revisions.

    Returns
    -------
    list[str]
        Stable-ID-qualified completeness failures.
    """

    failures: list[str] = []
    for record in records:
        stable_id = str(record.get("stable_id"))
        status = record.get("status", {})
        code = status.get("code")
        kind = status.get("kind")
        if code not in TERMINAL_STATUS_CODES or kind != str(code).split(":", 1)[0]:
            failures.append(f"{stable_id}:invalid-code-kind")
            continue
        transition = _STATUS_TRANSITIONS[StatusKind(kind)]
        if transition.requires_failure_evidence:
            stage = str(code).split(":", 1)[1]
            if status.get("stage") != stage or not status.get("reason_code"):
                failures.append(f"{stable_id}:missing-stage-reason")
            if status.get("traceback") is None and status.get("no_traceback_reason") is None:
                failures.append(f"{stable_id}:missing-traceback-disposition")
            if status.get("root_cause_fingerprint") is None:
                failures.append(f"{stable_id}:missing-root-cause")
            if code == "failed:accuracy-gate" and not status.get("human_review", {}).get(
                "required"
            ):
                failures.append(f"{stable_id}:missing-human-review")
        elif transition.forbids_failure_fields and (
            status.get("stage") is not None or status.get("reason_code") is not None
        ):
            failures.append(f"{stable_id}:nonfailure-has-failure-fields")
    return failures


def filter_funnel(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    query: FunnelQuery,
) -> list[JsonObject]:
    """Filter current records by exact framework/rung/status/flag criteria.

    Parameters
    ----------
    current_records:
        Current model revisions.
    query:
        Exact optional filters.

    Returns
    -------
    list[dict[str, Any]]
        Matching records in stable-ID order.
    """

    matching: list[JsonObject] = []
    required_flags = set(query.flags)
    for record_mapping in _records(current_records):
        record = dict(record_mapping)
        if query.framework is not None and (
            record.get("implementation", {}).get("run_framework") != query.framework
        ):
            continue
        if query.rung is not None and record.get("source_resolution", {}).get("rung") != query.rung:
            continue
        if (
            query.status_code is not None
            and record.get("status", {}).get("code") != query.status_code
        ):
            continue
        if not required_flags.issubset(set(record.get("flags", []))):
            continue
        matching.append(record)
    return sorted(matching, key=lambda record: str(record["stable_id"]))


def funnel_counts(
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> Mapping[str, int]:
    """Aggregate queryable current-record funnel dimensions.

    Parameters
    ----------
    current_records:
        Current model revisions.

    Returns
    -------
    Mapping[str, int]
        Namespaced count keys for status, rung, framework, metadata, and modes.
    """

    counts: Counter[str] = Counter()
    for record in _records(current_records):
        counts["models:total"] += 1
        counts[f"status:{record['status']['code']}"] += 1
        counts[f"metadata:{record['authored_metadata_state']}"] += 1
        counts[f"rung:{record['source_resolution']['rung']}"] += 1
        counts[f"framework:{record['implementation']['run_framework']}"] += 1
        for mode in record["modes"]["meaningful_modes"]:
            counts[f"mode:{mode}"] += 1
        if (
            record["status"]["kind"] == "runs"
            and not record["completeness"]["source_read_fields_complete"]
        ):
            counts["runs:metadata-pending"] += 1
        if (
            record["status"]["kind"] != "runs"
            and record["fidelity"]["required"]
            and not record["fidelity"]["current"]
            and record["execution"]["accepted_attempt_ids"]
        ):
            counts["forward_observed_but_blocked"] += 1
    return dict(sorted(counts.items()))


def completeness_report(
    intake_ids: Iterable[str],
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    *,
    workflow_states: Sequence[str] = (),
) -> CompletenessReport:
    """Compute the final-release partition and crawl-completion gates.

    Parameters
    ----------
    intake_ids:
        Trusted intake stable IDs.
    current_records:
        Current terminal revisions.
    workflow_states:
        Scheduler/operational states still present.

    Returns
    -------
    CompletenessReport
        Exact partition, issue membership, workflow counts, and funnel totals.
    """

    records = [dict(record) for record in _records(current_records)]
    partition = partition_report(intake_ids, records)
    issues = _record_completion_issues(records)
    workflow_counts = Counter(workflow_states)
    unknown_workflows = set(workflow_counts) - WORKFLOW_STATES
    for workflow in unknown_workflows:
        issues[f"unknown_workflow:{workflow}"].append("<campaign>")
    complete = partition.valid and not issues and not workflow_counts
    return CompletenessReport(
        partition=partition,
        incomplete_by_issue={key: tuple(sorted(value)) for key, value in sorted(issues.items())},
        workflow_counts=dict(sorted(workflow_counts.items())),
        funnel_counts=funnel_counts(records),
        complete=complete,
    )


def checkpoint_consistency_report(
    intake_ids: Iterable[str],
    current_records: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
) -> CompletenessReport:
    """Validate the represented terminal prefix without requiring crawl completion.

    Missing intake IDs are valid progress at per-environment and daily checkpoint
    cadence. Represented IDs must still be unique, in intake, terminal, and
    internally complete for their terminal status.

    Parameters
    ----------
    intake_ids:
        Trusted full intake stable IDs.
    current_records:
        Current terminal revisions in the represented prefix.

    Returns
    -------
    CompletenessReport
        Prefix partition diagnostics; ``complete`` means checkpoint-consistent,
        not final-release complete.
    """

    records = [dict(record) for record in _records(current_records)]
    partition = partition_report(intake_ids, records)
    issues = _record_completion_issues(records)
    invalid = sorted(code for code in partition.buckets if code.startswith("invalid:"))
    for code in invalid:
        issues[f"partition:{code}"].extend(sorted(partition.buckets[code]))
    consistent = not (
        partition.extra_ids or partition.duplicate_ids or invalid or any(issues.values())
    )
    return CompletenessReport(
        partition=partition,
        incomplete_by_issue={key: tuple(sorted(value)) for key, value in sorted(issues.items())},
        workflow_counts={},
        funnel_counts=funnel_counts(records),
        complete=consistent,
    )


def record_is_release_eligible(
    record: Mapping[str, Any], current_records: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Derive release eligibility without trusting the persisted release flag.

    Parameters
    ----------
    record:
        Candidate current canonical model revision.
    current_records:
        Independently selected current revisions keyed by stable ID.

    Returns
    -------
    bool
        True only when current metadata, gates, execution, and family dependencies hold.
    """

    projected = current_records.get(str(record.get("stable_id")))
    if (
        not isinstance(projected, Mapping)
        or projected.get("record_revision") != record.get("record_revision")
        or record.get("status", {}).get("kind") != "runs"
        or record.get("authored_metadata_state") != "accepted"
        or record.get("status", {}).get("human_review", {}).get("required") is True
    ):
        return False
    source_fields = (
        "taxonomy",
        "external_metadata",
        "website",
        "people_and_origin",
        "dates",
        "citation",
        "licenses",
    )
    if any(record.get(field) is None for field in source_fields):
        return False
    coverage = record.get("evidence", {}).get("coverage", {})
    accuracy = record.get("accuracy_gate", {})
    fidelity = record.get("fidelity", {})
    execution = record.get("execution", {})
    meaningful = set(record.get("modes", {}).get("meaningful_modes", []))
    outcomes = set(record.get("modes", {}).get("per_mode_run", {}))
    return bool(
        coverage.get("all_agent_fields_have_support") is True
        and coverage.get("family_grounding_complete") is True
        and coverage.get("missing_support") == []
        and accuracy.get("current") is True
        and accuracy.get("verdict") == "accurate"
        and accuracy.get("gate_id")
        and accuracy.get("vet_identity")
        and (not fidelity.get("required") or fidelity.get("current") is True)
        and execution.get("current") is True
        and meaningful
        and meaningful == outcomes
        and family_variant_currency_error(record, current_records) is None
    )


def _record_completion_issues(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return status-appropriate completeness issues for represented records.

    ``runs`` records must satisfy every runnable-release gate. Honest terminal
    failures, skips, and deferrals are complete as terminal outcomes even though
    they intentionally remain ineligible for the runnable release view.

    Parameters
    ----------
    records:
        Current terminal model revisions.

    Returns
    -------
    dict[str, list[str]]
        Issue name to affected stable IDs.
    """

    issues: dict[str, list[str]] = defaultdict(list)
    base_required = ("schema_valid", "mandatory_source_present")
    run_required = (
        "source_read_fields_complete",
        "evidence_coverage_complete",
        "accuracy_gate_current",
        "required_fidelity_current",
        "execution_current",
        "family_template_valid",
        "release_eligible",
    )
    current_by_id = {str(record.get("stable_id")): record for record in records}
    for failure in _status_completeness_failures(records):
        stable_id, issue = failure.split(":", 1)
        issues[f"status:{issue}"].append(stable_id)
    for record in records:
        stable_id = str(record["stable_id"])
        status = record.get("status", {})
        status_code = str(status.get("code"))
        kind = status.get("kind")
        try:
            transition = _STATUS_TRANSITIONS[StatusKind(kind)]
        except (TypeError, ValueError):
            transition = None
        if status.get("human_review", {}).get("required") is True:
            issues["human_review_pending"].append(stable_id)
        completeness = record.get("completeness", {})
        required_true = (
            (*base_required, *run_required)
            if transition is not None and transition.requires_run_completeness
            else base_required
        )
        for field in required_true:
            if not completeness.get(field, False):
                issues[field].append(stable_id)
        expected_terminal_issues = (
            {status_code} if transition is not None and transition.expects_terminal_issue else set()
        )
        for issue in completeness.get("issues", []):
            if str(issue) not in expected_terminal_issues:
                issues[f"record:{issue}"].append(stable_id)
        if transition is not None and transition.requires_run_completeness:
            meaningful = set(record.get("modes", {}).get("meaningful_modes", []))
            outcomes = set(record.get("modes", {}).get("per_mode_run", {}))
            if meaningful != outcomes:
                issues["meaningful_mode_outcomes_incomplete"].append(stable_id)
            family_error = family_variant_currency_error(record, current_by_id)
            if family_error is not None:
                issues["stale_family_variant"].append(stable_id)
            elif not record_is_release_eligible(record, current_by_id):
                issues["derived_release_ineligible"].append(stable_id)
    return issues
