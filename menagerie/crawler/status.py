"""Terminal partition enforcement and queryable crawler funnel reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from menagerie.crawler.constants import SKIPPED_STATUS_CODES, TERMINAL_STATUS_CODES, WORKFLOW_STATES
from menagerie.crawler.models import (
    CompletenessReport,
    FunnelQuery,
    JsonObject,
    PartitionReport,
)


class PartitionError(ValueError):
    """Raised when current terminal records do not exactly partition intake."""


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
    """Compute partition and all represented crawl-completion gates.

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
    issues: dict[str, list[str]] = defaultdict(list)
    required_true = (
        "schema_valid",
        "mandatory_source_present",
        "source_read_fields_complete",
        "evidence_coverage_complete",
        "accuracy_gate_current",
        "required_fidelity_current",
        "execution_current",
        "family_template_valid",
        "release_eligible",
    )
    for record in records:
        stable_id = str(record["stable_id"])
        completeness = record.get("completeness", {})
        for field in required_true:
            if not completeness.get(field, False):
                issues[field].append(stable_id)
        for issue in completeness.get("issues", []):
            issues[f"record:{issue}"].append(stable_id)
        meaningful = set(record.get("modes", {}).get("meaningful_modes", []))
        outcomes = set(record.get("modes", {}).get("per_mode_run", {}))
        if meaningful != outcomes:
            issues["meaningful_mode_outcomes_incomplete"].append(stable_id)
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
