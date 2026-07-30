"""Read-only reduction and exact partition proof for the four crawler campaigns."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from menagerie.crawler.authority import build_authority_context
from menagerie.crawler.constants import ACCESS_BLOCKED_STATUS_CODE
from menagerie.crawler.checkpoint import canonical_operational_ledger_path
from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes, hash_bytes
from menagerie.crawler.intake import IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.partitioner import (
    CampaignBinding,
    assert_campaign_partition,
    load_campaign_bindings,
)
from menagerie.crawler.promotion import (
    PromotionError,
    load_promotion_rows,
    promotion_extension_path,
    promotion_intake_identity,
    promotion_records_root,
)
from menagerie.crawler.recordio import scan_jsonl
from menagerie.crawler.reducer import default_ledger_paths, project_dependency_current
from menagerie.crawler.status import (
    checkpoint_consistency_report,
    completeness_report,
    record_is_release_eligible,
)
from menagerie.crawler.tools.rebuild_views import _access_blocked_row
from menagerie.crawler.tools.throughput_report import build_throughput_report


class CampaignMergeError(RuntimeError):
    """Raised when a campaign merge cannot prove its integrity obligations."""


@dataclass(frozen=True)
class CampaignSource:
    """Physical roots for one completed single-writer campaign.

    Parameters
    ----------
    campaign_id:
        Frozen campaign identifier from ``campaigns.json``.
    records_root:
        Campaign-local canonical records root.
    runtime_root:
        Campaign-local private runtime root containing W1.4 instrumentation.
    """

    campaign_id: str
    records_root: Path
    runtime_root: Path


@dataclass(frozen=True)
class MergeResult:
    """Validated merged projection and its persisted report.

    Parameters
    ----------
    current_records:
        Globally ordered dependency-current terminal records.
    report:
        Persisted merge, partition, quality, and throughput report.
    view_digests:
        Exact digest of every rebuilt merged view.
    """

    current_records: tuple[JsonObject, ...]
    report: JsonObject
    view_digests: Mapping[str, str]


def _read_json_object(path: Path) -> JsonObject:
    """Read one JSON object with a merge-specific diagnostic.

    Parameters
    ----------
    path:
        JSON document path.

    Returns
    -------
    dict[str, Any]
        Parsed object.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignMergeError(f"cannot read merge input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CampaignMergeError(f"merge input must be a JSON object: {path}")
    return value


def _read_jsonl_objects(path: Path) -> list[JsonObject]:
    """Read roster or partition JSONL without treating it as a canonical ledger.

    Parameters
    ----------
    path:
        Roster-shaped JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed object rows.
    """

    rows: list[JsonObject] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CampaignMergeError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignMergeError(f"cannot read merge input {path}: {exc}") from exc
    return rows


def _manifest_path(root: Path, relative: str, label: str) -> Path:
    """Resolve one manifest path while refusing traversal outside its records root.

    Parameters
    ----------
    root:
        Canonical records root containing ``campaigns.json``.
    relative:
        Manifest-relative path.
    label:
        Diagnostic field name.

    Returns
    -------
    pathlib.Path
        Resolved in-root path.
    """

    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise CampaignMergeError(f"campaign manifest {label} escapes records root: {relative}")
    return path


def _validate_frozen_partition(
    manifest_path: Path,
    bindings: Sequence[CampaignBinding],
) -> Mapping[str, IntakeSnapshot]:
    """Re-prove manifest bytes, roster coverage, and frozen intake identities.

    Parameters
    ----------
    manifest_path:
        Primary clone's frozen ``campaigns.json``.
    bindings:
        Strictly parsed four campaign bindings.

    Returns
    -------
    Mapping[str, IntakeSnapshot]
        Campaign ID to verified primary intake snapshot.
    """

    manifest = _read_json_object(manifest_path)
    records_root = manifest_path.resolve().parent
    roster_binding = manifest.get("roster")
    if not isinstance(roster_binding, Mapping):
        raise CampaignMergeError("campaign manifest has no roster binding")
    roster_relative = roster_binding.get("path")
    if not isinstance(roster_relative, str) or not roster_relative:
        raise CampaignMergeError("campaign manifest roster path is invalid")
    roster_path = (records_root / roster_relative).resolve()
    roster_bytes = roster_path.read_bytes()
    roster = _read_jsonl_objects(roster_path)
    if hash_bytes(roster_bytes) != roster_binding.get("sha256") or len(
        roster
    ) != roster_binding.get("row_count"):
        raise CampaignMergeError("campaign manifest roster bytes or row count changed")

    partitions: dict[str, list[JsonObject]] = {}
    snapshots: dict[str, IntakeSnapshot] = {}
    for binding in bindings:
        campaign_id = binding.spec.campaign_id
        partition_path = _manifest_path(
            records_root, binding.partition_path, f"{campaign_id} partition path"
        )
        partition_bytes = partition_path.read_bytes()
        rows = _read_jsonl_objects(partition_path)
        if hash_bytes(partition_bytes) != binding.partition_sha256:
            raise CampaignMergeError(f"{campaign_id} partition digest changed")
        if len(rows) != binding.row_count:
            raise CampaignMergeError(f"{campaign_id} partition row count changed")
        partitions[campaign_id] = rows

        intake_root = _manifest_path(
            records_root, binding.intake_path, f"{campaign_id} intake path"
        )
        snapshot = load_intake_snapshot(intake_root)
        if (
            snapshot.snapshot_id != binding.intake_snapshot_id
            or snapshot.snapshot_sha256 != binding.intake_snapshot_sha256
            or len(snapshot.items) != binding.row_count
        ):
            raise CampaignMergeError(f"{campaign_id} frozen intake binding changed")
        snapshots[campaign_id] = snapshot

    assert_campaign_partition(roster, partitions)
    return snapshots


def _canonical_input_paths(records_root: Path) -> tuple[Path, ...]:
    """Return every campaign canonical ledger path covered by immutability checks.

    Parameters
    ----------
    records_root:
        Campaign-local records root.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Existing JSONL ledgers plus the fixed core ledger paths.
    """

    ledgers = default_ledger_paths(records_root)
    fixed = {
        ledgers.models,
        ledgers.attempts,
        ledgers.gates,
        ledgers.artifacts,
        canonical_operational_ledger_path(ledgers.models),
        records_root / "operational" / "requeue-grants.jsonl",
        promotion_extension_path(records_root),
    }
    promoted = default_ledger_paths(promotion_records_root(records_root))
    fixed.update(
        {
            promoted.models,
            promoted.attempts,
            promoted.gates,
            promoted.artifacts,
            canonical_operational_ledger_path(promoted.models),
        }
    )
    discovered: set[Path] = set()
    for directory in ("models", "attempts", "gates", "artifacts", "operational"):
        discovered.update((records_root / directory).glob("*.jsonl"))
    return tuple(sorted(fixed | discovered))


def _ledger_fingerprint(sources: Sequence[CampaignSource]) -> Mapping[str, str]:
    """Hash all canonical campaign ledgers without opening a writer or lock file.

    Parameters
    ----------
    sources:
        Exact campaign inputs.

    Returns
    -------
    Mapping[str, str]
        Campaign-qualified path to content digest or the ``missing`` marker.
    """

    fingerprints: dict[str, str] = {}
    for source in sources:
        for path in _canonical_input_paths(source.records_root):
            key = f"{source.campaign_id}:{path.resolve()}"
            fingerprints[key] = hash_bytes(path.read_bytes()) if path.is_file() else "missing"
    return dict(sorted(fingerprints.items()))


def _assert_author_binding(
    campaign_id: str,
    records: Sequence[Mapping[str, Any]],
    binding: CampaignBinding,
) -> tuple[str, str]:
    """Validate frozen author/checker labels across all model history.

    Parameters
    ----------
    campaign_id:
        Campaign being validated.
    records:
        Every append-only model revision.
    binding:
        Frozen manifest binding.

    Returns
    -------
    tuple[str, str]
        Unique author and checker versions used to rebuild authority.
    """

    author_versions: set[str] = set()
    checker_versions: set[str] = set()
    failures: list[str] = []
    for record in records:
        stable_id = str(record.get("stable_id"))
        provenance = record.get("provenance")
        if not isinstance(provenance, Mapping):
            failures.append(f"{stable_id}:missing-provenance")
            continue
        actual_author = provenance.get("author_model")
        actual_checker = provenance.get("checker_model")
        if actual_author != binding.spec.author_model:
            failures.append(
                f"{stable_id}:author_model={actual_author!r},expected={binding.spec.author_model!r}"
            )
        if actual_checker != binding.spec.checker_model:
            failures.append(
                f"{stable_id}:checker_model={actual_checker!r},"
                f"expected={binding.spec.checker_model!r}"
            )
        author_version = provenance.get("author_version")
        checker_version = provenance.get("checker_version")
        if isinstance(author_version, str) and author_version:
            author_versions.add(author_version)
        else:
            failures.append(f"{stable_id}:missing-author-version")
        if isinstance(checker_version, str) and checker_version:
            checker_versions.add(checker_version)
        else:
            failures.append(f"{stable_id}:missing-checker-version")
    if failures:
        raise CampaignMergeError(
            f"{campaign_id} author_model_identity binding invalid: {sorted(failures)}"
        )
    if len(author_versions) != 1 or len(checker_versions) != 1:
        raise CampaignMergeError(
            f"{campaign_id} model identity versions are not frozen: "
            f"author={sorted(author_versions)}, checker={sorted(checker_versions)}"
        )
    return next(iter(author_versions)), next(iter(checker_versions))


def _environment_generations(
    attempts: Sequence[Mapping[str, Any]],
    operational: Sequence[Mapping[str, Any]],
) -> Mapping[str, str]:
    """Restore the latest durable environment generation exactly as the driver does.

    Parameters
    ----------
    attempts, operational:
        Canonical attempt and operational rows in ledger order.

    Returns
    -------
    Mapping[str, str]
        Environment intent to latest durable generation identity.
    """

    generations: dict[str, str] = {}
    for attempt in attempts:
        environment = attempt.get("environment")
        identities = attempt.get("identities")
        family = environment.get("family") if isinstance(environment, Mapping) else None
        generation = identities.get("environment") if isinstance(identities, Mapping) else None
        if isinstance(family, str) and family and isinstance(generation, str) and generation:
            generations[family] = generation
    for event in operational:
        if event.get("event_kind") != "campaign-health":
            continue
        details = event.get("details")
        if not isinstance(details, Mapping):
            continue
        intent = details.get("intent")
        disposition = details.get("disposition")
        generation = details.get("env_generation")
        if disposition == "environment-cleanup-quarantined":
            environment = details.get("environment")
            generation = (
                environment.get("env_generation") if isinstance(environment, Mapping) else None
            )
        elif disposition != "environment-integrity-quarantined":
            continue
        if isinstance(intent, str) and intent and isinstance(generation, str) and generation:
            generations[intent] = generation
    return generations


def _campaign_current(
    source: CampaignSource,
    binding: CampaignBinding,
    snapshot: IntakeSnapshot,
    raw_models: Sequence[Mapping[str, Any]],
) -> Mapping[str, JsonObject]:
    """Replay one campaign through the read-only dependency-current reducer.

    Parameters
    ----------
    source, binding, snapshot:
        Physical campaign input and its frozen authority.
    raw_models:
        Already scanned model history used for identity checking.

    Returns
    -------
    Mapping[str, dict[str, Any]]
        Valid dependency-current terminal records.
    """

    author_version, checker_version = _assert_author_binding(
        source.campaign_id, raw_models, binding
    )
    ledgers = default_ledger_paths(source.records_root)
    attempts = scan_jsonl(ledgers.attempts)
    operational = scan_jsonl(canonical_operational_ledger_path(ledgers.models))
    context = build_authority_context(
        active_intake_snapshot_id=snapshot.snapshot_id,
        active_intake_snapshot_sha256=snapshot.snapshot_sha256,
        intake_rows=(item.to_dict() for item in snapshot.items),
        author_model=binding.spec.author_model,
        author_version=author_version,
        checker_model=binding.spec.checker_model,
        checker_version=checker_version,
        environment_generations=_environment_generations(attempts, operational),
    )
    projection = project_dependency_current(ledgers, context=context)
    if projection.stale_reasons:
        raise CampaignMergeError(
            f"{source.campaign_id} reducer replay rejected current rows: "
            f"{dict(sorted(projection.stale_reasons.items()))}"
        )
    consistency = checkpoint_consistency_report(
        (item.stable_id for item in snapshot.items), projection.current_records
    )
    if not consistency.complete:
        raise CampaignMergeError(
            f"{source.campaign_id} checkpoint validation failed: "
            f"extra={sorted(consistency.partition.extra_ids)}, "
            f"duplicates={sorted(consistency.partition.duplicate_ids)}, "
            f"issues={dict(consistency.incomplete_by_issue)}"
        )
    return projection.current_records


def _global_partition_precheck(
    expected_ids: set[str],
    raw_ids_by_campaign: Mapping[str, set[str]],
) -> None:
    """Fail before reduction when actual processed IDs overlap or omit roster IDs.

    Parameters
    ----------
    expected_ids:
        Stable IDs from the four frozen intake snapshots.
    raw_ids_by_campaign:
        Campaign to stable IDs represented in its model ledger history.
    """

    owners: dict[str, list[str]] = defaultdict(list)
    for campaign_id, stable_ids in raw_ids_by_campaign.items():
        for stable_id in stable_ids:
            owners[stable_id].append(campaign_id)
    duplicates = sorted(stable_id for stable_id, campaigns in owners.items() if len(campaigns) > 1)
    observed = set(owners)
    missing = sorted(expected_ids - observed)
    extra = sorted(observed - expected_ids)
    if duplicates or missing or extra:
        raise CampaignMergeError(
            "global processed partition invalid: "
            f"duplicate_stable_ids={duplicates}, missing_stable_ids={missing}, "
            f"extra_stable_ids={extra}"
        )


def _promotion_current(
    source: CampaignSource,
    binding: CampaignBinding,
    promotions: Sequence[Mapping[str, Any]],
) -> Mapping[str, JsonObject]:
    """Replay consumed C3 promotions under an isolated Opus authority root.

    Parameters
    ----------
    source:
        Physical C3 campaign source.
    binding:
        Frozen C3 author/checker identity binding.
    promotions:
        Typed promotion intake rows collected from the Sonnet campaigns.

    Returns
    -------
    Mapping[str, dict[str, Any]]
        Dependency-current consumed promotion terminals. Missing rows are honest,
        unconsumed deferrals and therefore are not an error.

    Raises
    ------
    CampaignMergeError
        If promoted ledgers contain ungranted work, stale authority, or a nonterminal
        result.
    """

    promoted_root = promotion_records_root(source.records_root)
    ledgers = default_ledger_paths(promoted_root)
    raw_models = scan_jsonl(ledgers.models, validate=False)
    if not raw_models:
        return {}
    promotion_ids = {str(row["stable_id"]) for row in promotions}
    raw_ids = {str(record.get("stable_id")) for record in raw_models}
    if not raw_ids.issubset(promotion_ids):
        raise CampaignMergeError(
            "C3 promotion records contain work outside the typed intake extension: "
            f"{sorted(raw_ids - promotion_ids)}"
        )
    author_version, checker_version = _assert_author_binding(
        binding.spec.campaign_id, raw_models, binding
    )
    snapshot_id, snapshot_sha256, intake_rows = promotion_intake_identity(promotions)
    attempts = scan_jsonl(ledgers.attempts)
    operational = scan_jsonl(canonical_operational_ledger_path(ledgers.models))
    context = build_authority_context(
        active_intake_snapshot_id=snapshot_id,
        active_intake_snapshot_sha256=snapshot_sha256,
        intake_rows=intake_rows,
        author_model=binding.spec.author_model,
        author_version=author_version,
        checker_model=binding.spec.checker_model,
        checker_version=checker_version,
        environment_generations=_environment_generations(attempts, operational),
    )
    projection = project_dependency_current(ledgers, context=context)
    if projection.stale_reasons:
        raise CampaignMergeError(
            "C3 promotion reducer replay rejected current rows: "
            f"{dict(sorted(projection.stale_reasons.items()))}"
        )
    consumed = completeness_report(raw_ids, projection.current_records)
    if not consumed.complete:
        raise CampaignMergeError(
            "C3 promotion checkpoint validation failed: "
            f"missing={sorted(consumed.partition.missing_ids)}, "
            f"extra={sorted(consumed.partition.extra_ids)}, "
            f"issues={dict(consumed.incomplete_by_issue)}"
        )
    deferred = sorted(
        stable_id
        for stable_id, record in projection.current_records.items()
        if str(record["status"]["code"]) == "deferred:needs-opus-tier"
    )
    if deferred:
        raise CampaignMergeError(
            f"C3 promotion terminals cannot defer to their own author tier: {deferred}"
        )
    return projection.current_records


def resolve_promotion_supersession(
    current_by_campaign: Mapping[str, Mapping[str, JsonObject]],
    promotions: Sequence[Mapping[str, Any]],
    c3_promoted: Mapping[str, JsonObject],
) -> tuple[Mapping[str, JsonObject], tuple[str, ...], tuple[str, ...]]:
    """Resolve C3 terminals over their exact originating deferrals.

    Parameters
    ----------
    current_by_campaign:
        Dependency-current base campaign records.
    promotions:
        Typed C3 intake-extension rows.
    c3_promoted:
        Dependency-current terminals authored under the isolated C3 promotion intake.

    Returns
    -------
    tuple[Mapping[str, dict[str, Any]], tuple[str, ...], tuple[str, ...]]
        Final stable-ID map, superseded promotion IDs, and unconsumed promotion IDs.

    Raises
    ------
    CampaignMergeError
        If a promotion does not name an exact originating Opus-tier deferral or a C3
        terminal lacks a promotion row.
    """

    merged: dict[str, JsonObject] = {
        stable_id: record
        for campaign_id in sorted(current_by_campaign)
        for stable_id, record in current_by_campaign[campaign_id].items()
    }
    promotion_by_id = {str(row["stable_id"]): row for row in promotions}
    if set(c3_promoted) - set(promotion_by_id):
        raise CampaignMergeError(
            "C3 terminal lacks a typed promotion row: "
            f"{sorted(set(c3_promoted) - set(promotion_by_id))}"
        )
    superseded: list[str] = []
    unconsumed: list[str] = []
    for stable_id, promotion in sorted(promotion_by_id.items()):
        source_campaign = str(promotion["source_campaign_id"])
        source = current_by_campaign.get(source_campaign, {}).get(stable_id)
        if source is None or source.get("status", {}).get("code") != "deferred:needs-opus-tier":
            raise CampaignMergeError(
                f"promotion {promotion['promotion_id']} has no matching source-campaign "
                "deferred:needs-opus-tier terminal"
            )
        c3_terminal = c3_promoted.get(stable_id)
        if c3_terminal is None:
            unconsumed.append(str(promotion["promotion_id"]))
            continue
        merged[stable_id] = c3_terminal
        superseded.append(str(promotion["promotion_id"]))
    return (
        {stable_id: merged[stable_id] for stable_id in sorted(merged)},
        tuple(superseded),
        tuple(unconsumed),
    )


def _status_report(records: Sequence[Mapping[str, Any]]) -> JsonObject:
    """Build exact terminal status counts and explicitly defined rates.

    Parameters
    ----------
    records:
        Current terminal records.

    Returns
    -------
    dict[str, Any]
        Status counts plus quality, reject, and deferral rates.
    """

    status_counts = Counter(str(record["status"]["code"]) for record in records)
    total = len(records)
    runs = status_counts["runs"]
    rejected = sum(
        count
        for status, count in status_counts.items()
        if status.startswith("failed:") or status.startswith("skipped:")
    )
    deferred = sum(
        count for status, count in status_counts.items() if status.startswith("deferred:")
    )
    denominator = float(total) if total else 1.0
    return {
        "terminal_count": total,
        "status_counts": dict(sorted(status_counts.items())),
        "quality_count": runs,
        "quality_rate": runs / denominator,
        "reject_count": rejected,
        "reject_rate": rejected / denominator,
        "deferred_count": deferred,
        "deferred_rate": deferred / denominator,
    }


def _view_payloads(
    current: Sequence[JsonObject],
    report: JsonObject,
) -> Mapping[str, bytes]:
    """Build deterministic merged view bytes after all validation passes.

    Parameters
    ----------
    current:
        Globally ordered current records.
    report:
        Complete merge report.

    Returns
    -------
    Mapping[str, bytes]
        Relative merged-view path to complete bytes.
    """

    current_by_id = {str(record["stable_id"]): record for record in current}
    release = [record for record in current if record_is_release_eligible(record, current_by_id)]
    deferred = [
        record
        for record in current
        if str(record["status"]["code"]).startswith("deferred:")
    ]
    deferred_linux = [
        record
        for record in deferred
        if str(record["status"]["code"]) in {"deferred:needs-cuda", "deferred:needs-x86"}
    ]
    deferred_promotions = [
        record
        for record in deferred
        if str(record["status"]["code"]) == "deferred:needs-opus-tier"
    ]
    # The cross-campaign access worklist. It has to exist at MERGE level too: the
    # institutional-access batch that recovers these models is assembled once, over
    # every campaign, by someone reading the merged output rather than four
    # per-campaign views.
    blocked_on_access = [
        _access_blocked_row(record)
        for record in deferred
        if str(record["status"]["code"]) == ACCESS_BLOCKED_STATUS_CODE
    ]

    def jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
        """Serialize one ordered derived view as canonical JSONL."""

        return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)

    status_summary = {
        "current_count": len(current),
        "release_count": len(release),
        "deferred_count": len(deferred),
        "blocked_on_access_count": len(blocked_on_access),
        "terminal": report["total"],
    }
    return {
        "current-models/current.jsonl": jsonl(current),
        "release-models.jsonl": jsonl(release),
        "deferred-linux.jsonl": jsonl(deferred_linux),
        "deferred-promotions.jsonl": jsonl(deferred_promotions),
        "blocked-on-access.jsonl": jsonl(blocked_on_access),
        "status-summary.json": canonical_json_bytes(status_summary) + b"\n",
        "merge-report.json": canonical_json_bytes(report) + b"\n",
    }


def merge_campaigns(
    manifest_path: Path,
    sources: Sequence[CampaignSource],
    output_root: Path,
) -> MergeResult:
    """Merge four completed campaigns without mutating any campaign ledger.

    This function performs no ``JsonlLedger`` construction and no reducer append.
    It scans immutable bytes, replays each campaign with the read-only dependency
    projector, proves exact global ownership, rebuilds derived views, and verifies
    source-ledger digests before and after the operation.

    Parameters
    ----------
    manifest_path:
        Primary clone's frozen campaign manifest.
    sources:
        Exactly one physical source for each frozen campaign.
    output_root:
        Disposable merged-view destination outside all campaign records roots.

    Returns
    -------
    MergeResult
        Validated merged projection, report, and view digests.
    """

    bindings = load_campaign_bindings(manifest_path)
    binding_by_id = {binding.spec.campaign_id: binding for binding in bindings}
    source_by_id = {source.campaign_id: source for source in sources}
    if len(source_by_id) != len(sources) or set(source_by_id) != set(binding_by_id):
        raise CampaignMergeError(
            "merge sources must name each frozen campaign exactly once: "
            f"expected={sorted(binding_by_id)}, actual={sorted(source_by_id)}"
        )
    resolved_output = output_root.resolve()
    for source in sources:
        resolved_records = source.records_root.resolve()
        if resolved_output == resolved_records or resolved_records in resolved_output.parents:
            raise CampaignMergeError(
                f"merged views cannot be written inside campaign records: {resolved_records}"
            )

    primary_snapshots = _validate_frozen_partition(manifest_path, bindings)
    expected_ids = {
        item.stable_id for snapshot in primary_snapshots.values() for item in snapshot.items
    }
    if len(expected_ids) != sum(len(snapshot.items) for snapshot in primary_snapshots.values()):
        raise CampaignMergeError("frozen campaign intake snapshots contain duplicate stable_ids")

    before = _ledger_fingerprint(sources)
    try:
        promotions = load_promotion_rows(
            tuple(
                promotion_extension_path(source.records_root)
                for source in sorted(sources, key=lambda item: item.campaign_id)
            )
        )
    except PromotionError as exc:
        raise CampaignMergeError(str(exc)) from exc
    raw_models_by_id: dict[str, list[JsonObject]] = {}
    actual_snapshots: dict[str, IntakeSnapshot] = {}
    for campaign_id, source in sorted(source_by_id.items()):
        binding = binding_by_id[campaign_id]
        actual_snapshot = load_intake_snapshot(source.records_root / binding.intake_path)
        if (
            actual_snapshot.snapshot_id != binding.intake_snapshot_id
            or actual_snapshot.snapshot_sha256 != binding.intake_snapshot_sha256
        ):
            raise CampaignMergeError(f"{campaign_id} campaign intake differs from its binding")
        actual_snapshots[campaign_id] = actual_snapshot
        raw_models_by_id[campaign_id] = scan_jsonl(
            default_ledger_paths(source.records_root).models, validate=False
        )

    raw_ids_by_campaign = {
        campaign_id: {str(record.get("stable_id")) for record in records}
        for campaign_id, records in raw_models_by_id.items()
    }
    _global_partition_precheck(expected_ids, raw_ids_by_campaign)
    for campaign_id, snapshot in actual_snapshots.items():
        assigned = {item.stable_id for item in snapshot.items}
        processed = raw_ids_by_campaign[campaign_id]
        if assigned != processed:
            raise CampaignMergeError(
                f"{campaign_id} processed ownership invalid: "
                f"missing_stable_ids={sorted(assigned - processed)}, "
                f"extra_stable_ids={sorted(processed - assigned)}"
            )

    current_by_campaign: dict[str, Mapping[str, JsonObject]] = {}
    for campaign_id, source in sorted(source_by_id.items()):
        current_by_campaign[campaign_id] = _campaign_current(
            source,
            binding_by_id[campaign_id],
            actual_snapshots[campaign_id],
            raw_models_by_id[campaign_id],
        )

    c3_promoted = _promotion_current(
        source_by_id["c3-classics"],
        binding_by_id["c3-classics"],
        promotions,
    )
    merged_by_id, superseded_promotions, unconsumed_promotions = (
        resolve_promotion_supersession(current_by_campaign, promotions, c3_promoted)
    )
    final = completeness_report(expected_ids, merged_by_id)
    if not final.complete:
        raise CampaignMergeError(
            "final checkpoint validation failed: "
            f"missing={sorted(final.partition.missing_ids)}, "
            f"extra={sorted(final.partition.extra_ids)}, "
            f"duplicates={sorted(final.partition.duplicate_ids)}, "
            f"issues={dict(final.incomplete_by_issue)}"
        )

    throughput = build_throughput_report(
        {source.campaign_id: (source.records_root, source.runtime_root) for source in sources}
    )
    per_campaign = {
        campaign_id: _status_report([records[stable_id] for stable_id in sorted(records)])
        for campaign_id, records in sorted(current_by_campaign.items())
    }
    current = tuple(merged_by_id[stable_id] for stable_id in sorted(merged_by_id))
    report: JsonObject = {
        "format": "menagerie.crawler.campaign-merge-report.v1",
        "manifest_sha256": hash_bytes(manifest_path.read_bytes()),
        "proof": {
            "frozen_partition_rechecked": True,
            "actual_processed_pairwise_disjoint": True,
            "actual_processed_union_equals_roster": True,
            "author_model_identity_matches_manifest": True,
            "campaign_ledgers_read_only": True,
            "final_checkpoint_validation": "passed",
        },
        "rate_definitions": {
            "quality_rate": "runs / terminal_count",
            "reject_rate": "(failed + skipped) / terminal_count",
            "deferred_rate": "deferred / terminal_count",
        },
        "campaigns": per_campaign,
        "total": _status_report(current),
        "throughput": throughput,
        "promotions": {
            "intake_count": len(promotions),
            "consumed_count": len(superseded_promotions),
            "superseded_promotion_ids": list(superseded_promotions),
            "unconsumed_count": len(unconsumed_promotions),
            "unconsumed_promotion_ids": list(unconsumed_promotions),
        },
    }
    payloads = _view_payloads(current, report)
    after_reads = _ledger_fingerprint(sources)
    if after_reads != before:
        raise CampaignMergeError("campaign canonical ledgers changed during read-only merge")
    for relative, payload in payloads.items():
        atomic_replace_bytes(resolved_output / relative, payload)
    after_writes = _ledger_fingerprint(sources)
    if after_writes != before:
        raise CampaignMergeError(
            "campaign canonical ledgers changed while merged views were written"
        )
    view_digests = {relative: hash_bytes(payload) for relative, payload in sorted(payloads.items())}
    return MergeResult(current, report, view_digests)
