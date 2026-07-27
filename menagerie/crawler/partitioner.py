"""Deterministic tier-aligned campaign partitioning for the crawler roster."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.identity import atomic_replace_bytes, canonical_json_bytes, hash_bytes
from menagerie.crawler.intake import IntakeSnapshot, create_intake_snapshot
from menagerie.crawler.models import JsonObject

CAMPAIGN_MANIFEST_FORMAT = "menagerie.crawler.campaigns.v1"
DEFAULT_CAMPAIGN_MANIFEST = Path("menagerie/crawler/records/campaigns.json")
REVIEW_COHORT_SIZE = 1_000

_DISCOVERED_PYTORCH_ZOO = "discovered-pytorch"
_CLASSICS_ZOO = "unregistered-classics-pytorch"
_NATIVE_ZOOS = frozenset(
    {
        "discovered-tensorflow",
        "discovered-jax",
        "discovered-keras",
    }
)


class CampaignPartitionError(ValueError):
    """Raised when campaign partition inputs or outputs violate exact coverage."""


@dataclass(frozen=True)
class CampaignSpec:
    """Frozen configuration for one independent single-writer campaign.

    Parameters
    ----------
    campaign_id:
        Stable campaign identifier.
    workload_class:
        Human-readable workload partition.
    phase:
        Ordered crawler framework phase.
    author_model, checker_model:
        Exact wrapper model labels used to derive authority identities.
    review_checkpoint_at:
        One-shot review threshold, or zero when disabled after the canary.
    """

    campaign_id: str
    workload_class: str
    phase: str
    author_model: str
    checker_model: str
    review_checkpoint_at: int


@dataclass(frozen=True)
class CampaignBinding:
    """Manifest binding between one campaign and its immutable intake snapshot."""

    spec: CampaignSpec
    row_count: int
    partition_path: str
    partition_sha256: str
    intake_path: str
    intake_snapshot_id: str
    intake_snapshot_sha256: str
    review_cohort_size: int


CAMPAIGN_SPECS: tuple[CampaignSpec, ...] = (
    CampaignSpec(
        campaign_id="c1-mech",
        workload_class="library-zoo-mechanical",
        phase="pytorch",
        author_model="claude-sonnet",
        checker_model="gpt-5.6-terra",
        review_checkpoint_at=REVIEW_COHORT_SIZE,
    ),
    CampaignSpec(
        campaign_id="c2-disco",
        workload_class="discovered-pytorch",
        phase="pytorch",
        author_model="claude-sonnet",
        checker_model="gpt-5.6-terra",
        review_checkpoint_at=0,
    ),
    CampaignSpec(
        campaign_id="c3-classics",
        workload_class="unregistered-classics-and-family-companions",
        phase="pytorch",
        author_model="claude-opus-5",
        checker_model="gpt-5.6-sol",
        review_checkpoint_at=0,
    ),
    CampaignSpec(
        campaign_id="c4-native",
        workload_class="discovered-tensorflow-jax-keras",
        phase="native-tail",
        author_model="claude-sonnet",
        checker_model="gpt-5.6-terra",
        review_checkpoint_at=0,
    ),
)

_SPEC_BY_ID = {spec.campaign_id: spec for spec in CAMPAIGN_SPECS}


def _read_jsonl(path: Path) -> list[JsonObject]:
    """Read one JSONL file as an ordered list of object rows.

    Parameters
    ----------
    path:
        JSONL file to read.

    Returns
    -------
    list[dict[str, Any]]
        Parsed object rows in file order.

    Raises
    ------
    CampaignPartitionError
        If a row is malformed or not a JSON object.
    """

    rows: list[JsonObject] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CampaignPartitionError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignPartitionError(f"cannot read campaign input {path}: {exc}") from exc
    return rows


def _natural_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the roster row's immutable natural key.

    Parameters
    ----------
    row:
        Roster row.

    Returns
    -------
    tuple[str, str, str]
        Name, zoo, and variant.

    Raises
    ------
    CampaignPartitionError
        If the natural key is incomplete.
    """

    name = row.get("name")
    zoo = row.get("zoo")
    variant = row.get("variant", "")
    if not isinstance(name, str) or not name:
        raise CampaignPartitionError("roster row has no non-empty name")
    if not isinstance(zoo, str) or not zoo:
        raise CampaignPartitionError(f"roster row {name!r} has no non-empty zoo")
    if not isinstance(variant, str):
        raise CampaignPartitionError(f"roster row {name!r} has a non-string variant")
    return (name, zoo, variant)


def _initial_campaign(row: Mapping[str, Any]) -> str:
    """Classify one row by workload class and framework phase.

    Parameters
    ----------
    row:
        Validated roster row.

    Returns
    -------
    str
        Campaign identifier before family co-location.
    """

    zoo = str(row["zoo"])
    if zoo == _DISCOVERED_PYTORCH_ZOO:
        return "c2-disco"
    if zoo == _CLASSICS_ZOO:
        return "c3-classics"
    if zoo in _NATIVE_ZOOS:
        return "c4-native"
    return "c1-mech"


def _stratified_prefix(
    rows: Sequence[JsonObject], *, prefix_size: int = REVIEW_COHORT_SIZE
) -> tuple[JsonObject, ...]:
    """Put a deterministic zoo-proportional review cohort first.

    Hamilton apportionment keeps each zoo's prefix allocation within one row of
    its exact proportional share. Selection and ordering use canonical row bytes,
    so source-file order cannot bias the canary.

    Parameters
    ----------
    rows:
        C1 mechanical rows.
    prefix_size:
        Number of rows in the representative prefix.

    Returns
    -------
    tuple[dict[str, Any], ...]
        All input rows exactly once, with the representative cohort first.
    """

    if prefix_size <= 0 or len(rows) <= prefix_size:
        return tuple(rows)
    by_zoo: dict[str, list[JsonObject]] = defaultdict(list)
    for row in rows:
        by_zoo[str(row["zoo"])].append(row)
    for zoo_rows in by_zoo.values():
        zoo_rows.sort(key=canonical_json_bytes)

    exact_shares = {
        zoo: prefix_size * len(zoo_rows) / len(rows) for zoo, zoo_rows in by_zoo.items()
    }
    quotas = {zoo: int(share) for zoo, share in exact_shares.items()}
    remaining = prefix_size - sum(quotas.values())
    remainder_order = sorted(
        by_zoo,
        key=lambda zoo: (
            -(exact_shares[zoo] - quotas[zoo]),
            zoo,
        ),
    )
    for zoo in remainder_order[:remaining]:
        quotas[zoo] += 1

    selected: list[JsonObject] = []
    deferred: list[JsonObject] = []
    for zoo, zoo_rows in sorted(by_zoo.items()):
        quota = quotas[zoo]
        selected.extend(zoo_rows[:quota])
        deferred.extend(zoo_rows[quota:])
    selected.sort(key=canonical_json_bytes)
    deferred.sort(key=canonical_json_bytes)
    return tuple((*selected, *deferred))


def assert_campaign_partition(
    roster: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    """Prove pairwise disjointness, total coverage, and family co-location.

    Parameters
    ----------
    roster:
        Complete source roster.
    partitions:
        Campaign ID to emitted rows.

    Raises
    ------
    CampaignPartitionError
        If campaign names, row membership, coverage, uniqueness, or family
        co-location differ from the frozen contract.
    """

    if set(partitions) != set(_SPEC_BY_ID):
        raise CampaignPartitionError("campaign partition names differ from the frozen four")
    roster_keys = [_natural_key(row) for row in roster]
    if len(roster_keys) != len(set(roster_keys)):
        raise CampaignPartitionError("source roster contains duplicate natural keys")

    owner_by_key: dict[tuple[str, str, str], str] = {}
    family_owners: dict[str, set[str]] = defaultdict(set)
    for campaign_id, rows in partitions.items():
        for row in rows:
            key = _natural_key(row)
            prior = owner_by_key.setdefault(key, campaign_id)
            if prior != campaign_id:
                raise CampaignPartitionError(
                    f"roster row {key!r} appears in both {prior} and {campaign_id}"
                )
            family = row.get("family")
            if isinstance(family, str) and family:
                family_owners[family].add(campaign_id)

    expected = set(roster_keys)
    observed = set(owner_by_key)
    if expected != observed:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise CampaignPartitionError(
            f"campaign union differs from roster: missing={missing[:5]}, extra={extra[:5]}"
        )
    split_families = sorted(family for family, owners in family_owners.items() if len(owners) > 1)
    if split_families:
        raise CampaignPartitionError(
            f"non-null families span campaigns: {split_families[:5]}"
        )


def partition_roster(
    roster: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[JsonObject, ...]]:
    """Partition the roster into four tier-aligned campaigns.

    Families containing an unregistered-classics row are promoted in full to C3.
    This is the conservative resolution when a family otherwise crosses the
    mechanical/classics boundary: hard work may be promoted, but no classic is
    silently down-tiered and no non-null family is split.

    Parameters
    ----------
    roster:
        Complete roster rows.

    Returns
    -------
    dict[str, tuple[dict[str, Any], ...]]
        Ordered rows for C1 through C4.
    """

    normalized = [dict(row) for row in roster]
    keys = [_natural_key(row) for row in normalized]
    if len(keys) != len(set(keys)):
        raise CampaignPartitionError("source roster contains duplicate natural keys")
    classic_families = {
        str(row["family"])
        for row in normalized
        if row["zoo"] == _CLASSICS_ZOO
        and isinstance(row.get("family"), str)
        and row["family"]
    }
    mutable: dict[str, list[JsonObject]] = {spec.campaign_id: [] for spec in CAMPAIGN_SPECS}
    for row in normalized:
        family = row.get("family")
        campaign_id = (
            "c3-classics"
            if isinstance(family, str) and family in classic_families
            else _initial_campaign(row)
        )
        mutable[campaign_id].append(row)
    mutable["c1-mech"] = list(_stratified_prefix(mutable["c1-mech"]))
    partitions = {campaign_id: tuple(rows) for campaign_id, rows in mutable.items()}
    assert_campaign_partition(normalized, partitions)
    return partitions


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    """Serialize object rows as canonical newline-terminated JSONL.

    Parameters
    ----------
    rows:
        Rows to serialize.

    Returns
    -------
    bytes
        Canonical JSONL bytes.
    """

    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def emit_campaign_partitions(
    roster_path: Path,
    output_root: Path,
    *,
    stable_ids_path: Optional[Path] = None,
) -> tuple[CampaignBinding, ...]:
    """Emit roster partitions, immutable intake snapshots, and ``campaigns.json``.

    Parameters
    ----------
    roster_path:
        Complete source roster JSONL.
    output_root:
        Canonical crawler records root.
    stable_ids_path:
        Optional stable-ID registry to preserve.

    Returns
    -------
    tuple[CampaignBinding, ...]
        Frozen campaign bindings in C1--C4 order.
    """

    roster = _read_jsonl(roster_path)
    partitions = partition_roster(roster)
    partitions_root = output_root / "partitions"
    intake_root = output_root / "intake"
    empty_deferred = partitions_root / "empty-deferred.jsonl"
    atomic_replace_bytes(empty_deferred, b"\n")

    bindings: list[CampaignBinding] = []
    for spec in CAMPAIGN_SPECS:
        rows = partitions[spec.campaign_id]
        partition_path = partitions_root / f"{spec.campaign_id}.jsonl"
        partition_bytes = _jsonl_bytes(rows)
        atomic_replace_bytes(partition_path, partition_bytes)
        snapshot = create_intake_snapshot(
            partition_path,
            empty_deferred,
            intake_root,
            stable_ids_path=stable_ids_path,
        )
        bindings.append(
            CampaignBinding(
                spec=spec,
                row_count=len(rows),
                partition_path=partition_path.relative_to(output_root).as_posix(),
                partition_sha256=hash_bytes(partition_bytes),
                intake_path=snapshot.root.relative_to(output_root).as_posix(),
                intake_snapshot_id=snapshot.snapshot_id,
                intake_snapshot_sha256=snapshot.snapshot_sha256,
                review_cohort_size=spec.review_checkpoint_at,
            )
        )

    manifest = {
        "format": CAMPAIGN_MANIFEST_FORMAT,
        "roster": {
            "path": Path(os.path.relpath(roster_path, output_root)).as_posix(),
            "sha256": hash_bytes(roster_path.read_bytes()),
            "row_count": len(roster),
        },
        "proof": {
            "pairwise_disjoint": True,
            "union_equals_roster": True,
            "non_null_families_colocated": True,
        },
        "campaigns": [
            {
                "campaign_id": binding.spec.campaign_id,
                "workload_class": binding.spec.workload_class,
                "phase": binding.spec.phase,
                "author_model": binding.spec.author_model,
                "checker_model": binding.spec.checker_model,
                "review_checkpoint_at": binding.spec.review_checkpoint_at,
                "row_count": binding.row_count,
                "partition_path": binding.partition_path,
                "partition_sha256": binding.partition_sha256,
                "intake_path": binding.intake_path,
                "intake_snapshot_id": binding.intake_snapshot_id,
                "intake_snapshot_sha256": binding.intake_snapshot_sha256,
                "review_cohort": {
                    "size": binding.review_cohort_size,
                    "stratified_by": ["zoo"] if binding.review_cohort_size else [],
                    "source_order": "partition-prefix",
                },
            }
            for binding in bindings
        ],
    }
    atomic_replace_bytes(output_root / "campaigns.json", canonical_json_bytes(manifest) + b"\n")
    return tuple(bindings)


def load_campaign_bindings(path: Path) -> tuple[CampaignBinding, ...]:
    """Load and validate a campaign manifest.

    Parameters
    ----------
    path:
        ``campaigns.json`` path.

    Returns
    -------
    tuple[CampaignBinding, ...]
        Validated bindings in frozen campaign order.

    Raises
    ------
    CampaignPartitionError
        If the manifest does not exactly bind the frozen campaigns and models.
    """

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CampaignPartitionError(f"cannot read campaign manifest {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("format") != CAMPAIGN_MANIFEST_FORMAT:
        raise CampaignPartitionError("campaign manifest format is invalid")
    if value.get("proof") != {
        "pairwise_disjoint": True,
        "union_equals_roster": True,
        "non_null_families_colocated": True,
    }:
        raise CampaignPartitionError("campaign manifest coverage proof is invalid")
    roster = value.get("roster")
    if (
        not isinstance(roster, dict)
        or not isinstance(roster.get("path"), str)
        or not roster["path"]
        or not isinstance(roster.get("row_count"), int)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", str(roster.get("sha256"))) is None
    ):
        raise CampaignPartitionError("campaign manifest roster binding is invalid")
    raw_campaigns = value.get("campaigns")
    if not isinstance(raw_campaigns, list):
        raise CampaignPartitionError("campaign manifest has no campaign list")

    bindings: list[CampaignBinding] = []
    for raw in raw_campaigns:
        if not isinstance(raw, dict):
            raise CampaignPartitionError("campaign binding must be an object")
        campaign_id = raw.get("campaign_id")
        if not isinstance(campaign_id, str) or campaign_id not in _SPEC_BY_ID:
            raise CampaignPartitionError(f"unknown campaign binding: {campaign_id!r}")
        spec = _SPEC_BY_ID[campaign_id]
        for field in (
            "workload_class",
            "phase",
            "author_model",
            "checker_model",
            "review_checkpoint_at",
        ):
            if raw.get(field) != getattr(spec, field):
                raise CampaignPartitionError(
                    f"{campaign_id} has stale {field}: {raw.get(field)!r}"
                )
        cohort = raw.get("review_cohort")
        if (
            not isinstance(cohort, dict)
            or cohort.get("size") != spec.review_checkpoint_at
            or cohort.get("stratified_by") != (["zoo"] if spec.review_checkpoint_at else [])
            or cohort.get("source_order") != "partition-prefix"
        ):
            raise CampaignPartitionError(f"{campaign_id} review cohort is malformed")
        row_count = raw.get("row_count")
        if not isinstance(row_count, int) or row_count < 1:
            raise CampaignPartitionError(f"{campaign_id} row count is invalid")
        string_fields = (
            "partition_path",
            "partition_sha256",
            "intake_path",
            "intake_snapshot_id",
            "intake_snapshot_sha256",
        )
        if not all(isinstance(raw.get(field), str) and raw[field] for field in string_fields):
            raise CampaignPartitionError(f"{campaign_id} binding paths or identities are invalid")
        for field in ("partition_sha256", "intake_snapshot_sha256"):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", str(raw[field])) is None:
                raise CampaignPartitionError(f"{campaign_id} {field} is invalid")
        bindings.append(
            CampaignBinding(
                spec=spec,
                row_count=row_count,
                partition_path=str(raw["partition_path"]),
                partition_sha256=str(raw["partition_sha256"]),
                intake_path=str(raw["intake_path"]),
                intake_snapshot_id=str(raw["intake_snapshot_id"]),
                intake_snapshot_sha256=str(raw["intake_snapshot_sha256"]),
                review_cohort_size=spec.review_checkpoint_at,
            )
        )
    if [binding.spec.campaign_id for binding in bindings] != [
        spec.campaign_id for spec in CAMPAIGN_SPECS
    ]:
        raise CampaignPartitionError("campaign manifest order or membership is invalid")
    if sum(binding.row_count for binding in bindings) != roster["row_count"]:
        raise CampaignPartitionError("campaign row counts do not cover the roster count")
    snapshot_ids = [binding.intake_snapshot_id for binding in bindings]
    if len(snapshot_ids) != len(set(snapshot_ids)):
        raise CampaignPartitionError("campaign manifest reuses an intake snapshot")
    return tuple(bindings)


def campaign_binding_for_snapshot(
    manifest_path: Path,
    snapshot: IntakeSnapshot,
) -> CampaignBinding:
    """Resolve the unique campaign bound to an intake snapshot.

    Parameters
    ----------
    manifest_path:
        Valid campaign manifest path.
    snapshot:
        Loaded immutable intake snapshot.

    Returns
    -------
    CampaignBinding
        Exact matching campaign.

    Raises
    ------
    CampaignPartitionError
        If zero or multiple bindings match.
    """

    binding = find_campaign_binding(manifest_path, snapshot)
    if binding is None:
        raise CampaignPartitionError(
            f"intake snapshot {snapshot.snapshot_id} has 0 campaign bindings"
        )
    return binding


def find_campaign_binding(
    manifest_path: Path,
    snapshot: IntakeSnapshot,
) -> Optional[CampaignBinding]:
    """Find a campaign binding for an intake snapshot when one exists.

    Parameters
    ----------
    manifest_path:
        Valid campaign manifest path.
    snapshot:
        Loaded immutable intake snapshot.

    Returns
    -------
    CampaignBinding or None
        Unique matching campaign, or ``None`` for an unrelated snapshot.

    Raises
    ------
    CampaignPartitionError
        If multiple bindings claim the snapshot.
    """

    matches = tuple(
        binding
        for binding in load_campaign_bindings(manifest_path)
        if binding.intake_snapshot_id == snapshot.snapshot_id
        and binding.intake_snapshot_sha256 == snapshot.snapshot_sha256
    )
    if len(matches) > 1:
        raise CampaignPartitionError(
            f"intake snapshot {snapshot.snapshot_id} has {len(matches)} campaign bindings"
        )
    return matches[0] if matches else None


def campaign_counts(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, int]:
    """Return frozen-order campaign row counts.

    Parameters
    ----------
    partitions:
        Valid campaign partition.

    Returns
    -------
    Mapping[str, int]
        Campaign ID to row count.
    """

    return {
        spec.campaign_id: len(partitions[spec.campaign_id])
        for spec in CAMPAIGN_SPECS
    }
