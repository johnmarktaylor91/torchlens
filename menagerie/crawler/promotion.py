"""Typed, append-only Sonnet-to-Opus intake promotions."""

from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from menagerie.crawler.author_dispatch import BlockedRecommendation
from menagerie.crawler.constants import PROMOTION_SCHEMA_VERSION
from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    fsync_directory,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.intake import IntakeItem, IntakeSnapshot, load_intake_snapshot
from menagerie.crawler.models import JsonObject
from menagerie.crawler.schema import PayloadValidationError, validate_payload

PROMOTION_DESTINATION_CAMPAIGN = "c3-classics"
PROMOTION_SOURCE_CAMPAIGNS = frozenset({"c1-mech", "c2-disco", "c4-native"})
PROMOTION_EXTENSION_RELATIVE_PATH = Path("intake-extensions") / "c3-classics.jsonl"
PROMOTION_RECORDS_RELATIVE_PATH = Path("promotion-campaigns") / "c3-classics"


class PromotionError(ValueError):
    """Raised when an intake promotion is malformed, conflicting, or cannot persist."""


def promotion_extension_path(records_root: Path) -> Path:
    """Return the canonical C3 intake-extension path below one records root.

    Parameters
    ----------
    records_root:
        Campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Append-only C3 intake-extension path.
    """

    return records_root / PROMOTION_EXTENSION_RELATIVE_PATH


def promotion_records_root(records_root: Path) -> Path:
    """Return the isolated C3 promotion-campaign records root.

    Parameters
    ----------
    records_root:
        C3 base campaign canonical records directory.

    Returns
    -------
    pathlib.Path
        Separate canonical ledger root for promoted C3 work.
    """

    return records_root / PROMOTION_RECORDS_RELATIVE_PATH


def promotion_intake_identity(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str, tuple[JsonObject, ...]]:
    """Derive the content-addressed authority root for one C3 promotion batch.

    Parameters
    ----------
    rows:
        Complete validated promotion extension.

    Returns
    -------
    tuple[str, str, tuple[dict[str, Any], ...]]
        Snapshot ID, full digest, and trusted promoted intake rows.

    Raises
    ------
    PromotionError
        If rows conflict or target something other than C3.
    """

    validated = tuple(validate_promotion_row(row) for row in rows)
    by_id = {str(row["stable_id"]): row for row in validated}
    if len(by_id) != len(validated):
        raise PromotionError("promotion intake duplicates a stable_id")
    ordered = tuple(by_id[stable_id] for stable_id in sorted(by_id))
    basis, _promotion_bytes = _promotion_intake_basis(ordered)
    digest = stable_hash(basis)
    snapshot_id = f"intake-{digest.removeprefix('sha256:')[:20]}"
    intake_rows = tuple(deepcopy(row["source_intake"]["item"]) for row in ordered)
    return snapshot_id, digest, intake_rows


def materialize_promotion_intake(
    rows: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> IntakeSnapshot:
    """Materialize a loadable C3 snapshot directly from typed promotion rows.

    Parameters
    ----------
    rows:
        Complete promotion extension rows.
    output_root:
        Parent directory for content-addressed promotion intake snapshots.

    Returns
    -------
    IntakeSnapshot
        Immutable snapshot whose item rows are exact copies of the originating
        frozen intake members.

    Raises
    ------
    PromotionError
        If promotions conflict or an existing content-addressed snapshot differs.
    """

    validated = tuple(validate_promotion_row(row) for row in rows)
    by_id = {str(row["stable_id"]): row for row in validated}
    if len(by_id) != len(validated):
        raise PromotionError("promotion intake duplicates a stable_id")
    ordered = tuple(by_id[stable_id] for stable_id in sorted(by_id))
    basis, promotion_bytes = _promotion_intake_basis(ordered)
    snapshot_sha256 = stable_hash(basis)
    snapshot_id = f"intake-{snapshot_sha256.removeprefix('sha256:')[:20]}"
    snapshot_root = output_root / snapshot_id
    intake_rows = tuple(deepcopy(row["source_intake"]["item"]) for row in ordered)
    manifest: JsonObject = {
        **basis,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "item_count": len(intake_rows),
    }
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    items_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in intake_rows)
    manifest_path = snapshot_root / "manifest.json"
    created = not manifest_path.exists()
    if created:
        atomic_replace_bytes(snapshot_root / "sources" / "promotions.jsonl", promotion_bytes)
        atomic_replace_bytes(snapshot_root / "items.jsonl", items_bytes)
        atomic_replace_bytes(manifest_path, manifest_bytes)
    elif (
        manifest_path.read_bytes() != manifest_bytes
        or (snapshot_root / "items.jsonl").read_bytes() != items_bytes
        or (snapshot_root / "sources" / "promotions.jsonl").read_bytes() != promotion_bytes
    ):
        raise PromotionError(f"promotion intake identity collision at {snapshot_root}")
    loaded = load_intake_snapshot(snapshot_root)
    return IntakeSnapshot(
        loaded.snapshot_id,
        loaded.snapshot_sha256,
        loaded.root,
        loaded.items,
        created,
    )


def promotion_snapshot_destination(snapshot: IntakeSnapshot) -> str | None:
    """Return the validated destination campaign for a promotion snapshot.

    Parameters
    ----------
    snapshot:
        Already hash-verified ordinary intake snapshot.

    Returns
    -------
    str | None
        ``c3-classics`` for a typed promotion snapshot, otherwise ``None``.

    Raises
    ------
    PromotionError
        If a snapshot claims promotion semantics without exact typed extension rows.
    """

    manifest_path = snapshot.root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PromotionError(f"cannot read promotion intake manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise PromotionError("promotion intake manifest must be an object")
    destination = manifest.get("promotion_destination_campaign")
    if destination is None:
        return None
    if destination != PROMOTION_DESTINATION_CAMPAIGN:
        raise PromotionError(f"unknown promotion destination campaign: {destination!r}")
    promotion_path = snapshot.root / "sources" / "promotions.jsonl"
    promotions = _decode_promotion_lines(promotion_path.read_bytes(), promotion_path)
    expected_ids = [item.stable_id for item in snapshot.items]
    observed_ids = [str(row["stable_id"]) for row in promotions]
    if observed_ids != expected_ids or manifest.get("promotion_sha256s") != [
        row["promotion_sha256"] for row in promotions
    ]:
        raise PromotionError("promotion snapshot rows do not bind its exact intake members")
    return PROMOTION_DESTINATION_CAMPAIGN


def _promotion_intake_basis(
    ordered: Sequence[Mapping[str, Any]],
) -> tuple[JsonObject, bytes]:
    """Build the exact ordinary-intake basis for ordered promotion rows.

    Parameters
    ----------
    ordered:
        Stable-ID-ordered validated promotion rows.

    Returns
    -------
    tuple[dict[str, Any], bytes]
        Loadable intake manifest basis and exact typed extension bytes.
    """

    promotion_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in ordered)
    intake_rows = [deepcopy(row["source_intake"]["item"]) for row in ordered]
    stable_ids = [
        {
            "name": row["name"],
            "zoo": row["zoo"],
            "variant": row["variant"],
            "stable_id": row["stable_id"],
        }
        for row in intake_rows
    ]
    basis: JsonObject = {
        "format": "menagerie.crawler.intake.v1",
        "sources": {"promotions.jsonl": hash_bytes(promotion_bytes)},
        "stable_ids": stable_hash(stable_ids),
        "items": intake_rows,
        "promotion_destination_campaign": PROMOTION_DESTINATION_CAMPAIGN,
        "promotion_sha256s": [row["promotion_sha256"] for row in ordered],
    }
    return basis, promotion_bytes


def build_promotion_row(
    *,
    source_campaign_id: str,
    snapshot: IntakeSnapshot,
    item: IntakeItem,
    result: BlockedRecommendation,
    source_manifest: Mapping[str, Any],
    created_at: str,
) -> JsonObject:
    """Build one self-hashed promotion row from an accepted higher-tier deferral.

    Parameters
    ----------
    source_campaign_id:
        Frozen Sonnet campaign that produced the deferral.
    snapshot, item:
        Exact originating intake trust root and member.
    result:
        Checked ``BLOCKED(needs-higher-tier)`` author result.
    source_manifest:
        Frozen source manifest retained for C3.
    created_at:
        UTC row creation timestamp.

    Returns
    -------
    dict[str, Any]
        Schema-valid, content-addressed C3 intake promotion.

    Raises
    ------
    PromotionError
        If the source campaign, result, intake, manifest, or research summary is
        inconsistent.
    """

    if source_campaign_id not in PROMOTION_SOURCE_CAMPAIGNS:
        raise PromotionError(
            f"promotion source campaign must be one of {sorted(PROMOTION_SOURCE_CAMPAIGNS)}"
        )
    if result.reason_code != "needs-higher-tier":
        raise PromotionError("only BLOCKED(needs-higher-tier) can enter the C3 extension")
    if result.binding.stable_id != item.stable_id:
        raise PromotionError("promotion author result does not match its intake item")
    if result.binding.intake_snapshot_id != snapshot.snapshot_id or (
        result.binding.intake_snapshot_sha256 != snapshot.snapshot_sha256
    ):
        raise PromotionError("promotion author result does not match its intake snapshot")
    item_value = item.to_dict()
    item_sha256 = stable_hash(item_value)
    if result.binding.intake_item_sha256 != item_sha256:
        raise PromotionError("promotion author result does not match its frozen intake bytes")
    manifest = deepcopy(dict(source_manifest))
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        raise PromotionError("promotion source manifest has no source list")
    manifest_identity = stable_hash(sources)
    if manifest.get("manifest_sha256") != manifest_identity or (
        result.binding.source_manifest_identity != manifest_identity
    ):
        raise PromotionError("promotion source manifest identity is inconsistent")
    if result.research_summary is None:
        raise PromotionError("higher-tier promotion requires the stage-1 research summary")
    if created_at != result.binding.created_at:
        raise PromotionError("promotion timestamp must equal the prior attempt timestamp")
    body: JsonObject = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "stable_id": item.stable_id,
        "source_campaign_id": source_campaign_id,
        "destination_campaign_id": PROMOTION_DESTINATION_CAMPAIGN,
        "created_at": created_at,
        "source_intake": {
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_sha256": snapshot.snapshot_sha256,
            "item_sha256": item_sha256,
            "item": item_value,
        },
        "source_manifest": manifest,
        "source_manifest_identity": manifest_identity,
        "stage1_research_summary": deepcopy(result.research_summary),
        "prior_attempt": {
            "result_id": result.binding.result_id,
            "result_sha256": result.binding.result_sha256,
            "work_id": result.binding.work_id,
            "kind": "BLOCKED",
            "reason_code": result.reason_code,
            "author_result": deepcopy(result.binding.raw_result),
        },
    }
    promotion_sha256 = stable_hash(body)
    row = {
        **body,
        "promotion_id": f"promotion-{promotion_sha256.removeprefix('sha256:')[:20]}",
        "promotion_sha256": promotion_sha256,
    }
    return validate_promotion_row(row)


def validate_promotion_row(value: Mapping[str, Any]) -> JsonObject:
    """Validate one promotion schema and every redundant identity binding.

    Parameters
    ----------
    value:
        Candidate promotion row.

    Returns
    -------
    dict[str, Any]
        Defensive validated copy.

    Raises
    ------
    PromotionError
        If schema validation or a redundant content binding fails.
    """

    row = deepcopy(dict(value))
    try:
        validate_payload(row, PROMOTION_SCHEMA_VERSION)
    except PayloadValidationError as exc:
        raise PromotionError(str(exc)) from exc
    unhashed = {
        key: value
        for key, value in row.items()
        if key not in {"promotion_id", "promotion_sha256"}
    }
    expected_hash = stable_hash(unhashed)
    expected_id = f"promotion-{expected_hash.removeprefix('sha256:')[:20]}"
    if row["promotion_sha256"] != expected_hash or row["promotion_id"] != expected_id:
        raise PromotionError("promotion row content identity is inconsistent")
    source_intake = row["source_intake"]
    if stable_hash(source_intake["item"]) != source_intake["item_sha256"]:
        raise PromotionError("promotion source intake item digest is inconsistent")
    if source_intake["item"].get("stable_id") != row["stable_id"]:
        raise PromotionError("promotion source intake stable_id is inconsistent")
    manifest = row["source_manifest"]
    manifest_identity = stable_hash(manifest["sources"])
    if (
        manifest["manifest_sha256"] != manifest_identity
        or row["source_manifest_identity"] != manifest_identity
    ):
        raise PromotionError("promotion source manifest digest is inconsistent")
    prior_attempt = row["prior_attempt"]
    author_result = prior_attempt["author_result"]
    if (
        author_result.get("result_id") != prior_attempt["result_id"]
        or author_result.get("result_sha256") != prior_attempt["result_sha256"]
        or author_result.get("stable_id") != row["stable_id"]
        or author_result.get("work_id") != prior_attempt["work_id"]
        or author_result.get("kind") != "BLOCKED"
        or author_result.get("payload", {}).get("reason_code") != "needs-higher-tier"
    ):
        raise PromotionError("promotion prior-attempt binding is inconsistent")
    return row


def append_promotion_row(path: Path, row: Mapping[str, Any]) -> JsonObject:
    """Append one promotion durably and idempotently under an exclusive file lock.

    Parameters
    ----------
    path:
        Destination C3 intake-extension JSONL.
    row:
        Complete typed promotion.

    Returns
    -------
    dict[str, Any]
        Validated persisted row.

    Raises
    ------
    PromotionError
        If the same stable ID already carries a different promotion.
    """

    validated = validate_promotion_row(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = _decode_promotion_lines(handle.read(), path)
        matching = [item for item in existing if item["stable_id"] == validated["stable_id"]]
        if matching:
            if len(matching) != 1 or matching[0] != validated:
                raise PromotionError(
                    f"conflicting C3 promotion for stable_id {validated['stable_id']}"
                )
            return matching[0]
        handle.seek(0, os.SEEK_END)
        handle.write(canonical_json_bytes(validated) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(path.parent)
    return validated


def load_promotion_rows(paths: Sequence[Path]) -> tuple[JsonObject, ...]:
    """Load and deduplicate promotion extensions from multiple campaign roots.

    Parameters
    ----------
    paths:
        Candidate extension paths. Missing files are empty extensions.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Stable-ID-ordered unique promotion rows.

    Raises
    ------
    PromotionError
        If extensions conflict for one stable ID.
    """

    by_id: dict[str, JsonObject] = {}
    for path in paths:
        if not path.is_file():
            continue
        rows = _decode_promotion_lines(path.read_bytes(), path)
        for row in rows:
            stable_id = str(row["stable_id"])
            prior = by_id.setdefault(stable_id, row)
            if prior != row:
                raise PromotionError(f"conflicting C3 promotions for stable_id {stable_id}")
    return tuple(by_id[stable_id] for stable_id in sorted(by_id))


def _decode_promotion_lines(data: bytes, path: Path) -> list[JsonObject]:
    """Decode and validate canonical promotion JSONL bytes.

    Parameters
    ----------
    data:
        Complete extension bytes.
    path:
        Diagnostic source path.

    Returns
    -------
    list[dict[str, Any]]
        Validated rows in append order.

    Raises
    ------
    PromotionError
        If a line is malformed, non-canonical, or duplicated.
    """

    rows: list[JsonObject] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(data.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            decoded = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PromotionError(f"{path}:{line_number} is invalid JSON") from exc
        if not isinstance(decoded, Mapping):
            raise PromotionError(f"{path}:{line_number} is not an object")
        row = validate_promotion_row(decoded)
        if raw_line != canonical_json_bytes(row):
            raise PromotionError(f"{path}:{line_number} is not canonical JSON")
        stable_id = str(row["stable_id"])
        if stable_id in seen:
            raise PromotionError(f"{path} duplicates promotion stable_id {stable_id}")
        seen.add(stable_id)
        rows.append(row)
    return rows
