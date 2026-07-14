"""Trusted discovery snapshots with durable, idempotent model identities."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.identity import load_stable_ids

from menagerie.crawler.identity import canonical_json_bytes, hash_bytes, stable_hash
from menagerie.crawler.models import JsonObject


class IntakeError(ValueError):
    """Raised when trusted discovery inputs cannot form one consistent snapshot."""


@dataclass(frozen=True)
class IntakeItem:
    """One trusted roster member in an intake snapshot.

    Parameters
    ----------
    stable_id:
        Durable model identifier.
    name, zoo, variant:
        Minimal natural key needed to preserve model identity.
    discovery_source:
        Input stream in which the roster member was observed.
    legacy_row_sha256:
        Hash of the complete untrusted legacy row.
    """

    stable_id: str
    name: str
    zoo: str
    variant: str
    discovery_source: str
    legacy_row_sha256: str

    @property
    def natural_key(self) -> tuple[str, str, str]:
        """Return the immutable legacy natural key.

        Returns
        -------
        tuple[str, str, str]
            Name, zoo, and variant.
        """

        return (self.name, self.zoo, self.variant)

    def to_dict(self) -> JsonObject:
        """Return the canonical JSON representation.

        Returns
        -------
        dict[str, Any]
            JSON-compatible intake item.
        """

        return {
            "stable_id": self.stable_id,
            "name": self.name,
            "zoo": self.zoo,
            "variant": self.variant,
            "discovery_source": self.discovery_source,
            "legacy_row_sha256": self.legacy_row_sha256,
        }


@dataclass(frozen=True)
class IntakeSnapshot:
    """Immutable content-addressed discovery snapshot.

    Parameters
    ----------
    snapshot_id, snapshot_sha256:
        Content-derived snapshot identity and full digest.
    root:
        Snapshot directory containing the manifest, items, and exact source bytes.
    items:
        Stable-ID-assigned trusted roster.
    created:
        Whether this call materialized a new snapshot.
    """

    snapshot_id: str
    snapshot_sha256: str
    root: Path
    items: tuple[IntakeItem, ...]
    created: bool


def _read_jsonl(path: Path) -> list[JsonObject]:
    """Read a JSONL discovery stream as object rows.

    Parameters
    ----------
    path:
        Source JSONL path.

    Returns
    -------
    list[dict[str, Any]]
        Parsed rows in source order.

    Raises
    ------
    IntakeError
        If a line is malformed or is not an object.
    """

    rows: list[JsonObject] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise IntakeError(f"{path}:{line_number} must contain a JSON object")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise IntakeError(f"cannot read discovery input {path}: {exc}") from exc
    return rows


def _natural_key(row: Mapping[str, Any], source: str) -> tuple[str, str, str]:
    """Extract the minimal durable identity key from an untrusted row.

    Parameters
    ----------
    row:
        Legacy discovery row.
    source:
        Source name used in diagnostics.

    Returns
    -------
    tuple[str, str, str]
        Name, zoo, and variant.
    """

    name = row.get("name")
    zoo = row.get("zoo")
    if not isinstance(name, str) or not name.strip():
        raise IntakeError(f"{source} row has no non-empty name")
    if not isinstance(zoo, str) or not zoo.strip():
        raise IntakeError(f"{source} row {name!r} has no non-empty zoo")
    variant = row.get("variant", "")
    if not isinstance(variant, str):
        raise IntakeError(f"{source} row {name!r} has a non-string variant")
    return (name, zoo, variant)


def _atomic_write(path: Path, data: bytes) -> None:
    """Atomically write and fsync one snapshot file.

    Parameters
    ----------
    path:
        Destination path.
    data:
        Exact bytes to persist.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _assigned_id(
    key: tuple[str, str, str], preserved_ids: Mapping[tuple[str, str, str], str]
) -> str:
    """Return a preserved legacy ID or a deterministic ID for a new key.

    Parameters
    ----------
    key:
        Natural model key.
    preserved_ids:
        Existing durable ID mapping.

    Returns
    -------
    str
        Durable stable ID.
    """

    preserved = preserved_ids.get(key)
    if preserved is not None:
        return preserved
    digest = stable_hash({"namespace": "menagerie-crawler-v1", "natural_key": list(key)})
    return f"m_{digest.removeprefix('sha256:')[:20]}"


def _build_items(
    streams: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    preserved_ids: Mapping[tuple[str, str, str], str],
) -> tuple[IntakeItem, ...]:
    """Build the unique stable-ID roster across discovery streams.

    Parameters
    ----------
    streams:
        Named discovery row sequences.
    preserved_ids:
        Existing durable ID mapping.

    Returns
    -------
    tuple[IntakeItem, ...]
        Canonically sorted roster.

    Raises
    ------
    IntakeError
        If a natural key or stable ID conflicts.
    """

    by_key: dict[tuple[str, str, str], IntakeItem] = {}
    by_id: dict[str, tuple[str, str, str]] = {}
    for source, rows in streams:
        for row in rows:
            key = _natural_key(row, source)
            stable_id = _assigned_id(key, preserved_ids)
            prior_key = by_id.get(stable_id)
            if prior_key is not None and prior_key != key:
                raise IntakeError(f"stable ID {stable_id!r} maps to both {prior_key!r} and {key!r}")
            item = IntakeItem(
                stable_id=stable_id,
                name=key[0],
                zoo=key[1],
                variant=key[2],
                discovery_source=source,
                legacy_row_sha256=stable_hash(row),
            )
            existing = by_key.get(key)
            if existing is not None:
                if existing.legacy_row_sha256 != item.legacy_row_sha256:
                    raise IntakeError(f"conflicting discovery rows for natural key {key!r}")
                continue
            by_key[key] = item
            by_id[stable_id] = key
    return tuple(sorted(by_key.values(), key=lambda item: item.stable_id))


def create_intake_snapshot(
    master_catalog: Path,
    deferred_catalog: Path,
    output_root: Path,
    *,
    stable_ids_path: Optional[Path] = None,
    discovery_only: Iterable[Mapping[str, Any]] = (),
    additional_streams: Sequence[tuple[str, Sequence[Mapping[str, Any]]]] = (),
) -> IntakeSnapshot:
    """Snapshot trusted roster inputs and assign durable IDs idempotently.

    Only roster membership and the minimal natural key are promoted into intake.
    Every other legacy field survives solely through the row hash.

    Parameters
    ----------
    master_catalog, deferred_catalog:
        Frozen discovery JSONL inputs.
    output_root:
        Parent directory for content-addressed snapshots.
    stable_ids_path:
        Optional existing durable-ID table to preserve.
    discovery_only:
        Additional discovery-only roster rows.
    additional_streams:
        Named roster streams, such as classic registry hints.

    Returns
    -------
    IntakeSnapshot
        Existing or newly materialized immutable snapshot.
    """

    master_rows = _read_jsonl(master_catalog)
    deferred_rows = _read_jsonl(deferred_catalog)
    preserved = load_stable_ids(stable_ids_path) if stable_ids_path is not None else {}
    streams: list[tuple[str, Sequence[Mapping[str, Any]]]] = [
        ("master_catalog", master_rows),
        ("deferred", deferred_rows),
        ("discovery_only", tuple(discovery_only)),
        *additional_streams,
    ]
    items = _build_items(streams, preserved)
    source_bytes = {
        "master_catalog.jsonl": master_catalog.read_bytes(),
        "deferred.jsonl": deferred_catalog.read_bytes(),
    }
    preserved_rows = [
        {"name": key[0], "zoo": key[1], "variant": key[2], "stable_id": stable_id}
        for key, stable_id in sorted(preserved.items())
    ]
    snapshot_basis = {
        "format": "menagerie.crawler.intake.v1",
        "sources": {name: hash_bytes(data) for name, data in sorted(source_bytes.items())},
        "stable_ids": stable_hash(preserved_rows),
        "items": [item.to_dict() for item in items],
    }
    snapshot_sha256 = stable_hash(snapshot_basis)
    snapshot_id = f"intake-{snapshot_sha256.removeprefix('sha256:')[:20]}"
    snapshot_root = output_root / snapshot_id
    manifest = {
        **snapshot_basis,
        "snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_sha256,
        "item_count": len(items),
    }
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    items_bytes = b"".join(canonical_json_bytes(item.to_dict()) + b"\n" for item in items)
    manifest_path = snapshot_root / "manifest.json"
    if manifest_path.exists():
        if manifest_path.read_bytes() != manifest_bytes:
            raise IntakeError(f"snapshot identity collision at {snapshot_root}")
        if (snapshot_root / "items.jsonl").read_bytes() != items_bytes:
            raise IntakeError(f"snapshot items changed at {snapshot_root}")
        return IntakeSnapshot(snapshot_id, snapshot_sha256, snapshot_root, items, False)

    for name, data in source_bytes.items():
        _atomic_write(snapshot_root / "sources" / name, data)
    _atomic_write(snapshot_root / "items.jsonl", items_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    return IntakeSnapshot(snapshot_id, snapshot_sha256, snapshot_root, items, True)


def load_intake_snapshot(snapshot_root: Path) -> IntakeSnapshot:
    """Load and verify a previously materialized intake snapshot.

    Parameters
    ----------
    snapshot_root:
        Directory containing ``manifest.json`` and ``items.jsonl``.

    Returns
    -------
    IntakeSnapshot
        Verified immutable snapshot.
    """

    manifest_value = json.loads((snapshot_root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest_value, dict):
        raise IntakeError("intake manifest must be a JSON object")
    rows = _read_jsonl(snapshot_root / "items.jsonl")
    items = tuple(
        IntakeItem(
            stable_id=str(row["stable_id"]),
            name=str(row["name"]),
            zoo=str(row["zoo"]),
            variant=str(row["variant"]),
            discovery_source=str(row["discovery_source"]),
            legacy_row_sha256=str(row["legacy_row_sha256"]),
        )
        for row in rows
    )
    expected = stable_hash(
        {
            key: value
            for key, value in manifest_value.items()
            if key not in {"snapshot_id", "snapshot_sha256", "item_count"}
        }
    )
    if expected != manifest_value.get("snapshot_sha256"):
        raise IntakeError("intake manifest digest does not match its contents")
    return IntakeSnapshot(
        snapshot_id=str(manifest_value["snapshot_id"]),
        snapshot_sha256=expected,
        root=snapshot_root,
        items=items,
        created=False,
    )
