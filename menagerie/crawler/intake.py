"""Trusted discovery snapshots with durable, idempotent model identities."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.identity import load_stable_ids

from menagerie.crawler.identity import (
    atomic_replace_bytes,
    canonical_json_bytes,
    hash_bytes,
    stable_hash,
)
from menagerie.crawler.models import JsonObject
from menagerie.crawler.routing import ModelRequirements, requirements_from_zoo_era


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
    preserved_legacy_flags:
        Deterministic audit-routing risks derived from the immutable legacy row.
    variant_scope, family_representative_id:
        Explicit family membership used for representative-first scheduling. A
        representative names its own stable ID; a size variant names another row.
    """

    stable_id: str
    name: str
    zoo: str
    variant: str
    discovery_source: str
    legacy_row_sha256: str
    preserved_legacy_flags: tuple[str, ...]
    variant_scope: str = "family"
    family_representative_id: Optional[str] = None
    era: Optional[str] = None
    packages: frozenset[str] = frozenset()
    exact_repository: bool = False
    legacy_torch: bool = False

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
            "preserved_legacy_flags": list(self.preserved_legacy_flags),
            "variant_scope": self.variant_scope,
            "family_representative_id": self.family_representative_id or self.stable_id,
            "era": self.era,
            "routing_requirements": {
                "packages": sorted(self.packages),
                "exact_repository": self.exact_repository,
                "legacy_torch": self.legacy_torch,
            },
        }

    def to_model_requirements(self, framework: str) -> ModelRequirements:
        """Return the complete deterministic router input for this intake row.

        Parameters
        ----------
        framework:
            Framework inferred from the immutable name and zoo.

        Returns
        -------
        ModelRequirements
            All intake-derived facts consumed by the router.
        """

        return ModelRequirements(
            stable_id=self.stable_id,
            framework=framework,
            packages=self.packages,
            exact_repository=self.exact_repository,
            legacy_torch=self.legacy_torch,
        )


def derive_legacy_risk_flags(
    row: Mapping[str, Any], *, discovery_source: str = ""
) -> tuple[str, ...]:
    """Derive authority-free audit risks from one immutable legacy row.

    Parameters
    ----------
    row:
        Complete untrusted legacy source row.
    discovery_source:
        Snapshot stream containing the row.

    Returns
    -------
    tuple[str, ...]
        Canonically sorted risk flags used only to require fresh verification.
    """

    flags = {str(flag) for flag in row.get("flags", []) if isinstance(flag, str)}
    recipe = row.get("recipe")
    recipe_type = recipe.get("type") if isinstance(recipe, Mapping) else None
    if recipe_type in {"statement", "expression", "exec-string"}:
        flags.add("legacy-opaque-recipe")
    if bool(row.get("quarantine")) or (
        isinstance(recipe, Mapping) and bool(recipe.get("quarantine"))
    ):
        flags.add("legacy-quarantined-recipe")

    claim_text = " ".join(
        str(row.get(field, "")) for field in ("notes", "verified", "verification_expectation")
    ).lower()
    if any(token in claim_text for token in ("verified", "trace", "runs", "forward")):
        flags.add("legacy-run-claim")
    if (
        "faithful" in claim_text
        or "reimplement" in claim_text
        or re.search(r"\bport(?:ed|ing)?\b", claim_text) is not None
    ):
        flags.add("legacy-fidelity-claim")

    recipe_text = ""
    if isinstance(recipe, Mapping):
        recipe_text = " ".join(
            str(recipe.get(field, "")) for field in ("code", "module", "symbol")
        ).lower()
    classic_text = " ".join(
        (
            discovery_source,
            str(row.get("zoo", "")),
            claim_text,
            recipe_text,
        )
    ).lower()
    if "classic" in classic_text:
        flags.add("legacy-classic-requires-fidelity-audit")

    audit_text = " ".join((*flags, claim_text)).lower()
    if "slop" in audit_text:
        flags.add("legacy-slop-requires-fidelity-audit")
    if not row.get("source_url"):
        flags.add("legacy-source-unresolved")
    if row.get("deferral") is not None:
        flags.add("legacy-deferred")
    return tuple(sorted(flags))


def legacy_requires_fidelity_audit(flags: Iterable[str]) -> bool:
    """Return whether legacy risks require a current five-way fidelity gate.

    Parameters
    ----------
    flags:
        Preserved authority-free legacy risk markers.

    Returns
    -------
    bool
        Whether the row belongs to a classic, faithful/fidelity, or slop audit class.
    """

    normalized = {str(flag).strip().lower() for flag in flags}
    return any(
        token in flag
        for flag in normalized
        for token in ("classic", "faithful", "fidelity", "slop")
    )


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


def _snapshot_id(snapshot_sha256: str) -> str:
    """Derive the canonical display identity from a full snapshot digest.

    Parameters
    ----------
    snapshot_sha256:
        Canonical prefixed SHA-256 digest of the snapshot basis.

    Returns
    -------
    str
        Canonical ``intake-<20 hex>`` identity.
    """

    return f"intake-{snapshot_sha256.removeprefix('sha256:')[:20]}"


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
            raw_era = row.get("era")
            if raw_era is not None and not isinstance(raw_era, str):
                raise IntakeError(f"{source} row {key[0]!r} has a non-string era")
            routing = requirements_from_zoo_era(key[1], raw_era)
            item = IntakeItem(
                stable_id=stable_id,
                name=key[0],
                zoo=key[1],
                variant=key[2],
                discovery_source=source,
                legacy_row_sha256=stable_hash(row),
                preserved_legacy_flags=derive_legacy_risk_flags(row, discovery_source=source),
                variant_scope=str(row.get("variant_scope", "family")),
                family_representative_id=str(row.get("family_representative_id") or stable_id),
                era=raw_era,
                packages=routing.packages,
                exact_repository=routing.exact_repository,
                legacy_torch=routing.legacy_torch,
            )
            existing = by_key.get(key)
            if existing is not None:
                if existing.legacy_row_sha256 != item.legacy_row_sha256:
                    raise IntakeError(f"conflicting discovery rows for natural key {key!r}")
                continue
            by_key[key] = item
            by_id[stable_id] = key
    for item in by_key.values():
        if item.variant_scope != "family":
            raise IntakeError(
                f"unsupported variant scope for {item.stable_id}: {item.variant_scope}"
            )
        if item.family_representative_id not in by_id:
            raise IntakeError(
                f"family representative {item.family_representative_id!r} for "
                f"{item.stable_id!r} is not in intake"
            )
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

    Only roster membership, the minimal natural key, and conservative audit-risk
    markers are promoted into intake. No legacy claim gains authority.

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
    snapshot_id = _snapshot_id(snapshot_sha256)
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
        atomic_replace_bytes(snapshot_root / "sources" / name, data)
    atomic_replace_bytes(snapshot_root / "items.jsonl", items_bytes)
    atomic_replace_bytes(manifest_path, manifest_bytes)
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
    source_inventory = manifest_value.get("sources")
    if not isinstance(source_inventory, dict) or not all(
        isinstance(name, str)
        and name
        and isinstance(digest, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None
        for name, digest in source_inventory.items()
    ):
        raise IntakeError("intake manifest source inventory is malformed")
    source_root = (snapshot_root / "sources").resolve()
    observed_names = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if observed_names != set(source_inventory):
        raise IntakeError(
            "intake source inventory differs from the immutable manifest: "
            f"missing={sorted(set(source_inventory) - observed_names)}, "
            f"extra={sorted(observed_names - set(source_inventory))}"
        )
    for name, expected_digest in source_inventory.items():
        relative = Path(name)
        if relative.is_absolute():
            raise IntakeError("intake source path must be relative")
        candidate_path = source_root / relative
        source_path = candidate_path.resolve()
        if not source_path.is_relative_to(source_root) or candidate_path.is_symlink():
            raise IntakeError("intake source path escapes its immutable snapshot")
        try:
            observed_digest = hash_bytes(source_path.read_bytes())
        except OSError as exc:
            raise IntakeError(f"intake source byte is unavailable: {name}") from exc
        if observed_digest != expected_digest:
            raise IntakeError(f"intake source digest changed: {name}")
    rows = _read_jsonl(snapshot_root / "items.jsonl")
    manifest_items = manifest_value.get("items")
    if not isinstance(manifest_items, list) or rows != manifest_items:
        raise IntakeError("intake items do not match the immutable manifest")
    source_rows: dict[str, dict[str, Mapping[str, Any]]] = {}
    for discovery_source, filename in (
        ("master_catalog", "master_catalog.jsonl"),
        ("deferred", "deferred.jsonl"),
    ):
        source_path = snapshot_root / "sources" / filename
        if not source_path.is_file():
            continue
        source_rows[discovery_source] = {
            stable_hash(source_row): source_row for source_row in _read_jsonl(source_path)
        }

    def bound_source_row(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        """Return the immutable legacy source row bound to one intake item."""

        discovery_source = str(row["discovery_source"])
        return source_rows.get(discovery_source, {}).get(str(row["legacy_row_sha256"]))

    def loaded_flags(row: Mapping[str, Any]) -> tuple[str, ...]:
        """Return stored flags or derive them from immutable snapshotted source bytes."""

        raw_flags = row.get("preserved_legacy_flags")
        if isinstance(raw_flags, list):
            return tuple(sorted(str(flag) for flag in raw_flags))
        discovery_source = str(row["discovery_source"])
        legacy_row = bound_source_row(row)
        if legacy_row is None:
            return derive_legacy_risk_flags({}, discovery_source=discovery_source)
        return derive_legacy_risk_flags(legacy_row, discovery_source=discovery_source)

    def loaded_routing(
        row: Mapping[str, Any],
    ) -> tuple[Optional[str], frozenset[str], bool, bool]:
        """Load stored routing facts or derive them from snapshotted source bytes."""

        legacy_row = bound_source_row(row)
        raw_era = row.get("era")
        if raw_era is None and legacy_row is not None:
            raw_era = legacy_row.get("era")
        if raw_era is not None and not isinstance(raw_era, str):
            raise IntakeError("intake routing era must be a string or null")
        raw_routing = row.get("routing_requirements")
        if raw_routing is None:
            derived = requirements_from_zoo_era(str(row["zoo"]), raw_era)
            return (
                raw_era,
                derived.packages,
                derived.exact_repository,
                derived.legacy_torch,
            )
        if not isinstance(raw_routing, Mapping):
            raise IntakeError("intake routing requirements must be an object")
        raw_packages = raw_routing.get("packages")
        exact_repository = raw_routing.get("exact_repository")
        legacy_torch = raw_routing.get("legacy_torch")
        if (
            not isinstance(raw_packages, list)
            or not all(isinstance(package, str) and package for package in raw_packages)
            or raw_packages != sorted(set(raw_packages))
            or not isinstance(exact_repository, bool)
            or not isinstance(legacy_torch, bool)
        ):
            raise IntakeError("intake routing requirements are malformed")
        return raw_era, frozenset(raw_packages), exact_repository, legacy_torch

    loaded_items: list[IntakeItem] = []
    for row in rows:
        era, packages, exact_repository, legacy_torch = loaded_routing(row)
        loaded_items.append(
            IntakeItem(
                stable_id=str(row["stable_id"]),
                name=str(row["name"]),
                zoo=str(row["zoo"]),
                variant=str(row["variant"]),
                discovery_source=str(row["discovery_source"]),
                legacy_row_sha256=str(row["legacy_row_sha256"]),
                preserved_legacy_flags=loaded_flags(row),
                variant_scope=str(row.get("variant_scope", "family")),
                family_representative_id=str(
                    row.get("family_representative_id") or row["stable_id"]
                ),
                era=era,
                packages=packages,
                exact_repository=exact_repository,
                legacy_torch=legacy_torch,
            )
        )
    items = tuple(loaded_items)
    loaded_ids = {item.stable_id for item in items}
    for item in items:
        if item.variant_scope != "family":
            raise IntakeError(
                f"unsupported variant scope for {item.stable_id}: {item.variant_scope}"
            )
        if item.family_representative_id not in loaded_ids:
            raise IntakeError(
                f"family representative {item.family_representative_id!r} for "
                f"{item.stable_id!r} is not in intake"
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
    if manifest_value.get("snapshot_id") != _snapshot_id(expected):
        raise IntakeError("intake snapshot_id is not derived from its immutable digest")
    return IntakeSnapshot(
        snapshot_id=str(manifest_value["snapshot_id"]),
        snapshot_sha256=expected,
        root=snapshot_root,
        items=items,
        created=False,
    )
