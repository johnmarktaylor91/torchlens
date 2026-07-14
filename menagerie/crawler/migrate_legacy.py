"""Total-authority legacy migration into untriaged, hash-only hints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from menagerie.crawler.identity import hash_bytes, stable_hash
from menagerie.crawler.intake import IntakeItem, IntakeSnapshot
from menagerie.crawler.models import JsonObject


@dataclass(frozen=True)
class LegacyModuleHint:
    """Hash-only classic-module association.

    Parameters
    ----------
    natural_key:
        Name, zoo, and variant belonging to the classic entry.
    path:
        Local module path used only while computing its hash.
    risk_flags:
        Existing audit flags associated with the module.
    """

    natural_key: tuple[str, str, str]
    path: Path
    risk_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class LegacyMigration:
    """One untriaged legacy hint with no inherited authority.

    Parameters
    ----------
    stable_id:
        Durable intake identity.
    snapshot_id, snapshot_sha256:
        Trusted roster snapshot parent.
    legacy_row_sha256, legacy_recipe_sha256, legacy_module_sha256, legacy_notes_sha256:
        Hashes of old bytes or values; legacy contents are not copied.
    preserved_legacy_flags:
        Risk markers retained for later audit routing.
    discovery_sources:
        Streams in which this roster member was observed.
    """

    stable_id: str
    snapshot_id: str
    snapshot_sha256: str
    legacy_row_sha256: Optional[str]
    legacy_recipe_sha256: Optional[str]
    legacy_module_sha256: Optional[str]
    legacy_notes_sha256: Optional[str]
    preserved_legacy_flags: tuple[str, ...]
    discovery_sources: tuple[str, ...]

    def to_dict(self) -> JsonObject:
        """Return the canonical untriaged hint payload.

        Returns
        -------
        dict[str, Any]
            JSON-compatible migration record.
        """

        return {
            "stable_id": self.stable_id,
            "workflow_state": "UNTRIAGED",
            "intake": {
                "snapshot_id": self.snapshot_id,
                "snapshot_sha256": self.snapshot_sha256,
                "legacy_row_sha256": self.legacy_row_sha256,
                "legacy_recipe_sha256": self.legacy_recipe_sha256,
                "legacy_module_sha256": self.legacy_module_sha256,
                "legacy_notes_sha256": self.legacy_notes_sha256,
                "legacy_claims_untrusted": True,
                "preserved_legacy_flags": list(self.preserved_legacy_flags),
                "discovery_sources": list(self.discovery_sources),
            },
            "inherited_claims": {
                "runs": False,
                "rung": None,
                "source_verified": False,
                "fidelity": None,
                "recipe_accepted": False,
            },
        }


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return a legacy row's natural key.

    Parameters
    ----------
    row:
        Legacy catalog row.

    Returns
    -------
    tuple[str, str, str]
        Name, zoo, and variant.
    """

    return (str(row.get("name", "")), str(row.get("zoo", "")), str(row.get("variant", "")))


def _legacy_flags(row: Mapping[str, Any]) -> set[str]:
    """Derive and preserve risk flags without trusting legacy claims.

    Parameters
    ----------
    row:
        Complete untrusted legacy row.

    Returns
    -------
    set[str]
        Audit-routing flags only.
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
    if any(token in claim_text for token in ("faithful", "reimplement", "port")):
        flags.add("legacy-fidelity-claim")
    if not row.get("source_url"):
        flags.add("legacy-source-unresolved")
    if row.get("deferral") is not None:
        flags.add("legacy-deferred")
    return flags


def _merge_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    """Index legacy rows while rejecting conflicting duplicate identities.

    Parameters
    ----------
    rows:
        Legacy catalog, deferred, and discovery-only rows.

    Returns
    -------
    dict[tuple[str, str, str], Mapping[str, Any]]
        Natural-key row index.

    Raises
    ------
    ValueError
        If one natural key has different legacy row bytes.
    """

    indexed: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        prior = indexed.get(key)
        if prior is not None and stable_hash(prior) != stable_hash(row):
            raise ValueError(f"conflicting legacy rows for {key!r}")
        indexed[key] = row
    return indexed


def _module_index(
    modules: Sequence[LegacyModuleHint],
) -> dict[tuple[str, str, str], LegacyModuleHint]:
    """Build a unique classic-module index.

    Parameters
    ----------
    modules:
        Classic module hints.

    Returns
    -------
    dict[tuple[str, str, str], LegacyModuleHint]
        Natural-key module mapping.
    """

    indexed: dict[tuple[str, str, str], LegacyModuleHint] = {}
    for module in modules:
        if module.natural_key in indexed:
            raise ValueError(f"duplicate classic module hint for {module.natural_key!r}")
        indexed[module.natural_key] = module
    return indexed


def _migrate_item(
    item: IntakeItem,
    snapshot: IntakeSnapshot,
    row: Optional[Mapping[str, Any]],
    module: Optional[LegacyModuleHint],
) -> LegacyMigration:
    """Convert one intake item into an authority-free migration hint.

    Parameters
    ----------
    item:
        Trusted roster identity.
    snapshot:
        Parent intake snapshot.
    row:
        Optional complete legacy row used only for hashes and flags.
    module:
        Optional classic module used only for its byte hash and flags.

    Returns
    -------
    LegacyMigration
        Untriaged hash-only hint.
    """

    flags: set[str] = set()
    legacy_row_sha256: Optional[str] = item.legacy_row_sha256
    recipe_sha256: Optional[str] = None
    notes_sha256: Optional[str] = None
    if row is not None:
        legacy_row_sha256 = stable_hash(row)
        flags.update(_legacy_flags(row))
        if "recipe" in row:
            recipe_sha256 = stable_hash(row["recipe"])
        if "notes" in row:
            notes_sha256 = stable_hash(row["notes"])
    module_sha256: Optional[str] = None
    if module is not None:
        module_sha256 = hash_bytes(module.path.read_bytes())
        flags.update(module.risk_flags)
        flags.add("legacy-classic-requires-fidelity-audit")
    if row is None and module is None:
        flags.add("discovery-only")
    return LegacyMigration(
        stable_id=item.stable_id,
        snapshot_id=snapshot.snapshot_id,
        snapshot_sha256=snapshot.snapshot_sha256,
        legacy_row_sha256=legacy_row_sha256,
        legacy_recipe_sha256=recipe_sha256,
        legacy_module_sha256=module_sha256,
        legacy_notes_sha256=notes_sha256,
        preserved_legacy_flags=tuple(sorted(flags)),
        discovery_sources=(item.discovery_source,),
    )


def migrate_legacy(
    snapshot: IntakeSnapshot,
    legacy_rows: Iterable[Mapping[str, Any]],
    *,
    classic_modules: Sequence[LegacyModuleHint] = (),
) -> tuple[LegacyMigration, ...]:
    """Migrate every intake ID as an untriaged, hash-only legacy hint.

    No inherited recipe, run, rung, source, or fidelity claim is accepted. The
    returned rows deliberately contain explicit false/null authority fields so
    downstream code cannot accidentally grandfather old results.

    Parameters
    ----------
    snapshot:
        Trusted, stable-ID-assigned roster snapshot.
    legacy_rows:
        Complete legacy catalog/deferred/discovery rows.
    classic_modules:
        Classic source modules to hash and flag.

    Returns
    -------
    tuple[LegacyMigration, ...]
        One untriaged hint for every intake stable ID.

    Raises
    ------
    ValueError
        If a supplied legacy row or classic is outside intake.
    """

    rows_by_key = _merge_rows(legacy_rows)
    modules_by_key = _module_index(classic_modules)
    intake_keys = {item.natural_key for item in snapshot.items}
    outside = (set(rows_by_key) | set(modules_by_key)) - intake_keys
    if outside:
        raise ValueError(f"legacy hints exist outside intake: {sorted(outside)!r}")
    return tuple(
        _migrate_item(
            item,
            snapshot,
            rows_by_key.get(item.natural_key),
            modules_by_key.get(item.natural_key),
        )
        for item in snapshot.items
    )
