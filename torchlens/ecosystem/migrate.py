"""``tl.migrate`` v1: closed, transactional, upgrade-only (MEMO 3.6, B2).

Accepts governed v6-current unified ``.tlspec`` directories (Trace and
Bundle products share the unified envelope). Adjacent-step, deterministic,
idempotent. It NEVER executes a model, replay, network action, entry point,
provider, or foreign callable, and never implements migration as
``tl.load(); tl.save()`` -- every v1 step is MANIFEST-LEVEL (stamp +
provenance + re-hash) because migration only ever creates the ABSENT-field
direction, which the reader tolerates by declared default-fill (the memo's
sizing argument; proven live by the deciding real-artifact test over the
harvested v2.31.0/v2.33.0/v2.34.1 goldens).

Mechanics: sibling staging with a PARTIAL marker, fsync, atomic publish via
rename with a retained backup, restore-on-failure, and stranded/concurrent
writer refusal BEFORE any mutation. Provenance is mandatory and append-only:
the migration witness sidecar (``tl_migration_provenance.json``) records the
origin identity and digests, ordered step IDs with their declared level and
absent fact families, tool identity, and the final manifest digest. Writer
identity is never forged -- the manifest keeps the original
``torchlens_version`` and the pair-consistency gate accepts the migrated
(writer, stamp) pair through the witness (``torchlens._io.compat_ledger``).
Facts are never backfilled; replay attestations become UNAVAILABLE by
construction (the envelope stamp changes nothing the attestation layer
reads, and the disclosure records it); migrated and native artifacts remain
distinguishable by the witness itself.

Out of v1 (typed refusals, MEMO 3.6): pre-floor migration (including the
genuine v2.16 ModelLog format), downgrade, repair, cross-backend
conversion, and third-party steps. Every spelling is DOCUMENTED-UNSTABLE
pending naming-session ratification.
"""

from __future__ import annotations

import datetime as _datetime
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._io import MIN_TLSPEC_VERSION, TLSPEC_VERSION, TorchLensIOError, above_ceiling_error
from .._io._json import read_bounded
from .._io.compat_ledger import (
    MIGRATION_WITNESS_FILENAME,
    MIGRATION_WITNESS_SCHEMA,
    bridge_reader_remedy,
    pair_is_governed,
    read_migration_witness,
    ungoverned_pair_error,
)

__tl_layer__ = "L8"

_STAGING_SUFFIX = ".tl-migrate-staging"
_BACKUP_SUFFIX = ".pre-migrate"
_PARTIAL_MARKER = "TL_MIGRATE_PARTIAL"

#: Fact families ABSENT from artifacts older than each target stamp
#: (disclosure data for the witness; the reader default-fills them at load).
#: Derived from the version history of record in
#: ``torchlens/_io/format_contract.py``.
_ABSENT_FAMILIES_BY_TARGET: dict[int, tuple[str, ...]] = {
    7: ("capture_outcome",),
    8: (
        "site_key",
        "grouping_policy",
        "edge_audit_families",
        "distributed_scope",
        "grad_fn_timing_provenance",
        "checkpoint_invocation_witness",
        "structure_only_marker",
        "primitive_op_profile",
        "bundle_member_relations",
        "episode_annotations",
    ),
    9: (
        "intervention_event_rows",
        "sidecar_annotations",
        "injection_provenance",
        "source_snapshots",
        "structure_evidence",
    ),
}


@dataclass(frozen=True)
class MigrationStep:
    """One declared adjacent migration step (MEMO 3.6 step protocol).

    Parameters
    ----------
    step_id:
        Stable step identifier (``"tlspec_6_to_7"``).
    level:
        ``"manifest"`` for every v1 step. The protocol is state-capable by
        declaration: a future STATE-LEVEL step names ``"state"`` here and
        uses only the restricted data vocabulary -- the guard test pins that
        no v1 step quietly claims state level.
    source_tlspec:
        Source schema stamp.
    target_tlspec:
        Target schema stamp (always ``source + 1``).
    source_contract_digest:
        Normative writer-contract digest of the source grammar when
        captured; ``None`` for historical grammars that predate the digest
        regime (honest absence, never backfilled).
    target_contract_digest:
        Always ``None`` in v1: a manifest-level restamp does NOT adopt the
        target writer's persisted grammar (the body stays the original
        writer's; absent families default-fill at load), so claiming the
        target contract digest would be false.
    absent_fact_families:
        Fact families the target schema declares that this artifact does
        not carry (disclosure, never backfilled).
    """

    step_id: str
    level: str
    source_tlspec: int
    target_tlspec: int
    source_contract_digest: str | None
    target_contract_digest: str | None
    absent_fact_families: tuple[str, ...]


@dataclass(frozen=True)
class MigrationReport:
    """Outcome report for one :func:`migrate` call.

    Parameters
    ----------
    status:
        ``"migrated"`` or ``"already-current"`` (idempotent no-op).
    path:
        The artifact directory (post-publish on success).
    backup_path:
        Retained pre-migration backup directory, or ``None`` for a no-op.
    source_tlspec:
        Stamp observed before migration.
    final_tlspec:
        Stamp after migration (the runtime schema on success).
    steps:
        The declared steps applied, in order (empty for a no-op).
    witness_path:
        The migration witness sidecar path, or ``None`` for a no-op on a
        never-migrated artifact.
    """

    status: str
    path: Path
    backup_path: Path | None
    source_tlspec: int
    final_tlspec: int
    steps: tuple[MigrationStep, ...]
    witness_path: Path | None


def _sha256_of_path(path: Path) -> str:
    """SHA-256 hex digest of one file's bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fsync_file(path: Path) -> None:
    """fsync one file's contents to disk."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _classify_source(path: Path) -> dict[str, Any]:
    """Classify and admit one migration source directory.

    Parameters
    ----------
    path:
        Artifact directory to migrate.

    Returns
    -------
    dict[str, Any]
        The parsed manifest.

    Raises
    ------
    TorchLensIOError
        Typed refusals for every out-of-scope source: not a governed
        unified artifact (``migration_source_unsupported``), a stranded or
        concurrent migration (``migration_concurrent_writer``), an
        above-runtime stamp, an ungoverned producer pair, or an invalid
        witness.
    """

    from ..io import detect_tlspec_format

    if not path.is_dir():
        raise TorchLensIOError(
            f"Migration source {path} is not a .tlspec directory. Remedy: pass "
            "the artifact directory written by tl.save.",
            code="migration_source_unsupported",
            remedy="pass the artifact directory written by tl.save",
            path=str(path),
        )
    staging = path.with_name(path.name + _STAGING_SUFFIX)
    if staging.exists():
        raise TorchLensIOError(
            f"Migration staging directory {staging} already exists: a prior "
            "migration was interrupted or another writer is migrating this "
            "artifact now. Remedy: verify no other process is migrating, then "
            "delete the staging directory and re-run tl.migrate.",
            code="migration_concurrent_writer",
            remedy="verify no concurrent migration, delete the staging directory, re-run",
            path=str(staging),
        )
    fmt = detect_tlspec_format(path)
    if fmt != "v2.0_unified":
        remedy = (
            bridge_reader_remedy("modellog", writer_release="2.16.0")
            if fmt == "v2.16_modellog_portable"
            else "only governed unified .tlspec directories (tlspec_version >= 6) migrate in v1"
        )
        raise TorchLensIOError(
            f"Migration source {path} has format {fmt!r}, outside tl.migrate v1 "
            f"scope (governed unified artifacts only; pre-floor migration is out "
            f"of v1 by design). Remedy: {remedy}.",
            code="migration_source_unsupported",
            remedy=remedy,
            path=str(path),
            detected_format=fmt,
        )
    manifest = read_bounded(path / "manifest.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tlspec_version"), int):
        raise TorchLensIOError(
            f"Migration source {path} has no integer tlspec_version in its "
            "manifest. Remedy: the artifact is corrupt; restore it from its "
            "source or backup.",
            code="migration_source_unsupported",
            remedy="the artifact is corrupt; restore it from its source or backup",
            path=str(path),
        )
    return manifest


def _admit_stamps(path: Path, manifest: dict[str, Any]) -> int:
    """Admit the source stamp and producer pair; return the source stamp."""

    source_tlspec = manifest["tlspec_version"]
    if source_tlspec > TLSPEC_VERSION:
        raise above_ceiling_error(
            observed=source_tlspec,
            subject="Migration source",
            code="artifact_version_above_runtime",
        )
    if source_tlspec < MIN_TLSPEC_VERSION:
        raise TorchLensIOError(
            f"Migration source {path} has tlspec_version={source_tlspec}, below "
            f"the rehydration floor {MIN_TLSPEC_VERSION}; pre-floor migration is "
            "out of tl.migrate v1 by design (MEMO 3.6). Remedy: "
            f"{bridge_reader_remedy('trace')}.",
            code="migration_source_unsupported",
            remedy=bridge_reader_remedy("trace"),
            path=str(path),
        )
    writer = manifest.get("torchlens_version", "")
    witness = read_migration_witness(path)
    if witness is None and not pair_is_governed(str(writer), source_tlspec):
        raise ungoverned_pair_error(
            str(writer),
            source_tlspec,
            subject="Migration source",
            code="artifact_producer_pair_ungoverned",
        )
    return source_tlspec


def _plan_steps(source_tlspec: int) -> tuple[MigrationStep, ...]:
    """Plan the adjacent manifest-level steps from ``source_tlspec`` to current."""

    steps = []
    for version in range(source_tlspec, TLSPEC_VERSION):
        steps.append(
            MigrationStep(
                step_id=f"tlspec_{version}_to_{version + 1}",
                level="manifest",
                source_tlspec=version,
                target_tlspec=version + 1,
                source_contract_digest=None,
                target_contract_digest=None,
                absent_fact_families=_ABSENT_FAMILIES_BY_TARGET.get(version + 1, ()),
            )
        )
    return tuple(steps)


def _step_payload(step: MigrationStep) -> dict[str, Any]:
    """Serialize one step for the witness sidecar."""

    return {
        "step_id": step.step_id,
        "level": step.level,
        "source_tlspec": step.source_tlspec,
        "target_tlspec": step.target_tlspec,
        "source_contract_digest": step.source_contract_digest,
        "target_contract_digest": step.target_contract_digest,
        "absent_fact_families": list(step.absent_fact_families),
    }


def _build_witness(
    *,
    manifest: dict[str, Any],
    original_manifest_sha256: str,
    prior_witness: dict[str, Any] | None,
    steps: tuple[MigrationStep, ...],
) -> dict[str, Any]:
    """Build the append-only witness payload (origin preserved across runs)."""

    from .. import __version__
    from .._io.writer_contract import writer_contract_digest

    if prior_witness is not None:
        origin = dict(prior_witness["origin"])
        prior_steps = list(prior_witness.get("steps", []))
    else:
        origin = {
            "torchlens_version": manifest.get("torchlens_version"),
            "tlspec_version": manifest["tlspec_version"],
            "manifest_sha256": original_manifest_sha256,
        }
        prior_steps = []
    return {
        "witness_schema": MIGRATION_WITNESS_SCHEMA,
        "origin": origin,
        "steps": prior_steps + [_step_payload(step) for step in steps],
        "tool": {
            "torchlens_version": __version__,
            "writer_contract_digest": writer_contract_digest(),
        },
        "migrated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "final_tlspec_version": TLSPEC_VERSION,
        "disclosures": {
            "replay_attestations": "unavailable",
            "facts_backfilled": "none",
            "level": "manifest",
        },
    }


def _stage_and_publish(
    path: Path,
    *,
    witness: dict[str, Any],
) -> tuple[Path, Path]:
    """Stage the migrated artifact beside the source and publish atomically.

    Parameters
    ----------
    path:
        Source artifact directory.
    witness:
        Witness payload; its ``final_manifest_sha256`` is filled here from
        the staged manifest bytes.

    Returns
    -------
    tuple[Path, Path]
        ``(backup_path, witness_path)`` after a successful publish.

    Raises
    ------
    TorchLensIOError
        With ``code="migration_staging_failed"`` on any filesystem failure;
        the source artifact is restored (or was never touched).
    """

    staging = path.with_name(path.name + _STAGING_SUFFIX)
    backup = path.with_name(path.name + _BACKUP_SUFFIX)
    if backup.exists():
        raise TorchLensIOError(
            f"Migration backup directory {backup} already exists from a prior "
            "run. Remedy: inspect and remove (or rename) the retained backup, "
            "then re-run tl.migrate.",
            code="migration_concurrent_writer",
            remedy="inspect and remove the retained backup, then re-run",
            path=str(backup),
        )
    try:
        shutil.copytree(path, staging, symlinks=False)
        marker = staging / _PARTIAL_MARKER
        marker.write_text("migration in progress\n", encoding="utf-8")
        manifest_path = staging / "manifest.json"
        manifest = read_bounded(manifest_path)
        manifest["tlspec_version"] = TLSPEC_VERSION
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        _fsync_file(manifest_path)
        witness["final_manifest_sha256"] = _sha256_of_path(manifest_path)
        witness_path = staging / MIGRATION_WITNESS_FILENAME
        witness_path.write_text(json.dumps(witness, indent=2, sort_keys=True), encoding="utf-8")
        _fsync_file(witness_path)
        marker.unlink()
    except (OSError, TorchLensIOError, json.JSONDecodeError) as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise TorchLensIOError(
            f"Migration staging for {path} failed before publish: {exc}. The "
            "source artifact was never modified. Remedy: resolve the "
            "filesystem failure and re-run tl.migrate.",
            code="migration_staging_failed",
            remedy="resolve the filesystem failure and re-run tl.migrate",
            path=str(path),
        ) from exc
    try:
        os.rename(path, backup)
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise TorchLensIOError(
            f"Migration publish for {path} could not move the source aside: "
            f"{exc}. The source artifact is unchanged. Remedy: resolve the "
            "filesystem failure and re-run tl.migrate.",
            code="migration_staging_failed",
            remedy="resolve the filesystem failure and re-run tl.migrate",
            path=str(path),
        ) from exc
    try:
        os.rename(staging, path)
    except OSError as exc:
        os.rename(backup, path)  # restore-on-failure: the source comes back
        shutil.rmtree(staging, ignore_errors=True)
        raise TorchLensIOError(
            f"Migration publish for {path} failed at the final rename: {exc}. "
            "The source artifact was restored in place. Remedy: resolve the "
            "filesystem failure and re-run tl.migrate.",
            code="migration_staging_failed",
            remedy="resolve the filesystem failure and re-run tl.migrate",
            path=str(path),
        ) from exc
    return backup, path / MIGRATION_WITNESS_FILENAME


def migrate(path: str | Path) -> MigrationReport:
    """Upgrade one governed ``.tlspec`` directory to the current schema.

    Adjacent-step, upgrade-only, deterministic, idempotent, transactional.
    Executes nothing: no model, replay, network action, entry point,
    provider, or foreign callable, and never ``tl.load(); tl.save()``. The
    pre-migration artifact is retained beside the result as
    ``<name>.pre-migrate``.

    Parameters
    ----------
    path:
        Unified ``.tlspec`` artifact directory (Trace or Bundle product).

    Returns
    -------
    MigrationReport
        ``status="already-current"`` when the artifact is at the runtime
        schema (no writes); ``status="migrated"`` after an atomic publish.

    Raises
    ------
    TorchLensIOError
        Typed refusals with stable codes: ``migration_source_unsupported``
        (non-unified/pre-floor/corrupt sources), ``migration_concurrent_writer``
        (stranded staging or retained backup in the way),
        ``migration_staging_failed`` (filesystem failure; source restored),
        ``migration_witness_invalid`` (broken provenance sidecar),
        ``artifact_producer_pair_ungoverned`` (unlawful writer/stamp pair),
        and ``artifact_version_above_runtime`` (downgrade refused).
    """

    artifact_path = Path(path)
    manifest = _classify_source(artifact_path)
    source_tlspec = _admit_stamps(artifact_path, manifest)
    if source_tlspec == TLSPEC_VERSION:
        return MigrationReport(
            status="already-current",
            path=artifact_path,
            backup_path=None,
            source_tlspec=source_tlspec,
            final_tlspec=source_tlspec,
            steps=(),
            witness_path=(
                artifact_path / MIGRATION_WITNESS_FILENAME
                if (artifact_path / MIGRATION_WITNESS_FILENAME).exists()
                else None
            ),
        )
    steps = _plan_steps(source_tlspec)
    witness = _build_witness(
        manifest=manifest,
        original_manifest_sha256=_sha256_of_path(artifact_path / "manifest.json"),
        prior_witness=read_migration_witness(artifact_path),
        steps=steps,
    )
    backup, witness_path = _stage_and_publish(artifact_path, witness=witness)
    return MigrationReport(
        status="migrated",
        path=artifact_path,
        backup_path=backup,
        source_tlspec=source_tlspec,
        final_tlspec=TLSPEC_VERSION,
        steps=steps,
        witness_path=witness_path,
    )
