"""The real-model ARTIFACT REGISTRY (testing MEMO section 4; build rows A1/A2).

One committed table of every real-model artifact the suite may touch:
``artifact_registry.jsonl``, one JSON row per line. Tests CONSUME rows by
``artifact_id``; nothing in the test suite constructs an artifact reference
ad hoc (no bare ``from_pretrained`` ids, no unpinned downloads). The registry
is the single authority for revisions, per-blob SHA-256 digests, licenses,
venues, and the deferred-capability columns, and its file digest is the exact
CI cache key (no prefix fallback).

Band vocabulary (memo section 4.1):

- ``R0`` (``REAL_CLASS``): installed upstream class, locally constructed
  config/weights, zero network. NEVER counts as pretrained evidence --
  :meth:`Registry.checkpoint_evidence` refuses R0 rows typed.
- ``R1`` (``REAL_CHECKPOINT``): immutable released weights, revision- and
  digest-pinned.
- ``R2`` (``EXTENDED_REALITY``): weekly breadth/scale/wrapper rows.

Every spelling here is test-infrastructure, not product surface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REGISTRY_DIR = Path(__file__).resolve().parent
REGISTRY_PATH = REGISTRY_DIR / "artifact_registry.jsonl"
VENDORED_DIR = REGISTRY_DIR / "vendored"
NATURAL_INPUTS_DIR = REGISTRY_DIR / "natural_inputs"

BANDS = frozenset({"R0", "R1", "R2"})
KINDS = frozenset({"hf_hub", "torchvision_weights", "local_class"})
VENUES = frozenset({"pr", "pr_telemetry", "nightly", "canary", "weekly", "release"})
BYTE_PROVENANCE = frozenset({"measured", "api_blob_set", "estimate"})
PERMISSIVE_LICENSES = frozenset({"mit", "apache-2.0", "bsd-3-clause"})

# Cache-cap formula (memo 4.3, FABLE r3 ceiling adopted over the OPUS 2 GB
# dissent): hard cap = nightly-set bytes + 20%.
CACHE_CAP_HEADROOM = 1.2
# Bump when the on-disk cache layout produced by the preflight fetcher
# changes shape (not when rows change -- row changes move the digest).
CACHE_SCHEMA_VERSION = 1

_REQUIRED_FIELDS = (
    "artifact_id",
    "band",
    "kind",
    "model_id",
    "revision",
    "license",
    "redistributable",
    "blobs",
    "param_count",
    "measured_bytes",
    "byte_provenance",
    "measured_peak_rss_bytes",
    "resolved_config_fingerprint",
    "natural_input_ids",
    "venues",
    "gallery_ids",
    "trust_remote_code",
    "requires_cuda",
    "dtype",
    "quantization",
    "distributed_world_size",
    "promoted_from_menagerie",
    "satisfies_checkpoint_claim",
    "notes",
)


class RegistryError(Exception):
    """A malformed registry file or a misuse of the registry contract."""


class CheckpointClaimError(RegistryError):
    """An R0/local row was offered as real-checkpoint evidence (never-count)."""


@dataclass(frozen=True)
class BlobSpec:
    """One pinned file belonging to an artifact row."""

    filename: str
    sha256: str | None
    size_bytes: int | None
    lfs: bool


@dataclass(frozen=True)
class ArtifactRow:
    """One artifact registry row (schema: testing MEMO section 4.2)."""

    artifact_id: str
    band: str
    kind: str
    model_id: str
    revision: str | None
    license: str | None
    redistributable: bool
    blobs: tuple[BlobSpec, ...]
    param_count: int | None
    measured_bytes: int | None
    byte_provenance: str | None
    measured_peak_rss_bytes: int | None
    resolved_config_fingerprint: str | None
    natural_input_ids: tuple[str, ...]
    venues: tuple[str, ...]
    gallery_ids: tuple[str, ...]
    trust_remote_code: bool
    requires_cuda: bool
    dtype: str
    quantization: str | None
    distributed_world_size: int
    promoted_from_menagerie: bool
    satisfies_checkpoint_claim: bool
    notes: str

    def vendored_paths(self) -> tuple[Path, ...]:
        """Absolute paths of this row's vendored (in-repo) blobs."""

        if self.kind != "local_class":
            return ()
        return tuple(REGISTRY_DIR / blob.filename for blob in self.blobs)


@dataclass(frozen=True)
class Registry:
    """The loaded, validated artifact registry."""

    rows: tuple[ArtifactRow, ...]
    raw_bytes: bytes = field(repr=False)

    def get(self, artifact_id: str) -> ArtifactRow:
        """Return the row for ``artifact_id`` or raise :class:`RegistryError`."""

        for row in self.rows:
            if row.artifact_id == artifact_id:
                return row
        raise RegistryError(
            f"unknown artifact_id {artifact_id!r}; tests consume committed registry"
            " rows only (consumes, never constructs). Remedy: add the row to"
            f" {REGISTRY_PATH.name} with its revision and per-blob sha256."
        )

    def rows_for_venue(self, venue: str) -> tuple[ArtifactRow, ...]:
        """All rows whose ``venues`` include ``venue``."""

        if venue not in VENUES:
            raise RegistryError(f"unknown venue {venue!r}; venues are {sorted(VENUES)}")
        return tuple(row for row in self.rows if venue in row.venues)

    def checkpoint_evidence(self, artifact_id: str) -> ArtifactRow:
        """Return a row usable as REAL-CHECKPOINT evidence, refusing R0 typed.

        This is the never-count enforcement hook (memo executive summary):
        a random-init/config-built row is mechanically forbidden from ever
        satisfying a real-checkpoint claim.
        """

        row = self.get(artifact_id)
        if not row.satisfies_checkpoint_claim:
            raise CheckpointClaimError(
                f"artifact {artifact_id!r} is band {row.band} kind {row.kind}:"
                " config-built/random-init artifacts never satisfy a"
                " real-checkpoint claim. Remedy: cite an R1/R2 hub or"
                " torchvision-weights row instead."
            )
        return row

    def digest(self) -> str:
        """SHA-256 over the exact committed registry bytes.

        This digest IS the CI artifact-cache key material: exact key, no
        prefix fallback (memo 4.3). Any row edit moves the key.
        """

        return hashlib.sha256(self.raw_bytes).hexdigest()

    def cache_key(self) -> str:
        """The exact CI cache key for the artifact cache."""

        return f"torchlens-artifacts-v{CACHE_SCHEMA_VERSION}-{self.digest()[:32]}"

    def nightly_cap_bytes(self) -> int:
        """The byte-budget hard cap: nightly-set total + 20% (memo 4.3)."""

        nightly = self.rows_for_venue("nightly")
        total = 0
        for row in nightly:
            if row.measured_bytes is not None:
                total += row.measured_bytes
            else:
                total += sum(blob.size_bytes or 0 for blob in row.blobs)
        return int(total * CACHE_CAP_HEADROOM)


#: Fields popped from a resolved config before fingerprinting: pure
#: provenance stamps that HF's own ``to_dict()`` embeds in every config but
#: that never change what gets captured. ``transformers_version`` is the
#: one member today (FK-transformers, 2026-10-04): a side-by-side dump of
#: every R0 family under transformers 5.14.1 vs 5.18.0 showed it as the
#: ONLY config diff for 20 of 24 families, with n_ops, recipe
#: classification, and the false-claims manifest byte-identical -- the
#: fingerprint drifted on a release bump that changed nothing the sweep
#: measures. This is a closed, by-name list: a field is added here only
#: after the same kind of evidence (changing it alone never changes graph
#: shape), never speculatively or by category.
_FINGERPRINT_PROVENANCE_ONLY_FIELDS = frozenset({"transformers_version"})


def resolved_config_fingerprint(model: Any = None, config: Any = None) -> str:
    """Fingerprint the RESOLVED model configuration (memo D7 class 4).

    A standard HF flag changed the captured graph 12% via a silent
    ``use_cache`` mutation, so op-count goldens, node floors, and cost
    budgets key on ``(checkpoint, revision, resolved-config)`` -- never on
    the checkpoint name alone. The fingerprint folds in the runtime state
    that changes graph shape without touching ``config`` serialization:
    the attention implementation, gradient checkpointing, and train mode.

    The fingerprint describes the resolved MODEL, not the exact package
    release that built it: fields in ``_FINGERPRINT_PROVENANCE_ONLY_FIELDS``
    are popped before hashing because they carry build provenance (which
    transformers release stamped this config) rather than anything that
    changes the captured graph. The version itself is never lost -- it is
    recorded separately in the expectations' provenance docstring, not
    folded into the golden key a routine point release would otherwise
    invalidate.

    Parameters
    ----------
    model:
        A live model carrying ``.config`` (transformers-style). Optional if
        ``config`` is given.
    config:
        A config object with ``to_dict()`` or a plain mapping.

    Returns
    -------
    str
        Hex SHA-256 over the canonical resolved-config JSON.
    """

    if config is None:
        if model is None:
            raise RegistryError("resolved_config_fingerprint needs a model or a config")
        config = getattr(model, "config", None)
        if config is None:
            raise RegistryError(f"{type(model).__name__} carries no .config")
    config_dict = dict(config.to_dict()) if hasattr(config, "to_dict") else dict(config)
    for provenance_field in _FINGERPRINT_PROVENANCE_ONLY_FIELDS:
        config_dict.pop(provenance_field, None)
    resolved: dict[str, Any] = {"config": config_dict}
    if model is not None:
        resolved["class"] = type(model).__name__
        resolved["attn_implementation"] = getattr(
            getattr(model, "config", None), "_attn_implementation", None
        )
        resolved["gradient_checkpointing"] = bool(
            getattr(model, "is_gradient_checkpointing", False)
        )
        resolved["training"] = bool(getattr(model, "training", False))
    canonical = json.dumps(resolved, sort_keys=True, default=repr)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_row(payload: dict[str, Any], line_no: int) -> ArtifactRow:
    missing = [name for name in _REQUIRED_FIELDS if name not in payload]
    extra = [name for name in payload if name not in _REQUIRED_FIELDS]
    if missing or extra:
        raise RegistryError(
            f"registry line {line_no}: schema mismatch (missing={missing},"
            f" extra={extra}); the row schema is closed."
        )
    if payload["band"] not in BANDS:
        raise RegistryError(f"registry line {line_no}: unknown band {payload['band']!r}")
    if payload["kind"] not in KINDS:
        raise RegistryError(f"registry line {line_no}: unknown kind {payload['kind']!r}")
    bad_venues = set(payload["venues"]) - VENUES
    if bad_venues:
        raise RegistryError(f"registry line {line_no}: unknown venues {sorted(bad_venues)}")
    if payload["byte_provenance"] is not None and payload["byte_provenance"] not in BYTE_PROVENANCE:
        raise RegistryError(
            f"registry line {line_no}: unknown byte_provenance {payload['byte_provenance']!r}"
        )
    if payload["redistributable"] and (payload["license"] or "") not in PERMISSIVE_LICENSES:
        raise RegistryError(
            f"registry line {line_no}: redistributable=True requires a permissive"
            f" license from {sorted(PERMISSIVE_LICENSES)}, got {payload['license']!r};"
            " only permissive rows may enter the Release asset."
        )
    if payload["satisfies_checkpoint_claim"] and payload["band"] == "R0":
        raise RegistryError(
            f"registry line {line_no}: an R0 row may never satisfy a checkpoint"
            " claim (never-count enforcement)."
        )
    blobs = tuple(
        BlobSpec(
            filename=blob["filename"],
            sha256=blob["sha256"],
            size_bytes=blob["size_bytes"],
            lfs=blob["lfs"],
        )
        for blob in payload["blobs"]
    )
    return ArtifactRow(
        artifact_id=payload["artifact_id"],
        band=payload["band"],
        kind=payload["kind"],
        model_id=payload["model_id"],
        revision=payload["revision"],
        license=payload["license"],
        redistributable=payload["redistributable"],
        blobs=blobs,
        param_count=payload["param_count"],
        measured_bytes=payload["measured_bytes"],
        byte_provenance=payload["byte_provenance"],
        measured_peak_rss_bytes=payload["measured_peak_rss_bytes"],
        resolved_config_fingerprint=payload["resolved_config_fingerprint"],
        natural_input_ids=tuple(payload["natural_input_ids"]),
        venues=tuple(payload["venues"]),
        gallery_ids=tuple(payload["gallery_ids"]),
        trust_remote_code=payload["trust_remote_code"],
        requires_cuda=payload["requires_cuda"],
        dtype=payload["dtype"],
        quantization=payload["quantization"],
        distributed_world_size=payload["distributed_world_size"],
        promoted_from_menagerie=payload["promoted_from_menagerie"],
        satisfies_checkpoint_claim=payload["satisfies_checkpoint_claim"],
        notes=payload["notes"],
    )


def load_registry(path: Path | None = None) -> Registry:
    """Load and validate the committed artifact registry."""

    registry_path = REGISTRY_PATH if path is None else path
    raw = registry_path.read_bytes()
    rows: list[ArtifactRow] = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        row = _parse_row(payload, line_no)
        if row.artifact_id in seen:
            raise RegistryError(f"duplicate artifact_id {row.artifact_id!r}")
        seen.add(row.artifact_id)
        rows.append(row)
    return Registry(rows=tuple(rows), raw_bytes=raw)
