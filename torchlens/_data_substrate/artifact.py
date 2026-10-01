"""Extraction ARTIFACT v2: bounded manifest + append-only fsynced ledger (extract D1).

Layout: ``manifest.json`` (bounded header: schema/tool versions, status, the
run signature, per-key logical metadata, terminal totals, ledger digest) +
``ledger.jsonl`` (append-only, ONE fsynced line per shard: index, global row
range, path, byte size, integrity facts, per-key physical shapes/dtypes,
ID-range digest) + a write-once ordered ``stimulus_ids.json`` sidecar +
immutable batch-major shards.

Commit order (the protocol): validate -> temp shard -> flush/fsync -> atomic
rename -> append/fsync ledger row. Torn final ledger lines and unledgered
orphans are never trusted; a missing or corrupt middle member ends the
trusted prefix TYPED; readers consume ledger order, never filename order.
The manifest is written at creation, once when batch zero freezes the plan,
and at terminal status — never per shard (whole-manifest rewriting was the
measured quadratic: 44 s / 597 MB at 4,000 shards vs a flat append).

Signature compare is FIELD BY FIELD against the KNOWN-FIELDS list (extract
D16): mismatch refuses typed naming the exact fields; missing semantic
fields never degrade to warnings.

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import hashlib
import json
import os
import zlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from .._errors import _actionable_message, _ActionableErrorMixin
from .._io import _json
from .._io._durability import fsync_dir, fsync_file
from ..errors._base import ConfigurationError

__tl_layer__ = "L3"

__all__ = [
    "LEDGER_FILENAME",
    "MANIFEST_SCHEMA_V1",
    "MANIFEST_SCHEMA_V2",
    "SIGNATURE_SEMANTIC_FIELDS",
    "STIMULUS_IDS_FILENAME",
    "UNRECORDED_V1",
    "ExtractionArtifactError",
    "ArtifactWriter",
    "compare_model_identity",
    "compare_signatures",
    "migrate_v1_artifact",
    "read_trusted_rows",
    "repair_ledger_tail",
    "shard_filename",
    "stimulus_ids_digest",
]

#: Legacy manifest schema id (whole-manifest ledger; superseded).
MANIFEST_SCHEMA_V1 = "tl_extract_manifest_v1"

#: Artifact v2 manifest schema id (bounded header + external ledger).
MANIFEST_SCHEMA_V2 = "tl_extract_manifest_v2"

#: Append-only per-shard ledger filename.
LEDGER_FILENAME = "ledger.jsonl"

#: Write-once ordered stimulus-identity sidecar filename.
STIMULUS_IDS_FILENAME = "stimulus_ids.json"


def shard_filename(index: int, extension: str = ".pt") -> str:
    """Return the canonical shard filename for a batch index.

    The artifact owns its layout: shard names are always derived from the
    shard index (resume validation refuses any drift between the two), so
    commit takes only the index and derives the name here. The extension
    follows the artifact's recorded shard format (a MANIFEST field, so a
    format change is a value change, never a layout break).

    Parameters
    ----------
    index:
        Zero-based shard index.
    extension:
        Shard filename extension (``".pt"`` or ``".safetensors"``).

    Returns
    -------
    str
        Filename of the form ``batch_00042.pt`` / ``batch_00042.safetensors``.
    """

    return f"batch_{index:05d}{extension}"


#: Sidecar schema id.
_STIMULUS_IDS_SCHEMA = "tl_extract_stimulus_ids_v1"

#: The D16 KNOWN-FIELDS list: every signature field compared individually on
#: resume. Every entry is SEMANTIC — a missing one refuses, never warns.
SIGNATURE_SEMANTIC_FIELDS: tuple[str, ...] = (
    "schema_version",
    "native_format",
    "layer_plan",
    "layers_kind",
    "batch_size",
    "transform_pipeline",
    "stimuli",
    "stimulus_ids_digest",
    "model_identity",
    "padding_side",
    "position_ids_source",
    "model_mode",
    "pool",
    "dtype_policy",
    "ragged",
    "integrity",
    # F18 extraction-runtime fields (extract D16's remaining KNOWN-FIELDS;
    # absent on pre-F18 v2 artifacts, where a missing semantic field
    # refuses — the fail-closed direction D16 mandates).
    "collate",
    "callable_identity",
    "selector_plan",
)

#: Sentinel marking a field a completed-v1 migration could not prove.
UNRECORDED_V1 = "unrecorded_v1"


class ExtractionArtifactError(_ActionableErrorMixin, ConfigurationError, RuntimeError):
    """Raised when an extraction artifact cannot be safely written, read, or resumed."""

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize an actionable artifact refusal.

        Parameters
        ----------
        problem:
            Description of the artifact state and why it was rejected.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


def _canonical_dumps(value: Any) -> str:
    """Serialize one JSON value canonically (sorted keys, minimal separators).

    Parameters
    ----------
    value:
        JSON-portable value.

    Returns
    -------
    str
        Canonical JSON string.
    """

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _atomic_write_json_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON document with file + directory fsync.

    Parameters
    ----------
    path:
        Destination path.
    payload:
        JSON-portable document.
    """

    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=False), encoding="utf-8")
    fsync_file(tmp)
    os.replace(tmp, path)
    fsync_dir(path.parent)


def stimulus_ids_digest(ids: list[str] | None) -> str | None:
    """Digest an ordered stimulus-id list for the signature (extract D2).

    Parameters
    ----------
    ids:
        Ordered identifiers, or ``None`` when none were supplied.

    Returns
    -------
    str | None
        ``"sha256:..."`` over the canonical JSON id array, or ``None``.
    """

    if ids is None:
        return None
    return "sha256:" + hashlib.sha256(_canonical_dumps(list(ids)).encode()).hexdigest()


def _crc32_file(path: Path) -> int:
    """Stream a file's CRC-32 (the D7 'fast' accidental-corruption fact).

    Parameters
    ----------
    path:
        File to checksum.

    Returns
    -------
    int
        CRC-32 over the FINAL FILE BYTES.
    """

    crc = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return crc
            crc = zlib.crc32(chunk, crc)


def compare_signatures(existing: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Compare two v2 signatures FIELD BY FIELD (extract D16).

    Parameters
    ----------
    existing:
        The artifact's recorded signature.
    current:
        The signature of the run asking to resume.

    Returns
    -------
    list[str]
        Names of every mismatched or missing semantic field, in KNOWN-FIELDS
        order (empty = compatible). ``model_identity`` is compared by its own
        level rules via :func:`compare_model_identity` and reported under its
        field name; fields a completed-v1 migration marked
        :data:`UNRECORDED_V1` are skipped (disclosed, not comparable).
    """

    mismatched: list[str] = []
    for field_name in SIGNATURE_SEMANTIC_FIELDS:
        recorded = existing.get(field_name, "__MISSING__")
        requested = current.get(field_name, "__MISSING__")
        if recorded == UNRECORDED_V1 or (
            isinstance(recorded, Mapping) and recorded.get("level") == UNRECORDED_V1
        ):
            continue
        if field_name == "model_identity":
            if compare_model_identity(recorded, requested):
                mismatched.append(field_name)
            continue
        if _canonical_dumps(recorded) != _canonical_dumps(requested):
            mismatched.append(field_name)
    return mismatched


def compare_model_identity(recorded: Any, current: Any) -> str | None:
    """Compare two model-identity records at the recorded level (extract D6).

    Parameters
    ----------
    recorded:
        The artifact's identity record.
    current:
        The resuming run's identity record.

    Returns
    -------
    str | None
        ``None`` when compatible; otherwise one of ``"level"`` (cross-level
        comparison refused), ``"digest"`` (measured digests differ — the
        T-MODELSWAP case), ``"assertion"`` (asserted claims differ), or
        ``"unavailable"`` (an unavailable identity cannot verify a resume).
    """

    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping):
        return "level"
    recorded_level = recorded.get("level")
    current_level = current.get("level")
    if recorded_level == "unavailable" or current_level == "unavailable":
        return "unavailable"
    if recorded_level != current_level:
        return "level"
    if recorded_level == "measured":
        same = (
            recorded.get("digest") == current.get("digest")
            and recorded.get("algorithm_id") == current.get("algorithm_id")
            and recorded.get("algorithm_version") == current.get("algorithm_version")
        )
        return None if same else "digest"
    if recorded_level == "asserted":
        same = recorded.get("assertion") == current.get("assertion")
        return None if same else "assertion"
    # level == "none": explicit recorded opt-out on both sides; the resume
    # proceeds with no comparison claimed.
    return None


def read_trusted_rows(container: Path) -> list[dict[str, Any]]:
    """Read the ledger in LEDGER ORDER and verify the trusted prefix.

    A torn FINAL line (crash debris from an interrupted append) is dropped —
    never trusted, never fatal. An unparseable MIDDLE line, or a ledgered
    shard that is missing or byte-size-mismatched, ends the trusted prefix
    TYPED: rows after it depend on the cumulative row ranges before it.

    Parameters
    ----------
    container:
        Artifact directory holding ``ledger.jsonl``.

    Returns
    -------
    list[dict[str, Any]]
        The verified ledger rows, in ledger order.

    Raises
    ------
    ExtractionArtifactError
        ``extraction_ledger_invalid`` on a mid-file break;
        ``extraction_ledger_prefix_broken`` on a missing/corrupt member.
    """

    ledger_path = container / LEDGER_FILENAME
    if not ledger_path.exists():
        return []
    raw_lines = ledger_path.read_bytes().split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines.pop()
    rows: list[dict[str, Any]] = []
    for position, line in enumerate(raw_lines):
        try:
            row = _json.loads_bounded(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("ledger row is not an object")
        except (ValueError, UnicodeDecodeError) as exc:
            if position == len(raw_lines) - 1:
                break  # torn final line: crash debris, dropped untrusted
            raise ExtractionArtifactError(
                f"Extraction ledger {str(ledger_path)!r} line {position + 1} is "
                f"unparseable ({exc}); a broken middle member ends the trusted "
                "prefix.",
                code="extraction_ledger_invalid",
                remedy=(
                    "delete the output directory and re-extract, or restore the "
                    "ledger from a backup"
                ),
                ledger_path=str(ledger_path),
                line=position + 1,
            ) from exc
        rows.append(row)
    for row in rows:
        shard_path = container / str(row.get("file", ""))
        expected = row.get("byte_size")
        if not shard_path.is_file() or (
            isinstance(expected, int) and shard_path.stat().st_size != expected
        ):
            raise ExtractionArtifactError(
                f"Ledgered shard {str(shard_path.name)!r} (index {row.get('index')}) "
                "is missing or byte-size-mismatched; later rows depend on its row "
                "range, so the trusted prefix ends here.",
                code="extraction_ledger_prefix_broken",
                remedy=(
                    "delete the output directory and re-extract (or restore the "
                    "missing shard file from a backup)"
                ),
                shard=str(shard_path.name),
                index=row.get("index"),
                expected_bytes=expected,
            )
    return rows


def repair_ledger_tail(container: Path) -> None:
    """Clear torn crash debris from the ledger tail before appends resume.

    A torn FINAL line is never trusted (D1) — but leaving its bytes in place
    would corrupt the next append by concatenation. Exactly two repairs are
    legal, both consistent with :func:`read_trusted_rows`'s trust decisions:
    an UNPARSEABLE final line is truncated away (its commit never completed),
    and a parseable final line missing only its newline gets the newline
    appended (the row's JSON is complete; only the terminator was lost).
    Committed rows are never rewritten — the ledger stays append-only.

    Parameters
    ----------
    container:
        Artifact directory holding ``ledger.jsonl``.
    """

    ledger_path = container / LEDGER_FILENAME
    if not ledger_path.exists():
        return
    data = ledger_path.read_bytes()
    if not data:
        return
    lines = data.split(b"\n")
    ends_with_newline = lines[-1] == b""
    if ends_with_newline:
        lines.pop()
    keep_bytes = 0
    torn = False
    for position, line in enumerate(lines):
        try:
            row = _json.loads_bounded(line.decode("utf-8"))
            if not isinstance(row, dict):
                raise ValueError("ledger row is not an object")
        except (ValueError, UnicodeDecodeError):
            # read_trusted_rows refuses typed on a mid-file break; only a
            # final-line tear reaches here untyped.
            torn = position == len(lines) - 1 and not ends_with_newline
            break
        keep_bytes += len(line) + 1
    if torn:
        with open(ledger_path, "r+b") as handle:
            handle.truncate(keep_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        return
    if not ends_with_newline and keep_bytes == len(data) + 1:
        # Every line parses but the last lost its terminator: complete it.
        with open(ledger_path, "ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())


class ArtifactWriter:
    """The v2 artifact writer implementing the commit protocol (extract D1).

    Parameters
    ----------
    container:
        Artifact directory (created if absent).
    manifest:
        The manifest document to own; written at creation, plan-freeze, and
        terminal status only.
    """

    def __init__(
        self,
        container: Path,
        manifest: dict[str, Any],
        *,
        shard_extension: str = ".pt",
        checksums: str = "fast",
    ) -> None:
        """Bind the writer to its directory, manifest, and shard policies.

        Parameters
        ----------
        container:
            Artifact directory.
        manifest:
            The manifest document to own.
        shard_extension:
            Filename extension for the artifact's shard format.
        checksums:
            The D7 integrity level (``"fast"`` / ``"crypto"`` / ``"none"``).
        """

        self.container = container
        self.manifest = manifest
        self.shard_extension = shard_extension
        self.checksums = checksums
        self._ledger_path = container / LEDGER_FILENAME

    def write_manifest(self) -> None:
        """Write the manifest atomically with fsync (one of the three writes)."""

        _atomic_write_json_fsync(self.container / "manifest.json", self.manifest)

    def write_stimulus_ids_sidecar(self, ids: list[str]) -> None:
        """Write the ordered write-once stimulus-id sidecar (extract D2).

        Parameters
        ----------
        ids:
            Ordered stimulus identifiers.

        Raises
        ------
        ExtractionArtifactError
            ``extraction_stimulus_ids_sidecar_invalid`` when a differing
            sidecar already exists (write-once means write once).
        """

        sidecar = self.container / STIMULUS_IDS_FILENAME
        payload = {"schema": _STIMULUS_IDS_SCHEMA, "ids": list(ids)}
        if sidecar.exists():
            try:
                existing = _json.read_bounded(sidecar)
            except Exception as exc:
                raise ExtractionArtifactError(
                    f"Stimulus-id sidecar {str(sidecar)!r} is unreadable ({exc}).",
                    code="extraction_stimulus_ids_sidecar_invalid",
                    remedy="delete the output directory and re-extract",
                    sidecar=str(sidecar),
                ) from exc
            if existing != payload:
                raise ExtractionArtifactError(
                    f"Stimulus-id sidecar {str(sidecar)!r} already records a "
                    "different ordered id list; the sidecar is write-once and "
                    "relabeling is a separate audited verb, never a resume "
                    "side effect.",
                    code="extraction_stimulus_ids_sidecar_invalid",
                    remedy=(
                        "resume with the artifact's original stimulus_ids, or "
                        "extract into a fresh directory"
                    ),
                    sidecar=str(sidecar),
                )
            return
        _atomic_write_json_fsync(sidecar, payload)

    def commit_shard(
        self,
        *,
        index: int,
        row_start: int,
        n_rows: int,
        save_payload: Callable[[Path], None],
        row_facts: dict[str, Any],
    ) -> dict[str, Any]:
        """Commit one shard under the protocol and append its ledger row.

        Order: (caller validated) -> temp shard -> flush/fsync -> atomic
        rename -> append/fsync ledger row. A shard file bearing its final
        name is durably complete; a ledger row exists only for renamed
        shards, so a crash at any point leaves either an ignorable temp
        file, an unledgered orphan, or a torn final ledger line — never a
        trusted lie. The shard filename is derived from ``index`` via
        :func:`shard_filename` (the artifact owns its layout).

        Parameters
        ----------
        index:
            Zero-based shard index.
        row_start:
            Global row index of this shard's first stimulus.
        n_rows:
            Number of stimulus rows in this shard (input-derived).
        save_payload:
            Callback writing the payload to the TEMP path it is given.
        row_facts:
            Additional per-shard ledger facts (per-key shapes/dtypes, value
            reductions, id-range digest, ...). The writer's construction-time
            ``shard_extension`` and ``checksums`` policies govern the shard
            filename and the D7 integrity facts.

        Returns
        -------
        dict[str, Any]
            The appended ledger row.
        """

        filename = shard_filename(index, self.shard_extension)
        final_path = self.container / filename
        tmp_path = self.container / (filename + ".tmp")
        save_payload(tmp_path)
        fsync_file(tmp_path)
        os.replace(tmp_path, final_path)
        fsync_dir(self.container)
        row: dict[str, Any] = {
            "index": index,
            "file": filename,
            "row_start": row_start,
            "n_rows": n_rows,
            "byte_size": final_path.stat().st_size,
            "crc32": _crc32_file(final_path) if self.checksums == "fast" else None,
            **row_facts,
        }
        if self.checksums == "crypto":
            file_hasher = hashlib.blake2b(digest_size=32)
            with open(final_path, "rb") as shard_handle:
                while True:
                    chunk = shard_handle.read(1 << 20)
                    if not chunk:
                        break
                    file_hasher.update(chunk)
            row["file_digest"] = f"blake2b:{file_hasher.hexdigest()}"
        line = _canonical_dumps(row) + "\n"
        with open(self._ledger_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def finalize(self, *, n_shards: int, n_stimuli: int) -> None:
        """Write the terminal manifest: status, totals, ledger digest.

        Parameters
        ----------
        n_shards:
            Total committed shards.
        n_stimuli:
            Total stimulus rows across shards.
        """

        ledger_digest = None
        if self._ledger_path.exists():
            ledger_digest = "sha256:" + hashlib.sha256(self._ledger_path.read_bytes()).hexdigest()
        self.manifest["status"] = "complete"
        self.manifest["totals"] = {"n_shards": n_shards, "n_stimuli": n_stimuli}
        self.manifest["ledger_digest"] = ledger_digest
        self.write_manifest()


def migrate_v1_artifact(
    container: Path,
    manifest_v1: Mapping[str, Any],
    *,
    acknowledge_in_progress: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Migrate a COMPLETED v1 artifact to v2 without a forward (extract D1).

    Builds ``ledger.jsonl`` from the v1 in-manifest batch ledger, writes the
    stimulus-id sidecar when v1 recorded ids, and rewrites the manifest as
    v2 with every unprovable semantic field marked :data:`UNRECORDED_V1`
    (disclosed, skipped by the signature compare on completed artifacts).

    Parameters
    ----------
    container:
        Artifact directory.
    manifest_v1:
        The parsed v1 manifest.
    acknowledge_in_progress:
        The explicit acknowledgment for IN-PROGRESS v1 artifacts (extract
        D16): the migrated artifact records the prefix as
        asserted-not-measured and carries ``unknown_v1_prefix_semantics:
        true`` PERMANENTLY, propagated by every exporter into the export
        contract — an assertion that expires when the data is converted is
        not an assertion.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        The v2 manifest document and the migrated ledger rows.

    Raises
    ------
    ExtractionArtifactError
        ``extraction_resume_v1_in_progress`` for in-progress v1 artifacts
        without the acknowledgment: v1 recorded neither model mode nor grad
        state nor model identity, so an unfinished prefix cannot be proven
        compatible with any resuming run (the random-init T-MODELSWAP
        hazard lives exactly here).
    """

    if manifest_v1.get("status") != "complete" and not acknowledge_in_progress:
        raise ExtractionArtifactError(
            "This artifact was written in progress by the v1 layout, which "
            "recorded neither model identity, model mode, grad state, nor "
            "padding policy; the completed prefix cannot be proven compatible "
            "with any resuming run (a random-init resume would silently mix "
            "checkpoint activations with random ones).",
            code="extraction_resume_v1_in_progress",
            remedy=(
                "read the trusted prefix with load_extraction() and re-extract "
                "into a fresh directory to finish the dataset"
            ),
            unprovable_fields=[
                "model_identity",
                "model_mode",
                "padding_side",
                "position_ids_source",
            ],
            status=manifest_v1.get("status"),
        )
    in_progress_acknowledged = manifest_v1.get("status") != "complete"
    v1_signature = dict(manifest_v1.get("signature") or {})
    provenance = dict(manifest_v1.get("stimulus_provenance") or {})
    ids = provenance.get("stimulus_ids")
    v1_transform = v1_signature.get("transform")
    transform_pipeline: Any = None if v1_transform is None else UNRECORDED_V1
    signature_v2: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_V2,
        "native_format": "pt_shards_v1",
        "layer_plan": v1_signature.get("layer_plan"),
        "layers_kind": v1_signature.get("layers_kind"),
        "batch_size": v1_signature.get("batch_size"),
        "transform_pipeline": transform_pipeline,
        "stimuli": v1_signature.get("stimuli"),
        "stimulus_ids_digest": stimulus_ids_digest(list(ids) if isinstance(ids, list) else None),
        "model_identity": {"level": UNRECORDED_V1},
        "padding_side": UNRECORDED_V1,
        "position_ids_source": UNRECORDED_V1,
        "model_mode": UNRECORDED_V1,
        "pool": None,
        "dtype_policy": None,
        "ragged": "refuse",
        "integrity": UNRECORDED_V1,
        # F18 runtime fields: v1 recorded none of them (disclosed, skipped
        # by the compare); a resume through opaque callables still refuses
        # through the D8 rules, which treat an unrecorded block as empty.
        "collate": UNRECORDED_V1,
        "callable_identity": UNRECORDED_V1,
        "selector_plan": UNRECORDED_V1,
    }
    rows: list[dict[str, Any]] = []
    row_start = 0
    for batch in manifest_v1.get("batches") or []:
        filename = str(batch.get("file"))
        shard_path = container / filename
        n_rows = int(batch.get("n_stimuli", 0))
        rows.append(
            {
                "index": int(batch.get("index", len(rows))),
                "file": filename,
                "row_start": row_start,
                "n_rows": n_rows,
                "byte_size": shard_path.stat().st_size if shard_path.is_file() else None,
                "crc32": None,
                "integrity": "migrated_v1",
            }
        )
        row_start += n_rows
    manifest_v2: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA_V2,
        "torchlens_version": manifest_v1.get("torchlens_version"),
        "status": "in_progress" if in_progress_acknowledged else "complete",
        "signature": signature_v2,
        "stimulus_provenance": {
            "order": provenance.get("order"),
            "n_stimuli": provenance.get("n_stimuli"),
            "ids_recorded": isinstance(ids, list),
            "ids_digest": signature_v2["stimulus_ids_digest"],
        },
        "storage": dict(manifest_v1.get("storage") or {}) | {"ledger": LEDGER_FILENAME},
        "layers": manifest_v1.get("layers"),
        "run": {},
        "totals": (
            None
            if manifest_v1.get("status") != "complete"
            else {"n_shards": len(rows), "n_stimuli": row_start}
        ),
        "ledger_digest": None,
        "migration": {
            "migrated_from": MANIFEST_SCHEMA_V1,
            "unprovable_fields": [
                "model_identity",
                "model_mode",
                "padding_side",
                "position_ids_source",
            ]
            + (["transform_pipeline"] if transform_pipeline == UNRECORDED_V1 else []),
            "legacy_signature": v1_signature,
        },
    }
    if in_progress_acknowledged:
        # The acknowledgment is PERMANENT: the prefix's semantics were
        # asserted, never measured, and every exporter propagates the flag.
        manifest_v2["unknown_v1_prefix_semantics"] = True
        manifest_v2["migration"]["in_progress_acknowledged"] = "asserted_not_measured"
    writer = ArtifactWriter(container, manifest_v2)
    ledger_path = container / LEDGER_FILENAME
    if not ledger_path.exists():
        with open(ledger_path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    if isinstance(ids, list):
        writer.write_stimulus_ids_sidecar([str(item) for item in ids])
    manifest_v2["ledger_digest"] = "sha256:" + hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    writer.write_manifest()
    return manifest_v2, rows
