"""Portable bundle manifest schema and compatibility policy.

This module defines the authoritative ``manifest.json`` structure for
TorchLens bundles, along with version and integrity helpers used by both save
and load paths. It records bundle-level metadata, one entry per persisted
tensor blob, and the compatibility checks that run before ``metadata.pkl`` is
trusted.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, TypeGuard

import torch
from packaging.version import InvalidVersion, Version

from .. import __version__ as TORCHLENS_VERSION
from ..errors._base import TorchLensWarning
from . import (
    MIN_TLSPEC_VERSION,
    TLSPEC_VERSION,
    ArtifactRuntimeIncompatibleError,
    ArtifactSchemaAgeWarning,
    TorchLensIOError,
    _json,
    above_ceiling_error,
    below_floor_error,
    compat_ledger,
)

LOGGER = logging.getLogger(__name__)

_CODEC_METADATA_TUPLE_TAG = "__torchlens_codec_tuple_v1__"

# Ceiling on manifest tensor entries. Load paths do per-entry work (sha256 of
# the referenced blob file plus a safetensors decode under ``lazy=False``), so
# an unbounded entry list let a KB-sized hostile manifest with millions of
# entries sharing one blob buy hours of CPU. Generous: real bundles carry a
# few entries per layer.
_MAX_MANIFEST_TENSOR_ENTRIES = 1_000_000

# Structural ceilings for tensor-entry shape metadata. ``shape`` is
# metadata-only at load, but downstream manifests multiply the dims into a
# public element count, and bool/negative/absurd dims are forgeries either
# way. torch tensors max out at 64 dims; per-dim 2**48 elements is far past
# any real tensor.
_MAX_TENSOR_DIMS = 256
_MAX_TENSOR_DIM_VALUE = 2**48


def _is_plain_nonnegative_int(value: Any) -> TypeGuard[int]:
    """Return whether ``value`` is a real non-negative int (bools excluded)."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


_SCHEMA_REMEDY = "re-save the artifact with tl.save(); do not hand-edit manifest.json"


def _schema_refuse(detail: str, **fields: Any) -> TorchLensIOError:
    """Typed refusal for a manifest that parses as JSON but violates the schema.

    One constructor for the whole malformed-manifest family (R65-1): every
    site stamps ``code="manifest_schema_invalid"`` plus the re-save remedy so
    callers branch on ``exc.fields["code"]``, never message text. ``read()``
    adds the artifact path to refusals that bubble out of ``from_dict``.
    """

    return TorchLensIOError(
        f"{detail} Remedy: {_SCHEMA_REMEDY}.",
        code="manifest_schema_invalid",
        remedy=_SCHEMA_REMEDY,
        **fields,
    )


@dataclass(frozen=True)
class TensorEntry:
    """One persisted tensor blob entry in ``manifest.json``.

    Parameters
    ----------
    blob_id:
        Opaque zero-padded blob identifier.
    kind:
        Logical tensor kind, for example ``"out"``.
    label:
        Human-readable TorchLens layer label associated with the blob.
    relative_path:
        Bundle-relative path to the blob file.
    backend:
        Storage backend name. Only ``"safetensors"`` is currently supported.
    shape:
        Tensor shape recorded at save time.
    dtype:
        Tensor dtype string recorded at save time.
    device_at_save:
        Original tensor device string.
    layout:
        Tensor layout string recorded at save time.
    bytes:
        Persisted tensor byte size after any ``.contiguous()`` conversion.
    sha256:
        SHA-256 digest of the written blob file bytes.
    requires_grad:
        Whether the logical torch tensor required gradients at save time.
    logical_backend:
        Optional logical backend that produced the payload before transport.
    codec:
        Optional payload codec name used to encode the logical payload.
    logical_dtype:
        Optional logical backend dtype string before transport conversion.
    logical_device:
        Optional logical backend device string before transport conversion.
    transport_backend:
        Optional physical transport backend used for persisted bytes.
    transport_dtype:
        Optional physical transport dtype string.
    codec_metadata:
        Optional best-effort JSON-ready codec metadata.
    """

    blob_id: str
    kind: str
    label: str
    relative_path: str
    backend: str
    shape: list[int]
    dtype: str
    device_at_save: str
    layout: str
    bytes: int
    sha256: str
    requires_grad: bool = False
    logical_backend: str | None = None
    codec: str | None = None
    logical_dtype: str | None = None
    logical_device: str | None = None
    transport_backend: str | None = None
    transport_dtype: str | None = None
    codec_metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorEntry:
        """Validate and build a ``TensorEntry`` from JSON-decoded data.

        Parameters
        ----------
        data:
            Raw decoded JSON mapping.

        Returns
        -------
        TensorEntry
            Validated tensor entry.

        Raises
        ------
        TorchLensIOError
            If the entry is missing required fields or uses invalid types.
        """

        required_str_fields = (
            "blob_id",
            "kind",
            "label",
            "relative_path",
            "backend",
            "dtype",
            "device_at_save",
            "layout",
            "sha256",
        )
        for field_name in required_str_fields:
            field_value = data.get(field_name)
            if not isinstance(field_value, str) or field_value == "":
                raise _schema_refuse(
                    f"Manifest tensor entry must include non-empty string {field_name!r}."
                )

        if not _is_sha256(data["sha256"]):
            raise _schema_refuse("Manifest tensor entry 'sha256' must be a SHA-256 hex digest.")

        shape = data.get("shape")
        if not isinstance(shape, list) or any(not _is_plain_nonnegative_int(dim) for dim in shape):
            raise _schema_refuse(
                "Manifest tensor entry 'shape' must be a list of non-negative ints."
            )
        if len(shape) > _MAX_TENSOR_DIMS or any(dim > _MAX_TENSOR_DIM_VALUE for dim in shape):
            raise _schema_refuse(
                "Manifest tensor entry 'shape' exceeds the structural dimension ceiling."
            )

        num_bytes = data.get("bytes")
        if not _is_plain_nonnegative_int(num_bytes):
            raise _schema_refuse("Manifest tensor entry 'bytes' must be a non-negative int.")

        optional_strings = {
            field_name: _optional_str(data, field_name)
            for field_name in (
                "logical_backend",
                "codec",
                "logical_dtype",
                "logical_device",
                "transport_backend",
                "transport_dtype",
            )
        }
        codec_metadata = data.get("codec_metadata")
        if codec_metadata is not None and not isinstance(codec_metadata, dict):
            raise _schema_refuse("Manifest tensor entry 'codec_metadata' must be an object.")
        if codec_metadata is not None:
            codec_metadata = _restore_codec_metadata_value(codec_metadata)
            if not isinstance(codec_metadata, dict):
                raise _schema_refuse(
                    "Manifest tensor entry 'codec_metadata' must decode to an object."
                )
        requires_grad = data.get("requires_grad", False)
        if not isinstance(requires_grad, bool):
            raise _schema_refuse("Manifest tensor entry 'requires_grad' must be a boolean.")

        return cls(
            blob_id=data["blob_id"],
            kind=data["kind"],
            label=data["label"],
            relative_path=data["relative_path"],
            backend=data["backend"],
            shape=shape,
            dtype=data["dtype"],
            device_at_save=data["device_at_save"],
            layout=data["layout"],
            bytes=num_bytes,
            sha256=data["sha256"],
            requires_grad=requires_grad,
            codec_metadata=codec_metadata,
            **optional_strings,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the entry into JSON-serializable data.

        Returns
        -------
        dict[str, Any]
            JSON-ready manifest entry.
        """

        data = {key: value for key, value in asdict(self).items() if value is not None}
        if self.codec_metadata is not None:
            data["codec_metadata"] = _json_ready_codec_metadata_value(self.codec_metadata)
        return data


def _json_ready_codec_metadata_value(value: Any) -> Any:
    """Encode tuple identity while making codec metadata JSON-ready.

    Parameters
    ----------
    value:
        Codec metadata value to encode.

    Returns
    -------
    Any
        JSON-ready value with tuples represented by an explicit tag.
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready_codec_metadata_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {
            _CODEC_METADATA_TUPLE_TAG: [_json_ready_codec_metadata_value(item) for item in value]
        }
    if isinstance(value, list):
        return [_json_ready_codec_metadata_value(item) for item in value]
    return str(value)


def _restore_codec_metadata_value(value: Any) -> Any:
    """Restore tagged container types in codec metadata.

    Parameters
    ----------
    value:
        JSON-decoded codec metadata value.

    Returns
    -------
    Any
        Value with tagged tuples reconstructed recursively.
    """

    if isinstance(value, list):
        return [_restore_codec_metadata_value(item) for item in value]
    if isinstance(value, dict):
        if set(value) == {_CODEC_METADATA_TUPLE_TAG}:
            items = value[_CODEC_METADATA_TUPLE_TAG]
            if not isinstance(items, list):
                raise _schema_refuse("Tagged codec metadata tuple must contain a list.")
            return tuple(_restore_codec_metadata_value(item) for item in items)
        return {key: _restore_codec_metadata_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Provenance:
    """Optional capture provenance certificate embedded in ``manifest.json``.

    Parameters
    ----------
    provenance_version:
        Version of this nested provenance schema.
    capture_devices:
        Sorted device strings observed on captured operations.
    dtype_policy:
        Recorded default-dtype and autocast facts; unavailable facts remain ``None``.
    rng_state_digests:
        Compact SHA-256 digests for capture-time RNG engines actually recorded;
        a failed digest records an ``unavailable:<ExceptionName>`` sentinel.
    input_hash:
        Content digest of materialized captured input tensors, when available;
        a failed digest (e.g. payloads discarded by a selective save) records
        an ``unavailable:<ExceptionName>`` sentinel.
    model_structure_hash:
        Address-free structural trace digest, when collection succeeds;
        a failed digest records an ``unavailable:<ExceptionName>`` sentinel.
    git_commit_hash:
        Commit of the user's current working directory, when it is a Git repository.
    """

    provenance_version: int
    capture_devices: list[str]
    dtype_policy: dict[str, Any]
    rng_state_digests: dict[str, str]
    input_hash: str | None
    model_structure_hash: str | None
    git_commit_hash: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Provenance:
        """Validate and construct one provenance certificate.

        Parameters
        ----------
        data:
            JSON-decoded provenance mapping.

        Returns
        -------
        Provenance
            Validated provenance certificate.

        Raises
        ------
        TorchLensIOError
            If the nested schema is malformed or unsupported.
        """

        if data.get("provenance_version") != 1:
            raise _schema_refuse("Manifest provenance_version must be exactly 1.")
        capture_devices = data.get("capture_devices")
        if not isinstance(capture_devices, list) or any(
            not isinstance(device, str) or not device for device in capture_devices
        ):
            raise _schema_refuse("Manifest provenance capture_devices must be strings.")
        dtype_policy = data.get("dtype_policy")
        if not isinstance(dtype_policy, dict):
            raise _schema_refuse("Manifest provenance dtype_policy must be an object.")
        rng_state_digests = data.get("rng_state_digests")
        if not isinstance(rng_state_digests, dict) or any(
            not isinstance(engine, str)
            or not (_is_sha256(digest) or _is_unavailable_sentinel(digest))
            for engine, digest in rng_state_digests.items()
        ):
            raise _schema_refuse(
                "Manifest provenance rng_state_digests must map names to SHA-256 digests "
                "or 'unavailable:<ExceptionName>' sentinels."
            )
        input_hash = _optional_sha256(data, "input_hash")
        model_structure_hash = _optional_sha256(data, "model_structure_hash")
        git_commit_hash = data.get("git_commit_hash")
        if git_commit_hash is not None and (
            not isinstance(git_commit_hash, str)
            or not 7 <= len(git_commit_hash) <= 64
            or any(character not in "0123456789abcdef" for character in git_commit_hash.lower())
        ):
            raise _schema_refuse("Manifest provenance git_commit_hash must be a Git hex hash.")
        return cls(
            provenance_version=1,
            capture_devices=list(capture_devices),
            dtype_policy=dict(dtype_policy),
            rng_state_digests=dict(rng_state_digests),
            input_hash=input_hash,
            model_structure_hash=model_structure_hash,
            git_commit_hash=git_commit_hash,
        )


def _optional_str(data: dict[str, Any], field_name: str) -> str | None:
    """Return an optional manifest string field after validation.

    Parameters
    ----------
    data:
        Manifest entry mapping.
    field_name:
        Optional field to read.

    Returns
    -------
    str | None
        Field value, or ``None`` when omitted.

    Raises
    ------
    TorchLensIOError
        If the optional value is present but not a non-empty string.
    """

    field_value = data.get(field_name)
    if field_value is None:
        return None
    if not isinstance(field_value, str) or field_value == "":
        raise _schema_refuse(
            f"Manifest tensor entry optional field {field_name!r} must be a non-empty string."
        )
    return field_value


@dataclass(frozen=True)
class Manifest:
    """Structured representation of ``manifest.json`` for portable bundles.

    Parameters
    ----------
    tlspec_version:
        Portable I/O schema version for TorchLens bundle metadata.
    torchlens_version:
        TorchLens runtime version that wrote the bundle.
    torch_version:
        PyTorch runtime version that wrote the bundle.
    python_version:
        Python runtime version that wrote the bundle.
    platform:
        Platform identifier string recorded at save time.
    created_at:
        UTC timestamp string for bundle creation.
    bundle_format:
        Bundle container format: ``"directory"`` for standard portable
        bundles written by ``Trace.save()``/``tl.save()``, or
        ``"fastlog-directory"`` for fastlog ring-buffer disk storage
        (see ``torchlens/fastlog/storage_disk.py``).
    n_layers:
        Total number of ``Op`` entries in the saved log.
    n_out_blobs:
        Count of persisted out tensor blobs.
    n_grad_blobs:
        Count of persisted grad tensor blobs.
    n_auxiliary_blobs:
        Count of persisted non-out, non-grad tensor blobs.
    tensors:
        Persisted tensor entries.
    unsupported_tensors:
        Best-effort records for tensors skipped under ``strict=False``.
    provenance:
        Optional versioned capture provenance certificate. Older manifests omit it.
    custom_attributes_disclosure:
        Optional save-time disclosure of the harvested module-attribute channel
        (``included`` flag, ``module_count``, bounded ``top_level_keys``). Older
        manifests omit it.
    buffer_values_disclosure:
        Optional save-time disclosure of the captured pre-forward buffer-value
        channel (``included`` flag, ``buffer_count``, bounded ``buffer_names``).
        Older manifests omit it.
    """

    tlspec_version: int
    torchlens_version: str
    torch_version: str
    python_version: str
    platform: str
    created_at: str
    bundle_format: str
    n_layers: int
    n_out_blobs: int
    n_grad_blobs: int
    n_auxiliary_blobs: int
    tensors: list[TensorEntry]
    unsupported_tensors: list[dict[str, str]]
    provenance: Provenance | None = None
    custom_attributes_disclosure: dict[str, Any] | None = None
    buffer_values_disclosure: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        """Validate decoded JSON data and build a manifest instance.

        Parameters
        ----------
        data:
            Raw JSON-decoded mapping.

        Returns
        -------
        Manifest
            Validated manifest.

        Raises
        ------
        ArtifactVersionBelowFloorError
            If the manifest predates the ``tlspec_version >= 6`` rehydration
            floor. Checked before the remaining required fields so a
            genuinely old manifest (which also lacks newer required fields)
            refuses with the floor named instead of a missing-field error.
        TorchLensIOError
            If required fields are missing or have invalid types.
        """

        raw_version = data.get("tlspec_version")
        if raw_version is None:
            # An absent tlspec_version predates portable I/O versioning; map it to
            # the below-floor refusal for consistency with the pickle-path
            # counterpart (R10-10) instead of a generic missing-field error. A
            # present-but-non-int value stays a required-field type error below.
            raise below_floor_error(
                observed="no tlspec_version (predates portable I/O versioning)",
                subject="Bundle manifest",
                code="artifact_version_below_floor",
            )
        if isinstance(raw_version, int) and raw_version < MIN_TLSPEC_VERSION:
            raise below_floor_error(
                observed=f"tlspec_version={raw_version}",
                subject="Bundle manifest",
                code="artifact_version_below_floor",
            )
        if isinstance(raw_version, int) and raw_version > TLSPEC_VERSION:
            # Symmetric with the below-floor check above (G3): a FUTURE
            # manifest may carry renamed/extra fields, and without this gate
            # it refuses with a misleading missing-required-field error
            # instead of the above-ceiling refusal that names the remedy.
            raise above_ceiling_error(
                observed=raw_version,
                subject="Bundle manifest",
                code="artifact_version_above_runtime",
            )

        required_int_fields = (
            "tlspec_version",
            "n_layers",
            "n_out_blobs",
            "n_grad_blobs",
            "n_auxiliary_blobs",
        )
        required_str_fields = (
            "torchlens_version",
            "torch_version",
            "python_version",
            "platform",
            "created_at",
            "bundle_format",
        )

        for field_name in required_int_fields:
            field_value = data.get(field_name)
            if not _is_plain_nonnegative_int(field_value):
                raise _schema_refuse(
                    f"Manifest field {field_name!r} must be a non-negative integer."
                )

        for field_name in required_str_fields:
            field_value = data.get(field_name)
            if not isinstance(field_value, str) or field_value == "":
                raise _schema_refuse(f"Manifest field {field_name!r} must be a non-empty string.")

        if data["bundle_format"] not in {"directory", "fastlog-directory"}:
            raise _schema_refuse(
                "Unsupported bundle_format="
                f"{data['bundle_format']!r}; expected 'directory' or 'fastlog-directory'."
            )

        raw_tensors = data.get("tensors")
        if not isinstance(raw_tensors, list):
            raise _schema_refuse("Manifest field 'tensors' must be a list.")
        if len(raw_tensors) > _MAX_MANIFEST_TENSOR_ENTRIES:
            raise _schema_refuse(
                f"Manifest declares {len(raw_tensors)} tensor entries, above the "
                f"{_MAX_MANIFEST_TENSOR_ENTRIES}-entry ceiling; refusing a "
                "structurally implausible artifact."
            )
        tensors = [TensorEntry.from_dict(entry) for entry in raw_tensors]
        # The save path writes exactly one blob file per entry, so duplicate
        # identities are forgeries: dup blob_ids silently last-win in the
        # load-side entry indexes, and dup relative_paths amplify per-entry
        # verification work against one file.
        seen_blob_ids: set[str] = set()
        seen_relative_paths: set[str] = set()
        for entry in tensors:
            if entry.blob_id in seen_blob_ids:
                raise _schema_refuse(
                    f"Manifest tensor entries duplicate blob_id {entry.blob_id!r}."
                )
            if entry.relative_path in seen_relative_paths:
                raise _schema_refuse(
                    f"Manifest tensor entries duplicate relative_path {entry.relative_path!r}."
                )
            seen_blob_ids.add(entry.blob_id)
            seen_relative_paths.add(entry.relative_path)

        unsupported_tensors = _validate_unsupported_tensors(data.get("unsupported_tensors"))
        raw_provenance = data.get("provenance")
        if raw_provenance is not None and not isinstance(raw_provenance, dict):
            raise _schema_refuse("Manifest field 'provenance' must be an object when present.")
        provenance = None if raw_provenance is None else Provenance.from_dict(raw_provenance)
        raw_disclosure = data.get("custom_attributes_disclosure")
        if raw_disclosure is not None:
            if not isinstance(raw_disclosure, dict):
                raise _schema_refuse(
                    "Manifest field 'custom_attributes_disclosure' must be an object when present."
                )
            if not isinstance(raw_disclosure.get("included"), bool):
                raise _schema_refuse(
                    "Manifest custom_attributes_disclosure.included must be a boolean."
                )
            raw_count = raw_disclosure.get("module_count")
            if not isinstance(raw_count, int) or raw_count < 0:
                raise _schema_refuse(
                    "Manifest custom_attributes_disclosure.module_count must be a "
                    "non-negative integer."
                )
            raw_keys = raw_disclosure.get("top_level_keys")
            if not isinstance(raw_keys, list) or not all(isinstance(key, str) for key in raw_keys):
                raise _schema_refuse(
                    "Manifest custom_attributes_disclosure.top_level_keys must be a "
                    "list of strings."
                )
        raw_buffer_disclosure = data.get("buffer_values_disclosure")
        if raw_buffer_disclosure is not None:
            if not isinstance(raw_buffer_disclosure, dict):
                raise _schema_refuse(
                    "Manifest field 'buffer_values_disclosure' must be an object when present."
                )
            if not isinstance(raw_buffer_disclosure.get("included"), bool):
                raise _schema_refuse(
                    "Manifest buffer_values_disclosure.included must be a boolean."
                )
            raw_buffer_count = raw_buffer_disclosure.get("buffer_count")
            if not isinstance(raw_buffer_count, int) or raw_buffer_count < 0:
                raise _schema_refuse(
                    "Manifest buffer_values_disclosure.buffer_count must be a non-negative integer."
                )
            raw_buffer_names = raw_buffer_disclosure.get("buffer_names")
            if not isinstance(raw_buffer_names, list) or not all(
                isinstance(name, str) for name in raw_buffer_names
            ):
                raise _schema_refuse(
                    "Manifest buffer_values_disclosure.buffer_names must be a list of strings."
                )
        manifest = cls(
            tlspec_version=data["tlspec_version"],
            torchlens_version=data["torchlens_version"],
            torch_version=data["torch_version"],
            python_version=data["python_version"],
            platform=data["platform"],
            created_at=data["created_at"],
            bundle_format=data["bundle_format"],
            n_layers=data["n_layers"],
            n_out_blobs=data["n_out_blobs"],
            n_grad_blobs=data["n_grad_blobs"],
            n_auxiliary_blobs=data["n_auxiliary_blobs"],
            tensors=tensors,
            unsupported_tensors=unsupported_tensors,
            provenance=provenance,
            custom_attributes_disclosure=raw_disclosure,
            buffer_values_disclosure=raw_buffer_disclosure,
        )
        manifest._validate_counts()
        return manifest

    @classmethod
    def read(cls, path: str | Path) -> Manifest:
        """Read, parse, and validate ``manifest.json``.

        Parameters
        ----------
        path:
            Path to the manifest file.

        Returns
        -------
        Manifest
            Validated manifest instance.

        Raises
        ------
        TorchLensIOError
            If the file cannot be read or does not decode into a valid manifest.
        """

        manifest_path = Path(path)
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                raw_data = _json.load_bounded(handle)
        except FileNotFoundError as exc:
            # Stable codes on the tl.load front door (R65): a missing, unreadable,
            # and malformed/over-limit manifest are distinct causes a caller (or
            # operator log reader) must be able to tell apart.
            raise TorchLensIOError(
                f"Manifest not found at {manifest_path}. Remedy: pass the bundle "
                "directory produced by tl.save() (it must contain manifest.json).",
                code="manifest_missing",
            ) from exc
        except OSError as exc:
            raise TorchLensIOError(
                f"Failed to read manifest at {manifest_path}: {type(exc).__name__}: {exc}. "
                "Remedy: check file permissions and that the bundle is fully copied.",
                code="manifest_unreadable",
            ) from exc
        except json.JSONDecodeError as exc:
            raise TorchLensIOError(
                f"Manifest at {manifest_path} does not parse as JSON within the "
                f"supported bounds: {exc}. Remedy: the artifact is corrupt or "
                "hand-edited; re-save it with tl.save().",
                code="manifest_unreadable",
            ) from exc
        if not isinstance(raw_data, dict):
            raise TorchLensIOError(
                f"Manifest root at {manifest_path} must be a JSON object. Remedy: "
                "the artifact is corrupt or hand-edited; re-save it with tl.save().",
                code="manifest_not_json_object",
            )
        try:
            return cls.from_dict(raw_data)
        except TorchLensIOError as exc:
            # Schema refusals from from_dict know the field, never the file;
            # the read seam is where the artifact path is known (R65-1).
            if exc.file_path is None:
                exc.file_path = str(manifest_path)
            raise

    def write(self, path: str | Path) -> None:
        """Write the manifest to disk using pretty-printed JSON.

        Parameters
        ----------
        path:
            Destination path for ``manifest.json``.

        Raises
        ------
        TorchLensIOError
            If the file cannot be written.
        """

        manifest_path = Path(path)
        try:
            text = json.dumps(
                self.to_dict(),
                indent=2,
                sort_keys=False,
                allow_nan=False,
            )
            with manifest_path.open("w", encoding="utf-8") as handle:
                handle.write(text + "\n")
                handle.flush()
                # Durability: a crash after the enclosing publish rename must
                # not leave a zero-length/partial manifest behind it.
                os.fsync(handle.fileno())
        except (OSError, ValueError) as exc:
            raise TorchLensIOError(
                f"Failed to write manifest at {manifest_path}: {type(exc).__name__}: "
                f"{exc}. Remedy: check disk space and directory permissions, then "
                "re-save.",
                code="manifest_write_failed",
                remedy="check disk space and directory permissions, then re-save",
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest into JSON-serializable data.

        Returns
        -------
        dict[str, Any]
            JSON-ready manifest mapping.
        """

        data = asdict(self)
        data["tensors"] = [entry.to_dict() for entry in self.tensors]
        if self.provenance is None:
            data.pop("provenance")
        if self.custom_attributes_disclosure is None:
            data.pop("custom_attributes_disclosure")
        if self.buffer_values_disclosure is None:
            data.pop("buffer_values_disclosure")
        return data

    def _validate_counts(self) -> None:
        """Ensure manifest tensor counts match the declared summary fields.

        Raises
        ------
        TorchLensIOError
            If the declared counts disagree with the tensor entries.
        """

        n_out_blobs = sum(1 for entry in self.tensors if entry.kind == "out")
        n_grad_blobs = sum(1 for entry in self.tensors if entry.kind == "grad")
        n_auxiliary_blobs = len(self.tensors) - n_out_blobs - n_grad_blobs
        if self.n_out_blobs != n_out_blobs:
            raise _schema_refuse("Manifest n_out_blobs does not match tensor entries.")
        if self.n_grad_blobs != n_grad_blobs:
            raise _schema_refuse("Manifest n_grad_blobs does not match tensor entries.")
        if self.n_auxiliary_blobs != n_auxiliary_blobs:
            raise _schema_refuse("Manifest n_auxiliary_blobs does not match tensor entries.")


def sha256_of_file(path: str | Path) -> str:
    """Compute the SHA-256 digest of one file's bytes.

    Parameters
    ----------
    path:
        File path to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """

    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    """Return whether a value is a lowercase or uppercase SHA-256 hex digest.

    Parameters
    ----------
    value:
        Candidate digest.

    Returns
    -------
    bool
        Whether ``value`` is a 64-character hexadecimal string.
    """

    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_unavailable_sentinel(value: Any) -> bool:
    """Return whether a value is a sanctioned could-not-compute digest sentinel.

    The save side records ``unavailable:<ExceptionName>`` when digest machinery
    fails (e.g. selective saves discard input payloads before save time), so a
    consumer can distinguish could-not-compute from does-not-apply. The grammar
    is closed: a single ``unavailable:`` prefix followed by one Python
    identifier (the exception class name), bounded in length, so arbitrary
    strings cannot ride the sentinel slot.

    Parameters
    ----------
    value:
        Candidate sentinel.

    Returns
    -------
    bool
        Whether ``value`` matches the closed sentinel grammar.
    """

    if not isinstance(value, str) or len(value) > 256:
        return False
    prefix, _, exception_name = value.partition(":")
    return prefix == "unavailable" and exception_name.isidentifier()


def _optional_sha256(data: dict[str, Any], field_name: str) -> str | None:
    """Validate an optional SHA-256 provenance digest field.

    Parameters
    ----------
    data:
        Provenance mapping.
    field_name:
        Field to validate.

    Returns
    -------
    str | None
        Validated digest or could-not-compute sentinel, or ``None``.

    Raises
    ------
    TorchLensIOError
        If the present value is neither a SHA-256 digest nor a sanctioned
        ``unavailable:<ExceptionName>`` sentinel.
    """

    value = data.get(field_name)
    if value is not None and not _is_sha256(value) and not _is_unavailable_sentinel(value):
        raise _schema_refuse(
            f"Manifest provenance {field_name} must be a SHA-256 digest "
            "or an 'unavailable:<ExceptionName>' sentinel."
        )
    return value


def enforce_version_policy(manifest: Manifest, *, bundle_path: Path | None = None) -> None:
    """Apply the bundle version and integrity compatibility policy for a loaded manifest.

    Parameters
    ----------
    manifest:
        Parsed manifest to validate against the current runtime.
    bundle_path:
        Artifact directory, when the caller has it in scope. Enables the
        migration-witness door for producer pair-consistency: a migrated
        artifact keeps its original writer identity and proves its
        (writer, stamp) lineage through ``tl_migration_provenance.json``.

    Raises
    ------
    ArtifactVersionBelowFloorError
        If the bundle predates the ``tlspec_version >= 6`` rehydration floor
        (first written by released torchlens 2.31.0).
    TorchLensIOError
        If the bundle targets a newer I/O format or an incompatible torch
        major version.

    Warns
    -----
    ArtifactSchemaAgeWarning
        If the bundle is between the rehydration floor and the current
        ``tlspec_version`` and therefore loads at its own recorded schema.
    """

    if manifest.tlspec_version > TLSPEC_VERSION:
        raise above_ceiling_error(
            observed=manifest.tlspec_version,
            subject="Bundle",
            code="artifact_version_above_runtime",
        )
    if manifest.tlspec_version < MIN_TLSPEC_VERSION:
        raise below_floor_error(
            observed=f"tlspec_version={manifest.tlspec_version}",
            subject="Bundle",
            code="artifact_version_below_floor",
        )
    if manifest.tlspec_version < TLSPEC_VERSION:
        # Honest between-floor-and-current advisory (r6 L7): the artifact loads
        # at its recorded schema; fields introduced by later schema versions are
        # absent, never default-filled (``Manifest.from_dict`` fails closed on
        # required fields). Categorized ``ArtifactSchemaAgeWarning`` (a visible
        # ``UserWarning`` subclass), never ``DeprecationWarning``: this advisory
        # deprecates no API, and the default warning filters hide
        # ``DeprecationWarning`` from end users (grind r3, R15-F1).
        warnings.warn(
            f"Bundle tlspec_version={manifest.tlspec_version} is older than "
            f"runtime tlspec_version={TLSPEC_VERSION}; loading at the recorded "
            "schema. Re-save the artifact with this release to upgrade it.",
            ArtifactSchemaAgeWarning,
            stacklevel=2,
        )

    runtime_torch = _parse_version(torch.__version__, label="runtime torch")
    manifest_torch = _parse_version(manifest.torch_version, label="manifest torch")
    if runtime_torch is not None and manifest_torch is not None:
        if runtime_torch.major != manifest_torch.major:
            raise ArtifactRuntimeIncompatibleError(
                "Bundle torch_version="
                f"{manifest.torch_version} is incompatible with runtime torch_version="
                f"{torch.__version__} (major version mismatch). Remedy: load the "
                "bundle under a torch runtime with the recorded major version.",
                code="bundle_torch_incompatible",
                remedy="load the bundle under a torch runtime with the recorded major version",
            )
        if runtime_torch.minor != manifest_torch.minor:
            warnings.warn(
                "Bundle torch_version="
                f"{manifest.torch_version} differs from runtime torch_version="
                f"{torch.__version__} (minor version mismatch).",
                TorchLensWarning,
                stacklevel=2,
            )
    elif manifest.torch_version != torch.__version__:
        raise ArtifactRuntimeIncompatibleError(
            "Bundle torch_version="
            f"{manifest.torch_version} could not be parsed compatibly with runtime "
            f"torch_version={torch.__version__}; refusing load. Remedy: load the "
            "bundle under a torch runtime with the recorded major version.",
            code="bundle_torch_incompatible",
            remedy="load the bundle under a torch runtime with the recorded major version",
        )

    runtime_torchlens = _parse_version(TORCHLENS_VERSION, label="runtime torchlens")
    manifest_torchlens = _parse_version(
        manifest.torchlens_version,
        label="manifest torchlens",
        warn_on_failure=False,
    )
    if manifest_torchlens is None:
        raise TorchLensIOError(
            "Bundle torchlens_version="
            f"{manifest.torchlens_version!r} could not be parsed under PEP 440; "
            "refusing a current-schema artifact with unverifiable producer "
            "provenance. Remedy: re-save the artifact with a released torchlens.",
            code="bundle_producer_unverifiable",
            remedy="re-save the artifact with a released torchlens",
        )
    # Producer PAIR-CONSISTENCY against the governed compatibility ledger
    # (ecosystem MEMO 3.1, gate G5). This replaced the hand-typed
    # ``< "2.33"`` inequality that orphaned lawful released v2.31.0/v2.32.4
    # artifacts: the first tlspec-6 writer was v2.31.0 (measured on genuine
    # wheels), so a (writer, stamp) pair refuses only when NO governed ledger
    # window says that writer emitted that stamp. A migrated artifact keeps
    # its original writer identity and proves its lineage through the
    # migration witness sidecar instead; an invalid witness refuses typed.
    if manifest_torchlens is not None and not compat_ledger.pair_is_governed(
        manifest.torchlens_version, manifest.tlspec_version
    ):
        witness = (
            compat_ledger.read_migration_witness(bundle_path) if bundle_path is not None else None
        )
        if witness is None or not compat_ledger.migrated_pair_is_governed(
            manifest.torchlens_version,
            manifest.tlspec_version,
            witness,
            bundle_path=bundle_path,
        ):
            raise compat_ledger.ungoverned_pair_error(
                manifest.torchlens_version,
                manifest.tlspec_version,
                subject="Bundle",
                code="artifact_producer_pair_ungoverned",
            )
    if runtime_torchlens is not None and manifest_torchlens is not None:
        if manifest_torchlens > runtime_torchlens:
            warnings.warn(
                "Bundle torchlens_version="
                f"{manifest.torchlens_version} is newer than runtime torchlens_version="
                f"{TORCHLENS_VERSION}.",
                TorchLensWarning,
                stacklevel=2,
            )
        elif manifest_torchlens < runtime_torchlens:
            LOGGER.info(
                "Bundle torchlens_version=%s is older than runtime torchlens_version=%s.",
                manifest.torchlens_version,
                TORCHLENS_VERSION,
            )

    runtime_python = _parse_version(_runtime_python_version(), label="runtime python")
    manifest_python = _parse_version(manifest.python_version, label="manifest python")
    if runtime_python is not None and manifest_python is not None:
        if runtime_python.major != manifest_python.major:
            warnings.warn(
                "Bundle python_version="
                f"{manifest.python_version} differs from runtime python_version="
                f"{_runtime_python_version()} (major version mismatch).",
                TorchLensWarning,
                stacklevel=2,
            )
    elif manifest.python_version != _runtime_python_version():
        warnings.warn(
            "Bundle python_version="
            f"{manifest.python_version} differs from runtime python_version="
            f"{_runtime_python_version()} and could not be parsed under PEP 440.",
            TorchLensWarning,
            stacklevel=2,
        )


def _parse_version(
    version_text: str,
    *,
    label: str,
    warn_on_failure: bool = True,
) -> Version | None:
    """Parse a version string under PEP 440 with warning fallback.

    Parameters
    ----------
    version_text:
        Raw version string to parse.
    label:
        Human-readable label for warnings.
    warn_on_failure:
        Whether an unparseable version should emit the legacy fallback warning.

    Returns
    -------
    Version | None
        Parsed version object, or ``None`` when parsing fails.
    """

    try:
        return Version(version_text)
    except InvalidVersion:
        if warn_on_failure:
            warnings.warn(
                f"Could not parse {label} version {version_text!r} under PEP 440; "
                "falling back to string comparison.",
                TorchLensWarning,
                stacklevel=3,
            )
        return None


def _runtime_python_version() -> str:
    """Return the current runtime Python version string.

    Returns
    -------
    str
        ``major.minor.micro`` string for the active interpreter.
    """

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _validate_unsupported_tensors(raw_value: Any) -> list[dict[str, str]]:
    """Validate the ``unsupported_tensors`` manifest field.

    Parameters
    ----------
    raw_value:
        Raw decoded JSON value.

    Returns
    -------
    list[dict[str, str]]
        Validated unsupported tensor records.

    Raises
    ------
    TorchLensIOError
        If the value is not a list of string-only mappings.
    """

    if raw_value is None:
        return []
    if not isinstance(raw_value, list):
        raise _schema_refuse("Manifest field 'unsupported_tensors' must be a list.")
    validated: list[dict[str, str]] = []
    for entry in raw_value:
        if not isinstance(entry, dict):
            raise _schema_refuse("Manifest unsupported_tensors entries must be objects.")
        validated_entry: dict[str, str] = {}
        for key, value in entry.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise _schema_refuse(
                    "Manifest unsupported_tensors entries must use string keys and values."
                )
            validated_entry[key] = value
        validated.append(validated_entry)
    return validated
