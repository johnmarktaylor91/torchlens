"""Portable bundle manifest schema and compatibility policy."""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from packaging.version import InvalidVersion, Version

from . import IO_FORMAT_VERSION, TorchLensIOError
from .. import __version__ as TORCHLENS_VERSION

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorEntry:
    """One persisted tensor blob entry in ``manifest.json``.

    Parameters
    ----------
    blob_id:
        Opaque zero-padded blob identifier.
    kind:
        Logical tensor kind, for example ``"activation"``.
    label:
        Human-readable TorchLens layer label associated with the blob.
    relative_path:
        Bundle-relative path to the blob file.
    backend:
        Storage backend name. S4 supports ``"safetensors"`` only.
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
                raise TorchLensIOError(
                    f"Manifest tensor entry must include non-empty string {field_name!r}."
                )

        shape = data.get("shape")
        if not isinstance(shape, list) or any(not isinstance(dim, int) for dim in shape):
            raise TorchLensIOError("Manifest tensor entry 'shape' must be a list of ints.")

        num_bytes = data.get("bytes")
        if not isinstance(num_bytes, int) or num_bytes < 0:
            raise TorchLensIOError("Manifest tensor entry 'bytes' must be a non-negative int.")

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
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the entry into JSON-serializable data.

        Returns
        -------
        dict[str, Any]
            JSON-ready manifest entry.
        """

        return asdict(self)


@dataclass(frozen=True)
class Manifest:
    """Structured representation of ``manifest.json`` for portable bundles.

    Parameters
    ----------
    io_format_version:
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
        Bundle container format. S4 supports ``"directory"`` only.
    n_layers:
        Total number of ``LayerPassLog`` entries in the saved log.
    n_activation_blobs:
        Count of persisted activation tensor blobs.
    n_gradient_blobs:
        Count of persisted gradient tensor blobs.
    n_auxiliary_blobs:
        Count of persisted non-activation, non-gradient tensor blobs.
    tensors:
        Persisted tensor entries.
    unsupported_tensors:
        Best-effort records for tensors skipped under ``strict=False``.
    """

    io_format_version: int
    torchlens_version: str
    torch_version: str
    python_version: str
    platform: str
    created_at: str
    bundle_format: str
    n_layers: int
    n_activation_blobs: int
    n_gradient_blobs: int
    n_auxiliary_blobs: int
    tensors: list[TensorEntry]
    unsupported_tensors: list[dict[str, str]]

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
        TorchLensIOError
            If required fields are missing or have invalid types.
        """

        required_int_fields = (
            "io_format_version",
            "n_layers",
            "n_activation_blobs",
            "n_gradient_blobs",
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
            if not isinstance(field_value, int) or field_value < 0:
                raise TorchLensIOError(
                    f"Manifest field {field_name!r} must be a non-negative integer."
                )

        for field_name in required_str_fields:
            field_value = data.get(field_name)
            if not isinstance(field_value, str) or field_value == "":
                raise TorchLensIOError(f"Manifest field {field_name!r} must be a non-empty string.")

        if data["bundle_format"] != "directory":
            raise TorchLensIOError(
                f"Unsupported bundle_format={data['bundle_format']!r}; expected 'directory'."
            )

        raw_tensors = data.get("tensors")
        if not isinstance(raw_tensors, list):
            raise TorchLensIOError("Manifest field 'tensors' must be a list.")
        tensors = [TensorEntry.from_dict(entry) for entry in raw_tensors]

        unsupported_tensors = _validate_unsupported_tensors(data.get("unsupported_tensors"))
        manifest = cls(
            io_format_version=data["io_format_version"],
            torchlens_version=data["torchlens_version"],
            torch_version=data["torch_version"],
            python_version=data["python_version"],
            platform=data["platform"],
            created_at=data["created_at"],
            bundle_format=data["bundle_format"],
            n_layers=data["n_layers"],
            n_activation_blobs=data["n_activation_blobs"],
            n_gradient_blobs=data["n_gradient_blobs"],
            n_auxiliary_blobs=data["n_auxiliary_blobs"],
            tensors=tensors,
            unsupported_tensors=unsupported_tensors,
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
                raw_data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise TorchLensIOError(f"Failed to read manifest at {manifest_path}.") from exc
        if not isinstance(raw_data, dict):
            raise TorchLensIOError("Manifest root must be a JSON object.")
        return cls.from_dict(raw_data)

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
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2, sort_keys=False)
                handle.write("\n")
        except OSError as exc:
            raise TorchLensIOError(f"Failed to write manifest at {manifest_path}.") from exc

    def to_dict(self) -> dict[str, Any]:
        """Convert the manifest into JSON-serializable data.

        Returns
        -------
        dict[str, Any]
            JSON-ready manifest mapping.
        """

        data = asdict(self)
        data["tensors"] = [entry.to_dict() for entry in self.tensors]
        return data

    def _validate_counts(self) -> None:
        """Ensure manifest tensor counts match the declared summary fields.

        Raises
        ------
        TorchLensIOError
            If the declared counts disagree with the tensor entries.
        """

        n_activation_blobs = sum(1 for entry in self.tensors if entry.kind == "activation")
        n_gradient_blobs = sum(1 for entry in self.tensors if entry.kind == "gradient")
        n_auxiliary_blobs = len(self.tensors) - n_activation_blobs - n_gradient_blobs
        if self.n_activation_blobs != n_activation_blobs:
            raise TorchLensIOError("Manifest n_activation_blobs does not match tensor entries.")
        if self.n_gradient_blobs != n_gradient_blobs:
            raise TorchLensIOError("Manifest n_gradient_blobs does not match tensor entries.")
        if self.n_auxiliary_blobs != n_auxiliary_blobs:
            raise TorchLensIOError("Manifest n_auxiliary_blobs does not match tensor entries.")


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


def enforce_version_policy(manifest: Manifest) -> None:
    """Apply the Fork F compatibility policy for a loaded bundle manifest.

    Parameters
    ----------
    manifest:
        Parsed manifest to validate against the current runtime.

    Raises
    ------
    TorchLensIOError
        If the bundle targets a newer I/O format or an incompatible torch
        major version.
    """

    if manifest.io_format_version > IO_FORMAT_VERSION:
        raise TorchLensIOError(
            "Bundle uses io_format_version="
            f"{manifest.io_format_version}, but this runtime only supports "
            f"{IO_FORMAT_VERSION}."
        )
    if manifest.io_format_version < IO_FORMAT_VERSION:
        warnings.warn(
            "Bundle io_format_version="
            f"{manifest.io_format_version} is older than runtime "
            f"io_format_version={IO_FORMAT_VERSION}; missing portable fields may "
            "default-fill during load.",
            DeprecationWarning,
            stacklevel=2,
        )

    runtime_torch = _parse_version(torch.__version__, label="runtime torch")
    manifest_torch = _parse_version(manifest.torch_version, label="manifest torch")
    if runtime_torch is not None and manifest_torch is not None:
        if runtime_torch.major != manifest_torch.major:
            raise TorchLensIOError(
                "Bundle torch_version="
                f"{manifest.torch_version} is incompatible with runtime torch_version="
                f"{torch.__version__} (major version mismatch)."
            )
        if runtime_torch.minor != manifest_torch.minor:
            warnings.warn(
                "Bundle torch_version="
                f"{manifest.torch_version} differs from runtime torch_version="
                f"{torch.__version__} (minor version mismatch).",
                UserWarning,
                stacklevel=2,
            )
    elif manifest.torch_version != torch.__version__:
        raise TorchLensIOError(
            "Bundle torch_version="
            f"{manifest.torch_version} could not be parsed compatibly with runtime "
            f"torch_version={torch.__version__}; refusing load."
        )

    runtime_torchlens = _parse_version(TORCHLENS_VERSION, label="runtime torchlens")
    manifest_torchlens = _parse_version(manifest.torchlens_version, label="manifest torchlens")
    if runtime_torchlens is not None and manifest_torchlens is not None:
        if manifest_torchlens > runtime_torchlens:
            warnings.warn(
                "Bundle torchlens_version="
                f"{manifest.torchlens_version} is newer than runtime torchlens_version="
                f"{TORCHLENS_VERSION}.",
                UserWarning,
                stacklevel=2,
            )
        elif manifest_torchlens < runtime_torchlens:
            LOGGER.info(
                "Bundle torchlens_version=%s is older than runtime torchlens_version=%s.",
                manifest.torchlens_version,
                TORCHLENS_VERSION,
            )
    elif manifest.torchlens_version != TORCHLENS_VERSION:
        warnings.warn(
            "Bundle torchlens_version="
            f"{manifest.torchlens_version} differs from runtime torchlens_version="
            f"{TORCHLENS_VERSION} and could not be parsed under PEP 440.",
            UserWarning,
            stacklevel=2,
        )

    runtime_python = _parse_version(_runtime_python_version(), label="runtime python")
    manifest_python = _parse_version(manifest.python_version, label="manifest python")
    if runtime_python is not None and manifest_python is not None:
        if runtime_python.major != manifest_python.major:
            warnings.warn(
                "Bundle python_version="
                f"{manifest.python_version} differs from runtime python_version="
                f"{_runtime_python_version()} (major version mismatch).",
                UserWarning,
                stacklevel=2,
            )
    elif manifest.python_version != _runtime_python_version():
        warnings.warn(
            "Bundle python_version="
            f"{manifest.python_version} differs from runtime python_version="
            f"{_runtime_python_version()} and could not be parsed under PEP 440.",
            UserWarning,
            stacklevel=2,
        )


def _parse_version(version_text: str, *, label: str) -> Version | None:
    """Parse a version string under PEP 440 with warning fallback.

    Parameters
    ----------
    version_text:
        Raw version string to parse.
    label:
        Human-readable label for warnings.

    Returns
    -------
    Version | None
        Parsed version object, or ``None`` when parsing fails.
    """

    try:
        return Version(version_text)
    except InvalidVersion:
        warnings.warn(
            f"Could not parse {label} version {version_text!r} under PEP 440; "
            "falling back to string comparison.",
            UserWarning,
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
        raise TorchLensIOError("Manifest field 'unsupported_tensors' must be a list.")
    validated: list[dict[str, str]] = []
    for entry in raw_value:
        if not isinstance(entry, dict):
            raise TorchLensIOError("Manifest unsupported_tensors entries must be objects.")
        validated_entry: dict[str, str] = {}
        for key, value in entry.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TorchLensIOError(
                    "Manifest unsupported_tensors entries must use string keys and values."
                )
            validated_entry[key] = value
        validated.append(validated_entry)
    return validated
