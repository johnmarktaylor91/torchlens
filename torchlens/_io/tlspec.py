"""Unified ``.tlspec`` manifest writer helpers."""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors.torch import save_file

from . import TorchLensIOError
from .manifest import Manifest, TensorEntry, sha256_of_file
from .. import __version__ as TORCHLENS_VERSION

TLSPEC_VERSION = 1
TLSPEC_SCHEMA_VERSION = 1
TLSPEC_MANIFEST_FILENAME = "manifest.json"
TLSPEC_VALID_SAVE_LEVELS = ("audit", "executable_with_callables", "portable")
TlspecKind = Literal["intervention", "model_log", "bundle"]
TlspecSaveLevel = Literal["audit", "executable_with_callables", "portable"]

_EMPTY_META_HASH = sha256(repr([]).encode("utf-8")).hexdigest()


class _TlSpecWriter:
    """Write Phase 11 unified ``.tlspec`` manifests and bundle payloads."""

    @classmethod
    def write_model_log_manifest(
        cls,
        *,
        path: Path,
        model_log: Any,
        legacy_manifest: Manifest,
        save_level: str,
    ) -> None:
        """Write a unified manifest for a saved ``ModelLog`` payload.

        Parameters
        ----------
        path:
            Destination ``manifest.json`` path.
        model_log:
            Source model log.
        legacy_manifest:
            Existing portable-I/O manifest fields required by the loader.
        save_level:
            Save level recorded in the public schema.
        """

        manifest = legacy_manifest.to_dict()
        manifest.update(
            cls.build_manifest(
                kind="model_log",
                source=model_log,
                tensor_entries=legacy_manifest.tensors,
                save_level=save_level,
                spec_compat_info=None,
                intervention_compat_metadata=None,
            )
        )
        cls.write_json(path, manifest)

    @classmethod
    def write_intervention_manifest(
        cls,
        *,
        path: Path,
        log: Any,
        spec_json: dict[str, Any],
        tensor_entries: list[TensorEntry],
        legacy_format_version: str,
        save_level: str,
    ) -> None:
        """Write a unified manifest for an intervention spec.

        Parameters
        ----------
        path:
            Destination ``manifest.json`` path.
        log:
            Source model log carrying the intervention recipe.
        spec_json:
            Serialized intervention spec payload.
        tensor_entries:
            Persisted tensor entries.
        legacy_format_version:
            Legacy intervention ``format_version`` value.
        save_level:
            Save level recorded in the public schema.
        """

        intervention_metadata = cls._intervention_compat_metadata(spec_json)
        manifest = {
            "format_version": legacy_format_version,
            "tensor_entries": [entry.to_dict() for entry in tensor_entries],
        }
        manifest.update(
            cls.build_manifest(
                kind="intervention",
                source=log,
                tensor_entries=tensor_entries,
                save_level=save_level,
                spec_compat_info=intervention_metadata,
                intervention_compat_metadata=intervention_metadata,
            )
        )
        cls.write_json(path, manifest)

    @classmethod
    def write_bundle(
        cls,
        *,
        bundle: Any,
        path: str | Path,
        save_level: str = "portable",
        overwrite: bool = False,
    ) -> None:
        """Write a ``Bundle`` as a unified ``.tlspec`` directory.

        Parameters
        ----------
        bundle:
            Bundle instance to persist.
        path:
            Destination directory.
        save_level:
            Save level recorded in the public schema.
        overwrite:
            Whether an existing destination may be replaced.
        """

        level = coerce_tlspec_save_level(save_level)
        target_path = Path(path)
        _reject_symlink_path(target_path, context="bundle tlspec target")
        tmp_path = target_path.parent / f"tmp.{uuid.uuid4().hex}"
        body_filename = "body.safetensors"
        try:
            if target_path.exists() and not overwrite:
                raise FileExistsError(f"Bundle path already exists: {target_path}")
            tmp_path.mkdir(parents=True)
            save_file({}, str(tmp_path / body_filename))
            member_records = cls._write_bundle_members(bundle, tmp_path=tmp_path, save_level=level)
            cls.write_json(
                tmp_path / "bundle.json",
                {
                    "members": member_records,
                    "baseline_name": getattr(bundle, "baseline_name", None),
                },
            )

            manifest = cls.build_manifest(
                kind="bundle",
                source=bundle,
                tensor_entries=[],
                save_level=level,
                spec_compat_info=None,
                intervention_compat_metadata=None,
            )
            manifest["body_index"] = [
                {
                    "filename": body_filename,
                    "dtype": "none",
                    "shape": [],
                    "num_elements": 0,
                    "intended_use": "bundle_marker",
                    "sha256": sha256_of_file(tmp_path / body_filename),
                }
            ]
            cls.write_json(tmp_path / TLSPEC_MANIFEST_FILENAME, manifest)
            if target_path.exists():
                shutil.rmtree(target_path)
            os.rename(tmp_path, target_path)
        except Exception:
            if tmp_path.exists():
                shutil.rmtree(tmp_path, ignore_errors=True)
            raise

    @classmethod
    def _write_bundle_members(
        cls,
        bundle: Any,
        *,
        tmp_path: Path,
        save_level: str,
    ) -> list[dict[str, str]]:
        """Write bundle members as nested unified ModelLog specs.

        Parameters
        ----------
        bundle:
            Bundle object.
        tmp_path:
            Temporary bundle directory.
        save_level:
            Save level forwarded to each member save.

        Returns
        -------
        list[dict[str, str]]
            Member metadata records.
        """

        members = getattr(bundle, "members", {})
        if not isinstance(members, dict):
            raise TorchLensIOError("Bundle writer expected a mapping of members.")
        members_path = tmp_path / "members"
        members_path.mkdir()
        records: list[dict[str, str]] = []
        for index, (name, model_log) in enumerate(members.items()):
            relative_path = Path("members") / f"{index:04d}.tlspec"
            model_log.save(tmp_path / relative_path, level=save_level)
            records.append({"name": str(name), "path": relative_path.as_posix()})
        return records

    @classmethod
    def build_manifest(
        cls,
        *,
        kind: TlspecKind,
        source: Any,
        tensor_entries: list[TensorEntry],
        save_level: str,
        spec_compat_info: dict[str, Any] | None,
        intervention_compat_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build the public unified manifest fields.

        Parameters
        ----------
        kind:
            Public object kind.
        source:
            Source object used to derive model identity and site descriptors.
        tensor_entries:
            Tensor sidecars written for the payload.
        save_level:
            Save level recorded in the manifest.
        spec_compat_info:
            Intervention compatibility metadata, or ``None``.
        intervention_compat_metadata:
            Intervention replay metadata, or ``None``.

        Returns
        -------
        dict[str, Any]
            JSON-serializable unified manifest fields.
        """

        level = coerce_tlspec_save_level(save_level)
        model_signature = cls._model_signature(source, kind=kind)
        return {
            "tlspec_version": TLSPEC_VERSION,
            "kind": kind,
            "created_at": _utc_timestamp(),
            "torchlens_version": TORCHLENS_VERSION,
            "python_version": _python_version(),
            "torch_version": torch.__version__,
            "schema_version": TLSPEC_SCHEMA_VERSION,
            "model_signature": model_signature,
            "model_fingerprint": cls._model_fingerprint(source, model_signature, kind=kind),
            "sites": cls._sites(source, kind=kind),
            "spec_compat_info": spec_compat_info,
            "body_format": "safetensors",
            "body_index": cls._body_index(tensor_entries),
            "save_level": level,
            "optional_dependencies": cls._optional_dependencies(source),
            "intervention_compat_metadata": intervention_compat_metadata,
        }

    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> None:
        """Write a JSON object with deterministic formatting.

        Parameters
        ----------
        path:
            Destination JSON path.
        data:
            JSON object to write.
        """

        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=False)
            handle.write("\n")

    @classmethod
    def _model_signature(cls, source: Any, *, kind: TlspecKind) -> str:
        """Return the module-qualified model identity for a source object.

        Parameters
        ----------
        source:
            Source object.
        kind:
            Public object kind.

        Returns
        -------
        str
            Module-qualified identity string.
        """

        if kind == "bundle":
            return "torchlens.intervention.bundle.Bundle"
        source_model_class = getattr(source, "source_model_class", None)
        if isinstance(source_model_class, str) and source_model_class:
            return source_model_class
        model = _source_model(source)
        if model is not None:
            model_type = type(model)
            return f"{model_type.__module__}.{model_type.__qualname__}"
        source_type = type(source)
        return f"{source_type.__module__}.{source_type.__qualname__}"

    @classmethod
    def _model_fingerprint(
        cls,
        source: Any,
        model_signature: str,
        *,
        kind: TlspecKind,
    ) -> dict[str, Any]:
        """Build the public model fingerprint object.

        Parameters
        ----------
        source:
            Source object.
        model_signature:
            Module-qualified model identity.
        kind:
            Public object kind.

        Returns
        -------
        dict[str, Any]
            Fingerprint metadata.
        """

        if kind == "bundle":
            return cls._bundle_fingerprint(source, model_signature)

        model = _source_model(source)
        if model is not None:
            parameter_hash = _hash_named_tensor_meta(model.named_parameters())
            buffer_hash = _hash_named_tensor_meta(model.named_buffers())
        else:
            raw_parameter_hash = getattr(source, "weight_fingerprint_at_capture", None)
            if isinstance(raw_parameter_hash, str) and len(raw_parameter_hash) == 64:
                parameter_hash = raw_parameter_hash
            else:
                parameter_hash = _EMPTY_META_HASH
            buffer_hash = _EMPTY_META_HASH
        return {
            "parameter_meta_hash": parameter_hash,
            "buffer_meta_hash": buffer_hash,
            "class_qualname": model_signature,
            "extra_metadata": _extra_metadata_for_source(source),
        }

    @classmethod
    def _bundle_fingerprint(cls, bundle: Any, model_signature: str) -> dict[str, Any]:
        """Build a combined fingerprint for a ``Bundle``.

        Parameters
        ----------
        bundle:
            Bundle object.
        model_signature:
            Bundle class identity.

        Returns
        -------
        dict[str, Any]
            Bundle-level fingerprint metadata.
        """

        members = getattr(bundle, "members", {})
        member_fingerprints = []
        if isinstance(members, dict):
            for name, member in members.items():
                signature = cls._model_signature(member, kind="model_log")
                member_fingerprints.append(
                    (name, cls._model_fingerprint(member, signature, kind="model_log"))
                )
        parameter_hash = sha256(repr(member_fingerprints).encode("utf-8")).hexdigest()
        return {
            "parameter_meta_hash": parameter_hash,
            "buffer_meta_hash": _EMPTY_META_HASH,
            "class_qualname": model_signature,
            "extra_metadata": {
                "member_names": list(members) if isinstance(members, dict) else [],
                "baseline_name": getattr(bundle, "baseline_name", None),
            },
        }

    @classmethod
    def _sites(cls, source: Any, *, kind: TlspecKind) -> list[dict[str, Any]]:
        """Return site descriptors for a source object.

        Parameters
        ----------
        source:
            Source object.
        kind:
            Public object kind.

        Returns
        -------
        list[dict[str, Any]]
            Site descriptor objects.
        """

        if kind == "bundle":
            sites: list[dict[str, Any]] = []
            members = getattr(source, "members", {})
            if isinstance(members, dict):
                for member_name, member in members.items():
                    for site in cls._sites(member, kind="model_log"):
                        site["bundle_member"] = member_name
                        sites.append(site)
            return sites
        layers = getattr(source, "layer_list", [])
        sites = []
        for layer in layers if isinstance(layers, list) else []:
            sites.append(
                {
                    "function_path": _json_str_or_none(getattr(layer, "func_name", None)),
                    "module_path": _json_str_or_none(getattr(layer, "containing_module", None)),
                    "layer_label": str(getattr(layer, "layer_label", "")),
                    "pass_index": _json_int_or_none(getattr(layer, "pass_num", None)),
                    "op_kind": _json_str_or_none(
                        getattr(layer, "operation_equivalence_type", None)
                        or getattr(layer, "func_name", None)
                    ),
                }
            )
        return sites

    @staticmethod
    def _body_index(tensor_entries: list[TensorEntry]) -> list[dict[str, Any]]:
        """Build the public body index from tensor entries.

        Parameters
        ----------
        tensor_entries:
            Persisted tensor entries.

        Returns
        -------
        list[dict[str, Any]]
            Body index entries.
        """

        index = []
        for entry in tensor_entries:
            num_elements = 1
            for dim in entry.shape:
                num_elements *= dim
            index.append(
                {
                    "filename": entry.relative_path,
                    "dtype": entry.dtype,
                    "shape": entry.shape,
                    "num_elements": num_elements,
                    "intended_use": entry.kind,
                    "sha256": entry.sha256,
                }
            )
        return index

    @staticmethod
    def _optional_dependencies(source: Any) -> list[str]:
        """Return optional extras needed to load a source object.

        Parameters
        ----------
        source:
            Source object.

        Returns
        -------
        list[str]
            Optional dependency extra names.
        """

        metadata = getattr(source, "metadata", None)
        if isinstance(metadata, dict):
            value = metadata.get("optional_dependencies")
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return sorted(set(value))
        return []

    @staticmethod
    def _intervention_compat_metadata(spec_json: dict[str, Any]) -> dict[str, Any]:
        """Extract replay compatibility metadata from an intervention payload.

        Parameters
        ----------
        spec_json:
            Serialized intervention spec payload.

        Returns
        -------
        dict[str, Any]
            Compatibility metadata for replay checks.
        """

        return {
            "format_version": spec_json.get("format_version"),
            "helper_registry_version": spec_json.get("helper_registry_version"),
            "executable": bool(spec_json.get("executable", False)),
            "target_manifest": spec_json.get("target_manifest", []),
            "function_registry_keys": spec_json.get("function_registry_keys", []),
            "append_state": spec_json.get("append_state", {}),
        }


def coerce_tlspec_save_level(save_level: str) -> TlspecSaveLevel:
    """Validate and normalize a ``.tlspec`` save level.

    Parameters
    ----------
    save_level:
        User-supplied save level string.

    Returns
    -------
    TlspecSaveLevel
        Normalized save level.

    Raises
    ------
    ValueError
        If the save level is unsupported.
    """

    if save_level not in TLSPEC_VALID_SAVE_LEVELS:
        levels = ", ".join(repr(level) for level in TLSPEC_VALID_SAVE_LEVELS)
        raise ValueError(
            f"Unsupported .tlspec save level {save_level!r}; expected one of {levels}."
        )
    return save_level  # type: ignore[return-value]


def _source_model(source: Any) -> torch.nn.Module | None:
    """Return the weakly-held source model when available.

    Parameters
    ----------
    source:
        ModelLog-like source object.

    Returns
    -------
    torch.nn.Module | None
        Source model, or ``None`` when unavailable.
    """

    source_ref = getattr(source, "_source_model_ref", None)
    if source_ref is None:
        return None
    model = source_ref()
    return model if isinstance(model, torch.nn.Module) else None


def _hash_named_tensor_meta(named_tensors: Iterable[tuple[str, torch.Tensor]]) -> str:
    """Hash ``(name, shape, dtype)`` metadata for named tensors.

    Parameters
    ----------
    named_tensors:
        Iterable of named tensor pairs.

    Returns
    -------
    str
        SHA-256 hex digest.
    """

    entries = [(name, tuple(tensor.shape), str(tensor.dtype)) for name, tensor in named_tensors]
    return sha256(repr(entries).encode("utf-8")).hexdigest()


def _extra_metadata_for_source(source: Any) -> dict[str, Any]:
    """Return JSON-safe extra metadata for a model-log-like source.

    Parameters
    ----------
    source:
        Source object.

    Returns
    -------
    dict[str, Any]
        Extra metadata object.
    """

    return {
        "model_name": _json_str_or_none(getattr(source, "model_name", None)),
        "graph_shape_hash": _json_str_or_none(getattr(source, "graph_shape_hash", None)),
        "input_shape_hash": _json_str_or_none(getattr(source, "input_shape_hash", None)),
    }


def _json_str_or_none(value: Any) -> str | None:
    """Return a string for JSON output, preserving ``None``.

    Parameters
    ----------
    value:
        Raw value.

    Returns
    -------
    str | None
        JSON-safe string or ``None``.
    """

    if value is None:
        return None
    return str(value)


def _json_int_or_none(value: Any) -> int | None:
    """Return an int for JSON output when possible.

    Parameters
    ----------
    value:
        Raw value.

    Returns
    -------
    int | None
        JSON-safe int or ``None``.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _python_version() -> str:
    """Return the runtime Python version string.

    Returns
    -------
    str
        ``major.minor.micro`` version string.
    """

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _utc_timestamp() -> str:
    """Return a UTC ISO 8601 timestamp.

    Returns
    -------
    str
        UTC timestamp with ``Z`` suffix.
    """

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _reject_symlink_path(path: Path, *, context: str) -> None:
    """Reject symlink paths before writing a ``.tlspec`` payload.

    Parameters
    ----------
    path:
        Path to inspect.
    context:
        Human-readable path role.

    Raises
    ------
    TorchLensIOError
        If ``path`` is a symlink.
    """

    if path.is_symlink():
        raise TorchLensIOError(f"Refusing to write through symlink {context}: {path}.")


__all__ = [
    "TLSPEC_MANIFEST_FILENAME",
    "TLSPEC_SCHEMA_VERSION",
    "TLSPEC_VERSION",
    "_TlSpecWriter",
    "coerce_tlspec_save_level",
]
