"""Validation subpackage for saved activations, backward capture, and invariants."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..intervention.save import check_spec_compat
from ..intervention.resolver import resolve_sites
from ..user_funcs import (
    validate_backward_pass,
    validate_batch_of_models_and_inputs,
    validate_forward_pass,
    validate_saved_activations,
)
from .core import validate_saved_activations as validate_model_log_saved_activations
from .consolidated import InterventionValidationReport, validate
from .invariants import MetadataInvariantError, check_metadata_invariants

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_tlspec(path: str | Path) -> None:
    """Validate a unified ``.tlspec`` manifest against schema v1.

    Legacy TorchLens 2.16 formats are intentionally accepted without schema
    validation so old artifacts remain loadable.

    Parameters
    ----------
    path:
        ``.tlspec`` directory path.

    Raises
    ------
    ValueError
        If a unified manifest violates the v1 schema.
    FileNotFoundError
        If a unified manifest file is missing.
    """

    from ..io import detect_tlspec_format, inspect_tlspec

    if detect_tlspec_format(path) != "v2.0_unified":
        return
    schema = _load_tlspec_manifest_schema()
    manifest = inspect_tlspec(path)
    _validate_manifest_against_schema(manifest, schema)


def _load_tlspec_manifest_schema() -> dict[str, Any]:
    """Load the bundled unified manifest schema.

    Returns
    -------
    dict[str, Any]
        Decoded JSON schema object.
    """

    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "tlspec_manifest_v1.json"
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError(f"TorchLens schema at {schema_path} is not a JSON object.")
    return schema


def _validate_manifest_against_schema(manifest: dict[str, Any], schema: dict[str, Any]) -> None:
    """Validate the schema subset TorchLens relies on at runtime.

    Parameters
    ----------
    manifest:
        Decoded manifest object.
    schema:
        Decoded schema object.

    Raises
    ------
    ValueError
        If validation fails.
    """

    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ValueError("Bundled tlspec schema has invalid 'required' field.")
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ValueError(f"Unified .tlspec manifest missing required fields: {missing}.")

    _require_int(manifest, "tlspec_version", expected=1)
    _require_str_enum(manifest, "kind", {"intervention", "model_log", "bundle"})
    _require_str(manifest, "created_at")
    _require_str(manifest, "torchlens_version")
    _require_str(manifest, "python_version")
    _require_str(manifest, "torch_version")
    _require_int(manifest, "schema_version", minimum=1)
    _require_str(manifest, "model_signature")
    _validate_model_fingerprint(manifest.get("model_fingerprint"))
    _validate_sites(manifest.get("sites"))
    _require_str_enum(manifest, "body_format", {"safetensors"})
    _validate_body_index(manifest.get("body_index"))
    _require_str_enum(manifest, "save_level", {"audit", "executable_with_callables", "portable"})
    _validate_optional_dependencies(manifest.get("optional_dependencies"))

    kind = manifest["kind"]
    if kind == "intervention":
        if not isinstance(manifest.get("spec_compat_info"), dict):
            raise ValueError("Intervention .tlspec manifests require object spec_compat_info.")
        if not isinstance(manifest.get("intervention_compat_metadata"), dict):
            raise ValueError(
                "Intervention .tlspec manifests require object intervention_compat_metadata."
            )
    else:
        if manifest.get("spec_compat_info") is not None:
            raise ValueError("Non-intervention .tlspec manifests require null spec_compat_info.")
        if manifest.get("intervention_compat_metadata") is not None:
            raise ValueError(
                "Non-intervention .tlspec manifests require null intervention_compat_metadata."
            )


def _validate_model_fingerprint(value: Any) -> None:
    """Validate a manifest model fingerprint object.

    Parameters
    ----------
    value:
        Raw fingerprint value.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    if not isinstance(value, dict):
        raise ValueError("Manifest model_fingerprint must be an object.")
    for field_name in ("parameter_meta_hash", "buffer_meta_hash"):
        field_value = value.get(field_name)
        if not isinstance(field_value, str) or _SHA256_RE.fullmatch(field_value) is None:
            raise ValueError(f"Manifest model_fingerprint.{field_name} must be a SHA-256 hex.")
    class_qualname = value.get("class_qualname")
    if not isinstance(class_qualname, str) or class_qualname == "":
        raise ValueError("Manifest model_fingerprint.class_qualname must be a non-empty string.")
    if not isinstance(value.get("extra_metadata"), dict):
        raise ValueError("Manifest model_fingerprint.extra_metadata must be an object.")


def _validate_sites(value: Any) -> None:
    """Validate manifest site descriptors.

    Parameters
    ----------
    value:
        Raw sites value.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    if not isinstance(value, list):
        raise ValueError("Manifest sites must be a list.")
    for index, site in enumerate(value):
        if not isinstance(site, dict):
            raise ValueError(f"Manifest sites[{index}] must be an object.")
        layer_label = site.get("layer_label")
        if not isinstance(layer_label, str):
            raise ValueError(f"Manifest sites[{index}].layer_label must be a string.")
        pass_index = site.get("pass_index")
        if pass_index is not None and not isinstance(pass_index, int):
            raise ValueError(f"Manifest sites[{index}].pass_index must be an int or null.")
        for field_name in ("function_path", "module_path", "op_kind"):
            field_value = site.get(field_name)
            if field_value is not None and not isinstance(field_value, str):
                raise ValueError(f"Manifest sites[{index}].{field_name} must be a string or null.")


def _validate_body_index(value: Any) -> None:
    """Validate manifest body index entries.

    Parameters
    ----------
    value:
        Raw body index value.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    if not isinstance(value, list):
        raise ValueError("Manifest body_index must be a list.")
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest body_index[{index}] must be an object.")
        for field_name in ("filename", "dtype", "intended_use"):
            field_value = entry.get(field_name)
            if not isinstance(field_value, str) or field_value == "":
                raise ValueError(
                    f"Manifest body_index[{index}].{field_name} must be a non-empty string."
                )
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(not isinstance(dim, int) for dim in shape):
            raise ValueError(f"Manifest body_index[{index}].shape must be a list of ints.")
        num_elements = entry.get("num_elements")
        if not isinstance(num_elements, int) or num_elements < 0:
            raise ValueError(
                f"Manifest body_index[{index}].num_elements must be a non-negative int."
            )


def _validate_optional_dependencies(value: Any) -> None:
    """Validate optional dependency metadata.

    Parameters
    ----------
    value:
        Raw optional dependency value.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    if not isinstance(value, list):
        raise ValueError("Manifest optional_dependencies must be a list.")
    if any(not isinstance(item, str) or item == "" for item in value):
        raise ValueError("Manifest optional_dependencies entries must be non-empty strings.")
    if len(value) != len(set(value)):
        raise ValueError("Manifest optional_dependencies entries must be unique.")


def _require_str(manifest: dict[str, Any], field_name: str) -> None:
    """Require one manifest field to be a non-empty string.

    Parameters
    ----------
    manifest:
        Manifest object.
    field_name:
        Field to validate.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    value = manifest.get(field_name)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"Manifest field {field_name!r} must be a non-empty string.")


def _require_int(
    manifest: dict[str, Any],
    field_name: str,
    *,
    expected: int | None = None,
    minimum: int | None = None,
) -> None:
    """Require one manifest field to be an integer.

    Parameters
    ----------
    manifest:
        Manifest object.
    field_name:
        Field to validate.
    expected:
        Exact required value, if any.
    minimum:
        Minimum accepted value, if any.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    value = manifest.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Manifest field {field_name!r} must be an integer.")
    if expected is not None and value != expected:
        raise ValueError(f"Manifest field {field_name!r} must equal {expected}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"Manifest field {field_name!r} must be >= {minimum}.")


def _require_str_enum(
    manifest: dict[str, Any],
    field_name: str,
    allowed_values: set[str],
) -> None:
    """Require one manifest field to be a string enum member.

    Parameters
    ----------
    manifest:
        Manifest object.
    field_name:
        Field to validate.
    allowed_values:
        Accepted string values.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    value = manifest.get(field_name)
    if not isinstance(value, str) or value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise ValueError(f"Manifest field {field_name!r} must be one of: {allowed}.")


__all__ = [
    "InterventionValidationReport",
    "validate_backward_pass",
    "validate_batch_of_models_and_inputs",
    "validate",
    "validate_forward_pass",
    "validate_saved_activations",
    "validate_model_log_saved_activations",
    "check_metadata_invariants",
    "check_spec_compat",
    "MetadataInvariantError",
    "resolve_sites",
    "validate_tlspec",
]
