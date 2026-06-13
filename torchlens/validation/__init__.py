"""Validation subpackage for saved outs, backward capture, and invariants."""

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
    validate_saved_outs,
)
from .core import validate_saved_outs as validate_trace_saved_outs
from .consolidated import InterventionValidationReport, validate
from .invariants import MetadataInvariantError, check_metadata_invariants

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_tlspec(path: str | Path) -> None:
    """Validate a unified ``.tlspec`` manifest against its manifest schema.

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
    manifest = inspect_tlspec(path)
    schema_version = _manifest_schema_version(manifest)
    schema = _load_tlspec_manifest_schema(schema_version)
    _validate_manifest_against_schema(manifest, schema)


def _manifest_schema_version(manifest: dict[str, Any]) -> int:
    """Return the manifest schema version from a decoded unified manifest.

    Parameters
    ----------
    manifest:
        Decoded manifest object.

    Returns
    -------
    int
        Manifest schema version.

    Raises
    ------
    ValueError
        If the schema version is missing or unsupported.
    """

    schema_version = manifest.get("schema_version", 1)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ValueError("Manifest field 'schema_version' must be an integer.")
    if schema_version not in {1, 2}:
        raise ValueError(
            f"Unsupported .tlspec manifest schema_version={schema_version}; expected 1 or 2."
        )
    return schema_version


def _load_tlspec_manifest_schema(schema_version: int) -> dict[str, Any]:
    """Load the bundled unified manifest schema.

    Parameters
    ----------
    schema_version:
        Manifest schema version to load.

    Returns
    -------
    dict[str, Any]
        Decoded JSON schema object.
    """

    schema_path = (
        Path(__file__).resolve().parents[1] / "schemas" / f"tlspec_manifest_v{schema_version}.json"
    )
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

    schema_version = _manifest_schema_version(manifest)
    _require_int(manifest, "tlspec_version", expected=1)
    _require_str_enum(manifest, "kind", {"intervention", "trace", "bundle"})
    _require_str(manifest, "created_at")
    _require_str(manifest, "torchlens_version")
    _require_str(manifest, "python_version")
    if schema_version == 1:
        _require_str(manifest, "torch_version")
    else:
        _validate_v2_backend_fields(manifest)
    _require_int(manifest, "schema_version", expected=schema_version)
    _require_str(manifest, "model_signature")
    _validate_model_fingerprint(manifest.get("model_fingerprint"), schema_version=schema_version)
    _validate_sites(manifest.get("sites"))
    body_formats = {"safetensors"} if schema_version == 1 else {"safetensors", "audit_only"}
    _require_str_enum(manifest, "body_format", body_formats)
    _validate_body_index(manifest.get("body_index"), schema_version=schema_version)
    _validate_backward_summary(manifest.get("backward_summary"), schema_version=schema_version)
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


def _validate_v2_backend_fields(manifest: dict[str, Any]) -> None:
    """Validate manifest schema v2 backend-owned fields.

    Parameters
    ----------
    manifest:
        Decoded manifest object.

    Raises
    ------
    ValueError
        If backend-aware fields are invalid.
    """

    from ..backends import UnknownBackendError, get_backend_spec

    backend_name = manifest.get("backend")
    if not isinstance(backend_name, str) or backend_name == "":
        raise ValueError("Manifest schema v2 requires non-empty backend.")
    try:
        spec = get_backend_spec(backend_name)
    except UnknownBackendError as exc:
        raise ValueError(f"Manifest schema v2 has unknown backend {backend_name!r}.") from exc

    runtime = manifest.get("backend_runtime")
    if not isinstance(runtime, dict):
        raise ValueError("Manifest schema v2 requires object backend_runtime.")
    for field_name in ("name", "version", "runtime_config", "device_summary", "compat_policy"):
        if field_name not in runtime:
            raise ValueError(f"Manifest backend_runtime missing {field_name!r}.")
    if not isinstance(runtime.get("name"), str) or runtime.get("name") == "":
        raise ValueError("Manifest backend_runtime.name must be a non-empty string.")
    if not isinstance(runtime.get("version"), str) or runtime.get("version") == "":
        raise ValueError("Manifest backend_runtime.version must be a non-empty string.")
    for field_name in ("runtime_config", "device_summary", "compat_policy"):
        if not isinstance(runtime.get(field_name), dict):
            raise ValueError(f"Manifest backend_runtime.{field_name} must be an object.")

    payload_policy = manifest.get("payload_policy")
    if not isinstance(payload_policy, dict):
        raise ValueError("Manifest schema v2 requires object payload_policy.")
    policy = payload_policy.get("policy")
    if policy not in {"full", "audit_only", "metadata_only"}:
        raise ValueError(
            "Manifest payload_policy.policy must be full, audit_only, or metadata_only."
        )
    if not isinstance(payload_policy.get("materialization_supported"), bool):
        raise ValueError("Manifest payload_policy.materialization_supported must be a bool.")
    payload_kinds = payload_policy.get("payload_kinds")
    if not isinstance(payload_kinds, list) or any(
        not isinstance(item, str) for item in payload_kinds
    ):
        raise ValueError("Manifest payload_policy.payload_kinds must be a list of strings.")

    torch_version = manifest.get("torch_version")
    if str(spec.name) == "torch":
        if not isinstance(torch_version, str) or torch_version == "":
            raise ValueError("Torch schema v2 manifests require non-empty torch_version.")
    elif torch_version is not None:
        raise ValueError("Non-torch schema v2 manifests require torch_version=null.")
    if str(spec.name) != "torch" and manifest.get("backward_summary") is not None:
        raise ValueError("Non-torch schema v2 manifests require backward_summary=null.")
    if "derived_gradient_summary" not in manifest:
        raise ValueError("Manifest schema v2 requires derived_gradient_summary.")


def _validate_model_fingerprint(value: Any, *, schema_version: int) -> None:
    """Validate a manifest model fingerprint object.

    Parameters
    ----------
    value:
        Raw fingerprint value.
    schema_version:
        Manifest schema version.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    if value is None and schema_version == 2:
        return
    if not isinstance(value, dict):
        raise ValueError("Manifest model_fingerprint must be an object.")
    if schema_version == 2 and not {"parameter_meta_hash", "buffer_meta_hash"} <= set(value):
        if not isinstance(value.get("backend_fingerprint"), dict):
            raise ValueError(
                "Manifest schema v2 backend-shaped model_fingerprint requires "
                "backend_fingerprint object."
            )
        return
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


def _validate_body_index(value: Any, *, schema_version: int) -> None:
    """Validate manifest body index entries.

    Parameters
    ----------
    value:
        Raw body index value.
    schema_version:
        Manifest schema version.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    if not isinstance(value, list):
        raise ValueError("Manifest body_index must be a list.")
    v1_intended_uses = {
        "bundle_marker",
        "buffer_initial_value",
        "captured_arg",
        "child_version",
        "func_config",
        "grad",
        "grad_fn_grad",
        "module_arg",
        "module_meta",
        "orphan_payload",
        "out",
        "rng_state",
        "transformed_grad",
        "transformed_out",
    }
    v2_intended_uses = v1_intended_uses | {
        "audit_record",
        "jax_const_leaf",
        "jax_derived_grad",
        "jax_equation_out",
        "jax_input_leaf",
        "jax_param_leaf",
    }
    allowed_intended_uses = v1_intended_uses if schema_version == 1 else v2_intended_uses
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
        intended_use = entry.get("intended_use")
        if intended_use not in allowed_intended_uses:
            raise ValueError(f"Manifest body_index[{index}].intended_use is invalid.")


def _validate_backward_summary(value: Any, *, schema_version: int) -> None:
    """Validate manifest backward summary metadata.

    Parameters
    ----------
    value:
        Raw backward summary object.
    schema_version:
        Manifest schema version.

    Raises
    ------
    ValueError
        If the value is invalid.
    """

    if value is None and schema_version == 2:
        return
    if not isinstance(value, dict):
        raise ValueError("Manifest backward_summary must be an object.")
    if not isinstance(value.get("has_backward_pass"), bool):
        raise ValueError("Manifest backward_summary.has_backward_pass must be a bool.")
    for field_name in (
        "num_backward_passes",
        "num_grad_fns",
        "num_grad_fn_calls",
        "num_saved_grad_records",
        "gradient_blob_count",
    ):
        field_value = value.get(field_name)
        if not isinstance(field_value, int) or field_value < 0:
            raise ValueError(f"Manifest backward_summary.{field_name} must be a non-negative int.")
    gradient_blob_kinds = value.get("gradient_blob_kinds")
    valid_kinds = {"grad", "grad_fn_grad", "transformed_grad"}
    if not isinstance(gradient_blob_kinds, list):
        raise ValueError("Manifest backward_summary.gradient_blob_kinds must be a list.")
    if any(kind not in valid_kinds for kind in gradient_blob_kinds):
        raise ValueError("Manifest backward_summary.gradient_blob_kinds contains an invalid kind.")
    if len(gradient_blob_kinds) != len(set(gradient_blob_kinds)):
        raise ValueError("Manifest backward_summary.gradient_blob_kinds entries must be unique.")
    old_bundle_policy = value.get("old_bundle_policy")
    if not isinstance(old_bundle_policy, str) or old_bundle_policy == "":
        raise ValueError("Manifest backward_summary.old_bundle_policy must be a non-empty string.")


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
    "validate_saved_outs",
    "validate_trace_saved_outs",
    "check_metadata_invariants",
    "check_spec_compat",
    "MetadataInvariantError",
    "resolve_sites",
    "validate_tlspec",
]
