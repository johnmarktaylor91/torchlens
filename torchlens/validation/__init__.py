"""Validation subpackage for saved outs, backward capture, and invariants."""

from __future__ import annotations

import copy
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
from .diagnostics import (
    ValidationDiagnostic,
    ValidationFailure,
    get_validation_diagnostics,
    get_validation_failure,
)
from .invariants import MetadataInvariantError, check_metadata_invariants
from .status import ValidationReplayState, ValidationReplayStatus

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_tlspec(
    path: str | Path,
    *,
    allow_unsupported_runnable_versions: bool = False,
) -> None:
    """Validate a unified ``.tlspec`` manifest against its manifest schema.

    Legacy TorchLens 2.16 formats are intentionally accepted without schema
    validation so old artifacts remain loadable.

    Parameters
    ----------
    path:
        ``.tlspec`` directory path.
    allow_unsupported_runnable_versions:
        Preserve structural validation while deferring runnable descriptor
        version ceilings to the structured readiness report. Public callers
        should retain the default tripwire behavior.

    Raises
    ------
    ValueError
        If a unified manifest violates the v1 schema.
    FileNotFoundError
        If a unified manifest file is missing.
    """

    from ..io import detect_tlspec_format, inspect_tlspec

    tlspec_path = Path(path)
    if not tlspec_path.exists():
        raise FileNotFoundError(f"TorchLens .tlspec path does not exist: {tlspec_path}.")
    if not tlspec_path.is_dir():
        raise ValueError(f"TorchLens .tlspec path must be a directory: {tlspec_path}.")

    manifest_path = tlspec_path / "manifest.json"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest_value = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse .tlspec manifest JSON at {manifest_path}.") from exc
        if not isinstance(manifest_value, dict):
            raise ValueError(f".tlspec manifest at {manifest_path} must be a JSON object.")

    tlspec_format = detect_tlspec_format(tlspec_path)
    if tlspec_format in {
        "v2.16_intervention_with_kind",
        "v2.16_intervention",
        "v2.16_modellog_portable",
    }:
        return
    if tlspec_format != "v2.0_unified":
        raise ValueError(f"Unrecognized TorchLens .tlspec format at {tlspec_path}.")

    manifest = inspect_tlspec(tlspec_path)
    schema_version = _manifest_schema_version(manifest)
    schema = _load_tlspec_manifest_schema(schema_version)
    _validate_manifest_against_schema(
        manifest,
        schema,
        allow_unsupported_runnable_versions=allow_unsupported_runnable_versions,
    )


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


def _validate_manifest_against_schema(
    manifest: dict[str, Any],
    schema: dict[str, Any],
    *,
    allow_unsupported_runnable_versions: bool = False,
) -> None:
    """Validate the schema subset TorchLens relies on at runtime.

    Parameters
    ----------
    manifest:
        Decoded manifest object.
    schema:
        Decoded schema object.
    allow_unsupported_runnable_versions:
        Whether runnable version equality is deferred to readiness preflight.

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

    # Walk the shipped JSON schema's own ``properties`` block so it is a real,
    # load-bearing contract rather than documentation the hand-written checks
    # below happen to agree with. Without this, an edit to only the JSON
    # schema file (the obvious place to look, given its name and the fact
    # that ``_load_tlspec_manifest_schema`` loads it) would silently no-op:
    # the schema is a package-data artifact external tooling may read
    # (``pyproject.toml``'s ``schemas/*.json`` package-data entry), so its
    # declared ``properties`` constraints must match what TorchLens itself
    # enforces.
    schema_for_validation = schema
    if allow_unsupported_runnable_versions and manifest.get("save_level") == "runnable":
        schema_for_validation = _schema_with_deferred_runnable_version_consts(schema)
    _validate_schema_properties(manifest, schema_for_validation, path="manifest")

    schema_version = _manifest_schema_version(manifest)
    # ``tlspec_version`` tracks the portable scrub/state format
    # (``torchlens._io.TLSPEC_VERSION``) and grows independently of
    # ``schema_version`` (which selects which of this file's schema variants
    # applies), so its supported range is enforced independently.
    _require_int(manifest, "tlspec_version", minimum=1)
    _validate_tlspec_version_ceiling(manifest["tlspec_version"])
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
    _require_str_enum(
        manifest,
        "save_level",
        {"audit", "executable_with_callables", "portable", "runnable"},
    )
    if manifest["save_level"] == "runnable":
        _validate_sparse_run_descriptor(
            manifest.get("run"),
            allow_unsupported_versions=allow_unsupported_runnable_versions,
        )
        _validate_runnable_payload_entries(manifest)
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


def _schema_with_deferred_runnable_version_consts(schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a manifest schema while deferring only runnable version constants.

    Parameters
    ----------
    schema:
        Bundled manifest schema used by the public strict validator.

    Returns
    -------
    dict[str, Any]
        Deep copy retaining every structural constraint while removing the
        exact-value checks reported by load-time structured readiness.
    """

    deferred = copy.deepcopy(schema)
    run_schema = deferred.get("properties", {}).get("run", {}).get("properties", {})
    for field_name in (
        "capability",
        "call_recipe",
        "callable_ref_schema",
        "state_binding",
        "input_binding",
        "control_witness",
        "initializer_policy_version",
    ):
        field_schema = run_schema.get(field_name)
        if isinstance(field_schema, dict):
            field_schema.pop("const", None)
    return deferred


def _validate_sparse_run_descriptor(
    value: Any,
    *,
    allow_unsupported_versions: bool = False,
) -> None:
    """Validate the frozen top-level sparse runnable descriptor contract.

    Parameters
    ----------
    value:
        Decoded ``manifest.run`` value.
    allow_unsupported_versions:
        Whether exact descriptor version checks are deferred to readiness.

    Raises
    ------
    ValueError
        If required versions, collections, preflight, or payload exclusions
        contradict the Stage-0 runnable schema.
    """

    from ..runnable import (
        RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        RUNNABLE_CALL_RECIPE_VERSION,
        RUNNABLE_INITIALIZER_POLICY_VERSION,
        RUNNABLE_TLSPEC_SCHEMA_VERSION,
    )

    if not isinstance(value, dict):
        raise ValueError("Runnable .tlspec manifests require object run descriptor.")
    required = {
        "capability",
        "backend",
        "call_recipe",
        "callable_ref_schema",
        "state_binding",
        "input_binding",
        "control_witness",
        "initializer_policy_version",
        "payload_layers",
        "callable_registry",
        "calls",
        "tensor_slots",
        "control_witnesses",
        "witness_completeness",
        "rng_profile",
        "compatibility",
        "preflight",
        "unsupported_sites",
    }
    if set(value) != required:
        missing = sorted(required - set(value))
        extra = sorted(set(value) - required)
        raise ValueError(
            f"Runnable run descriptor fields mismatch; missing={missing}, extra={extra}."
        )
    expected_versions = {
        "capability": RUNNABLE_TLSPEC_SCHEMA_VERSION,
        "call_recipe": RUNNABLE_CALL_RECIPE_VERSION,
        "callable_ref_schema": RUNNABLE_CALLABLE_REF_SCHEMA_VERSION,
        "initializer_policy_version": RUNNABLE_INITIALIZER_POLICY_VERSION,
        "state_binding": "module_path_role_v1",
        "input_binding": "model_site_io_role_v1",
        "control_witness": "scalar_bool_and_arm_entry_v1",
    }
    for field_name, expected in expected_versions.items():
        if allow_unsupported_versions:
            continue
        if value[field_name] != expected:
            raise ValueError(
                f"Runnable descriptor {field_name} must be {expected!r}; got {value[field_name]!r}."
            )
    for field_name in ("callable_registry", "calls", "tensor_slots", "control_witnesses"):
        if not isinstance(value[field_name], list):
            raise ValueError(f"Runnable descriptor {field_name} must be an array.")
    payload_layers = value["payload_layers"]
    if not isinstance(payload_layers, dict):
        raise ValueError("Runnable descriptor payload_layers must be an object.")
    legacy_payload_fields = frozenset({"weights", "activations"})
    current_payload_fields = frozenset({"weights", "nonpersistent_buffers", "activations"})
    if frozenset(payload_layers) not in {legacy_payload_fields, current_payload_fields}:
        raise ValueError("Runnable descriptor payload_layers fields mismatch.")
    expected_payload_schemas = {
        "weights": "state_dict_v1",
        "activations": "selected_activation_v1",
    }
    if "nonpersistent_buffers" in payload_layers:
        expected_payload_schemas["nonpersistent_buffers"] = "runnable_nonpersistent_buffer_v1"
    for layer_name, expected_schema in expected_payload_schemas.items():
        layer = payload_layers.get(layer_name)
        layer_present = layer.get("present") if isinstance(layer, dict) else None
        required_layer_fields = (
            {"present", "schema"}
            if layer_name in {"weights", "nonpersistent_buffers"} or layer_present is False
            else {
                "present",
                "schema",
                "members",
                "original_input_digests",
                "capture_state_digests",
            }
        )
        if not isinstance(layer, dict) or set(layer) != required_layer_fields:
            raise ValueError(f"Runnable descriptor payload_layers.{layer_name} fields mismatch.")
        if not isinstance(layer.get("present"), bool):
            raise ValueError(
                f"Runnable descriptor payload_layers.{layer_name}.present must be boolean."
            )
        if layer.get("schema") != expected_schema:
            raise ValueError(
                f"Runnable descriptor payload_layers.{layer_name}.schema must be "
                f"{expected_schema!r}."
            )
    activations = payload_layers["activations"]
    if activations["present"]:
        for field_name in ("members", "original_input_digests", "capture_state_digests"):
            if not isinstance(activations[field_name], list):
                raise ValueError(
                    f"Runnable descriptor payload_layers.activations.{field_name} must be an array."
                )
    preflight = value["preflight"]
    if not isinstance(preflight, dict) or preflight.get("passed") is not True:
        raise ValueError("Runnable descriptor must carry a passed producer preflight.")
    if value["unsupported_sites"] != []:
        raise ValueError("Runnable descriptor must not claim unsupported producer sites.")


def _validate_runnable_payload_entries(manifest: dict[str, Any]) -> None:
    """Cross-check optional runnable payload flags against body entries.

    Parameters
    ----------
    manifest:
        Decoded unified manifest already validated structurally.

    Raises
    ------
    ValueError
        If a weight blob appears without its flag or uses an invalid key.
    """

    run = manifest["run"]
    weights_present = run["payload_layers"]["weights"]["present"]
    nonpersistent_buffers = run["payload_layers"].get(
        "nonpersistent_buffers",
        {"present": False, "schema": "runnable_nonpersistent_buffer_v1"},
    )
    activations = run["payload_layers"]["activations"]
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list):
        raise ValueError("Runnable manifests require a tensors array.")
    weight_entries = [
        entry
        for entry in tensors
        if isinstance(entry, dict) and entry.get("kind") == "runnable_weight"
    ]
    if weight_entries and not weights_present:
        raise ValueError("Runnable weight blobs require payload_layers.weights.present=true.")
    labels: set[str] = set()
    for index, entry in enumerate(weight_entries):
        label = entry.get("label")
        if not isinstance(label, str) or label == "":
            raise ValueError(f"Runnable weight tensor entry {index} requires a canonical label.")
        if label in labels:
            raise ValueError(f"Runnable weight payload repeats canonical state name {label!r}.")
        labels.add(label)
    nonpersistent_buffer_entries = [
        entry
        for entry in tensors
        if isinstance(entry, dict) and entry.get("kind") == "runnable_nonpersistent_buffer"
    ]
    if nonpersistent_buffer_entries and not nonpersistent_buffers["present"]:
        raise ValueError("Runnable non-persistent buffer blobs require their payload declaration.")
    nonpersistent_names = {
        binding.get("state_dict_name")
        for slot in run.get("tensor_slots", [])
        if isinstance(slot, dict)
        and slot.get("role") == "buffer"
        and isinstance((binding := slot.get("state_binding")), dict)
        and binding.get("persistent") is False
        and isinstance(binding.get("state_dict_name"), str)
    }
    entry_names: set[str] = set()
    for index, entry in enumerate(nonpersistent_buffer_entries):
        label = entry.get("label")
        if not isinstance(label, str) or label == "":
            raise ValueError(
                f"Runnable non-persistent buffer entry {index} requires a canonical label."
            )
        if label in entry_names:
            raise ValueError(
                f"Runnable non-persistent buffer payload repeats canonical name {label!r}."
            )
        entry_names.add(label)
    if nonpersistent_buffers["present"] != bool(nonpersistent_names):
        raise ValueError("Runnable non-persistent buffer declaration disagrees with tensor slots.")
    if entry_names != nonpersistent_names:
        raise ValueError("Runnable non-persistent buffer entries disagree with tensor slots.")
    activation_entries = [
        entry
        for entry in tensors
        if isinstance(entry, dict) and entry.get("kind") == "runnable_activation"
    ]
    if activation_entries and not activations["present"]:
        raise ValueError(
            "Runnable activation blobs require payload_layers.activations.present=true."
        )
    entries_by_blob_id = {entry.get("blob_id"): entry for entry in activation_entries}
    valid_slot_ids = {
        slot.get("slot_id")
        for slot in run.get("tensor_slots", [])
        if isinstance(slot, dict) and isinstance(slot.get("slot_id"), str)
    }
    valid_call_ids = {
        call.get("call_id")
        for call in run.get("calls", [])
        if isinstance(call, dict) and isinstance(call.get("call_id"), str)
    }
    member_blob_ids: set[str] = set()
    for index, member in enumerate(activations.get("members", [])):
        if not isinstance(member, dict):
            raise ValueError(f"Runnable activation member {index} must be an object.")
        required = {"blob_id", "slot_id", "call_id", "op_label", "field", "byte_digest"}
        if set(member) != required:
            raise ValueError(f"Runnable activation member {index} fields mismatch.")
        blob_id = member.get("blob_id")
        if not isinstance(blob_id, str) or blob_id == "" or blob_id in member_blob_ids:
            raise ValueError(f"Runnable activation member {index} has invalid/repeated blob_id.")
        if blob_id not in entries_by_blob_id:
            raise ValueError(f"Runnable activation member {index} has no matching tensor entry.")
        if member.get("field") not in {"out", "transformed_out"}:
            raise ValueError(f"Runnable activation member {index} has invalid field.")
        for field_name in ("slot_id", "op_label", "byte_digest"):
            if not isinstance(member.get(field_name), str) or member[field_name] == "":
                raise ValueError(
                    f"Runnable activation member {index} requires non-empty {field_name}."
                )
        if member["slot_id"] not in valid_slot_ids:
            raise ValueError(f"Runnable activation member {index} references an unknown slot.")
        if _SHA256_RE.fullmatch(member["byte_digest"]) is None:
            raise ValueError(f"Runnable activation member {index} byte_digest must be SHA-256.")
        call_id = member.get("call_id")
        if call_id is not None and not isinstance(call_id, str):
            raise ValueError(f"Runnable activation member {index} call_id must be string/null.")
        if call_id is not None and call_id not in valid_call_ids:
            raise ValueError(f"Runnable activation member {index} references an unknown call.")
        if entries_by_blob_id[blob_id].get("label") != member["op_label"]:
            raise ValueError(f"Runnable activation member {index} label disagrees with its blob.")
        member_blob_ids.add(blob_id)
    if member_blob_ids != set(entries_by_blob_id):
        raise ValueError("Runnable activation tensor entries and declared members disagree.")
    _validate_runnable_digest_records(
        activations.get("original_input_digests", []),
        key_name="slot_id",
        valid_keys=valid_slot_ids,
    )
    state_names = {
        binding.get("state_dict_name")
        for slot in run.get("tensor_slots", [])
        if isinstance(slot, dict)
        and isinstance((binding := slot.get("state_binding")), dict)
        and isinstance(binding.get("state_dict_name"), str)
    }
    _validate_runnable_digest_records(
        activations.get("capture_state_digests", []),
        key_name="state_dict_name",
        valid_keys=state_names,
    )


def _validate_runnable_digest_records(
    records: Any,
    *,
    key_name: str,
    valid_keys: set[Any],
) -> None:
    """Validate one unique key-to-byte-digest runnable metadata array.

    Parameters
    ----------
    records:
        JSON-decoded digest record array.
    key_name:
        Record key field to validate.
    valid_keys:
        Descriptor keys that records may reference.

    Raises
    ------
    ValueError
        If record shape, membership, uniqueness, or digest syntax is invalid.
    """

    if not isinstance(records, list):
        raise ValueError("Runnable digest records must be an array.")
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {key_name, "byte_digest"}:
            raise ValueError(f"Runnable digest record {index} fields mismatch.")
        key = record.get(key_name)
        digest = record.get("byte_digest")
        if not isinstance(key, str) or key == "" or key in seen or key not in valid_keys:
            raise ValueError(f"Runnable digest record {index} has invalid/repeated {key_name}.")
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"Runnable digest record {index} byte_digest must be SHA-256.")
        seen.add(key)


def _validate_tlspec_version_ceiling(tlspec_version: int) -> None:
    """Reject portable states newer than the loader can interpret.

    Parameters
    ----------
    tlspec_version:
        Portable I/O state format version declared by the manifest.

    Raises
    ------
    ValueError
        If the declared version is newer than this runtime supports.
    """

    from .._io import TLSPEC_VERSION

    if tlspec_version > TLSPEC_VERSION:
        raise ValueError(
            f"Bundle uses tlspec_version={tlspec_version}, but this runtime only supports "
            f"{TLSPEC_VERSION}."
        )


# JSON Schema keywords enforced by ``_validate_schema_properties``. This is a
# deliberately narrow subset -- only the keywords the shipped
# ``schemas/tlspec_manifest_v*.json`` files actually use for scalar/array
# constraints (``type``, ``enum``, ``const``, ``minimum``, ``minLength``,
# ``pattern``, ``uniqueItems``, ``items``, ``properties``, ``required``).
# ``format`` is intentionally NOT enforced: JSON Schema draft 2020-12
# treats ``format`` as annotation-only unless the format-assertion
# vocabulary is explicitly enabled, and enforcing it here would exceed what
# the schema's own ``$schema`` declaration promises. ``additionalProperties``
# and ``allOf``/``if``/``then`` are also intentionally out of scope: the
# hand-written field validators above already enforce the conditional
# ``kind``-dependent rules the one ``allOf`` block encodes, and adding a
# generic ``additionalProperties: false`` check risks rejecting
# forward-compatible manifests the hand-written checks currently accept.
_JSON_SCHEMA_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _validate_schema_properties(value: Any, schema: Any, *, path: str) -> None:
    """Recursively enforce a JSON Schema ``properties``/``items`` subtree.

    Parameters
    ----------
    value:
        Decoded manifest value (or nested value) under validation.
    schema:
        Decoded JSON schema fragment describing ``value``.
    path:
        Dotted/bracketed path used in raised error messages.

    Raises
    ------
    ValueError
        If ``value`` violates a ``type``, ``enum``, ``const``, ``minimum``,
        ``minLength``, ``pattern``, ``uniqueItems``, nested ``properties``,
        nested ``required``, or ``items`` constraint declared in ``schema``.
    """

    if not isinstance(schema, dict):
        return

    schema_type = schema.get("type")
    if schema_type is not None:
        _check_json_schema_type(value, schema_type, path=path)

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']!r}; got {value!r}.")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path} must equal {schema['const']!r}; got {value!r}.")
    if (
        "minimum" in schema
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value < schema["minimum"]
    ):
        raise ValueError(f"{path} must be >= {schema['minimum']}; got {value}.")
    if "minLength" in schema and isinstance(value, str) and len(value) < schema["minLength"]:
        raise ValueError(f"{path} must have length >= {schema['minLength']}; got {len(value)}.")
    if (
        "pattern" in schema
        and isinstance(value, str)
        and re.search(schema["pattern"], value) is None
    ):
        raise ValueError(f"{path} does not match pattern {schema['pattern']!r}.")
    if schema.get("uniqueItems") and isinstance(value, list):
        seen: list[Any] = []
        for item in value:
            if item in seen:
                raise ValueError(f"{path} must have unique items; duplicate {item!r}.")
            seen.append(item)

    if isinstance(value, dict):
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, sub_schema in properties.items():
                if key in value:
                    _validate_schema_properties(value[key], sub_schema, path=f"{path}.{key}")
        nested_required = schema.get("required")
        if isinstance(nested_required, list):
            nested_missing = [field for field in nested_required if field not in value]
            if nested_missing:
                raise ValueError(f"{path} missing required fields: {nested_missing}.")
    elif isinstance(value, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_properties(item, items_schema, path=f"{path}[{index}]")


def _check_json_schema_type(value: Any, schema_type: Any, *, path: str) -> None:
    """Enforce a JSON Schema ``type`` keyword against a decoded value.

    Parameters
    ----------
    value:
        Decoded manifest value under validation.
    schema_type:
        Either a single JSON Schema type name or a list of allowed names.
    path:
        Dotted/bracketed path used in the raised error message.

    Raises
    ------
    ValueError
        If ``value`` does not match any of the allowed type names.
    """

    allowed_types = schema_type if isinstance(schema_type, list) else [schema_type]
    for allowed_type in allowed_types:
        if allowed_type == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return
        elif allowed_type == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return
        else:
            python_type = _JSON_SCHEMA_TYPE_MAP.get(allowed_type)
            if python_type is not None and isinstance(value, python_type):
                return
    raise ValueError(f"{path} must be of type {schema_type!r}; got {type(value).__name__}.")


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
    runtime_version = runtime.get("version")
    if runtime_version is not None and (
        not isinstance(runtime_version, str) or runtime_version == ""
    ):
        raise ValueError("Manifest backend_runtime.version must be null or a non-empty string.")
    for field_name in ("runtime_config", "device_summary", "compat_policy"):
        if not isinstance(runtime.get(field_name), dict):
            raise ValueError(f"Manifest backend_runtime.{field_name} must be an object.")

    payload_policy = manifest.get("payload_policy")
    if not isinstance(payload_policy, dict):
        raise ValueError("Manifest schema v2 requires object payload_policy.")
    policy = payload_policy.get("policy")
    if policy not in {"full", "audit_only", "metadata_only", "array_payloads"}:
        raise ValueError(
            "Manifest payload_policy.policy must be full, audit_only, metadata_only, "
            "or array_payloads."
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
        "annotation_blob",
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
        "pre_hook_input",
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
        "runnable_activation",
        "runnable_nonpersistent_buffer",
        "runnable_weight",
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
        _validate_optional_body_index_codec_fields(entry, index=index)


def _validate_optional_body_index_codec_fields(entry: dict[str, Any], *, index: int) -> None:
    """Validate optional schema-v2 codec vocabulary on one body index entry.

    Parameters
    ----------
    entry:
        Body index entry.
    index:
        Entry index used for error messages.

    Raises
    ------
    ValueError
        If an optional codec field is malformed.
    """

    for field_name in (
        "logical_backend",
        "codec",
        "logical_dtype",
        "logical_device",
        "transport_backend",
        "transport_dtype",
    ):
        if field_name not in entry:
            continue
        field_value = entry[field_name]
        if not isinstance(field_value, str) or field_value == "":
            raise ValueError(
                f"Manifest body_index[{index}].{field_name} must be a non-empty string."
            )
    if "codec_metadata" in entry and not isinstance(entry["codec_metadata"], dict):
        raise ValueError(f"Manifest body_index[{index}].codec_metadata must be an object.")


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
    "ValidationFailure",
    "ValidationDiagnostic",
    "ValidationReplayState",
    "ValidationReplayStatus",
    "get_validation_failure",
    "get_validation_diagnostics",
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
