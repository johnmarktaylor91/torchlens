"""Persistence for TorchLens intervention specifications."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from enum import Enum
import importlib
import json
import os
from pathlib import Path
import shutil
import uuid
import warnings
from typing import Any, Literal

import torch
from safetensors.torch import load_file, save_file

from .._io.manifest import TensorEntry, sha256_of_file
from .._io.tensor_policy import Ok, is_supported_for_save
from .errors import (
    DirectActivationWriteWarning,
    DirectWriteInExecutableSaveError,
    GraphShapeMismatchError,
    OpaqueCallableInExecutableSaveError,
    ReplayPreconditionError,
    SiteResolutionError,
)
from .helpers import HELPER_REGISTRY_VERSION, helper_from_serialized
from .resolver import (
    function_registry_key_from_callable,
    resolve_function_registry_key,
    resolve_sites,
)
from .types import (
    FrozenTargetSpec,
    FunctionRegistryKey,
    HelperSpec,
    HookSpec,
    InterventionSpec,
    TargetSpec,
    TargetValueSpec,
    TensorSliceSpec,
)

TLSPEC_FORMAT_VERSION = "1"
_SPEC_FILE = "spec.json"
_MANIFEST_FILE = "manifest.json"
_README_FILE = "README.md"
_TENSOR_DIR = "tensors"
_BLOB_TENSOR_KEY = "data"


class SaveLevel(str, Enum):
    """Supported intervention spec save levels."""

    AUDIT = "audit"
    EXECUTABLE_WITH_CALLABLES = "executable_with_callables"
    PORTABLE = "portable"


@dataclass(frozen=True)
class TargetManifestDiff:
    """Diff between saved target manifest labels and a new model log."""

    matched: list[str]
    new_labels: list[str]
    missing_labels: list[str]
    selector_resolution_diffs: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class SpecCompat:
    """Compatibility result for applying a saved spec to a model log."""

    outcome: Literal["EXACT", "COMPATIBLE_WITH_CONFIRMATION", "FAIL"]
    diff: TargetManifestDiff
    targets_resolve_identically: bool


@dataclass(frozen=True)
class _SerializedState:
    """Internal state accumulated during spec serialization."""

    tensor_entries: list[TensorEntry]
    tensor_refs: dict[str, torch.Tensor]


def save_intervention(
    log: Any,
    path: str | Path,
    *,
    level: str | SaveLevel = SaveLevel.EXECUTABLE_WITH_CALLABLES,
    allow_direct_writes: bool = False,
    overwrite: bool = False,
    _write_tensor_blob_fn: Callable[..., TensorEntry] | None = None,
) -> None:
    """Save a model log's intervention recipe to a ``.tlspec`` directory.

    Parameters
    ----------
    log:
        ModelLog-like object whose ``_intervention_spec`` should be persisted.
    path:
        Destination ``.tlspec`` directory path.
    level:
        Save level: ``"audit"``, ``"executable_with_callables"``, or
        ``"portable"``.
    allow_direct_writes:
        Whether executable saves may proceed when direct activation writes were
        detected.
    overwrite:
        Whether an existing target directory may be replaced.
    _write_tensor_blob_fn:
        Test injection hook used to simulate tensor-write crashes.
    """

    save_level = _coerce_save_level(level)
    _enforce_direct_write_policy(log, save_level, allow_direct_writes=allow_direct_writes)
    target_path = Path(path)
    _reject_symlink_path(target_path, context="intervention spec target")
    tmp_path = target_path.parent / f"tmp.{uuid.uuid4().hex}"
    tensor_entries: list[TensorEntry] = []
    state = _SerializedState(tensor_entries=tensor_entries, tensor_refs={})

    try:
        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Intervention spec path already exists: {target_path}")
        tmp_path.mkdir(parents=True)
        (tmp_path / _TENSOR_DIR).mkdir()

        spec = getattr(log, "_intervention_spec", None) or InterventionSpec()
        serialized_spec = _serialize_intervention_spec(spec, save_level, state)
        function_keys = _serialize_function_registry_keys(log)
        target_manifest = _build_target_manifest(log, spec)
        _write_tensor_sidecars(
            tmp_path,
            state.tensor_refs,
            tensor_entries,
            write_tensor_blob_fn=_write_tensor_blob_fn,
        )

        spec_json = {
            "format_version": TLSPEC_FORMAT_VERSION,
            "helper_registry_version": HELPER_REGISTRY_VERSION,
            "save_level": save_level.value,
            "executable": save_level != SaveLevel.AUDIT and not _spec_has_opaque(serialized_spec),
            "target_manifest": target_manifest,
            "helpers": _collect_helpers(serialized_spec),
            "intervention_spec": serialized_spec,
            "function_registry_keys": function_keys,
        }
        manifest_json = {
            "format_version": TLSPEC_FORMAT_VERSION,
            "tensor_entries": [entry.to_dict() for entry in tensor_entries],
        }

        _write_json_file(tmp_path / _SPEC_FILE, spec_json)
        _write_json_file(tmp_path / _MANIFEST_FILE, manifest_json)
        _write_text_file(tmp_path / _README_FILE, _readme_text(spec_json, tensor_entries))
        _fsync_directory(tmp_path)
        if target_path.exists():
            shutil.rmtree(target_path)
        os.rename(tmp_path, target_path)
        _fsync_directory(target_path.parent)
    except Exception:
        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise


def load_intervention_spec(path: str | Path) -> InterventionSpec:
    """Load an intervention spec from a ``.tlspec`` directory.

    Parameters
    ----------
    path:
        Directory containing ``spec.json`` and tensor sidecars.

    Returns
    -------
    InterventionSpec
        Loaded intervention spec with persistence metadata attached.
    """

    spec_path = Path(path)
    _reject_symlink_path(spec_path, context="intervention spec path")
    data = _read_json_file(spec_path / _SPEC_FILE)
    manifest = _read_json_file(spec_path / _MANIFEST_FILE)
    tensor_entries = [TensorEntry.from_dict(entry) for entry in manifest.get("tensor_entries", [])]
    tensors = _load_tensor_refs(spec_path, tensor_entries)
    spec_payload = data["intervention_spec"]
    spec = _deserialize_intervention_spec(spec_payload, tensors)
    metadata = dict(spec.metadata)
    metadata.update(
        {
            "format_version": data.get("format_version"),
            "helper_registry_version": data.get("helper_registry_version"),
            "save_level": data.get("save_level"),
            "executable": bool(data.get("executable", False)),
            "target_manifest": data.get("target_manifest", []),
            "function_registry_keys": data.get("function_registry_keys", []),
            "loaded_from_tlspec": str(spec_path),
        }
    )
    spec.metadata = metadata
    _verify_loaded_function_keys(data.get("function_registry_keys", []))
    return spec


def check_spec_compat(spec: InterventionSpec, new_log: Any) -> SpecCompat:
    """Check whether a loaded intervention spec targets a new model log.

    Parameters
    ----------
    spec:
        Loaded or in-memory intervention spec.
    new_log:
        Model log to check.

    Returns
    -------
    SpecCompat
        Compatibility classification and target diff.
    """

    target_manifest = list(spec.metadata.get("target_manifest", []))
    graph_hash = getattr(new_log, "graph_shape_hash", None)
    all_saved: set[str] = set()
    all_resolved: set[str] = set()
    selector_diffs: dict[str, dict[str, Any]] = {}
    unresolved = False
    graph_matches = True

    for index, entry in enumerate(target_manifest):
        saved_labels = list(entry.get("resolved_labels", []))
        all_saved.update(saved_labels)
        selector = _target_spec_from_json(entry["selector"])
        selector_key = f"selector_{index}"
        saved_hash = entry.get("graph_shape_hash")
        if saved_hash != graph_hash:
            graph_matches = False
        try:
            resolved_labels = list(resolve_sites(new_log, selector, strict=True).labels())
        except SiteResolutionError as exc:
            selector_diffs[selector_key] = {
                "selector": entry["selector"],
                "saved_labels": saved_labels,
                "resolved_labels": [],
                "error": str(exc),
            }
            unresolved = True
            continue
        all_resolved.update(resolved_labels)
        if resolved_labels != saved_labels:
            selector_diffs[selector_key] = {
                "selector": entry["selector"],
                "saved_labels": saved_labels,
                "resolved_labels": resolved_labels,
            }

    matched = sorted(all_saved & all_resolved)
    new_labels = sorted(all_resolved - all_saved)
    missing_labels = sorted(all_saved - all_resolved)
    targets_identical = not selector_diffs and not new_labels and not missing_labels
    diff = TargetManifestDiff(
        matched=matched,
        new_labels=new_labels,
        missing_labels=missing_labels,
        selector_resolution_diffs=selector_diffs,
    )

    if unresolved or missing_labels:
        outcome: Literal["EXACT", "COMPATIBLE_WITH_CONFIRMATION", "FAIL"] = "FAIL"
    elif targets_identical and graph_matches:
        outcome = "EXACT"
    elif all_saved.issubset(all_resolved) or not graph_matches:
        outcome = "COMPATIBLE_WITH_CONFIRMATION"
    else:
        outcome = "FAIL"

    if outcome == "FAIL" and bool(spec.metadata.get("executable", False)) and not graph_matches:
        raise GraphShapeMismatchError(
            "Saved spec's graph_shape_hash doesn't match target log; refusing to apply at "
            "executable level."
        )
    return SpecCompat(outcome, diff, targets_identical)


def _coerce_save_level(level: str | SaveLevel) -> SaveLevel:
    """Normalize a save-level input.

    Parameters
    ----------
    level:
        String or enum save level.

    Returns
    -------
    SaveLevel
        Normalized enum value.
    """

    return level if isinstance(level, SaveLevel) else SaveLevel(level)


def _reject_symlink_path(path: Path, *, context: str) -> None:
    """Reject symlink paths before reading or writing specs.

    Parameters
    ----------
    path:
        Path to inspect.
    context:
        Human-readable path role.
    """

    if path.is_symlink():
        raise ReplayPreconditionError(f"Refusing to use symlink {context}: {path}")


def _enforce_direct_write_policy(
    log: Any,
    save_level: SaveLevel,
    *,
    allow_direct_writes: bool,
) -> None:
    """Apply Phase 10 direct-write save policy.

    Parameters
    ----------
    log:
        Model log being saved.
    save_level:
        Requested save level.
    allow_direct_writes:
        Whether executable saves may proceed despite direct writes.
    """

    if not getattr(log, "_has_direct_writes", False):
        return
    if save_level == SaveLevel.AUDIT:
        warnings.warn(
            "Direct activation writes are audit-only evidence in saved specs.",
            DirectActivationWriteWarning,
            stacklevel=3,
        )
        return
    if not allow_direct_writes:
        raise DirectWriteInExecutableSaveError(
            "Direct activation writes cannot be saved as executable interventions. "
            "Pass allow_direct_writes=True only if the recipe semantics are intentional."
        )


def _serialize_intervention_spec(
    spec: InterventionSpec,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize an intervention spec to JSON-safe data.

    Parameters
    ----------
    spec:
        Intervention spec to serialize.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe spec payload.
    """

    return {
        "targets": [_target_spec_to_json(target) for target in spec.targets],
        "helper": _serialize_value(spec.helper, save_level, state),
        "value": _serialize_value(spec.value, save_level, state),
        "hook": _serialize_value(spec.hook, save_level, state),
        "target_value_specs": [
            _serialize_target_value_spec(value_spec, save_level, state)
            for value_spec in spec.target_value_specs
        ],
        "hook_specs": [
            _serialize_hook_spec(hook_spec, save_level, state) for hook_spec in spec.hook_specs
        ],
        "records": [asdict(record) for record in spec.records],
        "metadata": _jsonish_metadata(spec.metadata),
    }


def _serialize_target_value_spec(
    value_spec: TargetValueSpec,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a target-value spec.

    Parameters
    ----------
    value_spec:
        Target-value spec.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload.
    """

    return {
        "site_target": _target_spec_to_json(value_spec.site_target),
        "value": _serialize_value(value_spec.value, save_level, state),
        "metadata": _jsonish_metadata(value_spec.metadata),
    }


def _serialize_hook_spec(
    hook_spec: HookSpec,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a hook spec.

    Parameters
    ----------
    hook_spec:
        Hook spec.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        JSON-safe payload.
    """

    helper = hook_spec.helper if hook_spec.helper is not None else None
    hook_value = helper if helper is not None else hook_spec.hook
    return {
        "site_target": _target_spec_to_json(hook_spec.site_target),
        "hook": _serialize_value(hook_value, save_level, state),
        "helper": _serialize_value(helper, save_level, state),
        "handle": hook_spec.handle,
        "metadata": _jsonish_metadata(hook_spec.metadata),
    }


def _serialize_value(value: Any, save_level: SaveLevel, state: _SerializedState) -> Any:
    """Serialize tensors, helpers, callables, and JSON-safe literals.

    Parameters
    ----------
    value:
        Runtime value.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    Any
        JSON-safe serialized value.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, torch.Tensor):
        tensor_id = f"value_{len(state.tensor_refs)}"
        state.tensor_refs[tensor_id] = value.detach().cpu()
        return {"__tensor_ref__": tensor_id}
    if isinstance(value, HelperSpec):
        return {"__helper__": _serialize_helper(value, save_level, state)}
    if isinstance(value, tuple):
        return [_serialize_value(item, save_level, state) for item in value]
    if isinstance(value, list):
        return [_serialize_value(item, save_level, state) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_value(item, save_level, state) for key, item in value.items()}
    if callable(value):
        return {"__callable__": _serialize_callable(value, save_level)}
    return _serialize_opaque(value, save_level)


def _serialize_helper(
    helper: HelperSpec,
    save_level: SaveLevel,
    state: _SerializedState,
) -> dict[str, Any]:
    """Serialize a helper spec with save-level enforcement.

    Parameters
    ----------
    helper:
        Helper spec.
    save_level:
        Requested save level.
    state:
        Serialization tensor state.

    Returns
    -------
    dict[str, Any]
        Serialized helper payload.
    """

    portability = helper.portability
    if save_level == SaveLevel.PORTABLE and portability != "builtin":
        raise OpaqueCallableInExecutableSaveError(
            f"Portable intervention specs cannot save {portability} helper {helper.name!r}."
        )
    if save_level == SaveLevel.EXECUTABLE_WITH_CALLABLES and portability == "opaque_audit":
        raise OpaqueCallableInExecutableSaveError(
            f"Executable intervention specs cannot save opaque helper {helper.name!r}."
        )
    if portability == "import_ref":
        import_path = dict(helper.metadata).get("import_path") or _import_path_for_callable(
            helper.factory
        )
        return {"portability": "import_ref", "name": helper.name, "import_path": import_path}
    if portability == "opaque_audit":
        return {"portability": "opaque_audit", "name": helper.name, "repr": repr(helper)}

    return {
        "portability": "builtin",
        "name": helper.name,
        "kind": helper.kind,
        "args": [_serialize_value(arg, save_level, state) for arg in helper.args],
        "kwargs": {key: _serialize_value(value, save_level, state) for key, value in helper.kwargs},
        "metadata": _jsonish_metadata(dict(helper.metadata)),
    }


def _serialize_callable(value: Callable[..., Any], save_level: SaveLevel) -> dict[str, Any]:
    """Serialize a callable as import-ref or audit-only repr.

    Parameters
    ----------
    value:
        Callable to serialize.
    save_level:
        Requested save level.

    Returns
    -------
    dict[str, Any]
        Callable payload.
    """

    import_path = _import_path_for_callable(value)
    if import_path is not None and _callable_round_trips(value, import_path):
        if save_level == SaveLevel.PORTABLE:
            raise OpaqueCallableInExecutableSaveError(
                f"Portable intervention specs cannot save import-ref callable {import_path}."
            )
        return {"portability": "import_ref", "import_path": import_path, "repr": repr(value)}
    if save_level != SaveLevel.AUDIT:
        raise OpaqueCallableInExecutableSaveError(
            f"Callable {value!r} is opaque and can only be saved at audit level."
        )
    return {"portability": "opaque_audit", "repr": repr(value)}


def _serialize_opaque(value: Any, save_level: SaveLevel) -> dict[str, Any]:
    """Serialize a non-JSON object as audit-only data.

    Parameters
    ----------
    value:
        Opaque value.
    save_level:
        Requested save level.

    Returns
    -------
    dict[str, Any]
        Audit-only payload.
    """

    if save_level != SaveLevel.AUDIT:
        raise OpaqueCallableInExecutableSaveError(
            f"Value {value!r} is not portable and can only be saved at audit level."
        )
    return {"__opaque_audit__": {"type": type(value).__name__, "repr": repr(value)}}


def _deserialize_intervention_spec(
    data: dict[str, Any], tensors: dict[str, torch.Tensor]
) -> InterventionSpec:
    """Deserialize JSON-safe spec data.

    Parameters
    ----------
    data:
        Spec payload from ``spec.json``.
    tensors:
        Loaded tensor refs.

    Returns
    -------
    InterventionSpec
        Runtime intervention spec.
    """

    spec = InterventionSpec(metadata=dict(data.get("metadata", {})))
    spec.targets = [_target_spec_from_json(item) for item in data.get("targets", [])]
    spec.helper = _deserialize_value(data.get("helper"), tensors)
    spec.value = _deserialize_value(data.get("value"), tensors)
    spec.hook = _deserialize_value(data.get("hook"), tensors)
    for item in data.get("target_value_specs", []):
        spec.target_value_specs.append(
            TargetValueSpec(
                site_target=_target_spec_from_json(item["site_target"]),
                value=_deserialize_value(item.get("value"), tensors),
                metadata=dict(item.get("metadata", {})),
            )
        )
    for item in data.get("hook_specs", []):
        helper = _deserialize_value(item.get("helper"), tensors)
        hook = _deserialize_value(item.get("hook"), tensors)
        spec.hook_specs.append(
            HookSpec(
                site_target=_target_spec_from_json(item["site_target"]),
                hook=hook,
                helper=helper if isinstance(helper, HelperSpec) else None,
                handle=item.get("handle"),
                metadata=dict(item.get("metadata", {})),
            )
        )
    return spec


def _deserialize_value(value: Any, tensors: dict[str, torch.Tensor]) -> Any:
    """Deserialize a value from ``spec.json``.

    Parameters
    ----------
    value:
        JSON-decoded value.
    tensors:
        Loaded tensor refs.

    Returns
    -------
    Any
        Runtime value.
    """

    if isinstance(value, dict) and "__tensor_ref__" in value:
        return tensors[str(value["__tensor_ref__"])]
    if isinstance(value, dict) and "__helper__" in value:
        return helper_from_serialized(
            value["__helper__"],
            tensor_loader=lambda tensor_id: tensors[tensor_id],
            import_resolver=_resolve_import_ref,
        )
    if isinstance(value, dict) and "__callable__" in value:
        callable_payload = value["__callable__"]
        if callable_payload["portability"] == "import_ref":
            return _resolve_import_ref(callable_payload["import_path"])
        return HelperSpec(
            helper_name="opaque_audit",
            portability="opaque_audit",
            metadata=(("repr", callable_payload.get("repr", "")), ("executable", False)),
        )
    if isinstance(value, dict) and "__opaque_audit__" in value:
        payload = value["__opaque_audit__"]
        return HelperSpec(
            helper_name=str(payload.get("type", "opaque_audit")),
            portability="opaque_audit",
            metadata=(("repr", payload.get("repr", "")), ("executable", False)),
        )
    if isinstance(value, list):
        return [_deserialize_value(item, tensors) for item in value]
    if isinstance(value, dict):
        return {key: _deserialize_value(item, tensors) for key, item in value.items()}
    return value


def _target_spec_to_json(target: TargetSpec | FrozenTargetSpec) -> dict[str, Any]:
    """Serialize a target spec.

    Parameters
    ----------
    target:
        Target spec.

    Returns
    -------
    dict[str, Any]
        JSON-safe target payload.
    """

    metadata = dict(target.metadata) if isinstance(target.metadata, tuple) else target.metadata
    return {
        "selector_kind": target.selector_kind,
        "selector_value": _selector_value_to_json(target.selector_value),
        "strict": bool(target.strict),
        "slice_spec": asdict(target.slice_spec) if target.slice_spec is not None else None,
        "metadata": _jsonish_metadata(metadata),
    }


def _target_spec_from_json(data: dict[str, Any]) -> TargetSpec:
    """Deserialize a target spec.

    Parameters
    ----------
    data:
        JSON target payload.

    Returns
    -------
    TargetSpec
        Runtime target spec.
    """

    slice_data = data.get("slice_spec")
    slice_spec = TensorSliceSpec(**slice_data) if isinstance(slice_data, dict) else None
    return TargetSpec(
        selector_kind=data["selector_kind"],
        selector_value=_selector_value_from_json(data.get("selector_value")),
        strict=bool(data.get("strict", False)),
        slice_spec=slice_spec,
        metadata=dict(data.get("metadata", {})),
    )


def _selector_value_to_json(value: Any) -> Any:
    """Serialize selector payloads.

    Parameters
    ----------
    value:
        Selector payload.

    Returns
    -------
    Any
        JSON-safe payload.
    """

    if isinstance(value, TargetSpec | FrozenTargetSpec):
        return {"__target_spec__": _target_spec_to_json(value)}
    if isinstance(value, tuple):
        return [_selector_value_to_json(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return {"__repr__": repr(value), "__type__": type(value).__name__}


def _selector_value_from_json(value: Any) -> Any:
    """Deserialize selector payloads.

    Parameters
    ----------
    value:
        JSON-safe selector payload.

    Returns
    -------
    Any
        Runtime selector payload.
    """

    if isinstance(value, dict) and "__target_spec__" in value:
        return _target_spec_from_json(value["__target_spec__"])
    if isinstance(value, list):
        return tuple(_selector_value_from_json(item) for item in value)
    if isinstance(value, dict) and "__repr__" in value:
        return value["__repr__"]
    return value


def _build_target_manifest(log: Any, spec: InterventionSpec) -> list[dict[str, Any]]:
    """Build the saved target manifest for all recipe selectors.

    Parameters
    ----------
    log:
        Source model log.
    spec:
        Intervention spec.

    Returns
    -------
    list[dict[str, Any]]
        Manifest entries.
    """

    targets: list[TargetSpec] = []
    targets.extend(spec.targets)
    targets.extend(value_spec.site_target for value_spec in spec.target_value_specs)
    targets.extend(hook_spec.site_target for hook_spec in spec.hook_specs)
    manifest = []
    for target in targets:
        resolved = resolve_sites(log, target, strict=True)
        manifest.append(
            {
                "selector": _target_spec_to_json(target),
                "resolved_labels": list(resolved.labels()),
                "graph_shape_hash": getattr(log, "graph_shape_hash", None),
                "module_address_normalized": _normalized_module_address(target),
            }
        )
    return manifest


def _normalized_module_address(target: TargetSpec) -> str | None:
    """Return normalized module selector data when applicable.

    Parameters
    ----------
    target:
        Target spec.

    Returns
    -------
    str | None
        Normalized module address.
    """

    if target.selector_kind not in {"module", "in_module"}:
        return None
    return str(target.selector_value).strip(".")


def _serialize_function_registry_keys(log: Any) -> list[dict[str, Any]]:
    """Serialize function registry keys from a model log.

    Parameters
    ----------
    log:
        Model log.

    Returns
    -------
    list[dict[str, Any]]
        Function registry key entries.
    """

    entries: list[dict[str, Any]] = []
    for layer in getattr(log, "layer_list", []):
        key = _function_key_for_layer(layer)
        if key is None:
            continue
        entries.append({"layer_label": str(layer.layer_label), "key": asdict(key)})
    return entries


def _function_key_for_layer(layer: Any) -> FunctionRegistryKey | None:
    """Return or infer a layer's function registry key.

    Parameters
    ----------
    layer:
        Layer pass log-like object.

    Returns
    -------
    FunctionRegistryKey | None
        Function registry key or ``None`` for source nodes.
    """

    template = getattr(layer, "captured_arg_template", None)
    key = getattr(template, "func_id", None)
    if isinstance(key, FunctionRegistryKey):
        return key
    func = getattr(layer, "func_applied", None)
    if func is None:
        return None
    return function_registry_key_from_callable(func)


def _verify_loaded_function_keys(entries: Iterable[dict[str, Any]]) -> None:
    """Fail closed when saved function keys are unresolvable.

    Parameters
    ----------
    entries:
        Serialized function key entries.
    """

    for entry in entries:
        key_data = entry.get("key", {})
        key = FunctionRegistryKey(**key_data)
        resolve_function_registry_key(key)


def _write_tensor_sidecars(
    tmp_path: Path,
    tensor_refs: dict[str, torch.Tensor],
    tensor_entries: list[TensorEntry],
    *,
    write_tensor_blob_fn: Callable[..., TensorEntry] | None,
) -> None:
    """Write safetensors sidecars for serialized tensors.

    Parameters
    ----------
    tmp_path:
        Temporary spec directory.
    tensor_refs:
        Mapping from tensor ID to tensor payload.
    tensor_entries:
        Manifest entry accumulator.
    write_tensor_blob_fn:
        Optional test injection writer.
    """

    writer = write_tensor_blob_fn or _write_tlspec_tensor_blob
    for tensor_id, tensor in tensor_refs.items():
        decision = is_supported_for_save(tensor, strict=True)
        if not isinstance(decision, Ok):
            raise ValueError(f"Unsupported tensor for intervention save {tensor_id}: {decision}")
        tensor_entries.append(
            writer(
                tmp_path=tmp_path,
                blob_id=tensor_id,
                tensor=tensor,
                kind="intervention_value",
                label=tensor_id,
            )
        )


def _write_tlspec_tensor_blob(
    *,
    tmp_path: Path,
    blob_id: str,
    tensor: torch.Tensor,
    kind: str,
    label: str,
) -> TensorEntry:
    """Write one intervention tensor sidecar.

    Parameters
    ----------
    tmp_path:
        Temporary spec directory.
    blob_id:
        Tensor identifier.
    tensor:
        Tensor payload.
    kind:
        Logical tensor kind.
    label:
        Human-readable label.

    Returns
    -------
    TensorEntry
        Manifest entry.
    """

    contiguous = tensor.contiguous()
    relative_path = Path(_TENSOR_DIR) / f"{blob_id}.safetensors"
    blob_path = tmp_path / relative_path
    save_file({_BLOB_TENSOR_KEY: contiguous}, str(blob_path))
    _fsync_file(blob_path)
    return TensorEntry(
        blob_id=blob_id,
        kind=kind,
        label=label,
        relative_path=relative_path.as_posix(),
        backend="safetensors",
        shape=[int(dim) for dim in contiguous.shape],
        dtype=str(contiguous.dtype).replace("torch.", ""),
        device_at_save=str(tensor.device),
        layout=str(contiguous.layout).replace("torch.", ""),
        bytes=int(contiguous.numel() * contiguous.element_size()),
        sha256=sha256_of_file(blob_path),
    )


def _load_tensor_refs(spec_path: Path, entries: list[TensorEntry]) -> dict[str, torch.Tensor]:
    """Load tensor refs from safetensors sidecars.

    Parameters
    ----------
    spec_path:
        Spec directory.
    entries:
        Tensor manifest entries.

    Returns
    -------
    dict[str, torch.Tensor]
        Loaded tensors by blob ID.
    """

    tensors: dict[str, torch.Tensor] = {}
    for entry in entries:
        path = spec_path / entry.relative_path
        if sha256_of_file(path) != entry.sha256:
            raise ReplayPreconditionError(f"Tensor sidecar checksum mismatch: {entry.blob_id}")
        tensors[entry.blob_id] = load_file(str(path))[_BLOB_TENSOR_KEY]
    return tensors


def _collect_helpers(spec_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect helper payloads for the top-level helper index.

    Parameters
    ----------
    spec_payload:
        Serialized intervention spec.

    Returns
    -------
    list[dict[str, Any]]
        Helper payloads.
    """

    helpers: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        """Append helper leaves while walking serialized data."""

        if isinstance(value, dict) and "__helper__" in value:
            helpers.append(value["__helper__"])
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(spec_payload)
    return helpers


def _spec_has_opaque(spec_payload: dict[str, Any]) -> bool:
    """Return whether serialized spec data contains audit-only opaque payloads.

    Parameters
    ----------
    spec_payload:
        Serialized spec.

    Returns
    -------
    bool
        ``True`` when opaque payloads are present.
    """

    encoded = json.dumps(spec_payload, sort_keys=True)
    return "opaque_audit" in encoded or "__opaque_audit__" in encoded


def _jsonish_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe metadata dictionary.

    Parameters
    ----------
    metadata:
        Runtime metadata.

    Returns
    -------
    dict[str, Any]
        JSON-safe metadata.
    """

    return {str(key): _metadata_value(value) for key, value in metadata.items()}


def _metadata_value(value: Any) -> Any:
    """Serialize one metadata value conservatively.

    Parameters
    ----------
    value:
        Runtime metadata value.

    Returns
    -------
    Any
        JSON-safe value.
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list):
        return [_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    return repr(value)


def _import_path_for_callable(value: Any) -> str | None:
    """Return ``module:qualname`` for a callable when available.

    Parameters
    ----------
    value:
        Callable-like object.

    Returns
    -------
    str | None
        Import path or ``None``.
    """

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname:
        return None
    return f"{module}:{qualname}"


def _callable_round_trips(value: Callable[..., Any], import_path: str) -> bool:
    """Return whether an import path resolves to the same callable.

    Parameters
    ----------
    value:
        Callable being serialized.
    import_path:
        Candidate import path.

    Returns
    -------
    bool
        ``True`` when the import path resolves identically.
    """

    try:
        return _resolve_import_ref(import_path) is value
    except (AttributeError, ImportError, ValueError, TypeError):
        return False


def _resolve_import_ref(import_path: str) -> Callable[..., Any]:
    """Resolve a ``module:qualname`` import reference.

    Parameters
    ----------
    import_path:
        Import reference.

    Returns
    -------
    Callable[..., Any]
        Resolved callable.
    """

    module_name, sep, qualname = import_path.partition(":")
    if not sep:
        raise ValueError(f"Invalid import path {import_path!r}")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise TypeError(f"{import_path!r} did not resolve to a callable")
    return obj


def _write_json_file(path: Path, data: dict[str, Any]) -> None:
    """Write and fsync canonical JSON.

    Parameters
    ----------
    path:
        File path.
    data:
        JSON payload.
    """

    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text_file(path: Path, text: str) -> None:
    """Write and fsync text.

    Parameters
    ----------
    path:
        File path.
    text:
        Text payload.
    """

    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read JSON object data.

    Parameters
    ----------
    path:
        JSON file path.

    Returns
    -------
    dict[str, Any]
        Decoded JSON object.
    """

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ReplayPreconditionError(f"{path} must contain a JSON object")
    return data


def _fsync_file(path: Path) -> None:
    """Fsync an existing file.

    Parameters
    ----------
    path:
        File path.
    """

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_directory(path: Path) -> None:
    """Fsync a directory when the platform allows it.

    Parameters
    ----------
    path:
        Directory path.
    """

    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _readme_text(spec_json: dict[str, Any], tensor_entries: list[TensorEntry]) -> str:
    """Build the human-readable spec README.

    Parameters
    ----------
    spec_json:
        Serialized spec payload.
    tensor_entries:
        Tensor manifest entries.

    Returns
    -------
    str
        README text.
    """

    return (
        "# TorchLens intervention spec\n\n"
        f"- format_version: {spec_json['format_version']}\n"
        f"- helper_registry_version: {spec_json['helper_registry_version']}\n"
        f"- save_level: {spec_json['save_level']}\n"
        f"- executable: {spec_json['executable']}\n"
        f"- targets: {len(spec_json['target_manifest'])}\n"
        f"- tensor_sidecars: {len(tensor_entries)}\n"
    )


__all__ = [
    "SaveLevel",
    "SpecCompat",
    "TargetManifestDiff",
    "check_spec_compat",
    "load_intervention_spec",
    "resolve_function_registry_key",
    "save_intervention",
]
