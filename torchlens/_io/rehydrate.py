"""Portable state rehydration for TorchLens model logs.

This module restores scrubbed portable metadata into working TorchLens object
graphs. It rebuilds eager or lazy tensor fields from ``BlobRef`` placeholders,
reconstructs accessors, and supports the expert nested-materialization flow
used by ``torchlens.load(..., materialize_nested=False)``.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Literal, Mapping

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from . import BlobRef, FieldPolicy, TorchLensIOError
from .accessor_rebuild import rebuild_trace_accessors
from .lazy import LazyActivationRef
from .manifest import Manifest, TensorEntry, sha256_of_file
from .payload_codec import materialize_transport_tensor
from .paths import resolve_bundle_blob_path
from ..backends import BackendRuntimeCompatibilityError
from ..data_classes._state_adapter import state_items
from ..data_classes.trace import Trace

_LEGACY_CAPTURE_TRACE_KEYS = {
    "_raw_layer_dict",
    "_raw_layer_labels_list",
    "_layer_counter",
    "_raw_layer_type_counter",
    "_current_func_barcode",
    "_mod_entered",
    "_mod_exited",
    "_mod_call_index",
    "_mod_call_labels",
    "_module_build_data",
    "_module_metadata",
    "_module_forward_args",
    "_module_containment_engine",
    "_exhaustive_module_stack",
    "_grad_fn_strong_refs",
    "_in_exhaustive_pass",
    "_pending_live_fire_records",
}
_TORCH_BACKEND_NAME = "torch"


def rehydrate_trace(
    scrubbed_state: dict[str, Any],
    manifest: Manifest | dict[str, Any],
    bundle_path: str | Path,
    *,
    lazy: bool,
    map_location: str | torch.device,
    materialize_nested: bool,
) -> Trace:
    """Restore a scrubbed portable ``Trace`` state.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed metadata dict returned by :func:`scrub_for_save`.
    manifest:
        Portable manifest describing the on-disk tensor blobs.
    bundle_path:
        Root bundle directory containing the ``blobs/`` subdirectory.
    lazy:
        Whether direct out/grad blob refs should remain lazy.
    map_location:
        Target device passed through to ``safetensors`` materialization.
    materialize_nested:
        Whether nested blob refs inside containers should be materialized when
        ``lazy=True``.

    Returns
    -------
    Trace
        Rehydrated model log.
    """

    state_for_load = dict(scrubbed_state)
    source_version = _source_io_format_version(state_for_load, manifest)
    state_for_load = _normalize_legacy_trace_state(state_for_load, source_version)
    module_accessor_state = state_for_load.pop("_io_module_accessor_state", None)
    portable_key_order = tuple(state_for_load)

    trace = Trace.__new__(Trace)
    trace.__setstate__(state_for_load)
    _apply_manifest_backend(trace, manifest)
    _drop_capture_only_trace_fields(trace)

    if module_accessor_state is not None:
        rebuild_trace_accessors(
            trace,
            module_accessor_state._dict,
            module_accessor_state._list,
            module_accessor_state._pass_dict,
        )

    manifest_index = _build_manifest_index(manifest)
    audit_only_payloads = _manifest_uses_audit_only_payloads(manifest)
    payload_statuses: list[str] = []
    if audit_only_payloads:
        payload_statuses.append("audit_only")
    _rehydrate_object(
        trace,
        manifest_index=manifest_index,
        bundle_path=Path(bundle_path),
        lazy=lazy,
        map_location=map_location,
        materialize_nested=materialize_nested,
        audit_only_payloads=audit_only_payloads,
        payload_statuses=payload_statuses,
        seen=set(),
    )
    _set_payload_load_status(trace, manifest_index, payload_statuses)
    _restore_trace_state_order(trace, portable_key_order)
    return trace


def _source_io_format_version(
    state: dict[str, Any],
    manifest: Manifest | dict[str, Any],
) -> int:
    """Return the serialized I/O version from manifest metadata.

    Parameters
    ----------
    state:
        Scrubbed metadata dict being loaded.
    manifest:
        Portable manifest describing the on-disk bundle.

    Returns
    -------
    int
        Source bundle ``tlspec_version``. Falls back to state metadata for
        plain test fixtures.
    """

    if isinstance(manifest, Manifest):
        return manifest.tlspec_version
    version = manifest.get("tlspec_version", state.get("tlspec_version", 0))
    return int(version) if isinstance(version, int) else 0


def _normalize_legacy_trace_state(state: dict[str, Any], source_version: int) -> dict[str, Any]:
    """Normalize a v3-and-earlier scrubbed Trace state dict into v4 shape.

    v4 dropped these capture-only fields from ``Trace.__dict__``:
    ``_raw_layer_dict``, ``_raw_layer_labels_list``, ``_layer_counter``,
    ``_raw_layer_type_counter``, ``_current_func_barcode``, ``_mod_entered``,
    ``_mod_exited``, ``_mod_call_index``, ``_mod_call_labels``,
    ``_module_build_data``, ``_module_metadata``, ``_module_forward_args``,
    ``_module_containment_engine``, ``_exhaustive_module_stack``,
    ``_grad_fn_strong_refs``, ``_in_exhaustive_pass``, and
    ``_pending_live_fire_records``.

    For v3 artifacts being loaded into v4, strip these keys if present. The
    dropped state was capture-time-only; user-facing data is preserved.

    Parameters
    ----------
    state:
        Scrubbed ``Trace`` state loaded from ``metadata.pkl``.
    source_version:
        Source bundle ``tlspec_version``.

    Returns
    -------
    dict[str, Any]
        State suitable for ``Trace.__setstate__``.
    """

    if source_version >= 4:
        return state
    return {key: value for key, value in state.items() if key not in _LEGACY_CAPTURE_TRACE_KEYS}


def _drop_capture_only_trace_fields(trace: Trace) -> None:
    """Remove capture-only scratch fields that older load defaults may add.

    Parameters
    ----------
    trace:
        Rehydrated trace whose public state should match the v4 shape.
    """

    for field_name in _LEGACY_CAPTURE_TRACE_KEYS:
        trace.__dict__.pop(field_name, None)


def _restore_trace_state_order(trace: Trace, portable_key_order: tuple[str, ...]) -> None:
    """Restore loaded Trace ``__dict__`` order to the serialized metadata order.

    Parameters
    ----------
    trace:
        Rehydrated trace whose state order should be restored.
    portable_key_order:
        Key order from the source portable metadata.
    """

    state = trace.__dict__
    ordered_state = {
        field_name: state[field_name] for field_name in portable_key_order if field_name in state
    }
    ordered_state.update(
        {
            field_name: value
            for field_name, value in state.items()
            if field_name not in ordered_state
        }
    )
    state.clear()
    state.update(ordered_state)


def _apply_manifest_backend(trace: Trace, manifest: Manifest | dict[str, Any]) -> None:
    """Prefer schema-v2 manifest backend metadata over legacy scrubbed state."""

    if isinstance(manifest, Manifest):
        backend_name = getattr(manifest, "_tl_logical_backend", None)
    else:
        backend_name = manifest.get("backend")
    if isinstance(backend_name, str) and backend_name:
        setattr(trace, "backend", backend_name)


def _build_manifest_index(
    manifest: Manifest | dict[str, Any],
) -> Mapping[str, dict[str, Any] | TensorEntry]:
    """Index manifest tensor entries by blob id."""

    if isinstance(manifest, Manifest):
        return {entry.blob_id: entry for entry in manifest.tensors}

    tensors = manifest.get("tensors", [])
    if not isinstance(tensors, list):
        raise TorchLensIOError("Portable manifest must contain a list under 'tensors'.")
    index: dict[str, dict[str, Any]] = {}
    for entry in tensors:
        blob_id = entry.get("blob_id")
        if not isinstance(blob_id, str):
            raise TorchLensIOError("Portable manifest tensor entries must include string blob_id.")
        index[blob_id] = entry
    return index


def _rehydrate_object(
    value: Any,
    *,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    lazy: bool,
    map_location: str | torch.device,
    materialize_nested: bool,
    audit_only_payloads: bool,
    payload_statuses: list[str],
    seen: set[int],
) -> Any:
    """Walk a rehydrated object graph and materialize blob refs in place."""

    if isinstance(value, (str, int, float, bool, type(None), torch.dtype, torch.device, BlobRef)):
        return value
    if isinstance(value, tuple):
        return tuple(
            _rehydrate_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                lazy=lazy,
                map_location=map_location,
                materialize_nested=materialize_nested,
                audit_only_payloads=audit_only_payloads,
                payload_statuses=payload_statuses,
                seen=seen,
            )
            for item in value
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _rehydrate_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                lazy=lazy,
                map_location=map_location,
                materialize_nested=materialize_nested,
                audit_only_payloads=audit_only_payloads,
                payload_statuses=payload_statuses,
                seen=seen,
            )
        return value
    if isinstance(value, OrderedDict):
        for key, item in list(value.items()):
            value[key] = _rehydrate_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                lazy=lazy,
                map_location=map_location,
                materialize_nested=materialize_nested,
                audit_only_payloads=audit_only_payloads,
                payload_statuses=payload_statuses,
                seen=seen,
            )
        return value
    if isinstance(value, defaultdict):
        for key, item in list(value.items()):
            value[key] = _rehydrate_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                lazy=lazy,
                map_location=map_location,
                materialize_nested=materialize_nested,
                audit_only_payloads=audit_only_payloads,
                payload_statuses=payload_statuses,
                seen=seen,
            )
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _rehydrate_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                lazy=lazy,
                map_location=map_location,
                materialize_nested=materialize_nested,
                audit_only_payloads=audit_only_payloads,
                payload_statuses=payload_statuses,
                seen=seen,
            )
        return value
    if isinstance(value, set):
        return {
            _rehydrate_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                lazy=lazy,
                map_location=map_location,
                materialize_nested=materialize_nested,
                audit_only_payloads=audit_only_payloads,
                payload_statuses=payload_statuses,
                seen=seen,
            )
            for item in value
        }

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is None:
        return value

    obj_id = id(value)
    if obj_id in seen:
        return value
    seen.add(obj_id)

    for field_name, field_value in list(state_items(value)):
        if field_name not in spec:
            continue
        policy = spec[field_name]
        if policy == FieldPolicy.BLOB:
            if isinstance(field_value, BlobRef):
                ref_field_name = _lazy_ref_field_name(field_name)
                if ref_field_name is not None:
                    if audit_only_payloads:
                        _assign_rehydrated_field(value, field_name, None)
                        continue
                    tensor_ref = _build_lazy_tensor_ref(
                        field_value,
                        manifest_index,
                        bundle_path,
                        kind=_lazy_ref_kind(field_name),
                    )
                    if tensor_ref is not None:
                        _assign_rehydrated_field(value, ref_field_name, tensor_ref)
                    if lazy:
                        _assign_rehydrated_field(value, field_name, None)
                    else:
                        try:
                            materialized = _materialize_blob_ref(
                                field_value,
                                manifest_index,
                                bundle_path,
                                map_location,
                            )
                        except BackendRuntimeCompatibilityError:
                            payload_statuses.append("audit_only_missing_runtime")
                            materialized = None
                        _assign_rehydrated_field(value, field_name, materialized)
                elif not lazy or field_name in {"transformed_out", "transformed_grad"}:
                    if audit_only_payloads:
                        _assign_rehydrated_field(value, field_name, None)
                        continue
                    try:
                        materialized = _materialize_blob_ref(
                            field_value, manifest_index, bundle_path, map_location
                        )
                    except BackendRuntimeCompatibilityError:
                        payload_statuses.append("audit_only_missing_runtime")
                        materialized = None
                    _assign_rehydrated_field(value, field_name, materialized)
        elif policy == FieldPolicy.BLOB_RECURSIVE:
            if audit_only_payloads or (lazy and not materialize_nested):
                continue
            _assign_rehydrated_field(
                value,
                field_name,
                _materialize_recursive_blob_refs(
                    field_value,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    map_location=map_location,
                    payload_statuses=payload_statuses,
                ),
            )
        elif policy == FieldPolicy.KEEP:
            _assign_rehydrated_field(
                value,
                field_name,
                _rehydrate_object(
                    field_value,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    lazy=lazy,
                    map_location=map_location,
                    materialize_nested=materialize_nested,
                    audit_only_payloads=audit_only_payloads,
                    payload_statuses=payload_statuses,
                    seen=seen,
                ),
            )
    return value


def _assign_rehydrated_field(value: Any, field_name: str, field_value: Any) -> None:
    """Assign a rehydrated field without treating it as a user direct write.

    Parameters
    ----------
    value:
        Object being rehydrated.
    field_name:
        Field name to write.
    field_value:
        Rehydrated field value.
    """

    internal_set = getattr(value, "_internal_set", None)
    if callable(internal_set):
        internal_set(field_name, field_value)
    else:
        setattr(value, field_name, field_value)


def _materialize_recursive_blob_refs(
    value: Any,
    *,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    map_location: str | torch.device,
    payload_statuses: list[str],
) -> Any:
    """Materialize ``BlobRef`` objects inside nested containers and portable objects."""

    if isinstance(value, BlobRef):
        try:
            return _materialize_blob_ref(value, manifest_index, bundle_path, map_location)
        except BackendRuntimeCompatibilityError:
            payload_statuses.append("audit_only_missing_runtime")
            return value
    if isinstance(value, list):
        return [
            _materialize_recursive_blob_refs(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _materialize_recursive_blob_refs(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
            )
            for item in value
        )
    if isinstance(value, OrderedDict):
        return OrderedDict(
            (
                key,
                _materialize_recursive_blob_refs(
                    item,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    map_location=map_location,
                    payload_statuses=payload_statuses,
                ),
            )
            for key, item in value.items()
        )
    if isinstance(value, defaultdict):
        materialized: defaultdict[Any, Any] = defaultdict(value.default_factory)
        for key, item in value.items():
            materialized[key] = _materialize_recursive_blob_refs(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
            )
        return materialized
    if isinstance(value, dict):
        return {
            key: _materialize_recursive_blob_refs(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
            )
            for key, item in value.items()
        }
    if isinstance(value, set):
        return {
            _materialize_recursive_blob_refs(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
            )
            for item in value
        }
    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is not None and type(value).__name__ == "GradientRecord":
        for field_name, field_value in list(state_items(value)):
            if field_name not in spec:
                continue
            _assign_rehydrated_field(
                value,
                field_name,
                _materialize_recursive_blob_refs(
                    field_value,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    map_location=map_location,
                    payload_statuses=payload_statuses,
                ),
            )
        return value
    return value


def _materialize_blob_ref(
    blob_ref: BlobRef,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    map_location: str | torch.device,
) -> Any:
    """Load one payload blob from disk using safetensors and its codec.

    Parameters
    ----------
    blob_ref:
        Portable blob reference to materialize.
    manifest_index:
        Manifest tensor entries indexed by blob id.
    bundle_path:
        Root bundle directory containing the blob files.
    map_location:
        Target device for decoded tensors.

    Returns
    -------
    Any
        Materialized logical payload.
    """

    tensor_ref = _build_lazy_tensor_ref(
        blob_ref,
        manifest_index,
        bundle_path,
        kind=_lazy_ref_kind(blob_ref.kind),
    )
    if tensor_ref is not None:
        return tensor_ref.materialize(map_location=map_location)

    if blob_ref.blob_id not in manifest_index:
        raise TorchLensIOError(f"Manifest is missing blob_id={blob_ref.blob_id}.")
    entry = manifest_index[blob_ref.blob_id]
    relative_path = (
        entry.relative_path
        if isinstance(entry, TensorEntry)
        else entry.get("relative_path", f"blobs/{blob_ref.blob_id}.safetensors")
    )
    blob_path = resolve_bundle_blob_path(bundle_path, relative_path)
    if not blob_path.exists():
        raise TorchLensIOError(f"Tensor blob not found at {blob_path}.")

    observed_sha256 = sha256_of_file(blob_path)
    expected_sha256 = entry.sha256 if isinstance(entry, TensorEntry) else entry.get("sha256")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise TorchLensIOError(
            f"blob at {blob_path} sha256 mismatch; expected {expected_sha256} got {observed_sha256}"
        )

    tensor = _load_safetensors_tensor(blob_path, map_location, entry)
    return materialize_transport_tensor(tensor, entry, map_location=map_location)


def _build_lazy_tensor_ref(
    blob_ref: BlobRef,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    *,
    kind: Literal["out", "grad"],
) -> LazyActivationRef | None:
    """Build a ``LazyActivationRef`` from one manifest entry.

    Parameters
    ----------
    blob_ref:
        Activation blob reference from scrubbed metadata.
    manifest_index:
        Manifest tensor entries indexed by blob id.
    bundle_path:
        Root bundle directory.
    kind:
        Logical tensor kind for the lazy ref.

    Returns
    -------
    LazyActivationRef | None
        Lazy out placeholder, or ``None`` when the manifest entry uses the
        older minimal schema that does not include enough metadata yet.
    """

    entry = _manifest_entry_for_blob_ref(blob_ref, manifest_index)
    if entry is None:
        return None
    return LazyActivationRef(
        blob_id=entry.blob_id,
        shape=tuple(entry.shape),
        dtype=_dtype_from_manifest_string(entry.dtype),
        device_at_save=entry.device_at_save,
        source_bundle_path=bundle_path,
        relative_path=entry.relative_path,
        kind=kind,
        expected_sha256=entry.sha256,
        logical_backend=entry.logical_backend or "torch",
        codec=entry.codec or "torch_safetensors_v1",
        logical_dtype=entry.logical_dtype,
        logical_device=entry.logical_device,
        codec_metadata=entry.codec_metadata,
    )


def _lazy_ref_field_name(field_name: str) -> str | None:
    """Return the corresponding lazy-ref field for a direct blob field.

    Parameters
    ----------
    field_name:
        Direct blob field name on the owning data class.

    Returns
    -------
    str | None
        Matching lazy-ref field name, or ``None`` when not applicable.
    """

    if field_name == "out":
        return "out_ref"
    if field_name == "grad":
        return "grad_ref"
    return None


def _lazy_ref_kind(blob_kind: str) -> Literal["out", "grad"]:
    """Normalize a blob kind into a lazy direct-tensor kind.

    Parameters
    ----------
    blob_kind:
        Blob kind stored in the portable manifest.

    Returns
    -------
    str
        Direct tensor kind expected by ``LazyActivationRef``.
    """

    if blob_kind == "grad":
        return "grad"
    return "out"


def _manifest_entry_for_blob_ref(
    blob_ref: BlobRef,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
) -> TensorEntry | None:
    """Return the manifest entry corresponding to one ``BlobRef``.

    Parameters
    ----------
    blob_ref:
        Blob reference to resolve.
    manifest_index:
        Manifest tensor entries indexed by blob id.

    Returns
    -------
    TensorEntry | None
        Resolved manifest entry, or ``None`` when the manifest only contains the
        older minimal tensor schema.

    Raises
    ------
    TorchLensIOError
        If the blob id is missing from the manifest.
    """

    if blob_ref.blob_id not in manifest_index:
        raise TorchLensIOError(f"Manifest is missing blob_id={blob_ref.blob_id}.")
    entry = manifest_index[blob_ref.blob_id]
    if isinstance(entry, TensorEntry):
        return entry
    required_fields = {
        "backend",
        "shape",
        "dtype",
        "device_at_save",
        "layout",
        "bytes",
        "sha256",
    }
    if not required_fields.issubset(entry.keys()):
        return None
    return TensorEntry.from_dict(entry)


def _dtype_from_manifest_string(dtype_name: str) -> torch.dtype:
    """Resolve a manifest dtype string into a ``torch.dtype``.

    Parameters
    ----------
    dtype_name:
        Manifest dtype name without the ``torch.`` prefix.

    Returns
    -------
    torch.dtype
        Resolved dtype object.

    Raises
    ------
    TorchLensIOError
        If the dtype string is unknown to the runtime.
    """

    dtype_obj = getattr(torch, dtype_name, None)
    if not isinstance(dtype_obj, torch.dtype):
        raise TorchLensIOError(f"Unsupported dtype string in manifest: {dtype_name}.")
    return dtype_obj


def _normalize_map_location(map_location: str | torch.device) -> str:
    """Normalize ``map_location`` to the string form expected by safetensors."""

    return str(map_location)


def _load_safetensors_tensor(
    blob_path: Path,
    map_location: str | torch.device,
    entry: TensorEntry | dict[str, Any],
) -> torch.Tensor:
    """Load a single tensor from one safetensors blob.

    Parameters
    ----------
    blob_path:
        Blob file path to decode.
    map_location:
        Target device for the decoded tensor.
    entry:
        Manifest entry used to choose the physical decode device.

    Returns
    -------
    torch.Tensor
        Decoded tensor payload.

    Raises
    ------
    TorchLensIOError
        If the blob does not contain exactly one tensor.
    """

    logical_backend = _entry_logical_backend(entry)
    device = (
        _normalize_map_location(map_location) if logical_backend == _TORCH_BACKEND_NAME else "cpu"
    )
    try:
        tensor_map = load_file(blob_path, device=device)
    except ImportError as exc:
        raise TorchLensIOError(
            "Portable bundle load requires the safetensors backend. Install safetensors>=0.4."
        ) from exc
    except (OSError, SafetensorError, ValueError) as exc:
        raise TorchLensIOError(f"Failed to materialize blob at {blob_path}.") from exc

    if len(tensor_map) != 1:
        raise TorchLensIOError(f"Expected a single tensor in blob file {blob_path}.")
    return next(iter(tensor_map.values()))


def _set_payload_load_status(
    trace: Trace,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    payload_statuses: list[str],
) -> None:
    """Attach public payload load status metadata to a rehydrated trace."""

    if "audit_only_missing_runtime" in payload_statuses:
        setattr(trace, "payload_load_status", "audit_only_missing_runtime")
        return
    if "audit_only" in payload_statuses:
        setattr(trace, "payload_load_status", "audit_only")
        return
    if any(
        _entry_logical_backend(entry) != _TORCH_BACKEND_NAME for entry in manifest_index.values()
    ):
        setattr(trace, "payload_load_status", "loaded_device_best_effort")
        return
    setattr(trace, "payload_load_status", "loaded")


def _manifest_uses_audit_only_payloads(manifest: Manifest | dict[str, Any]) -> bool:
    """Return whether the manifest declares metadata-only payload loading."""

    if isinstance(manifest, Manifest):
        backend_name = getattr(manifest, "_tl_logical_backend", "torch")
        materializes = getattr(manifest, "_tl_payload_materialization_supported", True)
        return backend_name != _TORCH_BACKEND_NAME and not bool(materializes)
    if manifest.get("backend", _TORCH_BACKEND_NAME) == _TORCH_BACKEND_NAME:
        return False
    payload_policy = manifest.get("payload_policy", {})
    if not isinstance(payload_policy, dict):
        return False
    return not bool(payload_policy.get("materialization_supported", False))


def _entry_logical_backend(entry: dict[str, Any] | TensorEntry) -> str:
    """Return the manifest entry logical backend, defaulting legacy entries to torch."""

    if isinstance(entry, TensorEntry):
        return entry.logical_backend or _TORCH_BACKEND_NAME
    return str(entry.get("logical_backend") or _TORCH_BACKEND_NAME)


def rehydrate_nested(
    trace: Trace,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Replace any remaining nested ``BlobRef`` objects with materialized tensors.

    This function is a no-op unless the ``Trace`` was loaded with
    ``lazy=True, materialize_nested=False``. In the default load mode, nested
    tensors are already materialized.

    Typical workflow:

    >>> import torchlens as tl
    >>> log = tl.load("demo_bundle", lazy=True, materialize_nested=False)
    >>> tl.rehydrate_nested(log)

    Parameters
    ----------
    trace:
        Model log loaded from a portable bundle.
    map_location:
        Target device for the materialized tensors.

    Raises
    ------
    TorchLensIOError
        If the source bundle is unavailable or has drifted since load.
    """

    bundle_path = _source_bundle_path_for_trace(trace)
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        raise TorchLensIOError(f"Source bundle manifest not found at {manifest_path}.")

    expected_manifest_sha256 = getattr(trace, "_source_bundle_manifest_sha256", None)
    if expected_manifest_sha256 is not None:
        observed_manifest_sha256 = sha256_of_file(manifest_path)
        if observed_manifest_sha256 != expected_manifest_sha256:
            raise TorchLensIOError(
                "source bundle manifest has changed since load; materialize refs and retry"
            )

    manifest = Manifest.read(manifest_path)
    manifest_index = _build_manifest_index(manifest)
    payload_statuses: list[str] = []
    _rehydrate_nested_object(
        trace,
        manifest_index=manifest_index,
        bundle_path=bundle_path,
        map_location=map_location,
        payload_statuses=payload_statuses,
        seen=set(),
    )
    if payload_statuses:
        _set_payload_load_status(trace, manifest_index, payload_statuses)


def _rehydrate_nested_object(
    value: Any,
    *,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    map_location: str | torch.device,
    payload_statuses: list[str],
    seen: set[int],
) -> Any:
    """Walk an object graph and materialize only nested ``BlobRef`` fields.

    Parameters
    ----------
    value:
        Object graph node to inspect.
    manifest_index:
        Manifest tensor entries indexed by blob id.
    bundle_path:
        Root bundle directory containing the blob files.
    map_location:
        Target device for decoded tensors.
    seen:
        Identity set used to avoid infinite recursion on shared objects.

    Returns
    -------
    Any
        Original value, potentially with nested fields replaced in place.
    """

    if isinstance(value, (str, int, float, bool, type(None), torch.dtype, torch.device, BlobRef)):
        return value
    if isinstance(value, tuple):
        return tuple(
            _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
                seen=seen,
            )
            for item in value
        )
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
                seen=seen,
            )
        return value
    if isinstance(value, OrderedDict):
        for key, item in list(value.items()):
            value[key] = _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
                seen=seen,
            )
        return value
    if isinstance(value, defaultdict):
        for key, item in list(value.items()):
            value[key] = _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
                seen=seen,
            )
        return value
    if isinstance(value, dict):
        for key, item in list(value.items()):
            value[key] = _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
                seen=seen,
            )
        return value
    if isinstance(value, set):
        return {
            _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
                payload_statuses=payload_statuses,
                seen=seen,
            )
            for item in value
        }

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is None:
        return value

    obj_id = id(value)
    if obj_id in seen:
        return value
    seen.add(obj_id)

    for field_name, field_value in list(state_items(value)):
        if field_name not in spec:
            continue
        policy = spec[field_name]
        if policy == FieldPolicy.BLOB_RECURSIVE:
            _assign_rehydrated_field(
                value,
                field_name,
                _materialize_recursive_blob_refs(
                    field_value,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    map_location=map_location,
                    payload_statuses=payload_statuses,
                ),
            )
        elif policy == FieldPolicy.KEEP:
            _assign_rehydrated_field(
                value,
                field_name,
                _rehydrate_nested_object(
                    field_value,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    map_location=map_location,
                    payload_statuses=payload_statuses,
                    seen=seen,
                ),
            )
    return value


def _source_bundle_path_for_trace(trace: Trace) -> Path:
    """Resolve the source bundle path recorded on a portable-loaded ``Trace``.

    Parameters
    ----------
    trace:
        Model log whose source bundle should be resolved.

    Returns
    -------
    Path
        Source bundle directory.

    Raises
    ------
    TorchLensIOError
        If the model log does not retain a source bundle reference.
    """

    bundle_path = getattr(trace, "_source_bundle_path", None)
    if isinstance(bundle_path, Path):
        return bundle_path

    for layer in getattr(trace, "layer_list", []):
        out_ref = getattr(layer, "out_ref", None)
        if isinstance(out_ref, LazyActivationRef):
            return out_ref.source_bundle_path
        grad_ref = getattr(layer, "grad_ref", None)
        if isinstance(grad_ref, LazyActivationRef):
            return grad_ref.source_bundle_path

    raise TorchLensIOError(
        "Trace does not retain a source bundle path for nested blob rehydration."
    )
