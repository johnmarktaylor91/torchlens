"""Portable state rehydration for TorchLens model logs."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

from . import BlobRef, FieldPolicy, TorchLensIOError
from .accessor_rebuild import rebuild_model_log_accessors
from .lazy import LazyActivationRef
from .manifest import Manifest, TensorEntry
from ..data_classes.model_log import ModelLog


def rehydrate_model_log(
    scrubbed_state: dict[str, Any],
    manifest: Manifest | dict[str, Any],
    bundle_path: str | Path,
    *,
    lazy: bool,
    map_location: str | torch.device,
    materialize_nested: bool,
) -> ModelLog:
    """Restore a scrubbed portable ``ModelLog`` state.

    Parameters
    ----------
    scrubbed_state:
        Scrubbed metadata dict returned by :func:`scrub_for_save`.
    manifest:
        Portable manifest describing the on-disk tensor blobs.
    bundle_path:
        Root bundle directory containing the ``blobs/`` subdirectory.
    lazy:
        Whether direct activation/gradient blob refs should remain lazy.
    map_location:
        Target device passed through to ``safetensors`` materialization.
    materialize_nested:
        Whether nested blob refs inside containers should be materialized when
        ``lazy=True``.

    Returns
    -------
    ModelLog
        Rehydrated model log.
    """

    state_for_load = dict(scrubbed_state)
    module_accessor_state = state_for_load.pop("_io_module_accessor_state", None)

    model_log = ModelLog.__new__(ModelLog)
    model_log.__setstate__(state_for_load)

    if module_accessor_state is not None:
        rebuild_model_log_accessors(
            model_log,
            module_accessor_state._dict,
            module_accessor_state._list,
            module_accessor_state._pass_dict,
        )

    manifest_index = _build_manifest_index(manifest)
    _rehydrate_object(
        model_log,
        manifest_index=manifest_index,
        bundle_path=Path(bundle_path),
        lazy=lazy,
        map_location=map_location,
        materialize_nested=materialize_nested,
        seen=set(),
    )
    return model_log


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

    for field_name, field_value in list(vars(value).items()):
        if field_name not in spec:
            continue
        policy = spec[field_name]
        if policy == FieldPolicy.BLOB:
            if isinstance(field_value, BlobRef):
                if field_name == "activation":
                    activation_ref = _build_lazy_activation_ref(
                        field_value,
                        manifest_index,
                        bundle_path,
                    )
                    if activation_ref is not None:
                        setattr(value, "activation_ref", activation_ref)
                    if lazy:
                        setattr(value, field_name, None)
                    else:
                        setattr(
                            value,
                            field_name,
                            _materialize_blob_ref(
                                field_value,
                                manifest_index,
                                bundle_path,
                                map_location,
                            ),
                        )
                elif not lazy:
                    setattr(
                        value,
                        field_name,
                        _materialize_blob_ref(
                            field_value, manifest_index, bundle_path, map_location
                        ),
                    )
        elif policy == FieldPolicy.BLOB_RECURSIVE:
            if lazy and not materialize_nested:
                continue
            setattr(
                value,
                field_name,
                _materialize_recursive_blob_refs(
                    field_value,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    map_location=map_location,
                ),
            )
        elif policy == FieldPolicy.KEEP:
            setattr(
                value,
                field_name,
                _rehydrate_object(
                    field_value,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    lazy=lazy,
                    map_location=map_location,
                    materialize_nested=materialize_nested,
                    seen=seen,
                ),
            )
    return value


def _materialize_recursive_blob_refs(
    value: Any,
    *,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    map_location: str | torch.device,
) -> Any:
    """Materialize ``BlobRef`` objects inside nested containers."""

    if isinstance(value, BlobRef):
        return _materialize_blob_ref(value, manifest_index, bundle_path, map_location)
    if isinstance(value, list):
        return [
            _materialize_recursive_blob_refs(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
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
            )
        return materialized
    if isinstance(value, dict):
        return {
            key: _materialize_recursive_blob_refs(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                map_location=map_location,
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
            )
            for item in value
        }
    return value


def _materialize_blob_ref(
    blob_ref: BlobRef,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    map_location: str | torch.device,
) -> torch.Tensor:
    """Load one tensor blob from disk using safetensors."""

    if blob_ref.blob_id not in manifest_index:
        raise TorchLensIOError(f"Manifest is missing blob_id={blob_ref.blob_id}.")
    entry = manifest_index[blob_ref.blob_id]
    relative_path = (
        entry.relative_path
        if isinstance(entry, TensorEntry)
        else entry.get("relative_path", f"blobs/{blob_ref.blob_id}.safetensors")
    )
    blob_path = bundle_path / relative_path
    if not blob_path.exists():
        raise TorchLensIOError(f"Tensor blob not found at {blob_path}.")
    tensor_map = load_file(blob_path, device=_normalize_map_location(map_location))
    if len(tensor_map) != 1:
        raise TorchLensIOError(f"Expected a single tensor in blob file {blob_path}.")
    return next(iter(tensor_map.values()))


def _build_lazy_activation_ref(
    blob_ref: BlobRef,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
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

    Returns
    -------
    LazyActivationRef | None
        Lazy activation placeholder, or ``None`` when the manifest entry uses the
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
        kind="activation",
        expected_sha256=entry.sha256,
    )


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
