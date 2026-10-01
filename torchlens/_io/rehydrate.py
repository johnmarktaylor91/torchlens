"""Portable state rehydration for TorchLens model logs.

This module restores scrubbed portable metadata into working TorchLens object
graphs. It rebuilds eager or lazy tensor fields from ``BlobRef`` placeholders,
reconstructs accessors, and supports the expert nested-materialization flow
used by ``torchlens.load(..., materialize_nested=False)``.
"""

from __future__ import annotations

import dataclasses
import types
import weakref
from collections import OrderedDict, defaultdict
from collections.abc import Iterable, Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import SafetensorError
from safetensors.torch import load_file

from ..backends import BackendRuntimeCompatibilityError
from ..data_classes._state_adapter import state_items
from ..data_classes.trace import Trace
from ..ir.workspaces import LEGACY_TRACE_BUILD_STATE_KEYS
from . import BlobRef, FieldPolicy, PayloadLoadHints, TorchLensIOError, prerelease as _prerelease
from ._torch_symbols import torch_attr
from .accessor_rebuild import rebuild_trace_accessors
from .lazy import LazyActivationRef, _file_identity
from .manifest import Manifest, TensorEntry, sha256_of_file
from .paths import resolve_bundle_blob_path, resolve_bundle_blobs_dir
from .payload_codec import materialize_transport_tensor
from .scrub import (
    _RAW_IMAGE_SENTINEL,
    _RAW_INPUT_IMAGE_BYTES_LIMIT,
    _RAW_INPUT_IMAGE_MAX_EDGE,
    _pin_in_memo,
)
from .state_contract import governed_artifact_load
from .state_keys import invalidate_static_class_attr_cache, static_class_attr

_LEGACY_CAPTURE_TRACE_KEYS = {
    *LEGACY_TRACE_BUILD_STATE_KEYS,
    "_build_state",
    "_raw_graph_ws",
    "_module_capture_ws",
    "_wrapper_runtime_ws",
    "_pending_live_fire_records",
}
_TORCH_BACKEND_NAME = "torch"
_PORTABLE_WALK_MAX_DEPTH = 200
_REHYDRATE_IN_PROGRESS = object()


def rehydrate_trace(
    scrubbed_state: dict[str, Any],
    manifest: Manifest | dict[str, Any],
    bundle_path: str | Path,
    *,
    lazy: bool,
    map_location: str | torch.device,
    materialize_nested: bool,
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
    resolved_blobs_dir: Path | None = None,
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
    payload_hints:
        Optional backend payload hints used during materialization.
    resolved_blobs_dir:
        Canonical blob containment root already resolved for this load
        operation. When omitted, it is resolved once here.

    Returns
    -------
    Trace
        Rehydrated model log.
    """

    # Load boundary: force every memoized class-owned static lookup to re-validate
    # its class-definition fingerprint before this artifact's fields are assigned.
    invalidate_static_class_attr_cache()
    state_for_load = dict(scrubbed_state)
    module_accessor_state = state_for_load.pop("_io_module_accessor_state", None)
    portable_key_order = tuple(state_for_load)

    trace = Trace.__new__(Trace)
    # The root Trace state is governed artifact bytes: arm the unknown-field
    # partition (state contract, MEMO 3.3) around its ``__setstate__`` exactly
    # like the metadata unpickler arms it for nested records.
    with governed_artifact_load():
        trace.__setstate__(state_for_load)
    trace.raw_input = _rehydrate_small_raw_images(getattr(trace, "raw_input", None))
    trace.raw_output = _rehydrate_small_raw_images(getattr(trace, "raw_output", None))
    _apply_manifest_backend(trace, manifest)
    _drop_capture_only_trace_fields(trace)

    manifest_index = _build_manifest_index(manifest)
    audit_only_payloads = _manifest_uses_audit_only_payloads(manifest)
    payload_statuses: list[str] = []
    if audit_only_payloads:
        payload_statuses.append("audit_only")
    seen: dict[int, Any] = {}
    bundle_root = Path(bundle_path)
    canonical_blobs_dir = (
        resolve_bundle_blobs_dir(bundle_root) if resolved_blobs_dir is None else resolved_blobs_dir
    )
    _rehydrate_object(
        trace,
        manifest_index=manifest_index,
        bundle_path=bundle_root,
        resolved_blobs_dir=canonical_blobs_dir,
        lazy=lazy,
        map_location=map_location,
        materialize_nested=materialize_nested,
        payload_hints=payload_hints,
        audit_only_payloads=audit_only_payloads,
        payload_statuses=payload_statuses,
        seen=seen,
    )
    if module_accessor_state is not None:
        _rehydrate_object(
            module_accessor_state,
            manifest_index=manifest_index,
            bundle_path=bundle_root,
            resolved_blobs_dir=canonical_blobs_dir,
            lazy=lazy,
            map_location=map_location,
            materialize_nested=materialize_nested,
            payload_hints=payload_hints,
            audit_only_payloads=audit_only_payloads,
            payload_statuses=payload_statuses,
            seen=seen,
        )

    # F44 injections stage 2: the rows split out of layer_list during
    # __setstate__ live in the session-transient injection state, which the
    # policy-driven walk above never materializes (FieldPolicy.DROP), so
    # their payload references materialize explicitly here -- eagerly even
    # under lazy=True, the injected family is small by construction -- and
    # the records finalize with attestation="unattested" (loading never
    # attests; the replay door is attest_injected_ops).
    from .injection_codec import finalize_loaded_injected_ops

    finalize_loaded_injected_ops(
        trace,
        materialize=lambda blob_ref: _materialize_blob_ref(
            blob_ref,
            manifest_index,
            bundle_root,
            map_location,
            payload_hints,
            canonical_blobs_dir,
        ),
    )

    # rebuild_trace_accessors() MUST run AFTER _rehydrate_object(), not before.
    # accessor_rebuild.py bakes `trace._buffer_initial_values.get(address)`
    # into each Buffer's `_initial_value` at construction time (Buffer.__init__
    # stores it directly; `initial_value` is not a dynamically-recomputed
    # property). `_buffer_initial_values` is a FieldPolicy.BLOB_RECURSIVE field
    # that only _rehydrate_object() resolves from raw BlobRef placeholders into
    # real tensors (for a default eager load; lazy=True/materialize_nested=False
    # intentionally leaves it as BlobRef for the expert nested-materialization
    # flow). Building accessors first captured the still-unresolved BlobRef
    # permanently, so `Buffer.initial_value` returned a raw BlobRef forever
    # after a plain `tl.load()` instead of a torch.Tensor. Nothing
    # rebuild_trace_accessors() reads (trace.layer_list, module_accessor_state)
    # depends on _rehydrate_object() having run first, so this reorder is safe.
    if module_accessor_state is not None:
        rebuild_trace_accessors(
            trace,
            module_accessor_state._dict,
            module_accessor_state._list,
            module_accessor_state._pass_dict,
        )

    serialized_tlspec_version = (
        manifest.tlspec_version
        if isinstance(manifest, Manifest)
        else manifest.get("tlspec_version")
    )
    if isinstance(serialized_tlspec_version, int) and not isinstance(
        serialized_tlspec_version, bool
    ):
        trace.tlspec_version = serialized_tlspec_version

    _bind_conditional_arms(trace)
    _set_payload_load_status(trace, manifest_index, payload_statuses)
    _restore_trace_state_order(trace, portable_key_order)
    return trace


def _bind_conditional_arms(trace: Trace) -> None:
    """Bind loaded conditional-arm convenience accessors to their owning trace.

    Parameters
    ----------
    trace:
        Rehydrated trace whose runtime-only conditional bindings should be restored.
    """

    from ..data_classes.trace import ConditionalAccessor

    accessor = getattr(trace, "conditionals", None)
    if not isinstance(accessor, ConditionalAccessor):
        return
    for conditional in accessor.values():
        for arm_index, arm in enumerate(conditional.arms):
            arm._bind(trace, conditional.id, arm_index)


def _rehydrate_small_raw_images(value: Any) -> Any:
    """Restore bounded raw-image sentinels to PIL images when possible.

    Parameters
    ----------
    value:
        Scrubbed raw input or output value.

    Returns
    -------
    Any
        Value with TorchLens small-image sentinels replaced by PIL images when
        Pillow can decode them; otherwise the original sentinel is retained.
    """

    if isinstance(value, dict) and value.get(_RAW_IMAGE_SENTINEL) is True:
        data = value.get("data")
        if not isinstance(data, bytes):
            return value
        # SECURITY (secF-1). ``data`` is attacker-controlled bytes on a plain
        # ``tl.load()``: a hand-edited ``metadata.pkl`` is NOT bound by the
        # save-time "small" policy, so this sink must re-impose the SAME canonical
        # bounds the writer applies (never the attacker-echoed ``bytes_limit`` /
        # ``max_edge`` fields sitting in ``value``) and fail CLOSED. Without this,
        # a few-hundred-byte blob declaring huge dimensions forced a large
        # ``Image.load()`` allocation (decompression-bomb DoS), and malformed
        # bytes raised an UNCAUGHT Pillow error that crashed the whole load.
        # 1) Byte cap: refuse to even hand oversized bytes to Pillow's C codecs.
        if len(data) > _RAW_INPUT_IMAGE_BYTES_LIMIT:
            return value
        try:
            from PIL import Image
        except ImportError:
            return value
        # 2) Header-only dimension check BEFORE decode: ``Image.open`` reads the
        #    format header (giving ``.size``) without allocating pixel buffers, so
        #    an oversized *declared* image is rejected here -- before ``.load()``
        #    can allocate -- and bounded to the documented save-time max edge.
        # 3) Fail closed: any decoder error (malformed / truncated / unsupported
        #    bytes, or an early decompression-bomb guard) degrades to the inert raw
        #    sentinel dict instead of propagating out of ``tl.load()``.
        try:
            image = Image.open(BytesIO(data))
            width, height = image.size
            if (
                width > _RAW_INPUT_IMAGE_MAX_EDGE
                or height > _RAW_INPUT_IMAGE_MAX_EDGE
                or width < 1
                or height < 1
            ):
                return value
            image.load()
        except Exception:
            return value
        return image
    if isinstance(value, list):
        return [_rehydrate_small_raw_images(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_rehydrate_small_raw_images(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_rehydrate_small_raw_images(item) for item in value)
    if isinstance(value, dict):
        return {key: _rehydrate_small_raw_images(item) for key, item in value.items()}
    return value


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


_REHYDRATE_LEAF = 0
_REHYDRATE_TUPLE = 1
_REHYDRATE_LIST = 2
_REHYDRATE_MAPPING = 3
_REHYDRATE_SET = 4
_REHYDRATE_FROZENSET = 5
_REHYDRATE_OBJECT = 6

_REHYDRATE_LEAF_TYPES = (str, int, float, bool, type(None), torch.dtype, torch.device, BlobRef)
_REHYDRATE_KINDS: weakref.WeakKeyDictionary[type, int] = weakref.WeakKeyDictionary()


def _rebuild_tuple_value(value: tuple[Any, ...], items: Iterable[Any]) -> tuple[Any, ...]:
    """Rebuild a tuple-like container without erasing its public type.

    Parameters
    ----------
    value:
        Source tuple-like container.
    items:
        Rehydrated child values.

    Returns
    -------
    tuple[Any, ...]
        Rebuilt tuple subclass, or a plain tuple when reconstruction is unsupported.
    """

    materialized = tuple(items)
    if isinstance(value, torch.Size):
        return torch.Size(materialized)
    maker = getattr(type(value), "_make", None)
    if callable(maker):
        try:
            return maker(materialized)
        except (TypeError, ValueError):
            return materialized
    if type(value) is not tuple:
        try:
            return type(value)(materialized)
        except (TypeError, ValueError):
            return materialized
    return materialized


def _rehydrate_node_kind(value_type: type) -> int:
    """Classify one node type for :func:`_rehydrate_object`, memoized per type.

    Same branch order as the ``isinstance`` chain it replaces: leaf types (which
    include ``BlobRef``, resolved by the caller, not walked) first, then ``tuple``,
    ``list``, mappings, and ``set``.
    """

    if issubclass(value_type, _REHYDRATE_LEAF_TYPES):
        kind = _REHYDRATE_LEAF
    elif issubclass(value_type, tuple):
        kind = _REHYDRATE_TUPLE
    elif issubclass(value_type, list):
        kind = _REHYDRATE_LIST
    elif issubclass(value_type, dict):
        kind = _REHYDRATE_MAPPING
    elif issubclass(value_type, set):
        kind = _REHYDRATE_SET
    elif issubclass(value_type, frozenset):
        kind = _REHYDRATE_FROZENSET
    else:
        kind = _REHYDRATE_OBJECT
    _REHYDRATE_KINDS[value_type] = kind
    return kind


def _rehydrate_object(
    value: Any,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    resolved_blobs_dir: Path,
    lazy: bool,
    map_location: str | torch.device,
    materialize_nested: bool,
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None,
    audit_only_payloads: bool,
    payload_statuses: list[str],
    seen: dict[int, Any],
    depth: int = 0,
) -> Any:
    """Walk a rehydrated object graph and materialize blob refs in place."""

    if depth > _PORTABLE_WALK_MAX_DEPTH:
        raise TorchLensIOError(
            f"Portable metadata exceeds the maximum depth of {_PORTABLE_WALK_MAX_DEPTH}."
        )

    # One cached type lookup replaces the eight-way ``isinstance`` chain this branch
    # table re-ran for every one of the ~226k nodes a ResNet load walks. The three
    # former mapping branches (``OrderedDict`` / ``defaultdict`` / ``dict``) had
    # byte-identical in-place bodies and are one branch; immutable frozensets use
    # their own rebuilding branch so nested blob references are not skipped.
    value_type = type(value)
    kind = _REHYDRATE_KINDS.get(value_type)
    if kind is None:
        kind = _rehydrate_node_kind(value_type)
    if kind == _REHYDRATE_LEAF:
        return value
    obj_id = id(value)
    cached = seen.get(obj_id)
    if cached is _REHYDRATE_IN_PROGRESS:
        raise TorchLensIOError("Portable metadata contains a cycle through an immutable container.")
    if cached is not None:
        return cached
    if kind == _REHYDRATE_TUPLE:
        _pin_in_memo(seen, value)
        seen[obj_id] = _REHYDRATE_IN_PROGRESS
        rebuilt_tuple = _rebuild_tuple_value(
            value,
            (
                _rehydrate_object(
                    item,
                    manifest_index,
                    bundle_path,
                    resolved_blobs_dir,
                    lazy,
                    map_location,
                    materialize_nested,
                    payload_hints,
                    audit_only_payloads,
                    payload_statuses,
                    seen,
                    depth + 1,
                )
                for item in value
            ),
        )
        seen[obj_id] = rebuilt_tuple
        return rebuilt_tuple
    if kind == _REHYDRATE_LIST:
        seen[obj_id] = value
        for index, item in enumerate(value):
            value[index] = _rehydrate_object(
                item,
                manifest_index,
                bundle_path,
                resolved_blobs_dir,
                lazy,
                map_location,
                materialize_nested,
                payload_hints,
                audit_only_payloads,
                payload_statuses,
                seen,
                depth + 1,
            )
        return value
    if kind == _REHYDRATE_MAPPING:
        seen[obj_id] = value
        for key, item in list(value.items()):
            value[key] = _rehydrate_object(
                item,
                manifest_index,
                bundle_path,
                resolved_blobs_dir,
                lazy,
                map_location,
                materialize_nested,
                payload_hints,
                audit_only_payloads,
                payload_statuses,
                seen,
                depth + 1,
            )
        return value
    if kind == _REHYDRATE_SET:
        rebuilt_set: set[Any] = set()
        _pin_in_memo(seen, value)
        seen[obj_id] = rebuilt_set
        rebuilt_set.update(
            _rehydrate_object(
                item,
                manifest_index,
                bundle_path,
                resolved_blobs_dir,
                lazy,
                map_location,
                materialize_nested,
                payload_hints,
                audit_only_payloads,
                payload_statuses,
                seen,
                depth + 1,
            )
            for item in value
        )
        return rebuilt_set
    if kind == _REHYDRATE_FROZENSET:
        _pin_in_memo(seen, value)
        seen[obj_id] = _REHYDRATE_IN_PROGRESS
        rebuilt_frozenset = frozenset(
            _rehydrate_object(
                item,
                manifest_index,
                bundle_path,
                resolved_blobs_dir,
                lazy,
                map_location,
                materialize_nested,
                payload_hints,
                audit_only_payloads,
                payload_statuses,
                seen,
                depth + 1,
            )
            for item in value
        )
        seen[obj_id] = rebuilt_frozenset
        return rebuilt_frozenset

    seen[obj_id] = value

    spec = getattr(value_type, "PORTABLE_STATE_SPEC", None)
    if spec is None:
        return value

    for field_name, field_value in list(state_items(value)):
        if field_name not in spec:
            continue
        # Registered pre-release fields materialize under their SWITCHED-ON
        # persisting policy, mirroring scrub's write-side override -- a
        # declared-DROP field saved BLOB_RECURSIVE under the switch must
        # rehydrate its nested BlobRefs, not hand them back dead.
        policy = _prerelease.effective_policy(value_type, field_name, spec[field_name])
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
                        payload_hints=payload_hints,
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
                                payload_hints,
                                resolved_blobs_dir,
                            )
                        except BackendRuntimeCompatibilityError:
                            if _payload_hints_are_explicit(payload_hints):
                                raise
                            payload_statuses.append("audit_only_missing_runtime")
                            materialized = None
                        _assign_rehydrated_field(value, field_name, materialized)
                elif not lazy or field_name in {"transformed_out", "transformed_grad"}:
                    if audit_only_payloads:
                        _assign_rehydrated_field(value, field_name, None)
                        continue
                    try:
                        materialized = _materialize_blob_ref(
                            field_value,
                            manifest_index,
                            bundle_path,
                            map_location,
                            payload_hints,
                            resolved_blobs_dir,
                        )
                    except BackendRuntimeCompatibilityError:
                        if _payload_hints_are_explicit(payload_hints):
                            raise
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
                    resolved_blobs_dir=resolved_blobs_dir,
                    map_location=map_location,
                    payload_hints=payload_hints,
                    payload_statuses=payload_statuses,
                ),
            )
        elif policy == FieldPolicy.KEEP:
            _assign_rehydrated_field(
                value,
                field_name,
                _rehydrate_object(
                    field_value,
                    manifest_index,
                    bundle_path,
                    resolved_blobs_dir,
                    lazy,
                    map_location,
                    materialize_nested,
                    payload_hints,
                    audit_only_payloads,
                    payload_statuses,
                    seen,
                    depth + 1,
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

    # r54 sec_3: resolve the ``_internal_set`` protocol setter off the CLASS, never
    # off the (attacker-controllable) instance ``__dict__``. ``getattr_static``
    # walks the class MRO without triggering descriptors, so a planted instance
    # ``_internal_set`` key can never substitute for the real slotted-class method.
    # Only a genuine class-owned plain method (the sole real one is
    # ``Op._internal_set``) is bound and invoked; non-slotted classes have no
    # ``_internal_set`` on the class and fall through to the frozen/``setattr``
    # branch exactly as before.
    internal_set = static_class_attr(type(value), "_internal_set", None)
    if isinstance(internal_set, types.FunctionType):
        internal_set.__get__(value, type(value))(field_name, field_value)
    elif dataclasses.is_dataclass(value) and getattr(type(value), "__dataclass_params__").frozen:
        object.__setattr__(value, field_name, field_value)
    else:
        setattr(value, field_name, field_value)


def _materialize_recursive_blob_refs(
    value: Any,
    *,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    resolved_blobs_dir: Path,
    map_location: str | torch.device,
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None,
    payload_statuses: list[str],
    active_ids: frozenset[int] = frozenset(),
    depth: int = 0,
) -> Any:
    """Materialize ``BlobRef`` objects inside nested containers and portable objects."""

    if depth > _PORTABLE_WALK_MAX_DEPTH:
        raise TorchLensIOError(
            f"Portable payload exceeds the maximum depth of {_PORTABLE_WALK_MAX_DEPTH}."
        )

    if isinstance(value, BlobRef):
        try:
            return _materialize_blob_ref(
                value,
                manifest_index,
                bundle_path,
                map_location,
                payload_hints,
                resolved_blobs_dir,
            )
        except BackendRuntimeCompatibilityError:
            if _payload_hints_are_explicit(payload_hints):
                raise
            payload_statuses.append("audit_only_missing_runtime")
            return value
    if isinstance(value, (list, tuple, dict, set, frozenset)):
        value_id = id(value)
        if value_id in active_ids:
            raise TorchLensIOError("Portable payload contains a cyclic container.")
        child_active_ids = active_ids | {value_id}
    else:
        child_active_ids = active_ids

    def recurse(item: Any) -> Any:
        """Materialize one child at the next portable-walk depth."""

        return _materialize_recursive_blob_refs(
            item,
            manifest_index=manifest_index,
            bundle_path=bundle_path,
            resolved_blobs_dir=resolved_blobs_dir,
            map_location=map_location,
            payload_hints=payload_hints,
            payload_statuses=payload_statuses,
            active_ids=child_active_ids,
            depth=depth + 1,
        )

    if isinstance(value, list):
        return [recurse(item) for item in value]
    if isinstance(value, tuple):
        return _rebuild_tuple_value(
            value,
            (recurse(item) for item in value),
        )
    if isinstance(value, OrderedDict):
        return OrderedDict((key, recurse(item)) for key, item in value.items())
    if isinstance(value, defaultdict):
        materialized: defaultdict[Any, Any] = defaultdict(value.default_factory)
        for key, item in value.items():
            materialized[key] = recurse(item)
        return materialized
    if isinstance(value, dict):
        return {key: recurse(item) for key, item in value.items()}
    if isinstance(value, set):
        return {recurse(item) for item in value}
    if isinstance(value, frozenset):
        return frozenset(recurse(item) for item in value)
    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is not None and type(value).__name__ == "GradientRecord":
        for field_name, field_value in list(state_items(value)):
            if field_name not in spec:
                continue
            _assign_rehydrated_field(
                value,
                field_name,
                recurse(field_value),
            )
        return value
    return value


def _materialize_blob_ref(
    blob_ref: BlobRef,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    map_location: str | torch.device,
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None,
    resolved_blobs_dir: Path,
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
    payload_hints:
        Optional backend payload hints used during materialization.
    resolved_blobs_dir:
        Canonical blob containment root for this load operation.

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
        payload_hints=payload_hints,
    )
    if tensor_ref is not None:
        return tensor_ref.materialize(
            map_location=map_location,
            payload_hints=payload_hints,
            resolved_blobs_dir=resolved_blobs_dir,
        )

    if blob_ref.blob_id not in manifest_index:
        raise TorchLensIOError(f"Manifest is missing blob_id={blob_ref.blob_id}.")
    entry = manifest_index[blob_ref.blob_id]
    relative_path = (
        entry.relative_path
        if isinstance(entry, TensorEntry)
        else entry.get("relative_path", f"blobs/{blob_ref.blob_id}.safetensors")
    )
    blob_path = resolve_bundle_blob_path(
        bundle_path,
        relative_path,
        resolved_blobs_dir=resolved_blobs_dir,
    )
    if not blob_path.exists():
        raise TorchLensIOError(f"Tensor blob not found at {blob_path}.")

    # The digest and the load are two separate opens of the same path (the load
    # is mmap-backed). Bracket the hash with the file identity and re-check it
    # before the load -- the same R59 TOCTOU discipline as lazy.py -- so a
    # rename-replace between integrity check and load is refused rather than
    # admitting bytes that were never hashed.
    try:
        pre_hash_identity = _file_identity(blob_path.stat())
    except OSError as exc:
        raise TorchLensIOError(f"Failed to access blob at {blob_path}.") from exc
    observed_sha256 = sha256_of_file(blob_path)
    expected_sha256 = entry.sha256 if isinstance(entry, TensorEntry) else entry.get("sha256")
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise TorchLensIOError(
            f"blob at {blob_path} sha256 mismatch; expected {expected_sha256} got {observed_sha256}"
        )
    try:
        pre_load_identity = _file_identity(blob_path.stat())
    except OSError as exc:
        raise TorchLensIOError(f"Failed to access blob at {blob_path}.") from exc
    if pre_load_identity != pre_hash_identity:
        raise TorchLensIOError(f"blob at {blob_path} changed between integrity check and load.")

    tensor = _load_safetensors_tensor(blob_path, map_location, entry)
    return materialize_transport_tensor(
        tensor,
        entry,
        map_location=map_location,
        payload_hints=payload_hints,
    )


def _build_lazy_tensor_ref(
    blob_ref: BlobRef,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    *,
    kind: Literal["out", "grad"],
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None,
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
    payload_hints:
        Optional backend payload hints stored for deferred materialization.

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
        requires_grad=entry.requires_grad,
        logical_backend=entry.logical_backend or "torch",
        codec=entry.codec or "torch_safetensors_v1",
        logical_dtype=entry.logical_dtype,
        logical_device=entry.logical_device,
        codec_metadata=entry.codec_metadata,
        payload_hints=payload_hints,
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

    # r45 secC_1: the manifest ``dtype`` field is attacker-controlled. ``torch_attr`` reads
    # ``torch.__dict__`` directly, so an arbitrary attribute name (``onnx`` / ``_dynamo`` /
    # deprecated ``has_cuda``) fires NO lazy submodule import, NO deprecated ``replacement()``,
    # and leaks NO raw ``ImportError`` before the ``isinstance`` value gate below. Every real
    # dtype is a ``torch.__dict__`` entry and still resolves.
    dtype_obj = torch_attr(dtype_name)
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
    tensor = next(iter(tensor_map.values()))
    requires_grad = (
        entry.requires_grad if isinstance(entry, TensorEntry) else entry.get("requires_grad", False)
    )
    if not isinstance(requires_grad, bool):
        raise TorchLensIOError("Manifest tensor entry 'requires_grad' must be a boolean.")
    if requires_grad:
        tensor.requires_grad_(True)
    return tensor


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
    if _has_pending_lazy_payload(trace):
        # Distinct lazy state (agent memo P0 part 4): "loaded" used to mean
        # both payloads-resident and payloads-behind-lazy-refs, so consumers
        # could not tell a materialized artifact from one whose reads hit
        # disk. Reads on a loaded_lazy trace either materialize sha-verified
        # blobs or refuse typed -- never a silent None.
        setattr(trace, "payload_load_status", "loaded_lazy")
        return
    setattr(trace, "payload_load_status", "loaded")


def _has_pending_lazy_payload(trace: Trace) -> bool:
    """Return whether any op still holds an unmaterialized lazy payload ref.

    Reads through the private slot accessor: a public ``op.out`` read on a
    lazy op MATERIALIZES the blob (the P0 honesty gate), which a status probe
    must never trigger.

    Parameters
    ----------
    trace:
        Rehydrated trace to inspect.

    Returns
    -------
    bool
        Whether an out/grad lazy ref exists whose payload slot is empty.
    """

    for op in getattr(trace, "layer_list", []) or []:
        slot = getattr(op, "_slot", None)
        if not callable(slot):
            continue
        if slot("out_ref") is not None and slot("out") is None:
            return True
        if slot("grad_ref") is not None and slot("grad") is None:
            return True
    return False


def _payload_hints_are_explicit(
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None,
) -> bool:
    """Return whether payload hints request fail-closed backend behavior."""

    if payload_hints is None:
        return False
    jax_hint = payload_hints.jax if isinstance(payload_hints, PayloadLoadHints) else None
    if isinstance(payload_hints, Mapping):
        raw_hint = payload_hints.get("jax")
        if isinstance(raw_hint, Mapping):
            return bool(
                raw_hint.get("sharding") is not None or raw_hint.get("reconstruct_sharding")
            )
        if raw_hint is not None:
            jax_hint = raw_hint
    if jax_hint is None:
        return False
    sharding = getattr(jax_hint, "sharding", None)
    reconstruct = bool(getattr(jax_hint, "reconstruct_sharding", False))
    return sharding is not None or reconstruct


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
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None = None,
) -> None:
    """Replace any remaining nested ``BlobRef`` objects with materialized tensors.

    This function is a no-op unless the ``Trace`` was loaded with
    ``lazy=True, materialize_nested=False``. In the default load mode, nested
    tensors are already materialized.

    Typical workflow:

    >>> import torchlens as tl
    >>> log = tl.load("demo_bundle", lazy=True, materialize_nested=False)
    >>> tl.io.rehydrate_nested(log)

    Parameters
    ----------
    trace:
        Model log loaded from a portable bundle.
    map_location:
        Target device for the materialized tensors.
    payload_hints:
        Optional backend payload hints used during materialization.

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
    resolved_blobs_dir = resolve_bundle_blobs_dir(bundle_path)
    _rehydrate_nested_object(
        trace,
        manifest_index=manifest_index,
        bundle_path=bundle_path,
        resolved_blobs_dir=resolved_blobs_dir,
        map_location=map_location,
        payload_hints=payload_hints,
        payload_statuses=payload_statuses,
        seen={},
    )
    module_logs = getattr(trace, "_module_logs", None)
    if module_logs is not None:
        _rehydrate_nested_object(
            module_logs,
            manifest_index=manifest_index,
            bundle_path=bundle_path,
            resolved_blobs_dir=resolved_blobs_dir,
            map_location=map_location,
            payload_hints=payload_hints,
            payload_statuses=payload_statuses,
            seen={},
        )
    if payload_statuses:
        _set_payload_load_status(trace, manifest_index, payload_statuses)


def _rehydrate_nested_object(
    value: Any,
    *,
    manifest_index: Mapping[str, dict[str, Any] | TensorEntry],
    bundle_path: Path,
    resolved_blobs_dir: Path,
    map_location: str | torch.device,
    payload_hints: PayloadLoadHints | Mapping[str, Any] | None,
    payload_statuses: list[str],
    seen: dict[int, Any],
    depth: int = 0,
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
    resolved_blobs_dir:
        Canonical blob containment root for this materialization operation.
    map_location:
        Target device for decoded tensors.
    payload_hints:
        Optional backend payload hints used during materialization.
    seen:
        Identity set used to avoid infinite recursion on shared objects.

    Returns
    -------
    Any
        Original value, potentially with nested fields replaced in place.
    """

    if depth > _PORTABLE_WALK_MAX_DEPTH:
        raise TorchLensIOError(
            f"Portable metadata exceeds the maximum depth of {_PORTABLE_WALK_MAX_DEPTH}."
        )
    if isinstance(value, (str, int, float, bool, type(None), torch.dtype, torch.device, BlobRef)):
        return value
    obj_id = id(value)
    cached = seen.get(obj_id)
    if cached is _REHYDRATE_IN_PROGRESS:
        raise TorchLensIOError("Portable metadata contains a cycle through an immutable container.")
    if cached is not None:
        return cached
    if isinstance(value, tuple):
        _pin_in_memo(seen, value)
        seen[obj_id] = _REHYDRATE_IN_PROGRESS
        rebuilt_tuple = _rebuild_tuple_value(
            value,
            (
                _rehydrate_nested_object(
                    item,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    resolved_blobs_dir=resolved_blobs_dir,
                    map_location=map_location,
                    payload_hints=payload_hints,
                    payload_statuses=payload_statuses,
                    seen=seen,
                    depth=depth + 1,
                )
                for item in value
            ),
        )
        seen[obj_id] = rebuilt_tuple
        return rebuilt_tuple
    if isinstance(value, list):
        seen[obj_id] = value
        for index, item in enumerate(value):
            value[index] = _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                resolved_blobs_dir=resolved_blobs_dir,
                map_location=map_location,
                payload_hints=payload_hints,
                payload_statuses=payload_statuses,
                seen=seen,
                depth=depth + 1,
            )
        return value
    if isinstance(value, OrderedDict):
        seen[obj_id] = value
        for key, item in list(value.items()):
            value[key] = _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                resolved_blobs_dir=resolved_blobs_dir,
                map_location=map_location,
                payload_hints=payload_hints,
                payload_statuses=payload_statuses,
                seen=seen,
                depth=depth + 1,
            )
        return value
    if isinstance(value, defaultdict):
        seen[obj_id] = value
        for key, item in list(value.items()):
            value[key] = _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                resolved_blobs_dir=resolved_blobs_dir,
                map_location=map_location,
                payload_hints=payload_hints,
                payload_statuses=payload_statuses,
                seen=seen,
                depth=depth + 1,
            )
        return value
    if isinstance(value, dict):
        seen[obj_id] = value
        for key, item in list(value.items()):
            value[key] = _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                resolved_blobs_dir=resolved_blobs_dir,
                map_location=map_location,
                payload_hints=payload_hints,
                payload_statuses=payload_statuses,
                seen=seen,
                depth=depth + 1,
            )
        return value
    if isinstance(value, set):
        rebuilt_set: set[Any] = set()
        _pin_in_memo(seen, value)
        seen[obj_id] = rebuilt_set
        rebuilt_set.update(
            _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                resolved_blobs_dir=resolved_blobs_dir,
                map_location=map_location,
                payload_hints=payload_hints,
                payload_statuses=payload_statuses,
                seen=seen,
                depth=depth + 1,
            )
            for item in value
        )
        return rebuilt_set
    if isinstance(value, frozenset):
        _pin_in_memo(seen, value)
        seen[obj_id] = _REHYDRATE_IN_PROGRESS
        rebuilt_frozenset = frozenset(
            _rehydrate_nested_object(
                item,
                manifest_index=manifest_index,
                bundle_path=bundle_path,
                resolved_blobs_dir=resolved_blobs_dir,
                map_location=map_location,
                payload_hints=payload_hints,
                payload_statuses=payload_statuses,
                seen=seen,
                depth=depth + 1,
            )
            for item in value
        )
        seen[obj_id] = rebuilt_frozenset
        return rebuilt_frozenset

    seen[obj_id] = value

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is None:
        return value

    for field_name, field_value in list(state_items(value)):
        if field_name not in spec:
            continue
        # Same switched-on override as the top-level walk: nested records'
        # registered pre-release fields rehydrate with their persisting policy.
        policy = _prerelease.effective_policy(type(value), field_name, spec[field_name])
        if policy == FieldPolicy.BLOB_RECURSIVE:
            _assign_rehydrated_field(
                value,
                field_name,
                _materialize_recursive_blob_refs(
                    field_value,
                    manifest_index=manifest_index,
                    bundle_path=bundle_path,
                    resolved_blobs_dir=resolved_blobs_dir,
                    map_location=map_location,
                    payload_hints=payload_hints,
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
                    resolved_blobs_dir=resolved_blobs_dir,
                    map_location=map_location,
                    payload_hints=payload_hints,
                    payload_statuses=payload_statuses,
                    seen=seen,
                    depth=depth + 1,
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
