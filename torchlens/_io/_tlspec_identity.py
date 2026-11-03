"""Model identity and backend runtime fields of the unified ``.tlspec`` manifest.

The manifest writer in ``tlspec.py`` builds the envelope and publishes the
artifact; this module derives the fields that describe what was traced: the
model signature and fingerprint (per trace, or combined for a bundle) and the
schema v2 backend fields a non-torch or sparse runnable trace adds.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, Any

import torch

from .._data_classes._trace_missing_attrs import is_live_trace
from ..backends.registry import get_backend_spec
from ._tlspec_json_helpers import _hash_named_tensor_meta, _json_str_or_none

if TYPE_CHECKING:
    from .manifest import TensorEntry
    from .tlspec import TlspecKind

_EMPTY_META_HASH = sha256(repr([]).encode("utf-8")).hexdigest()


def _safe_len(value: Any) -> int:
    """Return ``len(value)`` or zero when unavailable.

    Parameters
    ----------
    value:
        Object to measure.

    Returns
    -------
    int
        Length or zero.
    """

    try:
        return len(value)
    except TypeError:
        return 0


def _model_signature(source: Any, *, kind: TlspecKind) -> str:
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
    model_class_qualname = source._source_class_qualname if is_live_trace(source) else None
    if isinstance(model_class_qualname, str) and model_class_qualname:
        return model_class_qualname
    model = _source_model(source)
    if model is not None:
        model_type = type(model)
        return f"{model_type.__module__}.{model_type.__qualname__}"
    source_type = type(source)
    return f"{source_type.__module__}.{source_type.__qualname__}"


def _model_fingerprint(
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
        return _bundle_fingerprint(source, model_signature)

    source_fingerprint = getattr(source, "_source_bundle_model_fingerprint", None)
    if isinstance(source_fingerprint, dict):
        return dict(source_fingerprint)

    model = _source_model(source)
    if model is not None:
        parameter_hash = _hash_named_tensor_meta(model.named_parameters())
        buffer_hash = _hash_named_tensor_meta(model.named_buffers())
    else:
        raw_parameter_hash = source._provenance_param_hash_quick if is_live_trace(source) else None
        if isinstance(raw_parameter_hash, str) and len(raw_parameter_hash) == 64:
            parameter_hash = raw_parameter_hash
        else:
            parameter_hash = _EMPTY_META_HASH
        buffer_hash = _hash_named_tensor_meta(_carried_buffer_values(source))
    return {
        "parameter_meta_hash": parameter_hash,
        "buffer_meta_hash": buffer_hash,
        "class_qualname": model_signature,
        "extra_metadata": _extra_metadata_for_source(source),
    }


def _carried_buffer_values(source: Any) -> list[tuple[str, Any]]:
    """The registered buffers a trace without a live model still knows.

    The pre-forward value channel (``Trace._buffer_initial_values``) is keyed by every
    registered buffer in model traversal order, each value a tensor carrying the buffer's
    shape and dtype, so a loaded trace recomputes the same buffer fingerprint the live
    capture hashed from ``named_buffers()``. Empty when the channel was not persisted
    (``include_buffer_values=False``), which the manifest discloses.
    """

    captured = getattr(source, "_buffer_initial_values", None)
    if not isinstance(captured, Mapping):
        return []
    return [
        (str(name), value)
        for name, value in captured.items()
        if hasattr(value, "shape") and hasattr(value, "dtype")
    ]


def _bundle_fingerprint(bundle: Any, model_signature: str) -> dict[str, Any]:
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
            signature = _model_signature(member, kind="trace")
            member_fingerprints.append((name, _model_fingerprint(member, signature, kind="trace")))
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


def _source_model(source: Any) -> torch.nn.Module | None:
    """Return the weakly-held source model when available.

    Parameters
    ----------
    source:
        Trace-like source object.

    Returns
    -------
    torch.nn.Module | None
        Source model, or ``None`` when unavailable.
    """

    source_ref = source._source_model_ref if is_live_trace(source) else None
    if source_ref is None:
        return None
    model = source_ref()
    return model if isinstance(model, torch.nn.Module) else None


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
        "model_class_name": _json_str_or_none(
            source._source_class_name if is_live_trace(source) else None
        ),
        "graph_shape_hash": _json_str_or_none(
            source._provenance_fingerprint if is_live_trace(source) else None
        ),
        "input_signature_hash": _json_str_or_none(
            source._provenance_input_signature_hash if is_live_trace(source) else None
        ),
    }


def _backend_v2_fields(
    source: Any, *, backend_name: str, tensor_entries: list[TensorEntry]
) -> dict[str, Any]:
    """Return the schema v2 fields of a non-torch or sparse runnable trace manifest.

    Parameters
    ----------
    source:
        Trace being serialized.
    backend_name:
        Registered backend name.
    tensor_entries:
        Saved tensor entries used to derive payload kinds.

    Returns
    -------
    dict[str, Any]
        Fields that replace the schema v1 values in the trace manifest.
    """

    spec = get_backend_spec(backend_name)
    return {
        "schema_version": 2,
        "backend": backend_name,
        "backend_runtime": _backend_runtime(source, backend_name=backend_name),
        "torch_version": torch.__version__ if backend_name == "torch" else None,
        "model_fingerprint": _backend_model_fingerprint(source, backend_name=backend_name),
        "body_format": spec.serialization_policy.body_format,
        "backward_summary": None,
        "derived_gradient_summary": _derived_gradient_summary(source),
        "payload_policy": _payload_policy(backend_name=backend_name, tensor_entries=tensor_entries),
    }


def _backend_runtime(source: Any, *, backend_name: str) -> dict[str, Any]:
    """Build a schema v2 backend runtime fingerprint.

    Parameters
    ----------
    source:
        Trace being serialized.
    backend_name:
        Registered backend name.

    Returns
    -------
    dict[str, Any]
        JSON-ready backend runtime metadata.
    """

    spec = get_backend_spec(backend_name)
    runtime_name = spec.serialization_policy.runtime_name or backend_name
    runtime_config = source._provenance_backend_config
    if not isinstance(runtime_config, dict):
        runtime_config = {}
    device_summary = getattr(source, "backend_runtime_device_summary", None)
    if not isinstance(device_summary, dict):
        device_summary = {}
    runtime_version = source._provenance_backend_version
    return {
        "name": runtime_name,
        "version": None if runtime_version is None else str(runtime_version),
        "runtime_config": runtime_config,
        "device_summary": device_summary,
        "compat_policy": {"policy": "metadata-only"},
    }


def _payload_policy(
    *,
    backend_name: str,
    tensor_entries: list[TensorEntry],
) -> dict[str, Any]:
    """Build a schema v2 payload policy object.

    Parameters
    ----------
    backend_name:
        Registered backend name.
    tensor_entries:
        Saved tensor entries used to derive payload kinds.

    Returns
    -------
    dict[str, Any]
        JSON-ready payload policy metadata.
    """

    spec = get_backend_spec(backend_name)
    payload_kinds = []
    if spec.capabilities.payload_materialization:
        payload_kinds = sorted({entry.kind for entry in tensor_entries})
    return {
        "policy": spec.serialization_policy.payload_policy,
        "materialization_supported": spec.capabilities.payload_materialization,
        "payload_kinds": payload_kinds,
    }


def _derived_gradient_summary(source: Any) -> dict[str, Any] | None:
    """Build schema-v2 derived-gradient summary metadata.

    Parameters
    ----------
    source:
        Trace being serialized.

    Returns
    -------
    dict[str, Any] | None
        JSON-ready derived-gradient summary, or ``None`` when absent.
    """

    leaf_records = source._autograd_derived_grads
    intermediate_records = getattr(
        getattr(source, "autograd", None), "intermediate_derived_grads", {}
    )
    leaf_count = _safe_len(leaf_records)
    intermediate_count = _safe_len(intermediate_records)
    if leaf_count == 0 and intermediate_count == 0:
        return None
    provenance_kinds = sorted(
        {
            str(record.provenance.get("kind"))
            for accessor in (leaf_records, intermediate_records)
            for record in getattr(accessor, "values", lambda: ())()
            if getattr(record, "provenance", None) is not None
            and record.provenance.get("kind") is not None
        }
    )
    return {
        "num_leaf_records": leaf_count,
        "num_intermediate_records": intermediate_count,
        "provenance_kinds": provenance_kinds,
    }


def _backend_model_fingerprint(source: Any, *, backend_name: str) -> dict[str, Any]:
    """Build a schema v2 backend-owned model fingerprint.

    Parameters
    ----------
    source:
        Trace being serialized.
    backend_name:
        Registered backend name.

    Returns
    -------
    dict[str, Any]
        JSON-ready backend fingerprint.
    """

    model_signature = _model_signature(source, kind="trace")
    graph_shape_hash = source._provenance_fingerprint
    return {
        "backend_fingerprint": {
            "backend": backend_name,
            "model_signature": model_signature,
            "graph_digest": graph_shape_hash or "metadata-only",
            "module_identity_mode": getattr(source, "module_identity_mode", None),
            "param_source": source._provenance_param_source,
        }
    }
