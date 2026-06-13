"""Portable state scrubbing for TorchLens model logs.

This module converts a live ``Trace`` object graph into portable metadata
plus a list of tensor blob specs. It is the save-side counterpart to
rehydration: every class-specific ``PORTABLE_STATE_SPEC`` is applied here so
tensor payloads become ``BlobRef`` placeholders and non-portable live objects
are dropped or stringified before writing ``metadata.pkl``.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch

from . import BlobRef, FieldPolicy, TLSPEC_VERSION, TorchLensIOError
from ..data_classes.trace import Trace

BlobSpec: TypeAlias = tuple[str, torch.Tensor, str, str]

_SIMPLE_KEEP_TYPES = (str, int, float, bool, type(None), torch.dtype, torch.device)
_RAW_INPUT_TEXT_LIMIT = 10_000
_RAW_INPUT_TENSOR_BYTES_LIMIT = 1_000_000
_RAW_OUTPUT_TEXT_LIMIT = _RAW_INPUT_TEXT_LIMIT
_RAW_OUTPUT_TENSOR_BYTES_LIMIT = _RAW_INPUT_TENSOR_BYTES_LIMIT
_RAW_CONTAINER_ITEM_LIMIT = 20
_LOGGER = logging.getLogger(__name__)


@dataclass
class _ScrubOptions:
    """Flags controlling which optional tensor payloads are preserved."""

    include_outs: bool
    include_grads: bool
    include_saved_args: bool
    include_rng_states: bool
    unsupported_tensor_records: list[dict[str, str]] = field(default_factory=list)


def scrub_for_save(
    trace: Trace,
    *,
    include_outs: bool = True,
    include_grads: bool = True,
    include_saved_args: bool = False,
    include_rng_states: bool = False,
) -> tuple[dict[str, Any], list[BlobSpec], list[dict[str, str]]]:
    """Scrub a ``Trace`` into portable metadata plus tensor blob specs.

    Parameters
    ----------
    trace:
        Live model log to scrub.
    include_outs:
        Whether saved outs should be replaced with blob references.
    include_grads:
        Whether grads should be replaced with blob references.
    include_saved_args:
        Whether captured args/kwargs and related tensor payloads should be
        preserved via blob references.
    include_rng_states:
        Whether captured RNG state tensors should be preserved via blob
        references.

    Returns
    -------
    tuple[dict[str, Any], list[BlobSpec], list[dict[str, str]]]
        Scrubbed top-level state dict plus a list of blob specs
        ``(blob_id, tensor, kind, label)`` and unsupported tensor audit records.
    """

    options = _ScrubOptions(
        include_outs=include_outs,
        include_grads=include_grads,
        include_saved_args=include_saved_args,
        include_rng_states=include_rng_states,
    )
    memo: dict[int, Any] = {}
    blob_specs: list[BlobSpec] = []
    blob_counter = [0]

    scrubbed_model = _scrub_value(trace, options, memo, blob_specs, blob_counter)
    if not isinstance(scrubbed_model, Trace):
        raise TorchLensIOError("Portable scrub expected a scrubbed Trace instance.")

    scrubbed_state = dict(scrubbed_model.__dict__)
    scrubbed_state["tlspec_version"] = TLSPEC_VERSION

    module_accessor = getattr(trace, "_module_logs", None)
    if module_accessor is not None:
        scrubbed_state["_io_module_accessor_state"] = _scrub_value(
            module_accessor,
            options,
            memo,
            blob_specs,
            blob_counter,
        )
    else:
        scrubbed_state["_io_module_accessor_state"] = None
    return scrubbed_state, blob_specs, options.unsupported_tensor_records


def _scrub_value(
    value: Any,
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
) -> Any:
    """Recursively scrub a value while preserving shared object identity."""

    if isinstance(value, _SIMPLE_KEEP_TYPES):
        return value
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, BlobRef):
        return value
    if isinstance(value, list):
        return [_scrub_value(item, options, memo, blob_specs, blob_counter) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(item, options, memo, blob_specs, blob_counter) for item in value)
    if isinstance(value, set):
        return {_scrub_value(item, options, memo, blob_specs, blob_counter) for item in value}
    if isinstance(value, OrderedDict):
        return OrderedDict(
            (
                key,
                _scrub_value(item, options, memo, blob_specs, blob_counter),
            )
            for key, item in value.items()
        )
    if isinstance(value, defaultdict):
        return {
            key: _scrub_value(item, options, memo, blob_specs, blob_counter)
            for key, item in value.items()
        }
    if isinstance(value, dict):
        return {
            key: _scrub_value(item, options, memo, blob_specs, blob_counter)
            for key, item in value.items()
        }

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is None:
        return value
    obj_id = id(value)
    if obj_id in memo:
        return memo[obj_id]

    scrubbed_obj = object.__new__(type(value))
    memo[obj_id] = scrubbed_obj
    scrubbed_state: dict[str, Any] = {}
    for field_name, field_value in vars(value).items():
        if field_name not in spec:
            raise TorchLensIOError(
                f"{type(value).__name__}.{field_name} is missing from PORTABLE_STATE_SPEC."
            )
        policy = _effective_policy(value, field_name, spec[field_name], options)
        scrubbed_state[field_name] = _scrub_field(
            owner=value,
            field_name=field_name,
            field_value=field_value,
            policy=policy,
            options=options,
            memo=memo,
            blob_specs=blob_specs,
            blob_counter=blob_counter,
        )

    if isinstance(value, Trace):
        scrubbed_state["_activation_transform_repr"] = (
            repr(value.activation_transform) if value.activation_transform is not None else None
        )
        scrubbed_state["tlspec_version"] = TLSPEC_VERSION

    scrubbed_obj.__dict__.update(scrubbed_state)
    return scrubbed_obj


def _effective_policy(
    owner: Any,
    field_name: str,
    base_policy: FieldPolicy,
    options: _ScrubOptions,
) -> FieldPolicy:
    """Resolve the runtime policy for a field after include-flag overrides."""

    if field_name in {"out", "transformed_out"} and not options.include_outs:
        return FieldPolicy.DROP
    if field_name in {"grad", "transformed_grad"} and not options.include_grads:
        return FieldPolicy.DROP
    if field_name in {"saved_args", "saved_kwargs", "out_versions_by_child"}:
        return FieldPolicy.BLOB_RECURSIVE if options.include_saved_args else FieldPolicy.DROP
    if field_name in {"forward_args", "forward_kwargs"}:
        return FieldPolicy.BLOB_RECURSIVE if options.include_saved_args else FieldPolicy.DROP
    if field_name == "func_rng_states":
        return FieldPolicy.BLOB_RECURSIVE if options.include_rng_states else FieldPolicy.DROP
    return base_policy


def _scrub_field(
    *,
    owner: Any,
    field_name: str,
    field_value: Any,
    policy: FieldPolicy,
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
) -> Any:
    """Scrub one object field according to its effective field policy."""

    if isinstance(owner, Trace) and field_name in {"raw_input", "raw_output"}:
        return _scrub_raw_value_for_save(
            owner,
            field_name,
            field_value,
            options,
            memo,
            blob_specs,
            blob_counter,
        )
    if policy in {FieldPolicy.DROP, FieldPolicy.WEAKREF_STRIP}:
        return None
    if policy == FieldPolicy.STRINGIFY:
        return _stringify_value(field_value)
    if policy == FieldPolicy.BLOB:
        return _blobify_tensor_field(
            owner, field_name, field_value, blob_specs, blob_counter, options
        )
    if policy == FieldPolicy.BLOB_RECURSIVE:
        return _blobify_recursive_value(
            owner=owner,
            field_name=field_name,
            value=field_value,
            options=options,
            memo=memo,
            blob_specs=blob_specs,
            blob_counter=blob_counter,
        )
    return _scrub_value(field_value, options, memo, blob_specs, blob_counter)


def _scrub_raw_value_for_save(
    owner: Trace,
    field_name: str,
    value: Any,
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
) -> Any:
    """Apply a raw-value save policy before portable metadata serialization.

    Parameters
    ----------
    owner:
        Trace carrying the raw-value save policy.
    field_name:
        Raw-value field being serialized.
    value:
        Raw user input or output metadata to serialize.
    options:
        Active scrub options.
    memo:
        Object-identity memo for recursive scrubbing.
    blob_specs:
        Accumulated tensor blob specs.
    blob_counter:
        Mutable blob id counter.

    Returns
    -------
    Any
        Scrubbed raw value, a bounded placeholder, or ``None``.
    """

    policy_name = f"save_{field_name}"
    policy = getattr(owner, policy_name, "small")
    if policy is False:
        return None
    if policy is True:
        return _scrub_value(value, options, memo, blob_specs, blob_counter)
    if policy != "small":
        raise TorchLensIOError(f"{policy_name} must be 'small', True, or False.")
    return _small_raw_value(value, field_name=field_name)


def _small_raw_value(value: Any, *, field_name: str) -> Any:
    """Return a bounded portable representation of a raw value.

    Parameters
    ----------
    value:
        Raw user input or output metadata.
    field_name:
        Raw-value field being serialized.

    Returns
    -------
    Any
        Truncated string, small tensor, recursively bounded container, or
        ``None`` when the value is too large or unsupported.
    """

    if value is None:
        return None
    if isinstance(value, str):
        text_limit = _RAW_OUTPUT_TEXT_LIMIT if field_name == "raw_output" else _RAW_INPUT_TEXT_LIMIT
        return value[:text_limit]
    if isinstance(value, torch.Tensor):
        tensor_bytes = value.nelement() * value.element_size()
        tensor_limit = (
            _RAW_OUTPUT_TENSOR_BYTES_LIMIT
            if field_name == "raw_output"
            else _RAW_INPUT_TENSOR_BYTES_LIMIT
        )
        if tensor_bytes <= tensor_limit:
            return value
        _LOGGER.debug(
            "Dropping %s tensor over small-policy cap: %s bytes", field_name, tensor_bytes
        )
        return None
    if isinstance(value, list):
        return [
            _small_raw_value(item, field_name=field_name)
            for item in value[:_RAW_CONTAINER_ITEM_LIMIT]
        ]
    if isinstance(value, tuple):
        return tuple(
            _small_raw_value(item, field_name=field_name)
            for item in value[:_RAW_CONTAINER_ITEM_LIMIT]
        )
    if isinstance(value, dict):
        return {
            key: _small_raw_value(item, field_name=field_name)
            for key, item in list(value.items())[:_RAW_CONTAINER_ITEM_LIMIT]
        }
    _LOGGER.debug(
        "Dropping unsupported %s type under small policy: %s",
        field_name,
        type(value).__name__,
    )
    return None


def _blobify_tensor_field(
    owner: Any,
    field_name: str,
    field_value: Any,
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
    options: _ScrubOptions,
) -> Any:
    """Replace a tensor field with a ``BlobRef`` and record its blob spec."""

    if field_value is None:
        return None
    if _is_mlx_array(field_value):
        options.unsupported_tensor_records.append(
            {
                "owner_type": type(owner).__name__,
                "owner_label": _blob_label_for_owner(owner),
                "field": field_name,
                "kind": _blob_kind_for_field(owner, field_name),
                "reason": "mlx_array_audit_null",
            }
        )
        return None
    if not isinstance(field_value, torch.Tensor):
        raise TorchLensIOError(
            f"{type(owner).__name__}.{field_name} expected a tensor for portable blobification, "
            f"got {type(field_value).__name__}."
        )
    blob_id = _next_blob_id(blob_counter)
    kind = _blob_kind_for_field(owner, field_name)
    label = _blob_label_for_owner(owner)
    tensor_payload = (
        field_value.detach() if isinstance(field_value, torch.nn.Parameter) else field_value
    )
    blob_specs.append((blob_id, tensor_payload, kind, label))
    return BlobRef(blob_id=blob_id, kind=kind)


def _blobify_recursive_value(
    *,
    owner: Any,
    field_name: str,
    value: Any,
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
) -> Any:
    """Blobify tensors recursively inside nested containers."""

    if isinstance(value, _SIMPLE_KEEP_TYPES):
        return value
    if isinstance(value, torch.Size):
        return tuple(value)
    if isinstance(value, BlobRef):
        return value
    if isinstance(value, torch.Tensor):
        return _blobify_tensor_field(owner, field_name, value, blob_specs, blob_counter, options)
    if isinstance(value, list):
        return [
            _blobify_recursive_value(
                owner=owner,
                field_name=field_name,
                value=item,
                options=options,
                memo=memo,
                blob_specs=blob_specs,
                blob_counter=blob_counter,
            )
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _blobify_recursive_value(
                owner=owner,
                field_name=field_name,
                value=item,
                options=options,
                memo=memo,
                blob_specs=blob_specs,
                blob_counter=blob_counter,
            )
            for item in value
        )
    if isinstance(value, OrderedDict):
        return OrderedDict(
            (
                key,
                _blobify_recursive_value(
                    owner=owner,
                    field_name=field_name,
                    value=item,
                    options=options,
                    memo=memo,
                    blob_specs=blob_specs,
                    blob_counter=blob_counter,
                ),
            )
            for key, item in value.items()
        )
    if isinstance(value, defaultdict):
        return {
            key: _blobify_recursive_value(
                owner=owner,
                field_name=field_name,
                value=item,
                options=options,
                memo=memo,
                blob_specs=blob_specs,
                blob_counter=blob_counter,
            )
            for key, item in value.items()
        }
    if isinstance(value, dict):
        return {
            key: _blobify_recursive_value(
                owner=owner,
                field_name=field_name,
                value=item,
                options=options,
                memo=memo,
                blob_specs=blob_specs,
                blob_counter=blob_counter,
            )
            for key, item in value.items()
        }
    if isinstance(value, set):
        return {
            _blobify_recursive_value(
                owner=owner,
                field_name=field_name,
                value=item,
                options=options,
                memo=memo,
                blob_specs=blob_specs,
                blob_counter=blob_counter,
            )
            for item in value
        }

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is not None:
        return _scrub_value(value, options, memo, blob_specs, blob_counter)
    return _stringify_value(value)


def _is_mlx_array(value: Any) -> bool:
    """Return whether ``value`` is an MLX array without requiring MLX."""

    try:
        import mlx.core as mx
    except ImportError:
        return False
    array_type = getattr(mx, "array", None)
    return array_type is not None and isinstance(value, array_type)


def _stringify_value(value: Any) -> str:
    """Convert a non-portable object into a stable placeholder string."""

    return f"<scrubbed:{type(value).__name__}>"


def _next_blob_id(blob_counter: list[int]) -> str:
    """Allocate the next monotonically increasing zero-padded blob id."""

    blob_counter[0] += 1
    return f"{blob_counter[0]:010d}"


def _blob_kind_for_field(owner: Any, field_name: str) -> str:
    """Map an object field name to the portable manifest tensor kind."""

    if field_name == "out":
        return "out"
    if field_name == "transformed_out":
        return "transformed_out"
    if field_name == "grad":
        return "grad"
    if field_name == "transformed_grad":
        return "transformed_grad"
    if field_name in {"saved_args", "saved_kwargs"}:
        return "captured_arg"
    if field_name == "out_versions_by_child":
        return "child_version"
    if field_name == "func_rng_states":
        return "rng_state"
    if field_name in {"forward_args", "forward_kwargs"}:
        return "module_arg"
    if field_name == "func_config":
        return "func_config"
    if field_name == "custom_attributes":
        return "module_meta"
    if field_name in {"grad_inputs", "grad_outputs"}:
        return "grad_fn_grad"
    if field_name == "_buffer_initial_values":
        return "buffer_initial_value"
    if field_name == "orphan_records":
        return "orphan_payload"
    raise TorchLensIOError(f"No blob kind mapping defined for {type(owner).__name__}.{field_name}.")


def _blob_label_for_owner(owner: Any) -> str:
    """Return the human-readable label stored alongside a blob spec."""

    if hasattr(owner, "label") and getattr(owner, "label") is not None:
        return str(getattr(owner, "label"))
    if hasattr(owner, "call_label") and getattr(owner, "call_label") is not None:
        return str(getattr(owner, "call_label"))
    if hasattr(owner, "address") and getattr(owner, "address") is not None:
        return str(getattr(owner, "address"))
    return type(owner).__name__
