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
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from . import BlobRef, FieldPolicy, IO_FORMAT_VERSION, TorchLensIOError
from ..data_classes.model_log import Trace

BlobSpec: TypeAlias = tuple[str, torch.Tensor, str, str]

_SIMPLE_KEEP_TYPES = (str, int, float, bool, type(None), torch.dtype, torch.device)
_RAW_INPUT_TEXT_LIMIT = 10_000
_RAW_INPUT_TENSOR_BYTES_LIMIT = 1_000_000
_LOGGER = logging.getLogger(__name__)


@dataclass
class _ScrubOptions:
    """Flags controlling which optional tensor payloads are preserved."""

    include_outs: bool
    include_grads: bool
    include_saved_args: bool
    include_rng_states: bool


def scrub_for_save(
    trace: Trace,
    *,
    include_outs: bool = True,
    include_grads: bool = True,
    include_saved_args: bool = False,
    include_rng_states: bool = False,
) -> tuple[dict[str, Any], list[BlobSpec]]:
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
    tuple[dict[str, Any], list[BlobSpec]]
        Scrubbed top-level state dict plus a list of blob specs
        ``(blob_id, tensor, kind, label)``.
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
    scrubbed_state["io_format_version"] = IO_FORMAT_VERSION

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
    return scrubbed_state, blob_specs


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
        scrubbed_state["_out_transform_repr"] = (
            repr(value.out_postfunc) if value.out_postfunc is not None else None
        )
        scrubbed_state["io_format_version"] = IO_FORMAT_VERSION

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
    if field_name in {"saved_args", "saved_kwargs", "output_versions_per_child"}:
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

    if isinstance(owner, Trace) and field_name == "raw_input":
        return _scrub_raw_input_for_save(
            owner, field_value, options, memo, blob_specs, blob_counter
        )
    if policy in {FieldPolicy.DROP, FieldPolicy.WEAKREF_STRIP}:
        return None
    if policy == FieldPolicy.STRINGIFY:
        return _stringify_value(field_value)
    if policy == FieldPolicy.BLOB:
        return _blobify_tensor_field(owner, field_name, field_value, blob_specs, blob_counter)
    if policy == FieldPolicy.BLOB_RECURSIVE:
        return _blobify_recursive_value(
            owner=owner,
            field_name=field_name,
            value=field_value,
            memo=memo,
            blob_specs=blob_specs,
            blob_counter=blob_counter,
        )
    return _scrub_value(field_value, options, memo, blob_specs, blob_counter)


def _scrub_raw_input_for_save(
    owner: Trace,
    value: Any,
    options: _ScrubOptions,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
) -> Any:
    """Apply ``Trace.save_raw_input`` before portable metadata serialization.

    Parameters
    ----------
    owner:
        Trace carrying the raw-input save policy.
    value:
        Raw user input to serialize.
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
        Scrubbed raw input, a bounded placeholder, or ``None``.
    """

    policy = getattr(owner, "save_raw_input", "small")
    if policy is False:
        return None
    if policy is True:
        return _scrub_value(value, options, memo, blob_specs, blob_counter)
    if policy != "small":
        raise TorchLensIOError("save_raw_input must be 'small', True, or False.")
    return _small_raw_input_value(value)


def _small_raw_input_value(value: Any) -> Any:
    """Return a bounded portable representation of a raw input value.

    Parameters
    ----------
    value:
        Raw user input.

    Returns
    -------
    Any
        Truncated string, small tensor, recursively bounded container, or
        ``None`` when the value is too large or unsupported.
    """

    if value is None:
        return None
    if isinstance(value, str):
        return value[:_RAW_INPUT_TEXT_LIMIT]
    if isinstance(value, torch.Tensor):
        tensor_bytes = value.nelement() * value.element_size()
        if tensor_bytes <= _RAW_INPUT_TENSOR_BYTES_LIMIT:
            return value
        _LOGGER.debug("Dropping raw_input tensor over small-policy cap: %s bytes", tensor_bytes)
        return None
    if isinstance(value, list):
        return [_small_raw_input_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_small_raw_input_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _small_raw_input_value(item) for key, item in value.items()}
    _LOGGER.debug(
        "Dropping unsupported raw_input type under small policy: %s", type(value).__name__
    )
    return None


def _blobify_tensor_field(
    owner: Any,
    field_name: str,
    field_value: Any,
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
) -> Any:
    """Replace a tensor field with a ``BlobRef`` and record its blob spec."""

    if field_value is None:
        return None
    if not isinstance(field_value, torch.Tensor):
        raise TorchLensIOError(
            f"{type(owner).__name__}.{field_name} expected a tensor for portable blobification, "
            f"got {type(field_value).__name__}."
        )
    blob_id = _next_blob_id(blob_counter)
    kind = _blob_kind_for_field(owner, field_name)
    label = _blob_label_for_owner(owner)
    blob_specs.append((blob_id, field_value, kind, label))
    return BlobRef(blob_id=blob_id, kind=kind)


def _blobify_recursive_value(
    *,
    owner: Any,
    field_name: str,
    value: Any,
    memo: dict[int, Any],
    blob_specs: list[BlobSpec],
    blob_counter: list[int],
) -> Any:
    """Blobify tensors recursively inside nested containers."""

    if isinstance(value, BlobRef):
        return value
    if isinstance(value, torch.Tensor):
        return _blobify_tensor_field(owner, field_name, value, blob_specs, blob_counter)
    if isinstance(value, list):
        return [
            _blobify_recursive_value(
                owner=owner,
                field_name=field_name,
                value=item,
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
                memo=memo,
                blob_specs=blob_specs,
                blob_counter=blob_counter,
            )
            for item in value
        }

    spec = getattr(type(value), "PORTABLE_STATE_SPEC", None)
    if spec is not None:
        return _scrub_value(
            value, _ScrubOptions(True, True, True, True), memo, blob_specs, blob_counter
        )
    return _stringify_value(value)


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
    if field_name == "output_versions_per_child":
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
    raise TorchLensIOError(f"No blob kind mapping defined for {type(owner).__name__}.{field_name}.")


def _blob_label_for_owner(owner: Any) -> str:
    """Return the human-readable label stored alongside a blob spec."""

    if hasattr(owner, "layer_label_w_pass") and getattr(owner, "layer_label_w_pass") is not None:
        return str(getattr(owner, "layer_label_w_pass"))
    if hasattr(owner, "call_label") and getattr(owner, "call_label") is not None:
        return str(getattr(owner, "call_label"))
    if hasattr(owner, "address") and getattr(owner, "address") is not None:
        return str(getattr(owner, "address"))
    return type(owner).__name__
