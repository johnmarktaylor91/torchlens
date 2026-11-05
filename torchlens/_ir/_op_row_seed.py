"""The values ``Op`` construction derives and converts, for a fields dict or a row.

``Op.__init__`` fills four reference defaults (``dtype_ref``, ``device_ref``,
``backend_address``, ``resolver_status``) and converts the byte, duration and
FLOP fields to their quantity types. Row ingest (``_postprocess/_row_ingest.py``)
adopts a capture row without a fields dict and must apply the same derivations
exactly once. Both call :func:`seed_op_values`: ``Op.__init__`` with
:data:`NAME_KEYS` over its fields dict, ingest with :func:`layout_keys` over the
row list, so the two paths share one implementation. This module imports no
module of the package's largest import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch

from .._quantities import Bytes, as_bytes, as_duration, as_flops
from .refs import DeviceRef, DtypeRef

__all__: tuple[str, ...] = ()

#: The byte-count fields converted with ``as_bytes`` at construction.
INIT_BYTES_FIELDS: tuple[str, ...] = (
    "activation_memory",
    "transformed_activation_memory",
    "autograd_memory",
    "bytes_delta_at_call",
    "bytes_peak_at_call",
    "gradient_memory",
    "transformed_gradient_memory",
)

#: The reference fields ``Op.__init__`` defaults when they are ``None``.
REF_DEFAULT_FIELDS: tuple[str, ...] = (
    "dtype_ref",
    "device_ref",
    "backend_address",
    "resolver_status",
)


class OpSeedKeys(NamedTuple):
    """Where each seeded value lives: field names (a fields dict) or field ids (a row)."""

    dtype_ref: Any
    dtype: Any
    device_ref: Any
    out: Any
    output_device: Any
    backend_address: Any
    address: Any
    resolver_status: Any
    bytes_fields: tuple[Any, ...]
    param_memory: Any
    func_duration: Any
    flops: tuple[Any, ...]


#: The keys of a fields dict (each key is the field name).
NAME_KEYS = OpSeedKeys(
    dtype_ref="dtype_ref",
    dtype="dtype",
    device_ref="device_ref",
    out="out",
    output_device="output_device",
    backend_address="backend_address",
    address="address",
    resolver_status="resolver_status",
    bytes_fields=INIT_BYTES_FIELDS,
    param_memory="param_memory",
    func_duration="func_duration",
    flops=("flops_forward", "flops_backward"),
)


def layout_keys(fid_by_name: Mapping[str, int]) -> OpSeedKeys:
    """Return :data:`NAME_KEYS` translated to the field ids of a row layout."""

    def fid(key: Any) -> Any:
        """Return the field id (or tuple of ids) for one name key."""

        if isinstance(key, tuple):
            return tuple(fid_by_name[name] for name in key)
        return fid_by_name[key]

    return OpSeedKeys(*(fid(key) for key in NAME_KEYS))


def dtype_ref_or_none(value: Any) -> DtypeRef | None:
    """Return a neutral dtype reference for a dtype-like value."""

    return DtypeRef.from_value(value)


def device_ref_from_metadata(out: Any, output_device: Any) -> DeviceRef | None:
    """Return a neutral device reference from payload or output-device metadata."""

    if isinstance(out, torch.Tensor):
        return DeviceRef.from_value(out.device)
    if output_device in (None, "same"):
        return None
    return DeviceRef.from_value(output_device)


def seed_op_values(values: Any, keys: OpSeedKeys) -> None:
    """Apply ``Op`` construction's reference defaults and quantity conversions in place.

    Parameters
    ----------
    values
        A complete fields dict (with :data:`NAME_KEYS`) or a layout-ordered
        row list (with :func:`layout_keys`); every key must be present.
    keys
        Where each value lives in ``values``.
    """

    if values[keys.dtype_ref] is None:
        values[keys.dtype_ref] = dtype_ref_or_none(values[keys.dtype])
    if values[keys.device_ref] is None:
        values[keys.device_ref] = device_ref_from_metadata(
            values[keys.out], values[keys.output_device]
        )
    if values[keys.backend_address] is None:
        values[keys.backend_address] = values[keys.address]
    if values[keys.resolver_status] is None:
        values[keys.resolver_status] = "resolved"
    for key in keys.bytes_fields:
        values[key] = as_bytes(values[key])
    values[keys.param_memory] = Bytes(values[keys.param_memory] or 0)
    values[keys.func_duration] = as_duration(values[keys.func_duration])
    for key in keys.flops:
        values[key] = as_flops(values[key])
