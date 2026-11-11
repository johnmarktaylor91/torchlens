"""The values ``Op`` construction derives and converts, applied to a capture row.

``Op.__init__`` (``_data_classes/op.py``) fills four reference defaults
(``dtype_ref``, ``device_ref``, ``backend_address``, ``resolver_status``) and
converts the byte, duration and FLOP fields to their quantity types. Row
ingest (``_postprocess/_row_ingest.py``) adopts a capture row without a fields
dict or ``Op.__init__`` and applies the same derivations exactly once, with
:func:`seed_op_values` over the row list (:func:`layout_keys`). ``Op.__init__``
keeps its own copy: ``_data_classes`` sits a layer below ``_ir`` and may not
import more from it. ``tests/test_capture_row_ingest.py`` pins the two to the
same fields and the same cells. This module imports no module of the
package's largest import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import torch

from .._quantities import Bytes, as_bytes, as_duration, as_flops
from .refs import DeviceRef, DtypeRef

__all__: tuple[str, ...] = ()

#: The byte-count fields ``Op.__init__`` converts with ``as_bytes`` (its ``_INIT_BYTES_FIELDS``).
INIT_BYTES_FIELDS: tuple[str, ...] = (
    "activation_memory",
    "transformed_activation_memory",
    "autograd_memory",
    "bytes_delta_at_call",
    "bytes_peak_at_call",
    "gradient_memory",
    "transformed_gradient_memory",
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


def seed_op_values(values: Any, keys: OpSeedKeys, cache: dict[Any, Any] | None = None) -> None:
    """Apply ``Op`` construction's reference defaults and quantity conversions in place.

    Parameters
    ----------
    values
        A layout-ordered row list (with :func:`layout_keys`) or a complete
        fields dict (with :data:`NAME_KEYS`); every key must be present.
    keys
        Where each value lives in ``values``.
    cache
        One trace's write-time intern table (row ingest passes its plan's), or
        ``None`` to build every value afresh. With it, equal sources give the
        same value object (:func:`_seed_interned`).
    """

    if cache is not None:
        _seed_interned(values, keys, cache)
        return
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


def _seed_interned(values: Any, keys: OpSeedKeys, cache: dict[Any, Any]) -> None:
    """:func:`seed_op_values` with one trace's intern table (part C1 of compaction interning).

    Each converted value is a pure function of its source's exact type and
    value, so equal sources share one immutable object: ``DtypeRef`` by dtype,
    ``DeviceRef`` by device (an exact ``torch.Tensor`` payload's device, or the
    output-device metadata when there is no payload), ``Bytes`` and ``Flops`` by
    their raw count. A ``Duration`` is shared only for an exact ``int`` zero or
    ``None`` (the placeholder of rows that ran nothing); measured timings stay one
    object per row. Other shapes and unhashable sources take the uncached path.
    A hit is a plain subscript, so it costs no call. The table belongs to one
    ingest of one trace and is dropped with it.
    """

    if values[keys.dtype_ref] is None:
        source = values[keys.dtype]
        key: Any = ("dtype", source.__class__, source)
        try:
            values[keys.dtype_ref] = cache[key]
        except (KeyError, TypeError):
            values[keys.dtype_ref] = _cached_or_built(cache, key, dtype_ref_or_none, source)
    if values[keys.device_ref] is None:
        out = values[keys.out]
        if out is None or out.__class__ is torch.Tensor:
            source = values[keys.output_device] if out is None else out.device
            key = ("device", out is None, source.__class__, source)
            try:
                values[keys.device_ref] = cache[key]
            except (KeyError, TypeError):
                values[keys.device_ref] = _cached_or_built(
                    cache, key, _device_ref_of, (out, values[keys.output_device])
                )
        else:
            values[keys.device_ref] = device_ref_from_metadata(out, values[keys.output_device])
    if values[keys.backend_address] is None:
        values[keys.backend_address] = values[keys.address]
    if values[keys.resolver_status] is None:
        values[keys.resolver_status] = "resolved"
    _seed_interned_quantities(values, keys, cache)


def _seed_interned_quantities(values: Any, keys: OpSeedKeys, cache: dict[Any, Any]) -> None:
    """Convert the byte, duration and FLOP fields through ``cache`` (:func:`_seed_interned`)."""

    for key in keys.bytes_fields:
        raw = values[key]
        try:
            values[key] = cache["bytes", raw.__class__, raw]
        except (KeyError, TypeError):
            values[key] = _quantity(cache, "bytes", raw, as_bytes)
    raw = values[keys.param_memory] or 0
    try:
        values[keys.param_memory] = cache["param", raw.__class__, raw]
    except (KeyError, TypeError):
        values[keys.param_memory] = _quantity(cache, "param", raw, Bytes)
    raw = values[keys.func_duration]
    if raw is None or (raw.__class__ is int and raw == 0):
        try:
            values[keys.func_duration] = cache["duration", raw.__class__, raw]
        except KeyError:
            values[keys.func_duration] = _quantity(cache, "duration", raw, as_duration)
    else:
        values[keys.func_duration] = as_duration(raw)
    for key in keys.flops:
        raw = values[key]
        try:
            values[key] = cache["flops", raw.__class__, raw]
        except (KeyError, TypeError):
            values[key] = _quantity(cache, "flops", raw, as_flops)


def _device_ref_of(sources: tuple[Any, Any]) -> DeviceRef | None:
    """Build the device reference of ``(out, output_device)``."""

    return device_ref_from_metadata(*sources)


def _cached_or_built(cache: dict[Any, Any], key: Any, build: Any, source: Any) -> Any:
    """Build a value, caching it under ``key`` when ``key`` is hashable."""

    value = build(source)
    try:
        cache[key] = value
    except TypeError:
        pass
    return value


def _quantity(cache: dict[Any, Any], kind: str, raw: Any, build: Any) -> Any:
    """Build the quantity for ``raw`` and cache it by ``raw``'s exact type and value."""

    return _cached_or_built(cache, (kind, raw.__class__, raw), build, raw)
