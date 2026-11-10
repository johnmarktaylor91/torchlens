"""One row write for a new ``Param`` record.

``Param.__init__`` used to assign its thirty stored fields one at a time, each
through a row-cell descriptor (three calls per field, about ninety per
parameter per trace). :func:`init_param_row` builds the same row as one list in
the class layout's order and hands it to the store in a single write. The
values, their order of evaluation and the objects stored are the ones the
field-by-field ``__init__`` produced; ``param.py`` checks at import that
:data:`PARAM_ROW_FIELDS` is its stored-field tuple, so the two cannot drift.

``_param_ref`` starts as ``None``. The preparing backend sets it to the live
``nn.Parameter`` for lazy grad access; it keeps the parameter alive while the
``Param`` is alive (trace lifetime <= model lifetime) and ``cleanup()`` clears it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._quantities import Bytes
from .._trace_core.record_rows import bind_record_row

if TYPE_CHECKING:
    import torch

    from .._ir.refs import DeviceRef, DtypeRef
    from .op import GradientRecord

#: The order of the row :func:`init_param_row` builds (``Param``'s stored fields).
PARAM_ROW_FIELDS: tuple[str, ...] = (
    "module_address",
    "name",
    "shape",
    "dtype",
    "dtype_ref",
    "device_ref",
    "backend_address",
    "resolver_status",
    "num_params",
    "param_memory",
    "is_trainable",
    "address",
    "all_addresses",
    "all_module_addresses",
    "barcode",
    "has_optimizer",
    "_param_ref",
    "_param_ref_released",
    "_source_trace_ref",
    "num_calls",
    "used_by_ops",
    "used_by_layers",
    "co_parent_params",
    "_has_grad",
    "_grad_shape",
    "_grad_dtype",
    "_grad_memory",
    "_grad_records",
    "_derived_grad_payload",
    "_derived_grad_record_path",
)


class ParamRowFields:
    """Static types of ``Param``'s stored fields, for the type checker only.

    The fields are row-cell descriptors installed at import, which a type
    checker cannot see, and ``Param.__init__`` no longer assigns them one by
    one. ``Param`` names this class as its base under ``TYPE_CHECKING`` only;
    at runtime its base stays ``object``.
    """

    address: str
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    dtype_ref: DtypeRef | None
    device_ref: DeviceRef | None
    backend_address: str | None
    resolver_status: str
    num_params: int
    param_memory: Bytes
    is_trainable: bool
    module_address: str
    all_addresses: list[str]
    all_module_addresses: list[str]
    barcode: str
    has_optimizer: bool | None
    _param_ref: torch.nn.Parameter | None
    _param_ref_released: bool
    _source_trace_ref: Any
    num_calls: int
    used_by_ops: list[str]
    used_by_layers: list[str]
    co_parent_params: list[str]
    _has_grad: bool
    _grad_shape: tuple[int, ...] | None
    _grad_dtype: torch.dtype | None
    _grad_memory: Bytes
    _grad_records: list[GradientRecord]
    _derived_grad_payload: Any | None
    _derived_grad_record_path: str | None


def init_param_row(record: Any, dtype_ref: DtypeRef | None, *given: Any) -> None:
    """Write a new ``Param``'s stored fields in one row write.

    Parameters
    ----------
    record
        The ``Param`` being initialised.
    dtype_ref
        The dtype reference for the parameter's dtype, built by the caller.
    *given
        ``Param.__init__``'s ten arguments, unchanged and in its order:
        ``module_address``, ``name``, ``shape``, ``dtype``, ``num_params``,
        ``param_memory``, ``trainable``, ``address``, ``barcode`` and
        ``has_optimizer``. Any other count raises ``ValueError``.
    """

    module_address, name, shape, dtype, num_params = given[:5]
    param_memory, trainable, address, barcode, has_optimizer = given[5:]
    bind_record_row(
        record,
        [
            module_address,
            name,
            shape,
            dtype,
            dtype_ref,
            None,  # device_ref
            address,  # backend_address
            "resolved",  # resolver_status
            num_params,
            Bytes(param_memory),
            trainable,  # is_trainable
            address,
            [address],  # all_addresses
            [module_address],  # all_module_addresses
            barcode,
            has_optimizer,
            None,  # _param_ref (see the module docstring)
            False,  # _param_ref_released
            None,  # _source_trace_ref
            1,  # num_calls: forward ops that used this param; from here on, set by postprocess
            [],  # used_by_ops: op labels that used this param
            [],  # used_by_layers: layer labels that used this param
            [],  # co_parent_params: other param addresses used by the same op
            False,  # _has_grad: one-shot flag, once True no further checks
            None,  # _grad_shape
            None,  # _grad_dtype
            Bytes(0),  # _grad_memory
            [],  # _grad_records
            None,  # _derived_grad_payload
            None,  # _derived_grad_record_path
        ],
    )
