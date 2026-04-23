"""Compatibility checks for tensors saved into portable bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


@dataclass(frozen=True)
class Ok:
    """Successful tensor policy decision."""


@dataclass(frozen=True)
class SkipReason:
    """Best-effort skip decision for unsupported tensors under ``strict=False``."""

    text: str


@dataclass(frozen=True)
class FailReason:
    """Hard-fail decision for unsupported tensors under ``strict=True``."""

    text: str


TensorPolicyDecision = Union[Ok, SkipReason, FailReason]

_SPARSE_LAYOUTS = {
    torch.sparse_coo,
    getattr(torch, "sparse_csr", object()),
    getattr(torch, "sparse_csc", object()),
    getattr(torch, "sparse_bsr", object()),
    getattr(torch, "sparse_bsc", object()),
}
_SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
    torch.complex64,
    torch.complex128,
}


def is_supported_for_save(
    tensor: torch.Tensor,
    *,
    strict: bool = True,
) -> TensorPolicyDecision:
    """Decide whether a tensor can be persisted in a portable bundle.

    Parameters
    ----------
    tensor:
        Tensor candidate to validate.
    strict:
        Whether unsupported tensors should return ``FailReason`` (default) or
        ``SkipReason`` for best-effort bundle creation.

    Returns
    -------
    TensorPolicyDecision
        ``Ok`` for supported tensors, otherwise ``FailReason`` or
        ``SkipReason`` depending on ``strict``.
    """

    reason = _unsupported_reason(tensor)
    if reason is None:
        return Ok()
    if strict:
        return FailReason(reason)
    return SkipReason(reason)


def _unsupported_reason(tensor: torch.Tensor) -> str | None:
    """Return the first incompatibility reason for a tensor, if any.

    Parameters
    ----------
    tensor:
        Tensor candidate to inspect.

    Returns
    -------
    str | None
        Unsupported reason text, or ``None`` when the tensor is supported.
    """

    if type(tensor) is not torch.Tensor:
        return f"tensor subclass {type(tensor).__name__} is not supported this release"
    if tensor.layout in _SPARSE_LAYOUTS or tensor.is_sparse:
        return f"{str(tensor.layout).replace('torch.', '')} layout is not supported this release"
    if tensor.is_quantized:
        return "quantized tensors are not supported this release"
    if tensor.device.type == "meta":
        return "meta tensors do not have materialized data to save"
    if bool(getattr(tensor, "is_nested", False)):
        return "nested tensors are not supported this release"
    if _is_distributed_shard_tensor(tensor):
        return "DTensor/FSDP shard tensors are not supported this release"
    complex32_dtype = getattr(torch, "complex32", None)
    if complex32_dtype is not None and tensor.dtype == complex32_dtype:
        return "complex32 tensors are not supported this release"
    if tensor.device.type not in {"cpu", "cuda"}:
        return f"{tensor.device.type} tensors are not supported this release"
    if tensor.dtype not in _SUPPORTED_DTYPES:
        return f"dtype {tensor.dtype} is not supported this release"
    return None


def _is_distributed_shard_tensor(tensor: torch.Tensor) -> bool:
    """Best-effort detection for DTensor and sharded distributed tensors.

    Parameters
    ----------
    tensor:
        Tensor candidate to inspect.

    Returns
    -------
    bool
        True when the tensor appears to be a DTensor or sharded tensor.
    """

    tensor_type = type(tensor)
    module_name = tensor_type.__module__.lower()
    type_name = tensor_type.__name__.lower()
    return (
        "dtensor" in type_name
        or "shardedtensor" in type_name
        or ("distributed" in module_name and ("dtensor" in module_name or "shard" in module_name))
    )
