"""Storage resolution for fastlog tensor payloads."""

from __future__ import annotations

import torch

from .._state import pause_logging
from ..utils.tensor_utils import safe_copy
from .exceptions import PredicateError
from .types import CaptureSpec, StorageIntent

_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
}


def _apply_payload_transforms(tensor: torch.Tensor, spec: CaptureSpec) -> torch.Tensor:
    """Apply device and dtype transforms after safe_copy."""

    payload = tensor
    with pause_logging():
        if spec.dtype is not None:
            payload = payload.to(dtype=spec.dtype)
        if spec.device is not None:
            payload = payload.to(device=spec.device)
    return payload


def _resolve_storage(
    tensor: torch.Tensor,
    spec: CaptureSpec,
    intent: StorageIntent,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Resolve RAM and disk tensor payloads for one capture decision.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None]
        ``(ram_payload, disk_payload)``. Either payload may be None.

    Raises
    ------
    PredicateError
        If the requested storage policy is invalid for the tensor.
    """

    if spec.keep_grad and intent.on_disk and not intent.in_ram:
        raise PredicateError("keep_grad=True is not valid for disk-only fastlog storage")
    if spec.keep_grad and (tensor.dtype in _INTEGER_DTYPES or spec.dtype in _INTEGER_DTYPES):
        raise PredicateError("keep_grad=True is not valid for integer or bool tensors")

    ram_payload = None
    disk_payload = None
    if intent.in_ram:
        ram_payload = safe_copy(tensor, detach_tensor=not spec.keep_grad)
        ram_payload = _apply_payload_transforms(ram_payload, spec)
    if intent.on_disk:
        disk_payload = safe_copy(tensor, detach_tensor=True)
        disk_payload = _apply_payload_transforms(disk_payload, spec)
    return ram_payload, disk_payload
