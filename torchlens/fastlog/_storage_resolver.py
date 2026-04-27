"""Storage resolution for fastlog tensor payloads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from .._errors import TorchLensPostfuncError
from .._state import pause_logging
from .._training_validation import TrainingModeConfigError
from ..utils.tensor_utils import safe_copy
from .exceptions import PredicateError
from .types import CaptureSpec, RecordContext, StorageIntent

if TYPE_CHECKING:
    from ..types import ActivationPostfunc

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
    *,
    activation_postfunc: "ActivationPostfunc | None" = None,
    save_raw_activation: bool = True,
    ctx: RecordContext | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Resolve RAM and disk tensor payloads for one capture decision.

    Returns
    -------
    tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]
        ``(ram_payload, disk_payload, transformed_ram_payload, transformed_disk_payload)``.
        Any payload may be ``None``.

    Parameters
    ----------
    tensor:
        Tensor selected for capture.
    spec:
        Capture policy for the tensor.
    intent:
        Storage intent resolved from streaming options.
    activation_postfunc:
        Optional callable applied to RAM/disk payloads after dtype/device
        transforms. Errors are wrapped in :class:`TorchLensPostfuncError`.
    save_raw_activation:
        When False and ``activation_postfunc`` is set, raw payloads are
        suppressed and only the transformed copy is retained.
    ctx:
        Record context used to enrich postfunc error messages.

    Raises
    ------
    PredicateError
        If the requested storage policy is invalid for the tensor.
    TorchLensPostfuncError
        If the activation postfunc raises while transforming a payload.
    TrainingModeConfigError
        If ``keep_grad=True`` and the transformed RAM payload is not a
        gradient-capable tensor connected to the autograd graph.
    """

    if spec.keep_grad and intent.on_disk and not intent.in_ram:
        raise PredicateError("keep_grad=True is not valid for disk-only fastlog storage")
    if spec.keep_grad and (tensor.dtype in _INTEGER_DTYPES or spec.dtype in _INTEGER_DTYPES):
        raise PredicateError("keep_grad=True is not valid for integer or bool tensors")

    ram_payload: torch.Tensor | None = None
    disk_payload: torch.Tensor | None = None
    transformed_ram: torch.Tensor | None = None
    transformed_disk: torch.Tensor | None = None
    postfunc = activation_postfunc
    keep_raw = save_raw_activation or postfunc is None

    if intent.in_ram:
        raw_ram = safe_copy(tensor, detach_tensor=not spec.keep_grad)
        raw_ram = _apply_payload_transforms(raw_ram, spec)
        if postfunc is not None:
            transformed_ram = _invoke_postfunc(
                raw_ram,
                postfunc,
                ctx=ctx,
                spec=spec,
                intent=intent,
                target="ram",
            )
            if spec.keep_grad:
                _validate_train_mode_transformed(
                    raw_ram,
                    transformed_ram,
                    ctx=ctx,
                    spec=spec,
                )
        if keep_raw:
            ram_payload = raw_ram
    if intent.on_disk:
        raw_disk = safe_copy(tensor, detach_tensor=True)
        raw_disk = _apply_payload_transforms(raw_disk, spec)
        if postfunc is not None:
            transformed_disk = _invoke_postfunc(
                raw_disk,
                postfunc,
                ctx=ctx,
                spec=spec,
                intent=intent,
                target="disk",
            )
        if keep_raw:
            disk_payload = raw_disk
    return ram_payload, disk_payload, transformed_ram, transformed_disk


def _invoke_postfunc(
    tensor: torch.Tensor,
    postfunc: Callable[[torch.Tensor], torch.Tensor],
    *,
    ctx: RecordContext | None,
    spec: CaptureSpec,
    intent: StorageIntent,
    target: str,
) -> torch.Tensor:
    """Apply a fastlog activation postfunc with logging paused."""

    try:
        with pause_logging():
            return postfunc(tensor)
    except Exception as exc:
        raise TorchLensPostfuncError(
            _postfunc_error_message(
                ctx=ctx,
                spec=spec,
                intent=intent,
                target=target,
            )
        ) from exc


def _postfunc_error_message(
    *,
    ctx: RecordContext | None,
    spec: CaptureSpec,
    intent: StorageIntent,
    target: str,
) -> str:
    """Build context for an activation postfunc failure."""

    storage_target = _describe_storage_target(intent, target)
    if ctx is None:
        return (
            "activation_postfunc raised while resolving a fastlog payload "
            f"(storage_target={storage_target}, keep_grad={spec.keep_grad})."
        )
    return (
        f"activation_postfunc raised for fastlog event "
        f"label={ctx.label!r} kind={ctx.kind} func={ctx.func_name} "
        f"shape={tuple(ctx.tensor_shape) if ctx.tensor_shape is not None else None} "
        f"dtype={ctx.tensor_dtype} storage_target={storage_target} "
        f"keep_grad={spec.keep_grad}."
    )


def _describe_storage_target(intent: StorageIntent, target: str) -> str:
    """Return a human-readable storage-target description."""

    if intent.in_ram and intent.on_disk:
        return f"mirror:{target}"
    if intent.in_ram:
        return "ram"
    if intent.on_disk:
        return "disk"
    return target


def _validate_train_mode_transformed(
    raw_tensor: torch.Tensor,
    transformed: torch.Tensor | None,
    *,
    ctx: RecordContext | None,
    spec: CaptureSpec,
) -> None:
    """Validate differentiability requirements for a transformed RAM payload.

    The transformed payload must remain a gradient-capable tensor that stays
    graph-connected when the raw RAM payload retained autograd history.
    Disk transformed payloads are detached inspection copies and are not
    validated here.
    """

    label = ctx.label if ctx is not None else "<unknown>"
    if not isinstance(transformed, torch.Tensor):
        raise TrainingModeConfigError(
            "activation_postfunc must return a torch.Tensor while keep_grad=True "
            f"for fastlog event {label!r}."
        )
    if transformed.dtype in _INTEGER_DTYPES:
        raise TrainingModeConfigError(
            f"train_mode=True with non-grad dtype {transformed.dtype} on fastlog "
            f"event {label!r}. Integer and bool dtypes cannot propagate gradients. "
            "Adjust activation_postfunc to return a floating dtype."
        )
    if raw_tensor.requires_grad and transformed.grad_fn is None:
        raise TrainingModeConfigError(
            "activation_postfunc returned a tensor disconnected from the autograd "
            "graph (grad_fn is None) while keep_grad=True. The transformed activation "
            f"for fastlog event {label!r} must remain differentiable."
        )
    _ = spec
