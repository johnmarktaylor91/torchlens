"""Storage resolution for fastlog tensor payloads."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

from .._errors import TorchLensPostfuncError
from .._state import pause_logging
from .._training_validation import TrainingModeConfigError
from ..utils.tensor_utils import safe_copy
from ..utils.tensor_utils import SaveMode
from .exceptions import InvalidStorageError, PredicateError
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
_WARNED_REFERENCE_SAVE_MODE = False


def _apply_payload_transforms(tensor: torch.Tensor, spec: CaptureSpec) -> torch.Tensor:
    """Apply device and dtype transforms after safe_copy."""

    payload = tensor
    with pause_logging():
        if spec.dtype is not None:
            payload = payload.to(dtype=spec.dtype)
        if spec.device is not None:
            payload = payload.to(device=spec.device)
    return payload


def _warn_reference_save_mode_once() -> None:
    """Emit the reference-mode mutation warning once per process."""

    global _WARNED_REFERENCE_SAVE_MODE
    if _WARNED_REFERENCE_SAVE_MODE:
        return
    warnings.warn(
        "save_mode='reference' stores source tensors by reference; reading a mutated "
        "saved tensor raises MutatedReferenceError.",
        UserWarning,
        stacklevel=3,
    )
    _WARNED_REFERENCE_SAVE_MODE = True


def _save_mode_for_payload(spec: CaptureSpec, *, target: Literal["ram", "disk"]) -> SaveMode:
    """Return the effective save mode for a storage target."""

    if target == "disk" and spec.save_mode in {"reference", "view"}:
        return "copy"
    return spec.save_mode


def _resolve_storage(
    tensor: torch.Tensor,
    spec: CaptureSpec,
    intent: StorageIntent,
    *,
    activation_transform: "ActivationPostfunc | None" = None,
    save_raw_activations: bool = True,
    ctx: Any | None = None,
    kind: Literal["activation", "grad"] = "activation",
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
    activation_transform:
        Optional callable applied to RAM/disk payloads after dtype/device
        transforms. Errors are wrapped in :class:`TorchLensPostfuncError`.
    save_raw_activations:
        When False and ``activation_transform`` is set, raw payloads are
        suppressed and only the transformed copy is retained.
    ctx:
        Record context used to enrich transform error messages.

    Raises
    ------
    PredicateError
        If the requested storage policy is invalid for the tensor.
    TorchLensPostfuncError
        If the out transform raises while transforming a payload.
    TrainingModeConfigError
        If ``keep_grad=True`` and the transformed RAM payload is not a
        grad-capable tensor connected to the autograd graph.
    """

    if spec.keep_grad and intent.on_disk and not intent.in_ram:
        message = f"keep_grad=True is not valid for disk-only {kind} storage"
        if kind == "grad":
            raise InvalidStorageError(message)
        raise PredicateError(message)
    if spec.keep_grad and (tensor.dtype in _INTEGER_DTYPES or spec.dtype in _INTEGER_DTYPES):
        raise PredicateError("keep_grad=True is not valid for integer or bool tensors")
    if spec.save_mode not in {"copy", "reference", "view", "cpu_async"}:
        raise PredicateError("save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'")
    if spec.save_mode == "reference":
        _warn_reference_save_mode_once()
    if spec.save_mode == "view" and not spec.keep_grad:
        spec = CaptureSpec(
            save_out=spec.save_out,
            save_metadata=spec.save_metadata,
            keep_grad=True,
            device=spec.device,
            dtype=spec.dtype,
            save_mode=spec.save_mode,
        )

    ram_payload: torch.Tensor | None = None
    disk_payload: torch.Tensor | None = None
    transformed_ram: torch.Tensor | None = None
    transformed_disk: torch.Tensor | None = None
    transform = activation_transform
    keep_raw = save_raw_activations or transform is None

    if intent.in_ram:
        raw_ram = safe_copy(
            tensor,
            detach_tensor=not spec.keep_grad,
            save_mode=_save_mode_for_payload(spec, target="ram"),
        )
        raw_ram = _apply_payload_transforms(raw_ram, spec)
        if transform is not None:
            transformed_ram = _invoke_transform(
                raw_ram,
                transform,
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
        raw_disk = safe_copy(
            tensor,
            detach_tensor=True,
            save_mode=_save_mode_for_payload(spec, target="disk"),
        )
        raw_disk = _apply_payload_transforms(raw_disk, spec)
        if transform is not None:
            transformed_disk = _invoke_transform(
                raw_disk,
                transform,
                ctx=ctx,
                spec=spec,
                intent=intent,
                target="disk",
            )
        if keep_raw:
            disk_payload = raw_disk
    return ram_payload, disk_payload, transformed_ram, transformed_disk


def _invoke_transform(
    tensor: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
    *,
    ctx: Any | None,
    spec: CaptureSpec,
    intent: StorageIntent,
    target: str,
) -> torch.Tensor:
    """Apply a fastlog out transform with logging paused."""

    try:
        with pause_logging():
            return transform(tensor)
    except Exception as exc:
        raise TorchLensPostfuncError(
            _transform_error_message(
                ctx=ctx,
                spec=spec,
                intent=intent,
                target=target,
            )
        ) from exc


def _transform_error_message(
    *,
    ctx: Any | None,
    spec: CaptureSpec,
    intent: StorageIntent,
    target: str,
) -> str:
    """Build context for an out transform failure."""

    storage_target = _describe_storage_target(intent, target)
    if ctx is None:
        return (
            "activation_transform raised while resolving a fastlog payload "
            f"(storage_target={storage_target}, keep_grad={spec.keep_grad})."
        )
    return (
        f"activation_transform raised for fastlog event "
        f"label={ctx.label!r} kind={ctx.kind} func={ctx.func_name} "
        f"shape={tuple(ctx.shape) if ctx.shape is not None else None} "
        f"dtype={ctx.dtype} storage_target={storage_target} "
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

    The transformed payload must remain a grad-capable tensor that stays
    graph-connected when the raw RAM payload retained autograd history.
    Disk transformed payloads are detached inspection copies and are not
    validated here.
    """

    label = ctx.label if ctx is not None else "<unknown>"
    if not isinstance(transformed, torch.Tensor):
        raise TrainingModeConfigError(
            "activation_transform must return a torch.Tensor while keep_grad=True "
            f"for fastlog event {label!r}."
        )
    if transformed.dtype in _INTEGER_DTYPES:
        raise TrainingModeConfigError(
            f"backward_ready=True with non-grad dtype {transformed.dtype} on fastlog "
            f"event {label!r}. Integer and bool dtypes cannot propagate grads. "
            "Adjust activation_transform to return a floating dtype."
        )
    if raw_tensor.requires_grad and transformed.grad_fn is None:
        raise TrainingModeConfigError(
            "activation_transform returned a tensor disconnected from the autograd "
            "graph (grad_fn is None) while keep_grad=True. The transformed out "
            f"for fastlog event {label!r} must remain differentiable."
        )
    _ = spec
