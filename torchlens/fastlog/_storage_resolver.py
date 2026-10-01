"""Storage resolution for fastlog tensor payloads."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import torch

from .._errors import TorchLensPostfuncError
from .._state import pause_logging
from .._training_validation import TrainingModeConfigError
from ..ir.summary_role import ensure_summary_output_owns_storage, is_summary_transform
from ..utils.tensor_utils import SaveMode, safe_copy
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


def _raw_payload_source(
    tensor: torch.Tensor,
    spec: CaptureSpec,
    *,
    target: Literal["ram", "disk"],
    reduce_only: bool,
    detach: bool,
) -> torch.Tensor:
    """Raw payload for one target: the live view on the reduce-only path, else a safe copy.

    Explorer P2 (fastlog leg): the reduce-only path hands the reducer the
    live tensor at the save point (the transient safe_copy is the cost the
    path exists to skip); every other path keeps the copy + payload
    respellings.
    """

    if reduce_only:
        return tensor
    copied = safe_copy(
        tensor,
        detach_tensor=detach,
        save_mode=_save_mode_for_payload(spec, target=target),
    )
    return _apply_payload_transforms(copied, spec)


def _resolve_storage(
    tensor: torch.Tensor,
    spec: CaptureSpec,
    intent: StorageIntent,
    *,
    activation_transform: ActivationPostfunc | None = None,
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
        message = (
            f"keep_grad=True is not valid for disk-only {kind} storage. "
            "Remedy: set retain_in_memory=True or drop keep_grad=True."
        )
        if kind == "grad":
            raise InvalidStorageError(message, code="predicate_storage_conflict")
        raise PredicateError(message, code="predicate_storage_conflict")
    if spec.keep_grad and (tensor.dtype in _INTEGER_DTYPES or spec.dtype in _INTEGER_DTYPES):
        raise PredicateError(
            "keep_grad=True is not valid for integer or bool tensors. "
            "Remedy: drop keep_grad=True or keep the payload in a floating dtype.",
            code="predicate_storage_conflict",
        )
    if spec.save_mode not in {"copy", "reference", "view", "cpu_async"}:
        raise PredicateError(
            "save_mode must be one of 'copy', 'reference', 'view', or 'cpu_async'. "
            "Remedy: set save_mode to one of those documented modes.",
            code="save_mode_invalid",
        )
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
    # Explorer P2 (fastlog leg): a declared summary reducer with raw
    # retention off and no dtype/device respelling skips the transient
    # safe_copy -- the reducer sees a detached view of the live tensor at
    # the save point. The dtype/device respellings keep the copy path: they
    # allocate anyway, so there is nothing to skip.
    reduce_only = (
        not keep_raw
        and is_summary_transform(transform)
        and spec.dtype is None
        and spec.device is None
    )

    if intent.in_ram:
        raw_ram = _raw_payload_source(
            tensor, spec, target="ram", reduce_only=reduce_only, detach=not spec.keep_grad
        )
        if transform is not None:
            transformed_ram = _invoke_transform(
                raw_ram,
                transform,
                ctx=ctx,
                spec=spec,
                intent=intent,
                target="ram",
            )
            if reduce_only:
                transformed_ram = ensure_summary_output_owns_storage(transformed_ram, tensor)
            if spec.keep_grad:
                _validate_train_mode_transformed(
                    raw_ram,
                    transformed_ram,
                    ctx=ctx,
                    spec=spec,
                    transform=transform,
                )
        if keep_raw:
            ram_payload = raw_ram
    if intent.on_disk:
        raw_disk: torch.Tensor | None = None
        if intent.in_ram:
            if raw_ram is None:
                raise RuntimeError("RAM mirror payload missing for disk-backed fastlog capture")
            # The mirror consumers write their blobs synchronously before the
            # forward advances, so the disk slot can alias the retained RAM
            # payload instead of cloning it: the writer reads the same bytes a
            # copy taken here would hold. A detached view keeps the disk mirror's
            # documented detached-inspection contract (and the manifest's
            # requires_grad=False) without a data copy.
            if keep_raw:
                disk_payload = _detached_write_alias(raw_ram)
            if transformed_ram is not None:
                transformed_disk = _detached_write_alias(transformed_ram)
        else:
            raw_disk = _raw_payload_source(
                tensor, spec, target="disk", reduce_only=reduce_only, detach=True
            )
            if transform is not None:
                transformed_disk = _invoke_transform(
                    raw_disk,
                    transform,
                    ctx=ctx,
                    spec=spec,
                    intent=intent,
                    target="disk",
                )
                if reduce_only:
                    transformed_disk = ensure_summary_output_owns_storage(transformed_disk, tensor)
            if keep_raw:
                disk_payload = raw_disk
    return ram_payload, disk_payload, transformed_ram, transformed_disk


def _detached_write_alias(tensor: torch.Tensor) -> torch.Tensor:
    """Return a zero-copy detached handle on a RAM payload for a synchronous write.

    Parameters
    ----------
    tensor:
        Retained RAM payload whose bytes the blob writer will read immediately.

    Returns
    -------
    torch.Tensor
        The payload itself when already detached, else a detached view sharing
        its storage (``save_mode="reference"`` through the sanctioned copy
        primitive; only :func:`safe_copy` may detach on fastlog paths).
    """

    if not tensor.requires_grad:
        return tensor
    return safe_copy(tensor, detach_tensor=True, save_mode="reference")


def _invoke_transform(
    tensor: torch.Tensor,
    transform: Callable[[torch.Tensor], torch.Tensor],
    *,
    ctx: Any | None,
    spec: CaptureSpec,
    intent: StorageIntent,
    target: str,
) -> torch.Tensor:
    """Apply a fastlog out transform with logging paused.

    A transform declaring ``_tl_wants_ctx = True`` (duck-typed, F25 op
    tier) additionally receives the frozen ``RecordContext`` as ``ctx=``
    so a label-aware reducer (facet splitting) needs no side channel;
    ordinary transforms keep the one-positional-tensor contract.
    """

    try:
        with pause_logging():
            if getattr(transform, "_tl_wants_ctx", False):
                from typing import cast

                return cast("Callable[..., torch.Tensor]", transform)(tensor, ctx=ctx)
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
    transform: Any | None = None,
) -> None:
    """Validate differentiability requirements for a transformed RAM payload.

    The transformed payload must remain a grad-capable tensor that stays
    graph-connected when the raw RAM payload retained autograd history.
    Disk transformed payloads are detached inspection copies and are not
    validated here. A transform DECLARING the non-differentiable summary
    role (explorer P1) is contractually outside the autograd graph and is
    exempt; undeclared detached transforms keep refusing -- the carve-out is
    the declared role, never the output's shape or dtype.
    """

    if transform is not None and is_summary_transform(transform):
        return
    label = ctx.label if ctx is not None else "<unknown>"
    if not isinstance(transformed, torch.Tensor):
        raise TrainingModeConfigError(
            "activation_transform must return a torch.Tensor while keep_grad=True "
            f"for fastlog event {label!r}. "
            "Remedy: return a differentiable torch.Tensor from activation_transform, "
            "or declare a non-differentiable reducer with "
            "torchlens.observability.summary(fn).",
            code="transform_not_differentiable",
        )
    if transformed.dtype in _INTEGER_DTYPES:
        raise TrainingModeConfigError(
            f"backward_ready=True with non-grad dtype {transformed.dtype} on fastlog "
            f"event {label!r}. Integer and bool dtypes cannot propagate grads. "
            "Remedy: adjust activation_transform to return a floating dtype, or "
            "declare a non-differentiable reducer with "
            "torchlens.observability.summary(fn).",
            code="transform_not_differentiable",
        )
    if raw_tensor.requires_grad and transformed.grad_fn is None:
        raise TrainingModeConfigError(
            "activation_transform returned a tensor disconnected from the autograd "
            "graph (grad_fn is None) while keep_grad=True. The transformed out "
            f"for fastlog event {label!r} must remain differentiable. "
            "Remedy: keep activation_transform on the autograd graph (no detach/no_grad), "
            "or declare a non-differentiable reducer with "
            "torchlens.observability.summary(fn).",
            code="transform_not_differentiable",
        )
    _ = spec
