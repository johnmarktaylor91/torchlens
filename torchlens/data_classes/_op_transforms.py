"""User-transform application and validation helpers for saved payloads.

Split out of ``op.py`` (T43 size-ratchet split): these four helpers apply a
user ``activation_transform``/``grad_transform`` with logging paused and
validate its output against the backward-ready and streaming-save contracts.
``op.py`` re-exposes them unchanged (the ``Op`` transform methods and the
torch backend import through it), so the public import site is unmoved.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .._errors import TorchLensPostfuncError

# _io / _state / _training_validation sit at or above this module's layer
# (arch-spine layer lint): their names defer inside the one function each
# serves rather than importing eagerly here.


def apply_transform(
    *,
    label: str | None,
    tensor: torch.Tensor,
    transform: Callable[..., Any],
    transform_kind: str,
    streaming_active: bool = False,
    raw_label: str | None = None,
    func_name: str | None = None,
) -> Any:
    """Apply a user transform with logging paused and contextual errors.

    Parameters
    ----------
    label:
        Raw layer label for error context, or ``None`` when unavailable.
    tensor:
        Raw tensor passed to the user transform.
    transform:
        Callable applied to ``tensor``.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active.
    raw_label:
        Raw layer label for error context when it differs from ``label``.
    func_name:
        Function name for error context.

    Returns
    -------
    Any
        Value returned by ``transform``.
    """

    # R36: a cpu_async payload may still be an in-flight pinned buffer; a
    # user transform is a host-side byte read and must never observe partial
    # bytes. No-op unless async fence events are actually pending.
    from .._state import pause_logging
    from ..utils.tensor_utils import synchronize_pending_cpu_async_copies

    synchronize_pending_cpu_async_copies()
    try:
        with pause_logging():
            return transform(tensor)
    except Exception as exc:
        raise TorchLensPostfuncError(
            transform_error_message(
                label=label,
                raw_label=raw_label,
                func_name=func_name,
                tensor=tensor,
                transform_kind=transform_kind,
                streaming_active=streaming_active,
            )
        ) from exc


def transform_error_message(
    *,
    label: str | None,
    raw_label: str | None = None,
    func_name: str | None = None,
    tensor: torch.Tensor,
    transform_kind: str,
    streaming_active: bool,
) -> str:
    """Build context for an out or grad transform failure.

    Parameters
    ----------
    label:
        Raw layer label for error context, or ``None`` when unavailable.
    raw_label:
        Raw layer label for error context when it differs from ``label``.
    func_name:
        Function name for error context.
    tensor:
        Raw tensor passed to the transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active.

    Returns
    -------
    str
        Contextual error message.
    """

    return (
        f"{transform_kind}_transform raised for layer {label} "
        f"(raw={raw_label or label}, func={func_name}, "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"streaming_active={streaming_active})."
    )


def train_mode_tripwire_armed(*, backward_ready: bool, transform: Any = None) -> bool:
    """Whether the train-mode differentiability tripwire applies at all.

    Parameters
    ----------
    backward_ready:
        Whether TorchLens is preserving autograd graph connectivity.
    transform:
        The transform callable itself, when the caller has it. A transform
        DECLARING the non-differentiable summary role (explorer P1;
        ``torchlens.ir.summary_role``) is contractually outside the autograd
        graph -- it reduces a detached view after the save point -- so the
        differentiability tripwire does not apply to its output. Undeclared
        transforms keep the full validation: the carve-out is the declared
        role, never the output's shape or dtype.

    Returns
    -------
    bool
        ``True`` when transformed outputs must stay on the autograd graph.
    """

    if not backward_ready:
        return False
    if transform is not None:
        from ..ir.summary_role import is_summary_transform

        if is_summary_transform(transform):
            return False
    return True


def validate_train_mode_transform_output(
    *,
    raw_tensor: torch.Tensor,
    transformed_tensor: Any,
    transform_kind: str,
    tripwire_armed: bool,
    label: str | None = None,
) -> None:
    """Validate differentiability requirements for train-mode transform outputs.

    Parameters
    ----------
    raw_tensor:
        Raw tensor passed to the transform.
    transformed_tensor:
        Value returned by the transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    tripwire_armed:
        Whether the differentiability tripwire applies; compute with
        :func:`train_mode_tripwire_armed` (folds ``backward_ready`` and the
        declared summary-role carve-out at the caller, where the transform
        callable lives).
    label:
        Raw layer label for error context, or ``None`` when unavailable.

    Returns
    -------
    None
        Raises if the transformed value violates train-mode requirements.
    """

    from .._training_validation import _NON_GRAD_DTYPES, TrainingModeConfigError

    if not tripwire_armed or not raw_tensor.requires_grad:
        return
    if not isinstance(transformed_tensor, torch.Tensor):
        raise TrainingModeConfigError(
            f"{transform_kind}_transform must return a torch.Tensor while backward_ready=True "
            f"for layer {label}. "
            "Remedy: return a differentiable torch.Tensor from the transform, or "
            "declare a non-differentiable reducer with "
            "torchlens.observability.summary(fn).",
            code="transform_not_differentiable",
        )
    if transformed_tensor.dtype in _NON_GRAD_DTYPES:
        raise TrainingModeConfigError(
            f"backward_ready=True with non-grad dtype {transformed_tensor.dtype} on layer "
            f"{label}. Integer and bool dtypes cannot propagate grads. "
            "Remedy: return a floating-dtype tensor from the transform, or declare a "
            "non-differentiable reducer with torchlens.observability.summary(fn).",
            code="transform_not_differentiable",
        )
    if not transformed_tensor.requires_grad or (
        transformed_tensor.grad_fn is None and transformed_tensor is not raw_tensor
    ):
        raise TrainingModeConfigError(
            f"{transform_kind}_transform returned a tensor disconnected from the autograd "
            "graph (grad_fn is None) while backward_ready=True. The transformed out "
            "must remain differentiable. "
            "Remedy: keep the transform on the autograd graph (no detach/no_grad), or "
            "declare a non-differentiable reducer with "
            "torchlens.observability.summary(fn).",
            code="transform_not_differentiable",
        )


def validate_streaming_transform_output(
    *,
    transformed_tensor: Any,
    transform_kind: str,
    streaming_active: bool,
    label: str | None = None,
) -> None:
    """Validate transformed tensors before streaming bundle finalization.

    Parameters
    ----------
    transformed_tensor:
        Value returned by the user transform.
    transform_kind:
        Transform kind, either ``"out"`` or ``"grad"``.
    streaming_active:
        Whether a streaming bundle writer is active for this trace.
    label:
        Raw layer label for error context, or ``None`` when unavailable.

    Returns
    -------
    None
        Raises if streaming cannot serialize the transformed value.
    """

    from .._io import TorchLensIOError

    if not streaming_active:
        return
    if not isinstance(transformed_tensor, torch.Tensor):
        raise TorchLensIOError(
            f"Streaming save requires {transform_kind}_transform outputs to be "
            f"torch.Tensor instances, but layer {label} produced "
            f"{type(transformed_tensor).__name__}."
        )
    if transformed_tensor.layout != torch.strided:
        raise TorchLensIOError(
            f"Streaming save does not support sparse {transform_kind}_transform outputs "
            f"for layer {label}."
        )
