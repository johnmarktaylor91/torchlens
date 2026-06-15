"""Shared validation helpers for training-compatible capture modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .errors._base import ConfigurationError

_NON_GRAD_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
}
for _dtype_name in ("uint16", "uint32", "uint64"):
    if hasattr(torch, _dtype_name):
        _NON_GRAD_DTYPES.add(getattr(torch, _dtype_name))


class TrainingModeConfigError(ConfigurationError, ValueError):
    """Raised when ``backward_ready=True`` conflicts with capture configuration.

    This is the slow/replay sibling of ``tl.fastlog.RecordingConfigError``:
    both are ``ValueError`` subclasses for user configuration mistakes, but
    this class specifically covers autograd-retaining training captures.
    """


def reject_compiled_model(model: nn.Module, *, api_name: str) -> None:
    """Reject ``torch.compile`` wrappers for training-mode capture APIs.

    Parameters
    ----------
    model:
        Candidate model supplied by the user.
    api_name:
        Public API name used in the error message.

    Raises
    ------
    RuntimeError
        If ``model`` is a compiled ``OptimizedModule`` wrapper.
    """

    try:
        from torch._dynamo.eval_frame import OptimizedModule
    except ImportError:
        optimized_module_type: type[nn.Module] | None = None
    else:
        optimized_module_type = OptimizedModule

    is_optimized_module = optimized_module_type is not None and isinstance(
        model, optimized_module_type
    )
    if not is_optimized_module:
        is_optimized_module = model.__class__.__name__ == "OptimizedModule"

    if is_optimized_module:
        raise RuntimeError(
            f"{api_name} does not support torch.compile'd models in backward_ready: "
            "dynamo replaces the Python forward with a compiled graph that "
            "byops TorchLens' function wrappers. Call the API on the "
            "original un-compiled model."
        )


def validate_training_compatibility(
    *,
    backward_ready: bool,
    streaming: Any,
    detach_saved_activations: bool | None = None,
    save_outs_to: str | Path | None = None,
    inference_mode_active: bool | None = None,
    inference_only: bool = False,
    inference_only_conflicts: tuple[str, ...] = (),
) -> None:
    """Validate user options for autograd-retaining training captures.

    Parameters
    ----------
    backward_ready:
        Whether the caller requested training-compatible out retention.
    streaming:
        Streaming options object, or ``None``.
    detach_saved_activations:
        Whether saved tensors are explicitly detached.
    save_outs_to:
        Legacy disk-save path, if supplied outside grouped streaming options.
    inference_mode_active:
        Whether PyTorch inference mode is active at validation time.
    inference_only:
        Whether the user requested no-grad forward capture.
    inference_only_conflicts:
        Explicit backward-related flags that require a retained autograd graph.

    Raises
    ------
    TrainingModeConfigError
        If ``backward_ready=True`` conflicts with disk persistence, explicit
        detaching, active inference mode, or if ``inference_only=True``
        conflicts with explicit backward-related capture flags.
    """

    if inference_only and inference_only_conflicts:
        offending = ", ".join(inference_only_conflicts)
        raise TrainingModeConfigError(
            "inference_only=True cannot be combined with backward-related capture "
            f"({offending}); these require the autograd graph that inference_only discards. "
            "Drop inference_only or drop the backward flag."
        )

    if not backward_ready:
        return

    streaming_bundle_path = getattr(streaming, "bundle_path", None)
    if save_outs_to is not None or streaming_bundle_path is not None:
        raise TrainingModeConfigError(
            "backward_ready=True is not compatible with slow/replay out disk saves"
        )
    if detach_saved_activations is True:
        raise TrainingModeConfigError(
            "backward_ready=True requires detach_saved_activations=False so grads can propagate"
        )
    if inference_mode_active is True:
        raise TrainingModeConfigError(
            "backward_ready=True cannot run while PyTorch inference mode is active; "
            "inference tensors cannot retain autograd history"
        )
