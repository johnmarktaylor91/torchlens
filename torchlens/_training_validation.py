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
    """Raised when ``train_mode=True`` conflicts with capture configuration.

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
            f"{api_name} does not support torch.compile'd models in train_mode: "
            "dynamo replaces the Python forward with a compiled graph that "
            "bypasses TorchLens' function wrappers. Call the API on the "
            "original un-compiled model."
        )


def validate_training_compatibility(
    *,
    train_mode: bool,
    streaming: Any,
    detach_saved_tensors: bool | None = None,
    save_activations_to: str | Path | None = None,
    inference_mode_active: bool | None = None,
) -> None:
    """Validate user options for autograd-retaining training captures.

    Parameters
    ----------
    train_mode:
        Whether the caller requested training-compatible activation retention.
    streaming:
        Streaming options object, or ``None``.
    detach_saved_tensors:
        Whether saved tensors are explicitly detached.
    save_activations_to:
        Legacy disk-save path, if supplied outside grouped streaming options.
    inference_mode_active:
        Whether PyTorch inference mode is active at validation time.

    Raises
    ------
    TrainingModeConfigError
        If ``train_mode=True`` conflicts with disk persistence, explicit
        detaching, or active inference mode.
    """

    if not train_mode:
        return

    streaming_bundle_path = getattr(streaming, "bundle_path", None)
    if save_activations_to is not None or streaming_bundle_path is not None:
        raise TrainingModeConfigError(
            "train_mode=True is not compatible with slow/replay activation disk saves"
        )
    if detach_saved_tensors is True:
        raise TrainingModeConfigError(
            "train_mode=True requires detach_saved_tensors=False so gradients can propagate"
        )
    if inference_mode_active is True:
        raise TrainingModeConfigError(
            "train_mode=True cannot run while PyTorch inference mode is active; "
            "inference tensors cannot retain autograd history"
        )
