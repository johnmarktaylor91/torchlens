"""Shared validation helpers for training-compatible capture modes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

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


class TrainingModeConfigError(ValueError):
    """Raised when ``train_mode=True`` conflicts with capture configuration.

    This is the slow/replay sibling of ``tl.fastlog.RecordingConfigError``:
    both are ``ValueError`` subclasses for user configuration mistakes, but
    this class specifically covers autograd-retaining training captures.
    """


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
