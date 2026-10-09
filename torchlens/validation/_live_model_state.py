"""Restore a validated model's state without writing into unchanged tensors.

Validation snapshots ``model.state_dict()`` before its runs and restores it
afterwards. The restore (``load_state_dict`` or the resilient in-place
fallback) copies the snapshot into every parameter and buffer, which moves each
tensor's autograd version counter even when the bytes come back equal. A user
who holds an autograd graph across ``tl.validate`` then fails at backward with
torch's "modified by an inplace operation" error. The helpers here restore
only when the live state actually differs from the snapshot, and keep the
user's accumulated gradients by reattaching the original ``.grad`` objects.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn


def _tensor_bits_equal(live: torch.Tensor, saved: torch.Tensor) -> bool:
    """Return whether two dense tensors hold bit-identical contents.

    Parameters
    ----------
    live:
        Tensor currently registered on the model.
    saved:
        Snapshot clone taken before validation.

    Returns
    -------
    bool
        ``True`` only when shape, dtype, device and every byte match. Any
        tensor whose bytes cannot be compared (meta, sparse, quantized, a
        subclass that refuses the view) reports ``False``, so the caller
        restores it exactly as before.
    """

    if (
        live.shape != saved.shape
        or live.dtype != saved.dtype
        or live.device != saved.device
        or live.layout != torch.strided
        or saved.layout != torch.strided
        or live.device.type == "meta"
        or live.is_quantized
        or saved.is_quantized
    ):
        return False
    try:
        with torch.no_grad():
            live_bytes = live.detach().reshape(-1).view(torch.uint8)
            saved_bytes = saved.detach().reshape(-1).view(torch.uint8)
            return bool(torch.equal(live_bytes, saved_bytes))
    except (RuntimeError, TypeError, NotImplementedError):
        return False


def state_dict_unchanged(model: nn.Module, snapshot: Mapping[str, Any]) -> bool:
    """Return whether ``model``'s state equals ``snapshot`` bit for bit.

    Parameters
    ----------
    model:
        Live model whose state may have been changed by a validation run.
    snapshot:
        Cloned ``state_dict`` taken before the run.

    Returns
    -------
    bool
        ``True`` when the key set matches and every entry is a tensor with
        identical bytes. Non-tensor entries (``get_extra_state`` payloads,
        quantization sidecars) report ``False`` so they always take the full
        restore.
    """

    live_state = model.state_dict(keep_vars=True)
    if set(live_state) != set(snapshot):
        return False
    for name, saved in snapshot.items():
        live = live_state[name]
        if not isinstance(saved, torch.Tensor) or not isinstance(live, torch.Tensor):
            return False
        if not _tensor_bits_equal(live, saved):
            return False
    return True


def restore_state_dict_if_changed(
    model: nn.Module,
    snapshot: Mapping[str, Any],
    restore: Callable[[nn.Module, Any], Any],
) -> None:
    """Run ``restore(model, snapshot)`` only when the live state differs.

    Parameters
    ----------
    model:
        Live model to restore.
    snapshot:
        Cloned ``state_dict`` taken before the validation run.
    restore:
        The full restore to run when anything changed (``load_state_dict`` or
        the resilient restore). When something changed it runs over the whole
        snapshot, unchanged from before, so restore fidelity is identical.
    """

    if state_dict_unchanged(model, snapshot):
        return
    restore(model, snapshot)


def load_state_dict_restore(model: nn.Module, snapshot: Any) -> None:
    """Restore ``snapshot`` with the module's own strict ``load_state_dict``.

    Parameters
    ----------
    model:
        Model to restore.
    snapshot:
        Cloned ``state_dict``.
    """

    model.load_state_dict(snapshot)


def held_parameter_grads(model: nn.Module) -> tuple[tuple[nn.Parameter, Any], ...]:
    """Record each parameter with the ``.grad`` object it holds now.

    Parameters
    ----------
    model:
        Model about to run validation passes that reset its gradients.

    Returns
    -------
    tuple[tuple[nn.Parameter, Any], ...]
        Every parameter paired with its current ``.grad`` (possibly ``None``).
    """

    return tuple((parameter, parameter.grad) for parameter in model.parameters())


def reattach_parameter_grads(held: tuple[tuple[nn.Parameter, Any], ...]) -> None:
    """Put each parameter's pre-validation ``.grad`` object back by reference.

    Reassigning the attribute never writes into the user's gradient tensors,
    so accumulated gradients survive validation with their identity, storage
    and version counter.

    Parameters
    ----------
    held:
        Output of :func:`held_parameter_grads` taken before the passes.
    """

    for parameter, grad in held:
        parameter.grad = grad
