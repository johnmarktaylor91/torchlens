"""Pure batching and device-move helpers behind the extraction engine."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any

import torch

from .envelope import BatchEnvelope


def _move_nested_to_device(value: Any, device: torch.device | str | None) -> Any:
    """Move tensors in a nested value to a device.

    Parameters
    ----------
    value:
        Tensor or nested Python container.
    device:
        Target device, or ``None`` to leave values unchanged.

    Returns
    -------
    Any
        Value with tensors moved to ``device``.
    """

    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_nested_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_nested_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move_nested_to_device(item, device) for key, item in value.items()}
    return value


def _move_envelope_to_device(
    envelope: BatchEnvelope, device: torch.device | str | None
) -> BatchEnvelope:
    """Move an envelope's payloads to a device (mask included).

    Parameters
    ----------
    envelope:
        Collated envelope.
    device:
        Target device, or ``None``.

    Returns
    -------
    BatchEnvelope
        The moved envelope.
    """

    if device is None:
        return envelope
    return dataclasses.replace(
        envelope,
        args=tuple(_move_nested_to_device(item, device) for item in envelope.args),
        kwargs={key: _move_nested_to_device(item, device) for key, item in envelope.kwargs.items()},
        mask=envelope.mask.to(device) if envelope.mask is not None else None,
    )


def _iter_item_batches(stimuli: Any, batch_size: int) -> Iterator[Any]:
    """Yield raw per-batch material: tensor slices or item lists.

    Parameters
    ----------
    stimuli:
        Tensor with batch dimension or iterable stimulus set.
    batch_size:
        Number of items per batch.

    Yields
    ------
    Any
        A tensor slice, or a list of stimulus items.
    """

    if isinstance(stimuli, torch.Tensor):
        for start in range(0, stimuli.shape[0], batch_size):
            yield stimuli[start : start + batch_size]
        return
    batch: list[Any] = []
    for item in stimuli:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _snapshot(tensor: torch.Tensor, *, host_copy: bool) -> torch.Tensor:
    """Detach one stored payload into its contiguous snapshot.

    Disk mode (``host_copy=True``) takes the CPU copy the shard writer
    consumes; memory mode (``host_copy=False``) keeps the payload on the
    device the model produced it on, so in-memory results follow the
    model's device like :func:`torchlens.extract` results do.

    Parameters
    ----------
    tensor:
        The stored payload after pool/postprocess/cast.
    host_copy:
        Whether the snapshot moves to the host.

    Returns
    -------
    torch.Tensor
        The detached, contiguous snapshot.
    """

    detached = tensor.detach()
    if host_copy:
        detached = detached.cpu()
    return detached.contiguous()
