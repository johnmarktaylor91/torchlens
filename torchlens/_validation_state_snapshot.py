"""Restore snapshots and copy-failure reporting for forward validation.

Forward validation runs the model on deep copies and restores the live model
afterwards from a snapshot of its ``state_dict``. The snapshot used to be a
full clone on the model's own device, held beside the deep copy: on an
accelerator that put two extra copies of the weights on the card before any
activation. Here the snapshot of every plain dense accelerator tensor lives in
host memory instead. The comparison and the restore move one tensor at a time
back to the device it came from, so the restore writes exactly the bytes a
device clone would have held, through the same restore function, and only when
the live state actually changed.

The copy-failure helpers name the exception a failed ``copy.deepcopy`` raised
(for example a CUDA out-of-memory error), so a later refusal is never reported
only as a missing-capability message about unrelated plain attributes.
"""

from __future__ import annotations

import copy
import dataclasses
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .errors._base import TorchLensWarning

_HOST_DEVICE = torch.device("cpu")
COPY_FAILED_CODE = "validation_model_copy_failed"
_COPY_FAILED_POINTER = (
    "See the validation_model_copy_failed row of docs/reference/warning_contract.md."
)


class _ValidationCopyFailedWarning(TorchLensWarning, RuntimeWarning):
    """Validation could not deep-copy the model and fell back to the live model.

    A ``RuntimeWarning`` so existing filters keep matching, and a
    ``TorchLensWarning`` so it carries ``fields["code"]`` and ``fields["remedy"]``.
    """


@dataclasses.dataclass(frozen=True)
class StateSnapshot:
    """Bit-exact copy of a module ``state_dict`` taken before validation runs.

    Attributes
    ----------
    tensors:
        Snapshot tensors by ``state_dict`` key. Plain dense accelerator tensors
        are held in host memory; every other tensor (CPU, meta, quantized,
        sparse, tensor subclasses) is a detached clone on its own device, as
        before.
    source_devices:
        The device each entry was read from, which is where a restore puts it.
    metadata:
        The ``state_dict``'s private ``_metadata`` (module versions), or None.
    """

    tensors: OrderedDict[str, torch.Tensor]
    source_devices: dict[str, torch.device]
    metadata: Any


def _holds_on_host(tensor: torch.Tensor) -> bool:
    """Return whether a snapshot entry can be held in host memory bit for bit."""

    return (
        type(tensor) is torch.Tensor
        and tensor.layout == torch.strided
        and tensor.device.type not in ("cpu", "meta")
        and not tensor.is_quantized
    )


def snapshot_state_for_restore(model: nn.Module) -> StateSnapshot:
    """Snapshot ``model.state_dict()`` for a later bit-exact restore.

    Parameters
    ----------
    model:
        Module whose state should be snapshotted.

    Returns
    -------
    StateSnapshot
        Detached copies of every entry: host copies for plain dense
        accelerator tensors, same-device clones for everything else.
    """

    original_state = model.state_dict()
    tensors: OrderedDict[str, torch.Tensor] = OrderedDict()
    source_devices: dict[str, torch.device] = {}
    for name, tensor in original_state.items():
        detached = tensor.detach()
        source_devices[name] = detached.device
        if _holds_on_host(detached):
            tensors[name] = detached.to(_HOST_DEVICE, copy=True)
        else:
            tensors[name] = detached.clone()
    metadata = copy.deepcopy(getattr(original_state, "_metadata", None))
    return StateSnapshot(tensors=tensors, source_devices=source_devices, metadata=metadata)


def _on_source_device(snapshot: StateSnapshot, name: str) -> torch.Tensor:
    """Return one snapshot entry on the device it was read from."""

    saved = snapshot.tensors[name]
    source = snapshot.source_devices[name]
    return saved if saved.device == source else saved.to(source)


def state_matches_snapshot(model: nn.Module, snapshot: StateSnapshot) -> bool:
    """Return whether ``model``'s state equals ``snapshot`` bit for bit.

    Same rule as the device-local check it replaces: matching key sets and
    identical bytes, shape, dtype and device for every entry, each compared on
    the device it was snapshotted from (one entry is staged at a time).

    Parameters
    ----------
    model:
        Live model whose state may have been changed by a validation run.
    snapshot:
        Snapshot taken before the run.

    Returns
    -------
    bool
        ``True`` only when nothing changed.
    """

    from .validation._live_model_state import _tensor_bits_equal

    live_state = model.state_dict(keep_vars=True)
    if set(live_state) != set(snapshot.tensors):
        return False
    for name in snapshot.tensors:
        live = live_state[name]
        if not isinstance(live, torch.Tensor):
            return False
        if not _tensor_bits_equal(live, _on_source_device(snapshot, name)):
            return False
    return True


def _restore_entries(model: nn.Module, snapshot: StateSnapshot) -> OrderedDict[str, torch.Tensor]:
    """Build the ``state_dict`` a restore loads, staging only changed entries.

    An entry whose live tensor already holds the snapshot's bytes is passed as
    a detached alias of that live tensor (loading it is a no-op copy); every
    other entry is the snapshot moved back to its source device.
    """

    from .validation._live_model_state import _tensor_bits_equal

    live_state = model.state_dict(keep_vars=True)
    entries: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name in snapshot.tensors:
        staged = _on_source_device(snapshot, name)
        live = live_state.get(name)
        if isinstance(live, torch.Tensor) and _tensor_bits_equal(live, staged):
            entries[name] = live.detach()
        else:
            # Never hand out the snapshot's own tensor: a module whose
            # ``_load_from_state_dict`` assigns would alias the snapshot.
            entries[name] = staged.clone() if staged is snapshot.tensors[name] else staged
    if snapshot.metadata is not None:
        entries._metadata = copy.deepcopy(snapshot.metadata)  # type: ignore[attr-defined]
    return entries


def restore_snapshot_if_changed(
    model: nn.Module,
    snapshot: StateSnapshot,
    restore: Callable[[nn.Module, Any], Any],
) -> None:
    """Run ``restore(model, entries)`` only when the live state differs.

    Parameters
    ----------
    model:
        Live model to restore.
    snapshot:
        Snapshot taken before the validation run.
    restore:
        The full restore (``load_state_dict`` or the resilient restore). It
        receives every key on its source device, as with a device clone.
    """

    if state_matches_snapshot(model, snapshot):
        return
    restore(model, _restore_entries(model, snapshot))


def describe_copy_failure(exc: BaseException) -> str:
    """Return ``"ExceptionType: message"`` for a failed model deep copy."""

    detail = str(exc).strip().splitlines()
    first_line = detail[0] if detail else ""
    if len(first_line) > 300:
        first_line = first_line[:297] + "..."
    return f"{type(exc).__name__}: {first_line}" if first_line else type(exc).__name__


def _is_out_of_memory(exc: BaseException) -> bool:
    """Return whether a deep-copy failure was a device or host allocation failure."""

    out_of_memory = getattr(torch, "OutOfMemoryError", None)
    if out_of_memory is not None and isinstance(exc, out_of_memory):
        return True
    return isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()


def copy_failure_remedy(exc: BaseException) -> str:
    """Return the spelling that works for a failed validation deep copy."""

    if _is_out_of_memory(exc):
        return (
            "free device memory so a second copy of the weights fits, or validate "
            "on the host: tl.validate(model.to('cpu'), inputs, scope='forward')"
        )
    return (
        "make copy.deepcopy(model) succeed (for example define __deepcopy__ on the "
        "class holding the attribute it fails on), then call "
        "tl.validate(model, inputs, scope='forward') again"
    )


def warn_copy_failed(model: nn.Module, exc: BaseException, *, run: str) -> None:
    """Warn that a validation deep copy failed, naming the exception and the remedy.

    Parameters
    ----------
    model:
        The model the user passed to validation.
    exc:
        The exception ``copy.deepcopy`` raised.
    run:
        Which validation run lost its copy: ``"ground-truth"`` or ``"replay"``.
    """

    model_name = type(model).__name__
    if run == "ground-truth":
        lead = (
            "TorchLens validate_forward_pass could not deepcopy the model "
            f"({model_name}) for the ground-truth run"
        )
        fallback = (
            "it falls back to state_dict snapshot/restore of the live model, and "
            "non-registered mutable state may cause false negatives"
        )
    else:
        lead = (
            "TorchLens validation replay against live model state; model could not "
            f"be copied ({model_name})"
        )
        fallback = "the pristine re-trace check is skipped"
    warnings.warn(
        _ValidationCopyFailedWarning(
            f"{lead}: {describe_copy_failure(exc)}. Validation runs the model on a copy "
            f"so the live model is never touched; without one {fallback}. "
            f"{_COPY_FAILED_POINTER} Remedy: {copy_failure_remedy(exc)}",
            code="validation_model_copy_failed",
            copy_error=describe_copy_failure(exc),
            model_type=f"{type(model).__module__}.{type(model).__qualname__}",
        ),
        stacklevel=4,
    )


def copy_failure_clause(exc: BaseException | None) -> str:
    """Return the refusal clause naming why the copy failed, or an empty string."""

    if exc is None:
        return ""
    return (
        f" The deepcopy failed with {describe_copy_failure(exc)}; "
        f"remedy: {copy_failure_remedy(exc)}."
    )
