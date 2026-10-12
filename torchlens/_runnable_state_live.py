"""Live-model state restore for the run() and capture brackets.

The default live ``run()`` snapshots the model's declared state (parameters and
buffers) before executing and restores it afterwards (``_runnable_state``
takes the snapshot; this module restores it).
"""

from __future__ import annotations

import torch

from . import _state
from ._runnable_state import LiveDeclaredStateSnapshot

__all__ = (
    "LiveStateRestoreFailure",
    "restore_live_declared_state",
)


class LiveStateRestoreFailure(RuntimeError):
    """Internal carrier for a declared-state restore that failed mid-bracket.

    The transaction converts this into the typed ``run_state_restore_failed``
    refusal with the structured fields the contract promises; the failing
    restore exception is chained as ``__cause__``.
    """

    def __init__(self, state_dict_name: str, groups_restored: int) -> None:
        super().__init__(
            f"declared-state restore failed at {state_dict_name!r} after "
            f"{groups_restored} alias group(s) were restored"
        )
        self.state_dict_name = state_dict_name
        self.groups_restored = groups_restored


def _declared_slot_unchanged(original: torch.Tensor, clone: torch.Tensor) -> bool:
    """Return whether a live declared-state slot still holds its snapshot bytes.

    Byte-exact and NaN-safe (the shared state-write audit compare). Any compare
    failure answers ``False`` so the caller restores rather than trusts.
    """

    from .backends.torch.rescue import _state_bytes_equal

    try:
        return (
            original.shape == clone.shape
            and original.dtype == clone.dtype
            and original.device == clone.device
            and _state_bytes_equal(original, clone)
        )
    except Exception:
        return False


def restore_live_declared_state(snapshot: LiveDeclaredStateSnapshot) -> None:
    """Restore a declared-state snapshot onto the live model (finally bracket).

    Runs under ``no_grad`` and ``pause_logging``, one value restore per alias
    group (distinct live object), then re-binds any slot the forward reassigned
    to a different object back to its ORIGINAL object -- so repeated run()
    calls leave the model bit-identical with ``a is b`` alias semantics intact.
    Raises :class:`LiveStateRestoreFailure` (original exception chained) on the
    first failed restore; the caller owns the state-compromised consequence
    (L4 5.4).

    Only a slot whose bytes differ from its snapshot is written back. An
    unconditional ``copy_`` bumps the version counter of every parameter and
    buffer the forward left untouched, and autograd saved those tensors (conv
    weights, eval-mode BatchNorm running statistics) at their old version, so a
    later ``backward()`` through the run's graph-attached activations failed
    with torch's in-place modification error.
    """

    restored: set[int] = set()
    with _state.pause_logging(), torch.no_grad():
        for _module, _registry_attr, _local_name, canonical, original in snapshot.bindings:
            key = id(original)
            if key in restored:
                continue
            try:
                clone = snapshot.clones[key]
                if not _declared_slot_unchanged(original, clone):
                    original.copy_(clone)
            except Exception as exc:
                raise LiveStateRestoreFailure(canonical, len(restored)) from exc
            restored.add(key)
        for module, registry_attr, local_name, canonical, original in snapshot.bindings:
            try:
                registry = getattr(module, registry_attr, None)
                if isinstance(registry, dict) and registry.get(local_name) is not original:
                    registry[local_name] = original
            except Exception as exc:
                raise LiveStateRestoreFailure(canonical, len(restored)) from exc
