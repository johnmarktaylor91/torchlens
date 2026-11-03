"""Engine hook table: callbacks the settle and read layers register at import.

The engine (capture, recording, backends) must not import the layers above it.
Where it needs a service those layers provide, it calls a slot here; the owning
layer fills the slot when it is imported. Slots fail closed: a service the engine
relies on for correctness (the metadata-invariant tripwire) is loaded on demand
when nothing registered it, never skipped.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._data_classes.trace import Trace

__all__ = [
    "register_metadata_invariants",
    "registered_metadata_invariants",
    "run_metadata_invariants",
]

MetadataInvariantsHook = Callable[["Trace"], bool]

_METADATA_INVARIANTS: MetadataInvariantsHook | None = None


def register_metadata_invariants(hook: MetadataInvariantsHook) -> MetadataInvariantsHook:
    """Install ``hook`` as the metadata-invariant tripwire and return it.

    ``torchlens.validation`` registers ``check_metadata_invariants`` when it is
    imported; tests may install a counting wrapper around it.
    """
    global _METADATA_INVARIANTS
    _METADATA_INVARIANTS = hook
    return hook


def registered_metadata_invariants() -> MetadataInvariantsHook | None:
    """Return the installed tripwire, or ``None`` before registration."""
    return _METADATA_INVARIANTS


def run_metadata_invariants(trace: Any) -> bool:
    """Run the metadata-invariant tripwire on ``trace``.

    Raises the tripwire's ``MetadataInvariantError`` on the first failing
    invariant and returns ``True`` otherwise. When nothing has registered yet,
    the validation package is imported so the check always runs: an empty slot
    never means "skip".
    """
    hook = _METADATA_INVARIANTS
    if hook is None:
        import torchlens.validation  # noqa: F401  # deferred: cycle validation (registers the hook)

        hook = _METADATA_INVARIANTS
    if hook is None:  # pragma: no cover - the validation package always registers
        raise RuntimeError("torchlens.validation did not register the metadata-invariant hook")
    return hook(trace)
