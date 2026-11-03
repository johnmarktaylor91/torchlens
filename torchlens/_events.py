"""Engine hook table: callbacks the settle and read layers register at import.

The engine (capture, recording, backends) must not import the layers above it.
Where it needs a service those layers provide, it calls a slot here; the owning
layer fills the slot when it is imported. Slots fail closed: a service the engine
relies on for correctness (the metadata-invariant tripwire) raises when nothing
registered it, never skips. This module imports nothing above the basis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._data_classes.trace import Trace

__all__ = [
    "active_profiler_session",
    "register_active_profiler_session",
    "register_metadata_invariants",
    "registered_metadata_invariants",
    "run_metadata_invariants",
]

MetadataInvariantsHook = Callable[["Trace"], bool]

_METADATA_INVARIANTS: MetadataInvariantsHook | None = None
_ACTIVE_PROFILER_SESSION: Callable[[], Any] | None = None


def register_active_profiler_session(probe: Callable[[], Any]) -> Callable[[], Any]:
    """Install the observability substrate's active-session probe and return it."""
    global _ACTIVE_PROFILER_SESSION
    _ACTIVE_PROFILER_SESSION = probe
    return probe


def active_profiler_session() -> Any | None:
    """Return the active profiler session, or ``None``.

    A session exists only once ``torchlens._observability._session`` has been
    imported, and that module registers the probe at import, so an empty slot
    truthfully means no session.
    """
    probe = _ACTIVE_PROFILER_SESSION
    return None if probe is None else probe()


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
    invariant and returns ``True`` otherwise. An empty slot raises ``LookupError``
    rather than skipping; ``torchlens.backends._invariants`` loads the
    validation package first so the check always runs.
    """
    hook = _METADATA_INVARIANTS
    if hook is None:
        raise LookupError(
            "the metadata-invariant tripwire is not registered; import torchlens.validation"
        )
    return hook(trace)
