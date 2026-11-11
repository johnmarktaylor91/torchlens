"""Process-wide slots of the host-nondeterminism monitor: the active window and patch stacks.

Kept apart from the monitor class so the stream helpers (:mod:`.streams`, :mod:`.seeding`)
can bracket their own torch RNG touches under :func:`_suppress_active_monitor_marks`
without importing the monitor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .monitor import host_nondeterminism_monitor

import sys as _sys_module
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

_ACTIVE_MONITOR: host_nondeterminism_monitor | None = None
"""The capture-scoped monitor currently installed, or ``None``.

Published as the LAST statement of ``__enter__`` and restored to the PREVIOUS occupant
FIRST in the teardown, so readers never observe a partially-installed window and a
nested window's exit cannot null the slot mid-outer-window. Captures do not nest
(``active_logging`` rejects nested captures), so a single slot is sufficient.
"""

_WARMUPS: list[Callable[[], None]] = []
"""Callables a monitor window runs before it arms, in registration order.

Higher layers register here, at import, the lazy imports a first capture would otherwise
trigger INSIDE the window (an in-window import marks the finders' entropy draws and walks
their frames as inventory roots). The monitor never imports upward to warm them itself.
"""


def register_warmup(fn: Callable[[], None]) -> None:
    """Register ``fn`` to run before every monitor window arms (idempotent).

    Parameters
    ----------
    fn:
        A no-argument callable; a raise is benign (the window arms regardless and the
        in-window retry's draws are then marked honestly).
    """

    if fn not in _WARMUPS:
        _WARMUPS.append(fn)


class _PatchStackEntry:
    """One live monitor patch on ``(holder, name)``, spliceable out of order.

    ``holder`` is retained STRONGLY so its ``id()`` cannot be reused by another object
    while the entry is stacked (the stack is keyed by ``(id(holder), name)``).
    """

    __slots__ = ("holder", "name", "original", "wrapper")

    def __init__(self, holder: Any, name: str, original: Any, wrapper: Any) -> None:
        self.holder = holder
        self.name = name
        self.original = original
        self.wrapper = wrapper


_PATCH_STACKS: dict[tuple[int, str], list[_PatchStackEntry]] = {}
"""Live monitor patches per ``(id(holder), name)``, innermost last.

Restoration is SPLICE-aware. A monitor unwinding out of order (a non-LIFO overlap) used
to hand the true original back and then have the outer window's restore write ITS
snapshot -- which was the inner wrapper -- leaving ``time.time`` / ``os.urandom`` /
``np.random.default_rng`` wrapped for the life of the process. Splicing instead rewrites
the successor entry's recorded original, so whoever restores last always writes the
genuine pre-monitor value.
"""


@contextmanager
def _suppress_active_monitor_marks() -> Iterator[None]:
    """Suppress channel marks for a TorchLens-OWNED RNG bookkeeping bracket.

    TorchLens's own seed / snapshot / restore helpers legitimately touch the torch RNG
    APIs that the registry rows monitor -- per-op state logging and in-forward
    intervention replay run INSIDE the capture window (``set_rng_from_saved_states``
    calls the patched ``torch.random.set_rng_state``). A TorchLens-initiated
    restore is NOT model host nondeterminism, so these helpers bracket themselves with
    the active monitor's ``_mark``-choke-point suppression (surface-complete: wrapper
    marks, held-code call marks, and default-generator receiver marks all funnel
    through the same choke). USER code never executes inside these brackets -- the
    replayed op itself runs OUTSIDE the bracket in
    :func:`execute_with_restored_rng_autocast`. No active monitor -> no-op.
    """

    monitor = _ACTIVE_MONITOR
    if monitor is None:
        yield
        return
    with monitor._monitor_internal_probe():
        yield


def _skip_retired_hooks(candidate: Any) -> Any:
    """Return the first chain link that is not a torn-down (or freed) monitor's hook.

    A non-LIFO overlap used to restore an already-retired window's hook into a
    profile slot; the sys slot self-heals on its next event, but the
    ``threading`` registration only seeds NEW threads, so a dead hook parked
    there never fires again on the main thread and taxes (and misclassifies
    into) every later capture. Each hook names its owning monitor weakly and its
    own predecessor, so the restore skips straight to the newest LIVE link.
    """

    seen: set[int] = set()
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        owner_ref = getattr(candidate, "_tl_owner_ref", None)
        if owner_ref is None:
            break  # not a monitor hook
        owner = owner_ref()
        if owner is not None and not getattr(owner, "_hooks_retired", False):
            break
        candidate = getattr(candidate, "_tl_predecessor", None)
    return candidate


def _retired_hook_event(
    own: Any,
    setter: Any,
    predecessor: Any,
    frame: Any,
    event: str,
    arg: Any,
) -> None:
    """Handle one profile event reaching a hook whose window is over (or whose monitor is freed).

    ``threading.setprofile`` only seeds NEW threads, so a worker started in-window keeps
    the hook as its thread-local profile function forever: a permanent per-call tax, and
    (worse) a later capture's draws on that thread classifying into this dead window's
    discarded result. A non-LIFO overlap can likewise hand the hook back to the process.
    Self-uninstall on the first event after teardown, but ONLY when this hook is the one
    currently installed on this thread: while it is merely a LINK in a live successor's
    chain, uninstalling would tear down that successor's window too. ``own`` is a weak
    handle to the hook and ``setter`` the held pre-patch ``sys.setprofile`` (a LIVE later
    window may have its swap-detection wrapper on the module attr), so neither the hook
    nor its monitor is kept alive by this path.
    """

    try:
        own_hook = own()
        if own_hook is not None and _sys_module.getprofile() is own_hook:
            setter(predecessor)
    except Exception:
        pass
    if predecessor is not None:
        try:
            predecessor(frame, event, arg)
        except Exception:
            pass
    return None


def _install_spawned_thread_hook(owner: Any) -> None:
    """Install the window's threading hook on a raw ``_thread`` spawn, if the window is open.

    ``owner`` is a weak handle, so a raw thread running past the window keeps neither the
    monitor nor a strong local naming it (the target runs after this frame returns). The
    held pre-patch ``sys.setprofile`` is used: the module attr may hold a (this or a later)
    window's swap-detection wrapper.
    """

    monitor = owner()
    hook = None if monitor is None else monitor._threading_hook
    if monitor is not None and hook is not None and not monitor._torn_down:
        (monitor._orig_sys_setprofile or _sys_module.setprofile)(hook)
