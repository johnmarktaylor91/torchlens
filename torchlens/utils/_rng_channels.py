"""Thread routing, owner-join settlement, and global-engine state rows for the monitor.

Split from ``torchlens.utils.rng`` under the R43 file-size ratchet. This module owns:

* the monitor's result record (:class:`HostRngMonitorResult`);
* the foreign-thread routing of THREAD-LOCAL channel touches -- pure READS (the clock
  family, the OS-entropy/construction funnels) and PRIVATE-instance draw primitives --
  plus the two wrapper builders that feed it;
* the owner-thread SYNCHRONIZATION join (W051 / AUD-CODE 2.16): the closed vocabulary
  of blocking stdlib primitives the capture's owner thread can wait on, their ``call``
  classification, and the teardown settlement that promotes disclosed foreign reads to
  monitor UNCERTAINTY exactly when the owner joined another thread in-window;
* the Python / NumPy GLOBAL-engine state surface (W051 / AUD-CODE 2.17): module-attr
  rows for ``random.getstate/setstate/seed`` and ``numpy.random.get_state/set_state/
  seed`` plus the held-reference ``c_call`` prefixes, all ``replayable_read``-class.

``rng.py`` keeps the monitor itself and thin delegates; nothing here is public surface.

Thread model. The monitor's module/class patches are process-wide, so a benign background
thread left running by an earlier test (a tracker flush loop, an asyncio manager) can read
``time.time()`` or draw a private ``random.Random`` (``tempfile.mkstemp``) inside another
capture's monitor window. The runnable contract rules that a benign background thread
never ceilings a capture -- but a PRE-EXISTING thread whose value the forward WAITS FOR
(``PRE_POOL.submit(time.time).result()``, ``run_coroutine_threadsafe(...).result()``) is
host nondeterminism fed into the forward, and settling it ``verified`` is the contract's
forbidden class (the FLAKEHUNT over-correction). The split that keeps both honest: a
foreign thread's thread-local touch lands in ``foreign_thread_reads`` (never a ceiling by
itself); the owner's in-window blocking synchronizations land in ``owner_thread_waits``;
at teardown the two JOIN -- every disclosed foreign read is promoted to monitor
uncertainty (``foreign_thread_read_joined:<channel>``) iff some wait's counterpart was
outside the capture's thread universe or unresolvable. A flush loop never makes the
capture wait; every feed-in topology does. Owner-thread and
in-window-started-thread touches ceiling exactly as before; mutating surfaces (global
engines, generator draws on shared engines) stay thread-agnostic and never route here.
"""

from __future__ import annotations

import importlib
import random as _random_module
import sys as _sys_module
import threading as _threading_module
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - the import cycle is typing-only
    from .rng import host_nondeterminism_monitor


class HostRngMonitorResult:
    """Outcome of one capture-scoped host-nondeterminism monitoring window."""

    __slots__ = (
        "channels",
        "replayable_reads",
        "uncertain",
        "uncertain_detail",
        "foreign_thread_reads",
        "owner_thread_waits",
        "owner_unhooked_joins",
        "owner_unattributed_waits",
    )

    def __init__(self) -> None:
        self.channels: set[str] = set()
        # r65 CLUSTER Z: torch RNG reads fully determined by the capture seed
        # (``initial_seed`` family) and, as of W051, the Python/NumPy GLOBAL-engine
        # state surface (``random.getstate``/``setstate``/``seed`` and the legacy
        # ``numpy.random`` twins). A member sets ``host_rng_consumed`` WITHOUT
        # poisoning the capture seed: a run at the capture seed stays verified and any
        # other/absent seed ceilings -- exactly the python-``random`` branch semantics.
        # Kept apart from ``channels``, whose members ceiling permanently.
        self.replayable_reads: set[str] = set()
        self.uncertain: bool = False
        # Actionable named-thread / failure detail for the INCOMPLETE ceiling so the
        # readiness diagnostic and ``tl.compat.report()`` can name the offending domain.
        self.uncertain_detail: tuple[str, ...] = ()
        # Thread-local channel touches (clock family / OS-entropy funnels / PRIVATE
        # ``random.Random`` instance draws) made by a thread that is neither the
        # capture owner nor in-window-started: ambient host activity outside the
        # single-threaded capture model (a tracker flush loop, an asyncio manager, a
        # GUI timer, a ``tempfile`` name sequence). DISCLOSED here, session-only; never
        # a ceiling BY THEMSELVES. They settle at teardown against
        # ``owner_thread_waits`` (see :func:`settle_foreign_reads`).
        self.foreign_thread_reads: set[str] = set()
        # W051 (AUD-CODE 2.16): the blocking stdlib synchronization primitives the OWNER
        # thread entered in-window from a non-TorchLens frame (``threading.Event.wait``,
        # ``concurrent.futures.Future.result``, ...), session-only. ``owner_unhooked_joins``
        # is the subset whose resolved counterpart thread lies OUTSIDE the capture's
        # thread universe (settles uncertain by itself); ``owner_unattributed_waits`` the
        # subset with no resolvable counterpart (promotes disclosed foreign reads). The
        # remainder -- joins on the capture's own in-window workers -- is disclosure only.
        self.owner_thread_waits: set[str] = set()
        self.owner_unhooked_joins: set[str] = set()
        self.owner_unattributed_waits: set[str] = set()


def foreign_thread_call(monitor: host_nondeterminism_monitor) -> bool:
    """Whether the current thread is outside the capture's thread universe.

    The capture's thread universe is the owner thread plus every thread STARTED
    inside the window (registered race-free at bootstrap by the in-window
    threading/raw-spawn hooks). Anything else is ambient host activity a
    module-attr patch can still observe -- a tracker flush loop, an asyncio
    manager -- whose thread-local touches must not ceiling the capture BY
    THEMSELVES (the smoke-gate UNVERIFIABLE flake family: a background thread's
    ``time.time()`` landing inside a short monitor window ceilinged an honest
    deterministic capture). Fail-closed interplay: if the profile-hook install
    step failed, in-window threads are unregistered and their reads would
    classify foreign, but that same failure already flagged
    ``monitor_install_failed:profile_hooks`` uncertainty, so completeness is
    INCOMPLETE regardless -- a dropped mark can never rescue a verdict.
    """

    ident = _threading_module.get_ident()
    return ident != monitor._owner_thread and ident not in monitor._in_window_thread_idents


def mark_thread_routed(monitor: host_nondeterminism_monitor, channel: str) -> None:
    """Route one THREAD-LOCAL channel touch by thread origin.

    Owner-thread and in-window-thread touches CEILING through the monitor's
    ``_mark`` exactly as before. Foreign-thread touches land in the session-only
    ``foreign_thread_reads`` disclosure instead: these channels read no shared
    replayable engine state, so a foreign touch cannot desync the capture's replay
    -- the only escape is value feed-in through a synchronization channel, which
    :func:`settle_foreign_reads` closes by joining the disclosure with the owner's
    in-window waits at teardown. Mutating surfaces (global engines, generator draws
    on shared engines) deliberately do NOT route through here: a foreign draw on a
    shared engine desyncs replay no matter which thread made it.
    """

    if foreign_thread_call(monitor):
        if not monitor._suppress_self_marks:
            monitor.result.foreign_thread_reads.add(channel)
        return
    monitor._mark(channel)


def mark_pure_read(monitor: host_nondeterminism_monitor, channel: str) -> None:
    """Route one pure-READ channel touch (clock / OS entropy) by thread origin."""

    mark_thread_routed(monitor, channel)


def entropy_wrapper(monitor: host_nondeterminism_monitor, original: Any, channel: str) -> Any:
    """Build a marking passthrough wrapper for one OS-entropy channel."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Mark the entropy channel (thread-routed), then delegate to the original."""

        mark_pure_read(monitor, channel)
        return original(*args, **kwargs)

    return wrapper


def clock_wrapper(
    monitor: host_nondeterminism_monitor,
    original: Any,
    channel: str,
    time_arg_index: int | None,
) -> Any:
    """Build a marking passthrough wrapper for one clock channel.

    ``time_arg_index`` names the positional argument that makes the call a pure
    transform of a caller-supplied time (``localtime(ts)``); when it is
    supplied and non-``None`` the call reads no clock and is not marked.
    """

    tl_ids = monitor._tl_globals_ids

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Mark the clock channel for an implicit-now read from a non-TorchLens frame.

        Frames owned by TorchLens module globals are exempt: the per-op capture
        clock reads would otherwise self-ceiling every capture. An unreadable
        caller frame is treated as foreign, which over-marks rather than under-marks.
        """

        explicit_time = (
            time_arg_index is not None
            and len(args) > time_arg_index
            and args[time_arg_index] is not None
        )
        if not explicit_time:
            try:
                caller_globals_id = id(_sys_module._getframe(1).f_globals)
            except Exception:
                caller_globals_id = -1
            if caller_globals_id not in tl_ids:
                mark_pure_read(monitor, channel)
        return original(*args, **kwargs)

    return wrapper


# ---- Owner-thread synchronization vocabulary (W051 / AUD-CODE 2.16) --------------------
#
# Blocking Python-level entry points through which the capture's OWNER thread can wait
# for another thread's result. Rows are ``(display, module, qualname-or-None, method,
# condition_path)``. Two observation layers, both built from this ONE table:
#
# * CLASS PATCH (counterpart-resolving): every class-method row is wrapped in-window; on
#   the owner thread the wrapper runs the original and then reads WHICH threads notified
#   the primitive's underlying ``threading.Condition`` (``condition_path`` walks from the
#   receiver to it; ``Condition.notify``/``notify_all`` are stamped per window in a weak
#   registry, never on the user's object; ``Thread.join`` resolves to the joined thread's
#   ident directly). A counterpart outside the capture's thread universe is an UNHOOKED
#   join -- the evidence that promotes disclosed foreign reads to uncertainty (and the
#   disclosure of the clause-(iv) residual: that thread's profile-only channels are
#   unobservable). A hooked-only counterpart needs nothing extra (its reads ceiling
#   directly). No resolvable counterpart (a pre-window-notified primitive, an asyncio
#   loop run, a module-level ``wait``) is an UNATTRIBUTED wait, promoting like unhooked.
# * CALL-EVENT (entry witness, held-reference safe): the function's CODE object rides the
#   held-code registry, so a pre-bound ``wait = event.wait`` alias that bypasses the class
#   patch is still witnessed at entry -- as an unattributed wait.
#
# ``queue.SimpleQueue.get`` is a C method descriptor and is classified by receiver type
# on ``c_call`` instead (unattributed). The vocabulary is CLOSED and self-describing;
# every row must resolve on a supported CPython (a stdlib rename is a failing registry
# test, never a silent under-witness).

_THREAD_COUNTERPART = "@thread"
"""``condition_path`` token: the counterpart is the receiver thread's own ident."""

OWNER_SYNC_PRIMITIVES: tuple[tuple[str, str, str | None, str, tuple[str, ...] | None], ...] = (
    ("threading.Event.wait", "threading", "Event", "wait", ("_cond",)),
    ("threading.Condition.wait", "threading", "Condition", "wait", ()),
    ("threading.Condition.wait_for", "threading", "Condition", "wait_for", ()),
    ("threading.Thread.join", "threading", "Thread", "join", (_THREAD_COUNTERPART,)),
    ("threading.Semaphore.acquire", "threading", "Semaphore", "acquire", ("_cond",)),
    ("threading.Barrier.wait", "threading", "Barrier", "wait", ("_cond",)),
    (
        "concurrent.futures.Future.result",
        "concurrent.futures._base",
        "Future",
        "result",
        ("_condition",),
    ),
    (
        "concurrent.futures.Future.exception",
        "concurrent.futures._base",
        "Future",
        "exception",
        ("_condition",),
    ),
    ("concurrent.futures.wait", "concurrent.futures._base", None, "wait", None),
    ("concurrent.futures.as_completed", "concurrent.futures._base", None, "as_completed", None),
    ("queue.Queue.get", "queue", "Queue", "get", ("not_empty",)),
    (
        "asyncio.BaseEventLoop.run_until_complete",
        "asyncio.base_events",
        "BaseEventLoop",
        "run_until_complete",
        None,
    ),
    (
        "asyncio.BaseEventLoop.run_forever",
        "asyncio.base_events",
        "BaseEventLoop",
        "run_forever",
        None,
    ),
    (
        "multiprocessing.pool.ApplyResult.get",
        "multiprocessing.pool",
        "ApplyResult",
        "get",
        ("_event", "_cond"),
    ),
    (
        "multiprocessing.pool.ApplyResult.wait",
        "multiprocessing.pool",
        "ApplyResult",
        "wait",
        ("_event", "_cond"),
    ),
)
"""Closed vocabulary of owner-thread blocking synchronization entry points."""

OWNER_SYNC_DISPOSITION = "owner_sync"
"""Disposition token under which sync code objects ride the held-code registry."""

# Top-level stdlib packages whose frames sit BETWEEN a sync primitive and the code that
# initiated the wait (``Event.wait`` -> ``Condition.wait``; ``Future.result`` called from
# ``concurrent.futures.thread``). The initiator is the first frame outside them.
_SYNC_STDLIB_TOPS: frozenset[str] = frozenset(
    {"threading", "queue", "concurrent", "asyncio", "multiprocessing", "selectors", "_thread"}
)
_INITIATOR_WALK_LIMIT = 64


def owner_sync_code_marks() -> dict[int, tuple[str, str]]:
    """Resolve every :data:`OWNER_SYNC_PRIMITIVES` row to ``id(code) -> (display, disposition)``.

    Modules are imported here (stdlib only, once per process) so a forward that imports
    ``concurrent.futures`` IN-WINDOW is still classified -- resolving only already-loaded
    modules would be a silent under-witness. Runs before any patch installs, so the
    import-time side effects of these stdlib modules are never mis-marked. A row that
    fails to resolve is skipped here and caught by :func:`unresolved_owner_sync_rows`
    (the registry meta-test's tripwire).
    """

    marks: dict[int, tuple[str, str]] = {}
    for display, module_name, qualname, method, _path in OWNER_SYNC_PRIMITIVES:
        function = _resolve_sync_function(module_name, qualname, method)
        code = getattr(function, "__code__", None)
        if code is not None:
            marks[id(code)] = (display, OWNER_SYNC_DISPOSITION)
    return marks


def unresolved_owner_sync_rows() -> tuple[str, ...]:
    """Display names of vocabulary rows whose function does not resolve on this Python."""

    return tuple(
        display
        for display, module_name, qualname, method, _path in OWNER_SYNC_PRIMITIVES
        if getattr(_resolve_sync_function(module_name, qualname, method), "__code__", None) is None
    )


def _resolve_sync_holder(module_name: str, qualname: str | None) -> Any | None:
    """Return the module or class that owns one vocabulary row's method, or ``None``."""

    try:
        module = _sys_module.modules.get(module_name) or importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - an unresolvable vocabulary row is unwitnessed, never a capture failure
        return None
    return module if qualname is None else getattr(module, qualname, None)


def _resolve_sync_function(module_name: str, qualname: str | None, method: str) -> Any | None:
    """Return one vocabulary row's function object, or ``None``."""

    holder = _resolve_sync_holder(module_name, qualname)
    return None if holder is None else getattr(holder, method, None)


def owner_sync_c_call_primitive(receiver: Any, name: Any) -> str | None:
    """Classify a ``c_call`` receiver/method as a C-level owner sync primitive, or ``None``."""

    if name != "get":
        return None
    simple_queue = getattr(_sys_module.modules.get("_queue"), "SimpleQueue", None)
    if simple_queue is not None and isinstance(receiver, simple_queue):
        return "queue.SimpleQueue.get"
    return None


class _OwnerSyncSession:
    """Per-window state of the owner-synchronization layer.

    ``stamps`` maps each notified ``threading.Condition`` (weakly held -- the user's
    object is never mutated) to the idents of the threads that notified it inside the
    window; ``depth`` is the owner thread's nesting inside a class-patched wait wrapper
    (a thread-local, so a foreign thread's wait can never mask the owner's), which the
    call-event layer consults to leave attribution to the wrapper.
    """

    __slots__ = ("monitor", "stamps", "lock", "_depth")

    def __init__(self, monitor: host_nondeterminism_monitor) -> None:
        self.monitor = monitor
        self.stamps: weakref.WeakKeyDictionary[Any, set[int]] = weakref.WeakKeyDictionary()
        self.lock = _threading_module.Lock()
        self._depth = _threading_module.local()

    @property
    def depth(self) -> int:
        """This thread's owner-sync nesting depth (0 outside a session)."""

        return int(getattr(self._depth, "value", 0))

    @depth.setter
    def depth(self, value: int) -> None:
        """Clamp negative depths to 0 on the way in (thread-local)."""

        self._depth.value = max(0, value)

    def stamp(self, condition: Any) -> None:
        """Record the current thread as a notifier of ``condition`` (best effort)."""

        try:
            with self.lock:
                idents = self.stamps.get(condition)
                if idents is None:
                    idents = set()
                    self.stamps[condition] = idents
                idents.add(_threading_module.get_ident())
        except TypeError:
            # An un-weakref-able Condition subclass: the primitive stays unattributed.
            return

    def notifiers(self, condition: Any) -> set[int] | None:
        """Idents that notified ``condition`` in-window, or ``None`` when never stamped."""

        try:
            with self.lock:
                idents = self.stamps.get(condition)
                return None if idents is None else set(idents)
        except TypeError:
            return None


def install_owner_sync_surfaces(monitor: host_nondeterminism_monitor) -> None:
    """Install the counterpart-resolving class patches of the owner-sync layer (W051 2.16)."""

    session = _OwnerSyncSession(monitor)
    monitor._owner_sync_session = session
    condition_class = _threading_module.Condition
    for name in ("notify", "notify_all"):
        monitor._patch_attr(
            condition_class, name, _notify_wrapper(session, getattr(condition_class, name))
        )
    for display, module_name, qualname, method, condition_path in OWNER_SYNC_PRIMITIVES:
        if qualname is None:
            continue  # module-level functions ride the call-event layer only
        holder = _resolve_sync_holder(module_name, qualname)
        original = None if holder is None else vars(holder).get(method)
        if original is None:
            continue
        monitor._patch_attr(
            holder, method, _wait_wrapper(session, original, display, condition_path)
        )


def _notify_wrapper(session: _OwnerSyncSession, original: Any) -> Any:
    """Stamp the notifying thread on the Condition, then delegate."""

    def wrapper(self_condition: Any, *args: Any, **kwargs: Any) -> Any:
        """Record this thread as a notifier of ``self_condition`` and notify."""

        session.stamp(self_condition)
        return original(self_condition, *args, **kwargs)

    return wrapper


def _in_thread_universe(monitor: host_nondeterminism_monitor) -> bool:
    """Whether the current thread is the owner or an in-window-started (hooked) thread.

    Waits are classified for the WHOLE capture thread universe, not the owner alone: a
    hooked in-window worker that waits on a pre-existing pool (``Thread(target=lambda:
    box.append(PRE_POOL.submit(time.time).result()))``) and is then joined by the owner
    would otherwise relay a foreign value through a hooked-only join.
    """

    ident = _threading_module.get_ident()
    return ident == monitor._owner_thread or ident in monitor._in_window_thread_idents


def _wait_wrapper(
    session: _OwnerSyncSession, original: Any, display: str, condition_path: tuple[str, ...] | None
) -> Any:
    """Build the counterpart-resolving wrapper for one wait primitive."""

    monitor = session.monitor

    def wrapper(self_primitive: Any, *args: Any, **kwargs: Any) -> Any:
        """Run the wait; inside the capture's thread universe, classify its counterpart."""

        if not _in_thread_universe(monitor) or monitor._suppress_self_marks:
            return original(self_primitive, *args, **kwargs)
        session.depth += 1
        try:
            return original(self_primitive, *args, **kwargs)
        finally:
            session.depth -= 1
            try:
                classify_owner_wait(
                    session,
                    OwnerWait(
                        display,
                        self_primitive,
                        condition_path,
                        _sys_module._getframe(1),
                        args,
                        kwargs,
                    ),
                )
            except Exception:  # noqa: BLE001 - a classifier fault degrades to uncertainty, never breaks the wait
                monitor._flag_uncertain("owner_sync_classifier_error")

    return wrapper


def _wait_initiator(frame: Any) -> tuple[Any, bool]:
    """Walk to the frame that initiated a wait; also report a producer-side ``Queue.put``.

    Returns ``(initiator_frame_or_None, is_producer_wait)``. A ``Condition.wait`` reached
    THROUGH ``queue.Queue.put`` blocks for SPACE and receives nothing, so it is not a join.
    """

    queue_module = _sys_module.modules.get("queue")
    put_code = getattr(getattr(getattr(queue_module, "Queue", None), "put", None), "__code__", None)
    initiator = frame
    for _ in range(_INITIATOR_WALK_LIMIT):
        if initiator is None:
            break
        if put_code is not None and initiator.f_code is put_code:
            return initiator, True
        try:
            module_name = initiator.f_globals.get("__name__", "")
        except (AttributeError, TypeError):
            module_name = ""
        if str(module_name).partition(".")[0] not in _SYNC_STDLIB_TOPS:
            break
        initiator = initiator.f_back
    return initiator, False


def _torchlens_initiated(monitor: host_nondeterminism_monitor, initiator: Any) -> bool:
    """Whether the wait's initiator frame is TorchLens-owned (exact module-globals identity)."""

    if initiator is None:
        return False
    try:
        return id(initiator.f_globals) in monitor._tl_globals_ids
    except (AttributeError, TypeError):
        return False


def _probe_only_acquire(display: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """A non-blocking ``Semaphore.acquire`` returns only a bool and can carry no value."""

    if display != "threading.Semaphore.acquire":
        return False
    blocking = args[0] if args else kwargs.get("blocking", True)
    timeout = args[1] if len(args) > 1 else kwargs.get("timeout")
    return blocking is False or timeout == 0


def _resolve_counterparts(
    session: _OwnerSyncSession, receiver: Any, condition_path: tuple[str, ...] | None
) -> set[int] | None:
    """Idents of the threads the owner's wait received from, or ``None`` when unknown."""

    if condition_path is None:
        return None
    if condition_path == (_THREAD_COUNTERPART,):
        ident = getattr(receiver, "ident", None)
        return None if ident is None else {int(ident)}
    target = receiver
    for attribute in condition_path:
        target = getattr(target, attribute, None)
        if target is None:
            return None
    return session.notifiers(target)


@dataclass(frozen=True)
class OwnerWait:
    """One completed wait on an owner-sync primitive, as the classifier receives it.

    ``display`` is the vocabulary row's display name, ``receiver`` the primitive the
    wait ran on, ``condition_path`` the row's counterpart path, ``caller_frame`` the
    frame that called the primitive, and ``args``/``kwargs`` the wait call's own
    arguments (a non-blocking semaphore probe is recognized from them).
    """

    display: str
    receiver: Any
    condition_path: tuple[str, ...] | None
    caller_frame: Any
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None


def classify_owner_wait(session: _OwnerSyncSession, wait: OwnerWait) -> None:
    """Attribute one completed owner-thread wait to its counterpart thread(s).

    Records the primitive in ``owner_thread_waits``; additionally in
    ``owner_unhooked_joins`` when any counterpart is outside the capture's thread
    universe, or in ``owner_unattributed_waits`` when no counterpart resolves. Runs for
    the owner AND in-window hooked threads (a relay through a hooked worker must not
    launder a foreign value into a hooked-only join). A wait initiated by a TorchLens-owned frame (the async disk writer's backpressure block), a
    producer-side ``Queue.put`` block, or a non-blocking semaphore probe is not a join.
    """

    monitor = session.monitor
    display = wait.display
    initiator, producer_wait = _wait_initiator(wait.caller_frame)
    if producer_wait or _torchlens_initiated(monitor, initiator):
        return
    if _probe_only_acquire(display, wait.args, wait.kwargs or {}):
        return
    result = monitor.result
    result.owner_thread_waits.add(display)
    counterparts = _resolve_counterparts(session, wait.receiver, wait.condition_path)
    if counterparts is None:
        result.owner_unattributed_waits.add(display)
        return
    foreign = {
        ident
        for ident in counterparts
        if ident != monitor._owner_thread and ident not in monitor._in_window_thread_idents
    }
    if foreign:
        result.owner_unhooked_joins.add(display)


def classify_consumed_only_receiver(
    monitor: host_nondeterminism_monitor, receiver: Any, arg: Any, frame: Any
) -> bool:
    """Route a W051 consumed-only ``c_call`` receiver; ``True`` when claimed here.

    The C-level owner sync primitive (``queue.SimpleQueue.get``, 2.16) is an owner wait;
    the Python / legacy-NumPy GLOBAL-engine singletons (2.17) are draw-exempt receivers
    whose held-ref STATE methods are consumed-only (replayable reads).
    """

    method_name = getattr(arg, "__name__", None)
    sync_primitive = owner_sync_c_call_primitive(receiver, method_name)
    if sync_primitive is not None:
        note_owner_wait(monitor, sync_primitive, frame)
        return True
    engine_prefix = monitor._global_engine_prefixes.get(id(receiver))
    if engine_prefix is None:
        return False
    state_channel = global_engine_state_channel(engine_prefix, method_name)
    if state_channel is not None:
        monitor._mark_replayable(state_channel)
    return True


def note_owner_wait(monitor: host_nondeterminism_monitor, primitive: str, frame: Any) -> None:
    """Record one in-window sync-primitive ENTRY by the OWNER thread (call-event layer).

    ``frame`` is the frame of the primitive's own body for ``call`` events and the
    CALLER frame for ``c_call`` events; the initiator walk handles both. When the entry
    is nested inside this window's class-patched wait wrapper the wrapper owns the
    attribution and this layer stays silent; otherwise (a pre-bound alias bypassing the
    class patch, the C ``SimpleQueue.get``) the wait is recorded UNATTRIBUTED.
    """

    if not _in_thread_universe(monitor) or monitor._suppress_self_marks:
        return
    session = monitor._owner_sync_session
    if session is not None and session.depth > 0:
        return
    initiator, producer_wait = _wait_initiator(frame)
    if producer_wait or _torchlens_initiated(monitor, initiator):
        return
    monitor.result.owner_thread_waits.add(primitive)
    monitor.result.owner_unattributed_waits.add(primitive)


def settle_foreign_reads(monitor: host_nondeterminism_monitor) -> None:
    """Join the in-window waits with the foreign-thread disclosure (W051 2.16).

    Disclosed foreign reads are promoted to UNCERTAIN iff the capture's thread universe
    had at least one UNHOOKED join (received from a thread outside the universe) or
    UNATTRIBUTED wait (no resolvable counterpart) -- a flush loop never makes the capture
    wait; every feed-in topology does. Hooked-only joins (the owner waiting on its OWN
    in-window workers) promote nothing: those workers' reads ceiling directly.

    INCOMPLETE is the honest verdict -- the monitor saw a join and a touch but cannot
    attribute the value flow -- and the detail names both halves of the mechanism so the
    readiness diagnostic teaches the fix. An unhooked join WITHOUT any disclosed read is
    disclosure only (``owner_unhooked_joins``): the joined thread may have consumed a
    profile-only channel (``datetime.now``, a held clock builtin) that nothing can observe
    on an unhooked thread -- the clause-(iv) residual of the contract. Settling that case
    uncertain is the named design fork "unhooked join alone ceilings" (one call below),
    left unruled because it changes the r41 doctrine that a digest-witnessed draw on a
    pre-existing worker is CERTAIN.
    """

    result = monitor.result
    joins = result.owner_unhooked_joins | result.owner_unattributed_waits
    if not result.foreign_thread_reads or not joins:
        return
    for channel in sorted(result.foreign_thread_reads):
        monitor._flag_uncertain(f"foreign_thread_read_joined:{channel}")
    for primitive in sorted(joins):
        monitor._flag_uncertain(f"owner_thread_waited:{primitive}")


# ---- Python / NumPy global-engine state surface (W051 / AUD-CODE 2.17) -----------------
#
# The seed-replayable global engines are witnessed by a before/after state snapshot at
# the capture boundary. An owner-thread ``getstate -> draw -> setstate`` (or a
# ``seed``/``set_state`` from a pre-window value) leaves the snapshot compare EQUAL while
# the forward consumed a host scalar -- a false ``verified`` when the replay's ambient
# engine state differs. These rows carry the ``replayable_read`` disposition (the torch
# ``initial_seed`` analog): they set ``host_rng_consumed`` so the replay REQUIRES the
# capture seed (at which the fresh oracle reproduces the same draw) and never ceiling --
# the contract's own asymmetry note: a python/numpy in-forward reseed is self-reproducing
# on-seed, unlike a torch-engine mutation, which desyncs replay-re-executed DAG RNG ops.

GLOBAL_ENGINE_STATE_ROWS: tuple[tuple[str, str, str], ...] = (
    # (public module spelling, method, channel)
    ("random", "getstate", "random.getstate"),
    ("random", "setstate", "random.setstate"),
    ("random", "seed", "random.seed"),
    ("numpy.random", "get_state", "numpy.random.get_state"),
    ("numpy.random", "set_state", "numpy.random.set_state"),
    ("numpy.random", "seed", "numpy.random.seed"),
)
"""Module-attr rows of the Python / legacy-NumPy global-engine STATE surface."""

GLOBAL_ENGINE_STATE_DISPOSITION = "replayable_read"
"""Every global-engine state row is consumed-only: replay at the capture seed reproduces."""


def registry_rows() -> tuple[tuple[str, str, str, str, str], ...]:
    """``HostNondeterminismRow`` tuples for the global-engine state surface.

    One ``module_patch`` row per spelling (thread-independent -- a foreign thread's
    ``random.seed()`` desyncs the shared engine exactly like the owner's) plus one
    ``c_call_identity`` row per Python-``random`` spelling for the held-reference layer
    (``from random import getstate`` calls the singleton's bound Python method, whose
    body enters the C base ``_random.Random.getstate`` -- classified by receiver identity
    on the owner and every in-window hooked thread). The legacy NumPy singleton's bound
    Cython methods emit no ``c_call`` on numpy>=2, so their held-reference spelling is a
    disclosed residual of the same class as the numpy>=2 instance-draw shape.
    """

    rows: list[tuple[str, str, str, str, str]] = []
    for _module_name, _method, channel in GLOBAL_ENGINE_STATE_ROWS:
        rows.append(
            ("rng_global_state", channel, "module_patch", "any", GLOBAL_ENGINE_STATE_DISPOSITION)
        )
        if channel.startswith("random."):
            rows.append(
                (
                    "rng_global_state",
                    channel,
                    "c_call_identity",
                    "hooked",
                    GLOBAL_ENGINE_STATE_DISPOSITION,
                )
            )
    return tuple(rows)


def global_engine_state_prefixes() -> dict[int, str]:
    """``id(global engine singleton) -> channel prefix`` for the held-ref ``c_call`` layer."""

    prefixes: dict[int, str] = {}
    inst = getattr(_random_module, "_inst", None)
    if inst is not None:
        prefixes[id(inst)] = "random"
    np_singleton = getattr(getattr(np.random, "mtrand", None), "_rand", None)
    if np_singleton is not None:
        prefixes[id(np_singleton)] = "numpy.random"
    return prefixes


_GLOBAL_ENGINE_STATE_METHODS: frozenset[str] = frozenset(
    {"getstate", "setstate", "seed", "get_state", "set_state"}
)


def global_engine_state_channel(prefix: str, name: Any) -> str | None:
    """Channel for a state method ``c_call`` on a global-engine singleton, or ``None``."""

    if isinstance(name, str) and name in _GLOBAL_ENGINE_STATE_METHODS:
        return f"{prefix}.{name}"
    return None


def install_global_engine_state_surfaces(monitor: host_nondeterminism_monitor) -> None:
    """Patch the module-attr spellings of the Python / legacy-NumPy state surface.

    ``numpy.random.mtrand`` re-exports the same bound methods as ``numpy.random``; both
    module attributes are patched so either spelling marks. TorchLens's own in-window
    snapshot/restore/seed helpers (``log_current_rng_states``,
    ``set_rng_from_saved_states``, ``set_random_seed``) run under the active monitor's
    mark suppression and never self-mark through these wrappers.
    """

    holders: dict[str, tuple[Any, ...]] = {
        "random": (_random_module,),
        "numpy.random": tuple(
            module
            for module in (np.random, getattr(np.random, "mtrand", None))
            if module is not None
        ),
    }
    for module_name, method, channel in GLOBAL_ENGINE_STATE_ROWS:
        for holder in holders[module_name]:
            original = getattr(holder, method, None)
            if original is None or not callable(original):
                continue
            monitor._patch_attr(holder, method, _replayable_wrapper(monitor, original, channel))


def _replayable_wrapper(monitor: host_nondeterminism_monitor, original: Any, channel: str) -> Any:
    """Build a consumed-only marking passthrough for one global-engine state method."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Mark the replayable-read channel at entry, then delegate to the original."""

        monitor._mark_replayable(channel)
        return original(*args, **kwargs)

    return wrapper
