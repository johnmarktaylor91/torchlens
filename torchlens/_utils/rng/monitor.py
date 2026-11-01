"""The host-nondeterminism monitor: the capture-window context manager.

Installs, from :data:`~torchlens._utils.rng.registry.HOST_NONDETERMINISM_REGISTRY`, the
module and class patches, profile hooks, model generator digests and frame-reachable
inventories that witness host entropy, clock and RNG-instance consumption during a
capture, and settles them into a :class:`~torchlens._utils.rng.channels.HostRngMonitorResult`.
"""

from __future__ import annotations

import _random as _c_random_module
import _thread as _c_thread_module
import collections as _collections_module
import contextvars as _contextvars_module
import datetime as _datetime_module
import functools as _functools_module
import gc as _gc_module
import os as _os_module
import random
import sys as _sys_module
import threading as _threading_module
import time as _time_module
import uuid as _uuid_module
import weakref as _weakref_module
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from types import (
    BuiltinFunctionType,
    CodeType,
    FrameType,
    FunctionType,
    GetSetDescriptorType,
    MappingProxyType,
    MethodType,
    MethodWrapperType,
    ModuleType,
    SimpleNamespace,
)
from typing import Any, cast

import numpy as np
import torch

from . import channels, engine_state, inventory_types as _types

try:  # ``resource`` is POSIX-only; feature-detected for the clock family.
    import resource as _resource_module
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource_module = None  # type: ignore[assignment]

from .._torch_compat import (
    warm_lazy_torch_imports,
)
from . import monitor_state
from .call_sites import _call_site_time_arg_proof
from .channels import HostRngMonitorResult
from .inventory_types import (
    _AMBIENT_BRIDGE_LEAF_TYPES,
    _CUSTOM_HOLDER_SKIP_MODULES,
    _INERT_PRIMITIVE_LEAF_TYPES,
    _INVENTORY_LEAF_TYPES,
    _LRU_CACHE_WRAPPER_TYPE,
    _NUMERIC_PAYLOAD_LEAF_TYPES,
    _NUMPY_RNG_INSTANCE_TYPES,
    _PARTIAL_SLOT_DESCRIPTORS,
    _STDLIB_CLASS_LEAF_MODULES,
    _STDLIB_PATH_PREFIX,
    _UNCERTAIN_DETAIL_CAP,
    _WEAKREF_PROXY_TYPES,
    _NotADigestableRng,
    _numpy_rng_receiver,
    _resolve_slot_member,
    _rng_exempt_instances,
    _torchlens_module_globals_ids,
    _untrusted_rng_override,
)
from .monitor_state import _PATCH_STACKS, _PatchStackEntry, _skip_retired_hooks
from .registry import (
    _CLOCK_COUNTER_NAMES,
    _CLOCK_IMPLICIT_NOW,
    _DATETIME_CLOCK_READERS,
    _GENERATOR_METHOD_ROWS_BY_NAME,
    TORCH_RNG_SURFACE,
    _resolve_default_generator_ids,
    _torch_rng_holder_module,
)


class host_nondeterminism_monitor:
    """Context manager installing the registry-driven host-nondeterminism monitor.

    Mechanisms (r39 CLASS A + r41, all built FROM :data:`HOST_NONDETERMINISM_REGISTRY`):

    * **Model-attribute state digest (thread-independent belt).** A cycle-safe sweep of
      every numpy ``Generator`` / ``RandomState`` / bare ``BitGenerator`` / ``random``
      generator the MODEL itself holds -- submodule ``__dict__`` values INCLUDING builtin
      container nesting (list/tuple/set/frozenset elements and dict keys AND values, r41)
      -- never a process-wide ``gc`` scan. Before/after state digests catch a draw on ANY
      thread, including a pre-existing worker. Exhaustion of the defensive
      :data:`_INVENTORY_NODE_CAP` flags ``inventory_budget_exhausted`` (INCOMPLETE),
      never a silent truncation.
    * **Frame-reachable deep state digest (whole-window belt, B4).** The same
      before/after digest for numpy RNG receivers DEEPLY reachable from profiled
      user-frame roots -- named globals including module namespaces, fast locals,
      helper returns -- through exact builtin containers, plain-object
      ``__dict__``/``__slots__`` values, nested name-referenced modules, and direct
      class attributes (:meth:`_deep_inventory_frame_reachable`) -- so a
      profile-silent numpy>=2 draw through a foreign module attribute chain
      (``helpers.RNG.random()``) or a nested holder (``HOLDER.inner.gen``) is
      witnessed. Digest at first reference, ONE compare at ``__exit__`` --
      thread-independent for every receiver the window's code reaches. Cap
      exhaustion flags ``deep_inventory_budget_exhausted`` (INCOMPLETE), never a
      silent truncation.
    * **Class patches (thread-independent belt).** ``random.Random`` / ``random.SystemRandom``
      / ``_random.Random`` draw primitives (the bare-``_random.Random()`` channel; measured
      E1 patchable), plus the ``os.urandom`` / ``os.getrandom`` / ``random._urandom`` entropy
      funnel and the ``numpy.random.default_rng`` factory.
    * **Construction entropy.** The writable ``numpy.random.bit_generator.randbits`` alias
      (measured E5): an UNSEEDED BitGenerator/``default_rng()`` construction on ANY thread
      marks, closing the ephemeral-unseeded-generator channel structurally.
    * **Clock family.** Module-attr wrappers for every ``time.*`` current-clock reader (the
      implicit-now converters mark only with no explicit-time argument), ``os.times`` and
      feature-detected ``resource.getrusage``; c_call identity for the immutable
      ``datetime`` current readers (E1: unpatchable extension types).
    * **Held-reference identity (r41 hon1_1).** Every module-patched original's ``id()``
      is registered BEFORE its attribute is replaced, so a pre-window held reference
      (``from time import time`` / ``from os import urandom`` in a model or helper
      module) marks by ``c_call`` identity on the owner and every in-window hooked
      thread. The implicit-now converters decode the call site's bytecode
      (:func:`_call_site_explicit_time_value`), keeping a held
      ``localtime(1234)`` literal a pure transform; an explicit ``None``
      argument, a variable (could be ``None``), or an undecodable site
      (star-call) marks fail-closed. TorchLens's own frames are exempt by exact
      module-globals ownership
      (its per-op clock reads route patched-attr -> wrapper -> original, emitting
      ``c_call`` for the original from the wrapper's frame).
    * **Dual chained profile hooks (belt).** ``sys.setprofile`` (owner thread) AND
      ``threading.setprofile`` (threads STARTED in-window; measured E2: a pre-existing
      worker is unreachable). Each hook chains its own exact predecessor and is
      identity-restored on success and exception. The threading hook additionally
      records each hooked thread's ident into an in-window DIAGNOSTIC registry
      (its r41 escape-belt 3-class consumer was deleted in r43, replaced by the
      binary owner/non-owner check in ``_completeness_cross_thread.py``).

    Entropy / instance / construction / clock positives mark from any COVERED thread. A
    REALISTIC pre-existing-thread RNG use (a background worker drawing from a MODEL-HELD
    Generator/RandomState/BitGenerator/``torch.Generator`` -- held anywhere the inert-reachability walk can
    follow WITHOUT executing user code, incl. class descriptors, weakrefs, and callable
    interiors; r53 corr/F1 -- or reachable from an in-window profiled frame's roots; B4)
    is witnessed thread-independently by the state digests, and an unseeded construction
    on any thread by the module/class patches. The residual is only an EXTERNALLY-HELD
    generator drawn on a pre-existing (non-hooked) thread -- or, for the profile-silent
    numpy>=2 method shape, on ANY thread -- that is reachable from NO digest root (the
    model, or any in-window profiled frame's locals/named globals/returns and their deep
    inert closure) except BY EXECUTING USER CODE (a property/descriptor ``__get__`` body,
    ``__getattr__``, or a callable's return value) or through a leafed edge (a function
    attribute, a hostile container subclass's elements, a stdlib/internal-package
    namespace stash, a computed module attribute, a ``deque``'s C buffer), of the
    same class as the adversarial draw+``state`` RESTORE (E4) -- a self-cleaning sequence
    no py<=3.11 mechanism can witness -- documented in contract s11, NOT a blanket
    ceiling: the r38 draft's thread-presence INCOMPLETE over-triggered every capture
    running alongside a benign background thread (DataLoader/Jupyter/pytest), so it is
    intentionally not applied. Future all-thread coverage is ``sys.monitoring`` (PEP 669,
    3.12+, interpreter-wide).
    """

    def __init__(self, model: Any = None) -> None:
        # The model is swept for held numpy/`random` generators (a cheap container-aware
        # thread-independent digest belt) -- NOT a process-wide ``gc.get_objects()`` scan.
        self._model = model
        self.result = HostRngMonitorResult()
        # O(1) dedupe for ``_flag_uncertain``: per-frame failure paths repeat
        # one reason millions of times on a persistently-raising profiled
        # object; without this set each repeat re-copied the detail tuple.
        self._uncertain_seen: set[str] = set()
        self._restores: list[Callable[[], None]] = []
        self._owner_thread = _threading_module.get_ident()
        self._previous_sys_profile: Any = None
        self._previous_threading_profile: Any = None
        self._orig_sys_setprofile: Any = None
        self._orig_threading_setprofile: Any = None
        self._sys_hook: Any = None
        self._threading_hook: Any = None
        self._sys_profile_installed = False
        self._threading_profile_installed = False
        # Window lifecycle. ``_entered`` refuses a second arm on the same instance and
        # ``_torn_down`` makes the unwind idempotent (an ``ExitStack`` double-close used
        # to clobber a later window's profile hook); the profile hooks also read
        # ``_torn_down`` to SELF-uninstall on threads ``__exit__`` cannot reach.
        self._entered = False
        self._torn_down = False
        # Set only AFTER the profile slots have been handed back, so the owner thread's
        # own teardown frames do not trip the self-uninstall and then read as
        # "someone replaced our hook".
        self._hooks_retired = False
        self._previous_active_monitor: host_nondeterminism_monitor | None = None
        self._generator_states: list[tuple[Any, str]] = []
        # B4: whole-window digests of numpy RNG receivers DEEPLY reachable from
        # profiled-frame roots (named globals incl. module namespaces, fast locals,
        # helper returns), digested at first reference and compared at ``__exit__``.
        # This is the belt for a pre-existing generator the model does NOT hold:
        # numpy>=2 draw methods emit no profile event, and the per-frame digest only
        # reaches receivers the drawing frame names directly (plus one inert edge),
        # so a draw through a foreign module attribute chain
        # (``helpers.RNG.random()``) or a nested holder (``HOLDER.inner.gen``) was
        # otherwise unwitnessed -> false VERIFIED.
        self._deep_generator_states: list[tuple[Any, str]] = []
        self._deep_walk_seen_ids: set[int] = set()
        self._deep_inventory_visited: int = 0
        self._deep_inventory_exhausted: bool = False
        self._tl_globals_ids: frozenset[int] = frozenset()
        self._exempt_ids: frozenset[int] = frozenset()
        self._clock_ccall_keys: dict[tuple[int, str], str] = {}
        # r41 hon1_1: id(original builtin) -> (channel, time_arg_index). ``None`` index
        # marks unconditionally; an int index marks only when the observed call site's
        # positional argcount leaves the explicit-time argument absent (or undecodable).
        self._held_ref_marks: dict[int, tuple[str, int | None]] = {}
        # r65 CLUSTER Z: id(innermost python function __code__) -> (channel,
        # disposition) for the torch RNG module-patch rows. torch.manual_seed & co are
        # PYTHON functions (they emit ``call`` events, not ``c_call``), so the held-ref
        # layer classifies them by code identity; the innermost ``__wrapped__`` unwind
        # keeps a TorchLens capture wrapper's shared code object from ever being
        # registered (which would misattribute every wrapped torch call).
        self._held_code_marks: dict[int, tuple[str, str]] = {}
        self._global_engine_prefixes: dict[int, str] = {}  # W051 2.17: singleton id -> prefix
        self._owner_sync_session: Any = None  # W051 2.16: per-window owner-join state
        # r67 C1: identity ROUTING CACHE of the process/device default generators --
        # it selects WHICH GENERATOR_METHOD_TABLE column applies, never WHETHER a
        # Generator receiver is classified (every ``isinstance(receiver,
        # torch.Generator)`` c_call dispatches through the table). Seeded at install
        # and re-resolved dynamically on a receiver miss, so a device default
        # populated mid-forward still selects the default column.
        self._default_generator_ids: frozenset[int] = frozenset()
        # r41 hon2_1: idents of threads hooked by the in-window threading profile
        # hook (registered race-free at thread bootstrap). Verdict-relevant again
        # since the foreign-thread pure-read split: ``channels.mark_pure_read``
        # consults it to keep in-window-started threads CEILING while ambient
        # pre-existing threads only disclose. Absence is fail-closed: a failed
        # profile install already flagged uncertainty, so an unregistered
        # in-window thread can never rescue a verdict.
        self._in_window_thread_idents: set[int] = set()
        # NumPy 2.x binds Generator/RandomState Cython callables as Python methods
        # that emit no profile ``c_call`` event. The feature-detected fallback
        # snapshots only RNG receivers directly referenced by each profiled Python
        # frame or one inert holder edge below them and compares them at return,
        # preserving the no-method-name invariant.
        self._numpy_frame_rng_states: dict[int, list[tuple[Any, str]]] = {}
        # Value RETAINS the code object and globals mapping strongly (like the
        # sibling ``_numpy_frame_digest_scope_cache`` below) so their ``id()``
        # cannot be reused by a different frame within the monitoring window. A
        # bare ``tuple[str, ...]`` value (the former shape) retained neither, so
        # an id collision after GC returned STALE ``co_names`` and snapshotted
        # the wrong RNG receivers -- an under-witness.
        self._numpy_global_name_cache: dict[
            tuple[int, int], tuple[CodeType, dict[str, Any], tuple[str, ...]]
        ] = {}
        # Code objects compare structurally and ignore ``co_filename``. Key by identity
        # and retain the code object strongly in the value so an id cannot be reused
        # during the monitoring window and an internal structural twin cannot suppress
        # the digest for a user frame.
        self._numpy_frame_digest_scope_cache: dict[int, tuple[CodeType, bool]] = {}
        # Per-window cache: RNG holder type -> the first untrusted draw/state
        # override found on it, or None when the type's witnessed surface is
        # entirely library-defined (see ``_digest_rng_witnessable``). Window-
        # scoped (not module-level) so it never joins the process-global state
        # census.
        self._rng_override_cache: dict[type, str | None] = {}
        # r49 hon1_1: re-entrancy depth for monitor-INTERNAL probes. While > 0 the monitor is
        # reading through its OWN inventory probe, so any channel a probe transitively touches
        # must NOT be marked as a model host read. Guarded at the single ``_mark`` choke point
        # -> surface-complete over every clock/entropy channel.
        #
        # THREAD-LOCAL, not a shared int. The bracket is NOT owner-thread-only in practice:
        # ``log_current_rng_states`` raises it once per logged op from the wrapper hot path
        # while the forward runs, so a concurrent thread's ``+=``/``-=`` could interleave and
        # store a NEGATIVE resting value -- truthy forever, silently dropping EVERY later
        # ``_mark``/``_mark_replayable`` on every thread while ``uncertain`` stayed False. That
        # is a direct false-VERIFIED path. Per-thread depth makes a lost update impossible and
        # is also semantically right: a probe on thread A never exempts thread B's host reads.
        self._suppress_state = _threading_module.local()

    # -- helpers -----------------------------------------------------------------

    @property
    def _suppress_self_marks(self) -> int:
        """This THREAD's monitor-internal probe depth (0 outside any probe)."""

        return int(getattr(self._suppress_state, "depth", 0))

    def _mark(self, channel: str) -> None:
        """Record a host nondeterminism channel touch, unless a monitor probe is active.

        The single choke point for CEILING-class marks (see :meth:`_mark_replayable`
        for the non-ceiling set). Marks made while ``_suppress_self_marks`` is raised
        are the monitor reading through its OWN inventory probe, never a model host
        read, and are dropped.
        """

        if self._suppress_self_marks:
            return
        self.result.channels.add(channel)

    def _mark_replayable(self, channel: str) -> None:
        """Record a torch RNG read reproduced by the capture seed (r65 CLUSTER Z).

        Same ``_suppress_self_marks`` choke point as :meth:`_mark`, but lands in the
        NON-ceiling ``replayable_reads`` set: the read sets ``host_rng_consumed``
        without discarding the capture seed.
        """

        if self._suppress_self_marks:
            return
        self.result.replayable_reads.add(channel)

    def _mark_disposition(self, channel: str, disposition: str) -> None:
        """Route a torch RNG surface touch to its disposition's result set."""

        if disposition == "replayable_read":
            self._mark_replayable(channel)
        else:
            self._mark(channel)

    @contextmanager
    def _monitor_internal_probe(self) -> Iterator[None]:
        """Suppress the monitor's OWN transitive channel marks during an internal probe (r49 hon1_1).

        A monitor-initiated read is NOT a model host read. The opaque-queue emptiness proof calls
        ``multiprocessing.Queue.empty()``, which reads ``time.monotonic`` through
        ``multiprocessing.connection`` -- without this guard that TorchLens-initiated probe would
        self-mark the clock channel and over-trigger a deterministic model merely holding an empty
        ``mp.Queue`` to UNVERIFIABLE (the r48 hon1_1 regression). Guarding at the single ``_mark``
        choke point is surface-complete: ANY channel a future inventory probe transitively touches
        is auto-exempt. The bracket is owner-thread and ``__enter__``-scoped (BEFORE the user
        forward runs), so no user/model/worker host read is ever inside it.
        """

        self._suppress_state.depth = self._suppress_self_marks + 1
        try:
            yield
        finally:
            # Never let the resting depth go negative (a negative depth is truthy and
            # would suppress every later mark on this thread with no uncertainty stamp).
            self._suppress_state.depth = max(0, self._suppress_self_marks - 1)

    def _flag_uncertain(self, reason: str) -> None:
        """Downgrade monitor completeness, optionally recording one reason.

        Uncertainty is never read as absence of consumption: install, chain,
        restore, and inventory failures all land here so the verdict degrades
        instead of silently blessing the capture.

        Detail accumulation is DEDUPED and CAPPED: several callers fire PER
        PROFILE EVENT (``profile_rng_state_read_failed``,
        ``profile_classifier_error``, ...), so a persistently-raising profiled
        object used to grow ``uncertain_detail`` by a full tuple copy per frame
        -- measured O(N^2), turning a real forward (~1e5-1e6 profiled frames)
        into minutes-to-hours of tuple-copy churn while the verdict was already
        settled INCOMPLETE by the boolean. A repeated reason is dropped in
        O(1); past the distinct-reason cap one overflow marker records that
        further DISTINCT reasons were suppressed. The ``uncertain`` boolean --
        the only verdict-steering output -- is stamped unconditionally first.
        """

        self.result.uncertain = True
        if not reason or reason in self._uncertain_seen:
            return
        if len(self._uncertain_seen) >= _UNCERTAIN_DETAIL_CAP:
            overflow = "uncertain_detail_capped"
            if overflow not in self._uncertain_seen:
                self._uncertain_seen.add(overflow)
                self.result.uncertain_detail = (*self.result.uncertain_detail, overflow)
            return
        self._uncertain_seen.add(reason)
        self.result.uncertain_detail = (*self.result.uncertain_detail, reason)

    def _patch_attr(self, holder: Any, name: str, wrapper: Any) -> None:
        """Patch one module or class attribute and queue its exact restoration.

        The queued restore flags uncertainty when the attribute no longer holds this
        wrapper at teardown -- someone replaced the patch mid-window, so exact
        restoration cannot be proven -- and restores the original regardless.

        Restoration is SPLICE-aware through :data:`_PATCH_STACKS`: unwinding out of
        order hands our recorded original to the entry stacked ABOVE us instead of
        writing it under a live patch, so the genuine pre-monitor value always reaches
        the attribute and no wrapper survives the last unwind.
        """

        original = getattr(holder, name)
        entry = _PatchStackEntry(holder, name, original, wrapper)
        _PATCH_STACKS.setdefault((id(holder), name), []).append(entry)
        setattr(holder, name, wrapper)

        def _restore(entry: _PatchStackEntry = entry) -> None:
            """Restore this entry's original, or splice it out from under a live patch."""

            key = (id(entry.holder), entry.name)
            stack = _PATCH_STACKS.get(key) or []
            try:
                index = stack.index(entry)
            except ValueError:
                index = -1
            if index >= 0 and index != len(stack) - 1:
                # A LATER monitor patched over us and is still live. Writing our
                # original now would strip its wrapper, and its own restore would then
                # write OUR wrapper back -- the historical permanent leak. Hand our
                # original to the successor and leave the attribute alone.
                stack[index + 1].original = entry.original
                stack.pop(index)
                self._flag_uncertain(
                    f"patch_spliced:{getattr(entry.holder, '__name__', entry.holder)}.{entry.name}"
                )
                return
            if index >= 0:
                stack.pop(index)
            if not stack:
                _PATCH_STACKS.pop(key, None)
            try:
                if getattr(entry.holder, entry.name, None) is not entry.wrapper:
                    # Someone outside the monitor replaced our patch mid-window:
                    # restoration cannot be proven exact -> uncertainty (fail closed),
                    # restore anyway.
                    self._flag_uncertain(
                        f"patch_replaced:"
                        f"{getattr(entry.holder, '__name__', entry.holder)}.{entry.name}"
                    )
            finally:
                # The tamper check reads through the holder's own attribute machinery
                # (a PEP-562 module ``__getattr__`` can raise). Restoring in a
                # ``finally`` keeps a raising probe from stranding the wrapper on a
                # stdlib module for the life of the process.
                setattr(entry.holder, entry.name, entry.original)

        self._restores.append(_restore)

    def _register_held_ref(
        self, original: Any, channel: str, time_arg_index: int | None = None
    ) -> None:
        """Register a module-patched ORIGINAL builtin for held-reference c_call marking.

        Called at each module-attr patch site BEFORE :meth:`_patch_attr` replaces the
        attribute, so a pre-window ``from module import name`` alias -- which calls the
        original object directly, bypassing the patch -- is classified by identity in
        :meth:`_classify_c_call` (r41 hon1_1). Class-patch originals are deliberately
        NOT registered: a held bound draw method hits the existing receiver-isinstance
        branch already.
        """

        # ``setdefault``: some registry channels alias ONE builtin object
        # (``random._urandom`` IS ``os.urandom``); the first-registered (canonical)
        # channel name wins for the shared identity.
        self._held_ref_marks.setdefault(id(original), (channel, time_arg_index))

    def _entropy_wrapper(self, original: Any, channel: str) -> Any:
        """Build a marking passthrough wrapper for one OS-entropy channel (thread-routed)."""

        return channels.entropy_wrapper(self, original, channel)

    def _raw_thread_spawn_wrapper(self, original: Any) -> Any:
        """Build a passthrough spawn wrapper that profile-hooks the NEW thread.

        ``_thread.start_new_thread`` / ``start_joinable_thread`` bootstrap the
        target directly (no ``threading.Thread`` bootstrap, so
        ``threading.setprofile`` never fires for them). The wrapped target
        installs this window's threading hook on the new thread before running,
        making an in-window raw-thread host draw witnessed exactly like a
        ``threading.Thread`` one. Everything else passes through untouched.
        """

        monitor = self

        def wrapper(function: Any, *rest: Any, **spawn_kwargs: Any) -> Any:
            """Spawn with the target wrapped to self-install the profile hook."""

            def hooked_target(*fargs: Any, **fkwargs: Any) -> Any:
                """Install the in-window threading hook, then run the target."""

                hook = monitor._threading_hook
                if hook is not None and not monitor._torn_down:
                    # Held original: the module attr may hold a (this or a
                    # later) window's swap-detection wrapper.
                    setter = monitor._orig_sys_setprofile or _sys_module.setprofile
                    setter(hook)
                return function(*fargs, **fkwargs)

            return original(hooked_target, *rest, **spawn_kwargs)

        return wrapper

    def _clock_wrapper(self, original: Any, channel: str, time_arg_index: int | None) -> Any:
        """Build a marking passthrough wrapper for one clock channel (thread-routed).

        ``time_arg_index`` names the positional argument that makes the call a pure
        transform of a caller-supplied time (``localtime(ts)``); when it is
        supplied and non-``None`` the call reads no clock and is not marked.
        """

        return channels.clock_wrapper(self, original, channel, time_arg_index)

    def _instance_method_wrapper(self, original: Any, channel: str) -> Any:
        """Build a thread-routed marking passthrough for one RNG-instance draw method.

        W051 (2.18): a PRIVATE instance draw is thread-local, so it routes like the
        clock/entropy reads (foreign threads disclose, then settle against the owner's
        in-window waits -- ``_rng_channels``); model-held instances stay digest-belted.
        """

        exempt_ids = self._exempt_ids

        def wrapper(self_rng: Any, *args: Any, **kwargs: Any) -> Any:
            """Mark the channel unless the receiver is an exempt (TorchLens-owned) instance."""

            if id(self_rng) not in exempt_ids:
                channels.mark_thread_routed(self, channel)
            return original(self_rng, *args, **kwargs)

        return wrapper

    def _torch_rng_wrapper(self, original: Any, channel: str, disposition: str) -> Any:
        """Module-attr wrapper for a torch RNG API row (r65 CLUSTER Z).

        Marks AT ENTRY (before delegation) so a device-module endpoint that raises on a
        deviceless host still records the touch -- fail-closed, never under-marking.
        TorchLens's own in-window uses run under :func:`_suppress_active_monitor_marks`
        (suppressed at the ``_mark`` choke point), so no caller-frame exemption exists
        here: a user callback invoked FROM a TorchLens frame still marks.
        """

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Mark the torch RNG channel at entry, then delegate to the original."""

            self._mark_disposition(channel, disposition)
            return original(*args, **kwargs)

        return wrapper

    def _register_held_code(self, original: Any, channel: str, disposition: str) -> None:
        """Register a torch RNG PYTHON function for held-reference ``call`` marking.

        The r41 held-ref layer registers builtin identity for ``c_call`` events; torch's
        RNG APIs are Python functions, which emit ``call`` events keyed by CODE identity
        instead. The ``__wrapped__`` chain is unwound to the INNERMOST function and only
        a torch-owned code object is ever registered: registering a TorchLens capture
        wrapper's shared code object would misattribute every wrapped torch call, and a
        user holding either spelling (the raw torch function OR the TorchLens-wrapped
        module attribute, whose delegation still enters the raw function) is classified
        through the same innermost code. ``setdefault`` keeps the first-registered
        (canonical) channel name for aliased spellings (``torch.manual_seed`` IS
        ``torch.random.manual_seed``).
        """

        innermost = original
        for _ in range(8):
            wrapped = getattr(innermost, "__wrapped__", None)
            if wrapped is None:
                break
            innermost = wrapped
        code = getattr(innermost, "__code__", None)
        module_name = getattr(innermost, "__module__", None)
        if code is None or not isinstance(module_name, str):
            return
        if module_name != "torch" and not module_name.startswith("torch."):
            return
        self._held_code_marks.setdefault(id(code), (channel, disposition))

    def _classify_call(self, frame: Any) -> None:
        """Classify a ``call`` profile event against the held torch RNG code registry.

        A pre-window ``from torch import manual_seed`` alias calls the original Python
        function directly, bypassing the module-attr patch; the innermost code object
        was registered at monitor install. No caller-frame exemption: TorchLens-owned
        in-window uses are suppressed at the ``_mark`` choke point instead, and the
        patched-attr delegation double-mark is an idempotent set-add of the same
        channel (fail-closed over-marking, never under-marking).
        """

        marks = self._held_code_marks
        if not marks:
            return
        mark = marks.get(id(frame.f_code))
        if mark is None:
            return
        channel, disposition = mark
        if disposition == channels.OWNER_SYNC_DISPOSITION:
            channels.note_owner_wait(self, channel, frame)
            return
        self._mark_disposition(channel, disposition)

    @staticmethod
    def _numpy_frame_candidate_children(value: Any) -> tuple[Any, ...]:
        """Return one inert holder edge below a profiled-frame value.

        Exact built-in container types are read through their concrete C
        implementations, so a hostile subclass cannot execute an overridden iterator.
        Plain-object attributes are read only through a concrete getset descriptor for
        the instance ``__dict__``; a property or other hostile descriptor named
        ``__dict__`` makes the value a leaf. Callable receivers remain gated by
        :func:`_numpy_rng_receiver`, which reads ``__self__`` only after an exact
        built-in-bound-callable or ``MethodType`` check.

        Parameters
        ----------
        value:
            Candidate frame local or referenced global.

        Returns
        -------
        tuple[Any, ...]
            Immediate inert children that may themselves be concrete NumPy RNG
            receivers.
        """

        value_type = type(value)
        if value_type is dict:
            return (*dict.keys(value), *dict.values(value))
        if value_type in (list, tuple, set, frozenset):
            return tuple(value)
        # r39: match weakref PROXIES on EXACT type BEFORE any isinstance branch below
        # -- ``isinstance`` consults the proxy's forwarded ``__class__``, so a proxy
        # whose referent is a deque/local/partial would otherwise be dispatched into a
        # base-C read that raises (or, worse, forwards). NO inert deref exists (see
        # :data:`_WEAKREF_PROXY_TYPES`); leaf here -- the deep frame walk fail-closes
        # this shape to INCOMPLETE.
        if type(value) in _WEAKREF_PROXY_TYPES:
            return ()
        # r38 stdlib-instance holder edges, mirroring the deep-inventory parity
        # branches: every read below is base-C (deref / member descriptor /
        # ``tp_traverse`` / base ``__iter__``), so no user code can fire.
        if isinstance(value, _weakref_module.ref):
            try:
                referent = _weakref_module.ref.__call__(value)
            except Exception:
                return ()
            return () if referent is None else (referent,)
        if isinstance(value, _c_thread_module._local):
            children: list[Any] = []
            for referent in _gc_module.get_referents(value):
                if type(referent) is not dict:
                    continue
                for per_thread in dict.values(referent):
                    if type(per_thread) is dict:
                        children.extend(dict.values(per_thread))
            return tuple(children)
        if isinstance(value, _functools_module.partial):
            interior: list[Any] = []
            for descriptor in _PARTIAL_SLOT_DESCRIPTORS:
                try:
                    interior.append(descriptor.__get__(value, value_type))
                except Exception:
                    continue
            return tuple(interior)
        if isinstance(value, _collections_module.deque):
            return tuple(_collections_module.deque.__iter__(value))
        # r39 C-holder parity branches (round-39 executed false-VERIFIEDs V9a-V9f):
        # a generator reached ONLY through a C-implemented holder with a C accessor
        # path never enters a Python frame. Every read below is base-C
        # (``tp_traverse`` / base getset / C ``Context`` lookup); no user code fires.
        if _LRU_CACHE_WRAPPER_TYPE is not None and isinstance(value, _LRU_CACHE_WRAPPER_TYPE):
            # A WARM wrapper's ``tp_traverse`` exposes its cache dict; flatten one
            # dict level so this single-edge belt sees the cached values directly.
            cached_values: list[Any] = []
            for referent in _gc_module.get_referents(value):
                if type(referent) is dict:
                    cached_values.extend(dict.values(referent))
            return tuple(cached_values)
        if type(value) is MappingProxyType:
            # ``tp_traverse`` yields the BACKING mapping without invoking its (possibly
            # user-defined) ``keys``/``values``; flatten an exact-dict backing one level.
            proxied: list[Any] = []
            for referent in _gc_module.get_referents(value):
                if type(referent) is dict:
                    proxied.extend(dict.keys(referent))
                    proxied.extend(dict.values(referent))
                else:
                    proxied.append(referent)
            return tuple(proxied)
        if type(value) is _contextvars_module.ContextVar:
            # The VALUE lives in the per-thread ``Context``, off the reference graph;
            # the base C ``get`` (set value or declared default, owner thread) is the
            # only inert edge. Pre-existing OTHER threads' contexts stay the documented
            # foreign-thread residual.
            try:
                return (_contextvars_module.ContextVar.get(value),)
            except LookupError:
                return ()
        if isinstance(value, type):
            # Class-attribute surface across the OWN MRO and the METACLASS MRO --
            # ``CLS.gen`` on a base class or user metaclass resolves entirely in C.
            return host_nondeterminism_monitor._class_attr_surface(value)
        if isinstance(value, np.ndarray):
            # An OBJECT-dtype ndarray holds ordinary Python references (numpy's own
            # parallel-streams idiom: arrays of spawned Generators) and ``ARR[0]`` is
            # a profile-silent C subscript. Element-iterate ONLY object dtypes through
            # the base getsets; non-object dtypes stay hard leaves. This belt has no
            # budget of its own, so an over-cap object array is left to the budgeted
            # (fail-closed) deep frame walk.
            try:
                dtype = np.ndarray.dtype.__get__(value)
                if dtype.hasobject and int(np.ndarray.size.__get__(value)) <= (
                    _types._DEEP_INVENTORY_NODE_CAP
                ):
                    return tuple(np.ndarray.flat.__get__(value))
            except Exception:
                return ()
            return ()
        if isinstance(
            value,
            (
                ModuleType,
                FunctionType,
                BuiltinFunctionType,
                MethodType,
                MethodWrapperType,
                np.generic,
                torch.Tensor,
                torch.nn.Module,
            ),
        ):
            return ()
        try:
            module = type.__dict__["__module__"].__get__(value_type)
        except Exception:
            return ()
        if not isinstance(module, str):
            return ()
        module_root = module.split(".", 1)[0]
        if module_root in _CUSTOM_HOLDER_SKIP_MODULES or module_root in _STDLIB_CLASS_LEAF_MODULES:
            return ()
        instance_dict_getter = None
        try:
            mro = type.__dict__["__mro__"].__get__(value_type)
        except Exception:
            return ()
        for klass in mro:
            try:
                class_dict = type.__dict__["__dict__"].__get__(klass)
            except Exception:
                return ()
            descriptor = class_dict.get("__dict__")
            if descriptor is None:
                continue
            if not isinstance(descriptor, GetSetDescriptorType):
                return ()
            instance_dict_getter = descriptor
            break
        if instance_dict_getter is None:
            return ()
        try:
            instance_dict = instance_dict_getter.__get__(value, value_type)
        except (AttributeError, TypeError, ValueError):
            return ()
        if not isinstance(instance_dict, dict):
            return ()
        return tuple(instance_dict.values())

    def _numpy_frame_needs_rng_snapshot(self, code: CodeType) -> bool:
        """Return whether a code object's frames need the NumPy RNG digest.

        Torch, TorchLens, and NumPy package frames are implementation frames: a
        user-owned NumPy RNG draw originates in the calling user frame, which remains
        fully snapshotted, or in a user callback, which receives its own profile
        ``call`` event. Caching by code object makes the per-call decision an identity
        lookup after the first invocation without trusting mutable module metadata.

        Parameters
        ----------
        code:
            Code object for the entering Python frame.

        Returns
        -------
        bool
            Whether the frame may originate a user-owned NumPy RNG draw.
        """

        cache_key = id(code)
        cached = self._numpy_frame_digest_scope_cache.get(cache_key)
        if cached is not None and cached[0] is code:
            return cached[1]
        filename = code.co_filename
        needs_snapshot = not any(
            filename.startswith(prefix)
            for prefix in _types._NUMPY_FRAME_DIGEST_INTERNAL_PATH_PREFIXES
        )
        self._numpy_frame_digest_scope_cache[cache_key] = (code, needs_snapshot)
        return needs_snapshot

    def _snapshot_numpy_frame_rngs(self, frame: FrameType) -> None:
        """Snapshot NumPy RNG receivers inertly reachable from a Python frame.

        This fallback is active only for the feature-detected NumPy Cython method
        shape that emits no profile ``c_call`` event. It covers every materialized
        fast local directly plus globals named by the frame's code object (its
        ``co_names`` plus its string ``co_consts`` -- r38: the latter resolve
        ``globals()["name"]``-style dynamic-name subscripts), descending
        one inert edge below those globals through exact built-in containers or a
        plain-object ``__dict__``. It never walks a shared module namespace and never
        expands locals edges (both are the whole-window
        :meth:`_deep_inventory_frame_reachable` belt's job -- B4) or invokes user
        attribute/iteration hooks. Referenced global names, rather than resolved
        objects, are cached per code/globals identity so a rebound global cannot
        evade a later snapshot.

        Parameters
        ----------
        frame:
            Python frame entering under the profile hook.
        """

        if not _types._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST:
            return
        if not self._numpy_frame_needs_rng_snapshot(frame.f_code):
            return
        cache_key = (id(frame.f_code), id(frame.f_globals))
        cached = self._numpy_global_name_cache.get(cache_key)
        if cached is None:
            # r38: string CONSTANTS join the referenced-name set. A dynamic-name
            # namespace read -- ``globals()["name"]`` (and the ``vars()`` /
            # ``eval("name")`` spellings) -- carries the name in ``co_consts``,
            # not ``co_names``, so a generator reached ONLY through such a
            # subscript steered a branch and replayed false-VERIFIED (executed
            # repro, round 37). A string constant naming no global is one cheap
            # dict miss per frame entry; a COMPUTED name stays the documented
            # dynamic-name residual.
            global_names = tuple(
                dict.fromkeys(
                    (
                        *frame.f_code.co_names,
                        *(const for const in frame.f_code.co_consts if isinstance(const, str)),
                    )
                )
            )
            # Retain the code object AND globals mapping in the value so their
            # id()s cannot be reused mid-window (see the field comment); an id
            # collision would otherwise return stale co_names -> under-witness.
            self._numpy_global_name_cache[cache_key] = (
                frame.f_code,
                frame.f_globals,
                global_names,
            )
        else:
            global_names = cached[2]
        global_candidates = [
            frame.f_globals[name] for name in global_names if name in frame.f_globals
        ]
        snapshots: list[tuple[Any, str]] = []
        seen_ids: set[int] = set()
        for candidate in global_candidates:
            for reachable in (candidate, *self._numpy_frame_candidate_children(candidate)):
                receiver = _numpy_rng_receiver(reachable)
                holder = receiver if receiver is not None else reachable
                if (
                    id(holder) in seen_ids
                    or id(holder) in self._exempt_ids
                    or not isinstance(holder, _NUMPY_RNG_INSTANCE_TYPES)
                ):
                    continue
                seen_ids.add(id(holder))
                try:
                    snapshots.append((holder, self._digest_rng_witnessable(holder)))
                except Exception:
                    self._flag_uncertain("profile_rng_state_read_failed")
        local_candidates = list(frame.f_locals.values())
        for candidate in local_candidates:
            # Locals stay DIRECT-only here (any per-local edge expansion costs ~25%
            # of a small capture across the thousands of stdlib helper frames the
            # capture machinery enters); a receiver NESTED below a local is covered
            # by the B4 window-level deep inventory seeded right below.
            receiver = _numpy_rng_receiver(candidate)
            holder = receiver if receiver is not None else candidate
            if (
                id(holder) in seen_ids
                or id(holder) in self._exempt_ids
                or not isinstance(holder, _NUMPY_RNG_INSTANCE_TYPES)
            ):
                continue
            seen_ids.add(id(holder))
            try:
                snapshots.append((holder, self._digest_rng_witnessable(holder)))
            except Exception:
                self._flag_uncertain("profile_rng_state_read_failed")
        if snapshots:
            self._numpy_frame_rng_states[id(frame)] = snapshots
        # B4: window-level deep inventory of everything this frame can reach (module
        # namespaces, nested holders, class attributes) -- first reference walks and
        # digests, repeats cost only set lookups, comparison happens once at __exit__.
        if self._deep_inventory_seeds_from(frame.f_code):
            self._deep_inventory_frame_reachable(
                (*global_candidates, *local_candidates), frame.f_code
            )

    def _compare_numpy_frame_rngs(self, frame: FrameType) -> None:
        """Mark a NumPy RNG receiver whose state changed within a profiled frame.

        Parameters
        ----------
        frame:
            Python frame returning under the profile hook.
        """

        snapshots = self._numpy_frame_rng_states.pop(id(frame), ())
        for holder, before in snapshots:
            try:
                changed = self._digest_rng_witnessable(holder) != before
            except Exception:
                self._flag_uncertain("profile_rng_state_read_failed")
                continue
            if changed:
                self._mark("c_rng_instance_draw")
                return

    def _snapshot_returned_numpy_rngs(self, frame: FrameType, returned: Any) -> None:
        """Snapshot NumPy RNG receivers returned into a still-running user frame.

        NumPy 2.x Cython RNG methods emit no profile event. A receiver obtained from a
        Python helper after its caller entered therefore misses the caller's entry
        snapshot unless it is transferred at the helper's ``return`` event. Tracking
        the returned receiver and one inert holder edge is method-name-independent and
        lets the caller's ordinary return comparison witness any later state change.

        Parameters
        ----------
        frame:
            Python frame returning ``returned``.
        returned:
            Value delivered to the caller.
        """

        if not _types._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST:
            return
        caller = frame.f_back
        if caller is None or not self._numpy_frame_needs_rng_snapshot(caller.f_code):
            return
        snapshots = self._numpy_frame_rng_states.setdefault(id(caller), [])
        seen_ids = {id(holder) for holder, _ in snapshots}
        for candidate in (returned, *self._numpy_frame_candidate_children(returned)):
            receiver = _numpy_rng_receiver(candidate)
            holder = receiver if receiver is not None else candidate
            if (
                id(holder) in seen_ids
                or id(holder) in self._exempt_ids
                or not isinstance(holder, _NUMPY_RNG_INSTANCE_TYPES)
            ):
                continue
            seen_ids.add(id(holder))
            try:
                snapshots.append((holder, self._digest_rng_witnessable(holder)))
            except Exception:
                self._flag_uncertain("profile_rng_state_read_failed")
        if not snapshots:
            self._numpy_frame_rng_states.pop(id(caller), None)
        # B4: a returned holder's nested receivers join the window-level deep
        # inventory (the one-edge transfer above only reaches direct children).
        if self._deep_inventory_seeds_from(caller.f_code):
            self._deep_inventory_frame_reachable((returned,), caller.f_code)

    def _classify_c_call(self, frame: Any, arg: Any) -> None:
        """Classify one ``c_call`` profile event against the held-builtin registry.

        Held-reference identity is checked FIRST, so a pre-window
        ``from time import time`` alias -- which bypasses the module-attr patch
        by calling the original builtin -- is still marked. TorchLens's own
        frames are exempt by exact module-globals ownership. For an
        implicit-now converter the call-site argument count decides whether the
        call reads a clock at all; an undecodable call site marks fail-closed.
        """

        # r41 hon1_1: held-reference identity FIRST. A pre-window ``from time import
        # time`` alias calls the ORIGINAL builtin, bypassing the module-attr patch; the
        # original was identity-registered before patching. TorchLens's own frames are
        # exempt by EXACT module-globals ownership -- its per-op clock reads route
        # patched-attr -> ``_clock_wrapper`` -> original, emitting ``c_call`` for the
        # original FROM the wrapper's frame; without the exemption every capture would
        # self-ceiling.
        mark = self._held_ref_marks.get(id(arg))
        if mark is not None:
            try:
                caller_globals_id = id(frame.f_globals)
            except Exception:
                caller_globals_id = -1
            if caller_globals_id not in self._tl_globals_ids:
                held_channel, time_arg_index = mark
                if time_arg_index is None:
                    self._mark(held_channel)
                else:
                    # Implicit-now converter: a call site providing the explicit-time
                    # argument is a pure transform ONLY when that argument is provably
                    # non-None -- ``localtime(None)`` reads the clock exactly like
                    # ``localtime()`` (grind-r5 b8 R57). Undecodable (star-call /
                    # unknown opcode) marks fail-closed; a literal ``None`` marks; a
                    # computed argument flags uncertainty (runtime-dependent value:
                    # neither a clock-draw claim nor a clean pass is provable).
                    argcount = channels.call_site_argcount(frame)
                    if argcount is None or argcount <= time_arg_index:
                        self._mark(held_channel)
                    else:
                        proof = _call_site_time_arg_proof(frame, argcount, time_arg_index)
                        if proof == "now_read":
                            self._mark(held_channel)
                        elif proof == "unknown":
                            self._flag_uncertain(f"held_ref_time_arg_unproven:{held_channel}")
        receiver = getattr(arg, "__self__", None)
        if receiver is None:
            return
        # Profile hooks observe every C call made by capture internals. More than 90%
        # of those calls are methods on these five exact built-in receiver types. The
        # held-reference identity check above must still run first because a monitored
        # module function also has a module receiver. After an identity miss, however,
        # these exact types cannot be a torch/NumPy/Python RNG or datetime class, so
        # bypassing the remaining receiver classifiers is behavior-preserving.
        if type(receiver) in _types._INERT_PROFILE_C_CALL_RECEIVER_TYPES:
            return
        if channels.classify_consumed_only_receiver(self, receiver, arg, frame):
            return
        # r67 C1: method c_calls on ANY ``torch.Generator`` receiver -- process/device
        # defaults, user-constructed, model-held, RETURNED clones, and subclasses (an
        # inherited C method's c_call receiver IS the subclass instance) -- dispatch
        # through the one authoritative GENERATOR_METHOD_TABLE. The default identity
        # set only routes WHICH column applies; on a miss it is re-resolved against
        # the currently populated ``torch.{cuda,xpu,mtia}.default_generators`` (no
        # imports, no device init) so a default populated mid-forward still selects
        # the default column. Scalar-returning methods carry a marking disposition in
        # BOTH columns (return closure -- r66 free-F1/corr1-1/hon1-F5 can no longer
        # escape); Generator-returning methods are structural because the returned
        # object lands back HERE. An UNKNOWN public method on any Generator receiver
        # is unclassifiable host-RNG semantics -> monitor uncertainty (INCOMPLETE),
        # never a silent miss; the return-closure meta-test additionally goes RED at
        # torch-upgrade time for any new method name.
        if isinstance(receiver, torch.Generator):
            method_name = getattr(arg, "__name__", None)
            if isinstance(method_name, str) and not method_name.startswith("__"):
                row = _GENERATOR_METHOD_ROWS_BY_NAME.get(method_name)
                if row is None:
                    self._flag_uncertain(f"generator_method:{method_name}")
                    return
                is_default = id(receiver) in self._default_generator_ids
                if not is_default:
                    refreshed = _resolve_default_generator_ids()
                    self._default_generator_ids = refreshed
                    is_default = id(receiver) in refreshed
                disposition = row.default_disposition if is_default else row.nondefault_disposition
                if disposition is not None:
                    spelling = "torch.default_generator" if is_default else "torch.Generator"
                    self._mark_disposition(f"{spelling}.{method_name}", disposition)
            return
        # numpy instance draws + bare ``_random.Random`` draws (receiver typing -- no
        # draw-method NAME enumeration, so new draw methods need no detector edit).
        if id(receiver) not in self._exempt_ids and isinstance(
            receiver, (_c_random_module.Random, *_NUMPY_RNG_INSTANCE_TYPES)
        ):
            self._mark("c_rng_instance_draw")
            return
        # Immutable ``datetime`` current-clock readers (r42 hon1_1: subclass-safe). The base
        # readers are class methods on unpatchable C extension types, so a ``datetime.datetime``
        # / ``datetime.date`` SUBCLASS inherits them and reads the SAME wall clock -- but the
        # exact ``id(receiver)`` key misses the subclass. Classify subclass-safe, mirroring the
        # numpy instance-draw ``isinstance`` branch, while guarding a genuinely re-implemented
        # subclass method (an override that does not call the inherited reader is not attributed).
        name = getattr(arg, "__name__", None)
        if name is not None and isinstance(receiver, type):
            channel = self._datetime_clock_channel(receiver, name)
            if channel is not None:
                self._mark(channel)

    def _datetime_clock_channel(self, receiver: type, name: str) -> str | None:
        """Classify a ``datetime``/``date`` current-clock ``c_call`` receiver (r42 hon1_1)."""

        # Exact base receiver: the original registered key (fast path, unchanged).
        exact = self._clock_ccall_keys.get((id(receiver), name))
        if exact is not None:
            return exact
        # Inherited C reader on a subclass: ``now``/``utcnow``/``today`` on a ``datetime``
        # subclass, ``today`` on a (non-datetime) ``date`` subclass. Not attributed when the
        # subclass genuinely OVERRIDES the reader (its own ``__dict__`` defines the name before
        # the base in the MRO) -- an override that returns a fixed value is not a clock read; if
        # it calls the inherited reader, that inner call is caught by the exact-key path.
        if name in ("now", "utcnow", "today") and issubclass(receiver, _datetime_module.datetime):
            if not self._subclass_overrides_reader(receiver, name, _datetime_module.datetime):
                return f"datetime.datetime.{name}"
            return None
        if name == "today" and issubclass(receiver, _datetime_module.date):
            if not self._subclass_overrides_reader(receiver, name, _datetime_module.date):
                return "datetime.date.today"
        return None

    @staticmethod
    def _subclass_overrides_reader(receiver: type, name: str, base: type) -> bool:
        """Return whether ``receiver`` redefines ``name`` above ``base`` in its MRO (r42 hon1_1)."""

        for klass in receiver.__mro__:
            if klass is base:
                return False
            if name in getattr(klass, "__dict__", {}):
                return True
        return False

    def _make_profile_hook(self, predecessor: Any, *, records_thread_ident: bool = False) -> Any:
        """Build the ``sys``/``threading`` profile hook, chained ahead of ``predecessor``.

        The hook classifies ``c_call`` events against the held-builtin registry and
        ``call`` events against the held torch-RNG code registry, and snapshots
        numpy generator state per frame. ``records_thread_ident`` is set for the
        ``threading`` copy so every hooked thread registers its ident during
        bootstrap, before its first user statement, making the escape belt's
        in-window classification race-free. Any classifier error degrades
        completeness rather than propagating into the traced program.
        """

        def hook(frame: Any, event: str, arg: Any) -> Any:
            """Classify one profile event, then chain to the predecessor hook."""

            if self._hooks_retired:
                # The window that installed this hook is over. ``threading.setprofile``
                # only seeds NEW threads, so a worker started in-window keeps this hook
                # as its thread-local profile function forever -- a permanent per-call
                # tax, a permanent strong reference to the model graph, and (worse) a
                # later capture's draws on that thread classifying into this dead
                # window's discarded result. A non-LIFO overlap can likewise hand this
                # hook back to the process. Self-uninstall on the first event after
                # teardown -- but ONLY when this hook is the one currently installed on
                # this thread: while it is merely a LINK in a live successor's chain,
                # uninstalling would tear down that successor's window too.
                try:
                    if _sys_module.getprofile() is hook:
                        # Held original: a LIVE later window may have its
                        # swap-detection wrapper on the module attr, and this
                        # dead-window self-uninstall must not flag it.
                        setter = self._orig_sys_setprofile or _sys_module.setprofile
                        setter(predecessor)
                except Exception:
                    pass
                if predecessor is not None:
                    try:
                        predecessor(frame, event, arg)
                    except Exception:
                        pass
                return None
            if records_thread_ident:
                # r41 hon2_1: the threading hook registers every hooked thread's ident
                # (idempotent set.add, GIL-atomic) during thread bootstrap -- BEFORE the
                # thread's first user statement -- so the escape belt's in-window
                # classification is race-free even for an escape-first thread.
                try:
                    self._in_window_thread_idents.add(_threading_module.get_ident())
                except Exception:
                    self._flag_uncertain("profile_classifier_error")
            try:
                if event == "c_call":
                    # The classifier's own first exits, inlined: a call that is not a
                    # held reference and whose receiver is absent or one of the exact
                    # inert built-in types returns without classifying. Everything
                    # else enters the unchanged classifier.
                    receiver = getattr(arg, "__self__", None)
                    if id(arg) in self._held_ref_marks or (
                        receiver is not None
                        and type(receiver) not in _types._INERT_PROFILE_C_CALL_RECEIVER_TYPES
                    ):
                        self._classify_c_call(frame, arg)
                elif event == "call":
                    # Avoid two Python helper calls for the overwhelmingly common
                    # internal frame whose cached digest scope is false. Cache misses
                    # and positive scopes still enter the unchanged snapshot helper.
                    digest_scope = self._numpy_frame_digest_scope_cache.get(id(frame.f_code))
                    if (
                        digest_scope is None
                        or digest_scope[0] is not frame.f_code
                        or digest_scope[1] is not False
                    ):
                        self._snapshot_numpy_frame_rngs(frame)
                    # r65 CLUSTER Z: held-reference torch RNG spellings are Python
                    # functions -- classified by code identity on ``call`` events (the
                    # r41 builtin-identity layer only ever sees ``c_call``).
                    if id(frame.f_code) in self._held_code_marks:
                        self._classify_call(frame)
                elif event == "return":
                    if id(frame) in self._numpy_frame_rng_states:
                        self._compare_numpy_frame_rngs(frame)
                    # Same cached scope test as the ``call`` branch, applied to the
                    # caller the value returns into: a cached implementation caller
                    # is exactly the helper's own early return.
                    if _types._NUMPY_RNG_METHODS_NEED_FRAME_DIGEST:
                        caller = frame.f_back
                        if caller is not None:
                            caller_scope = self._numpy_frame_digest_scope_cache.get(
                                id(caller.f_code)
                            )
                            if (
                                caller_scope is None
                                or caller_scope[0] is not caller.f_code
                                or caller_scope[1] is not False
                            ):
                                self._snapshot_returned_numpy_rngs(frame, arg)
            except Exception:
                self._flag_uncertain("profile_classifier_error")
            if predecessor is not None:
                try:
                    predecessor(frame, event, arg)
                except Exception:
                    self._flag_uncertain("profile_predecessor_error")

        cast(Any, hook)._tl_owner = self  # dead-chain restore walks use this (non-LIFO fix).
        return hook

    @staticmethod
    def _is_recursable_custom_holder(value: Any) -> bool:
        """Return whether ``value`` is a holder to recurse into (r42 corr2_1 / r51 hon1_1).

        Recurse into plain user objects that carry an attribute surface AND (r51 hon1_1) into
        every reachable ``nn.Module`` -- its own ``__dict__`` / ``__slots__`` (which hold
        ``_parameters`` / ``_buffers`` / ``_modules`` plus arbitrary user attributes). A REGISTERED
        submodule is additionally seeded via the top ``model.modules()`` loop (and premarked in
        ``seen_container_ids`` so it is not double-walked); an UNREGISTERED submodule (held in a
        plain attribute, ``list`` / ``dict`` / nested container, or custom holder -- the "modules
        must live in an ``nn.ModuleList``" footgun) is reachable ONLY through this branch, so a
        numpy Generator behind it is finally witnessed. Skips scalars, classes (walked through
        the class-surface edge instead, r53 corr), modules, tensors, ndarrays, and any
        stdlib/torch/numpy implementation object (:data:`_CUSTOM_HOLDER_SKIP_MODULES`). A
        digestable RNG never reaches here (it is snapshotted before this branch).

        r53 corr/F1: the former blanket ``callable(value) -> False`` gate is GONE -- it made a
        user-defined CALLABLE INSTANCE (``self.op = CallableSampler()``, the idiomatic callable
        transform/sampler object) a hard leaf, leaving its held generator unwitnessed (false
        VERIFIED). r55 C6: this predicate now gates ONLY the dual-walk of container/weakref
        SUBCLASS own-attribute surfaces (the Mapping / Collection / ``weakref.ref`` branches);
        the sweep's terminal fallback expands every other node through the authoritative
        ``gc.get_referents`` enumerator (:meth:`_inert_gc_children`) instead of this
        module-gated attribute-surface check.
        """

        if value is None or isinstance(value, (str, bytes, bytearray, int, float, bool, complex)):
            return False
        # r51 hon1_1: an nn.Module is a recursable holder. This branch MUST precede the
        # module / torch-namespace gates below because a torch built-in module (``nn.Linear``)
        # lives in the skipped ``torch`` top-level namespace -- the gate would otherwise
        # short-circuit it to a hard leaf, the hon_1 hole.
        if isinstance(value, torch.nn.Module):
            return True
        if isinstance(value, type):
            return False
        if isinstance(value, ModuleType):
            return False
        if isinstance(value, (np.ndarray, np.generic)):
            return False
        if isinstance(value, torch.Tensor):  # r51 hon1_1: was ``(torch.Tensor, torch.nn.Module)``;
            return False  # nn.Module is now recursable via the branch above.
        module = getattr(type(value), "__module__", "") or ""
        if module.split(".", 1)[0] in _CUSTOM_HOLDER_SKIP_MODULES:
            return False
        try:
            has_dict = isinstance(object.__getattribute__(value, "__dict__"), dict)
        except (AttributeError, TypeError):
            has_dict = False
        has_slots = any(
            "__slots__" in getattr(klass, "__dict__", {}) for klass in type(value).__mro__
        )
        return has_dict or has_slots

    @staticmethod
    def _custom_holder_children(value: Any) -> list[Any]:
        """Return an inert custom holder's attribute values (r42 corr2_1).

        Reads only ``__dict__`` values and ``__slots__`` slot values (via the slot member
        descriptor, never ``getattr`` -- so no property getter and no ``__getattr__`` fires).
        r61 corr_1: slot descriptors resolve through :func:`_resolve_slot_member`, so a
        PRIVATE slot (``__slots__ = ("__rng",)`` -- class-dict key ``_Class__rng`` via
        CPython name mangling) is read like any other; every caller (registered-module
        seed, Mapping/Collection dual-walk, weakref-subclass, numeric-payload branch)
        inherits the fix at this single choke point. r63: the resolver prefers the
        mangled private descriptor over any raw class-dict shadow and returns ONLY an
        inert ``MemberDescriptorType`` (or ``None`` -- slot skipped), so a planted raw
        shadow can neither hide the real slot value nor execute a hostile ``__get__``.
        """

        children: list[Any] = []
        try:
            instance_dict = object.__getattribute__(value, "__dict__")
        except (AttributeError, TypeError):
            instance_dict = None
        if isinstance(instance_dict, dict):
            children.extend(instance_dict.values())
        for klass in type(value).__mro__:
            slots = klass.__dict__.get("__slots__")
            if slots is None:
                continue
            if isinstance(slots, str):
                slots = (slots,)
            try:
                slot_names = list(slots)
            except TypeError:
                continue
            for slot in slot_names:
                if not isinstance(slot, str) or slot in ("__dict__", "__weakref__"):
                    continue
                getter = getattr(_resolve_slot_member(klass, slot), "__get__", None)
                if getter is None:
                    continue
                try:
                    children.append(getter(value, klass))
                except (AttributeError, TypeError, ValueError):
                    continue
        return children

    @staticmethod
    def _is_trusted_leaf_class(klass: type) -> bool:
        """Return whether the class-surface edge treats ``klass`` as a trusted leaf (r53 corr).

        ``object`` / ``type`` and any class whose OWN raw ``__module__`` roots in
        :data:`_CUSTOM_HOLDER_SKIP_MODULES` (torch / numpy / stdlib runtime internals) OR in
        :data:`_STDLIB_CLASS_LEAF_MODULES` (r56 amb_1: the WHOLE stdlib, structurally, not a
        hand-maintained denylist) are leaves -- their class dicts are implementation noise,
        mirroring the instance-side module gate. The r45 removal of ``collections`` from the
        INSTANCE-side skip set (so ``deque`` / ``UserList`` contents descend) must not walk
        stdlib CLASS dicts: ``collections.abc`` ABCs in a container subclass's MRO carry
        ``_abc_impl`` whose caches are process-global registries (the ambient escape).
        Instance-side descent never needed the stdlib class-dict surface; a generator
        monkeypatched ONTO a stdlib class object is shared runtime state, the same
        documented residual family as a shared module global. ``__main__`` is explicitly
        NOT a leaf even though ``sys.stdlib_module_names`` lists it -- a model class defined
        in a script/notebook must keep its r53 class-attribute coverage. ``__module__`` is
        read through the base ``type`` getset (``type.__dict__["__module__"].__get__``),
        never ``getattr``, so a hostile metaclass property never fires (probed). The getset
        is pure C on both class shapes: a HEAP class reads its ``tp_dict`` entry (identical
        to the former raw-dict read), while a STATIC C extension type (``np.ndarray`` /
        ``np.float64`` / ``builtin_function_or_method``) derives it from ``tp_name`` -- those
        types store NO dict entry, so the former raw-dict read returned ``None`` and walked
        every C extension implementation class; r63 requires the stock numeric-payload
        classes leafed once the payload branch enqueues ``type(value)``. An
        unreadable or non-string ``__module__`` fails toward WALKING the class -- the walk
        is inert, so the unprovable case errs toward MORE coverage, never code execution.
        """

        if klass is object or klass is type:
            return True
        try:
            module = type.__dict__["__module__"].__get__(klass)
        except Exception:
            return False
        if not isinstance(module, str):
            return False
        root = module.split(".", 1)[0]
        return root in _CUSTOM_HOLDER_SKIP_MODULES or root in _STDLIB_CLASS_LEAF_MODULES

    @staticmethod
    def _shared_namespace_dict_ids() -> frozenset[int]:
        """Identity keys of every loaded module's ``__dict__`` (r55 C6 exclusion set).

        A function's ``__globals__`` / ``__builtins__`` and any other reference
        into a live module namespace are SHARED state, not model state: expanding
        one would walk every framework's globals from a single model attribute.
        Excluding them by ``__dict__`` IDENTITY (never by name heuristics) makes
        "a generator held in a shared module global, drawn on a pre-existing
        non-hooked thread" the explicit documented residual of the inert sweep.

        r57 C6 follow-up: this registry is NOT the only wall for a function's
        namespace -- ``_inert_gc_children`` additionally drops a ``FunctionType``
        node's own ``__globals__`` / ``__builtins__`` referents by ROLE, so an
        ``exec``/``torch.fx``-generated function whose synthetic globals dict is
        absent from ``sys.modules`` is walled all the same.
        """

        ids = set()
        for module in list(_sys_module.modules.values()):
            module_dict = getattr(module, "__dict__", None)
            if module_dict is not None:
                ids.add(id(module_dict))
        return frozenset(ids)

    @staticmethod
    def _inert_gc_children(value: Any, shared_namespace_ids: frozenset[int]) -> list[Any]:
        """Return a node's AUTHORITATIVE inert reference edges (r55 C6, corr_2/corr_4).

        ``gc.get_referents(value)`` runs CPython ``tp_traverse`` -- pure C field
        enumeration that NEVER executes Python: no property getter, descriptor
        ``__get__``, ``__getattr__``, or user callable can fire (probed: a hostile
        ``@property``/``__getattr__`` counter stays at zero). Unlike the replaced
        hand-maintained callable-interior vocabulary (closure/defaults/kwdefaults/
        partial/property/...bound-method fields), the traverse exposes EVERY inert
        reference field an object type declares -- ``__annotations__`` (r54
        corr_2), ``functools`` wrapper ``__wrapped__`` chains (r54 corr_4),
        ``__dict__``, slots, cells -- so a new hiding field is unreachable only if
        CPython itself cannot reach it for garbage collection. This is a ROOTED
        per-object enumerator feeding the existing cycle-guarded, node-capped
        model walk -- NOT a process-wide ``gc.get_objects()`` scan.

        Dropped edges are exactly the three documented exclusion families: referents
        whose identity is a loaded module's ``__dict__`` (shared namespaces --
        ``shared_namespace_ids``), instances of the ambient-bridge
        :data:`_AMBIENT_BRIDGE_LEAF_TYPES`, and a ``FunctionType`` node's own
        ``__globals__`` / ``__builtins__`` namespace edges (dropped by ROLE and
        identity, registry-independent). Everything else is enqueued and handled
        by the sweep's typed branches (a referent dict via the Mapping protocol, a
        referent class via the trusted-leaf-gated class-surface branch, a referent
        RNG via the digest). r61 hon_1: a :data:`_NUMERIC_PAYLOAD_LEAF_TYPES`
        node (tensor / ndarray / numpy scalar) is NOT dropped -- as a referent it
        is enqueued like any other node, and on its own visit it contributes
        EXACTLY its Python instance ``__dict__`` / ``__slots__`` values via
        :meth:`_custom_holder_children` PLUS its class-surface edge
        (``type(value)``, r63 -- routed through the trusted-leaf-gated class
        branch so a class-attribute generator on a user-defined subclass is
        inventoried while stock torch/numpy classes stay leaves); never
        ``gc.get_referents`` on the C object, so the numeric buffer / autograd
        internals stay unwalked while ``weight.rng = default_rng()`` is
        inventoried.

        r57 C6 (r56 hon_1): a BOUND C callable (``BuiltinFunctionType`` /
        ``MethodWrapperType``) contributes exactly its ``__self__`` RECEIVER --
        for ``gen.random`` / ``RandomState().rand`` /
        ``random.Random(0).random`` the receiver IS the RNG state holder, and
        ``gc.get_referents(bound_method)`` exposes it inertly (zero Python
        executed). The receiver passes through the SAME walls as every other
        node: a module-level C function (``math.sqrt`` / ``np.array`` -- module
        ``__self__``; ``torch.relu`` / ``torch.zeros`` -- ``None``/absent
        ``__self__``) stays a leaf, a shared-namespace or expansion-leaf
        receiver stays walled (the r47/r56 ambient-escape closes are untouched),
        and a class receiver (``int.from_bytes``) is enqueued into the
        trusted-leaf-gated class branch. The node NEVER expands generically:
        ``gc.get_referents(torch.relu)`` would enqueue its defining
        ``torch._C`` type -- targeted receiver-enqueue beats blanket expansion.

        r57 C6 follow-up (callable-ambient-escape close): a ``FunctionType``
        node -- a plain Python function, a lambda, a ``functools.wraps``-style
        wrapper, or a TorchLens-wrapped torch op (after ANY capture in the
        process, ``wrap_torch()`` rebinds ``torch.relu`` / ``torch.zeros`` from
        ``BuiltinFunctionType`` to a ``FunctionType`` closure wrapper, so
        ``self.act = torch.relu`` IS a FunctionType model attribute in
        production) -- expands through the SAME authoritative traverse minus
        its NAMESPACE edges dropped BY ROLE AND IDENTITY: ``__globals__`` and
        ``__builtins__``. Registry membership (``shared_namespace_ids``) is NOT
        relied on for these two edges, because an ``exec``/``torch.fx``/dynamo-
        generated function carries a SYNTHETIC globals dict that is no loaded
        module's ``__dict__`` -- pre-fix, one such attribute walked its entire
        namespace (the r47/r56 ambient-escape class: under a large ambient
        graph the node cap is exhausted BEFORE the model's own generator).
        Every OTHER function edge stays enqueued, exactly like the generic
        fallback: ``__defaults__`` / ``__kwdefaults__`` / ``__closure__`` /
        ``__annotations__`` (r54 corr_2) / the function's own ``__dict__``
        (``fn.stash = gen``) keep their pinned generator coverage, and the
        ``__module__`` / ``__name__`` / ``__qualname__`` / ``__doc__`` metadata
        referents are inert terminals (a ``str`` has no referents). Builtin-vs-
        wrapped torch ops therefore contribute the same thing to the sweep:
        nothing ambient. Non-function callables (bound Python methods,
        ``functools.partial`` / ``staticmethod`` / ``classmethod`` /
        ``lru_cache`` wrappers, ``OpOverload``-style callable instances,
        user callable instances) NEED no namespace drop: their ``tp_traverse``
        enumerates only per-instance interior fields (``__self__`` /
        ``__func__`` / ``func``-``args``-``keywords`` / ``__wrapped__`` /
        instance ``__dict__``), never a namespace dict -- any FUNCTION among
        those children is caught by this branch on its own visit, and a
        callable INSTANCE's ``__dict__`` (which can hold the model's generator)
        must stay generically walked.
        """

        numpy_receiver = _numpy_rng_receiver(value)
        if numpy_receiver is not None:
            if id(numpy_receiver) in shared_namespace_ids or isinstance(
                numpy_receiver, _AMBIENT_BRIDGE_LEAF_TYPES
            ):
                return []
            return [numpy_receiver]
        if isinstance(value, (BuiltinFunctionType, MethodWrapperType)):
            receiver = getattr(value, "__self__", None)
            if (
                receiver is None
                or id(receiver) in shared_namespace_ids
                or isinstance(receiver, _AMBIENT_BRIDGE_LEAF_TYPES)
            ):
                return []
            # r61 hon_1: a NUMERIC-PAYLOAD receiver (``tensor.add.__self__`` is the
            # tensor) is enqueued -- its own visit walks instance state only.
            return [receiver]
        if isinstance(value, FunctionType):
            # Namespace edges by ROLE (identity match), never by registry lookup:
            # a synthetic (exec/fx/dynamo) globals dict is walled even though it
            # is absent from ``shared_namespace_ids``. Both reads are C member
            # accesses on the static ``FunctionType`` (no descriptor can fire).
            namespace_edge_ids = {id(value.__globals__)}
            function_builtins = getattr(value, "__builtins__", None)
            if function_builtins is not None:
                namespace_edge_ids.add(id(function_builtins))
            function_children: list[Any] = []
            for referent in _gc_module.get_referents(value):
                if id(referent) in namespace_edge_ids:
                    continue
                if id(referent) in shared_namespace_ids:
                    continue
                if isinstance(referent, _AMBIENT_BRIDGE_LEAF_TYPES):
                    continue
                function_children.append(referent)
            return function_children
        if isinstance(value, _AMBIENT_BRIDGE_LEAF_TYPES):
            return []
        if isinstance(value, _NUMERIC_PAYLOAD_LEAF_TYPES):
            # r61 hon_1: never ``gc.get_referents`` on a tensor/ndarray (the C
            # buffer / autograd internals stay unwalked), but the Python instance
            # ``__dict__`` / ``__slots__`` surfaces are real inert holder state.
            # r63 (r62 class-surface MED): the node's CLASS surface flows too --
            # ``type(value)`` is enqueued into the sweep's trusted-leaf-gated
            # class branch (raw mappingproxy reads, per-class dedup), so a
            # class-attribute generator on a USER-defined Tensor / Parameter /
            # ndarray subclass is inventoried while stock torch/numpy
            # implementation classes stay trusted leaves. This makes the branch
            # structurally identical to every other holder branch: instance
            # state via ``_custom_holder_children`` PLUS the class edge.
            payload_children = host_nondeterminism_monitor._custom_holder_children(value)
            payload_children.append(type(value))
            return payload_children
        children: list[Any] = []
        for referent in _gc_module.get_referents(value):
            if id(referent) in shared_namespace_ids:
                continue
            if isinstance(referent, _AMBIENT_BRIDGE_LEAF_TYPES):
                continue
            children.append(referent)
        return children

    @staticmethod
    def _exposes_queue_protocol(value: Any) -> bool:
        """Return whether ``value``'s TYPE implements the standard queue protocol (r45 hon1_1).

        Duck-typed on the CLASS (never the instance, so no property getter or ``__getattr__``
        side effect fires): ``get`` + ``put`` + (``qsize`` or ``empty``) as callables. This
        matches ``queue.Queue`` / ``LifoQueue`` / ``PriorityQueue`` (an inspectable ``.queue``
        deque) AND ``queue.SimpleQueue`` / ``multiprocessing.Queue`` / any future opaque queue
        (no non-mutating buffer) by construction -- NOT a concrete ``SimpleQueue`` type list.
        Ordinary ``Mapping`` / ``Collection`` holders are caught by earlier descent branches and
        never reach this predicate.
        """

        cls = type(value)
        if not (callable(getattr(cls, "get", None)) and callable(getattr(cls, "put", None))):
            return False
        return callable(getattr(cls, "qsize", None)) or callable(getattr(cls, "empty", None))

    @staticmethod
    def _opaque_queue_provably_empty(value: Any) -> bool:
        """Return whether an opaque queue is NON-MUTATINGLY provably empty (r47 hon1_2).

        ``queue.SimpleQueue`` / ``multiprocessing.Queue`` expose no non-mutating payload snapshot
        (``.get`` would DRAIN), so their contents cannot be inventoried. But ``empty()`` and
        ``qsize()`` are NON-MUTATING (probed), so a queue they report EMPTY provably holds no
        generator and can safely stay VERIFIED. True IFF ``empty()`` is exactly ``True`` (authoritative
        non-empty when it is exactly ``False``), else ``qsize()`` is integer ``0``. Any exception,
        non-bool ``empty()``, non-int/negative ``qsize()``, or unsupported value fails closed to
        ``False`` -- ``mp.Queue`` can raise/flake, so a non-empty or unknown queue is NEVER read as
        empty.
        """

        empty_fn = getattr(value, "empty", None)
        if callable(empty_fn):
            try:
                empty_result = empty_fn()
            except Exception:
                empty_result = None
            if empty_result is True:
                return True
            if empty_result is False:
                return False
        qsize_fn = getattr(value, "qsize", None)
        if callable(qsize_fn):
            try:
                size = qsize_fn()
            except Exception:
                return False
            if type(size) is int and size == 0:
                return True
        return False

    def _sweep_model_generators(self) -> list[tuple[Any, str]]:
        """Digest numpy/`random` generators reachable from the MODEL's submodule attributes.

        This is the CHEAP thread-independent belt (model attributes only), NOT a
        process-wide ``gc.get_objects()`` scan (the r39-draft GC-wide inventory cost
        ~900 ms/capture, perturbed the peak-memory bracket, and over-trigger-risked
        unrelated generators -- removed for cause and never reintroduced).

        r41 hon1_2 / r45 hon1_1: the sweep is an ITERATIVE, cycle-safe recursion by
        container PROTOCOL (not a fixed concrete-type list) -- every submodule ``__dict__``
        value, descending through every ``collections.abc.Mapping`` (KEYS and VALUES) and
        every non-leaf ``collections.abc.Collection`` (elements), so ``self.pool = [gen]``,
        ``{"g": gen}``, ``[[gen]]``, a dict-key generator, and a generator inside a ``deque``
        / ``ChainMap`` / ``UserList`` / ``UserDict`` / namedtuple / custom ``Sequence`` or
        ``Mapping`` are all digested. Descent is gated on ``Collection`` (Sized), NEVER bare
        ``Iterable``, so a one-shot generator / ``map`` / ``itertools`` attribute is never
        consumed. A safe queue (``queue.Queue`` / ``LifoQueue`` / ``PriorityQueue``) is reached
        through a non-mutating snapshot of its internal ``.queue`` deque; an opaque queue
        (``SimpleQueue`` / ``mp.Queue``) with no inspectable buffer fails closed to INCOMPLETE
        (``inventory_opaque_container``). r42 corr2_1: it ADDITIONALLY recurses into inert
        CUSTOM object holders (``self.holder.rng``) through their ``__dict__`` / ``__slots__``
        values, skipping tensors, ndarrays, callables, modules, and stdlib/torch/numpy
        implementation objects. r51 hon1_1: an ``nn.Module`` is NO LONGER a hard leaf -- every
        reachable module (REGISTERED via the top ``model.modules()`` loop AND an UNREGISTERED
        submodule held in a plain attribute / ``list`` / ``dict`` / nested container / custom
        holder) is descended through the same ``__dict__`` / ``__slots__`` protocol, so a numpy
        Generator behind an unregistered submodule is witnessed (registered ids are premarked so
        there is no double-walk; r59 hon_1: the registered seed loop routes through
        :meth:`_custom_holder_children`, so a registered module's ``__slots__`` slot values are
        seeded alongside its ``__dict__`` -- the former dict-only seed left a slot-held generator
        unwitnessed on a pre-existing thread). The former
        ``budget = 2000`` early return truncated SILENTLY (a generator on a late
        submodule of any >~120-module model was missed -> false VERIFIED); the
        replacement :data:`_INVENTORY_NODE_CAP` is defensive only -- exhaustion flags
        ``inventory_budget_exhausted`` (INCOMPLETE), never a silent partial snapshot.

        The realistic hon1_1/corr2_2 CROSS-THREAD draw (an externally-held numpy
        Generator drawn on an in-window helper thread) is caught by the
        ``threading.setprofile`` receiver classifier; owner-thread instance draws and
        the immutable ``datetime`` readers by the ``sys.setprofile`` classifier;
        unseeded construction and Python ``random`` by the construction/class patches.
        This digest is the thread-independent belt for a generator the model itself
        HOLDS (caught even on a pre-existing thread; r42 corr2_1 extends this to a
        generator behind a custom holder attribute; r51 hon1_1 to a generator behind an
        UNREGISTERED submodule).

        r53 corr_1/corr_2/F1: the walk's structural invariant is REACHABILITY --
        it follows EVERY reference edge that can be followed WITHOUT executing
        user-defined code. Beyond the instance/container surfaces above it walks:
        (1) class-MRO ``__dict__`` surfaces of user-defined classes (raw mappingproxy
        reads through the base ``type`` getsets -- the descriptor protocol only fires
        on getattr, NEVER on a ``values()`` read -- so a class-descriptor-held
        submodule/generator and a plain class-attribute generator are reached;
        torch/stdlib/numpy implementation classes are trusted leaves);
        (2) ``weakref.ref`` / ``WeakMethod`` referents through ONE base-C dereference
        (``weakref.ref.__call__``, immune to hostile ``__call__`` overrides; weak
        CONTAINERS descend via the ordinary Mapping/Collection protocols);
        (3) r55 C6 (corr_2/corr_4): every OTHER node's reference edges through the
        AUTHORITATIVE ``gc.get_referents`` enumerator (:meth:`_inert_gc_children`)
        -- CPython ``tp_traverse``, pure C, zero Python executed -- minus the three
        documented exclusion families (loaded-module ``__dict__`` identities,
        :data:`_AMBIENT_BRIDGE_LEAF_TYPES`, and a ``FunctionType`` node's own
        ``__globals__`` / ``__builtins__`` namespace edges by identity -- so a
        TorchLens-wrapped or ``exec``/fx-generated function never walks a
        namespace); a :data:`_NUMERIC_PAYLOAD_LEAF_TYPES` node (tensor /
        ndarray / numpy scalar) contributes exactly its instance
        ``__dict__`` / ``__slots__`` values, never its C buffer (r61 hon_1);
        a BOUND C callable contributes exactly
        its ``__self__`` receiver through those same walls (r57 C6 -- a cached
        ``self.sample = self.rng.random`` bound method, alone or behind
        a ``partial`` / closure cell / ``staticmethod`` / ``classmethod`` / dict
        / list, recovers the generator; ``math.sqrt``-style module functions
        stay leafed). This SUBSUMES the r53 hand-maintained
        callable-interior vocabulary (closure cells, defaults, kwdefaults,
        ``partial``/``property``/``static``/``classmethod`` fields, bound-method
        ``__func__``/``__self__``, callable-instance ``__dict__``/``__slots__``)
        and closes its whole drift class: ``__annotations__`` (r54 corr_2),
        ``functools`` wrapper ``__wrapped__`` chains (r54 corr_4), and any future
        inert field a type declares are reached by construction, not by table
        maintenance. The walk never invokes a property, a descriptor ``__get__``,
        ``__getattr__``, or any user callable. Residual (contract s11): a generator
        reachable ONLY BY EXECUTING USER CODE (a property/descriptor ``__get__``
        body, ``__getattr__``, or a callable's return value) or held ONLY in a
        SHARED module-global namespace, drawn on a PRE-EXISTING (non-hooked)
        thread.
        """

        snapshots: list[tuple[Any, str]] = []
        model = self._model
        # Enumerate the registered module tree through the AUTHORITATIVE base
        # implementation, bypassing any user override of ``modules()``. An
        # nn.Module subclass that overrides ``modules()`` to return an empty (or
        # otherwise lying) iterable would otherwise hide every submodule -- and
        # any model-held RNG -- from this sweep, producing a clean false VERIFIED
        # (``channels=[] uncertain=False``). Reading through the base method is
        # the same posture as the class-surface reads below that go through base
        # ``type`` getsets so a hostile override never fires. Materialize once;
        # the module set is consumed twice below.
        if isinstance(model, torch.nn.Module):
            registered_modules = list(torch.nn.Module.modules(model))
        else:
            modules = getattr(model, "modules", None)
            if not callable(modules):
                return snapshots
            registered_modules = list(modules())
        try:
            # r55 C6: shared-namespace exclusion set for the gc-referent fallback,
            # computed ONCE per sweep (bounded by loaded-module count).
            shared_namespace_ids = self._shared_namespace_dict_ids()
            pending: list[Any] = []
            for module in registered_modules:
                # r59 hon_1: seed BOTH the instance ``__dict__`` values AND the
                # ``__slots__`` slot values of every REGISTERED module through
                # ``_custom_holder_children`` (slot reads go through the slot member
                # descriptor -- inert, no property / ``__getattr__`` / user code), the
                # same protocol unregistered holders already get. The former
                # ``__dict__``-only seed left a registered module's SLOT-held generator
                # invisible: the r51 premark below (KEPT -- coverage-neutral dedup)
                # skips a registered module when it is later reached as a walk node,
                # and the ROOT module is never a walk node at all, so
                # ``_custom_holder_children`` never ran for the ``model.modules()``
                # set -> a pre-existing-thread draw from a slot generator was
                # unwitnessed (false VERIFIED). Dropping the premark instead is
                # DISQUALIFIED: it still misses the top module's slots and re-walks
                # every registered submodule (~2.2x).
                pending.extend(self._custom_holder_children(module))
                # r53 corr_1 (class-surface edge): the registered module's CLASS is a holder
                # surface too -- a class descriptor or a class-attribute generator on the
                # user model class is invisible to the instance-``__dict__`` seed above.
                pending.append(type(module))
            # r51 hon1_1: nn.Module is now a recursable holder (``_is_recursable_custom_holder``),
            # so a REGISTERED submodule reached during descent would be re-walked even though the
            # top loop already seeded it. Premark every registered module id so the
            # ``id in seen_container_ids`` guards skip that re-walk -- a coverage-NEUTRAL dedup (a
            # generator directly on a registered module is seeded and digested from its
            # ``__dict__`` / ``__slots__`` values -- r59 hon_1 -- before the module OBJECT is ever
            # reached) that kills the ~2x double-walk. UNREGISTERED
            # submodules are absent from ``modules()`` and stay un-premarked, so they ARE descended.
            seen_container_ids: set[int] = {id(module) for module in registered_modules}
            visited_nodes = 0
            while pending:
                value = pending.pop()
                visited_nodes += 1
                if visited_nodes > _types._INVENTORY_NODE_CAP:
                    self._flag_uncertain("inventory_budget_exhausted")
                    return snapshots
                # r39 (frame-walk parity, executed false-VERIFIED V9a): a weakref
                # PROXY forwards EVERY read -- including ``isinstance`` (via
                # ``__class__``) and the Mapping/Collection protocol reads below --
                # through its referent's own attribute machinery, and NO inert deref
                # exists (``tp_traverse`` does not yield the referent; see
                # :data:`_WEAKREF_PROXY_TYPES`). Match the EXACT type BEFORE any
                # isinstance branch can be spoofed into a forwarding read, and fail
                # CLOSED by design (pre-r39 this only ceilinged via an incidental
                # ``inventory_scan_failed`` exception).
                if type(value) in _WEAKREF_PROXY_TYPES:
                    self._flag_uncertain("inventory_opaque_container")
                    continue
                # r39 (frame-walk parity, executed false-VERIFIED V9c-model): a
                # ``ContextVar``'s VALUE lives in the per-thread ``Context``, off the
                # reference graph -- ``tp_traverse`` does not expose it, so the gc
                # fallback below is blind. The base C ``get`` (set value or declared
                # default, owner thread) is the only inert edge; pre-existing OTHER
                # threads' contexts stay the documented foreign-thread residual.
                if type(value) is _contextvars_module.ContextVar:
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    try:
                        pending.append(_contextvars_module.ContextVar.get(value))
                    except LookupError:
                        pass
                    continue
                # r53 corr_1 (class-surface edge): a CLASS node contributes its raw
                # ``__dict__`` values -- a mappingproxy read reaches descriptor OBJECTS
                # (class-descriptor-held submodules/generators) and plain class-attribute
                # generators WITHOUT ever firing ``__get__`` (the descriptor protocol only
                # fires on getattr, never on a ``values()`` read) -- plus its remaining MRO
                # classes. Reads go through the base ``type`` getsets so a hostile
                # metaclass property on ``__dict__``/``__mro__`` never fires;
                # torch/stdlib/numpy implementation classes are trusted leaves. Per-class
                # dedup keeps this trivially bounded (distinct user classes, not modules).
                if isinstance(value, type):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    # r39 (executed false-VERIFIED V9d): ``CLS.gen`` can resolve
                    # through ``type(CLS).__mro__`` -- a user METACLASS class-var --
                    # entirely in C. Enqueue the metaclass into this same
                    # trusted-leaf-gated branch (``type`` itself is a trusted leaf,
                    # so the extra edge is dedup-bounded to user metaclasses).
                    pending.append(type(value))
                    if self._is_trusted_leaf_class(value):
                        continue
                    try:
                        raw_dict = type.__dict__["__dict__"].__get__(value)
                        mro = type.__dict__["__mro__"].__get__(value)
                    except Exception:
                        continue  # inert read failed: skip posture (nothing executed)
                    pending.extend(raw_dict.values())
                    pending.extend(base for base in mro if base is not value)
                    continue
                # r53 corr_2 (weakref edge): dereference ONCE at base-C level
                # (``weakref.ref.__call__`` -- immune to a hostile subclass ``__call__``
                # override) and enqueue the live referent; a dead ref contributes nothing.
                # Covers ``WeakMethod`` (its base ref points at ``__self__``); weak
                # CONTAINERS (``WeakSet``/``WeakValueDictionary``/``WeakKeyDictionary``)
                # already descend via the Mapping/Collection protocol branches below. A
                # SUBCLASS instance additionally contributes its own inert ``__dict__`` /
                # ``__slots__`` values and its class surface. A deref failure fails closed
                # (``inventory_state_read_failed``), never reads as no-referent.
                if isinstance(value, _weakref_module.ref):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    try:
                        referent = _weakref_module.ref.__call__(value)
                    except Exception:
                        self._flag_uncertain("inventory_state_read_failed")
                        continue
                    if referent is not None:
                        pending.append(referent)
                    if type(value) is not _weakref_module.ref:
                        pending.extend(self._custom_holder_children(value))
                        pending.append(type(value))
                    continue
                # r39 (executed false-VERIFIED V9f): an OBJECT-dtype ndarray's ELEMENTS
                # are ordinary Python references -- numpy's own parallel-streams idiom
                # stores spawned Generators this way -- and ``ARR[i]`` is a
                # profile-silent C subscript. Element-iterate ONLY object dtypes,
                # through the base getsets so a subclass property never fires; numeric
                # dtypes keep their buffer unwalked. An over-cap object array fails
                # closed (INCOMPLETE), never a silent partial read. The instance
                # ``__dict__``/``__slots__`` and class edges mirror the gc fallback's
                # numeric-payload branch (r61/r63), which this branch pre-empts for
                # ndarrays only.
                if isinstance(value, np.ndarray):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    try:
                        dtype = np.ndarray.dtype.__get__(value)
                    except Exception:
                        self._flag_uncertain("inventory_state_read_failed")
                        continue
                    if dtype.hasobject:
                        try:
                            size = int(np.ndarray.size.__get__(value))
                        except Exception:
                            self._flag_uncertain("inventory_state_read_failed")
                            continue
                        if visited_nodes + size > _types._INVENTORY_NODE_CAP:
                            self._flag_uncertain("inventory_budget_exhausted")
                            return snapshots
                        try:
                            pending.extend(np.ndarray.flat.__get__(value))
                        except Exception:
                            self._flag_uncertain("inventory_state_read_failed")
                            continue
                    pending.extend(self._custom_holder_children(value))
                    pending.append(type(value))
                    continue
                # r45 hon1_1: descend by container PROTOCOL, not a fixed concrete-type list, so a
                # model-held generator inside a ``deque`` / ``ChainMap`` / ``UserList`` /
                # ``UserDict`` / namedtuple / custom ``Sequence`` / custom ``Mapping`` is reached
                # (the r44 hon1_1 gap). ``Mapping`` first (a ``Mapping`` is also a ``Collection``);
                # then any non-leaf ``Collection``. The gate is ``Collection`` (Sized+Iterable+
                # Container), NEVER bare ``Iterable`` -- a generator / ``map`` / ``zip`` /
                # ``itertools`` object is ``Iterable`` but not ``Collection``, so it is NEVER
                # iterated and a one-shot iterator is never consumed / corrupted.
                if isinstance(value, Mapping):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    pending.extend(value.keys())
                    pending.extend(value.values())
                    # r47 hon1_1: a Mapping that is ALSO a custom (non-stdlib) inert holder can
                    # carry a generator as its OWN attribute (``self.rng`` on a ``UserDict`` /
                    # custom ``Mapping`` subclass), which is NOT among its keys/values. Walk its
                    # own ``__dict__`` / ``__slots__`` too so the held generator is digest-
                    # witnessed. ``_is_recursable_custom_holder`` excludes stdlib/torch/numpy, so a
                    # plain ``dict`` is never double-walked; cycle-guarded via ``seen_container_ids``
                    # (already added above), node-capped, and never invokes a property/``__getattr__``.
                    if self._is_recursable_custom_holder(value):
                        pending.extend(self._custom_holder_children(value))
                        pending.append(type(value))  # r53 corr_1: class surface of the subclass
                    continue
                if isinstance(value, Collection) and not isinstance(value, _INVENTORY_LEAF_TYPES):
                    if id(value) in seen_container_ids:
                        continue
                    seen_container_ids.add(id(value))
                    pending.extend(value)
                    # r47 hon1_1: same dual-walk for a non-leaf Collection that is ALSO a custom
                    # inert holder -- ``self.rng`` on a ``Sequence`` / ``MutableSequence`` /
                    # ``UserList`` / ``__slots__`` collection subclass, not among its elements.
                    if self._is_recursable_custom_holder(value):
                        pending.extend(self._custom_holder_children(value))
                        pending.append(type(value))  # r53 corr_1: class surface of the subclass
                    continue
                if id(value) in self._exempt_ids:
                    continue
                try:
                    digest = self._digest_rng_witnessable(value)
                except _NotADigestableRng:
                    if id(value) in seen_container_ids:
                        continue
                    # r45 hon1_1: a model-held generator can sit inside a QUEUE whose buffer is a
                    # non-mutating-inspectable deque (``queue.Queue`` / ``LifoQueue`` /
                    # ``PriorityQueue`` expose ``.queue``). Snapshot that deque non-destructively.
                    # A queue with NO inspectable buffer (``SimpleQueue`` / ``mp.Queue`` / any
                    # opaque queue) cannot be inventoried without draining it -> FAIL CLOSED to
                    # INCOMPLETE (``inventory_opaque_container``) rather than reading as
                    # no-consumption. Checked BEFORE the gc-referent fallback so an opaque
                    # queue still fail-closes instead of being walked into its
                    # threading/connection internals.
                    if self._exposes_queue_protocol(value):
                        seen_container_ids.add(id(value))
                        inner = getattr(value, "queue", None)
                        if isinstance(inner, Collection) and not isinstance(
                            inner, _INVENTORY_LEAF_TYPES
                        ):
                            pending.append(inner)
                        else:
                            # r49 hon1_1: the emptiness proof is a MONITOR-INTERNAL read. Bracket
                            # ONLY the probe so a clock touched TRANSITIVELY by ``mp.Queue.empty()``
                            # (via ``multiprocessing.connection``) is self-suppressed at the
                            # ``_mark`` choke point, not mis-marked as a model host read. The
                            # ``_flag_uncertain`` branch stays OUTSIDE the bracket so the NON-EMPTY
                            # opaque-queue INCOMPLETE residual is preserved.
                            with self._monitor_internal_probe():
                                provably_empty = self._opaque_queue_provably_empty(value)
                            if provably_empty:
                                # r47 hon1_2: a non-mutatingly PROVABLY-EMPTY opaque queue
                                # (``SimpleQueue`` / ``mp.Queue``) cannot hold a generator, so it
                                # stays VERIFIED -- no over-trigger for a deterministic model that
                                # merely holds an empty queue.
                                pass
                            else:
                                # A NON-EMPTY or unknown opaque queue fails closed (a queue-held
                                # generator drawn on a pre-existing worker would otherwise be
                                # unwitnessed).
                                self._flag_uncertain("inventory_opaque_container")
                        continue
                    # r55 C6 (corr_2/corr_4): AUTHORITATIVE inert fallback. Every node that
                    # is neither a container, a weak reference, a class, a digestable RNG,
                    # nor a queue expands through ``gc.get_referents`` (CPython
                    # ``tp_traverse`` -- zero Python executed) minus the documented
                    # shared-namespace/leaf exclusions and, for a ``FunctionType`` node,
                    # its identity-dropped ``__globals__``/``__builtins__`` namespace
                    # edges (r57 C6 follow-up). This replaces BOTH the r42
                    # custom-holder recursion and the r53 hand-listed callable-interior
                    # vocabulary at this terminal position: ``__dict__``/``__slots__``
                    # values, closure cells, defaults, kwdefaults, ``__annotations__``,
                    # ``functools`` wrapper ``__wrapped__`` chains, bound-method
                    # ``__func__``/``__self__`` (incl. a bound C method's ``__self__``
                    # receiver -- r57 C6), and any future inert field are enqueued by
                    # construction into the same cycle-guarded, node-capped walk (a
                    # referent dict descends via the Mapping branch; a referent class via
                    # the trusted-leaf-gated class branch; a referent generator is
                    # digested). The before/after digest then catches a draw on ANY thread
                    # once the generator is FOUND.
                    seen_container_ids.add(id(value))
                    pending.extend(self._inert_gc_children(value, shared_namespace_ids))
                    continue
                except Exception:
                    self._flag_uncertain("inventory_state_read_failed")
                    continue
                snapshots.append((value, digest))
        except Exception:
            self._flag_uncertain("inventory_scan_failed")
        return snapshots

    @staticmethod
    def _exact_state_repr(state: Any) -> str:
        """Render an RNG state tree EXACTLY, independent of display options.

        ``repr`` of an ndarray obeys the user-global ``np.set_printoptions``
        ``threshold`` (commonly small in notebooks), truncating the 624-word
        MT19937 key to head/tail -- hanging a verdict-steering digest off a
        DISPLAY knob. Arrays render as ``(dtype, shape, tobytes)`` and
        containers recurse, so the digest is bytes-exact and
        printoptions-independent.
        """

        if isinstance(state, np.ndarray):
            return f"ndarray({state.dtype!s},{state.shape!r},{state.tobytes()!r})"
        if isinstance(state, dict):
            rendered = ",".join(
                f"{key!r}:{host_nondeterminism_monitor._exact_state_repr(value)}"
                for key, value in state.items()
            )
            return "{" + rendered + "}"
        if isinstance(state, (tuple, list)):
            rendered = ",".join(
                host_nondeterminism_monitor._exact_state_repr(item) for item in state
            )
            return f"{type(state).__name__}({rendered})"
        return repr(state)

    def _digest_rng_witnessable(self, holder: Any) -> str:
        """Digest one RNG holder, fail-closing on an unwitnessable subclass.

        A ``random.Random`` / numpy-RNG SUBCLASS that overrides a draw or
        state method in USER code escapes every witness the monitor has: the
        class patches sit on the library base (shadowed by the override), the
        ``c_call`` profile classifier never fires for a pure-Python method,
        and the C-state digest does not advance when the override draws from
        its own attributes -- a probe-proven FALSE-CLEAN (grind p5 §3.9,
        rng-subclass-override). Possession of such an engine therefore
        downgrades completeness (``uncertain``, never "no consumption"),
        exactly like the opaque-queue and inventory-failure paths. Library-
        defined subclasses (``SystemRandom``, numpy's ``PCG64``/``MT19937``/
        ... bit generators) stay trusted by defining module, so no shipped
        type over-triggers; the deliberate trade is that HOLDING a user-
        overridden engine ceilings to ``unverifiable`` even undrawn, because
        unlike ``SystemRandom`` (whose draws the class patches still witness)
        a draw here would be invisible.
        """

        holder_type = type(holder)
        if holder_type not in self._rng_override_cache:
            self._rng_override_cache[holder_type] = _untrusted_rng_override(holder_type)
        override = self._rng_override_cache[holder_type]
        if override is not None:
            self._flag_uncertain(
                "rng_subclass_override_unwitnessable:"
                f"{holder_type.__module__}.{holder_type.__qualname__}.{override}"
            )
        return self._digest_rng_instance(holder)

    @staticmethod
    def _seed_seq_spawn_fragment(bit_generator: Any) -> str:
        """Digest the spawn-relevant SeedSequence state behind one BitGenerator.

        ``Generator.spawn()`` / ``BitGenerator.spawn()`` advance NO sampled
        state: they mutate ``seed_seq._n_children_spawned``, which is
        verdict-steering hidden state (a fresh oracle-1 run spawns a
        differently-keyed child). Folding the spawn counter plus the seeding
        identity into the digest makes an in-window spawn on a digest-rooted
        engine witnessable (grind-r5 b8 R57 HIGH). Absent/opaque seed
        sequences digest to the empty fragment (nothing spawnable to hide).
        """

        seed_seq = getattr(bit_generator, "seed_seq", None)
        if seed_seq is None:
            seed_seq = getattr(bit_generator, "_seed_seq", None)
        if seed_seq is None:
            return ""
        return host_nondeterminism_monitor._seed_seq_state_fragment(seed_seq)

    @staticmethod
    def _seed_seq_state_fragment(seed_seq: Any) -> str:
        """Digest one SeedSequence's seeding identity and spawn counter."""

        return host_nondeterminism_monitor._exact_state_repr(
            (
                "seed_seq",
                getattr(seed_seq, "entropy", None),
                getattr(seed_seq, "spawn_key", None),
                getattr(seed_seq, "pool_size", None),
                getattr(seed_seq, "n_children_spawned", None),
            )
        )

    @staticmethod
    def _digest_rng_instance(holder: Any) -> str:
        """Return a comparable state digest for one RNG holder.

        Covers numpy ``Generator``/``RandomState``/bare ``BitGenerator``,
        bare ``SeedSequence`` holders, ``torch.Generator`` (state bytes plus
        device identity), and ``random.Random``. Generator and
        BitGenerator digests fold in the underlying SeedSequence spawn state
        so ``spawn()`` -- which advances no sampled state -- is witnessed. A
        stateless ``Random`` subclass whose ``getstate()``
        raises ``NotImplementedError`` (``SystemRandom``) is classified
        monitored-not-digestible rather than an inventory error: possessing an
        undrawn stateless engine is not nondeterminism.
        """

        exact = host_nondeterminism_monitor._exact_state_repr
        spawn_fragment = host_nondeterminism_monitor._seed_seq_spawn_fragment
        if isinstance(holder, np.random.Generator):
            bit_generator = holder.bit_generator
            return exact(bit_generator.state) + spawn_fragment(bit_generator)
        if isinstance(holder, np.random.RandomState):
            return exact(holder.get_state())
        # r41 (review): a BARE model-held BitGenerator (``self.bg = PCG64(...)`` drawn
        # through a wrapping Generator) advances its own ``state``; digest it directly
        # so the registry's BitGenerator claim is digest-true.
        if isinstance(holder, np.random.BitGenerator):
            return exact(holder.state) + spawn_fragment(holder)
        # grind-r5 b8 R57: a model-held bare ``SeedSequence`` is a spawnable
        # entropy root; ``seed_seq.spawn()`` mid-window is the same hidden
        # verdict-steering mutation as ``Generator.spawn()``.
        if isinstance(holder, np.random.SeedSequence):
            return host_nondeterminism_monitor._seed_seq_state_fragment(holder)
        if isinstance(holder, torch.Generator):
            # r7 b8-sol R57: a model-held ``torch.Generator`` drawn on a
            # pre-existing (non-hooked) thread advanced state with NO witness
            # while the numpy analog was digest-caught, so the residual
            # enumeration's "only an EXTERNALLY-HELD generator" claim was
            # false. Digest the exact state bytes plus device identity so the
            # before/after sweeps witness any draw thread-independently. A
            # state-read failure propagates to the fail-closed inventory
            # error path, downgrading completeness rather than reading clean.
            state_bytes = holder.get_state().cpu().numpy().tobytes()
            return exact(("torch.Generator", str(holder.device), state_bytes))
        if isinstance(holder, random.Random):
            try:
                state = holder.getstate()
            except NotImplementedError:
                # r55 corr_1: ``random.SystemRandom`` (and any stateless ``Random``
                # subclass following its documented protocol) INTENTIONALLY has no
                # digestible state -- ``getstate()`` raises ``NotImplementedError``
                # by design. Possession of an UNDRAWN stateless engine is not
                # nondeterminism: classify monitored-not-digestible (the sweep then
                # walks it structurally like any holder) instead of letting the
                # generic inventory error path over-trigger
                # ``inventory_state_read_failed`` on a deterministic model. Actual
                # draws stay witnessed by the class-method patches on
                # ``random.SystemRandom.{random,getrandbits,randbytes}``. Any OTHER
                # exception from ``getstate()`` (a genuinely broken state read)
                # still propagates to the fail-closed inventory error path.
                raise _NotADigestableRng from None
            return host_nondeterminism_monitor._exact_state_repr(state)
        raise _NotADigestableRng

    @staticmethod
    def _module_namespace_walk_eligible(module: Any) -> dict[str, Any] | None:
        """Return a loaded module's raw namespace when the B4 deep inventory may walk it.

        Eligibility is decided from RAW reads only (the base ``ModuleType`` getset and
        plain ``dict.get``), so a lazy-loading module ``__getattr__`` (PEP 562) never
        fires. Skipped namespaces: the three internal package roots
        (:data:`_NUMPY_FRAME_DIGEST_INTERNAL_PATH_PREFIXES` -- their held engines are
        exempt singletons or replayable state, and numpy's own namespace is
        implementation noise), and standard-library modules matched by BOTH name
        (``sys.stdlib_module_names``) and location (:data:`_STDLIB_PATH_PREFIX`), so a
        user module that merely SHADOWS a stdlib name stays walked. ``__main__`` is
        never skipped -- scripts and notebooks hold their generators there.
        """

        if not isinstance(module, ModuleType):
            return None
        try:
            namespace = vars(ModuleType)["__dict__"].__get__(module)
        except Exception:
            return None
        if not isinstance(namespace, dict):
            return None
        name = namespace.get("__name__")
        top = name.split(".", 1)[0] if isinstance(name, str) else ""
        filename = namespace.get("__file__")
        if top != "__main__" and top in _sys_module.stdlib_module_names:
            if not isinstance(filename, str) or filename.startswith(_STDLIB_PATH_PREFIX):
                return None
        if isinstance(filename, str) and filename.startswith(
            _types._NUMPY_FRAME_DIGEST_INTERNAL_PATH_PREFIXES
        ):
            return None
        return namespace

    @staticmethod
    def _deep_inventory_seeds_from(code: CodeType) -> bool:
        """Return whether a profiled frame's code may seed the B4 deep inventory.

        Stdlib-source frames (``contextlib``/``warnings``/``typing``/... invoked by
        capture machinery thousands of times per forward) and synthetic-source frames
        (``<string>`` dataclass shims, ``<eval_with_key>`` fx codegen) never seed it:
        no realistic numpy receiver chain is WRITTEN in stdlib source (stdlib
        ``random`` draws are class-patch witnessed and the global engines are
        replayable state), and exec'd-from-string user code is the documented
        exec-namespace residual. Without this gate the walk crawls the in-progress
        capture graph through stdlib helper frames' locals -- measured +230 ms on a
        400 ms capture with zero coverage gained. The per-frame one-edge digest keeps
        running for these frames unchanged.
        """

        filename = code.co_filename
        return not filename.startswith(_STDLIB_PATH_PREFIX) and not filename.startswith("<")

    def _deep_inventory_frame_reachable(self, candidates: tuple[Any, ...], code: CodeType) -> None:
        """Digest numpy RNG receivers deeply reachable from profiled-frame roots (B4).

        NumPy>=2 binds its RNG draw methods as profile-silent Cython callables, so a
        draw is witnessed ONLY by a before/after state digest of a known receiver.
        The per-frame digest reaches receivers the drawing frame names directly (plus
        one inert edge) and the model sweep reaches receivers the MODEL holds -- but a
        pre-existing generator held in a foreign module's namespace
        (``helpers.RNG.random()``), behind a nested plain holder
        (``HOLDER.inner.gen``), inside nested builtin containers, or as a direct user
        class attribute was reachable by user code yet witnessed by NEITHER belt: a
        provably wrong capture validated VERIFIED+ATTESTED (the B4 false-VERIFIED
        class). This inventory closes it: every receiver reachable from a profiled
        user frame's fast locals, named globals, or helper return values through
        exact builtin containers (subclasses read via the base C implementations, so
        a hostile override never executes), plain-object ``__dict__``/``__slots__``
        values (:meth:`_custom_holder_children` -- slot reads through the member
        descriptor, never ``getattr``), eligible module namespaces, and direct
        class-attribute values of non-trusted-leaf classes (raw mappingproxy reads;
        the descriptor protocol never fires on a ``values()`` read) is digested at
        FIRST REFERENCE and compared once at ``__exit__``; any net state change marks
        the ceiling channel ``frame_reachable_generator``. The compare itself is
        thread-independent: a pre-existing worker's draw from a receiver the OWNER's
        in-window code also reaches is witnessed.

        Frame-triggered (never an unconditional ``sys.modules`` sweep -- measured at
        ~150 ms/capture in a bare torch env, dominated by torch's own dependency
        stack) and window-memoized by root id, so a capture that never references an
        RNG-bearing namespace pays only set lookups. Nested MODULE values descend
        only when their binding name appears in the referencing code object's
        ``co_names`` (``pkg.sub.RNG`` names ``sub``), keeping package fan-out
        proportional to what the code can actually reach; a computed module attribute
        (``getattr(pkg, name)``) is a documented residual. Budget exhaustion flags
        ``deep_inventory_budget_exhausted`` (INCOMPLETE) -- never a silent partial
        snapshot.

        r38 (round-37 re-attack): the walk now has PARITY with the model-rooted
        sweep for the stdlib instance holders it used to leaf -- ``weakref.ref``
        referents (base-C deref), ``threading.local`` per-thread namespaces
        (``tp_traverse``, ALL threads' dicts), ``functools.partial`` interiors
        (base member descriptors), and ``deque`` buffers (base ``__iter__``) --
        each a previously executed frame-rooted false-VERIFIED (V6/V7/V8 +
        deque). An OPAQUE queue (``SimpleQueue`` / ``mp.Queue``) reachable from a
        frame and not non-mutatingly provably empty fails CLOSED
        (``inventory_opaque_container``): an unreadable C-internal holder is a
        typed INCOMPLETE, never a silent leaf. Dynamic-name namespace subscripts
        (``globals()["name"]``) are witnessed by resolving the code object's
        string CONSTANTS as global roots (:meth:`_snapshot_numpy_frame_rngs`).
        r39 (round-39 re-attack): six more executed C-holder false-VERIFIEDs closed
        by bounded parity branches -- weakref PROXIES (fail-closed; no inert deref),
        warm ``functools.lru_cache`` wrappers and ``MappingProxyType`` (both via
        ``tp_traverse``), owner-thread ``ContextVar`` values (base C ``get``),
        OBJECT-dtype ndarray elements (base getsets, budget-gated), and class-var
        resolution through the MRO and a user METACLASS (parity with the model
        sweep). These are STOPGAPS for found shapes, not closure: a generator
        reachable ONLY through a C-implemented holder whose accessor path contains
        no Python frame remains structurally outside this witness's scope on
        numpy>=2 + CPython<3.12 (contract s11; the sys.monitoring/PEP-669 receiver
        classifier is the py>=3.12 architectural closure, under owner review).

        Remaining documented residuals (contract s11): receivers in namespaces no
        in-window frame references, stdlib/internal-package namespace stashes,
        function-attribute holders, COMPUTED (non-constant) dynamic names, the
        self-cleaning draw+state-restore, and an externally-held generator drawn
        only on a pre-existing non-hooked thread (reachable from NO digest root).

        Parameters
        ----------
        candidates:
            Frame-visible root values (fast locals, resolved named globals, or a
            helper return value).
        code:
            Code object of the referencing frame (its ``co_names`` guide nested
            module descent).
        """

        if self._deep_inventory_exhausted:
            return
        snapshots = self._deep_generator_states
        seen_ids = self._deep_walk_seen_ids
        visited_nodes = self._deep_inventory_visited
        try:
            pending = [
                value
                for value in candidates
                if type(value) not in _INERT_PRIMITIVE_LEAF_TYPES and id(value) not in seen_ids
            ]
            if not pending:
                return
            co_names: frozenset[str] | None = None
            while pending:
                value = pending.pop()
                value_type = type(value)
                if value_type in _INERT_PRIMITIVE_LEAF_TYPES or id(value) in seen_ids:
                    continue
                seen_ids.add(id(value))
                visited_nodes += 1
                if visited_nodes > _types._DEEP_INVENTORY_NODE_CAP:
                    self._deep_inventory_exhausted = True
                    self._flag_uncertain("deep_inventory_budget_exhausted")
                    return
                # r39 (executed false-VERIFIED V9a): a weakref PROXY forwards EVERY
                # read -- including ``isinstance`` (via ``__class__``) and attribute
                # access -- through its referent's own attribute machinery, and NO
                # inert deref exists (see :data:`_WEAKREF_PROXY_TYPES`). Match the
                # EXACT type BEFORE any isinstance branch below can be spoofed into a
                # forwarding read, and fail CLOSED: an unreadable C holder is a typed
                # INCOMPLETE, never a silent leaf.
                if value_type in _WEAKREF_PROXY_TYPES:
                    self._flag_uncertain("inventory_opaque_container")
                    continue
                if isinstance(value, _NUMPY_RNG_INSTANCE_TYPES):
                    if id(value) in self._exempt_ids:
                        continue
                    try:
                        snapshots.append((value, self._digest_rng_witnessable(value)))
                    except Exception:
                        self._flag_uncertain("inventory_state_read_failed")
                    continue
                if value_type in (BuiltinFunctionType, MethodType):
                    receiver = _numpy_rng_receiver(value)
                    if (
                        receiver is not None
                        and id(receiver) not in seen_ids
                        and id(receiver) not in self._exempt_ids
                    ):
                        seen_ids.add(id(receiver))
                        try:
                            snapshots.append((receiver, self._digest_rng_witnessable(receiver)))
                        except Exception:
                            self._flag_uncertain("inventory_state_read_failed")
                    continue
                # Container SUBCLASSES read through the base C implementations, so a
                # hostile override never executes (a ``UserDict``'s ``.data`` arrives
                # via the recursable-holder branch instead).
                if isinstance(value, dict):
                    pending.extend(dict.keys(value))
                    pending.extend(dict.values(value))
                    if value_type is not dict and self._is_recursable_custom_holder(value):
                        pending.extend(self._custom_holder_children(value))
                    continue
                if isinstance(value, (list, tuple)):
                    pending.extend(
                        list.__iter__(value) if isinstance(value, list) else tuple.__iter__(value)
                    )
                    continue
                if isinstance(value, (set, frozenset)):
                    pending.extend(
                        set.__iter__(value) if isinstance(value, set) else frozenset.__iter__(value)
                    )
                    continue
                # r38 parity branches: stdlib INSTANCE holders the MODEL-rooted sweep
                # already walks but this frame walk leafed -- ``weakref.ref`` referents,
                # ``threading.local`` per-thread namespaces, ``functools.partial``
                # interiors, and ``deque`` buffers. Each was an executed frame-rooted
                # false-VERIFIED (round 37: V6/V7/V8 + deque); every read below is
                # base-C (deref / ``tp_traverse`` / member descriptor / base
                # ``__iter__``), so no user code can fire.
                if isinstance(value, _weakref_module.ref):
                    # Mirror of the model sweep's r53 corr_2 branch: ONE base-C deref
                    # (immune to a hostile subclass ``__call__`` override); a dead ref
                    # contributes nothing; a deref failure fails CLOSED, never reads
                    # as no-referent. A SUBCLASS also walks its own attribute surface.
                    try:
                        referent = _weakref_module.ref.__call__(value)
                    except Exception:
                        self._flag_uncertain("inventory_state_read_failed")
                        continue
                    if referent is not None:
                        pending.append(referent)
                    if type(value) is not _weakref_module.ref:
                        pending.extend(self._custom_holder_children(value))
                        pending.append(value_type)
                    continue
                if isinstance(value, _c_thread_module._local):
                    # ``tp_traverse`` of a ``threading.local`` exposes its class plus
                    # EVERY thread's per-thread attribute dict (pure C -- no
                    # ``__getattribute__`` / property can fire), so ``TLS.gen`` set by
                    # any thread joins the digest, not just the walking thread's view.
                    pending.extend(_gc_module.get_referents(value))
                    continue
                if isinstance(value, _functools_module.partial):
                    # ``func`` / ``args`` / ``keywords`` are C slots (no ``__dict__``
                    # entry); read them through the BASE member descriptors so a
                    # subclass shadow never executes. The ``func`` edge recovers a
                    # ``partial(gen.random)`` receiver via the bound-callable branch.
                    for descriptor in _PARTIAL_SLOT_DESCRIPTORS:
                        try:
                            pending.append(descriptor.__get__(value, value_type))
                        except Exception:
                            self._flag_uncertain("inventory_state_read_failed")
                    if type(value) is not _functools_module.partial:
                        pending.extend(self._custom_holder_children(value))
                        pending.append(value_type)
                    continue
                if isinstance(value, _collections_module.deque):
                    # Base-C iteration, symmetric with the set/frozenset reads above:
                    # read-only ``deque.__iter__`` never mutates the buffer and never
                    # dispatches to a subclass override.
                    pending.extend(_collections_module.deque.__iter__(value))
                    if type(value) is not _collections_module.deque:
                        pending.extend(self._custom_holder_children(value))
                        pending.append(value_type)
                    continue
                # r39 C-holder parity branches (executed false-VERIFIEDs V9b/V9c/V9e/
                # V9f): a generator reached ONLY through a C-implemented holder with a
                # C accessor path never enters a Python frame. Every read below is
                # base-C (``tp_traverse`` / base getset / C ``Context`` lookup).
                if _LRU_CACHE_WRAPPER_TYPE is not None and isinstance(
                    value, _LRU_CACHE_WRAPPER_TYPE
                ):
                    # A cache WARMED pre-capture returns its cached generator with no
                    # Python frame; ``tp_traverse`` exposes the cache dict (walked by
                    # the dict branch above) plus the wrapped function (callable leaf).
                    pending.extend(_gc_module.get_referents(value))
                    continue
                if value_type is MappingProxyType:
                    # ``tp_traverse`` yields the BACKING mapping object without
                    # invoking its (possibly user-defined) ``keys``/``values``; the
                    # walk then descends it through its own typed branch.
                    pending.extend(_gc_module.get_referents(value))
                    continue
                if value_type is _contextvars_module.ContextVar:
                    # The VALUE lives in the per-thread ``Context``, off the reference
                    # graph (``tp_traverse`` does not expose it); the base C ``get``
                    # (set value or declared default, owner thread) is the only inert
                    # edge. Pre-existing OTHER threads' contexts stay the documented
                    # foreign-thread residual.
                    try:
                        pending.append(_contextvars_module.ContextVar.get(value))
                    except LookupError:
                        pass
                    continue
                if isinstance(value, np.ndarray):
                    # An OBJECT-dtype ndarray holds ordinary Python references
                    # (numpy's own parallel-streams idiom: arrays of spawned
                    # Generators) and ``ARR[0]`` is a profile-silent C subscript.
                    # Element-iterate ONLY object dtypes, through the base getsets so
                    # a subclass property never fires; non-object dtypes stay hard
                    # leaves. An over-cap object array exhausts the budget FIRST
                    # (INCOMPLETE, fail-closed), never a silent partial read.
                    try:
                        dtype = np.ndarray.dtype.__get__(value)
                    except Exception:
                        self._flag_uncertain("inventory_state_read_failed")
                        continue
                    if dtype.hasobject:
                        try:
                            size = int(np.ndarray.size.__get__(value))
                        except Exception:
                            self._flag_uncertain("inventory_state_read_failed")
                            continue
                        if visited_nodes + size > _types._DEEP_INVENTORY_NODE_CAP:
                            self._deep_inventory_exhausted = True
                            self._flag_uncertain("deep_inventory_budget_exhausted")
                            return
                        try:
                            pending.extend(np.ndarray.flat.__get__(value))
                        except Exception:
                            self._flag_uncertain("inventory_state_read_failed")
                    continue
                if isinstance(value, ModuleType):
                    namespace = self._module_namespace_walk_eligible(value)
                    if namespace is None:
                        continue
                    if co_names is None:
                        co_names = frozenset(code.co_names)
                    for name, member in namespace.items():
                        # A nested module descends only when the referencing code can
                        # actually reach it by name; everything else walks normally.
                        if isinstance(member, ModuleType) and name not in co_names:
                            continue
                        pending.append(member)
                    continue
                if isinstance(value, type):
                    # r39 (executed false-VERIFIEDs V9c/V9d): ``CLS.gen`` resolves
                    # through ``CLS.__mro__`` AND ``type(CLS).__mro__`` entirely in C,
                    # and -- unlike the model sweep -- this walk never expands a shared
                    # module namespace, so a BASE class or user METACLASS is NOT
                    # otherwise reachable (the r38 "reachable through its own defining
                    # namespace" rationale was false for the frame walk). Cascade the
                    # MRO and enqueue the metaclass, matching the model sweep; both
                    # are dedup-bounded, and ``type`` itself plus stdlib/torch/numpy
                    # classes stay trusted leaves on their own visit.
                    pending.append(value_type)  # the metaclass (``type(value)``)
                    if self._is_trusted_leaf_class(value):
                        continue
                    try:
                        raw_dict = type.__dict__["__dict__"].__get__(value)
                        mro = type.__dict__["__mro__"].__get__(value)
                    except Exception:
                        continue
                    pending.extend(
                        attr
                        for attr in raw_dict.values()
                        if self._module_namespace_class_attr_eligible(attr)
                    )
                    pending.extend(base for base in mro if base is not value)
                    continue
                if value_type is SimpleNamespace:
                    # ``types.SimpleNamespace`` is the idiomatic config-object shape but
                    # its type roots in the stdlib leaf set, so the holder gate below
                    # would leaf it; its plain ``__dict__`` is inert to read.
                    pending.extend(object.__getattribute__(value, "__dict__").values())
                    continue
                if isinstance(value, (FunctionType, MethodWrapperType)):
                    # Callable leaf (documented residual: function-ATTRIBUTE-held
                    # receivers). RNG-bound callables were already extracted above.
                    continue
                if self._is_recursable_custom_holder(value):
                    pending.extend(self._custom_holder_children(value))
                    # An instance's CLASS is a holder surface too (dynamically created
                    # classes are in no namespace); dedup keeps this a single filtered
                    # dict read per distinct class.
                    pending.append(value_type)
                    continue
                if self._exposes_queue_protocol(value):
                    # r38 mirror of the model sweep's r45/r47 queue posture: an
                    # inspectable queue (``queue.Queue`` family) was already walked as
                    # a recursable holder above (its ``.queue`` deque descends via the
                    # deque branch); an OPAQUE queue (``SimpleQueue`` / ``mp.Queue`` --
                    # no readable buffer, ``get`` would drain) that is not
                    # non-mutatingly provably empty fails CLOSED. A frame-reachable
                    # queue-held generator is a typed INCOMPLETE, never a silent leaf.
                    with self._monitor_internal_probe():
                        provably_empty = self._opaque_queue_provably_empty(value)
                    if not provably_empty:
                        self._flag_uncertain("inventory_opaque_container")
        except Exception:
            self._flag_uncertain("inventory_scan_failed")
        finally:
            self._deep_inventory_visited = visited_nodes

    @staticmethod
    def _module_namespace_class_attr_eligible(value: Any) -> bool:
        """Return whether a raw class-dict value can be or inertly hold a receiver.

        Methods, descriptors, properties, nested classes, and module references are
        implementation noise on the class surface (nested classes and modules are
        reached through their own roots); receivers, RNG-bound callables, exact
        builtin containers, and plain holder objects stay walked.
        """

        value_type = type(value)
        if value_type in _INERT_PRIMITIVE_LEAF_TYPES:
            return False
        if isinstance(value, _NUMPY_RNG_INSTANCE_TYPES) or _numpy_rng_receiver(value) is not None:
            return True
        if value_type in (dict, list, tuple, set, frozenset):
            return True
        return not isinstance(
            value,
            (
                type,
                ModuleType,
                FunctionType,
                BuiltinFunctionType,
                MethodType,
                MethodWrapperType,
                staticmethod,
                classmethod,
                property,
                GetSetDescriptorType,
            ),
        )

    @staticmethod
    def _class_attr_surface(klass: type) -> tuple[Any, ...]:
        """Eligible raw class-dict values across ``klass``'s MRO plus its metaclass MRO (r39).

        ``CLS.gen`` resolves through ``type(CLS).__mro__`` and then ``CLS.__mro__``
        entirely in C -- no Python frame -- so a generator stored as a BASE-class or
        METACLASS class-var steers user code with no profile event (executed round-39
        false-VERIFIEDs V9c/V9d). All reads go through the base ``type`` getsets (a
        hostile metaclass property on ``__dict__``/``__mro__`` never fires); trusted
        stdlib/torch/numpy classes and ``type`` itself contribute nothing, so the cost
        is bounded by the referencing code's USER classes only.
        """

        surface: list[Any] = []
        seen: set[int] = set()
        stack: list[type] = [klass]
        while stack:
            current = stack.pop()
            if id(current) in seen or not isinstance(current, type):
                continue
            seen.add(id(current))
            stack.append(type(current))
            if host_nondeterminism_monitor._is_trusted_leaf_class(current):
                continue
            try:
                raw_dict = type.__dict__["__dict__"].__get__(current)
                mro = type.__dict__["__mro__"].__get__(current)
            except Exception:
                continue
            surface.extend(
                attr
                for attr in raw_dict.values()
                if host_nondeterminism_monitor._module_namespace_class_attr_eligible(attr)
            )
            stack.extend(base for base in mro if base is not current)
        return tuple(surface)

    # -- context protocol ---------------------------------------------------------

    def _install_steps(self) -> tuple[tuple[str, Callable[[], None]], ...]:
        """Return the ordered install steps, each guarded INDEPENDENTLY by ``__enter__``.

        One raising surface used to abort every later surface (a single hostile model
        attribute cost the whole profile belt), which is a silent under-witness rather
        than a fail-closed degradation: the failed step flags uncertainty and the rest
        still install. Profile hooks stay LAST so the classifier never observes a
        half-patched surface set.
        """

        return (
            ("prologue", self._install_prologue),
            ("rng_primitive", self._install_python_rng_primitives),
            ("global_engine_state", self._install_global_engine_state_surfaces),
            ("owner_sync", lambda: channels.install_owner_sync_surfaces(self)),
            ("entropy", self._install_entropy_surfaces),
            ("construction", self._install_construction_surfaces),
            ("clock", self._install_clock_surfaces),
            ("torch_rng", self._install_torch_rng_surfaces),
            ("generator_belt", self._install_generator_belt),
            ("profile_hooks", self._install_profile_hooks),
        )

    def __enter__(self) -> HostRngMonitorResult:
        if self._entered:
            # Re-arming the SAME instance would snapshot its own patches as the
            # "prior" state and hand a torn-down window's hook back to the process at
            # exit. The window is already live: degrade completeness, install nothing.
            self._flag_uncertain("monitor_reentered")
            return self.result
        self._entered = True
        if monitor_state._ACTIVE_MONITOR is not None:
            # Overlapping windows are outside the single-threaded capture model: this
            # window's ``_patch_attr`` snapshots the OUTER window's wrappers as its
            # "originals", so a non-LIFO unwind cannot prove exact restoration. Degrade
            # completeness (the capture ceilings) rather than claim a clean window.
            self._flag_uncertain("monitor_overlap")
        # BEFORE any patch installs: force torch's lazy torch._compile /
        # torch._dynamo import cascade (first wrapped op of a selective
        # runnable-capable capture) to draw its module-exec entropy
        # (uuid.uuid4/getrandbits) OUTSIDE the window. In-window it marked
        # os.urandom channels and permanently ceilinged a pure deterministic
        # model's first runnable artifact to UNVERIFIABLE. A failed warm is
        # benign: the in-window retry's draws are then honestly marked.
        # A different hazard, same fix, for the other imports a first capture
        # triggers (torch's ``torch.backends.opt_einsum`` inside ``torch.einsum``,
        # warmed by it, and TorchLens's observability session reader on the first
        # wrapped op): an in-window import runs the ``sys.meta_path`` finders'
        # frames, which the frame-reachable inventory walks as roots, so a large
        # graph behind a finder (pytest's assertion rewriter) exhausted the budget.
        try:
            warm_lazy_torch_imports()  # first, so a failed import below cannot skip it
            from .._state import pause_logging
            from ..backends.torch._op_markers import resolve_active_session_fn

            with pause_logging():
                resolve_active_session_fn()
        except Exception:
            pass
        try:
            for step_name, step in self._install_steps():
                try:
                    step()
                except Exception:
                    self._flag_uncertain(f"monitor_install_failed:{step_name}")
        except BaseException:
            # Python never calls ``__exit__`` when ``__enter__`` raises, so a
            # BaseException (a Ctrl-C landing in the O(model-size) generator sweep,
            # a thread kill) would leave ~40 process-wide patches and both profile
            # hooks installed FOREVER with no owner -- and the next window would
            # snapshot the leaked wrapper as its own "original", stacking the leak
            # monotonically. Unwind before re-raising.
            self._flag_uncertain("monitor_install_interrupted")
            self._teardown()
            raise
        # r41 hon2_1: publish the in-window registry LAST so the escape belt never
        # observes a partially-installed window (restored FIRST in the teardown).
        self._previous_active_monitor = monitor_state._ACTIVE_MONITOR
        monitor_state._ACTIVE_MONITOR = self
        return self.result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._teardown()

    def _teardown(self) -> None:
        """Unwind everything :meth:`__enter__` installed, exactly once.

        Idempotent by contract: a second call (an ``ExitStack`` double-close, a manual
        ``__exit__`` after a ``with``) used to clobber a LATER monitor's -- or a
        debugger's -- profile hook and uninstall a live window's wrappers mid-forward,
        so every repeat is a no-op. Also called from ``__enter__``'s BaseException arm.
        """
        if self._torn_down:
            return
        self._torn_down = True
        # grind-p3 T11.8: ``_torn_down`` latches BEFORE the unwind runs, so a
        # BaseException escaping mid-teardown (a Ctrl-C landing in the registry
        # or profile-hook steps, or inside one patch restore -- the per-restore
        # guard caught only ``Exception``) used to strand the WHOLE remaining
        # restore queue forever: the retry no-opped on the latch, ~40
        # process-wide patches leaked, and the next window snapshotted the
        # leaked wrappers as its own originals, stacking monotonically. Every
        # teardown stage is now BaseException-isolated so the restore queue
        # ALWAYS drains; the first BaseException re-raises only after the
        # unwind completes (Ctrl-C semantics are preserved, never swallowed).
        deferred: BaseException | None = None
        try:
            # Restore the PREVIOUS registry entry instead of clearing unconditionally: a
            # nested monitor's exit used to null the slot mid-outer-window, after which
            # TorchLens's own per-op RNG restores marked a false ``mutation`` channel and
            # every in-window thread reclassified as foreign.
            if monitor_state._ACTIVE_MONITOR is self or monitor_state._ACTIVE_MONITOR is None:
                # Skip torn-down ancestors: a non-LIFO overlap otherwise parks a
                # dead monitor in the module slot and every later capture flags a
                # phantom overlap (same dead-chain class as the profile slots).
                candidate = self._previous_active_monitor
                seen: set[int] = set()
                while (
                    candidate is not None
                    and id(candidate) not in seen
                    and getattr(candidate, "_torn_down", False)
                ):
                    seen.add(id(candidate))
                    candidate = candidate._previous_active_monitor
                monitor_state._ACTIVE_MONITOR = candidate
            else:
                self._flag_uncertain("active_monitor_replaced")
        except BaseException as exc:  # noqa: BLE001 -- drain first, re-raise after
            self._flag_uncertain("teardown_interrupted")
            deferred = exc
        try:
            self._restore_profile_hooks()
        except BaseException as exc:  # noqa: BLE001 -- drain first, re-raise after
            self._flag_uncertain("teardown_interrupted")
            if deferred is None:
                deferred = exc
        self._hooks_retired = True
        for restore in reversed(self._restores):
            try:
                restore()
            except Exception:
                self._flag_uncertain("patch_restore_failed")
            except BaseException as exc:  # noqa: BLE001 -- keep draining the queue
                self._flag_uncertain("patch_restore_interrupted")
                if deferred is None:
                    deferred = exc
        self._restores.clear()
        try:
            self._compare_generator_inventories()
            channels.settle_foreign_reads(self)
        except BaseException as exc:  # noqa: BLE001 -- unwind already complete
            self._flag_uncertain("teardown_interrupted")
            if deferred is None:
                deferred = exc
        # A thread parked in an in-window wait wrapper, or still holding a retired profile
        # hook, keeps this monitor reachable; past the settled verdict it must not pin the model.
        self._model = None
        self._generator_states, self._deep_generator_states = [], []
        self._numpy_frame_rng_states = {}
        if deferred is not None:
            raise deferred

    def _compare_generator_inventories(self) -> None:
        """Unwind stage: mark any inventoried generator whose state digest moved in-window."""

        for holder, before in self._generator_states:
            try:
                if self._digest_rng_witnessable(holder) != before:
                    self._mark("model_attribute_generator")
            except Exception:
                self._flag_uncertain("inventory_compare_failed")
        for holder, before in self._deep_generator_states:
            try:
                if self._digest_rng_witnessable(holder) != before:
                    self._mark("frame_reachable_generator")
            except Exception:
                self._flag_uncertain("inventory_compare_failed")

    def _restore_profile_hooks(self) -> None:
        """Hand the profile slots back, never overwriting a hook that is not ours.

        A non-LIFO overlap (monitor A exiting inside monitor B's window) used to
        install A's saved predecessor OVER B's live hook, permanently destroying it.
        When the slot no longer holds our hook we flag uncertainty and LEAVE it --
        our hook is already gone, and whoever replaced it owns the slot. Threads that
        still carry a torn-down hook self-uninstall on their next profile event (see
        :meth:`_make_profile_hook`), which is also what un-strands the in-window
        worker threads ``threading.setprofile`` cannot reach.
        """

        if self._threading_profile_installed:
            try:
                if (
                    hasattr(_threading_module, "getprofile")
                    and _threading_module.getprofile() is not self._threading_hook
                ):
                    self._flag_uncertain("threading_profile_replaced")
                else:
                    # Held original: the swap-detection wrapper is still on
                    # the module attr at this point (its _patch_attr restore
                    # unwinds after this method returns).
                    setter = self._orig_threading_setprofile or _threading_module.setprofile
                    setter(
                        _skip_retired_hooks(
                            self._previous_threading_profile, "_previous_threading_profile"
                        )
                    )
            except Exception:
                self._flag_uncertain("threading_profile_restore_failed")
        if self._sys_profile_installed:
            try:
                if _sys_module.getprofile() is not self._sys_hook:
                    self._flag_uncertain("sys_profile_replaced")
                else:
                    setter = self._orig_sys_setprofile or _sys_module.setprofile
                    setter(_skip_retired_hooks(self._previous_sys_profile, "_previous_sys_profile"))
            except Exception:
                self._flag_uncertain("sys_profile_restore_failed")

    def _install_prologue(self) -> None:
        """Resolve the identity exemption sets consulted by every classifier."""

        self._tl_globals_ids = _torchlens_module_globals_ids()
        self._exempt_ids = frozenset(id(item) for item in _rng_exempt_instances())
        self._global_engine_prefixes = engine_state.global_engine_state_prefixes()
        # W051 (2.16): owner-thread sync primitives ride the held-code ``call`` layer.
        self._held_code_marks.update(channels.owner_sync_code_marks())

    def _install_python_rng_primitives(self) -> None:
        """Patch the Python RNG class primitives (instances, subclasses, the bare C base)."""

        # rng_primitive: Python RNG class primitives (instances + subclasses +
        # the bare C base for ``_random.Random()``).
        for holder in (random.Random, random.SystemRandom, _c_random_module.Random):
            for method_name in ("random", "getrandbits", "randbytes"):
                if method_name in vars(holder):
                    self._patch_attr(
                        holder,
                        method_name,
                        self._instance_method_wrapper(
                            getattr(holder, method_name),
                            f"{holder.__module__}.{holder.__qualname__}.{method_name}",
                        ),
                    )

    def _install_global_engine_state_surfaces(self) -> None:
        """Patch the Python / legacy-NumPy global-engine STATE surface (W051 2.17)."""

        engine_state.install_global_engine_state_surfaces(self)

    def _install_entropy_surfaces(self) -> None:
        """Patch the OS-entropy funnels (``os``/``random._urandom``) with held-ref registration."""

        # entropy: OS entropy + the secrets funnel alias + uuid4's feed. Each
        # original is identity-registered BEFORE patching (r41 held-ref layer).
        self._register_held_ref(_os_module.urandom, "os.urandom")
        self._patch_attr(
            _os_module, "urandom", self._entropy_wrapper(_os_module.urandom, "os.urandom")
        )
        if hasattr(_os_module, "getrandom"):
            self._register_held_ref(_os_module.getrandom, "os.getrandom")
            self._patch_attr(
                _os_module,
                "getrandom",
                self._entropy_wrapper(_os_module.getrandom, "os.getrandom"),
            )
        if hasattr(random, "_urandom"):
            self._register_held_ref(random._urandom, "random._urandom")
            self._patch_attr(
                random,
                "_urandom",
                self._entropy_wrapper(random._urandom, "random._urandom"),
            )
        # entropy: uuid1's platform C funnels. On Linux ``uuid.uuid1`` resolves
        # ``uuid._generate_time_safe`` (libuuid: wall clock + clock-seq entropy
        # + node) and touches NO other monitored surface, so an in-window
        # ``uuid.uuid1()`` was a clean false-VERIFIED escape; the Python
        # fallback path IS caught through getrandbits/clocks. Windows routes
        # through ``uuid._UuidCreate``. ``uuid.uuid1`` reads these as module
        # globals at call time, so a pre-window ``from uuid import uuid1``
        # alias cannot bypass the patch.
        for uuid_funnel_name in ("_generate_time_safe", "_UuidCreate"):
            uuid_funnel = getattr(_uuid_module, uuid_funnel_name, None)
            if uuid_funnel is None or not callable(uuid_funnel):
                continue
            self._register_held_ref(uuid_funnel, "uuid.uuid1")
            self._patch_attr(
                _uuid_module,
                uuid_funnel_name,
                self._entropy_wrapper(uuid_funnel, "uuid.uuid1"),
            )

    def _install_construction_surfaces(self) -> None:
        """Patch the NumPy generator factory and the unseeded-construction entropy alias."""

        # construction: the modern NumPy generator factory + the writable
        # construction-entropy alias for unseeded BitGenerator construction (E5).
        self._register_held_ref(np.random.default_rng, "np.random.default_rng")
        self._patch_attr(
            np.random,
            "default_rng",
            self._entropy_wrapper(np.random.default_rng, "np.random.default_rng"),
        )
        bit_generator_module = getattr(np.random, "bit_generator", None)
        if bit_generator_module is not None and hasattr(bit_generator_module, "randbits"):
            self._register_held_ref(bit_generator_module.randbits, "np_bit_generator_randbits")
            self._patch_attr(
                bit_generator_module,
                "randbits",
                self._entropy_wrapper(bit_generator_module.randbits, "np_bit_generator_randbits"),
            )

    def _install_clock_surfaces(self) -> None:
        """Patch the clock family and register the immutable ``datetime`` readers."""

        # clock: the frozen ``time.*`` readers (thread-independent module patches).
        for clock_name in _CLOCK_COUNTER_NAMES:
            if hasattr(_time_module, clock_name):
                self._register_held_ref(getattr(_time_module, clock_name), f"time.{clock_name}")
                self._patch_attr(
                    _time_module,
                    clock_name,
                    self._clock_wrapper(
                        getattr(_time_module, clock_name), f"time.{clock_name}", None
                    ),
                )
        for clock_name, time_arg_index in _CLOCK_IMPLICIT_NOW:
            if hasattr(_time_module, clock_name):
                self._register_held_ref(
                    getattr(_time_module, clock_name), f"time.{clock_name}", time_arg_index
                )
                self._patch_attr(
                    _time_module,
                    clock_name,
                    self._clock_wrapper(
                        getattr(_time_module, clock_name), f"time.{clock_name}", time_arg_index
                    ),
                )
        if hasattr(_os_module, "times"):
            self._register_held_ref(_os_module.times, "os.times")
            self._patch_attr(
                _os_module, "times", self._clock_wrapper(_os_module.times, "os.times", None)
            )
        if _resource_module is not None and hasattr(_resource_module, "getrusage"):
            self._register_held_ref(_resource_module.getrusage, "resource.getrusage")
            self._patch_attr(
                _resource_module,
                "getrusage",
                self._clock_wrapper(_resource_module.getrusage, "resource.getrusage", None),
            )
        # clock: immutable ``datetime`` current readers via c_call identity.
        for receiver, method in _DATETIME_CLOCK_READERS:
            if hasattr(receiver, method):
                self._clock_ccall_keys[(id(receiver), method)] = (
                    f"datetime.{receiver.__name__}.{method}"
                )

    def _install_torch_rng_surfaces(self) -> None:
        """Patch the frozen torch RNG surface table and seed the default-generator routing cache."""

        # r65 CLUSTER Z: torch RNG API module patches, derived from the ONE frozen
        # disposition table (entropy/mutation ceiling permanently; replayable_read
        # sets the consumed flag only). Each ORIGINAL is held-code registered
        # BEFORE its attribute is replaced, mirroring the r41 held-ref layer, so a
        # pre-window ``from torch import manual_seed`` alias cannot bypass.
        for surface_row in TORCH_RNG_SURFACE:
            if surface_row.disposition not in ("entropy", "mutation", "replayable_read"):
                continue
            module_path, _, attr_name = surface_row.target.rpartition(".")
            rng_holder_module = _torch_rng_holder_module(module_path)
            if rng_holder_module is None or not hasattr(rng_holder_module, attr_name):
                continue
            original = getattr(rng_holder_module, attr_name)
            self._register_held_code(original, surface_row.target, surface_row.disposition)
            self._patch_attr(
                rng_holder_module,
                attr_name,
                self._torch_rng_wrapper(original, surface_row.target, surface_row.disposition),
            )
        # r67 C1: seed the default-generator ROUTING CACHE for the all-receiver
        # c_call classifier (process default + every populated cuda/xpu/mtia
        # device default). Never forces device init, and never an honesty
        # boundary: the classifier re-resolves on any miss.
        self._default_generator_ids = _resolve_default_generator_ids()

    def _install_generator_belt(self) -> None:
        """Digest the generators the MODEL holds (the cheap thread-independent belt)."""

        # BELT (thread-independent, CHEAP -- model attributes + builtin-container
        # nesting): digest generators the model HOLDS, so a draw on ANY thread
        # (incl. a pre-existing worker) is caught by the before/after state
        # comparison at __exit__. Deliberately NOT a process-wide
        # ``gc.get_objects()`` scan -- the r39-draft GC-wide inventory cost
        # ~900 ms/capture, perturbed the tracemalloc peak, and could over-trigger
        # on unrelated generators (removed for cause). Realistic CROSS-THREAD
        # external draws (hon1_1/corr2_2) are caught by ``threading.setprofile``
        # below; an EXTERNALLY-held generator drawn on a PRE-EXISTING (non-hooked)
        # thread is the documented residual (contract s11), same class as the
        # adversarial draw+state-restore.
        self._generator_states = self._sweep_model_generators()

    def _install_profile_hooks(self) -> None:
        """Install the dual chained profile hooks -- ALWAYS the last step."""

        # Held pre-patch setprofile originals: every MONITOR-internal slot
        # write (teardown restore, raw-thread hook install, post-window hook
        # self-uninstall) routes through these so it can never trip the
        # swap-detection wrappers installed below -- including a LATER
        # window's wrappers on an overlapping monitor.
        self._orig_sys_setprofile = _sys_module.setprofile
        self._orig_threading_setprofile = getattr(_threading_module, "setprofile", None)
        # BELT: dual chained profile hooks (owner thread + threads started in-window). These
        # are the r37/base mechanism (base runs them and is fast); the owner hook catches
        # owner-thread numpy Generator instance draws and the immutable ``datetime`` readers,
        # the threading hook catches an in-window helper-thread draw (hon1_1/corr2_2) and
        # records each hooked thread's ident in the diagnostic registry (the r41
        # escape-belt 3-class consumer was deleted r43; owner/non-owner is binary now).
        self._previous_sys_profile = _sys_module.getprofile()
        self._sys_hook = self._make_profile_hook(self._previous_sys_profile)
        _sys_module.setprofile(self._sys_hook)
        self._sys_profile_installed = True
        self._previous_threading_profile = (
            _threading_module.getprofile() if hasattr(_threading_module, "getprofile") else None
        )
        self._threading_hook = self._make_profile_hook(
            self._previous_threading_profile, records_thread_ident=True
        )
        _threading_module.setprofile(self._threading_hook)
        self._threading_profile_installed = True
        # Raw ``_thread`` spawns bypass ``threading.setprofile`` entirely (that
        # hook rides ``threading.Thread``'s bootstrap), so an in-window
        # ``_thread.start_new_thread`` thread drawing an externally-held
        # generator was a clean false-VERIFIED escape -- outside the documented
        # residual, which covers only PRE-EXISTING threads. Patch the spawn
        # entry points to install this window's hook on the new thread before
        # the target runs; the hook self-uninstalls on its first event after
        # teardown, so a spawned thread outliving the window sheds it.
        # ``threading`` itself holds a pre-patch ``_start_new_thread`` ref, so
        # Thread starts are unaffected (no double hook).
        for spawn_name in ("start_new_thread", "start_joinable_thread"):
            if hasattr(_c_thread_module, spawn_name):
                self._patch_attr(
                    _c_thread_module,
                    spawn_name,
                    self._raw_thread_spawn_wrapper(getattr(_c_thread_module, spawn_name)),
                )
        # R57: a BALANCED in-window swap (user code saves our hook, installs
        # its own profile function, restores ours before window exit) left NO
        # teardown evidence -- the slot held our hook at exit -- so entropy/
        # clock draws inside the swapped sub-window escaped uncertain=False:
        # a false-VERIFIED (and, downstream, false-ATTESTED) window. Any
        # in-window slot write by non-monitor code makes the window
        # unprovable, so the setprofile entry points themselves are patched
        # to flag uncertainty and pass through. Monitor-internal writes use
        # the held originals above and never trip these. A pre-window
        # ``from sys import setprofile`` alias or a C-level
        # ``PyEval_SetProfile`` (cProfile.enable) bypasses the module attr
        # AND both endpoint identity checks (a balanced held-ref swap
        # restores our hook before teardown), re-opening the blind
        # sub-window for the profile-only channel class. No Python-level
        # fail-closed spelling exists for a pre-window held slot-writer;
        # both are DOCUMENTED residuals -- contract residual-tail row (vi)
        # in docs/reference/runnable_tlspec_contract.md -- until the
        # sys.monitoring (PEP 669) port -- interpreter-global,
        # slot-swap-immune -- closes them on py>=3.12 (grind-r5 b8 R57).
        self._patch_attr(
            _sys_module,
            "setprofile",
            self._profile_slot_swap_wrapper(self._orig_sys_setprofile, "sys.setprofile"),
        )
        if self._orig_threading_setprofile is not None:
            self._patch_attr(
                _threading_module,
                "setprofile",
                self._profile_slot_swap_wrapper(
                    self._orig_threading_setprofile, "threading.setprofile"
                ),
            )

    def _profile_slot_swap_wrapper(self, original: Any, channel: str) -> Any:
        """Build a flagging passthrough for an in-window profile-slot write.

        The write itself is honored untouched (the user's profiler works);
        the window's completeness degrades because draws made while a foreign
        profile function holds the slot are structurally unwitnessable.
        """

        def wrapper(function: Any) -> Any:
            """Flag the unprovable sub-window, then delegate the slot write."""

            if function is not None and (
                function is self._sys_hook or function is self._threading_hook
            ):
                # Installing THIS monitor's own hook is machinery, not a
                # swap: ``Thread._bootstrap_inner`` re-installs the window's
                # threading hook (``sys.setprofile(threading._profile_hook)``)
                # on every in-window thread start, and a user restoring our
                # hook closes, not opens, a blind window.
                return original(function)
            self._flag_uncertain(f"profile_slot_swapped_in_window:{channel}")
            return original(function)

        return wrapper
