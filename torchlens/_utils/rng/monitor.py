"""The host-nondeterminism monitor: the capture-window context manager.

Installs, from :data:`~torchlens._utils.rng.registry.HOST_NONDETERMINISM_REGISTRY`, the
module and class patches, profile hooks, model generator digests and frame-reachable
inventories that witness host entropy, clock and RNG-instance consumption during a
capture, and settles them into a :class:`~torchlens._utils.rng.channels.HostRngMonitorResult`.
"""

from __future__ import annotations

import _random as _c_random_module
import _thread as _c_thread_module
import datetime as _datetime_module
import os as _os_module
import random
import sys as _sys_module
import threading as _threading_module
import time as _time_module
import uuid as _uuid_module
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import (
    CodeType,
    FrameType,
)
from typing import Any, cast

import numpy as np
import torch

from .._torch_compat import (
    warm_lazy_torch_imports,
)
from . import (
    channels,
    deep_inventory,
    digest,
    engine_state,
    generator_sweep,
    inventory_types as _types,
    inventory_walk,
    monitor_state,
    numpy_frames,
)
from .call_sites import _call_site_time_arg_proof
from .channels import HostRngMonitorResult
from .inventory_types import (
    _NUMPY_RNG_INSTANCE_TYPES,
    _UNCERTAIN_DETAIL_CAP,
    _numpy_rng_receiver,
    _resource_module,
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

    # Moved method groups (phase 2 of the rng rebuild); the class keeps the call shapes.
    _is_recursable_custom_holder = staticmethod(inventory_walk._is_recursable_custom_holder)
    _custom_holder_children = staticmethod(inventory_walk._custom_holder_children)
    _is_trusted_leaf_class = staticmethod(inventory_walk._is_trusted_leaf_class)
    _shared_namespace_dict_ids = staticmethod(inventory_walk._shared_namespace_dict_ids)
    _inert_gc_children = staticmethod(inventory_walk._inert_gc_children)
    _exposes_queue_protocol = staticmethod(inventory_walk._exposes_queue_protocol)
    _opaque_queue_provably_empty = staticmethod(inventory_walk._opaque_queue_provably_empty)
    _module_namespace_walk_eligible = staticmethod(inventory_walk._module_namespace_walk_eligible)
    _deep_inventory_seeds_from = staticmethod(inventory_walk._deep_inventory_seeds_from)
    _module_namespace_class_attr_eligible = staticmethod(
        inventory_walk._module_namespace_class_attr_eligible
    )
    _class_attr_surface = staticmethod(inventory_walk._class_attr_surface)
    _exact_state_repr = staticmethod(digest._exact_state_repr)
    _seed_seq_spawn_fragment = staticmethod(digest._seed_seq_spawn_fragment)
    _seed_seq_state_fragment = staticmethod(digest._seed_seq_state_fragment)
    _digest_rng_instance = staticmethod(digest._digest_rng_instance)
    _numpy_frame_candidate_children = staticmethod(numpy_frames._numpy_frame_candidate_children)

    def _deep_inventory_frame_reachable(self, *args: Any, **kwargs: Any) -> Any:
        return deep_inventory._deep_inventory_frame_reachable(self, *args, **kwargs)

    def _sweep_model_generators(self, *args: Any, **kwargs: Any) -> Any:
        return generator_sweep._sweep_model_generators(self, *args, **kwargs)

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
        # warmed by it, and TorchLens's observability session reader, which the op markers
        # resolve on the first wrapped op): an in-window import runs the ``sys.meta_path`` finders'
        # frames, which the frame-reachable inventory walks as roots, so a large
        # graph behind a finder (pytest's assertion rewriter) exhausted the budget.
        try:
            warm_lazy_torch_imports()  # first, so a failed import below cannot skip it
            from ..._observability._session import active_session
            from ..._state import pause_logging

            with pause_logging():
                active_session()
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
