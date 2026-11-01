"""Frame-reachable deep inventory of RNG instances at capture entry."""

from __future__ import annotations

import _thread as _c_thread_module
import collections as _collections_module
import contextvars as _contextvars_module
import functools as _functools_module
import gc as _gc_module
import weakref as _weakref_module
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    MappingProxyType,
    MethodType,
    MethodWrapperType,
    ModuleType,
    SimpleNamespace,
)
from typing import TYPE_CHECKING, Any

import numpy as np

from . import inventory_types as _types
from .inventory_types import (
    _INERT_PRIMITIVE_LEAF_TYPES,
    _LRU_CACHE_WRAPPER_TYPE,
    _NUMPY_RNG_INSTANCE_TYPES,
    _PARTIAL_SLOT_DESCRIPTORS,
    _WEAKREF_PROXY_TYPES,
    _numpy_rng_receiver,
)

if TYPE_CHECKING:
    from .monitor import host_nondeterminism_monitor


def _deep_inventory_frame_reachable(
    monitor: host_nondeterminism_monitor, candidates: tuple[Any, ...], code: CodeType
) -> None:
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
    monitor-cleaning draw+state-restore, and an externally-held generator drawn
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

    if monitor._deep_inventory_exhausted:
        return
    snapshots = monitor._deep_generator_states
    seen_ids = monitor._deep_walk_seen_ids
    visited_nodes = monitor._deep_inventory_visited
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
                monitor._deep_inventory_exhausted = True
                monitor._flag_uncertain("deep_inventory_budget_exhausted")
                return
            # r39 (executed false-VERIFIED V9a): a weakref PROXY forwards EVERY
            # read -- including ``isinstance`` (via ``__class__``) and attribute
            # access -- through its referent's own attribute machinery, and NO
            # inert deref exists (see :data:`_WEAKREF_PROXY_TYPES`). Match the
            # EXACT type BEFORE any isinstance branch below can be spoofed into a
            # forwarding read, and fail CLOSED: an unreadable C holder is a typed
            # INCOMPLETE, never a silent leaf.
            if value_type in _WEAKREF_PROXY_TYPES:
                monitor._flag_uncertain("inventory_opaque_container")
                continue
            if isinstance(value, _NUMPY_RNG_INSTANCE_TYPES):
                if id(value) in monitor._exempt_ids:
                    continue
                try:
                    snapshots.append((value, monitor._digest_rng_witnessable(value)))
                except Exception:
                    monitor._flag_uncertain("inventory_state_read_failed")
                continue
            if value_type in (BuiltinFunctionType, MethodType):
                receiver = _numpy_rng_receiver(value)
                if (
                    receiver is not None
                    and id(receiver) not in seen_ids
                    and id(receiver) not in monitor._exempt_ids
                ):
                    seen_ids.add(id(receiver))
                    try:
                        snapshots.append((receiver, monitor._digest_rng_witnessable(receiver)))
                    except Exception:
                        monitor._flag_uncertain("inventory_state_read_failed")
                continue
            # Container SUBCLASSES read through the base C implementations, so a
            # hostile override never executes (a ``UserDict``'s ``.data`` arrives
            # via the recursable-holder branch instead).
            if isinstance(value, dict):
                pending.extend(dict.keys(value))
                pending.extend(dict.values(value))
                if value_type is not dict and monitor._is_recursable_custom_holder(value):
                    pending.extend(monitor._custom_holder_children(value))
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
                    monitor._flag_uncertain("inventory_state_read_failed")
                    continue
                if referent is not None:
                    pending.append(referent)
                if type(value) is not _weakref_module.ref:
                    pending.extend(monitor._custom_holder_children(value))
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
                        monitor._flag_uncertain("inventory_state_read_failed")
                if type(value) is not _functools_module.partial:
                    pending.extend(monitor._custom_holder_children(value))
                    pending.append(value_type)
                continue
            if isinstance(value, _collections_module.deque):
                # Base-C iteration, symmetric with the set/frozenset reads above:
                # read-only ``deque.__iter__`` never mutates the buffer and never
                # dispatches to a subclass override.
                pending.extend(_collections_module.deque.__iter__(value))
                if type(value) is not _collections_module.deque:
                    pending.extend(monitor._custom_holder_children(value))
                    pending.append(value_type)
                continue
            # r39 C-holder parity branches (executed false-VERIFIEDs V9b/V9c/V9e/
            # V9f): a generator reached ONLY through a C-implemented holder with a
            # C accessor path never enters a Python frame. Every read below is
            # base-C (``tp_traverse`` / base getset / C ``Context`` lookup).
            if _LRU_CACHE_WRAPPER_TYPE is not None and isinstance(value, _LRU_CACHE_WRAPPER_TYPE):
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
                    monitor._flag_uncertain("inventory_state_read_failed")
                    continue
                if dtype.hasobject:
                    try:
                        size = int(np.ndarray.size.__get__(value))
                    except Exception:
                        monitor._flag_uncertain("inventory_state_read_failed")
                        continue
                    if visited_nodes + size > _types._DEEP_INVENTORY_NODE_CAP:
                        monitor._deep_inventory_exhausted = True
                        monitor._flag_uncertain("deep_inventory_budget_exhausted")
                        return
                    try:
                        pending.extend(np.ndarray.flat.__get__(value))
                    except Exception:
                        monitor._flag_uncertain("inventory_state_read_failed")
                continue
            if isinstance(value, ModuleType):
                namespace = monitor._module_namespace_walk_eligible(value)
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
                if monitor._is_trusted_leaf_class(value):
                    continue
                try:
                    raw_dict = type.__dict__["__dict__"].__get__(value)
                    mro = type.__dict__["__mro__"].__get__(value)
                except Exception:
                    continue
                pending.extend(
                    attr
                    for attr in raw_dict.values()
                    if monitor._module_namespace_class_attr_eligible(attr)
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
            if monitor._is_recursable_custom_holder(value):
                pending.extend(monitor._custom_holder_children(value))
                # An instance's CLASS is a holder surface too (dynamically created
                # classes are in no namespace); dedup keeps this a single filtered
                # dict read per distinct class.
                pending.append(value_type)
                continue
            if monitor._exposes_queue_protocol(value):
                # r38 mirror of the model sweep's r45/r47 queue posture: an
                # inspectable queue (``queue.Queue`` family) was already walked as
                # a recursable holder above (its ``.queue`` deque descends via the
                # deque branch); an OPAQUE queue (``SimpleQueue`` / ``mp.Queue`` --
                # no readable buffer, ``get`` would drain) that is not
                # non-mutatingly provably empty fails CLOSED. A frame-reachable
                # queue-held generator is a typed INCOMPLETE, never a silent leaf.
                with monitor._monitor_internal_probe():
                    provably_empty = monitor._opaque_queue_provably_empty(value)
                if not provably_empty:
                    monitor._flag_uncertain("inventory_opaque_container")
    except Exception:
        monitor._flag_uncertain("inventory_scan_failed")
    finally:
        monitor._deep_inventory_visited = visited_nodes
