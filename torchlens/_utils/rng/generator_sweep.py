"""Model-held torch generator sweep: digests every generator a model can reach."""

from __future__ import annotations

import contextvars as _contextvars_module
import weakref as _weakref_module
from collections.abc import Collection, Mapping
from typing import Any

import numpy as np
import torch

from . import inventory_types as _types

try:  # ``resource`` is POSIX-only; feature-detected for the clock family.
    import resource as _resource_module
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource_module = None  # type: ignore[assignment]

from typing import TYPE_CHECKING

from .inventory_types import (
    _INVENTORY_LEAF_TYPES,
    _WEAKREF_PROXY_TYPES,
    _NotADigestableRng,
)

if TYPE_CHECKING:
    from .monitor import host_nondeterminism_monitor


def _sweep_model_generators(monitor: host_nondeterminism_monitor) -> list[tuple[Any, str]]:
    """Digest numpy/`random` generators reachable from the MODEL's submodule attributes.

    This is the CHEAP thread-independent belt (model attributes only), NOT a
    process-wide ``gc.get_objects()`` scan (the r39-draft GC-wide inventory cost
    ~900 ms/capture, perturbed the peak-memory bracket, and over-trigger-risked
    unrelated generators -- removed for cause and never reintroduced).

    r41 hon1_2 / r45 hon1_1: the sweep is an ITERATIVE, cycle-safe recursion by
    container PROTOCOL (not a fixed concrete-type list) -- every submodule ``__dict__``
    value, descending through every ``collections.abc.Mapping`` (KEYS and VALUES) and
    every non-leaf ``collections.abc.Collection`` (elements), so ``monitor.pool = [gen]``,
    ``{"g": gen}``, ``[[gen]]``, a dict-key generator, and a generator inside a ``deque``
    / ``ChainMap`` / ``UserList`` / ``UserDict`` / namedtuple / custom ``Sequence`` or
    ``Mapping`` are all digested. Descent is gated on ``Collection`` (Sized), NEVER bare
    ``Iterable``, so a one-shot generator / ``map`` / ``itertools`` attribute is never
    consumed. A safe queue (``queue.Queue`` / ``LifoQueue`` / ``PriorityQueue``) is reached
    through a non-mutating snapshot of its internal ``.queue`` deque; an opaque queue
    (``SimpleQueue`` / ``mp.Queue``) with no inspectable buffer fails closed to INCOMPLETE
    (``inventory_opaque_container``). r42 corr2_1: it ADDITIONALLY recurses into inert
    CUSTOM object holders (``monitor.holder.rng``) through their ``__dict__`` / ``__slots__``
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
    ``monitor.sample = monitor.rng.random`` bound method, alone or behind
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
    model = monitor._model
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
        shared_namespace_ids = monitor._shared_namespace_dict_ids()
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
            pending.extend(monitor._custom_holder_children(module))
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
                monitor._flag_uncertain("inventory_budget_exhausted")
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
                monitor._flag_uncertain("inventory_opaque_container")
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
                if monitor._is_trusted_leaf_class(value):
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
                    monitor._flag_uncertain("inventory_state_read_failed")
                    continue
                if referent is not None:
                    pending.append(referent)
                if type(value) is not _weakref_module.ref:
                    pending.extend(monitor._custom_holder_children(value))
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
                    monitor._flag_uncertain("inventory_state_read_failed")
                    continue
                if dtype.hasobject:
                    try:
                        size = int(np.ndarray.size.__get__(value))
                    except Exception:
                        monitor._flag_uncertain("inventory_state_read_failed")
                        continue
                    if visited_nodes + size > _types._INVENTORY_NODE_CAP:
                        monitor._flag_uncertain("inventory_budget_exhausted")
                        return snapshots
                    try:
                        pending.extend(np.ndarray.flat.__get__(value))
                    except Exception:
                        monitor._flag_uncertain("inventory_state_read_failed")
                        continue
                pending.extend(monitor._custom_holder_children(value))
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
                # carry a generator as its OWN attribute (``monitor.rng`` on a ``UserDict`` /
                # custom ``Mapping`` subclass), which is NOT among its keys/values. Walk its
                # own ``__dict__`` / ``__slots__`` too so the held generator is digest-
                # witnessed. ``_is_recursable_custom_holder`` excludes stdlib/torch/numpy, so a
                # plain ``dict`` is never double-walked; cycle-guarded via ``seen_container_ids``
                # (already added above), node-capped, and never invokes a property/``__getattr__``.
                if monitor._is_recursable_custom_holder(value):
                    pending.extend(monitor._custom_holder_children(value))
                    pending.append(type(value))  # r53 corr_1: class surface of the subclass
                continue
            if isinstance(value, Collection) and not isinstance(value, _INVENTORY_LEAF_TYPES):
                if id(value) in seen_container_ids:
                    continue
                seen_container_ids.add(id(value))
                pending.extend(value)
                # r47 hon1_1: same dual-walk for a non-leaf Collection that is ALSO a custom
                # inert holder -- ``monitor.rng`` on a ``Sequence`` / ``MutableSequence`` /
                # ``UserList`` / ``__slots__`` collection subclass, not among its elements.
                if monitor._is_recursable_custom_holder(value):
                    pending.extend(monitor._custom_holder_children(value))
                    pending.append(type(value))  # r53 corr_1: class surface of the subclass
                continue
            if id(value) in monitor._exempt_ids:
                continue
            try:
                digest = monitor._digest_rng_witnessable(value)
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
                if monitor._exposes_queue_protocol(value):
                    seen_container_ids.add(id(value))
                    inner = getattr(value, "queue", None)
                    if isinstance(inner, Collection) and not isinstance(
                        inner, _INVENTORY_LEAF_TYPES
                    ):
                        pending.append(inner)
                    else:
                        # r49 hon1_1: the emptiness proof is a MONITOR-INTERNAL read. Bracket
                        # ONLY the probe so a clock touched TRANSITIVELY by ``mp.Queue.empty()``
                        # (via ``multiprocessing.connection``) is monitor-suppressed at the
                        # ``_mark`` choke point, not mis-marked as a model host read. The
                        # ``_flag_uncertain`` branch stays OUTSIDE the bracket so the NON-EMPTY
                        # opaque-queue INCOMPLETE residual is preserved.
                        with monitor._monitor_internal_probe():
                            provably_empty = monitor._opaque_queue_provably_empty(value)
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
                            monitor._flag_uncertain("inventory_opaque_container")
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
                pending.extend(monitor._inert_gc_children(value, shared_namespace_ids))
                continue
            except Exception:
                monitor._flag_uncertain("inventory_state_read_failed")
                continue
            snapshots.append((value, digest))
    except Exception:
        monitor._flag_uncertain("inventory_scan_failed")
    return snapshots
