"""Receiver, holder and leaf type tables for the monitor's RNG instance inventory.

Which Python objects can hold or be a witnessed RNG (NumPy generators, ``random``
instances), which are inert leaves the inventory never descends into, how slot members and
exempt TorchLens-owned instances are resolved, and the inventory node caps.
"""

from __future__ import annotations

import functools as _functools_module
import os as _os_module
import random
import sys as _sys_module
import weakref as _weakref_module
from collections.abc import Collection
from types import (
    BuiltinFunctionType,
    CodeType,
    FrameType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodType,
    ModuleType,
)
from typing import Any

import numpy as np
import torch

try:  # ``resource`` is POSIX-only; feature-detected for the clock family.
    import resource as _resource_module
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource_module = None  # type: ignore[assignment]

_NUMPY_RNG_INSTANCE_TYPES: tuple[type, ...] = (
    np.random.Generator,
    np.random.RandomState,
    np.random.BitGenerator,
    # grind-r5 b8 R57: SeedSequence is a spawnable entropy root whose
    # ``spawn()`` advances only ``_n_children_spawned`` -- digestable hidden
    # state, so it is a first-class monitored holder.
    np.random.SeedSequence,
)
"""Public NumPy RNG receiver types covered by the host-nondeterminism witness.

``SeedSequence`` is a first-class member (r5 b8-fable R57): ``spawn()`` on a
model-held sequence (or on a generator's underlying sequence) advances
``n_children_spawned`` -- verdict-steering hidden state that keys every
future child's stream -- without touching ``bit_generator.state``.
"""

_INERT_PROFILE_C_CALL_RECEIVER_TYPES: frozenset[type] = frozenset(
    {ModuleType, dict, list, set, str}
)
"""Exact receiver types that cannot own any monitored host-nondeterminism channel."""


def _numpy_rng_receiver(value: Any) -> Any | None:
    """Return the NumPy RNG receiver bound to ``value``, when present.

    Parameters
    ----------
    value:
        Candidate bound callable.

    Returns
    -------
    Any or None
        The owning NumPy RNG instance, or ``None`` when ``value`` is not bound
        to a public NumPy RNG type.
    """

    if not isinstance(value, (BuiltinFunctionType, MethodType)):
        return None
    receiver = value.__self__
    return receiver if isinstance(receiver, _NUMPY_RNG_INSTANCE_TYPES) else None


def _numpy_rng_methods_need_frame_digest() -> bool:
    """Feature-detect NumPy RNG methods that do not emit profile ``c_call`` events.

    NumPy 1.x binds its public RNG C methods as ``BuiltinFunctionType`` objects,
    which the existing receiver-profile classifier observes directly. NumPy 2.x
    binds Cython callables as Python ``MethodType`` objects; those calls do not
    emit ``c_call`` events. Inspecting the public class surfaces avoids version
    parsing and draw-method name enumeration.

    Returns
    -------
    bool
        Whether any public NumPy RNG receiver surface uses Python method binding.
    """

    probes = (
        np.random.Generator(np.random.PCG64(0)),
        np.random.RandomState(0),
    )
    for probe in probes:
        for descriptor in vars(type(probe)).values():
            descriptor_get = getattr(descriptor, "__get__", None)
            if not callable(descriptor_get):
                continue
            try:
                bound = descriptor_get(probe, type(probe))
            except (AttributeError, TypeError, ValueError):
                continue
            if isinstance(bound, MethodType) and _numpy_rng_receiver(bound) is probe:
                return True
    return False


_NUMPY_RNG_METHODS_NEED_FRAME_DIGEST = _numpy_rng_methods_need_frame_digest()
"""Whether NumPy RNG draws need the profiled-frame state-digest fallback."""

_NUMPY_FRAME_DIGEST_INTERNAL_PATH_PREFIXES = (
    f"{_os_module.path.dirname(_os_module.path.dirname(_os_module.path.dirname(__file__)))}{_os_module.sep}",
    f"{_os_module.path.dirname(np.__file__)}{_os_module.sep}",
    f"{_os_module.path.dirname(torch.__file__)}{_os_module.sep}",
)
"""Package roots whose internal frames cannot originate user-owned NumPy RNG draws."""

_STDLIB_PATH_PREFIX = f"{_os_module.path.dirname(_os_module.__file__)}{_os_module.sep}"
"""Filesystem root of the running interpreter's standard library."""

_DEEP_INVENTORY_NODE_CAP = 1_000_000
"""Defensive per-window walk budget for the frame-reachable RNG inventory (B4).

Matches :data:`_INVENTORY_NODE_CAP` semantics exactly: exhaustion flags
``deep_inventory_budget_exhausted`` (INCOMPLETE, fail-closed), never a silent
partial snapshot. Realistic captures walk at most a few thousand nodes (only
roots the profiled code actually references are expanded); the cap only exists
so a pathological object graph terminates.
"""

_INERT_PRIMITIVE_LEAF_TYPES: frozenset[type] = frozenset(
    {str, bytes, bytearray, memoryview, bool, int, float, complex, type(None)}
)
"""Exact value types that can neither be nor inertly hold an RNG receiver."""

_PARTIAL_SLOT_DESCRIPTORS: tuple[Any, ...] = tuple(
    descriptor
    for descriptor in (
        vars(_functools_module.partial).get(name) for name in ("func", "args", "keywords")
    )
    if isinstance(descriptor, (MemberDescriptorType, GetSetDescriptorType))
)
"""Base ``functools.partial`` C slot descriptors for ``func``/``args``/``keywords`` (r38).

``partial`` interiors are C slots invisible to ``__dict__`` reads; the frame-reachable
inventory reads them through these BASE descriptors so a hostile subclass shadow never
executes. The ``func`` edge recovers ``partial(gen.random)``-style receivers through the
bound-callable extraction."""

_WEAKREF_PROXY_TYPES: tuple[type, ...] = (
    _weakref_module.ProxyType,
    _weakref_module.CallableProxyType,
)
"""The two weakref PROXY C types (r39 -- executed false-VERIFIED V9a).

A proxy is NOT a ``weakref.ref`` subclass and has NO inert dereference: CPython's proxy
``tp_traverse`` does not yield the referent (``gc.get_referents(proxy)`` is empty --
verified on py3.10), and EVERY other read, including ``isinstance`` (via ``__class__``)
and attribute access, forwards through the referent's own attribute machinery, where a
hostile ``__getattribute__`` could fire. Both walks therefore match on the EXACT
``type(value)`` (which never forwards) and fail CLOSED."""

_LRU_CACHE_WRAPPER_TYPE: type | None = getattr(_functools_module, "_lru_cache_wrapper", None)
"""The C ``functools.lru_cache`` wrapper type (r39 -- executed false-VERIFIED V9b).

A cache WARMED before the capture returns its cached generator with no Python frame (the
wrapped function never runs), so the draw is witnessable only by digesting the cache
contents, exposed inertly by ``tp_traverse``. ``None`` on a runtime whose ``lru_cache``
is the pure-Python fallback -- there the wrapper is a plain ``FunctionType`` whose cache
hits still enter a profiled Python frame, so the return-transfer digest covers it."""


class _NotADigestableRng(Exception):
    """Internal sentinel: the value is not a digestable numpy/`random` generator."""


_RNG_TRUSTED_DEFINER_TOPS: frozenset[str] = frozenset(
    {"random", "_random", "numpy", "builtins", "torch"}
)
"""Top-level modules whose classes may define an RNG's witnessed draw/state surface.

The monitor's three witnesses (base-class method patches, ``c_call`` receiver
classification, C-state digests) all assume the draw/state methods are the
LIBRARY'S. Library-shipped subclasses keep that true (``SystemRandom`` routes
through the patched primitives; numpy's ``PCG64``/``MT19937``/... advance the
digested C state), so any method first defined by a class from these modules is
witnessable. A method first defined by a class from anywhere else is user code
the witnesses cannot see.
"""


def _untrusted_rng_override(holder_type: type) -> str | None:
    """Return the first draw/state method a USER class overrides, or ``None``.

    Parameters
    ----------
    holder_type:
        Concrete type of a digestable RNG holder (``random.Random`` or numpy
        ``Generator``/``RandomState``/``BitGenerator`` lineage).

    Returns
    -------
    str | None
        Name of the first witnessed-surface method whose defining class is not
        library code, or ``None`` when the whole surface is library-defined.
        The surface is every non-dunder attribute of the recognized library
        bases present in the MRO -- draw helpers included, because a helper
        overridden in user code can draw without ever reaching the patched
        primitives or advancing the digested C state.
    """

    if holder_type in (
        random.Random,
        random.SystemRandom,
        np.random.Generator,
        np.random.RandomState,
        torch.Generator,
    ):
        return None
    base_names: set[str] = set()
    trusted_bases = (random.Random, np.random.Generator, np.random.RandomState)
    for base in trusted_bases:
        if issubclass(holder_type, base):
            base_names.update(name for name in dir(base) if not name.startswith("_"))
    if issubclass(holder_type, np.random.BitGenerator):
        base_names.update(name for name in dir(np.random.BitGenerator) if not name.startswith("_"))
    if issubclass(holder_type, torch.Generator):
        # r7 b8-sol R57: torch.Generator is subclassable; a user override of
        # its draw/state surface (get_state/manual_seed/seed/...) would
        # shadow the C state the digest reads, exactly the numpy false-clean.
        base_names.update(name for name in dir(torch.Generator) if not name.startswith("_"))
    for name in sorted(base_names):
        for cls in holder_type.__mro__:
            if name in vars(cls):
                if cls.__module__.split(".", 1)[0] not in _RNG_TRUSTED_DEFINER_TOPS:
                    return name
                break
    return None


_UNCERTAIN_DETAIL_CAP: int = 64
"""Max DISTINCT ``uncertain_detail`` reasons retained per monitoring window.

The boolean ``uncertain`` verdict is unconditional; the detail is a diagnostic.
Past the cap one ``uncertain_detail_capped`` marker discloses the suppression.
"""


def _torchlens_module_globals_ids() -> frozenset[int]:
    """Identity keys of every loaded torchlens module's globals dict.

    TorchLens's own capture machinery reads clocks constantly (per-op timing); its
    reads are excluded by EXACT module ownership -- the caller frame's globals dict
    identity against this registry -- never by filename strings or frame ancestry.
    A user callback's code keeps its own module globals even when invoked from a
    TorchLens frame, so user reads are always detected.
    """

    ids = set()
    for name, module in list(_sys_module.modules.items()):
        if module is not None and (name == "torchlens" or name.startswith("torchlens.")):
            module_dict = getattr(module, "__dict__", None)
            if module_dict is not None:
                ids.add(id(module_dict))
    return frozenset(ids)


def _rng_exempt_instances() -> tuple[Any, ...]:
    """Replayable global singletons + TorchLens's private barcode RNG (identity-exempt).

    These are the load-bearing no-over-trigger gate: the legacy ``mtrand._rand``
    singleton and module ``random`` engine keep their SEEDED-reproduction semantics
    (a seeded model stays VERIFIED), and TorchLens's own barcode RNG draws during the
    forward must never ceiling the capture.
    """

    exempt: list[Any] = []
    inst = getattr(random, "_inst", None)
    if inst is not None:
        exempt.append(inst)
    np_singleton = getattr(getattr(np.random, "mtrand", None), "_rand", None)
    if np_singleton is not None:
        exempt.append(np_singleton)
    # r7 b8-sol R57: torch.Generator holders are digestable now, so the
    # REPLAYABLE global torch engine must be identity-exempt exactly like the
    # random/numpy module singletons -- its state is seeded and snapshot-
    # replayed by capture, and a seeded torch draw must never ceiling.
    torch_singleton = getattr(torch, "default_generator", None)
    if torch_singleton is not None:
        exempt.append(torch_singleton)
    try:
        from ..hashing import _BARCODE_RNG

        exempt.append(_BARCODE_RNG)
    except Exception:  # pragma: no cover - hashing is a first-party import
        pass
    # A numpy ``Generator``/``RandomState`` OWNS a distinct ``BitGenerator`` object that
    # advances on every draw and is itself GC-visible. Exempt those underlying bit
    # generators transitively so the process-wide inventory does not falsely ceiling a
    # SEEDED replayable-singleton draw (the load-bearing over-trigger gate; a seeded
    # ``np.random.random()`` model must stay VERIFIED).
    for candidate in list(exempt):
        bit_generator = None
        if isinstance(candidate, np.random.Generator):
            bit_generator = getattr(candidate, "bit_generator", None)
        elif isinstance(candidate, np.random.RandomState):
            bit_generator = getattr(candidate, "_bit_generator", None) or getattr(
                candidate, "bit_generator", None
            )
        if bit_generator is not None:
            exempt.append(bit_generator)
    return tuple(exempt)


_CUSTOM_HOLDER_SKIP_MODULES = frozenset(
    {
        # Stdlib runtime primitives + torch/numpy implementation objects the custom-holder
        # generator sweep (r42 corr2_1) must NOT recurse into: threading/asyncio/file/
        # socket handles and torch/numpy internals (a digestable RNG is snapshotted BEFORE this
        # skip check; everything else in those namespaces is impl noise). Keyed on the top-level
        # module of the value's TYPE. r45 hon1_1: ``collections`` and ``queue`` are REMOVED --
        # ``deque`` / ``ChainMap`` / ``UserList`` / ``UserDict`` / ``Counter`` and the ``queue.*``
        # containers are now reached structurally (the ``Mapping`` / non-leaf ``Collection`` descent
        # and the queue-protocol snapshot) BEFORE this skip check ever runs, so their held
        # generators are no longer silently missed.
        "builtins",
        "threading",
        "asyncio",
        "socket",
        "io",
        "_io",
        "selectors",
        "subprocess",
        "multiprocessing",
        "concurrent",
        "ctypes",
        "weakref",
        "typing",
        "functools",
        "pathlib",
        "logging",
        "numpy",
        "torch",
    }
)
"""Top-level module names whose instances the custom-holder generator sweep skips (r42 corr2_1)."""

_STDLIB_CLASS_LEAF_MODULES: frozenset[str] = frozenset(_sys_module.stdlib_module_names) - {
    "__main__"
}
"""Stdlib top-level module names whose CLASSES the class-surface walk leafs (r56 amb_1).

Structural (``sys.stdlib_module_names``), never a hand-maintained denylist -- closing the
whole "stdlib class dict carries a process-global registry" family at once (``collections.abc``
ABC ``_abc_impl`` caches were the proven member: r45 removed ``collections`` from the
INSTANCE-side skip set so containers descend, which silently re-opened the CLASS-side walk of
ABC dicts). Applies ONLY to :meth:`host_nondeterminism_monitor._is_trusted_leaf_class` -- the
instance-side container/holder descent is untouched, so a ``deque`` / ``UserList`` /
``UserDict`` subclass instance and its held generators stay fully reachable. ``__main__`` is
carved out: model classes defined in scripts/notebooks keep r53 class-attribute coverage.
"""

_INVENTORY_LEAF_TYPES: tuple[type, ...] = (
    str,
    bytes,
    bytearray,
    memoryview,
    np.ndarray,
    np.generic,
    torch.Tensor,
    torch.nn.Module,
    ModuleType,
)
"""Types the sweep's ``Collection`` branch never ELEMENT-ITERATES (r45 hon1_1).

``str`` / ``bytes`` / ``bytearray`` / ``memoryview`` / ``np.ndarray`` / a bare ``torch.Tensor``
satisfy ``collections.abc.Collection`` yet must NOT be descended element-wise: a string would
explode into per-character strings (huge / cyclic node blow-up) and a tensor / ndarray holds no
Python-level RNG among its ELEMENTS. r61 hon_1: this tuple now means exactly "do not iterate
the buffer" -- it is NOT an instance-state wall. A tensor / ndarray-subclass node still walks
its Python instance ``__dict__`` / ``__slots__`` through the gc fallback's numeric-payload
branch (:data:`_NUMERIC_PAYLOAD_LEAF_TYPES`), so ``weight.rng = default_rng()`` is inventoried.
``torch.nn.Module`` and ``ModuleType`` are likewise excluded from the
``Collection`` descent branch (this entry is inert for ``nn.Module`` -- it is not a
``collections.abc.Collection`` -- and documents intent). r51 hon1_1: an ``nn.Module`` is NO LONGER
a hard inventory leaf; it is descended via ``_is_recursable_custom_holder`` (its own
``__dict__`` / ``__slots__``), reaching a generator behind an UNREGISTERED submodule.
"""


def _derive_abc_impl_type() -> type | None:
    """The C ``_abc._abc_data`` type carried by every ``ABCMeta`` class (r56 amb_1).

    Derived from a live ABC rather than importing the private ``_abc`` module. ``None`` on
    a runtime using the pure-Python ``_py_abc`` fallback (no CPython ``_abc_impl`` slot),
    where this exclusion is inapplicable by construction.
    """

    probe = Collection.__dict__.get("_abc_impl")
    if probe is not None and type(probe).__module__ == "_abc":
        return type(probe)
    return None


_ABC_IMPL_TYPE: type | None = _derive_abc_impl_type()

_AMBIENT_BRIDGE_LEAF_TYPES: tuple[type, ...] = (
    ModuleType,
    CodeType,
    FrameType,
) + ((_ABC_IMPL_TYPE,) if _ABC_IMPL_TYPE is not None else ())
"""Types the authoritative ``gc.get_referents`` fallback SEVERS entirely (r55 C6, r61 split).

r61 hon_1 split the former ``_GC_EXPANSION_LEAF_TYPES`` by the REASON each member was
excluded. This tuple keeps exactly the members that BRIDGE the rooted walk into ambient
process state -- severed to ``[]``, never partially walked:

- ``ModuleType``: a module is a SHARED namespace; expanding it drops the rooted
  walk into every imported framework's globals. A generator held in a shared
  module global is the explicit documented residual (contract section 11).
- ``CodeType``: code objects hold only compile-time constants (``co_consts``)
  -- a live generator cannot exist there.
- ``FrameType``: frames chain through ``f_back`` into the ENTIRE interpreter
  call stack (TorchLens's own frames included) -- shared, unbounded state.
- ``_abc._abc_data`` (r56 amb_1): every ``ABCMeta`` class's ``_abc_impl`` slot.
  Its ``tp_traverse`` exposes the ABC registry/cache/NEGATIVE-cache weakref sets
  -- PROCESS-GLOBAL bookkeeping accumulating a weakref to every type ever
  ``isinstance``-checked against that ABC. Expanding one bridges the rooted walk
  out of the model subgraph into the ambient object graph (every cached class's
  dict, then its attribute webs), making sweep completeness ambient-state-
  dependent: under a large ambient graph the walk digested foreign generators
  and exhausted the node cap BEFORE the model's own generator (the r55 full-run
  regression on ``test_r47_generator_container_subclass``). The caches hold only
  weakrefs to TYPES plus registered classes, so a model-held generator can never
  live inside one -- excluding them loses zero legitimate coverage.

``types.TracebackType`` is deliberately in NEITHER r61 tuple (unchanged posture): a
walked traceback contributes only ``tb_frame`` / ``tb_next``, the frame is severed at
the referent filter, a traceback has no instance ``__dict__`` -- probed, no ambient
escape either way.

Deliberately NOT severed: ``torch.nn.Module`` (r51 hon1_1 -- an unregistered
submodule must stay reachable), ``type`` objects (a referent class is enqueued
and handled by the r53 class-surface branch with its trusted-leaf gate), and
stdlib wrapper instances such as ``functools._lru_cache_wrapper`` (r54 corr_4:
the ``__wrapped__`` edge must be walked; boundedness is owned by the node cap,
inertness by ``tp_traverse`` itself).

``BuiltinFunctionType`` was REMOVED in r57 C6 (r56 hon_1): its old
justification ("a C function's only referents are its ``__self__`` module")
is FALSE for a *bound* C method, whose ``__self__`` is the RECEIVER instance
-- ``gen.random.__self__`` IS the numpy ``Generator``, so blanket-
leafing dropped a real inert edge ``gc.get_referents`` exposes (a model
caching ``self.sample = self.rng.random`` read false VERIFIED).
Bound C callables are now special-cased at the top of
:meth:`host_nondeterminism_monitor._inert_gc_children`: exactly the
``__self__`` receiver is enqueued, gated by the SAME shared-namespace /
ambient-bridge walls as every other node, and the node never expands
generically (a module-level C function -- module or absent ``__self__`` --
contributes nothing).

r57 C6 follow-up: ``FunctionType`` gets the matching callable-specific
treatment in the same method -- authoritative traverse minus its
``__globals__`` / ``__builtins__`` namespace edges dropped BY IDENTITY
(registry-independent, so an ``exec``/``torch.fx``-generated function with a
SYNTHETIC globals dict is walled too). After any capture, ``wrap_torch()``
rebinds torch ops (``torch.relu`` / ``torch.zeros``) from builtins to
``FunctionType`` wrappers, so without this branch a cached
``self.act = torch.relu`` fell through to the generic fallback and enqueued
function namespace/metadata edges -- the same ambient-escape class the r47/
r56 walls close. Builtin-vs-wrapped op shape no longer changes the sweep.
"""

_NUMERIC_PAYLOAD_LEAF_TYPES: tuple[type, ...] = (np.ndarray, np.generic, torch.Tensor)
"""Numeric-payload types: buffer never walked, instance STATE walked (r61 hon_1).

These are NO LONGER hard expansion leaves. The r45/r55 posture blanket-leafed them
("a tensor's traverse reaches autograd internals, and neither holds a Python-level
RNG"), which silently dropped REAL inert instance state: arbitrary user attributes on
a tensor / ``nn.Parameter`` / ndarray-subclass instance are valid CPython
(``self.lin.weight.rng = default_rng()`` -- the r60 hon_1 false-VERIFIED repro).

The r61 rule is per-EDGE, not per-type: the C numeric buffer / autograd / storage
internals are never walked (``gc.get_referents`` is never called on these types), but
the Python instance ``__dict__`` / ``__slots__`` surfaces ARE walked via
:meth:`host_nondeterminism_monitor._custom_holder_children` -- the same inert
protocol every other holder gets (slot member descriptors, never ``getattr``; zero
user code) -- AND (r63) the node's class-surface edge is enqueued like every other
holder branch's: ``type(value)`` flows through the trusted-leaf-gated class branch,
so a class-attribute generator on a USER-defined Tensor / Parameter / ndarray
subclass is inventoried while stock ``torch.Tensor`` / ``nn.Parameter`` /
``np.ndarray`` / ``np.generic`` remain trusted leaves (no implementation-class
expansion). ``np.generic`` scalars carry no instance surface and contribute only
their (leafed) class.

Distinct from :data:`_INVENTORY_LEAF_TYPES`, which means "do not element-iterate this
buffer-like Collection" (a tensor still never explodes into per-element nodes) -- the
two tuples diverge on purpose: element iteration and instance-state walking are
different edges.
"""


def _resolve_slot_member(klass: type, slot_name: str) -> MemberDescriptorType | None:
    """Return ``klass``'s slot member descriptor, resolving CPython name mangling (r61 corr_1).

    A ``__slots__`` entry spelled ``"__rng"`` (two leading underscores, at most one
    trailing) is stored in the class dict under its MANGLED key ``_Class__rng`` --
    while ``__slots__`` itself still lists the RAW string. The former
    ``klass.__dict__.get(slot)`` lookup therefore returned ``None`` for every private
    slot and its held generator was silently unwitnessed (false VERIFIED).

    Mangling rules pinned against CPython ground truth (6/6 probe cases): ``__rng`` ->
    ``_Class__rng``; ``___x`` -> ``_Class___x``; a trailing-dunder name (``__d__``) is
    NOT mangled; leading underscores are stripped from the class name (``_F`` ->
    ``_F__rng``); an all-underscore class name mangles NOTHING; a single-underscore
    name (``_x``) is not mangled. The DECLARING class from the MRO is the mangling
    authority (each MRO level mangles against its own name).

    r63 (r62 raw-shadow HIGH): the MANGLED private descriptor is preferred BEFORE any
    raw class-dict key. Name mangling is a compile-time class-body transform, so a
    post-hoc raw entry (``M.__rng = shadow`` via module-level ``setattr``) lands at the
    UNMANGLED key while the real slot descriptor stays at ``_M__rng`` -- the former
    raw-key-first lookup let that shadow win and silently dropped the slot-held
    generator (false VERIFIED). Every candidate key is TYPECHECKED: only an inert
    ``types.MemberDescriptorType`` (the C slot descriptor, whose ``__get__`` is pure C
    field access and executes no Python) is ever returned -- never an arbitrary
    class-dict value whose ``__get__`` could invoke user code (a property or hostile
    descriptor planted at either key). Returns ``None`` when neither key holds a
    member descriptor; the caller skips the slot (nothing executed, nothing read).
    Slot STORAGE keys are mangled by ``type.__new__`` itself (probed), so compiled and
    ``type()``-built classes agree on the mangled key; the raw-key fallback covers
    exactly the shapes mangling leaves untouched (non-private names, trailing-dunder
    names, and an all-underscore class name, where the descriptor lives at the raw key).
    """

    class_dict = klass.__dict__
    candidate_keys: list[str] = []
    if slot_name.startswith("__") and not slot_name.endswith("__"):
        stripped = klass.__name__.lstrip("_")
        if stripped:
            candidate_keys.append("_" + stripped + slot_name)
    candidate_keys.append(slot_name)
    for key in candidate_keys:
        member = class_dict.get(key)
        if isinstance(member, MemberDescriptorType):
            return member
    return None


_INVENTORY_NODE_CAP = 1_000_000
"""Defensive node cap for the model-attribute generator sweep (r41 hon1_2).

Realistic models sit 2-3 orders of magnitude below this (a 400-block toy holds ~7.7k
module-dict values; the cap admits ~1M visited nodes in ~300 ms measured), so no
realistic deterministic model can be ceilinged by it. Exhaustion NEVER truncates
silently: it flags ``inventory_budget_exhausted`` monitor uncertainty, downgrading
capture completeness to INCOMPLETE -- a truncated inventory never reads as
no-consumption. Read at call time so the cap-exhaustion invariant is testable at any
cap value.
"""
