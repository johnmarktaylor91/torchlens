"""Holder, leaf and namespace walk rules of the monitor's RNG instance inventory."""

from __future__ import annotations

import gc as _gc_module
import sys as _sys_module
from types import (
    BuiltinFunctionType,
    CodeType,
    FunctionType,
    GetSetDescriptorType,
    MethodType,
    MethodWrapperType,
    ModuleType,
)
from typing import Any

import numpy as np
import torch

from . import inventory_types as _types

try:  # ``resource`` is POSIX-only; feature-detected for the clock family.
    import resource as _resource_module
except ImportError:  # pragma: no cover - non-POSIX platforms
    _resource_module = None  # type: ignore[assignment]

from .inventory_types import (
    _AMBIENT_BRIDGE_LEAF_TYPES,
    _CUSTOM_HOLDER_SKIP_MODULES,
    _INERT_PRIMITIVE_LEAF_TYPES,
    _NUMERIC_PAYLOAD_LEAF_TYPES,
    _NUMPY_RNG_INSTANCE_TYPES,
    _STDLIB_CLASS_LEAF_MODULES,
    _STDLIB_PATH_PREFIX,
    _numpy_rng_receiver,
    _resolve_slot_member,
)


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
    has_slots = any("__slots__" in getattr(klass, "__dict__", {}) for klass in type(value).__mro__)
    return has_dict or has_slots


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
        payload_children = _custom_holder_children(value)
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
        if _is_trusted_leaf_class(current):
            continue
        try:
            raw_dict = type.__dict__["__dict__"].__get__(current)
            mro = type.__dict__["__mro__"].__get__(current)
        except Exception:
            continue
        surface.extend(
            attr for attr in raw_dict.values() if _module_namespace_class_attr_eligible(attr)
        )
        stack.extend(base for base in mro if base is not current)
    return tuple(surface)
