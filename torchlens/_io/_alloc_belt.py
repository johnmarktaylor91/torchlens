"""Allocation belt of the bundle unpickler: what a stream may name, construct or reach.

The vocabulary :mod:`torchlens._io._safe_unpickle` enforces on every pickled global and
every REDUCE / NEWOBJ / NEWOBJ_EX / INST / OBJ, kept apart from the two unpickler
machines that apply it:

* the torch I/O / serialization / packaging / jit TYPE denylist
  (``_torch_type_denied``), the load-bearing structural guard that keeps file readers,
  writers, importers and picklers out of a resolved ``torch.*`` type;
* the storage and tensor constructor sets (``_is_torch_storage_type``,
  ``_is_torch_storage_instance``, ``_is_alloc_constructor_type``) and the mediated
  allocation belt (``_alloc_refusal_reason``) that refuses an attacker-sized allocation
  whether the stream names the constructor, reaches it through a torch callable or a
  numpy reconstructor, or sizes a module construction;
* the exact ``(module, name)`` globals resolved directly (``_SAFE_EXPLICIT_GLOBALS``),
  the restricted ``getattr`` admitted in their place (``_safe_getattr``) and the
  resolved-module ownership test behind the torch branch (``_is_torch_owned``).

Enforced by ``tests/test_r49_unpickler_weights_only_baseline.py``,
``tests/test_r63_construct_deny.py``, ``tests/test_r18m_safe_unpickle_alloc.py`` and
``tests/test_io_mediated_allocation_bypass.py``; the anti-drift invariant is that the
set of admitted ``(module, name)`` for which the unpickler CONSTRUCTS a storage / tensor
/ ndarray object is EMPTY.
"""

from __future__ import annotations

import pickle
from typing import Any

import torch

__all__ = [
    "_SAFE_EXPLICIT_GLOBALS",
    "_alloc_refusal_reason",
    "_is_alloc_constructor_type",
    "_is_torch_owned",
    "_is_torch_storage_instance",
    "_is_torch_storage_type",
    "_safe_getattr",
    "_torch_type_denied",
]

# torch I/O / serialization / packaging / jit TYPE denylist.
#
# The torch-``type`` branch below admits any resolved ``torch.*`` ``type`` on the
# premise that "resolving/constructing a torch data type executes no attacker
# code". That premise HOLDS for the tensor / storage / Size / Parameter / dtype /
# nn-module classes legit metadata references, but it is FALSE for torch's
# I/O-capable types: constructing one opens/reads/writes a filesystem PATH taken
# from the pickle stream at ``tl.load`` time -- e.g.
# ``torch._C.PyTorchFileReader('/etc/passwd')`` (arbitrary-path read + zip-bomb
# DoS) or ``torch.package.PackageImporter(path)`` (directory traversal read).
# torch's own ``weights_only`` allowlist deliberately EXCLUDES these types; we
# align with that intent and deny them here so they are never constructed.
#
# Denial uses two independent signals, because ``__module__`` alone is
# insufficient: ``torch._C.PyTorchFileReader.__module__`` is ``"torch"`` -- the
# SAME module string as ``torch.Size`` / ``torch.Tensor`` -- so a module test
# cannot separate the file reader from a benign shape type. Hence:
#   (a) a MODULE-PREFIX denylist over the pickled path AND the resolved real
#       ``__module__`` (catches ``torch.serialization`` / ``torch.jit`` /
#       ``torch.package`` / ``torch.hub`` families), plus
#   (b) a STRUCTURAL TYPE-NAME guard (the load-bearing one) that denies any type
#       whose pickled name or real ``__qualname__`` contains a file / reader /
#       writer / package / directory / zip / (file)check / importer / pickler
#       token -- this is what stops ``PyTorchFileReader`` / ``PyTorchFileWriter``
#       / ``FileCheck`` / ``PackageImporter`` / ``DirectoryReader``.
#
# Residual (stated honestly): a positive allowlist of the torch DATA types is
# impractical because legit metadata may reference ANY ``nn.Module`` subclass
# (an open-ended set) as a ``module_type``. So this is a denylist: a
# hypothetical future torch type whose construction does I/O yet whose name
# contains none of the tokens and whose module is outside the denied families
# would still be admitted. The token + module families cover the entire known
# torch file-I/O / serialization / packaging / jit surface (audited against
# torch's ``weights_only`` exclusions), so the residual is bounded and small.
_DENIED_TORCH_TYPE_MODULE_PREFIXES: tuple[str, ...] = (
    "torch.serialization",
    "torch.jit",
    "torch.package",
    "torch.hub",
    "torch.utils.tensorboard",
    "torch.utils.cpp_extension",
    "torch.utils.bottleneck",
    "torch.distributed",
    "torch.multiprocessing",
)

_DENIED_TORCH_TYPE_NAME_TOKENS: tuple[str, ...] = (
    "file",  # PyTorchFileReader / PyTorchFileWriter / *FileStore
    "reader",  # DirectoryReader / *Reader
    "writer",  # *Writer
    "package",  # PackageImporter / PackageExporter
    "importer",
    "exporter",
    "directory",
    "zip",  # *ZipFile / zip archive helpers
    "check",  # FileCheck
    "pickler",  # Pickler / Unpickler
    "unpickler",
)


def _torch_type_denied(pickled_module: str, pickled_name: str, obj: type) -> bool:
    """Return whether a resolved torch ``type`` is an I/O / serialization gadget.

    Denies torch's file-reader / file-writer / packaging / serialization / jit
    types whose construction is NOT inert (opens/reads/writes a filesystem path
    from the pickle stream). See ``_DENIED_TORCH_TYPE_MODULE_PREFIXES`` /
    ``_DENIED_TORCH_TYPE_NAME_TOKENS`` for the policy rationale.

    Parameters
    ----------
    pickled_module:
        Attacker-controlled pickled module path (e.g. ``"torch._C"``).
    pickled_name:
        Attacker-controlled pickled global name (e.g. ``"PyTorchFileReader"``).
    obj:
        The resolved torch ``type`` object.

    Returns
    -------
    bool
        ``True`` if the type must be denied (never constructed).
    """

    real_module = str(getattr(obj, "__module__", "") or "")
    for candidate in (pickled_module, real_module):
        if any(
            candidate == prefix or candidate.startswith(prefix + ".")
            for prefix in _DENIED_TORCH_TYPE_MODULE_PREFIXES
        ):
            return True
    real_name = str(getattr(obj, "__qualname__", "") or getattr(obj, "__name__", "") or "")
    haystack = f"{pickled_name} {real_name}".lower()
    return any(token in haystack for token in _DENIED_TORCH_TYPE_NAME_TOKENS)


# STORAGE-CONSTRUCTOR identity denylist (r49, secA_1). The torch-``type`` branch of
# ``find_class`` admits any resolved torch DATA ``type`` on the premise that
# "constructing a torch data type executes no attacker code". That premise HOLDS for
# tensor / Size / Parameter / dtype / nn-module classes, but is FALSE for a torch
# STORAGE class: a pickle ``REDUCE`` of ``torch.UntypedStorage`` with an attacker
# integer -- ``UntypedStorage(6_000_000_000)`` -- allocates that many raw bytes at
# unpickle time (a multi-GiB out-of-memory DoS), and a constructed storage is the only
# source of an uninitialized-heap tensor that a following BUILD/``set_`` could rebind.
# torch's own ``weights_only`` baseline (``_weights_only_unpickler``) DOES list the
# real ``torch.storage.{Typed,Untyped}Storage`` classes, so denying them here makes
# this unpickler a documented STRICT SUBSET of that baseline. No legitimate ``.tlspec``
# metadata references a storage type in the outer stream: tensor payloads are
# ``BlobRef``s, and an inline metadata tensor is rebuilt whole by
# ``rebuild_inline_tensor`` over a non-resizable storage, never by an outer storage
# global. The set is built from ``torch._storage_classes`` (all 31
# typed/untyped classes) plus the canonical ``TypedStorage`` / ``UntypedStorage``, and
# an ``issubclass`` belt covers any future storage subclass (e.g. a legacy typed
# ``FloatStorage`` is a ``TypedStorage`` subclass).
_TORCH_STORAGE_BASE_CLASSES: tuple[type, ...] = tuple(
    cls
    for cls in (
        getattr(torch.storage, "TypedStorage", None),
        getattr(torch, "UntypedStorage", None),
        getattr(torch.storage, "_StorageBase", None),
    )
    if isinstance(cls, type)
)

_TORCH_STORAGE_CLASSES: frozenset[type] = frozenset(
    set(getattr(torch, "_storage_classes", frozenset())) | set(_TORCH_STORAGE_BASE_CLASSES)
)


def _is_torch_storage_type(obj: Any) -> bool:
    """Return whether ``obj`` is a torch STORAGE *class* (its ctor allocates memory).

    Surface-complete + version-robust: an identity membership test over
    ``torch._storage_classes`` (plus ``TypedStorage`` / ``UntypedStorage`` /
    ``_StorageBase``) and an ``issubclass`` belt so any future storage subclass is
    also caught. Never raises (a non-type ``obj`` returns ``False``).
    """

    if not isinstance(obj, type):
        return False
    if obj in _TORCH_STORAGE_CLASSES:
        return True
    if not _TORCH_STORAGE_BASE_CLASSES:
        return False
    try:
        return issubclass(obj, _TORCH_STORAGE_BASE_CLASSES)
    except TypeError:  # pragma: no cover - defensive; issubclass on odd metaclasses.
        return False


def _is_torch_storage_instance(obj: Any) -> bool:
    """Return whether ``obj`` is a torch STORAGE *instance* (a BUILD/rebind target)."""

    if not _TORCH_STORAGE_BASE_CLASSES:
        return False
    try:
        return isinstance(obj, _TORCH_STORAGE_BASE_CLASSES)
    except TypeError:  # pragma: no cover - defensive.
        return False


# ALLOCATION-CONSTRUCTOR TYPE surface (r63, tensor/ndarray-construct regap). The r49
# belt denies only STORAGE construction, but a torch storage is NOT the only
# pickle-reachable attacker-sized allocator: a bare REDUCE/NEWOBJ of ``torch.Tensor(N)``
# / ``torch.FloatTensor(N)`` / ``numpy.ndarray((N,))`` with an attacker-supplied size
# allocates an attacker-sized, UNINITIALIZED-heap buffer at plain ``tl.load()`` time (no
# trust, no ``.run()``) -- the same alloc-DoS + uninitialized-heap-read class the
# storage deny closes, which the storage-only belt let through. The TYPE OBJECTS stay
# RESOLVABLE at ``find_class`` (legit metadata references these types as inert dtype /
# module_type values); ONLY their CONSTRUCTION on the three belts is refused.
#
# Legacy ``torch.<dtype>Tensor`` classes (``torch.FloatTensor``, the ``torch.cuda.*`` /
# ``torch.sparse.*`` families) are ``tensortype``-metaclass instances whose MRO is
# ``[FloatTensor, object]`` -- they are NOT ``torch.Tensor`` subclasses, so
# ``issubclass(obj, torch.Tensor)`` MISSES them, yet ``torch.FloatTensor(5)`` still
# allocates a 5-element uninitialized tensor. They are caught by identity membership over
# ``torch._tensor_classes`` (surface-complete over the 38 current classes) plus an
# ``isinstance``-of-metaclass belt so any future/variant ``tensortype`` class is caught
# too. numpy is a torch dependency and is imported by the time this module loads
# (``import torch`` above), so ``numpy.ndarray`` resolves in practice; the reference is
# guarded so this early-imported security front door never hard-fails if numpy is absent
# (the predicate then degrades to storage + tensor coverage). No ``np`` alias existed in
# this module (numpy globals are admitted by STRING pair in ``_SAFE_EXPLICIT_GLOBALS``,
# never by import), so a guarded reference is introduced here.
try:
    import numpy as _numpy

    _NUMPY_NDARRAY_TYPE: type | None = _numpy.ndarray
except Exception:  # pragma: no cover - numpy is a torch dependency in practice.
    _NUMPY_NDARRAY_TYPE = None


_TORCH_LEGACY_TENSOR_CLASSES: frozenset[type] = frozenset(
    cls for cls in getattr(torch, "_tensor_classes", frozenset()) if isinstance(cls, type)
)

# The ``tensortype`` metaclass(es) that build the legacy tensor classes (``type(...)`` of
# each), excluding plain ``type``. An ``isinstance`` belt over these catches any legacy /
# future tensor class the identity set above does not enumerate.
_TORCH_TENSORTYPE_METACLASSES: tuple[type, ...] = tuple(
    {
        metacls
        for metacls in (type(cls) for cls in _TORCH_LEGACY_TENSOR_CLASSES)
        if metacls is not type
    }
)


def _is_alloc_constructor_type(obj: Any) -> bool:
    """Return whether ``obj`` is a type whose bare construction allocates attacker-sized memory.

    Superset of ``_is_torch_storage_type`` (r49) that ALSO catches the two other
    pickle-reachable attacker-sized allocators (r63): a torch tensor type (a
    ``torch.Tensor`` subclass such as ``Parameter``, OR a legacy ``tensortype``-metaclass
    ``torch.<dtype>Tensor`` such as ``torch.FloatTensor``) and a ``numpy.ndarray``
    subclass. REDUCE/NEWOBJ-constructing any of these with a pickle-supplied size
    allocates an attacker-sized, UNINITIALIZED-heap buffer at plain ``tl.load()`` time.
    The type OBJECTS stay RESOLVABLE at ``find_class`` (legit metadata references
    ``torch.Tensor`` / ``numpy.ndarray`` as inert values); ONLY their CONSTRUCTION on the
    three belts is refused. Never raises (a non-type ``obj`` returns ``False``).

    ``torch.nn.Module`` subclasses are deliberately NOT listed here even though
    ``nn.Linear(65536, 65536)`` allocates ~16 GiB of parameters: that construction is only
    dangerous when the pickle SUPPLIES the sizes, and a real artifact legitimately
    NEWOBJ-constructs a zero-argument module (``torch.nn.modules.linear.Identity()`` from a
    ``splice_module`` intervention helper). An arg-blind belt here would refuse that honest
    artifact, so the module rule lives in the arg-aware ``_alloc_refusal_reason`` instead.
    """

    if _is_torch_storage_type(obj):
        return True
    if not isinstance(obj, type):
        return False
    try:
        if issubclass(obj, torch.Tensor):
            return True
    except TypeError:  # pragma: no cover - defensive; issubclass on odd metaclasses.
        pass
    if obj in _TORCH_LEGACY_TENSOR_CLASSES:
        return True
    if _TORCH_TENSORTYPE_METACLASSES and isinstance(obj, _TORCH_TENSORTYPE_METACLASSES):
        return True
    if _NUMPY_NDARRAY_TYPE is not None:
        try:
            if issubclass(obj, _NUMPY_NDARRAY_TYPE):
                return True
        except TypeError:  # pragma: no cover - defensive.
            pass
    return False


# Resolved IDENTITIES of the numpy array-reconstruction helper admitted by
# ``_SAFE_EXPLICIT_GLOBALS`` under both the legacy (<2.0) and modern (>=2.0) private
# module spellings. Matching on IDENTITY -- never on the attacker-supplied pickled
# ``(module, name)`` -- is the same rule the torch / torchlens branches of
# ``find_class`` already follow. Both spellings alias the SAME function object on every
# numpy that ships the compatibility shim; resolving both keeps the set correct if a
# future numpy splits them.
def _build_numpy_reconstruct_funcs() -> frozenset[Any]:
    """Resolve the numpy ``_reconstruct`` helpers admitted by the explicit allowlist.

    Returns
    -------
    frozenset[Any]
        Every resolvable ``numpy(.|._)core.multiarray._reconstruct`` function object.
        Empty when numpy is absent or the private module layout changed.
    """

    import importlib

    resolved: list[Any] = []
    for module_path in ("numpy.core.multiarray", "numpy._core.multiarray"):
        try:
            candidate = getattr(importlib.import_module(module_path), "_reconstruct", None)
        except Exception:  # pragma: no cover - numpy absent / private layout drift.
            continue
        if callable(candidate):
            resolved.append(candidate)
    return frozenset(resolved)


_NUMPY_RECONSTRUCT_FUNCS: frozenset[Any] = _build_numpy_reconstruct_funcs()


def _is_vetted_torch_reduce_callable(func: Any) -> bool:
    """Return whether a torch-owned callable may legitimately be INVOKED by a REDUCE.

    The ONLY torch-owned callables a legitimate ``.tlspec`` stream ever CALLS are the
    ``torch._utils._rebuild*`` reconstructors, which ``find_class`` already gates
    (undotted name + torch-owned non-type callable). Every other torch-owned callable
    reaches the stack as an inert VALUE -- a function-registry reference resolved through
    ``_safe_getattr`` -- and is never called by an honest stream.

    Parameters
    ----------
    func:
        Callable sitting on the pickle stack as a REDUCE target.

    Returns
    -------
    bool
        ``True`` only for the vetted ``torch._utils._rebuild*`` reconstructor family.
    """

    return str(getattr(func, "__module__", "") or "") == "torch._utils" and str(
        getattr(func, "__name__", "") or ""
    ).startswith("_rebuild")


def _is_torch_reachable_callable(func: Any) -> bool:
    """Return whether a REDUCE target is a torch callable reachable by this unpickler.

    Covers BOTH routes a torch callable can take onto the pickle stack:

    * ``find_class`` -- a torch-owned function (real ``__module__`` inside the torch
      package), e.g. the ``torch._utils._rebuild*`` reconstructors; and
    * ``_safe_getattr`` -- a method/function DESCRIPTOR fetched off one of the admitted
      torch C holder classes. Those descriptors carry ``__module__ is None``, so a
      ``__module__``-only test MISSES them, and they are the second half of the mediation
      class: ``getattr(torch._C.TensorBase, "new_empty")`` applied to a small tensor
      rebuilt from an embedded blob amplifies a ~1.7 KiB pickle into an arbitrarily large
      UNINITIALIZED tensor. Ownership is therefore read from ``__objclass__`` /
      ``__self__`` -- the descriptor's real owner -- as well as ``__module__``.

    Parameters
    ----------
    func:
        Callable sitting on the pickle stack as a REDUCE target.

    Returns
    -------
    bool
        ``True`` when the callable belongs to torch by module, owning class, or binding.
    """

    if _is_torch_owned(func):
        return True
    for owner in (getattr(func, "__objclass__", None), getattr(func, "__self__", None)):
        if owner is None:
            continue
        # IDENTITY, never ``in``: ``owner`` is attacker-reachable, and a value that is
        # UNHASHABLE (an ``ndarray``) or carries a broadcasting ``__eq__`` makes a
        # frozenset membership test RAISE instead of decide -- a security belt that
        # crashes is a belt that did not answer. Same rule the numpy/bytes branch of
        # ``_alloc_refusal_reason`` follows.
        if any(owner is allowed for allowed in _ALLOWED_GETATTR_OBJECTS):
            return True
        if _is_torch_owned(owner):
            return True
        if isinstance(owner, type) and _is_torch_owned(owner):
            return True
        if _is_torch_owned(type(owner)):
            return True
    return False


def _requests_integer_sized_buffer(args: Any) -> bool:
    """Return whether ``args`` asks a ``bytes``/``bytearray`` ctor for a SIZE, not data.

    ``bytes(b"...")`` / ``bytearray(b"...")`` copy a buffer already present in the stream
    (proportional, harmless). ``bytes(N)`` / ``bytearray(N)`` allocate N zero bytes from a
    handful of pickle bytes -- an out-of-memory DoS with no buffer to pay for it.

    Parameters
    ----------
    args:
        Argument tuple popped for the REDUCE / NEWOBJ opcode.

    Returns
    -------
    bool
        ``True`` when the sole argument is an integer size.
    """

    if not isinstance(args, tuple) or len(args) != 1:
        return False
    return isinstance(args[0], int) and not isinstance(args[0], bool)


def _is_sized_module_construction(func: Any, args: Any, kwargs: Any) -> bool:
    """Return whether a ``torch.nn.Module`` is being constructed WITH pickle-supplied sizes.

    Constructing a module runs its ``__init__``, which allocates PARAMETERS on the belt's
    behalf -- ``torch.nn.modules.linear.Linear(65536, 65536)`` is ~16 GiB from a 48-byte
    REDUCE, and the module class itself is an ordinary torch-owned data type that
    ``find_class`` admits as an inert ``module_type`` reference.

    The rule is ARGUMENT-BEARING construction only, and that boundary is load-bearing in
    both directions. A pickled module INSTANCE always arrives as ``cls.__new__(cls)`` with
    an EMPTY argument tuple followed by a BUILD (protocol >= 2 ``copyreg.__newobj__``), and
    real artifacts do exactly that -- a ``splice_module`` intervention helper round-trips
    ``torch.nn.modules.linear.Identity()``. A zero-argument construction also cannot be
    SCALED by the attacker: whatever it allocates is fixed by the class. Every amplifying
    construction, by contrast, must pass the sizes through the pickle.

    Parameters
    ----------
    func:
        Class the opcode is about to construct.
    args:
        Positional argument tuple popped for it.
    kwargs:
        Keyword argument mapping popped for it (NEWOBJ_EX only), or ``None``.

    Returns
    -------
    bool
        ``True`` for an argument-bearing ``nn.Module`` construction.
    """

    if not isinstance(func, type):
        return False
    try:
        if not issubclass(func, torch.nn.Module):
            return False
    except TypeError:  # pragma: no cover - defensive; issubclass on odd metaclasses.
        return False
    has_positional = bool(args) if isinstance(args, (tuple, list)) else args is not None
    has_keyword = bool(kwargs) if isinstance(kwargs, dict) else kwargs is not None
    return has_positional or has_keyword


def _alloc_refusal_reason(func: Any, args: Any, kwargs: Any = None) -> str | None:
    """Return why calling ``func(*args)`` would MEDIATE an attacker-sized allocation.

    ``_is_alloc_constructor_type`` inspects the TYPE BEING CONSTRUCTED, so it is blind to
    any admitted callable that constructs on that type's behalf: at REDUCE time the stack
    holds the MEDIATOR, not the allocating type. This closes that indirection CLASS.

    Enumerated mediators (r6; every ``_SAFE_EXPLICIT_GLOBALS`` entry and every other
    surface ``find_class`` admits was assessed -- see the audit ledger):

    * numpy ``_reconstruct`` -- ``_reconstruct(numpy.ndarray, (N,), b"b")`` runs
      ``ndarray.__new__`` and returns an N-element UNINITIALIZED array; the bare
      ``numpy.ndarray`` REDUCE the belt already refuses is reached through a plain
      allowlisted FUNCTION. An 88-byte pickle yielded a 2,000,000-element array; at
      N=2e9 roughly 16 GiB of uninitialized heap, allocated before any structural check.
      The mediated TYPE (``args[0]``) is run through the existing belt policy.
    * torch-owned callables reached as REDUCE targets -- ``_safe_getattr`` admits
      ``getattr(torch._C._VariableFunctionsClass, "empty")`` because a legit stream stores
      that reference as a VALUE, but a following REDUCE CALLS it:
      ``torch.empty(N)`` / ``empty_strided`` / ``zeros`` / ``rand`` allocate an
      attacker-sized tensor (``empty`` uninitialized) from a 73-byte pickle. The same
      surface reaches tensor-constructor METHOD DESCRIPTORS: a small tensor legitimately
      rebuilt from an embedded blob, fed to ``getattr(TensorBase, "new_empty")``, turned a
      1,681-byte pickle into a 16,000,000-byte uninitialized tensor. Only the vetted
      ``torch._utils._rebuild*`` reconstructors may be invoked.
    * ``bytes`` / ``bytearray`` given an integer SIZE rather than a buffer -- a direct
      zero-fill allocator the type-keyed belt never covered.
    * an ARGUMENT-BEARING ``torch.nn.Module`` construction, whose ``__init__`` allocates
      attacker-sized parameters (``_is_sized_module_construction``).

    Parameters
    ----------
    func:
        Callable / class the opcode is about to invoke.
    args:
        Argument tuple the opcode popped for it.
    kwargs:
        Keyword argument mapping the opcode popped (NEWOBJ_EX only), or ``None``.

    Returns
    -------
    str | None
        A refusal reason, or ``None`` when the call is not an allocation mediator.
    """

    if _NUMPY_RECONSTRUCT_FUNCS and any(func is helper for helper in _NUMPY_RECONSTRUCT_FUNCS):
        # Fail CLOSED on a malformed arg tuple: a security belt never admits a call it
        # cannot inspect.
        if not isinstance(args, tuple) or not args or _is_alloc_constructor_type(args[0]):
            mediated_type = args[0] if isinstance(args, tuple) and args else None
            return (
                "numpy _reconstruct mediating an allocation-constructor "
                f"({getattr(mediated_type, '__module__', '?')}."
                f"{getattr(mediated_type, '__qualname__', mediated_type)}) "
                "-- _reconstruct runs ndarray.__new__ on a pickle-supplied shape and "
                "returns attacker-sized, uninitialized-heap memory, reaching the same "
                "allocation the bare numpy.ndarray REDUCE is refused for"
            )
        return None
    if callable(func) and not isinstance(func, type) and _is_torch_reachable_callable(func):
        if not _is_vetted_torch_reduce_callable(func):
            return (
                "a torch callable "
                f"{getattr(func, '__module__', '') or getattr(func, '__objclass__', '')}."
                f"{getattr(func, '__name__', func)!r} invoked by a REDUCE -- torch tensor "
                "factories (empty / empty_strided / zeros / rand / ...) and tensor "
                "constructor METHODS (new_empty / new_zeros / ...) allocate attacker-sized "
                "tensors on the belt's behalf. A legit stream resolves torch callables as "
                "inert VALUES and only ever CALLS the vetted torch._utils._rebuild* "
                "reconstructors"
            )
        return None
    # IDENTITY, never ``in``/``==``: ``func`` is attacker-controlled and a resolved value
    # with a broadcasting ``__eq__`` (an ``ndarray``) would make a membership test raise
    # instead of decide.
    if (func is bytes or func is bytearray) and _requests_integer_sized_buffer(args):
        return (
            f"{getattr(func, '__name__', func)}(N) with an integer SIZE -- it allocates N "
            "zero bytes from a handful of pickle bytes (an out-of-memory DoS); a legit "
            "stream only ever copies a buffer already present in the stream"
        )
    if _is_sized_module_construction(func, args, kwargs):
        return (
            "an argument-bearing torch.nn.Module construction "
            f"{getattr(func, '__module__', '')}.{getattr(func, '__qualname__', func)} -- "
            "its __init__ allocates attacker-sized PARAMETERS on the belt's behalf "
            "(nn.Linear(65536, 65536) is ~16 GiB from a 48-byte pickle). A pickled module "
            "INSTANCE always arrives as a zero-argument cls.__new__(cls) plus a BUILD, so "
            "this refuses nothing a real artifact does"
        )
    return None


def _is_torch_owned(obj: Any) -> bool:
    """Return whether a resolved object genuinely lives under the ``torch`` package.

    Trust in the torch branch of ``find_class`` is decided by the RESOLVED object's
    real ``__module__``, NEVER by the attacker-controlled pickled path. Pickle's
    ``STACK_GLOBAL`` / ``GLOBAL`` resolution attribute-walks a DOTTED name off the
    named module, so ``module="torch"`` with ``name="utils.collect_env.subprocess.Popen"``
    escapes the torch package into stdlib ``subprocess`` and returns
    ``subprocess.Popen`` (real module ``subprocess``) -- a ``type`` that a following
    pickle ``REDUCE`` would INVOKE to spawn a process (RCE). Requiring the resolved
    type's real module to be within ``torch`` denies that escape, mirroring the
    ``_is_torchlens_owned`` real-module check.
    """

    owner = str(getattr(obj, "__module__", "") or "")
    return owner == "torch" or owner.startswith("torch.")


_SAFE_EXPLICIT_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("collections", "Counter"),
        # Pure-data builtin constructors / base type. Explicitly EXCLUDES getattr
        # / eval / exec / __import__ / compile / open and every other code-exec
        # builtin. ``object`` is the base type used by default ``__reduce_ex__``.
        ("builtins", "object"),
        ("builtins", "list"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "dict"),
        ("builtins", "tuple"),
        ("builtins", "bytearray"),
        ("builtins", "bytes"),
        ("builtins", "complex"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "str"),
        ("builtins", "bool"),
        ("builtins", "slice"),
        # The ``Ellipsis`` singleton (``...``). Pickle resolves it as a bare
        # ``builtins.Ellipsis`` global (no dedicated opcode, unlike ``None``);
        # resolving the name just returns the existing singleton object, so this
        # is strictly inert -- no different in kind from admitting the ``slice``
        # constructor above.
        ("builtins", "Ellipsis"),
        # torch C callable-holder classes (used as getattr targets and, rarely,
        # as bare globals). They are types; returning them is inert -- the only
        # exploit path (getattr pivot) is closed by ``_safe_getattr``.
        ("torch._C", "TensorBase"),
        ("torch._C", "_TensorBase"),
        ("torch._C", "_VariableFunctionsClass"),
        # NumPy numeric-array reconstruction helpers (raw-input ndarrays and
        # preview-backend numeric payloads). These rebuild arrays from a raw
        # numeric buffer / dtype and execute no attacker code for numeric dtypes.
        # Both the legacy (<2.0) and new (>=2.0) private module paths are covered.
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy._core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        # NumPy scalar reconstruction (``np.float64(...)``, ``np.int64(...)``, a
        # bare numpy scalar living anywhere in user metadata -- e.g. an
        # ``input_structure`` dataclass field). Unlike ``_reconstruct`` /
        # ``_frombuffer`` (an attacker-sized shape/count), ``scalar(dtype, obj)``
        # is bounded by the fixed dtype itemsize (at most a few bytes): there is
        # no attacker-controlled allocation size to refuse.
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "scalar"),
    }
)

# Preview-backend array module roots whose resolved *types* / dtype instances are
# admitted (MLX / Paddle / JAX numeric array + dtype reconstruction). Only ``type``
# objects and framework dtype-like instances are admitted; callables are not.
_PREVIEW_ARRAY_MODULE_ROOTS: tuple[str, ...] = ("mlx.core",)


def _build_allowed_getattr_objects() -> frozenset[Any]:
    """Resolve the torch C callable-holder classes admitted as getattr targets.

    Returns
    -------
    frozenset[Any]
        The torch C classes off which ``getattr`` may legitimately fetch a
        tensor-method reference. Missing names (torch version drift) are skipped.
    """

    objects: list[Any] = []
    for name in ("TensorBase", "_TensorBase", "_VariableFunctionsClass"):
        holder = getattr(torch._C, name, None)
        if isinstance(holder, type):
            objects.append(holder)
    return frozenset(objects)


_ALLOWED_GETATTR_OBJECTS: frozenset[Any] = _build_allowed_getattr_objects()


def _safe_getattr(obj: Any, name: Any) -> Any:
    """Restricted ``getattr`` admitted in place of ``builtins.getattr``.

    Legitimate bundle metadata reconstructs torch C tensor-method references via
    ``getattr(torch._C.TensorBase, "sum")`` and similar. Admitting raw
    ``builtins.getattr`` would enable the classic pickle pivot
    ``getattr(<any function>, "__globals__")`` -> ``eval`` / ``exec``. This
    wrapper structurally blocks that pivot by restricting the target object to
    the enumerated torch C callable-holder classes; any other target -- including
    an already-resolved torchlens recipe function -- fails closed.

    Restricting the TARGET is not enough (round-5): the admitted
    ``_VariableFunctionsClass`` holder also exposes side-effecting file-I/O
    builtins -- notably ``torch.from_file`` -- so ``getattr(_VariableFunctionsClass,
    "from_file")`` would return a callable that CREATES/TRUNCATES or READS an
    arbitrary file when used as the callable of a following pickle ``REDUCE``. The
    resolved callable is therefore additionally gated through the shared
    callable-safety policy (``is_pure_forward_callable``): a pure tensor-method /
    factory reference (``sum`` / ``from_numpy`` / ``frombuffer``) is returned, while
    ``from_file`` and any other side-effecting builtin fail closed. The policy is
    imported lazily to keep this early-imported security front door free of
    import-order coupling.

    Parameters
    ----------
    obj:
        Attribute-access target supplied by the pickle stream.
    name:
        Attribute name supplied by the pickle stream.

    Returns
    -------
    Any
        ``getattr(obj, name)`` for an admitted torch C target and a non-side-effecting
        result.

    Raises
    ------
    pickle.UnpicklingError
        If ``obj`` is not an admitted torch C callable-holder class, ``name`` is not
        a string, or the resolved attribute is a side-effecting callable.
    """

    if not isinstance(name, str) or obj not in _ALLOWED_GETATTR_OBJECTS:
        raise pickle.UnpicklingError(
            f"Blocked getattr during bundle metadata unpickle: getattr on {obj!r} is not permitted."
        )
    resolved = getattr(obj, name)
    if callable(resolved):
        from .._utils._callable_safety import is_pure_forward_callable

        if not is_pure_forward_callable(resolved):
            raise pickle.UnpicklingError(
                "Blocked side-effecting torch callable during bundle metadata "
                f"unpickle: getattr(..., {name!r}) is not a pure forward/tensor op."
            )
    return resolved
