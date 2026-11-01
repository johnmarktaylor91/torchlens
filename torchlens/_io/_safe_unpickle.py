"""Restricted unpickler for TorchLens portable-bundle metadata.

A loaded ``.tlspec`` bundle is UNTRUSTED input. Its ``metadata.pkl`` payload was
historically read with an *unrestricted* :class:`pickle.Unpickler`, so a crafted
bundle whose ``metadata.pkl`` carried a ``__reduce__`` / ``os.system`` gadget
executed arbitrary code at ``pickle.load`` time -- BEFORE the resolver / callable
safety defenses (:mod:`torchlens._utils._callable_safety` and the runnable
resolver) could ever fire. That is a load-time RCE sitting in front of the entire
prior hardening effort.

This module closes that front door with the standard safe-unpickle defense: a
class allowlist (default-deny), mirroring the intent of ``torch.load(...,
weights_only=True)``. Only the exact set of classes / values that legitimately
appear in TorchLens ``Trace`` metadata is admitted; every other global -- and in
particular any ``os`` / ``sys`` / ``subprocess`` / ``posix`` / ``nt`` /
``builtins`` code-exec callable, ``importlib``, or ``torch.serialization``
loader -- raises :class:`pickle.UnpicklingError` and executes nothing.

Allowlist policy (fail-closed):

* Objects resolved from the ``torchlens`` package are admitted when they are a
  ``type`` ON THE POSITIVE ``_SAFE_TORCHLENS_TYPES`` ALLOWLIST (vetted-inert metadata
  dataclasses / enums / accessors), or a VETTED-INERT first-party callable: a PUBLIC,
  side-effect-free ``torchlens.*`` facet recipe / transform / helper (e.g.
  ``torchlens._utils.display.identity``), decided by
  ``torchlens._utils._callable_safety.is_inert_first_party_callable``. A torchlens
  ``type`` is NO LONGER admitted unconditionally: a pickle ``REDUCE`` INVOKES the
  admitted type (or its reconstructed instance) with attacker args, so an
  import-sink type such as ``torchlens.intervention._spec_values.LazyImportRef`` (whose
  ``__call__`` resolves an arbitrary import path from its own pickled field) is a
  load-time RCE and is DENIED by the default-deny type allowlist. Trust is
  decided by the resolved object's real module, never the pickled path, so a key
  that walks attributes off a torchlens module to reach ``os.system`` (real module
  ``posix``) is denied. Crucially, a pickle ``REDUCE`` INVOKES the admitted callable
  with attacker args, so the pre-r21 blanket "any torchlens-owned callable" policy
  was a load-time RCE: it trusted the PRIVATE ``torchlens.report._environment._module_is_installed``
  (which ``importlib.import_module``s its argument). The narrowed gate is
  deterministic (public-name + side-effect-name guard, no live-registry lookup) and
  admits our vetted first-party code without admitting that gadget.
* A FOREIGN facet-recipe callable (a user/test ``@tl.facets.register`` recipe
  embedded by identity in a ``FacetRegistrySnapshot``) is handled DETERMINISTICALLY
  by the same trust model the resolver uses. Known-dangerous modules (``os`` /
  ``sys`` / ``subprocess`` / ``pickle`` / ``builtins`` / ``importlib`` / ...) are
  hard-denied unconditionally. Otherwise, under an explicit trust opt-in
  (``trust_custom_callables`` / ``allowed_custom_callable_modules``) the module is
  imported and the real callable resolved; by default it is LOAD-TOLERATED as an
  inert ``_DeferredForeignCallable`` placeholder -- the foreign module is NEVER
  imported at unpickle, the trace still loads structurally, and any attempt to CALL
  the recipe (facet computation, or a ``__reduce__`` gadget) fails closed. Admission
  depends only on ``(module, name)`` + trust flags, never on process state (an
  import-time / live-registry snapshot was the ordering-dependent load-failure bug).
* A curated set of safe stdlib data/value types (``collections.OrderedDict``, the
  pure-data builtin container types plus ``object``, and the NumPy numeric-array
  reconstruction helpers).
* Any object resolved from the ``torch`` package that is a DATA ``type`` (tensor /
  Module / storage classes) or a ``torch`` dtype / layout / memory_format
  *instance*, plus the pure ``torch._utils._rebuild_*`` reconstructors. Resolving
  a global only references it; constructing a torch data type executes no attacker
  code. Torch's I/O-capable types are the exception -- constructing
  ``torch._C.PyTorchFileReader`` / ``PyTorchFileWriter`` / ``FileCheck`` /
  ``torch.package.PackageImporter`` / ``torch.serialization.*`` / ``torch.jit.*``
  opens/reads/writes an attacker-named filesystem path (arbitrary file access +
  zip-bomb DoS) at unpickle time, so these TYPES are denied by a module-prefix +
  structural-name guard aligned with torch's own ``weights_only`` exclusions.
  Non-type torch callables (e.g. ``torch.serialization.load``) are denied.
* Preview-backend (MLX / Paddle / JAX) numeric array/dtype ``type`` objects and
  framework dtype-like instances.
* ``builtins.getattr`` is legitimately used to reconstruct references to torch C
  tensor methods (``getattr(TensorBase, "sum")``). It is admitted only through a
  *safe* wrapper that restricts the target object to the enumerated torch C
  callable-holder classes, which structurally blocks the classic
  ``getattr(obj, "__globals__")`` pivot to ``eval`` / ``exec``.
* ``torch.storage._load_from_bytes`` (the storage leg of torch's default tensor
  reduction) runs a nested ``torch.load`` over bytes taken from the file. It is
  REFUSED by name: no file bytes ever reach torch's unpickler. A plain tensor that
  genuinely lives in metadata (an annotation value, a small raw input/output) is
  written by the canonical metadata pickler through the TorchLens-owned inline
  reduction ``_inline_tensor.rebuild_inline_tensor`` (raw bytes, dtype, shape,
  ``requires_grad``, device), admitted here by exact pair; it allocates exactly the
  bytes present in the stream into a non-resizable storage. A file written by an
  earlier TorchLens that embedded a tensor storage refuses typed, with a re-save
  remedy.

If a future legitimate class is not yet covered, loading it fails closed with a
clear error; we NEVER admit a code-exec gadget to make a load succeed.

STRICT-SUBSET OF THE TORCH ``weights_only`` BASELINE (r49 storage, r63 tensor/ndarray).
This unpickler's effective admit+CONSTRUCT surface is a documented STRICT SUBSET of
torch's ``_weights_only_unpickler._get_allowed_globals()`` baseline. The invariant is
about CONSTRUCTION, NOT resolution: the tensor / storage / ndarray TYPE OBJECTS stay
RESOLVABLE at ``find_class`` (legit metadata references ``torch.Tensor`` /
``torch.FloatTensor`` / ``torch.nn.Parameter`` / ``numpy.ndarray`` as inert dtype /
module_type values -- resolving a global only references it), but their BARE
CONSTRUCTION from pickle-supplied args is DENIED. Two layers enforce it:
(1) torch STORAGE constructors are denied by IDENTITY at ``find_class``
(``_is_torch_storage_type``); an inline metadata tensor arrives through
``rebuild_inline_tensor`` as a finished tensor over a non-resizable storage, and the
embedded-storage route ``torch.storage._load_from_bytes`` is refused by name.
(2) the opcode-level belt: a BUILD may not apply a ``__setstate__`` / storage-rebind to a
torch Tensor / Storage / Parameter, and a REDUCE / NEWOBJ / NEWOBJ_EX / INST / OBJ may
not CONSTRUCT (or mediate) an attacker-sized ALLOCATION type
(``_is_alloc_constructor_type`` / ``_alloc_refusal_reason``). The belt runs on a static
pre-scan of the opcode stream plus call-time guards around every callable a REDUCE
invokes (:mod:`torchlens._io._pickle_prescan`), so the load itself runs on the C
``pickle.Unpickler``; a stream the pre-scan cannot classify runs through the per-opcode
pure-Python ``_PurePythonBundleUnpickler`` with the same belt. Enforced by
``tests/test_r49_unpickler_weights_only_baseline.py`` and ``tests/test_r63_construct_deny.py``;
the anti-drift invariant is that the set of admitted ``(module, name)`` for which this
unpickler CONSTRUCTS a storage / tensor / ndarray object is EMPTY -- a
CONSTRUCTOR-SEMANTICS assertion, NOT a string-subset of ``_get_allowed_globals()``.
"""

from __future__ import annotations

import io
import pickle
import struct
from collections.abc import Callable, Collection
from dataclasses import dataclass
from types import ModuleType
from typing import Any, BinaryIO

import torch

from .._utils._callable_safety import (
    _ALLOWED_STDLIB_ROOTS as _CALLABLE_ALLOWED_STDLIB_ROOTS,
    _APPLIANCE_MODULES as _CALLABLE_APPLIANCE_MODULES,
    _DENIED_MODULES as _CALLABLE_DENIED_MODULES,
    _STDLIB_AND_BUILTIN_TOP_LEVEL as _CALLABLE_STDLIB_AND_BUILTIN_TOP_LEVEL,
    is_denied_stdlib_or_builtin_module,
)
from ._durability import _stream_bytes_left
from ._inline_tensor import INLINE_TENSOR_GLOBAL, rebuild_inline_tensor
from ._pickle_prescan import (
    _CORRUPT_PREFIX,
    _EXT_REFUSAL,
    _FRAME_REFUSAL,
    GuardedCallable,
    PrescanPlan,
    StreamNeedsInterpreter,
    prescan_stream,
    scan_opcodes,
)
from ._pre3_module_paths import refuse_pre_3_module_path
from .format_errors import ArtifactTooOldError

__all__ = ["SafeBundleUnpickler"]


# The callable resolver owns the security vocabulary. This front door imports
# the SAME frozen object so a denylist addition cannot land on only one path.
_DENIED_FOREIGN_MODULES = _CALLABLE_DENIED_MODULES


def _module_denied(module: str) -> bool:
    """Return whether ``module`` is (nested under) a hard-denied dangerous module."""

    return any(
        module == denied or module.startswith(denied + ".") for denied in _DENIED_FOREIGN_MODULES
    )


# STRUCTURAL close of the denylist-completeness class (r31). The detector,
# carve-outs, and predicate are imported from ``utils._callable_safety`` so the
# unpickler and callable resolver make one decision from one authority. The explicit
# ``_DENIED_FOREIGN_MODULES`` alias stays as belt-and-suspenders; this positive rule
# closes the CLASS: any FOREIGN global whose real top-level module is a Python
# STANDARD-LIBRARY or BUILTIN module is denied EVEN under trust. The pure-forward
# ``operator`` / ``_operator`` root is carved out; non-stdlib torch / torchlens /
# numpy / user packages are naturally allowed (not in the detector set). NOTE: this is
# applied ONLY in the FOREIGN tail of ``find_class`` -- the ``torch`` / ``torchlens``
# / preview / ``_SAFE_EXPLICIT_GLOBALS`` branches (which legitimately resolve
# ``collections`` / ``builtins`` pure-data globals) return BEFORE the tail.
_ALLOWED_STDLIB_ROOTS = _CALLABLE_ALLOWED_STDLIB_ROOTS
_STDLIB_AND_BUILTIN_TOP_LEVEL = _CALLABLE_STDLIB_AND_BUILTIN_TOP_LEVEL
_stdlib_or_builtin_denied = is_denied_stdlib_or_builtin_module


def _name_has_dunder_walk(name: str) -> bool:
    """Return whether a dotted pickled ``name`` walks through a dunder attribute.

    Pickle protocol >= 4 attribute-walks a DOTTED ``name`` off the resolved module
    (``STACK_GLOBAL``). Every escape from a value/callable object into a module's
    globals / builtins / class / reduce machinery traverses a DUNDER attribute
    segment: ``<func>.__globals__`` returns a module-globals MAPPING,
    ``__builtins__`` the builtins, and ``__dict__`` / ``__class__`` / ``__reduce__``
    / ``__subclasses__`` / ``__mro__`` / ``__base__`` the reflective pivots. A
    LEGITIMATE pickled global name is a module-level class/function qualname -- at
    most an ``Outer.Inner`` nested-class dotted qualname -- so any dunder segment is
    a walk vector and is refused, EXCEPT a small allowlist of provably-inert
    introspection dunders (``__name__`` / ``__doc__`` / ``__qualname__``) that
    resolve to a plain ``str``/``None`` and cannot pivot into globals/builtins/reduce
    machinery. (A trusted appliance recipe legitimately reads e.g.
    ``torchlens.neuro.__name__``; the resolved-mapping backstop below still catches
    any dict/globals escape the name check does not.) Used to gate the trusted-foreign
    branch, where the resolved-real-module denylist recheck is defeated by a resolved
    module-globals mapping (a ``dict`` has no ``__module__``, so the recheck falls back
    to the non-denied pickled module).
    """

    inert_introspection_dunders = {"__name__", "__doc__", "__qualname__"}
    return any(
        segment.startswith("__") and segment not in inert_introspection_dunders
        for segment in name.split(".")
    )


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


@dataclass(frozen=True)
class _DeferredForeignCallable:
    """Inert placeholder for a foreign recipe callable retained on an untrusted load.

    A saved ``FacetRegistrySnapshot`` may reference a user/test-registered facet
    recipe by identity (``@tl.facets.register``). Importing that foreign module at
    unpickle time would execute its top-level code -- exactly the load-time RCE
    surface this module closes. So under a default (untrusted) load the reference
    is LOAD-TOLERATED as this placeholder instead of imported: the trace loads
    structurally (the recipe reference is preserved for inspection / re-save), but
    the foreign module is NEVER imported and any attempt to CALL the recipe (facet
    computation, or a ``__reduce__`` construction gadget) fails closed.

    This mirrors the resolver's load-tolerate / execute-deny least-privilege split
    (``resolve_function_registry_key``): a foreign custom callable is retained as an
    unresolved reference on load and only imported+resolved under an explicit trust
    opt-in (``trust_custom_callables`` / ``allowed_custom_callable_modules``).

    Admission here is DETERMINISTIC: it depends only on the pickled ``(module,
    name)`` and the caller's trust flags -- never on process state (whether the
    module happens to be imported, or whether the recipe is currently in the live
    registry), which was the ordering-dependent load-failure bug.
    """

    module: str
    qualname: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Fail closed: an untrusted foreign recipe is never executed.

        Raises
        ------
        pickle.UnpicklingError
            Always. If invoked during unpickle (a ``__reduce__`` gadget) the load
            fails closed; if invoked later (facet computation) it signals that a
            trust opt-in is required to run the foreign recipe.
        """

        raise pickle.UnpicklingError(
            "Refusing to execute foreign custom callable "
            f"{self.module}:{self.qualname} resolved from an untrusted bundle. "
            "Pass trust_custom_callables=True or allowed_custom_callable_modules "
            "to import and run a trusted foreign recipe."
        )


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


def _is_torchlens_owned(obj: Any) -> bool:
    """Return whether a resolved object genuinely lives under the ``torchlens`` package.

    Trust for a callable is decided by the RESOLVED object's real ``__module__``,
    never by the pickled import path. This closes the "walk ``os.system`` off a
    torchlens module" pivot (``getattr(torchlens._io.tlspec, 'os').system`` has
    ``__module__ == 'posix'`` and is therefore denied) -- mirroring the resolver's
    identically-named check.
    """

    owner = str(getattr(obj, "__module__", "") or "")
    return owner == "torchlens" or owner.startswith("torchlens.")


# Extras-gated "appliance" subpackages whose ``__init__`` imports heavy FOREIGN
# third-party dependencies (rsatoolbox / brainscore_core for ``torchlens.neuro``,
# IPython / jupyter_client for ``torchlens._notebook``). Unlike the rest of the
# first-party ``torchlens.*`` namespace -- which is safe to import + inspect
# because it runs only our own already-installed code (see
# ``_is_torchlens_owned`` above) -- importing one of these two runs foreign
# top-level code with NO trust opt-in. A bundle-supplied global naming one of
# them (or a submodule) must therefore be EXCLUDED from the "torchlens is our
# own code" fast path below and instead fall through to receive the exact same
# deny-by-default / trust-opt-in / load-tolerate treatment already applied to a
# genuinely foreign module at the tail of ``find_class`` -- never passed to
# ``super().find_class`` (which imports the module) to make that determination.
_TORCHLENS_APPLIANCE_MODULES = _CALLABLE_APPLIANCE_MODULES


def _is_torchlens_appliance_module(module: str) -> bool:
    """Return whether ``module`` is (nested under) an extras-gated appliance package."""

    return any(
        module == appliance or module.startswith(appliance + ".")
        for appliance in _TORCHLENS_APPLIANCE_MODULES
    )


# Positive allowlist of VETTED-INERT first-party ``(module, qualname)`` DATA types
# (default-deny). The torchlens ``type`` branch of ``find_class`` historically
# admitted ANY resolved ``torchlens.*`` type unconditionally on the premise that a
# resolved type is "reconstructed via normal unpickling, never INVOKED with attacker
# args". That premise is FALSE: a pickle ``REDUCE`` may INVOKE any admitted type
# (its instance, or the type itself) with attacker-chosen arguments. The concrete
# load-time RCE (r23): ``torchlens.intervention._spec_values.LazyImportRef`` is a frozen
# dataclass whose ``__call__(import_path, ...)`` resolves an ARBITRARY import path
# under a trust flag read from its OWN pickled field, so a crafted ``metadata.pkl``
# reconstructs ``LazyImportRef(import_path="os:system", trust_custom_callables=True)``
# and REDUCE-invokes it -> ``os.system`` on plain ``tl.load``.
#
# This closes that with the same default-deny discipline as the r22 callable gate: a
# torchlens type is admitted ONLY if it is on this frozen allowlist of pure DATA
# types -- dataclasses / enums / NamedTuples / plain accessor+value classes whose
# reconstruction runs no import / exec / filesystem / process side effect and which
# is NOT an import-sink / instance-callable gadget. Every other torchlens type --
# above all ``LazyImportRef`` and any callable/import-sink/exec type -- is DENIED.
#
# Derivation (default-deny): the set is the vetted-inert DATA-type inventory of the
# fixed set of first-party DATA modules that make up a scrubbed ``Trace``'s
# ``metadata.pkl`` (data_classes.* / ir.* / intervention.types / intervention._super
# / semantic.facets / quantities / capture.config+stop / _trace_state / _io.BlobRef /
# visualization.code_panel). It was materialized by enumerating those modules and
# admitting each class OWNED by the module (real ``__module__`` match, so no aliasing
# to a logic module) that (a) is NOT instance-callable and (b) carries NO custom
# ``__reduce__`` / ``__reduce_ex__``, then EXCLUDING live-view / error / exception
# classes (``Facet`` / ``FacetView`` / ``MissingFacet`` / ``*Error`` / ...) that never
# appear in scrubbed metadata. ``Edit`` is the single vetted instance-callable
# exception (``__call__`` takes no attacker args; its ``factory`` is scrubbed to
# ``None`` before pickling, so it is not a reduce-invocable import sink). The result
# was cross-checked against the torchlens types the real unpickler resolves across the
# io / tlspec-runnable / semantic-facet / intervention / capture-oracle suites. The
# import-sink / logic modules -- above all ``torchlens.intervention._spec_values`` (home of
# ``LazyImportRef``) -- are NOT data modules and are absent, so ``LazyImportRef`` is
# denied on both counts (module not curated; instance-callable import sink). A legit
# type not yet covered fails CLOSED (a load error surfacing in a test) and is added
# only after the same vetting -- never by re-admitting the blanket "any torchlens
# type" rule.
_SAFE_TORCHLENS_TYPES: frozenset[tuple[str, str]] = frozenset(
    {
        # Trace runtime-state enum + capture config / stop directives.
        ("torchlens._trace_state", "TraceState"),
        ("torchlens._capture.config", "InternalCaptureConfig"),
        ("torchlens._capture.stop", "StopDirective"),
        # Portable I/O blob reference (NamedTuple of ids).
        ("torchlens._io", "BlobRef"),
        # Buffer / grad / param / op / layer / module data records + accessors.
        ("torchlens._data_classes.buffer", "Buffer"),
        ("torchlens._data_classes.buffer", "BufferAccessor"),
        ("torchlens._data_classes.derived_grad", "DerivedGradAccessor"),
        ("torchlens._data_classes.derived_grad", "DerivedGradRecord"),
        ("torchlens._data_classes.derived_grad", "IntermediateDerivedGradAccessor"),
        ("torchlens._data_classes.derived_grad", "IntermediateDerivedGradRecord"),
        ("torchlens._data_classes.func_call_location", "FuncCallLocation"),
        ("torchlens._data_classes.grad_fn", "GradFn"),
        ("torchlens._data_classes.grad_fn", "GradFnAccessor"),
        ("torchlens._data_classes.grad_fn", "GradFnCallAccessor"),
        ("torchlens._data_classes.grad_fn_call", "GradFnCall"),  # locked rename target
        ("torchlens._data_classes.layer", "Layer"),
        # The accessor classes live in _layer_accessors.py but pin ``__module__`` to the
        # layer module, so every artifact pickles them under these rows.
        ("torchlens._data_classes.layer", "LayerAccessor"),
        ("torchlens._data_classes.layer", "OpAccessor"),
        ("torchlens._data_classes.module", "HookInfo"),
        ("torchlens._data_classes.module", "Module"),
        ("torchlens._data_classes.module", "ModuleAccessor"),
        ("torchlens._data_classes.module", "ModuleCall"),
        ("torchlens._data_classes.module", "ModuleCallAccessor"),
        ("torchlens._data_classes.op", "GradientRecord"),
        ("torchlens._data_classes.op", "GradientRecordAccessor"),
        ("torchlens._data_classes.op", "Op"),
        ("torchlens._data_classes.param", "Param"),
        ("torchlens._data_classes.param", "ParamAccessor"),
        # Prehook-provenance snapshots (frozen dataclasses, inert data).
        ("torchlens._data_classes.prehook", "TensorInputObservation"),
        ("torchlens._data_classes.prehook", "ModuleInputSnapshot"),
        ("torchlens._data_classes.prehook", "PreHookEffect"),
        # Backward-pass records. __setstate__ is the same inert version-compat
        # field-refill pattern Trace/Op use (default_fill_state); no side effects.
        ("torchlens._data_classes.backward_pass", "BackwardPass"),
        ("torchlens._data_classes.backward_pass", "BackwardPassAccessor"),
        # DROP-gated primitive-profile records (pre-release switch only).
        ("torchlens._data_classes.aten_op", "AtenOp"),
        ("torchlens._data_classes.aten_op", "OpRef"),
        ("torchlens._data_classes.aten_op", "_ModePausedInteriorGap"),
        ("torchlens._data_classes.aten_op", "_PrimitiveOpProfile"),
        ("torchlens._ir.events", "_AtenTensorFact"),
        ("torchlens._ir.events", "_AtenExecutionContext"),
        # Trace + conditional-control-flow data classes.
        ("torchlens._data_classes.trace", "Conditional"),
        ("torchlens._data_classes.trace", "ConditionalAccessor"),
        ("torchlens._data_classes.trace", "ConditionalArm"),
        ("torchlens._data_classes.trace", "ConditionalEvent"),
        ("torchlens._data_classes.trace", "ConditionalRoleRef"),
        ("torchlens._data_classes.trace", "ResolvedPostprocessing"),
        ("torchlens._data_classes.trace", "ResolvedPreprocessing"),
        ("torchlens._data_classes.trace", "Trace"),
        # Bundle Super* aligned-view wrappers (locked ``NodeView`` rename target).
        ("torchlens.intervention._super.super_op", "SuperLayer"),
        ("torchlens.intervention._super.super_op", "SuperLayerAccessor"),
        ("torchlens.intervention._super.super_op", "SuperOp"),  # locked rename target
        ("torchlens.intervention._super.super_op", "SuperOpAccessor"),
        ("torchlens.intervention._super.super_op", "TraceAccessor"),
        # Intervention-spec DATA types (portable recipe descriptors, not runtime
        # callables). ``Edit`` is instance-callable but INERT: ``__call__``
        # takes no attacker args and its ``factory`` is scrubbed to ``None`` before
        # pickling, so it is not a reduce-invocable import sink.
        ("torchlens.intervention.types", "CapturedArgTemplate"),
        ("torchlens.intervention.types", "EdgeUseRecord"),
        ("torchlens.intervention.types", "FireRecord"),
        ("torchlens.intervention.types", "ForkFieldPolicy"),
        ("torchlens.intervention.types", "FrozenHookSpec"),
        ("torchlens.intervention.types", "FrozenInterventionSpec"),
        ("torchlens.intervention.types", "FrozenTargetSpec"),
        ("torchlens.intervention.types", "FrozenTargetValueSpec"),
        ("torchlens.intervention.types", "FunctionRegistryKey"),
        ("torchlens.intervention.types", "Edit"),
        ("torchlens.intervention.types", "HookSpec"),
        ("torchlens.intervention.types", "InterventionDecision"),
        ("torchlens.intervention.types", "InterventionSpec"),
        ("torchlens.intervention.types", "LiteralTensor"),
        ("torchlens.intervention.types", "LiteralValue"),
        ("torchlens.intervention.types", "ParentRef"),
        ("torchlens.intervention.types", "Relationship"),
        ("torchlens.intervention.types", "TargetSpec"),
        ("torchlens.intervention.types", "TargetValueSpec"),
        ("torchlens.intervention.types", "TensorSliceSpec"),
        ("torchlens.intervention.types", "Unsupported"),
        # IR container / container-registry structural descriptors + enums.
        ("torchlens._ir.container", "ContainerSpec"),
        ("torchlens._ir.container", "DataclassField"),
        ("torchlens._ir.container", "DictKey"),
        ("torchlens._ir.container", "HFKey"),
        ("torchlens._ir.container", "NamedField"),
        ("torchlens._ir.container", "TupleIndex"),
        ("torchlens._ir.container_registry", "ContainerLeafOccurrence"),
        ("torchlens._ir.container_registry", "ContainerRecord"),
        ("torchlens._ir.container_registry", "ContainerRegistry"),
        ("torchlens._ir.container_registry", "ContainerSnapshot"),
        ("torchlens._ir.container_registry", "FuncSite"),
        ("torchlens._ir.container_registry", "IdentityEntry"),
        ("torchlens._ir.container_registry", "ModelSite"),
        ("torchlens._ir.container_registry", "ModuleSite"),
        ("torchlens._ir.container_registry", "Phase"),
        ("torchlens._ir.container_registry", "Role"),
        ("torchlens._ir.container_registry", "WalkResult"),
        # Device / dtype / tensor / param reference value types.
        ("torchlens._ir.refs", "DeferredRef"),
        ("torchlens._ir.refs", "DeviceRef"),
        ("torchlens._ir.refs", "DtypeRef"),
        ("torchlens._ir.refs", "ParamRef"),
        ("torchlens._ir.refs", "ReservedLabel"),
        ("torchlens._ir.refs", "TensorRef"),
        # Physical-quantity value types.
        ("torchlens._quantities", "Bytes"),
        ("torchlens._quantities", "Duration"),
        ("torchlens._quantities", "Flops"),
        ("torchlens._quantities", "Macs"),
        ("torchlens._quantities", "Quantity"),
        ("torchlens._quantities", "_FloatQuantity"),
        ("torchlens._quantities", "_IntQuantity"),
        # Semantic facet snapshot + recipe descriptor data types.
        ("torchlens._semantic.facets", "AbsenceReason"),
        ("torchlens._semantic.facets", "FacetCapabilityFlags"),
        ("torchlens._semantic.facets", "FacetContribution"),
        ("torchlens._semantic.facets", "FacetFunction"),
        ("torchlens._semantic.facets", "FacetMenuItem"),
        ("torchlens._semantic.facets", "FacetRegistrySnapshot"),
        ("torchlens._semantic.facets", "FacetSpec"),
        ("torchlens._semantic.facets", "TransformPrimitive"),
        ("torchlens._semantic.facets", "_RegisteredRecipe"),
        # Visualization code-panel source text value type.
        ("torchlens._visualization.code_panel", "SourceText"),
    }
)


# Exact (module, name) globals resolved directly. Every entry is either a pure
# data container/value type or a pure tensor-reconstruction helper -- none can
# execute attacker code merely by being resolved or called on pickle-provided
# arguments.
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


# Base pure-Python opcode handlers (typeshed does not name them on ``pickle._Unpickler``).
_BASE_LOAD_BUILD = pickle._Unpickler.dispatch[pickle.BUILD[0]]
_BASE_LOAD_REDUCE = pickle._Unpickler.dispatch[pickle.REDUCE[0]]
_BASE_LOAD_NEWOBJ = pickle._Unpickler.dispatch[pickle.NEWOBJ[0]]
_BASE_LOAD_NEWOBJ_EX = pickle._Unpickler.dispatch[pickle.NEWOBJ_EX[0]]


def _exception_arose_outside_vm(exc: BaseException) -> bool:
    """Return whether any traceback frame of ``exc`` lies outside the unpickle VM.

    LOCUS discriminator for the load boundary: a VM-internal corruption failure (stack
    underflow, truncated read, missing memo key, bad utf-8, truncation) is raised by VM
    code in stdlib ``pickle.py``, the pre-scan module or this module, or by a C call that
    adds NO Python frame; its traceback never leaves those files. An APPLICATION /
    reconstruction failure (a REDUCE callable's body, a ``__setstate__``, a ``find_class``
    module import) necessarily executes at least one Python frame outside them. The file
    names come from live code objects, so the test matches the code that actually runs.
    """

    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename not in _VM_INTERNAL_CODE_FILES:
            return True
        tb = tb.tb_next
    return False


_VM_INTERNAL_CODE_FILES: frozenset[str] = frozenset(
    {
        # The stdlib pure-Python VM (pickle.py), via the executing code object.
        pickle._Unpickler.load.__code__.co_filename,
        # The static pre-scan and the call-time guards.
        prescan_stream.__code__.co_filename,
        # This module: the policy, the belt and the load boundaries.
        _exception_arose_outside_vm.__code__.co_filename,
    }
)


class _DenyMissingDispatch(dict):
    """Opcode dispatch table that raises ``UnpicklingError`` on an unknown opcode.

    The C ``pickle.Unpickler`` raises ``pickle.UnpicklingError`` ("invalid load key") on
    an unknown opcode; the pure-Python ``pickle._Unpickler`` raises a bare ``KeyError``
    from its ``dispatch[key]`` lookup. The interpreter keeps the C failure mode; the other
    corrupt-input facets are normalized by the load boundary (``_boundary``).
    """

    def __missing__(self, key: int) -> None:
        raise pickle.UnpicklingError(f"invalid load key, {chr(key)!r}.")


# --- Belt refusal text (one source of truth for the pre-scan, the guards and the ------
# --- pure-Python handlers) -------------------------------------------------------------

_ALLOC_TYPE_TAIL: dict[str, str] = {
    "REDUCE": (
        " (constructing it from pickle-supplied args allocates attacker-sized, "
        "uninitialized-heap memory -- an out-of-memory DoS and the source of an "
        "uninitialized-heap tensor; .tlspec payloads travel as BlobRefs and reconstruct "
        "through the wrapped _rebuild/_frombuffer helpers, which are functions, not types)."
    ),
    "NEWOBJ": ".",
    "NEWOBJ_EX": ".",
    "INST/OBJ": (
        " (constructing it from pickle-supplied args allocates attacker-sized, "
        "uninitialized-heap memory -- an out-of-memory DoS and the source of an "
        "uninitialized-heap tensor. .tlspec metadata is protocol >= 2 and never emits the "
        "legacy INST/OBJ opcodes; payloads travel as BlobRefs)."
    ),
}
_MEDIATED_TAIL: dict[str, str] = {
    "REDUCE": (
        ". The allocation-constructor belt inspects the TYPE being constructed, so an "
        "admitted callable that constructs on its behalf must be refused here; .tlspec "
        "payloads travel as BlobRefs."
    ),
    "NEWOBJ": ".",
    "NEWOBJ_EX": ".",
    "INST/OBJ": (
        ". The legacy INST/OBJ construction opcodes route through the same allocation "
        "belt as REDUCE/NEWOBJ; .tlspec payloads travel as BlobRefs."
    ),
}


def _construction_refusal(opcode: str, func: Any, args: Any, kwargs: Any = None) -> str | None:
    """Return why ``opcode`` may not construct ``func(*args, **kwargs)``, or ``None``.

    The allocation belt for every constructing opcode (REDUCE, NEWOBJ, NEWOBJ_EX, INST,
    OBJ): first the mediated-allocation class (``_alloc_refusal_reason``), then the
    allocation-constructor type class (``_is_alloc_constructor_type``). The pre-scan
    calls it with literal arguments, the call-time guard with the real ones, and the
    pure-Python handlers from their opcode methods, so the three paths cannot drift.
    """

    label = "INST/OBJ" if opcode in ("INST", "OBJ") else opcode
    mediated = _alloc_refusal_reason(func, args, kwargs)
    if mediated is not None:
        return (
            f"Blocked mediated allocation via {label} during bundle metadata unpickle: "
            f"{mediated}{_MEDIATED_TAIL[label]}"
        )
    if _is_alloc_constructor_type(func):
        return (
            "Blocked allocation-constructor (torch storage/tensor or numpy ndarray) via "
            f"{label} during bundle metadata unpickle: {getattr(func, '__module__', '')}."
            f"{getattr(func, '__qualname__', func)!r}{_ALLOC_TYPE_TAIL[label]}"
        )
    return None


def _build_refusal_text(module: str, qualname: str) -> str:
    """Return the BUILD refusal text for a torch tensor / storage class ``module.qualname``."""

    return (
        f"Blocked BUILD applied to a torch {module}.{qualname} during bundle metadata "
        "unpickle: a pickle BUILD invokes __setstate__ / storage rebind (Tensor.set_) on "
        "the target, which can rebind it onto an uninitialized or oversized storage. "
        ".tlspec metadata never BUILDs a torch tensor/storage/parameter (payloads are "
        "BlobRefs)."
    )


def _build_refusal(inst: Any) -> str | None:
    """Return the BUILD refusal for a torch tensor / storage target, else ``None``."""

    if isinstance(inst, torch.Tensor) or _is_torch_storage_instance(inst):
        return _build_refusal_text(type(inst).__module__, type(inst).__qualname__)
    return None


def _build_type_refusal(cls: type) -> str | None:
    """Return the BUILD refusal when instances of ``cls`` are torch tensors / storages."""

    if not isinstance(cls, type):
        return None
    if issubclass(cls, torch.Tensor) or _is_torch_storage_type(cls):
        return _build_refusal_text(cls.__module__, cls.__qualname__)
    return None


_Importer = Callable[[str, str], Any]


class _BundleAllowlistPolicy:
    """The ``find_class`` policy shared by the C-backed and pure-Python unpicklers.

    Refuses every global that only torchlens 2.x wrote with the one 2.x error, then
    gates every resolved global through the module allowlist. Anything outside the
    allowlist raises :class:`pickle.UnpicklingError` without importing or calling
    attacker-chosen code. ``_resolve_global`` is the policy; each concrete class wires
    it to the ``find_class`` hook of its base machine.
    """

    _trust_custom_callables: bool
    _allowed_custom_callable_modules: frozenset[str] | None

    def _init_policy(
        self,
        trust_custom_callables: bool,
        allowed_custom_callable_modules: Collection[str] | None,
    ) -> None:
        """Store the trust flags the policy consults for foreign callables."""

        self._trust_custom_callables = trust_custom_callables
        self._allowed_custom_callable_modules = (
            frozenset(allowed_custom_callable_modules)
            if allowed_custom_callable_modules is not None
            else None
        )

    def _import_global(self, module: str, name: str) -> Any:
        """Resolve ``module.name`` with the base machine's importer (concrete classes bind it)."""

        raise NotImplementedError

    def _boundary(self, run: Callable[[], Any]) -> Any:
        """Run ``run`` normalizing ONLY VM-internal corruption failures.

        The contract has TWO halves: (1) every corrupt / truncated / malformed STREAM
        failure surfaces as ``pickle.UnpicklingError`` (which the load sites translate
        into a clean :class:`~torchlens._io._errors.ArtifactError`); and (2) an
        exception raised by APPLICATION code the stream invokes (a REDUCE callable, a
        ``__setstate__``, a ``find_class`` module import) propagates RAW with its
        identity intact, so the load sites keep the original exception as the DIRECT
        ``__cause__`` of ``ArtifactError``. Classification is by LOCUS
        (``_exception_arose_outside_vm``), never by exception type: a failure whose
        traceback stays inside the unpickler modules is stream corruption; one that
        executed application frames keeps its identity. Every deliberate refusal
        already raises ``pickle.UnpicklingError`` and passes through unchanged, and
        ``except Exception`` preserves ``KeyboardInterrupt`` / ``SystemExit``.
        """

        try:
            return run()
        except (pickle.UnpicklingError, ArtifactTooOldError):
            # The earlier-major refusal is a typed load verdict, not corruption.
            raise
        except Exception as exc:  # noqa: BLE001 -- locus-classified; see docstring
            if _exception_arose_outside_vm(exc):
                raise
            raise pickle.UnpicklingError(
                f"corrupt or malformed pickle stream: {type(exc).__name__}: {exc}"
            ) from exc

    def _custom_module_trusted(self, module: str) -> bool:
        """Return whether a foreign module may be imported+resolved via trust flags.

        Parameters
        ----------
        module:
            Foreign pickled module path.

        Returns
        -------
        bool
            ``True`` if the module is explicitly allowlisted, or broad trust is on.
        """

        if self._allowed_custom_callable_modules is not None:
            return module in self._allowed_custom_callable_modules
        return self._trust_custom_callables

    def _find_torchlens_global(self, module: str, name: str, import_global: _Importer) -> Any:
        """Admit a torchlens-owned pickled global, allowlist first.

        ``_SAFE_TORCHLENS_TYPES`` is consulted BEFORE anything is imported or
        resolved; an allowlisted pair is admitted when it still resolves to a type.
        Every other pair is denied as a type (a pickle ``REDUCE`` INVOKES it with
        attacker args: ``LazyImportRef`` would be a load-time RCE) and admitted as a
        callable only when the RESOLVED object passes ``is_inert_first_party_callable``
        (facet marker or the frozen vetted set; never the live registry). Trust is the
        resolved ``__module__``, never the pickled path (``torchlens._io.tlspec`` /
        ``os.system`` walks off to ``posix``: denied). Resolving a torchlens submodule
        runs first-party code only and tells a vetted callable from a type; a pair
        that resolves to nothing is refused with ``UnpicklingError``. ``import_global``
        is the calling machine's importer, so both machines resolve identically.
        """

        if (module, name) in _SAFE_TORCHLENS_TYPES:
            obj = import_global(module, name)
            if isinstance(obj, type):
                return obj
            raise pickle.UnpicklingError(
                "Blocked allowlisted torchlens name that no longer resolves to a type "
                f"during bundle metadata unpickle: {module}.{name}."
            )
        try:
            obj = import_global(module, name)
        except (AttributeError, ImportError) as exc:
            raise pickle.UnpicklingError(
                "Blocked unresolvable torchlens global during bundle metadata "
                f"unpickle: {module}.{name} names no importable torchlens global "
                "(a stale module path or class name is refused, never resolved)."
            ) from exc
        if isinstance(obj, type):
            raise pickle.UnpicklingError(
                "Blocked non-allowlisted torchlens type during bundle metadata "
                f"unpickle: {module}.{name}. Only vetted-inert first-party DATA "
                "types are admitted; import-sink / instance-callable types such "
                "as LazyImportRef are denied (a pickle REDUCE would INVOKE the "
                "reconstructed object with attacker arguments)."
            )
        from .._utils._callable_safety import is_inert_first_party_callable  # import-order

        if callable(obj) and _is_torchlens_owned(obj) and is_inert_first_party_callable(obj):
            return obj
        raise pickle.UnpicklingError(
            f"Blocked disallowed torchlens global during bundle metadata unpickle: {module}.{name}."
        )

    def _resolve_global(  # noqa: C901 -- explicit policy
        self, module: str, name: str, import_global: _Importer | None = None
    ) -> Any:
        """Resolve a pickled global only if it is on the metadata allowlist.

        Parameters
        ----------
        module:
            Pickled module path.
        name:
            Pickled global name.
        import_global:
            The base machine's importer to resolve an admitted pair with; it carries the
            protocol state the stream set (a dotted name attribute-walks only at protocol
            4 and above, so the walk must be the machine's, read at the opcode). Defaults
            to this object's own importer (a direct ``find_class`` probe).

        Returns
        -------
        Any
            The resolved class / value / safe wrapper.

        Raises
        ------
        pickle.UnpicklingError
            If the global is not admitted by the allowlist.
        """

        if import_global is None:
            import_global = self._import_global
        refuse_pre_3_module_path(module, name)

        # Safe wrapper for the one legitimately-needed but individually-dangerous
        # callable.
        if (module, name) == ("builtins", "getattr"):
            return _safe_getattr
        # The storage leg of torch's default tensor reduction is a nested
        # ``torch.load`` over file bytes. Refused by name, before anything else
        # resolves: no byte from the file ever reaches torch's unpickler.
        if (module, name) == ("torch.storage", "_load_from_bytes"):
            raise pickle.UnpicklingError(
                "Blocked torch.storage._load_from_bytes during bundle metadata "
                "unpickle: this file embeds a tensor in its metadata in the format "
                "TorchLens 2.36.2 and earlier wrote (an annotation value such as one "
                "logged with tl.observers.log_value, or a small raw input or output), "
                "which this release no longer reads. To recover it, re-run the capture "
                "with this release and save it again; or load the file with the "
                "release that wrote it, replace the tensors in trace.annotations with "
                "plain values (for example tensor.tolist()), set "
                "trace._options_save_raw_input = False and trace._options_save_transformed_output = False, and "
                "save: the re-saved file loads here. If no TorchLens release wrote "
                "this file, it is corrupt."
            )
        # The TorchLens-owned inline tensor reduction: validated arguments, an exact
        # byte-length check and a non-resizable buffer-backed storage.
        if (module, name) == INLINE_TENSOR_GLOBAL:
            return rebuild_inline_tensor

        # Exact-pair allowlist of inert data/value types and tensor reconstructors.
        if (module, name) in _SAFE_EXPLICIT_GLOBALS:
            return import_global(module, name)

        if module == "torch._utils" and name.startswith("_rebuild"):
            return self._find_torch_rebuild(module, name, import_global)

        # torch package: admit value instances (dtype / layout / memory_format /
        # ``torch.Size`` etc.) and a resolved DATA ``type`` (tensor / Module /
        # storage classes, e.g. ``torch.nn.modules.linear.Identity``). Resolving a
        # global merely references it, and constructing a torch DATA type executes
        # no attacker code -- but constructing an I/O-capable torch type
        # (``PyTorchFileReader`` / ``PyTorchFileWriter`` / ``FileCheck`` /
        # ``PackageImporter`` / ``torch.serialization.*`` / ``torch.jit.*``) opens/
        # reads/writes an attacker-named filesystem path at unpickle time. Those
        # I/O / serialization / packaging / jit types are DENIED via
        # ``_torch_type_denied`` (module-prefix + structural name guard), aligning
        # with torch's own ``weights_only`` allowlist which excludes them. Non-type
        # callables (e.g. ``torch.serialization.load``) are denied; the two
        # individually-dangerous ones are wrapped above.
        if module == "torch" or module.startswith("torch."):
            obj = import_global(module, name)
            if isinstance(obj, (torch.dtype, torch.layout, torch.memory_format)):
                return obj
            if isinstance(obj, type):
                # POSITIVELY require the RESOLVED type's real module to be within the
                # torch package. A DOTTED pickled name attribute-walks OFF torch and
                # can escape into stdlib -- e.g. ``torch`` /
                # ``utils.collect_env.subprocess.Popen`` returns ``subprocess.Popen``
                # (real module ``subprocess``), which a following ``REDUCE`` would
                # INVOKE to spawn a process (RCE). ``_torch_type_denied`` (a name /
                # module-prefix denylist) would NOT catch it -- ``Popen`` matches no
                # token and ``subprocess`` no prefix -- so the escape is denied here
                # by the positive real-module gate BEFORE the I/O-type denylist runs.
                if not _is_torch_owned(obj):
                    raise pickle.UnpicklingError(
                        "Blocked non-torch-owned type resolved via the torch "
                        f"namespace during bundle metadata unpickle: {module}.{name} "
                        f"resolved to a type owned by "
                        f"{str(getattr(obj, '__module__', '') or '')!r} (a dotted "
                        "name that escapes the torch package is refused)."
                    )
                # STORAGE-CONSTRUCTOR deny (r49, secA_1). A torch STORAGE class is a
                # torch-owned DATA ``type`` and would otherwise pass here, but its
                # construction is NOT inert: a following ``REDUCE`` of
                # ``torch.UntypedStorage(N)`` allocates N raw bytes (a multi-GiB alloc
                # DoS) and is the sole source of an uninitialized-heap storage a BUILD
                # could rebind onto a tensor. torch's own ``weights_only`` baseline
                # lists the real ``torch.storage.{Typed,Untyped}Storage`` classes, so
                # denying them makes this unpickler a STRICT SUBSET of that baseline.
                # ``.tlspec`` never needs one (payloads are BlobRefs; inline metadata
                # tensors arrive whole through ``rebuild_inline_tensor``).
                if _is_torch_storage_type(obj):
                    raise pickle.UnpicklingError(
                        "Blocked torch storage constructor during bundle metadata "
                        f"unpickle: {module}.{name} (constructing a storage allocates "
                        "raw memory -- an UntypedStorage(N) out-of-memory DoS and the "
                        "source of an uninitialized-heap tensor; .tlspec tensor "
                        "payloads are BlobRefs and inline metadata tensors arrive "
                        "whole through the TorchLens inline tensor reduction)."
                    )
                if _torch_type_denied(module, name, obj):
                    raise pickle.UnpicklingError(
                        "Blocked torch I/O / serialization type during bundle "
                        f"metadata unpickle: {module}.{name} (construction performs "
                        "filesystem access; not an inert data type)."
                    )
                return obj
            raise pickle.UnpicklingError(
                f"Blocked disallowed torch global during bundle metadata unpickle: {module}.{name}."
            )

        # Preview-backend numeric array/dtype reconstruction (MLX / Paddle / JAX):
        # admit resolved ``type`` objects and framework dtype-like instances only.
        if any(
            module == root or module.startswith(root + ".") for root in _PREVIEW_ARRAY_MODULE_ROOTS
        ):
            obj = import_global(module, name)
            if isinstance(obj, type):
                # As in the torch branch: a resolved TYPE is admitted only when its
                # OWN real module is within a preview-array root. A DOTTED pickled
                # name (e.g. ``mlx.core`` / ``...subprocess.Popen``) attribute-walks
                # OFF the preview package and would otherwise return an escaped,
                # unrelated (possibly code-exec) type unconditionally.
                type_module = str(getattr(obj, "__module__", "") or "")
                if any(
                    type_module == root or type_module.startswith(root + ".")
                    for root in _PREVIEW_ARRAY_MODULE_ROOTS
                ):
                    return obj
                raise pickle.UnpicklingError(
                    "Blocked non-preview-owned type resolved via a preview-backend "
                    f"namespace during bundle metadata unpickle: {module}.{name} "
                    f"resolved to a type owned by {type_module!r}."
                )
            resolved_module = getattr(type(obj), "__module__", "") or ""
            if any(
                resolved_module == root or resolved_module.startswith(root + ".")
                for root in _PREVIEW_ARRAY_MODULE_ROOTS
            ):
                return obj
            raise pickle.UnpicklingError(
                "Blocked disallowed preview-backend global during bundle metadata "
                f"unpickle: {module}.{name}."
            )

        # TorchLens-owned metadata (not the extras-gated appliance packages, whose
        # import runs foreign code): allowlisted DATA types and vetted-inert
        # first-party callables only. Policy and refusal order: the method.
        if (
            module == "torchlens" or module.startswith("torchlens.")
        ) and not _is_torchlens_appliance_module(module):
            return self._find_torchlens_global(module, name, import_global)

        # Foreign global -- e.g. a user/test-registered ``@tl.facets.register``
        # recipe embedded by identity in a FacetRegistrySnapshot. Admission here is
        # DETERMINISTIC: it depends ONLY on the pickled ``(module, name)`` and the
        # caller's trust flags -- never on process state (whether the module is
        # already imported, or whether the recipe is in the live registry). Keying
        # off live state was the ordering-dependent load-failure bug.
        #
        # 1. Hard-deny known dangerous modules (os / sys / subprocess / pickle /
        #    builtins / importlib / ...) unconditionally -- they are never a
        #    legitimate metadata global and importing / resolving them is the RCE.
        if _module_denied(module):
            raise pickle.UnpicklingError(
                f"Blocked dangerous global during bundle metadata unpickle: {module}.{name}."
            )
        # 1b. STRUCTURAL close of the denylist-completeness class (r31): hard-deny any
        #     FOREIGN global whose pickled module is a Python STANDARD-LIBRARY /
        #     BUILTIN module (keyed on the top-level package), regardless of trust.
        #     This closes the whole class the explicit denylist kept chasing (io /
        #     _imp / zipimport / linecache / gc / mmap / ...). The pure-data stdlib
        #     globals that legit bundles need (``collections`` / ``builtins``) resolved
        #     via ``_SAFE_EXPLICIT_GLOBALS`` far above; only genuinely-foreign globals
        #     reach here, so a stdlib/builtin owner is never legitimate.
        if _stdlib_or_builtin_denied(module):
            raise pickle.UnpicklingError(
                "Blocked standard-library / builtin global during bundle metadata "
                f"unpickle: {module}.{name}. Stdlib/builtin modules are denied even "
                "under trust_custom_callables or an explicit module allowlist."
            )
        # 2. Explicit trust opt-in: import + resolve the real foreign callable,
        #    mirroring ``resolve_function_registry_key`` under trust. Importing the
        #    module executes its top-level code, so this is gated behind the flags.
        if self._custom_module_trusted(module):
            # DiD (r28, DiD-1): a DOTTED ``name`` under a TRUSTED module can
            # attribute-walk into a module-globals MAPPING
            # (``<trusted_mod>.<func>.__globals__``) whose real ``__module__`` is
            # undefined, so the resolved-real-module denylist recheck below falls back
            # to the (non-denied) pickled module and ADMITS it -- sidestepping the E2
            # denylist recheck. Every such escape traverses a dunder attribute segment;
            # legit foreign recipe names never do. Refuse a dunder-walk BEFORE resolving
            # (never trigger the walk).
            if _name_has_dunder_walk(name):
                raise pickle.UnpicklingError(
                    "Blocked dunder attribute-walk in a trusted foreign global during "
                    f"bundle metadata unpickle: {module}.{name} (a dotted name walking "
                    "into __globals__/__builtins__/__dict__ escapes the resolved-module "
                    "denylist recheck)."
                )
            obj = import_global(module, name)
            # A resolved MAPPING (e.g. a module's ``__globals__`` / ``__dict__`` reached
            # by a dotted walk that the dunder guard above did not already refuse) is
            # never a legitimate foreign recipe callable/type, and its real
            # ``__module__`` is undefined -- defeating the denylist recheck. Refuse it.
            if isinstance(obj, dict):
                raise pickle.UnpicklingError(
                    "Blocked mapping (module globals) resolved via a dotted name during "
                    f"trusted bundle metadata unpickle: {module}.{name}."
                )
            # A resolved MODULE object (A-R32-2): a bare module has no ``__module__``, so
            # the resolved-real-module denylist recheck below falls back to the (trusted,
            # non-stdlib) PICKLED module name and ADMITS it -- defeating
            # ``_stdlib_or_builtin_denied``. A bare module is never a legitimate foreign
            # recipe callable/type. Refuse it outright, mirroring the mapping refusal
            # above (``isinstance(obj, dict)`` misses modules).
            if isinstance(obj, ModuleType):
                raise pickle.UnpicklingError(
                    "Blocked module object resolved via a dotted name during trusted "
                    f"bundle metadata unpickle: {module}.{name}. A bare module is never a "
                    "legitimate foreign recipe callable/type."
                )
            # RE-ENFORCE the DENYLIST on the RESOLVED object's REAL module, NEVER the
            # pickled string. A DOTTED ``name`` attribute-walks off the trusted module
            # and can land on a callable from a DIFFERENT, denied module:
            # ``module="some_trusted_mod"`` / ``name="os.system"`` resolves
            # ``os.system`` (real module ``posix``) whenever the trusted module did
            # ``import os``. ``_module_denied`` on the pickled ``module`` alone (step
            # 1) never sees ``posix``; keying on resolved identity closes the walk. We
            # re-enforce the DENYLIST (never hand back a dangerous callable, even under
            # trust) but NOT the allowlist -- the allowlist governs which MODULE may be
            # IMPORTED (already enforced), and a resolved ``__module__`` may legitimately
            # differ from the imported module (e.g. a C-accelerator re-export).
            resolved_owner = str(getattr(obj, "__module__", "") or module)
            if _module_denied(resolved_owner):
                raise pickle.UnpicklingError(
                    "Blocked dangerous global (resolved real module "
                    f"{resolved_owner!r}) during trusted bundle metadata unpickle: "
                    f"{module}.{name}."
                )
            # STRUCTURAL stdlib/builtin close (r31): a dotted ``name`` can walk OFF a
            # trusted module and land on a stdlib/builtin callable (e.g. a trusted
            # ``mymod.io.open`` resolving ``io.open``, real module ``io``). Deny the
            # whole stdlib/builtin class on the resolved real module, even under trust.
            if _stdlib_or_builtin_denied(resolved_owner):
                raise pickle.UnpicklingError(
                    "Blocked standard-library / builtin global (resolved real module "
                    f"{resolved_owner!r}) reached via a dotted name during trusted "
                    f"bundle metadata unpickle: {module}.{name}."
                )
            # PURITY PARITY (secE-1 / secE-r36-1). A callable that walked back into the
            # torch namespace via a dotted name -- OR a module-less C tensor method
            # (``resize_`` / ``set_`` / ``apply_`` / ``map_``) -- must be a PURE forward
            # op: ``torch`` also hosts side-effecting builtins (``torch.from_file``) and
            # the tensor-method family rebinds/reallocates storage or runs an arbitrary
            # callable per element. This gate keys on the callable's REAL
            # (capture-unwrapped) module, NEVER ``resolved_owner``: that fallback string
            # is spoofed two ways -- it lands on the trusted pickled ``module`` for a
            # module-less tensor method (real ``__module__ is None``), and on
            # ``"torchlens.backends.torch.wrappers"`` for ANY torch op TorchLens has
            # capture-wrapped (the near-universal live state) -- so a raw string
            # torch/prefix check misses a wrapped ``resize_`` / ``torch.load`` (the r36
            # hole). Mirror the r35 torchlens-walk fix / the intervention resolver on the
            # REAL identity: when the real owner is module-less OR torch, hold it to
            # ``is_pure_forward_callable`` (which unwraps + covers the
            # tensor-method-descriptor case, the storage-unsafe / ``apply_`` / ``map_``
            # name guards, and torch name/purity). Legit foreign recipe callables/types
            # carry a real, non-torch ``__module__`` and are NOT subject to this gate.
            from .._utils._callable_safety import (
                is_pure_forward_callable,
                real_callable_module,
            )

            real_owner = real_callable_module(obj) if callable(obj) else ""
            if (
                callable(obj)
                and (real_owner == "" or real_owner == "torch" or real_owner.startswith("torch."))
                and not is_pure_forward_callable(obj)
            ):
                raise pickle.UnpicklingError(
                    "Blocked torch-namespace or module-less callable (chiefly a C-level "
                    "tensor method such as resize_/set_/apply_/map_ or a side-effecting "
                    f"torch builtin) reached via a dotted name during trusted bundle "
                    f"metadata unpickle: {module}.{name}. Only pure forward/tensor ops "
                    "resolve from the torch namespace or a module-less C callable, even "
                    "under trust."
                )
            # OPERATOR GADGET name-scope (r33, A-R32-1 / E-r32-1): the ``operator`` /
            # ``_operator`` root is carved out of the stdlib denial so ``operator.neg``
            # survives, but that carve-out must NOT re-admit the generic operator
            # gadgets (``attrgetter`` / ``methodcaller`` / ``call`` / ``getitem`` /
            # ``setitem`` / ``delitem`` / the in-place ``iadd`` / ``imul`` / ...
            # mutators) that enable an RCE chain (``attrgetter('__globals__')`` ->
            # ``__import__`` -> ``os.system``). Deny any resolved ``operator`` /
            # ``_operator`` callable whose terminal name is not a pure forward operator,
            # even under trust. Parity with the intervention resolver.
            if callable(obj):
                from .._utils._callable_safety import is_denied_operator_gadget

                if is_denied_operator_gadget(obj):
                    raise pickle.UnpicklingError(
                        "Blocked generic operator gadget (resolved real module "
                        f"{resolved_owner!r}) reached during trusted bundle metadata "
                        f"unpickle: {module}.{name}. Only pure arithmetic / comparison / "
                        "bitwise / index operators resolve from operator/_operator."
                    )
            return obj
        # 3. Default (untrusted): LOAD-TOLERATE as an inert deferred reference. The
        #    foreign module is NEVER imported here (import executes code); the recipe
        #    reference is preserved so the trace loads structurally, and any attempt
        #    to CALL it (facet computation, or a ``__reduce__`` gadget) fails closed.
        return _DeferredForeignCallable(module, name)

    def _find_torch_rebuild(self, module: str, name: str, import_global: _Importer) -> Any:
        """Resolve a ``torch._utils._rebuild*`` reconstructor for ``find_class``."""
        # Pure torch tensor reconstruction helpers. ``_rebuild_*`` functions only
        # reassemble tensors from already-resolved (allowlisted) components; any
        # class they receive as an argument was itself gated through find_class.
        #
        # HARDENING (r28, A-R28-1 CRITICAL default-victim RCE): the prior fast-path
        # returned ``super().find_class`` for ANY ``name`` starting with ``_rebuild``
        # with NO further gate. Under pickle protocol >= 4 the pickled ``name`` is
        # attribute-walked, so a DOTTED name such as
        # ``_rebuild_tensor_v2.__globals__.get`` walks OFF the reconstructor function
        # into ``torch._utils`` module globals and returns a REDUCE-invocable bound
        # ``dict.get`` -> ``_import_dotted_name`` -> ``os.system`` -- a load-time RCE
        # (fires at metadata unpickle, no ``.run()`` needed). This is the SAME
        # dotted-walk escape class the r27 torch-type / preview branches close; the
        # ``_rebuild`` branch was MISSED. Legit ``_rebuild_*`` reconstructor names NEVER
        # contain a dot, so (a) refuse any DOTTED ``name`` BEFORE resolving (never
        # trigger the walk), and (b) after resolving, positively require a torch-owned,
        # non-type CALLABLE (the reconstructors are functions of ``torch._utils``).
        if "." in name:
            raise pickle.UnpicklingError(
                "Blocked dotted torch._utils._rebuild name during bundle metadata "
                f"unpickle: {module}.{name} (a dotted name attribute-walks off the "
                "reconstructor into module globals; legit _rebuild_* names have no dot)."
            )
        obj = import_global(module, name)
        if not (callable(obj) and _is_torch_owned(obj) and not isinstance(obj, type)):
            raise pickle.UnpicklingError(
                "Blocked non-torch-owned / non-callable torch._utils._rebuild "
                f"resolution during bundle metadata unpickle: {module}.{name} "
                f"(resolved to {str(getattr(obj, '__module__', '') or '')!r})."
            )
        return obj


class SafeBundleUnpickler(_BundleAllowlistPolicy, pickle.Unpickler):
    """Default-deny unpickler for untrusted TorchLens bundle ``metadata.pkl``.

    The stream is read once and pre-scanned by a dry run of the C machine over
    recording probes (:mod:`torchlens._io._pickle_prescan`: allowlist resolution of
    every global, the allocation belt for every statically known construction, the
    BUILD-target classification; the opcode ceiling is counted separately on streams
    large enough to reach it), then loaded by the C ``pickle.Unpickler``. Its ``find_class`` hands back each pre-resolved global, wrapped
    in a :class:`~torchlens._io._pickle_prescan.GuardedCallable` when a REDUCE will call it so the belt runs on the
    real arguments at call time. A stream shape the pre-scan cannot classify runs
    through :class:`_PurePythonBundleUnpickler` instead, which enforces the same checks
    per opcode; no TorchLens writer produces such a stream. After ``load()``,
    ``load_path`` is ``"c"`` or ``"pure_python"``, ``prescan_opcodes`` the opcode count
    of the reader pass, ``global_lookups`` the C phase's ``find_class`` calls and
    ``guarded_calls`` the REDUCEs the guards ran.
    """

    def __init__(
        self,
        file: BinaryIO,
        *,
        trust_custom_callables: bool = False,
        allowed_custom_callable_modules: Collection[str] | None = None,
        max_opcodes: int | None = None,
        count_opcodes_above: int = 0,
    ) -> None:
        """Initialize the restricted unpickler.

        Parameters
        ----------
        file:
            Binary file object positioned at the start of a pickle stream; read in full.
            A source that cannot report its length (a pipe, a socket) refuses every
            FRAME, as the pure-Python machine does.
        trust_custom_callables:
            Whether to admit a foreign (non-torch/torchlens/numpy) callable by
            importing and resolving it. Mirrors the intervention-resolver contract:
            ``False`` (default) denies unless the callable is already registered in
            the live facet registry (a positive allowlist against trusted live
            state that never triggers a fresh import).
        allowed_custom_callable_modules:
            Optional narrow allowlist of foreign module names whose callables may be
            imported and resolved even when ``trust_custom_callables`` is ``False``.
        max_opcodes:
            Opcode-count ceiling; a longer stream refuses with
            :class:`~torchlens._io._pickle_prescan.OpcodeCeilingExceeded` before any
            object is built.
        count_opcodes_above:
            Only streams longer than this many bytes are held to the ceiling (below it
            the ceiling is unreachable); the memo-index check runs on every stream.
        """

        self._source_sized = _stream_bytes_left(file) is not None
        data = file.read()
        self._data = bytes(data)
        super().__init__(io.BytesIO(self._data))
        self._init_policy(trust_custom_callables, allowed_custom_callable_modules)
        self._max_opcodes = max_opcodes
        self._count_opcodes_above = count_opcodes_above
        self._plan: PrescanPlan | None = None
        self.load_path: str | None = None
        self.prescan_opcodes = 0
        self.global_lookups = 0
        self.guarded_calls = 0
        self.fallback_reason: str | None = None

    def _import_global(self, module: str, name: str) -> Any:
        """Resolve with the C machine's importer (``fix_imports`` mapping included)."""

        return super().find_class(module, name)

    def find_class(self, module: str, name: str) -> Any:
        """Return the pre-resolved global for the C machine, guarded when a REDUCE calls it.

        Outside a ``load()`` (a direct probe) this is the bare policy. Inside one, every
        global the stream names was already resolved by the pre-scan through the same
        policy (or refused by it: the refusal is raised here, at the GLOBAL that named
        it); a name the pre-scan did not see is a stream that changed under the loader
        and is refused.
        """

        plan = self._plan
        if plan is None:
            return self._resolve_global(module, name)
        pair = (module, name)
        plan.global_lookups += 1
        if plan.refused is not None and pair == plan.refused[0]:
            # The pre-scan's deferred refusal: raised here, at the GLOBAL that named it,
            # after every earlier opcode ran for real (the pure-Python machine's order).
            raise plan.refused[1]
        try:
            obj = plan.resolved[pair]
        except KeyError:
            raise pickle.UnpicklingError(
                "corrupt or malformed pickle stream: the stream named a global the "
                "pre-scan did not see."
            ) from None
        if pair in plan.wrap_pairs:
            return GuardedCallable(obj, plan, pair=pair)
        return obj

    def _prescan(self) -> PrescanPlan | None:
        """Return the pre-scan plan, or ``None`` when the interpreter must run instead."""

        ceiling = self._max_opcodes if len(self._data) > self._count_opcodes_above else None
        try:
            self.prescan_opcodes = scan_opcodes(
                self._data, ceiling, source_sized=self._source_sized
            )
            plan = prescan_stream(
                self._data,
                resolve=self._resolve_global,
                construction_refusal=_construction_refusal,
                build_refusal=_build_refusal,
                build_type_refusal=_build_type_refusal,
            )
        except StreamNeedsInterpreter as why:
            self.fallback_reason = str(why)
            return None
        return plan

    def _interpret(self) -> Any:
        """Run the stream through the pure-Python machine (the exact per-opcode checks)."""

        interpreter = _PurePythonBundleUnpickler(
            io.BytesIO(self._data),
            trust_custom_callables=self._trust_custom_callables,
            allowed_custom_callable_modules=self._allowed_custom_callable_modules,
            source_sized=self._source_sized,
        )
        return interpreter.load()

    def load(self) -> Any:
        """Pre-scan the stream, then load it with the C machine (or the interpreter)."""

        plan = self._boundary(self._prescan)
        if plan is None:
            self.load_path = "pure_python"
            return self._interpret()
        self._plan = plan
        self.load_path = "c"
        try:
            return self._boundary(super().load)
        finally:
            self._plan = None
            self.global_lookups = plan.global_lookups
            self.guarded_calls = plan.cursor


class _PurePythonBundleUnpickler(_BundleAllowlistPolicy, pickle._Unpickler):
    """The per-opcode form of the same policy, on the pure-Python ``pickle._Unpickler``.

    Runs the streams the pre-scan cannot classify statically. The pure-Python VM reads a
    CLASS-LEVEL ``dispatch`` dict bound at class-body time (a plain ``def load_build``
    override is inert), so the class body REBUILDS ``dispatch`` to the overrides below.
    """

    def __init__(
        self,
        file: BinaryIO,
        *,
        trust_custom_callables: bool = False,
        allowed_custom_callable_modules: Collection[str] | None = None,
        source_sized: bool = True,
    ) -> None:
        """Initialize the interpreter with the same trust flags as the fast path.

        ``source_sized`` is ``False`` when the fast path hands over a stream it read from
        a source that could not report its length: ``load_frame`` then refuses every
        FRAME, as it does on such a source read directly.
        """

        super().__init__(file)
        self._stream = file
        self._source_sized = source_sized
        self._init_policy(trust_custom_callables, allowed_custom_callable_modules)

    def load(self) -> Any:
        """Run the pure-Python VM inside the corruption-normalizing boundary."""

        return self._boundary(super().load)

    def _import_global(self, module: str, name: str) -> Any:
        """Resolve with the pure-Python machine's importer."""

        return super().find_class(module, name)

    def find_class(self, module: str, name: str) -> Any:
        """Resolve a global through the shared allowlist policy."""

        return self._resolve_global(module, name)

    # Layer 2: opcode-level policy. ``find_class`` gates which globals resolve, but BUILD,
    # REDUCE, NEWOBJ, NEWOBJ_EX, INST and OBJ act on objects already on the stack; the
    # belt text and decisions live in ``_construction_refusal`` and ``_build_refusal``.

    def load_build(self) -> None:
        """Refuse a BUILD (``__setstate__`` / storage-rebind) on a torch tensor/storage."""

        # Base ``load_build`` is ``state = stack.pop(); inst = stack[-1]`` -- so before
        # it runs, the target instance is ``stack[-2]`` (``state`` is ``stack[-1]``).
        reason = _build_refusal(self.stack[-2])  # type: ignore[attr-defined]
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        return _BASE_LOAD_BUILD(self)  # type: ignore[arg-type]

    def load_reduce(self) -> None:
        """Refuse a REDUCE that constructs -- or MEDIATES -- an attacker-sized allocation."""

        # Base ``load_reduce`` is ``args = stack.pop(); func = stack[-1]`` -- so before
        # it runs, ``func`` is ``stack[-2]`` (``args`` is ``stack[-1]``).
        reason = _construction_refusal("REDUCE", self.stack[-2], self.stack[-1])  # type: ignore[attr-defined]
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        return _BASE_LOAD_REDUCE(self)  # type: ignore[arg-type]

    def load_newobj(self) -> None:
        """Refuse a NEWOBJ (``cls.__new__``) that would allocate a storage/tensor/ndarray."""

        reason = _construction_refusal("NEWOBJ", self.stack[-2], self.stack[-1])  # type: ignore[attr-defined]
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        return _BASE_LOAD_NEWOBJ(self)  # type: ignore[arg-type]

    def load_newobj_ex(self) -> None:
        """Refuse a NEWOBJ_EX (``cls.__new__``) that would allocate a storage/tensor/ndarray."""

        reason = _construction_refusal(
            "NEWOBJ_EX",
            self.stack[-3],  # type: ignore[attr-defined]
            self.stack[-2],  # type: ignore[attr-defined]
            self.stack[-1],  # type: ignore[attr-defined]
        )
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        return _BASE_LOAD_NEWOBJ_EX(self)  # type: ignore[arg-type]

    def _instantiate(self, klass: Any, args: Any) -> None:
        """Refuse an INST / OBJ construction that would allocate attacker-sized memory.

        ``load_inst`` / ``load_obj`` both funnel through this primitive (normal MRO, not
        the dispatch table), so one override gates both legacy construction opcodes.
        They hand ``args`` as a LIST (from ``pop_mark()``); the belt gates on a TUPLE,
        so normalize first so a security belt never silently no-ops on a container-type
        mismatch.
        """

        arg_tuple = tuple(args) if isinstance(args, (list, tuple)) else args
        reason = _construction_refusal("INST", klass, arg_tuple)
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        return super()._instantiate(klass, args)  # type: ignore[misc]

    def load_ext(self) -> None:
        """Refuse EXT1 / EXT2 / EXT4 before any extension-registry lookup.

        They resolve a code through the process-wide ``copyreg`` registry and cache the
        result process-wide. The writer emits one only for a code registered in the
        writing process, which no other process could load, so metadata never uses them.
        The C path refuses the same opcodes in the reader pass (``scan_opcodes``).
        """

        raise pickle.UnpicklingError(_CORRUPT_PREFIX + _EXT_REFUSAL)

    def load_frame(self) -> None:
        """Refuse a FRAME declaring more bytes than remain in the stream, before reading it.

        The base VM passes the declared size to ``file.read``, which a buffered file
        allocates up front. A stream that cannot report its length is refused. The C
        path holds the whole stream in memory, bounds every FRAME in the reader pass and
        refuses every FRAME of a source that could not report its length: one decision.
        """

        (frame_size,) = struct.unpack("<Q", self.read(8))  # type: ignore[attr-defined]
        left = _stream_bytes_left(self._stream) if self._source_sized else None
        if left is None or frame_size > left:
            raise pickle.UnpicklingError(_CORRUPT_PREFIX + _FRAME_REFUSAL)
        self._unframer.load_frame(frame_size)  # type: ignore[attr-defined]

    dispatch = _DenyMissingDispatch(pickle._Unpickler.dispatch)
    dispatch[pickle.BUILD[0]] = load_build
    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.NEWOBJ_EX[0]] = load_newobj_ex
    dispatch[pickle.EXT1[0]] = load_ext
    dispatch[pickle.EXT2[0]] = load_ext
    dispatch[pickle.EXT4[0]] = load_ext
    dispatch[pickle.FRAME[0]] = load_frame
