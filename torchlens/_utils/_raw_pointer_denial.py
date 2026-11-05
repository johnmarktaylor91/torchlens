"""Structural raw-pointer denial for resolved callables and bundle keywords.

A saved bundle is UNTRUSTED input: its callable keys and its per-call keyword
names are both attacker-chosen. Torch exposes a whole FAMILY of callables that
take a RAW MEMORY ADDRESS as an argument and build a live Python object around
it -- above all the legacy tensor constructors' hidden keyword-only ``cdata``
overload (``torch.FloatTensor(cdata=<int>)``, ``torch.Tensor.new(cdata=<int>)``),
the storage classes' ``_new_with_weak_ptr`` / ``_set_cdata`` family, and the
pointer-adopting free functions (``torch._C._aoti.alloc_tensor_by_stealing_from_void_ptr``,
``torch.xpu.get_stream_from_external(data_ptr=...)``,
``torch.cuda.streams.ExternalStream(stream_ptr=...)``,
``torch.cuda.Event.from_ipc_handle(handle)``). Handing any of them an
attacker-chosen integer dereferences an arbitrary process address: the probe on
torch 2.13 segfaults the interpreter on a NULL ``cdata`` (``torch.FloatTensor(cdata=0)``,
``torch.Tensor(cdata=0)``, ``torch.Tensor.new(cdata=0)`` all exit -11), and a
non-NULL address is an arbitrary-memory read / type-confusion primitive.

Denying these ONE NAME AT A TIME does not terminate: the legacy constructors
alone are 38 classes across ``torch`` / ``torch.cuda`` / ``torch.sparse`` /
``torch.cuda.sparse`` that torch regenerates per dtype, so every new dtype
sibling is a new gadget. This module closes the CLASS with two structural rules
and no enumeration of individual names:

1. CONSTRUCTOR-TYPE rule (:func:`is_raw_pointer_constructor_type`). Every torch
   tensor / storage CLASS is denied as a resolvable callable, recognized
   STRUCTURALLY: a ``torch.Tensor`` subclass, a member of torch's own
   ``_tensor_classes`` / ``_storage_classes`` registries, anything built by the
   ``tensortype`` METACLASS behind the legacy ``torch.<dtype>Tensor`` families
   (so a future dtype or device sibling is denied the moment torch adds it), and
   the storage base classes. A class is a CONSTRUCTOR, never a captured forward
   op: TorchLens logs the legacy constructors as ops whose replay is refused
   (``tests/test_legacy_tensor_constructors.py``), so nothing legitimate is lost.

2. RAW-POINTER PARAMETER rule (:data:`RAW_POINTER_PARAMETER_NAMES`,
   :func:`signature_takes_raw_pointer`, :func:`raw_pointer_keyword_name`).
   A callable whose inspectable signature exposes a pointer / address / foreign
   handle parameter is denied; and -- because the dangerous C overloads carry NO
   inspectable signature -- a bundle-supplied KEYWORD ARGUMENT whose name is one
   of these is refused at the replay argument boundary, whatever callable it
   targets. This is what keeps the legitimate spellings working: the wrapped,
   genuinely captured allocator ``Tensor.new(*sizes)`` / ``Tensor.new(data)``
   (an ``IGNORED_FUNCS`` roster row that TorchLens does decorate and log, with
   uninitialized-memory semantics tracked in ``_utils/_uninit_alloc.py``) keeps
   resolving and running, while the keyword-only ``cdata=`` spelling of the SAME
   callable is refused before any call happens.

The parameter names were chosen to have zero overlap with the pure forward-op
surface: no captured torch forward op takes a parameter named ``cdata``,
``ptr``, ``*_ptr``, ``address`` or ``handle`` (verified by enumerating every
callable reachable through the loader's allowlisted roots and enumerated torch
namespaces -- ``tests/test_sec_raw_pointer_family.py`` re-enumerates at runtime
so a future admission regression fails). ``data_ptr`` as a RESULT (the pure read
``Tensor.data_ptr()``, which consumes no address) is deliberately NOT denied.

Leaf module: imports ``torch`` and the standard library only, so the early
security front doors (``_io/_safe_unpickle``, ``_io/_alloc_belt``) can use it
without import-order coupling.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

__tl_layer__ = "L0"

# Parameter / keyword names that denote a RAW MEMORY ADDRESS or a foreign resource
# handle adopted by the callee. Exact names plus the ``*_ptr`` / ``*_handle`` suffix
# shapes, so a future sibling spelling (``stream_ptr`` / ``mem_ptr`` / ``event_handle``)
# is covered without being enumerated.
RAW_POINTER_PARAMETER_NAMES: frozenset[str] = frozenset(
    {
        "cdata",
        "_cdata",
        "ptr",
        "pointer",
        "address",
        "addr",
        "handle",
        "from_address",
        "data_ptr",
        "void_ptr",
        "storage_ptr",
        "stream_ptr",
        "mem_ptr",
        "weak_ptr",
        "ipc_handle",
        "range_handle",
    }
)

# Suffix shapes closing the same class for names never enumerated above.
_RAW_POINTER_NAME_SUFFIXES: tuple[str, ...] = ("_ptr", "_cdata", "_address", "_handle")


def is_raw_pointer_parameter_name(name: str) -> bool:
    """Return whether a parameter / keyword name denotes a raw pointer or foreign handle.

    Parameters
    ----------
    name:
        Parameter or bundle-supplied keyword name.

    Returns
    -------
    bool
        ``True`` when the name is an enumerated pointer name or carries a pointer
        suffix shape (``*_ptr`` / ``*_cdata`` / ``*_address`` / ``*_handle``).
    """

    lowered = name.lower()
    if lowered in RAW_POINTER_PARAMETER_NAMES:
        return True
    return any(lowered.endswith(suffix) for suffix in _RAW_POINTER_NAME_SUFFIXES)


def raw_pointer_keyword_name(names: Iterable[str]) -> str | None:
    """Return the first bundle-supplied keyword name that denotes a raw pointer.

    Parameters
    ----------
    names:
        Keyword argument names carried by an untrusted call descriptor.

    Returns
    -------
    str | None
        The offending name, or ``None`` when every name is ordinary.
    """

    for name in names:
        if isinstance(name, str) and is_raw_pointer_parameter_name(name):
            return name
    return None


def signature_takes_raw_pointer(func: Callable[..., Any]) -> bool:
    """Return whether an inspectable callable accepts a raw-pointer parameter.

    Fails SAFE in the permissive direction for non-inspectable C builtins (no
    signature -> ``False``): those are closed by the constructor-type rule and by
    the keyword refusal at the replay argument boundary, not here.

    Parameters
    ----------
    func:
        Capture-unwrapped callable under the security gate.

    Returns
    -------
    bool
        ``True`` when any declared parameter names a pointer / address / handle.
    """

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # C builtins expose no signature.
        return False
    return any(is_raw_pointer_parameter_name(name) for name in signature.parameters)


def _storage_base_classes() -> tuple[type, ...]:
    """Return the torch storage base classes available on this torch."""

    bases: list[type] = []
    for name in ("UntypedStorage", "TypedStorage"):
        candidate = torch.__dict__.get(name)
        if isinstance(candidate, type):
            bases.append(candidate)
    storage_module = getattr(torch, "storage", None)
    for name in ("UntypedStorage", "TypedStorage", "_StorageBase", "_LegacyStorage"):
        candidate = getattr(storage_module, name, None)
        if isinstance(candidate, type) and candidate not in bases:
            bases.append(candidate)
    return tuple(bases)


def _registry_classes(attribute: str) -> frozenset[type]:
    """Return the classes in one of torch's own type registries."""

    registry = getattr(torch, attribute, None)
    if not isinstance(registry, (set, frozenset, tuple, list)):
        return frozenset()
    return frozenset(cls for cls in registry if isinstance(cls, type))


_TENSOR_REGISTRY_CLASSES: frozenset[type] = _registry_classes("_tensor_classes")
_STORAGE_REGISTRY_CLASSES: frozenset[type] = _registry_classes("_storage_classes")
_STORAGE_BASE_CLASSES: tuple[type, ...] = _storage_base_classes()

# The ``tensortype`` metaclass(es) torch builds the legacy ``torch.<dtype>Tensor``
# families with (``torch`` / ``torch.cuda`` / ``torch.sparse`` / ``torch.cuda.sparse``).
# An ``isinstance`` belt over these denies every CURRENT and FUTURE dtype or device
# sibling without enumerating a single class name.
_TENSORTYPE_METACLASSES: tuple[type, ...] = tuple(
    {
        metacls
        for metacls in (type(cls) for cls in _TENSOR_REGISTRY_CLASSES | _STORAGE_REGISTRY_CLASSES)
        if metacls is not type
    }
)


def is_raw_pointer_constructor_type(obj: Any) -> bool:
    """Return whether ``obj`` is a torch tensor / storage CLASS (a raw-pointer constructor).

    Structural recognition, never an enumeration of class names: a
    ``torch.Tensor`` subclass, a member of torch's ``_tensor_classes`` /
    ``_storage_classes`` registries, an instance of the ``tensortype`` metaclass
    behind the legacy ``torch.<dtype>Tensor`` families, or a storage base class.
    Every one of them accepts a keyword-only ``cdata`` / ``_cdata`` raw address
    (or a ``wrap_storage`` donor) that builds a live object over an arbitrary
    process address.

    Parameters
    ----------
    obj:
        Resolved object under the security gate.

    Returns
    -------
    bool
        ``True`` for a torch tensor / storage class; ``False`` for every
        non-type and for ordinary callables. Never raises.
    """

    if not isinstance(obj, type):
        return False
    try:
        if issubclass(obj, torch.Tensor):
            return True
        if _STORAGE_BASE_CLASSES and issubclass(obj, _STORAGE_BASE_CLASSES):
            return True
    except TypeError:  # pragma: no cover - defensive; odd metaclasses.
        pass
    if obj in _TENSOR_REGISTRY_CLASSES or obj in _STORAGE_REGISTRY_CLASSES:
        return True
    return bool(_TENSORTYPE_METACLASSES) and isinstance(obj, _TENSORTYPE_METACLASSES)


def raw_pointer_constructor_family() -> tuple[type, ...]:
    """Return the live raw-pointer constructor family, for the runtime guard test.

    Returns
    -------
    tuple[type, ...]
        Every torch tensor / storage class this torch exposes, sorted by dotted
        name so the guard test reports a stable, readable failure.
    """

    family: set[type] = set(_TENSOR_REGISTRY_CLASSES | _STORAGE_REGISTRY_CLASSES)
    family.add(torch.Tensor)
    family.update(_STORAGE_BASE_CLASSES)
    return tuple(sorted(family, key=lambda cls: f"{cls.__module__}.{cls.__qualname__}"))


def pointer_keyword_refusal_detail(name: str, where: str) -> str:
    """Return the shared refusal message for a bundle-supplied pointer keyword.

    Parameters
    ----------
    name:
        The offending keyword name.
    where:
        Short description of the refusing boundary, used in the message.

    Returns
    -------
    str
        One refusal sentence naming the keyword and the rule.
    """

    return (
        f"{where} carries keyword argument {name!r}, which names a raw memory "
        "address or foreign handle. Bundle-supplied pointer keywords (the legacy "
        "constructors' keyword-only 'cdata' overload and its siblings) are refused: "
        "a saved call may never hand torch an attacker-chosen address."
    )


def mapping_pointer_keyword(kwargs: Mapping[str, Any]) -> str | None:
    """Return the first raw-pointer keyword present in a reconstructed kwargs mapping.

    Parameters
    ----------
    kwargs:
        Keyword mapping assembled for one replayed call.

    Returns
    -------
    str | None
        The offending keyword name, or ``None``.
    """

    return raw_pointer_keyword_name(kwargs.keys())
