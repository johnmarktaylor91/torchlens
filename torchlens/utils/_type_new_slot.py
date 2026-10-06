"""CPython ``tp_new`` patching for torch's immutable legacy tensor-constructor types.

``torch.FloatTensor`` and its dtype siblings are ``torch.tensortype`` instances:
static extension types flagged ``Py_TPFLAGS_IMMUTABLETYPE``, so ``setattr`` on
them (or on their metaclass) is refused, and calling one runs a C ``tp_new``
that no Python wrapper or ``TorchFunctionMode`` ever sees. Replacing the
namespace attribute instead would break class identity (``isinstance``,
``x.type(torch.FloatTensor)``) and would miss every alias bound before the first
capture. This module patches the class object's ``__new__`` in place: it clears
the immutable bit for the one ``setattr`` and puts it back, keeps the original C
``tp_new`` pointer so the wrapper calls exactly the eager constructor, and
restores the slot pointer and the class ``__dict__`` entry on removal.

Everything here reads private CPython layout, so callers gate on
``torchlens.utils._torch_compat.HAS_LEGACY_CONSTRUCTOR_NEW_PATCH`` (whose probe is
:func:`probe_type_new_slot_patch`).
"""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_PY_TPFLAGS_IMMUTABLETYPE = 1 << 8
_ABSENT = object()


class _PyTypeObjectToNew(ctypes.Structure):
    """Partial ctypes mirror of CPython's ``PyTypeObject`` up to ``tp_new``."""

    _fields_ = [
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.c_void_p),
        ("ob_size", ctypes.c_ssize_t),
        ("tp_name", ctypes.c_char_p),
        ("tp_basicsize", ctypes.c_ssize_t),
        ("tp_itemsize", ctypes.c_ssize_t),
        ("tp_dealloc", ctypes.c_void_p),
        ("tp_vectorcall_offset", ctypes.c_ssize_t),
        ("tp_getattr", ctypes.c_void_p),
        ("tp_setattr", ctypes.c_void_p),
        ("tp_as_async", ctypes.c_void_p),
        ("tp_repr", ctypes.c_void_p),
        ("tp_as_number", ctypes.c_void_p),
        ("tp_as_sequence", ctypes.c_void_p),
        ("tp_as_mapping", ctypes.c_void_p),
        ("tp_hash", ctypes.c_void_p),
        ("tp_call", ctypes.c_void_p),
        ("tp_str", ctypes.c_void_p),
        ("tp_getattro", ctypes.c_void_p),
        ("tp_setattro", ctypes.c_void_p),
        ("tp_as_buffer", ctypes.c_void_p),
        ("tp_flags", ctypes.c_ulong),
        ("tp_doc", ctypes.c_char_p),
        ("tp_traverse", ctypes.c_void_p),
        ("tp_clear", ctypes.c_void_p),
        ("tp_richcompare", ctypes.c_void_p),
        ("tp_weaklistoffset", ctypes.c_ssize_t),
        ("tp_iter", ctypes.c_void_p),
        ("tp_iternext", ctypes.c_void_p),
        ("tp_methods", ctypes.c_void_p),
        ("tp_members", ctypes.c_void_p),
        ("tp_getset", ctypes.c_void_p),
        ("tp_base", ctypes.c_void_p),
        ("tp_dict", ctypes.c_void_p),
        ("tp_descr_get", ctypes.c_void_p),
        ("tp_descr_set", ctypes.c_void_p),
        ("tp_dictoffset", ctypes.c_ssize_t),
        ("tp_init", ctypes.c_void_p),
        ("tp_alloc", ctypes.c_void_p),
        ("tp_new", ctypes.c_void_p),
    ]


# ``newfunc``: PyObject *(*)(PyTypeObject *, PyObject *args, PyObject *kwds).
# PYFUNCTYPE keeps the GIL and re-raises the error a NULL return leaves set, so
# the original constructor's own exception reaches the caller unchanged.
_TP_NEW_FUNCTYPE = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object
)


def _type_view(cls: type) -> _PyTypeObjectToNew:
    """Return the ctypes view of ``cls``'s type object (CPython only)."""

    return _PyTypeObjectToNew.from_address(id(cls))


class _InheritedNewProbe:
    """Reference class whose ``tp_new`` is inherited from ``object`` (``object_new``)."""


class _PythonNewProbe:
    """Reference class with a Python ``__new__``: its ``tp_new`` is ``slot_tp_new``."""

    def __new__(cls) -> _PythonNewProbe:
        """Return a plain instance (never called; only the slot pointer is read)."""

        return object.__new__(cls)


class _PythonNewProbeTwin:
    """Second Python-``__new__`` class: shares ``slot_tp_new`` with the first."""

    def __new__(cls) -> _PythonNewProbeTwin:
        """Return a plain instance (never called; only the slot pointer is read)."""

        return object.__new__(cls)


def _layout_matches(cls: type) -> bool:
    """Return whether the mirrored fields agree with ``cls``'s own type attributes.

    Every compared field is read by ``type``'s own members straight from the
    struct (``__basicsize__`` is ``tp_basicsize``, ``__weakrefoffset__`` is
    ``tp_weaklistoffset``, ``__base__`` is ``tp_base``, ...), so equality pins
    each field's offset; the last three sit between ``tp_doc`` and ``tp_new``.
    """

    view = _type_view(cls)
    name = view.tp_name or b""
    base = cls.__base__
    return (
        name.rsplit(b".", 1)[-1] == cls.__name__.encode()
        and view.tp_basicsize == cls.__basicsize__
        and view.tp_itemsize == cls.__itemsize__
        and view.tp_flags == cls.__flags__
        and view.tp_weaklistoffset == cls.__weakrefoffset__
        and view.tp_base == (id(base) if base is not None else None)
        and view.tp_dictoffset == cls.__dictoffset__
        and bool(view.tp_new)
    )


def _tp_new_slot_verified() -> bool:
    """Return whether the mirrored ``tp_new`` field behaves like CPython's slot.

    Read-only: ``object`` and a class inheriting its constructor must share one
    ``tp_new``, and two classes defining a Python ``__new__`` must share a
    different one (``slot_tp_new``). A mirror whose ``tp_new`` offset drifted
    lands on a neighbouring field (``tp_alloc``, ``tp_free``, ...) where those
    identities do not hold.
    """

    object_new = _type_view(object).tp_new
    trampoline = _type_view(_PythonNewProbe).tp_new
    return (
        bool(object_new)
        and bool(trampoline)
        and _type_view(_InheritedNewProbe).tp_new == object_new
        and _type_view(_PythonNewProbeTwin).tp_new == trampoline
        and trampoline != object_new
    )


def probe_type_new_slot_patch(cls: Any) -> bool:
    """Return whether ``cls``'s type-object layout matches the ctypes mirror.

    Read-only; nothing is written before every check passes.

    Parameters
    ----------
    cls:
        A class to patch (``torch.FloatTensor``, ``Variable``).

    Returns
    -------
    bool
        True on CPython when the mirrored ``tp_name``, ``tp_basicsize``,
        ``tp_itemsize``, ``tp_flags``, ``tp_weaklistoffset``, ``tp_base`` and
        ``tp_dictoffset`` agree with the live class and with the reference
        classes, ``tp_new`` is populated, and the ``tp_new`` field itself passes
        :func:`_tp_new_slot_verified`.
    """

    if sys.implementation.name != "cpython" or not isinstance(cls, type):
        return False
    try:
        return (
            all(_layout_matches(ref) for ref in (cls, _InheritedNewProbe, _PythonNewProbe))
            and _tp_new_slot_verified()
        )
    except Exception:
        return False


def has_python_new_trampoline(cls: type) -> bool:
    """Return whether ``cls``'s constructor slot is CPython's ``slot_tp_new``.

    True when a Python-level ``__new__`` (``staticmethod`` or function) is in
    effect for ``cls``, either its own or one set on a base and propagated:
    the C ``tp_new`` is then no longer reachable through the slot. Only valid
    after :func:`probe_type_new_slot_patch` passed.

    Parameters
    ----------
    cls:
        Class to inspect.

    Returns
    -------
    bool
        Whether ``tp_new`` equals the reference Python-``__new__`` trampoline.
    """

    return bool(_type_view(cls).tp_new == _type_view(_PythonNewProbe).tp_new)


def _set_or_del_type_attr(cls: type, name: str, value: Any) -> None:
    """Set (or delete, for ``_ABSENT``) a class attribute, even on an immutable type.

    Only the immutable bit is toggled: ``type.__setattr__`` itself rewrites
    other flag bits (``PyType_Modified`` drops the version-tag bit), so the
    saved flag word is never written back wholesale.
    """

    view = _type_view(cls)
    was_immutable = bool(view.tp_flags & _PY_TPFLAGS_IMMUTABLETYPE)
    if was_immutable:
        view.tp_flags = view.tp_flags & ~_PY_TPFLAGS_IMMUTABLETYPE
    try:
        if value is _ABSENT:
            type.__delattr__(cls, name)
        else:
            type.__setattr__(cls, name, value)
    finally:
        if was_immutable:
            view.tp_flags = view.tp_flags | _PY_TPFLAGS_IMMUTABLETYPE


@dataclass(frozen=True)
class TypeNewPatch:
    """Everything needed to undo one ``__new__`` override exactly.

    Attributes
    ----------
    cls:
        The patched class.
    original_tp_new:
        The C ``tp_new`` pointer before patching.
    original_dict_new:
        ``cls.__dict__["__new__"]`` before patching, or the module sentinel
        when the class inherited ``__new__``.
    installed_new:
        The ``staticmethod`` this patch put in ``cls.__dict__``.
    """

    cls: type
    original_tp_new: int
    original_dict_new: Any
    installed_new: staticmethod


def original_new_caller(cls: type) -> Callable[[type, tuple[Any, ...], dict[str, Any]], Any]:
    """Return a callable running ``cls``'s current C ``tp_new`` directly.

    Parameters
    ----------
    cls:
        Class whose (not yet patched) constructor slot to capture.

    Returns
    -------
    Callable[[type, tuple[Any, ...], dict[str, Any]], Any]
        ``call(subtype, args, kwargs)``, the eager constructor; an empty
        ``kwargs`` is passed as NULL, exactly as a plain call passes it.
    """

    tp_new = _TP_NEW_FUNCTYPE(_type_view(cls).tp_new)

    def call(subtype: type, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Run the captured C constructor."""

        return tp_new(subtype, args, kwargs if kwargs else ctypes.py_object())

    return call


def install_new_override(cls: type, new_impl: Callable[..., Any]) -> TypeNewPatch:
    """Install ``new_impl`` as ``cls.__new__`` and return the undo record.

    Parameters
    ----------
    cls:
        Class to patch.
    new_impl:
        ``new_impl(subtype, *args, **kwargs)``.

    Returns
    -------
    TypeNewPatch
        Record for :func:`restore_new_override`.
    """

    installed = staticmethod(new_impl)
    patch = TypeNewPatch(
        cls=cls,
        original_tp_new=int(_type_view(cls).tp_new or 0),
        original_dict_new=cls.__dict__.get("__new__", _ABSENT),
        installed_new=installed,
    )
    _set_or_del_type_attr(cls, "__new__", installed)
    return patch


def restore_new_override(patch: TypeNewPatch) -> bool:
    """Undo one override when it is still the class's ``__new__``.

    Parameters
    ----------
    patch:
        Record returned by :func:`install_new_override`.

    Returns
    -------
    bool
        True when restored; False when a foreign patch now sits in
        ``__new__`` (it is left untouched, never clobbered).
    """

    cls = patch.cls
    if cls.__dict__.get("__new__") is not patch.installed_new:
        return False
    _set_or_del_type_attr(cls, "__new__", patch.original_dict_new)
    # ``type.__setattr__`` cannot put a C ``tp_new`` back (it keeps the
    # ``slot_tp_new`` trampoline when the restored entry is the C wrapper), so
    # the saved pointer is written back directly.
    _type_view(cls).tp_new = patch.original_tp_new
    ctypes.pythonapi.PyType_Modified(ctypes.py_object(cls))
    _restore_inheriting_subclasses(cls, patch.original_tp_new)
    return True


def _restore_inheriting_subclasses(cls: type, original_tp_new: int) -> None:
    """Put the C ``tp_new`` back on Python subclasses that inherited the override.

    Installing a Python ``__new__`` propagates ``slot_tp_new`` into every
    subclass without its own ``__new__`` (a user ``class V(Variable)``), and
    the ``type.__setattr__`` restore leaves them on that trampoline. Each such
    subclass (recursively) gets the saved pointer back, as at its creation; a
    subclass defining its own ``__new__`` keeps its slot, and so do its
    descendants.
    """

    for sub in cls.__subclasses__():
        if "__new__" in sub.__dict__ or not has_python_new_trampoline(sub):
            continue
        _type_view(sub).tp_new = original_tp_new
        ctypes.pythonapi.PyType_Modified(ctypes.py_object(sub))
        _restore_inheriting_subclasses(sub, original_tp_new)
