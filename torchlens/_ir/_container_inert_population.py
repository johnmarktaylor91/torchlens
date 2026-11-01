"""Inert-population probes for output-container reconstruction.

Reconstruction allocates dataclass / ``ModelOutput`` containers through the type's builtin
``__new__`` and writes captured fields straight into ``obj.__dict__``. These helpers decide
whether a resolved type permits that without running attacker-reachable code (an overridden
``__new__``, a ``__slots__`` type, or a data descriptor would). ``container.py`` consumes them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

# The ONLY ``__new__`` allocators that run no Python code at instance allocation:
# ``object.__new__`` (plain dataclasses) and the builtin ``dict.__new__`` (shared by
# ``OrderedDict``/``defaultdict`` and thus inherited by ``OrderedDict``-derived ModelOutput
# classes). A type that OVERRIDES ``__new__`` matches neither of these identities and is
# refused, so non-invoking reconstruction never executes an attacker-supplied allocator.
_INERT_NEW_ALLOCATORS: tuple[Any, ...] = (object.__new__, dict.__new__)


def _type_has_safe_new(container_type: type[Any]) -> bool:
    """Return whether constructing ``container_type`` via ``__new__`` runs no attacker code.

    Reconstructing a ``dataclass`` / ``hf_model_output`` output from an untrusted spec must
    NEVER invoke arbitrary constructor code. We build the instance with ``cls.__new__(cls)``
    and then set only the CAPTURED fields inertly (``object.__setattr__`` / the builtin dict
    populator), so no attacker ``__init__`` / ``__post_init__`` runs. This guard makes sure
    ``cls.__new__(cls)`` itself cannot run attacker code: a normal dataclass leaves ``__new__``
    as ``object.__new__`` and an ``OrderedDict``-derived ModelOutput inherits the inert builtin
    ``dict.__new__``, whereas a type that OVERRIDES ``__new__`` is exotic and is refused here
    (the caller then falls back to the plain namespace / mapping shape rather than executing
    that ``__new__``).

    Parameters
    ----------
    container_type:
        Already-resolved, admissible container class.

    Returns
    -------
    bool
        True when ``container_type.__new__`` is an inert builtin allocator.
    """

    return any(container_type.__new__ is inert_new for inert_new in _INERT_NEW_ALLOCATORS)


def _inert_new(container_type: type[Any]) -> Any:
    """Allocate an instance via the type's ``__new__`` WITHOUT running ``__init__``.

    The caller has already verified (``_type_has_safe_new``) that ``container_type.__new__``
    is an inert builtin allocator, so this executes no attacker code. We call it through
    ``container_type.__new__`` rather than ``object.__new__(container_type)`` because
    ``object.__new__`` refuses ``dict``/``OrderedDict`` subclasses ("not safe"), while the
    inherited builtin ``__new__`` on the class allocates them correctly.

    Parameters
    ----------
    container_type:
        Admissible container class whose ``__new__`` was checked inert.

    Returns
    -------
    Any
        A freshly allocated, uninitialized instance of ``container_type``.
    """

    new = cast("Callable[[type[Any]], Any]", container_type.__new__)
    return new(container_type)


def _name_is_data_descriptor(container_type: type[Any], name: Any) -> bool:
    """Return whether ``name`` resolves to a DATA descriptor in ``container_type``'s MRO.

    A data descriptor (``property`` with a setter, or any class attribute whose type
    defines ``__set__`` / ``__delete__``) HONORS ``__set__`` even through
    ``object.__setattr__``. Populating such a name during reconstruction would fire
    attacker-reachable descriptor code (the SEC1 surface) AND could not be set faithfully
    (a plain ``obj.__dict__`` write is shadowed by the data descriptor on read). We resolve
    from the FIRST MRO class that defines the name -- the one that wins attribute lookup.

    Parameters
    ----------
    container_type:
        Resolved container class.
    name:
        Captured field / key name to classify.

    Returns
    -------
    bool
        True when ``name`` is a class-defined data descriptor.
    """

    for klass in getattr(container_type, "__mro__", (container_type,)):
        class_dict = getattr(klass, "__dict__", {})
        if name in class_dict:
            attribute = class_dict[name]
            attribute_type = type(attribute)
            return hasattr(attribute_type, "__set__") or hasattr(attribute_type, "__delete__")
    return False


def _type_has_instance_dict(container_type: type[Any]) -> bool:
    """Return whether instances of ``container_type`` carry a per-instance ``__dict__``.

    Non-invoking field population writes directly to ``obj.__dict__``; a ``__slots__`` type
    with no instance ``__dict__`` cannot be populated inertly (a plain slot write would go
    through the slot data descriptor). Such a type is refused so reconstruction stays inert.

    Parameters
    ----------
    container_type:
        Resolved container class.

    Returns
    -------
    bool
        True when instances expose a writable ``__dict__``.
    """

    return any(
        "__dict__" in getattr(klass, "__dict__", {})
        for klass in getattr(container_type, "__mro__", (container_type,))
    )


def _fields_are_inert_settable(container_type: type[Any], names: tuple[Any, ...]) -> bool:
    """Return whether every captured field/key can be set on ``container_type`` inertly.

    Inert population requires (1) a per-instance ``__dict__`` to write into and (2) no
    captured name that resolves to a data descriptor (its ``__set__`` would fire attacker
    code and the write would not be faithful). When either fails, reconstruction falls back
    to the plain namespace / mapping shape rather than executing any descriptor code.

    Parameters
    ----------
    container_type:
        Resolved container class.
    names:
        Captured field / key names to populate.

    Returns
    -------
    bool
        True when all names can be populated via ``obj.__dict__`` without firing code.
    """

    if not _type_has_instance_dict(container_type):
        return False
    return not any(_name_is_data_descriptor(container_type, name) for name in names)
