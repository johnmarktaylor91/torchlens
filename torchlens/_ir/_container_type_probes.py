"""Type-structural construction probes for output-container reconstruction.

Reconstruction never invokes a container type's ``__new__``, ``__init__``,
``__post_init__`` or metaclass ``__call__`` (the SEC1 construction-gadget
surface). These probes read a resolved type's structure instead: whether it is a
genuine generated namedtuple (faithfulness), and whether a dataclass or its
metaclass defines construction code that could compute state the non-invoking
rebuild drops (lossiness). ``container.py`` consumes them.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import sys
import types
from collections.abc import Callable
from typing import Any, cast


def _is_generated_namedtuple_type(container_type: type[Any]) -> bool:
    """Return whether ``container_type`` is a genuine compiler-generated ``namedtuple`` class.

    A real ``collections.namedtuple`` is a ``tuple`` subclass carrying a ``_fields`` tuple of
    identifier strings, a resolved ``__module__``, and a compiler-GENERATED Python ``__new__``
    (a :class:`types.FunctionType`, unlike the inert C ``tuple.__new__`` or a structseq's
    builtin ``__new__``).

    This gate does NOT decide safety -- reconstruction ALWAYS goes through the inert builtin
    ``tuple.__new__`` and NEVER invokes ``container_type.__new__``, so even a weaponized
    look-alike ``__new__`` cannot run. It decides FAITHFULNESS: a resolved type that looks like
    a genuine namedtuple reconstructs as its own type, while anything else (a bare ``tuple``
    subclass with no generated ``__new__``, a structseq, an arbitrary tuple gadget) falls back
    to a plain ``tuple``.

    Parameters
    ----------
    container_type:
        Already-resolved, admissible container class.

    Returns
    -------
    bool
        True when the class presents the standard generated-namedtuple shape.
    """

    if not issubclass(container_type, tuple):
        return False
    fields = getattr(container_type, "_fields", None)
    if not isinstance(fields, tuple) or not all(
        isinstance(name, str) and name.isidentifier() for name in fields
    ):
        return False
    if not isinstance(getattr(container_type, "__module__", None), str):
        return False
    return isinstance(getattr(container_type, "__new__", None), types.FunctionType)


def _dataclass_defines_post_init(container_type: type[Any]) -> bool:
    """Return whether a dataclass ``container_type`` defines its own ``__post_init__``.

    A user ``__post_init__`` can compute NON-field instance attributes (e.g. a value derived
    from a tensor) that the non-invoking rebuild -- which sets only captured fields -- silently
    drops. Because we never run ``__post_init__`` (that would be the SEC1 construction-gadget
    surface), its mere presence is the type-structural signal that reconstruction MAY be lossy.

    Parameters
    ----------
    container_type:
        Resolved dataclass container class.

    Returns
    -------
    bool
        True when ``__post_init__`` is defined anywhere in the MRO (excluding ``object``).
    """

    for klass in getattr(container_type, "__mro__", (container_type,)):
        if klass is object:
            continue
        if "__post_init__" in getattr(klass, "__dict__", {}):
            return True
    return False


def _generated_dataclass_init_marker() -> str | None:
    """Feature-detect the ``co_filename`` marker of a dataclasses-GENERATED ``__init__`` (r47 secB_1).

    ``dataclasses`` builds a generated ``__init__`` via ``exec`` -- on CPython 3.11 / 3.12 its
    ``__code__.co_filename`` is the literal ``"<string>"`` (a USER-authored ``__init__`` reads the
    real source-file path). Rather than HARD-CODE ``"<string>"`` (a version-parse), probe a fresh
    trivial dataclass on THIS interpreter and read its generated init's ``co_filename``. If a future
    CPython emits a different marker, the probe returns that marker so the discriminator still
    matches THIS interpreter's generated inits; a user dataclass in a different module then reads as
    foreign -> a benign OVER-trigger (fail-safe), never a false VERIFIED.
    """

    @dataclasses.dataclass
    class _Probe:
        """Throwaway dataclass whose generated ``__init__`` supplies the marker."""

        _x: int

    for klass in _Probe.__mro__:
        init = klass.__dict__.get("__init__")
        if init is not None and klass is not object:
            return getattr(getattr(init, "__code__", None), "co_filename", None)
    return None


_GENERATED_DC_INIT_MARKER: str | None = _generated_dataclass_init_marker()
"""``co_filename`` of a dataclasses-generated ``__init__`` on this interpreter (``"<string>"``)."""


def _dataclass_has_foreign_init(container_type: type[Any]) -> bool:
    """Return whether a dataclass's winning ``__init__`` is NOT the dataclasses-generated one (r47 secB_1).

    Sparse reconstruction sets only captured fields and never invokes ``__init__`` (that would be the
    SEC1 construction-gadget RCE surface), so a USER-authored ``__init__`` that computes a dropped
    tensor-derived non-field extra cannot be proven faithful -- exactly like ``__post_init__``. This
    is TYPE-observable: the winning (most-derived) ``__init__`` is the dataclasses-GENERATED one only
    when ``__dataclass_params__.init`` is true AND its ``__code__.co_filename`` equals the feature-
    detected generated marker. A generated field-mirroring init (including a dataclass that generates
    its OWN init shadowing an evil base) stays lossless; anything else -- a user init, ``init=False``
    with a custom init, an undetectable/absent code object -- is foreign (lossy).

    Fails CLOSED: an unknown or undetectable shape is treated as foreign (over-triggers to
    UNVERIFIABLE), never a false VERIFIED, because it returns ``not generated`` only after
    POSITIVELY confirming the generated marker.
    """

    for klass in getattr(container_type, "__mro__", (container_type,)):
        if "__init__" not in getattr(klass, "__dict__", {}):
            continue
        # First (most-derived) class that defines ``__init__`` -- the one that would run.
        if klass is object:
            return False
        init = klass.__dict__["__init__"]
        code = getattr(init, "__code__", None)
        params = getattr(container_type, "__dataclass_params__", None)
        generated = (
            getattr(params, "init", False)
            and _GENERATED_DC_INIT_MARKER is not None
            and getattr(code, "co_filename", None) == _GENERATED_DC_INIT_MARKER
        )
        return not generated
    return False


def _metaclass_defines_foreign_call(container_type: type[Any]) -> bool:
    """Return whether ``container_type``'s METACLASS defines a non-``type`` ``__call__`` (r51 secB_1).

    A custom metaclass ``__call__`` runs at construction and can compute a dropped tensor-derived
    non-field/non-key instance attribute that non-invoking reconstruction (``cls.__new__(cls)`` +
    inert writes) bypasses -- like ``__post_init__`` / a foreign ``__init__``, and not otherwise
    type-observable without INVOKING the metaclass (the SEC1 construction surface). Walks the
    metaclass MRO for the first class defining ``__call__``; ``type`` (the builtin, always present
    since every metaclass derives from ``type``) -> not foreign; anything else -> foreign (lossy).
    An ``ABCMeta`` metaclass defines no ``__call__`` of its own, so the walk falls through to
    ``type.__call__`` -> not foreign (no over-trigger for the ``ABCMeta``-metaclass control).
    """

    metaclass = type(container_type)
    for klass in getattr(metaclass, "__mro__", (metaclass,)):
        if "__call__" in getattr(klass, "__dict__", {}):
            return klass is not type
    return False


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


def _container_type_ref(value: Any) -> tuple[str | None, str | None]:
    """Return the importable type reference for ``value``."""

    cls = type(value)
    return cls.__module__, cls.__qualname__


def _is_namedtuple_instance(value: Any) -> bool:
    """Return whether ``value`` is a namedtuple instance."""

    return isinstance(value, tuple) and hasattr(value, "_fields")


def _is_hf_model_output(value: Any) -> bool:
    """Return whether ``value`` looks like a HuggingFace ``ModelOutput``."""

    cls = type(value)
    if any(
        base.__module__.startswith("transformers") and base.__name__ == "ModelOutput"
        for base in cls.__mro__
    ):
        return True
    return (
        (cls.__module__.startswith("transformers") or cls.__name__.endswith("ModelOutput"))
        and hasattr(value, "keys")
        and hasattr(value, "__getitem__")
    )


def _torch_return_type_fields(value: Any) -> tuple[str, ...]:
    """Best-effort field names for tuple-like torch return types."""

    if not isinstance(value, tuple):
        return ()
    n_fields = getattr(value, "n_fields", None)
    n_unnamed = getattr(value, "n_unnamed_fields", 0)
    if not isinstance(n_fields, int) or n_fields <= 0 or n_unnamed:
        return ()
    field_names = tuple(
        match.group(1)
        for match in re.finditer(r"^\s*([A-Za-z_]\w*)=", repr(value), flags=re.MULTILINE)
    )
    if len(field_names) != n_fields:
        return ()
    return field_names


def _is_hf_model_output_type(container_type: type[Any]) -> bool:
    """Return whether ``container_type`` is a HuggingFace ``ModelOutput`` container class.

    Mirrors the capture-side ``ir.container_registry._is_hf_model_output`` heuristic at
    the class level. Inspects only the ALREADY-RESOLVED class (its MRO / name / mapping
    protocol); it never imports ``transformers`` to satisfy an attacker.

    Parameters
    ----------
    container_type:
        Resolved class to classify.

    Returns
    -------
    bool
        True when the class is a real ``transformers`` ``ModelOutput`` subclass or a
        HuggingFace-style mapping output class.
    """

    for base in getattr(container_type, "__mro__", ()):
        if base.__module__.startswith("transformers") and base.__name__ == "ModelOutput":
            return True
    return (
        (
            container_type.__module__.startswith("transformers")
            or container_type.__name__.endswith("ModelOutput")
        )
        and hasattr(container_type, "keys")
        and hasattr(container_type, "__getitem__")
    )


def _is_trusted_transformers_output_type(container_type: type[Any], spec: Any | None) -> bool:
    """Return whether ``container_type`` is a genuine ``transformers`` output by resolution authority.

    r42 secB_1: mirrors :func:`_is_trusted_structseq_type`. The trust decision keys on the SPEC's
    resolution authority (``spec.type_module`` / ``spec.type_qualname``), never the resolved
    class's spoofable ``__module__`` string nor a loose ``startswith("transformers")`` prefix
    (which matches ``transformers_evil``). Trust requires ALL of: the spec named the genuine
    package (``transformers`` or ``transformers.``-prefixed), that module is actually loaded in
    ``sys.modules``, and static re-resolution of ``spec.type_qualname`` from it returns the
    IDENTICAL class object. A ``None`` spec (a legacy direct call with no resolution authority)
    is never trusted -- fail closed.
    """

    if spec is None:
        return False
    module = spec.type_module or ""
    if module != "transformers" and not module.startswith("transformers."):
        return False
    if not spec.type_qualname:
        return False
    loaded = sys.modules.get(module)
    if loaded is None:
        return False
    resolved: Any = loaded
    try:
        for attribute_name in spec.type_qualname.split("."):
            resolved = inspect.getattr_static(resolved, attribute_name)
    except AttributeError:
        return False
    return resolved is container_type
