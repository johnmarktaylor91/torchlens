"""Leaf dataclasses for TorchLens output container structure."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass
import inspect
import sys
import types
from typing import Any, ClassVar, Literal, TypeAlias, cast

from .._io import FieldPolicy

# ``defaultdict`` factory callables restorable on load WITHOUT importing an
# arbitrary callable. Mirrors ``torchlens.backends.torch.ops._SAFE_DEFAULT_FACTORIES``;
# a factory outside this allowlist is recorded opaque at capture, never here.
_SAFE_DEFAULT_FACTORIES: dict[str, Any] = {
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "frozenset": frozenset,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "bytes": bytes,
    "bytearray": bytearray,
    "complex": complex,
}


@dataclass(frozen=True)
class TupleIndex:
    """Index component for tuple/list output paths."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {"index": FieldPolicy.KEEP}

    index: int


@dataclass(frozen=True)
class DictKey:
    """Key component for dict output paths."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {"key": FieldPolicy.KEEP}

    key: Any


@dataclass(frozen=True)
class NamedField:
    """Field-name component for namedtuple output paths."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {"name": FieldPolicy.KEEP}

    name: str


@dataclass(frozen=True)
class DataclassField:
    """Field-name component for dataclass output paths."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {"name": FieldPolicy.KEEP}

    name: str


@dataclass(frozen=True)
class HFKey:
    """Key component for HuggingFace ``ModelOutput`` output paths."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {"key": FieldPolicy.KEEP}

    key: Any


OutputPathComponent: TypeAlias = (
    TupleIndex | DictKey | NamedField | DataclassField | HFKey | str | int
)


@dataclass(frozen=True)
class ContainerSpec:
    """Portable description of an output container seen during capture."""

    PORTABLE_STATE_SPEC: ClassVar[dict[str, FieldPolicy]] = {
        "kind": FieldPolicy.KEEP,
        "length": FieldPolicy.KEEP,
        "keys": FieldPolicy.BLOB_RECURSIVE,
        "fields": FieldPolicy.KEEP,
        "type_module": FieldPolicy.KEEP,
        "type_qualname": FieldPolicy.KEEP,
        "child_specs": FieldPolicy.BLOB_RECURSIVE,
        "literal_value": FieldPolicy.BLOB_RECURSIVE,
        "aux_data": FieldPolicy.BLOB_RECURSIVE,
        "lossy_reconstruction": FieldPolicy.KEEP,
    }

    kind: Literal[
        "tuple",
        "list",
        "dict",
        "namedtuple",
        "dataclass",
        "hf_model_output",
        "literal",
        "opaque",
        "registered",
    ]
    length: int | None = None
    keys: tuple[Any, ...] = ()
    fields: tuple[str, ...] = ()
    type_module: str | None = None
    type_qualname: str | None = None
    child_specs: tuple[tuple[OutputPathComponent, "ContainerSpec"], ...] = ()
    literal_value: Any = None
    aux_data: Any = None
    lossy_reconstruction: bool = False


def rebuild_container_from_spec(spec: ContainerSpec, leaves: list[Any] | tuple[Any, ...]) -> Any:
    """Rebuild an output container from a spec and flat leaves.

    Parameters
    ----------
    spec:
        Container shape captured during output traversal.
    leaves:
        Flat leaf values in the same DFS order emitted by output traversal.

    Returns
    -------
    Any
        Container matching ``spec`` with tensor leaves filled from ``leaves``.

    Raises
    ------
    ValueError
        If the number of leaves does not match the container specification.
    """

    leaf_iter = iter(leaves)
    rebuilt = _rebuild_container_from_spec(spec, leaf_iter)
    sentinel = object()
    if next(leaf_iter, sentinel) is not sentinel:
        raise ValueError("Too many leaves supplied for ContainerSpec.")
    return rebuilt


def _rebuild_container_from_spec(spec: ContainerSpec, leaf_iter: Any) -> Any:
    """Rebuild one container node from a leaf iterator.

    Parameters
    ----------
    spec:
        Container node to rebuild.
    leaf_iter:
        Iterator over flat leaf values.

    Returns
    -------
    Any
        Rebuilt container node.
    """

    child_by_key = dict(spec.child_specs)
    if spec.kind == "literal":
        return spec.literal_value
    if spec.kind == "opaque":
        raise ValueError("Opaque ContainerSpec cannot be reconstructed.")
    if spec.kind in {"tuple", "list"}:
        values = [
            _rebuild_child_or_leaf(child_by_key, TupleIndex(index), leaf_iter)
            for index in range(spec.length or 0)
        ]
        return tuple(values) if spec.kind == "tuple" else values
    if spec.kind == "dict":
        rebuilt_items = {
            key: _rebuild_child_or_leaf(child_by_key, DictKey(key), leaf_iter) for key in spec.keys
        }
        if spec.type_module is None or spec.type_qualname is None:
            return rebuilt_items
        return _rebuild_dict_subtype(spec, rebuilt_items)
    if spec.kind == "namedtuple":
        values = [
            _rebuild_child_or_leaf(child_by_key, NamedField(field_name), leaf_iter)
            for field_name in spec.fields
        ]
        container_type = _resolve_container_type(spec)
        # r49 secB_1: the SHARED substitution predicate decides plain-tuple fallback, so the
        # load-time lossy gate and this reconstruction can never disagree. ``not substitute``
        # means the type is a trusted structseq OR a generated namedtuple; the inner check only
        # selects the correct INERT constructor for those two non-substituting cases.
        if container_type is not None and not _reconstruction_would_substitute_plain(
            container_type, "namedtuple", spec.fields, spec
        ):
            if _is_trusted_structseq_type(container_type, spec):
                # Genuine torch structseq (``torch.return_types.*``): reconstruct through its
                # own INERT builtin ``__new__`` (a single-iterable C constructor that runs no
                # arbitrary Python). Admissibility already pinned the RESOLVED ``__module__`` to
                # ``torch.return_types`` (not the attacker's spec string), so this is a genuine
                # torch C type, never an attacker-named look-alike.
                return container_type(values)
            # Real compiler-generated namedtuple: allocate the tuple INERTLY via the builtin
            # ``tuple.__new__`` and NEVER invoke ``container_type(*values)`` -- this is the RCE
            # fix. The old ``container_type(*values)`` ran the resolved type's ``__new__`` /
            # ``__init__``, so a spec naming a ``tuple``-subclass "namedtuple-like" type whose
            # ``__new__`` executes code was a load/run construction gadget. ``tuple.__new__``
            # bypasses that ``__new__`` entirely, mirroring the inert non-invoking reconstruction
            # used for the dataclass / HF branches.
            return tuple.__new__(container_type, values)
        return tuple(values)
    if spec.kind == "dataclass":
        field_values = {
            field_name: _rebuild_child_or_leaf(child_by_key, DataclassField(field_name), leaf_iter)
            for field_name in spec.fields
        }
        container_type = _resolve_container_type(spec)
        # r49 secB_1: shared substitution predicate (== the old inline
        # ``_type_has_safe_new and _fields_are_inert_settable`` conjunction) so gate and
        # reconstruction stay coupled.
        if container_type is not None and not _reconstruction_would_substitute_plain(
            container_type, "dataclass", spec.fields, spec
        ):
            return _construct_dataclass_without_init(container_type, field_values)
        return field_values
    if spec.kind == "hf_model_output":
        key_values = {
            key: _rebuild_child_or_leaf(child_by_key, HFKey(key), leaf_iter) for key in spec.keys
        }
        container_type = _resolve_container_type(spec)
        # r49 secB_1: shared substitution predicate (== the old inline
        # ``_type_has_safe_new and issubclass(dict) and _fields_are_inert_settable`` conjunction)
        # so gate and reconstruction stay coupled.
        if container_type is not None and not _reconstruction_would_substitute_plain(
            container_type, "hf_model_output", spec.keys, spec
        ):
            return _construct_model_output_without_init(container_type, key_values)
        return key_values
    if spec.kind == "registered":
        values = [
            _rebuild_child_or_leaf(child_by_key, TupleIndex(index), leaf_iter)
            for index in range(spec.length or 0)
        ]
        container_type = _resolve_container_type(spec)
        if container_type is None:
            raise ValueError(
                f"Registered container type {spec.type_module}.{spec.type_qualname} "
                "could not be imported."
            )
        registration = get_registered_container(container_type)
        if registration is None:
            raise ValueError(
                f"Container type {container_type.__qualname__!r} is not registered in this runtime."
            )
        return registration.unflatten(spec.aux_data, values)
    raise ValueError(f"Unsupported ContainerSpec kind {spec.kind!r}.")


def _rebuild_dict_subtype(spec: ContainerSpec, items: dict[Any, Any]) -> Any:
    """Rebuild a faithfully reconstructable mapping subtype (OrderedDict/defaultdict).

    Parameters
    ----------
    spec:
        ``dict``-kind spec whose ``type_module``/``type_qualname`` name a concrete
        mapping subclass and whose ``aux_data`` carries any reconstruction metadata.
    items:
        Already-rebuilt ``{key: value}`` pairs in captured order.

    Returns
    -------
    Any
        Instance of the exact mapping subtype.

    Raises
    ------
    ValueError
        If the subtype cannot be imported or a defaultdict factory is not on the
        safe allowlist (should not occur: such mappings are recorded opaque at
        capture and never reach reconstruction).
    """

    container_type = _resolve_container_type(spec)
    if container_type is None:
        raise ValueError(
            f"Mapping type {spec.type_module}.{spec.type_qualname} could not be imported."
        )
    if issubclass(container_type, defaultdict):
        aux = spec.aux_data if isinstance(spec.aux_data, dict) else {}
        factory_name = aux.get("default_factory")
        if factory_name is None:
            return container_type(None, items)
        factory = _SAFE_DEFAULT_FACTORIES.get(factory_name)
        if factory is None:
            raise ValueError(f"Unsafe defaultdict factory {factory_name!r} for reconstruction.")
        return container_type(factory, items)
    return container_type(items)


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


def reconstruction_is_lossy(value: Any, captured_names: tuple[Any, ...]) -> bool:
    """Return whether the non-invoking rebuild would DROP live output instance state.

    The field/key-only rebuild restores exactly ``captured_names`` into ``obj.__dict__``. It
    is LOSSY -- so the run must be reported UNVERIFIABLE, never a false VERIFIED -- when the
    live output carries state that rebuild cannot faithfully restore:

    * a ``__slots__`` layout with no instance ``__dict__`` (fields cannot be set inertly), or
    * a captured name that resolves to a data descriptor (cannot be set faithfully), or
    * a computed non-field/non-key instance attribute (e.g. a ``__post_init__`` value derived
      from a tensor) that rebuild neither captured nor can safely recompute.

    A ``None``-valued extra attribute (e.g. a standard HuggingFace ``ModelOutput`` field left
    unset and excluded from the mapping) is NOT treated as lossy: it carries no derived-from-
    tensor state that could go stale, keeping pure/standard outputs VERIFIED.

    Parameters
    ----------
    value:
        The live output container instance seen at capture time.
    captured_names:
        Field names (dataclass) or mapping keys (``ModelOutput``) the rebuild will restore.

    Returns
    -------
    bool
        True when reconstruction would be lossy for this output.
    """

    container_type = type(value)
    instance_dict = getattr(value, "__dict__", None)
    if instance_dict is None:
        return True
    for name in captured_names:
        if _name_is_data_descriptor(container_type, name):
            return True
    captured = set(captured_names)
    for attribute_name, attribute_value in instance_dict.items():
        if attribute_name in captured:
            continue
        if attribute_value is None:
            continue
        return True
    return False


def _is_trusted_structseq_type(container_type: type[Any], spec: ContainerSpec) -> bool:
    """Return whether ``container_type`` is a genuine torch structseq (r39 secB_1, spec-aware).

    The structseq branch is the ONLY container-reconstruction path that invokes an arbitrary
    type's own ``__new__`` (``container_type(values)``), so its trust decision must key on the
    RESOLUTION AUTHORITY that supplied the type -- never the resolved class's ``__module__``
    attribute, which is an ordinary spoofable class attribute ANY tuple subclass can set to
    ``"torch.return_types"``. This gate requires ALL of:

    * ``spec.type_module == "torch.return_types"`` (the spec named the genuine module, not an
      alias module pointing look-alikes at it);
    * resolution begins from the REAL already-loaded ``sys.modules["torch.return_types"]``;
    * re-resolving ``spec.type_qualname`` from that module yields the IDENTICAL class object
      (identity, not a name/attr match) -- defeating both the spoofed-``__module__`` gadget and
      an alias-module trick; and
    * the class is a ``tuple`` subclass carrying the structseq ``n_fields`` marker.

    A genuine ``torch.return_types`` structseq is only ever reachable via
    ``spec.type_module == "torch.return_types"``, so this loses no legitimate capture while
    closing the latent arbitrary-``__new__`` sink.

    Parameters
    ----------
    container_type:
        Already-resolved, admissible container class.
    spec:
        The container spec being reconstructed (the resolution-authority evidence).

    Returns
    -------
    bool
        True only when the class is a genuine ``torch.return_types`` structseq by resolution
        authority AND structural markers.
    """

    if spec.type_module != "torch.return_types" or not spec.type_qualname:
        return False
    module = sys.modules.get("torch.return_types")
    if module is None:
        return False
    resolved: Any = module
    try:
        for attribute_name in spec.type_qualname.split("."):
            resolved = inspect.getattr_static(resolved, attribute_name)
    except AttributeError:
        return False
    if resolved is not container_type:
        return False
    return (
        isinstance(container_type, type)
        and issubclass(container_type, tuple)
        and hasattr(container_type, "n_fields")
    )


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


# Classes whose ``__init__`` only mirrors declared fields into the mapping/attribute views of a
# ``ModelOutput`` (or allocates the mapping) and therefore cannot silently compute a dropped
# tensor-derived extra attribute: the builtin mapping bases and the ``transformers`` ModelOutput
# family. A resolved ``__init__`` provided by any of these is trusted; anything else is a custom
# constructor and is treated as a lossy black box (see ``_model_output_has_foreign_init``).
_TRUSTED_MODEL_OUTPUT_INIT_BASES: tuple[type[Any], ...] = (object, dict, OrderedDict)


def _is_trusted_transformers_output_type(
    container_type: type[Any], spec: "ContainerSpec | None"
) -> bool:
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


def _model_output_has_foreign_init(
    container_type: type[Any], spec: "ContainerSpec | None" = None
) -> bool:
    """Return whether an ``hf_model_output`` type's own ``__init__`` is an untrusted constructor.

    A custom (non-dataclass) ``dict``-subclass ``ModelOutput`` can compute, in its ``__init__``,
    a NON-``None`` non-key instance attribute derived from a tensor (e.g. ``self.double = x * 2``)
    that the non-invoking rebuild -- which restores only the captured mapping keys -- silently
    drops. Unlike the ``__post_init__`` / ``__slots__`` / data-descriptor signals, that computed
    extra is NOT observable from the type alone: the ONLY way to see it is to RUN ``__init__``,
    which IS the construction-gadget RCE surface we must never invoke on an untrusted bundle. So
    we use a TYPE-observable PROXY: the class that provides the winning (most-derived) ``__init__``.
    If that ``__init__`` comes from a trusted populator -- the builtin mapping bases
    (``object`` / ``dict`` / ``OrderedDict``) or a ``transformers`` ModelOutput-family class whose
    generated / field-mirroring ``__init__`` only reflects declared fields -- reconstruction is
    faithful. Any OTHER class defining ``__init__`` is a custom constructor whose effect we cannot
    verify without running it, so the node is flagged lossy.

    This DELIBERATELY over-triggers: a genuine hand-rolled custom ``ModelOutput`` whose ``__init__``
    happens to set only its keys is still marked lossy (reported UNVERIFIABLE, never a false
    VERIFIED). That is the intended honest fail-closed -- the tripwire refuses to bless an output it
    cannot prove was reconstructed faithfully. Real ``transformers`` outputs and plain no-``__init__``
    mapping subclasses stay eligible for VERIFIED.

    Parameters
    ----------
    container_type:
        Resolved ``hf_model_output`` container class.

    Returns
    -------
    bool
        True when the winning ``__init__`` is defined by an untrusted (non-mapping,
        non-``transformers``) class.
    """

    for klass in getattr(container_type, "__mro__", (container_type,)):
        if "__init__" not in getattr(klass, "__dict__", {}):
            continue
        # First (most-derived) class that defines ``__init__`` -- the one that would run.
        if klass in _TRUSTED_MODEL_OUTPUT_INIT_BASES:
            return False
        # r42 secB_1: trust a non-base field-mirroring init ONLY when the CONTAINER type is a
        # genuine ``transformers`` ModelOutput by RESOLUTION AUTHORITY (identity re-resolution
        # from the real loaded package), never the spoofable / loose ``__module__`` prefix.
        # Unresolved or non-identical -> foreign (fail closed to lossy).
        return not _is_trusted_transformers_output_type(container_type, spec)
    return False


def _reconstruction_would_substitute_plain(
    container_type: type[Any],
    kind: str,
    names: tuple[Any, ...],
    spec: "ContainerSpec | None",
) -> bool:
    """Return whether ``_rebuild_container_from_spec`` would substitute a PLAIN container (r49 secB_1).

    The SINGLE shared substitution criterion consulted by BOTH the reconstruction path
    (:func:`_rebuild_container_from_spec`) AND the load-time forged-flag lossy gate
    (:func:`reconstruction_is_lossy_by_type` / ``_spec_node_reconstruction_lossy``). Reconstruction
    falls back to a plain ``dict`` / ``tuple`` -- dropping the recorded container TYPE and any
    ``__new__``-computed state -- exactly when the type's ``__new__`` is not an inert allocator, its
    fields are not inertly settable, or (namedtuple) it is neither a generated namedtuple nor a
    trusted structseq. The gate must treat every such case as lossy, else a forged
    ``lossy_reconstruction=False`` yields a false ``VERIFIED`` on a type-substituted output (the r48
    secB_1 ``__new__`` / generated-namedtuple hole).

    Coupling is enforced by SHARED CODE: because reconstruction and the gate call THIS predicate,
    a future gate/reconstruction divergence is a failing coverage meta-test rather than a silent
    false ``VERIFIED``. Over-trigger is safe -- a plain dataclass (``object.__new__``, inert fields)
    and a real ``collections.namedtuple`` / ``typing.NamedTuple`` stay not-substituted -> VERIFIED.
    """

    if kind == "dataclass":
        return not (
            _type_has_safe_new(container_type) and _fields_are_inert_settable(container_type, names)
        )
    if kind == "hf_model_output":
        return not (
            _type_has_safe_new(container_type)
            and issubclass(container_type, dict)
            and _fields_are_inert_settable(container_type, names)
        )
    if kind == "namedtuple":
        return not (
            (spec is not None and _is_trusted_structseq_type(container_type, spec))
            or _is_generated_namedtuple_type(container_type)
        )
    return False


def reconstruction_is_lossy_by_type(
    container_type: type[Any],
    captured_names: tuple[Any, ...],
    kind: str,
    spec: "ContainerSpec | None" = None,
) -> bool:
    """Recompute reconstruction lossiness from the RESOLVED type at LOAD time.

    The persisted :attr:`ContainerSpec.lossy_reconstruction` flag is computed at CAPTURE and
    is attacker-controlled in an untrusted bundle: a forged ``False`` would force a false
    ``VERIFIED`` on a genuinely lossy reconstruction. This function re-derives lossiness
    INDEPENDENTLY from the resolved type (the one trustworthy input at load), mirroring the
    r25 :func:`reconstruction_is_lossy` criteria that are TYPE-observable:

    * a ``__slots__`` layout with no instance ``__dict__`` (fields cannot be set inertly), or
    * a captured name that resolves to a data descriptor (cannot be set faithfully), or
    * (dataclass kind only) a user ``__post_init__`` that may compute dropped non-field state, or
    * (dataclass kind only, r47 secB_1) a winning ``__init__`` that is NOT the dataclasses-generated
      one (a user-authored / foreign constructor that may compute dropped non-field state -- detected
      by the generated init's feature-detected ``co_filename`` marker, see
      ``_dataclass_has_foreign_init``). This defeats a forged ``lossy_reconstruction=False`` naming a
      custom-init dataclass without invoking its constructor.
    * (dataclass + ``hf_model_output`` kinds, r51 secB_1) a METACLASS (``type(container_type)``) that
      defines a ``__call__`` other than the builtin ``type.__call__`` -- a fourth computed-dropped-
      state constructor hook that non-invoking reconstruction (``cls.__new__(cls)``) bypasses,
      applied GATE-DIRECT so reconstruction still rebuilds the correct inert type while the verdict
      fail-closes to lossy (see ``_metaclass_defines_foreign_call``). A plain-``type`` metaclass, a
      ``__call__``-free ``ABCMeta``, and a real namedtuple stay VERIFIED-eligible (no over-trigger);
      the ``namedtuple`` kind is separately covered by ``namedtuple_type_can_carry_instance_state``.

    The ``__post_init__`` signal is applied ONLY to the plain-``dataclass`` kind: HuggingFace
    ``ModelOutput`` dataclasses use ``__post_init__`` solely to populate their mapping from
    fields (attrs == keys), so applying it to the ``hf_model_output`` kind would over-trigger
    the standard/real ModelOutput case that must stay VERIFIED.

    For the ``hf_model_output`` kind, the r27 hardening adds an INDEPENDENT type-observable proxy
    for the r25 "computed non-``None`` extra attr on a custom (non-dataclass) ModelOutput
    ``__init__``" case, which is otherwise invisible without INVOKING ``__init__`` (the RCE
    construction-gadget surface): if the resolved type's own (most-derived) ``__init__`` is a
    custom constructor -- one NOT provided by the trusted mapping bases (``dict`` / ``OrderedDict``)
    or a ``transformers`` ModelOutput-family class -- reconstruction is flagged lossy (see
    ``_model_output_has_foreign_init``). This defeats a forged ``lossy_reconstruction=False`` on
    such a custom ModelOutput without executing its constructor, at the honest cost of marking
    some genuine hand-rolled custom ModelOutputs UNVERIFIABLE. The persisted flag remains a
    supplementary (never sole) signal so any remaining not-type-observable case still reports lossy.

    Parameters
    ----------
    container_type:
        Resolved container class named by the spec.
    captured_names:
        Field names (dataclass) or mapping keys (``ModelOutput``) the rebuild will restore.
    kind:
        The spec ``kind`` (``"dataclass"`` or ``"hf_model_output"``).

    Returns
    -------
    bool
        True when reconstructing ``container_type`` from ``captured_names`` is lossy.
    """

    if not _type_has_instance_dict(container_type):
        return True
    if any(_name_is_data_descriptor(container_type, name) for name in captured_names):
        return True
    if kind == "dataclass" and (
        _dataclass_defines_post_init(container_type) or _dataclass_has_foreign_init(container_type)
    ):
        return True
    if kind == "hf_model_output" and _model_output_has_foreign_init(container_type, spec):
        return True
    # r51 secB_1: a custom metaclass ``__call__`` is a FOURTH computed-dropped-state constructor
    # hook (alongside ``__post_init__`` / foreign ``__init__`` / non-inert ``__new__``). It can
    # compute a dropped tensor-derived extra that non-invoking reconstruction (``cls.__new__(cls)``,
    # which bypasses the metaclass ``__call__`` entirely) drops, and is not type-observable without
    # INVOKING the metaclass (the SEC1 surface). Applied GATE-DIRECT (mirroring
    # ``_dataclass_has_foreign_init``), NOT via ``_reconstruction_would_substitute_plain`` -- the
    # metaclass leaves ``__new__``/fields inert, so reconstruction still rebuilds the correct inert
    # type (higher fidelity) while the honesty verdict fail-closes to ``unverifiable``. The
    # ``namedtuple`` kind is deliberately excluded (already covered by
    # ``namedtuple_type_can_carry_instance_state``: a ``__slots__=()`` namedtuple has no instance
    # ``__dict__`` for the extra and an unslotted tuple subclass is separately flagged).
    if kind in ("dataclass", "hf_model_output") and _metaclass_defines_foreign_call(container_type):
        return True
    # r49 secB_1: couple the gate to reconstruction's OWN substitution criterion. If
    # ``_rebuild_container_from_spec`` would substitute a PLAIN container for this type (its
    # ``__new__`` is not an inert allocator, or its fields are not inertly settable), the
    # recorded type and any ``__new__``-computed state are dropped -- treat it as lossy,
    # mirroring reconstruction exactly through the shared predicate.
    if _reconstruction_would_substitute_plain(container_type, kind, captured_names, spec):
        return True
    return False


def _construct_dataclass_without_init(
    container_type: type[Any], field_values: dict[str, Any]
) -> Any:
    """Rebuild a dataclass output WITHOUT invoking its ``__init__`` / ``__post_init__``.

    Uses ``cls.__new__(cls)`` and writes each CAPTURED field DIRECTLY into ``obj.__dict__``.
    Writing through ``obj.__dict__`` -- NOT ``object.__setattr__`` -- bypasses ANY data
    descriptor ``__set__`` (the SEC1 construction-gadget surface) and a frozen dataclass's
    ``__setattr__``, so no attacker-chosen code runs and frozen outputs rebuild faithfully.
    The caller has already verified via ``_fields_are_inert_settable`` that the type carries a
    per-instance ``__dict__`` and that no captured field is a data descriptor.

    Parameters
    ----------
    container_type:
        Admissible dataclass type with an inert ``__new__`` (see ``_type_has_safe_new``).
    field_values:
        Captured ``{field_name: value}`` pairs to set on the instance.

    Returns
    -------
    Any
        Reconstructed dataclass instance with the captured field values.
    """

    obj = _inert_new(container_type)
    instance_dict = obj.__dict__
    for field_name, value in field_values.items():
        instance_dict[field_name] = value
    return obj


def _construct_model_output_without_init(
    container_type: type[Any], key_values: dict[Any, Any]
) -> Any:
    """Rebuild a HuggingFace ``ModelOutput`` output WITHOUT invoking attacker constructor code.

    A ``ModelOutput`` is a ``dict`` subclass whose ``__init__``/``__post_init__`` populate its
    mapping view (and HuggingFace-style attribute aliases) from its set fields. We replicate
    that populated state inertly: ``cls.__new__(cls)`` (an empty ``dict``-subclass instance),
    then for each captured ``(key, value)`` set the mapping entry via the BASE
    ``dict``/``OrderedDict`` ``__setitem__`` (never the subclass's possibly-overridden
    ``__setitem__``) and the attribute alias via a DIRECT ``obj.__dict__`` write (never
    ``object.__setattr__``, which fires data descriptors) -- WITHOUT running the type's own
    ``__setitem__`` / ``__setattr__`` / ``__init__`` / ``__post_init__``.

    The mapping entry uses ``OrderedDict.__setitem__`` for ``OrderedDict`` subclasses (real
    transformers ``ModelOutput``) and the builtin ``dict.__setitem__`` for plain ``dict``
    subclasses; ``dict.__setitem__`` on an ``OrderedDict`` corrupts its ordering bookkeeping,
    so the correct inert populator is selected per base. The caller has already verified via
    ``_fields_are_inert_settable`` that the type carries a per-instance ``__dict__`` and that
    no captured key is a data descriptor.

    Parameters
    ----------
    container_type:
        Admissible ``dict``-derived ModelOutput type with an inert ``__new__``.
    key_values:
        Captured ``{key: value}`` pairs in capture order.

    Returns
    -------
    Any
        Reconstructed ModelOutput with the captured mapping view and attributes.
    """

    setitem = (
        OrderedDict.__setitem__ if issubclass(container_type, OrderedDict) else dict.__setitem__
    )
    obj = _inert_new(container_type)
    instance_dict = obj.__dict__
    for key, value in key_values.items():
        setitem(obj, key, value)
        instance_dict[key] = value
    return obj


def _rebuild_child_or_leaf(
    child_by_key: dict[OutputPathComponent, ContainerSpec],
    component: OutputPathComponent,
    leaf_iter: Any,
) -> Any:
    """Return a rebuilt child container or the next flat leaf.

    Parameters
    ----------
    child_by_key:
        Mapping from child path component to nested container spec.
    component:
        Path component to rebuild.
    leaf_iter:
        Iterator over flat leaf values.

    Returns
    -------
    Any
        Rebuilt nested container or next leaf value.

    Raises
    ------
    ValueError
        If a required leaf is missing.
    """

    child_spec = child_by_key.get(component)
    if child_spec is not None:
        return _rebuild_container_from_spec(child_spec, leaf_iter)
    try:
        return next(leaf_iter)
    except StopIteration as exc:
        raise ValueError("Not enough leaves supplied for ContainerSpec.") from exc


class ContainerReconstructionError(ValueError):
    """Raised when an output-container spec names a type that is not admissible.

    The output ``ContainerSpec`` is portable, attacker-influenceable data. Its
    ``type_module`` / ``type_qualname`` name the concrete container class to rebuild.
    This error is the default-deny tripwire: a spec that names a type outside the
    benign container allowlist for its ``kind`` is refused BEFORE any construction,
    mirroring the safe-unpickler's global denial. Subclasses ``ValueError`` so the
    runnable run wrapper reports it as a typed ``RunPreconditionError`` denial.
    """


# ``dict``-subtype output containers are only ever recorded for the two mapping
# classes ``torchlens.backends.torch.ops._mapping_reconstruction`` can faithfully
# rebuild; every other mapping is recorded ``opaque`` (honest-reject at save) and so
# never reaches reconstruction. Any other ``(module, qualname)`` for a ``dict`` kind
# is therefore a tampered / corrupt spec and is refused.
_ALLOWED_MAPPING_TYPE_REFS: frozenset[tuple[str, str]] = frozenset(
    {("collections", "OrderedDict"), ("collections", "defaultdict")}
)


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


def _container_type_is_admissible(container_type: type[Any], spec: ContainerSpec) -> bool:
    """Return whether ``container_type`` may be constructed for ``spec.kind``.

    Default-deny: a resolved type is admissible only when it structurally matches the
    benign container kind that capture recorded, so constructing it cannot execute
    arbitrary attacker code (e.g. ``subprocess.Popen`` fails every branch).

    Parameters
    ----------
    container_type:
        Already-resolved candidate class.
    spec:
        Container spec being reconstructed.

    Returns
    -------
    bool
        True when the type is an admissible container class for the spec's kind.
    """

    kind = spec.kind
    if kind == "namedtuple":
        # Default-deny STRUCTURALLY regardless of the attacker-supplied ``spec.type_module``
        # string (SEC2): the resolved type must genuinely be a ``tuple`` subclass, either a
        # real namedtuple (carries ``_fields``) or a torch structseq (``torch.return_types.*``
        # exposes ``n_fields`` and whose RESOLVED ``__module__`` -- not the spec string -- is
        # ``torch.return_types``). A type reached by dotted getattr traversal that is not a
        # tuple subclass (or a tuple subclass from any other module posing as structseq) is
        # refused before any construction, so trusting the module name cannot skip the check.
        if not issubclass(container_type, tuple):
            return False
        if hasattr(container_type, "_fields"):
            return True
        # r39 secB_1: a tuple subclass posing as a structseq is admissible ONLY when the
        # spec-aware resolution-authority gate confirms it -- never the spoofable
        # ``__module__`` class attribute.
        return _is_trusted_structseq_type(container_type, spec)
    if kind == "dataclass":
        return dataclasses.is_dataclass(container_type)
    if kind == "hf_model_output":
        return _is_hf_model_output_type(container_type)
    if kind == "dict":
        return (spec.type_module, spec.type_qualname) in _ALLOWED_MAPPING_TYPE_REFS
    if kind == "registered":
        return get_registered_container(container_type) is not None
    return False


# A container ``type_qualname`` names a concrete class produced by the traced model
# (``Outer.Inner`` at most a few components deep). A pathologically deep attacker qualname
# has no legitimate container use and only widens the static-traversal surface, so the walk
# is capped and an over-long reference falls back to the plain container shape.
_MAX_QUALNAME_DEPTH = 10


def _resolve_container_type(spec: ContainerSpec) -> type[Any] | None:
    """Resolve the concrete container type named by a spec under a default-deny gate.

    Security contract (this is the fix for the output-container reconstruction RCE and the
    r27 lazy-import / descriptor gadget on the dotted-name walk):

    * NEVER imports an attacker-named module -- importing runs top-level module code.
      The type is resolved ONLY from a module already present in ``sys.modules``; every
      legit container type was produced by the traced model, so its module is loaded.
    * Resolves the dotted ``type_qualname`` with :func:`inspect.getattr_static`, which reads
      each hop STATICALLY from the object's ``__dict__`` / MRO and NEVER invokes a module
      PEP-562 ``__getattr__`` or a class/metaclass descriptor / ``property.__get__``. Plain
      ``getattr`` would fire those on the ATTACKER-chosen name BEFORE the admissibility gate:
      e.g. ``type_module="torch"`` + ``type_qualname="onnx"`` triggered ``torch.__getattr__``
      and lazily imported ~48 torch submodules (running their top-level code) during a benign
      ``.run()`` -- and the same read fires on the honesty gate too. An unmaterialized lazy
      submodule is absent from ``__dict__``, so the static walk fails closed to the fallback
      and imports nothing. The qualname depth is also capped (see ``_MAX_QUALNAME_DEPTH``).
    * A resolved type is returned ONLY when it structurally matches the benign container
      kind recorded in the spec (see ``_container_type_is_admissible``); a resolved type
      that is NOT admissible (e.g. ``subprocess.Popen`` under a ``namedtuple`` kind) is a
      tampered spec and is refused with a typed ``ContainerReconstructionError`` BEFORE
      any construction.
    * An unresolvable reference (module not loaded / attribute missing / not a type)
      returns ``None`` so the caller falls back to its plain container shape, preserving
      the historical graceful behavior for genuinely-unavailable types.

    Parameters
    ----------
    spec:
        Container spec with optional type reference metadata.

    Returns
    -------
    type[Any] | None
        The admissible container type, or ``None`` when no type is recorded or the
        reference cannot be resolved to a loaded type.

    Raises
    ------
    ContainerReconstructionError
        If the reference resolves to a loaded type that is not admissible for the
        recorded container kind.
    """

    module_name = spec.type_module
    qualname = spec.type_qualname
    if module_name is None or qualname is None:
        return None
    module = sys.modules.get(module_name)
    if module is None:
        # Refuse to import an arbitrary bundle-named module; fall back to the plain
        # container shape rather than executing that module's top-level code.
        return None
    parts = qualname.split(".")
    if not parts or len(parts) > _MAX_QUALNAME_DEPTH:
        return None
    resolved: Any = module
    try:
        for attribute_name in parts:
            # STATIC resolution only: never invokes ``__getattr__`` / descriptor ``__get__``,
            # and each hop must be present in the object's ``__dict__`` / MRO or this raises.
            resolved = inspect.getattr_static(resolved, attribute_name)
    except AttributeError:
        return None
    if not isinstance(resolved, type):
        return None
    if not _container_type_is_admissible(resolved, spec):
        raise ContainerReconstructionError(
            "Refusing to reconstruct output container from disallowed type "
            f"{module_name}.{qualname} for container kind {spec.kind!r}."
        )
    return resolved


class RegisteredContainer:
    """Flatten/unflatten hooks for a user-registered container type."""

    def __init__(
        self,
        flatten: Callable[[Any], tuple[list[Any] | tuple[Any, ...], Any]],
        unflatten: Callable[[Any, list[Any]], Any],
        *,
        state_complete: bool = False,
    ) -> None:
        """Create a registered container hook pair.

        Parameters
        ----------
        flatten:
            Callable returning ``(children, aux_data)``.
        unflatten:
            Callable accepting ``(aux_data, children)``.
        state_complete:
            Explicit trusted declaration (r37 R11) that ``unflatten(aux_data,
            children)`` restores EVERY piece of per-instance state ``flatten``
            observed -- nothing is dropped. Defaults ``False``: a registration
            without the declaration keeps working for analysis capture, but a
            RUNNABLE save of an instance carrying extra ``__dict__`` state
            refuses rather than silently dropping it on replay.
        """

        self.flatten = flatten
        self.unflatten = unflatten
        self.state_complete = bool(state_complete)


_CONTAINER_REGISTRY: dict[type[Any], RegisteredContainer] = {}


def register_container(
    container_type: type[Any],
    flatten: Callable[[Any], tuple[list[Any] | tuple[Any, ...], Any]],
    unflatten: Callable[[Any, list[Any]], Any],
    *,
    state_complete: bool = False,
) -> None:
    """Register a custom container type for capture and reconstruction.

    Parameters
    ----------
    container_type:
        Runtime class to recognize as a container.
    flatten:
        Callable returning ``(children, aux_data)`` for an instance.
    unflatten:
        Callable accepting ``(aux_data, children)`` and returning an instance.
    state_complete:
        Trusted declaration that the hook pair round-trips ALL per-instance
        state (r37 R11); without it, runnable saves refuse instances carrying
        extra ``__dict__`` state instead of dropping it silently on replay.
    """

    _CONTAINER_REGISTRY[container_type] = RegisteredContainer(
        flatten, unflatten, state_complete=state_complete
    )


def get_registered_container(container_type: type[Any]) -> RegisteredContainer | None:
    """Return the registration for a container type, if present.

    Parameters
    ----------
    container_type:
        Runtime class to look up.

    Returns
    -------
    RegisteredContainer | None
        Registered hook pair or ``None``.
    """

    for registered_type, registration in _CONTAINER_REGISTRY.items():
        if issubclass(container_type, registered_type):
            return registration
    return None


def namedtuple_extra_instance_state(value: Any) -> bool:
    """Return whether a namedtuple INSTANCE carries non-field ``__dict__`` state.

    r37 secB_1: a namedtuple subclass without ``__slots__ = ()`` can stash
    tensor-derived attributes on the instance (``self.total = a + b`` in a custom
    ``__new__``); the inert ``tuple.__new__`` rebuild drops them. This is the
    NAMEDTUPLE-SPECIFIC capture-side signal -- never route namedtuples through the
    dataclass helper (:func:`reconstruction_is_lossy`), whose no-``__dict__``
    interpretation is inverted for tuple storage (it would mark every plain
    namedtuple lossy). ``None``-valued extras carry no droppable derived state.
    """

    instance_dict = getattr(value, "__dict__", None)
    if not isinstance(instance_dict, dict):
        return False
    return any(item is not None for item in instance_dict.values())


def namedtuple_type_can_carry_instance_state(container_type: type[Any]) -> bool:
    """Return whether a RESOLVED namedtuple type can hold per-instance state (r37).

    The load-time forged-flag defense: a persisted ``lossy_reconstruction=False``
    on a namedtuple spec is only honored when the resolved type structurally
    CANNOT carry dropped state (plain ``collections.namedtuple`` /
    ``typing.NamedTuple`` / a ``__slots__ = ()`` subclass -> no instance
    ``__dict__``). A resolved type WITH an instance ``__dict__`` is treated as
    lossy even when the persisted flag says otherwise.
    """

    return _type_has_instance_dict(container_type)


def mapping_extra_instance_state(value: Any) -> bool:
    """Return whether a trusted-base mapping INSTANCE carries extra ``__dict__`` state.

    Exact ``dict`` has no instance ``__dict__`` (structurally stateless);
    ``OrderedDict``/``defaultdict`` instances can carry arbitrary attributes the
    key/value rebuild drops (r37 3-ADJ-2). Capture-side signal, mirrored by the
    save-side losslessness refusal.
    """

    instance_dict = getattr(value, "__dict__", None)
    if not isinstance(instance_dict, dict):
        return False
    return any(item is not None for item in instance_dict.values())


CONTAINER_KIND_CAPABILITIES: dict[str, dict[str, str | bool]] = {
    # r37 R11 -- THE per-kind reconstruction capability table. One truth consumed
    # by spec construction (torchlens/backends/torch/ops.py::_build_container_spec),
    # the producer losslessness proof (_prove_runnable_output_lossless), and the
    # runtime independent recompute (_runnable_execution::_spec_node_reconstruction_lossy).
    # ``instance_state_rule`` answers "can this kind's resolved type carry
    # per-instance state beyond its recorded fields/keys/items, and how is that
    # policed": builtin_stateless (structurally impossible), instance_refused
    # (capture-side instance check + save refusal; persisted flag supplementary),
    # type_recompute (load-time type-level recompute overrides a forged flag),
    # declaration_required (explicit trusted state_complete registration), and
    # refused (the kind itself is a typed save refusal).
    "tuple": {
        "children": "positional",
        "literal_support": True,
        "exact_type": True,
        "instance_state_rule": "builtin_stateless",
        "addressable": True,
    },
    "list": {
        "children": "positional",
        "literal_support": True,
        "exact_type": True,
        "instance_state_rule": "builtin_stateless",
        "addressable": True,
    },
    "dict": {
        "children": "keys",
        "literal_support": True,
        "exact_type": True,
        "instance_state_rule": "instance_refused",
        "addressable": True,
    },
    "namedtuple": {
        "children": "fields",
        "literal_support": True,
        "exact_type": True,
        "instance_state_rule": "type_recompute",
        "addressable": True,
    },
    "dataclass": {
        "children": "fields",
        "literal_support": True,
        "exact_type": True,
        "instance_state_rule": "type_recompute",
        "addressable": True,
    },
    "hf_model_output": {
        "children": "keys",
        "literal_support": True,
        "exact_type": True,
        "instance_state_rule": "type_recompute",
        "addressable": True,
    },
    "literal": {
        "children": "none",
        "literal_support": True,
        "exact_type": True,
        "instance_state_rule": "builtin_stateless",
        "addressable": False,
    },
    "opaque": {
        "children": "none",
        "literal_support": False,
        "exact_type": False,
        "instance_state_rule": "refused",
        "addressable": False,
    },
    "registered": {
        "children": "registered",
        "literal_support": True,
        "exact_type": True,
        "instance_state_rule": "declaration_required",
        "addressable": True,
    },
}
"""Frozen capability rows for every ``ContainerSpec.kind`` (r37 R11)."""


__all__ = [
    "CONTAINER_KIND_CAPABILITIES",
    "ContainerReconstructionError",
    "ContainerSpec",
    "DataclassField",
    "DictKey",
    "HFKey",
    "NamedField",
    "OutputPathComponent",
    "RegisteredContainer",
    "TupleIndex",
    "get_registered_container",
    "mapping_extra_instance_state",
    "namedtuple_extra_instance_state",
    "namedtuple_type_can_carry_instance_state",
    "reconstruction_is_lossy",
    "reconstruction_is_lossy_by_type",
    "register_container",
    "rebuild_container_from_spec",
    "resolve_container_type",
]


# Public alias so other bundle-reachable sinks (e.g. ``Op.multi_output_type`` in
# ``torchlens.data_classes.op``) route container-type resolution through THIS single
# default-deny resolver rather than re-deriving a second, drift-prone one. There must be
# exactly one resolver that decides a container type by name from ``sys.modules`` and NEVER
# imports an attacker-controlled bundle string.
resolve_container_type = _resolve_container_type
