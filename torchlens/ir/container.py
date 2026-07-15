"""Leaf dataclasses for TorchLens output container structure."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass
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
        if container_type is not None:
            if _is_trusted_structseq_type(container_type):
                # Genuine torch structseq (``torch.return_types.*``): reconstruct through its
                # own INERT builtin ``__new__`` (a single-iterable C constructor that runs no
                # arbitrary Python). Admissibility already pinned the RESOLVED ``__module__`` to
                # ``torch.return_types`` (not the attacker's spec string), so this is a genuine
                # torch C type, never an attacker-named look-alike.
                return container_type(values)
            if _is_generated_namedtuple_type(container_type):
                # Real compiler-generated namedtuple: allocate the tuple INERTLY via the builtin
                # ``tuple.__new__`` and NEVER invoke ``container_type(*values)`` -- this is the
                # RCE fix. The old ``container_type(*values)`` ran the resolved type's
                # ``__new__`` / ``__init__``, so a spec naming a ``tuple``-subclass "namedtuple-
                # like" type whose ``__new__`` executes code was a load/run construction gadget.
                # ``tuple.__new__`` bypasses that ``__new__`` entirely, mirroring the inert
                # non-invoking reconstruction used for the dataclass / HF branches.
                return tuple.__new__(container_type, values)
        return tuple(values)
    if spec.kind == "dataclass":
        field_values = {
            field_name: _rebuild_child_or_leaf(child_by_key, DataclassField(field_name), leaf_iter)
            for field_name in spec.fields
        }
        container_type = _resolve_container_type(spec)
        if (
            container_type is not None
            and _type_has_safe_new(container_type)
            and _fields_are_inert_settable(container_type, spec.fields)
        ):
            return _construct_dataclass_without_init(container_type, field_values)
        return field_values
    if spec.kind == "hf_model_output":
        key_values = {
            key: _rebuild_child_or_leaf(child_by_key, HFKey(key), leaf_iter) for key in spec.keys
        }
        container_type = _resolve_container_type(spec)
        if (
            container_type is not None
            and _type_has_safe_new(container_type)
            and issubclass(container_type, dict)
            and _fields_are_inert_settable(container_type, spec.keys)
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


def _is_trusted_structseq_type(container_type: type[Any]) -> bool:
    """Return whether ``container_type`` is a genuine torch structseq (``torch.return_types.*``).

    Structseq classes are ``tuple`` subclasses whose ``__new__`` is an inert builtin (C)
    single-iterable constructor. We key on the RESOLVED ``__module__`` -- not the attacker's
    ``spec.type_module`` string -- so only genuine torch structseqs (which cannot be forged
    into ``sys.modules['torch.return_types']``) reconstruct through their own ``__new__``.

    Parameters
    ----------
    container_type:
        Already-resolved, admissible container class.

    Returns
    -------
    bool
        True when the class is a genuine ``torch.return_types`` structseq.
    """

    return getattr(container_type, "__module__", None) == "torch.return_types" and hasattr(
        container_type, "n_fields"
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


def reconstruction_is_lossy_by_type(
    container_type: type[Any], captured_names: tuple[Any, ...], kind: str
) -> bool:
    """Recompute reconstruction lossiness from the RESOLVED type at LOAD time.

    The persisted :attr:`ContainerSpec.lossy_reconstruction` flag is computed at CAPTURE and
    is attacker-controlled in an untrusted bundle: a forged ``False`` would force a false
    ``VERIFIED`` on a genuinely lossy reconstruction. This function re-derives lossiness
    INDEPENDENTLY from the resolved type (the one trustworthy input at load), mirroring the
    r25 :func:`reconstruction_is_lossy` criteria that are TYPE-observable:

    * a ``__slots__`` layout with no instance ``__dict__`` (fields cannot be set inertly), or
    * a captured name that resolves to a data descriptor (cannot be set faithfully), or
    * (dataclass kind only) a user ``__post_init__`` that may compute dropped non-field state.

    The ``__post_init__`` signal is applied ONLY to the plain-``dataclass`` kind: HuggingFace
    ``ModelOutput`` dataclasses use ``__post_init__`` solely to populate their mapping from
    fields (attrs == keys), so applying it to the ``hf_model_output`` kind would over-trigger
    the standard/real ModelOutput case that must stay VERIFIED. The purely instance-level r25
    "computed non-None extra attr on a custom (non-dataclass) ModelOutput ``__init__``" case is
    not type-observable without invoking that ``__init__``; the caller keeps the persisted flag
    as a supplementary (never sole) signal so genuine such captures still report lossy.

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
    if kind == "dataclass" and _dataclass_defines_post_init(container_type):
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
        return getattr(container_type, "__module__", None) == "torch.return_types" and hasattr(
            container_type, "n_fields"
        )
    if kind == "dataclass":
        return dataclasses.is_dataclass(container_type)
    if kind == "hf_model_output":
        return _is_hf_model_output_type(container_type)
    if kind == "dict":
        return (spec.type_module, spec.type_qualname) in _ALLOWED_MAPPING_TYPE_REFS
    if kind == "registered":
        return get_registered_container(container_type) is not None
    return False


def _resolve_container_type(spec: ContainerSpec) -> type[Any] | None:
    """Resolve the concrete container type named by a spec under a default-deny gate.

    Security contract (this is the fix for the output-container reconstruction RCE):

    * NEVER imports an attacker-named module -- importing runs top-level module code.
      The type is resolved ONLY from a module already present in ``sys.modules``; every
      legit container type was produced by the traced model, so its module is loaded.
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
    resolved: Any = module
    try:
        for attribute_name in qualname.split("."):
            resolved = getattr(resolved, attribute_name)
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
    ) -> None:
        """Create a registered container hook pair.

        Parameters
        ----------
        flatten:
            Callable returning ``(children, aux_data)``.
        unflatten:
            Callable accepting ``(aux_data, children)``.
        """

        self.flatten = flatten
        self.unflatten = unflatten


_CONTAINER_REGISTRY: dict[type[Any], RegisteredContainer] = {}


def register_container(
    container_type: type[Any],
    flatten: Callable[[Any], tuple[list[Any] | tuple[Any, ...], Any]],
    unflatten: Callable[[Any, list[Any]], Any],
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
    """

    _CONTAINER_REGISTRY[container_type] = RegisteredContainer(flatten, unflatten)


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


__all__ = [
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
