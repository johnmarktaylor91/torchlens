"""Leaf dataclasses for TorchLens output container structure."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass
import sys
from typing import Any, ClassVar, Literal, TypeAlias

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
            if spec.type_module == "torch.return_types":
                # torch structseq classes (``torch.return_types.*``) reject
                # positional ``*args``; they take a single iterable. Reconstruct
                # nested structseq the same way at any container depth.
                return container_type(values)
            return container_type(*values)
        return tuple(values)
    if spec.kind == "dataclass":
        field_values = {
            field_name: _rebuild_child_or_leaf(child_by_key, DataclassField(field_name), leaf_iter)
            for field_name in spec.fields
        }
        container_type = _resolve_container_type(spec)
        if container_type is not None:
            return container_type(**field_values)
        return field_values
    if spec.kind == "hf_model_output":
        key_values = {
            key: _rebuild_child_or_leaf(child_by_key, HFKey(key), leaf_iter) for key in spec.keys
        }
        container_type = _resolve_container_type(spec)
        if container_type is not None:
            return container_type(**key_values)
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
        # torch structseq classes (``torch.return_types.*``) that torch itself emits,
        # or any genuine namedtuple class (a ``tuple`` subclass carrying ``_fields``).
        if spec.type_module == "torch.return_types":
            return True
        return issubclass(container_type, tuple) and hasattr(container_type, "_fields")
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
    "ContainerSpec",
    "DataclassField",
    "DictKey",
    "HFKey",
    "NamedField",
    "OutputPathComponent",
    "RegisteredContainer",
    "TupleIndex",
    "get_registered_container",
    "register_container",
    "rebuild_container_from_spec",
]
