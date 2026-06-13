"""Equinox module helpers for the JAX backend."""

from __future__ import annotations

import base64
import inspect
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from ...data_classes.param import Param
from ...ir.refs import DeviceRef, DtypeRef

MODULE_SCOPE_PREFIX = "tlm_"


@dataclass(frozen=True)
class EquinoxModuleTree:
    """Static metadata discovered from an Equinox module pytree.

    Parameters
    ----------
    root
        Root Equinox module instance.
    metadata
        TorchLens module metadata keyed by module address.
    address_by_id
        Primary module address keyed by object identity.
    modules_by_class
        Module addresses keyed by module class and object identity.
    param_owner_by_address
        Owning module address keyed by pytree parameter address.
    param_address_by_value_id
        Parameter address keyed by array object identity.
    """

    root: Any
    metadata: dict[str, dict[str, Any]]
    address_by_id: dict[int, str]
    modules_by_class: dict[type[Any], dict[int, str]]
    param_owner_by_address: dict[str, str]
    param_address_by_value_id: dict[int, str]


def is_equinox_module(value: object) -> bool:
    """Return whether ``value`` is an Equinox module instance.

    Parameters
    ----------
    value
        Candidate root callable.

    Returns
    -------
    bool
        True when Equinox is importable and ``value`` is an ``eqx.Module``.
    """

    try:
        import equinox as eqx
    except ImportError:
        return False
    return isinstance(value, eqx.Module)


def discover_equinox_module_tree(model: Any) -> EquinoxModuleTree | None:
    """Discover module hierarchy and parameter ownership for an Equinox model.

    Parameters
    ----------
    model
        Candidate JAX callable.

    Returns
    -------
    EquinoxModuleTree | None
        Module tree for Equinox roots, otherwise ``None``.
    """

    if not is_equinox_module(model):
        return None
    metadata: dict[str, dict[str, Any]] = {}
    address_by_id: dict[int, str] = {}
    modules_by_class: defaultdict[type[Any], dict[int, str]] = defaultdict(dict)
    _walk_equinox_modules(
        module=model,
        address="self",
        metadata=metadata,
        address_by_id=address_by_id,
        modules_by_class=modules_by_class,
    )
    param_owner_by_address, param_address_by_value_id = _equinox_param_maps(
        model,
        set(metadata),
    )
    return EquinoxModuleTree(
        root=model,
        metadata=metadata,
        address_by_id=address_by_id,
        modules_by_class=dict(modules_by_class),
        param_owner_by_address=param_owner_by_address,
        param_address_by_value_id=param_address_by_value_id,
    )


@contextmanager
def scoped_equinox_module_calls(tree: EquinoxModuleTree) -> Iterator[None]:
    """Temporarily wrap Equinox module ``__call__`` methods with named scopes.

    Parameters
    ----------
    tree
        Discovered Equinox module tree.

    Yields
    ------
    None
        Control while class-level wrappers are installed.
    """

    import jax

    originals: dict[type[Any], Any] = {}
    for module_class, address_by_instance_id in tree.modules_by_class.items():
        original_call = getattr(module_class, "__call__")
        originals[module_class] = original_call

        def wrapper(
            self: Any,
            *args: Any,
            __address_by_id: dict[int, str] = address_by_instance_id,
            __original: Any = original_call,
            **kwargs: Any,
        ) -> Any:
            """Call the original module under a TorchLens named scope when known."""

            address = __address_by_id.get(id(self))
            if address is None:
                return __original(self, *args, **kwargs)
            with jax.named_scope(encode_module_scope(address)):
                return __original(self, *args, **kwargs)

        setattr(module_class, "__call__", wrapper)
    try:
        yield
    finally:
        for module_class, original_call in originals.items():
            setattr(module_class, "__call__", original_call)


def encode_module_scope(address: str) -> str:
    """Return a reversible ``jax.named_scope`` marker for a module address.

    Parameters
    ----------
    address
        TorchLens module address.

    Returns
    -------
    str
        Named-scope-safe marker string.
    """

    encoded = base64.urlsafe_b64encode(address.encode("utf-8")).decode("ascii")
    return MODULE_SCOPE_PREFIX + encoded.rstrip("=")


def decode_module_scope(scope_name: str) -> str | None:
    """Decode a TorchLens module named-scope marker.

    Parameters
    ----------
    scope_name
        Name-stack component.

    Returns
    -------
    str | None
        Decoded module address, or ``None`` for non-TorchLens scopes.
    """

    if not scope_name.startswith(MODULE_SCOPE_PREFIX):
        return None
    payload = scope_name.removeprefix(MODULE_SCOPE_PREFIX)
    padded = payload + "=" * (-len(payload) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def equinox_param_logs(tree: EquinoxModuleTree, trace: Any) -> dict[str, Param]:
    """Build TorchLens parameter logs from Equinox pytree array leaves.

    Parameters
    ----------
    tree
        Discovered Equinox module tree.
    trace
        Trace receiving the parameter logs.

    Returns
    -------
    dict[str, Param]
        Parameter logs keyed by pytree address.
    """

    import equinox as eqx
    import jax

    param_logs: dict[str, Param] = {}
    leaves_with_paths, _treedef = jax.tree_util.tree_flatten_with_path(tree.root)
    for path, value in leaves_with_paths:
        if not eqx.is_array(value):
            continue
        address = _path_to_string(path)
        module_address = tree.param_owner_by_address.get(address, "self")
        shape = tuple(getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", ""))
        param = Param(
            module_address=module_address,
            name=address.rsplit(".", 1)[-1],
            shape=shape,
            dtype=dtype,  # type: ignore[arg-type]
            num_params=_numel(shape),
            param_memory=_nbytes(value) or 0,
            trainable=bool(eqx.is_inexact_array(value)),
            address=address,
            barcode=f"jax:{address}",
            has_optimizer=None,
        )
        param.dtype_ref = DtypeRef(backend="jax", name=dtype)
        param.device_ref = DeviceRef.from_value(getattr(value, "device", None))
        param.backend_address = f"pytree:{address}"
        param.resolver_status = "metadata_only"
        param._param_ref = None
        param.source_trace = trace
        setattr(param, "address_kind", "pytree_path")
        setattr(param, "param_source", "pytree-derived")
        param_logs[address] = param
    return param_logs


def _walk_equinox_modules(
    *,
    module: Any,
    address: str,
    metadata: dict[str, dict[str, Any]],
    address_by_id: dict[int, str],
    modules_by_class: defaultdict[type[Any], dict[int, str]],
) -> None:
    """Walk Equinox dataclass fields and populate module metadata.

    Parameters
    ----------
    module
        Equinox module instance.
    address
        TorchLens address for ``module``.
    metadata
        Metadata mapping being populated.
    address_by_id
        Primary address mapping being populated.
    modules_by_class
        Class-level wrapper mapping being populated.

    Returns
    -------
    None
        Mappings are updated in place.
    """

    module_id = id(module)
    primary = address_by_id.get(module_id)
    if primary is not None:
        metadata[primary].setdefault("all_addresses", [primary]).append(address)
        return

    address_by_id[module_id] = address
    modules_by_class[type(module)][module_id] = address
    child_addresses = [
        child_address
        for child_address, child_module in _iter_equinox_module_children(module, address)
        if is_equinox_module(child_module)
    ]
    metadata[address] = {
        **_module_source_metadata(module),
        "cls": type(module),
        "class_name": type(module).__name__,
        "class_qualname": f"{type(module).__module__}.{type(module).__qualname__}",
        "address_children": child_addresses,
        "all_addresses": [address],
        "training": False,
        "forward_pre_hooks": [],
        "forward_hooks": [],
        "backward_pre_hooks": [],
        "backward_hooks": [],
        "full_backward_pre_hooks": [],
        "full_backward_hooks": [],
        "custom_attributes": {},
        "custom_methods": [],
    }
    for child_address, child_module in _iter_equinox_module_children(module, address):
        if is_equinox_module(child_module):
            _walk_equinox_modules(
                module=child_module,
                address=child_address,
                metadata=metadata,
                address_by_id=address_by_id,
                modules_by_class=modules_by_class,
            )


def _iter_equinox_module_children(module: Any, address: str) -> Iterator[tuple[str, Any]]:
    """Yield direct Equinox module children from dataclass fields.

    Parameters
    ----------
    module
        Equinox module instance.
    address
        Parent TorchLens address.

    Yields
    ------
    tuple[str, Any]
        Child address and child module.
    """

    if not is_dataclass(module):
        return
    for field in fields(module):
        if not hasattr(module, field.name):
            continue
        child = getattr(module, field.name)
        if is_equinox_module(child):
            yield _join_module_address(address, field.name), child


def _equinox_param_maps(
    model: Any,
    module_addresses: set[str],
) -> tuple[dict[str, str], dict[int, str]]:
    """Return owner and value-id maps for Equinox pytree array leaves.

    Parameters
    ----------
    model
        Equinox module root.
    module_addresses
        Known TorchLens module addresses.

    Returns
    -------
    tuple[dict[str, str], dict[int, str]]
        Owner mapping keyed by pytree leaf address and address mapping keyed by
        array object identity.
    """

    import equinox as eqx
    import jax

    owners: dict[str, str] = {}
    value_ids: dict[int, str] = {}
    leaves_with_paths, _treedef = jax.tree_util.tree_flatten_with_path(model)
    for path, value in leaves_with_paths:
        if not eqx.is_array(value):
            continue
        address = _path_to_string(path)
        owner = _deepest_module_prefix(address, module_addresses)
        owners[address] = owner or "self"
        value_ids[id(value)] = address
    return owners, value_ids


def _deepest_module_prefix(address: str, module_addresses: set[str]) -> str | None:
    """Return the deepest module address that owns a parameter path.

    Parameters
    ----------
    address
        Pytree leaf address.
    module_addresses
        Known module addresses.

    Returns
    -------
    str | None
        Deepest owning module address, if any.
    """

    parts = address.split(".")
    candidates = [".".join(parts[:index]) for index in range(len(parts), 0, -1)]
    for candidate in candidates:
        if candidate in module_addresses:
            return candidate
    return "self" if "self" in module_addresses else None


def _module_source_metadata(module: Any) -> dict[str, Any]:
    """Return best-effort source metadata for an Equinox module.

    Parameters
    ----------
    module
        Equinox module instance.

    Returns
    -------
    dict[str, Any]
        Source metadata compatible with TorchLens module logs.
    """

    cls = type(module)
    init = getattr(cls, "__init__", None)
    call = getattr(cls, "__call__", None)
    return {
        "class_source_file": inspect.getsourcefile(cls),
        "class_source_line": _source_line(cls),
        "init_source_file": inspect.getsourcefile(init) if init is not None else None,
        "init_source_line": _source_line(init),
        "forward_source_file": inspect.getsourcefile(call) if call is not None else None,
        "forward_source_line": _source_line(call),
        "class_docstring": inspect.getdoc(cls),
        "init_signature": _signature_string(init),
        "init_docstring": inspect.getdoc(init) if init is not None else None,
        "forward_signature": _signature_string(call),
        "forward_docstring": inspect.getdoc(call) if call is not None else None,
    }


def _source_line(obj: Any) -> int | None:
    """Return the first source line for ``obj`` when available.

    Parameters
    ----------
    obj
        Object to inspect.

    Returns
    -------
    int | None
        First source line, or ``None``.
    """

    if obj is None:
        return None
    try:
        return inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        return None


def _signature_string(obj: Any) -> str | None:
    """Return ``obj``'s signature string when inspectable.

    Parameters
    ----------
    obj
        Callable object.

    Returns
    -------
    str | None
        Signature string, or ``None``.
    """

    if obj is None:
        return None
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _join_module_address(parent: str, child_name: str) -> str:
    """Return a TorchLens child module address.

    Parameters
    ----------
    parent
        Parent module address.
    child_name
        Child field name.

    Returns
    -------
    str
        Joined child address.
    """

    return child_name if parent == "self" else f"{parent}.{child_name}"


def _path_to_string(path: Any) -> str:
    """Convert a JAX pytree path to a dotted string.

    Parameters
    ----------
    path
        JAX pytree path entries.

    Returns
    -------
    str
        Dotted path string.
    """

    if not path:
        return "root"
    parts: list[str] = []
    for entry in path:
        name = getattr(entry, "name", None)
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        if name is not None:
            parts.append(str(name))
        elif key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        else:
            parts.append(str(entry).strip("[]'"))
    return ".".join(parts)


def _numel(shape: tuple[int, ...]) -> int:
    """Return number of elements for ``shape``.

    Parameters
    ----------
    shape
        Tensor shape.

    Returns
    -------
    int
        Product of dimensions.
    """

    result = 1
    for dim in shape:
        result *= int(dim)
    return result


def _nbytes(value: object) -> int | None:
    """Return byte size for an array-like value.

    Parameters
    ----------
    value
        Array-like value.

    Returns
    -------
    int | None
        Byte size when available.
    """

    nbytes = getattr(value, "nbytes", None)
    return None if nbytes is None else int(nbytes)
