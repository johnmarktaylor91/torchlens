"""TensorFlow module discovery, call-stack patching, and parameter logs."""

from __future__ import annotations

import inspect
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from ...data_classes.param import Param
from ...ir.events import ModuleFrame
from ...ir.refs import DeviceRef, DtypeRef


@dataclass(frozen=True)
class TFModuleTree:
    """Discovered TensorFlow module hierarchy.

    Parameters
    ----------
    root
        Root callable object.
    metadata
        TorchLens module metadata keyed by primary module address.
    address_by_id
        Primary module address keyed by object identity.
    modules_by_class
        Module object ids grouped by concrete class for raw ``tf.Module`` patching.
    param_owner_by_address
        Owning module address keyed by variable address.
    param_address_by_id
        Variable address keyed by object identity.
    variables_by_id
        Warmed model variables keyed by object identity.
    call_counts
        Per-module call counts observed during capture.
    forward_args_by_call
        Forward call args keyed by ``(address, call_index)``.
    """

    root: Any
    metadata: dict[str, dict[str, Any]]
    address_by_id: dict[int, str]
    modules_by_class: dict[type[Any], dict[int, str]]
    param_owner_by_address: dict[str, str]
    param_address_by_id: dict[int, str]
    variables_by_id: dict[int, Any]
    call_counts: dict[str, int]
    forward_args_by_call: dict[tuple[str, int], tuple[tuple[Any, ...], dict[str, Any]]]


def discover_tf_module_tree(model: Any, tf: Any) -> TFModuleTree | None:
    """Discover TensorFlow/Keras module metadata for object-module attribution.

    Parameters
    ----------
    model
        Candidate TensorFlow callable.
    tf
        Imported TensorFlow module.

    Returns
    -------
    TFModuleTree | None
        Discovered tree for Keras layers or raw ``tf.Module`` roots, otherwise ``None``.
    """

    if not _is_tf_module_like(model, tf):
        return None
    metadata: dict[str, dict[str, Any]] = {}
    address_by_id: dict[int, str] = {}
    modules_by_class: defaultdict[type[Any], dict[int, str]] = defaultdict(dict)
    _walk_tf_modules(
        module=model,
        address="self",
        tf=tf,
        metadata=metadata,
        address_by_id=address_by_id,
        modules_by_class=modules_by_class,
        seen=set(),
    )
    param_owner_by_address, param_address_by_id, variables_by_id = _tf_param_maps(
        model=model,
        metadata=metadata,
        address_by_id=address_by_id,
    )
    return TFModuleTree(
        root=model,
        metadata=metadata,
        address_by_id=address_by_id,
        modules_by_class=dict(modules_by_class),
        param_owner_by_address=param_owner_by_address,
        param_address_by_id=param_address_by_id,
        variables_by_id=variables_by_id,
        call_counts={},
        forward_args_by_call={},
    )


@contextmanager
def patched_tf_module_stack(
    tree: TFModuleTree | None,
    tf: Any,
    module_stack: list[ModuleFrame],
) -> Iterator[None]:
    """Temporarily patch TensorFlow module ``__call__`` methods to maintain a stack.

    Parameters
    ----------
    tree
        Discovered module tree. ``None`` leaves the stack unmodified.
    tf
        Imported TensorFlow module.
    module_stack
        Mutable stack receiving active module frames.

    Yields
    ------
    None
        Control while wrappers are installed.
    """

    if tree is None:
        yield
        return
    originals: dict[type[Any], Any] = {}
    keras_layer_class = _keras_layer_class(tf)
    if keras_layer_class is not None:
        _patch_class_call(keras_layer_class, tree, module_stack, originals)
    for module_class in tree.modules_by_class:
        if module_class is keras_layer_class:
            continue
        if "__call__" not in vars(module_class):
            continue
        _patch_class_call(module_class, tree, module_stack, originals)
    try:
        yield
    finally:
        for module_class, original in originals.items():
            setattr(module_class, "__call__", original)


def tf_param_logs(tree: TFModuleTree, trace: Any) -> dict[str, Param]:
    """Build TorchLens parameter logs from warmed TensorFlow variables.

    Parameters
    ----------
    tree
        Discovered TensorFlow module tree.
    trace
        Trace receiving the parameter records.

    Returns
    -------
    dict[str, Param]
        Parameter records keyed by stable TensorFlow variable address.
    """

    logs: dict[str, Param] = {}
    for variable_id, variable in tree.variables_by_id.items():
        address = tree.param_address_by_id.get(variable_id)
        if address is None or address in logs:
            continue
        owner = tree.param_owner_by_address.get(address, "self")
        shape = tuple(int(dim) for dim in getattr(variable, "shape", ()))
        dtype = str(getattr(variable, "dtype", ""))
        param = Param(
            module_address=owner,
            name=address.rsplit(".", 1)[-1],
            shape=shape,
            dtype=dtype,  # type: ignore[arg-type]
            num_params=_numel(shape),
            param_memory=_nbytes(variable) or 0,
            trainable=bool(getattr(variable, "trainable", False)),
            address=address,
            barcode=f"tf:{address}",
            has_optimizer=None,
        )
        param.dtype_ref = DtypeRef(backend="tf", name=dtype)
        param.device_ref = DeviceRef(backend="tf", name=str(getattr(variable, "device", "")))
        param.backend_address = f"object:{address}"
        param.resolver_status = "resolved"
        param._param_ref = variable
        param.source_trace = trace
        param.all_module_addresses = list(
            tree.metadata.get(owner, {}).get("all_addresses", [owner])
        )
        logs[address] = param
    return logs


def _patch_class_call(
    module_class: type[Any],
    tree: TFModuleTree,
    module_stack: list[ModuleFrame],
    originals: dict[type[Any], Any],
) -> None:
    """Patch one concrete module class ``__call__`` method.

    Parameters
    ----------
    module_class
        Concrete class to patch.
    tree
        Discovered module metadata.
    module_stack
        Active capture stack.
    originals
        Original methods keyed by patched class.

    Returns
    -------
    None
        Mutates the class for the active capture session.
    """

    if module_class in originals:
        return
    original_call = getattr(module_class, "__call__")
    originals[module_class] = original_call

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        address = tree.address_by_id.get(id(self))
        if address is None:
            return original_call(self, *args, **kwargs)
        tree.call_counts[address] = tree.call_counts.get(address, 0) + 1
        call_index = tree.call_counts[address]
        tree.forward_args_by_call[(address, call_index)] = (tuple(args), dict(kwargs))
        metadata = tree.metadata.get(address, {})
        frame = ModuleFrame(
            address=address,
            address_normalized=None,
            module_type=str(metadata.get("class_name", type(self).__name__)),
            call_index=call_index,
            fx_qualpath=None,
            entry_argnames=tuple(str(key) for key in kwargs),
        )
        module_stack.append(frame)
        try:
            return original_call(self, *args, **kwargs)
        finally:
            if module_stack and module_stack[-1] == frame:
                module_stack.pop()
            elif frame in module_stack:
                module_stack.remove(frame)

    setattr(module_class, "__call__", wrapper)


def _walk_tf_modules(
    *,
    module: Any,
    address: str,
    tf: Any,
    metadata: dict[str, dict[str, Any]],
    address_by_id: dict[int, str],
    modules_by_class: defaultdict[type[Any], dict[int, str]],
    seen: set[int],
) -> None:
    """Recursively walk Keras layers and raw ``tf.Module`` children.

    Parameters
    ----------
    module
        Module object currently being visited.
    address
        Stable address for ``module``.
    tf
        Imported TensorFlow module.
    metadata
        Metadata accumulator.
    address_by_id
        Object-id to address accumulator.
    modules_by_class
        Concrete-class patch accumulator.
    seen
        Object ids already visited.

    Returns
    -------
    None
        Mutates accumulators in place.
    """

    if id(module) in seen or not _is_tf_module_like(module, tf):
        return
    seen.add(id(module))
    address_by_id[id(module)] = address
    modules_by_class[type(module)][id(module)] = address
    children = _tf_module_children(module, tf)
    metadata[address] = _module_metadata(module, address, children)
    for child_name, child in children:
        child_address = f"{address}.{child_name}" if address != "self" else child_name
        _walk_tf_modules(
            module=child,
            address=child_address,
            tf=tf,
            metadata=metadata,
            address_by_id=address_by_id,
            modules_by_class=modules_by_class,
            seen=seen,
        )


def _module_metadata(module: Any, address: str, children: list[tuple[str, Any]]) -> dict[str, Any]:
    """Return TorchLens module metadata for one TensorFlow module.

    Parameters
    ----------
    module
        TensorFlow module object.
    address
        Stable module address.
    children
        Direct child module pairs.

    Returns
    -------
    dict[str, Any]
        Metadata record consumed by module-log finalization.
    """

    module_class = type(module)
    try:
        source_file = inspect.getsourcefile(module_class)
        source_line = inspect.getsourcelines(module_class)[1]
    except (OSError, TypeError):
        source_file = None
        source_line = None
    return {
        "cls": module_class,
        "class_name": module_class.__name__,
        "class_qualname": f"{module_class.__module__}.{module_class.__qualname__}",
        "all_addresses": [address],
        "address_children": [
            f"{address}.{name}" if address != "self" else name for name, _child in children
        ],
        "training": bool(getattr(module, "training", False)),
        "class_source_file": source_file,
        "class_source_line": source_line,
        "forward_source_file": None,
        "forward_source_line": None,
    }


def _tf_module_children(module: Any, tf: Any) -> list[tuple[str, Any]]:
    """Return named direct child TensorFlow modules.

    Parameters
    ----------
    module
        Parent TensorFlow module.
    tf
        Imported TensorFlow module.

    Returns
    -------
    list[tuple[str, Any]]
        Named direct children.
    """

    children: list[tuple[str, Any]] = []
    layers = getattr(module, "layers", None)
    if isinstance(layers, list):
        for index, child in enumerate(layers):
            name = str(getattr(child, "name", f"layer_{index}"))
            if _is_tf_module_like(child, tf):
                children.append((_safe_address_part(name), child))
    private_layers = getattr(module, "_layers", None)
    if isinstance(private_layers, list):
        for index, child in enumerate(private_layers):
            name = str(getattr(child, "name", f"layer_{index}"))
            if _is_tf_module_like(child, tf) and all(
                id(existing) != id(child) for _child_name, existing in children
            ):
                children.append((_safe_address_part(name), child))
    for name, value in vars(module).items():
        if name.startswith("_") or not _is_tf_module_like(value, tf):
            continue
        address_part = _safe_address_part(name)
        if all(id(existing) != id(value) for _child_name, existing in children):
            children.append((address_part, value))
    return children


def _tf_param_maps(
    *,
    model: Any,
    metadata: dict[str, dict[str, Any]],
    address_by_id: dict[int, str],
) -> tuple[dict[str, str], dict[int, str], dict[int, Any]]:
    """Return variable ownership maps for a warmed TensorFlow model.

    Parameters
    ----------
    model
        Root TensorFlow module.
    metadata
        Module metadata keyed by address.
    address_by_id
        Module object id to primary address.

    Returns
    -------
    tuple[dict[str, str], dict[int, str], dict[int, Any]]
        Owner, address, and variable maps.
    """

    param_owner_by_address: dict[str, str] = {}
    param_address_by_id: dict[int, str] = {}
    variables_by_id: dict[int, Any] = {}
    module_by_address = {address: _module for _module, address in _iter_module_addresses(model)}
    if not module_by_address:
        module_by_address = {"self": model}
    seen: set[int] = set()
    for address, module in module_by_address.items():
        variables = getattr(module, "variables", ())
        for index, variable in enumerate(list(variables)):
            variable_id = id(variable)
            if variable_id in seen:
                continue
            seen.add(variable_id)
            raw_name = str(getattr(variable, "path", getattr(variable, "name", f"var_{index}")))
            name = raw_name.split(":")[0].replace("/", ".")
            if "." not in name:
                name = f"{address}.{name}" if address != "self" else name
            owner = _owner_from_variable_name(name, set(metadata), address)
            variables_by_id[variable_id] = variable
            param_address_by_id[variable_id] = name
            param_owner_by_address[name] = owner
    return param_owner_by_address, param_address_by_id, variables_by_id


def _iter_module_addresses(model: Any) -> Iterator[tuple[Any, str]]:
    """Yield module-address pairs from a Keras model when available.

    Parameters
    ----------
    model
        Root model.

    Yields
    ------
    tuple[Any, str]
        Module object and stable address.
    """

    yield model, "self"
    layers = getattr(model, "layers", None)
    if isinstance(layers, list):
        for layer in layers:
            yield layer, _safe_address_part(str(getattr(layer, "name", type(layer).__name__)))


def _owner_from_variable_name(
    variable_name: str,
    module_addresses: set[str],
    fallback: str,
) -> str:
    """Infer the owning module address for a TensorFlow variable name.

    Parameters
    ----------
    variable_name
        Stable variable path.
    module_addresses
        Known module addresses.
    fallback
        Fallback owner address.

    Returns
    -------
    str
        Module address.
    """

    owner = variable_name.rsplit(".", 1)[0] if "." in variable_name else fallback
    if owner in module_addresses:
        return owner
    parts = owner.split(".")
    for start in range(len(parts)):
        suffix = ".".join(parts[start:])
        if suffix in module_addresses:
            return suffix
    if parts and parts[-1] in module_addresses:
        return parts[-1]
    return owner or fallback


def _is_tf_module_like(value: Any, tf: Any) -> bool:
    """Return whether ``value`` is a Keras layer/model or raw ``tf.Module``.

    Parameters
    ----------
    value
        Candidate object.
    tf
        Imported TensorFlow module.

    Returns
    -------
    bool
        True for TensorFlow module-like objects.
    """

    module_type = getattr(tf, "Module", None)
    keras_layer_class = _keras_layer_class(tf)
    return bool(
        (module_type is not None and isinstance(value, module_type))
        or (keras_layer_class is not None and isinstance(value, keras_layer_class))
    )


def _keras_layer_class(tf: Any) -> type[Any] | None:
    """Return the active Keras ``Layer`` class if importable.

    Parameters
    ----------
    tf
        Imported TensorFlow module.

    Returns
    -------
    type[Any] | None
        Keras layer class or ``None``.
    """

    del tf
    try:
        import keras
    except ImportError:
        return None
    layers = getattr(keras, "layers", None)
    layer_class = getattr(layers, "Layer", None)
    return layer_class if isinstance(layer_class, type) else None


def _safe_address_part(value: str) -> str:
    """Return a dot-address-safe component.

    Parameters
    ----------
    value
        Raw component.

    Returns
    -------
    str
        Sanitized component.
    """

    cleaned = value.replace("/", ".").replace(":", "_")
    return cleaned or "module"


def _numel(shape: tuple[int, ...]) -> int:
    """Return the number of elements represented by ``shape``.

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


def _nbytes(value: Any) -> int | None:
    """Return TensorFlow tensor memory in bytes when available.

    Parameters
    ----------
    value
        TensorFlow variable or tensor.

    Returns
    -------
    int | None
        Byte size, or ``None`` when unavailable.
    """

    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    dtype = getattr(value, "dtype", None)
    size = getattr(dtype, "size", None)
    if size is None:
        return None
    return _numel(shape) * int(size)


def monotonic_time() -> float:
    """Return a wall-clock timestamp for module call timing.

    Returns
    -------
    float
        Current timestamp.
    """

    return time.time()
