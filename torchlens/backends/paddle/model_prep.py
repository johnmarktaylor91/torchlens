"""Paddle model preparation helpers for technical-preview capture."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from ...ir.events import ModuleFrame


@dataclass
class PaddleModuleTree:
    """Discovered Paddle object-module tree.

    Parameters
    ----------
    root
        Root Paddle layer.
    metadata
        TorchLens module metadata keyed by primary address.
    address_by_id
        Primary module address keyed by object identity.
    param_owner_by_address
        Owning module address keyed by parameter address.
    param_address_by_id
        Primary parameter address keyed by exact Paddle tensor identity.
    call_counts
        Per-primary-address call counts recorded during capture.
    forward_args_by_call
        Forward args keyed by ``(primary_address, call_index)``.
    hook_handles
        Paddle hook handles installed for this capture session.
    """

    root: object
    metadata: dict[str, dict[str, Any]]
    address_by_id: dict[int, str]
    param_owner_by_address: dict[str, str]
    param_address_by_id: dict[int, str]
    call_counts: dict[str, int] = field(default_factory=dict)
    forward_args_by_call: dict[tuple[str, int], tuple[tuple[Any, ...], dict[str, Any]]] = field(
        default_factory=dict
    )
    hook_handles: list[Any] = field(default_factory=list)


def discover_paddle_module_tree(model: object) -> PaddleModuleTree | None:
    """Discover a Paddle layer hierarchy with deterministic aliases.

    Parameters
    ----------
    model
        Candidate Paddle model.

    Returns
    -------
    PaddleModuleTree | None
        Discovered module tree, or ``None`` for raw functions/plain callables.
    """

    if inspect.isfunction(model) or inspect.ismethod(model) or not callable(model):
        return None
    if not _is_paddle_layer(model):
        return None

    modules_by_id: dict[int, object] = {}
    addresses_by_id: dict[int, list[str]] = {}
    named_sublayers = getattr(model, "named_sublayers")
    for raw_address, module in named_sublayers(include_self=True):
        if not _is_paddle_layer(module):
            continue
        address = "self" if raw_address == "" else str(raw_address)
        modules_by_id[id(module)] = module
        addresses_by_id.setdefault(id(module), []).append(address)
    if id(model) not in addresses_by_id:
        modules_by_id[id(model)] = model
        addresses_by_id[id(model)] = ["self"]

    address_by_id: dict[int, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    alias_to_primary: dict[str, str] = {}
    for module_id, addresses in addresses_by_id.items():
        primary = _primary_address(addresses)
        module = modules_by_id[module_id]
        address_by_id[module_id] = primary
        all_addresses = sorted(set(addresses), key=_address_sort_key)
        for alias in all_addresses:
            alias_to_primary[alias] = primary
        metadata[primary] = {
            "cls": type(module),
            "class_name": type(module).__name__,
            "class_qualname": f"{type(module).__module__}.{type(module).__qualname__}",
            "address_children": [],
            "all_addresses": all_addresses,
            "training": bool(getattr(module, "training", False)),
            "forward_pre_hooks": [],
            "forward_hooks": [],
            "backward_pre_hooks": [],
            "backward_hooks": [],
            "full_backward_pre_hooks": [],
            "full_backward_hooks": [],
            "custom_attributes": {},
            "custom_methods": [],
            "_module_object": module,
        }

    for primary, item in metadata.items():
        item["address_children"] = _direct_children(primary, metadata)

    param_owner_by_address: dict[str, str] = {}
    param_address_by_id: dict[int, str] = {}
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        for raw_param_address, value in named_parameters():
            param_address = str(raw_param_address)
            owner_alias = param_address.rsplit(".", 1)[0] if "." in param_address else "self"
            owner = alias_to_primary.get(owner_alias, owner_alias)
            primary_param_address = _join_module_address(owner, param_address.rsplit(".", 1)[-1])
            param_owner_by_address[primary_param_address] = owner
            param_address_by_id.setdefault(id(value), primary_param_address)

    return PaddleModuleTree(
        root=model,
        metadata=metadata,
        address_by_id=address_by_id,
        param_owner_by_address=param_owner_by_address,
        param_address_by_id=param_address_by_id,
    )


def prepare_model_once(model: object) -> object:
    """Apply one-time Paddle preparation.

    Parameters
    ----------
    model
        Paddle model.

    Returns
    -------
    object
        The unchanged model.
    """

    return model


def prepare_model_session(session: object, model: object, tree: PaddleModuleTree | None) -> object:
    """Install per-session Paddle forward hooks.

    Parameters
    ----------
    session
        Active trace object.
    model
        Paddle model.
    tree
        Discovered Paddle module tree, if object-module mode is active.

    Returns
    -------
    object
        The unchanged model.
    """

    del session
    if tree is None:
        return model
    for module_id, address in tree.address_by_id.items():
        module = tree.metadata[address]["_module_object"]
        if id(module) != module_id:
            continue
        pre_hook = _make_pre_hook(tree, address)
        post_hook = _make_post_hook()
        tree.hook_handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        tree.hook_handles.append(module.register_forward_post_hook(post_hook, with_kwargs=True))
    return model


def cleanup_model_session(
    session: object, prepared_model: object, tree: PaddleModuleTree | None
) -> None:
    """Clean up per-session Paddle preparation.

    Parameters
    ----------
    session
        Active trace object.
    prepared_model
        Prepared model object.
    tree
        Discovered Paddle module tree, if any.
    """

    del session, prepared_model
    if tree is None:
        return
    for handle in tree.hook_handles:
        remove = getattr(handle, "remove", None)
        if callable(remove):
            remove()
    tree.hook_handles.clear()


def _make_pre_hook(tree: PaddleModuleTree, address: str) -> Any:
    """Build a Paddle forward pre-hook for ``address``.

    Parameters
    ----------
    tree
        Discovered module tree.
    address
        Primary module address.

    Returns
    -------
    Any
        Hook callable accepted by Paddle.
    """

    def hook(layer: object, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Push module frame metadata for one Paddle forward call."""

        del layer
        call_index = tree.call_counts.get(address, 0) + 1
        tree.call_counts[address] = call_index
        tree.forward_args_by_call[(address, call_index)] = (tuple(args), dict(kwargs))
        from ... import _state

        trace = _state._active_trace
        if trace is not None and hasattr(trace, "_paddle_module_stack"):
            metadata = tree.metadata[address]
            trace._paddle_module_stack.append(
                ModuleFrame(
                    address=address,
                    address_normalized=address,
                    module_type=str(metadata.get("class_name", "")),
                    call_index=call_index,
                    fx_qualpath=None,
                    entry_argnames=tuple(str(index) for index in range(len(args)))
                    + tuple(str(key) for key in kwargs),
                )
            )

    return hook


def _make_post_hook() -> Any:
    """Build a Paddle forward post-hook that pops the active module stack.

    Returns
    -------
    Any
        Hook callable accepted by Paddle.
    """

    def hook(
        layer: object,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: object,
    ) -> None:
        """Pop module frame metadata after one Paddle forward call."""

        del layer, args, kwargs, output
        from ... import _state

        trace = _state._active_trace
        if trace is not None and getattr(trace, "_paddle_module_stack", None):
            trace._paddle_module_stack.pop()

    return hook


def _is_paddle_layer(value: object) -> bool:
    """Return whether ``value`` is a ``paddle.nn.Layer``.

    Parameters
    ----------
    value
        Candidate object.

    Returns
    -------
    bool
        True when Paddle is installed and ``value`` is a Paddle layer.
    """

    try:
        import paddle
    except ImportError:
        return False
    return isinstance(value, paddle.nn.Layer)


def _primary_address(addresses: list[str]) -> str:
    """Return the deterministic primary address for module aliases.

    Parameters
    ----------
    addresses
        All observed addresses for one module object.

    Returns
    -------
    str
        Primary TorchLens address.
    """

    if "self" in addresses:
        return "self"
    return sorted(set(addresses), key=_address_sort_key)[0]


def _address_sort_key(address: str) -> tuple[int, str]:
    """Sort root-like addresses before nested addresses.

    Parameters
    ----------
    address
        Module address.

    Returns
    -------
    tuple[int, str]
        Stable sort key.
    """

    return (address.count("."), address)


def _direct_children(address: str, metadata: dict[str, dict[str, Any]]) -> list[str]:
    """Return direct metadata children for ``address``.

    Parameters
    ----------
    address
        Parent address.
    metadata
        Module metadata by address.

    Returns
    -------
    list[str]
        Sorted child addresses.
    """

    children = [
        candidate
        for candidate in metadata
        if candidate != address and _nearest_metadata_parent(candidate, metadata) == address
    ]
    return sorted(children, key=_address_sort_key)


def _nearest_metadata_parent(address: str, metadata: dict[str, dict[str, Any]]) -> str | None:
    """Return the closest existing parent address for ``address``.

    Parameters
    ----------
    address
        Child address.
    metadata
        Module metadata keyed by address.

    Returns
    -------
    str | None
        Parent address, or ``None`` for root.
    """

    if address == "self":
        return None
    parts = address.split(".")
    while len(parts) > 1:
        parts.pop()
        candidate = ".".join(parts)
        if candidate in metadata:
            return candidate
    return "self" if "self" in metadata else None


def _join_module_address(parent: str, child_name: str) -> str:
    """Return a TorchLens child module address.

    Parameters
    ----------
    parent
        Parent module address.
    child_name
        Child name.

    Returns
    -------
    str
        Joined module address.
    """

    return child_name if parent in {"", "self"} else f"{parent}.{child_name}"


__all__ = [
    "PaddleModuleTree",
    "cleanup_model_session",
    "discover_paddle_module_tree",
    "prepare_model_once",
    "prepare_model_session",
]
