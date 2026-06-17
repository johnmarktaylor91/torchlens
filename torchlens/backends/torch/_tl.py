"""TorchLens private metadata namespace helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from typing import Any, Iterable, List, Optional, cast

from torch import nn
from torch.utils.weak import WeakIdKeyDictionary

__all__ = [
    "TorchLensMeta",
    "TensorMeta",
    "ParamMeta",
    "ModuleMeta",
    "DecorationTag",
    "TorchLensTLCollisionError",
    "get",
    "is_tracked",
    "clear_meta",
    "get_tensor_meta",
    "set_tensor_label",
    "get_tensor_label",
    "clear_tensor_label",
    "promote_label_to_buffer_source_and_clear_label",
    "set_buffer_address",
    "get_buffer_address",
    "get_label_list",
    "set_param_meta",
    "get_param_meta",
    "increment_param_call_index",
    "restore_param_requires_grad",
    "set_module_meta",
    "get_module_meta",
    "mark_decorated_function",
    "is_decorated_function",
    "mark_forward_call_decorated",
    "is_forward_call_decorated",
    "mark_tensor_replacement_wrapped",
    "is_tensor_replacement_wrapped",
    "copy_replacement_meta",
]


class TorchLensMeta:
    """Branded base for TorchLens-owned ``._tl`` metadata."""


@dataclass
class TensorMeta(TorchLensMeta):
    """Metadata attached to non-Parameter tensors during a capture session."""

    label_raw: Optional[str] = None
    address: Optional[str] = None
    buffer_source: Optional[str] = None


@dataclass
class ParamMeta(TorchLensMeta):
    """Metadata attached to parameters during a capture session."""

    param_barcode: Optional[str] = None
    param_address: Optional[str] = None
    call_index: int = 0
    requires_grad_before_capture: Optional[bool] = None


@dataclass
class ModuleMeta(TorchLensMeta):
    """Permanent metadata attached to modules after model preparation."""

    address: Optional[str] = None
    module_type: Optional[str] = None


@dataclass
class DecorationTag(TorchLensMeta):
    """Sentinel metadata attached to decorated callables."""

    is_decorated_function: bool = False
    forward_call_is_decorated: bool = False
    tensor_replacement_wrapped: bool = False


class TorchLensTLCollisionError(AttributeError):
    """Raised when an existing ``._tl`` is foreign or the wrong metadata kind."""


# --------------------------------------------------------------------------- #
# Identity-keyed registries for USER-OWNED objects (modules + parameters).
#
# Modules and parameters are the user's objects; tagging them with a ``._tl``
# attribute pollutes them (state_dict / serialization / introspection leakage)
# and, for parameters, writes an attribute onto an ``nn.Parameter`` tensor. We
# instead store their TorchLens metadata in process-wide identity-keyed weak
# registries, leaving the user objects untouched. TorchLens-OWNED objects
# (plain activation tensors -> ``TensorMeta``; decorated callables ->
# ``DecorationTag``) keep using the ``._tl`` attribute -- no pollution concern,
# and that path is hot.
#
# ``WeakIdKeyDictionary`` (not stdlib ``WeakKeyDictionary``) is REQUIRED for the
# parameter registry: ``nn.Parameter`` is a tensor whose ``__eq__`` is
# elementwise, which breaks ``WeakKeyDictionary``'s internal ref equality. The
# id-keyed variant keys by object identity, is tensor-safe, and auto-removes
# dead entries via weakrefs (no manual finalizers, no id-reuse hazard). Entry
# lifetime == key-object lifetime, identical to the former attribute. Modules'
# metadata is permanent/cross-session; a process-wide registry preserves that.
# --------------------------------------------------------------------------- #
# Values are ModuleMeta / ParamMeta respectively (WeakIdKeyDictionary is not
# typing-subscriptable, so the value type is documented rather than annotated).
_MODULE_REGISTRY: "WeakIdKeyDictionary" = WeakIdKeyDictionary()
_PARAM_REGISTRY: "WeakIdKeyDictionary" = WeakIdKeyDictionary()


def get(obj: Any) -> Optional[TorchLensMeta]:
    """Return TorchLens metadata attached to an object.

    Parameters
    ----------
    obj : Any
        Object that may carry a ``._tl`` namespace.

    Returns
    -------
    Optional[TorchLensMeta]
        TorchLens metadata if present, otherwise ``None``.

    Raises
    ------
    TorchLensTLCollisionError
        If ``obj._tl`` exists but is not TorchLens-owned metadata.
    """
    meta = getattr(obj, "_tl", None)
    if meta is None:
        return None
    if not isinstance(meta, TorchLensMeta):
        raise TorchLensTLCollisionError(
            f"Foreign _tl attribute on {type(obj).__name__}: {type(meta).__name__}"
        )
    return meta


def is_tracked(obj: Any) -> bool:
    """Return whether an object has TorchLens-owned ``._tl`` metadata.

    Parameters
    ----------
    obj : Any
        Object to inspect.

    Returns
    -------
    bool
        True when TorchLens metadata is present.
    """
    if isinstance(obj, nn.Module):
        return obj in _MODULE_REGISTRY
    if isinstance(obj, nn.Parameter):
        return obj in _PARAM_REGISTRY
    return get(obj) is not None


def clear_meta(obj: Any) -> None:
    """Remove TorchLens-owned ``._tl`` metadata from an object.

    Parameters
    ----------
    obj : Any
        Object whose TorchLens metadata should be cleared.

    Notes
    -----
    Foreign ``._tl`` values are preserved. Module/parameter metadata lives in the
    identity-keyed registries and is removed there.
    """
    existing = getattr(obj, "_tl", None)
    if isinstance(existing, TorchLensMeta):
        # TorchLens-owned tensor/callable: metadata is on the attribute.
        try:
            delattr(obj, "_tl")
        except AttributeError:
            pass
        return
    # No TorchLens attribute (any foreign ``_tl`` is left untouched). Registry-stored
    # module/param metadata, if any, is removed here. ``isinstance(_, nn.Parameter)``
    # is the expensive check, so it is reached only for attribute-less non-module
    # objects -- never the hot plain-tensor cleanup path.
    if isinstance(obj, nn.Module):
        _MODULE_REGISTRY.pop(obj, None)
    elif isinstance(obj, nn.Parameter):
        _PARAM_REGISTRY.pop(obj, None)


def get_tensor_meta(t: Any) -> Optional[TensorMeta]:
    """Return tensor metadata, raising on foreign or wrong-kind metadata.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.

    Returns
    -------
    Optional[TensorMeta]
        Tensor metadata if present.
    """
    meta = get(t)
    if meta is None:
        return None
    if not isinstance(meta, TensorMeta):
        raise TorchLensTLCollisionError(
            f"Expected TensorMeta on {type(t).__name__}, found {type(meta).__name__}"
        )
    return meta


def _ensure_tensor_meta(t: Any) -> TensorMeta:
    """Return existing tensor metadata or attach a new tensor namespace.

    Parameters
    ----------
    t : Any
        Tensor-like object to mutate.

    Returns
    -------
    TensorMeta
        Tensor metadata namespace.
    """
    meta = get_tensor_meta(t)
    if meta is None:
        meta = TensorMeta()
        t._tl = meta
    return meta


def set_tensor_label(t: Any, label: str) -> None:
    """Set the raw capture label on a tensor.

    Parameters
    ----------
    t : Any
        Tensor-like object to tag.
    label : str
        Raw TorchLens label.
    """
    _ensure_tensor_meta(t).label_raw = label


def get_tensor_label(t: Any) -> Optional[str]:
    """Return a tensor's raw capture label.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.

    Returns
    -------
    Optional[str]
        Raw label if present.
    """
    meta = get_tensor_meta(t)
    return None if meta is None else meta.label_raw


def clear_tensor_label(t: Any) -> None:
    """Clear only the raw capture label on a tensor.

    Parameters
    ----------
    t : Any
        Tensor-like object to update.
    """
    meta = get_tensor_meta(t)
    if meta is not None:
        meta.label_raw = None


def promote_label_to_buffer_source_and_clear_label(t: Any) -> None:
    """Move a tensor label into ``buffer_source`` and clear the raw label.

    Parameters
    ----------
    t : Any
        Tensor-like object to update.
    """
    meta = get_tensor_meta(t)
    if meta is not None and meta.label_raw is not None:
        meta.buffer_source = meta.label_raw
        meta.label_raw = None


def set_buffer_address(t: Any, address: str) -> None:
    """Set a tensor's buffer address.

    Parameters
    ----------
    t : Any
        Tensor-like object to tag.
    address : str
        Dotted buffer address.
    """
    _ensure_tensor_meta(t).address = address


def get_buffer_address(t: Any) -> Optional[str]:
    """Return a tensor's buffer address.

    Parameters
    ----------
    t : Any
        Tensor-like object to inspect.

    Returns
    -------
    Optional[str]
        Dotted buffer address if present.
    """
    meta = get_tensor_meta(t)
    return None if meta is None else meta.address


def get_label_list(tensors: Iterable[Any]) -> List[str]:
    """Return sparse raw labels from a tensor iterable.

    Parameters
    ----------
    tensors : Iterable[Any]
        Tensor-like objects to scan.

    Returns
    -------
    List[str]
        Labels for tensors with ``TensorMeta.label_raw`` set.

    Raises
    ------
    TorchLensTLCollisionError
        If a tensor has a foreign non-TorchLens ``._tl`` value.
    """
    out: List[str] = []
    for t in tensors:
        meta = getattr(t, "_tl", None)
        if meta is None:
            continue
        if not isinstance(meta, TorchLensMeta):
            raise TorchLensTLCollisionError(f"Foreign _tl on tensor: {type(meta).__name__}")
        if isinstance(meta, TensorMeta) and meta.label_raw is not None:
            out.append(meta.label_raw)
    return out


def get_param_meta(p: Any) -> Optional[ParamMeta]:
    """Return parameter metadata, raising on foreign or wrong-kind metadata.

    Parameters
    ----------
    p : Any
        Parameter-like object to inspect.

    Returns
    -------
    Optional[ParamMeta]
        Parameter metadata if present.
    """
    return _PARAM_REGISTRY.get(p)


def _ensure_param_meta(p: Any) -> ParamMeta:
    """Return existing parameter metadata or attach a new parameter namespace.

    Parameters
    ----------
    p : Any
        Parameter-like object to mutate.

    Returns
    -------
    ParamMeta
        Parameter metadata namespace.
    """
    meta = _PARAM_REGISTRY.get(p)
    if meta is None:
        meta = ParamMeta()
        _PARAM_REGISTRY[p] = meta
    return meta


def set_param_meta(p: Any, *, barcode: str, address: str, requires_grad_before: bool) -> None:
    """Set all session metadata on a parameter.

    Parameters
    ----------
    p : Any
        Parameter-like object to tag.
    barcode : str
        Parameter-sharing barcode.
    address : str
        Dotted parameter address.
    requires_grad_before : bool
        ``requires_grad`` value before TorchLens changed it.
    """
    meta = _ensure_param_meta(p)
    meta.param_barcode = barcode
    meta.param_address = address
    meta.call_index = 0
    meta.requires_grad_before_capture = requires_grad_before


def increment_param_call_index(p: Any) -> int:
    """Increment and return a parameter's call index.

    Parameters
    ----------
    p : Any
        Parameter-like object to update.

    Returns
    -------
    int
        New call index.
    """
    meta = _ensure_param_meta(p)
    meta.call_index += 1
    return meta.call_index


def restore_param_requires_grad(p: Any) -> None:
    """Restore a parameter's pre-capture ``requires_grad`` flag.

    Parameters
    ----------
    p : Any
        Parameter-like object to restore.
    """
    meta = get_param_meta(p)
    if meta is not None and meta.requires_grad_before_capture is not None:
        p.requires_grad = meta.requires_grad_before_capture


def get_module_meta(m: Any) -> Optional[ModuleMeta]:
    """Return module metadata, raising on foreign or wrong-kind metadata.

    Parameters
    ----------
    m : Any
        Module-like object to inspect.

    Returns
    -------
    Optional[ModuleMeta]
        Module metadata if present.
    """
    return _MODULE_REGISTRY.get(m)


def _ensure_module_meta(m: Any) -> ModuleMeta:
    """Return existing module metadata or attach a new module namespace.

    Parameters
    ----------
    m : Any
        Module-like object to mutate.

    Returns
    -------
    ModuleMeta
        Module metadata namespace.
    """
    meta = _MODULE_REGISTRY.get(m)
    if meta is None:
        meta = ModuleMeta()
        _MODULE_REGISTRY[m] = meta
    return meta


def set_module_meta(m: Any, *, address: str, module_type: str) -> None:
    """Set permanent module metadata.

    Parameters
    ----------
    m : Any
        Module-like object to tag.
    address : str
        Dotted module address.
    module_type : str
        Module class name.
    """
    meta = _ensure_module_meta(m)
    meta.address = address
    meta.module_type = module_type


def _ensure_decoration_tag(fn: Any) -> DecorationTag:
    """Return existing callable metadata or attach a new decoration namespace.

    Parameters
    ----------
    fn : Any
        Callable-like object to tag.

    Returns
    -------
    DecorationTag
        Decoration metadata namespace.
    """
    meta = get(fn)
    if meta is None:
        meta = DecorationTag()
        fn._tl = meta
        return meta
    if not isinstance(meta, DecorationTag):
        raise TorchLensTLCollisionError(
            f"Expected DecorationTag on {type(fn).__name__}, found {type(meta).__name__}"
        )
    return meta


def _get_decoration_tag(fn: Any) -> Optional[DecorationTag]:
    """Return callable decoration metadata if present.

    Parameters
    ----------
    fn : Any
        Callable-like object to inspect.

    Returns
    -------
    Optional[DecorationTag]
        Decoration metadata if present.
    """
    meta = get(fn)
    if meta is None:
        return None
    if not isinstance(meta, DecorationTag):
        raise TorchLensTLCollisionError(
            f"Expected DecorationTag on {type(fn).__name__}, found {type(meta).__name__}"
        )
    return meta


def mark_decorated_function(fn: Any) -> None:
    """Mark a wrapped torch function as decorated.

    Parameters
    ----------
    fn : Any
        Callable-like object to tag.
    """
    _ensure_decoration_tag(fn).is_decorated_function = True


def is_decorated_function(fn: Any) -> bool:
    """Return whether a callable is a decorated torch function.

    Parameters
    ----------
    fn : Any
        Callable-like object to inspect.

    Returns
    -------
    bool
        True when marked as a decorated torch function.
    """
    meta = _get_decoration_tag(fn)
    return False if meta is None else meta.is_decorated_function


def mark_forward_call_decorated(fwd: Any) -> None:
    """Mark a module ``forward`` replacement as decorated.

    Parameters
    ----------
    fwd : Any
        Callable-like object to tag.
    """
    _ensure_decoration_tag(fwd).forward_call_is_decorated = True


def is_forward_call_decorated(fwd: Any) -> bool:
    """Return whether a module ``forward`` replacement is decorated.

    Parameters
    ----------
    fwd : Any
        Callable-like object to inspect.

    Returns
    -------
    bool
        True when the callable is a decorated forward replacement.
    """
    meta = _get_decoration_tag(fwd)
    return False if meta is None else meta.forward_call_is_decorated


def mark_tensor_replacement_wrapped(hook: Any) -> None:
    """Mark an intervention hook as wrapped for tensor replacement.

    Parameters
    ----------
    hook : Any
        Callable-like object to tag.
    """
    _ensure_decoration_tag(hook).tensor_replacement_wrapped = True


def is_tensor_replacement_wrapped(hook: Any) -> bool:
    """Return whether an intervention hook is wrapped for tensor replacement.

    Parameters
    ----------
    hook : Any
        Callable-like object to inspect.

    Returns
    -------
    bool
        True when the hook has the tensor replacement wrapper sentinel.
    """
    meta = _get_decoration_tag(hook)
    return False if meta is None else meta.tensor_replacement_wrapped


def copy_replacement_meta(src: Any, dst: Any) -> None:
    """Copy TorchLens metadata from one replacement tensor to another.

    Parameters
    ----------
    src : Any
        Source object whose metadata should be copied.
    dst : Any
        Destination object that should receive a shallow dataclass copy.
    """
    src_meta = get(src)
    if src_meta is not None:
        dst._tl = dataclass_replace(cast(Any, src_meta))
