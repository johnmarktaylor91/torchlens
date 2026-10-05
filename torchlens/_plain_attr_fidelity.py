"""Plain-attribute fidelity facts for the validation deepcopy and fallback paths.

Validation runs its ground truth and its replay on deep copies of the model so the
caller's model state is untouched. Two facts decide whether that is sound:

* whether ``copy.deepcopy`` reproduced every plain attribute's instance namespace
  (:func:`assert_copy_kept_instance_attrs`); the ``self.__dict__ = self`` AttrDict
  idiom used by HiFi-GAN-family configs copies its items but leaves the copy's
  attribute namespace empty, so the copy's forward raises where the source's does not;
* whether a plain attribute merely aliases the module's own registered tensors
  (:func:`is_registered_tensor_alias`), as ``nn.RNNBase._flat_weights`` does; such
  values are restored by the state_dict restore, so only their identity is plain state.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

_FIDELITY_MAX_DEPTH = 4
_FIDELITY_MAX_CONTAINER_ITEMS = 128
_OBJECT_GETSTATE = getattr(object, "__getstate__", None)


def assert_copy_kept_instance_attrs(source: nn.Module, copied: nn.Module) -> None:
    """Raise when a deepcopy dropped instance attributes from a plain attribute value.

    Parameters
    ----------
    source:
        Original module tree.
    copied:
        ``copy.deepcopy(source)``.

    Raises
    ------
    ValueError
        If the module trees differ in arity, or if a plain attribute value whose
        type uses the default copy protocol lost instance-attribute names in the
        copy. Callers treat this exactly like a failed deepcopy and take the
        disclosed live-model fallback.
    """

    from ._capture_state_helpers import _module_plain_attr_names

    source_modules = list(source.modules())
    copied_modules = list(copied.modules())
    if len(source_modules) != len(copied_modules):
        raise ValueError(
            "Validation deepcopy changed the module tree arity: source has "
            f"{len(source_modules)} modules but the copy has {len(copied_modules)}."
        )
    for source_module, copied_module in zip(source_modules, copied_modules, strict=True):
        shared_names = _module_plain_attr_names(source_module) & _module_plain_attr_names(
            copied_module
        )
        for name in sorted(shared_names):
            _compare_instance_attrs(
                source_module.__dict__[name],
                copied_module.__dict__[name],
                f"{type(source_module).__name__}.{name}",
                depth=0,
                seen=frozenset(),
            )


def _compare_instance_attrs(
    source: Any,
    copied: Any,
    attr_path: str,
    *,
    depth: int,
    seen: frozenset[int],
) -> None:
    """Recursively compare instance-attribute names of a source value and its copy.

    Parameters
    ----------
    source:
        Plain attribute value (or nested item) on the source module.
    copied:
        The corresponding value on the copied module.
    attr_path:
        Human-readable attribute path for the error message.
    depth:
        Current container depth; recursion stops past ``_FIDELITY_MAX_DEPTH``.
    seen:
        Ids of source values already on the recursion path (cycle guard).

    Raises
    ------
    ValueError
        If ``copied`` lacks instance-attribute names that ``source`` has.
    """

    if depth > _FIDELITY_MAX_DEPTH or id(source) in seen:
        return
    if isinstance(source, (nn.Module, torch.Tensor, type)):
        return
    if callable(source) and not isinstance(source, (list, tuple, dict)):
        return
    child_seen = seen | {id(source)}
    if isinstance(source, (list, tuple)) and isinstance(copied, (list, tuple)):
        if len(source) == len(copied) and len(source) <= _FIDELITY_MAX_CONTAINER_ITEMS:
            for index, (source_item, copied_item) in enumerate(zip(source, copied, strict=True)):
                _compare_instance_attrs(
                    source_item,
                    copied_item,
                    f"{attr_path}[{index}]",
                    depth=depth + 1,
                    seen=child_seen,
                )
    elif isinstance(source, dict) and isinstance(copied, dict):
        if len(source) <= _FIDELITY_MAX_CONTAINER_ITEMS:
            for key, source_item in source.items():
                if key in copied:
                    _compare_instance_attrs(
                        source_item,
                        copied[key],
                        f"{attr_path}[{key!r}]",
                        depth=depth + 1,
                        seen=child_seen,
                    )
    _compare_namespace(source, copied, attr_path, depth=depth, seen=child_seen)


def _compare_namespace(
    source: Any,
    copied: Any,
    attr_path: str,
    *,
    depth: int,
    seen: frozenset[int],
) -> None:
    """Compare the ``__dict__`` names of one value and its copy, then recurse into them.

    Parameters
    ----------
    source:
        Source value.
    copied:
        Copied value.
    attr_path:
        Human-readable attribute path for the error message.
    depth:
        Current container depth.
    seen:
        Ids of source values on the recursion path, including ``source``.

    Raises
    ------
    ValueError
        If ``copied`` lacks instance-attribute names that ``source`` has.
    """

    try:
        source_attrs = getattr(source, "__dict__", None)
        copied_attrs = getattr(copied, "__dict__", None)
    except Exception:  # noqa: BLE001 -- an exotic __getattr__ gives no fidelity fact
        return
    if not isinstance(source_attrs, dict) or not _uses_default_copy_protocol(type(source)):
        return
    copied_names = set(copied_attrs) if isinstance(copied_attrs, dict) else set()
    missing = sorted(str(name) for name in set(source_attrs) - copied_names)
    if missing:
        raise ValueError(
            f"Validation deepcopy dropped instance attributes {missing[:8]!r} of plain "
            f"attribute '{attr_path}' ({type(source).__name__}); the copy would not run "
            "like the source."
        )
    if source_attrs is source or len(source_attrs) > _FIDELITY_MAX_CONTAINER_ITEMS:
        return  # an AttrDict's namespace is its items, already walked above
    for name, source_item in source_attrs.items():
        _compare_instance_attrs(
            source_item,
            copied_attrs[name],
            f"{attr_path}.{name}",
            depth=depth + 1,
            seen=seen,
        )


def _uses_default_copy_protocol(value_type: type) -> bool:
    """Return whether ``copy.deepcopy`` copies instances of a type by its default protocol.

    Parameters
    ----------
    value_type:
        Type of the plain attribute value.

    Returns
    -------
    bool
        ``False`` when the type customizes copying (``__deepcopy__``, ``__reduce__``,
        ``__reduce_ex__``, ``__getstate__`` or ``__setstate__``); such a type may drop
        cached attributes on purpose, so a missing name is not evidence of a broken copy.
    """

    if getattr(value_type, "__deepcopy__", None) is not None:
        return False
    if value_type.__reduce_ex__ is not object.__reduce_ex__:
        return False
    if value_type.__reduce__ is not object.__reduce__:
        return False
    if getattr(value_type, "__getstate__", None) is not _OBJECT_GETSTATE:
        return False
    return getattr(value_type, "__setstate__", None) is None


def is_registered_tensor_alias(module: nn.Module, value: Any) -> bool:
    """Return whether a plain attribute only aliases the module's own registered tensors.

    Parameters
    ----------
    module:
        Module that owns the plain attribute.
    value:
        Attribute value.

    Returns
    -------
    bool
        ``True`` for a tensor that is (by identity) one of ``module``'s registered
        parameters or buffers, or an exact ``list``/``tuple`` of such tensors and
        ``None`` with at least one tensor (``nn.RNNBase._flat_weights``).
    """

    registered = {
        id(tensor)
        for tensor in (*module._parameters.values(), *module._buffers.values())
        if tensor is not None
    }
    if isinstance(value, torch.Tensor):
        return id(value) in registered
    if type(value) not in (list, tuple) or not any(item is not None for item in value):
        return False
    return all(item is None or id(item) in registered for item in value)


def restore_simple_plain_attrs_on_copy(source: nn.Module, copied: nn.Module) -> None:
    """Align simple plain attributes on a validation copy with its source.

    Parameters
    ----------
    source:
        Original module tree.
    copied:
        Deep-copied module tree that should represent ``source`` at validation
        entry.
    """

    from ._capture_state_helpers import _module_plain_attr_names

    simple_types = (type(None), bool, int, float, complex, str, bytes)
    # The two trees are zipped POSITIONALLY, so a deepcopy that adds or drops a
    # submodule (a __deepcopy__ hook, lazy child materialization) used to
    # silently shift every later pair and align attributes onto the WRONG
    # modules (T11.10). Arity is checked BEFORE the loop (a lazy strict zip
    # would fire only at exhaustion, after shifted pairs already mutated the
    # copy); on mismatch the caller's existing fallback validates against the
    # live model with a disclosure.
    source_modules = list(source.modules())
    copied_modules = list(copied.modules())
    if len(source_modules) != len(copied_modules):
        raise ValueError(
            "Validation deepcopy changed the module tree arity: source has "
            f"{len(source_modules)} modules but the copy has {len(copied_modules)}; "
            "positional attribute restoration would misalign."
        )
    for source_module, copied_module in zip(source_modules, copied_modules, strict=True):
        source_names = _module_plain_attr_names(source_module)
        copied_names = _module_plain_attr_names(copied_module)
        for name in sorted(source_names & copied_names):
            source_value = getattr(source_module, name)
            copied_value = getattr(copied_module, name)
            if not isinstance(source_value, simple_types):
                continue
            if not isinstance(copied_value, simple_types):
                continue
            if source_value != copied_value:
                setattr(copied_module, name, source_value)
