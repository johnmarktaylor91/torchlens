"""Semantic role hints for structured module outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch import nn

from ..intervention.types import (
    DataclassField,
    DictKey,
    HFKey,
    NamedField,
    OutputPathComponent,
    TupleIndex,
)

ROLE_HINTS_BY_MODULE_TYPE: dict[type[nn.Module], dict[tuple[Any, ...], str]] = {
    nn.LSTM: {
        (0,): "output",
        (1,): "h_n",
        (2,): "c_n",
        (1, 0): "h_n",
        (1, 1): "c_n",
    },
    nn.GRU: {
        (0,): "output",
        (1,): "h_n",
    },
    nn.RNN: {
        (0,): "output",
        (1,): "h_n",
    },
    nn.MultiheadAttention: {
        (0,): "attn_output",
        (1,): "attn_output_weights",
    },
}


def role_hints_for_module(module: nn.Module | None) -> Mapping[tuple[Any, ...], str] | None:
    """Return semantic output-role hints for a module instance.

    Parameters
    ----------
    module:
        Module whose return structure is being described.

    Returns
    -------
    Mapping[tuple[Any, ...], str] | None
        Primitive path tuple to role-name mapping, or ``None`` when no known
        semantic hints apply.
    """

    if module is None:
        return None
    for module_type, hints in ROLE_HINTS_BY_MODULE_TYPE.items():
        if isinstance(module, module_type):
            return hints
    return None


def role_hints_for_module_class(
    module_class: type[nn.Module] | None,
) -> Mapping[tuple[Any, ...], str] | None:
    """Return semantic output-role hints for a module class.

    Parameters
    ----------
    module_class:
        Module class whose return structure is being described.

    Returns
    -------
    Mapping[tuple[Any, ...], str] | None
        Primitive path tuple to role-name mapping, or ``None`` when no known
        semantic hints apply.
    """

    if module_class is None:
        return None
    for module_type, hints in ROLE_HINTS_BY_MODULE_TYPE.items():
        if issubclass(module_class, module_type):
            return hints
    return None


def normalize_output_path(
    path: tuple[OutputPathComponent, ...] | tuple[Any, ...],
) -> tuple[Any, ...]:
    """Convert output path components to primitive keys.

    Parameters
    ----------
    path:
        Path emitted by structured output traversal.

    Returns
    -------
    tuple[Any, ...]
        Primitive tuple suitable for stable role-hint lookup.
    """

    normalized: list[Any] = []
    for component in path:
        if isinstance(component, TupleIndex):
            normalized.append(component.index)
        elif isinstance(component, DictKey):
            normalized.append(component.key)
        elif isinstance(component, NamedField | DataclassField):
            normalized.append(component.name)
        elif isinstance(component, HFKey):
            normalized.append(component.key)
        else:
            normalized.append(component)
    return tuple(normalized)


def multi_output_role_from_path(
    path: tuple[OutputPathComponent, ...] | tuple[Any, ...],
    index: int | None,
    *,
    hints: Mapping[tuple[Any, ...], str] | None = None,
) -> str | None:
    """Return a stable role string for one structured output leaf.

    Parameters
    ----------
    path:
        Container path emitted by output traversal.
    index:
        Zero-based fallback output index.
    hints:
        Optional semantic role hints keyed by primitive path tuples.

    Returns
    -------
    str | None
        Role string for selector and display use. ``None`` is returned for a
        direct single-tensor output with no container path.
    """

    normalized = normalize_output_path(path)
    if hints is not None and normalized in hints:
        return hints[normalized]
    if normalized:
        return ".".join(str(component) for component in normalized)
    if index is not None:
        return str(index)
    return None
