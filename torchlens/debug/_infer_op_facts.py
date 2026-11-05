"""Executed-op fact readers for the input-shape guesser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import record_member

if TYPE_CHECKING:
    pass


def _shape_tuple(shape: Any) -> tuple[int, ...] | None:
    """Convert a Torch/TorchLens shape-like object to an integer tuple.

    Parameters
    ----------
    shape:
        Shape-like object.

    Returns
    -------
    tuple[int, ...] | None
        Converted shape, or ``None`` when conversion is not possible.
    """

    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _shape_op_label(op: Any) -> str | None:
    """Return an op label from a TorchLens layer/op object.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Label when present.
    """

    label = getattr(op, "layer_label", None) or getattr(op, "op_label", None)
    return str(label) if label is not None else None


def _shape_op_module(op: Any) -> str | None:
    """Return a module address from a TorchLens op-like object.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Module address when present.
    """

    module = getattr(op, "module_address", None) or getattr(op, "module", None)
    return str(module) if module is not None else None


def _shape_op_source_line(op: Any) -> str | None:
    """Return an op source-line string when TorchLens exposes one.

    Parameters
    ----------
    op:
        TorchLens op-like object.

    Returns
    -------
    str | None
        Source-line string, or ``None``.
    """

    source_line = getattr(op, "source_line", None)
    return str(source_line) if source_line is not None else None


def _shape_op_config_int(op: Any, key: str) -> int | None:
    """Read an integer from an op's ``func_config``.

    Parameters
    ----------
    op:
        TorchLens op-like object.
    key:
        Config key.

    Returns
    -------
    int | None
        Integer value when present.
    """

    config = record_member(op, "func_config", {}) or {}
    value = config.get(key)
    return int(value) if value is not None else None
