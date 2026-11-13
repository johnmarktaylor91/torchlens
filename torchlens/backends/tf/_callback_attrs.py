"""TensorFlow op-callback attribute and op-type normalization helpers."""

from __future__ import annotations

from typing import Any


def _normalize_op_type(op_type: Any) -> str:
    """Normalize TensorFlow callback op type to text.

    Parameters
    ----------
    op_type
        Raw callback op type.

    Returns
    -------
    str
        Normalized operation type.
    """

    if isinstance(op_type, bytes):
        return op_type.decode("utf-8", errors="replace")
    return str(op_type)


def _attrs_to_public_dict(attrs: tuple[Any, ...] | dict[str, Any]) -> dict[str, str]:
    """Convert TensorFlow attrs into a compact serializable diagnostic mapping.

    Parameters
    ----------
    attrs
        Raw callback attrs.

    Returns
    -------
    dict[str, str]
        Stringified attributes.
    """

    if isinstance(attrs, dict):
        return {str(key): repr(value) for key, value in attrs.items()}
    if isinstance(attrs, tuple):
        return {str(index): repr(value) for index, value in enumerate(attrs)}
    return {}


def _attrs_to_raw_dict(attrs: tuple[Any, ...] | dict[str, Any]) -> dict[str, Any]:
    """Convert TensorFlow callback attrs into a raw replay mapping.

    Parameters
    ----------
    attrs
        Raw callback attrs.

    Returns
    -------
    dict[str, Any]
        Attribute mapping preserving runtime values for raw-op replay.
    """

    if isinstance(attrs, dict):
        return {str(key): value for key, value in attrs.items()}
    if isinstance(attrs, tuple):
        result: dict[str, Any] = {}
        iterator = iter(attrs)
        for key in iterator:
            try:
                value = next(iterator)
            except StopIteration:
                break
            result[str(key)] = value
        return result
    return {}
