"""Canonical comparison helpers for TorchLens golden pickle parity tests."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any
import weakref

import numpy as np
import torch

from _pickle_compare_allowlist import allowed_pickle_diff_fields
from torchlens.data_classes._state_adapter import state_items


def _tensor_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return whether two tensors are exactly equal for parity purposes.

    Parameters
    ----------
    a
        Left tensor.
    b
        Right tensor.

    Returns
    -------
    bool
        True when shape, dtype, device, and values match exactly, with NaNs
        treated as equal when they appear in the same positions.
    """

    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    left = a.detach()
    right = b.detach()
    if a.dtype in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        return bool(torch.equal(left, right))
    if a.is_floating_point() or a.is_complex():
        mask_a = torch.isnan(left)
        mask_b = torch.isnan(right)
        if not torch.equal(mask_a, mask_b):
            return False
        finite_a = left[~mask_a]
        finite_b = right[~mask_b]
        return bool(torch.allclose(finite_a, finite_b, atol=0, rtol=0))
    return bool(torch.equal(left, right))


def _canonical_pickle_diff(a: Any, b: Any) -> list[str]:
    """Return semantic pickle differences after applying the committed allow-list.

    Parameters
    ----------
    a
        Current object.
    b
        Golden object.

    Returns
    -------
    list[str]
        Human-readable difference paths. Empty means canonical parity passed.
    """

    return _diff(a, b, "trace", set())


def _diff(a: Any, b: Any, path: str, seen: set[tuple[int, int]]) -> list[str]:
    """Recursively compare two Python objects.

    Parameters
    ----------
    a
        Current object.
    b
        Golden object.
    path
        Human-readable object path.
    seen
        Object id pairs already compared.

    Returns
    -------
    list[str]
        Differences found under ``path``.
    """

    pair = (id(a), id(b))
    if pair in seen:
        return []
    seen.add(pair)
    if isinstance(a, weakref.ReferenceType) and isinstance(b, weakref.ReferenceType):
        return []
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return [] if _tensor_equal(a, b) else [f"{path}: tensors differ"]
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return [] if np.array_equal(a, b, equal_nan=True) else [f"{path}: arrays differ"]
    if type(a) is not type(b):
        return [f"{path}: type {type(a)!r} != {type(b)!r}"]
    if is_dataclass(a) and not isinstance(a, type):
        names = [field.name for field in fields(a)]
        return _diff_attrs(a, b, names, path, seen)
    if getattr(type(a), "PORTABLE_STATE_SPEC", None) is not None:
        left_state = dict(state_items(a))
        right_state = dict(state_items(b))
        left_keys = set(left_state) - _ignored_fields(a)
        right_keys = set(right_state) - _ignored_fields(a)
        diffs: list[str] = []
        if left_keys != right_keys:
            diffs.append(f"{path}: keys differ {sorted(left_keys ^ right_keys)!r}")
        for key in sorted(left_keys & right_keys):
            diffs.extend(_diff(left_state[key], right_state[key], f"{path}.{key}", seen))
        return diffs
    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        left_keys = set(a.__dict__)
        right_keys = set(b.__dict__)
        ignored = _ignored_fields(a)
        left_keys -= ignored
        right_keys -= ignored
        diffs: list[str] = []
        if left_keys != right_keys:
            diffs.append(f"{path}: keys differ {sorted(left_keys ^ right_keys)!r}")
        for key in sorted(left_keys & right_keys):
            diffs.extend(_diff(a.__dict__[key], b.__dict__[key], f"{path}.{key}", seen))
        return diffs
    if isinstance(a, dict):
        return _diff_mapping(a, b, path, seen)
    if isinstance(a, (list, tuple)):
        return _diff_sequence(a, b, path, seen)
    if isinstance(a, set):
        return [] if sorted(map(repr, a)) == sorted(map(repr, b)) else [f"{path}: sets differ"]
    try:
        equal = a == b
    except Exception:
        equal = repr(a) == repr(b)
    if isinstance(equal, torch.Tensor):
        equal = bool(torch.all(equal).item())
    if isinstance(equal, np.ndarray):
        equal = bool(np.all(equal))
    return [] if bool(equal) else [f"{path}: {a!r} != {b!r}"]


def _ignored_fields(obj: Any) -> frozenset[str]:
    """Return allow-listed field names for an object.

    Parameters
    ----------
    obj
        Object being compared.

    Returns
    -------
    frozenset[str]
        Field names ignored for this object's class.
    """

    return allowed_pickle_diff_fields().get(type(obj).__name__, frozenset())


def _diff_attrs(
    a: Any, b: Any, names: list[str], path: str, seen: set[tuple[int, int]]
) -> list[str]:
    """Compare named attributes on two objects.

    Parameters
    ----------
    a
        Current object.
    b
        Golden object.
    names
        Attribute names to compare.
    path
        Human-readable object path.
    seen
        Object id pairs already compared.

    Returns
    -------
    list[str]
        Differences found under ``path``.
    """

    ignored = _ignored_fields(a)
    diffs: list[str] = []
    for name in names:
        if name in ignored:
            continue
        diffs.extend(_diff(getattr(a, name), getattr(b, name), f"{path}.{name}", seen))
    return diffs


def _diff_mapping(
    a: dict[Any, Any], b: dict[Any, Any], path: str, seen: set[tuple[int, int]]
) -> list[str]:
    """Compare two mappings.

    Parameters
    ----------
    a
        Current mapping.
    b
        Golden mapping.
    path
        Human-readable object path.
    seen
        Object id pairs already compared.

    Returns
    -------
    list[str]
        Differences found under ``path``.
    """

    if set(a) != set(b):
        return [f"{path}: dict keys differ {sorted(set(a) ^ set(b))!r}"]
    diffs: list[str] = []
    for key in sorted(a, key=repr):
        diffs.extend(_diff(a[key], b[key], f"{path}[{key!r}]", seen))
    return diffs


def _diff_sequence(
    a: list[Any] | tuple[Any, ...],
    b: list[Any] | tuple[Any, ...],
    path: str,
    seen: set[tuple[int, int]],
) -> list[str]:
    """Compare two sequences.

    Parameters
    ----------
    a
        Current sequence.
    b
        Golden sequence.
    path
        Human-readable object path.
    seen
        Object id pairs already compared.

    Returns
    -------
    list[str]
        Differences found under ``path``.
    """

    if len(a) != len(b):
        return [f"{path}: len {len(a)} != {len(b)}"]
    diffs: list[str] = []
    for index, (left, right) in enumerate(zip(a, b)):
        diffs.extend(_diff(left, right, f"{path}[{index}]", seen))
    return diffs
