"""Label naming for autograd grad_fn nodes: normalized type names and label parts."""

from __future__ import annotations

import re
from typing import Any


def _normalize_grad_fn_type(grad_fn_handle: Any) -> str:
    """Normalize an autograd grad_fn_handle class name for TorchLens labels.

    Parameters
    ----------
    grad_fn_handle:
        Autograd function object.

    Returns
    -------
    str
        Lowercased class name with a trailing ``Backward<digits>`` suffix removed.
    """
    return re.sub(r"Backward\d*$", "", type(grad_fn_handle).__name__).lower()


def _grad_fn_label_parts(
    trace: Any,
    type: str,
    _layer_label: str | None,
    type_counter: dict[str, int],
    total_num: int,
) -> tuple[int, int, str]:
    """Build numeric label fields for one grad_fn_handle.

    Parameters
    ----------
    trace:
        Trace being updated.
    type:
        Normalized grad_fn_handle type.
    _layer_label:
        Matching forward layer label, or ``None`` for intervening grad_fns.
    type_counter:
        Running per-type counter for intervening grad_fns.
    total_num:
        One-based discovery index in the backward graph.

    Returns
    -------
    tuple[int, int, str]
        GradFn type index, total index, and user-facing label.
    """
    del trace, _layer_label
    type_counter[type] = type_counter.get(type, 0) + 1
    type_num = type_counter[type]
    label = f"{type}_back_{type_num}_{total_num}"
    return type_num, total_num, label


def _grad_fn_type_from_class_name(class_name: str) -> str:
    """Normalize an autograd class name for native backward labels."""

    return re.sub(r"Backward\d*$", "", class_name).lower()
