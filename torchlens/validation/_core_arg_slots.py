"""Slot-level evidence readers for the unattributed-arg-slot sweep."""

from __future__ import annotations

from typing import Any

import torch

from .._data_classes.op import Op
from .._utils.tensor_utils import tensor_all_nan


def _tensor_arg_value_is_trivial(value: torch.Tensor) -> bool:
    """Return whether a saved arg tensor value is too generic to attribute.

    Mirrors the triviality exemptions of Case 1 in
    ``_check_arglocs_correct_for_arg`` exactly: empty, bool, all-NaN, all-zero,
    and all-abs-one tensors match producers coincidentally all the time, so a
    value-identity match on them proves nothing.

    Parameters
    ----------
    value:
        Saved argument tensor leaf.

    Returns
    -------
    bool
        True when value-identity evidence on this tensor is not probative.
    """

    return bool(
        value.numel() == 0
        or value.dtype == torch.bool
        or tensor_all_nan(value)
        or torch.all(value == 0)
        or torch.all(torch.abs(value) == 1)
    )


def _matches_own_parameter(target_layer: Op, value: torch.Tensor) -> bool:
    """Return whether a saved arg value is one of the op's own parameters.

    Parameters passed positionally (``F.linear(x, self.weight, self.bias)``)
    are captured in ``saved_args`` but are deliberately NOT graph parents, so
    their slots are legitimately unattributed.

    Parameters
    ----------
    target_layer:
        Operation whose arg slot is being classified.
    value:
        Saved argument tensor leaf at an unattributed slot.

    Returns
    -------
    bool
        True when the value matches one of the op's recorded parameters.
    """

    for param_log in target_layer._param_logs or ():
        param_value = getattr(param_log, "value", None)
        if not isinstance(param_value, torch.Tensor):
            continue
        if param_value.device.type == "meta":
            # Offloaded model (accelerate hooks, lane F37): the live handle is
            # meta between forwards and carries no value evidence, so it can
            # never CONFIRM a match. Fail closed: the slot keeps whatever
            # classification it already has.
            continue
        if (
            param_value.shape == value.shape
            and param_value.dtype == value.dtype
            and torch.equal(param_value, value)
        ):
            return True
    return False


def _saved_arg_leaf_slots(arg_type: str, container: Any) -> list[tuple[Any, str, torch.Tensor]]:
    """Return (argloc_key, witness_path, tensor) leaf slots for a container."""

    if container is None:
        return []
    items = enumerate(container) if arg_type == "args" else container.items()
    prefix = "arg" if arg_type == "args" else "kw:"
    slots: list[tuple[Any, str, torch.Tensor]] = []
    for key, val in items:
        if isinstance(val, torch.Tensor):
            slots.append((key, f"{prefix}{key}", val))
        elif type(val) in (list, tuple):
            for sub_index, sub_val in enumerate(val):
                if isinstance(sub_val, torch.Tensor):
                    slots.append(((key, sub_index), f"{prefix}{key}.{sub_index}", sub_val))
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, torch.Tensor):
                    slots.append(((key, sub_key), f"{prefix}{key}.{sub_key}", sub_val))
    return slots
