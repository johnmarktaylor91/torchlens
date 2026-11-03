"""Input tensor leaves for a torch capture: find them, move them to the model device, address them.

The torch backend's ``_fetch_label_move_input_tensors`` runs these three phases in
order on the copied forward inputs: collect every tensor leaf with its nested
address, move each distinct leaf to the model device (writing it back into its
container), and flatten the leaves into one de-duplicated list of tensors and
``input.<argname>[.<nested_path>]`` labels.
"""

from __future__ import annotations

from typing import Any

import torch

from ..._utils.input_leaves import INPUT_WAS_PARAMETER_ATTR
from ..._utils.introspection import (
    INPUT_SEARCH_DEPTH_LIMIT,
    get_vars_of_type_from_obj,
)
from ._outputs import _assign_nested_input_value


def collect_input_tensor_leaves(
    input_args: list[Any],
    input_arg_names: list[str],
    input_kwargs: dict[Any, Any],
) -> tuple[list[list[Any]], list[list[Any]], list[str]]:
    """Find the tensor leaves of every positional and keyword input.

    Returns the per-argument leaf lists (``(tensor, addr, addr_full)`` triples),
    the per-keyword leaf lists, and the input paths whose traversal reached
    ``INPUT_SEARCH_DEPTH_LIMIT`` unresolved.
    """
    input_arg_tensors = []
    input_kwarg_tensors = []
    traversal_gaps: list[str] = []
    for arg_index, arg in enumerate(input_args):
        unresolved: list[str] = []
        input_arg_tensors.append(
            get_vars_of_type_from_obj(
                arg,
                torch.Tensor,
                search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                return_addresses=True,
                depth_exceeded_paths=unresolved,
            )
        )
        arg_name = input_arg_names[arg_index]
        traversal_gaps.extend(
            f"input.{arg_name}{f'.{path}' if path else ''}" for path in unresolved
        )
    for key, kwarg in input_kwargs.items():
        unresolved = []
        input_kwarg_tensors.append(
            get_vars_of_type_from_obj(
                kwarg,
                torch.Tensor,
                search_depth=INPUT_SEARCH_DEPTH_LIMIT,
                return_addresses=True,
                depth_exceeded_paths=unresolved,
            )
        )
        traversal_gaps.extend(f"input.{key}{f'.{path}' if path else ''}" for path in unresolved)
    return input_arg_tensors, input_kwarg_tensors, traversal_gaps


def move_input_tensors_to_device(
    input_args: list[Any],
    input_kwargs: dict[Any, Any],
    input_arg_tensors: list[list[Any]],
    input_kwarg_tensors: list[list[Any]],
    model_device: object,
) -> None:
    """Move each distinct input tensor leaf to ``model_device``, in place.

    Rewrites the leaf lists and the input containers with the moved tensors; a
    tensor reached twice is moved once, and positional inputs go before keywords.
    """
    moved_tensors_by_id: dict[int, torch.Tensor] = {}
    _move_positional_input_tensors(input_args, input_arg_tensors, model_device, moved_tensors_by_id)
    for kwarg_idx, (key, _val) in enumerate(input_kwargs.items()):
        for tensor_idx, (tensor, addr, addr_full) in enumerate(input_kwarg_tensors[kwarg_idx]):
            moved_tensor = moved_tensors_by_id.get(id(tensor))
            if moved_tensor is None:
                moved_tensor = tensor if model_device is None else tensor.to(model_device)
                moved_tensors_by_id[id(tensor)] = moved_tensor
            if bool(getattr(tensor, INPUT_WAS_PARAMETER_ATTR, False)):
                setattr(moved_tensor, INPUT_WAS_PARAMETER_ATTR, True)
            input_kwarg_tensors[kwarg_idx][tensor_idx] = (moved_tensor, addr, addr_full)
            if not addr_full:
                input_kwargs[key] = moved_tensor
            else:
                input_kwargs[key] = _assign_nested_input_value(
                    input_kwargs[key], addr_full, moved_tensor
                )


def _move_positional_input_tensors(
    input_args: list[Any],
    input_arg_tensors: list[list[Any]],
    model_device: object,
    moved_tensors_by_id: dict[int, torch.Tensor],
) -> None:
    """Move the positional inputs' tensor leaves, recording each move by tensor id."""
    # Move each tensor to model device.  Plain tuples must be temporarily
    # converted to lists for item assignment, then converted back to
    # preserve type.  This roundtrip only applies to *exact* ``tuple``
    # instances: they are the only ones addressed positionally
    # (``("ind", i)``) by ``get_vars_of_type_from_obj``, which treats
    # tuple *subclasses* (e.g. a NamedTuple-based GNN batch container) as
    # plain attribute-bearing objects instead, addressed via
    # ``("attr", name)``.  Applying the list roundtrip to a subclass would
    # silently discard its identity and break downstream named-field
    # access (``batch.edge_features``); ``_assign_nested_input_value``
    # already knows how to mutate those in place via ``attr`` addressing.
    for arg_idx, arg in enumerate(input_args):
        was_tuple = type(arg) is tuple
        if was_tuple:
            input_args[arg_idx] = list(arg)
        for tensor_idx, (tensor, addr, addr_full) in enumerate(input_arg_tensors[arg_idx]):
            moved_tensor = moved_tensors_by_id.get(id(tensor))
            if moved_tensor is None:
                # ``model_device is None`` = the model pins no device
                # (no parameters or buffers); inputs stay on their own
                # devices, matching eager semantics.
                moved_tensor = tensor if model_device is None else tensor.to(model_device)
                moved_tensors_by_id[id(tensor)] = moved_tensor
            if bool(getattr(tensor, INPUT_WAS_PARAMETER_ATTR, False)):
                setattr(moved_tensor, INPUT_WAS_PARAMETER_ATTR, True)
            input_arg_tensors[arg_idx][tensor_idx] = (moved_tensor, addr, addr_full)
            if not addr_full:
                input_args[arg_idx] = moved_tensor
            else:
                input_args[arg_idx] = _assign_nested_input_value(
                    input_args[arg_idx], addr_full, moved_tensor
                )
        if was_tuple and isinstance(input_args[arg_idx], list):
            input_args[arg_idx] = tuple(input_args[arg_idx])


def flatten_input_tensor_addresses(
    input_arg_tensors: list[list[Any]],
    input_kwarg_tensors: list[list[Any]],
    input_arg_names: list[str],
    input_kwargs: dict[Any, Any],
) -> tuple[list[Any], list[str]]:
    """Flatten the leaf lists into distinct tensors and their source-address labels."""
    # Build flat lists of (tensor, address) for both positional and keyword args.
    # Address format: "input.<argname>" or "input.<argname>.<nested_path>"
    input_tensors = []
    input_tensor_addresses = []
    seen_tensor_ids: set[int] = set()
    for arg_idx, arg_tensors in enumerate(input_arg_tensors):
        for tensor, addr, addr_full in arg_tensors:
            if id(tensor) in seen_tensor_ids:
                continue
            seen_tensor_ids.add(id(tensor))
            input_tensors.append(tensor)
            tensor_addr = f"input.{input_arg_names[arg_idx]}"
            if addr != "":
                tensor_addr += f".{addr}"
            input_tensor_addresses.append(tensor_addr)

    for arg_idx, kwarg_tensors in enumerate(input_kwarg_tensors):
        for tensor, addr, addr_full in kwarg_tensors:
            if id(tensor) in seen_tensor_ids:
                continue
            seen_tensor_ids.add(id(tensor))
            input_tensors.append(tensor)
            tensor_addr = f"input.{list(input_kwargs.keys())[arg_idx]}"
            if addr != "":
                tensor_addr += f".{addr}"
            input_tensor_addresses.append(tensor_addr)

    return input_tensors, input_tensor_addresses
