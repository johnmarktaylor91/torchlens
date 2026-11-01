"""Output tensor extraction shared by the torch wrapper families.

``_collect_output_tensors`` is the fast inline path ``wrapped_func`` and the
transform wrappers run on every logged output; it lives below both so neither
imports the other.
"""

from __future__ import annotations

from typing import Any

import torch

from ..._utils.introspection import get_vars_of_type_from_obj


def _collect_output_tensors(out: Any) -> list[torch.Tensor]:
    """Fast inline output tensor extraction.

    Most torch functions return a single tensor. This handles that case
    with a simple isinstance check, falling back to BFS for compound outputs.
    """
    if isinstance(out, torch.Tensor):
        if isinstance(out, torch.nn.Parameter):
            return []
        return [out]
    if isinstance(out, (list, tuple)):
        tensors = []
        for item in out:
            if isinstance(item, torch.Tensor) and not isinstance(item, torch.nn.Parameter):
                tensors.append(item)
        return tensors
    if out is None:
        return []
    if type(out) is torch.utils.hooks.RemovableHandle:
        # ``register_hook``-family output: holds only ints and weakrefs
        # (weakrefs are callable, so the BFS skips them) — provably tensor-free.
        return []
    # Rare: dict, custom object, etc. — fall back to BFS.
    return get_vars_of_type_from_obj(
        out, which_type=torch.Tensor, subclass_exceptions=[torch.nn.Parameter]
    )
