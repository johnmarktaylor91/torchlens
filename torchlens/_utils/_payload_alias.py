"""The alias a deferred payload holds in place of an eager copy."""

from __future__ import annotations

import torch

from .tensor_utils import _mint_graph_connected_alias


def mint_payload_alias(x: torch.Tensor, graph_connected: bool) -> torch.Tensor:
    """Return the deferred-payload alias of ``x``, as ``_try_defer_payload_alias`` mints it.

    ``graph_connected`` mints the alias connected to the autograd graph; otherwise
    the alias is the historical ``detach()``. The direct payload path calls this so
    the detach stays with the tensor utilities, outside the capture paths.
    """

    return _mint_graph_connected_alias(x) if graph_connected else x.detach()
