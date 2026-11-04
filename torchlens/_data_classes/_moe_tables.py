"""Read saved routing tensors into dependency-free table rows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .._errors import PayloadUnavailableError
from ._moe_structure import _Route, _routes

if TYPE_CHECKING:
    from .trace import Trace


def _activation(trace: Trace, label: str) -> torch.Tensor:
    """Require a saved, raw routing output and name its exact missing site."""
    op = trace.ops[label]
    if not op.has_saved_activation:
        raise PayloadUnavailableError(
            f"Routing activation {label!r} was not saved.",
            code="moe_routing_activation_missing",
            remedy="capture with save='all' to retain router weights and expert indices",
            activation=label,
        )
    value = op.out
    if not isinstance(value, torch.Tensor):
        raise PayloadUnavailableError(
            f"Routing activation {label!r} is unavailable as a raw tensor.",
            code="moe_routing_activation_missing",
            remedy="capture with save='all' and no activation transform",
            activation=label,
        )
    return value


def _row(trace: Trace, route: _Route) -> dict[str, Any]:
    """Read one invocation, preserving the saved routing numbers verbatim."""
    indices = _activation(trace, route.indices)
    weights = _activation(trace, route.weights)
    if indices.shape == weights.shape[:-1]:
        indices = indices.unsqueeze(-1)
    selected = indices.detach().cpu().reshape(-1, indices.shape[-1]).tolist()
    gates = weights.detach().cpu().reshape(-1, weights.shape[-1]).tolist()
    counts = [sum(row.count(expert) for row in selected) for expert in range(route.expert_count)]
    assignments = sum(counts)
    return {
        "module": route.block or "self",
        "router": route.gate,
        "call": trace.ops[route.indices].func_call_id,
        "token_shape": tuple(indices.shape[:-1]),
        "expert_indices": selected,
        "gate_weights": gates,
        "tokens_per_expert": counts,
        "load": [count / assignments if assignments else 0.0 for count in counts],
    }


def _routing_table(trace: Trace, address: str | None = None) -> list[dict[str, Any]]:
    """Collect rows for all blocks, or exactly one requested module address."""
    address = "" if address == "self" else address
    return [
        _row(trace, route) for route in _routes(trace) if address is None or route.block == address
    ]
