"""Conservative read-time proof of gate, selection and expert dataflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from ..intervention.types import CapturedArgTemplate, LiteralValue
from ._moe_layouts import _Layout, _layouts
from ._runtime_handles import source_model_from_trace

if TYPE_CHECKING:
    from .op import Op
    from .trace import Trace


@dataclass(frozen=True)
class _Route:
    """One gate call's two saved selection outputs and structural identity."""

    block: str
    gate: str
    experts: tuple[str, ...]
    expert_count: int
    weights: str
    indices: str
    routers: frozenset[str]


def _inside(op: Op, address: str) -> bool:
    """Test captured module-call membership, never a substring match."""
    return not address or any(label.rsplit(":", 1)[0] == address for label in op.modules)


def _ancestors(ops: dict[str, Op], label: str) -> set[str]:
    """Walk recorded parents, including the starting op, without recursion."""
    seen: set[str] = set()
    pending = [label]
    while pending:
        current = pending.pop()
        if current in seen or current not in ops:
            continue
        seen.add(current)
        pending.extend(ops[current].parents)
    return seen


def _last_axis(op: Op, position: int, default: int, rank: int | None = None) -> bool:
    """Require a statically recorded expert-axis reduction/selection."""
    template = op.args_template
    if not isinstance(template, CapturedArgTemplate):
        return False
    args = template.args
    kwargs = dict(template.kwargs)
    dim = kwargs.get("dim", args[position] if len(args) > position else LiteralValue(default))
    dim = dim.value if isinstance(dim, LiteralValue) else None
    rank = len(op.shape) if rank is None else rank
    return isinstance(dim, int) and rank > 0 and -rank <= dim < rank and dim % rank == rank - 1


def _selected_outputs(ops: dict[str, Op]) -> list[tuple[Op, Op]]:
    """Pair top-k outputs or top-one softmax gathers by recorded dependency."""
    calls: dict[Any, dict[int, Op]] = {}
    for op in ops.values():
        if op.func_name == "topk" and _last_axis(op, 2, -1):
            calls.setdefault(op.func_call_id, {})[op.multi_output_index] = op
    pairs = [(rows[0], rows[1]) for rows in calls.values() if 0 in rows and 1 in rows]
    for indices in ops.values():
        input_rank = max(
            (len(ops[label].shape) for label in indices.parents if label in ops), default=0
        )
        if indices.func_name != "argmax" or not _last_axis(indices, 1, -1, input_rank):
            continue
        for values in ops.values():
            if (
                values.func_name == "gather"
                and _last_axis(values, 1, -1)
                and indices.layer_label in _ancestors(ops, values.layer_label)
            ):
                pairs.append((values, indices))
    return pairs


def _weight_output(ops: dict[str, Op], values: Op) -> str | None:
    """Recognize softmax-before-topk or softmax of selected logits."""
    for label in values.parents:
        parent = ops.get(label)
        if parent is not None and parent.func_name == "softmax" and _last_axis(parent, 1, -1):
            return values.layer_label
    for op in ops.values():
        if op.func_name == "softmax" and _last_axis(op, 1, -1) and values.layer_label in op.parents:
            return op.layer_label
    return None


def _routes(trace: Trace) -> list[_Route]:
    """Prove routing solely on demand; never register capture hooks or fields."""
    model = source_model_from_trace(trace)
    if not isinstance(model, nn.Module):
        return []
    containers = _layouts(model)
    if not containers:
        return []
    ops = {op.layer_label: op for op in trace.ops}
    routes = []
    for values, indices in _selected_outputs(ops):
        weights = _weight_output(ops, values)
        if weights is None:
            continue
        upstream = _ancestors(ops, values.layer_label)
        for layout in containers:
            if not _inside(values, layout.block):
                continue
            for gate in layout.gates:
                gate_ops = {label for label in upstream if _inside(ops[label], gate)}
                if (
                    not gate_ops
                    or not _expert_width(ops, values, layout.count)
                    or not _feeds_experts(ops, indices, layout)
                ):
                    continue
                routers = gate_ops | {values.layer_label, indices.layer_label, weights}
                routers.update(
                    label
                    for label in upstream
                    if any(seed in _ancestors(ops, label) for seed in gate_ops)
                )
                routes.append(
                    _Route(
                        layout.block,
                        gate,
                        layout.experts,
                        layout.count,
                        weights,
                        indices.layer_label,
                        frozenset(routers),
                    )
                )
    return routes


def _expert_width(ops: dict[str, Op], values: Op, count: int) -> bool:
    """Require the selection's probability/logit input to span the expert axis."""
    return any(
        ops[label].shape and ops[label].shape[-1] == count
        for label in values.parents
        if label in ops
    )


def _feeds_experts(ops: dict[str, Op], indices: Op, layout: _Layout) -> bool:
    """Require recorded selection-index dependence inside each called expert."""
    called = []
    block_calls = {label for label in indices.modules if label.rsplit(":", 1)[0] == layout.block}
    for expert in layout.experts:
        members = [
            op
            for op in ops.values()
            if _inside(op, expert) and (not layout.block or block_calls.intersection(op.modules))
        ]
        if members:
            called.append(
                any(indices.layer_label in _ancestors(ops, op.layer_label) for op in members)
            )
    return (
        all(called)
        if called
        else _feeds_fused(ops, indices, layout.parameter_prefixes, block_calls)
    )


def _feeds_fused(
    ops: dict[str, Op], indices: Op, prefixes: tuple[str, ...], block_calls: set[str]
) -> bool:
    """Recognize batched expert math only with indexed expert-parameter ancestry."""
    for op in ops.values():
        if op.func_name not in {"bmm", "matmul", "einsum"} or (
            block_calls and not block_calls.intersection(op.modules)
        ):
            continue
        upstream = _ancestors(ops, op.layer_label)
        if indices.layer_label not in upstream:
            continue
        if _indexed_parameters(ops, upstream, indices.layer_label, prefixes):
            return True
    return False


def _indexed_parameters(
    ops: dict[str, Op], labels: set[str], indices: str, prefixes: tuple[str, ...]
) -> bool:
    """Prove that the routing indices select expert parameters before batched math."""
    for label in labels:
        if ops[label].func_name not in {"__getitem__", "getitem", "gather", "index_select"}:
            continue
        upstream = _ancestors(ops, label)
        if indices not in upstream:
            continue
        addresses = {param.address for ancestor in upstream for param in ops[ancestor].params}
        if all(any(address.startswith(prefix) for address in addresses) for prefix in prefixes):
            return True
    return False


def _identity(op: Op) -> tuple[bool, int | None]:
    """Derive the two identity facts without reading any activation."""
    routes = _routes(op.source_trace)
    router = any(op.layer_label in route.routers for route in routes)
    for route in routes:
        for index, address in enumerate(route.experts):
            if _inside(op, address):
                return router, index
    return router, None
