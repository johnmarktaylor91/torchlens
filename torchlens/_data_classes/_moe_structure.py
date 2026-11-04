"""Conservative read-time proof of gate, selection and expert dataflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch.nn as nn

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
    weights: str
    indices: str
    routers: frozenset[str]


def _inside(op: Op, address: str) -> bool:
    """Test captured module-call membership, never a substring match."""
    return not address or any(label.rsplit(":", 1)[0] == address for label in op.modules)


def _join(address: str, child: str) -> str:
    """Join a root-relative module address."""
    return f"{address}.{child}" if address else child


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


def _last_axis(op: Op, position: int, default: int) -> bool:
    """Require a statically recorded expert-axis reduction/selection."""
    args = op.args_template or ()
    kwargs = op.kwargs_template or {}
    dim = kwargs.get("dim", args[position] if len(args) > position else default)
    return isinstance(dim, int) and bool(op.shape) and dim % len(op.shape) == len(op.shape) - 1


def _containers(model: nn.Module) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Find homogeneous containers and sibling gates without naming heuristics."""
    found = []
    for address, block in model.named_modules():
        children = tuple(block.named_children())
        for name, container in children:
            if not isinstance(container, (nn.ModuleList, nn.ModuleDict)):
                continue
            entries = tuple(container.named_children())
            if len(entries) < 2 or len({type(child) for _, child in entries}) != 1:
                continue
            experts = tuple(_join(_join(address, name), key) for key, _ in entries)
            gates = tuple(
                _join(address, key)
                for key, child in children
                if child is not container and not isinstance(child, (nn.ModuleList, nn.ModuleDict))
            )
            found.append((address, experts, gates))
    return found


def _selected_outputs(ops: dict[str, Op]) -> list[tuple[Op, Op]]:
    """Pair top-k outputs or top-one softmax gathers by recorded dependency."""
    calls: dict[Any, dict[int, Op]] = {}
    for op in ops.values():
        if op.func_name == "topk" and _last_axis(op, 2, -1):
            calls.setdefault(op.func_call_id, {})[op.multi_output_index] = op
    pairs = [(rows[0], rows[1]) for rows in calls.values() if 0 in rows and 1 in rows]
    for indices in ops.values():
        if indices.func_name != "argmax" or not _last_axis(indices, 1, -1):
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
    containers = _containers(model)
    if not containers:
        return []
    ops = {op.layer_label: op for op in trace.ops}
    routes = []
    for values, indices in _selected_outputs(ops):
        weights = _weight_output(ops, values)
        if weights is None:
            continue
        upstream = _ancestors(ops, values.layer_label)
        for block, experts, gates in containers:
            if not _inside(values, block):
                continue
            for gate in gates:
                gate_ops = {label for label in upstream if _inside(ops[label], gate)}
                if (
                    not gate_ops
                    or not _expert_width(ops, values, len(experts))
                    or not _feeds_experts(ops, indices, experts, block)
                ):
                    continue
                routers = gate_ops | {values.layer_label, indices.layer_label, weights}
                routers.update(
                    label
                    for label in upstream
                    if any(seed in _ancestors(ops, label) for seed in gate_ops)
                )
                routes.append(
                    _Route(block, gate, experts, weights, indices.layer_label, frozenset(routers))
                )
    return routes


def _expert_width(ops: dict[str, Op], values: Op, count: int) -> bool:
    """Require the selection's probability/logit input to span the expert axis."""
    return any(
        ops[label].shape and ops[label].shape[-1] == count
        for label in values.parents
        if label in ops
    )


def _feeds_experts(ops: dict[str, Op], indices: Op, experts: tuple[str, ...], block: str) -> bool:
    """Require recorded selection-index dependence inside each called expert."""
    called = []
    block_calls = {label for label in indices.modules if label.rsplit(":", 1)[0] == block}
    for expert in experts:
        members = [
            op
            for op in ops.values()
            if _inside(op, expert) and (not block or block_calls.intersection(op.modules))
        ]
        if members:
            called.append(
                any(indices.layer_label in _ancestors(ops, op.layer_label) for op in members)
            )
    return all(called) if called else _feeds_fused(ops, indices, experts, block_calls)


def _feeds_fused(
    ops: dict[str, Op], indices: Op, experts: tuple[str, ...], block_calls: set[str]
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
        addresses = {param.address for label in upstream for param in ops[label].params}
        if all(
            any(address.startswith(expert + ".") for address in addresses) for expert in experts
        ):
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
