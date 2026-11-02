"""Faithful replay of the in-forward ``torch.autograd.backward`` boundary op.

An ``autogradbackward`` op records one leaf gradient written by a
``torch.autograd.backward`` (or ``Tensor.backward``) call made inside a model's
forward. Its saved roots are detached snapshots, so the gradient cannot be
re-derived from the saved arguments alone: it is a function of the recorded
GRAPH from the op's leaf (``inputs``) to its roots (``tensors``).

This module checks the op the way ``_autograd_grad_replay`` checks an
``autogradgrad`` op: it replays the recorded forward subgraph from a fresh
leaf holding the saved leaf value to the roots, calls the original
``torch.autograd.grad`` on that replay with the recorded ``grad_tensors``, adds
the previous grad for an accumulating pass (the op's in-place receiver, so a
perturbed previous grad reaches the result), and hands the gradient back to the
ordinary replay comparison. A wrong recorded parent, a wrong func, a tampered
saved gradient, or a subgraph that cannot be replayed all fail validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ..backends.torch._autograd_backward_boundary import recorder_signature
from ._autograd_grad_replay import (
    AutogradGradReplayError,
    _bind_inputs,
    _concrete_label,
    _ParamLeaves,
    _replayed_outputs,
    _slot_root,
    _SubgraphReplay,
    _write_path,
    original_autograd_grad,
)

if TYPE_CHECKING:
    from .._data_classes.op import Op


def _replay_backward_boundary(
    layer: Op,
    input_args: dict[str, Any],
    layers_to_perturb: list[str],
    outer: _ParamLeaves | None,
) -> torch.Tensor:
    """Replay one autograd.backward boundary op; ``outer`` is an enclosing replay's leaves.

    Parameters
    ----------
    layer:
        The ``autogradbackward`` op being replayed.
    input_args:
        Prepared replay arguments (``{"args": [...], "kwargs": {...}}``), parents
        already substituted with their saved or perturbed values.
    layers_to_perturb:
        Parent labels perturbed for this replay.
    outer:
        An enclosing subgraph replay's parameter leaves, or ``None`` at top level.

    Returns
    -------
    torch.Tensor
        The re-derived gradient of the op's leaf, accumulated onto the previous
        grad when the recorded pass accumulated.
    """

    from .._state import pause_logging

    trace = layer._source_trace
    if trace is None:
        raise AutogradGradReplayError("autograd.backward boundary op has no source trace")
    args = list(input_args["args"])
    bound = recorder_signature().bind(*args, **dict(input_args["kwargs"]))
    accumulate_into = bound.arguments.get("accumulate_into")
    inputs_value, op_slots, input_values, param_leaves = _bind_inputs(
        layer, trace, bound.arguments["inputs"], _slot_root(bound, "inputs", len(args)), outer
    )
    perturbed = frozenset(_concrete_label(trace, label) for label in layers_to_perturb)
    replay = _SubgraphReplay(trace, input_values, perturbed, param_leaves, outer is not None)
    with pause_logging(), torch.enable_grad():
        roots = _replayed_outputs(
            replay,
            layer,
            bound.arguments["tensors"],
            _slot_root(bound, "tensors", len(args)),
            perturbed,
        )
        for path, label in op_slots.items():
            inputs_value = _write_path(inputs_value, path, replay.value(label))
        grads = original_autograd_grad()(
            roots,
            inputs_value,
            grad_outputs=bound.arguments.get("grad_tensors"),
            allow_unused=True,
        )
        grad = grads[0] if grads else None
        if grad is None:
            raise AutogradGradReplayError(
                f"autograd.backward boundary {layer.label!r}: its leaf is not reached from "
                "the replayed roots"
            )
        if outer is None:
            grad = grad.detach()
        if isinstance(accumulate_into, torch.Tensor):
            grad = grad + accumulate_into.to(device=grad.device, dtype=grad.dtype)
    return grad


def replay_autograd_backward_boundary(
    layer: Op, input_args: dict[str, Any], layers_to_perturb: list[str]
) -> torch.Tensor:
    """Replay a top-level ``autogradbackward`` op from its recorded subgraph.

    Parameters
    ----------
    layer:
        The op being replayed.
    input_args:
        Prepared replay arguments.
    layers_to_perturb:
        Parent labels perturbed for this replay.

    Returns
    -------
    torch.Tensor
        The re-derived (detached) gradient, compared by the caller to the saved one.
    """

    return _replay_backward_boundary(layer, input_args, layers_to_perturb, None)
