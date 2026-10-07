"""Faithful replay of an in-forward ``torch.autograd.grad`` boundary op.

An ``autogradgrad`` op records a ``torch.autograd.grad`` call made inside a
model's forward. Its saved ``outputs`` argument is a detached snapshot, so
re-calling ``torch.autograd.grad`` on the saved arguments cannot work: the
gradient is a function of the recorded GRAPH from the op's ``inputs`` to its
``outputs``, not of the outputs' values.

This module checks the op faithfully instead of exempting it: it replays the
recorded forward subgraph from fresh leaves built from the saved ``inputs``
values to the ``outputs`` roots (each op re-executed from its own saved
arguments, with recomputed values spliced in at its parent slots; an input that
itself depends on another input stays a recomputed node), calls the
original ``torch.autograd.grad`` on that replay with the recorded
``grad_outputs`` and engine flags, and hands the result back to the ordinary
replay comparison. A wrong recorded parent, a wrong func, a tampered saved
gradient, or a subgraph that cannot be replayed all fail validation.

Helpers from ``core`` are imported lazily (``core`` imports this module).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from ..backends.torch._autograd_grad_boundary import grad_signature, is_autograd_grad_recorder
from ..utils.rng import execute_with_restored_rng_autocast

if TYPE_CHECKING:
    from ..data_classes.op import Op
    from ..data_classes.trace import Trace

_ParamLeaves = dict[int, torch.Tensor]
"""``id(live parameter) -> replay leaf`` for the parameters a replay differentiates."""


class AutogradGradReplayError(RuntimeError):
    """The recorded subgraph of an autograd.grad boundary op cannot be replayed.

    Raised only inside a replay; the replay executor turns it into a failed
    validation, so it never reaches a caller.
    """


def original_autograd_grad() -> Callable[..., Any]:
    """Return the pre-TorchLens ``torch.autograd.grad`` callable.

    Returns
    -------
    Callable[..., Any]
        The snapshot taken when the autograd wrappers were installed, else the
        current ``torch.autograd.grad``.
    """

    from ..backends.torch import backward as torch_backward

    original = torch_backward._ORIGINAL_AUTOGRAD_GRAD
    return original if original is not None else torch.autograd.grad


def _substitute_params(value: Any, mapping: _ParamLeaves) -> Any:
    """Replace live parameter leaves in a nested argument by their replay leaves."""

    if isinstance(value, torch.Tensor):
        return mapping.get(id(value), value)
    if isinstance(value, list):
        return [_substitute_params(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(_substitute_params(item, mapping) for item in value)
    if isinstance(value, dict):
        return {key: _substitute_params(item, mapping) for key, item in value.items()}
    return value


def _tensor_leaf_paths(value: Any, path: tuple[Any, ...]) -> list[tuple[tuple[Any, ...], Any]]:
    """Return ``(path, tensor)`` for every tensor leaf of a nested argument."""

    if isinstance(value, torch.Tensor):
        return [(path, value)]
    if isinstance(value, (list, tuple)):
        found: list[tuple[tuple[Any, ...], Any]] = []
        for index, item in enumerate(value):
            found.extend(_tensor_leaf_paths(item, (*path, index)))
        return found
    return []


def _write_path(container: Any, path: tuple[Any, ...], value: Any) -> Any:
    """Return ``container`` with the leaf at ``path`` replaced (tuples rebuilt)."""

    from ..utils.collections import assign_to_sequence_or_dict

    if not path:
        return value
    child = _write_path(container[path[0]], path[1:], value)
    return assign_to_sequence_or_dict(container, path[0], child)


def _read_path(container: Any, path: tuple[Any, ...]) -> Any:
    """Return the leaf of a nested argument at ``path``."""

    value = container
    for key in path:
        value = value[key]
    return value


def _slot_root(bound: inspect.BoundArguments, name: str, args_len: int) -> tuple[str, Any]:
    """Return the ``(arg_domain, key)`` holding ``name`` in the original call."""

    if name not in bound.arguments:
        raise AutogradGradReplayError(f"autograd.grad call has no {name!r} argument")
    position = list(bound.signature.parameters).index(name)
    if position < args_len:
        return ("args", position)
    return ("kwargs", name)


def _parent_slots_under(layer: Op, slot: tuple[str, Any]) -> dict[tuple[Any, ...], str]:
    """Return ``{path below slot: parent label}`` for parents recorded inside ``slot``."""

    arg_domain, root_key = slot
    found: dict[tuple[Any, ...], str] = {}
    positions = (getattr(layer, "parent_arg_positions", {}) or {}).get(arg_domain, {})
    for key, parent_label in positions.items():
        key_path = key if isinstance(key, tuple) else (key,)
        if key_path[0] == root_key:
            found[tuple(key_path[1:])] = parent_label
    return found


def _concrete_label(trace: Trace, label: str) -> str:
    """Return the concrete op label a (possibly pass-less) graph label resolves to."""

    from .core import _op_for_validation_label

    return _op_for_validation_label(trace, label).label


def _fresh_leaf(value: torch.Tensor, substitutions: _ParamLeaves, nested: bool) -> torch.Tensor:
    """Return the differentiable leaf an autograd.grad input slot replays from.

    A top-level replay always starts from a fresh leaf holding the saved value.
    A replay nested inside another one's subgraph keeps an input that already
    carries the enclosing replay's graph, so its gradients stay connected.
    """

    if id(value) in substitutions:
        return substitutions[id(value)]
    if nested and value.requires_grad:
        return value
    return value.detach().clone().requires_grad_(True)


def _perturbed_node(replayed: torch.Tensor, perturbed: torch.Tensor) -> torch.Tensor:
    """Return a replay node whose VALUE is the perturbed value, still tied to the graph.

    A gradient never reads its root's value, only the root's dependence on the
    inputs, so an additive offset would test nothing. The perturbed node is
    ``replayed * ratio + offset``: at the replay point it equals ``perturbed``
    exactly, and its derivative is scaled by ``ratio`` (``perturbed / saved``,
    or 2 where the saved value is zero), so the perturbation reaches the
    gradient through the recorded subgraph.
    """

    saved = replayed.detach()
    perturbed = perturbed.to(device=saved.device, dtype=saved.dtype)
    ratio = torch.where(saved != 0, perturbed / torch.where(saved != 0, saved, 1), 2.0)
    offset = perturbed - saved * ratio
    return replayed * ratio + offset


def _select_output(op: Op, output: Any) -> Any:
    """Pick the leaf one op represents out of its func's (possibly multi) return."""

    from .core import _slice_recomputed_output_by_path

    container_path = tuple(getattr(op, "container_path", ()) or ())
    if container_path and not isinstance(output, torch.Tensor):
        return _slice_recomputed_output_by_path(output, container_path)
    if isinstance(output, (list, tuple)):
        return output[op.multi_output_index]
    return output


class _SubgraphReplay:
    """Recompute the recorded forward subgraph between a boundary's inputs and roots.

    Parameter inputs are fresh leaves. An op input is a fresh leaf when it does
    not itself depend on another input, and otherwise a recomputed node, so a
    gradient with respect to an upstream input still flows through it (as
    ``torch.autograd.grad`` differentiates through every path).
    """

    def __init__(
        self,
        trace: Trace,
        input_values: dict[str, torch.Tensor],
        perturbed_inputs: frozenset[str],
        param_leaves: _ParamLeaves,
        nested: bool,
    ) -> None:
        self.trace = trace
        self.input_values = input_values
        self.perturbed_inputs = perturbed_inputs
        self.param_leaves = param_leaves
        self.nested = nested
        self.values: dict[str, torch.Tensor] = {}
        self.depends: dict[str, bool] = {}

    def resolve(self, label: str) -> Op:
        """Resolve a parent label to its concrete op."""

        from .core import _op_for_validation_label

        return _op_for_validation_label(self.trace, label)

    def value(self, label: str) -> torch.Tensor:
        """Return the replayed (input-connected) value of the op named ``label``."""

        op = self.resolve(label)
        self._replay_ancestors(op)
        if not self.depends.get(op.label, False):
            raise AutogradGradReplayError(
                f"autograd.grad root {op.label!r} does not depend on any recorded input"
            )
        return self.values[op.label]

    def _reads_input_param(self, op: Op) -> bool:
        """Return whether ``op`` consumed a parameter that is a boundary input."""

        for param_log in tuple(getattr(op, "_param_logs", ()) or ()):
            try:
                handle = getattr(param_log, "handle", None)
            except Exception:
                handle = None
            if handle is not None and id(handle) in self.param_leaves:
                return True
        return False

    def _replay_ancestors(self, root: Op) -> None:
        """Post-order walk from ``root``: replay every op that depends on an input."""

        stack: list[tuple[Op, bool]] = [(root, False)]
        while stack:
            op, parents_done = stack.pop()
            if op.label in self.depends:
                continue
            parents = [self.resolve(label) for label in tuple(op.parents)]
            if not parents_done:
                stack.append((op, True))
                stack.extend((parent, False) for parent in parents)
                continue
            depends = self._reads_input_param(op) or any(
                self.depends.get(parent.label, False) for parent in parents
            )
            if op.label in self.input_values:
                self.values[op.label] = self._input_node(op, depends)
                self.depends[op.label] = True
                continue
            self.depends[op.label] = depends
            if depends:
                self.values[op.label] = self._replay_op(op)

    def _input_node(self, op: Op, depends_upstream: bool) -> torch.Tensor:
        """Return the replay node of an op passed in ``inputs``."""

        slot_value = self.input_values[op.label]
        if not depends_upstream:
            return _fresh_leaf(slot_value, self.param_leaves, self.nested)
        node = self._replay_op(op)
        if op.label in self.perturbed_inputs:
            return _perturbed_node(node, slot_value)
        return node

    def _connected_value(self, label: str) -> torch.Tensor | None:
        """Return the replayed value for a parent label, or None when it is constant."""

        parent = self.resolve(label)
        if self.depends.get(parent.label, False):
            return self.values[parent.label]
        return None

    def _splice_parents(self, op: Op, input_args: dict[str, Any]) -> None:
        """Write replayed parents and parameter leaves into ``op``'s replay args."""

        is_inplace = bool(getattr(op, "is_inplace", False))
        for arg_domain in ("args", "kwargs"):
            positions = (getattr(op, "parent_arg_positions", {}) or {}).get(arg_domain, {})
            for key, parent_label in positions.items():
                replayed = self._connected_value(parent_label)
                if replayed is None:
                    continue
                if is_inplace:
                    replayed = replayed.clone()
                key_path = key if isinstance(key, tuple) else (key,)
                input_args[arg_domain] = _write_path(input_args[arg_domain], key_path, replayed)
        input_args["args"] = list(_substitute_params(list(input_args["args"]), self.param_leaves))
        input_args["kwargs"] = _substitute_params(dict(input_args["kwargs"]), self.param_leaves)

    def _replay_op(self, op: Op) -> torch.Tensor:
        """Re-execute one op from its saved arguments with replayed parents spliced in."""

        from .core import (
            _execute_func_with_restored_state,
            _prepare_input_args_for_validating_layer,
        )

        input_args, reason = _prepare_input_args_for_validating_layer(self.trace, op, [])
        if input_args is None:
            raise AutogradGradReplayError(
                f"subgraph op {op.label!r} has no saved arguments ({reason})"
            )
        self._splice_parents(op, input_args)
        if is_autograd_grad_recorder(op.func):
            # A recorded grad inside this grad's subgraph: replay it against this
            # replay's leaves so its gradients stay connected to them.
            output = _select_output(op, _replay_boundary(op, input_args, [], self.param_leaves))
        else:
            output = _execute_func_with_restored_state(op, input_args, [], op.label, False)
        if not isinstance(output, torch.Tensor):
            raise AutogradGradReplayError(f"subgraph op {op.label!r} did not replay a tensor")
        return output


def _bind_inputs(
    layer: Op,
    trace: Trace,
    inputs_value: Any,
    inputs_slot: tuple[str, Any],
    outer: _ParamLeaves | None,
) -> tuple[Any, dict[tuple[Any, ...], str], dict[str, torch.Tensor], _ParamLeaves]:
    """Split the ``inputs`` argument into op inputs and parameter leaves.

    Returns the inputs argument with fresh leaves at its non-op slots, the op
    slots (path -> op label), each op input's slot value, and the parameter
    leaves (an enclosing replay's leaves overlaid with this call's own).
    """

    substitutions: _ParamLeaves = dict(outer or {})
    op_slots = {
        path: _concrete_label(trace, label)
        for path, label in _parent_slots_under(layer, inputs_slot).items()
    }
    input_values: dict[str, torch.Tensor] = {}
    for path, tensor in _tensor_leaf_paths(inputs_value, ()):
        if path in op_slots:
            input_values[op_slots[path]] = tensor
            continue
        leaf = _fresh_leaf(tensor, substitutions, outer is not None)
        inputs_value = _write_path(inputs_value, path, leaf)
        if isinstance(tensor, torch.nn.Parameter):
            substitutions[id(tensor)] = leaf
    return inputs_value, op_slots, input_values, substitutions


def _replayed_outputs(
    replay: _SubgraphReplay,
    layer: Op,
    outputs_value: Any,
    outputs_slot: tuple[str, Any],
    perturbed: frozenset[str],
) -> Any:
    """Return the ``outputs`` argument with each recorded root replaced by its replay."""

    from ..utils.tensor_utils import tensor_nanequal

    for path, root_label in _parent_slots_under(layer, outputs_slot).items():
        label = replay.resolve(root_label).label
        replayed_root = replay.value(label)
        saved_root = _read_path(outputs_value, path)
        if label in perturbed:
            replayed_root = _perturbed_node(replayed_root, saved_root)
        elif not perturbed and not tensor_nanequal(
            replayed_root.detach(), saved_root.detach(), allow_tolerance=True
        ):
            raise AutogradGradReplayError(
                f"replayed autograd.grad root {label!r} does not reproduce its saved value"
            )
        outputs_value = _write_path(outputs_value, path, replayed_root)
    return outputs_value


def _replay_boundary(
    layer: Op,
    input_args: dict[str, Any],
    layers_to_perturb: list[str],
    outer: _ParamLeaves | None,
) -> tuple[Any, ...]:
    """Replay one autograd.grad boundary op; ``outer`` is an enclosing replay's leaves."""

    from .._state import pause_logging

    trace = layer._source_trace
    if trace is None:
        raise AutogradGradReplayError("autograd.grad boundary op has no source trace")
    args = list(input_args["args"])
    bound = grad_signature(original_autograd_grad()).bind(*args, **dict(input_args["kwargs"]))
    inputs_value, op_slots, input_values, param_leaves = _bind_inputs(
        layer, trace, bound.arguments["inputs"], _slot_root(bound, "inputs", len(args)), outer
    )
    perturbed = frozenset(_concrete_label(trace, label) for label in layers_to_perturb)
    replay = _SubgraphReplay(trace, input_values, perturbed, param_leaves, outer is not None)
    with pause_logging(), torch.enable_grad():
        bound.arguments["outputs"] = _replayed_outputs(
            replay,
            layer,
            bound.arguments["outputs"],
            _slot_root(bound, "outputs", len(args)),
            perturbed,
        )
        for path, label in op_slots.items():
            inputs_value = _write_path(inputs_value, path, replay.value(label))
        bound.arguments["inputs"] = inputs_value
        grads = original_autograd_grad()(*bound.args, **bound.kwargs)
    if outer is not None:
        # Inside an enclosing replay the gradients stay graph-connected, as the
        # recorded create_graph call's outputs were.
        return tuple(grads)
    return tuple(g.detach() if isinstance(g, torch.Tensor) else g for g in grads)


def replay_autograd_grad_boundary(
    layer: Op,
    input_args: dict[str, Any],
    layers_to_perturb: list[str],
) -> tuple[Any, ...]:
    """Recompute an autograd.grad boundary op's gradients from its recorded subgraph.

    Parameters
    ----------
    layer:
        The ``autogradgrad`` op being validated (any of its output ops).
    input_args:
        The op's prepared replay arguments (saved values with parent outputs
        swapped in, live parameters restored, perturbations applied).
    layers_to_perturb:
        Parent labels perturbed for this replay; empty for plain replay.

    Returns
    -------
    tuple[Any, ...]
        The recomputed gradients, detached, in ``torch.autograd.grad`` order.

    Raises
    ------
    AutogradGradReplayError
        If the subgraph cannot be replayed or the replayed roots disagree with
        the saved roots on a plain replay.
    """

    return _replay_boundary(layer, input_args, layers_to_perturb, None)


def execute_replay_func(layer: Op, input_args: dict[str, Any], layers_to_perturb: list[str]) -> Any:
    """Run one op's replay: its recorded func, or the autograd.grad subgraph replay.

    Parameters
    ----------
    layer:
        Op being replayed.
    input_args:
        Prepared replay arguments (``{"args": [...], "kwargs": {...}}``).
    layers_to_perturb:
        Parent labels perturbed for this replay.

    Returns
    -------
    Any
        The func's return value, restored RNG and autocast state applied.
    """

    if is_autograd_grad_recorder(layer.func):
        return replay_autograd_grad_boundary(layer, input_args, layers_to_perturb)
    with torch.set_grad_enabled(captured_grad_enabled(layer)):
        return execute_with_restored_rng_autocast(
            layer.func,
            tuple(input_args["args"]),
            dict(input_args["kwargs"]),
            rng_states=layer.func_rng_states,
            autocast_state=layer.func_autocast_state,
        )


def captured_grad_enabled(layer: Op) -> bool:
    """Return the grad mode the op's original call ran under.

    Capture records ``torch.is_grad_enabled()`` per op under the reserved
    ``"__execution__"`` key of ``func_autocast_state``. Replay must run under
    that mode: ATen's backend choice can depend on it (on macOS arm64 a
    depthwise 3x3 conv picks ``Slow2d`` with grad enabled and
    ``Winograd3x3Depthwise`` without), so replaying under a different mode can
    call a different kernel than the captured run did.

    Parameters
    ----------
    layer:
        Op being replayed.

    Returns
    -------
    bool
        The recorded grad mode, or the caller's current grad mode when the op
        carries no execution record (source ops and synthesized records).
    """

    autocast_state = getattr(layer, "func_autocast_state", None)
    execution = autocast_state.get("__execution__") if isinstance(autocast_state, dict) else None
    recorded = execution.get("grad_enabled") if isinstance(execution, dict) else None
    if isinstance(recorded, bool):
        return recorded
    return bool(torch.is_grad_enabled())
