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

import heapq
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch

from .._state import pause_logging
from .._utils._raw_pointer_denial import refuse_pointer_keywords
from .._utils._record_group_table import record_member
from .._utils._tensor_layout import CaptureLayout, layout_of, relay_to_layout
from .._utils.collections import read_at_path, replace_at_path
from .._utils.rng import execute_with_restored_rng_autocast
from ..backends.torch._autograd_grad_boundary import grad_signature, recorder_signature
from ..backends.torch._autograd_grad_marker import (
    is_autograd_backward_recorder,
    is_autograd_grad_recorder,
)
from ._replay_device import align_parent_to_slot_device, saved_args_on_consumed_devices
from ._replay_storage_context import (
    alias_provenance_holds,
    alias_value_holds,
    check_view_context_receiver,
    rebuilt_alias,
    receiver_lineage,
    replayed_storage_alias,
    slot_capture_layout,
    storage_key,
    writes_through_view,
)

if TYPE_CHECKING:
    from .._data_classes.op import Op
    from .._data_classes.trace import Trace

_ParamLeaves = dict[int, torch.Tensor]


class _Enclosing(NamedTuple):
    """What a nested replay inherits from the replay whose subgraph holds it."""

    param_leaves: _ParamLeaves
    perturbed: bool
    storages: frozenset[tuple[torch.device, int]]


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
    positions = (record_member(layer, "parent_arg_positions", {}) or {}).get(arg_domain, {})
    for key, parent_label in positions.items():
        key_path = key if isinstance(key, tuple) else (key,)
        if key_path[0] == root_key:
            found[tuple(key_path[1:])] = parent_label
    return found


def _concrete_label(trace: Trace, label: str) -> str:
    """Return the concrete op label a (possibly pass-less) graph label resolves to."""

    from ._core import _op_for_validation_label  # deferred: cycle validation

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

    from ._core import _slice_recomputed_output_by_path  # deferred: cycle validation

    container_path = tuple(getattr(op, "container_path", ()) or ())
    if container_path and not isinstance(output, torch.Tensor):
        return _slice_recomputed_output_by_path(output, container_path)
    if isinstance(output, (list, tuple)):
        return output[op.multi_output_index]
    return output


def _capture_index_key(op: Op) -> tuple[int, str]:
    """Return an op's place in capture's event order, ``-1`` when it has no index."""

    index = record_member(op, "raw_index", None)
    return (index if isinstance(index, int) else -1, op.label)


def _recorded_order(pending: dict[str, Op], resolve: Callable[[str], Op]) -> list[Op]:
    """Return ``pending`` ops parents-first, in capture's execution order where it allows.

    The order key is the op's capture event index (``raw_index``). A parent's
    event always precedes its children's, so sorting by it alone is already
    topological; the ready-set walk keeps parents first even for an op without a
    usable index.
    """

    waiting = {
        label: {parent.label for parent in map(resolve, tuple(op.parents))} & pending.keys()
        for label, op in pending.items()
    }
    children: dict[str, list[str]] = {label: [] for label in pending}
    for label, parents in waiting.items():
        for parent in parents:
            children[parent].append(label)
    ready = [
        (_capture_index_key(pending[label]), label)
        for label, parents in waiting.items()
        if not parents
    ]
    heapq.heapify(ready)
    ordered: list[Op] = []
    while ready:
        _, label = heapq.heappop(ready)
        ordered.append(pending[label])
        for child in children[label]:
            waiting[child].discard(label)
            if not waiting[child]:
                heapq.heappush(ready, (_capture_index_key(pending[child]), child))
    return ordered


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
        enclosing: _Enclosing | None,
    ) -> None:
        self.trace = trace
        self.input_values = input_values
        self.perturbed_inputs = perturbed_inputs
        self.param_leaves = param_leaves
        self.nested = enclosing is not None
        # Values equal capture's only when neither this replay nor an enclosing one
        # is perturbed; otherwise an alias is proven by its storage provenance.
        self.perturbed_run = bool(perturbed_inputs) or bool(enclosing and enclosing.perturbed)
        self.enclosing_storages = enclosing.storages if enclosing is not None else frozenset()
        self.values: dict[str, torch.Tensor] = {}
        self.depends: dict[str, bool] = {}
        self.unperturbed: dict[str, torch.Tensor] = {}

    def resolve(self, label: str) -> Op:
        """Resolve a parent label to its concrete op."""

        from ._core import _op_for_validation_label  # deferred: cycle validation

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

        for param_log in tuple(op._param_logs or ()):
            try:
                handle = getattr(param_log, "handle", None)
            except Exception:
                handle = None
            if handle is not None and id(handle) in self.param_leaves:
                return True
        return False

    def replay_roots(self, labels: list[str]) -> None:
        """Replay every op the roots named by ``labels`` depend on, all in one ordered pass."""

        self._replay_ancestors(*(self.resolve(label) for label in labels))

    def _replay_ancestors(self, *roots: Op) -> None:
        """Replay the not-yet-replayed ancestors of ``roots`` in the recorded execution order.

        Autograd's engine runs ready backward nodes newest first (by creation
        sequence number), and the gradient contributions reaching one node are
        summed in the order its consumers' backward nodes run. Building the
        replayed subgraph in any other order sums the same terms in another order,
        which moves a gradient made of cancelling terms by a rounding step. So the
        subgraph is rebuilt in the order capture executed it (see
        :func:`_recorded_order`).
        """

        pending: dict[str, Op] = {}
        stack = list(roots)
        while stack:
            op = stack.pop()
            if op.label in self.depends or op.label in pending:
                continue
            pending[op.label] = op
            stack.extend(self.resolve(label) for label in tuple(op.parents))
        for op in _recorded_order(pending, self.resolve):
            self._replay_one(op)

    def _replay_one(self, op: Op) -> None:
        """Replay one op whose parents are all replayed or known constant."""

        parents = [self.resolve(label) for label in tuple(op.parents)]
        depends = self._reads_input_param(op) or any(
            self.depends.get(parent.label, False) for parent in parents
        )
        if op.label in self.input_values:
            self.values[op.label] = self._input_node(op, depends)
            self.depends[op.label] = True
            return
        self.depends[op.label] = depends
        if depends:
            self.values[op.label] = self._replay_op(op)

    def grad_input(self, label: str) -> torch.Tensor:
        """Return the tensor the ``inputs`` slot of op input ``label`` differentiates against.

        For a perturbed input this is its unperturbed node, upstream of the
        perturbed node the subgraph consumes, so the gradient carries the
        perturbation's derivative scaling (see ``_input_node``).
        """

        value = self.value(label)
        return self.unperturbed.get(self.resolve(label).label, value)

    def _input_node(self, op: Op, depends_upstream: bool) -> torch.Tensor:
        """Return the replay node of an op passed in ``inputs``.

        A perturbed input is replayed as ``_perturbed_node`` over its unperturbed
        node, and the gradient is taken against that unperturbed node. Taking it
        against the perturbed value itself would leave a gradient that does not
        read its input's value (any root linear in the input) unchanged, so the
        perturbation could never reach the boundary's output.
        """

        slot_value = self.input_values[op.label]
        perturbed = op.label in self.perturbed_inputs
        if depends_upstream:
            node = self._replay_op(op)
        elif perturbed and isinstance(op.out, torch.Tensor):
            saved = op.out.to(device=slot_value.device, dtype=slot_value.dtype)
            node = _fresh_leaf(saved, self.param_leaves, self.nested)
        else:
            return _fresh_leaf(slot_value, self.param_leaves, self.nested)
        if not perturbed:
            return node
        self.unperturbed[op.label] = node
        return _perturbed_node(node, slot_value)

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
            positions = (record_member(op, "parent_arg_positions", {}) or {}).get(arg_domain, {})
            for key, parent_label in positions.items():
                replayed = self._connected_value(parent_label)
                if replayed is None:
                    continue
                layout = slot_capture_layout(op, arg_domain, key)
                key_path = key if isinstance(key, tuple) else (key,)
                saved = read_at_path(input_args[arg_domain], key_path)
                replayed = self._written_alias(parent_label, replayed, (saved, layout))
                slot = (arg_domain, key)
                if is_inplace and not writes_through_view(slot, replayed, self.enclosing_storages):
                    # Only a write through a view is replayed in place, as capture
                    # wrote it, so it reaches the replayed base; every other in-place
                    # slot (a replay leaf would raise and be corrupted) gets a clone.
                    replayed = replayed.clone()
                replayed = align_parent_to_slot_device(input_args, arg_domain, key, replayed)
                # Differentiable copy into the recorded layout (grad mode is on here).
                replayed = cast(torch.Tensor, relay_to_layout(replayed, layout))
                input_args[arg_domain] = replace_at_path(input_args[arg_domain], key_path, replayed)
        input_args["args"] = list(_substitute_params(list(input_args["args"]), self.param_leaves))
        input_args["kwargs"] = _substitute_params(dict(input_args["kwargs"]), self.param_leaves)

    def _written_alias(
        self,
        parent_label: str,
        replayed: torch.Tensor,
        slot: tuple[Any, CaptureLayout | None],
    ) -> torch.Tensor:
        """Return the replayed tensor a slot read when its parent wrote through a view.

        An in-place op on a view (``y[:, 5:6].add_(v)``) advances the label of every
        storage alias it wrote through, so a later reader of the base ``y`` records the
        in-place op as its parent while it read the whole post-write base, not the
        op's view-shaped output. ``slot`` holds the slot's saved value (its shape is
        the shape read) and its recorded layout (``None`` for a row-major contiguous
        read). The replay writes in place as capture did, so the alias the slot read
        is a replayed tensor on the same storage with that geometry, searched along
        the in-place op's receiver lineage first (``replayed_storage_alias``). An
        alias the replay never built (a sibling view read only after the write) is
        rebuilt on the written storage when its full layout was recorded and the
        written view sits where capture's did. Otherwise the parent's output stands,
        so it serves a read of another layout only. The alias must hold the slot's
        saved value; a perturbed replay proves it by storage provenance instead (the
        unperturbed replay, which validation always runs first, proves the value).

        Raises
        ------
        AutogradGradReplayError
            When the alias does not hold what the reader read, or no alias has the
            shape it read.
        """

        saved, layout = slot
        parent = self.resolve(parent_label)
        if not (bool(getattr(parent, "is_inplace", False)) and isinstance(saved, torch.Tensor)):
            return replayed
        shape = tuple(saved.shape)
        if tuple(replayed.shape) == shape and layout in (None, layout_of(replayed)):
            return replayed
        lineage = [self.values.get(label) for label in receiver_lineage(parent, self.resolve)]
        others = [value for value in self.values.values() if value.requires_grad]
        found = replayed_storage_alias(replayed, (shape, layout), lineage, others)
        if found is None:
            found = rebuilt_alias(replayed, layout, slot_capture_layout(parent, "args", 0))
        if found is None and tuple(replayed.shape) == shape:
            return replayed  # another layout: relay_to_layout copies it into the recorded one
        if found is None:
            raise AutogradGradReplayError(f"no alias after {parent.label!r} has the shape read")
        if self.perturbed_run:
            proven = alias_provenance_holds(found, replayed, (shape, layout))
        else:
            proven = alias_value_holds(found, saved)
        if not proven:
            raise AutogradGradReplayError(f"the alias after {parent.label!r} differs from the read")
        return found

    def _enclosing(self) -> _Enclosing:
        """Return what a replay nested in this one's subgraph inherits."""

        owned = [
            value
            for value in (*self.values.values(), *self.param_leaves.values())
            if value.layout == torch.strided
        ]
        storages = self.enclosing_storages | {storage_key(value) for value in owned}
        return _Enclosing(self.param_leaves, self.perturbed_run, frozenset(storages))

    def _replay_op(self, op: Op) -> torch.Tensor:
        """Re-execute one op from its saved arguments with replayed parents spliced in."""

        from ._core import (  # deferred: cycle validation
            _execute_func_with_restored_state,
            _prepare_input_args_for_validating_layer,
        )

        with saved_args_on_consumed_devices(self.trace, op.label):
            input_args, reason = _prepare_input_args_for_validating_layer(self.trace, op, [])
            if input_args is None:
                raise AutogradGradReplayError(
                    f"subgraph op {op.label!r} has no saved arguments ({reason})"
                )
            self._splice_parents(op, input_args)
            if is_autograd_grad_recorder(op.func):
                # A recorded grad inside this grad's subgraph: replay it against this
                # replay's leaves so its gradients stay connected to them.
                output = _select_output(op, _replay_boundary(op, input_args, [], self._enclosing()))
            elif is_autograd_backward_recorder(op.func):
                output = _replay_backward_boundary(op, input_args, [], self._enclosing())
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
        inputs_value = replace_at_path(inputs_value, path, leaf)
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

    from .._utils.tensor_utils import tensor_nanequal

    root_slots = _parent_slots_under(layer, outputs_slot)
    replay.replay_roots(list(root_slots.values()))
    for path, root_label in root_slots.items():
        label = replay.resolve(root_label).label
        replayed_root = replay.value(label)
        saved_root = read_at_path(outputs_value, path)
        if label in perturbed:
            replayed_root = _perturbed_node(replayed_root, saved_root)
        elif not perturbed and not tensor_nanequal(
            replayed_root.detach(), saved_root.detach(), allow_tolerance=True
        ):
            raise AutogradGradReplayError(
                f"replayed autograd.grad root {label!r} does not reproduce its saved value"
            )
        outputs_value = replace_at_path(outputs_value, path, replayed_root)
    return outputs_value


def _boundary_source_trace(layer: Op) -> Trace:
    """Return the trace a boundary op (``autogradgrad`` or ``autogradbackward``) replays in."""

    trace = layer._source_trace
    if trace is None:
        raise AutogradGradReplayError(f"{layer.func_name} boundary op has no source trace")
    return trace


def _replay_boundary(
    layer: Op,
    input_args: dict[str, Any],
    layers_to_perturb: list[str],
    outer: _Enclosing | None,
) -> tuple[Any, ...]:
    """Replay one autograd.grad boundary op; ``outer`` describes an enclosing replay."""
    trace = _boundary_source_trace(layer)
    args = list(input_args["args"])
    bound = grad_signature(original_autograd_grad()).bind(*args, **dict(input_args["kwargs"]))
    inputs_value, op_slots, input_values, param_leaves = _bind_inputs(
        layer,
        trace,
        bound.arguments["inputs"],
        _slot_root(bound, "inputs", len(args)),
        outer.param_leaves if outer is not None else None,
    )
    perturbed = frozenset(_concrete_label(trace, label) for label in layers_to_perturb)
    replay = _SubgraphReplay(trace, input_values, perturbed, param_leaves, outer)
    with pause_logging(), torch.enable_grad():
        bound.arguments["outputs"] = _replayed_outputs(
            replay,
            layer,
            bound.arguments["outputs"],
            _slot_root(bound, "outputs", len(args)),
            perturbed,
        )
        for path, label in op_slots.items():
            inputs_value = replace_at_path(inputs_value, path, replay.grad_input(label))
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
    """Run one op's replay: its recorded func, or a boundary op's subgraph replay.

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

    refuse_pointer_keywords(layer, input_args["kwargs"])
    if is_autograd_grad_recorder(layer.func):
        # An autograd.grad call cannot have run under inference_mode, and an
        # ambient one would leave the rebuilt subgraph without autograd graph.
        with torch.inference_mode(False):
            return replay_autograd_grad_boundary(layer, input_args, layers_to_perturb)
    if is_autograd_backward_recorder(layer.func):
        with torch.inference_mode(False):
            return _replay_backward_boundary(layer, input_args, layers_to_perturb, None)
    check_view_context_receiver(layer, input_args)
    grad_enabled, inference_mode = captured_grad_modes(layer)
    with torch.inference_mode(inference_mode), torch.set_grad_enabled(grad_enabled):
        return execute_with_restored_rng_autocast(
            layer.func,
            tuple(input_args["args"]),
            dict(input_args["kwargs"]),
            rng_states=layer.call.rng_states,
            autocast_state=layer.call.autocast_state,
        )


def captured_grad_modes(layer: Op) -> tuple[bool, bool]:
    """Return the ``(grad_enabled, inference_mode)`` the op's original call ran under.

    Capture records both per op under the reserved ``"__execution__"`` key of
    ``func_autocast_state``. Replay must run under them: ATen's backend choice
    can depend on grad mode (on macOS arm64 a depthwise 3x3 conv picks
    ``Slow2d`` with grad enabled and ``Winograd3x3Depthwise`` without), so
    replaying under a different mode can call a different kernel than the
    captured run did.

    Parameters
    ----------
    layer:
        Op being replayed.

    Returns
    -------
    tuple[bool, bool]
        The recorded modes; each falls back to the caller's current mode when
        the op carries no execution record (source ops, synthesized records).
    """

    autocast_state = record_member(layer, "func_autocast_state", None)
    execution = autocast_state.get("__execution__") if isinstance(autocast_state, dict) else None
    record = execution if isinstance(execution, dict) else {}
    grad_enabled = record.get("grad_enabled")
    inference_mode = record.get("inference_mode")
    return (
        grad_enabled if isinstance(grad_enabled, bool) else bool(torch.is_grad_enabled()),
        (
            inference_mode
            if isinstance(inference_mode, bool)
            else bool(torch.is_inference_mode_enabled())
        ),
    )


def _replay_backward_boundary(
    layer: Op,
    input_args: dict[str, Any],
    layers_to_perturb: list[str],
    outer: _Enclosing | None,
) -> torch.Tensor:
    """Replay one ``autogradbackward`` op; ``outer`` describes an enclosing replay.

    The op's saved roots are detached snapshots, so its leaf gradient is a function
    of the recorded GRAPH from the leaf (``inputs``) to the roots (``tensors``): the
    forward subgraph is replayed from a fresh leaf holding the saved leaf value, the
    original ``torch.autograd.grad`` runs on that replay with the recorded
    ``grad_tensors``, and an accumulating pass adds the previous grad (the op's
    in-place receiver, so a perturbed previous grad reaches the result). A wrong
    parent, a wrong func, a tampered saved gradient or an unreplayable subgraph fails.

    Parameters
    ----------
    layer:
        The ``autogradbackward`` op being replayed.
    input_args:
        Prepared replay arguments, parents already substituted with saved or perturbed values.
    layers_to_perturb:
        Parent labels perturbed for this replay.
    outer:
        An enclosing replay's leaves, perturbation state and storages, or ``None`` at
        top level (result detached).

    Returns
    -------
    torch.Tensor
        The re-derived leaf gradient, accumulated onto the previous grad when the pass did.
    """

    trace = _boundary_source_trace(layer)
    args = list(input_args["args"])
    bound = recorder_signature().bind(*args, **dict(input_args["kwargs"]))
    inputs_value, op_slots, input_values, param_leaves = _bind_inputs(
        layer,
        trace,
        bound.arguments["inputs"],
        _slot_root(bound, "inputs", len(args)),
        outer.param_leaves if outer is not None else None,
    )
    perturbed = frozenset(_concrete_label(trace, label) for label in layers_to_perturb)
    replay = _SubgraphReplay(trace, input_values, perturbed, param_leaves, outer)
    with pause_logging(), torch.enable_grad():
        roots = _replayed_outputs(
            replay,
            layer,
            bound.arguments["tensors"],
            _slot_root(bound, "tensors", len(args)),
            perturbed,
        )
        for path, label in op_slots.items():
            inputs_value = replace_at_path(inputs_value, path, replay.value(label))
        grad = _replayed_leaf_grad(roots, inputs_value, bound.arguments.get("grad_tensors"))
        if outer is None:
            grad = grad.detach()
        accumulate_into = bound.arguments.get("accumulate_into")
        if isinstance(accumulate_into, torch.Tensor):
            grad = grad + accumulate_into.to(device=grad.device, dtype=grad.dtype)
    return grad


def _replayed_leaf_grad(roots: Any, leaf: Any, grad_outputs: Any) -> torch.Tensor:
    """Return the gradient of ``roots`` with respect to the replayed ``leaf``.

    A leaf the replayed roots do not reach raises torch's own unused-input error
    (``allow_unused=False``), which fails the replay like any other replay error.
    """

    return original_autograd_grad()(roots, leaf, grad_outputs=grad_outputs)[0]
