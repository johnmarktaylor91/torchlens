"""Read a trace's recorded module calls into a backend-neutral form for pairing.

Every backend records module calls as ``ModuleCall`` records (address, call index,
parent call, op labels, recorded output ops). Backends differ in two spellings that this
reader absorbs, without changing capture: a top-level call's ``call_parent`` is the root
call (``"self:1"``: torch for calls it lists as top level, preview backends for all) or
unset, read alike as the root; and never-called container segments (a torch ``ModuleList``,
a plain MLX list) appear in addresses as ``layers.0`` on one side and ``0`` on the other.

A call that returns its input unchanged (an identity module, dropout in eval mode) is an
op inside the call on torch, which inserts an identity op at module exit, but a recorded
passthrough on the preview backends: the returned input's op is flagged as the call's
output and stays out of ``output_ops``. The reader appends those passthrough ops after
the call's own outputs, the order torch's exit-time identity ops take, so both spellings
give the same output roster. Two more reads fill what preview backends do not record:
the root call's outputs are the trace's outputs (on every backend, in the trace's output
order), its inputs without recorded input ops are the trace's inputs, and a call with no
recorded ``input_ops`` takes as inputs the ops outside it that feed its own ops, the
producers torch records as ``input_ops``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._data_classes._module_passthrough import passthrough_op_labels
from .._fast_live_plans import _module_output_paths
from ._common import _op_from_label, _op_label, _ordered_ops

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

ROOT_ADDRESS = "self"

#: Plain Python containers: a preview backend records one holding sub-modules as a
#: module (with a call, since its address prefixes its children's), but it is never
#: called, like a torch ``ModuleList``.
_CONTAINER_CLASSES = frozenset({"list", "tuple", "dict"})


@dataclass(frozen=True)
class CallInfo:
    """One recorded module call, read for cross-trace pairing.

    Parameters
    ----------
    label:
        Pass-qualified call label (``"layers.0:1"``).
    address, call_index, class_name:
        As recorded on the ``ModuleCall``.
    parent:
        Label of the calling module call (the root call for top-level calls), or
        ``None`` for a root call.
    enter, exit:
        Execution positions of the call's first and last op (exit orders rows).
    depth:
        Nesting depth under the root (root is 0).
    outputs, inputs:
        Recorded output and input op labels.
    training:
        The module's recorded training flag, or ``None`` when unrecorded.
    """

    label: str
    address: str
    call_index: int
    class_name: str
    parent: str | None
    enter: int
    exit: int
    depth: int
    outputs: tuple[str, ...]
    inputs: tuple[str, ...]
    training: bool | None
    ordered: bool = True


def _raw_calls(trace: Trace) -> list[tuple[Any, Any]]:
    """Return ``(module, call)`` pairs for every recorded module call."""

    return [(module, call) for module in trace._module_logs for call in module.calls.values()]


def _drop_containers(
    raw: list[tuple[Any, Any]], parents: dict[str, str | None]
) -> tuple[list[tuple[Any, Any]], dict[str, str | None]]:
    """Drop calls recorded for plain containers; their children take the container's parent."""

    containers = {
        str(call.label)
        for module, call in raw
        if (getattr(call, "class_name", "") or getattr(module, "class_name", ""))
        in _CONTAINER_CLASSES
    }

    def lift(label: str | None) -> str | None:
        """The nearest non-container ancestor (bounded by the call count)."""

        steps = 0
        while label in containers and steps <= len(parents):
            label, steps = parents.get(str(label)), steps + 1
        return label

    kept = [(module, call) for module, call in raw if str(call.label) not in containers]
    return kept, {
        label: lift(parent) for label, parent in parents.items() if label not in containers
    }


def _root_parent_of(label: str, roots: list[Any]) -> str | None:
    """Return the root call that owns a top-level call (``call_parent`` unset)."""

    if not roots:
        return None
    for root in roots:
        if label in (getattr(root, "call_children", None) or ()):
            return str(root.label)
    return str(roots[0].label)


def _parent_label(call: Any, roots: list[Any]) -> str | None:
    """Normalize a call's parent: ``None`` and ``"self:<n>"`` both mean the root."""

    if call.address == ROOT_ADDRESS:
        return None
    parent = call.call_parent
    if parent is None:
        return _root_parent_of(str(call.label), roots)
    return str(parent)


def _return_ordered(call: Any, trace: Trace, own: tuple[str, ...]) -> tuple[str, ...] | None:
    """The call's own output ops in RETURN order, or ``None`` when that order is unknown.

    ``output_ops`` is in op-creation order (HF eager attention creates its weights
    before its output, then returns ``(output, weights)``). Each op's
    ``multi_output_name``, minted from its return path at capture, ties it to its
    position in ``output_paths`` (the return traversal order).
    """

    ops = [_op_from_label(trace, label) for label in own]
    if any(op is None for op in ops):
        return None
    paths = _module_output_paths(trace, call, str(call.address), ops)
    if paths is None:
        return None
    position = {tuple(path): index for index, path in enumerate(call.output_paths or ())}
    return tuple(
        label
        for _path, label in sorted(
            zip(paths, own, strict=True), key=lambda pl: position[tuple(pl[0])]
        )
    )


def _resolved_labels(trace: Trace, labels: Any) -> tuple[str, ...]:
    """Return the pass-qualified op labels of ``labels`` in order, once each."""

    resolved = (_op_from_label(trace, str(label)) for label in labels)
    return tuple(dict.fromkeys(_op_label(op) for op in resolved if op is not None))


def _outputs(call: Any, trace: Trace) -> tuple[tuple[str, ...], bool]:
    """The call's recorded outputs and whether they are in return order.

    Own output ops come first, then a returned input (the order torch's exit-time
    identity op takes). Two or more returned inputs give no outputs (the boundary reads
    as unrecorded): their producers' execution order need not be the return order. More
    than one leaf is in return order only when the return paths resolve it.
    """

    if call.address == ROOT_ADDRESS and trace.output_layers:
        return _resolved_labels(trace, trace.output_layers), True
    passthrough = passthrough_op_labels(call, trace)
    if len(passthrough) > 1:
        return (), True
    own = tuple(str(label) for label in (call.output_ops or ()))
    leaves = (*own, *passthrough)
    if len(leaves) <= 1:
        return leaves, True
    ordered = None if passthrough else _return_ordered(call, trace, own)
    return (leaves, False) if ordered is None else (ordered, True)


def _inputs(
    call: Any, trace: Trace, outputs: tuple[str, ...], order: dict[str, int]
) -> tuple[str, ...]:
    """The call's recorded input ops, else the ops outside it that feed its own ops.

    A root call without recorded input ops reads the trace's input ops (JAX records
    neither root inputs nor nested ops on the root). A call with no ops of its own (a recorded pass-through) returned its input, so
    its outputs are its inputs.
    """

    if call.input_ops:
        return tuple(str(label) for label in call.input_ops)
    if call.address == ROOT_ADDRESS and trace.input_layers:
        return _resolved_labels(trace, trace.input_layers)
    inside = {str(label) for label in (call.ops or ())}
    if not inside:
        return outputs
    return tuple(sorted(_feeders(trace, inside), key=lambda label: order.get(label, -1)))


def _feeders(trace: Trace, inside: set[str]) -> set[str]:
    """Return the labels of ops outside ``inside`` that are parents of ops in it."""

    feeders: set[str] = set()
    for label in inside:
        op = _op_from_label(trace, label)
        for parent in getattr(op, "parents", None) or ():
            producer = _op_from_label(trace, str(parent))
            if producer is not None and _op_label(producer) not in inside:
                feeders.add(_op_label(producer))
    return feeders


def _span(call: Any, outputs: tuple[str, ...], order: dict[str, int]) -> tuple[int, int]:
    """Return ``(enter, exit)`` execution positions from the call's ops, else its outputs."""

    positions = [order[label] for label in (call.ops or ()) if label in order]
    if not positions:
        positions = [order[label] for label in outputs if label in order]
    if not positions:
        return -1, -1
    return min(positions), max(positions)


def read_calls(trace: Trace) -> list[CallInfo]:
    """Return every recorded module call of ``trace`` in module-exit order.

    Exit order puts a child before the parent that returns its value, so the first
    differing row is the innermost module where the outputs separate.

    Parameters
    ----------
    trace:
        Completed TorchLens trace (live or loaded).

    Returns
    -------
    list[CallInfo]
        Calls sorted by (exit position, deeper first, enter position).
    """

    order = {_op_label(op): index for index, op in enumerate(_ordered_ops(trace))}
    raw = _raw_calls(trace)
    roots = [call for _module, call in raw if call.address == ROOT_ADDRESS]
    parents = {str(call.label): _parent_label(call, roots) for _module, call in raw}
    raw, parents = _drop_containers(raw, parents)

    def depth(label: str) -> int:
        """Nesting depth by walking parents (bounded by the call count)."""

        steps, current = 0, parents.get(label)
        while current is not None and steps <= len(parents):
            steps, current = steps + 1, parents.get(current)
        return steps

    infos = []
    for module, call in raw:
        outputs, ordered = _outputs(call, trace)
        enter, exit_ = _span(call, outputs, order)
        training = getattr(module, "training", None)
        infos.append(
            CallInfo(
                label=str(call.label),
                address=str(call.address),
                call_index=int(call.call_index),
                class_name=str(
                    getattr(call, "class_name", "") or getattr(module, "class_name", "") or ""
                ),
                parent=parents[str(call.label)],
                enter=enter,
                exit=exit_,
                depth=depth(str(call.label)),
                outputs=outputs,
                inputs=_inputs(call, trace, outputs, order),
                training=None if training is None else bool(training),
                ordered=ordered,
            )
        )
    return sorted(infos, key=lambda info: (info.exit, -info.depth, info.enter))


def normalized_address(address: str, called: frozenset[str]) -> str:
    """Drop never-called container segments that precede a numeric index.

    ``model.layers.0.mlp`` reads as ``model.0.mlp`` when ``model.layers`` has no
    recorded call (a ``ModuleList`` or a plain list), so MLX ``layers.0`` and a
    torch ``Sequential`` child ``0`` meet.

    Parameters
    ----------
    address:
        Module address.
    called:
        Every address with at least one recorded call on the same trace.

    Returns
    -------
    str
        The normalized address.
    """

    segments = address.split(".")
    kept = []
    for index, segment in enumerate(segments):
        prefix = ".".join(segments[: index + 1])
        follows_index = index + 1 < len(segments) and segments[index + 1].isdigit()
        if follows_index and prefix not in called and not segment.isdigit():
            continue
        kept.append(segment)
    return ".".join(kept)
