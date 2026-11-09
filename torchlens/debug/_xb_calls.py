"""Read a trace's recorded module calls into a backend-neutral form for pairing.

Every backend records module calls as ``ModuleCall`` records (address, call index,
parent call, op labels, recorded output ops). Backends differ in two spellings
that this reader absorbs, without changing capture: the root's own children carry
``call_parent=None`` on torch and ``"self:1"`` on the preview backends, and
container segments that are never called (a torch ``ModuleList``, a plain MLX
list) appear in addresses as ``layers.0`` on one side and ``0`` on the other.

A call that returns its input unchanged (an identity module, dropout in eval mode) is an
op inside the call on torch, which inserts an identity op at module exit, but a recorded
passthrough on the preview backends: the returned input's op is flagged as the call's
output and stays out of ``output_ops``. The reader appends those passthrough ops after
the call's own outputs, the order torch's exit-time identity ops take, so both spellings
give the same output roster.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._data_classes._module_passthrough import passthrough_op_labels
from ._common import _op_label, _ordered_ops

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

ROOT_ADDRESS = "self"


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


def _raw_calls(trace: Trace) -> list[tuple[Any, Any]]:
    """Return ``(module, call)`` pairs for every recorded module call."""

    return [(module, call) for module in trace._module_logs for call in module.calls.values()]


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


def _outputs(call: Any, trace: Trace) -> tuple[str, ...]:
    """The call's recorded outputs: its own output ops, then any returned inputs.

    Two or more returned inputs give no outputs (the boundary reads as unrecorded):
    their producers' execution order need not be the order the call returned them in.
    """

    passthrough = passthrough_op_labels(call, trace)
    if len(passthrough) > 1:
        return ()
    return tuple(str(label) for label in (*(call.output_ops or ()), *passthrough))


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

    def depth(label: str) -> int:
        """Nesting depth by walking parents (bounded by the call count)."""

        steps, current = 0, parents.get(label)
        while current is not None and steps <= len(parents):
            steps, current = steps + 1, parents.get(current)
        return steps

    infos = []
    for module, call in raw:
        outputs = _outputs(call, trace)
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
                inputs=tuple(str(label) for label in (call.input_ops or ())),
                training=None if training is None else bool(training),
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
