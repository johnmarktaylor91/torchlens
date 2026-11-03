"""Prove from a trace that a backward ran without loss scaling.

``grad_health`` refuses magnitude verdicts when the loss scale is unknown:
gradients captured under ``torch.amp.GradScaler`` carry the scale (~2**16),
and unscaled thresholds would call every op exploding. A plain run, though,
has no scale, and refusing a verdict there hides every finding. This module
decides, from the trace alone, whether the selected backward provably ran
with scale 1.0. The proof is conservative: any fact it cannot establish
leaves the scale unknown.

The proof holds when all of these do:

- no captured forward op ran with autocast enabled on any device, and each
  op recorded its autocast state (so "no autocast" is observed, not assumed);
- the backward was started without an explicit ``gradient=`` seed;
- every autograd node between the backward root (the loss) and the first
  node of the captured forward is recorded, and none of them is a multiply,
  a divide or a custom autograd function. ``GradScaler.scale(loss)`` is a
  multiply outside the model, so a scaled loss always fails this check.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torchlens._data_classes.trace import Trace

#: Autograd node classes that can rescale every gradient below them.
_SCALING_GRAD_FNS = frozenset(
    {
        "MulBackward0",
        "MulBackward1",
        "DivBackward0",
        "DivBackward1",
        "DivBackward2",
        "DivBackward3",
    }
)

#: Engine keyword arguments that seed the backward with a caller tensor.
_SEED_KEYS = ("gradient", "grad_tensors")


def _autocast_reason(trace: Trace) -> str | None:
    """Return why autocast cannot be ruled out for the forward, or ``None``."""

    ops = [op for op in trace._layer_list if int(getattr(op, "step_index", 0) or 0) > 0]
    if not ops:
        return "the trace has no compute ops"
    for op in ops:
        state = getattr(op, "func_autocast_state", None)
        if not isinstance(state, dict):
            return "an op recorded no autocast state"
        devices = {key: value for key, value in state.items() if key != "__execution__"}
        if not devices:
            return "an op recorded no autocast state"
        for value in devices.values():
            if isinstance(value, dict) and value.get("enabled"):
                return "an op ran under autocast"
    return None


def _backward_pass(trace: Trace, bwd: int) -> Any:
    """Return the backward-pass record numbered ``bwd``, or ``None``."""

    for record in list(trace.autograd.backward_passes):
        if getattr(record, "pass_number", None) == bwd:
            return record
    return None


def _seed_reason(record: Any) -> str | None:
    """Return why the backward seed may be scaled, or ``None``."""

    if getattr(record, "root_grad_arguments", None) is not None:
        return "the backward was seeded with an explicit gradient"
    flags = getattr(record, "engine_flags", None) or {}
    if any(flags.get(key) is not None for key in _SEED_KEYS):
        return "the backward was seeded with an explicit gradient"
    return None


def _loss_path_reason(trace: Trace, record: Any) -> str | None:
    """Return why the loss-side autograd path may scale, or ``None``.

    Walks from the backward roots through nodes outside the captured forward,
    stopping at the first node that belongs to a captured op.
    """

    logs = getattr(trace, "_grad_fn_logs", None) or {}
    roots = list(getattr(record, "root_grad_fn_ids", None) or ())
    if not roots or not logs:
        return "the backward graph was not recorded"
    queue: deque[int] = deque(roots)
    seen: set[int] = set()
    while queue:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        node = logs.get(node_id)
        if node is None:
            return "the backward graph was not fully recorded"
        if bool(getattr(node, "has_op", False)):
            continue
        class_name = str(getattr(node, "class_name", ""))
        if class_name in _SCALING_GRAD_FNS:
            return f"{class_name} sits between the loss and the model output"
        if bool(getattr(node, "is_custom", False)):
            return (
                f"custom autograd function {class_name} sits between the loss and the model output"
            )
        queue.extend(getattr(node, "next_grad_fn_ids", None) or ())
    return None


def unscaled_proof(trace: Trace, bwd: int) -> tuple[bool, str]:
    """Decide whether backward pass ``bwd`` provably ran with loss scale 1.0.

    Parameters
    ----------
    trace:
        Completed TorchLens trace with at least one backward pass.
    bwd:
        One-based backward pass number.

    Returns
    -------
    tuple[bool, str]
        ``(True, why)`` when the proof holds, else ``(False, why_not)``.
    """

    reason = _autocast_reason(trace)
    if reason is not None:
        return False, reason
    record = _backward_pass(trace, bwd)
    if record is None:
        return False, f"backward pass {bwd} was not recorded"
    reason = _seed_reason(record) or _loss_path_reason(trace, record)
    if reason is not None:
        return False, reason
    return True, (
        "no op ran under autocast and no multiply, divide or custom function "
        "sits between the loss and the model output"
    )
