"""``tl.grad`` edit sites on the Paddle backend.

A ``trace(intervene=tl.when(tl.grad(site), edit))`` clause edits the gradient
of ``site``'s output, not its activation. Paddle records no backward pass, so
the clause cannot fire during capture. Instead the capture splits it off the
live forward predicate (:func:`split_gradient_rules`), resolves its sites on
the finished trace and arms it there through the backend-neutral gradient-site
module (:func:`arm_paddle_grad_sites`). Gradient computations then apply the
armed edits:

- The derived-gradient replay (``GradOptions``) registers each armed edit as a
  Paddle gradient hook on the replayed site output (``_edit_replay``), so
  ``paddle.grad`` returns the edited gradients for inputs and parameters.
- :func:`paddle_grad_site_records` recomputes the same pass with the neutral
  reverse engine over the replay adapter's recipes. Its gradients are the
  published intermediate gradients (the edited value at a site, what PyTorch
  saves as ``Op.grad``), and its input-leaf gradients must agree with Paddle's
  own; a disagreement refuses instead of publishing either. An op the adapter
  cannot replay refuses by name (``run_capability_unavailable``).

This module imports without Paddle installed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..._data_classes.derived_grad import (
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
)
from ..._ir.refs import DtypeRef
from ..._utils._record_group_table import record_member
from ...intervention._grad_site_neutral import (
    armed_grad_sites,
    run_armed_grad_sites,
)
from .._protocols.paddle import PADDLE_REFUSAL_BACKWARD
from .._replay_adapters import get_adapter
from ..registry import BackendUnsupportedError


def _backward_refusal(message: str) -> BackendUnsupportedError:
    """Return the typed refusal for a gradient address Paddle cannot arm."""

    return BackendUnsupportedError(message, code=PADDLE_REFUSAL_BACKWARD)


def armed_hooks_by_call(trace: Any) -> dict[int, tuple[tuple[tuple[Any, ...], Any, str], ...]]:
    """The armed gradient edits keyed by the wrapped-call ordinal of their site.

    Parameters
    ----------
    trace
        Finished Paddle trace with armed gradient sites.

    Returns
    -------
    dict[int, tuple[tuple[tuple[Any, ...], Any, str], ...]]
        ``(output path, armed edit, raw label)`` per call ordinal, in arming order.
    """

    ops = {getattr(op, "_label_raw", None): op for op in getattr(trace, "_layer_list", ())}
    hooks: dict[int, tuple[tuple[tuple[Any, ...], Any, str], ...]] = {}
    for edit in armed_grad_sites(trace):
        op = ops.get(edit.label)
        call_id = None if op is None else record_member(op, "func_call_id", None)
        if call_id is None:
            raise _backward_refusal(
                f"the paddle backend cannot apply the tl.grad edit at {edit.label!r}: the "
                "site is not the output of a recorded wrapped call (an input leaf or a "
                "constant), so the derived-gradient replay has no call to attach it to."
            )
        path = tuple(record_member(op, "container_path", ()) or ())
        hooks[int(call_id)] = (*hooks.get(int(call_id), ()), (path, edit, edit.label))
    return hooks


def _is_boundary_output(op: Any) -> bool:
    """Whether ``op`` is an output record mirroring its single parent."""

    return bool(record_member(op, "is_output", False)) and len(tuple(op.parents or ())) == 1


def _seed_label(trace: Any, halt_call_id: int | None) -> str:
    """Raw label of the op whose output the loss is computed from."""

    ops = tuple(getattr(trace, "_layer_list", ()))
    if halt_call_id is not None:
        found = [op for op in ops if record_member(op, "func_call_id", None) == halt_call_id]
    else:
        found = [op for op in ops if record_member(op, "is_output", False)]
    if len(found) != 1:
        raise _backward_refusal(
            "paddle tl.grad(...) edits with derived gradients need a single tensor model "
            f"output (or halted frontier); this capture has {len(found)} output ops."
        )
    op = found[0]
    if _is_boundary_output(op):
        op = trace._layer_dict_all_keys[tuple(op.parents)[0]]
    return str(op._label_raw)


def _engine_ops(trace: Any) -> tuple[Any, ...]:
    """The trace's ops in forward order without inputs and output mirror records."""

    return tuple(
        op
        for op in getattr(trace, "_layer_list", ())
        if not record_member(op, "is_input", False) and not _is_boundary_output(op)
    )


def _loss_cotangent(backend: Any, output: Any, loss_fn: Any) -> Any:
    """``d loss / d output`` at the replay output, on a detached copy."""

    paddle = backend.paddle
    if loss_fn is None:
        return paddle.ones_like(output)
    leaf = output.detach()
    leaf.stop_gradient = False
    (cotangent,) = paddle.grad(loss_fn(leaf), [leaf], allow_unused=False)
    return cotangent


def _check_input_leaves(
    adapter: Any, trace: Any, grads: Mapping[str, Any], leaves: Sequence[Any], native: Sequence[Any]
) -> None:
    """Refuse when the engine's input-leaf gradients disagree with Paddle's own."""

    inputs = [op for op in trace._layer_list if record_member(op, "is_input", False)]
    for leaf, native_grad in zip(leaves, native, strict=False):
        if leaf.source != "inputs" or native_grad is None:
            continue
        match = next(
            (op for op in inputs if op.out is leaf.tensor or _same_value(adapter, op.out, leaf)),
            None,
        )
        engine_grad = None if match is None else grads.get(str(match._label_raw))
        if engine_grad is not None and not adapter.close(native_grad, engine_grad):
            raise _backward_refusal(
                f"paddle tl.grad(...) edits: the edited gradient of {leaf.path!r} from "
                "paddle.grad disagrees with the reverse-engine recomputation; refusing to "
                "publish derived gradients."
            )


def _same_value(adapter: Any, value: Any, leaf: Any) -> bool:
    """Whether a saved input value equals a leaf tensor (same shape, close values)."""

    try:
        return bool(tuple(value.shape) == tuple(leaf.tensor.shape)) and bool(
            adapter.close(value, leaf.tensor)
        )
    except Exception:  # noqa: BLE001 - an incomparable value is simply not a match
        return False


def _engine_record(op: Any, grad: Any, grad_options: Any) -> IntermediateDerivedGradRecord:
    """One intermediate derived-gradient record computed by the reverse engine."""

    loss_fn = grad_options.loss_fn
    return IntermediateDerivedGradRecord(
        op_label=op.label,
        layer_label=op.layer_label,
        aval=f"Tensor(shape={tuple(getattr(grad, 'shape', ()))}, dtype={grad.dtype})",
        dtype_ref=DtypeRef(backend="paddle", name=str(grad.dtype)),
        grad=grad,
        provenance={
            "backend": "paddle",
            "kind": "intermediate_derived_gradient",
            "mechanism": "reverse_engine.tl_grad_sites",
            "loss_id": None if loss_fn is None else getattr(loss_fn, "__qualname__", None),
            "save_predicate_id": "trace.saved_ops",
            "status": "exact",
            "match_key": "label_raw",
            "max_intermediate_grads": grad_options.max_intermediate_grads,
        },
    )


def _engine_pass(backend: Any, trace: Any, replay_output: Any, loss_fn: Any) -> Any:
    """The armed pass recomputed by the reverse engine from the captured values."""

    if not backend.is_tensor(replay_output):
        raise _backward_refusal(
            "paddle tl.grad(...) edits with derived gradients need a single tensor model "
            "output (or halted frontier)."
        )
    runtime = getattr(trace, "_paddle_intervention_runtime", None)
    seed_label = _seed_label(trace, None if runtime is None else runtime.halt_call_id)
    return run_armed_grad_sites(
        trace,
        "paddle",
        _engine_ops(trace),
        lambda label: _saved_value(trace, label),
        {seed_label: _loss_cotangent(backend, replay_output, loss_fn)},
    )


def _saved_value(trace: Any, label: str) -> Any:
    """The captured value of a raw label; refuse when retention dropped it."""

    op = next(op for op in trace._layer_list if getattr(op, "_label_raw", None) == label)
    if op.out is None:
        raise _backward_refusal(
            f"paddle tl.grad(...) edits with derived gradients need the activation of "
            f"{op.label!r}, which this capture did not save; capture with save=True."
        )
    return op.out


def paddle_grad_site_records(
    backend: Any,
    trace: Any,
    replay_output: Any,
    grad_options: Any,
    leaves: Sequence[Any],
    leaf_grads: Sequence[Any],
) -> IntermediateDerivedGradAccessor | None:
    """Recompute an armed pass with the reverse engine, check it, return intermediate records.

    Parameters
    ----------
    backend
        Active ``PaddleBackend``.
    trace
        Finished trace and its live intervention runtime.
    replay_output
        Raw output (or halted frontier) of the derived-gradient replay.
    grad_options
        Paddle ``GradOptions``.
    leaves, leaf_grads
        The differentiated leaves (inputs first) and their ``paddle.grad``
        results, edited through the replay hooks.

    Returns
    -------
    IntermediateDerivedGradAccessor | None
        ``None`` when no gradient edit is armed on ``trace``; otherwise the
        edited intermediate gradients by op label (empty unless requested).
    """

    if not armed_grad_sites(trace):
        return None
    result = _engine_pass(backend, trace, replay_output, grad_options.loss_fn)
    _check_input_leaves(get_adapter("paddle"), trace, result.grads, leaves, leaf_grads)
    if not grad_options.intermediate_grads:
        return IntermediateDerivedGradAccessor()
    return IntermediateDerivedGradAccessor(
        {
            op.label: _engine_record(op, result.grads[op._label_raw], grad_options)
            for op in _engine_ops(trace)
            if result.grads.get(getattr(op, "_label_raw", None)) is not None
        }
    )


__all__ = (
    "armed_hooks_by_call",
    "paddle_grad_site_records",
)
