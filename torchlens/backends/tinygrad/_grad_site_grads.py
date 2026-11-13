"""``tl.grad`` edit sites on a tinygrad capture, applied in derived gradients.

``trace(intervene=tl.when(tl.grad(site), edit), grad_options=GradOptions(...))``
edits the gradient of ``site``'s output, not its activation. tinygrad has no
gradient hooks and the capture records no backward pass, so the clause cannot
fire during capture. The capture runs the plain forward, resolves the clause's
sites on the finished trace and arms them there through the backend-neutral
gradient-site module. The derived gradients are then computed by its reverse
engine over the replay adapter's recipes, from the input and parameter values
the capture saved: the armed edit replaces the gradient at each site and the
edited gradient flows upstream, which is what PyTorch's gradient hook does.
An op the adapter cannot replay refuses by name (``run_capability_unavailable``).

Input gradients and intermediate gradients are published from that pass;
parameter gradients are not (the reverse engine sees parameters as saved
buffers). This module imports no selector or hook package: the edit plan
(which alone imports it) hands in the label resolver.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..._data_classes.derived_grad import (
    DerivedGradAccessor,
    DerivedGradRecord,
    IntermediateDerivedGradAccessor,
    IntermediateDerivedGradRecord,
    set_intermediate_derived_grads,
)
from ..._ir.refs import DtypeRef
from ...intervention._grad_site_neutral import (
    GradSitePass,
    GradSiteResult,
    arm_grad_sites,
    build_grad_site_edits,
    run_armed_grad_sites,
)
from .._replay_adapters import get_adapter
from ..registry import BackendUnsupportedError
from ._derived_grads import _differentiated_input_leaves, _normalize_tinygrad_input_grad_argnums

__tl_layer__ = "L4"

_MECHANISM = "reverse_engine.tl_grad_sites"
_REFUSAL = "backward_capture_unsupported"


def _refuse_grad_site(message: str, *, code: str) -> BackendUnsupportedError:
    """A typed refusal of a ``tl.grad`` edit the tinygrad backend cannot compute."""

    return BackendUnsupportedError(f"tinygrad backend: {message}", code=code)


def selector_operands(target: Any) -> tuple[str | None, tuple[Any, ...]]:
    """Read a composite selector as ``(connective, children)``; a leaf is ``(None, ())``."""

    kind = getattr(target, "selector_kind", None)
    if kind in ("and", "or") and hasattr(target, "selectors"):
        return kind, tuple(target.selectors)
    if kind == "not" and hasattr(target, "selector"):
        return "not", (target.selector,)
    return None, ()


def _replay_values(trace: Any, adapter: Any) -> tuple[dict[str, Any], list[Any]]:
    """Every label's value, replayed from the saved sources; the graph ops in forward order."""

    values: dict[str, Any] = {}
    graph_ops: list[Any] = []
    for op in trace._layer_list:
        recipe = adapter.recipe(trace, op)
        if recipe is None:
            if op.out is None:
                raise _refuse_grad_site(
                    f"tl.grad(...) edits need the value of source {op.label!r}, which this "
                    "capture did not save.",
                    code=_REFUSAL,
                )
            values.update(dict.fromkeys((str(op._label_raw), str(op.label)), op.out))
            continue
        graph_ops.append(op)
        if recipe.output_slots[0] not in values:
            outputs = adapter.forward(trace, op, [values[p] for p in recipe.parent_labels])
            values.update(zip(recipe.output_slots, outputs, strict=True))
    return values, graph_ops


def _seed(trace: Any, adapter: Any, values: dict[str, Any], loss_fn: Any) -> dict[str, Any]:
    """``{output label: d loss / d output}`` for the one tensor output of the capture."""

    outputs = [op for op in trace._layer_list if op.is_output]
    if len(outputs) != 1:
        raise _refuse_grad_site(
            f"tl.grad(...) edits need a single tensor model output; this capture has {len(outputs)}.",
            code=_REFUSAL,
        )
    recipe = adapter.recipe(trace, outputs[0])
    label = str(outputs[0]._label_raw) if recipe is None else recipe.output_slots[0]
    from tinygrad import Tensor

    leaf = Tensor(values[label].numpy())
    leaf.requires_grad = True
    loss = loss_fn(leaf) if loss_fn is not None else leaf
    if tuple(loss.shape) not in ((), (1,)):
        raise ValueError(
            "tinygrad derived gradients require loss_fn(raw_output) to be scalar unless "
            "the traced output is already scalar."
        )
    loss.sum().backward()
    return {label: leaf.grad}


def _record_fields(backend: Any, grad: Any) -> dict[str, Any]:
    """The shared fields of a derived-gradient record of ``grad``."""

    copied = backend._realized_copy(grad)
    dtype = str(getattr(copied, "dtype", ""))
    return {
        "aval": f"Tensor(shape={tuple(getattr(copied, 'shape', ()))}, dtype={dtype})",
        "dtype_ref": DtypeRef(backend="tinygrad", name=dtype),
        "grad": copied,
    }


def _input_records(
    backend: Any, trace: Any, args: Sequence[Any], grad_options: Any, result: GradSiteResult
) -> dict[str, DerivedGradRecord]:
    """Input-leaf gradient records, matched to the captured input op by container path."""

    argnums = _normalize_tinygrad_input_grad_argnums(grad_options.input_grad_argnums, len(args))
    label_by_path = {
        "inputs." + ".".join(op.container_path): str(op._label_raw)
        for op in trace._layer_list
        if op.is_input
    }
    provenance = {"backend": "tinygrad", "kind": "derived_gradient", "mechanism": _MECHANISM}
    records: dict[str, DerivedGradRecord] = {}
    for path, argnum, tensor in _differentiated_input_leaves(backend, args, argnums):
        grad = result.grads.get(label_by_path.get(path, ""))
        if grad is None:
            grad = tensor.zeros_like()  # the loss does not depend on this input
        records[path] = DerivedGradRecord(
            path=path,
            source="inputs",
            argnum=argnum,
            input_argnum=argnum,
            provenance=provenance,
            **_record_fields(backend, grad),
        )
    return records


def _intermediate_records(
    backend: Any, graph_ops: Sequence[Any], result: GradSiteResult
) -> dict[str, IntermediateDerivedGradRecord]:
    """Intermediate gradient records of the saved graph ops the pass reached."""

    provenance = {
        "backend": "tinygrad",
        "kind": "intermediate_derived_gradient",
        "mechanism": _MECHANISM,
        "status": "exact",
    }
    return {
        op.label: IntermediateDerivedGradRecord(
            op_label=op.label,
            layer_label=op.layer_label,
            provenance=dict(provenance),
            **_record_fields(backend, result.grads[str(op._label_raw)]),
        )
        for op in graph_ops
        if op.has_saved_activation and result.grads.get(str(op._label_raw)) is not None
    }


def attach_grad_site_derived_grads(
    backend: Any,
    trace: Any,
    call: dict[str, Any],
    plan: Any,
    resolve_labels: Callable[[Any, Any], Sequence[str]],
) -> GradSiteResult:
    """Arm one ``tl.when(tl.grad(...), edit)`` clause on ``trace`` and attach its gradients.

    Parameters
    ----------
    backend
        The ``TinygradBackend`` that produced ``trace``.
    trace
        The plain capture of the forward.
    call
        The ``capture_trace`` keyword arguments of the capture; its
        ``grad_options`` is the tinygrad ``GradOptions``.
    plan
        The clause's ``TinygradEditPlan``: ``plan.selector`` is the gradient
        site (``tl.grad(s)``, unions, a pass qualifier) and ``plan.edit.apply``
        the resolved ``tensor -> tensor`` edit.
    resolve_labels
        ``resolve_labels(trace, forward_selector)``: raw labels it names.

    Returns
    -------
    GradSiteResult
        The edited pass: gradient per reached label and the fires.
    """

    grad_options = call["grad_options"]
    adapter = get_adapter("tinygrad")
    edits = build_grad_site_edits(
        plan.selector,
        lambda grad, _call: plan.edit.apply(grad),
        lambda inner: resolve_labels(trace, inner),
        operands=selector_operands,
    )
    arm_grad_sites(trace, edits)
    values, graph_ops = _replay_values(trace, adapter)
    seeds = _seed(trace, adapter, values, grad_options.loss_fn)
    result = run_armed_grad_sites(
        trace, "tinygrad", graph_ops, values.__getitem__, GradSitePass(seeds)
    )
    args = backend._normalize_input_args(call["input_args"])
    trace._autograd_derived_grads = DerivedGradAccessor(
        _input_records(backend, trace, args, grad_options, result)
    )
    if grad_options.intermediate_grads:
        records = _intermediate_records(backend, graph_ops, result)
        set_intermediate_derived_grads(trace, IntermediateDerivedGradAccessor(records))
    return result
