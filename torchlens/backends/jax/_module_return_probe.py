"""Recorded module returns for JAX pytree modules (Equinox and Flax NNX).

The shared seam (:mod:`torchlens.backends._module_returns`) wants, at each module exit, the
value the call returned and the op that produced each array leaf. A JAX module call runs
while ``jax.make_jaxpr`` traces the model, so the value it returns is a tracer, and ops get
their labels only later, when TorchLens interprets the jaxpr. This module bridges the two:

The backend keeps calling its own ``derive_closed_jaxpr``, interpreter and module-log builder;
this module only adds steps around them, so it imports nothing from the backend's import
cycle.

1. Tracing. :func:`extend` wraps the model so that it returns the model output followed by
   every array leaf a module call returned (module exits are noted by
   :func:`._return_ledger.note_return` from the scoped call wrappers in ``modules.py``). Each
   such leaf becomes an extra jaxpr output, so after interpretation its value is a concrete
   array whose producing op is known. :func:`strip` gives the backend the same jaxpr with
   the extra outputs removed (what it stamps and checks); the equations are the same
   objects, so no op is added.
2. Interpretation. :func:`jaxpr_to_interpret` hands the interpreter the extended jaxpr and
   :func:`split_outputs` removes the extra outputs from the result. Only the root output
   list differs between the two jaxprs, so the result is exactly what the stripped jaxpr
   gives.
3. Labels. :func:`note_return_labels` runs while capture events exist and resolves each
   extra output to the raw label of the op that produced it, the way trace outputs resolve.
4. Module logs. :func:`build_module_logs_with_returns` records each call in the ledger with
   final op labels, then runs the backend's builder between the seam's flag and path steps.

Honesty rule: a call is recorded only when every array leaf it returned is a tracer of the
top-level trace. A module called under ``jax.vmap`` returns tracers of the batching trace:
one trace-time call stands for every batch member, so no single op is its output. Such calls
are left unrecorded (their ``output_ops`` stay empty and ``Module.out`` raises) and disclosed
with a ``CaptureGapWarning``, never filled with a guessed op. A module call inside a
``jax.lax.scan``, ``cond`` or ``while_loop`` body returns tracers of the body's trace, and
loop bodies run once per iteration, so those calls are left unrecorded and disclosed too.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from ..._errors import CaptureGapWarning
from .._module_returns import attach_output_paths, flag_returned_ops, iter_returned_leaves
from ._return_ledger import JaxReturnLedger


def extend(model: Callable[..., Any], tree: Any) -> Callable[..., Any]:
    """Return ``model`` extended to also return the top-level leaves module calls returned.

    Parameters
    ----------
    model
        Module root about to be traced (its module calls are already scope-wrapped).
    tree
        Equinox or NNX module tree holding a :class:`JaxReturnLedger`.

    Returns
    -------
    Callable[..., Any]
        A callable returning ``(model_output, [leaf, ...])`` for ``jax.make_jaxpr``.
    """

    ledger = tree.module_returns
    ledger.release()

    def extended(*call_args: Any) -> tuple[Any, list[Any]]:
        """Run the model, then append the top-level tracer leaves module calls returned."""

        ledger.top_trace = _top_trace(call_args)
        ledger.armed = True
        try:
            output = model(*call_args)
        finally:
            ledger.armed = False
        ledger.leaves = _top_level_leaves(ledger)
        return output, list(ledger.leaves)

    return extended


def strip(closed: Any, tree: Any) -> Any:
    """Keep the extended closed jaxpr on the ledger; return it without the extra outputs."""

    ledger = tree.module_returns
    ledger.extended_jaxpr = closed
    return _strip_outvars(closed, len(closed.jaxpr.outvars) - len(ledger.leaves))


def jaxpr_to_interpret(tree: Any, closed_jaxpr: Any) -> Any:
    """Return the extended jaxpr when this capture derived one, else ``closed_jaxpr``."""

    ledger = getattr(tree, "module_returns", None)
    if isinstance(ledger, JaxReturnLedger) and ledger.extended_jaxpr is not None:
        return ledger.extended_jaxpr
    return closed_jaxpr


def _apply_call_renumbering(tree: Any, ledger: Any, renamed: Any) -> None:
    """Move trace-time module calls to their runtime call index.

    Module calls inside loop or jit bodies run once per iteration or bind (the
    interpreter's ``CallNumbering``), which shifts the index of later calls of the same
    module; their forward arguments and noted returns follow.
    """

    if tree is None or not renamed:
        return
    moved = {
        (address, renamed.get((address, call), call)): value
        for (address, call), value in tree.forward_args_by_call.items()
    }
    tree.forward_args_by_call.clear()  # the tree is frozen; its dict is the shared record
    tree.forward_args_by_call.update(moved)
    if isinstance(ledger, JaxReturnLedger):
        ledger.pending = [
            (address, renamed.get((address, call), call), value)
            for address, call, value in ledger.pending
        ]


def split_outputs(result: Any, tree: Any) -> Any:
    """Move the extra outputs of an extended interpretation onto the ledger.

    Parameters
    ----------
    result
        Interpreter result for the jaxpr :func:`jaxpr_to_interpret` returned.
    tree
        Active module tree, or ``None`` for a function-root capture.

    Returns
    -------
    Any
        ``result`` with the model's outputs only.
    """

    ledger = getattr(tree, "module_returns", None)
    _apply_call_renumbering(tree, ledger, getattr(result, "module_call_renumbering", None))
    if not isinstance(ledger, JaxReturnLedger) or ledger.extended_jaxpr is None:
        return result
    ledger.extended_jaxpr = None
    if getattr(result, "halt_frontier", None) is not None:
        # A halted run never reaches the extra outputs; no module return is recorded.
        ledger.release()
        return result
    kept = len(result.outputs) - len(ledger.leaves)
    ledger.extra_values = tuple(result.outputs[kept:])
    producer_by_id = {
        id(value): capture.index for capture in result.captures for value in capture.output_values
    }
    ledger.producers = tuple(producer_by_id.get(id(value)) for value in ledger.extra_values)
    return replace(result, outputs=tuple(result.outputs[:kept]))


def note_return_labels(trace: Any, tree: Any) -> None:
    """Resolve each extra output to its producing op's raw label (capture events alive)."""

    ledger = getattr(tree, "module_returns", None)
    if not isinstance(ledger, JaxReturnLedger) or not ledger.extra_values:
        return
    by_payload = {
        id(event.output.tensor.payload): event.label_raw
        for event in trace.capture_events.amended_op_records()
        if event.output.tensor.payload is not None
    }
    by_capture = getattr(trace, "_jax_capture_index_to_raw_op_label", {})
    for leaf, value, producer in zip(
        ledger.leaves, ledger.extra_values, ledger.producers, strict=True
    ):
        label = by_payload.get(id(value))
        if label is None and producer is not None:
            label = by_capture.get(producer)
        if label is not None:
            ledger.raw_labels[id(leaf)] = label


def build_module_logs_with_returns(
    trace: Any, tree: Any, build_module_logs: Callable[[Any], None]
) -> None:
    """Record each honest module return, then build module logs through the seam."""

    ledger = getattr(tree, "module_returns", None)
    if isinstance(ledger, JaxReturnLedger):
        _record_pending(trace, ledger)
    flag_returned_ops(trace, ledger)
    build_module_logs(trace)
    attach_output_paths(trace, ledger)


def _record_pending(trace: Any, ledger: JaxReturnLedger) -> None:
    """Move every pending call whose leaves are all top-level tracers into the ledger."""

    final_by_raw = {op._label_raw: op.label for op in trace._layer_list}
    allowed = {id(leaf) for leaf in ledger.leaves}

    def label_of(leaf: object) -> str | None:
        """Return the final label of the op that produced a recorded leaf."""

        raw = ledger.raw_labels.get(id(leaf))
        return final_by_raw.get(raw) if raw is not None else None

    for address, call_index, value in ledger.pending:
        tracers = [leaf for leaf in _array_leaves(value) if _is_tracer(leaf)]
        if all(id(leaf) in allowed for leaf in tracers):
            ledger.record(address, call_index, value, is_leaf=_is_array, label_of=label_of)
        else:
            warnings.warn(
                CaptureGapWarning(
                    f"JAX module call '{address}:{call_index}' returned arrays of an inner trace "
                    "(jax.vmap, or a jit, scan, cond or while body: one trace-time call stands "
                    "for every batch member, bind or iteration), so no op is its output; its "
                    "return is not recorded, so this call has no "
                    "output_ops, output_paths or Module.out. Remedy: read the values through "
                    "the enclosing module's output",
                    code="module_return_unrecorded",
                ),
                stacklevel=2,
            )
    ledger.release()


def _top_level_leaves(ledger: JaxReturnLedger) -> list[Any]:
    """Return each distinct top-level tracer leaf of the pending returns, in exit order."""

    seen: dict[int, Any] = {}
    for _address, _call_index, value in ledger.pending:
        for leaf in _array_leaves(value):
            if _is_top_level_tracer(leaf, ledger.top_trace):
                seen.setdefault(id(leaf), leaf)
    return list(seen.values())


def _array_leaves(value: Any) -> list[Any]:
    """Return the array leaves of a returned value, in seam container order."""

    return [leaf for _path, leaf in iter_returned_leaves(value, _is_array)]


def _is_array(value: object) -> bool:
    """Return whether ``value`` is a JAX array or tracer."""

    import jax

    return isinstance(value, (jax.Array, _tracer_type()))


def _is_tracer(value: object) -> bool:
    """Return whether ``value`` is a JAX tracer."""

    return isinstance(value, _tracer_type())


def _tracer_type() -> Any:
    """Return JAX's tracer class, or an empty tuple when this JAX does not expose it."""

    import jax

    return getattr(getattr(jax, "core", None), "Tracer", ())


def _top_trace(call_args: Sequence[Any]) -> Any:
    """Return the trace of the top-level dynamic arguments, or ``None`` when unknown.

    ``Tracer._trace`` is a private attribute present on every JAX tracer the supported range
    ships; when it is missing no leaf counts as top-level and nothing is recorded.
    """

    import jax

    for leaf in jax.tree_util.tree_leaves(call_args):
        if isinstance(leaf, _tracer_type()):
            return getattr(leaf, "_trace", None)
    return None


def _is_top_level_tracer(value: object, top_trace: Any) -> bool:
    """Return whether ``value`` is a tracer of the top-level trace."""

    if top_trace is None or not _is_tracer(value):
        return False
    return getattr(value, "_trace", None) is top_trace


def _strip_outvars(closed: Any, kept: int) -> Any:
    """Return ``closed`` with only its first ``kept`` outputs (equations untouched)."""

    jaxpr = closed.jaxpr
    changes: dict[str, Any] = {"outvars": list(jaxpr.outvars[:kept])}
    debug_info = getattr(jaxpr, "debug_info", None)
    result_paths = getattr(debug_info, "result_paths", None)
    replace_info = getattr(debug_info, "_replace", None)
    if isinstance(result_paths, tuple) and callable(replace_info):
        changes["debug_info"] = replace_info(result_paths=result_paths[:kept])
    return closed.replace(jaxpr=jaxpr.replace(**changes))
