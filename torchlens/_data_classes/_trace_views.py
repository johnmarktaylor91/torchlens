"""Trace group-view and field-derived accessor functions.

Bodies of the ``Trace`` view properties (``ops``, ``layers``, ``module_calls``,
the saved and compute subsets, the graph degree counts and the autograd
accessors) live here as ``f(trace)`` functions; the record keeps one thin
property per name with its public docstring and forwards to these.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from .layer import Layer

if TYPE_CHECKING:
    from .trace import Trace

from collections import OrderedDict
from typing import TYPE_CHECKING, cast

from ._accessor_base import Accessor
from ._backend_capability_guards import raise_if_no_backward_capture
from ._trace_accessors import (
    _TRACE_LAYER_ACCESSOR_ATTR,
    _TRACE_MODULE_CALL_ACCESSOR_ATTR,
    _TRACE_OP_ACCESSOR_ATTR,
    OrphanAccessor,
    TraceGradFnCallAccessor,
    TraceModuleCallAccessor,
    TraceOpAccessor,
    boundary_label_ops,
)
from .backward_pass import BackwardPassAccessor
from .derived_grad import IntermediateDerivedGradAccessor
from .grad_fn import GradFnAccessor
from .layer import LayerAccessor
from .module import ModuleAccessor
from .op import Op

if TYPE_CHECKING:
    from .trace import Trace

__all__: list[str] = []


def layers_recurrent(trace: Trace) -> LayerAccessor:
    """Body of ``Trace._layers_recurrent``; the record keeps the thin spelling and docstring."""

    return LayerAccessor(
        OrderedDict(
            (label, layer) for label, layer in trace._layer_logs.items() if layer.num_passes > 1
        ),
        source_trace=trace,
    )


def graph_tied_param_groups(trace: Trace) -> tuple[tuple[str, ...], ...]:
    """Body of ``Trace._graph_tied_param_groups``; the record keeps the thin spelling and docstring."""

    groups = []
    for pl in trace._param_logs:
        addresses = tuple(getattr(pl, "all_addresses", None) or (pl.address,))
        if len(addresses) > 1:
            groups.append(addresses)
    return tuple(groups)


def ops(trace: Trace) -> TraceOpAccessor:
    """Body of ``Trace.ops``; the record keeps the thin spelling and docstring."""

    # Memoized on the instance, never in a module global: a global
    # weak-keyed cache value reaches this Trace through the held records
    # and would pin it forever (the R37 ``trace.run()`` fork leak).
    # Keyed on container IDENTITY (held strongly in the entry, so a
    # recycled ``id`` can never mistranslate) plus length: every internal
    # rebind (build, rerun refresh) changes identity, every removal
    # changes length, and the in-place rename/refresh paths call
    # ``_invalidate_trace_op_layer_accessor_caches`` explicitly. The r3
    # by-value label-tuple key (98909fc7) additionally auto-detected
    # DIRECT equal-length in-place edits of ``layer_list``, but priced
    # every memo HIT at O(n), turning hot ``trace.ops`` sweeps O(n^2)
    # (measured exponent ~2.1) and regressing ``tl.trace`` itself
    # 6-10% (r3 R52-1/R28-1); the r3 fixplan prescribes this identity+len
    # key. Direct user in-place mutation of this build product is outside
    # the memo's auto-detection contract -- the supported spellings are
    # reassignment or the explicit invalidator.
    container = trace._layer_list
    cache_entry = trace.__dict__.get(_TRACE_OP_ACCESSOR_ATTR)
    if cache_entry is not None and cache_entry[0] is container and cache_entry[1] == len(container):
        return cast(TraceOpAccessor, cache_entry[2])
    accessor = TraceOpAccessor(container, trace._layer_num_calls, counted_view=True)
    trace.__dict__[_TRACE_OP_ACCESSOR_ATTR] = (container, len(container), accessor)
    return accessor


def transforms(trace: Trace) -> tuple[Op, ...]:
    """Body of ``Trace.transforms``; the record keeps the thin spelling and docstring."""

    return tuple(op for op in trace._layer_list if getattr(op, "is_transform", False))


def layers(trace: Trace) -> LayerAccessor:
    """Body of ``Trace.layers``; the record keeps the thin spelling and docstring."""

    # Memoized on the instance, never in a module global: a global
    # weak-keyed cache value reaches this Trace through the held records
    # and would pin it forever (the R37 ``trace.run()`` fork leak).
    # Same identity+len key scheme (and contract) as ``ops`` above:
    # reassignment changes identity, removal changes length, and the
    # in-place rename/refresh paths call the explicit invalidator
    # (r3 R52-1 -- the by-value key priced every HIT at O(n)).
    container = trace._layer_logs
    cache_entry = trace.__dict__.get(_TRACE_LAYER_ACCESSOR_ATTR)
    if cache_entry is not None and cache_entry[0] is container and cache_entry[1] == len(container):
        return cast("LayerAccessor", cache_entry[2])
    accessor = LayerAccessor(container, source_trace=trace)
    trace.__dict__[_TRACE_LAYER_ACCESSOR_ATTR] = (container, len(container), accessor)
    return accessor


def module_calls(trace: Trace) -> TraceModuleCallAccessor:
    """Body of ``Trace.module_calls``; the record keeps the thin spelling and docstring."""

    # Memoized on the instance, never in a module global: the accessor
    # reaches this Trace through ``ModuleCall._source_trace``, so a global
    # weak-keyed cache would pin every Trace forever.
    cached = trace.__dict__.get(_TRACE_MODULE_CALL_ACCESSOR_ATTR)
    if cached is not None:
        return cached
    calls: OrderedDict[str, Any] = OrderedDict()
    for module in trace._module_logs:
        for call in module.calls.values():
            calls[call.label] = call
    accessor = TraceModuleCallAccessor(calls)
    trace.__dict__[_TRACE_MODULE_CALL_ACCESSOR_ATTR] = accessor
    return accessor


def num_edges(trace: Trace) -> int:
    """Body of ``Trace.num_edges``; the record keeps the thin spelling and docstring."""

    # ``_resolved_op`` is the memoized form of ``trace.ops[child_label]``: a
    # stored child label is not an Op-accessor dict key, so resolving it per
    # edge otherwise repeats one full accessor lookup for every edge.
    resolve = trace.ops._resolved_op
    return len(
        {
            (op.label, resolve(child_label).label)
            for op in trace._layer_list
            for child_label in op.children
        }
    )


def graph_num_compute_edges(trace: Trace) -> int:
    """Body of ``Trace._graph_num_compute_edges``; the record keeps the thin spelling and docstring."""

    resolve = trace.ops._resolved_op
    compute_ops = trace.compute_ops
    compute_labels = {op.label for op in compute_ops}
    return len(
        {
            (op.label, resolve(child_label).label)
            for op in compute_ops
            for child_label in op.children
            if resolve(child_label).label in compute_labels
        }
    )


def graph_num_buffer_edges(trace: Trace) -> int:
    """Body of ``Trace._graph_num_buffer_edges``; the record keeps the thin spelling and docstring."""

    resolve = trace.ops._resolved_op
    return len(
        {
            (op.label, resolve(child_label).label)
            for op in trace._layer_list
            for child_label in op.children
            if op.is_buffer or resolve(child_label).is_buffer
        }
    )


def graph_num_layer_edges(trace: Trace) -> int:
    """Body of ``Trace._graph_num_layer_edges``; the record keeps the thin spelling and docstring."""

    return len(
        {
            (layer.layer_label, child_label)
            for layer in trace.layers
            for child_label in layer.children
        }
    )


def autograd_num_backward_edges(trace: Trace) -> int | None:
    """Body of ``Trace._autograd_num_backward_edges``; the record keeps the thin spelling and docstring."""

    if not trace._autograd_has_backward_pass:
        return None
    return len(
        {
            (grad_fn.label, child_label)
            for grad_fn in trace.autograd.grad_fns
            for child_label in grad_fn.children
        }
    )


def graph_branching_factor(trace: Trace) -> float:
    """Body of ``Trace._graph_branching_factor``; the record keeps the thin spelling and docstring."""

    compute_ops = list(trace.compute_ops)
    if not compute_ops:
        return 0.0
    return sum(len(op.children) for op in compute_ops) / len(compute_ops)


def graph_max_in_degree(trace: Trace) -> int:
    """Body of ``Trace._graph_max_in_degree``; the record keeps the thin spelling and docstring."""

    return max((len(op.parents) for op in trace.compute_ops), default=0)


def graph_max_out_degree(trace: Trace) -> int:
    """Body of ``Trace._graph_max_out_degree``; the record keeps the thin spelling and docstring."""

    return max((len(op.children) for op in trace.compute_ops), default=0)


def autograd_intermediate_derived_grads(trace: Trace) -> IntermediateDerivedGradAccessor:
    """Body of ``Trace._autograd_intermediate_derived_grads``; the record keeps the thin spelling."""

    records = trace.__dict__.get("_intermediate_derived_grads")
    if isinstance(records, IntermediateDerivedGradAccessor):
        return records
    return IntermediateDerivedGradAccessor()


def graph_orphans(trace: Trace) -> OrphanAccessor:
    """Body of ``Trace._graph_orphans``; the record keeps the thin spelling and docstring."""

    orphan_dict = OrderedDict(
        (log.layer_label, log) for log in trace._orphan_logs if getattr(log, "is_orphan", False)
    )
    return OrphanAccessor(orphan_dict)


def autograd_grad_fns(trace: Trace) -> GradFnAccessor:
    """Body of ``Trace._autograd_grad_fns``; the record keeps the thin spelling and docstring."""

    trace._sync_backward_projection_if_needed()
    return GradFnAccessor(trace._grad_fn_logs, trace._autograd_grad_fn_order)


def autograd_grad_fn_calls(trace: Trace) -> TraceGradFnCallAccessor:
    """Body of ``Trace._autograd_grad_fn_calls``; the record keeps the thin spelling and docstring."""

    trace._sync_backward_projection_if_needed()
    calls: OrderedDict[str, Any] = OrderedDict()
    for grad_fn_handle in trace.autograd.grad_fns:
        for call_index, call in grad_fn_handle.calls.items():
            call._source_trace = trace
            calls[f"{grad_fn_handle.label}:{call_index}"] = call
    return TraceGradFnCallAccessor(calls)


def autograd_backward_passes(trace: Trace) -> BackwardPassAccessor:
    """Body of ``Trace._autograd_backward_passes``; the record keeps the thin spelling and docstring."""

    raise_if_no_backward_capture(trace, plural_subject="backward_passes")
    trace._sync_backward_projection_if_needed()
    return BackwardPassAccessor(trace._backward_pass_logs)


def saved_ops(trace: Trace) -> Accessor[Op]:
    """Body of ``Trace.saved_ops``; the record keeps the thin spelling and docstring."""

    return TraceOpAccessor(
        [op for op in trace._layer_list if op.has_saved_activation],
        trace._layer_num_calls,
    )


def saved_grad_ops(trace: Trace) -> Accessor[Op]:
    """Body of ``Trace.saved_grad_ops``; the record keeps the thin spelling and docstring."""

    raise_if_no_backward_capture(trace, plural_subject="op-level saved gradients")
    return TraceOpAccessor(
        [op for op in trace._layer_list if op.has_grad],
        trace._layer_num_calls,
    )


def saved_layers(trace: Trace) -> Accessor[Layer]:
    """Body of ``Trace.saved_layers``; the record keeps the thin spelling and docstring."""

    return LayerAccessor(
        OrderedDict(
            (label, layer)
            for label, layer in trace._layer_logs.items()
            if any(op.has_saved_activation for op in layer.ops.values())
        ),
        source_trace=trace,
    )


def layers_saved_grad(trace: Trace) -> Accessor[Layer]:
    """Body of ``Trace._layers_saved_grad``; the record keeps the thin spelling and docstring."""

    return LayerAccessor(
        OrderedDict(
            (label, layer)
            for label, layer in trace._layer_logs.items()
            if any(op.has_grad for op in layer.ops.values())
        ),
        source_trace=trace,
    )


def saved_modules(trace: Trace) -> Accessor[Any]:
    """Body of ``Trace.saved_modules``; the record keeps the thin spelling and docstring."""

    saved_addresses = {call.address for call in trace.saved_module_calls}
    return ModuleAccessor(
        OrderedDict(
            (module.address, module)
            for module in trace.modules
            if module.address in saved_addresses
        )
    )


def saved_grad_modules(trace: Trace) -> Accessor[Any]:
    """Body of ``Trace.saved_grad_modules``; the record keeps the thin spelling and docstring."""

    saved_addresses = {call.address for call in trace.saved_grad_module_calls}
    return ModuleAccessor(
        OrderedDict(
            (module.address, module)
            for module in trace.modules
            if module.address in saved_addresses
        )
    )


def autograd_saved_grad_fn_calls(trace: Trace) -> Accessor[Any]:
    """Body of ``Trace._autograd_saved_grad_fn_calls``; the record keeps the thin spelling and docstring."""

    calls: OrderedDict[str, Any] = OrderedDict()
    for label, call in trace.autograd.grad_fn_calls.items():
        if (
            getattr(call, "grad_inputs", None) is not None
            or getattr(call, "grad_outputs", None) is not None
        ):
            calls[label] = call
    return TraceGradFnCallAccessor(calls)


def autograd_saved_grad_fns(trace: Trace) -> GradFnAccessor:
    """Body of ``Trace._autograd_saved_grad_fns``; the record keeps the thin spelling and docstring."""

    items = OrderedDict(
        (grad_fn_object_id, grad_fn_handle)
        for grad_fn_object_id, grad_fn_handle in trace._grad_fn_logs.items()
        if any(
            getattr(call, "grad_inputs", None) is not None
            or getattr(call, "grad_outputs", None) is not None
            for call in grad_fn_handle.calls.values()
        )
    )
    return GradFnAccessor(items, list(items))


def compute_ops(trace: Trace) -> Accessor[Op]:
    """Body of ``Trace.compute_ops``; the record keeps the thin spelling and docstring."""

    return TraceOpAccessor(
        [op for op in trace._layer_list if not (op.is_input or op.is_output or op.is_buffer)],
        trace._layer_num_calls,
    )


def layers_compute(trace: Trace) -> Accessor[Layer]:
    """Body of ``Trace._layers_compute``; the record keeps the thin spelling and docstring."""

    return LayerAccessor(
        OrderedDict(
            (label, layer)
            for label, layer in trace._layer_logs.items()
            if label in {op.layer_label for op in trace.compute_ops}
        ),
        source_trace=trace,
    )


def input_ops(trace: Trace) -> Accessor[Op]:
    """Body of ``Trace.input_ops``; the record keeps the thin spelling and docstring."""

    return TraceOpAccessor(
        boundary_label_ops(trace.input_layers, trace._layer_logs.__getitem__),
        trace._layer_num_calls,
    )


def output_ops(trace: Trace) -> Accessor[Op]:
    """Body of ``Trace.output_ops``; the record keeps the thin spelling and docstring."""

    return TraceOpAccessor(
        boundary_label_ops(trace.output_layers, trace._layer_logs.__getitem__),
        trace._layer_num_calls,
    )


def buffer_read_ops(trace: Trace) -> list[str]:
    """Body of ``Trace.buffer_read_ops``; the record keeps the thin spelling and docstring."""

    return [op.label for op in trace._layer_list if op.is_buffer and op.buffer_write_kind is None]


def buffer_write_ops(trace: Trace) -> list[str]:
    """Body of ``Trace.buffer_write_ops``; the record keeps the thin spelling and docstring."""

    return [
        op.label for op in trace._layer_list if op.is_buffer and op.buffer_write_kind is not None
    ]


def layers_internal_source(trace: Trace) -> Accessor[Layer]:
    """Body of ``Trace._layers_internal_source``; the record keeps the thin spelling and docstring."""

    return LayerAccessor(
        OrderedDict(
            (trace[label].layer_label, trace.layers[trace[label].layer_label])
            for label in trace.internal_source_ops
        ),
        source_trace=trace,
    )


def layers_internal_sink(trace: Trace) -> Accessor[Layer]:
    """Body of ``Trace._layers_internal_sink``; the record keeps the thin spelling and docstring."""

    return LayerAccessor(
        OrderedDict(
            (trace[label].layer_label, trace.layers[trace[label].layer_label])
            for label in trace.internal_sink_ops
        ),
        source_trace=trace,
    )


def num_uncalled_modules(trace: Trace) -> int | None:
    """Body of ``Trace.num_uncalled_modules``; the record keeps the thin spelling and docstring."""

    stored = trace.__dict__.get("_uncalled_module_addresses")
    if stored is not None:
        return len(stored)
    source_ref = None if trace.__dict__.get("_tl_cleaned_up", False) else trace._source_model_ref
    if source_ref is None or source_ref() is None:
        return None
    return len(trace.uncalled_modules)


def ops_with_params(trace: Trace) -> Accessor[Op]:
    """Body of ``Trace.ops_with_params``; the record keeps the thin spelling and docstring."""

    return TraceOpAccessor(
        [op for op in trace._layer_list if op.num_params > 0],
        trace._layer_num_calls,
    )


def module_calls_with_output_in(trace: Trace, op_labels: set[str]) -> Accessor[Any]:
    """Access ModuleCalls with at least one output Op among ``op_labels``.

    The root call's layer-space output labels resolve to their Op labels
    first (``op_label_aliases``), so the root call counts like any other.

    Parameters
    ----------
    trace:
        The owning trace.
    op_labels:
        Pass-qualified Op labels to match against each call's outputs.

    Returns
    -------
    Accessor[Any]
        Matching ModuleCalls keyed by call label, in call order.
    """

    from .._postprocess.saved_summary import op_label_aliases

    aliases = op_label_aliases(trace)
    calls: OrderedDict[str, Any] = OrderedDict()
    for call in trace.module_calls:
        if any(aliases.get(label, label) in op_labels for label in getattr(call, "output_ops", [])):
            calls[call.label] = call
    return TraceModuleCallAccessor(calls)


class _CallableList(list[Any]):
    """List that returns a plain list when called.

    This keeps rare report surfaces callable for user ergonomics without adding
    extra callable custom_methods to the Trace method state_history.
    """

    def __call__(self) -> list[Any]:
        """Return a plain-list copy of this report.

        Returns
        -------
        list[Any]
            Plain list containing this report's items.
        """

        return list(self)


def find_sites(trace: Trace, selector: Any, *, strict: bool = False, max_fanout: int = 8) -> Any:
    """Return a table of intervention sites matching a query.

    Parameters
    ----------
    trace:
        The owning trace.
    selector:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    max_fanout:
        Maximum number of matching sites.

    Returns
    -------
    _SiteTable
        Ordered table of matching layer-pass records.
    """

    from .._selection.resolver import find_sites

    return find_sites(trace, selector, strict=strict, max_fanout=max_fanout)


def suggest_layer_labels(trace: Trace, selector: str, *, limit: int = 10) -> list[str]:
    """Return layer labels matching a fuzzy query (the did-you-mean candidates of a refusal).

    Parameters
    ----------
    trace:
        The owning trace.
    selector:
        Layer-label substring or approximate layer name.
    limit:
        Maximum number of labels to return.

    Returns
    -------
    List[str]
        Matching no-pass layer labels in execution order, followed by close
        fuzzy matches when substring matches are insufficient.
    """

    query_text = str(selector).lower()
    labels = list(trace.layer_labels)
    substring_matches = [label for label in labels if query_text in label.lower()]
    if len(substring_matches) >= limit:
        return substring_matches[:limit]
    fuzzy_matches = difflib.get_close_matches(str(selector), labels, n=limit, cutoff=0.25)
    result = substring_matches[:]
    for label in fuzzy_matches:
        if label not in result:
            result.append(label)
        if len(result) >= limit:
            break
    return result


def find_nan(trace: Trace) -> Any:
    """Return the first NaN or Inf among saved outputs in execution order.

    Selectively saved traces identify this as the first finding among
    saved tensors and report unsaved ancestor operations as an uncertainty
    zone rather than claiming the true birth operation.

    Returns
    -------
    FindNanReport
        Structured non-finite diagnostic. Source locations are surfaced
        from the operation's existing recorded code context when available.
    """

    from ..debug._nan import find_nan_in_trace

    return find_nan_in_trace(trace)


def check_plan(trace: Trace, plan: Any) -> Any:
    """Audit-only plan check (nnsight-scan parity).

    Resolves each plan entry's selector against this trace, checks
    multiplicity, and compares declared replacement geometry (a meta
    tensor or any shape/dtype-bearing spec) against the recorded
    shape/dtype hypotheses. The report carries the source comparison
    digest, the evidence status, every checked site, and
    ``executable=False`` — checking a plan NEVER sets
    ``replayable``, arms, replays, or lifts the late-bind
    refusal. Refuses typed on callables, value-derived selection, and
    REFUTED sources. Spelling DOCUMENTED-UNSTABLE (the public verb name
    belongs to the UI sprint).

    Parameters
    ----------
    trace:
        The owning trace.
    plan:
        Iterable of selector entries or ``(selector, replacement)``
        pairs.

    Returns
    -------
    torchlens._capture._plan_check.PlanCheckReport
        Frozen audit report.
    """

    from .._capture._plan_check import check_plan

    return check_plan(trace, plan)


def uncalled_modules(trace: Trace) -> _CallableList:
    """Return registered modules that were not exercised in the captured pass.

    Returns
    -------
    _CallableList
        Module addresses present on the source model but absent from the
        captured module accessor, recorded when the capture finished.
        Returns an empty list when that record is missing (a loaded or
        unpickled trace) and the source model is no longer available;
        ``num_uncalled_modules`` reads ``None`` in that case.
    """

    stored = trace.__dict__.get("_uncalled_module_addresses")
    if stored is not None:
        return _CallableList(stored)
    source_ref = None if trace.__dict__.get("_tl_cleaned_up", False) else trace._source_model_ref
    model = source_ref() if source_ref is not None else None
    if model is None:
        return _CallableList()
    registered = {address or "trace" for address, _module in model.named_modules()}
    called = set(getattr(trace._module_logs, "_dict", {}).keys())
    called.update(getattr(trace._module_logs, "_alias_dict", {}).keys())
    return _CallableList(sorted(registered - called))


def resolve_site_table(
    trace: Trace, selector: Any, *, strict: bool = False, max_fanout: int = 8
) -> Any:
    """Resolve the record-grain site table for internal planning (private).

    The public door is ``trace[selector]``: a set-naming selector returns a
    ``tl.Selection`` of every match, never capped; a one-thing spelling
    returns its one record.

    Parameters
    ----------
    trace:
        The owning trace.
    selector:
        Selector, target spec, frozen target spec, or non-strict bare string.
    strict:
        Whether to reject non-portable query forms.
    max_fanout:
        Maximum number of matching sites.

    Returns
    -------
    _SiteTable
        Ordered table of matching layer-pass records.
    """

    from .._selection.resolver import _resolve_sites

    return _resolve_sites(trace, selector, strict=strict, max_fanout=max_fanout)
