"""Concern-group views of a ``Trace``: ``trace.cost``, ``trace.graph``, ...

Each group is a light read-only view over its Trace; the facts it
exposes are computed by the Trace's private implementation members.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trace import Trace


class _TraceGroup:
    """Base for the read-only concern groups of one Trace."""

    __slots__ = ("_trace",)

    def __init__(self, trace: Trace) -> None:
        """Bind the group to ``trace``."""

        self._trace = trace

    def __repr__(self) -> str:
        """Name the group and its Trace."""

        label = getattr(self._trace, "trace_label", None)
        return f"<{type(self).__name__} of Trace {label!r}>"

    def __dir__(self) -> list[str]:
        """List the group's public members."""

        return sorted(n for n in dir(type(self)) if not n.startswith("_"))


class TraceCost(_TraceGroup):
    """Compute, memory and timing totals of a Trace."""

    __slots__ = ()

    @property
    def flops_by_op_type(self) -> Any:
        """Group FLOPs by layer type."""

        return self._trace._cost_flops_by_op_type

    @property
    def forward_peak_memory_pair(self) -> Any:
        """Session-time (live, resident) forward peak pair with backend named."""

        return self._trace._cost_forward_peak_memory_pair

    @property
    def last_backward_duration(self) -> Any:
        """Most recent backward-pass duration, if any."""

        return self._trace._cost_last_backward_duration

    @property
    def macs_by_kind(self) -> Any:
        """Group TRUE forward MACs by layer type."""

        return self._trace._cost_macs_by_kind

    @property
    def overhead_duration(self) -> Any:
        """Time spent on TorchLens overhead (total minus function calls)."""

        return self._trace._cost_overhead_duration

    @property
    def total_backward_duration(self) -> Any:
        """Sum of all captured backward-pass durations."""

        return self._trace._cost_total_backward_duration

    @property
    def total_flops(self) -> Any:
        """Total FLOPs (forward + backward)."""

        return self._trace._cost_total_flops

    @property
    def total_flops_backward(self) -> Any:
        """Total backward FLOPs across all layers (skipping None/unknown)."""

        return self._trace._cost_total_flops_backward

    @property
    def total_flops_forward(self) -> Any:
        """Total forward FLOPs (the canonical identity-partition aggregation)."""

        return self._trace._cost_total_flops_forward

    @property
    def total_macs(self) -> Any:
        """Total MACs are not derivable and always ``None`` (see total_macs_backward)."""

        return self._trace._cost_total_macs

    @property
    def total_macs_backward(self) -> Any:
        """Backward MACs are not derivable and always ``None`` (estimates only)."""

        return self._trace._cost_total_macs_backward

    @property
    def total_macs_forward(self) -> Any:
        """Total TRUE forward MACs over ops with a known MAC split."""

        return self._trace._cost_total_macs_forward


class TraceGraph(_TraceGroup):
    """Graph-shape facts of a Trace: edges, degrees, branching and conditionals."""

    __slots__ = ()

    @property
    def branching_factor(self) -> Any:
        """Mean fan-out: children (consumers) per compute Op."""

        return self._trace._graph_branching_factor

    @property
    def edges(self) -> Any:
        """Return this trace's dataflow edge family (finalized edge views)."""

        return self._trace._graph_edges

    @property
    def has_conditional_branching(self) -> Any:
        """Whether any layer is in a conditional branch."""

        return self._trace._graph_has_conditional_branching

    @property
    def has_conditionals(self) -> Any:
        """Whether this Trace contains conditional-flow records."""

        return self._trace._graph_has_conditionals

    @property
    def is_branching(self) -> Any:
        """Whether any layer has more than one child."""

        return self._trace._graph_is_branching

    @property
    def is_dynamic(self) -> Any:
        """Whether execution depends on runtime tensor values."""

        return self._trace._graph_is_dynamic

    @property
    def is_recurrent(self) -> Any:
        """Whether any layer has more than one pass."""

        return self._trace._graph_is_recurrent

    @property
    def max_in_degree(self) -> Any:
        """Maximum number of parents over compute Ops."""

        return self._trace._graph_max_in_degree

    @property
    def max_ops_per_layer(self) -> Any:
        """Maximum number of ops for any layer."""

        return self._trace._graph_max_ops_per_layer

    @property
    def max_out_degree(self) -> Any:
        """Maximum number of children over compute Ops."""

        return self._trace._graph_max_out_degree

    @property
    def num_buffer_edges(self) -> Any:
        """Distinct Op graph edges with at least one buffer endpoint."""

        return self._trace._graph_num_buffer_edges

    @property
    def num_compute_edges(self) -> Any:
        """Distinct Op graph edges whose endpoints are both compute Ops."""

        return self._trace._graph_num_compute_edges

    @property
    def num_layer_edges(self) -> Any:
        """Distinct edges in the aggregate Layer graph."""

        return self._trace._graph_num_layer_edges

    @property
    def orphans(self) -> Any:
        """Access retained orphan island operations by raw or final label."""

        return self._trace._graph_orphans

    @property
    def tied_param_groups(self) -> Any:
        """Alias-address groups of parameters registered at more than one path."""

        return self._trace._graph_tied_param_groups


class TraceAutograd(_TraceGroup):
    """Backward-pass records and grad_fn facts of a Trace."""

    __slots__ = ()

    @property
    def backward_passes(self) -> Any:
        """Access backward pass records by 0-based position or named pass number."""

        return self._trace._autograd_backward_passes

    @property
    def grad_fn_calls(self) -> Any:
        """Access per-invocation GradFnCall records by qualified label or index."""

        return self._trace._autograd_grad_fn_calls

    @property
    def grad_fn_fire_timings(self) -> Any:
        """Return live per-fire backward timing spans keyed like ``grad_fn_calls``."""

        return self._trace._autograd_grad_fn_fire_timings

    @property
    def grad_fns(self) -> Any:
        """Access backward grad_fn_handle metadata by label, index, pass label, or substring."""

        return self._trace._autograd_grad_fns

    @property
    def intermediate_derived_grads(self) -> Any:
        """Access exact op-level derived gradient records, when a backend provides them."""

        return self._trace._autograd_intermediate_derived_grads

    @property
    def last_backward_pass(self) -> Any:
        """Return the most recent backward pass record, if any."""

        return self._trace._autograd_last_backward_pass

    @property
    def last_backward_root_grad_fn_object_id(self) -> Any:
        """Most recent backward root grad_fn_handle object id, if any."""

        return self._trace._autograd_last_backward_root_grad_fn_object_id

    @property
    def num_backward_edges(self) -> Any:
        """Distinct edges in the backward GradFn graph, or ``None`` if ungated."""

        return self._trace._autograd_num_backward_edges

    @property
    def saved_grad_fn_calls(self) -> Any:
        """Access GradFnCall records with saved gradient inputs or outputs."""

        return self._trace._autograd_saved_grad_fn_calls

    @property
    def saved_grad_fns(self) -> Any:
        """Access GradFns containing at least one saved GradFnCall."""

        return self._trace._autograd_saved_grad_fns


class TraceSource(_TraceGroup):
    """Source facts of the traced model class: locations, signatures, docstrings."""

    __slots__ = ()

    @property
    def class_location(self) -> Any:
        """Combined model class source location."""

        return self._trace._source_class_location

    @property
    def cls(self) -> Any:
        """Return the live source model class when the model is still alive."""

        return self._trace._source_cls

    @property
    def forward_location(self) -> Any:
        """Combined forward source location."""

        return self._trace._source_forward_location

    @property
    def init_location(self) -> Any:
        """Combined model ``__init__`` source location."""

        return self._trace._source_init_location
