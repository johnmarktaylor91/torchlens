"""The graph group of an ``Op`` or ``Layer``: ``op.graph.has_parents``, ...

Where a record sits in the computational graph: parent, child, sibling and
co-parent relations, input and output ancestry, distances from inputs and to
outputs, buffer sources and sinks, and equivalence classes. Stored members read
the record's private storage attribute (``_graph_<leaf>``, same persisted
column as before); derived members compute on the record when read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._utils._record_group_table import OP_GROUP_VIEWS
from ._record_group_base import _LayerGroup, _RecordGroup

if TYPE_CHECKING:
    from .layer import Layer
    from .op import Op


class OpGraph(_RecordGroup):
    """Graph position of one Op: relations, ancestry, distances, buffers and equivalence.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.ops["linear_1_1"].graph, tl.types.OpGraph)
    True
    """

    __slots__ = ()
    _record: Op

    @property
    def buffer_sink_ops(self) -> Any:
        """Accessor over the buffer-write Ops that directly consume this Op."""

        return self._record._graph_buffer_sink_ops

    @property
    def buffer_source(self) -> Any:
        """Label of the Op that wrote this buffer value, or `None` for a static buffer."""

        return self._record._graph_buffer_source

    @property
    def buffer_source_func_name(self) -> Any:
        """Name of the function that produced this buffer value."""

        return self._record._graph_buffer_source_func_name

    @property
    def buffer_source_ops(self) -> Any:
        """Accessor over the buffer-source Ops that feed this Op."""

        return self._record._graph_buffer_source_ops

    @property
    def co_parents(self) -> Any:
        """Labels of other Ops that share at least one child with this Op."""

        return self._record._graph_co_parents

    @property
    def edge_substitutions(self) -> Any:
        """Recorded parent-edge substitutions for this Op, keyed by argument slot."""

        return self._record._graph_edge_substitutions

    @property
    def equivalence_class(self) -> Any:
        """Key of the equivalence class that loop detection assigned to this Op."""

        return self._record._graph_equivalence_class

    @property
    def equivalent_ops(self) -> Any:
        """Labels of other Ops in the same equivalence class (same function and arguments)."""

        return self._record._graph_equivalent_ops

    @property
    def has_children(self) -> Any:
        """Whether this Op has any child Ops."""

        return self._record._graph_has_children

    @property
    def has_co_parents(self) -> Any:
        """Whether this Op shares children with other Ops."""

        return self._record._graph_has_co_parents

    @property
    def has_input_ancestor(self) -> Any:
        """Whether any model input is an ancestor of this Op."""

        return self._record._graph_has_input_ancestor

    @property
    def has_internal_source_ancestor(self) -> Any:
        """Whether any internally initialized source Op is an ancestor of this Op."""

        return self._record._graph_has_internal_source_ancestor

    @property
    def has_output_descendant(self) -> Any:
        """Whether any model output is a descendant of this Op."""

        return self._record._graph_has_output_descendant

    @property
    def has_parents(self) -> Any:
        """Whether this Op has any parent Ops."""

        return self._record._graph_has_parents

    @property
    def has_siblings(self) -> Any:
        """Whether this Op shares parents with other Ops."""

        return self._record._graph_has_siblings

    @property
    def internal_source_ancestors(self) -> Any:
        """Labels of the internally initialized source Ops that are ancestors of this Op."""

        return self._record._graph_internal_source_ancestors

    @property
    def internal_source_parents(self) -> Any:
        """Labels of the internally initialized source Ops that are direct parents of this Op."""

        return self._record._graph_internal_source_parents

    @property
    def io_role(self) -> Any:
        """Input or output address this boundary Op stands for, or ``None``."""

        return self._record._graph_io_role

    @property
    def is_buffer(self) -> Any:
        """Stored flag marking this Op as a buffer boundary; `is_buffer_source` exposes it."""

        return self._record._graph_is_buffer

    @property
    def is_compute(self) -> Any:
        """Whether this Op executed a torch function rather than being a boundary sentinel."""

        return self._record._graph_is_compute

    @property
    def is_final_output(self) -> Any:
        """Whether this Op is a final model output."""

        return self._record._graph_is_final_output

    @property
    def is_internal_source(self) -> Any:
        """Whether this Op was created inside the model, such as by a factory call, rather than fed from an input."""

        return self._record._graph_is_internal_source

    @property
    def is_orphan(self) -> Any:
        """Whether this Op is disconnected from both the input and output flows."""

        return self._record._graph_is_orphan

    @property
    def is_output_parent(self) -> Any:
        """Whether this Op is directly returned from `forward`."""

        return self._record._graph_is_output_parent

    @property
    def max_distance_from_input(self) -> Any:
        """Longest graph distance from any model input to this Op."""

        return self._record._graph_max_distance_from_input

    @property
    def max_distance_to_output(self) -> Any:
        """Longest graph distance from this Op to any model output."""

        return self._record._graph_max_distance_to_output

    @property
    def min_distance_from_input(self) -> Any:
        """Shortest graph distance from any model input to this Op."""

        return self._record._graph_min_distance_from_input

    @property
    def min_distance_to_output(self) -> Any:
        """Shortest graph distance from this Op to any model output."""

        return self._record._graph_min_distance_to_output

    @property
    def num_inputs(self) -> Any:
        """Number of parent Ops of this Op."""

        return self._record._graph_num_inputs

    @property
    def parent_param_ops(self) -> Any:
        """Mapping from each consumed parameter's identifying barcode to its pass number."""

        return self._record._graph_parent_param_ops

    @property
    def parent_params(self) -> Any:
        """The `nn.Parameter` objects consumed by this Op."""

        return self._record._graph_parent_params

    @property
    def recurrent_ops(self) -> Any:
        """Labels of the Ops in the same recurrent group: equivalent and at a structurally corresponding position."""

        return self._record._graph_recurrent_ops

    @property
    def root_ancestors(self) -> Any:
        """Labels of the root Ops at the start of the graph that this Op descends from."""

        return self._record._graph_root_ancestors

    @property
    def siblings(self) -> Any:
        """Labels of other Ops that share at least one parent with this Op."""

        return self._record._graph_siblings


class LayerGraph(_LayerGroup):
    """Graph position of one Layer: relations, ancestry, per-pass edges and equivalence.

    Examples
    --------
    >>> import torch, torchlens as tl
    >>> trace = tl.trace(torch.nn.Linear(3, 2), torch.randn(1, 3))
    >>> isinstance(trace.layers["linear_1"].graph, tl.types.LayerGraph)
    True
    """

    __slots__ = ()
    _record: Layer
    _GROUP = "graph"

    @property
    def buffer_pass(self) -> Any:
        """Sequential number of this buffer write among the writes to the same buffer address, starting at 1, or `None` for non-writes."""

        return self._record._graph_buffer_pass

    @property
    def buffer_source(self) -> Any:
        """The label of the op that wrote this buffer value, or `None` for static buffers."""

        return self._record._graph_buffer_source

    @property
    def buffer_source_func_name(self) -> Any:
        """Name of the function that produced this buffer write, or `None` when this layer is not a buffer write."""

        return self._record._graph_buffer_source_func_name

    @property
    def child_ops_per_layer(self) -> Any:
        """Mapping from each child layer label to the pass numbers of this layer's ops that connect to it."""

        return self._record._graph_child_ops_per_layer

    @property
    def children_per_pass(self) -> Any:
        """Mapping from pass number to the child layer labels (without pass suffixes) of that pass."""

        return self._record._graph_children_per_pass

    @property
    def equivalence_class(self) -> Any:
        """Fingerprint string that groups ops repeating the same operation, used for loop detection."""

        return self._record._graph_equivalence_class

    @property
    def equivalent_ops(self) -> Any:
        """Labels of the ops that share this layer's equivalence class."""

        return self._record._graph_equivalent_ops

    @property
    def has_children(self) -> Any:
        """Whether any pass of this layer has child layers."""

        return self._record._graph_has_children

    @property
    def has_co_parents(self) -> Any:
        """Whether this layer has any co-parent layers besides itself."""

        return self._record._graph_has_co_parents

    @property
    def has_input_ancestor(self) -> Any:
        """Whether any model input is an ancestor of this layer."""

        return self._record._graph_has_input_ancestor

    @property
    def has_parents(self) -> Any:
        """Whether any pass of this layer has parent layers."""

        return self._record._graph_has_parents

    @property
    def has_siblings(self) -> Any:
        """Whether this layer has any sibling layers besides itself."""

        return self._record._graph_has_siblings

    @property
    def io_role(self) -> Any:
        """Input or output address this boundary Layer stands for, or ``None``."""

        return self._record._graph_io_role

    @property
    def is_buffer(self) -> Any:
        """Whether this layer is a registered-buffer version node, the flag `is_buffer_source` reports."""

        return self._record._graph_is_buffer

    @property
    def is_compute(self) -> Any:
        """Whether this layer's representative op is a compute op."""

        return self._record._graph_is_compute

    @property
    def is_final_output(self) -> Any:
        """Whether this layer is one of the model's final outputs."""

        return self._record._graph_is_final_output

    @property
    def is_internal_sink(self) -> Any:
        """Whether this layer's value ends inside the model without reaching any output."""

        return self._record._graph_is_internal_sink

    @property
    def is_internal_source(self) -> Any:
        """Whether this layer creates a tensor inside the model instead of receiving it from a model input."""

        return self._record._graph_is_internal_source

    @property
    def is_orphan(self) -> Any:
        """Whether any op in this layer is disconnected from the main graph."""

        return self._record._graph_is_orphan

    @property
    def parent_ops_per_layer(self) -> Any:
        """Mapping from each parent layer label to the pass numbers of this layer's ops that connect from it."""

        return self._record._graph_parent_ops_per_layer

    @property
    def parents_per_pass(self) -> Any:
        """Mapping from pass number to the parent layer labels (without pass suffixes) of that pass."""

        return self._record._graph_parents_per_pass

    @property
    def site_peers(self) -> Any:
        """Layers sharing any of this Layer's site keys (live index)."""

        return self._record._graph_site_peers


OP_GROUP_VIEWS["graph"] = OpGraph

__all__ = ["LayerGraph", "OpGraph"]
