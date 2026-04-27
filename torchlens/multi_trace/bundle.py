"""TraceBundle: container for N ModelLog instances with supergraph access.

The bundle is the core abstraction for any multi-pass analysis: aggregate
statistics across many forward passes, comparison of two or more traces,
or coverage analysis on dynamic networks.  One class handles both
shared-topology bundles (every trace traverses the same nodes) and
divergent-topology bundles (e.g. dynamic branching, MoE routing); the
shared case is just a degenerate Overlay where every node is universal.

Construction is cheap -- the bundle holds references, never copies tensors,
and the supergraph is built once in :meth:`__init__` then cached.

Visualization, counterfactual branch enumeration, intervention APIs, and
streaming aggregate are explicitly out of scope for this phase. See
``.project-context/todos.md`` for the deferred Phase 2 items.
"""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
)

import torch

from .metrics import is_scalar_like, relative_l1_scalar, resolve_metric
from .node_view import NodeView
from .topology import Supergraph, build_supergraph

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..data_classes.model_log import ModelLog


# Threshold above which we surface a "consider stats-only storage" warning.
# Chosen to match the spec; users with explicit intent (long-running sweeps)
# can suppress the warning via the standard ``warnings`` machinery.
_LARGE_BUNDLE_THRESHOLD = 100


class TraceBundle:
    """Holds N ``ModelLog`` instances and exposes aggregate / comparison APIs.

    Construct from a list of ``ModelLog`` objects (held by reference, never
    copied or mutated).  Optionally provide ``names`` to label each trace
    and ``groups`` to partition them.

    The supergraph is built at construction time.  Selective and universal
    node views are derived from it; per-node accessors are returned via
    ``bundle[node_name]``.

    Examples
    --------

    >>> import torchlens as tl
    >>> ml1 = tl.log_forward_pass(model, x1)
    >>> ml2 = tl.log_forward_pass(model, x2)
    >>> bundle = tl.bundle([ml1, ml2])
    >>> bundle.is_shared_topology
    True
    >>> bundle['relu_1_2'].activation.shape
    torch.Size([12, 4])  # batch_size 6 * 2 traces
    >>> bundle.most_changed(top_k=3)
    [('linear_1_1', 0.42), ('relu_1_2', 0.18), ('output_1', 0.05)]
    """

    def __init__(
        self,
        traces: List["ModelLog"],
        names: Optional[List[str]] = None,
        groups: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        if not isinstance(traces, list) or any(t is None for t in traces):
            raise TypeError("traces must be a list of ModelLog instances")
        if len(traces) == 0:
            raise ValueError("TraceBundle requires at least one ModelLog; got empty list.")

        # Default names
        if names is None:
            names = [f"trace_{i}" for i in range(len(traces))]
        else:
            if len(names) != len(traces):
                raise ValueError(
                    f"names length ({len(names)}) must equal traces length ({len(traces)})"
                )
            if len(set(names)) != len(names):
                raise ValueError("names must be unique")
        # Validate groups
        if groups is None:
            groups = {}
        else:
            name_set = set(names)
            for group_name, members in groups.items():
                if not isinstance(members, list):
                    raise TypeError(f"groups['{group_name}'] must be a list of trace names")
                unknown = [m for m in members if m not in name_set]
                if unknown:
                    raise ValueError(
                        f"groups['{group_name}'] references unknown trace names: "
                        f"{unknown}. Known: {sorted(name_set)}"
                    )

        self._traces: List["ModelLog"] = list(traces)
        self._names: List[str] = list(names)
        self._groups: Dict[str, List[str]] = {k: list(v) for k, v in groups.items()}

        if len(self._traces) > _LARGE_BUNDLE_THRESHOLD:
            warnings.warn(
                f"TraceBundle was constructed with {len(self._traces)} traces. "
                "Holding many full ModelLog instances in memory can be expensive; "
                "consider using stats-only storage upstream (e.g. trace with "
                "save_outputs reduced) if you only need aggregates.",
                stacklevel=2,
            )

        # Build supergraph once, cache.
        self._supergraph: Supergraph = build_supergraph(self._traces, self._names)

    # ------------------------------------------------------------------
    # Iteration / indexing
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of bundled ``ModelLog`` traces."""

        return len(self._traces)

    def __iter__(self) -> Iterator["ModelLog"]:
        """Iterate the underlying ``ModelLog`` instances in bundle order."""

        return iter(self._traces)

    def __getitem__(self, key: Union[str, int]) -> Union[NodeView, "ModelLog"]:
        """Look up a node by name or a trace by index.

        ``str`` arguments resolve to a :class:`NodeView` for the supergraph
        node with that canonical name.  ``int`` arguments resolve to the
        i-th ``ModelLog`` in bundle order.
        """

        if isinstance(key, int):
            return self._traces[key]
        if isinstance(key, str):
            node = self._supergraph.nodes.get(key)
            if node is None:
                raise KeyError(f"Node '{key}' not found in supergraph. Use `bundle.nodes` to list.")
            return NodeView(key, node, self._names)
        raise TypeError(f"TraceBundle indices must be int or str, got {type(key).__name__}")

    def __contains__(self, node_name: str) -> bool:
        """Whether ``node_name`` is present in the supergraph."""

        return node_name in self._supergraph.nodes

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def traces(self) -> List["ModelLog"]:
        """The bundled ``ModelLog`` instances, in construction order."""

        return list(self._traces)

    @property
    def names(self) -> List[str]:
        """Trace names in construction order."""

        return list(self._names)

    @property
    def groups(self) -> Dict[str, List[str]]:
        """A copy of the configured trace groups."""

        return {k: list(v) for k, v in self._groups.items()}

    @property
    def is_shared_topology(self) -> bool:
        """Whether every trace traversed every supergraph node."""

        n = len(self._names)
        for node in self._supergraph.nodes.values():
            if len(node.traces) != n:
                return False
        return True

    @property
    def nodes(self) -> List[str]:
        """Supergraph node names in topological order."""

        return list(self._supergraph.topological_order)

    @property
    def universal_nodes(self) -> List[str]:
        """Supergraph nodes traversed by ALL bundled traces."""

        n = len(self._names)
        return [
            name
            for name in self._supergraph.topological_order
            if len(self._supergraph.nodes[name].traces) == n
        ]

    @property
    def has_gradients(self) -> bool:
        """Whether any bundled trace has backward gradient data."""

        return any(getattr(ml, "has_gradients", False) for ml in self._traces)

    # ------------------------------------------------------------------
    # Selection / coverage
    # ------------------------------------------------------------------

    def selective_nodes(self, group: Optional[str] = None) -> List[str]:
        """Nodes traversed only by a subset of traces.

        With ``group=None`` (default), returns all nodes that are NOT
        universal -- i.e. trace-selective overall.  With ``group=<name>``,
        returns nodes traversed exactly by the traces in that group (and
        no others).  Empty list if topology is shared and ``group`` is
        ``None``.
        """

        if group is None:
            n = len(self._names)
            return [
                name
                for name in self._supergraph.topological_order
                if len(self._supergraph.nodes[name].traces) != n
            ]
        if group not in self._groups:
            raise KeyError(f"Unknown group '{group}'. Known groups: {sorted(self._groups)}")
        members = set(self._groups[group])
        out: List[str] = []
        for name in self._supergraph.topological_order:
            traces_at = set(self._supergraph.nodes[name].traces)
            # "Only traversed by traces in `group`" = traces_at is non-empty
            # AND a subset of members AND has at least one member.
            if traces_at and traces_at.issubset(members):
                out.append(name)
        return out

    def coverage(self, trace_name: str) -> float:
        """Fraction of supergraph nodes that ``trace_name`` traversed.

        Returns 0.0 if the supergraph is empty.  Raises ``KeyError`` if
        ``trace_name`` is not a known trace.
        """

        if trace_name not in self._names:
            raise KeyError(f"Unknown trace '{trace_name}'. Known: {self._names}")
        total = len(self._supergraph.nodes)
        if total == 0:
            return 0.0
        count = sum(1 for node in self._supergraph.nodes.values() if trace_name in node.layer_refs)
        return count / total

    # ------------------------------------------------------------------
    # Aggregate analysis
    # ------------------------------------------------------------------

    def most_changed(
        self,
        top_k: int = 10,
        metric: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "cosine",
        on: Literal["activation", "gradient"] = "activation",
    ) -> List[tuple[str, float]]:
        """Rank nodes by mean pairwise distance across traversing traces.

        Skips nodes traversed by fewer than 2 traces (no pair to compare).
        Returns ``(node_name, score)`` tuples sorted by descending score.
        ``top_k`` may be larger than the number of eligible nodes; the
        result simply truncates to as many as exist.
        """

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        metric_fn = resolve_metric(metric)
        scored: List[tuple[str, float]] = []
        for name in self._supergraph.topological_order:
            node = self._supergraph.nodes[name]
            tensors: List[torch.Tensor] = []
            for trace_name in node.traces:
                layer = node.layer_refs[trace_name]
                if on == "activation":
                    has = getattr(layer, "has_saved_activations", False)
                    value = getattr(layer, "activation", None) if has else None
                else:
                    has = getattr(layer, "has_gradient", False)
                    value = getattr(layer, "gradient", None) if has else None
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
            if len(tensors) < 2:
                continue
            distances: List[float] = []
            for i in range(len(tensors)):
                for j in range(i + 1, len(tensors)):
                    ti, tj = tensors[i], tensors[j]
                    if is_scalar_like(ti) or is_scalar_like(tj):
                        d = relative_l1_scalar(ti, tj)
                    else:
                        d = metric_fn(ti, tj)
                    distances.append(float(d.detach().item()))
            if not distances:
                continue
            mean_d = sum(distances) / len(distances)
            scored.append((name, mean_d))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def aggregate(
        self,
        node: str,
        statistic: Literal["mean", "std", "var", "norm"] = "mean",
        on: Literal["activation", "gradient"] = "activation",
    ) -> torch.Tensor:
        """Compute an aggregate statistic at a single supergraph node.

        Thin wrapper around ``bundle[node].aggregate(...)``.  Raises
        ``KeyError`` if the node is unknown.
        """

        view = self[node]
        if not isinstance(view, NodeView):  # pragma: no cover - defensive
            raise TypeError("aggregate requires a node name (str), not a trace index")
        return view.aggregate(statistic=statistic, on=on)

    # ------------------------------------------------------------------
    # Hard checks
    # ------------------------------------------------------------------

    def assert_shared_topology(self) -> None:
        """Raise ``ValueError`` if topology is not shared across all traces.

        Useful as a guard in code that assumes universal coverage and would
        otherwise fail with a less informative error from a downstream
        accessor.
        """

        if not self.is_shared_topology:
            n = len(self._names)
            divergent = [
                name for name, node in self._supergraph.nodes.items() if len(node.traces) != n
            ]
            raise ValueError(
                f"TraceBundle is not shared-topology: {len(divergent)} "
                f"node(s) are traversed by only some traces. Examples: "
                f"{divergent[:5]}"
            )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_traces = len(self._traces)
        n_nodes = len(self._supergraph.nodes)
        n_universal = len(self.universal_nodes)
        shared = self.is_shared_topology
        has_grads = self.has_gradients
        return (
            f"TraceBundle(N={n_traces} traces, {n_nodes} supergraph nodes, "
            f"{n_universal} universal, shared_topology={shared}, "
            f"has_gradients={has_grads})"
        )
