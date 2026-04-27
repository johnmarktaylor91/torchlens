"""NodeView: per-node accessor for a TraceBundle.

A :class:`NodeView` is a lightweight, view-style object returned by
``bundle[node_name]``.  It carries enough context to answer per-trace queries
without holding any tensor copies; tensors are read on demand from the
underlying ``LayerLog`` references in the supergraph.

Two flavours of accessor are provided per data field:

* ``.activations`` / ``.gradients`` -- list view, length == number of traces
  that traversed this node, never raises on partial coverage; entries are
  ``None`` when a traversing trace has no stored tensor.
* ``.activation`` / ``.gradient`` -- stacked tensor view, requires every
  bundled trace to have traversed the node and produced consistent shapes.
  Raises ``ValueError`` with a hint pointing the caller to the list form
  otherwise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Literal, Optional, Set, Union

import torch

from .metrics import (
    is_scalar_like,
    relative_l1_scalar,
    resolve_metric,
)

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from .topology import SupergraphNode


_TENSOR_FIELD_LITERAL = Literal["activation", "gradient"]


class NodeView:
    """Read-only view of one supergraph node across all bundled traces.

    Construct via ``bundle[node_name]`` -- not directly. Holds references
    only; never copies tensors. Cheap to recreate, so callers should not
    cache instances.
    """

    def __init__(
        self,
        node_name: str,
        node: "SupergraphNode",
        bundle_trace_names: List[str],
    ) -> None:
        self._node_name = node_name
        self._node = node
        # The full bundle ordering of trace names so list views align even
        # when some traces are absent from this node.
        self._bundle_trace_names = list(bundle_trace_names)

    # ------------------------------------------------------------------
    # Identity / metadata
    # ------------------------------------------------------------------

    @property
    def node_name(self) -> str:
        """Canonical supergraph node name."""

        return self._node_name

    @property
    def traces(self) -> Set[str]:
        """Names of bundled traces that traversed this node."""

        return set(self._node.traces)

    @property
    def coverage(self) -> float:
        """Fraction of bundled traces that traversed this node, in ``[0, 1]``."""

        if not self._bundle_trace_names:
            return 0.0
        return len(self._node.traces) / len(self._bundle_trace_names)

    @property
    def op_type(self) -> str:
        """Operation type, e.g. ``'relu'``, ``'conv2d'``, ``'matmul'``."""

        return self._node.op_type

    @property
    def module_path(self) -> Optional[str]:
        """``containing_module`` from the underlying LayerLog, or ``None``."""

        return self._node.module_path

    @property
    def shape(self) -> tuple:
        """Tensor shape excluding the batch dim.

        Verifies consistency across all traversing traces; raises
        ``ValueError`` on mismatch.  Returns an empty tuple ``()`` for
        scalar (0-d) outputs.
        """

        shapes: list[tuple] = []
        for trace_name in self._node.traces:
            layer = self._node.layer_refs[trace_name]
            t_shape = getattr(layer, "tensor_shape", None)
            if t_shape is None:
                continue
            t_shape_tuple = tuple(t_shape)
            if len(t_shape_tuple) == 0:
                shapes.append(())
            else:
                shapes.append(tuple(t_shape_tuple[1:]))
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            raise ValueError(
                f"Shape mismatch across traces at node '{self._node_name}': {sorted(unique_shapes)}"
            )
        if not shapes:
            return ()
        return shapes[0]

    # ------------------------------------------------------------------
    # Tensor accessors -- list form (always works)
    # ------------------------------------------------------------------

    def _tensor_list(self, field: _TENSOR_FIELD_LITERAL) -> List[Optional[torch.Tensor]]:
        """Return the per-trace tensor list for the requested field.

        Iterates in bundle-trace order; entries for traces that did not
        traverse this node are simply omitted (NOT padded with None) so the
        list length equals ``len(self.traces)``. For traversing traces that
        lack a stored value (e.g. ``save_outputs=False`` at trace time), the
        entry is ``None``.
        """

        out: List[Optional[torch.Tensor]] = []
        # Iterate in bundle-trace order to keep the list deterministic.
        for trace_name in self._bundle_trace_names:
            if trace_name not in self._node.layer_refs:
                continue
            layer = self._node.layer_refs[trace_name]
            if field == "activation":
                has = getattr(layer, "has_saved_activations", False)
                value = getattr(layer, "activation", None) if has else None
            else:  # gradient
                has = getattr(layer, "has_gradient", False)
                value = getattr(layer, "gradient", None) if has else None
            out.append(value if isinstance(value, torch.Tensor) else None)
        return out

    @property
    def activations(self) -> List[Optional[torch.Tensor]]:
        """Per-trace activation tensors, in bundle-trace order.

        Length equals ``len(self.traces)``. Entries are ``None`` for
        traversing traces with no stored activation.
        """

        return self._tensor_list("activation")

    @property
    def gradients(self) -> List[Optional[torch.Tensor]]:
        """Per-trace gradient tensors, in bundle-trace order.

        Same contract as :attr:`activations` but for gradients.
        """

        return self._tensor_list("gradient")

    # ------------------------------------------------------------------
    # Tensor accessors -- stacked form (strict)
    # ------------------------------------------------------------------

    def _stacked(self, field: _TENSOR_FIELD_LITERAL) -> torch.Tensor:
        """Return a single stacked tensor across all bundled traces.

        Strict contract: every trace in the bundle must have traversed this
        node AND have a stored tensor AND share the same shape excluding
        the batch dim. Otherwise raises ``ValueError`` with a hint pointing
        to the list form.
        """

        missing: List[str] = []
        no_stored: List[str] = []
        tensors: List[torch.Tensor] = []
        per_trace_shapes: List[tuple] = []

        for trace_name in self._bundle_trace_names:
            if trace_name not in self._node.layer_refs:
                missing.append(trace_name)
                continue
            layer = self._node.layer_refs[trace_name]
            if field == "activation":
                has = getattr(layer, "has_saved_activations", False)
                value = getattr(layer, "activation", None) if has else None
            else:
                has = getattr(layer, "has_gradient", False)
                value = getattr(layer, "gradient", None) if has else None
            if not isinstance(value, torch.Tensor):
                no_stored.append(trace_name)
                continue
            tensors.append(value)
            shape = tuple(value.shape)
            per_trace_shapes.append(shape[1:] if len(shape) > 0 else ())

        list_attr = "activations" if field == "activation" else "gradients"

        if missing:
            raise ValueError(
                f"Cannot stack '{field}' for node '{self._node_name}': not all "
                f"bundled traces traversed it ({len(missing)} missing). Use "
                f"`bundle['{self._node_name}'].{list_attr}` for the list form."
            )
        if no_stored:
            raise ValueError(
                f"Cannot stack '{field}' for node '{self._node_name}': "
                f"{len(no_stored)} trace(s) have no stored {field}. Use "
                f"`bundle['{self._node_name}'].{list_attr}` for the list form."
            )
        unique_shapes = set(per_trace_shapes)
        if len(unique_shapes) > 1:
            raise ValueError(
                f"Cannot stack '{field}' for node '{self._node_name}': shapes "
                f"differ across traces (excluding batch dim): {sorted(unique_shapes)}."
                f" Use `bundle['{self._node_name}'].{list_attr}` for the list form."
            )

        if not tensors:
            raise ValueError(
                f"Cannot stack '{field}' for node '{self._node_name}': no traces"
                f" provided this node."
            )

        # 0-d outputs cannot be concatenated along a "batch" dim -- stack instead.
        if tensors[0].dim() == 0:
            return torch.stack(tensors, dim=0)
        return torch.cat(tensors, dim=0)

    @property
    def activation(self) -> torch.Tensor:
        """Single stacked activation tensor across all traces.

        Concatenated along the batch dim (dim 0). Raises ``ValueError`` if
        not every trace traversed this node, any traversing trace lacks a
        stored activation, or shapes disagree.
        """

        return self._stacked("activation")

    @property
    def gradient(self) -> torch.Tensor:
        """Single stacked gradient tensor across all traces.

        Same contract as :attr:`activation` but for gradients.
        """

        return self._stacked("gradient")

    # ------------------------------------------------------------------
    # Per-node analysis
    # ------------------------------------------------------------------

    def _select_tensors(self, on: _TENSOR_FIELD_LITERAL) -> tuple[list[str], list[torch.Tensor]]:
        """Return (trace_names, tensors) restricted to traces with usable data."""

        usable_names: List[str] = []
        usable_tensors: List[torch.Tensor] = []
        for trace_name in self._bundle_trace_names:
            if trace_name not in self._node.layer_refs:
                continue
            layer = self._node.layer_refs[trace_name]
            if on == "activation":
                has = getattr(layer, "has_saved_activations", False)
                value = getattr(layer, "activation", None) if has else None
            else:
                has = getattr(layer, "has_gradient", False)
                value = getattr(layer, "gradient", None) if has else None
            if isinstance(value, torch.Tensor):
                usable_names.append(trace_name)
                usable_tensors.append(value)
        return usable_names, usable_tensors

    def diff(
        self,
        other: Optional[str] = None,
        metric: Union[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = "cosine",
        on: _TENSOR_FIELD_LITERAL = "activation",
    ) -> torch.Tensor:
        """Pairwise distances between traversing traces at this node.

        Parameters
        ----------
        other:
            Name of a single trace.  When ``None`` (default), returns a
            ``KxK`` matrix of pairwise distances across the K traces that
            traversed this node.  When provided, returns a ``1xK`` row of
            distances from ``other`` to every traversing trace.
        metric:
            ``'cosine'`` (default), ``'relative_l2'``, ``'pearson'``, or any
            callable taking ``(Tensor, Tensor) -> Tensor``.
        on:
            ``'activation'`` (default) or ``'gradient'``.

        Notes
        -----
        For scalar outputs (0-d or 1-element tensors), automatically falls
        back to :func:`relative_l1_scalar` regardless of the requested
        metric. Cosine and Pearson are meaningless on scalars.
        """

        metric_fn = resolve_metric(metric)
        usable_names, usable_tensors = self._select_tensors(on)
        k = len(usable_tensors)

        if other is not None:
            if other not in self._bundle_trace_names:
                raise ValueError(
                    f"Unknown trace name '{other}'. Bundle traces: {self._bundle_trace_names}"
                )
            if other not in usable_names:
                raise ValueError(
                    f"Trace '{other}' has no usable {on} at node "
                    f"'{self._node_name}' (did not traverse, or no stored value)."
                )
            row = torch.zeros(1, max(k, 1))
            ref_idx = usable_names.index(other)
            ref = usable_tensors[ref_idx]
            for j, t in enumerate(usable_tensors):
                if j == ref_idx:
                    # Self-comparison is exactly zero by construction; skip
                    # the metric to avoid floating-point near-zero noise.
                    continue
                if is_scalar_like(ref) or is_scalar_like(t):
                    val = relative_l1_scalar(ref, t)
                else:
                    val = metric_fn(ref, t)
                row[0, j] = float(val.detach().item())
            return row

        # Square pairwise matrix
        out = torch.zeros(k, k)
        for i in range(k):
            for j in range(i, k):
                if i == j:
                    out[i, j] = 0.0
                    continue
                ti = usable_tensors[i]
                tj = usable_tensors[j]
                if is_scalar_like(ti) or is_scalar_like(tj):
                    val = relative_l1_scalar(ti, tj)
                else:
                    val = metric_fn(ti, tj)
                fval = float(val.detach().item())
                out[i, j] = fval
                out[j, i] = fval
        return out

    def aggregate(
        self,
        statistic: Literal["mean", "std", "var", "norm"] = "mean",
        on: _TENSOR_FIELD_LITERAL = "activation",
    ) -> torch.Tensor:
        """Compute an aggregate statistic across all traces at this node.

        Stacks the per-trace tensors along a new leading axis and reduces
        along it.  Requires every traversing trace to have a stored tensor
        and matching shapes; raises ``ValueError`` otherwise.
        """

        usable_names, usable_tensors = self._select_tensors(on)
        if not usable_tensors:
            raise ValueError(
                f"No traces have stored {on} at node '{self._node_name}'; cannot aggregate."
            )
        # Verify shape compatibility.
        shapes = {tuple(t.shape) for t in usable_tensors}
        if len(shapes) > 1:
            raise ValueError(
                f"Cannot aggregate '{on}' at node '{self._node_name}': shapes "
                f"differ across traces: {sorted(shapes)}"
            )

        stacked = torch.stack(usable_tensors, dim=0).to(torch.float32)
        if statistic == "mean":
            return stacked.mean(dim=0)
        if statistic == "std":
            return stacked.std(dim=0)
        if statistic == "var":
            return stacked.var(dim=0)
        if statistic == "norm":
            return torch.linalg.vector_norm(stacked, dim=0)
        raise ValueError(f"Unknown statistic '{statistic}'. Valid: 'mean', 'std', 'var', 'norm'.")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        cov_pct = self.coverage * 100
        return (
            f"NodeView(name='{self._node_name}', op_type='{self.op_type}', "
            f"traces={sorted(self._node.traces)}, coverage={cov_pct:.0f}%)"
        )
