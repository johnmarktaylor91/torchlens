"""NodeView accessors for bundle sites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import torch

from .metrics import is_scalar_like, relative_l1_scalar, resolve_metric

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..data_classes.layer_pass_log import LayerPassLog
    from .topology import SupergraphNode

_TENSOR_FIELD_LITERAL = Literal["activation", "gradient"]


class NodeView:
    """View of a single site across all bundle members."""

    def __init__(
        self,
        node_name: str,
        node: "SupergraphNode | None" = None,
        bundle_trace_names: list[str] | None = None,
        *,
        members: dict[str, Any] | None = None,
        query: Any | None = None,
    ) -> None:
        """Initialize a node view.

        Parameters
        ----------
        node_name:
            Display node name.
        node:
            Optional legacy supergraph node.
        bundle_trace_names:
            Optional legacy trace-name order.
        members:
            Dict keyed by bundle member name.
        query:
            Original site query.
        """

        self._node_name = node_name
        self._query = query
        self._node = node
        if members is not None:
            self._members = dict(members)
            self._bundle_member_names = list(members)
        elif node is not None and bundle_trace_names is not None:
            self._members = {
                name: node.layer_refs[name]
                for name in bundle_trace_names
                if name in node.layer_refs
            }
            self._bundle_member_names = list(bundle_trace_names)
        else:
            self._members = {}
            self._bundle_member_names = []

    @classmethod
    def from_members(cls, query: Any, members: dict[str, "LayerPassLog"]) -> "NodeView":
        """Build a NodeView from resolved member layer passes.

        Parameters
        ----------
        query:
            Original site query.
        members:
            Mapping from bundle member name to layer pass.

        Returns
        -------
        NodeView
            New node view.
        """

        first_label = next(iter(members.values())).layer_label if members else repr(query)
        return cls(str(first_label), members=members, query=query)

    @property
    def node_name(self) -> str:
        """Return the representative node name.

        Returns
        -------
        str
            Node name.
        """

        return self._node_name

    @property
    def labels(self) -> dict[str, str]:
        """Return resolved labels keyed by member name.

        Returns
        -------
        dict[str, str]
            Per-member resolved labels.
        """

        return {name: str(layer.layer_label) for name, layer in self._members.items()}

    @property
    def members(self) -> dict[str, Any]:
        """Return resolved layer pass records keyed by member name.

        Returns
        -------
        dict[str, LayerPassLog]
            Per-member layer pass records.
        """

        return dict(self._members)

    @property
    def traces(self) -> set[str]:
        """Return names of members that resolved this node.

        Returns
        -------
        set[str]
            Member names.
        """

        return set(self._members)

    @property
    def coverage(self) -> float:
        """Return the fraction of bundle members represented by this view.

        Returns
        -------
        float
            Coverage in ``[0, 1]``.
        """

        if not self._bundle_member_names:
            return 1.0 if self._members else 0.0
        return len(self._members) / len(self._bundle_member_names)

    @property
    def op_type(self) -> str:
        """Return the representative operation type.

        Returns
        -------
        str
            Function name.
        """

        if self._node is not None:
            return self._node.op_type
        first = next(iter(self._members.values()), None)
        return "" if first is None else str(getattr(first, "func_name", "") or "")

    @property
    def module_path(self) -> str | None:
        """Return the representative containing module.

        Returns
        -------
        str | None
            Module path.
        """

        if self._node is not None:
            return self._node.module_path
        first = next(iter(self._members.values()), None)
        module = None if first is None else getattr(first, "containing_module", None)
        return None if module is None else str(module)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the common tensor shape excluding the batch dimension.

        Returns
        -------
        tuple[int, ...]
            Common non-batch shape.
        """

        shapes = {
            self._shape_excluding_batch(layer)
            for layer in self._members.values()
            if getattr(layer, "tensor_shape", None) is not None
        }
        if len(shapes) > 1:
            raise ValueError(
                f"Shape mismatch across bundle members at node {self._node_name!r}: "
                f"{sorted(shapes)}"
            )
        return next(iter(shapes), ())

    @property
    def activations(self) -> dict[str, torch.Tensor | None]:
        """Return activation tensors keyed by member name.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Per-member activations.
        """

        return self._tensor_dict("activation")

    @property
    def gradients(self) -> dict[str, torch.Tensor | None]:
        """Return gradient tensors keyed by member name.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Per-member gradients.
        """

        return self._tensor_dict("gradient")

    @property
    def activation(self) -> torch.Tensor:
        """Return a stacked activation tensor.

        Returns
        -------
        torch.Tensor
            Activations concatenated along batch dimension when possible.
        """

        return self._stacked("activation")

    @property
    def gradient(self) -> torch.Tensor:
        """Return a stacked gradient tensor.

        Returns
        -------
        torch.Tensor
            Gradients concatenated along batch dimension when possible.
        """

        return self._stacked("gradient")

    def diff(
        self,
        other: str | None = None,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
        on: _TENSOR_FIELD_LITERAL = "activation",
    ) -> torch.Tensor:
        """Return pairwise distances between member tensors at this node.

        Parameters
        ----------
        other:
            Optional member name for a one-row comparison.
        metric:
            Metric name or callable.
        on:
            Tensor field to compare.

        Returns
        -------
        torch.Tensor
            Pairwise distance matrix.
        """

        metric_fn = resolve_metric(metric)
        tensor_dict = self._tensor_dict(on)
        names = [name for name, tensor in tensor_dict.items() if isinstance(tensor, torch.Tensor)]
        tensors = [tensor_dict[name] for name in names]
        if other is not None:
            if other not in tensor_dict:
                raise ValueError(f"Unknown bundle member {other!r}. Known: {list(tensor_dict)}")
            if other not in names:
                raise ValueError(f"Bundle member {other!r} has no usable {on} at this node.")
            ref_idx = names.index(other)
            return self._diff_row(tensors, ref_idx, metric_fn)
        return self._diff_matrix(tensors, metric_fn)

    def aggregate(
        self,
        statistic: Literal["mean", "std", "var", "norm"] = "mean",
        on: _TENSOR_FIELD_LITERAL = "activation",
    ) -> torch.Tensor:
        """Aggregate member tensors at this node.

        Parameters
        ----------
        statistic:
            Reduction to apply.
        on:
            Tensor field to aggregate.

        Returns
        -------
        torch.Tensor
            Aggregated tensor.
        """

        tensors = [
            tensor for tensor in self._tensor_dict(on).values() if isinstance(tensor, torch.Tensor)
        ]
        if not tensors:
            raise ValueError(f"No bundle members have stored {on} at node {self._node_name!r}.")
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) > 1:
            raise ValueError(f"Cannot aggregate tensors with different shapes: {sorted(shapes)}")
        stacked = torch.stack([tensor.to(torch.float32) for tensor in tensors], dim=0)
        if statistic == "mean":
            return stacked.mean(dim=0)
        if statistic == "std":
            return stacked.std(dim=0)
        if statistic == "var":
            return stacked.var(dim=0)
        if statistic == "norm":
            return torch.linalg.vector_norm(stacked, dim=0)
        raise ValueError("statistic must be one of 'mean', 'std', 'var', or 'norm'.")

    def _tensor_dict(self, field: _TENSOR_FIELD_LITERAL) -> dict[str, torch.Tensor | None]:
        """Return a tensor field keyed by member name.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Per-member tensor values.
        """

        output: dict[str, torch.Tensor | None] = {}
        for name, layer in self._members.items():
            if field == "activation":
                has_value = getattr(layer, "has_saved_activations", False)
                value = getattr(layer, "activation", None) if has_value else None
            else:
                has_value = getattr(layer, "has_gradient", False)
                value = getattr(layer, "gradient", None) if has_value else None
            output[name] = value if isinstance(value, torch.Tensor) else None
        return output

    def _stacked(self, field: _TENSOR_FIELD_LITERAL) -> torch.Tensor:
        """Stack or concatenate a tensor field across members.

        Returns
        -------
        torch.Tensor
            Stacked tensor.
        """

        tensors = [
            value for value in self._tensor_dict(field).values() if isinstance(value, torch.Tensor)
        ]
        if len(tensors) != len(self._members):
            raise ValueError(
                f"Cannot stack {field!r} for node {self._node_name!r}: "
                "not every member has a stored tensor."
            )
        shapes = {tuple(tensor.shape[1:]) if tensor.dim() > 0 else () for tensor in tensors}
        if len(shapes) > 1:
            raise ValueError(
                f"Cannot stack tensors with different non-batch shapes: {sorted(shapes)}"
            )
        if not tensors:
            raise ValueError(f"Cannot stack {field!r}: no tensors are available.")
        if tensors[0].dim() == 0:
            return torch.stack(tensors, dim=0)
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _shape_excluding_batch(layer: Any) -> tuple[int, ...]:
        """Return a layer shape excluding leading batch dimension.

        Returns
        -------
        tuple[int, ...]
            Non-batch shape.
        """

        shape = tuple(getattr(layer, "tensor_shape", ()) or ())
        return () if len(shape) <= 1 else shape[1:]

    @staticmethod
    def _diff_matrix(
        tensors: list[torch.Tensor | None],
        metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute a square pairwise distance matrix.

        Returns
        -------
        torch.Tensor
            Pairwise distance matrix.
        """

        usable = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
        out = torch.zeros(len(usable), len(usable))
        for i, left in enumerate(usable):
            for j, right in enumerate(usable):
                if i == j:
                    continue
                value = (
                    relative_l1_scalar(left, right)
                    if is_scalar_like(left)
                    else metric_fn(left, right)
                )
                out[i, j] = float(value.detach().item())
        return out

    @staticmethod
    def _diff_row(
        tensors: list[torch.Tensor | None],
        ref_idx: int,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute one row of distances from a reference tensor.

        Returns
        -------
        torch.Tensor
            Distance row.
        """

        usable = [tensor for tensor in tensors if isinstance(tensor, torch.Tensor)]
        ref = usable[ref_idx]
        out = torch.zeros(1, len(usable))
        for idx, tensor in enumerate(usable):
            if idx == ref_idx:
                continue
            value = (
                relative_l1_scalar(ref, tensor) if is_scalar_like(ref) else metric_fn(ref, tensor)
            )
            out[0, idx] = float(value.detach().item())
        return out

    def __repr__(self) -> str:
        """Return a compact representation.

        Returns
        -------
        str
            Representation.
        """

        return f"NodeView(name={self._node_name!r}, members={list(self._members)!r})"


__all__ = ["NodeView"]
