"""Generic base classes for bundle-level Super views."""

from __future__ import annotations

from typing import Any, Callable, Generic, Literal, TypeVar, cast

import torch

from .._metrics import is_scalar_like, relative_l1_scalar, resolve_metric

T = TypeVar("T")
S = TypeVar("S", bound="Super[Any]")
_TENSOR_FIELD_LITERAL = Literal["out", "grad"]


class Super(Generic[T]):
    """Generic aligned view of one trace object across bundle members."""

    def __init__(
        self,
        node_name: str,
        members: dict[str, T],
        *,
        query: Any = None,
        bundle_member_names: list[str] | None = None,
    ) -> None:
        """Initialize an aligned Super view.

        Parameters
        ----------
        node_name:
            Representative node or label name.
        members:
            Resolved objects keyed by bundle member name.
        query:
            Original user query, when available.
        bundle_member_names:
            Full bundle member-name order. Missing names are treated as sparse
            alignment gaps and lower ``coverage``.
        """

        self._node_name = node_name
        self._query = query
        self._members = dict(members)
        self._bundle_member_names = (
            list(bundle_member_names) if bundle_member_names is not None else list(members)
        )

    @classmethod
    def from_members(cls: type[S], query: Any, members: dict[str, T]) -> S:
        """Build a Super view from resolved member objects.

        Parameters
        ----------
        query:
            Original user query.
        members:
            Mapping from bundle member name to resolved object.

        Returns
        -------
        Super
            New aligned view.
        """

        first_label = (
            getattr(
                next(iter(members.values())),
                "label",
                getattr(next(iter(members.values())), "layer_label", repr(query)),
            )
            if members
            else repr(query)
        )
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

        return {
            name: str(getattr(member, "label", getattr(member, "layer_label", "")))
            for name, member in self._members.items()
        }

    @property
    def members(self) -> dict[str, T]:
        """Return resolved objects keyed by member name.

        Returns
        -------
        dict[str, T]
            Per-member resolved objects.
        """

        return dict(self._members)

    @property
    def traces(self) -> set[str]:
        """Return names of bundle members represented by this view.

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

    def __repr__(self) -> str:
        """Return a compact representation.

        Returns
        -------
        str
            Representation.
        """

        return (
            f"{self.__class__.__name__}(name={self._node_name!r}, members={list(self._members)!r})"
        )


class _TensorBearing:
    """Mixin for Super views whose members expose tensor-like fields."""

    _members: dict[str, Any]
    _node_name: str

    @property
    def op_type(self) -> str:
        """Return the representative operation type.

        Returns
        -------
        str
            Function name.
        """

        node = getattr(self, "_node", None)
        if node is not None:
            return str(node.op_type)
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

        node = getattr(self, "_node", None)
        if node is not None:
            return node.module_path
        first = next(iter(self._members.values()), None)
        module = None if first is None else getattr(first, "module", None)
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
            self._shape_excluding_batch(member)
            for member in self._members.values()
            if getattr(member, "shape", None) is not None
        }
        if len(shapes) > 1:
            raise ValueError(
                f"Shape mismatch across bundle members at node {self._node_name!r}: "
                f"{sorted(shapes)}"
            )
        return next(iter(shapes), ())

    @property
    def outs(self) -> dict[str, torch.Tensor | None]:
        """Return out tensors keyed by member name.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Per-member outs.
        """

        return self._tensor_dict("out")

    @property
    def grads(self) -> dict[str, torch.Tensor | None]:
        """Return grad tensors keyed by member name.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Per-member grads.
        """

        return self._tensor_dict("grad")

    @property
    def out(self) -> torch.Tensor:
        """Return a stacked out tensor.

        Returns
        -------
        torch.Tensor
            Activations concatenated along batch dimension when possible.
        """

        return self._stacked("out")

    @property
    def grad(self) -> torch.Tensor:
        """Return a stacked grad tensor.

        Returns
        -------
        torch.Tensor
            Gradients concatenated along batch dimension when possible.
        """

        return self._stacked("grad")

    def diff_pair(
        self,
        other: str | None = None,
        metric: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "cosine",
        on: _TENSOR_FIELD_LITERAL = "out",
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
        on: _TENSOR_FIELD_LITERAL = "out",
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
            return cast(torch.Tensor, torch.linalg.vector_norm(stacked, dim=0))
        raise ValueError("statistic must be one of 'mean', 'std', 'var', or 'norm'.")

    def _tensor_dict(self, field: _TENSOR_FIELD_LITERAL) -> dict[str, torch.Tensor | None]:
        """Return a tensor field keyed by member name.

        Parameters
        ----------
        field:
            Tensor field to collect.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Per-member tensor values.
        """

        output: dict[str, torch.Tensor | None] = {}
        for name, member in self._members.items():
            value = self._get_tensor(member, field)
            output[name] = value if isinstance(value, torch.Tensor) else None
        return output

    def _get_tensor(self, member: Any, field: _TENSOR_FIELD_LITERAL) -> torch.Tensor | None:
        """Return one tensor-bearing field from ``member``.

        Parameters
        ----------
        member:
            Tensor-bearing member object.
        field:
            Tensor field to collect.

        Returns
        -------
        torch.Tensor | None
            Tensor value when available.
        """

        if field == "out":
            has_value = getattr(member, "has_saved_activation", False)
            value = getattr(member, "out", None) if has_value else None
        else:
            has_value = getattr(member, "has_saved_gradient", False)
            value = getattr(member, "grad", None) if has_value else None
        return value if isinstance(value, torch.Tensor) else None

    def _stacked(self, field: _TENSOR_FIELD_LITERAL) -> torch.Tensor:
        """Stack or concatenate a tensor field across members.

        Parameters
        ----------
        field:
            Tensor field to stack.

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
    def _shape_excluding_batch(member: Any) -> tuple[int, ...]:
        """Return a member shape excluding leading batch dimension.

        Parameters
        ----------
        member:
            Tensor-bearing member object.

        Returns
        -------
        tuple[int, ...]
            Non-batch shape.
        """

        shape = tuple(getattr(member, "shape", ()) or ())
        return () if len(shape) <= 1 else shape[1:]

    @staticmethod
    def _diff_matrix(
        tensors: list[torch.Tensor | None],
        metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Compute a square pairwise distance matrix.

        Parameters
        ----------
        tensors:
            Tensors to compare.
        metric_fn:
            Pairwise tensor metric.

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

        Parameters
        ----------
        tensors:
            Tensors to compare.
        ref_idx:
            Reference tensor index.
        metric_fn:
            Pairwise tensor metric.

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
