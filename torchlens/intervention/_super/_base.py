"""Generic base classes for bundle-level Super views."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast

import torch

from ..._errors import InvalidArgumentError, PayloadUnavailableError
from ...errors.episode import CheckpointSeriesLiveParamsError
from .._metrics import is_scalar_like, relative_l1_scalar, resolve_metric

T = TypeVar("T")
S = TypeVar("S", bound="Super[Any]")
_TENSOR_FIELD_LITERAL = Literal["out", "grad"]


class SuperMemberAccessor(Generic[T]):
    """Dict-like accessor for one Super view's represented members."""

    def __init__(self, members: dict[str, T]) -> None:
        """Initialize the member accessor.

        Parameters
        ----------
        members:
            Resolved objects keyed by bundle member name.
        """

        self._members = dict(members)
        self._names = list(members)

    def __getitem__(self, key: int | str) -> T:
        """Return a represented member by 0-based position or trace name.

        Parameters
        ----------
        key:
            Integer position or bundle member name.

        Returns
        -------
        T
            Resolved member object.
        """

        if isinstance(key, int):
            return self._members[self._names[key]]
        return self._members[key]

    def __contains__(self, key: object) -> bool:
        """Return whether a trace name is represented.

        Parameters
        ----------
        key:
            Candidate trace name.

        Returns
        -------
        bool
            Whether ``key`` names a represented member.
        """

        return isinstance(key, str) and key in self._members

    def __iter__(self) -> Iterator[str]:
        """Iterate represented trace names.

        Returns
        -------
        Iterator[str]
            Iterator over represented trace names.
        """

        return iter(self._names)

    def __len__(self) -> int:
        """Return the number of represented members.

        Returns
        -------
        int
            Represented member count.
        """

        return len(self._members)

    def keys(self) -> list[str]:
        """Return represented trace names.

        Returns
        -------
        list[str]
            Trace names in bundle order.
        """

        return list(self._names)

    def values(self) -> list[T]:
        """Return represented member objects.

        Returns
        -------
        list[T]
            Resolved member objects in bundle order.
        """

        return [self._members[name] for name in self._names]

    def items(self) -> list[tuple[str, T]]:
        """Return represented ``(trace_name, member)`` pairs.

        Returns
        -------
        list[tuple[str, T]]
            Member pairs in bundle order.
        """

        return [(name, self._members[name]) for name in self._names]


class Super(Generic[T]):
    """Generic aligned view of one trace object across bundle members."""

    def __init__(
        self,
        label: str,
        members: dict[str, T],
        *,
        query: Any = None,
        bundle_member_names: list[str] | None = None,
    ) -> None:
        """Initialize an aligned Super view.

        Parameters
        ----------
        label:
            Representative label.
        members:
            Resolved objects keyed by bundle member name.
        query:
            Original user query, when available.
        bundle_member_names:
            Full bundle member-name order. Missing names are treated as sparse
            alignment gaps and lower ``coverage``.
        """

        self._label = label
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
    def label(self) -> str:
        """Return the representative label.

        Returns
        -------
        str
            Label.
        """

        return self._label

    @property
    def members(self) -> SuperMemberAccessor[T]:
        """Return resolved objects keyed by member name.

        Returns
        -------
        SuperMemberAccessor[T]
            Accessor over represented member objects.
        """

        return SuperMemberAccessor(self._members)

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
    def absent_traces(self) -> set[str]:
        """Return bundle member names not represented by this view.

        Returns
        -------
        set[str]
            Bundle member names where this label did not resolve.
        """

        return set(self._bundle_member_names) - set(self._members)

    @property
    def num_traces(self) -> int:
        """Return the number of represented bundle members.

        Returns
        -------
        int
            Represented member count.
        """

        return len(self.traces)

    @property
    def num_absent_traces(self) -> int:
        """Return the number of bundle members not represented.

        Returns
        -------
        int
            Absent member count.
        """

        return len(self.absent_traces)

    @property
    def is_complete_coverage(self) -> bool:
        """Return whether every bundle member is represented.

        Returns
        -------
        bool
            Whether this label resolved in every bundle member.
        """

        return not self.absent_traces

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

        return f"{self.__class__.__name__}(label={self._label!r}, members={list(self._members)!r})"


class _TensorBearing:
    """Mixin for Super views whose members expose tensor-like fields."""

    _members: dict[str, Any]
    _label: str

    #: Claim class of this view's per-member tensors. ``True`` (SuperParam)
    #: means a cross-member read asserts capture-time PARAMETER values -- a
    #: historical claim TorchLens cannot prove from live handles -- so the
    #: ``_tensor_dict`` funnel refuses before tensor lookup unless every
    #: member carries immutable parameter evidence (A-CKPT; foldB D7).
    _cross_member_param_claim: ClassVar[bool] = False

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
            raise InvalidArgumentError(
                f"Shape mismatch across bundle members at label {self._label!r}: {sorted(shapes)}",
                code="bundle_shape_mismatch",
                remedy="query members with matching shapes at this label",
                label=str(self._label),
            )
        return next(iter(shapes), ())

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
                raise InvalidArgumentError(
                    f"Unknown bundle member {other!r}. Known: {list(tensor_dict)}",
                    code="bundle_member_unknown",
                    remedy="pass one of the known bundle member names",
                    member=str(other),
                )
            if other not in names:
                raise PayloadUnavailableError(
                    f"Bundle member {other!r} has no usable {on} at this node",
                    code="bundle_member_payload_missing",
                    remedy="compare against a member with a stored tensor at this node",
                    member=str(other),
                )
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
            raise PayloadUnavailableError(
                f"No bundle members have stored {on} at label {self._label!r}",
                code="bundle_member_payload_missing",
                remedy="aggregate at a label where members retained tensors",
                label=str(self._label),
            )
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) > 1:
            raise InvalidArgumentError(
                f"Cannot aggregate tensors with different shapes: {sorted(shapes)}",
                code="bundle_shape_mismatch",
                remedy="aggregate members with matching shapes",
            )
        stacked = torch.stack([tensor.to(torch.float32) for tensor in tensors], dim=0)
        if statistic == "mean":
            return stacked.mean(dim=0)
        if statistic == "std":
            return stacked.std(dim=0)
        if statistic == "var":
            return stacked.var(dim=0)
        if statistic == "norm":
            return cast(torch.Tensor, torch.linalg.vector_norm(stacked, dim=0))
        raise InvalidArgumentError(
            f"statistic must be one of 'mean', 'std', 'var', or 'norm'; received {statistic!r}",
            code="bundle_statistic_invalid",
            remedy="pass statistic='mean', 'std', 'var', or 'norm'",
            argument="statistic",
        )

    def _tensor_dict(self, field: _TENSOR_FIELD_LITERAL) -> dict[str, torch.Tensor | None]:
        """Return a tensor field keyed by member name.

        This is the ONE funnel every cross-member tensor read passes through
        (``diff_pair``, ``aggregate``, ``out``/``grad`` via ``_stacked``, and
        the SuperParam ``weight_norm_diff``), so the checkpoint live-ref
        guard lives here: for parameter-claim views with two or more members
        it refuses BEFORE any member tensor is resolved.

        Parameters
        ----------
        field:
            Tensor field to collect.

        Returns
        -------
        dict[str, torch.Tensor | None]
            Per-member tensor values.
        """

        if self._cross_member_param_claim and len(self._members) >= 2:
            self._require_immutable_param_evidence(field)
        output: dict[str, torch.Tensor | None] = {}
        for name, member in self._members.items():
            value = self._get_tensor(member, field)
            output[name] = value if isinstance(value, torch.Tensor) else None
        return output

    def _require_immutable_param_evidence(self, field: _TENSOR_FIELD_LITERAL) -> None:
        """Refuse a cross-member parameter read without immutable evidence.

        The guard is keyed on the CLAIM (a cross-member, cross-time parameter
        value), never on Python object identity: save/load manufactures
        relationship-rank upgrades, and a reloaded artifact has no model ref
        to detect. Every member's derived value basis must be immutable
        capture-time evidence (``snapshot``, R8(b)) for the read to proceed;
        ``live_ref`` and ``absent`` bases refuse typed.

        Parameters
        ----------
        field:
            Tensor field the caller asked for (``out`` or ``grad``).

        Raises
        ------
        CheckpointSeriesLiveParamsError
            With stable code ``checkpoint_series_live_params`` when any
            member lacks immutable parameter evidence.
        """

        bases: dict[str, Any] = {
            name: getattr(member, "value_basis", None) for name, member in self._members.items()
        }
        if all(basis is not None and basis.is_immutable for basis in bases.values()):
            return
        member_names = list(self._members)
        first = next(iter(self._members.values()))
        param_address = str(getattr(first, "address", None) or self._label)
        quoted = [f"'{name}'" for name in member_names]
        named = " and ".join(quoted) if len(quoted) == 2 else ", ".join(quoted)
        any_live = any(basis is not None and basis.basis == "live_ref" for basis in bases.values())
        if any_live:
            mechanism = (
                "TorchLens records which parameter a run used, not its bytes, so a "
                "parameter read resolves through the live model and returns TODAY'S "
                "weights, not the weights at capture. Two members of a checkpoint "
                "series will therefore report identical weights."
            )
        else:
            mechanism = (
                "TorchLens records which parameter a run used, not its bytes, and "
                "this artifact was saved without parameter snapshots, so the "
                "per-member values are gone. Reads would degrade to NaN or empty "
                "results that look like 'nothing to compare'."
            )
        remedy = (
            "snapshot parameters at capture, or bind each member to an immutable "
            "checkpoint -- the ordering is still valid; only the weight claim is refused"
        )
        raise CheckpointSeriesLiveParamsError(
            f"Members {named} have no immutable capture-time parameter evidence "
            f"for parameter '{param_address}'. {mechanism} "
            f"To read a weight trajectory: {remedy}.",
            code="checkpoint_series_live_params",
            members=member_names,
            param_address=param_address,
            field=str(field),
            bases={
                name: str(basis) if basis is not None else None for name, basis in bases.items()
            },
            remedy=remedy,
        )

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
            has_value = getattr(member, "has_grad", False)
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
            raise PayloadUnavailableError(
                f"Cannot stack {field!r} for label {self._label!r}: "
                "not every member has a stored tensor",
                code="bundle_stack_incomplete",
                remedy="stack at a label where every member retained a tensor",
                label=str(self._label),
                field=str(field),
            )
        shapes = {tuple(tensor.shape[1:]) if tensor.dim() > 0 else () for tensor in tensors}
        if len(shapes) > 1:
            raise InvalidArgumentError(
                f"Cannot stack tensors with different non-batch shapes: {sorted(shapes)}",
                code="bundle_shape_mismatch",
                remedy="stack members with matching non-batch shapes",
            )
        if not tensors:
            raise PayloadUnavailableError(
                f"Cannot stack {field!r}: no tensors are available",
                code="bundle_stack_incomplete",
                remedy="stack at a label where members retained tensors",
                field=str(field),
            )
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
                    if is_scalar_like(left) and is_scalar_like(right)
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
                relative_l1_scalar(ref, tensor)
                if is_scalar_like(ref) and is_scalar_like(tensor)
                else metric_fn(ref, tensor)
            )
            out[0, idx] = float(value.detach().item())
        return out
