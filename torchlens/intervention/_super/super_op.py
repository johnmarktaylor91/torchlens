"""SuperOp accessors for bundle sites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ._accessor_base import SuperAccessor
from ._base import Super, _TensorBearing

if TYPE_CHECKING:
    from ..._data_classes.aten_op import AtenOp
    from ..._data_classes.layer import Layer
    from ..._data_classes.op import Op
    from .._topology.topology import SupergraphNode


class SuperOp(Super["Op"], _TensorBearing):
    """View of a single site across all bundle members."""

    def __init__(
        self,
        label: str,
        node: SupergraphNode | None = None,
        bundle_trace_names: list[str] | None = None,
        *,
        members: dict[str, Any] | None = None,
        query: Any | None = None,
    ) -> None:
        """Initialize a node view.

        Parameters
        ----------
        label:
            Display label.
        node:
            Optional legacy supergraph node.
        bundle_trace_names:
            Optional legacy trace-name order.
        members:
            Dict keyed by bundle member name.
        query:
            Original site query.
        """

        self._node = node
        resolved_members: dict[str, Any]
        if members is not None:
            resolved_members = dict(members)
            bundle_member_names = list(members)
        elif node is not None and bundle_trace_names is not None:
            resolved_members = {
                name: node.layer_refs[name]
                for name in bundle_trace_names
                if name in node.layer_refs
            }
            bundle_member_names = list(bundle_trace_names)
        else:
            resolved_members = {}
            bundle_member_names = []
        super().__init__(
            label,
            cast(dict[str, "Op"], resolved_members),
            query=query,
            bundle_member_names=bundle_member_names,
        )

    def __repr__(self) -> str:
        """Return a compact representation.

        Returns
        -------
        str
            Representation.
        """

        return f"SuperOp(label={self._label!r}, members={list(self._members)!r})"


class SuperLayer(SuperOp):
    """View of a single aggregate layer label across all bundle members."""


class SuperAtenOp(Super["AtenOp"]):
    """Positional, coverage-qualified alignment of one ATen slot across members."""

    def __init__(
        self,
        *,
        label: str,
        decomposition_slot: int,
        members: dict[str, AtenOp] | None = None,
        bundle_member_names: list[str] | None = None,
        has_observation_gap: bool = False,
    ) -> None:
        """Initialize one lazy primitive-slot alignment.

        Parameters
        ----------
        label
            Display label for the aligned slot.
        decomposition_slot
            Zero-based decomposition slot.
        members
            Primitive rows keyed by bundle member name.
        bundle_member_names
            Complete bundle member order, including missing rows.
        has_observation_gap
            Whether any member carries an observation gap or absent profile.
        """

        resolved_members = dict(members or {})
        resolved_names = list(bundle_member_names or resolved_members)
        super().__init__(
            label,
            resolved_members,
            query=decomposition_slot,
            bundle_member_names=resolved_names,
        )
        self.decomposition_slot = decomposition_slot
        self.comparison_status = self._comparison_status(
            resolved_members,
            resolved_names,
            has_observation_gap=has_observation_gap,
        )

    @staticmethod
    def _comparison_status(
        members: dict[str, AtenOp],
        member_names: list[str],
        *,
        has_observation_gap: bool,
    ) -> str:
        """Return the closed alignment status for one positional slot.

        Parameters
        ----------
        members
            Present primitive rows.
        member_names
            Complete bundle member order.
        has_observation_gap
            Whether gaps prevent a proven missing-row claim.

        Returns
        -------
        str
            One documented-unstable alignment status token.
        """

        if has_observation_gap:
            return "coverage_indeterminate"
        if len(members) != len(member_names):
            return "sparse"
        schemas = {
            (row.namespace, row.operator, row.overload, row.schema_fingerprint)
            for row in members.values()
        }
        return "all_present_same_schema" if len(schemas) <= 1 else "all_present_different_schema"

    def __repr__(self) -> str:
        """Return a compact alignment representation."""

        return (
            f"SuperAtenOp(label={self._label!r}, slot={self.decomposition_slot}, "
            f"status={self.comparison_status!r})"
        )


class SuperOpAccessor(SuperAccessor["Op", SuperOp]):
    """Dict-like Bundle accessor returning SuperOp objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize an op accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperOp)

    def _resolve_in_member(self, trace: Any, label: str) -> Op | None:
        """Resolve ``label`` to an Op within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate Op label.

        Returns
        -------
        Op | None
            Matching Op, or ``None`` when unresolved.
        """
        try:
            resolved = trace.layers[label]
        except (KeyError, ValueError):
            return None
        if type(resolved).__name__ == "Op":
            return cast("Op", resolved)
        if type(resolved).__name__ == "Layer" and len(resolved.ops) == 1:
            return cast("Op", resolved.ops[0])
        return None


class SuperLayerAccessor(SuperAccessor["Layer", SuperLayer]):
    """Dict-like Bundle accessor returning SuperLayer objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a layer accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, super_cls=SuperLayer)

    def _resolve_in_member(self, trace: Any, label: str) -> Layer | None:
        """Resolve ``label`` to a Layer within one member trace.

        Parameters
        ----------
        trace:
            Bundle member trace.
        label:
            Candidate layer label.

        Returns
        -------
        Layer | None
            Matching Layer, or ``None`` when unresolved.
        """
        try:
            resolved = trace.layers[label]
        except (KeyError, ValueError):
            return None
        return cast("Layer", resolved) if type(resolved).__name__ == "Layer" else None


class TraceAccessor:
    """Dict-like accessor for Bundle member traces."""

    def __init__(self, members: dict[str, Any]) -> None:
        """Initialize a trace accessor.

        Parameters
        ----------
        members:
            Bundle member mapping.
        """

        self._members = members

    def __getitem__(self, name: str) -> Any:
        """Return a trace by member name.

        Parameters
        ----------
        name:
            Bundle member name.

        Returns
        -------
        Any
            Matching Trace.
        """

        return self._members[name]

    def __iter__(self) -> Any:
        """Iterate member names.

        Returns
        -------
        Any
            Iterator over member names.
        """

        return iter(self._members)

    def __len__(self) -> int:
        """Return the number of traces.

        Returns
        -------
        int
            Number of traces.
        """

        return len(self._members)

    def items(self) -> Any:
        """Return member items.

        Returns
        -------
        Any
            Dict-items view.
        """

        return self._members.items()


__all__ = [
    "SuperAtenOp",
    "SuperLayer",
    "SuperLayerAccessor",
    "SuperOp",
    "SuperOpAccessor",
    "TraceAccessor",
]
