"""SuperOp accessors for bundle sites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ._accessor_base import SuperAccessor
from ._base import Super, _TensorBearing

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ...data_classes.layer_log import Layer
    from ...data_classes.op_log import Op
    from .._topology.topology import SupergraphNode


class SuperOp(Super["Op"], _TensorBearing):
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
            node_name,
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

        return f"SuperOp(name={self._node_name!r}, members={list(self._members)!r})"


class SuperLayer(SuperOp):
    """View of a single aggregate layer label across all bundle members."""


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
            return cast("Op", resolved.ops[1])
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


__all__ = ["SuperLayer", "SuperLayerAccessor", "SuperOp", "SuperOpAccessor", "TraceAccessor"]
