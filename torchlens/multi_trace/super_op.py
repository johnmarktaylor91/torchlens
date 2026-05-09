"""SuperOp accessors for bundle sites."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from ._super_base import Super, _TensorBearing

if TYPE_CHECKING:  # pragma: no cover - typing-only
    from ..data_classes.op_log import OpLog
    from .topology import SupergraphNode


class SuperOp(Super["OpLog"], _TensorBearing):
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
            cast(dict[str, "OpLog"], resolved_members),
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


class _BundleLabelAccessor:
    """Dict-like accessor for cross-member bundle labels."""

    def __init__(self, bundle: Any, *, pass_qualified: bool) -> None:
        """Initialize a bundle label accessor.

        Parameters
        ----------
        bundle:
            Bundle instance that owns the member traces.
        pass_qualified:
            Whether this accessor resolves pass-qualified Op labels.
        """

        self._bundle = bundle
        self._pass_qualified = pass_qualified

    def __getitem__(self, label: str) -> SuperOp | SuperLayer:
        """Return a cross-member view for ``label``.

        Parameters
        ----------
        label:
            Layer or Op label to resolve in each bundle member.

        Returns
        -------
        SuperOp | SuperLayer
            Cross-member view.
        """

        members: dict[str, Any] = {}
        for name, trace in self._bundle.members.items():
            members[name] = trace.layers[label]
        if self._pass_qualified:
            return SuperOp.from_members(label, members)
        return SuperLayer.from_members(label, members)

    def __contains__(self, label: object) -> bool:
        """Return whether every member contains ``label``.

        Parameters
        ----------
        label:
            Candidate label.

        Returns
        -------
        bool
            Whether the label resolves in every member.
        """

        if not isinstance(label, str):
            return False
        try:
            self[label]
        except (KeyError, ValueError):
            return False
        return True


class SuperOpAccessor(_BundleLabelAccessor):
    """Dict-like Bundle accessor returning SuperOp objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize an op accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, pass_qualified=True)


class SuperLayerAccessor(_BundleLabelAccessor):
    """Dict-like Bundle accessor returning SuperLayer objects."""

    def __init__(self, bundle: Any) -> None:
        """Initialize a layer accessor for ``bundle``.

        Parameters
        ----------
        bundle:
            Bundle instance.
        """

        super().__init__(bundle, pass_qualified=False)


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
