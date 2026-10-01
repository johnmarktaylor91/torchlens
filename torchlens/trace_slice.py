"""``TraceSlice``: the sub-DAG view presenter (L6 graph slice, P2).

Composition, never inheritance (the ``MergedTrace`` precedent): a slice of
a trace is NOT a ``Trace``. A ``Trace`` promises inputs, outputs, module
hierarchy, validation state, save/replay; a slice has DANGLING EDGES whose
other ends live outside it, so calling it a Trace would imply capabilities
it cannot honour. ``TraceSlice`` therefore exposes exactly what a region
honestly has:

- its MEMBER ops (full typed ``Op`` records, execution order) and the
  dataflow edges INTERNAL to the region;
- an EXPLICIT BOUNDARY: every dataflow edge crossing into the region
  (``boundary_in_edges`` — the external dependencies) and out of it
  (``boundary_out_edges``), plus the region's entry/exit ops
  (``source_ops`` / ``sink_ops``);
- a lift back into the selection algebra (``__selection__`` + the operator
  mixin), so a slice composes with any other region and feeds ``do()``.

It deliberately offers NO save/replay/validate: those would be lies about
a region with unresolved external inputs. ``tl.save`` refuses a slice
typed (``slice_save_unsupported``) — save the underlying trace and
re-derive the view.

Built by ``trace.between(sources, sinks)`` (the influence region — the
same machinery and member set as the ``tl.between`` producer, one idea in
two binding modes) and by the general door ``trace.subgraph(selection)``,
which presents ANY ACT region — an n-hop neighborhood, a hand-built
selection, or a future motif matcher's hits — as the same view.

Every spelling here ships DOCUMENTED-UNSTABLE pending naming-session
ratification (megasprint provisional-name protocol).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .selection import (
    Selection,
    SelectionError,
    _lift,
    _SelectionOperand,
)
from .selection_graph import (
    _edge_partition,
    _lift_region_group,
    _region_labels,
    _resolve_region,
    _TraceGraph,
    resolve_between_labels,
)

__all__ = [
    "TraceSlice",
]


class TraceSlice(_SelectionOperand):
    """Frozen presenter over one trace's executed sub-DAG.

    Exposes member ops, internal dataflow edges, and the region's explicit
    boundary. Never a ``Trace``; offers no save/replay/validate. Lift back
    into the selection algebra with ``__selection__`` (implicit under the
    ``| & - ~`` operators). Session-time only; never persisted.
    """

    __slots__ = (
        "_boundary_in",
        "_boundary_out",
        "_graph",
        "_internal_edges",
        "_labels",
        "_member_set",
        "_source",
        "_trace",
    )

    _boundary_in: tuple[tuple[str, str], ...]
    _boundary_out: tuple[tuple[str, str], ...]
    _graph: _TraceGraph
    _internal_edges: tuple[tuple[str, str], ...]
    _labels: tuple[str, ...]
    _member_set: frozenset[str]
    _source: str
    _trace: Any

    def __init__(
        self,
        trace: Any,
        graph: _TraceGraph,
        members: frozenset[str],
        source: str,
    ) -> None:
        """Freeze one slice view (internal constructor).

        Users build slices through ``trace.between(...)`` /
        ``trace.subgraph(...)``, never directly.
        """

        internal, boundary_in, boundary_out = _edge_partition(graph, members)
        object.__setattr__(self, "_trace", trace)
        object.__setattr__(self, "_graph", graph)
        object.__setattr__(self, "_labels", graph.execution_ordered(members))
        object.__setattr__(self, "_member_set", members)
        object.__setattr__(self, "_internal_edges", internal)
        object.__setattr__(self, "_boundary_in", boundary_in)
        object.__setattr__(self, "_boundary_out", boundary_out)
        object.__setattr__(self, "_source", source)

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse mutation after freeze."""

        raise AttributeError("TraceSlice is frozen; derive a new slice instead.")

    # Membership --------------------------------------------------------------

    @property
    def source_trace(self) -> Any:
        """Return the trace this slice views (the single truth; never copied)."""

        return self._trace

    @property
    def labels(self) -> tuple[str, ...]:
        """Return member op labels (pass-qualified, execution order)."""

        return self._labels

    @property
    def ops(self) -> tuple[Any, ...]:
        """Return the member ``Op`` records in execution order."""

        return tuple(self._graph.ops[label] for label in self._labels)

    def __len__(self) -> int:
        """Return the member count."""

        return len(self._labels)

    def __iter__(self) -> Iterator[Any]:
        """Iterate member ``Op`` records in execution order."""

        return iter(self.ops)

    def __contains__(self, item: Any) -> bool:
        """Test membership by pass-qualified label, bare layer label, or Op."""

        if isinstance(item, str):
            if item in self._member_set:
                return True
            return any(
                getattr(self._graph.ops[label], "layer_label", None) == item
                for label in self._labels
            )
        return getattr(item, "label", None) in self._member_set

    def __getitem__(self, label: str) -> Any:
        """Return one member op by label (teaching KeyError on a miss).

        Accepts the exact pass-qualified label, or a bare layer label when
        it names exactly one member op.
        """

        if not isinstance(label, str):
            raise KeyError(
                f"TraceSlice indexes members by label string; got {type(label).__name__}."
            )
        if label in self._member_set:
            return self._graph.ops[label]
        bare_hits = [
            member
            for member in self._labels
            if getattr(self._graph.ops[member], "layer_label", None) == label
        ]
        if len(bare_hits) == 1:
            return self._graph.ops[bare_hits[0]]
        if len(bare_hits) > 1:
            spellings = ", ".join(repr(member) for member in bare_hits)
            raise KeyError(
                f"{label!r} names {len(bare_hits)} member passes of this slice; "
                f"use a pass-qualified label ({spellings})."
            )
        if label in self._graph.ops or any(
            getattr(op, "layer_label", None) == label for op in self._graph.ops.values()
        ):
            raise KeyError(
                f"{label!r} is an op of the underlying trace but not a member of "
                f"this slice ({self._source}); read it off the trace directly."
            )
        preview = ", ".join(repr(member) for member in self._labels[:5])
        if len(self._labels) > 5:
            preview += f", ... ({len(self._labels)} members)"
        raise KeyError(
            f"{label!r} is not an op of this slice's underlying trace; "
            f"members: {preview or '(none)'}."
        )

    # Structure ---------------------------------------------------------------

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        """Return internal dataflow edges as ``(parent_label, child_label)``."""

        return self._internal_edges

    @property
    def boundary_in_edges(self) -> tuple[tuple[str, str], ...]:
        """Return edges entering the slice: ``(external_parent, member)``.

        The slice's declared external dependencies — every member input
        produced OUTSIDE the region is named here, never silently dropped.
        """

        return self._boundary_in

    @property
    def boundary_out_edges(self) -> tuple[tuple[str, str], ...]:
        """Return edges leaving the slice: ``(member, external_child)``."""

        return self._boundary_out

    @property
    def source_ops(self) -> tuple[Any, ...]:
        """Return members with no parent inside the slice (entry ops)."""

        internal_children = {child for _, child in self._internal_edges}
        return tuple(
            self._graph.ops[label] for label in self._labels if label not in internal_children
        )

    @property
    def sink_ops(self) -> tuple[Any, ...]:
        """Return members with no child inside the slice (exit ops)."""

        internal_parents = {parent for parent, _ in self._internal_edges}
        return tuple(
            self._graph.ops[label] for label in self._labels if label not in internal_parents
        )

    # Disclosure --------------------------------------------------------------

    @property
    def empty(self) -> bool:
        """Return whether the slice has no members (emptiness is disclosure)."""

        return not self._labels

    def summary(self) -> str:
        """Return a human-readable disclosure of members and boundary.

        Opens with the source trace's honesty banner when that capture is
        non-clean: a slice is a VIEW, so summarizing a poisoned, unverified,
        structure-only, or episode capture without the banner would launder
        the parent's honesty facts away.
        """

        from ._capture_honesty import honesty_banner_lines

        lines = [
            f"TraceSlice ({self._source}): {len(self._labels)} ops, "
            f"{len(self._internal_edges)} internal edges",
        ]
        lines.extend(
            f"  [capture honesty] {banner_line}"
            for banner_line in honesty_banner_lines(self._trace)
        )
        if not self._labels:
            lines.append(
                "  empty region: no member ops (for source->sink slices this "
                "means no directed path exists)."
            )
            return "\n".join(lines)
        entry = ", ".join(op.label for op in self.source_ops)
        exit_ = ", ".join(op.label for op in self.sink_ops)
        lines.append(f"  entry ops: {entry}")
        lines.append(f"  exit ops: {exit_}")
        lines.append(
            f"  boundary: {len(self._boundary_in)} edges in, {len(self._boundary_out)} edges out"
        )
        for parent, child in self._boundary_in[:8]:
            lines.append(f"    in: {parent} -> {child}")
        if len(self._boundary_in) > 8:
            lines.append(f"    in: ... {len(self._boundary_in) - 8} more")
        for parent, child in self._boundary_out[:8]:
            lines.append(f"    out: {parent} -> {child}")
        if len(self._boundary_out) > 8:
            lines.append(f"    out: ... {len(self._boundary_out) - 8} more")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a compact honest summary (members + boundary counts)."""

        return (
            f"TraceSlice({len(self._labels)} ops, "
            f"{len(self._internal_edges)} internal edges, "
            f"boundary {len(self._boundary_in)} in / {len(self._boundary_out)} out; "
            f"{self._source})"
        )

    # Region door (F01) --------------------------------------------------------

    def as_region(self, *, exits: Any = None) -> Any:
        """Derive an admissible region operated on as ONE unit (F01).

        Runs the derived admissibility checks (complete exits, convexity,
        pass-instance partition) and returns the typed
        ``torchlens.intervention.regions.RegionTarget``, whose ``do()``
        lowering substitutes the region's EXIT values on the replay engine
        without replaying the interior. Contrast ``do(slice, edit)``, which
        keeps the shipped member-site family semantics (each member value
        edited in place).

        Parameters
        ----------
        exits:
            Optional user-declared exit list, checked for completeness
            (a leaking list refuses, naming the leaked edges).
        """

        from .intervention.regions import region as _region

        return _region(self, exits=exits)

    # Algebra lift ------------------------------------------------------------

    def __selection__(self) -> Selection:
        """Lift the slice's members back into the selection algebra.

        Returns the whole-site QUERY over the member family, addressed by
        ``(layer_label, pass_index)`` — exactly how Op/Layer lifts work —
        so ``slice & other`` composes and ``fork.do(slice, edit)``
        re-resolves the member sites on the fork by name. Resolution
        refuses typed on a trace missing a member site
        (``site_not_in_trace``) or on member sites with no output index
        space (``non_tensor_site`` / ``no_index_space``).
        """

        from .selection_graph import _SliceMembersTerm

        sites = tuple(
            (
                getattr(self._graph.ops[label], "layer_label", None) or label,
                getattr(self._graph.ops[label], "pass_index", 1) or 1,
            )
            for label in self._labels
        )
        return Selection(_SliceMembersTerm(sites=sites, display=self._source), kind="ACT")


def build_slice_between(trace: Any, sources: Any, sinks: Any) -> TraceSlice:
    """Build the influence-region slice for ``trace.between(sources, sinks)``.

    Shares the exact member-set machinery with the ``tl.between`` producer
    (:func:`torchlens.selection_graph.resolve_between_labels`): the
    presenter and the Selection denote the same region.
    """

    lifted_sources = _lift_region_group(sources, "between", "sources")
    lifted_sinks = _lift_region_group(sinks, "between", "sinks")
    members, graph = resolve_between_labels(trace, lifted_sources, lifted_sinks)
    return TraceSlice(trace, graph, members, "between(sources -> sinks)")


def build_slice_from_selection(trace: Any, selection_like: Any) -> TraceSlice:
    """Build the slice view of any ACT region for ``trace.subgraph(...)``.

    The general door: whatever produced the region — ``tl.neighborhood``,
    ``tl.between``, explicit ``tl.units``, or a future motif matcher — its
    touched-site FAMILY becomes the member set (element masks never shrink
    a graph region; the family level of the two-level denotation).
    """

    lifted = _lift(selection_like)
    if lifted is None:
        if isinstance(selection_like, str):
            from .selection import _WholeSiteTerm

            lifted = Selection(
                _WholeSiteTerm(site_label=selection_like, pass_index=None), kind="ACT"
            )
        else:
            raise ValueError(
                "trace.subgraph(...) takes a selection-shaped region (a "
                "Selection, ResolvedSelection, Op/Layer, receptive-field "
                f"region, or site label string); got {type(selection_like).__name__}."
            )
    if lifted.kind != "ACT":
        raise SelectionError(
            "trace.subgraph(...) presents regions of the executed op DAG, so "
            f"the region must be an ACT selection; got a {lifted.kind} selection.",
            code="selection_kind_incompatible",
            left_kind="ACT",
            right_kind=lifted.kind,
            operator="subgraph",
        )
    graph = _TraceGraph.from_trace(trace)
    members = _region_labels(graph, _resolve_region(lifted, trace), "subgraph")
    return TraceSlice(trace, graph, members, f"subgraph({lifted!r})")
