"""Renderer-neutral source graph extraction for visualization."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .request import ResolvedRenderRequest

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


@dataclass(frozen=True)
class SourceGraph:
    """Normalized forward graph before collapse or presentation decisions.

    Parameters
    ----------
    entries_to_plot:
        Focus-rewritten render entries in deterministic source order.
    edge_map:
        Skip-filtered normalized edge occurrences.
    skipped_labels:
        Labels removed by the request's skip predicate.
    module_ancestry:
        Module paths for each source entry.
    container_ancestry:
        Typed container paths for each source entry.
    """

    entries_to_plot: dict[str, Any]
    edge_map: dict[str, list[Any]]
    skipped_labels: set[str]
    module_ancestry: Mapping[str, tuple[str, ...]]
    container_ancestry: Mapping[str, tuple[Any, ...]]


def build_source_graph(trace: "Trace", request: ResolvedRenderRequest) -> SourceGraph:
    """Extract the normalized forward source graph for one draw request.

    Parameters
    ----------
    trace:
        Captured trace to render.
    request:
        Fully resolved semantic render request.

    Returns
    -------
    SourceGraph
        Focus-rewritten entries, skip-filtered edges, and source ancestry
        indexes used by downstream rendering stages.
    """

    from ._render_flow import _build_module_focus_entries, _build_skip_filtered_edge_map
    from ._render_nodes import _normalize_buffer_visibility

    if request.vis_mode == "unrolled":
        entries_to_plot: dict[str, Any] = dict(trace.layer_dict_main_keys)
    elif request.vis_mode == "rolled":
        entries_to_plot = dict(trace.layer_logs)
    else:
        raise ValueError("vis_mode must be either 'rolled' or 'unrolled'")

    if request.module is not None:
        target_module = _resolve_focus_module(trace, request.module)
        entries_to_plot = _build_module_focus_entries(
            trace,
            entries_to_plot,
            target_module,
            vis_mode=request.vis_mode,
        )

    show_buffer_layers = _normalize_buffer_visibility(request.show_buffer_layers)
    edge_map, skipped_labels = _build_skip_filtered_edge_map(
        trace,
        entries_to_plot,
        vis_mode=request.vis_mode,
        show_buffer_layers=show_buffer_layers,
        skip_fn=request.skip_fn,
    )
    return SourceGraph(
        entries_to_plot=entries_to_plot,
        edge_map=edge_map,
        skipped_labels=set(skipped_labels),
        module_ancestry=_SourceAncestryIndex(entries_to_plot, "modules"),
        container_ancestry=_SourceAncestryIndex(entries_to_plot, "container_path"),
    )


def _resolve_focus_module(trace: "Trace", module: Any) -> Any:
    """Resolve and validate a module focus argument.

    Parameters
    ----------
    trace:
        Model log being rendered.
    module:
        Module instance or module address string.
    Returns
    -------
    Any
        Module to focus.

    Raises
    ------
    ValueError
        If the module cannot be found or belongs to a different trace.
    """

    from ..data_classes.module import Module

    if isinstance(module, str):
        if module not in trace.modules:
            raise ValueError(f"Module address '{module}' was not found in this Trace.")
        resolved = trace.modules[module]
        if not isinstance(resolved, Module):
            raise ValueError(f"Module address '{module}' resolved to a module pass, not a Module.")
        return resolved
    if not isinstance(module, Module):
        raise ValueError("module must be a Module, module address string, or None.")
    if module._source_trace is not trace:
        raise ValueError("Module focus must belong to the Trace being rendered.")
    return module


class _SourceAncestryIndex(Mapping[str, tuple[Any, ...]]):
    """Lookup-only ancestry index over normalized source entries.

    Parameters
    ----------
    entries_to_plot:
        Normalized source entries keyed by their render labels.
    attribute:
        Entry attribute containing either module or container ancestry.

    Returns
    -------
    _SourceAncestryIndex
        Mapping that defers ancestry extraction until a downstream stage needs
        a particular entry, preserving the single eager source walk.
    """

    def __init__(self, entries_to_plot: Mapping[str, Any], attribute: str) -> None:
        """Initialize an ancestry index.

        Parameters
        ----------
        entries_to_plot:
            Normalized source entries keyed by their render labels.
        attribute:
            Entry attribute containing either module or container ancestry.
        """

        self._entries_to_plot = entries_to_plot
        self._attribute = attribute

    def __getitem__(self, label: str) -> tuple[Any, ...]:
        """Return one entry's ancestry tuple.

        Parameters
        ----------
        label:
            Render label identifying the source entry.

        Returns
        -------
        tuple[Any, ...]
            Requested module or container ancestry.
        """

        try:
            return tuple(getattr(self._entries_to_plot[label], self._attribute, ()) or ())
        except ValueError:
            return ()

    def __iter__(self) -> Iterator[str]:
        """Iterate source labels in deterministic render order.

        Returns
        -------
        Iterator[str]
            Render labels from the normalized source entries.
        """

        return iter(self._entries_to_plot)

    def __len__(self) -> int:
        """Return the number of indexed source entries.

        Returns
        -------
        int
            Number of available ancestry records.
        """

        return len(self._entries_to_plot)
