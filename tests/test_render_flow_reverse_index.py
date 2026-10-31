"""Regression tests for render-flow reverse-index reuse."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.data_classes.trace import Trace
from torchlens.visualization import _render_flow


class _SkipChain(nn.Module):
    """Model containing consecutive skippable operations."""

    def __init__(self) -> None:
        """Initialize the test model."""

        super().__init__()
        self.input_projection = nn.Linear(4, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.output_projection = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the projection and consecutive activation chain."""

        hidden = self.input_projection(inputs)
        hidden = self.relu(hidden)
        hidden = self.sigmoid(hidden)
        return self.output_projection(hidden)


def _skip_activations(layer: Any) -> bool:
    """Return whether a layer is one of the consecutive test activations."""

    return layer.layer_type in {"relu", "sigmoid"}


def test_render_pass_reuses_layer_label_reverse_indexes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each render-flow pass should build and reuse one reverse index."""

    trace = tl.trace(_SkipChain(), torch.randn(1, 4))
    entries = dict(trace.layer_dict_main_keys)
    entry_snapshot = tuple(entries.items())

    hidden_indexes: list[Mapping[str, Any]] = []
    visible_mappings: list[Mapping[str, Any]] = []
    skip_indexes: list[Mapping[str, Any]] = []
    walk_indexes: list[Mapping[str, Any]] = []
    original_hidden = _render_flow._is_hidden_buffer_update_node
    original_expand = _render_flow._expand_edges_through_skipped
    original_walk = _render_flow._walk_skipped_successors

    def spy_hidden(
        owning_trace: Trace,
        node: Any,
        entries_to_plot: Mapping[str, Any],
        entries_by_layer_label: Mapping[str, Any],
        show_buffer_layers: Any,
        vis_mode: str,
    ) -> bool:
        """Record the reverse index used for hidden-buffer checks."""

        hidden_indexes.append(entries_by_layer_label)
        return original_hidden(
            owning_trace,
            node,
            entries_to_plot,
            entries_by_layer_label,
            show_buffer_layers,
            vis_mode,
        )

    def spy_expand(
        owning_trace: Trace,
        parent_node: Any,
        visible_entries: Mapping[str, Any],
        visible_entries_by_layer: Mapping[str, Any],
        skipped_labels: set[str],
        vis_mode: str,
    ) -> list[Any]:
        """Record the reverse index used for each visible parent."""

        visible_mappings.append(visible_entries)
        skip_indexes.append(visible_entries_by_layer)
        return original_expand(
            owning_trace,
            parent_node,
            visible_entries,
            visible_entries_by_layer,
            skipped_labels,
            vis_mode,
        )

    def spy_walk(
        owning_trace: Trace,
        node: Any,
        visible_entries: Mapping[str, Any],
        visible_entries_by_layer: Mapping[str, Any],
        skipped_labels: set[str],
        vis_mode: str,
        seen: set[str],
    ) -> list[Any]:
        """Record the reverse index threaded through skipped-node recursion."""

        walk_indexes.append(visible_entries_by_layer)
        return original_walk(
            owning_trace,
            node,
            visible_entries,
            visible_entries_by_layer,
            skipped_labels,
            vis_mode,
            seen,
        )

    monkeypatch.setattr(_render_flow, "_is_hidden_buffer_update_node", spy_hidden)
    monkeypatch.setattr(_render_flow, "_expand_edges_through_skipped", spy_expand)
    monkeypatch.setattr(_render_flow, "_walk_skipped_successors", spy_walk)

    _, skipped_labels = _render_flow._build_skip_filtered_edge_map(
        trace,
        entries,
        vis_mode="unrolled",
        show_buffer_layers="meaningful",
        skip_fn=_skip_activations,
    )
    _render_flow._enumerate_base_rendered_node_emissions(
        trace,
        entries,
        skipped_labels=skipped_labels,
        vis_mode="unrolled",
        vis_call_depth=1000,
        show_buffer_layers="meaningful",
        collapse_fn=None,
        repeat_folds=None,
        show_containers=False,
        collapsed_container_nodes={},
    )

    assert len(hidden_indexes) > 1
    assert len(skip_indexes) > 1
    assert len(walk_indexes) > 1
    assert len({id(index) for index in hidden_indexes}) == 1
    assert len({id(index) for index in skip_indexes + walk_indexes}) == 1
    assert len({id(mapping) for mapping in visible_mappings}) == 1
    assert hidden_indexes[0] == {node.layer_label: node for node in entries.values()}
    assert skip_indexes[0] == {node.layer_label: node for node in visible_mappings[0].values()}
    assert tuple(entries.items()) == entry_snapshot
    assert all(
        tuple(mapping.items()) == tuple(visible_mappings[0].items()) for mapping in visible_mappings
    )
