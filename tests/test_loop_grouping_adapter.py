"""Tests for backend-neutral loop grouping adapter."""

from typing import Any

import torch

import example_models
from torchlens import trace as trace_fn
from torchlens.postprocess.loop_grouping_adapter import (
    RecurrenceGroupingGraph,
    RecurrenceNode,
    group_recurrent_nodes,
)


def _raw_label(trace: Any, final_label: str) -> str:
    """Return the raw label corresponding to a final trace label.

    Parameters
    ----------
    trace:
        TorchLens trace with raw/final label maps.
    final_label:
        Final layer or op label.

    Returns
    -------
    str
        Raw label for ``final_label``.
    """
    return trace._final_to_raw_layer_labels[final_label]


def _raw_recurrent_member_sets(trace: Any) -> set[frozenset[str]]:
    """Collect expected recurrent member sets from a torch recurrent fixture.

    Parameters
    ----------
    trace:
        TorchLens trace for a recurrent torch model.

    Returns
    -------
    set[frozenset[str]]
        Raw-label recurrent groups with at least two members.
    """
    member_sets: set[frozenset[str]] = set()
    for op in trace.ops:
        if len(op.recurrent_ops) <= 1:
            continue
        raw_members = frozenset(_raw_label(trace, label) for label in op.recurrent_ops)
        member_sets.add(raw_members)
    return member_sets


def _neutral_graph_from_torch_recurrent_fixture(trace: Any) -> RecurrenceGroupingGraph:
    """Build a label-consistent neutral graph from a finalized torch trace.

    Parameters
    ----------
    trace:
        TorchLens trace for a recurrent torch model.

    Returns
    -------
    RecurrenceGroupingGraph
        Neutral graph using raw labels consistently for nodes and data edges.
    """
    nodes: dict[str, RecurrenceNode] = {}
    raw_labels: list[str] = []
    raw_label_set = {op._label_raw for op in trace.ops}

    for op in trace.ops:
        raw_label = op._label_raw
        raw_labels.append(raw_label)
        nodes[raw_label] = RecurrenceNode(
            label=raw_label,
            raw_order=op.raw_index,
            equivalence_key=op.equivalence_class,
            equivalent_labels=tuple(op.equivalent_ops),
            data_parents=tuple(
                _raw_label(trace, parent)
                for parent in op.parents
                if parent in trace.layer_dict_all_keys
            ),
            data_children=tuple(
                _raw_label(trace, child)
                for child in op.children
                if child in trace.layer_dict_all_keys
            ),
            layer_label=raw_label,
            recurrent_labels=(),
            uses_params=bool(op.uses_params),
            func_name=op.func_name,
            param_barcodes=tuple(op._param_barcodes),
            retain=raw_label in raw_label_set,
            pruned=False,
        )

    return RecurrenceGroupingGraph(
        nodes=nodes,
        raw_labels=tuple(raw_labels),
        source_labels=tuple(_raw_label(trace, label) for label in trace.input_layers),
        eligible_labels=tuple(raw_labels),
    )


def test_neutral_loop_grouping_matches_torch_recurrent_fixture() -> None:
    """Neutral grouping service reproduces torch recurrent member sets."""
    torch.manual_seed(0)
    traced = trace_fn(example_models.RecurrentParamsSimple(), torch.rand(5, 5))

    graph = _neutral_graph_from_torch_recurrent_fixture(traced)
    assignments = group_recurrent_nodes(graph)
    actual_groups = {
        frozenset(assignment.recurrent_labels)
        for assignment in assignments.values()
        if assignment.num_passes > 1
    }

    assert actual_groups == _raw_recurrent_member_sets(traced)
    assert all("control" not in node.data_parents for node in graph.nodes.values())
