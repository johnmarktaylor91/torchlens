"""Phase 4b intervention replay-template and output-path tests."""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.capture.output_tensors import _walk_output_tensors_with_paths
from torchlens.intervention.types import (
    ContainerSpec,
    DataclassField,
    DictKey,
    HFKey,
    LiteralTensor,
    LiteralValue,
    NamedField,
    ParentRef,
    TupleIndex,
)


@dataclass(frozen=True)
class _DataclassOutput:
    """Dataclass fixture for path-aware output traversal."""

    hidden: torch.Tensor
    score: torch.Tensor


class _HFModelOutput(dict):
    """Minimal HuggingFace-style output object for traversal tests."""

    def __init__(self, **kwargs: Any) -> None:
        """Store output values in insertion order.

        Parameters
        ----------
        **kwargs:
            Output keys and values.
        """

        super().__init__(kwargs)

    def keys(self) -> Any:
        """Return output keys.

        Returns
        -------
        Any
            Dict key view.
        """

        return super().keys()


def _component_values(paths: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Convert output path components to comparable payload tuples.

    Parameters
    ----------
    paths:
        Raw output paths.

    Returns
    -------
    list[tuple[Any, ...]]
        Path component payloads.
    """

    values: list[tuple[Any, ...]] = []
    for path in paths:
        path_values = []
        for component in path:
            if hasattr(component, "index"):
                path_values.append(component.index)
            elif hasattr(component, "key"):
                path_values.append(component.key)
            elif hasattr(component, "name"):
                path_values.append(component.name)
            else:
                path_values.append(component)
        values.append(tuple(path_values))
    return values


@pytest.mark.smoke
def test_path_walker_supports_required_output_containers() -> None:
    """Path-aware traversal preserves supported output container structure."""

    x = torch.randn(2, 3)
    pair_type = namedtuple("Pair", ["left", "right"])
    cases = [
        (x, [()], [None]),
        ((x, x + 1), [(TupleIndex(0),), (TupleIndex(1),)], ["tuple"]),
        ([x, x + 1], [(TupleIndex(0),), (TupleIndex(1),)], ["list"]),
        ({"a": x, "b": x + 1}, [(DictKey("a"),), (DictKey("b"),)], ["dict"]),
        (pair_type(x, x + 1), [(NamedField("left"),), (NamedField("right"),)], ["namedtuple"]),
        (
            _DataclassOutput(x, x + 1),
            [(DataclassField("hidden"),), (DataclassField("score"),)],
            ["dataclass"],
        ),
        (
            _HFModelOutput(last_hidden_state=x, logits=x + 1),
            [(HFKey("last_hidden_state"),), (HFKey("logits"),)],
            ["hf_model_output"],
        ),
    ]

    for output, expected_paths, expected_kinds in cases:
        entries = list(_walk_output_tensors_with_paths(output))
        paths = [entry[1] for entry in entries]
        specs = [entry[2] for entry in entries if entry[2] is not None]

        assert _component_values(paths) == _component_values(expected_paths)
        if expected_kinds == [None]:
            assert specs == []
        else:
            assert specs
            assert all(isinstance(spec, ContainerSpec) for spec in specs)
            assert {spec.kind for spec in specs} == set(expected_kinds)


@pytest.mark.smoke
def test_intervention_ready_records_unique_output_paths_for_multi_output_op() -> None:
    """Multi-output torch calls share call id while preserving per-output paths."""

    class SplitModel(torch.nn.Module):
        """Model with a tuple-returning torch operation."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Split and consume both outputs.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Sum of split outputs.
            """

            left, right = torch.split(x, 1, dim=0)
            return left + right

    log = tl.log_forward_pass(
        SplitModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )

    split_layers = [layer for layer in log.layer_list if layer.func_name == "split"]
    assert len(split_layers) == 2
    assert len({layer.func_call_id for layer in split_layers}) == 1
    assert len({layer.output_path for layer in split_layers}) == 2
    assert {type(layer.output_path[0]) for layer in split_layers} == {TupleIndex}
    assert all(layer.container_spec is not None for layer in split_layers)


@pytest.mark.smoke
def test_replay_templates_classify_parent_literals_and_literal_tensors() -> None:
    """Captured templates classify parent refs, literal values, and tensor literals."""

    class LinearShift(torch.nn.Module):
        """Parameterized model for replay-template classification."""

        def __init__(self) -> None:
            """Create one linear layer."""

            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply a linear layer and a literal shift.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Shifted linear output.
            """

            return self.linear(x) + 1

    log = tl.log_forward_pass(
        LinearShift(),
        torch.randn(4, 3),
        vis_opt="none",
        intervention_ready=True,
    )

    templates = [
        layer.captured_arg_template
        for layer in log.layer_list
        if layer.captured_arg_template is not None
    ]
    flattened = [component for template in templates for component in template.args]

    assert any(isinstance(component, ParentRef) for component in flattened)
    assert any(isinstance(component, LiteralTensor) for component in flattened)
    assert any(isinstance(component, LiteralValue) for component in flattened)


@pytest.mark.smoke
def test_edge_uses_extend_parent_arg_locs_without_replacing_them() -> None:
    """Edge provenance agrees with existing parent-layer arg locations."""

    class AddRelus(torch.nn.Module):
        """Model with repeated parent use."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add two relu calls from the same input.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Sum of relu outputs.
            """

            return torch.relu(x) + torch.relu(x)

    log = tl.log_forward_pass(
        AddRelus(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )

    edge_layers = [layer for layer in log.layer_list if layer.edge_uses]
    assert edge_layers
    for layer in edge_layers:
        loc_labels = set(layer.parent_layer_arg_locs["args"].values()) | set(
            layer.parent_layer_arg_locs["kwargs"].values()
        )
        edge_labels = {
            log._raw_to_final_layer_labels.get(edge.parent_label, edge.parent_label)
            for edge in layer.edge_uses
        }
        assert edge_labels == loc_labels
        assert all(
            log._raw_to_final_layer_labels.get(edge.child_label, edge.child_label)
            == layer.layer_label
            for edge in layer.edge_uses
        )
        assert all(edge.child_func_call_id == layer.func_call_id for edge in layer.edge_uses)


@pytest.mark.smoke
def test_non_intervention_ready_capture_leaves_templates_and_edges_empty() -> None:
    """Default capture avoids Phase 4b template and edge-provenance overhead."""

    class TinyModel(torch.nn.Module):
        """Simple model for non-ready gating."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply a relu and literal shift.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Shifted relu output.
            """

            return torch.relu(x) + 1

    log = tl.log_forward_pass(TinyModel(), torch.randn(2, 3), vis_opt="none")

    assert all(layer.captured_arg_template is None for layer in log.layer_list)
    assert all(layer.captured_kwarg_template is None for layer in log.layer_list)
    assert all(layer.edge_uses == [] for layer in log.layer_list)
