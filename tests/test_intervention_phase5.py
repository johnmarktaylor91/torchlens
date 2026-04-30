"""Phase 5 postprocess preservation tests for intervention metadata."""

from __future__ import annotations

import torch

import torchlens as tl
from torchlens import MetadataInvariantError, check_metadata_invariants
from torchlens.intervention.types import ParentRef, TupleIndex
from torchlens.validation.invariants import check_func_call_id_invariant


class _SplitModel(torch.nn.Module):
    """Model with a multi-output torch operation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split and consume both outputs.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sum of split chunks.
        """

        left, right = torch.split(x, 1, dim=0)
        return left + right


class _LinearSplitModel(torch.nn.Module):
    """Module-containing split fixture for graph hash tests."""

    def __init__(self) -> None:
        """Create a single linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a linear layer, split its output, and recombine.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Recombined split outputs.
        """

        left, right = torch.split(self.linear(x), 1, dim=0)
        return left + right


class _DifferentGraphModel(torch.nn.Module):
    """Different graph fixture for hash sensitivity."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a different operation sequence.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output scaled by two.
        """

        return torch.relu(x) * 2


class _RecurrentReluModel(torch.nn.Module):
    """Small recurrent-style fixture with repeated operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the same operation in a Python loop.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Recurrent relu result.
        """

        out = x
        for _ in range(3):
            out = torch.relu(out)
        return out


def _parent_refs(value: object) -> list[ParentRef]:
    """Collect ``ParentRef`` leaves from a nested template.

    Parameters
    ----------
    value:
        Template value to traverse.

    Returns
    -------
    list[ParentRef]
        Parent references found in the template.
    """

    if isinstance(value, ParentRef):
        return [value]
    if isinstance(value, tuple | list):
        refs: list[ParentRef] = []
        for item in value:
            refs.extend(_parent_refs(item))
        return refs
    if hasattr(value, "__dataclass_fields__"):
        refs = []
        for field_name in value.__dataclass_fields__:
            refs.extend(_parent_refs(getattr(value, field_name)))
        return refs
    return []


def test_graph_shape_hash_is_stable_and_sensitive() -> None:
    """Graph hash matches same graph/shape and changes for a different graph."""

    log1 = tl.log_forward_pass(
        _LinearSplitModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )
    log2 = tl.log_forward_pass(
        _LinearSplitModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )
    log3 = tl.log_forward_pass(
        _DifferentGraphModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )

    assert log1.graph_shape_hash is not None
    assert len(log1.graph_shape_hash) == 64
    assert log1.graph_shape_hash == log2.graph_shape_hash
    assert log3.graph_shape_hash != log1.graph_shape_hash


def test_label_rewrite_preserves_templates_edges_and_call_groups() -> None:
    """Step 11 rewrites replay templates and edge records to final labels."""

    log = tl.log_forward_pass(
        _SplitModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )

    split_layers = [layer for layer in log.layer_list if layer.func_name == "split"]
    assert len(split_layers) == 2
    assert len({layer.func_call_id for layer in split_layers}) == 1
    assert {type(layer.output_path[0]) for layer in split_layers} == {TupleIndex}

    final_labels = set(log.layer_labels)
    for layer in log.layer_list:
        for edge in layer.edge_uses:
            assert edge.parent_label in final_labels
            assert edge.child_label in final_labels
        for template in (layer.captured_arg_template, layer.captured_kwarg_template):
            for parent_ref in _parent_refs(template):
                assert parent_ref.parent_label in final_labels


def test_keep_unsaved_layers_false_preserves_replay_call_group_siblings() -> None:
    """Step 12 keeps multi-output call groups atomic when activations are pruned."""

    log = tl.log_forward_pass(
        _SplitModel(),
        torch.randn(2, 3),
        vis_opt="none",
        layers_to_save="output",
        keep_unsaved_layers=False,
        intervention_ready=True,
    )

    split_layers = [layer for layer in log.layer_list if layer.func_name == "split"]
    assert len(split_layers) == 2
    assert not all(layer.has_saved_activations for layer in split_layers)
    assert len({layer.func_call_id for layer in split_layers}) == 1
    assert check_metadata_invariants(log) is True


def test_recurrent_iterations_keep_distinct_func_call_ids() -> None:
    """Loop detection does not regenerate or merge per-wrapper-call ids."""

    log = tl.log_forward_pass(
        _RecurrentReluModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )

    relu_layers = [layer for layer in log.layer_list if layer.func_name == "relu"]
    assert len(relu_layers) == 3
    assert len({layer.func_call_id for layer in relu_layers}) == 3
    assert check_metadata_invariants(log) is True


def test_func_call_id_invariant_catches_duplicate_output_paths() -> None:
    """Invariant S fails when same-call outputs reuse an output path."""

    log = tl.log_forward_pass(
        _SplitModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )
    split_layers = [layer for layer in log.layer_list if layer.func_name == "split"]
    split_layers[1].output_path = split_layers[0].output_path

    try:
        check_func_call_id_invariant(log)
    except MetadataInvariantError as exc:
        assert exc.check_name == "func_call_id_consistency"
    else:
        raise AssertionError("Invariant S did not catch duplicate output_path")


def test_module_address_normalized_strips_pass_qualifiers() -> None:
    """Hash preparation records pass-normalized module addresses on layers."""

    log = tl.log_forward_pass(
        _LinearSplitModel(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )
    linear_layers = [layer for layer in log.layer_list if layer.containing_module is not None]

    assert linear_layers
    assert any(layer.module_address_normalized == "linear" for layer in linear_layers)
    assert all(
        layer.module_address_normalized is None or ":" not in layer.module_address_normalized
        for layer in linear_layers
    )
