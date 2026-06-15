"""Regression tests for Torch backend per-output field copying."""

from __future__ import annotations

import copy
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.backends.torch import ops as torch_ops


class _SingleOutputModel(torch.nn.Module):
    """Model with one ordinary tensor output from a wrapped torch call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a single-output torch operation.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU output.
        """

        return torch.relu(x)


class _SplitOutputModel(torch.nn.Module):
    """Model with a multi-output wrapped torch call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split an input tensor and consume both outputs.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sum of both split outputs.
        """

        left, right = torch.split(x, 1, dim=0)
        return left + right


def test_single_output_fast_path_traces_without_per_output_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-output torch calls reuse shared fields while preserving metadata."""

    copy_calls: list[dict[str, Any]] = []
    original_copy = torch_ops._copy_shared_fields_for_output

    def counted_copy(fields_dict: dict[str, Any]) -> dict[str, Any]:
        """Count calls to the multi-output isolation helper."""

        copy_calls.append(fields_dict)
        return original_copy(fields_dict)

    monkeypatch.setattr(torch_ops, "_copy_shared_fields_for_output", counted_copy)

    trace = tl.trace(_SingleOutputModel(), torch.randn(2, 3))

    relu_layers = [layer for layer in trace.layer_list if layer.func_name == "relu"]
    assert len(relu_layers) == 1
    relu_layer = relu_layers[0]
    assert relu_layer.in_multi_output is False
    assert relu_layer.multi_output_index is None
    assert relu_layer.container_path == ()
    assert relu_layer.container_spec is None
    assert relu_layer.parent_arg_positions["args"]
    assert copy_calls == []


def test_multi_output_fast_path_keeps_sibling_field_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Multi-output torch calls still isolate per-output mutable metadata."""

    copy_call_count = 0
    original_copy = torch_ops._copy_shared_fields_for_output

    def counted_copy(fields_dict: dict[str, Any]) -> dict[str, Any]:
        """Count calls to the multi-output isolation helper."""

        nonlocal copy_call_count
        copy_call_count += 1
        return original_copy(fields_dict)

    monkeypatch.setattr(torch_ops, "_copy_shared_fields_for_output", counted_copy)

    trace = tl.trace(_SplitOutputModel(), torch.randn(2, 3))

    split_layers = [layer for layer in trace.layer_list if layer.func_name == "split"]
    assert len(split_layers) == 2
    assert copy_call_count == 2

    first_positions = split_layers[0].parent_arg_positions
    sibling_positions = split_layers[1].parent_arg_positions
    sibling_positions_before = copy.deepcopy(sibling_positions)

    first_arg_key = next(iter(first_positions["args"]))
    first_positions["args"][first_arg_key] = "sentinel_parent"

    assert sibling_positions == sibling_positions_before
