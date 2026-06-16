"""Characterization tests for tensors that are both inputs and parameter-like."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.validation as tl_validation
from torchlens.intervention.types import ParentRef


class _TinyStimulusModel(nn.Module):
    """Tiny model that consumes a differentiable stimulus input."""

    def __init__(self) -> None:
        """Initialize deterministic weights."""

        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[0.25, -0.50], [0.75, 0.125]],
                dtype=torch.float32,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two tensor ops to the stimulus.

        Parameters
        ----------
        x:
            Stimulus tensor.

        Returns
        -------
        torch.Tensor
            Scalar model output.
        """

        return torch.relu(x @ self.weight).sum()


class _FanoutStimulusModel(nn.Module):
    """Model that consumes the same input parameter in two ops."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the stimulus in two separate branches.

        Parameters
        ----------
        x:
            Stimulus tensor.

        Returns
        -------
        torch.Tensor
            Scalar model output.
        """

        return (torch.relu(x) + torch.sigmoid(x)).sum()


class _NestedStimulusModel(nn.Module):
    """Model whose stimulus arrives inside a shallow container."""

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Consume the first tensor in ``inputs``.

        Parameters
        ----------
        inputs:
            Tensor-bearing input list.

        Returns
        -------
        torch.Tensor
            Scalar model output.
        """

        return torch.relu(inputs[0]).sum()


class _CudaElementwiseStimulusModel(nn.Module):
    """CUDA-resident elementwise model for cross-device input movement tests."""

    def __init__(self) -> None:
        """Initialize a CUDA buffer so TorchLens selects the CUDA model device."""

        super().__init__()
        self.register_buffer("shift", torch.tensor([[0.5, -0.25]], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply deterministic elementwise ops to a moved stimulus.

        Parameters
        ----------
        x:
            Stimulus tensor.

        Returns
        -------
        torch.Tensor
            Scalar model output.
        """

        return torch.relu(x + self.shift).sum()


def _stimulus() -> torch.Tensor:
    """Return a deterministic differentiable stimulus tensor.

    Returns
    -------
    torch.Tensor
        Leaf tensor requiring gradients.
    """

    return torch.tensor([[1.0, -2.0]], dtype=torch.float32, requires_grad=True)


def _parameter_stimulus() -> nn.Parameter:
    """Return a deterministic parameter-wrapped stimulus tensor.

    Returns
    -------
    nn.Parameter
        Leaf parameter used as a forward input.
    """

    return nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float32))


def _first_non_boundary_op(trace: tl.Trace) -> Any:
    """Return the first captured op that is not an input or output boundary.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.

    Returns
    -------
    Any
        First non-boundary operation.
    """

    return next(op for op in trace.layer_list if not op.is_input and not op.is_output)


def _assert_input_parent_edge(trace: tl.Trace) -> None:
    """Assert that the first input feeds at least one captured child op.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    """

    input_label = trace.input_layers[0]
    assert any(input_label in op.parents for op in trace.layer_list if not op.is_input)


def test_plain_requires_grad_leaf_input_links_and_backprops_to_original() -> None:
    """Case A works and propagates gradients to the caller's tensor."""

    model = _TinyStimulusModel()
    x = _stimulus()

    trace = tl.trace(model, x, save_grads="all")
    trace.log_backward(trace[trace.output_layers[0]].out)

    input_label = trace.input_layers[0]
    _assert_input_parent_edge(trace)
    assert trace[input_label].grad is not None
    assert x.grad is not None
    assert tl_validation.validate_forward_pass(model, x)


def test_same_device_parameter_input_traces_as_input_parent() -> None:
    """Case B traces on CPU with the parameter stimulus represented as an input."""

    model = _TinyStimulusModel()
    z = _parameter_stimulus()

    trace = tl.trace(model, z)

    _assert_input_parent_edge(trace)
    assert len(trace.params) == 1


def test_same_device_parameter_input_passes_validation() -> None:
    """Case B passes forward validation without any metadata-collision exemption."""

    model = _TinyStimulusModel()
    z = _parameter_stimulus()

    assert tl_validation.validate_forward_pass(model, z)


def test_parameter_input_links_and_backprops_to_original_after_fix() -> None:
    """Case B should become an input parent and backpropagate to the caller's parameter."""

    model = _TinyStimulusModel()
    z = _parameter_stimulus()

    trace = tl.trace(model, z, save_grads="all")
    trace.log_backward(trace[trace.output_layers[0]].out)

    input_label = trace.input_layers[0]
    _assert_input_parent_edge(trace)
    assert trace[input_label].grad is not None
    assert z.grad is not None
    assert tl_validation.validate_forward_pass(model, z)


def test_parameter_input_replay_template_uses_parent_ref() -> None:
    """Replay templates classify a parameter stimulus input as a parent reference."""

    model = _TinyStimulusModel()
    z = _parameter_stimulus()

    trace = tl.trace(model, z, intervention_ready=True)
    first_op = _first_non_boundary_op(trace)

    assert first_op.args_template is not None
    assert isinstance(first_op.args_template.args[0], ParentRef)
    assert first_op.args_template.args[0].parent_label == trace.input_layers[0]


def test_registered_model_parameter_fed_as_input_traces() -> None:
    """Case C traces when a registered parameter is also passed as input."""

    model = _TinyStimulusModel()
    model.stimulus = _parameter_stimulus()

    trace = tl.trace(model, model.stimulus)

    _assert_input_parent_edge(trace)


def test_parameter_input_fanout_links_to_each_consuming_op() -> None:
    """A parameter input consumed by two ops links to both child branches."""

    trace = tl.trace(_FanoutStimulusModel(), _parameter_stimulus())

    input_label = trace.input_layers[0]
    children = [op for op in trace.layer_list if input_label in op.parents]
    assert {child.func_name for child in children} >= {"relu", "sigmoid"}


def test_nested_parameter_input_links_to_child_op() -> None:
    """A shallow container holding a parameter input traces as a normal input."""

    trace = tl.trace(_NestedStimulusModel(), [_parameter_stimulus()])

    _assert_input_parent_edge(trace)


def test_inference_only_parameter_input_traces() -> None:
    """inference_only=True traces a parameter stimulus as a normal input."""

    trace = tl.trace(_TinyStimulusModel(), _parameter_stimulus(), inference_only=True)

    _assert_input_parent_edge(trace)


def test_backward_ready_parameter_input_traces() -> None:
    """backward_ready=True traces a parameter stimulus as a normal input."""

    trace = tl.trace(_TinyStimulusModel(), _parameter_stimulus(), backward_ready=True)

    _assert_input_parent_edge(trace)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cross-device check requires CUDA.")
def test_cross_device_parameter_input_matches_plain_tensor_path_if_cuda_available() -> None:
    """Cross-device Parameter-to-Tensor downgrade avoids the CPU crash when CUDA is available."""

    model = _CudaElementwiseStimulusModel().cuda()
    z = _parameter_stimulus()

    trace = tl.trace(model, z)

    input_label = trace.input_layers[0]
    assert any(input_label in op.parents for op in trace.layer_list if not op.is_input)
