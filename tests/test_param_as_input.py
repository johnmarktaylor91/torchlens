"""Characterization tests for tensors that are both inputs and parameter-like."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.validation as tl_validation
from torchlens.backends.torch._tl import TorchLensTLCollisionError


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


def test_plain_requires_grad_leaf_input_links_and_backprops_to_original() -> None:
    """Case A works and propagates gradients to the caller's tensor."""

    model = _TinyStimulusModel()
    x = _stimulus()

    trace = tl.trace(model, x, save_grads="all")
    trace.log_backward(trace[trace.output_layers[0]].out)

    input_label = trace.input_layers[0]
    first_op = _first_non_boundary_op(trace)

    assert input_label in first_op.parents
    assert trace[input_label].grad is not None
    assert x.grad is not None
    assert tl_validation.validate_forward_pass(model, x)


def test_same_device_parameter_input_currently_crashes_trace() -> None:
    """Case B currently crashes on CPU before a trace can be built."""

    model = _TinyStimulusModel()
    z = _parameter_stimulus()

    with pytest.raises(TorchLensTLCollisionError, match="Expected ParamMeta"):
        tl.trace(model, z)


def test_same_device_parameter_input_currently_crashes_validation() -> None:
    """Case B currently crashes validation with the same metadata collision."""

    model = _TinyStimulusModel()
    z = _parameter_stimulus()

    with pytest.raises(TorchLensTLCollisionError, match="Expected ParamMeta"):
        tl_validation.validate_forward_pass(model, z)


@pytest.mark.xfail(
    raises=TorchLensTLCollisionError,
    strict=True,
    reason="Parameter input clone keeps nn.Parameter type and collides TensorMeta with ParamMeta.",
)
def test_parameter_input_links_and_backprops_to_original_after_fix() -> None:
    """Case B should become an input parent and backpropagate to the caller's parameter."""

    model = _TinyStimulusModel()
    z = _parameter_stimulus()

    trace = tl.trace(model, z, save_grads="all")
    trace.log_backward(trace[trace.output_layers[0]].out)

    input_label = trace.input_layers[0]
    first_op = _first_non_boundary_op(trace)

    assert input_label in first_op.parents
    assert trace[input_label].grad is not None
    assert z.grad is not None
    assert tl_validation.validate_forward_pass(model, z)


@pytest.mark.xfail(
    raises=TorchLensTLCollisionError,
    strict=True,
    reason="Registered parameter fed as input currently hits the same TensorMeta/ParamMeta crash.",
)
def test_registered_model_parameter_fed_as_input_currently_broken() -> None:
    """Case C currently crashes before dual-role metadata can be inspected."""

    model = _TinyStimulusModel()
    model.stimulus = _parameter_stimulus()

    trace = tl.trace(model, model.stimulus)

    assert trace.input_layers


@pytest.mark.xfail(
    raises=TorchLensTLCollisionError,
    strict=True,
    reason="Fanout parameter inputs crash at first consumption before parent edges are recorded.",
)
def test_parameter_input_fanout_currently_broken() -> None:
    """A parameter input consumed by two ops currently crashes on CPU."""

    trace = tl.trace(_FanoutStimulusModel(), _parameter_stimulus())

    assert trace.input_layers


@pytest.mark.xfail(
    raises=TorchLensTLCollisionError,
    strict=True,
    reason="Nested parameter inputs are cloned as Parameters and crash at consumption.",
)
def test_nested_parameter_input_currently_broken() -> None:
    """A shallow container holding a parameter input currently crashes on CPU."""

    trace = tl.trace(_NestedStimulusModel(), [_parameter_stimulus()])

    assert trace.input_layers


@pytest.mark.xfail(
    raises=TorchLensTLCollisionError,
    strict=True,
    reason="Inference-only capture still labels a Parameter input before param classification.",
)
def test_inference_only_parameter_input_currently_broken() -> None:
    """inference_only=True does not avoid the current parameter-input crash."""

    trace = tl.trace(_TinyStimulusModel(), _parameter_stimulus(), inference_only=True)

    assert trace.input_layers


@pytest.mark.xfail(
    raises=TorchLensTLCollisionError,
    strict=True,
    reason="backward_ready capture still clones the Parameter input as a detached Parameter leaf.",
)
def test_backward_ready_parameter_input_currently_broken() -> None:
    """backward_ready=True does not avoid the current parameter-input crash."""

    trace = tl.trace(_TinyStimulusModel(), _parameter_stimulus(), backward_ready=True)

    assert trace.input_layers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cross-device check requires CUDA.")
def test_cross_device_parameter_input_matches_plain_tensor_path_if_cuda_available() -> None:
    """Cross-device Parameter-to-Tensor downgrade avoids the CPU crash when CUDA is available."""

    model = _CudaElementwiseStimulusModel().cuda()
    z = _parameter_stimulus()

    trace = tl.trace(model, z)

    input_label = trace.input_layers[0]
    assert any(input_label in op.parents for op in trace.layer_list if not op.is_input)
