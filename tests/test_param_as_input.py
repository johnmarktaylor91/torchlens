"""Characterization tests for tensors that are both inputs and parameter-like."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl


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


class _LoopStimulusModel(nn.Module):
    """Model with repeated ops fed by a parameter stimulus."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a fixed recurrent op pattern.

        Parameters
        ----------
        x:
            Stimulus tensor.

        Returns
        -------
        torch.Tensor
            Scalar model output.
        """

        for _ in range(3):
            x = torch.relu(x + 1)
        return x.sum()


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

    return next(op for op in trace._layer_list if not op.is_input and not op.is_output)


def _assert_input_parent_edge(trace: tl.Trace) -> None:
    """Assert that the first input feeds at least one captured child op.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    """

    input_label = trace.input_layers[0]
    assert any(input_label in op.parents for op in trace._layer_list if not op.is_input)


def test_parameter_input_tlspec_roundtrip_preserves_metadata(tmp_path: Path) -> None:
    """A parameter-input trace saves and loads through tlspec with metadata intact."""

    bundle_path = tmp_path / "parameter_input.tlspec"
    trace = tl.trace(_TinyStimulusModel(), _parameter_stimulus())

    trace.save(bundle_path)
    loaded = tl.load(bundle_path)

    input_label = loaded.input_layers[0]
    assert loaded[input_label].input_was_parameter is True
    assert any(input_label in op.parents for op in loaded._layer_list if not op.is_input)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cross-device check requires CUDA.")
def test_cross_device_parameter_input_matches_plain_tensor_path_if_cuda_available() -> None:
    """Cross-device Parameter-to-Tensor downgrade avoids the CPU crash when CUDA is available."""

    model = _CudaElementwiseStimulusModel().cuda()
    z = _parameter_stimulus()

    trace = tl.trace(model, z)

    input_label = trace.input_layers[0]
    assert any(input_label in op.parents for op in trace._layer_list if not op.is_input)
    assert trace[input_label].input_was_parameter is True
