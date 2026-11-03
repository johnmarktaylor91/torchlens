"""Behavioral tests for the SAE splice-experiment bridge helper.

The splice experiment composes fork + push + ``tl.splice_module``: these
tests pin the causal contract (site actually replaced, downstream actually
recomputed), the fidelity metrics, the latent-edit knob, and the typed
refusals -- all against an in-test SAE, so no optional package is needed.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bridge import sae as sae_bridge


class _TinyMLP(nn.Module):
    """Two-layer MLP with one ReLU splice site."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(8, 16)
        self.out_proj = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.relu(self.in_proj(x)))


class _TinySAE(nn.Module):
    """Minimal encode/decode pair standing in for a real SAE."""

    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.enc = nn.Linear(d_in, d_hidden)
        self.dec = nn.Linear(d_hidden, d_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.enc(x))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.dec(latents)


@pytest.fixture()
def splice_setup():
    """Deterministic model, inputs, and SAE."""

    torch.manual_seed(7)
    model = _TinyMLP().eval()
    x = torch.randn(3, 8)
    sae = _TinySAE(16, 32).eval()
    return model, x, sae


def test_splice_accepts_existing_trace_and_refuses_double_inputs(splice_setup) -> None:
    """A ready trace works directly; passing inputs alongside it refuses."""

    model, x, sae = splice_setup
    clean = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    from_trace = sae_bridge.splice(clean, site="relu_1_2", sae=sae)
    from_model = sae_bridge.splice(model, x, site="relu_1_2", sae=sae)
    assert from_trace.reconstruction_mse == pytest.approx(from_model.reconstruction_mse, rel=1e-6)

    with pytest.raises(TypeError, match="takes no `inputs`"):
        sae_bridge.splice(clean, x, site="relu_1_2", sae=sae)
    with pytest.raises(TypeError, match="requires `inputs`"):
        sae_bridge.splice(model, site="relu_1_2", sae=sae)


def test_splice_refuses_sae_without_codec(splice_setup) -> None:
    """Objects without encode/decode refuse before any capture work."""

    model, x, _ = splice_setup
    with pytest.raises(TypeError, match="encode"):
        sae_bridge.splice(model, x, site="relu_1_2", sae=object())


def test_splice_refuses_non_tensor_decode(splice_setup) -> None:
    """A decode returning a non-tensor refuses typed at splice time."""

    model, x, sae = splice_setup

    class _BadSAE:
        def encode(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def decode(self, value: torch.Tensor) -> str:
            return "not a tensor"

    with pytest.raises(TypeError, match="decode"):
        sae_bridge.splice(model, x, site="relu_1_2", sae=_BadSAE())
