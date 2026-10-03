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


def test_splice_replaces_site_and_recomputes_downstream(splice_setup) -> None:
    """The fork carries the reconstruction at the site and re-derived outputs."""

    model, x, sae = splice_setup
    result = sae_bridge.splice(model, x, site="relu_1_2", sae=sae)

    assert result.site == "relu_1_2:1"
    # The spliced fork's site payload IS the reconstruction, not the clean out.
    assert torch.allclose(result.spliced[result.site].out, result.reconstruction, atol=1e-6)
    assert torch.allclose(result.clean[result.site].out, result.clean_site_out)
    # Downstream really recomputed: outputs equal a manual forward from the splice.
    with torch.no_grad():
        expected = model.out_proj(sae.decode(sae.encode(result.clean_site_out)))
    assert torch.allclose(result.spliced_outputs[0], expected, atol=1e-5)
    assert result.output_delta_l2 > 0.0
    assert result.output_delta_max > 0.0


def test_splice_metrics_are_exact(splice_setup) -> None:
    """MSE and fraction-of-variance-explained match a hand computation."""

    model, x, sae = splice_setup
    result = sae_bridge.splice(model, x, site="relu_1_2", sae=sae)

    residual = result.reconstruction - result.clean_site_out
    expected_mse = float(residual.pow(2).mean())
    centered = result.clean_site_out - result.clean_site_out.mean()
    expected_fve = 1.0 - float(residual.pow(2).sum()) / float(centered.pow(2).sum())
    assert result.reconstruction_mse == pytest.approx(expected_mse, rel=1e-6)
    assert result.fraction_variance_explained == pytest.approx(expected_fve, rel=1e-6)


def test_splice_latents_edit_is_causal(splice_setup) -> None:
    """Editing latents changes the reconstruction the fork actually consumed."""

    model, x, sae = splice_setup
    full = sae_bridge.splice(model, x, site="relu_1_2", sae=sae)
    ablated = sae_bridge.splice(
        model, x, site="relu_1_2", sae=sae, latents_edit=lambda z: torch.zeros_like(z)
    )

    with torch.no_grad():
        zero_recon = sae.decode(torch.zeros_like(sae.encode(full.clean_site_out)))
    assert torch.allclose(ablated.reconstruction, zero_recon, atol=1e-6)
    assert torch.allclose(ablated.spliced[ablated.site].out, zero_recon, atol=1e-6)
    assert not torch.allclose(ablated.spliced_outputs[0], full.spliced_outputs[0])


@pytest.mark.smoke
def test_splice_accepts_existing_trace_and_refuses_double_inputs(splice_setup) -> None:
    """A ready trace works directly; passing inputs alongside it refuses."""

    model, x, sae = splice_setup
    clean = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
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
