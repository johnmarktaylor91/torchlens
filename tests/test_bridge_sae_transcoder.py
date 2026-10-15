"""Transcoder support: ``tl.splice_module(transcoder, input="in")`` on an MLP module.

A transcoder maps an MLP's input to its output. Spliced with ``input="in"`` at
the MLP module, it runs on the arguments the MLP was called with and its result
replaces the MLP's output; the residual stream and everything downstream then
recompute from it. The fixture's outputs are checked against a hand forward.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class _Block(nn.Module):
    """Pre-norm residual block with a named MLP, as in a transformer layer."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, width))
        self.head = nn.Linear(width, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.mlp(self.norm(x))
        return self.head(h)


class _Transcoder(nn.Module):
    """A sparse map from the MLP's input space to its output space."""

    def __init__(self, width: int = 8, latents: int = 24) -> None:
        super().__init__()
        self.enc = nn.Linear(width, latents)
        self.dec = nn.Linear(latents, width)

    def forward(self, mlp_input: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.relu(self.enc(mlp_input)))


@pytest.fixture()
def setup():
    torch.manual_seed(11)
    return _Block().eval(), torch.randn(4, 8), _Transcoder().eval()


def _hand(model: _Block, x: torch.Tensor, transcoder: _Transcoder) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        mlp_out = transcoder(model.norm(x))
        return mlp_out, model.head(x + mlp_out)


def test_transcoder_spliced_with_input_in_equals_a_hand_forward(setup) -> None:
    """The MLP output is the transcoder of the MLP input; the output follows."""

    model, x, transcoder = setup
    trace = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(
            intervention_ready=True,
            hooks={tl.module("mlp"): tl.splice_module(transcoder, input="in")},
        ),
    )
    mlp_out, out = _hand(model, x, transcoder)
    assert torch.allclose(trace["mlp"].out, mlp_out, atol=1e-6)
    assert torch.allclose(trace[trace.output_layers[0]].out, out, atol=1e-6)


def test_transcoder_reads_the_mlp_input_not_its_output(setup) -> None:
    """``input="out"`` would feed the MLP's own output: a different result."""

    model, x, transcoder = setup
    as_sae = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(
            intervention_ready=True,
            hooks={tl.module("mlp"): tl.splice_module(transcoder, input="out")},
        ),
    )
    with torch.no_grad():
        expected_sae = transcoder(model.mlp(model.norm(x)))
    assert torch.allclose(as_sae["mlp"].out, expected_sae, atol=1e-6)
    mlp_out, _out = _hand(model, x, transcoder)
    assert not torch.allclose(as_sae["mlp"].out, mlp_out, atol=1e-3)
