"""The opt-in logged splice: ``splice(..., log_latents=True)`` puts the latents in the trace.

On, ``result.spliced`` is a fresh capture of the live model with the SAE run
inside it as ordinary ops, so its encoder ops, the latents node and its
decoder ops sit between the site and its consumers; selectors and summaries
see them, and the latents node equals ``result.latents``. Off (the default),
the result and its spliced trace are the replayed fork they always were. The
spliced outputs are identical either way. Expected values come from a hand
forward of the same weights; the fixture is the hand-set SAE of
``test_sae4_latent_facets`` (dyadic weights, exact float32 arithmetic), so
equalities are exact (``torch.equal``) unless a tolerance is stated.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from tests.test_sae4_latent_facets import X, _HandSAE, _latents, _Model
from torchlens._errors import InvalidArgumentError


def _splice(model: nn.Module, x: torch.Tensor, sae: nn.Module, **kwargs):
    return tl.bridge.sae.splice(model, x, site="fc1", sae=sae, **kwargs)


def _labels(trace: tl.Trace) -> list[str]:
    return [op.label for op in trace._layer_list]


@pytest.fixture()
def setup():
    return _Model().eval(), torch.tensor(X), _HandSAE()


def test_logged_latents_are_trace_nodes_equal_to_the_result_latents(setup) -> None:
    model, x, sae = setup
    result = _splice(model, x, sae, log_latents=True)
    assert result.latents_site is not None
    node = result.spliced[result.latents_site]
    torch.testing.assert_close(node.out, result.latents, atol=0.0, rtol=0.0)
    torch.testing.assert_close(node.out, torch.tensor([_latents(r) for r in X]), atol=0.0, rtol=0.0)
    labels = _labels(result.spliced)
    site, consumer = labels.index("linear_1_1:1"), labels.index(result.latents_site)
    assert site < consumer < labels.index(next(lab for lab in labels if lab.startswith("linear_2")))
    # Selectors see the node: the latents op is a relu of the logged trace.
    relus = list(result.spliced[tl.func("relu")].labels)
    assert result.latents_site in relus
    assert result.latents_site.partition(":")[0] in str(result.spliced.summary())


def test_logged_splice_refuses_a_trace_door_with_a_running_remedy(setup) -> None:
    model, x, sae = setup
    clean = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    with pytest.raises(InvalidArgumentError) as caught:
        tl.bridge.sae.splice(clean, site="fc1", sae=sae, log_latents=True)
    assert caught.value.fields["code"] == "sae_splice_logged_needs_live_model"
    assert "splice(model, inputs, site=..., sae=..., log_latents=True)" in str(caught.value)
    assert _splice(model, x, sae, log_latents=True).latents_site is not None


class _Block(nn.Module):
    """Pre-norm residual block with a named MLP (the transcoder test's shape)."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x + self.mlp(self.norm(x)))


class _Transcoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 6)
        self.dec = nn.Linear(6, 4)

    def forward(self, mlp_input: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.relu(self.enc(mlp_input)))


def test_transcoder_splices_keep_working_beside_a_logged_splice() -> None:
    """``splice_module(transcoder, input_route="in")`` is unchanged after a logged splice ran."""

    torch.manual_seed(3)
    model, x, transcoder = _Block().eval(), torch.randn(3, 4), _Transcoder().eval()

    def transcoded() -> tl.Trace:
        return tl.trace(
            model,
            x,
            capture=tl.options.CaptureOptions(
                replayable=True,
                hooks={tl.module("mlp"): tl.splice_module(transcoder, input_route="in")},
            ),
        )

    with torch.no_grad():
        mlp_out = transcoder(model.norm(x))
        expected = model.head(x + mlp_out)
    before = transcoded()

    class _MlpSAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Linear(4, 6)
            self.dec = nn.Linear(6, 4)

        def encode(self, h: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.enc(h))

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.dec(z)

    logged = tl.bridge.sae.splice(model, x, site="mlp", sae=_MlpSAE(), log_latents=True)
    assert logged.latents_site is not None
    after = transcoded()
    for trace in (before, after):
        torch.testing.assert_close(trace["mlp"].out, mlp_out, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(trace[trace.output_layers[0]].out, expected, atol=1e-6, rtol=0.0)


class _PairOut(nn.Module):
    """Returns a tuple, so a hook on it sees a non-tensor output."""

    def __init__(self) -> None:
        super().__init__()
        self.inner = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.inner(x)
        return h, h + 1


class _TupleModel(nn.Module):
    """``pair`` returns its child's output inside a tuple."""

    def __init__(self) -> None:
        super().__init__()
        self.pair = _PairOut()
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.pair(x)
        return self.fc2(a * b)


def test_logged_splice_at_a_tuple_returning_module_refuses_and_the_unlogged_splice_runs() -> None:
    torch.manual_seed(0)
    model, x, sae = _TupleModel().eval(), torch.randn(2, 4), _HandSAE()
    with pytest.raises(InvalidArgumentError) as caught:
        tl.bridge.sae.splice(model, x, site="pair.inner", sae=sae, log_latents=True)
    assert caught.value.fields["code"] == "sae_splice_logged_output_not_tensor"
    result = tl.bridge.sae.splice(model, x, site="pair.inner", sae=sae)
    assert result.latents_site is None and result.latents.shape == (2, 3)
