"""Splice-result honesty: variance explained, latents, zero variance, CE recovered and KL.

Every new ``SpliceResult`` field is checked against a hand computation in
float64 on a tiny model with a hand-built SAE: the per-dimension (pooled and
by-dimension) variance explained, the latents (exactly the encoder output),
the zero-variance flag (``nan`` with a reason, never 0.0), and on a tiny
language model the next-token cross-entropy triple, the loss recovered and the
KL divergence, both from the result's own logits and from a direct forward.
"""

from __future__ import annotations

import math
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.bridge import sae as sae_bridge
from torchlens.errors import TorchLensWarning


class _TinyMLP(nn.Module):
    """Two-layer MLP with one ReLU splice site."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Linear(8, 16)
        self.out_proj = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(torch.relu(self.in_proj(x)))


class _TinyLM(nn.Module):
    """Embedding, one ReLU MLP site, unembedding: a next-token model."""

    def __init__(self, vocab: int = 11, width: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.hidden = nn.Linear(width, width)
        self.unembed = nn.Linear(width, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.unembed(torch.relu(self.hidden(self.embed(tokens))))


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


@pytest.fixture(autouse=True)
def _fresh_warning_codes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test sees the once-per-session warnings as not yet fired."""

    monkeypatch.setattr(sae_bridge, "_WARNED_CODES", set())


@pytest.fixture()
def mlp_setup():
    torch.manual_seed(7)
    return _TinyMLP().eval(), torch.randn(5, 8), _TinySAE(16, 32).eval()


@pytest.fixture()
def lm_setup():
    torch.manual_seed(3)
    model = _TinyLM().eval()
    tokens = torch.randint(0, 11, (2, 6))
    return model, tokens, _TinySAE(8, 24).eval()


def _hand_variance(clean: torch.Tensor, recon: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """Global-mean FVE, pooled per-dimension FVE and the per-dimension vector, float64."""

    x = clean.detach().double().reshape(-1, clean.shape[-1])
    r = recon.detach().double().reshape(-1, clean.shape[-1])
    sse_d = ((r - x) ** 2).sum(0)
    sst_d = ((x - x.mean(0)) ** 2).sum(0)
    global_fve = 1.0 - float(((r - x) ** 2).sum()) / float(((x - x.mean()) ** 2).sum())
    return global_fve, 1.0 - float(sse_d.sum()) / float(sst_d.sum()), 1.0 - sse_d / sst_d


def _hand_ce(logits: torch.Tensor, tokens: torch.Tensor, start: int) -> float:
    """Mean of logsumexp(l[t]) - l[t, token[t+1]] over predicting positions, float64."""

    total, count = 0.0, 0
    lg = logits.detach().double()
    for b in range(tokens.shape[0]):
        for t in range(start, tokens.shape[1] - 1):
            row = lg[b, t]
            total += float(torch.logsumexp(row, 0) - row[int(tokens[b, t + 1])])
            count += 1
    return total / count


def _hand_kl(clean_logits: torch.Tensor, spliced_logits: torch.Tensor, start: int) -> float:
    """Mean over positions of sum_v p_clean (log p_clean - log p_spliced), float64."""

    lc = torch.log_softmax(clean_logits.detach().double(), -1)[:, start:]
    ls = torch.log_softmax(spliced_logits.detach().double(), -1)[:, start:]
    per_position = (lc.exp() * (lc - ls)).sum(-1)
    return float(per_position.sum()) / per_position.numel()


def test_variance_fields_equal_a_float64_hand_computation(mlp_setup) -> None:
    """Global, pooled and per-dimension variance explained match float64 by hand."""

    model, x, sae = mlp_setup
    result = sae_bridge.splice(model, x, site="relu_1_2", sae=sae)

    global_fve, pooled, by_dim = _hand_variance(result.clean_site_out, result.reconstruction)
    residual = result.reconstruction.double() - result.clean_site_out.double()
    assert result.reconstruction_mse == pytest.approx(float((residual**2).mean()), rel=1e-12)
    assert result.fraction_variance_explained == pytest.approx(global_fve, rel=1e-12)
    assert result.fraction_variance_explained_pooled == pytest.approx(pooled, rel=1e-12)
    assert result.fraction_variance_explained_by_dim.dtype == torch.float64
    finite = torch.isfinite(by_dim)
    assert torch.equal(torch.isfinite(result.fraction_variance_explained_by_dim), finite)
    assert torch.allclose(
        result.fraction_variance_explained_by_dim[finite], by_dim[finite], rtol=1e-12, atol=0
    )
    # ReLU units dead on the whole batch have zero variance: nan there, never 0.0.
    assert torch.isnan(result.fraction_variance_explained_by_dim[~finite]).all()
    assert result.zero_variance is False
    assert "fraction_variance_explained" not in result.unavailable_metrics


def test_latents_are_the_encoder_output_exactly(mlp_setup) -> None:
    """Kept by default; equal to sae.encode(clean site); edits kept beside them."""

    model, x, sae = mlp_setup
    result = sae_bridge.splice(model, x, site="relu_1_2", sae=sae)
    with torch.no_grad():
        expected = sae.encode(result.clean_site_out)
    assert torch.equal(result.latents, expected)
    assert result.edited_latents is None

    def keep_first_four(z: torch.Tensor) -> torch.Tensor:
        return z * (torch.arange(z.shape[-1]) < 4)

    edited = sae_bridge.splice(model, x, site="relu_1_2", sae=sae, latents_edit=keep_first_four)
    assert torch.equal(edited.latents, expected)
    assert torch.equal(edited.edited_latents, keep_first_four(expected))
    with torch.no_grad():
        assert torch.allclose(edited.reconstruction, sae.decode(keep_first_four(expected)))


def test_keep_latents_false_drops_them_and_the_metrics_stay(mlp_setup) -> None:
    """The cheaper spelling keeps every metric and no latents."""

    model, x, sae = mlp_setup
    full = sae_bridge.splice(model, x, site="relu_1_2", sae=sae)
    lean = sae_bridge.splice(model, x, site="relu_1_2", sae=sae, keep_latents=False)
    assert lean.latents is None and lean.edited_latents is None
    assert lean.fraction_variance_explained_pooled == full.fraction_variance_explained_pooled


def test_large_latents_warn_once_and_name_a_spelling_that_runs(
    mlp_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the size guard: one coded warning per session; its remedy runs silently."""

    model, x, sae = mlp_setup
    monkeypatch.setattr(sae_bridge, "_LATENTS_WARN_BYTES", 16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sae_bridge.splice(model, x, site="relu_1_2", sae=sae)
        sae_bridge.splice(model, x, site="relu_1_2", sae=sae)
    coded = [w for w in caught if isinstance(w.message, TorchLensWarning)]
    assert [w.message.fields["code"] for w in coded] == ["sae_splice_latents_large"]
    message = str(coded[0].message)
    assert "'relu_1_2:1'" in message and "keep_latents=False" in message
    assert coded[0].filename == __file__

    monkeypatch.setattr(sae_bridge, "_WARNED_CODES", set())
    with warnings.catch_warnings(record=True) as quiet:
        warnings.simplefilter("always")
        sae_bridge.splice(model, x, site="relu_1_2", sae=sae, keep_latents=False)
    assert not [w for w in quiet if isinstance(w.message, TorchLensWarning)]


def test_zero_variance_sets_the_flag_and_never_returns_zero() -> None:
    """A constant site: nan, the flag, a reason per field and one coded warning."""

    model = _TinyMLP().eval()
    with torch.no_grad():
        model.in_proj.weight.zero_()
        model.in_proj.bias.fill_(0.5)
    sae = _TinySAE(16, 32).eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = sae_bridge.splice(model, torch.randn(4, 8), site="relu_1_2", sae=sae)
    assert result.zero_variance is True
    assert math.isnan(result.fraction_variance_explained)
    assert math.isnan(result.fraction_variance_explained_pooled)
    assert torch.isnan(result.fraction_variance_explained_by_dim).all()
    assert set(result.unavailable_metrics) >= {
        "fraction_variance_explained",
        "fraction_variance_explained_pooled",
    }
    coded = [w.message for w in caught if isinstance(w.message, TorchLensWarning)]
    assert [w.fields["code"] for w in coded] == ["sae_splice_zero_variance"]
    assert "Remedy:" in str(coded[0])


def test_single_token_has_no_per_dimension_variance(mlp_setup) -> None:
    """One token: the global number exists, the pooled one is nan and flagged."""

    model, _x, sae = mlp_setup
    result = sae_bridge.splice(model, torch.randn(1, 8), site="relu_1_2", sae=sae)
    assert math.isfinite(result.fraction_variance_explained)
    assert math.isnan(result.fraction_variance_explained_pooled)
    assert result.zero_variance is True
    assert "two tokens" in result.unavailable_metrics["fraction_variance_explained_pooled"]


def test_next_token_metrics_are_none_with_a_reason_without_tokens(mlp_setup) -> None:
    """No tokens=: every CE and KL field is None and says why."""

    model, x, sae = mlp_setup
    result = sae_bridge.splice(model, x, site="relu_1_2", sae=sae)
    for name in (
        "ce_loss_clean",
        "ce_loss_spliced",
        "ce_loss_zero_ablated",
        "ce_loss_recovered",
        "kl_divergence",
    ):
        assert getattr(result, name) is None
        assert "tokens=" in result.unavailable_metrics[name]


def _direct_logits(model: _TinyLM, tokens: torch.Tensor, site_value: torch.Tensor) -> torch.Tensor:
    """Run the downstream of the site directly in float64."""

    return nn.functional.linear(
        site_value.double(), model.unembed.weight.double(), model.unembed.bias.double()
    )


def test_ce_recovered_and_kl_equal_hand_calculations(lm_setup) -> None:
    """CE triple, loss recovered and KL: from the result's logits and from a direct run."""

    model, ids, sae = lm_setup
    result = sae_bridge.splice(model, ids, site=tl.func("relu"), sae=sae, tokens=ids)

    clean_logits, spliced_logits = result.clean_outputs[0], result.spliced_outputs[0]
    ce_clean = _hand_ce(clean_logits, ids, 0)
    ce_spliced = _hand_ce(spliced_logits, ids, 0)
    zero_logits = model.unembed.bias.detach().expand_as(clean_logits)
    ce_zero = _hand_ce(zero_logits, ids, 0)
    assert result.ce_loss_clean == pytest.approx(ce_clean, rel=1e-12)
    assert result.ce_loss_spliced == pytest.approx(ce_spliced, rel=1e-12)
    assert result.ce_loss_zero_ablated == pytest.approx(ce_zero, rel=1e-12)
    recovered = (ce_zero - ce_spliced) / (ce_zero - ce_clean)
    assert result.ce_loss_recovered == pytest.approx(recovered, rel=1e-10)
    assert result.kl_divergence == pytest.approx(
        _hand_kl(clean_logits, spliced_logits, 0), rel=1e-12
    )

    # Direct float64 forward from the site, independent of the replay.
    with torch.no_grad():
        site = torch.relu(model.hidden(model.embed(ids)))
        recon = sae.decode(sae.encode(site))
    direct_clean = _direct_logits(model, ids, site)
    direct_spliced = _direct_logits(model, ids, recon)
    assert result.ce_loss_clean == pytest.approx(_hand_ce(direct_clean, ids, 0), rel=1e-5)
    assert result.ce_loss_spliced == pytest.approx(_hand_ce(direct_spliced, ids, 0), rel=1e-5)
    assert result.kl_divergence == pytest.approx(
        _hand_kl(direct_clean, direct_spliced, 0), rel=1e-4
    )
    assert set(result.unavailable_metrics).isdisjoint({"ce_loss_recovered", "kl_divergence"})


def test_exclude_bos_keeps_position_zero_clean_and_out_of_every_metric(lm_setup) -> None:
    """The Gemma Scope convention: BOS untouched, metrics from position 1 on."""

    model, ids, sae = lm_setup
    result = sae_bridge.splice(
        model, ids, site=tl.func("relu"), sae=sae, tokens=ids, exclude_bos=True
    )
    clean = result.clean_site_out
    assert torch.equal(result.reconstruction[:, 0], clean[:, 0])
    assert torch.allclose(result.spliced[result.site].out[:, 0], clean[:, 0])
    with torch.no_grad():
        assert torch.equal(result.latents, sae.encode(clean))
    _global, pooled, _by_dim = _hand_variance(clean[:, 1:], result.reconstruction[:, 1:])
    assert result.fraction_variance_explained_pooled == pytest.approx(pooled, rel=1e-12)

    clean_logits, spliced_logits = result.clean_outputs[0], result.spliced_outputs[0]
    zero_logits = clean_logits.clone()
    zero_logits[:, 1:] = model.unembed.bias.detach()
    ce_clean = _hand_ce(clean_logits, ids, 1)
    ce_spliced = _hand_ce(spliced_logits, ids, 1)
    ce_zero = _hand_ce(zero_logits, ids, 1)
    assert result.ce_loss_clean == pytest.approx(ce_clean, rel=1e-12)
    assert result.ce_loss_zero_ablated == pytest.approx(ce_zero, rel=1e-6)
    assert result.ce_loss_recovered == pytest.approx(
        (ce_zero - ce_spliced) / (ce_zero - ce_clean), rel=1e-6
    )
    assert result.kl_divergence == pytest.approx(
        _hand_kl(clean_logits, spliced_logits, 1), rel=1e-12
    )


def test_short_sequence_gives_kl_and_a_reason_for_the_losses(lm_setup) -> None:
    """One position predicts nothing: CE fields None with a reason, KL still computed."""

    model, _ids, sae = lm_setup
    ids = torch.tensor([[4], [7]])
    result = sae_bridge.splice(model, ids, site=tl.func("relu"), sae=sae, tokens=ids)
    assert result.ce_loss_clean is None and result.ce_loss_recovered is None
    assert "positions" in result.unavailable_metrics["ce_loss_clean"]
    assert result.kl_divergence == pytest.approx(
        _hand_kl(result.clean_outputs[0], result.spliced_outputs[0], 0), rel=1e-12
    )


def test_tokens_matching_no_output_refuse_with_a_code(lm_setup) -> None:
    """Wrong shape or a float tensor: sae_splice_tokens_mismatch, not a guess."""

    model, ids, sae = lm_setup
    for bad in (ids[:, :3], ids.float()):
        with pytest.raises(InvalidArgumentError) as excinfo:
            sae_bridge.splice(model, ids, site=tl.func("relu"), sae=sae, tokens=bad)
        assert excinfo.value.fields["code"] == "sae_splice_tokens_mismatch"


def test_exclude_bos_without_a_sequence_axis_refuses(mlp_setup) -> None:
    """A (batch, features) site has no BOS position to keep."""

    model, x, sae = mlp_setup
    with pytest.raises(InvalidArgumentError) as excinfo:
        sae_bridge.splice(model, x, site="relu_1_2", sae=sae, exclude_bos=True)
    assert excinfo.value.fields["code"] == "sae_splice_bos_needs_sequence_axis"
    assert "(5, 16)" in str(excinfo.value)
