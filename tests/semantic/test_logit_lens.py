"""Tests for the logit-lens appliance and the language_model_head recipe."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.semantic import LogitLensError, logit_lens
from torchlens.utils._torch_compat import HAS_RMSNORM_MODULE

pytestmark = pytest.mark.smoke

_requires_rmsnorm = pytest.mark.skipif(
    not HAS_RMSNORM_MODULE, reason="torch.nn.RMSNorm postdates the torch 2.1 floor (added 2.4)"
)


class Block(nn.Module):
    """Tiny pre-LN transformer block with real residual adds."""

    def __init__(self, d: int = 8) -> None:
        """Initialize attention/MLP children and norms."""

        super().__init__()
        self.attn = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run pre-norm attention and MLP residual updates."""

        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class Stack(nn.Module):
    """Block stack with a final norm."""

    def __init__(self, d: int = 8, n: int = 2, norm: nn.Module | None = None) -> None:
        """Initialize blocks and the final norm."""

        super().__init__()
        self.h = nn.ModuleList([Block(d) for _ in range(n)])
        self.ln_f = norm if norm is not None else nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks then the final norm."""

        for block in self.h:
            x = block(x)
        return self.ln_f(x)


class TinyLM(nn.Module):
    """Tiny GPT-shaped language model with a conventional lm_head child."""

    def __init__(self, norm: nn.Module | None = None, head_bias: bool = False) -> None:
        """Initialize the transformer stack and the unembedding head."""

        super().__init__()
        self.transformer = Stack(norm=norm)
        self.lm_head = nn.Linear(8, 11, bias=head_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""

        return self.lm_head(self.transformer(x))


class GemmaStyleRMSNorm(nn.Module):
    """RMSNorm variant that scales by ``(1 + weight)`` like Gemma."""

    def __init__(self, d: int = 8) -> None:
        """Initialize the offset weight."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(d) * 0.1)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the nonstandard scaled RMS normalization."""

        normalized = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized * (1.0 + self.weight)


def _capture(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Trace a model with the exhaustive save the lens anchors need."""

    return tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))


def test_lm_head_recipe_exposes_unembedding_facets() -> None:
    """The language_model_head recipe anchors head and final-norm facets."""

    torch.manual_seed(0)
    model = TinyLM().eval()
    log = _capture(model, torch.randn(2, 3, 8))
    view = log.modules["self"].facets

    assert view.final_norm_kind == "layer_norm"
    assert view.final_norm_eps == pytest.approx(1e-5)
    assert view.unembed_weight.value.shape == (11, 8)
    assert view.final_norm_gamma.value.shape == (8,)
    assert view.logits.value.shape == (2, 3, 11)
    assert view.menu()["unembed_bias"].status == "structurally_absent"


@_requires_rmsnorm
def test_lm_head_recipe_classifies_rms_norm_without_beta() -> None:
    """RMSNorm-family final norms classify as rms_norm with beta absent."""

    torch.manual_seed(0)
    model = TinyLM(norm=nn.RMSNorm(8)).eval()
    log = _capture(model, torch.randn(2, 3, 8))
    view = log.modules["self"].facets

    assert view.final_norm_kind == "rms_norm"
    assert view.menu()["final_norm_beta"].status == "structurally_absent"


def test_logit_lens_validates_and_matches_real_logits_layer_norm() -> None:
    """The reconstructed LayerNorm lens validates and reproduces final logits."""

    torch.manual_seed(0)
    model = TinyLM().eval()
    log = _capture(model, torch.randn(2, 3, 8))
    result = logit_lens(log)

    assert result.lens_source == "model_head"
    assert result.validated is True
    assert [entry.address for entry in result.entries] == [
        "transformer.h.0",
        "transformer.h.1",
    ]
    assert result.stacked().shape == (2, 2, 3, 11)
    assert torch.allclose(result.entries[-1].logits, result.final_logits, rtol=1e-4, atol=1e-5)


@_requires_rmsnorm
def test_logit_lens_validates_rms_norm_and_head_bias() -> None:
    """RMSNorm reconstruction and unembedding bias both validate end-to-end."""

    torch.manual_seed(0)
    x = torch.randn(2, 3, 8)
    rms_model = TinyLM(norm=nn.RMSNorm(8)).eval()
    rms_result = logit_lens(_capture(rms_model, x))
    assert rms_result.validated is True
    assert torch.allclose(
        rms_result.entries[-1].logits, rms_result.final_logits, rtol=1e-4, atol=1e-5
    )

    bias_model = TinyLM(head_bias=True).eval()
    bias_result = logit_lens(_capture(bias_model, x))
    assert torch.allclose(
        bias_result.entries[-1].logits, bias_result.final_logits, rtol=1e-4, atol=1e-5
    )


def test_logit_lens_refuses_nonstandard_norm_scaling() -> None:
    """A Gemma-style (1 + weight) RMSNorm fails validation instead of mislabeling."""

    torch.manual_seed(0)
    model = TinyLM(norm=GemmaStyleRMSNorm()).eval()
    log = _capture(model, torch.randn(2, 3, 8))

    with pytest.raises(LogitLensError, match="failed validation"):
        logit_lens(log)
    unvalidated = logit_lens(log, validate=False)
    assert unvalidated.validated is False


def test_logit_lens_user_lens_and_layer_selection() -> None:
    """User lenses skip validation; bad layer addresses refuse with a teaching message."""

    torch.manual_seed(0)
    model = TinyLM().eval()
    log = _capture(model, torch.randn(2, 3, 8))

    result = logit_lens(log, lens=lambda hidden: hidden @ torch.zeros(8, 11))
    assert result.lens_source == "user"
    assert result.validated is False
    assert torch.equal(result.entries[0].logits, torch.zeros(2, 3, 11))

    subset = logit_lens(log, layers=["transformer.h.0"])
    assert [entry.address for entry in subset.entries] == ["transformer.h.0"]
    with pytest.raises(LogitLensError, match="do not expose facet"):
        logit_lens(log, layers=["transformer.ln_f"])
    with pytest.raises(LogitLensError, match="missing entries"):
        logit_lens(log, lens={"transformer.h.0": lambda hidden: hidden})


def test_logit_lens_refuses_without_head_facets() -> None:
    """A model with no unembedding facets refuses with recipe guidance."""

    torch.manual_seed(0)

    class HeadlessModel(nn.Module):
        """Transformer stack with no conventional unembedding child."""

        def __init__(self) -> None:
            """Initialize the stack."""

            super().__init__()
            self.transformer = Stack()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return normalized hidden states."""

            return self.transformer(x)

    model = HeadlessModel().eval()
    log = _capture(model, torch.randn(2, 3, 8))
    with pytest.raises(LogitLensError, match="unembed_weight") as head_exc:
        logit_lens(log)
    assert head_exc.value.fields["code"] == "logit_lens_head_unavailable"


def test_logit_lens_top_tokens_and_summary_shapes() -> None:
    """Result accessors decode top tokens and format the per-layer table."""

    torch.manual_seed(0)
    model = TinyLM().eval()
    log = _capture(model, torch.randn(2, 3, 8))
    result = logit_lens(log)

    rows = result.top_tokens(3)
    assert len(rows) == 2
    address, pairs = rows[0]
    assert address == "transformer.h.0"
    assert len(pairs) == 3
    assert all(isinstance(token, int) for token, _probability in pairs)
    assert all(0.0 <= probability <= 1.0 for _token, probability in pairs)

    class FakeTokenizer:
        """Minimal tokenizer stub with convert_ids_to_tokens."""

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            """Return string tokens for ids."""

            return [f"tok{token_id}" for token_id in ids]

    decoded = result.top_tokens(2, tokenizer=FakeTokenizer())
    assert decoded[0][1][0][0].startswith("tok")
    summary = result.summary(tokenizer=FakeTokenizer())
    assert "logit lens (resid_post, lens=model_head)" in summary
    assert "transformer.h.1" in summary
