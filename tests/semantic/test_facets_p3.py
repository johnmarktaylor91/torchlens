"""Phase 3 semantic facet tests."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from torchlens.intervention.errors import SiteResolutionError
from torchlens.semantic import MissingFacet
from torchlens.semantic import facets as facets_mod
from torchlens.semantic.recipes import _load_entrypoint_recipes


class LlamaSdpaAttention(nn.Module):
    """Tiny RoPE-like SDPA attention module."""

    def __init__(self, *, n_heads: int = 2, n_kv_heads: int = 2, causal: bool = False) -> None:
        """Initialize projections.

        Parameters
        ----------
        n_heads:
            Query head count.
        n_kv_heads:
            Key/value head count.
        causal:
            Whether SDPA should use causal masking.
        """

        super().__init__()
        self.num_heads = n_heads
        self.num_key_value_heads = n_kv_heads
        self.head_dim = 2
        self.causal = causal
        self.q_proj = nn.Linear(8, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(8, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(8, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run post-projection rotation before fused SDPA."""

        batch, seq, _hidden = x.shape
        q = self.q_proj(x).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        q = q + 0.125
        k = k - 0.25
        z = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=self.causal,
            dropout_p=0.0,
            enable_gqa=self.num_heads != self.num_key_value_heads,
        )
        return self.o_proj(z.transpose(1, 2).reshape(batch, seq, -1))


class _AttentionWrapper(nn.Module):
    """Wrapper exposing a named attention child."""

    def __init__(self, attn: nn.Module) -> None:
        """Initialize wrapper."""

        super().__init__()
        self.attn = attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention child."""

        return self.attn(x)


class TransformerBlock(nn.Module):
    """Tiny transformer block with real residual adds."""

    def __init__(self) -> None:
        """Initialize block."""

        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run attention and MLP residual updates."""

        resid_mid = x + self.attn(x)
        return resid_mid + self.mlp(resid_mid)


def _sdpa_op(log: Any) -> Any:
    """Return the single SDPA op from a trace."""

    return next(op for op in log.layer_list if "scaled_dot_product_attention" in str(op.func_name))


def _manual_pattern(op: Any) -> torch.Tensor:
    """Return a manual attention pattern from an SDPA op's actual saved inputs."""

    q, k, _v = op.saved_args[:3]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if op.saved_kwargs.get("is_causal", False):
        seq = scores.shape[-1]
        mask = torch.ones(seq, seq, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores.float(), dim=-1).to(q.dtype)


def test_reconstructed_sdpa_facets_validate_against_actual_post_rope_inputs() -> None:
    """Pattern and z reconstruct from actual SDPA args, not q_proj/k_proj outputs."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    x = torch.randn(2, 3, 8)
    log = tl.trace(model, x, layers_to_save="all", reconstruction_ready=True)
    view = log.modules["attn"].facets
    op = _sdpa_op(log)

    assert torch.allclose(view.pattern, _manual_pattern(op), atol=1e-5, rtol=1e-4)
    assert torch.allclose(view.z, op.out, atol=1e-5, rtol=1e-4)
    assert torch.allclose(
        view.result.sum(dim=-2), log.modules["attn.o_proj"].out, atol=1e-5, rtol=1e-4
    )
    q_pre = view.q.transpose(1, 2)
    k_pre = view.k.transpose(1, 2)
    wrong_pattern = torch.softmax(q_pre @ k_pre.transpose(-2, -1), dim=-1)
    assert not torch.allclose(view.pattern, wrong_pattern)


def test_reconstructed_sdpa_validation_is_non_vacuous_after_corruption() -> None:
    """Corrupting saved Q makes validation fail with a MissingFacet value."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    log = tl.trace(model, torch.randn(2, 3, 8), layers_to_save="all", reconstruction_ready=True)
    op = _sdpa_op(log)
    op.saved_args[0] = op.saved_args[0] + 10.0
    log.modules["attn"].facets.invalidate()

    value = log.modules["attn"].facets.pattern.value
    assert isinstance(value, MissingFacet)
    assert "validation failed" in value.reason


def test_reconstructed_sdpa_gqa_and_causal_mask_validate() -> None:
    """GQA expansion and causal masking reconstruct against fused SDPA output."""

    model = _AttentionWrapper(LlamaSdpaAttention(n_heads=4, n_kv_heads=2, causal=True))
    log = tl.trace(model, torch.randn(1, 4, 8), layers_to_save="all", reconstruction_ready=True)
    view = log.modules["attn"].facets
    op = _sdpa_op(log)

    assert view.pattern.shape == (1, 4, 4, 4)
    assert torch.allclose(view.z, op.out, atol=1e-5, rtol=1e-4)


def test_reconstructed_sdpa_missing_prerequisite_names_arg_capture() -> None:
    """Default capture names the missing arg-capture prerequisite."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    log = tl.trace(model, torch.randn(2, 3, 8), layers_to_save="all")
    value = log.modules["attn"].facets.pattern

    assert isinstance(value, MissingFacet)
    assert "save_arg_values=True" in value.reason


def test_fused_pattern_intervention_requires_real_eager_facet() -> None:
    """Read-only reconstructed pattern facets cannot be scatter-back edited."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    log = tl.trace(model, torch.randn(2, 3, 8), layers_to_save="all", reconstruction_ready=True)

    with pytest.raises(SiteResolutionError, match="read-only"):
        log.attach_hooks(tl.facet("pattern"), tl.zero_ablate())


def test_residual_facets_read_and_grad() -> None:
    """Transformer block residual facets are op-anchored and grad-capable."""

    model = TransformerBlock()
    x = torch.randn(2, 4, requires_grad=True)
    log = tl.trace(model, x, layers_to_save="all", backward_ready=True, gradients_to_save="all")
    log.log_backward(log[log.output_layers[0]].out.sum())
    facets = log.modules["self"].facets

    assert facets.resid_pre.shape == x.shape
    assert facets.resid_mid.shape == x.shape
    assert facets.resid_post.shape == x.shape
    assert torch.is_tensor(facets.resid_pre.grad)
    assert torch.is_tensor(facets.resid_mid.grad)
    assert torch.is_tensor(facets.resid_post.grad)


def test_module_path_fallback_on_unreciped_model() -> None:
    """A module with no semantic recipe still exposes structural output facets."""

    class CustomLeaf(nn.Module):
        """Unrecipe'd module."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Return a simple output."""

            return x.sin()

    class Model(nn.Module):
        """Wrapper with an addressable custom leaf."""

        def __init__(self) -> None:
            """Initialize child."""

            super().__init__()
            self.custom = CustomLeaf()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run child."""

            return self.custom(x)

    x = torch.randn(2, 3)
    log = tl.trace(Model(), x, layers_to_save="all")
    assert torch.equal(log.modules["custom"].facets["out"], x.sin())


def test_transformerlens_aliases_are_opt_in() -> None:
    """TransformerLens hook-name aliases resolve only after the enable call."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    x = torch.randn(2, 3, 8)
    log = tl.trace(model, x, layers_to_save="all", reconstruction_ready=True)
    assert "hook_pattern" not in log.modules["attn"].facets.keys()

    facets_mod.enable_transformerlens_aliases(True)
    try:
        aliased = tl.trace(model, x, layers_to_save="all", reconstruction_ready=True)
        view = aliased.modules["attn"].facets
        assert "hook_pattern" in view.keys()
        assert torch.allclose(view["hook_pattern"], view.pattern)
    finally:
        facets_mod.enable_transformerlens_aliases(False)


def test_entrypoint_loader_warns_and_skips_broken_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Broken recipe entry points warn instead of crashing import-time loading."""

    broken = EntryPoint(
        name="broken",
        value="not_a_real_torchlens_plugin:register",
        group="torchlens.recipes",
    )

    class _EntryPoints(list[EntryPoint]):
        """Minimal selectable entry-point collection."""

        def select(self, *, group: str) -> list[EntryPoint]:
            """Return points for one group."""

            return [point for point in self if point.group == group]

    monkeypatch.setattr(
        "torchlens.semantic.recipes.metadata.entry_points", lambda: _EntryPoints([broken])
    )
    with pytest.warns(UserWarning, match="Skipping broken torchlens.recipes entry point"):
        _load_entrypoint_recipes()
