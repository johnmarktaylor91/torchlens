"""Phase 3 semantic facet tests."""

from __future__ import annotations

from importlib.metadata import EntryPoint
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import torchlens as tl
from torchlens._semantic import facets as facets_mod
from torchlens._semantic.recipes import activate_entrypoint_recipes
from torchlens._utils._torch_compat import HAS_SDPA_GQA
from torchlens.errors import MissingFacetError, TraceCaptureError
from torchlens.intervention.errors import SiteResolutionError

pytestmark = pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword

#: Stage a sticky hook on the trace without propagating it.
_STAGE = tl.options.InterventionOptions(engine="set_only")


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
        # ``enable_gqa`` only when the head counts differ: the torch floor's SDPA lacks the kwarg.
        gqa = {"enable_gqa": True} if self.num_heads != self.num_key_value_heads else {}
        z = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=0.0, **gqa)
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

    return next(op for op in log._layer_list if "scaled_dot_product_attention" in str(op.func_name))


def _manual_pattern(op: Any) -> torch.Tensor:
    """Return a manual attention pattern from an SDPA op's actual saved inputs."""

    q, k, _v = op.call.saved_args[:3]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    if op.call.saved_kwargs.get("is_causal", False):
        seq = scores.shape[-1]
        mask = torch.ones(seq, seq, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores.float(), dim=-1).to(q.dtype)


def test_reconstructed_sdpa_facets_validate_against_actual_post_rope_inputs() -> None:
    """Pattern and z reconstruct from actual SDPA args, not q_proj/k_proj outputs."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    x = torch.randn(2, 3, 8)
    log = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(reconstruction_ready=True),
        save="all",
    )
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
    """Corrupting saved Q makes validation refuse the facet at access."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    log = tl.trace(
        model,
        torch.randn(2, 3, 8),
        capture=tl.options.CaptureOptions(reconstruction_ready=True),
        save="all",
    )
    op = _sdpa_op(log)
    op.call.saved_args[0] = op.call.saved_args[0] + 10.0
    log.modules["attn"].facets.invalidate()

    with pytest.raises(MissingFacetError, match="validation failed") as refusal:
        log.modules["attn"].facets.pattern
    assert refusal.value.reason.status == "needs_capture"


@pytest.mark.skipif(not HAS_SDPA_GQA, reason="SDPA enable_gqa postdates the torch 2.1 floor (2.5+)")
def test_reconstructed_sdpa_gqa_and_causal_mask_validate() -> None:
    """GQA expansion and causal masking reconstruct against fused SDPA output."""

    model = _AttentionWrapper(LlamaSdpaAttention(n_heads=4, n_kv_heads=2, causal=True))
    log = tl.trace(
        model,
        torch.randn(1, 4, 8),
        capture=tl.options.CaptureOptions(reconstruction_ready=True),
        save="all",
    )
    view = log.modules["attn"].facets
    op = _sdpa_op(log)

    assert view.pattern.shape == (1, 4, 4, 4)
    assert torch.allclose(view.z, op.out, atol=1e-5, rtol=1e-4)


class MistralAttention(nn.Module):
    """SDPA attention reading a boolean key mask passed in by its caller.

    The class name selects the built-in Llama-family attention recipe.
    """

    def __init__(self, dropout_p: float = 0.0) -> None:
        """Initialize projections and the train-mode dropout probability."""

        super().__init__()
        self.num_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = 4
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.dropout_p = dropout_p

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Run masked fused attention over two heads."""

        batch, seq, _hidden = x.shape
        q = self.q_proj(x).view(batch, seq, 2, 4).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, 2, 4).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, 2, 4).transpose(1, 2)
        dropout_p = self.dropout_p if self.training else 0.0
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        return z.transpose(1, 2).reshape(batch, seq, 8)


class _MaskBuildingWrapper(nn.Module):
    """Build the padding mask OUTSIDE the attention module, then attend."""

    def __init__(self, dropout_p: float = 0.0) -> None:
        """Initialize the attention child."""

        super().__init__()
        self.attn = MistralAttention(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mark rows whose features sum below -1 as padding keys."""

        keep = x.sum(dim=-1) > -1.0
        return self.attn(x, keep[:, None, None, :])


def _padded_input() -> torch.Tensor:
    """Return a batch whose second row ends in two padding positions."""

    x = torch.rand(2, 5, 8, generator=torch.Generator().manual_seed(0))
    x[1, 3:] = -5.0
    return x


def _manual_masked_pattern(model: _MaskBuildingWrapper, x: torch.Tensor) -> torch.Tensor:
    """Return the masked softmax map computed directly from the live modules."""

    attn = model.attn
    with torch.no_grad():
        q = attn.q_proj(x).view(2, 5, 2, 4).transpose(1, 2)
        k = attn.k_proj(x).view(2, 5, 2, 4).transpose(1, 2)
        keep = (x.sum(dim=-1) > -1.0)[:, None, None, :]
        scores = (q @ k.transpose(-2, -1)) / 2.0
        return torch.softmax(scores.masked_fill(~keep, float("-inf")), dim=-1)


def test_default_capture_reconstructs_sdpa_facets_without_capture_options() -> None:
    """A default capture serves the checked pattern from its recorded SDPA inputs."""

    model = _AttentionWrapper(LlamaSdpaAttention(n_heads=4, n_kv_heads=2, causal=True))
    x = torch.randn(1, 4, 8)
    plain = tl.trace(model, x)
    ready = tl.trace(model, x, capture=tl.options.CaptureOptions(reconstruction_ready=True))
    op = _sdpa_op(plain)
    assert op.call.saved_args is None, "premise: a default capture saves no argument copies"

    view = plain.modules["attn"].facets
    assert view.declared()["pattern"].status == "available_now"
    assert view["pattern"].spec.capability_flags.reconstructed is True
    assert view["pattern"].spec.capability_flags.write is False
    ready_pattern = ready.modules["attn"].facets.pattern.value
    assert torch.allclose(view.pattern.value, ready_pattern, atol=1e-7, rtol=0.0)
    assert torch.allclose(view.z, op.out, atol=1e-5, rtol=1e-4)


def test_default_capture_reconstructs_a_recorded_padding_mask() -> None:
    """A mask produced by captured ops rebuilds from its producer's saved output."""

    model = _MaskBuildingWrapper().eval()
    x = _padded_input()
    log = tl.trace(model, x)  # held: a module record's facets read through its live trace
    view = log.modules["attn"].facets

    pattern = view.pattern.value
    assert torch.allclose(pattern, _manual_masked_pattern(model, x), atol=1e-6)
    assert torch.all(pattern[1, :, :, 3:] == 0)


def test_unsaved_mask_producer_fails_closed() -> None:
    """A selective save that skips the mask producer refuses instead of guessing."""

    model = _MaskBuildingWrapper().eval()
    log = tl.trace(model, _padded_input(), save=tl.in_module("attn"))
    view = log.modules["attn"].facets

    assert view.declared()["pattern"].status == "needs_capture"
    with pytest.raises(MissingFacetError, match="attn_mask") as caught:
        _ = view.pattern
    assert "reconstruction_ready=True" in str(caught.value)


def test_train_mode_sdpa_dropout_fails_closed_on_default_capture() -> None:
    """Active SDPA dropout keeps the reconstruction refused on a default capture."""

    model = _MaskBuildingWrapper(dropout_p=0.5).train()
    log = tl.trace(model, _padded_input())
    view = log.modules["attn"].facets

    with pytest.raises(MissingFacetError, match="dropout"):
        _ = view.pattern


def test_fused_pattern_intervention_requires_real_eager_facet() -> None:
    """Read-only reconstructed pattern facets cannot be scatter-back edited."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    log = tl.trace(
        model,
        torch.randn(2, 3, 8),
        capture=tl.options.CaptureOptions(reconstruction_ready=True),
        save="all",
    )

    with pytest.raises(SiteResolutionError, match="read-only"):
        log.do({tl.facet("pattern"): tl.zero_ablate()}, intervention=_STAGE)


def test_transformerlens_alias_menu_mirrors_missing_native_facet() -> None:
    """TransformerLens aliases mirror native missing-facet menu status."""

    model = _AttentionWrapper(LlamaSdpaAttention())
    x = torch.randn(2, 3, 8)

    facets_mod.enable_transformer_lens_aliases(True)
    try:
        # Saving only the fused op leaves its Q/K inputs unrecorded, so the
        # native pattern (and its alias) cannot be reconstructed.
        log = tl.trace(model, x, save=tl.func("scaled_dot_product_attention"))
        view = log.modules["attn"].facets
        menu = view.declared()

        assert "pattern" not in view.keys()
        assert "hook_pattern" not in view.keys()
        assert menu["pattern"].status == "needs_capture"
        assert menu["hook_pattern"].status == "needs_capture"
        assert menu["hook_pattern"].save_hint == menu["pattern"].save_hint
        with pytest.raises(MissingFacetError, match="save_args=True"):
            view["hook_pattern"]
    finally:
        facets_mod.enable_transformer_lens_aliases(False)


def test_entrypoint_activation_warns_and_skips_broken_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broken recipe entry points warn instead of crashing explicit activation."""

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
        "torchlens._semantic.recipes.metadata.entry_points", lambda: _EntryPoints([broken])
    )
    with pytest.warns(UserWarning, match="Skipping broken torchlens.recipes entry point"):
        activated = activate_entrypoint_recipes()
    assert activated == ()


def test_entrypoint_activation_unknown_name_warns_coded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown requested provider names warn with the recipes_unknown_provider code."""

    class _EntryPoints(list[EntryPoint]):
        """Minimal selectable entry-point collection."""

        def select(self, *, group: str) -> list[EntryPoint]:
            """Return points for one group."""

            return [point for point in self if point.group == group]

    monkeypatch.setattr(
        "torchlens._semantic.recipes.metadata.entry_points", lambda: _EntryPoints([])
    )
    with pytest.warns(UserWarning, match="Unknown torchlens.recipes entry point") as caught:
        activated = activate_entrypoint_recipes(["nonexistent_provider"])
    assert activated == ()
    coded = [w.message for w in caught if getattr(w.message, "fields", {}).get("code")]
    assert coded, "the unknown-name warning must carry a machine-readable code"
    assert coded[0].fields["code"] == "recipes_unknown_provider"
    assert coded[0].fields["unknown_names"] == ["nonexistent_provider"]


def test_grouped_query_sdpa_without_enable_gqa_refuses_typed() -> None:
    """Graceful floor test for ``HAS_SDPA_GQA``: a GQA model traces only where SDPA has it.

    On a torch without the keyword the model's own call fails and capture reports it as
    a ``TraceCaptureError`` naming ``enable_gqa``; on newer torch the same model traces.
    """

    model = _AttentionWrapper(LlamaSdpaAttention(n_heads=4, n_kv_heads=2, causal=True))
    if HAS_SDPA_GQA:
        log = tl.trace(model, torch.randn(1, 4, 8))
        assert tuple(_sdpa_op(log).out.shape) == (1, 4, 4, 2)
        return
    with pytest.raises(TraceCaptureError, match="enable_gqa"):
        tl.trace(model, torch.randn(1, 4, 8))
