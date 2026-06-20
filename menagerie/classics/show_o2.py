"""Show-o2: unified multimodal transformer (autoregressive LM + flow image head).

Xie et al. (Show Lab), 2025.
Paper: https://arxiv.org/abs/2506.15564
Source: https://github.com/showlab/Show-o

Show-o2 is a single unified transformer that handles both understanding and
generation by sharing ONE backbone between two heads.  Its DISTINCTIVE
mechanism:

  - A packed token sequence interleaves [text tokens | image latent tokens].
    Text tokens are embedded from a (small) vocabulary; image latent tokens are
    continuous latent vectors.
  - A SHARED transformer stack processes the whole packed sequence.
  - Two heads read the shared hidden states:
      * an autoregressive LANGUAGE head (Linear -> vocab logits) applied on the
        text positions, predicting next-token distributions; and
      * a FLOW / diffusion IMAGE head (MLP) applied on the image positions,
        predicting a velocity / denoising target over the image latent (flow
        matching).
  - One backbone, two objectives (AR-LM next-token + flow-matching), enabling
    unified understanding and generation.

This faithful reimplementation captures the single shared backbone with the dual
AR-LM + flow heads at modest width (vocab=256, embed_dim=128, seq=24).  Random
init is the correct artifact for a structure atlas.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _SelfAttention(nn.Module):
    """Multi-head self-attention (full attention over the packed sequence)."""

    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class _Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ShowO2(nn.Module):
    """Show-o2: shared backbone with dual AR-language + flow-image heads."""

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        n_text: int = 12,
        n_image: int = 12,
        image_latent_dim: int = 16,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_text = n_text
        self.n_image = n_image
        self.image_latent_dim = image_latent_dim

        # Text embedding + image-latent projection into the shared space.
        self.text_embed = nn.Embedding(vocab_size, embed_dim)
        self.image_in = nn.Linear(image_latent_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_text + n_image, embed_dim))

        # Shared transformer backbone.
        self.blocks = nn.ModuleList([_Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Autoregressive language head (on text positions).
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        # Flow / diffusion image head (on image positions): predicts velocity.
        self.flow_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, image_latent_dim),
        )

    def forward(self, text_ids: torch.Tensor, image_latents: torch.Tensor) -> torch.Tensor:
        # text_ids: (B, n_text) long; image_latents: (B, n_image, image_latent_dim).
        _B = text_ids.shape[0]
        text_tok = self.text_embed(text_ids)  # (B, n_text, embed)
        img_tok = self.image_in(image_latents)  # (B, n_image, embed)
        x = torch.cat([text_tok, img_tok], dim=1)  # packed [text | image]
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        text_hidden = x[:, : self.n_text, :]
        image_hidden = x[:, self.n_text :, :]

        _lm_logits = self.lm_head(text_hidden)  # (B, n_text, vocab) -- traced head, leaf
        flow_pred = self.flow_head(image_hidden)  # (B, n_image, image_latent_dim)

        # Return the flow-matching image output (the generation branch).
        return flow_pred


class _ShowO2Wrapper(nn.Module):
    """Wrapper providing a single-tensor forward for tracing.

    The fixed text-token ids are synthesized inside forward; the model still
    runs both heads (lm + flow) -- we return the flow output.
    """

    def __init__(self, model: ShowO2) -> None:
        super().__init__()
        self.model = model

    def forward(self, image_latents: torch.Tensor) -> torch.Tensor:
        B = image_latents.shape[0]
        text_ids = torch.arange(self.model.n_text, device=image_latents.device).unsqueeze(0)
        text_ids = text_ids.expand(B, -1) % self.model.vocab_size
        return self.model(text_ids, image_latents)


def build_show_o2() -> nn.Module:
    """Build Show-o2 (shared backbone, AR-language + flow-image heads)."""
    model = ShowO2(
        vocab_size=256,
        embed_dim=128,
        depth=4,
        num_heads=4,
        n_text=12,
        n_image=12,
        image_latent_dim=16,
    )
    return _ShowO2Wrapper(model)


def example_input() -> torch.Tensor:
    """Example image-latent tensor ``(1, 12, 16)`` (text ids synthesized in forward)."""
    return torch.randn(1, 12, 16)


MENAGERIE_ENTRIES = [
    (
        "Show-o2 (unified transformer, AR-language + flow-image heads)",
        "build_show_o2",
        "example_input",
        "2025",
        "DC",
    ),
]
