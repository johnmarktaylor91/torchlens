"""Multimodal Diffusion Transformer (MMDiT), compact random-init form.

Stable Diffusion 3 (Esser et al., 2024) introduced MMDiT blocks with separate
modality-specific weights for image and text streams, followed by one joint
attention operation over the concatenated tokens. This compact reconstruction
keeps the load-bearing primitive: independent image/text normalization and
QKV/MLP projections, bidirectional joint attention, and diffusion timestep
conditioning.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TimestepMLP(nn.Module):
    """Fourier timestep embedding followed by an MLP."""

    def __init__(self, dim: int) -> None:
        """Initialize the embedding network.

        Parameters
        ----------
        dim:
            Hidden width.
        """
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(dim // 2), requires_grad=False)
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: Tensor) -> Tensor:
        """Embed scalar diffusion times.

        Parameters
        ----------
        t:
            Tensor of shape ``(batch,)``.

        Returns
        -------
        Tensor
            Conditioning tensor of shape ``(batch, dim)``.
        """
        angles = t[:, None] * self.freqs[None]
        emb = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.net(emb)


class ModalityBlock(nn.Module):
    """Per-modality projections used before and after joint attention."""

    def __init__(self, dim: int, heads: int) -> None:
        """Initialize modality-local layers.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of attention heads.
        """
        super().__init__()
        self.heads = heads
        self.dim_head = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.cond = nn.Linear(dim, dim * 4)

    def qkv_from(self, x: Tensor, cond: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Project normalized tokens into Q, K, and V tensors.

        Parameters
        ----------
        x:
            Modality token sequence.
        cond:
            Diffusion conditioning vector.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Query, key, and value tensors in ``(batch, heads, tokens, dim_head)``.
        """
        scale, shift, _, _ = self.cond(cond).chunk(4, dim=-1)
        x_norm = self.norm1(x) * (1.0 + scale[:, None]) + shift[:, None]
        batch, tokens, _ = x_norm.shape
        qkv = self.qkv(x_norm).view(batch, tokens, 3, self.heads, self.dim_head)
        q, k, v = qkv.unbind(dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def mlp_update(self, x: Tensor, attended: Tensor, cond: Tensor) -> Tensor:
        """Apply residual attention and modality-local MLP update.

        Parameters
        ----------
        x:
            Input tokens.
        attended:
            Attention output tokens for this modality.
        cond:
            Diffusion conditioning vector.

        Returns
        -------
        Tensor
            Updated tokens.
        """
        _, _, gate_attn, gate_mlp = self.cond(cond).chunk(4, dim=-1)
        x = x + torch.tanh(gate_attn)[:, None] * attended
        x = x + torch.tanh(gate_mlp)[:, None] * self.mlp(self.norm2(x))
        return x


class MMDiTBlock(nn.Module):
    """One MMDiT block with separate modality weights and joint attention."""

    def __init__(self, dim: int = 64, heads: int = 4) -> None:
        """Initialize the block.

        Parameters
        ----------
        dim:
            Token width.
        heads:
            Number of attention heads.
        """
        super().__init__()
        self.image = ModalityBlock(dim, heads)
        self.text = ModalityBlock(dim, heads)
        self.image_out = nn.Linear(dim, dim)
        self.text_out = nn.Linear(dim, dim)

    def forward(self, image: Tensor, text: Tensor, cond: Tensor) -> tuple[Tensor, Tensor]:
        """Run bidirectional joint attention over image and text tokens.

        Parameters
        ----------
        image:
            Image patch tokens.
        text:
            Text/context tokens.
        cond:
            Diffusion conditioning vector.

        Returns
        -------
        tuple[Tensor, Tensor]
            Updated image and text streams.
        """
        qi, ki, vi = self.image.qkv_from(image, cond)
        qt, kt, vt = self.text.qkv_from(text, cond)
        q = torch.cat((qi, qt), dim=2)
        k = torch.cat((ki, kt), dim=2)
        v = torch.cat((vi, vt), dim=2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).flatten(2)
        image_attn, text_attn = attn.split((image.shape[1], text.shape[1]), dim=1)
        image = self.image.mlp_update(image, self.image_out(image_attn), cond)
        text = self.text.mlp_update(text, self.text_out(text_attn), cond)
        return image, text


class CompactMMDiT(nn.Module):
    """Compact MMDiT denoiser for image tokens conditioned on text tokens."""

    def __init__(self, dim: int = 64, depth: int = 2, heads: int = 4) -> None:
        """Initialize token projections, MMDiT blocks, and output head.

        Parameters
        ----------
        dim:
            Token width.
        depth:
            Number of MMDiT blocks.
        heads:
            Number of attention heads.
        """
        super().__init__()
        self.time = TimestepMLP(dim)
        self.text_in = nn.Linear(dim, dim)
        self.image_in = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList([MMDiTBlock(dim, heads) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, text_tokens: Tensor) -> Tensor:
        """Denoise a learned image-token scaffold conditioned on text tokens.

        Parameters
        ----------
        text_tokens:
            Text tokens of shape ``(batch, text_tokens, dim)``.

        Returns
        -------
        Tensor
            Image-token prediction.
        """
        batch = text_tokens.shape[0]
        image_tokens = text_tokens.new_zeros(batch, 8, text_tokens.shape[-1])
        time = text_tokens.new_full((batch,), 0.5)
        cond = self.time(time)
        image = self.image_in(image_tokens)
        text = self.text_in(text_tokens)
        for block in self.blocks:
            image, text = block(image, text, cond)
        return self.out(image)


def build() -> nn.Module:
    """Build a compact MMDiT model.

    Returns
    -------
    nn.Module
        Random-init MMDiT reconstruction.
    """
    return CompactMMDiT()


def example_input() -> Tensor:
    """Return compact text token conditioning input.

    Returns
    -------
    Tensor
        Text token tensor.
    """
    return torch.randn(1, 16, 64)


MENAGERIE_ENTRIES = [
    ("multimodal_dit", "build", "example_input", "2024", "diffusion/multimodal"),
]
