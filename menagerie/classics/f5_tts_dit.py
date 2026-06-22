"""F5-TTS DiT text-to-speech flow-matching backbone.

Paper: F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching,
2024.  The released architecture uses a non-autoregressive conditional flow matching
decoder with a Diffusion Transformer (DiT) backbone, ConvNeXt V2 text conditioning, time
conditioning, noised mel input, and masked reference-audio conditioning.

This compact random-init version preserves those components for TorchLens rendering.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepMLP(nn.Module):
    """Sinusoidal timestep embedding followed by an MLP."""

    def __init__(self, dim: int) -> None:
        """Initialize the timestep MLP.

        Parameters
        ----------
        dim:
            Embedding dimension.
        """

        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed continuous flow-matching times.

        Parameters
        ----------
        t:
            Tensor of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Timestep embeddings of shape ``(batch, dim)``.
        """

        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(0.0, -torch.log(torch.tensor(10000.0)), half, device=t.device)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, 1))
        return self.net(emb)


class GlobalResponseNorm(nn.Module):
    """Global Response Normalization from ConvNeXt V2."""

    def __init__(self, dim: int) -> None:
        """Initialize GRN parameters.

        Parameters
        ----------
        dim:
            Feature dimension.
        """

        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global response normalization to sequence features.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, time, dim)``.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """

        gx = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtTextBlock(nn.Module):
    """ConvNeXt V2-style text conditioning block."""

    def __init__(self, dim: int) -> None:
        """Initialize a compact text block.

        Parameters
        ----------
        dim:
            Token feature dimension.
        """

        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 4)
        self.grn = GlobalResponseNorm(dim * 4)
        self.pw2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode text tokens with depthwise convolution and GRN MLP.

        Parameters
        ----------
        x:
            Text embeddings of shape ``(batch, time, dim)``.

        Returns
        -------
        torch.Tensor
            Encoded text features.
        """

        residual = x
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x)
        x = self.pw2(self.grn(F.gelu(self.pw1(x))))
        return residual + x


class AdaLayerNorm(nn.Module):
    """Adaptive LayerNorm modulation for DiT blocks."""

    def __init__(self, dim: int) -> None:
        """Initialize adaptive normalization.

        Parameters
        ----------
        dim:
            Feature dimension.
        """

        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mod = nn.Linear(dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply condition-dependent scale and shift.

        Parameters
        ----------
        x:
            Sequence tensor.
        cond:
            Conditioning tensor of shape ``(batch, dim)``.

        Returns
        -------
        torch.Tensor
            Modulated normalized tensor.
        """

        scale, shift = self.mod(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """Diffusion Transformer block with adaptive timestep conditioning."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Initialize a DiT block.

        Parameters
        ----------
        dim:
            Model dimension.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.norm1 = AdaLayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = AdaLayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Run one DiT residual block.

        Parameters
        ----------
        x:
            Sequence tensor.
        cond:
            Timestep conditioning tensor.

        Returns
        -------
        torch.Tensor
            Updated sequence tensor.
        """

        h = self.norm1(x, cond)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        x = x + self.ff(self.norm2(x, cond))
        return x


class CompactF5TTS(nn.Module):
    """Compact F5-TTS DiT flow-matching decoder."""

    def __init__(self, mel_bins: int = 16, vocab: int = 64, dim: int = 48) -> None:
        """Initialize compact F5-TTS.

        Parameters
        ----------
        mel_bins:
            Number of mel channels in the compact spectrogram.
        vocab:
            Text vocabulary size.
        dim:
            Transformer width.
        """

        super().__init__()
        self.text_embed = nn.Embedding(vocab, dim)
        self.text_blocks = nn.ModuleList([ConvNeXtTextBlock(dim) for _ in range(2)])
        self.noisy_proj = nn.Linear(mel_bins, dim)
        self.cond_proj = nn.Linear(mel_bins + 1, dim)
        self.time_mlp = TimestepMLP(dim)
        self.blocks = nn.ModuleList([DiTBlock(dim) for _ in range(2)])
        self.final_norm = nn.LayerNorm(dim)
        self.to_velocity = nn.Linear(dim, mel_bins)

    def forward(
        self,
        noisy_mel: torch.Tensor,
        text_tokens: torch.Tensor,
        cond_mel: torch.Tensor,
        cond_mask: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Predict flow velocity for a noised mel sequence.

        Parameters
        ----------
        noisy_mel:
            Noised mel tensor of shape ``(batch, frames, mel_bins)``.
        text_tokens:
            Token IDs aligned to mel frames.
        cond_mel:
            Reference/condition mel tensor.
        cond_mask:
            Reference mask of shape ``(batch, frames, 1)``.
        t:
            Flow-matching time tensor of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Predicted velocity with the same mel shape as ``noisy_mel``.
        """

        text = self.text_embed(text_tokens)
        for block in self.text_blocks:
            text = block(text)
        cond = self.cond_proj(torch.cat([cond_mel * cond_mask, cond_mask], dim=-1))
        x = self.noisy_proj(noisy_mel) + text + cond
        time = self.time_mlp(t)
        for block in self.blocks:
            x = block(x, time)
        return self.to_velocity(self.final_norm(x))


def build_f5_tts_dit() -> nn.Module:
    """Build compact F5-TTS DiT.

    Returns
    -------
    nn.Module
        Random-init F5-TTS DiT model.
    """

    return CompactF5TTS()


def example_input() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create compact F5-TTS flow-matching inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Noisy mel, text tokens, reference mel, mask, and time.
    """

    frames = 12
    noisy = torch.randn(1, frames, 16)
    text = torch.randint(0, 64, (1, frames))
    cond = torch.randn(1, frames, 16)
    mask = torch.zeros(1, frames, 1)
    mask[:, :4] = 1.0
    t = torch.tensor([0.35])
    return noisy, text, cond, mask, t


build = build_f5_tts_dit

MENAGERIE_ENTRIES = [
    ("f5_tts_dit", "build_f5_tts_dit", "example_input", "2024", "E6"),
]
