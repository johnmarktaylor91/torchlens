"""Evo 2 / StripedHyena-2 compact genomics language model.

Evo 2 (2025) scales StripedHyena-style sequence modeling for genomics.  This
classic is a random-init, CPU-sized atlas model: it keeps the faithful core
operators, not the proprietary speed kernels or trained weights.  The block
stack alternates FFT-based long causal Hyena convolutions with causal multi-head
attention, RMSNorm residuals, gated MLPs, and DNA byte-token inputs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square normalization used by StripedHyena-style blocks."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialize learnable scale and numerical epsilon.

        Parameters
        ----------
        dim:
            Hidden feature width.
        eps:
            Stabilizer added to the feature variance.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize an activation tensor over its final dimension.

        Parameters
        ----------
        x:
            Tensor with hidden features in the final dimension.

        Returns
        -------
        torch.Tensor
            RMS-normalized tensor with the same shape as ``x``.
        """

        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class SinusoidalFilter(nn.Module):
    """Implicit position MLP that emits channel-wise Hyena filters."""

    def __init__(self, dim: int, filter_order: int = 32) -> None:
        """Create a small implicit long-filter generator.

        Parameters
        ----------
        dim:
            Number of channels in the emitted convolution filter.
        filter_order:
            Hidden width of the positional MLP.
        """

        super().__init__()
        self.in_proj = nn.Linear(5, filter_order)
        self.out_proj = nn.Linear(filter_order, dim)
        self.decay = nn.Parameter(torch.linspace(0.1, 1.0, dim))

    def forward(self, length: int, device: torch.device) -> torch.Tensor:
        """Generate a causal convolution kernel for a sequence length.

        Parameters
        ----------
        length:
            Sequence length to cover.
        device:
            Device on which the kernel should be allocated.

        Returns
        -------
        torch.Tensor
            Filter tensor of shape ``(length, dim)``.
        """

        pos = torch.linspace(0.0, 1.0, length, device=device).unsqueeze(-1)
        features = torch.cat(
            [
                pos,
                torch.sin(2.0 * math.pi * pos),
                torch.cos(2.0 * math.pi * pos),
                torch.sin(8.0 * math.pi * pos),
                torch.cos(8.0 * math.pi * pos),
            ],
            dim=-1,
        )
        raw = self.out_proj(torch.tanh(self.in_proj(features)))
        envelope = torch.exp(-pos * self.decay.view(1, -1))
        return raw * envelope


class FFTCausalConv(nn.Module):
    """Channel-wise long causal convolution evaluated with real FFTs."""

    def __init__(self, dim: int) -> None:
        """Initialize the implicit Hyena filter.

        Parameters
        ----------
        dim:
            Hidden width for the channel-wise convolution.
        """

        super().__init__()
        self.filter = SinusoidalFilter(dim)
        self.short_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply long causal convolution to a batch of sequences.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, length, dim)``.

        Returns
        -------
        torch.Tensor
            Causally convolved tensor with the same shape as ``x``.
        """

        _, length, _ = x.shape
        fft_length = 2 * length
        kernel = self.filter(length, x.device)
        x_fft = torch.fft.rfft(x, n=fft_length, dim=1)
        k_fft = torch.fft.rfft(kernel, n=fft_length, dim=0).unsqueeze(0)
        y = torch.fft.irfft(x_fft * k_fft, n=fft_length, dim=1)[:, :length]
        return y + self.short_bias.view(1, 1, -1) * x


class HyenaMixer(nn.Module):
    """StripedHyena-style gated long-convolution mixer."""

    def __init__(self, dim: int) -> None:
        """Create projections, local depthwise convolution, and long mixer.

        Parameters
        ----------
        dim:
            Hidden feature width.
        """

        super().__init__()
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, groups=dim, padding=2)
        self.long_conv = FFTCausalConv(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix tokens with gated local and long causal convolutions.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, length, dim)``.

        Returns
        -------
        torch.Tensor
            Mixed sequence tensor of shape ``(batch, length, dim)``.
        """

        length = x.shape[1]
        value, long_gate, out_gate = self.in_proj(x).chunk(3, dim=-1)
        value = self.local_conv(value.transpose(1, 2))[..., :length].transpose(1, 2)
        mixed = self.long_conv(F.silu(value) * torch.sigmoid(long_gate))
        return self.out_proj(mixed * torch.sigmoid(out_gate))


class CausalAttention(nn.Module):
    """Plain Torch causal multi-head self-attention."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        """Create attention projections.

        Parameters
        ----------
        dim:
            Hidden feature width.
        heads:
            Number of attention heads.
        """

        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply masked self-attention.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, length, dim)``.

        Returns
        -------
        torch.Tensor
            Attention output with the same shape as ``x``.
        """

        batch, length, dim = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.full((length, length), float("-inf"), device=x.device).triu(1)
        attn = torch.softmax(scores + mask, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch, length, dim)
        return self.out_proj(out)


class GatedMLP(nn.Module):
    """SwiGLU-style feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        """Create gated feed-forward projections.

        Parameters
        ----------
        dim:
            Input and output feature width.
        hidden_dim:
            Intermediate feature width.
        """

        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated feed-forward transformation.

        Parameters
        ----------
        x:
            Sequence tensor with hidden features in the final dimension.

        Returns
        -------
        torch.Tensor
            Transformed tensor with the same leading dimensions as ``x``.
        """

        return self.down(F.silu(self.gate(x)) * self.up(x))


class StripedHyenaBlock(nn.Module):
    """Residual block using either a Hyena mixer or attention stripe."""

    def __init__(self, dim: int, mlp_dim: int, use_attention: bool) -> None:
        """Initialize a normalized mixer and MLP block.

        Parameters
        ----------
        dim:
            Hidden feature width.
        mlp_dim:
            MLP intermediate width.
        use_attention:
            Whether this stripe uses attention instead of Hyena convolution.
        """

        super().__init__()
        self.mixer_norm = RMSNorm(dim)
        self.mixer = CausalAttention(dim) if use_attention else HyenaMixer(dim)
        self.mlp_norm = RMSNorm(dim)
        self.mlp = GatedMLP(dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run residual sequence mixing followed by residual MLP.

        Parameters
        ----------
        x:
            Sequence tensor of shape ``(batch, length, dim)``.

        Returns
        -------
        torch.Tensor
            Updated sequence tensor.
        """

        x = x + self.mixer(self.mixer_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class Evo2StripedHyenaLM(nn.Module):
    """Compact DNA-token language model preserving Evo 2's hybrid block stack."""

    def __init__(
        self,
        vocab_size: int = 16,
        dim: int = 96,
        mlp_dim: int = 192,
        layers: int = 5,
    ) -> None:
        """Create a small StripedHyena-2 language model.

        Parameters
        ----------
        vocab_size:
            DNA byte-token vocabulary size.
        dim:
            Hidden feature width.
        mlp_dim:
            Feed-forward intermediate width.
        layers:
            Number of residual sequence blocks.
        """

        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList(
            [StripedHyenaBlock(dim, mlp_dim, use_attention=(idx % 3 == 2)) for idx in range(layers)]
        )
        self.final_norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map DNA token IDs to next-token logits.

        Parameters
        ----------
        token_ids:
            Integer DNA token IDs of shape ``(batch, length)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, length, vocab_size)``.
        """

        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def build() -> nn.Module:
    """Build the compact Evo 2 / StripedHyena-2 classic.

    Returns
    -------
    nn.Module
        Random-init compact StripedHyena genomics language model.
    """

    return Evo2StripedHyenaLM()


def example_input() -> torch.Tensor:
    """Return a short DNA byte-token example.

    Returns
    -------
    torch.Tensor
        Integer token IDs of shape ``(1, 32)`` over a 16-symbol DNA vocabulary.
    """

    return torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 2],
        dtype=torch.long,
    )


MENAGERIE_ENTRIES = [
    ("evo2", "build", "example_input", 2025, "DC"),
]
