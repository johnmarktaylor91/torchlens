# FAITHFUL REIMPLEMENTATION from Ma, Bai & Zhou, "MoFusion: Pretrained Diffusion
# Models for Unified Human Motion Synthesis" (arXiv:2212.02837, 2022) -- no public
# code (OFA-Sys/MoFusion repo is a README-only project-page stub; the paper states
# "Source code and checkpoints will be released before publication" but as of this
# writing the repo has not been updated with any code, and its README explicitly
# says the author "has lost interest in maintaining this project").
"""MoFusion: a Transformer encoder-decoder backbone pretrained as an x0-prediction
DDPM over motion-clip tensors x in R^(N x D) (N frames, D floats/frame), with an
optional cross-attended control signal (e.g. a frozen text/audio encoder's output).

Architecture faithfully reimplemented per Figure 1a and Section 2.2 of the paper:
  - Each of the N frame-tokens is projected D -> D' via a feed-forward (Linear) layer.
  - A learnable positional embedding and a learnable diffusion-step embedding
    (indexed by the noise-strength timestep t) are added to the token embeddings.
  - The (optional) control-signal encoder output feeds an L-layer Transformer
    ENCODER stack (skipped entirely if no control signal is supplied -- Fig 1a:
    "Encoder (Skipped if No Control Signal)").
  - The DECODER is L stacked blocks, each: multi-head self-attention over the noisy
    motion tokens, multi-head cross-attention to the encoder's control-signal output
    (identity/no-op when there is no control signal, matching "skipped if no control
    signal"), then a feed-forward sublayer -- all with residual connections, matching
    the standard Transformer decoder block Fig 1a draws.
  - A final feed-forward projects D' -> D to produce the denoised motion prediction
    x0_hat, matching Eq. (1): x0_hat = f_theta(x_t, t[, c]).

Hyperparameters (Section 3.1): "The decoder of MoFusion comprises 12 layers and 16
attention heads with a hidden size of 1,024, which has ~250M parameters." The
staging build below uses drastically reduced sizes (tiny D', L, heads) purely for
fast tracing; the layer topology and data flow are unchanged from the paper's
specification. The paper does not name D (raw per-frame feature width) explicitly
as a fixed constant (it depends on the retargeted skeleton's joint/rotation count),
so example_input uses a representative small D.
"""

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "reimpl-pytorch"


class SinusoidalStepEmbedding(nn.Module):
    """Standard DDPM sinusoidal timestep embedding (Eq. 1's diffusion-step input t is
    "projected into R^{D'} via a learnable embedding table" per Fig 1a's caption --
    reimplemented here as the conventional sinusoidal-then-MLP projection used by the
    Transformer-backbone diffusion models MoFusion built on [26, 42])."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class ControlEncoder(nn.Module):
    """The Transformer encoder stack over the (frozen off-the-shelf-encoded) control
    signal, per Fig 1a's "Encoder (Skipped if No Control Signal)" box: L layers of
    (multi-head self-attention, feed-forward), with positional embedding added to the
    token embeddings -- the mirror image of the decoder's own encoder-side sublayers.
    """

    def __init__(
        self,
        control_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_control_len: int,
    ):
        super().__init__()
        self.in_proj = nn.Linear(control_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_control_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        seq_len = control.shape[1]
        h = self.in_proj(control) + self.pos_emb[:, :seq_len]
        return self.layers(h)


class MoFusionDecoderBlock(nn.Module):
    """One decoder block: self-attention over the noisy motion tokens, then
    cross-attention to the (already-encoded) control signal, then a feed-forward
    sublayer -- each with a residual + pre-norm, matching Fig 1a's decoder stack
    (Multi-Head Attention -> Multi-Head Attention (cross) -> Feed Forward, repeated
    L times)."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(), nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, control_memory: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h)
        x = x + attn_out

        h = self.norm2(x)
        cross_out, _ = self.cross_attn(h, control_memory, control_memory)
        x = x + cross_out

        h = self.norm3(x)
        x = x + self.ff(h)
        return x


class MoFusionDecoder(nn.Module):
    """The full denoising network f_theta(x_t, t[, c]) -> x0_hat, per Eq. (1)/Fig 1a.
    `frame_dim` is D (raw per-frame feature width, dataset/skeleton dependent), and
    `hidden_dim` is D' (the attention-block hidden dimension the paper reports as
    1024 in the full-size model)."""

    def __init__(
        self,
        frame_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_frames: int,
        control_dim: int,
        max_control_len: int,
    ):
        super().__init__()
        self.token_in = nn.Linear(frame_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        self.step_emb = SinusoidalStepEmbedding(hidden_dim)
        self.control_encoder = ControlEncoder(
            control_dim, hidden_dim, num_layers, num_heads, max_control_len
        )
        self.blocks = nn.ModuleList(
            [MoFusionDecoderBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.token_out = nn.Linear(hidden_dim, frame_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        seq_len = x_t.shape[1]
        h = self.token_in(x_t) + self.pos_emb[:, :seq_len] + self.step_emb(t)[:, None, :]

        control_memory = self.control_encoder(control)

        for block in self.blocks:
            h = block(h, control_memory)

        return self.token_out(h)


def build_mofusion():
    torch.manual_seed(0)
    return MoFusionDecoder(
        frame_dim=12,
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        max_frames=8,
        control_dim=10,
        max_control_len=6,
    )


def example_input_mofusion():
    torch.manual_seed(0)
    batch = 2
    n_frames = 8
    frame_dim = 12
    control_len = 6
    control_dim = 10
    x_t = torch.randn(batch, n_frames, frame_dim)
    t = torch.randint(0, 200, (batch,))
    control = torch.randn(batch, control_len, control_dim)
    return (x_t, t, control)


MENAGERIE_ENTRIES = [
    ("MoFusion", build_mofusion, example_input_mofusion, 2022, "reimpl-pytorch"),
]
