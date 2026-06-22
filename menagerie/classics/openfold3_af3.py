"""OpenFold3 / AlphaFold3: Pairformer trunk + diffusion structure head.

Abramson et al. (DeepMind), AlphaFold3, Nature 2024.
https://www.nature.com/articles/s41586-024-07487-w
OpenFold3 (aqlaboratory) is the open PyTorch reimplementation of the AF3 architecture.
Source: https://github.com/aqlaboratory/openfold3 ; reference: https://github.com/lucidrains/alphafold3-pytorch

AlphaFold3 replaces AF2's Evoformer + IPA with:
  - a slimmed **MSA module** (small) feeding a
  - **Pairformer** trunk: stacks of (triangle multiplicative updates ×2 +
    triangle attention ×2 + pair transition) on the pair rep ``z`` plus a
    single-sequence attention-with-pair-bias + transition on the single rep ``s``;
    the MSA is no longer the central object (AF2's row/col MSA attention is gone),
  - and a **diffusion module** that generates all-atom coordinates by denoising:
    a conditioned transformer (DiffusionTransformer with AdaLN conditioning on the
    trunk ``s``/``z`` and the diffusion timestep) predicts the denoised positions
    from noised atom coordinates.

This is a faithful compact reimplementation: Pairformer block wiring (triangle
mult/attn, single-rep attention-with-pair-bias) and a conditioned diffusion
transformer denoiser are reproduced at small widths / few residues / few blocks /
one diffusion step, so the unrolled atlas graph renders quickly. Random init,
forward-only.

Faithful-core simplifications (honest, not lies):
  - 2 Pairformer blocks (vs 48); small c_s/c_z/N_token; the atom-level
    representation is collapsed to one token = one atom for the diffusion head.
  - a single diffusion denoising pass at a fixed timestep (vs the full sampler);
    the conditioned-transformer + AdaLN denoiser architecture is faithful.
  - input token features synthesized from a small random ``token_type`` index
    tensor (standing in for AF3's input embedder), so the example input is a
    single small integer tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _heads(x: torch.Tensor, h: int) -> torch.Tensor:
    *lead, c = x.shape
    return x.view(*lead, h, c // h)


class TriangleMultiplication(nn.Module):
    def __init__(self, c_z: int, c_hidden: int = 8, outgoing: bool = True) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.norm = nn.LayerNorm(c_z)
        self.ap = nn.Linear(c_z, c_hidden)
        self.ag = nn.Linear(c_z, c_hidden)
        self.bp = nn.Linear(c_z, c_hidden)
        self.bg = nn.Linear(c_z, c_hidden)
        self.norm_out = nn.LayerNorm(c_hidden)
        self.g = nn.Linear(c_z, c_z)
        self.o = nn.Linear(c_hidden, c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        a = torch.sigmoid(self.ag(z)) * self.ap(z)
        b = torch.sigmoid(self.bg(z)) * self.bp(z)
        x = (
            torch.einsum("ikc,jkc->ijc", a, b)
            if self.outgoing
            else torch.einsum("kic,kjc->ijc", a, b)
        )
        return torch.sigmoid(self.g(z)) * self.o(self.norm_out(x))


class TriangleAttention(nn.Module):
    def __init__(self, c_z: int, c_hidden: int = 8, n_head: int = 4, starting: bool = True) -> None:
        super().__init__()
        self.starting = starting
        self.h = n_head
        self.c = c_hidden
        self.norm = nn.LayerNorm(c_z)
        self.q = nn.Linear(c_z, c_hidden * n_head, bias=False)
        self.k = nn.Linear(c_z, c_hidden * n_head, bias=False)
        self.v = nn.Linear(c_z, c_hidden * n_head, bias=False)
        self.b = nn.Linear(c_z, n_head, bias=False)
        self.gate = nn.Linear(c_z, c_hidden * n_head)
        self.o = nn.Linear(c_hidden * n_head, c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.starting:
            z = z.transpose(0, 1)
        z = self.norm(z)
        q = _heads(self.q(z), self.h).permute(0, 2, 1, 3)
        k = _heads(self.k(z), self.h).permute(0, 2, 1, 3)
        v = _heads(self.v(z), self.h).permute(0, 2, 1, 3)
        bias = self.b(z).permute(2, 0, 1)
        a = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias.unsqueeze(0), dim=-1
        )
        out = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(z.shape[0], z.shape[1], -1)
        out = self.o(out * torch.sigmoid(self.gate(z)))
        return out if self.starting else out.transpose(0, 1)


class PairTransition(nn.Module):
    def __init__(self, c: int, n: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.l1 = nn.Linear(c, c * n)
        self.l2 = nn.Linear(c * n, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(F.relu(self.l1(self.norm(x))))


class AttentionPairBias(nn.Module):
    """Single-rep self-attention biased by the pair representation (AF3 trunk)."""

    def __init__(self, c_s: int, c_z: int, c_hidden: int = 8, n_head: int = 4) -> None:
        super().__init__()
        self.h = n_head
        self.c = c_hidden
        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)
        self.q = nn.Linear(c_s, c_hidden * n_head, bias=False)
        self.k = nn.Linear(c_s, c_hidden * n_head, bias=False)
        self.v = nn.Linear(c_s, c_hidden * n_head, bias=False)
        self.b = nn.Linear(c_z, n_head, bias=False)
        self.gate = nn.Linear(c_s, c_hidden * n_head)
        self.o = nn.Linear(c_hidden * n_head, c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        s = self.norm_s(s)
        q = _heads(self.q(s), self.h).permute(1, 0, 2)  # (H, N, d)
        k = _heads(self.k(s), self.h).permute(1, 0, 2)
        v = _heads(self.v(s), self.h).permute(1, 0, 2)
        bias = self.b(self.norm_z(z)).permute(2, 0, 1)  # (H, N, N)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(a, v).permute(1, 0, 2).reshape(s.shape[0], -1)
        return self.o(out * torch.sigmoid(self.gate(s)))


class PairformerBlock(nn.Module):
    """AF3 Pairformer block: pair triangle stack + single-rep attn-with-pair-bias."""

    def __init__(self, c_s: int, c_z: int) -> None:
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(c_z, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(c_z, outgoing=False)
        self.tri_attn_start = TriangleAttention(c_z, starting=True)
        self.tri_attn_end = TriangleAttention(c_z, starting=False)
        self.pair_trans = PairTransition(c_z)
        self.attn_pair_bias = AttentionPairBias(c_s, c_z)
        self.single_trans = PairTransition(c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_attn_start(z)
        z = z + self.tri_attn_end(z)
        z = z + self.pair_trans(z)
        s = s + self.attn_pair_bias(s, z)
        s = s + self.single_trans(s)
        return s, z


class DiffusionTransformerLayer(nn.Module):
    """Conditioned transformer layer with AdaLN conditioning on the trunk single rep."""

    def __init__(self, c_a: int, c_s: int, c_z: int, n_head: int = 4) -> None:
        super().__init__()
        self.h = n_head
        self.c = c_a // n_head
        self.ada_scale = nn.Linear(c_s, c_a)
        self.ada_shift = nn.Linear(c_s, c_a)
        self.norm = nn.LayerNorm(c_a, elementwise_affine=False)
        self.q = nn.Linear(c_a, c_a, bias=False)
        self.k = nn.Linear(c_a, c_a, bias=False)
        self.v = nn.Linear(c_a, c_a, bias=False)
        self.b = nn.Linear(c_z, n_head, bias=False)
        self.o = nn.Linear(c_a, c_a)
        self.ff = nn.Sequential(nn.Linear(c_a, c_a * 2), nn.GELU(), nn.Linear(c_a * 2, c_a))

    def forward(self, a: torch.Tensor, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # AdaLN conditioning
        h = self.norm(a) * (1 + self.ada_scale(s)) + self.ada_shift(s)
        q = _heads(self.q(h), self.h).permute(1, 0, 2)
        k = _heads(self.k(h), self.h).permute(1, 0, 2)
        v = _heads(self.v(h), self.h).permute(1, 0, 2)
        bias = self.b(z).permute(2, 0, 1)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(att, v).permute(1, 0, 2).reshape(a.shape[0], -1)
        a = a + self.o(out)
        a = a + self.ff(a)
        return a


class DiffusionModule(nn.Module):
    """AF3 diffusion head: noised atom coords -> conditioned transformer -> denoised coords."""

    def __init__(self, c_s: int, c_z: int, c_a: int = 16, n_layer: int = 2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(3, c_a)
        self.time_embed = nn.Sequential(nn.Linear(1, c_s), nn.SiLU(), nn.Linear(c_s, c_s))
        self.s_proj = nn.Linear(c_s, c_s)
        self.layers = nn.ModuleList(
            [DiffusionTransformerLayer(c_a, c_s, c_z) for _ in range(n_layer)]
        )
        self.out_proj = nn.Linear(c_a, 3)

    def forward(self, x_noised: torch.Tensor, s: torch.Tensor, z: torch.Tensor, t: float):
        a = self.in_proj(x_noised)
        tt = torch.full((s.shape[0], 1), float(t), device=s.device, dtype=s.dtype)
        s_cond = self.s_proj(s) + self.time_embed(tt)
        for layer in self.layers:
            a = layer(a, s_cond, z)
        return x_noised + self.out_proj(a)  # predicted denoised coordinates


class AlphaFold3(nn.Module):
    """Compact AlphaFold3 / OpenFold3: input embed -> Pairformer trunk -> diffusion head."""

    def __init__(
        self,
        c_s: int = 16,
        c_z: int = 16,
        n_block: int = 2,
        n_token: int = 32,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(n_token, c_s)
        self.left = nn.Embedding(n_token, c_z)
        self.right = nn.Embedding(n_token, c_z)
        # slim MSA -> pair contribution (one outer-product-mean-like projection)
        self.msa_to_pair = nn.Linear(c_s, c_z)
        self.blocks = nn.ModuleList([PairformerBlock(c_s, c_z) for _ in range(n_block)])
        self.diffusion = DiffusionModule(c_s, c_z)

    def forward(self, token_types: torch.Tensor) -> torch.Tensor:
        # token_types: (N,) integer per-token features
        s = self.token_embed(token_types)  # (N, c_s)
        z = self.left(token_types)[:, None, :] + self.right(token_types)[None, :, :]
        z = z + (self.msa_to_pair(s)[:, None, :] + self.msa_to_pair(s)[None, :, :])
        for blk in self.blocks:
            s, z = blk(s, z)
        # diffusion: start from random noised coordinates (one atom per token)
        x_noised = torch.randn(s.shape[0], 3, device=s.device, dtype=s.dtype)
        x0 = self.diffusion(x_noised, s, z, t=0.5)
        return x0


def build() -> nn.Module:
    return AlphaFold3()


def example_input() -> torch.Tensor:
    """Small token-type tensor ``(12,)`` = 12 tokens (one atom each)."""
    return torch.randint(0, 32, (12,))


MENAGERIE_ENTRIES = [
    (
        "OpenFold3 / AlphaFold3 (Pairformer trunk + diffusion structure head)",
        "build",
        "example_input",
        "2024",
        "DC",
    ),
]
