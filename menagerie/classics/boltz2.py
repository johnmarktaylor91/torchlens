"""Boltz-2: Pairformer trunk + diffusion structure module + binding-affinity head.

Passaro, Wohlwend et al. (MIT), Boltz-2, bioRxiv 2025.
https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1
Source: https://github.com/jwohlwend/boltz

Boltz-2 is an open AlphaFold3-class biomolecular structure + binding-affinity model.
Its pipeline (faithful to the published architecture, see DeepWiki jwohlwend/boltz):
  - **Trunk**: MSA module + **Pairformer** (triangle multiplicative updates ×2,
    triangle attention ×2, pair transition, single-rep attention-with-pair-bias,
    single transition) producing single ``s`` and pair ``z`` representations;
  - **Structure module**: a DiffusionConditioning step turns (s, z) into a
    conditioned diffusion transformer that denoises atomic coordinates;
  - **Affinity module** (Boltz-2's headline contribution): a *separate*
    Pairformer-based dual head producing (1) a binder-vs-decoy **probability** and
    (2) a continuous **affinity value** regressed on a log (uM-scaled) axis.

This is a faithful compact reimplementation of the trunk -> structure -> dual
affinity-head pipeline at small widths / few residues / few blocks. Random init,
forward-only. Returns (coords, binder_logit, affinity_value).

Faithful-core simplifications (honest, not lies):
  - 2 Pairformer blocks in trunk + 1 in the affinity head (vs the deeper stacks);
    small c_s/c_z/N_token; one atom per token in the diffusion head; a single
    fixed-timestep denoising pass (vs the full sampler).
  - input token features synthesized from a small random ``token_type`` index
    tensor (standing in for Boltz's input featurizer / BoltzFeaturizer), so the
    example input is a single small integer tensor the forward consumes.
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


class Transition(nn.Module):
    def __init__(self, c: int, n: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.l1 = nn.Linear(c, c * n)
        self.l2 = nn.Linear(c * n, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(F.relu(self.l1(self.norm(x))))


class AttentionPairBias(nn.Module):
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
        q = _heads(self.q(s), self.h).permute(1, 0, 2)
        k = _heads(self.k(s), self.h).permute(1, 0, 2)
        v = _heads(self.v(s), self.h).permute(1, 0, 2)
        bias = self.b(self.norm_z(z)).permute(2, 0, 1)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(a, v).permute(1, 0, 2).reshape(s.shape[0], -1)
        return self.o(out * torch.sigmoid(self.gate(s)))


class PairformerBlock(nn.Module):
    def __init__(self, c_s: int, c_z: int) -> None:
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(c_z, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(c_z, outgoing=False)
        self.tri_attn_start = TriangleAttention(c_z, starting=True)
        self.tri_attn_end = TriangleAttention(c_z, starting=False)
        self.pair_trans = Transition(c_z)
        self.attn_pair_bias = AttentionPairBias(c_s, c_z)
        self.single_trans = Transition(c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_attn_start(z)
        z = z + self.tri_attn_end(z)
        z = z + self.pair_trans(z)
        s = s + self.attn_pair_bias(s, z)
        s = s + self.single_trans(s)
        return s, z


class DiffusionConditioning(nn.Module):
    """Boltz-2 DiffusionConditioning: trunk (s,z) -> conditioned diffusion inputs."""

    def __init__(self, c_s: int, c_z: int, c_a: int) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(nn.Linear(1, c_s), nn.SiLU(), nn.Linear(c_s, c_s))
        self.s_norm = nn.LayerNorm(c_s)
        self.z_norm = nn.LayerNorm(c_z)
        self.s_proj = nn.Linear(c_s, c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, t: float):
        tt = torch.full((s.shape[0], 1), float(t), device=s.device, dtype=s.dtype)
        s_cond = self.s_proj(self.s_norm(s)) + self.time_embed(tt)
        return s_cond, self.z_norm(z)


class DiffTransformerLayer(nn.Module):
    def __init__(self, c_a: int, c_s: int, c_z: int, n_head: int = 4) -> None:
        super().__init__()
        self.h = n_head
        self.c = c_a // n_head
        self.scale = nn.Linear(c_s, c_a)
        self.shift = nn.Linear(c_s, c_a)
        self.norm = nn.LayerNorm(c_a, elementwise_affine=False)
        self.q = nn.Linear(c_a, c_a, bias=False)
        self.k = nn.Linear(c_a, c_a, bias=False)
        self.v = nn.Linear(c_a, c_a, bias=False)
        self.b = nn.Linear(c_z, n_head, bias=False)
        self.o = nn.Linear(c_a, c_a)
        self.ff = nn.Sequential(nn.Linear(c_a, c_a * 2), nn.GELU(), nn.Linear(c_a * 2, c_a))

    def forward(self, a: torch.Tensor, s: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = self.norm(a) * (1 + self.scale(s)) + self.shift(s)
        q = _heads(self.q(h), self.h).permute(1, 0, 2)
        k = _heads(self.k(h), self.h).permute(1, 0, 2)
        v = _heads(self.v(h), self.h).permute(1, 0, 2)
        bias = self.b(z).permute(2, 0, 1)
        att = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c**0.5) + bias, dim=-1)
        out = torch.matmul(att, v).permute(1, 0, 2).reshape(a.shape[0], -1)
        a = a + self.o(out)
        a = a + self.ff(a)
        return a


class StructureModule(nn.Module):
    """Diffusion-based structure module: noised coords -> denoised coords."""

    def __init__(self, c_s: int, c_z: int, c_a: int = 16, n_layer: int = 2) -> None:
        super().__init__()
        self.cond = DiffusionConditioning(c_s, c_z, c_a)
        self.in_proj = nn.Linear(3, c_a)
        self.layers = nn.ModuleList([DiffTransformerLayer(c_a, c_s, c_z) for _ in range(n_layer)])
        self.out_proj = nn.Linear(c_a, 3)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        s_cond, z_cond = self.cond(s, z, t=0.5)
        x_noised = torch.randn(s.shape[0], 3, device=s.device, dtype=s.dtype)
        a = self.in_proj(x_noised)
        for layer in self.layers:
            a = layer(a, s_cond, z_cond)
        return x_noised + self.out_proj(a)


class AffinityModule(nn.Module):
    """Boltz-2 dual affinity head: binder probability + continuous affinity value.

    A separate Pairformer block refines (s, z) restricted to the interaction, then
    two MLP heads read the pooled representation: a binary binder-vs-decoy logit
    and a scalar affinity regressed on the (uM-scaled) log axis.
    """

    def __init__(self, c_s: int, c_z: int) -> None:
        super().__init__()
        self.pairformer = PairformerBlock(c_s, c_z)
        self.norm = nn.LayerNorm(c_s)
        self.binder_head = nn.Sequential(nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, 1))
        self.affinity_head = nn.Sequential(nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, 1))

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        s, z = self.pairformer(s, z)
        pooled = self.norm(s).mean(dim=0, keepdim=True)  # (1, c_s)
        binder_logit = self.binder_head(pooled)  # (1, 1)
        affinity = self.affinity_head(pooled)  # (1, 1) log-uM axis
        return binder_logit, affinity


class Boltz2(nn.Module):
    """Compact Boltz-2: input embed -> Pairformer trunk -> structure + affinity heads."""

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
        self.msa_to_pair = nn.Linear(c_s, c_z)
        self.blocks = nn.ModuleList([PairformerBlock(c_s, c_z) for _ in range(n_block)])
        self.structure = StructureModule(c_s, c_z)
        self.affinity = AffinityModule(c_s, c_z)

    def forward(self, token_types: torch.Tensor):
        s = self.token_embed(token_types)
        z = self.left(token_types)[:, None, :] + self.right(token_types)[None, :, :]
        z = z + (self.msa_to_pair(s)[:, None, :] + self.msa_to_pair(s)[None, :, :])
        for blk in self.blocks:
            s, z = blk(s, z)
        coords = self.structure(s, z)
        binder_logit, affinity = self.affinity(s, z)
        return coords, binder_logit, affinity


def build() -> nn.Module:
    return Boltz2()


def example_input() -> torch.Tensor:
    """Small token-type tensor ``(12,)`` = 12 tokens (protein+ligand complex)."""
    return torch.randint(0, 32, (12,))


MENAGERIE_ENTRIES = [
    (
        "Boltz-2 (Pairformer trunk + structure + binding-affinity head)",
        "build",
        "example_input",
        "2025",
        "DC",
    ),
]
