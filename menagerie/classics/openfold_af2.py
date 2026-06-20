"""OpenFold / AlphaFold2: Evoformer trunk + Invariant Point Attention structure module.

Jumper et al. (DeepMind), AlphaFold2, Nature 2021. https://www.nature.com/articles/s41586-021-03819-2
OpenFold (Ahdritz et al. 2022) is the faithful open PyTorch reimplementation.
Source: https://github.com/aqlaboratory/openfold

AlphaFold2's distinctive contribution is the **Evoformer**: a stack of blocks that
co-evolve an MSA representation ``m`` (N_seq, N_res, c_m) and a pair representation
``z`` (N_res, N_res, c_z) through:
  - MSA row-wise gated self-attention **biased by the pair representation**
  - MSA column-wise gated self-attention
  - an MSA transition (2-layer MLP)
  - communication MSA -> pair via an **outer product mean**
  - **triangle multiplicative updates** (outgoing + incoming edges)
  - **triangle self-attention** (around starting node + around ending node)
  - a pair transition
followed by the **structure module**: 8 shared-weight iterations of **Invariant
Point Attention (IPA)** over a set of per-residue rigid frames, producing backbone
frames + sidechain torsion angles. IPA attends with three terms (scalar q.k,
pair bias, and squared-distance of attention "points" transformed into the global
frame) and is invariant to global rotation/translation.

This is a faithful compact reimplementation: the exact Evoformer block wiring and
IPA three-term attention are reproduced, at small channel widths / few residues /
few blocks so the unrolled atlas graph renders quickly. Random init, forward-only.

Faithful-core simplifications (honest, not lies):
  - frames are represented by their rotation matrix + translation directly rather
    than the quaternion+exp-map ``Rigid`` helper class; the IPA math is identical.
  - 2 Evoformer blocks (vs 48) and 2 IPA iterations (vs 8); small c_m/c_z/N_res.
  - the input MSA+pair tensors are produced from a small random ``aatype`` index
    sequence (standing in for AF2's feature-dict embedding stack), so the example
    input is a single small integer tensor the forward consumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    *lead, c = x.shape
    return x.view(*lead, n_head, c // n_head)


class MSARowAttentionWithPairBias(nn.Module):
    """Row-wise (per-sequence) gated MSA self-attention, biased by the pair rep."""

    def __init__(self, c_m: int, c_z: int, c_hidden: int = 8, n_head: int = 4) -> None:
        super().__init__()
        self.n_head = n_head
        self.c_hidden = c_hidden
        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_m, c_hidden * n_head, bias=False)
        self.linear_k = nn.Linear(c_m, c_hidden * n_head, bias=False)
        self.linear_v = nn.Linear(c_m, c_hidden * n_head, bias=False)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.linear_g = nn.Linear(c_m, c_hidden * n_head)
        self.linear_o = nn.Linear(c_hidden * n_head, c_m)

    def forward(self, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # m: (S, R, c_m); z: (R, R, c_z)
        m = self.norm_m(m)
        bias = self.linear_b(self.norm_z(z))  # (R, R, n_head)
        bias = bias.permute(2, 0, 1)  # (n_head, R, R)
        q = _gather_heads(self.linear_q(m), self.n_head)  # (S, R, H, d)
        k = _gather_heads(self.linear_k(m), self.n_head)
        v = _gather_heads(self.linear_v(m), self.n_head)
        # attention over residue axis, per head
        q = q.permute(0, 2, 1, 3)  # (S, H, R, d)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        a = torch.matmul(q, k.transpose(-1, -2)) / (self.c_hidden**0.5)
        a = a + bias.unsqueeze(0)  # broadcast over sequences
        a = torch.softmax(a, dim=-1)
        o = torch.matmul(a, v)  # (S, H, R, d)
        o = o.permute(0, 2, 1, 3).reshape(m.shape[0], m.shape[1], -1)
        g = torch.sigmoid(self.linear_g(m))
        return self.linear_o(o * g)


class MSAColumnAttention(nn.Module):
    """Column-wise (per-residue) gated MSA self-attention over the sequence axis."""

    def __init__(self, c_m: int, c_hidden: int = 8, n_head: int = 4) -> None:
        super().__init__()
        self.n_head = n_head
        self.c_hidden = c_hidden
        self.norm_m = nn.LayerNorm(c_m)
        self.linear_q = nn.Linear(c_m, c_hidden * n_head, bias=False)
        self.linear_k = nn.Linear(c_m, c_hidden * n_head, bias=False)
        self.linear_v = nn.Linear(c_m, c_hidden * n_head, bias=False)
        self.linear_g = nn.Linear(c_m, c_hidden * n_head)
        self.linear_o = nn.Linear(c_hidden * n_head, c_m)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        m = self.norm_m(m)  # (S, R, c_m)
        mt = m.transpose(0, 1)  # (R, S, c_m)
        q = _gather_heads(self.linear_q(mt), self.n_head).permute(0, 2, 1, 3)  # (R,H,S,d)
        k = _gather_heads(self.linear_k(mt), self.n_head).permute(0, 2, 1, 3)
        v = _gather_heads(self.linear_v(mt), self.n_head).permute(0, 2, 1, 3)
        a = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (self.c_hidden**0.5), dim=-1)
        o = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(mt.shape[0], mt.shape[1], -1)
        g = torch.sigmoid(self.linear_g(mt))
        return self.linear_o(o * g).transpose(0, 1)


class Transition(nn.Module):
    """2-layer ReLU MLP transition (used for both MSA and pair)."""

    def __init__(self, c: int, n: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.l1 = nn.Linear(c, c * n)
        self.l2 = nn.Linear(c * n, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(F.relu(self.l1(self.norm(x))))


class OuterProductMean(nn.Module):
    """MSA -> pair communication via outer product mean over sequences."""

    def __init__(self, c_m: int, c_z: int, c_hidden: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(c_m)
        self.linear_a = nn.Linear(c_m, c_hidden)
        self.linear_b = nn.Linear(c_m, c_hidden)
        self.linear_o = nn.Linear(c_hidden * c_hidden, c_z)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        m = self.norm(m)
        a = self.linear_a(m)  # (S, R, c)
        b = self.linear_b(m)
        # outer product over channel, mean over sequences
        outer = torch.einsum("sic,sjd->ijcd", a, b) / m.shape[0]
        outer = outer.reshape(outer.shape[0], outer.shape[1], -1)
        return self.linear_o(outer)


class TriangleMultiplication(nn.Module):
    """Triangle multiplicative update (outgoing if ``outgoing`` else incoming)."""

    def __init__(self, c_z: int, c_hidden: int = 8, outgoing: bool = True) -> None:
        super().__init__()
        self.outgoing = outgoing
        self.norm = nn.LayerNorm(c_z)
        self.linear_ap = nn.Linear(c_z, c_hidden)
        self.linear_ag = nn.Linear(c_z, c_hidden)
        self.linear_bp = nn.Linear(c_z, c_hidden)
        self.linear_bg = nn.Linear(c_z, c_hidden)
        self.norm_out = nn.LayerNorm(c_hidden)
        self.linear_g = nn.Linear(c_z, c_z)
        self.linear_o = nn.Linear(c_hidden, c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.norm(z)
        a = torch.sigmoid(self.linear_ag(z)) * self.linear_ap(z)  # (R,R,c)
        b = torch.sigmoid(self.linear_bg(z)) * self.linear_bp(z)
        if self.outgoing:
            x = torch.einsum("ikc,jkc->ijc", a, b)
        else:
            x = torch.einsum("kic,kjc->ijc", a, b)
        g = torch.sigmoid(self.linear_g(z))
        return g * self.linear_o(self.norm_out(x))


class TriangleAttention(nn.Module):
    """Triangle self-attention (around starting node if ``starting`` else ending)."""

    def __init__(self, c_z: int, c_hidden: int = 8, n_head: int = 4, starting: bool = True) -> None:
        super().__init__()
        self.starting = starting
        self.n_head = n_head
        self.c_hidden = c_hidden
        self.norm = nn.LayerNorm(c_z)
        self.linear_q = nn.Linear(c_z, c_hidden * n_head, bias=False)
        self.linear_k = nn.Linear(c_z, c_hidden * n_head, bias=False)
        self.linear_v = nn.Linear(c_z, c_hidden * n_head, bias=False)
        self.linear_b = nn.Linear(c_z, n_head, bias=False)
        self.linear_g = nn.Linear(c_z, c_hidden * n_head)
        self.linear_o = nn.Linear(c_hidden * n_head, c_z)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if not self.starting:
            z = z.transpose(0, 1)
        z = self.norm(z)
        q = _gather_heads(self.linear_q(z), self.n_head).permute(0, 2, 1, 3)  # (R,H,R,d)
        k = _gather_heads(self.linear_k(z), self.n_head).permute(0, 2, 1, 3)
        v = _gather_heads(self.linear_v(z), self.n_head).permute(0, 2, 1, 3)
        bias = self.linear_b(z).permute(2, 0, 1)  # (H, R, R)
        a = torch.matmul(q, k.transpose(-1, -2)) / (self.c_hidden**0.5)
        a = a + bias.unsqueeze(0)
        a = torch.softmax(a, dim=-1)
        o = torch.matmul(a, v).permute(0, 2, 1, 3).reshape(z.shape[0], z.shape[1], -1)
        g = torch.sigmoid(self.linear_g(z))
        out = self.linear_o(o * g)
        if not self.starting:
            out = out.transpose(0, 1)
        return out


class EvoformerBlock(nn.Module):
    """One Evoformer block: MSA stack -> outer-product-mean -> pair stack."""

    def __init__(self, c_m: int, c_z: int) -> None:
        super().__init__()
        self.msa_row = MSARowAttentionWithPairBias(c_m, c_z)
        self.msa_col = MSAColumnAttention(c_m)
        self.msa_trans = Transition(c_m)
        self.opm = OuterProductMean(c_m, c_z)
        self.tri_mul_out = TriangleMultiplication(c_z, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(c_z, outgoing=False)
        self.tri_attn_start = TriangleAttention(c_z, starting=True)
        self.tri_attn_end = TriangleAttention(c_z, starting=False)
        self.pair_trans = Transition(c_z)

    def forward(self, m: torch.Tensor, z: torch.Tensor):
        m = m + self.msa_row(m, z)
        m = m + self.msa_col(m)
        m = m + self.msa_trans(m)
        z = z + self.opm(m)
        z = z + self.tri_mul_out(z)
        z = z + self.tri_mul_in(z)
        z = z + self.tri_attn_start(z)
        z = z + self.tri_attn_end(z)
        z = z + self.pair_trans(z)
        return m, z


class InvariantPointAttention(nn.Module):
    """Invariant Point Attention: scalar + pair-bias + 3D-point attention terms."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        c_hidden: int = 8,
        n_head: int = 4,
        n_qk_points: int = 4,
        n_v_points: int = 4,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.c_hidden = c_hidden
        self.n_qk_points = n_qk_points
        self.n_v_points = n_v_points
        self.linear_q = nn.Linear(c_s, c_hidden * n_head)
        self.linear_kv = nn.Linear(c_s, 2 * c_hidden * n_head)
        self.linear_q_points = nn.Linear(c_s, n_head * n_qk_points * 3)
        self.linear_kv_points = nn.Linear(c_s, n_head * (n_qk_points + n_v_points) * 3)
        self.linear_b = nn.Linear(c_z, n_head)
        self.head_weight = nn.Parameter(torch.zeros(n_head))
        concat = n_head * (c_hidden + c_z + n_v_points * 4)
        self.linear_o = nn.Linear(concat, c_s)

    def forward(self, s: torch.Tensor, z: torch.Tensor, R: torch.Tensor, t: torch.Tensor):
        # s:(R,c_s) z:(R,R,c_z) R:(R,3,3) frame rot, t:(R,3) frame translation
        Nr = s.shape[0]
        H, C = self.n_head, self.c_hidden
        q = self.linear_q(s).view(Nr, H, C)
        kv = self.linear_kv(s).view(Nr, H, 2 * C)
        k, v = kv.split(C, dim=-1)

        def to_global(local: torch.Tensor) -> torch.Tensor:
            # local: (R, H, P, 3) in local frame -> global
            return torch.einsum("rij,rhpj->rhpi", R, local) + t[:, None, None, :]

        qp = self.linear_q_points(s).view(Nr, H, self.n_qk_points, 3)
        kvp = self.linear_kv_points(s).view(Nr, H, self.n_qk_points + self.n_v_points, 3)
        kp, vp = kvp.split([self.n_qk_points, self.n_v_points], dim=2)
        qp_g = to_global(qp)
        kp_g = to_global(kp)
        vp_g = to_global(vp)

        # scalar attention logits
        a_scalar = torch.einsum("ihc,jhc->hij", q, k) / (C**0.5)
        # pair bias
        b = self.linear_b(z).permute(2, 0, 1)  # (H, R, R)
        # point attention: -gamma_h/2 * sum ||q_i - k_j||^2
        diff = qp_g[:, None] - kp_g[None]  # (Ri, Rj, H, P, 3)
        sq = diff.pow(2).sum(dim=(-1, -2)).permute(2, 0, 1)  # (H, Ri, Rj)
        gamma = F.softplus(self.head_weight).view(H, 1, 1)
        a = a_scalar * (1.0 / 3.0) ** 0.5 + b * (1.0 / 3.0) ** 0.5 - 0.5 * gamma * sq
        a = torch.softmax(a, dim=-1)  # (H, Ri, Rj)

        # scalar output
        o_scalar = torch.einsum("hij,jhc->ihc", a, v).reshape(Nr, -1)
        # pair output
        o_pair = torch.einsum("hij,ijc->ihc", a, z).reshape(Nr, -1)
        # point output (global -> local of residue i)
        o_pt_g = torch.einsum("hij,jhpk->ihpk", a, vp_g)  # (R,H,Pv,3)
        Rt = R.transpose(-1, -2)
        o_pt_local = torch.einsum("rij,rhpj->rhpi", Rt, o_pt_g - t[:, None, None, :])
        o_pt_norm = torch.sqrt(o_pt_local.pow(2).sum(-1) + 1e-8)  # (R,H,Pv)
        o_pt = torch.cat([o_pt_local.reshape(Nr, -1), o_pt_norm.reshape(Nr, -1)], dim=-1)

        out = torch.cat([o_scalar, o_pair, o_pt], dim=-1)
        return self.linear_o(out)


class StructureModule(nn.Module):
    """Shared-weight IPA iterations producing backbone frames + torsion angles."""

    def __init__(self, c_s: int, c_z: int, n_iter: int = 2) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.norm_s = nn.LayerNorm(c_s)
        self.norm_z = nn.LayerNorm(c_z)
        self.linear_in = nn.Linear(c_s, c_s)
        self.ipa = InvariantPointAttention(c_s, c_z)
        self.norm_ipa = nn.LayerNorm(c_s)
        self.transition = nn.Sequential(
            nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, c_s)
        )
        self.norm_trans = nn.LayerNorm(c_s)
        self.bb_update = nn.Linear(c_s, 6)  # 3 rot (axis-angle) + 3 trans
        self.angle_head = nn.Sequential(nn.Linear(c_s, c_s), nn.ReLU(), nn.Linear(c_s, 7 * 2))

    @staticmethod
    def _axis_angle_to_matrix(v: torch.Tensor) -> torch.Tensor:
        # v: (R, 3) -> (R, 3, 3) via Rodrigues
        theta = torch.linalg.norm(v, dim=-1, keepdim=True) + 1e-8
        k = v / theta
        K = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
        K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
        K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
        K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
        eye = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0)
        th = theta.unsqueeze(-1)
        return eye + torch.sin(th) * K + (1 - torch.cos(th)) * torch.matmul(K, K)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        s = self.linear_in(self.norm_s(s))
        z = self.norm_z(z)
        Nr = s.shape[0]
        R = torch.eye(3, device=s.device, dtype=s.dtype).unsqueeze(0).repeat(Nr, 1, 1)
        t = torch.zeros(Nr, 3, device=s.device, dtype=s.dtype)
        for _ in range(self.n_iter):
            s = s + self.ipa(s, z, R, t)
            s = self.norm_ipa(s)
            s = s + self.transition(s)
            s = self.norm_trans(s)
            upd = self.bb_update(s)
            dR = self._axis_angle_to_matrix(upd[:, :3])
            R = torch.matmul(R, dR)
            t = t + torch.einsum("rij,rj->ri", R, upd[:, 3:])
        angles = self.angle_head(s).view(Nr, 7, 2)
        angles = angles / (torch.linalg.norm(angles, dim=-1, keepdim=True) + 1e-8)
        return t, R, angles


class AlphaFold2(nn.Module):
    """Compact AlphaFold2 / OpenFold: input embedder -> Evoformer -> structure module."""

    def __init__(
        self,
        c_m: int = 16,
        c_z: int = 16,
        c_s: int = 16,
        n_block: int = 2,
        n_token: int = 22,
    ) -> None:
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        # input feature embedding (stands in for AF2's preprocessing/embedding stack)
        self.msa_embed = nn.Embedding(n_token, c_m)
        self.left_embed = nn.Embedding(n_token, c_z)
        self.right_embed = nn.Embedding(n_token, c_z)
        self.blocks = nn.ModuleList([EvoformerBlock(c_m, c_z) for _ in range(n_block)])
        self.s_proj = nn.Linear(c_m, c_s)  # MSA first-row -> single rep
        self.structure = StructureModule(c_s, c_z)

    def forward(self, aatype_msa: torch.Tensor) -> torch.Tensor:
        # aatype_msa: (S, R) integer residue/MSA tokens
        m = self.msa_embed(aatype_msa)  # (S, R, c_m)
        seq = aatype_msa[0]  # (R,) target sequence
        z = self.left_embed(seq)[:, None, :] + self.right_embed(seq)[None, :, :]  # (R,R,c_z)
        for blk in self.blocks:
            m, z = blk(m, z)
        s = self.s_proj(m[0])  # single representation from first MSA row
        t, R, angles = self.structure(s, z)
        # return predicted CA coordinates (R, 3) as the model output
        return t


def build() -> nn.Module:
    return AlphaFold2()


def example_input() -> torch.Tensor:
    """Small MSA token tensor ``(4, 10)`` = 4 sequences x 10 residues."""
    return torch.randint(0, 22, (4, 10))


MENAGERIE_ENTRIES = [
    (
        "OpenFold / AlphaFold2 (Evoformer + IPA structure module)",
        "build",
        "example_input",
        "2021",
        "DC",
    ),
]
