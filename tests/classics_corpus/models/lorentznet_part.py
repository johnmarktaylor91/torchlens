"""LorentzNet and ParT: physics-motivated particle jet classifiers.

LorentzNet:
  Gong et al., "An Efficient Lorentz Equivariant Graph Neural Network for
  Jet Tagging." arXiv:2201.08187 (2022).
  Source: https://github.com/sdogsq/LorentzNet-release

ParT (Particle Transformer):
  Qu et al., "Particle Transformer for Jet Tagging." arXiv:2202.03772 (2022).
  Source: https://github.com/jet-universe/particle_transformer

------------------------------------------------------------------------------
LorentzNet distinctive primitive:
  Lorentz Group Equivariant Block (LGEB). Each block uses the Minkowski
  inner product  <p_i, p_j> = p_i^0 p_j^0 - p_i^1 p_j^1 - p_i^2 p_j^2 - p_i^3 p_j^3
  and Lorentz norm  <p_i, p_i>  as invariant scalars to form edge features,
  then applies a fully-connected message-passing network to update node
  features. No global-frame dependence; edge features are Lorentz scalars
  only. Final readout pools over nodes.

Faithful-compact simplifications:
  - LGEB: phi_e/phi_h/phi_x MLP widths = 32 vs 72 in paper.
  - 2 LGEB layers vs 6.
  - 8 particles, 4-momentum inputs (E, px, py, pz).
  - Classification head reduced to 2-class.

------------------------------------------------------------------------------
ParT (Particle Transformer) distinctive primitive:
  Standard transformer over N particles but with PAIRWISE INTERACTION
  features (U matrix) derived from the 4-momenta pair statistics:
    deltaR = sqrt((eta_i - eta_j)^2 + (phi_i - phi_j)^2)
    kT     = min(pT_i, pT_j) * deltaR
    z      = min(pT_i, pT_j) / (pT_i + pT_j)
    m^2    = (p_i + p_j)^2  (invariant mass squared)
  These four log scalars -> a small MLP -> U[i, j], which is added to the
  raw QK attention logits BEFORE softmax. This "interaction bias" makes the
  attention physics-aware.

Faithful-compact simplifications:
  - 2 transformer layers, d_model=32, 4 heads.
  - Interaction MLP: 4 scalar features -> 32 hidden -> n_heads per pair.
  - 12 particles.
  - Classification head: 2-class.
  - No particle-type embedding or momentum rescaling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LorentzNet
# =============================================================================


def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Minkowski inner product: x^0 y^0 - x^1 y^1 - x^2 y^2 - x^3 y^3.
    x, y: (..., 4)  ->  (...,)
    """
    # metric signature (+, -, -, -)
    metric = x.new_tensor([1.0, -1.0, -1.0, -1.0])
    return (x * y * metric).sum(dim=-1)


class LGEBlock(nn.Module):
    """Lorentz Group Equivariant Block (LGEB).

    Takes node features h (N, d_h) and 4-momenta p (N, 4).
    Builds all N*(N-1) pairs, computes Minkowski scalars as edge features,
    runs phi_e MLP on [h_i, h_j, <p_i,p_j>, <p_i,p_i>, <p_j,p_j>],
    aggregates messages, then updates h via phi_h MLP.
    """

    def __init__(self, d_h: int, d_e: int) -> None:
        super().__init__()
        self.phi_e = nn.Sequential(
            nn.Linear(2 * d_h + 3, d_e),
            nn.SiLU(),
            nn.Linear(d_e, d_e),
            nn.SiLU(),
        )
        self.phi_h = nn.Sequential(
            nn.Linear(d_h + d_e, d_h),
            nn.SiLU(),
            nn.Linear(d_h, d_h),
        )

    def forward(self, h: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """h: (N, d_h),  p: (N, 4)  ->  h': (N, d_h)"""
        N = h.size(0)
        # Pairwise Minkowski scalars
        p_i = p.unsqueeze(1).expand(N, N, 4)  # (N, N, 4)
        p_j = p.unsqueeze(0).expand(N, N, 4)  # (N, N, 4)
        dot_ij = minkowski_dot(p_i, p_j)  # (N, N)
        dot_ii = minkowski_dot(p, p)  # (N,)
        dot_jj = dot_ii.unsqueeze(0).expand(N, N)  # (N, N)
        dot_ii_2d = dot_ii.unsqueeze(1).expand(N, N)

        # Edge input: [h_i, h_j, <p_i,p_j>, <p_i,p_i>, <p_j,p_j>]
        h_i = h.unsqueeze(1).expand(N, N, -1)
        h_j = h.unsqueeze(0).expand(N, N, -1)
        # Stack scalars: (N, N, 3)
        scalars = torch.stack([dot_ij, dot_ii_2d, dot_jj], dim=-1)
        e_input = torch.cat([h_i, h_j, scalars], dim=-1)  # (N, N, 2*d_h+3)
        m = self.phi_e(e_input)  # (N, N, d_e)

        # Zero self-edges
        mask = 1.0 - torch.eye(N, device=h.device).unsqueeze(-1)
        m = m * mask

        # Aggregate messages
        agg = m.sum(dim=1)  # (N, d_e)
        h_new = self.phi_h(torch.cat([h, agg], dim=-1))
        return h_new


class LorentzNet(nn.Module):
    """LorentzNet for jet classification.

    Input: particles as 4-momenta (N, 4).
    Embeds to hidden dim, applies LGEB blocks, mean-pools, classifies.
    """

    def __init__(self, n_classes: int = 2, d_h: int = 32, d_e: int = 32, n_blocks: int = 2) -> None:
        super().__init__()
        self.embed = nn.Linear(4, d_h)
        self.blocks = nn.ModuleList([LGEBlock(d_h, d_e) for _ in range(n_blocks)])
        self.cls = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.SiLU(),
            nn.Linear(d_h, n_classes),
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """p: (N, 4)  ->  logits (n_classes,)"""
        h = F.silu(self.embed(p))  # (N, d_h)
        for blk in self.blocks:
            h = h + blk(h, p)
        out = h.mean(dim=0)  # (d_h,)
        return self.cls(out)


def build_lorentznet() -> nn.Module:
    return LorentzNet(n_classes=2, d_h=32, d_e=32, n_blocks=2)


def example_input_lorentznet() -> torch.Tensor:
    """8 particles, each with 4-momentum (E, px, py, pz)."""
    torch.manual_seed(0)
    return torch.randn(8, 4)


# =============================================================================
# ParT (Particle Transformer)
# =============================================================================


def _pairwise_interaction_features(p: torch.Tensor) -> torch.Tensor:
    """Compute 4 pairwise physics features from 4-momenta.

    p: (N, 4) columns = (E, px, py, pz).
    Returns (N, N, 4) with features: [log(deltaR+1e-6), log(kT+1e-6), log(z+1e-6), log(|m^2|+1e-6)]
    """
    N = p.size(0)
    E = p[:, 0]  # (N,)
    px = p[:, 1]
    py = p[:, 2]
    pz = p[:, 3]
    pT = torch.sqrt(px**2 + py**2 + 1e-8)

    # pseudo-rapidity  eta = -log(tan(theta/2)), theta from pz/|p|
    # simplified: eta approx atanh(pz / sqrt(pT^2 + pz^2))
    p3 = torch.sqrt(pT**2 + pz**2 + 1e-8)
    eta = 0.5 * torch.log((p3 + pz + 1e-8) / (p3 - pz + 1e-8))  # (N,)
    phi_angle = torch.atan2(py, px + 1e-8)  # (N,)

    eta_i = eta.unsqueeze(1).expand(N, N)
    eta_j = eta.unsqueeze(0).expand(N, N)
    phi_i = phi_angle.unsqueeze(1).expand(N, N)
    phi_j = phi_angle.unsqueeze(0).expand(N, N)
    pT_i = pT.unsqueeze(1).expand(N, N)
    pT_j = pT.unsqueeze(0).expand(N, N)

    deltaR = torch.sqrt((eta_i - eta_j) ** 2 + (phi_i - phi_j) ** 2 + 1e-8)
    kT = torch.min(pT_i, pT_j) * deltaR
    z_val = torch.min(pT_i, pT_j) / (pT_i + pT_j + 1e-8)

    # Invariant mass squared: (p_i + p_j)^2 in Minkowski
    p_sum = p.unsqueeze(1) + p.unsqueeze(0)  # (N, N, 4)
    m2 = p_sum[..., 0] ** 2 - p_sum[..., 1] ** 2 - p_sum[..., 2] ** 2 - p_sum[..., 3] ** 2

    feat = torch.stack(
        [
            torch.log(deltaR + 1e-6),
            torch.log(kT + 1e-6),
            torch.log(z_val + 1e-6),
            torch.log(m2.abs() + 1e-6),
        ],
        dim=-1,
    )  # (N, N, 4)
    return feat


class InteractionMLP(nn.Module):
    """Maps pairwise physics features -> per-head attention bias U[i, j, head]."""

    def __init__(self, n_heads: int, d_int: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, d_int),
            nn.SiLU(),
            nn.Linear(d_int, n_heads),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (N, N, 4)  ->  U: (N, N, n_heads)"""
        return self.net(feat)


class ParTBlock(nn.Module):
    """One transformer block with pairwise-interaction attention bias (U matrix)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.interaction_mlp = InteractionMLP(n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, U_feat: torch.Tensor) -> torch.Tensor:
        """x: (N, d_model),  U_feat: (N, N, 4)  ->  (N, d_model)"""
        N = x.size(0)
        res = x
        x = self.norm1(x)
        qkv = self.qkv(x).view(N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=1)  # each (N, n_heads, d_head)

        # Attention logits
        attn = torch.einsum("ihd,jhd->ijh", q, k) / self.scale  # (N, N, n_heads)

        # Add pairwise interaction bias U
        U = self.interaction_mlp(U_feat)  # (N, N, n_heads)
        attn = attn + U

        attn = F.softmax(attn, dim=1)  # normalize over keys

        # Aggregate values
        out = torch.einsum("ijh,jhd->ihd", attn, v)  # (N, n_heads, d_head)
        out = out.reshape(N, -1)
        out = self.proj(out)
        x = res + out

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class ParticleTransformer(nn.Module):
    """ParT: Particle Transformer for jet tagging.

    Input: 4-momenta (N, 4).
    Embeds particles, computes pairwise physics features once, runs transformer
    with interaction-bias attention, pools, classifies.
    """

    def __init__(
        self, n_classes: int = 2, d_model: int = 32, n_heads: int = 4, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(4, d_model)
        self.blocks = nn.ModuleList([ParTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, n_classes)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """p: (N, 4)  ->  logits (n_classes,)"""
        U_feat = _pairwise_interaction_features(p)  # (N, N, 4)
        x = F.gelu(self.embed(p))  # (N, d_model)
        for blk in self.blocks:
            x = blk(x, U_feat)
        x = self.norm(x)
        out = x.mean(dim=0)  # pool
        return self.cls(out)


def build_part() -> nn.Module:
    return ParticleTransformer(n_classes=2, d_model=32, n_heads=4, n_layers=2)


def example_input_part() -> torch.Tensor:
    """12 particles, 4-momentum each."""
    torch.manual_seed(1)
    return torch.randn(12, 4)


# =============================================================================
# Registry
# =============================================================================

MENAGERIE_ENTRIES = [
    (
        "LorentzNet",
        "build_lorentznet",
        "example_input_lorentznet",
        "2022",
        "DC",
    ),
    (
        "Particle Transformer (ParT)",
        "build_part",
        "example_input_part",
        "2022",
        "DC",
    ),
]
