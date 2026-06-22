"""FactFormer: Factorized Transformer for Spatio-Temporal PDE Operator Learning.

Li et al., 2023.
Paper: https://arxiv.org/abs/2305.17560
Source: https://github.com/alasdairtran/fourierflow (related); official: https://github.com/lucidrains/factorized-attention

FactFormer uses FACTORIZED AXIAL KERNEL-INTEGRAL ATTENTION:
  - Instead of full O(N^2) attention over the entire spatio-temporal token grid,
    it factorizes the kernel integral along each spatial/temporal axis SEPARATELY.
  - For a 2D+time grid (T, H, W), three separate 1D attention passes are performed:
      * Along H (rows) with shared K,V
      * Along W (cols) with shared K,V
      * Along T (time) with shared K,V
  - Each axis's attention is a KERNEL INTEGRAL: integral_Omega K(x,y) v(y) dy
    approximated as softmax attention along that axis.
  - The outputs of the three axis-attentions are combined (summed or concatenated+project).
  - This gives O(T*H*W*(T+H+W)) complexity vs O((T*H*W)^2).

The "low-rank" aspect: factorizing the full kernel K(x,y) = Kx(x1,y1)*Ky(x2,y2)*Kt(xt,yt)
into axis-wise products -- each axis shares the same feature map (K,V) across positions in
the other axes.

Simplifications: 2D spatial + 1 time step (H=8, W=8, T=4); 2 FactFormer blocks;
d_model=64; 4 heads per axis. The factorized axial attention is the central primitive.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def axial_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, axis: int) -> torch.Tensor:
    """Perform attention along a specific axis of a multi-dimensional tensor.

    q, k, v: (B, T, H, W, D) or similar -- multi-dimensional feature grids.
    axis: which spatial/temporal axis to attend over (0=T, 1=H, 2=W).
    Returns: same shape as q.
    """
    orig_shape = q.shape
    B = q.shape[0]
    D = q.shape[-1]

    # Axes: 1=T, 2=H, 3=W (after batch)
    # Move the target axis to dim=2, collapse others to dim=1 (batch-like)
    # shape: (B, T, H, W, D)
    axes = [1, 2, 3]  # T, H, W
    attend_axis = axes[axis]  # 1-indexed (excluding batch)
    other_axes = [a for a in axes if a != attend_axis]

    # Permute: (B, other..., attend_len, D)
    perm = [0] + other_axes + [attend_axis, 4]
    q_p = q.permute(*perm)
    k_p = k.permute(*perm)
    v_p = v.permute(*perm)

    # Flatten other spatial dims into batch
    *other_dims, attend_len, d = q_p.shape
    batch_size = 1
    for d_ in other_dims:
        batch_size *= d_
    q_flat = q_p.reshape(batch_size, attend_len, d)
    k_flat = k_p.reshape(batch_size, attend_len, d)
    v_flat = v_p.reshape(batch_size, attend_len, d)

    # Standard dot-product attention along the attend axis
    scale = d**-0.5
    scores = q_flat @ k_flat.transpose(-2, -1) * scale  # (batch, attend_len, attend_len)
    attn = torch.softmax(scores, dim=-1)
    out_flat = attn @ v_flat  # (batch, attend_len, d)

    # Reshape back
    out_p = out_flat.reshape(*other_dims, attend_len, d)
    # Inverse permute
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    out = out_p.permute(*inv_perm)
    return out


class FactorizedAxialAttention(nn.Module):
    """Factorized axial kernel-integral attention: separate 1D attention per axis.

    Operates on (B, T, H, W, d_model) grids. Performs attention separately along
    T, H, W axes and sums the results (low-rank factorization of kernel integral).
    """

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # One set of Q, K, V projections per axis (T, H, W)
        self.qkv_T = nn.Linear(d_model, 3 * d_model)
        self.qkv_H = nn.Linear(d_model, 3 * d_model)
        self.qkv_W = nn.Linear(d_model, 3 * d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) -> (..., n_heads, d_head), flatten heads into last 2."""
        return x  # Keep as-is; axial_attention handles flat last dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, d_model)
        def attend_axis(x_in: torch.Tensor, qkv_proj: nn.Linear, ax: int) -> torch.Tensor:
            qkv = qkv_proj(x_in)
            q, k, v = qkv.chunk(3, dim=-1)
            return axial_attention(q, k, v, ax)

        out_T = attend_axis(x, self.qkv_T, 0)  # attend along T
        out_H = attend_axis(x, self.qkv_H, 1)  # attend along H
        out_W = attend_axis(x, self.qkv_W, 2)  # attend along W

        # Sum factorized contributions
        out = out_T + out_H + out_W
        return self.out_proj(out)


class FactFormerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.attn = FactorizedAxialAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, d_model)
        x = x + self.attn(self.n1(x))
        x = x + self.ffn(self.n2(x))
        return x


class FactFormer(nn.Module):
    """FactFormer: Factorized Axial Attention PDE Operator Transformer.

    Input: spatio-temporal function grid (B, T, H, W, in_channels).
    Output: predicted field (B, T, H, W, out_channels).
    """

    def __init__(
        self,
        in_channels: int = 2,  # u + 1 (e.g., Navier-Stokes vorticity + forcing)
        out_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        T: int = 4,
        H: int = 8,
        W: int = 8,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        # Factorized axial position embeddings
        self.pos_T = nn.Embedding(T, d_model)
        self.pos_H = nn.Embedding(H, d_model)
        self.pos_W = nn.Embedding(W, d_model)
        self.blocks = nn.ModuleList([FactFormerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_channels),
        )
        self.T, self.H, self.W = T, H, W
        t_idx = torch.arange(T)
        h_idx = torch.arange(H)
        w_idx = torch.arange(W)
        self.register_buffer("t_idx", t_idx)
        self.register_buffer("h_idx", h_idx)
        self.register_buffer("w_idx", w_idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W, in_channels)
        h = self.input_proj(x)

        # Additive factorized positional encoding
        pt = self.pos_T(self.t_idx).view(self.T, 1, 1, -1)
        ph = self.pos_H(self.h_idx).view(1, self.H, 1, -1)
        pw = self.pos_W(self.w_idx).view(1, 1, self.W, -1)
        h = h + pt + ph + pw

        for blk in self.blocks:
            h = blk(h)

        return self.output_proj(h)


def build_factformer() -> nn.Module:
    return FactFormer(
        in_channels=2, out_channels=1, d_model=64, n_heads=4, n_layers=2, T=4, H=8, W=8
    )


def example_input_factformer() -> torch.Tensor:
    # (B=1, T=4, H=8, W=8, in_channels=2): vorticity + forcing field
    return torch.randn(1, 4, 8, 8, 2)


MENAGERIE_ENTRIES = [
    ("FactFormer", "build_factformer", "example_input_factformer", "2023", "DC"),
]
