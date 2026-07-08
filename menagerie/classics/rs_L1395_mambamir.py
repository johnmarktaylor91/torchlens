# FAITHFUL REIMPLEMENTATION from Huang et al., "MambaMIR: An Arbitrary-Masked Mamba for
# Joint Medical Image Reconstruction and Uncertainty Estimation" (MICCAI 2024,
# arXiv:2402.18451v3, https://arxiv.org/html/2402.18451v3) -- no public code.
#
# The paper states the code is at https://github.com/ayanglab/MambaMIR, but that repository
# no longer exists (returns HTTP 404 via both the GitHub website and the REST/GraphQL APIs;
# the `ayanglab` org is active with 28 other public repos, none named MambaMIR/MambaMIR-*, as
# of this writing). The candidate queue's duplicate row "MambaMRI" points to the same
# (nonexistent) repo/paper and is therefore a dedup of this same reimplementation, not a
# distinct architecture.
#
# Every mechanism below is transcribed directly from Section 2 ("Methodology") of the paper,
# which is unusually explicit about block-level structure:
#
#  - Sec 2.3.1 "Overall Architecture" (Fig. 2A): Input Module (stem, pixel space R^{h x w x c}
#    -> latent patch space R^{H x W x C}) -> N cascaded "AMSS Block Groups" (each: a conv
#    layer + a LayerNorm + M "AMSS Blocks", itself residual) -> Output Module (inverse map
#    back to pixel space), with a residual connection around the whole network.
#  - Sec 2.3.2 "AMSS Block" (Fig. 2B): LayerNorm -> split into two pathways. Primary path:
#    gating linear layer -> depth-wise 3x3 conv -> SiLU -> AMS6 Block -> LayerNorm.
#    Secondary path: linear layer -> SiLU. Merge the two paths by elementwise multiplication,
#    then a final gating linear layer produces the block output. (The paper explicitly notes
#    AMSS discards the "-Norm-MLP" tail of a standard ViT block "for a lighter network size".)
#  - Sec 2.2 "Arbitrary-Masked S6 Block" (Fig. 1, Fig. 2C): four sequential sub-modules.
#      (1) Scan Expanding Module: unfolds the H x W patch grid into 4 ordered 1-D sequences
#          -- row-major from top-left, row-major from bottom-right, column-major from
#          top-left, column-major from bottom-right (the "4x expansion" the paper describes,
#          matching VMamba's SS2M four-directional scan).
#      (2) Arbitrary-Masked Module: zeroes out a randomly-chosen (during training AND
#          inference, per the paper's Monte-Carlo-dropout framing for uncertainty estimation)
#          subset of the 4 scans, keeping the tensor's shape unchanged.
#      (3) S6 Module: the standard Mamba selective-scan SSM (Sec 2.1.2/2.1.3, Eq. 3-4) applied
#          independently to each surviving scan direction -- input-dependent projections to
#          Delta/B/C, zero-order-hold discretisation A_bar = exp(Delta*A),
#          B_bar ~= Delta*B, and the sequential recurrence h_k = A_bar h_{k-1} + B_bar x_k,
#          y_k = C h_k + D x_k. Implemented here as an explicit pure-torch sequential scan
#          (mathematically identical to, but not calling, the `mamba-ssm` CUDA kernel, which
#          is not part of the installed base env).
#      (4) Scan Merging Module: reverses each scan's ordering back to the canonical row-major
#          patch order, sums the (unmasked) directions, and reshapes back to the H x W grid.
#
# This is a from-scratch reimplementation of a *published, block-diagrammed* architecture,
# not a loose gist: every named sub-module in Fig. 1/Fig. 2 is present and wired in the paper's
# stated order. No mamba-ssm / causal-conv1d dependency is introduced (S6 recurrence is a
# plain torch loop over the sequence length), so the module runs in the installed base env.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "reimpl-pytorch"


class S6Module(nn.Module):
    """Selective State Space Model (S6) core of Mamba (Gu & Dao 2023), Sec 2.1.2-2.1.3.

    Pure-torch sequential recurrence over the discretised SSM (Eq. 4 in the paper):
        A_bar = exp(Delta * A), B_bar ~= Delta * B
        h_k = A_bar h_{k-1} + B_bar x_k,  y_k = C h_k + D x_k
    with Delta, B, C derived from the input (input-dependent / "selective").
    """

    def __init__(self, d_model, d_state=8, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(1, d_model // 8)

        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # A is a learned, per-channel negative-real diagonal (HiPPO-style init, log-parameterised)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # x: (B, L, d_model)
        b, seq_len, d = x.shape
        n = self.d_state

        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*n)
        delta, b_param, c_param = torch.split(x_dbl, [self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (B, L, d_model)

        a = -torch.exp(self.A_log)  # (d_model, n)

        # discretise: A_bar = exp(delta * A); B_bar ~= delta * B
        delta_a = torch.exp(delta.unsqueeze(-1) * a.unsqueeze(0).unsqueeze(0))  # (B, L, d, n)
        delta_b_x = delta.unsqueeze(-1) * b_param.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, d, n)

        h = torch.zeros(b, d, n, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(seq_len):
            h = delta_a[:, t] * h + delta_b_x[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, c_param[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, L, d)
        return y + x * self.D


class ScanExpandMerge:
    """Scan Expanding Module + Scan Merging Module (Sec 2.2, Fig. 1).

    4 directional flattenings of an (H, W) patch grid: row-major top-left, row-major
    bottom-right (both rows and columns reversed), column-major top-left (transpose then
    row-major), column-major bottom-right. Merging reverses each ordering and sums.
    """

    @staticmethod
    def expand(x):
        # x: (B, C, H, W) -> list of 4 (B, C, H*W) sequences
        b, c, h, w = x.shape
        scan0 = x.flatten(2)  # row-major, top-left -> bottom-right
        scan1 = x.flip(2).flip(3).flatten(2)  # row-major, bottom-right -> top-left
        scan2 = x.transpose(2, 3).flatten(2)  # column-major, top-left -> bottom-right
        scan3 = (
            x.transpose(2, 3).flip(2).flip(3).flatten(2)
        )  # column-major, bottom-right -> top-left
        return [scan0, scan1, scan2, scan3], (h, w)

    @staticmethod
    def merge(scans, hw):
        h, w = hw
        b, c, _ = scans[0].shape
        merged = scans[0].reshape(b, c, h, w)
        merged = merged + scans[1].reshape(b, c, h, w).flip(3).flip(2)
        merged = merged + scans[2].reshape(b, c, w, h).transpose(2, 3)
        merged = merged + scans[3].reshape(b, c, w, h).flip(3).flip(2).transpose(2, 3)
        return merged


class AMS6Block(nn.Module):
    """Arbitrary-Masked S6 Block (Sec 2.2, Fig. 1): Scan Expanding -> Arbitrary-Masked ->
    S6 (shared across the 4 directions) -> Scan Merging."""

    def __init__(self, d_model, d_state=8, n_mask=1):
        super().__init__()
        self.n_mask = n_mask  # number of the 4 scan directions arbitrarily zeroed each pass
        self.s6 = S6Module(d_model, d_state=d_state)

    def forward(self, x):
        # x: (B, C, H, W)
        scans, hw = ScanExpandMerge.expand(x)

        # Arbitrary-Masked Module: zero out `n_mask` of the 4 scans (every forward pass,
        # train and eval, matching the paper's Monte-Carlo-dropout-style uncertainty framing).
        keep = torch.ones(4, device=x.device)
        if self.n_mask > 0:
            drop_idx = torch.randperm(4, device=x.device)[: self.n_mask]
            keep[drop_idx] = 0.0

        processed = []
        for i, scan in enumerate(scans):
            seq = scan.transpose(1, 2)  # (B, L, C)
            out = self.s6(seq) * keep[i]
            processed.append(out.transpose(1, 2))  # back to (B, C, L)

        return ScanExpandMerge.merge(processed, hw)


class AMSSBlock(nn.Module):
    """AMSS Block (Sec 2.3.2, Fig. 2B): LN -> split -> [gate-linear, dwconv3x3, SiLU,
    AMS6, LN] (x) [linear, SiLU] -> multiply -> gate-linear."""

    def __init__(self, dim, d_state=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_gate = nn.Linear(dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act1 = nn.SiLU()
        self.ams6 = AMS6Block(dim, d_state=d_state)
        self.norm2 = nn.LayerNorm(dim)

        self.secondary_linear = nn.Linear(dim, dim)
        self.act2 = nn.SiLU()

        self.out_gate = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, h, w = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x_norm = self.norm(x_seq)

        # primary pathway
        primary = self.in_gate(x_norm)
        primary_img = primary.transpose(1, 2).reshape(b, c, h, w)
        primary_img = self.act1(self.dwconv(primary_img))
        primary_img = self.ams6(primary_img)
        primary = primary_img.flatten(2).transpose(1, 2)
        primary = self.norm2(primary)

        # secondary pathway
        secondary = self.act2(self.secondary_linear(x_norm))

        merged = primary * secondary
        out = self.out_gate(merged)
        out_img = out.transpose(1, 2).reshape(b, c, h, w)
        return x + out_img


class AMSSBlockGroup(nn.Module):
    """AMSS Block Group (Sec 2.3.1): conv layer + LayerNorm + M AMSS Blocks, residual."""

    def __init__(self, dim, n_blocks=2, d_state=8):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([AMSSBlock(dim, d_state=d_state) for _ in range(n_blocks)])

    def forward(self, x):
        residual = x
        out = self.conv(x)
        b, c, h, w = out.shape
        out_seq = self.norm(out.flatten(2).transpose(1, 2))
        out = out_seq.transpose(1, 2).reshape(b, c, h, w)
        for blk in self.blocks:
            out = blk(out)
        return residual + out


class MambaMIR(nn.Module):
    """MambaMIR (Sec 2.3.1, Fig. 2A): Input Module (stem) -> N AMSS Block Groups ->
    Output Module, with a global residual connection around the whole network."""

    def __init__(self, in_ch=1, dim=16, n_groups=2, n_blocks_per_group=1, d_state=8):
        super().__init__()
        self.input_module = nn.Conv2d(in_ch, dim, kernel_size=3, padding=1)
        self.groups = nn.ModuleList(
            [
                AMSSBlockGroup(dim, n_blocks=n_blocks_per_group, d_state=d_state)
                for _ in range(n_groups)
            ]
        )
        self.output_module = nn.Conv2d(dim, in_ch, kernel_size=3, padding=1)

    def forward(self, x):
        feat = self.input_module(x)
        out = feat
        for group in self.groups:
            out = group(out)
        out = self.output_module(out)
        return x + out  # global residual (Sec 2.3.1: "Residual connection is applied ... for
        # both the whole MambaMIR and the cascaded AMSS Blocks")


def build_mambamir():
    return MambaMIR(in_ch=1, dim=8, n_groups=2, n_blocks_per_group=1, d_state=4)


def example_input_mambamir():
    # Kept small (8x8 = 64 tokens/scan): the S6 Module's selective-scan recurrence is an
    # explicit sequential loop over the token sequence (see S6Module.forward), so trace/op
    # count scale directly with H*W; this is plenty to exercise every named sub-module
    # (Scan Expanding/Merging, Arbitrary-Masked, S6, AMSS gating) end to end.
    return torch.randn(1, 1, 8, 8)


MENAGERIE_ENTRIES = [
    ("MambaMIR", "build_mambamir", "example_input_mambamir", "2024", "reimpl-pytorch"),
]
