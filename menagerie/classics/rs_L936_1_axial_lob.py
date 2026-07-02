# SOURCE: vendored from LeonardoBerti00/Axial-LOB-High-Frequency-Trading-with-Axial-Attention @ main
#   (repo: https://github.com/LeonardoBerti00/Axial-LOB-High-Frequency-Trading-with-Axial-Attention)
#   AxialLOB_train_test.ipynb, model-definition cell (`GatedAxialAttention` +
#   `AxialLOB`), copied verbatim. This is the only PyTorch implementation in
#   the repo -- there is no separate `.py` model file, the whole project is a
#   single training notebook -- so the model cell is vendored as-is (only the
#   surrounding notebook/training/data-loading code is dropped).
#
# Axial-LOB (Kisiel & Gorse, "Axial-LOB: High-Frequency Trading with Axial
# Attention", 2022) predicts limit-order-book mid-price movement direction
# (down/stationary/up) from a window of raw LOB feature snapshots. A 1x1
# Conv2d lifts the single-channel LOB window to `c_in` channels, then two
# `GatedAxialAttention` blocks (the gated-axial-attention mechanism from
# Medical-Transformer's axialnet.py, explicitly credited in the source
# comment) apply self-attention along the width and height axes in sequence
# with learned relative positional embeddings and four learnable gating
# scalars (`f_qr`, `f_kr`, `f_sve`, `f_sv`) -- this is the actual novel
# architectural contribution (gated + factorized axial attention wired into
# a 2-stage residual CNN stack for LOB data), so this is a real
# architecture, not a bare pretrained-backbone reuse -> rung 2 (vendor),
# not rung 1. Two residual branches (1x1 Conv2d + BatchNorm2d) run in
# parallel with the axial-attention branches and are summed back in; a
# final 1x1 Conv2d + BatchNorm2d + AvgPool2d + Linear + softmax produces the
# 3-class direction forecast. Code copied verbatim from the notebook cell;
# only the module-level import statements were consolidated at the top of
# this file.

import math

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


def _conv1d1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm1d(out_channels),
    )


# class taken from https://github.com/jeya-maria-jose/Medical-Transformer/blob/main/lib/models/axialnet.py
class GatedAxialAttention(nn.Module):
    def __init__(self, in_channels, out_channels, heads, dim, flag):
        assert (in_channels % heads == 0) and (out_channels % heads == 0)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dim_head_v = out_channels // heads
        self.flag = flag  # if flag then we do the attention along width
        self.dim = dim
        self.dim_head_qk = self.dim_head_v // 2
        self.qkv_channels = self.dim_head_v + self.dim_head_qk * 2

        # Multi-head self attention
        self.to_qkv = _conv1d1x1(in_channels, self.heads * self.qkv_channels)
        self.bn_qkv = nn.BatchNorm1d(self.heads * self.qkv_channels)
        self.bn_similarity = nn.BatchNorm2d(heads * 3)
        self.bn_output = nn.BatchNorm1d(self.heads * self.qkv_channels)

        # Gating mechanism
        self.f_qr = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.3), requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(0.5), requires_grad=False)

        # Position embedding
        self.relative = nn.Parameter(
            torch.randn(self.dim_head_v * 2, dim * 2 - 1), requires_grad=True
        )
        query_index = torch.arange(dim).unsqueeze(0)
        key_index = torch.arange(dim).unsqueeze(1)
        relative_index = key_index - query_index + dim - 1
        self.register_buffer("flatten_index", relative_index.view(-1))

        self.reset_parameters()

    def forward(self, x):
        if self.flag:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        x = self.to_qkv(x)

        qkv = self.bn_qkv(x)
        q, k, v = torch.split(
            qkv.reshape(N * W, self.heads, self.dim_head_v * 2, H),
            [self.dim_head_v // 2, self.dim_head_v // 2, self.dim_head_v],
            dim=2,
        )

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(
            self.dim_head_v * 2, self.dim, self.dim
        )
        q_embedding, k_embedding, v_embedding = torch.split(
            all_embeddings, [self.dim_head_qk, self.dim_head_qk, self.dim_head_v], dim=0
        )
        qr = torch.einsum("bgci,cij->bgij", q, q_embedding)
        kr = torch.einsum("bgci,cij->bgij", k, k_embedding).transpose(2, 3)
        qk = torch.einsum("bgci, bgcj->bgij", q, k)

        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = (
            self.bn_similarity(stacked_similarity).view(N * W, 3, self.heads, H, H).sum(dim=1)
        )
        # (N, heads, H, H, W)
        similarity = torch.softmax(stacked_similarity, dim=3)
        sv = torch.einsum("bgij,bgcj->bgci", similarity, v)
        sve = torch.einsum("bgij,cij->bgci", similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_channels * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_channels, 2, H).sum(dim=-2)

        if self.flag:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        return output

    def reset_parameters(self):
        nn.init.normal_(self.relative, 0.0, math.sqrt(1.0 / self.dim_head_v))


class AxialLOB(nn.Module):
    def __init__(self, W, H, c_in, c_out, c_final, n_heads, pool_kernel, pool_stride):
        super().__init__()

        """
        Args:
          W and H:  the width and height of the input tensors
          c_in, c_out, and c_final:  the number of channels for the input, intermediate, and final convolutional layers
          n_heads:  the number of heads for the multi-head attention mechanism used in the GatedAxialAttention layers.
          pool_kernel and pool_stride:  the kernel size and stride of the final average pooling layer.
        """

        # channel output of the CNN_in is the channel input for the axial layer
        self.c_in = c_in
        self.c_out = c_out
        self.c_final = c_final

        self.CNN_in = nn.Conv2d(in_channels=1, out_channels=c_in, kernel_size=1)
        self.CNN_out = nn.Conv2d(in_channels=c_out, out_channels=c_final, kernel_size=1)
        self.CNN_res2 = nn.Conv2d(in_channels=c_out, out_channels=c_final, kernel_size=1)
        self.CNN_res1 = nn.Conv2d(in_channels=1, out_channels=c_out, kernel_size=1)

        self.norm = nn.BatchNorm2d(c_in)
        self.res_norm2 = nn.BatchNorm2d(c_final)
        self.res_norm1 = nn.BatchNorm2d(c_out)
        self.norm2 = nn.BatchNorm2d(c_final)
        self.axial_height_1 = GatedAxialAttention(c_out, c_out, n_heads, H, flag=False)
        self.axial_width_1 = GatedAxialAttention(c_out, c_out, n_heads, W, flag=True)
        self.axial_height_2 = GatedAxialAttention(c_out, c_out, n_heads, H, flag=False)
        self.axial_width_2 = GatedAxialAttention(c_out, c_out, n_heads, W, flag=True)

        self.activation = nn.ReLU()
        self.linear = nn.Linear(1600, 3)
        self.pooling = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x):
        # up branch
        # first convolution before the attention
        y = self.CNN_in(x)
        y = self.norm(y)
        y = self.activation(y)

        # attention mechanism through gated multi head axial layer
        y = self.axial_width_1(y)
        y = self.axial_height_1(y)

        # lower branch
        x = self.CNN_res1(x)
        x = self.res_norm1(x)
        x = self.activation(x)

        # first residual
        y = y + x
        z = y.detach().clone()

        # second axial layer
        y = self.axial_width_2(y)
        y = self.axial_height_2(y)

        # second convolution
        y = self.CNN_out(y)
        y = self.res_norm2(y)
        y = self.activation(y)

        # lower branch
        z = self.CNN_res2(z)
        z = self.norm2(z)
        z = self.activation(z)

        # second res connection
        y = y + z

        # final part
        y = self.pooling(y)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        forecast_y = torch.softmax(y, dim=1)
        return forecast_y


# ---------------------------------------------------------------------------
# Menagerie staging entry points
# ---------------------------------------------------------------------------
# The real notebook hyperparameters (cells 5 and 9): W=40, dim(H)=40,
# c_in_axial=32, c_out_axial=32, c_final=4, n_heads=4, pool_kernel=(1,4),
# pool_stride=(1,4). The hardcoded `nn.Linear(1600, 3)` in AxialLOB.__init__
# only matches this exact (W, H, c_final, pool) combination
# (c_final * H * (W // pool_stride[1]) == 4*40*10 == 1600), so these
# hyperparameters are kept exactly as in the original notebook rather than
# shrunk for a "tiny" staging config.


def build_axial_lob():
    return AxialLOB(
        W=40,
        H=40,
        c_in=32,
        c_out=32,
        c_final=4,
        n_heads=4,
        pool_kernel=(1, 4),
        pool_stride=(1, 4),
    )


def example_input_axial_lob():
    """A single-channel (N, 1, H=40, W=40) LOB feature window, matching the
    real `Dataset.__getitem__` -> DataLoader tensor shape fed to
    `AxialLOB.forward` in the source notebook (dim=40 LOB states x W=40
    features, batch size 1 here)."""
    torch.manual_seed(0)
    return (torch.randn(1, 1, 40, 40),)


MENAGERIE_ENTRIES = [
    (
        "Axial-LOB",
        build_axial_lob,
        example_input_axial_lob,
        2022,
        "CODE",
    ),
]
