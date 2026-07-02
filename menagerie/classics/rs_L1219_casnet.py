# SOURCE: vendored from Guaishou74851/CASNet @ main (model_test.py + utils.py, IEEE TIP 2022)
# https://github.com/Guaishou74851/CASNet
#
# CASNet: content-aware scalable image compressive sensing (CS) network with saliency-guided
# residual sampling and deep collaborative block-based reconstruction. Vendored verbatim from
# the official repo's `model_test.py` (module classes) and `utils.py` (the `H` orthogonal
# transform helper it imports), with only the import path adjusted (`from utils import H`
# stays local to this file since `H` is copied in below rather than imported from a sibling
# module). No architectural changes.
#
# ruff: noqa: E731, E741 -- vendored source uses lambda-assignment and single-letter names
# (l, P, D, H); preserved verbatim rather than renamed, matching the classics/ precedent for
# vendored/ported architecture files (see pyproject.toml per-file-ignores for menagerie/classics).

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# from utils.py (Guaishou74851/CASNet @ main) -- orthogonal transform helper
# ---------------------------------------------------------------------------
def H(img, mode, inv=False):
    if inv:
        mode = [0, 1, 2, 5, 4, 3, 6, 7][mode]
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])


# ---------------------------------------------------------------------------
# from model_test.py (Guaishou74851/CASNet @ main)
# ---------------------------------------------------------------------------
# classic residual block
class RB(nn.Module):
    def __init__(self, nf, bias, kz=3):
        super(RB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias),
        )

    def forward(self, x):
        return x + self.body(x)


# proximal mapping network (https://github.com/cszn/DPIR)
class P(nn.Module):
    def __init__(self, in_nf, out_nf):
        super(P, self).__init__()
        bias, block, nb, scale_factor = False, RB, 2, 2
        mid_nf = [16, 32, 64, 128]

        conv = lambda in_nf, out_nf: nn.Conv2d(in_nf, out_nf, 3, padding=1, bias=bias)
        up = lambda nf, scale_factor: nn.ConvTranspose2d(
            nf, nf, scale_factor, stride=scale_factor, bias=bias
        )
        down = lambda nf, scale_factor: nn.Conv2d(
            nf, nf, scale_factor, stride=scale_factor, bias=bias
        )

        self.down1 = nn.Sequential(
            conv(in_nf, mid_nf[0]),
            *[block(mid_nf[0], bias) for _ in range(nb)],
            down(mid_nf[0], scale_factor),
        )
        self.down2 = nn.Sequential(
            conv(mid_nf[0], mid_nf[1]),
            *[block(mid_nf[1], bias) for _ in range(nb)],
            down(mid_nf[1], scale_factor),
        )
        self.down3 = nn.Sequential(
            conv(mid_nf[1], mid_nf[2]),
            *[block(mid_nf[2], bias) for _ in range(nb)],
            down(mid_nf[2], scale_factor),
        )

        self.body = nn.Sequential(
            conv(mid_nf[2], mid_nf[3]),
            *[block(mid_nf[3], bias) for _ in range(nb)],
            conv(mid_nf[3], mid_nf[2]),
        )

        self.up3 = nn.Sequential(
            up(mid_nf[2], scale_factor),
            *[block(mid_nf[2], bias) for _ in range(nb)],
            conv(mid_nf[2], mid_nf[1]),
        )
        self.up2 = nn.Sequential(
            up(mid_nf[1], scale_factor),
            *[block(mid_nf[1], bias) for _ in range(nb)],
            conv(mid_nf[1], mid_nf[0]),
        )
        self.up1 = nn.Sequential(
            up(mid_nf[0], scale_factor),
            *[block(mid_nf[0], bias) for _ in range(nb)],
            conv(mid_nf[0], out_nf),
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.body(x3)
        x = self.up3(x + x3)  # three skip connections for the last three scales
        x = self.up2(x + x2)
        x = self.up1(x + x1)
        return x


class Phase(nn.Module):
    def __init__(self, img_nf, B):
        super(Phase, self).__init__()
        bias, nf, nb, onf = True, 8, 3, 3  # config of E

        self.rho = nn.Parameter(torch.Tensor([0.5]))
        self.P = P(img_nf + onf, img_nf)  # input: [Z | saliency_feature]
        self.B = B  # default: 32
        self.E = nn.Sequential(  # saliency feature extractor
            nn.Conv2d(1, nf, 1, bias=bias),
            *[RB(nf, bias, kz=1) for _ in range(nb)],
            nn.Conv2d(nf, onf, 1, bias=bias),
        )

    def forward(self, x, cs_ratio_map, PhiT_Phi, PhiT_y, mode, shape_info):
        b, l, h, w = shape_info

        # block gradient descent
        x = x - self.rho * (PhiT_Phi.matmul(x) - PhiT_y)

        # saliency information guided proximal mapping (with RTE strategy)
        x = x.reshape(b, l, -1).permute(0, 2, 1)
        x = F.fold(x, output_size=(h, w), kernel_size=self.B, stride=self.B)
        x_rotated = H(x, mode)
        cs_ratio_map_rotated = H(cs_ratio_map, mode)
        saliency_feature = self.E(cs_ratio_map_rotated)
        x_rotated = x_rotated + self.P(torch.cat([x_rotated, saliency_feature], dim=1))
        return H(x_rotated, mode, inv=True)  # inverse of H


# saliency detector
class D(nn.Module):
    def __init__(self, img_nf):
        super(D, self).__init__()
        bias, block, nb, mid_nf = False, RB, 3, 32
        conv = lambda in_nf, out_nf: nn.Conv2d(in_nf, out_nf, 3, padding=1, bias=bias)
        self.body = nn.Sequential(
            conv(img_nf, mid_nf), *[block(mid_nf, bias) for _ in range(nb)], conv(mid_nf, 1)
        )

    def forward(self, x):
        return self.body(x).reshape(*x.shape[:2], -1).softmax(dim=2).reshape_as(x)


# error correction of BRA
def batch_correct(Q, target_sum, N):
    b, l = Q.shape
    i, max_desc_step = 0, 10
    while True:
        i += 1
        Q = torch.clamp(Q, 0, N).round()
        d = Q.sum(dim=1) - target_sum  # batch delta
        if float(d.abs().sum()) == 0.0:
            break
        elif i < max_desc_step:  # 1: uniform descent
            Q = Q - (d / l).reshape(-1, 1).expand_as(Q)
        else:  # 2: random allocation
            for j in range(b):
                D_alloc = np.random.multinomial(int(d[j].abs().ceil()), [1.0 / l] * l, size=1)
                Q[j] -= int(d[j].sign()) * torch.Tensor(D_alloc).squeeze(0).to(Q.device)
    return Q


class CASNet(nn.Module):
    def __init__(self, phase_num, B, img_nf, Phi_init):
        super(CASNet, self).__init__()
        self.phase_num = phase_num
        self.phase_num_minus_1 = phase_num - 1
        self.B = B
        self.N = B * B
        self.Phi = nn.Parameter(Phi_init.reshape(self.N, self.N))
        self.RS = nn.ModuleList([Phase(img_nf, B) for _ in range(phase_num)])
        self.D = D(img_nf)
        self.index_mask = torch.arange(1, self.N + 1)

    def forward(self, x, q, modes, gamma=0.2822):
        b, c, h, w = x.shape

        x_unfold = F.unfold(x, kernel_size=self.B, stride=self.B).permute(
            0, 2, 1
        )  # shape: (b, l, img_nf * B * B)
        l = x_unfold.shape[1]  # block number of an image patch
        block_stack = x_unfold.reshape(-1, c * self.N, 1)  # shape: (b * l, img_nf * B * B, 1)
        block_volume = block_stack.shape[1]  # img_nf * B * B
        L = block_stack.shape[0]  # total block number of batch
        Phi_stack = self.Phi.unsqueeze(0).repeat(L, 1, 1)
        index_mask = self.index_mask.unsqueeze(0).repeat(L, 1).to(Phi_stack.device)

        # 1. basic uniform sampling
        q_basic = int(np.ceil(gamma * q))
        Phi_basic = Phi_stack.clone()
        Phi_basic[index_mask > q_basic] = 0
        PhiT_Phi_basic = Phi_basic.permute(0, 2, 1).matmul(Phi_basic)
        PhiT_y_basic = PhiT_Phi_basic.matmul(block_stack).reshape(b, l, -1).permute(0, 2, 1)
        PhiT_y_basic = F.fold(PhiT_y_basic, output_size=(h, w), kernel_size=self.B, stride=self.B)

        # 2. adaptive CS ratio allocation
        S = self.D(PhiT_y_basic)  # saliency map
        Q = (q - q_basic) * l * S  # measurement size map
        Q_unfold = (
            F.unfold(Q, kernel_size=self.B, stride=self.B).permute(0, 2, 1).sum(dim=2)
        )  # sumpooling, shape: (b, l)
        Q_unfold = (
            batch_correct(Q_unfold, (q - q_basic) * l, self.N - q_basic) + q_basic
        )  # error correction

        # 3. content-aware residual sampling
        q_stack = Q_unfold.reshape(-1, 1).repeat(1, Phi_stack.shape[1])
        Phi_stack[index_mask > q_stack] = 0
        Phi_stack[index_mask <= q_basic] = 0
        PhiT_Phi = Phi_stack.permute(0, 2, 1).matmul(Phi_stack)
        PhiT_y = PhiT_Phi.matmul(block_stack)

        # 4. deep collaborative reconstruction
        PhiT_Phi += PhiT_Phi_basic
        PhiT_y += (
            F.unfold(PhiT_y_basic, kernel_size=self.B, stride=self.B)
            .permute(0, 2, 1)
            .reshape(L, -1, 1)
        )
        x = PhiT_y
        cs_ratio_map = (
            (Q_unfold / self.N).unsqueeze(2).repeat(1, 1, block_volume).permute(0, 2, 1)
        )  # get expanded CS ratio map R'
        cs_ratio_map = F.fold(cs_ratio_map, output_size=(h, w), kernel_size=self.B, stride=self.B)
        shape_info = [b, l, h, w]
        for i in range(self.phase_num):  # recover step-by-step
            x = self.RS[i](x, cs_ratio_map, PhiT_Phi, PhiT_y, modes[i], shape_info)
            if i < self.phase_num_minus_1:
                x = F.unfold(x, kernel_size=self.B, stride=self.B).permute(0, 2, 1)
                x = x.reshape(L, -1, 1)
        return x


# ---------------------------------------------------------------------------
# menagerie staging entry point
# ---------------------------------------------------------------------------
# tiny config: 3 CS reconstruction phases (repo default is 13), 16x16 blocks
_PHASE_NUM, _B, _IMG_NF = 3, 16, 1


def build_casnet():
    N = _B * _B
    Phi_init = torch.zeros(N, N)
    return CASNet(_PHASE_NUM, _B, _IMG_NF, Phi_init)


def example_input_casnet():
    # matches `test.py`'s real call: model(x_input, q, rand_modes)
    # image must be H,W multiples of block_size B; tiny 32x32 single-channel image
    N = _B * _B
    x = torch.rand(1, 1, 32, 32)
    q = int(math.ceil(0.10 * N))  # cs_ratio=0.10, matches repo's example ratios
    modes = [0] * _PHASE_NUM  # fixed transform mode per phase (repo randomizes at test time)
    return x, q, modes


MENAGERIE_ENTRIES = [
    ("CASNet", "build_casnet", "example_input_casnet", 2022, "SOURCE_AVAILABLE"),
]
