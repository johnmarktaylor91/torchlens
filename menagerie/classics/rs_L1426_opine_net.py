# SOURCE: vendored from jianzhangcs/OPINE-Net @ 60c06bf6810cc84497054a630e3ada2674ff3bd2
# (TEST_CS_OPINE_Net_plus.py)
#
# OPINE-Net+ (Orthogonal-constrained Projection-based Iterative Network with a
# learned sampling subnet) -- a deep unrolled optimization network for image
# compressive sensing (CS). Jointly learns a binarized measurement matrix `Phi`
# (sampling-subnet, via a straight-through-estimator sign function `MySign`) and
# an iterative reconstruction network (`BasicBlock`, one ISTA-style proximal-
# gradient phase per unrolled iteration, each with a learned analysis/synthesis
# conv-transform pair and soft-thresholding). `BasicBlock`/`OPINENetplus`/
# `MySign`/`PhiTPhi_fun` below are the REAL model classes from
# `TEST_CS_OPINE_Net_plus.py`, copied verbatim (only base-lib deps used by the
# model itself: torch, torch.nn, torch.nn.functional, torch.nn.init -- the
# script-level imports scipy.io/cv2/skimage/argparse are data-loading/CLI-only
# and are dropped here, not part of the model architecture).
#
# Dropped: everything below `model = OPINENetplus(...)` in the original file
# (training/eval loop, CLI argparse, GPU env vars) -- script plumbing, not model.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

MENAGERIE_ZOO = "vendored-pytorch"

# Tiny-init constants (originals: n_output=1089 fixed 33x33 block; n_input from
# a cs_ratio -> measurement-count lookup table in the source script).
_BLOCK = 33
_N_OUTPUT = _BLOCK * _BLOCK  # 1089
_N_INPUT = 10  # cs_ratio=1 measurement count from the source ratio_dict
_LAYER_NO = 2  # tiny phase count (source default: 9)


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


MyBinarize = MySign.apply


def PhiTPhi_fun(x, PhiW, PhiTW):
    temp = F.conv2d(x, PhiW, padding=0, stride=_BLOCK, bias=None)
    temp = F.conv2d(temp, PhiTW, padding=0, bias=None)
    return torch.nn.PixelShuffle(_BLOCK)(temp)


# Define OPINE-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv_D = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))

        self.conv1_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_G = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv3_G = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiWeight, PhiTWeight, PhiTb):
        x = x - self.lambda_step * PhiTPhi_fun(x, PhiWeight, PhiTWeight)
        x = x + self.lambda_step * PhiTb
        x_input = x

        x_D = F.conv2d(x_input, self.conv_D, padding=1)

        x = F.conv2d(x_D, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x = F.conv2d(F.relu(x_backward), self.conv1_G, padding=1)
        x = F.conv2d(F.relu(x), self.conv2_G, padding=1)
        x_G = F.conv2d(x, self.conv3_G, padding=1)

        x_pred = x_input + x_G

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_D_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_D_est - x_D

        return [x_pred, symloss]


# Define OPINE-Net-plus
class OPINENetplus(torch.nn.Module):
    def __init__(self, LayerNo, n_input):
        super(OPINENetplus, self).__init__()

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, _N_OUTPUT)))
        self.Phi_scale = nn.Parameter(torch.Tensor([0.01]))

        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x):
        n_input = self.Phi.shape[0]

        # Sampling-subnet
        Phi_ = MyBinarize(self.Phi)
        Phi = self.Phi_scale * Phi_
        PhiWeight = Phi.contiguous().view(n_input, 1, _BLOCK, _BLOCK)
        Phix = F.conv2d(x, PhiWeight, padding=0, stride=_BLOCK, bias=None)  # Get measurements

        # Initialization-subnet
        PhiTWeight = Phi.t().contiguous().view(_N_OUTPUT, n_input, 1, 1)
        PhiTb = F.conv2d(Phix, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(_BLOCK)(PhiTb)
        x = PhiTb  # Conduct initialization

        # Recovery-subnet
        layers_sym = []  # for computing symmetric loss
        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiWeight, PhiTWeight, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym, Phi]


def build_opine_net():
    return OPINENetplus(_LAYER_NO, _N_INPUT)


def example_input_opine_net():
    return torch.randn(1, 1, _BLOCK, _BLOCK)


MENAGERIE_ENTRIES = [
    (
        "OPINE-Net+",
        "build_opine_net",
        "example_input_opine_net",
        2020,
        "vendored-pytorch",
    ),
]
