# SOURCE: vendored from https://github.com/jianzhangcs/ISTA-Net-PyTorch @ master (f6990406)
# (Train_CS_ISTA_Net.py :: BasicBlock, ISTANet)
#
# ISTA-Net: Interpretable Optimization-Inspired Deep Network for Image Compressive
# Sensing (Zhang & Ghanem, CVPR 2018). https://github.com/jianzhangcs/ISTA-Net-PyTorch
#
# The classes below are the REAL ISTA-Net model code: each `BasicBlock` unrolls one
# iteration of ISTA (a gradient-descent step against the CS measurement matrix Phi,
# followed by a learned analysis/soft-threshold/synthesis conv triple standing in for
# the sparsifying transform), and `ISTANet` stacks `LayerNo` such blocks. Only minimal,
# purely mechanical changes were made to lift the classes out of the training script:
# the argparse/data-loading/training-loop scaffolding was dropped (irrelevant to the
# model definition) and CUDA device placement was removed. No architecture was altered.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

MENAGERIE_ZOO = "vendored-pytorch"


# Define ISTA-Net Block
class BasicBlock(torch.nn.Module):
    def __init__(self):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))
        self.soft_thr = nn.Parameter(torch.Tensor([0.01]))

        self.conv1_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 1, 3, 3)))
        self.conv2_forward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv1_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(32, 32, 3, 3)))
        self.conv2_backward = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 32, 3, 3)))

    def forward(self, x, PhiTPhi, PhiTb):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, 33, 33)

        x = F.conv2d(x_input, self.conv1_forward, padding=1)
        x = F.relu(x)
        x_forward = F.conv2d(x, self.conv2_forward, padding=1)

        x = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.soft_thr))

        x = F.conv2d(x, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_backward = F.conv2d(x, self.conv2_backward, padding=1)

        x_pred = x_backward.view(-1, 1089)

        x = F.conv2d(x_forward, self.conv1_backward, padding=1)
        x = F.relu(x)
        x_est = F.conv2d(x, self.conv2_backward, padding=1)
        symloss = x_est - x_input

        return [x_pred, symloss]


# Define ISTA-Net
class ISTANet(torch.nn.Module):
    def __init__(self, LayerNo):
        super(ISTANet, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        for i in range(LayerNo):
            onelayer.append(BasicBlock())

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, Phix, Phi, Qinit):
        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)

        x = torch.mm(Phix, torch.transpose(Qinit, 0, 1))

        layers_sym = []  # for computing symmetric loss

        for i in range(self.LayerNo):
            [x, layer_sym] = self.fcs[i](x, PhiTPhi, PhiTb)
            layers_sym.append(layer_sym)

        x_final = x

        return [x_final, layers_sym]


# ---------------------------------------------------------------------------
# Tiny random-init build/example for TorchLens tracing.
#
# The real repo fixes n_output=1089 (33x33 image blocks) and derives n_input (the
# CS measurement dim) from a {1,4,10,25,30,40,50}% ratio_dict, then loads Phi/Qinit
# from precomputed .mat sampling matrices. We keep the model's hardcoded 33x33=1089
# patch geometry (baked into BasicBlock.forward via `x.view(-1, 1, 33, 33)`), pick a
# small measurement dim, and construct random Phi/Qinit tensors of the right shape
# in place of the .mat-file sampling matrices (matching Train_CS_ISTA_Net.py's own
# dtype/shape conventions: Phi is [n_input, n_output], Qinit is [n_output, n_input]).
# ---------------------------------------------------------------------------
_LAYER_NO = 3
_N_OUTPUT = 1089  # 33 * 33, hardcoded by BasicBlock.forward
_N_INPUT = 43  # CS ratio ~4%, matches ratio_dict[4] in the real training script
_BATCH = 2


def build_istanet():
    torch.manual_seed(0)
    model = ISTANet(_LAYER_NO)
    model.eval()
    return model


def example_input_istanet():
    torch.manual_seed(0)
    Phix = torch.randn(_BATCH, _N_INPUT)
    Phi = torch.randn(_N_INPUT, _N_OUTPUT)
    Qinit = torch.randn(_N_OUTPUT, _N_INPUT)
    return (Phix, Phi, Qinit)


MENAGERIE_ENTRIES = [
    ("ISTA-Net", "build_istanet", "example_input_istanet", 2018, MENAGERIE_ZOO),
]
