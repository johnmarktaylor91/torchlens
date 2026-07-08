# SOURCE: vendored from https://github.com/jianzhangcs/COAST @ main
# File: TRAIN_COAST.py (model classes CPMB, BasicBlock, COAST defined inline
# in the training script upstream). Only import/name changes made: model
# classes extracted from the training script into this standalone module;
# architecture and forward logic are byte-for-byte the upstream code.
"""COAST: COntrollable Arbitrary-Sampling neTwork for compressive sensing.

TIP 2021. An unrolled ISTA-Net-plus-style deep-unfolding network with a
condition-modulated proximal mapping block (CPMB) that lets a single model
handle arbitrary sampling ratios and sampling matrices at test time.
"""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


class CPMB(nn.Module):
    """Residual block with scale control
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, res_scale_linear, nf=32):
        super(CPMB, self).__init__()

        conv_bias = True

        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=conv_bias)
        self.res_scale = res_scale_linear
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        cond = x[1]
        content = x[0]
        cond = cond[:, 0:1]
        cond_repeat = cond.repeat((content.shape[0], 1))
        out = self.act(self.conv1(content))
        out = self.conv2(out)
        res_scale = self.res_scale(cond_repeat)
        alpha1 = res_scale.view(-1, 32, 1, 1)
        out1 = out * alpha1
        return content + out1, cond


class BasicBlock(torch.nn.Module):
    def __init__(self, res_scale_linear):
        super(BasicBlock, self).__init__()

        self.lambda_step = nn.Parameter(torch.Tensor([0.5]))

        self.head_conv = nn.Conv2d(1, 32, 3, 1, 1, bias=True)
        self.ResidualBlocks = nn.Sequential(
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32),
            CPMB(res_scale_linear=res_scale_linear, nf=32),
        )
        self.tail_conv = nn.Conv2d(32, 1, 3, 1, 1, bias=True)

    def forward(self, x, PhiTPhi, PhiTb, cond, block_size):
        x = x - self.lambda_step * torch.mm(x, PhiTPhi)
        x = x + self.lambda_step * PhiTb
        x_input = x.view(-1, 1, block_size, block_size)

        x_mid = self.head_conv(x_input)
        x_mid, cond = self.ResidualBlocks([x_mid, cond])
        x_mid = self.tail_conv(x_mid)
        x_pred = x_input + x_mid

        x_pred = x_pred.view(-1, block_size * block_size)

        return x_pred


class COAST(torch.nn.Module):
    def __init__(self, LayerNo):
        super(COAST, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo
        nf = 32
        scale_bias = True
        res_scale_linear = nn.Linear(1, nf, bias=scale_bias)

        for i in range(LayerNo):
            onelayer.append(BasicBlock(res_scale_linear=res_scale_linear))

        self.fcs = nn.ModuleList(onelayer)

    def forward(self, x, Phi, block_size=33):
        Phix = x[0]
        cond = x[1]

        PhiTPhi = torch.mm(torch.transpose(Phi, 0, 1), Phi)
        PhiTb = torch.mm(Phix, Phi)
        x = PhiTb.clone()

        for i in range(self.LayerNo):
            x = self.fcs[i](x, PhiTPhi, PhiTb, cond, block_size)

        x_final = x

        return x_final


# ---------------------------------------------------------------------------
# menagerie staging entrypoints
# ---------------------------------------------------------------------------


def build_coast():
    # Small LayerNo for a fast trace; architecture (CPMB/BasicBlock/COAST) unchanged.
    return COAST(LayerNo=2)


def example_input_coast():
    # forward(self, x, Phi, block_size) where x = [Phix, cond]:
    #   Phix: (batch, m) compressed measurements
    #   cond: (1, 2) sampling-ratio + noise-level condition vector
    #   Phi:  (m, block_size**2) sampling matrix
    torch.manual_seed(0)
    block_size = 8
    n_output = block_size * block_size
    m = 10
    batch = 2
    Phix = torch.randn(batch, m)
    cond = torch.randn(1, 2)
    Phi = torch.randn(m, n_output)
    return ([Phix, cond], Phi, block_size)


MENAGERIE_ENTRIES = [
    ("COAST", "build_coast", "example_input_coast", 2021, MENAGERIE_ZOO),
]
