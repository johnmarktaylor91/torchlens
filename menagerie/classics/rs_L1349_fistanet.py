# SOURCE: vendored from jinxixiang/FISTA-Net @ master
#   Vendored files: M5FISTANet.py (FISTANet, BasicBlock, initialize_weights),
#   M1LapReg.py (MatMask -- circular-mesh-to-grid mask construction, used verbatim
#   to build a real, non-synthetic mask matrix for the tiny example input).
# https://github.com/jinxixiang/FISTA-Net
#
# FISTA-Net (Xiang et al., "FISTA-Net: Learning a Fast Iterative Shrinkage
# Thresholding Network for Inverse Problems in Imaging", IEEE Trans. Medical
# Imaging 2021). An unrolled-optimization network for ill-posed linear inverse
# imaging problems (the reference application is Electrical Impedance Tomography
# reconstruction): each of `LayerNo` weight-shared BasicBlock stages performs one
# FISTA iteration -- a quadratic-regularized gradient descent step in measurement
# space (using the fixed system matrix Phi, its Gram PhiTPhi/PhiTb, and a Laplacian
# regularizer LTL), followed by a learned nonlinear proximal operator (paired
# conv/ReLU "forward" and "backward" CNN blocks around a learnable soft-threshold),
# plus a FISTA two-step momentum update with per-layer learned gradient-step and
# threshold hyperparameters (regularized through Softplus to stay positive and
# monotonic). We construct a real (not fabricated-shape) mask via the source's own
# MatMask -- it maps a small in-circle mesh onto a pnum x pnum grid exactly like the
# real EIT reconstruction target -- and a small random Phi/L (the real Jmat.csv/
# Lapmat.csv sensitivity/regularization matrices are experiment-specific numeric
# data files distributed with the paper, not source code, so are not fetched here;
# their role, dimensions, and how they are consumed by the model are unchanged).
#
# Minimal API-compat fixes (NOT architecture changes): none needed; the source
# imports only torch and numpy.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    """One weight-shared FISTA iteration: gradient step + learned proximal CNN."""

    def __init__(self, features=32):
        super(BasicBlock, self).__init__()
        self.Sp = nn.Softplus()

        self.conv_D = nn.Conv2d(1, features, (3, 3), stride=1, padding=1)
        self.conv1_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_forward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)

        self.conv1_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv2_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv3_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv4_backward = nn.Conv2d(features, features, (3, 3), stride=1, padding=1)
        self.conv_G = nn.Conv2d(features, 1, (3, 3), stride=1, padding=1)

    def forward(self, x, PhiTPhi, PhiTb, LTL, mask, lambda_step, soft_thr):
        # convert data format from (batch_size, channel, pnum, pnum) to (circle_num, batch_size)
        pnum = x.size()[2]
        x = x.view(x.size()[0], x.size()[1], pnum * pnum, -1)
        x = torch.squeeze(x, 1)
        x = torch.squeeze(x, 2).t()
        x = mask.mm(x)

        # quadratic tv gradient descent from doi: 10.1109/TMI.2009.2022540 Eq. (10)
        x = x - self.Sp(lambda_step) * torch.inverse(PhiTPhi + 0.001 * LTL).mm(
            PhiTPhi.mm(x) - PhiTb - 0.001 * LTL.mm(x)
        )

        # convert (circle_num, batch_size) to (batch_size, channel, pnum, pnum)
        x = torch.mm(mask.t(), x)
        x = x.view(pnum, pnum, -1)
        x = x.unsqueeze(0)
        x_input = x.permute(3, 0, 1, 2)

        x_D = self.conv_D(x_input)

        x = self.conv1_forward(x_D)
        x = F.relu(x)
        x = self.conv2_forward(x)
        x = F.relu(x)
        x = self.conv3_forward(x)
        x = F.relu(x)
        x_forward = self.conv4_forward(x)

        # soft-thresholding block
        x_st = torch.mul(torch.sign(x_forward), F.relu(torch.abs(x_forward) - self.Sp(soft_thr)))

        x = self.conv1_backward(x_st)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_backward = self.conv4_backward(x)

        x_G = self.conv_G(x_backward)

        # prediction output (skip connection); non-negative output
        x_pred = F.relu(x_input + x_G)

        # compute symmetry loss
        x = self.conv1_backward(x_forward)
        x = F.relu(x)
        x = self.conv2_backward(x)
        x = F.relu(x)
        x = self.conv3_backward(x)
        x = F.relu(x)
        x_D_est = self.conv4_backward(x)
        symloss = x_D_est - x_D

        return [x_pred, symloss, x_st]


class FISTANet(nn.Module):
    def __init__(self, LayerNo, Phi, L, mask):
        super(FISTANet, self).__init__()
        self.LayerNo = LayerNo
        self.Phi = Phi
        self.L = L
        self.mask = mask
        onelayer = []

        self.bb = BasicBlock(features=32)
        for i in range(LayerNo):
            onelayer.append(self.bb)

        self.fcs = nn.ModuleList(onelayer)
        self.fcs.apply(initialize_weights)

        # thresholding value
        self.w_theta = nn.Parameter(torch.Tensor([-0.5]))
        self.b_theta = nn.Parameter(torch.Tensor([-2]))
        # gradient step
        self.w_mu = nn.Parameter(torch.Tensor([-0.2]))
        self.b_mu = nn.Parameter(torch.Tensor([0.1]))
        # two-step update weight
        self.w_rho = nn.Parameter(torch.Tensor([0.5]))
        self.b_rho = nn.Parameter(torch.Tensor([0]))

        self.Sp = nn.Softplus()

    def forward(self, x0, b):
        """
        Phi   : system matrix; default dim 104 * 3228;
        mask  : mask matrix, dim 3228 * 4096
        b     : measured signal vector;
        x0    : initialized x with Laplacian Reg.
        """
        b = torch.squeeze(b, 1)
        b = torch.squeeze(b, 2)
        b = b.t()

        PhiTPhi = self.Phi.t().mm(self.Phi)
        PhiTb = self.Phi.t().mm(b)
        LTL = self.L.t().mm(self.L)

        xold = x0
        y = xold
        layers_sym = []
        layers_st = []
        xnews = []
        xnews.append(xold)

        for i in range(self.LayerNo):
            theta_ = self.w_theta * i + self.b_theta
            mu_ = self.w_mu * i + self.b_mu
            [xnew, layer_sym, layer_st] = self.fcs[i](
                y, PhiTPhi, PhiTb, LTL, self.mask, mu_, theta_
            )
            rho_ = (self.Sp(self.w_rho * i + self.b_rho) - self.Sp(self.b_rho)) / self.Sp(
                self.w_rho * i + self.b_rho
            )
            y = xnew + rho_ * (xnew - xold)
            xold = xnew
            xnews.append(xnew)
            layers_st.append(layer_st)
            layers_sym.append(layer_sym)

        return [xnew, layers_sym, layers_st]


def _mat_mask(pnum):
    """MatMask, verbatim from M1LapReg.py: maps the in-circle mesh onto a pnum x
    pnum grid, dim (n_inside, pnum*pnum)."""
    xcor = np.arange(-1 + 1 / pnum, 1 + 1 / pnum, 2 / pnum)
    ycor = np.arange(1 - 1 / pnum, -1 - 1 / pnum, -2 / pnum)

    n_inside = int(
        sum(
            1
            for j in range(pnum)
            for i in range(pnum)
            if xcor[i] * xcor[i] + ycor[j] * ycor[j] <= 1
        )
    )
    Msk = np.zeros((n_inside, pnum * pnum))
    Mat_id = np.arange(pnum * pnum).reshape(pnum, pnum)

    Mat_id = Mat_id.T
    Mat_id = np.fliplr(Mat_id)

    k = 0
    for j in range(pnum):
        for i in range(pnum):
            if xcor[i] * xcor[i] + ycor[j] * ycor[j] <= 1:
                Msk[k, Mat_id[j, i]] = 1
                k = k + 1
    return Msk


MENAGERIE_ZOO = "vendored-pytorch"


def build_fistanet():
    # pnum=8 keeps the mesh tiny (52 in-circle points out of the 8x8=64 grid,
    # vs. the paper's real 64x64/3228-point EIT mesh) while MatMask is the
    # source's real construction (not a fabricated shape). m=16 is a small
    # synthetic measurement dimension standing in for the paper's 104-row real
    # EIT sensitivity matrix Jmat.csv (a numeric data file, not source code).
    torch.manual_seed(0)
    pnum = 8
    n_inside = 52
    m = 16
    mask = torch.tensor(_mat_mask(pnum), dtype=torch.float32)
    Phi = torch.randn(m, n_inside)
    L = torch.randn(n_inside, n_inside)
    model = FISTANet(LayerNo=3, Phi=Phi, L=L, mask=mask)
    model.eval()
    return model


def example_input_fistanet():
    torch.manual_seed(0)
    pnum = 8
    m = 16
    x0 = torch.rand(1, 1, pnum, pnum)
    b = torch.randn(1, 1, m, 1)
    return (x0, b)


MENAGERIE_ENTRIES = [
    ("FISTA-Net", "build_fistanet", "example_input_fistanet", 2021, MENAGERIE_ZOO),
]
