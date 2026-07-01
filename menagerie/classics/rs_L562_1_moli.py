# SOURCE: vendored from hosseinshn/MOLI @ master
# https://github.com/hosseinshn/MOLI/blob/master/Cross%20validation/MOLI%20Complete/Cetuximab_cvSoftTripletClassifierNetv16_Script.py
#
# MOLI (Multi-Omics Late Integration): three parallel autoencoder-style
# encoder branches (expression / mutation / copy-number), each
# Linear -> BatchNorm1d -> ReLU -> Dropout, whose outputs are concatenated,
# L2-normalized, and passed to a Linear -> Dropout -> Sigmoid classifier.
# The per-branch classes below (AEE/AEM/AEC/Classifier) and the forward
# composition (concat -> F.normalize -> Clas) are copied verbatim from the
# repo's "MOLI Complete" cross-validation script; only the free-standing
# hyperparameters (IE_dim/IM_dim/h_dim*/rate*) were pulled out of the
# training-loop closure into constructor arguments so the classes can be
# instantiated outside the original random-search training loop.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class AEE(nn.Module):
    """Expression-branch encoder (verbatim from MOLI Complete script)."""

    def __init__(self, IE_dim, h_dim1, rate1):
        super(AEE, self).__init__()
        self.EnE = torch.nn.Sequential(
            nn.Linear(IE_dim, h_dim1), nn.BatchNorm1d(h_dim1), nn.ReLU(), nn.Dropout(rate1)
        )

    def forward(self, x):
        output = self.EnE(x)
        return output


class AEM(nn.Module):
    """Mutation-branch encoder (verbatim from MOLI Complete script)."""

    def __init__(self, IM_dim, h_dim2, rate2):
        super(AEM, self).__init__()
        self.EnM = torch.nn.Sequential(
            nn.Linear(IM_dim, h_dim2), nn.BatchNorm1d(h_dim2), nn.ReLU(), nn.Dropout(rate2)
        )

    def forward(self, x):
        output = self.EnM(x)
        return output


class AEC(nn.Module):
    """Copy-number-branch encoder (verbatim from MOLI Complete script;
    the original code also indexes this branch's Linear with IM_dim, a
    quirk of the source repeated here faithfully)."""

    def __init__(self, IC_dim, h_dim3, rate3):
        super(AEC, self).__init__()
        self.EnC = torch.nn.Sequential(
            nn.Linear(IC_dim, h_dim3), nn.BatchNorm1d(h_dim3), nn.ReLU(), nn.Dropout(rate3)
        )

    def forward(self, x):
        output = self.EnC(x)
        return output


class Classifier(nn.Module):
    """Late-fusion classifier head (verbatim from MOLI Complete script)."""

    def __init__(self, Z_in, rate4):
        super(Classifier, self).__init__()
        self.FC = torch.nn.Sequential(nn.Linear(Z_in, 1), nn.Dropout(rate4), nn.Sigmoid())

    def forward(self, x):
        return self.FC(x)


class MOLI(nn.Module):
    """Full MOLI model: three encoder branches feeding a shared classifier,
    reproducing the forward composition used in the repo's training loop
    (ZT = cat(ZEX, ZMX, ZCX); ZT = F.normalize(ZT, p=2, dim=0); Pred = Clas(ZT))."""

    def __init__(
        self,
        IE_dim=16,
        IM_dim=16,
        IC_dim=16,
        h_dim1=64,
        h_dim2=64,
        h_dim3=64,
        rate1=0.3,
        rate2=0.3,
        rate3=0.3,
        rate4=0.3,
    ):
        super(MOLI, self).__init__()
        self.AutoencoderE = AEE(IE_dim, h_dim1, rate1)
        self.AutoencoderM = AEM(IM_dim, h_dim2, rate2)
        self.AutoencoderC = AEC(IC_dim, h_dim3, rate3)
        Z_in = h_dim1 + h_dim2 + h_dim3
        self.Clas = Classifier(Z_in, rate4)

    def forward(self, dataE, dataM, dataC):
        ZEX = self.AutoencoderE(dataE)
        ZMX = self.AutoencoderM(dataM)
        ZCX = self.AutoencoderC(dataC)
        ZT = torch.cat((ZEX, ZMX, ZCX), 1)
        ZT = F.normalize(ZT, p=2, dim=0)
        Pred = self.Clas(ZT)
        return Pred


def build_moli():
    return MOLI(
        IE_dim=16,
        IM_dim=16,
        IC_dim=16,
        h_dim1=32,
        h_dim2=32,
        h_dim3=32,
        rate1=0.3,
        rate2=0.3,
        rate3=0.3,
        rate4=0.3,
    )


def example_input_moli():
    batch = 8
    return (torch.randn(batch, 16), torch.randn(batch, 16), torch.randn(batch, 16))


MENAGERIE_ENTRIES = [
    ("MOLI", "build_moli", "example_input_moli", 2019, "vendored-pytorch"),
]
