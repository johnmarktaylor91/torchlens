# SOURCE: vendored from BojarLab/CandyCrunch @ 7a54179db9cc4de27668a145d57c45c674aa864c
# File: candycrunch/model.py (CandyCrunch_CNN class + its ResUnit building block only;
# the dataset/augmentation helpers in the same source file are training-time utilities
# and are not part of the model architecture, so they are omitted here).
# CandyCrunch: a dilated-causal-convolution ResNet over MS/MS glycan fragment spectra,
# fused with precursor mass, retention time, and categorical metadata embeddings, to
# predict glycan structure directly from LC-MS/MS spectra (Nature Methods 2024).
# Vendored verbatim aside from import cleanup (only `torch`/`torch.nn.functional` needed
# by the model classes; the source file's `torchvision.transforms`-based augmentation
# pipeline for the training Dataset class is not part of the model itself).
import torch
import torch.nn.functional as F
from torch import flatten, nn


class ResUnit(nn.Module):
    def __init__(self, in_channels, size=3, dilation=1, causal=False, in_ln=True):
        super(ResUnit, self).__init__()
        self.size = size
        self.dilation = dilation
        self.causal = causal
        self.in_ln = in_ln
        if self.in_ln:
            self.ln1 = nn.InstanceNorm1d(in_channels, affine=True)
            self.ln1.weight.data.fill_(1.0)
        self.conv_in = nn.Conv1d(in_channels, in_channels // 2, 1)
        self.ln2 = nn.InstanceNorm1d(in_channels // 2, affine=True)
        self.ln2.weight.data.fill_(1.0)
        self.conv_dilated = nn.Conv1d(
            in_channels // 2,
            in_channels // 2,
            size,
            dilation=self.dilation,
            padding=((dilation * (size - 1)) if causal else (dilation * (size - 1) // 2)),
        )
        self.ln3 = nn.InstanceNorm1d(in_channels // 2, affine=True)
        self.ln3.weight.data.fill_(1.0)
        self.conv_out = nn.Conv1d(in_channels // 2, in_channels, 1)

    def forward(self, inp):
        x = inp
        if self.in_ln:
            x = self.ln1(x)
        x = nn.functional.leaky_relu(x)
        x = nn.functional.leaky_relu(self.ln2(self.conv_in(x)))
        x = self.conv_dilated(x)
        if self.causal and self.size > 1:
            x = x[:, :, : -self.dilation * (self.size - 1)]
        x = nn.functional.leaky_relu(self.ln3(x))
        x = self.conv_out(x)
        return x + inp


class CandyCrunch_CNN(torch.nn.Module):
    def __init__(self, input_dim, num_classes=1, hidden_dim=512, input_precursor_dim=None):
        super(CandyCrunch_CNN, self).__init__()

        self.input_dim = input_dim

        self.mz_lin1 = torch.nn.Linear(input_dim, 2 * hidden_dim)  # not used
        self.prec_lin1 = torch.nn.Linear(input_precursor_dim, 24)
        self.rt_lin1 = torch.nn.Linear(1, 24)
        self.comb_lin1 = torch.nn.Linear(2 * hidden_dim + 24 + 24 + 24 + 24 + 24 + 24 + 24, 2 * 512)
        self.comb_lin2 = torch.nn.Linear(2 * 512, 2 * 256)
        self.comb_lin3 = torch.nn.Linear(2 * 256, num_classes)

        self.type_emb = torch.nn.Embedding(5, 24)
        self.mode_emb = torch.nn.Embedding(3, 24)
        self.lc_emb = torch.nn.Embedding(4, 24)
        self.modification_emb = torch.nn.Embedding(4, 24)
        self.trap_emb = torch.nn.Embedding(5, 24)

        self.conv1 = torch.nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1)
        self.res1 = ResUnit(64, size=2, dilation=1, causal=True)
        self.res2 = ResUnit(64, size=2, dilation=2, causal=True)
        self.res3 = ResUnit(64, size=2, dilation=4, causal=True)
        self.res4 = ResUnit(64, size=2, dilation=8, causal=True)
        self.res5 = ResUnit(64, size=2, dilation=16, causal=True)
        self.res6 = ResUnit(64, size=2, dilation=32, causal=True)
        self.maxpool1 = torch.nn.MaxPool1d(kernel_size=20)
        self.fc1 = torch.nn.Linear(in_features=6528, out_features=1024)

        self.mz_bn1 = torch.nn.LayerNorm(2 * hidden_dim)  # not used
        self.prec_bn1 = torch.nn.LayerNorm(24)
        self.rt_bn1 = torch.nn.LayerNorm(24)
        self.comb_bn1 = torch.nn.LayerNorm(2 * 512)
        self.comb_bn2 = torch.nn.LayerNorm(2 * 256)
        self.mz_act1 = torch.nn.LeakyReLU()  # not used
        self.prec_act1 = torch.nn.LeakyReLU()
        self.rt_act1 = torch.nn.LeakyReLU()
        self.comb_act1 = torch.nn.LeakyReLU()
        self.comb_act2 = torch.nn.LeakyReLU()
        self.comb_dp1 = torch.nn.Dropout(0.2)
        self.comb_dp2 = torch.nn.Dropout(0.2)

    def forward(self, mz_list, precursor, glycan_type, rt, mode, lc, modification, trap, rep=False):
        glycan_type = self.type_emb(glycan_type).squeeze(1)
        mode = self.mode_emb(mode).squeeze(1)
        lc = self.lc_emb(lc).squeeze(1)
        modification = self.modification_emb(modification).squeeze(1)
        trap = self.trap_emb(trap).squeeze(1)
        precursor = self.prec_act1(self.prec_bn1(self.prec_lin1(precursor)))
        rt = self.rt_act1(self.rt_bn1(self.rt_lin1(rt)))
        mz = F.leaky_relu(self.conv1(mz_list))
        mz = self.res1(mz)
        mz = self.res2(mz)
        mz = self.res3(mz)
        mz = self.res4(mz)
        mz = self.res5(mz)
        mz = self.res6(mz)

        mz = self.maxpool1(mz)
        mz = F.leaky_relu(self.fc1(flatten(mz, start_dim=1)))

        comb = torch.cat([mz, precursor, glycan_type, rt, mode, lc, modification, trap], dim=1)
        comb = self.comb_dp1(self.comb_act1(self.comb_bn1(self.comb_lin1(comb))))
        comb_rep = self.comb_lin2(comb)
        comb = self.comb_dp2(self.comb_act2(self.comb_bn2(comb_rep)))
        comb = self.comb_lin3(comb)
        if rep:
            return comb, comb_rep
        else:
            return comb


MENAGERIE_ZOO = "vendored-pytorch"


def build_candycrunch():
    # Real repo default (candycrunch/prediction.py): CandyCrunch_CNN(2048,
    # num_classes=len(glycans), input_precursor_dim=12). Two internal layer widths are
    # hardcoded rather than derived from the constructor args, so they pin two of the
    # "default" values as effectively load-bearing: `fc1` has in_features=6528, which is
    # only consistent with a 2048-bin mz spectrum (64 channels * (2048 // maxpool
    # kernel=20) == 6528); and `fc1`'s out_features=1024 must equal `2*hidden_dim`
    # (comb_lin1's declared in_features), which only holds at the real default
    # hidden_dim=512. num_classes is shrunk for the menagerie's tiny-size convention
    # (it is a free dimension, unlike input_dim/hidden_dim above).
    return CandyCrunch_CNN(input_dim=2048, num_classes=16, hidden_dim=512, input_precursor_dim=12)


def example_input_candycrunch():
    torch.manual_seed(0)
    batch = 2
    mz_list = torch.randn(batch, 2, 2048)  # (batch, [mz, intensity] channels, n_peaks)
    precursor = torch.randn(batch, 12)
    glycan_type = torch.randint(0, 5, (batch, 1))
    rt = torch.randn(batch, 1)
    mode = torch.randint(0, 3, (batch, 1))
    lc = torch.randint(0, 4, (batch, 1))
    modification = torch.randint(0, 4, (batch, 1))
    trap = torch.randint(0, 5, (batch, 1))
    return (mz_list, precursor, glycan_type, rt, mode, lc, modification, trap)


MENAGERIE_ENTRIES = [
    ("CandyCrunch", "build_candycrunch", "example_input_candycrunch", 2024, "SOURCE_AVAILABLE"),
]
