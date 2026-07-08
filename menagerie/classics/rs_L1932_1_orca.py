# SOURCE: vendored from jzhoulab/orca @ main
# (https://raw.githubusercontent.com/jzhoulab/orca/main/orca_modules.py, class `Net`)
#
# Orca (Zhou, "Multiscale and integrative single-cell Hi-C analysis with Fast-Higashi" /
# Zhou 2022 "Genome sequence-based prediction of multiscale genome structure"; jzhoulab/orca)
# predicts multiscale 3D genome (Hi-C contact map) structure directly from DNA sequence.
# `Net` is Orca's self-contained 1Mb model: the integrated dilated-1D-conv encoder
# (bp resolution -> 4kb resolution) fused with the dilated-2D-conv decoder that produces
# a symmetric contact map, used standalone by orca_models.H1esc_1M / Hff_1M. The full
# multi-resolution Orca pipeline (H1esc/Hff/HCTnoc/*_256M classes in orca_models.py)
# composes several of these Encoder/Encoder2/Encoder3/Decoder stages together and loads
# pretrained state_dicts from data files not present in this environment; `Net` is the
# smallest fully self-contained real Orca architecture that requires no external weights
# or additional Orca modules, so it is vendored here at random init (tiny size) per the
# rung-2 "real repo code" path. Only the class body is copied verbatim from
# `orca_modules.py`; imports were trimmed to what `Net` itself needs (torch, checkpoint).

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

MENAGERIE_ZOO = "vendored-pytorch"


class Net(nn.Module):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model. The trained model weighted can be
        loaded into Encoder and Decoder_1m modules.

        Parameters
        ----------
        num_1d : int or None, optional
            The number of 1D targets used for the auxiliary
            task of predicting ChIP-seq profiles.
        """
        super(Net, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(96, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconvtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
            ]
        )

        self.convtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=(1, 1), padding=0),
        )
        if num_1d is not None:
            self.final_1d = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, num_1d, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
        self.num_1d = num_1d

    def forward(self, x):
        """Forward propagation of a batch."""

        def run0(x, dummy):
            lout1 = self.lconv1(x)
            out1 = self.conv1(lout1)
            lout2 = self.lconv2(out1 + lout1)
            out2 = self.conv2(lout2)
            lout3 = self.lconv3(out2 + lout2)
            out3 = self.conv3(lout3)
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = checkpoint(run0, x, dummy)
        else:
            cur = checkpoint(run0, x, dummy)

        def run1(cur):
            first = True
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        if self.num_1d:
            return cur, output1d
        else:
            return cur


def build_orca():
    torch.manual_seed(0)
    net = Net(num_1d=None)
    # eval mode: BatchNorm1d rejects batch=1 with spatial-size=1 in training mode
    # (the 1Mb encoder's final downsampling stage collapses to a single bin at
    # the minimum valid input length), so this only traces cleanly in eval mode.
    net.eval()
    return net


def example_input_orca():
    torch.manual_seed(0)
    # 4-channel one-hot DNA sequence; length must be a multiple of the encoder's
    # total downsampling factor (4*4*5*5*5*2 = 4000) for the pooling stages to
    # divide evenly. 4000 is the smallest valid length (1Mb model bins at 4kb).
    return torch.randn(1, 4, 4000)


MENAGERIE_ENTRIES = [
    ("Orca", build_orca, example_input_orca, 2022, MENAGERIE_ZOO),
]
