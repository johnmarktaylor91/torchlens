# SOURCE: vendored from KavrakiLab/Spec2Mol @ master
# https://raw.githubusercontent.com/KavrakiLab/Spec2Mol/master/spectra_encoder_model.py
# Net1D is the real mass-spectrum encoder used by Spec2Mol (Litsa et al., "Spec2Mol")
# to map a 4-channel tandem-MS spectrum vector into a 512-d embedding that is decoded
# into a SMILES string by a downstream Transformer decoder. Only imports/formatting are
# adapted for standalone staging; the architecture is untouched.
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_out_dim(length_in, kernel, stride, padding, dilation):
    length_out = (length_in + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    return length_out


class Net1D(nn.Module):
    def __init__(self, length_in):
        super(Net1D, self).__init__()
        out_1 = conv_out_dim(length_in, 200, 1, 100, 1)
        out_2 = conv_out_dim(out_1, 50, 50, 0, 1)
        out_3 = conv_out_dim(out_2, 200, 1, 100, 1)
        out_4 = conv_out_dim(out_3, 50, 50, 0, 1)
        self.cnn_out = 8 * out_4
        self.conv1 = nn.Conv1d(4, 4, 200, stride=1, padding=100, dilation=1)
        self.norm1 = nn.BatchNorm1d(4)
        self.pool1 = nn.MaxPool1d(50, stride=50, padding=0, dilation=1)
        self.pool2 = nn.MaxPool1d(50, stride=50, padding=0, dilation=1)
        self.conv2 = nn.Conv1d(4, 8, 200, stride=1, padding=100, dilation=1)
        self.norm2 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8 * out_4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.norm3 = nn.BatchNorm1d(512)

    def forward(self, x):
        x = self.pool1(F.relu(self.norm1(self.conv1(x))))
        x = self.pool2(F.relu(self.norm2(self.conv2(x))))
        x = x.view(-1, self.cnn_out)
        x = F.relu(self.norm3(self.fc1(x)))
        x = torch.tanh(self.fc2(x))
        return x


# Real deployed model uses length_in = int((499.97 - 50.0) * 100) = 44997 (0.01 Da
# resolution mass-spectrum vector, minmass=50, maxmass=499.97). We use a much smaller
# but still architecturally-valid length (must survive two conv+pool halvings without
# collapsing to 0) for a fast tiny-config trace.
_LENGTH_IN = 2600


def build_spec2mol_net1d():
    return Net1D(_LENGTH_IN)


def example_input_spec2mol_net1d():
    # (batch, 4 channels: [M+H]+ low/high, [M-H]- low/high energy spectra, length_in)
    return torch.randn(2, 4, _LENGTH_IN)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "Spec2Mol MS Encoder (Net1D)",
        "build_spec2mol_net1d",
        "example_input_spec2mol_net1d",
        2021,
        "vendored-pytorch",
    ),
]
