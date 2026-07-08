# SOURCE: vendored from https://github.com/lab-emi/OpenDPD @ main
#   Vendored file: backbones/dgru.py (the DGRU class in full). Import trimmed to what
#   the class actually uses (torch, torch.nn); no changes to the architecture or the
#   forward-pass math.
#
# OpenDPD (Wu, Gao, TU Delft) is an end-to-end PyTorch framework for RF power-amplifier
# (PA) behavioral modeling and digital pre-distortion (DPD). DGRU ("delta-feature GRU")
# is the framework's core DPD/PA backbone: it hand-engineers 6 instantaneous-envelope
# features from the complex I/Q input (I, Q, amplitude, amplitude^3, sin, cos of phase)
# before feeding a standard nn.GRU, then fuses the GRU's hidden state (through an extra
# ReLU-gated linear projection `fc_hid`) with the raw 6-feature vector via a residual
# skip-concatenation into the final linear output head `fc_out`, predicting 2 outputs
# (I/Q of the pre-distorted or amplifier response signal). We trace `DGRU.forward` with
# a small hidden size and a short random I/Q sequence (batch_first=True).

import torch
from torch import nn

MENAGERIE_ZOO = "vendored-pytorch"


# ---- from backbones/dgru.py, unmodified ----
class DGRU(nn.Module):
    def __init__(
        self, hidden_size, output_size, num_layers, bidirectional=False, batch_first=True, bias=True
    ):
        super(DGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = 6
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        # Instantiate NN Layers
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
            bias=self.bias,
        )
        self.fc_out = nn.Linear(
            in_features=hidden_size + self.input_size, out_features=self.output_size, bias=self.bias
        )
        self.fc_hid = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=self.bias)

    def reset_parameters(self):
        for name, param in self.rnn.named_parameters():
            num_gates = int(param.shape[0] / self.hidden_size)
            if "bias" in name:
                nn.init.constant_(param, 0)
            if "weight" in name:
                for i in range(0, num_gates):
                    nn.init.orthogonal_(param[i * self.hidden_size : (i + 1) * self.hidden_size, :])
            if "weight_ih_l0" in name:
                for i in range(0, num_gates):
                    nn.init.xavier_uniform_(
                        param[i * self.hidden_size : (i + 1) * self.hidden_size, :]
                    )

        for name, param in self.fc_out.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            if "bias" in name:
                nn.init.constant_(param, 0)

        for name, param in self.fc_hid.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param)
            if "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x, h_0):
        # Feature Extraction
        i_x = torch.unsqueeze(x[..., 0], dim=-1)
        q_x = torch.unsqueeze(x[..., 1], dim=-1)
        amp2 = torch.pow(i_x, 2) + torch.pow(q_x, 2)
        amp = torch.sqrt(amp2)
        amp3 = torch.pow(amp, 3)
        cos = i_x / amp
        sin = q_x / amp
        x = torch.cat((i_x, q_x, amp, amp3, sin, cos), dim=-1)
        # Regressor
        out, _ = self.rnn(x, h_0)
        out = torch.relu(self.fc_hid(out))
        out = torch.cat((out, x), dim=-1)
        out = self.fc_out(out)
        return out


# ---------------------------------------------------------------------------
# Menagerie staging harness
# ---------------------------------------------------------------------------
def build_opendpd_dgru():
    """DGRU DPD backbone, small hidden size / 1 layer for a tiny trace. Matches
    CoreModel's construction for backbone_type='dgru' with I/Q output_size=2."""
    torch.manual_seed(0)
    model = DGRU(
        hidden_size=16,
        output_size=2,
        num_layers=1,
        bidirectional=False,
        batch_first=True,
        bias=True,
    )
    model.reset_parameters()
    return model


def example_input_opendpd_dgru():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 2)  # (batch, time, I/Q)
    h_0 = torch.zeros(1, 1, 16)  # (num_layers, batch, hidden_size)
    return (x, h_0)


MENAGERIE_ENTRIES = [
    (
        "OpenDPD DGRU (Digital Pre-Distortion)",
        "build_opendpd_dgru",
        "example_input_opendpd_dgru",
        2024,
        "vendored-pytorch",
    ),
]
