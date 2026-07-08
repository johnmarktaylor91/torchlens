# SOURCE: vendored from lab-emi/OpenDPD @ main
#   Vendored files: models.py (CoreModel, gru backbone branch only), backbones/gru.py
#   (GRU).
# https://github.com/lab-emi/OpenDPD
#
# OpenDPD (Wu & Gao, TU Delft, arxiv PA-modeling/digital-predistortion framework).
# CoreModel is the repo's shared top-level DPD/PA-model wrapper: it dispatches to a
# selectable RNN "backbone" (GRU/LSTM/DeltaGRU/... -- OpenDPD supports many; this
# staged module vendors the plain GRU backbone, one of the paper's real named DPD
# variants) that consumes an (N, T, F) I/Q feature sequence and predicts (I, Q)
# baseband output samples via GRU + linear readout, with zero-initialized hidden
# state supplied automatically when not passed. This vendors the CoreModel dispatch
# class as used with backbone_type="gru" (rather than every backbone in the repo,
# to keep the staged module minimal); it is the real repo class, unmodified.
#
# NOTE (POTENTIAL_DEDUP): tierA/queue.tsv row 1323 ("DL-based Digital Pre-Distortion
# (DPD-Net)") points at the same lab-emi/OpenDPD repo under a different candidate
# name; whoever processes row 1323 should dedupe against this entry rather than
# re-vendoring the same source.
#
# No API-compat fixes were needed; CoreModel/GRU are unmodified from the source.

import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        num_layers,
        bidirectional=False,
        batch_first=True,
        bias=True,
    ):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.bias = bias

        # Instantiate NN Layers
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=self.bidirectional,
            batch_first=self.batch_first,
            bias=self.bias,
        )
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=self.output_size, bias=True)

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

    def forward(self, x, h_0):
        out, _ = self.rnn(x, h_0)
        out = self.fc_out(out)
        return out


class CoreModel(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        backbone_type,
        window_size=None,
        num_dvr_units=None,
        thx=0,
        thh=0,
    ):
        super(CoreModel, self).__init__()
        self.output_size = 2  # PA outputs: I & Q
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.backbone_type = backbone_type
        self.thx = thx
        self.thh = thh
        self.window_size = window_size
        self.num_dvr_units = num_dvr_units
        self.batch_first = True  # Force batch first
        self.bidirectional = False
        self.bias = True

        if backbone_type == "gru":
            self.backbone = GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                batch_first=self.batch_first,
                bias=self.bias,
            )
        else:
            raise ValueError(
                f"The backbone type '{self.backbone_type}' is not supported by this staged module (only 'gru' is "
                f"vendored here); see the full OpenDPD repo for the other backbones."
            )

        # Initialize backbone parameters
        try:
            self.backbone.reset_parameters()
        except AttributeError:
            pass

    def forward(self, x, h_0=None):
        device = x.device
        batch_size = x.size(0)  # NOTE: dim of x must be (batch, time, feat)/(N, T, F)

        if h_0 is None:  # Create initial hidden states if necessary
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)

        # Forward Propagate through the RNN
        out = self.backbone(x, h_0)

        return out


MENAGERIE_ZOO = "vendored-pytorch"


def build_dpd_opendpd():
    model = CoreModel(
        input_size=6,
        hidden_size=16,
        num_layers=1,
        backbone_type="gru",
    )
    model.eval()
    return model


def example_input_dpd_opendpd():
    # (batch, time, feat) I/Q feature sequence, matching CoreModel.forward's
    # documented input layout.
    return torch.randn(1, 20, 6)


MENAGERIE_ENTRIES = [
    (
        "DPD-Net (OpenDPD GRU backbone)",
        "build_dpd_opendpd",
        "example_input_dpd_opendpd",
        2024,
        MENAGERIE_ZOO,
    ),
]
