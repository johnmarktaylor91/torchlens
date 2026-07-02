# SOURCE: vendored from tobiascz/TeCNO @ master
# https://raw.githubusercontent.com/tobiascz/TeCNO/master/models/mstcn.py
#
# Czempiel, Paschali, Keicher, Simson, Feussner, Kim, Navab, 2020 (MICCAI 2020)
# "TeCNO: Surgical Phase Recognition with Multi-Stage Temporal Convolutional
# Networks". The real architectural contribution is `MultiStageModel`, a stack of
# `SingleStageModel`s built from dilated (optionally causal) `DilatedResidualLayer`
# 1D convolutions -- adapted from https://github.com/yabufarha/ms-tcn/blob/master/model.py
# (credited in the repo's own header comment) into TeCNO's cascaded-refinement
# multi-stage classification head over pre-extracted per-frame CNN features. This
# is pure `torch.nn` with no PyTorch-Lightning dependency (the Lightning
# `TeCNO(LightningModule)` wrapper lives separately in `modules/mstcn/tecno.py` and
# is purely a training harness -- it calls `self.model.forward(video_fe)` on exactly
# this `MultiStageModel`, so vendoring `models/mstcn.py` alone captures the real
# TeCNO architecture). Vendored verbatim below (`MultiStageModel`, `SingleStageModel`,
# `DilatedResidualLayer`); `hparams.*` argparse-namespace access replaced with a tiny
# local `_Hparams` shim carrying the same attribute names TeCNO's own
# `add_model_specific_args` registers (`mstcn_stages`, `mstcn_layers`, `mstcn_f_maps`,
# `mstcn_f_dim`, `out_features`, `mstcn_causal_conv`) -- no architecture change.

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class MultiStageModel(nn.Module):
    def __init__(self, hparams):
        self.num_stages = hparams.mstcn_stages
        self.num_layers = hparams.mstcn_layers
        self.num_f_maps = hparams.mstcn_f_maps
        self.dim = hparams.mstcn_f_dim
        self.num_classes = hparams.out_features
        self.causal_conv = hparams.mstcn_causal_conv
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(
            self.num_layers,
            self.num_f_maps,
            self.dim,
            self.num_classes,
            causal_conv=self.causal_conv,
        )
        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(
                        self.num_layers,
                        self.num_f_maps,
                        self.num_classes,
                        self.num_classes,
                        causal_conv=self.causal_conv,
                    )
                )
                for s in range(self.num_stages - 1)
            ]
        )
        self.smoothing = False

    def forward(self, x):
        out_classes = self.stage1(x)
        outputs_classes = out_classes.unsqueeze(0)
        for s in self.stages:
            out_classes = s(F.softmax(out_classes, dim=1))
            outputs_classes = torch.cat((outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return outputs_classes


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal_conv=False):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    DilatedResidualLayer(2**i, num_f_maps, num_f_maps, causal_conv=causal_conv)
                )
                for i in range(num_layers)
            ]
        )
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out_classes = self.conv_out_classes(out)
        return out_classes


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, causal_conv=False, kernel_size=3):
        super(DilatedResidualLayer, self).__init__()
        self.causal_conv = causal_conv
        self.dilation = dilation
        self.kernel_size = kernel_size
        if self.causal_conv:
            self.conv_dilated = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=(dilation * (kernel_size - 1)),
                dilation=dilation,
            )
        else:
            self.conv_dilated = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation
            )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        if self.causal_conv:
            out = out[:, :, : -(self.dilation * 2)]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


# ============================================================================
# build_/example_input_ harness
# ============================================================================


class _Hparams:
    """Minimal stand-in for the argparse `hparams` namespace TeCNO's
    `MultiStageModel.__init__` reads; carries the same attribute names TeCNO's
    own `add_model_specific_args` registers. Values shrunk from the real
    defaults (mstcn_stages=4, mstcn_layers=10, mstcn_f_maps=64, mstcn_f_dim=2048,
    out_features=7) for a fast trace."""

    def __init__(self):
        self.mstcn_stages = 2
        self.mstcn_layers = 3
        self.mstcn_f_maps = 16
        self.mstcn_f_dim = 32
        self.out_features = 7
        self.mstcn_causal_conv = True


def build_tecno_mstcn():
    hparams = _Hparams()
    model = MultiStageModel(hparams)
    model.eval()
    return model


def example_input_tecno_mstcn():
    torch.manual_seed(0)
    # (batch, dim=mstcn_f_dim, seq_len) -- per-frame CNN feature stem, channels-first.
    return torch.randn(1, 32, 20)


MENAGERIE_ENTRIES = [
    (
        "TeCNO",
        build_tecno_mstcn,
        example_input_tecno_mstcn,
        2020,
        "vendored-pytorch",
    ),
]
