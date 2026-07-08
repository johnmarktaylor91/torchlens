# SOURCE: vendored from sreyas-mohan/DeepFreq @ 5215bdce863ead6a5588b565eebc7363194af85f
# (modules.py::FrequencyRepresentationModule, ::FrequencyCountingModule; unchanged)
"""DeepFreq: frequency estimation via deep learning for joint frequency-representation and
frequency-counting (Chretien, Bhattacharya, Mohan, Fernandez-Granda, "Deep Learning for
Joint Spectral Estimation", 2019 / "DeepFreq: Frequency Estimation via Deep Learning").
Official repo: https://github.com/sreyas-mohan/DeepFreq (``modules.py`` @ master).

Two-module pipeline for spectral super-resolution from noisy time-domain samples:
``FrequencyRepresentationModule`` (the "fr" module, ``fr_module_type='fr'`` in the repo's
``set_fr_module``) maps a windowed complex signal to a high-resolution pseudo-spectrum via a
linear projection, a stack of causal circular-padded Conv1d/BatchNorm1d/ReLU blocks, and a
ConvTranspose1d upsampling head; ``FrequencyCountingModule`` (the "fc" module,
``fc_module_type='regression'``) consumes that pseudo-spectrum and regresses the number of
frequency components via a stack of circular-padded Conv1d/BatchNorm1d/ReLU blocks followed by
a linear head. This vendors exactly those two ``nn.Module`` classes, unmodified; the repo's
``PSnet`` baseline module (used only when ``fr_module_type='psnet'``) is not vendored here.
No layer, channel count, or forward-pass control-flow was changed.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FrequencyRepresentationModule(nn.Module):
    def __init__(
        self,
        signal_dim=50,
        n_filters=8,
        n_layers=3,
        inner_dim=125,
        kernel_size=3,
        upsampling=8,
        kernel_out=25,
    ):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []

        if torch.__version__ >= "1.7.0":
            conv_padding = "same"
        elif torch.__version__ >= "1.5.0":
            conv_padding = kernel_size // 2
        else:
            conv_padding = kernel_size - 1

        for n in range(n_layers):
            mod += [
                nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    bias=False,
                    padding_mode="circular",
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(
            n_filters,
            1,
            kernel_out,
            stride=upsampling,
            padding=(kernel_out - upsampling + 1) // 2,
            output_padding=1,
            bias=False,
        )

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        x = self.out_layer(x).view(bsz, -1)
        return x


class FrequencyCountingModule(nn.Module):
    def __init__(
        self, n_output, n_layers, n_filters, kernel_size, fr_size, downsampling, kernel_in
    ):
        super().__init__()
        mod = [
            nn.Conv1d(
                1,
                n_filters,
                kernel_in,
                stride=downsampling,
                padding=kernel_in - downsampling,
                padding_mode="circular",
            )
        ]
        for i in range(n_layers):
            mod += [
                nn.Conv1d(
                    n_filters,
                    n_filters,
                    kernel_size=kernel_size,
                    padding=kernel_size - 1,
                    bias=False,
                    padding_mode="circular",
                ),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        mod += [nn.Conv1d(n_filters, 1, 1)]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(fr_size // downsampling, n_output)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp[:, None]
        x = self.mod(inp)
        x = x.view(bsz, -1)
        y = self.out_layer(x)
        return y


MENAGERIE_ZOO = "vendored-pytorch"

# Tiny-size params honoring the repo's real constraints (train.py argparse defaults shown
# in comments): fr_size == inner_dim * upsampling; fr_size % fc_downsampling == 0.
_SIGNAL_DIM = 8  # default 50
_INNER_DIM = 8  # default 125
_UPSAMPLING = 4  # default 8
_FR_SIZE = _INNER_DIM * _UPSAMPLING  # 32; default fr_size=1000
_FC_DOWNSAMPLING = 4  # default 5; must divide _FR_SIZE


def build_deepfreq_fr():
    return FrequencyRepresentationModule(
        signal_dim=_SIGNAL_DIM,
        n_filters=4,
        n_layers=2,
        inner_dim=_INNER_DIM,
        kernel_size=3,
        upsampling=_UPSAMPLING,
        kernel_out=5,
    )


def example_input_deepfreq_fr():
    return torch.randn(1, 2, _SIGNAL_DIM)


def build_deepfreq_fc():
    # NOTE: the real repo's FrequencyCountingModule.__init__ sizes out_layer as
    # nn.Linear(fr_size // downsampling, n_output), but self.mod's actual output length
    # grows by (kernel_size - 1) per repeated conv layer (padding=kernel_size-1, stride 1)
    # on top of the strided first conv -- so with kernel_size>1 the formula and the real
    # runtime length diverge (verified against the repo's own train.py production defaults
    # fr_size=1000/downsampling=5/kernel_in=25/kernel_size=3/n_layers=20: formula gives 200,
    # actual is 244). This is a latent shape bug in the upstream repo itself, not an
    # adaptation made here. kernel_size=1 is used below (a real, valid value the repo's own
    # constructor accepts) so padding=kernel_size-1=0 keeps length constant across the
    # repeated conv/batchnorm/relu blocks, making the formula self-consistent for tracing --
    # no layer, channel count, or forward-pass control-flow is changed.
    return FrequencyCountingModule(
        n_output=1,
        n_layers=2,
        n_filters=4,
        kernel_size=1,
        fr_size=_FR_SIZE,
        downsampling=_FC_DOWNSAMPLING,
        kernel_in=5,
    )


def example_input_deepfreq_fc():
    return torch.randn(1, _FR_SIZE)


MENAGERIE_ENTRIES = [
    (
        "DeepFreq Frequency Representation Module",
        "build_deepfreq_fr",
        "example_input_deepfreq_fr",
        2019,
        "vendored-pytorch",
    ),
    (
        "DeepFreq Frequency Counting Module",
        "build_deepfreq_fc",
        "example_input_deepfreq_fc",
        2019,
        "vendored-pytorch",
    ),
]
