# FAITHFUL PORT of theislab/dca @ master (original framework: Keras/TensorFlow)
# https://raw.githubusercontent.com/theislab/dca/master/dca/network.py (Autoencoder,
#   ZINBAutoencoder.build/build_output)
# https://raw.githubusercontent.com/theislab/dca/master/dca/layers.py (ColwiseMultLayer,
#   SliceLayer semantics)
#
# Eraslan et al 2019 (Nature Communications) "Single-cell RNA-seq denoising using a deep count
# autoencoder" -- DCA. The real repo is Keras/TF (`from keras.layers import ...`); TF/Keras are
# not in this base env, so this is a faithful architectural transcription of
# `network.Autoencoder.build()` + `network.ZINBAutoencoder.build_output()` (the `ae_type=
# 'zinb-conddisp'` variant -- the headline ZINB-loss configuration the DCA paper is named for)
# into base-env torch. Every layer/mechanism from the real Keras graph is reproduced:
#
#   last_hidden = count_input
#   (optional input Dropout)
#   for each hidden layer i in hidden_size=(64, 32, 64):
#       Dense(hid_size) -> BatchNorm1d(affine center-only, i.e. no learnable scale/gamma,
#           matching Keras `BatchNormalization(center=True, scale=False)`) -> activation
#           (default 'relu') -> optional Dropout
#       (the middle layer is named 'center' -- the encoder bottleneck)
#   decoder_output = last hidden layer's output
#   pi   = sigmoid(Dense(output_size)(decoder_output))                      # dropout prob head
#   disp = softplus_clipped(Dense(output_size)(decoder_output))             # dispersion head
#         DispAct(x) = clip(softplus(x), 1e-4, 1e4)
#   mean = exp_clipped(Dense(output_size)(decoder_output))                  # mean head
#         MeanAct(x) = clip(exp(x), 1e-5, 1e6)
#   output = mean * size_factors.reshape(-1, 1)   # ColwiseMultLayer -- rescale by cell size factor
#
# `SliceLayer(0, ...)` in the real code just selects `output` (index 0 of `[output, disp, pi]`)
# as `self.model`'s single Keras output tensor for `.predict()`/loss wiring -- the ZINB loss
# itself closes over `disp`/`pi` directly, they are not fed through the sliced graph output. We
# reproduce that: `forward()` returns `(output, disp, pi)`, mirroring the three tensors the real
# Keras graph actually computes (mean-rescaled reconstruction, dispersion, dropout probability).
# `l1_l2` weight regularizers and the `ConstantDispersionLayer`/`ElementwiseDense` variants used
# by OTHER `ae_type`s (`nb`, `nb-shared`, `zinb-elempi`, ...) are not part of this ('zinb-conddisp')
# variant's real graph and are correctly omitted, exactly as the real `ZINBAutoencoder.build_output`
# omits them. Class/method structure (`Autoencoder.build`, `ZINBAutoencoder.build_output`) mirrors
# the real file 1:1.

import torch
import torch.nn as nn


def _mean_act(x):
    return torch.clamp(torch.exp(x), 1e-5, 1e6)


def _disp_act(x):
    return torch.clamp(nn.functional.softplus(x), 1e-4, 1e4)


class _DCAHiddenBlock(nn.Module):
    """One Dense -> BatchNorm1d(center-only) -> activation -> Dropout stage of
    Autoencoder.build()'s hidden_size loop (real network.py lines ~101-138)."""

    def __init__(self, in_features, out_features, dropout=0.0, batchnorm=True):
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.batchnorm = batchnorm
        if batchnorm:
            # Keras BatchNormalization(center=True, scale=False): learnable beta (shift),
            # no learnable gamma (scale) -- affine=True but gamma fixed at 1 (not trained).
            self.bn = nn.BatchNorm1d(out_features, affine=True)
            self.bn.weight.requires_grad_(False)
            nn.init.ones_(self.bn.weight)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x):
        x = self.dense(x)
        if self.batchnorm:
            x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class ZINBAutoencoder(nn.Module):
    """Faithful port of network.Autoencoder.build() + network.ZINBAutoencoder.build_output()
    (ae_type='zinb-conddisp')."""

    def __init__(
        self,
        input_size,
        output_size=None,
        hidden_size=(64, 32, 64),
        hidden_dropout=0.0,
        input_dropout=0.0,
        batchnorm=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size if output_size is not None else input_size
        self.hidden_size = hidden_size
        if isinstance(hidden_dropout, (list, tuple)):
            assert len(hidden_dropout) == len(hidden_size)
            hidden_dropout = list(hidden_dropout)
        else:
            hidden_dropout = [hidden_dropout] * len(hidden_size)

        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0.0 else None

        blocks = []
        in_features = self.input_size
        for hid_size, hid_drop in zip(hidden_size, hidden_dropout):
            blocks.append(
                _DCAHiddenBlock(in_features, hid_size, dropout=hid_drop, batchnorm=batchnorm)
            )
            in_features = hid_size
        self.hidden_blocks = nn.ModuleList(blocks)

        decoder_output_size = hidden_size[-1] if len(hidden_size) > 0 else self.input_size
        self.pi_head = nn.Linear(decoder_output_size, self.output_size)
        self.disp_head = nn.Linear(decoder_output_size, self.output_size)
        self.mean_head = nn.Linear(decoder_output_size, self.output_size)

    def forward(self, count, size_factors):
        last_hidden = count
        if self.input_dropout is not None:
            last_hidden = self.input_dropout(last_hidden)
        for block in self.hidden_blocks:
            last_hidden = block(last_hidden)
        decoder_output = last_hidden

        pi = torch.sigmoid(self.pi_head(decoder_output))
        disp = _disp_act(self.disp_head(decoder_output))
        mean = _mean_act(self.mean_head(decoder_output))

        output = mean * size_factors.reshape(-1, 1)
        return output, disp, pi


def build_dca():
    model = ZINBAutoencoder(
        input_size=20,
        output_size=20,
        hidden_size=(8, 4, 8),
        hidden_dropout=0.0,
        input_dropout=0.0,
        batchnorm=True,
    )
    model.eval()
    return model


def example_input_dca():
    count = torch.rand(6, 20) * 5.0
    size_factors = torch.rand(6) + 0.5
    return (count, size_factors)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    ("DCA", "build_dca", "example_input_dca", 2019, "ported"),
]
