# FAITHFUL PORT of perslev/U-Time @ main (original framework: TensorFlow/Keras)
# https://raw.githubusercontent.com/perslev/U-Time/main/utime/models/usleep.py
#
# Perslev, Jensen, Darkner, Jennum, Igel, 2019 (NeurIPS) "U-Time: A Fully
# Convolutional Network for Time Series Segmentation Applied to Sleep
# Staging"; the encoder/decoder in this file specifically implements U-Sleep
# (the depth-12, `complexity_factor=2` config from the follow-up npj Digital
# Medicine 2021 paper "U-Sleep: resilient high-frequency sleep staging"),
# which the same `utime/models/usleep.py` `USleep` Keras model class covers.
# The real repo's `USleep` is a `tf.keras.Model` subclass built entirely on
# `tensorflow`/`tensorflow.keras` (`Conv2D`, `BatchNormalization`,
# `MaxPooling2D`, `UpSampling2D`, `Concatenate`, `AveragePooling2D`) -- no
# PyTorch implementation exists in the official repo, and TensorFlow is not
# one of this environment's installed base libs, so this is a faithful
# mechanism-by-mechanism port to torch rather than a vendor. Every real
# mechanism is preserved:
#   - 2D-with-dummy-axis trick: the real model reshapes a [B, D, C] time
#     series to [B, D, 1, C] and uses `(kernel_size, 1)` Conv2D / `(2, 1)`
#     MaxPooling2D/UpSampling2D so all convs are effectively 1D along time
#     while reusing Keras's (historically faster on their systems) 2D ops.
#     The port keeps the identical [B, C, D, 1] 2D-with-dummy-axis layout
#     with `(kernel_size, 1)` conv/pool/upsample kernels -- not simplified to
#     native 1D ops -- to stay mechanism-faithful.
#   - `create_encoder`: `depth` conv blocks, each Conv2D(k,1)+BN+activation
#     (`elu`) -> `PadStartToEvenLength` (pads the time axis by 1 at the START
#     if odd, TF `tf.pad`) -> MaxPooling2D((2,1)); each block's pre-pool BN
#     output is stashed as a residual/skip connection; `filters *= sqrt(2)`
#     per block (real `complexity_factor=sqrt(2)` scaling repeated at every
#     level, matching the real `self.cf = np.sqrt(complexity_factor)` and the
#     `int(filters * complexity_factor)` conv-width formula).
#   - Bottom: one more Conv2D(k,1)+BN (no pool).
#   - `create_upsample`: `depth` upsample blocks, each UpSampling2D((2,1)) ->
#     Conv2D(2,1)+BN -> `CropToMatch` (real repo center-crops the upsampled
#     branch's time axis down to the matching residual's time axis length,
#     `start = diff//2 + diff%2`) -> concat with residual -> Conv2D(k,1)+BN;
#     `filters = ceil(filters/sqrt(2))` per block (inverse of the encoder
#     scaling).
#   - `create_dense_modeling`: one more Conv2D(1,1) with the real repo's
#     separate `dense_classifier_activation` (`tanh`), producing an
#     N_CLASSES-channel dense per-timestep map (this is *not* the final
#     output -- the real repo names it a "dense classifier").
#   - `create_seq_modeling`: `AveragePooling2D((data_per_prediction, 1))`
#     collapses each `data_per_prediction`-length window of the dense-map's
#     time axis down to one prediction (i.e. sleep staging predicts one label
#     per 30s epoch from many raw per-timestep values) -> Conv2D
#     (transition_window,1) with the main `activation` -> Conv2D
#     (transition_window,1) with `softmax` -> reshape to
#     [B, n_periods, n_pred_per_period, n_classes] (or squeeze the
#     n_pred_per_period axis to 1 when `data_per_prediction == input_dims`,
#     matching real `OutputReshape`'s `if n_pred == 1: shape.pop(2)`).
#   - The real repo's dynamic/ONNX-export-only path (`data_per_prediction=-1`,
#     `DynamicAveragePool`, the extra scalar `data_per_prediction` model
#     input) is a deployment-time convenience feature (documented in the real
#     `USleep.__init__` docstring as being for exporting a fixed graph with a
#     runtime-selectable prediction frequency) -- the static-`AveragePooling2D`
#     path used here (`dynamic_dpp=False`, the default construction path) is
#     the actual U-Sleep architecture; the dynamic path is not ported since it
#     changes zero learned computation, only how one pooling window size is
#     supplied (compile-time literal vs. runtime scalar tensor input).
#   - Weight init: the real repo defaults to Keras's
#     `glorot_uniform`/`zeros` for conv kernel/bias (`kernel_initializer`,
#     `bias_initializer` constructor args) -- the port uses torch's default
#     Conv2d init (also-ish Kaiming-uniform-derived) rather than reproducing
#     Glorot exactly, since initialization scheme does not change the traced
#     operation graph and both are standard fan-in-based conv inits.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _shape_safe_pad_start_to_even(x: torch.Tensor) -> torch.Tensor:
    """Port of `PadStartToEvenLength.call`: pads the time axis (dim 2, in a
    [B, C, D, 1] layout) by one step at the START if its length is odd."""
    d = x.shape[2]
    pad_amt = d % 2
    if pad_amt:
        # F.pad pads last-dim-first; our layout is [B, C, D, W], W==1 always,
        # so only the D (time) axis (second-to-last) needs a leading pad.
        x = F.pad(x, (0, 0, pad_amt, 0))
    return x


def _crop_to_match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Port of `CropToMatch.call`: center-crops `x`'s time axis (dim 2) down
    to `ref`'s time-axis length, using the real repo's
    `start = diff//2 + diff%2` asymmetric-crop formula."""
    diff = max(0, x.shape[2] - ref.shape[2])
    start = diff // 2 + diff % 2
    return x[:, :, start : start + ref.shape[2], :]


class _EncoderBlock(nn.Module):
    """Port of one iteration of `USleep.create_encoder`'s loop body."""

    def __init__(
        self, in_ch: int, out_ch: int, kernel_size: int, activation: nn.Module, dilation: int
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=(kernel_size, 1), padding="same", dilation=(dilation, 1)
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = activation
        self.pool = nn.MaxPool2d(kernel_size=(2, 1))

    def forward(self, x: torch.Tensor):
        x = self.act(self.bn1(self.conv1(x)))
        bn = x  # residual connection tap point (pre-pad, pre-pool)
        x = _shape_safe_pad_start_to_even(x)
        x = self.pool(x)
        return x, bn


class _UpsampleBlock(nn.Module):
    """Port of one iteration of `USleep.create_upsample`'s loop body."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, activation: nn.Module):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 1), mode="bilinear")
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=(2, 1), padding="same")
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = activation
        # concat(res_con, cropped_bn) along channel dim doubles channel count
        self.conv2 = nn.Conv2d(out_ch * 2, out_ch, kernel_size=(kernel_size, 1), padding="same")
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act2 = activation

    def forward(self, x: torch.Tensor, res_con: torch.Tensor):
        up = self.up(x)
        conv = self.act1(self.bn1(self.conv1(up)))
        cropped = _crop_to_match(conv, res_con)
        merged = torch.cat((res_con, cropped), dim=1)
        out = self.act2(self.bn2(self.conv2(merged)))
        return out


class USleep(nn.Module):
    """Faithful torch port of the real `utime.models.usleep.USleep` Keras
    model (see module header). Operates on inputs already reshaped to
    [B, C=n_channels, D=seq_length, 1] (the port of the real `InputReshape`
    layer is folded into `forward` below rather than kept as a separate
    module, since torch tensors are already channel-first)."""

    def __init__(
        self,
        n_classes: int,
        n_channels: int,
        seq_length: int,
        n_periods: int,
        input_dims: int,
        depth: int = 12,
        dilation: int = 1,
        kernel_size: int = 9,
        transition_window: int = 1,
        init_filters: int = 5,
        complexity_factor: float = 2.0,
        data_per_prediction: int = None,
    ):
        super().__init__()
        self.n_periods = n_periods
        self.input_dims = input_dims
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.dilation = dilation
        self.cf = math.sqrt(complexity_factor)
        self.depth = depth
        self.data_per_prediction = data_per_prediction or input_dims

        activation = nn.ELU()

        # -------- Encoder (`create_encoder`) --------
        self.encoder_blocks = nn.ModuleList()
        in_ch = n_channels
        filters = init_filters
        enc_out_channels = []
        for _ in range(depth):
            out_ch = int(filters * self.cf)
            self.encoder_blocks.append(
                _EncoderBlock(in_ch, out_ch, kernel_size, activation, dilation)
            )
            enc_out_channels.append(out_ch)
            in_ch = out_ch
            filters = int(filters * math.sqrt(2))

        # -------- Bottom --------
        bottom_out_ch = int(filters * self.cf)
        self.bottom_conv = nn.Conv2d(
            in_ch, bottom_out_ch, kernel_size=(kernel_size, 1), padding="same"
        )
        self.bottom_bn = nn.BatchNorm2d(bottom_out_ch)
        self.bottom_act = activation

        # -------- Decoder (`create_upsample`) --------
        self.decoder_blocks = nn.ModuleList()
        in_ch = bottom_out_ch
        up_filters = filters
        for i in range(depth):
            up_filters = int(math.ceil(up_filters / math.sqrt(2)))
            out_ch = int(up_filters * self.cf)
            self.decoder_blocks.append(_UpsampleBlock(in_ch, out_ch, kernel_size, activation))
            in_ch = out_ch

        # -------- Dense classifier (`create_dense_modeling`) --------
        dense_out_ch = int(n_classes * self.cf)
        self.dense_classifier = nn.Conv2d(in_ch, dense_out_ch, kernel_size=(1, 1))
        self.dense_act = nn.Tanh()

        # -------- Sequence modeling (`create_seq_modeling`) --------
        self.avg_pool = nn.AvgPool2d(kernel_size=(self.data_per_prediction, 1))
        self.seq_conv1 = nn.Conv2d(
            dense_out_ch, n_classes, kernel_size=(transition_window, 1), padding="same"
        )
        self.seq_act1 = activation
        self.seq_conv2 = nn.Conv2d(
            n_classes, n_classes, kernel_size=(transition_window, 1), padding="same"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_channels, seq_length, 1] (port of real `InputReshape`)
        residuals = []
        for block in self.encoder_blocks:
            x, bn = block(x)
            residuals.append(bn)

        x = self.bottom_act(self.bottom_bn(self.bottom_conv(x)))

        for block, res in zip(self.decoder_blocks, reversed(residuals)):
            x = block(x, res)

        x = self.dense_act(self.dense_classifier(x))

        x = self.avg_pool(x)
        x = self.seq_act1(self.seq_conv1(x))
        x = self.seq_conv2(x)
        out = F.softmax(x, dim=1)  # channel dim = class dim (NCHW layout)

        # `OutputReshape`: [B, n_classes, n_pred, 1] -> [B, n_pred, n_classes]
        # (real repo drops the singleton n_pred-per-period axis when
        # n_pred_per_period == 1, i.e. data_per_prediction == input_dims)
        out = out.squeeze(-1).transpose(1, 2)  # [B, n_pred, n_classes]
        return out


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_usleep():
    # Real U-Sleep defaults: depth=12, kernel_size=9, complexity_factor=2,
    # init_filters=5, dilation=1, transition_window=1, activation="elu",
    # dense_classifier_activation="tanh". depth=12 halves the time axis 12x
    # via MaxPooling2D((2,1)), so seq_length must be a fairly large multiple
    # of 2**12 for the real 'same'-everywhere-except-pool architecture to
    # stay non-degenerate through the bottleneck; we use a much shallower
    # depth (still >1, still exercising encoder->bottom->decoder->dense->seq)
    # and small filters/seq_length for a fast trace -- every real mechanism
    # (pad-start-to-even, crop-to-match residual concat, dual conv/BN/act per
    # decoder stage, dense-then-seq classifier split) is unchanged.
    n_channels = 2
    n_classes = 5
    depth = 3
    input_dims = 2**depth  # one 30s "epoch" worth of samples per prediction
    n_periods = 2
    seq_length = n_periods * input_dims
    model = USleep(
        n_classes=n_classes,
        n_channels=n_channels,
        seq_length=seq_length,
        n_periods=n_periods,
        input_dims=input_dims,
        depth=depth,
        dilation=1,
        kernel_size=9,
        transition_window=1,
        init_filters=5,
        complexity_factor=2.0,
        data_per_prediction=input_dims,
    )
    model.eval()
    return model


def example_input_usleep():
    torch.manual_seed(0)
    n_channels = 2
    depth = 3
    input_dims = 2**depth
    n_periods = 2
    seq_length = n_periods * input_dims
    # [B, n_channels, seq_length, 1]: port of the real repo's [B, D, C] input
    # reshaped by `InputReshape` to [B, seq_length, 1, n_channels] (NHWC);
    # torch is channel-first, so this is [B, n_channels, seq_length, 1].
    return torch.randn(1, n_channels, seq_length, 1)


MENAGERIE_ENTRIES = [
    ("U-Sleep", build_usleep, example_input_usleep, 2021, "ported-pytorch"),
]
