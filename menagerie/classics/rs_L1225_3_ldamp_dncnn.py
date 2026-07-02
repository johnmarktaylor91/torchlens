# FAITHFUL PORT of ricedsp/D-AMP_Toolbox @ master (original framework: TensorFlow 1.x)
# https://github.com/ricedsp/D-AMP_Toolbox/blob/master/LDAMP_TensorFlow/LearnedDAMP.py
# https://github.com/ricedsp/D-AMP_Toolbox/blob/master/LDAMP_TensorFlow/TrainDnCNN.py
#
# Metzler, Mousavi & Baraniuk, "Learned D-AMP: Principled Neural Network based
# Compressive Image Recovery," NeurIPS 2017 (the "D-AMP" / "Learned D-AMP"
# candidate). The official toolbox's D-AMP/D-VAMP/D-prGAMP unrolled-iteration
# algorithms (`LDAMP`, `LDIT`, `LDGAMP` in `LearnedDAMP.py`) are plain
# TF1.x graph code built around `tf.placeholder`/`tf.variable_scope`/manual
# `tf.Variable` weight dictionaries -- there is no PyTorch implementation
# anywhere in this repo (`LDAMP_TensorFlow/` is the only ML backend; the
# `Packages/rwt` and root-level MATLAB code are non-NN infrastructure). Per
# the ladder, TF1.x is a deprecated framework that cannot be reasonably
# installed/run standalone here, so the actual learned network -- the
# `DnCNN(r, rvar, theta_thislayer, training)` denoiser plugged into every one
# of the unrolled algorithms via `DnCNN_wrapper` -- is faithfully transcribed
# into base-env torch, module-for-module:
#   * first layer: 3x3 conv (bias-free, per `init_vars_DnCNN`: only `weights`
#     are created, no `biases`) + ReLU
#   * middle `n_DnCNN_layers - 2` layers: 3x3 conv (bias-free) + BatchNorm2d
#     + ReLU
#   * last layer: 3x3 conv (bias-free), no activation
#   * residual output: `x_hat = r - layers[-1]` (the network predicts the
#     noise residual, matching `DnCNN(...)`'s final `x_hat = r - layers[...]`
#     line)
# Fixed hyperparameters (`filter_height=filter_width=3`, `num_filters=64`,
# default `--DnCNN_layers 16`) are taken from `TrainDnCNN.py`'s
# `argparse` defaults / `## Network Parameters` block. This is a direct
# TF-graph-op-for-op transcription (`tf.nn.conv2d` -> `nn.Conv2d(bias=False)`,
# `tf.layers.batch_normalization` -> `nn.BatchNorm2d`, `tf.nn.relu` ->
# `nn.ReLU`); no mechanism was added, removed, or reordered.

import torch
import torch.nn as nn


class LDAMPDnCNN(nn.Module):
    """Faithful port of the DnCNN denoiser used inside Learned-D-AMP/D-IT/D-GAMP
    unrolled iterations (LDAMP_TensorFlow/LearnedDAMP.py: DnCNN())."""

    def __init__(self, channel_img=1, num_filters=64, filter_size=3, n_dncnn_layers=16):
        super().__init__()
        assert n_dncnn_layers >= 2, "need at least first + last layer"
        self.n_dncnn_layers = n_dncnn_layers
        padding = filter_size // 2

        # Layer 1: conv + relu (bias-free, matches init_vars_DnCNN weights[0])
        self.conv_first = nn.Conv2d(
            channel_img, num_filters, filter_size, padding=padding, bias=False
        )
        self.relu_first = nn.ReLU(inplace=True)

        # Layers 2..N-1: conv + BN + relu (bias-free convs)
        mid_convs = []
        mid_bns = []
        for _ in range(n_dncnn_layers - 2):
            mid_convs.append(
                nn.Conv2d(num_filters, num_filters, filter_size, padding=padding, bias=False)
            )
            mid_bns.append(nn.BatchNorm2d(num_filters))
        self.mid_convs = nn.ModuleList(mid_convs)
        self.mid_bns = nn.ModuleList(mid_bns)
        self.mid_relu = nn.ReLU(inplace=True)

        # Last layer: conv only, no activation (bias-free)
        self.conv_last = nn.Conv2d(
            num_filters, channel_img, filter_size, padding=padding, bias=False
        )

    def forward(self, r):
        """
        Args:
            r (float tensor, (bs, channel_img, H, W)): noisy input image
                (the AMP algorithm's running estimate `r` reshaped to image
                shape, per `LearnedDAMP.py: DnCNN()`'s `shape4D` reshape).

        Returns:
            x_hat (float tensor, (bs, channel_img, H, W)): denoised estimate,
                computed as `r - predicted_noise` (residual learning).
        """
        h = self.relu_first(self.conv_first(r))
        for conv, bn in zip(self.mid_convs, self.mid_bns):
            h = self.mid_relu(bn(conv(h)))
        noise = self.conv_last(h)
        x_hat = r - noise
        return x_hat


def build_ldamp_dncnn():
    """Tiny-config LDAMP DnCNN: 6 layers (vs paper default 16), 8 filters
    (vs paper default 64), grayscale (channel_img=1, matching TrainDnCNN.py
    defaults)."""
    return LDAMPDnCNN(channel_img=1, num_filters=8, filter_size=3, n_dncnn_layers=6)


def example_input_ldamp_dncnn():
    return (torch.randn(2, 1, 24, 24),)


MENAGERIE_ZOO = "ported-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Learned D-AMP DnCNN Denoiser",
        "build_ldamp_dncnn",
        "example_input_ldamp_dncnn",
        2017,
        "compressive-sensing",
    ),
]
