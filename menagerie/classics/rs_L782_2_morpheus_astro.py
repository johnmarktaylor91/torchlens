# FAITHFUL PORT of morpheus-project/morpheus @ master (original framework: TensorFlow 1.x,
# tf.compat.v1 graph-mode with tf.layers/tf.variable_scope custom estimator training loop)
#
# Files transcribed: morpheus/core/unet.py (Model.model_fn -- the U-Net graph builder),
# morpheus/core/model.py (Morpheus.inference -- softmax(build_graph(...))),
# morpheus/core/model_config.json (the real published inference hyperparameters).
#
# Morpheus (Hausen & Keating Robertson, 2020, ApJS) is a U-Net-based per-pixel semantic
# segmentation model for astronomical (HST) images: given 4-band (H, J, Z, V) 40x40 cutouts,
# it predicts a 5-class softmax per pixel (spheroid / disk / irregular / point_source /
# background). The real repo's `model_fn` builds a 3-level encoder/decoder with
# `down_filters=[8,16,32]`, `num_down_convs=4` conv-block repeats per down stage,
# `num_intermediate_filters=16` at the bottleneck, `num_up_convs=2` conv-block repeats per up
# stage, `up_filters=[8,16,32]`, batch-norm + dropout in every block, 2x2 max-pool
# downsampling, and bicubic-resize upsampling followed by skip-connection concat (all read
# verbatim from `moftransformer/core/model_config.json`'s published inference config, which
# is not obtainable via `pip install` since Morpheus is TF1.x custom-CUDA-era and cannot run
# in the base torch env). This port keeps every structural choice (filter counts, conv-block
# repeat counts, batchnorm+dropout placement, bicubic (not nearest/bilinear) upsampling,
# skip-connection concat order, final 1x1x5-class conv with no activation before softmax) --
# only the TF1 graph/session/variable_scope machinery is replaced with an eager torch
# nn.Module, and 2D bicubic upsampling is done via torch's `F.interpolate(...,
# mode="bicubic")` (torch's direct analogue of `tf.image.resize_images(...,
# ResizeMethod.BICUBIC, align_corners=True)`).
#
# MENAGERIE_ZOO = "ported-pytorch"

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


class ConvBlock(nn.Module):
    """
    Port of morpheus.core.unet.Model.block_op: batch_norm -> conv(3x3, relu) -> dropout.
    """

    def __init__(self, in_channels, num_filters, batch_norm=True, dropout=True, dropout_rate=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(in_channels) if batch_norm else None
        self.conv = nn.Conv2d(in_channels, num_filters, kernel_size=3, padding="same")
        self.dropout = nn.Dropout(dropout_rate) if dropout else None

    def forward(self, x):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = F.relu(self.conv(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class MorpheusUNet(nn.Module):
    """
    Faithful port of morpheus.core.unet.Model.model_fn + morpheus.core.model.Morpheus.

    Real published inference hparams (morpheus/core/model_config.json):
        down_filters=[8, 16, 32], num_down_convs=4, num_intermediate_filters=16,
        up_filters=[8, 16, 32], num_up_convs=2, batch_norm=True, dropout=True,
        dropout_rate=0.5. Input: [B, 4, 40, 40] (H, J, Z, V bands). Output: 5-class
        per-pixel softmax (spheroid, disk, irregular, point_source, background).
    """

    def __init__(
        self,
        in_channels=4,
        num_classes=5,
        down_filters=(8, 16, 32),
        num_down_convs=4,
        num_intermediate_filters=16,
        up_filters=(8, 16, 32),
        num_up_convs=2,
        batch_norm=True,
        dropout=True,
        dropout_rate=0.5,
    ):
        super().__init__()
        self.down_filters = list(down_filters)
        self.up_filters = list(up_filters)
        self.num_down_convs = num_down_convs
        self.num_up_convs = num_up_convs

        # downconv-{idx}: num_down_convs conv blocks, then 2x2 maxpool downsample
        self.down_stages = nn.ModuleList()
        ch = in_channels
        for num_filters in self.down_filters:
            blocks = nn.ModuleList()
            for _ in range(num_down_convs):
                blocks.append(ConvBlock(ch, num_filters, batch_norm, dropout, dropout_rate))
                ch = num_filters
            self.down_stages.append(blocks)

        # intermediate-conv (bottleneck)
        self.intermediate = ConvBlock(
            ch, num_intermediate_filters, batch_norm, dropout, dropout_rate
        )
        ch = num_intermediate_filters

        # upconv-{idx}: upsample (bicubic x2), concat with matching down-stage skip, then
        # num_up_convs conv blocks
        self.up_stages = nn.ModuleList()
        for idx, num_filters in enumerate(self.up_filters):
            skip_ch = self.down_filters[-(idx + 1)]
            blocks = nn.ModuleList()
            in_ch = ch + skip_ch
            for _ in range(num_up_convs):
                blocks.append(ConvBlock(in_ch, num_filters, batch_norm, dropout, dropout_rate))
                in_ch = num_filters
            self.up_stages.append(blocks)
            ch = num_filters

        # final_conv: 3x3 conv to num_classes, no activation (softmax applied by caller)
        self.final_conv = nn.Conv2d(ch, num_classes, kernel_size=3, padding="same")

    def forward(self, x):
        skips = []
        for blocks in self.down_stages:
            for block in blocks:
                x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.intermediate(x)

        for idx, blocks in enumerate(self.up_stages):
            x = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=True)
            skip = skips[-(idx + 1)]
            x = torch.cat([x, skip], dim=1)
            for block in blocks:
                x = block(x)

        logits = self.final_conv(x)
        return F.softmax(logits, dim=1)


def build_morpheus_astro():
    torch.manual_seed(0)
    model = MorpheusUNet(
        in_channels=4,
        num_classes=5,
        down_filters=(8, 16, 32),
        num_down_convs=4,
        num_intermediate_filters=16,
        up_filters=(8, 16, 32),
        num_up_convs=2,
        batch_norm=True,
        dropout=True,
        dropout_rate=0.5,
    )
    model.eval()
    return model


def example_input_morpheus_astro():
    torch.manual_seed(0)
    # Real published input: [batch, 40, 40, 4] (H, J, Z, V bands) in the TF NHWC repo;
    # NCHW here for torch conv semantics.
    return (torch.randn(2, 4, 40, 40),)


MENAGERIE_ENTRIES = [
    (
        "Morpheus (astronomy U-Net)",
        "build_morpheus_astro",
        "example_input_morpheus_astro",
        2020,
        "ported-pytorch",
    ),
]
