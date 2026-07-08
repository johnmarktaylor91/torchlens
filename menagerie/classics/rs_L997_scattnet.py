# FAITHFUL PORT of lehaifeng/SCAttNet @ master (original framework: TensorFlow 1.x)
#
# SCAttNet: Semantic Segmentation Network with Spatial and Channel Attention
# Mechanism for High-Resolution Remote Sensing Images (Xu, Fan, Cheng et al.,
# IEEE GRSL 2020, https://ieeexplore.ieee.org/document/8853134). Upstream repo
# only has TensorFlow 1.x source (`tf.variable_scope`, `tf.contrib.layers`,
# `tf.get_variable`) which cannot run on the TF>=2 stack installed here
# (`tf.contrib` was removed entirely in TF2) -- so this is a faithful,
# mechanism-for-mechanism transcription of the real upstream code:
#   - scattnet.py: `channel_spatial_block` / `channel_attention` /
#     `spatial_attention` (CBAM, Woo et al. 2018) + `encoder` / `decoder`
#     (SegNet-style, He et al. 2015: 5 VGG-style conv blocks with
#     max-pool-with-argmax downsampling, mirrored decoder with
#     max-unpooling using the saved indices) + `inference` (top-level
#     encoder -> decoder -> CBAM -> final 1x1 conv).
#   - ops.py: `conv2d` (conv + bias + activation), `batch_norm`
#     (moving-average BN), `relu`, `maxpool2d_with_argmax`, `maxunpool2d`.
#
# Every op present in the real TF graph has a 1:1 PyTorch counterpart here:
# conv2d+bias -> nn.Conv2d, batch_norm -> nn.BatchNorm2d, maxpool_with_argmax
# -> F.max_pool2d(..., return_indices=True), maxunpool2d -> F.max_unpool2d.
# train.py fixes batch_size=16, image=256x256x3, label classes=6 (ISPRS
# Vaihingen-style 6-class land-cover); the CBAM reduction ratio defaults to 8
# in ops.py's channel_attention. No architectural liberties were taken.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """channel_attention() in ops.py -- CBAM channel-attention submodule.

    TF used two `tf.layers.dense` ops (`mlp_0`, `mlp_1`) with weights shared
    (`reuse=True`) between the avg-pool and max-pool branches -- equivalent to
    one shared 2-layer MLP applied to both pooled descriptors, which is what
    a single `nn.Sequential` with shared parameters gives here.
    """

    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.mlp_0 = nn.Linear(channels, hidden)
        self.mlp_1 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (N, C, H, W)
        avg_pool = x.mean(dim=(2, 3))
        max_pool = x.amax(dim=(2, 3))

        avg_out = self.mlp_1(F.relu(self.mlp_0(avg_pool)))
        max_out = self.mlp_1(F.relu(self.mlp_0(max_pool)))

        scale = torch.sigmoid(avg_out + max_out)[:, :, None, None]
        return scale, x * scale


class SpatialAttention(nn.Module):
    """spatial_attention() in ops.py -- CBAM spatial-attention submodule."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.amax(dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        concat = torch.sigmoid(self.conv(concat))
        return concat, x * concat


class ChannelSpatialBlock(nn.Module):
    """channel_spatial_block() in scattnet.py -- CBAM: channel attention then
    spatial attention, applied sequentially."""

    def __init__(self, channels, ratio=8):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, x):
        _, x = self.channel_attention(x)
        _, x = self.spatial_attention(x)
        return x


class ConvBNReLU(nn.Module):
    """One (conv2d -> batch_norm -> relu) triple as used inside n_enc_block /
    n_dec_block in scattnet.py. `ops.conv2d` always uses SAME padding and a
    bias term; `ops.batch_norm` is standard batchnorm (moving-average
    running stats, matching nn.BatchNorm2d's default behavior)."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class NEncBlock(nn.Module):
    """n_enc_block() in scattnet.py -- n conv-bn-relu layers then
    max-pool-with-argmax (kernel=2, stride=2, SAME padding)."""

    def __init__(self, in_ch, out_ch, n):
        super().__init__()
        layers = []
        cur_in = in_ch
        for _ in range(n):
            layers.append(ConvBNReLU(cur_in, out_ch))
            cur_in = out_ch
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        pooled, indices = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        return pooled, indices, x.shape


class Encoder(nn.Module):
    """encoder() in scattnet.py -- 5 VGG-style blocks (SegNet encoder),
    channel progression 64/128/256/512/512, block depths 2/2/3/3/3."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.block_1 = NEncBlock(in_channels, 64, n=2)
        self.block_2 = NEncBlock(64, 128, n=2)
        self.block_3 = NEncBlock(128, 256, n=3)
        self.block_4 = NEncBlock(256, 512, n=3)
        self.block_5 = NEncBlock(512, 512, n=3)

    def forward(self, x):
        h, mask_1, shape_1 = self.block_1(x)
        h, mask_2, shape_2 = self.block_2(h)
        h, mask_3, shape_3 = self.block_3(h)
        h, mask_4, shape_4 = self.block_4(h)
        h, mask_5, shape_5 = self.block_5(h)
        masks = [mask_5, mask_4, mask_3, mask_2, mask_1]
        shapes = [shape_5, shape_4, shape_3, shape_2, shape_1]
        return h, masks, shapes


class NDecBlock(nn.Module):
    """n_dec_block() in scattnet.py -- max-unpool then n conv-bn-relu
    layers; the last layer of a block halves the channel count when
    `adj_k=True` (matching the TF `k / 2` on the final conv of blocks
    5->4, 4->3, 3->2, 2->1)."""

    def __init__(self, in_ch, out_ch, n, adj_k):
        super().__init__()
        layers = []
        cur_in = in_ch
        for i in range(n):
            is_last = i == n - 1
            cur_out = out_ch // 2 if (is_last and adj_k) else out_ch
            layers.append(ConvBNReLU(cur_in, cur_out))
            cur_in = cur_out
        self.layers = nn.ModuleList(layers)
        self.out_channels = cur_in

    def forward(self, x, mask, output_size):
        x = F.max_unpool2d(x, mask, kernel_size=2, stride=2, output_size=output_size)
        for layer in self.layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    """decoder() in scattnet.py -- mirror of the encoder using max-unpool
    (with the encoder's saved argmax indices) followed by the CBAM block
    and a final 1x1 conv to `num_classes` channels."""

    def __init__(self, num_classes=6):
        super().__init__()
        self.block_5 = NDecBlock(512, 512, n=3, adj_k=False)
        self.block_4 = NDecBlock(512, 512, n=3, adj_k=True)
        self.block_3 = NDecBlock(256, 256, n=3, adj_k=True)
        self.block_2 = NDecBlock(128, 128, n=2, adj_k=True)
        self.block_1 = NDecBlock(64, 64, n=2, adj_k=True)
        self.csb = ChannelSpatialBlock(self.block_1.out_channels, ratio=8)
        self.last_conv = nn.Conv2d(self.block_1.out_channels, num_classes, kernel_size=1)

    def forward(self, x, masks, shapes):
        h = self.block_5(x, masks[0], shapes[0][2:])
        h = self.block_4(h, masks[1], shapes[1][2:])
        h = self.block_3(h, masks[2], shapes[2][2:])
        h = self.block_2(h, masks[3], shapes[3][2:])
        h = self.block_1(h, masks[4], shapes[4][2:])
        h = self.csb(h)
        logits = self.last_conv(h)
        return logits


class SCAttNet(nn.Module):
    """inference() in scattnet.py -- top-level SegNet + CBAM segmentation
    network for high-resolution remote-sensing imagery."""

    def __init__(self, in_channels=3, num_classes=6):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(num_classes=num_classes)

    def forward(self, x):
        h, masks, shapes = self.encoder(x)
        logits = self.decoder(h, masks, shapes)
        return logits


MENAGERIE_ZOO = "ported-pytorch"


def build_scattnet():
    # train.py: img placeholder is [batch, 256, 256, 3], label classes=6.
    return SCAttNet(in_channels=3, num_classes=6)


def example_input_scattnet():
    return (torch.randn(1, 3, 64, 64),)


MENAGERIE_ENTRIES = [
    (
        "SCAttNet",
        "build_scattnet",
        "example_input_scattnet",
        2020,
        "ported-pytorch",
    ),
]
