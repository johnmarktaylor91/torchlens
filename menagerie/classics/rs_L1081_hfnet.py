# FAITHFUL PORT of ethz-asl/hfnet @ master (original framework: TensorFlow 1.x + tf.contrib.slim)
# https://github.com/ethz-asl/hfnet
# Ported files: hfnet/models/hf_net.py (HfNet._model, MOBILENET_DEF), hfnet/models/utils/layers.py
# (vlad, image_normalization), hfnet/models/backbones/mobilenet_v2.py (V2_DEF/mobilenet op spec),
# hfnet/models/backbones/utils/conv_blocks.py (expanded_conv).
# arXiv:1812.03506 (Sarlin et al., "From Coarse to Fine: Robust Hierarchical Localization at
# Large Scale", CVPR 2019) -- HF-Net.
#
# HF-Net is a *single shared* MobileNetV2 encoder with two heads read off two different
# intermediate depths: a SuperPoint-style local head (keypoint detector + dense descriptor
# map) tapped at `layer_7`, and a NetVLAD-style global-descriptor head tapped at `layer_18`.
# This ONE-ENCODER-TWO-HEADS-AT-DIFFERENT-DEPTHS design (for joint local+global feature
# extraction / hierarchical localization) is HF-Net's actual architectural contribution --
# not merely "MobileNet + a head", so this is ported as HF-Net's own model, not built from
# an unrelated base-lib class. tf.contrib.slim / tf1.x placeholders / `tf.variable_scope`
# cannot run in a modern TF or torch-only environment, so the graph is transcribed to torch.
#
# Faithfulness notes (framework-level substitutions with unchanged semantics):
#   - `MOBILENET_DEF` (hf_net.py) is HF-Net's OWN mobilenet spec (channel counts diverge
#     from stock MobileNetV2's `V2_DEF` starting at op index 6 -- HF-Net inserts an extra
#     64->128 "branch here" stage) -- reproduced exactly here as CFG, not the stock
#     torchvision channel list, so `layer_7`/`layer_18` taps land on the correct tensors.
#   - `expanded_conv` (conv_blocks.py): expansion(1x1, expand_ratio) -> depthwise(3x3,
#     stride) -> projection(1x1), + residual iff stride==1 and channels match -> ported as
#     `InvertedResidual`, the same structure as MobileNetV2's canonical block.
#   - `depth_multiplier`/`_make_divisible` (mobilenet.py): channel count =
#     round_to_divisible(num_outputs * depth_multiplier, 8, min=8) -> `_make_divisible` below,
#     transcribed verbatim (depth_multiplier=1.0 here, so it is a no-op at this width).
#   - `local_head` (hf_net.py): descriptor 3x3 conv -> 1x1 conv -> l2-normalize; detector
#     3x3 conv -> 1x1 conv (channels = grid^2 + 1 dustbin) -> softmax -> drop dustbin ->
#     depth_to_space(grid) to get a dense per-pixel score map -> ported 1:1 as `LocalHead`.
#   - `vlad`/NetVLAD (layers.py): 1x1 conv -> softmax cluster memberships; learned cluster
#     centers broadcast-subtracted from the feature map, weighted by membership, summed
#     spatially, intra-normalized per cluster then L2-normalized as a whole -> ported 1:1
#     as `NetVLAD`.
#   - `image_normalization`: `(x - 128) / 128` -> unchanged.
#   - TF NHWC -> torch NCHW throughout (pure layout change, no semantic difference).
#   - Inference-only losses/keypoint-extraction/NMS/descriptor-sampling (`Mode.PRED` branch,
#     `simple_nms`, `tf.contrib.resampler`) are training/export-time postprocessing outside
#     the differentiable trunk and are not part of the traced forward graph here, matching
#     how a plain eager forward pass over the model would be captured.

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_divisible(v, divisor=8, min_value=8):
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    """Faithful port of `expanded_conv` (conv_blocks.py): expansion(1x1) -> depthwise(3x3,
    stride) -> projection(1x1, linear), with a residual add iff stride==1 and channels match."""

    def __init__(self, in_ch, out_ch, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = _make_divisible(in_ch * expand_ratio, 8) if expand_ratio != 1 else in_ch
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU6(in_ch, hidden_dim, kernel_size=1))
        layers.extend(
            [
                ConvBNReLU6(
                    hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim
                ),
                nn.Conv2d(hidden_dim, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        if self.use_residual:
            out = out + x
        return out


# HF-Net's own MOBILENET_DEF spec (hf_net.py), as (kind, stride, out_channels, expand_ratio).
# The first expanded_conv uses expand_ratio=1 (expand_input_by_factor(1, divisible_by=1));
# all subsequent expanded_conv ops use the arg_scope default expand_ratio=6.
HFNET_MOBILENET_SPEC = [
    ("conv", 2, 32, None),
    ("expanded_conv", 1, 16, 1),
    ("expanded_conv", 2, 24, 6),
    ("expanded_conv", 1, 24, 6),
    ("expanded_conv", 2, 32, 6),
    ("expanded_conv", 1, 64, 6),
    ("expanded_conv", 1, 128, 6),  # layer_7 -- HF-Net's local_endpoint tap
    ("expanded_conv", 2, 64, 6),
    ("expanded_conv", 1, 64, 6),
    ("expanded_conv", 1, 64, 6),
    ("expanded_conv", 1, 64, 6),
    ("expanded_conv", 1, 96, 6),
    ("expanded_conv", 1, 96, 6),
    ("expanded_conv", 1, 96, 6),
    ("expanded_conv", 2, 160, 6),
    ("expanded_conv", 1, 160, 6),
    ("expanded_conv", 1, 160, 6),
    ("expanded_conv", 1, 320, 6),  # layer_18 -- HF-Net's global_endpoint tap
    ("conv1x1", 1, 1280, None),
]

LOCAL_ENDPOINT_LAYER = 7  # 1-indexed, matches TF-slim `layer_7`
GLOBAL_ENDPOINT_LAYER = 18  # 1-indexed, matches TF-slim `layer_18`


class MobileNetV2Encoder(nn.Module):
    """Faithful port of HF-Net's shared MobileNetV2 trunk (mobilenet_v2.py / hf_net.py
    MOBILENET_DEF), exposing the two intermediate endpoints HF-Net taps its heads from."""

    def __init__(self, in_ch=1, depth_multiplier=1.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = in_ch
        for kind, stride, out_c, expand_ratio in HFNET_MOBILENET_SPEC:
            out_c_scaled = _make_divisible(out_c * depth_multiplier, 8, 8)
            if kind == "conv":
                self.blocks.append(ConvBNReLU6(c_in, out_c_scaled, kernel_size=3, stride=stride))
            elif kind == "conv1x1":
                self.blocks.append(ConvBNReLU6(c_in, out_c_scaled, kernel_size=1, stride=stride))
            else:  # expanded_conv
                self.blocks.append(InvertedResidual(c_in, out_c_scaled, stride, expand_ratio))
            c_in = out_c_scaled
        self.local_channels = _make_divisible(128 * depth_multiplier, 8, 8)
        self.global_channels = _make_divisible(320 * depth_multiplier, 8, 8)

    def forward(self, x):
        local_feat, global_feat = None, None
        for i, block in enumerate(self.blocks, start=1):
            x = block(x)
            if i == LOCAL_ENDPOINT_LAYER:
                local_feat = x
            if i == GLOBAL_ENDPOINT_LAYER:
                global_feat = x
        return local_feat, global_feat


class LocalHead(nn.Module):
    """Faithful port of `local_head` (hf_net.py): descriptor branch (3x3 conv -> 1x1 conv ->
    l2-normalize) + detector branch (3x3 conv -> 1x1 conv over grid^2+1 dustbin classes ->
    softmax -> drop dustbin -> depth_to_space) producing a dense per-pixel score map."""

    def __init__(self, in_ch, descriptor_dim=256, detector_grid=8):
        super().__init__()
        self.detector_grid = detector_grid
        self.desc_conv1 = ConvBNReLU6(in_ch, 128, kernel_size=3)
        self.desc_conv2 = nn.Conv2d(128, descriptor_dim, 1)

        self.det_conv1 = ConvBNReLU6(in_ch, 128, kernel_size=3)
        self.det_conv2 = nn.Conv2d(128, 1 + detector_grid**2, 1)

    def forward(self, feat):
        desc = self.desc_conv1(feat)
        desc = self.desc_conv2(desc)
        desc = F.normalize(desc, p=2, dim=1)

        logits = self.det_conv1(feat)
        logits = self.det_conv2(logits)
        prob_full = F.softmax(logits, dim=1)
        prob = prob_full[:, :-1]  # strip the "no interest point" dustbin channel
        prob = F.pixel_shuffle(prob, self.detector_grid)  # depth_to_space equivalent
        prob = prob.squeeze(1)
        return desc, prob


class NetVLAD(nn.Module):
    """Faithful port of `vlad` (layers.py): soft-assignment NetVLAD pooling producing a
    single global descriptor per image."""

    def __init__(self, in_ch, n_clusters=64):
        super().__init__()
        self.n_clusters = n_clusters
        self.in_ch = in_ch
        self.membership_conv = nn.Conv2d(in_ch, n_clusters, 1)
        self.membership_bn = nn.BatchNorm2d(n_clusters)
        self.clusters = nn.Parameter(torch.randn(1, n_clusters, in_ch) * 0.01)

    def forward(self, feat):
        n, c, h, w = feat.shape
        memberships = self.membership_bn(self.membership_conv(feat))  # (N, K, H, W)
        memberships = F.softmax(memberships, dim=1)

        feat_flat = feat.view(n, c, h * w).transpose(1, 2)  # (N, HW, C)
        memberships_flat = memberships.view(n, self.n_clusters, h * w)  # (N, K, HW)

        # residuals[n, k, c] = sum_{hw} memberships[n,k,hw] * (cluster[k,c] - feat[n,hw,c])
        weighted_feat = torch.bmm(memberships_flat, feat_flat)  # (N, K, C)
        weighted_mass = memberships_flat.sum(dim=2, keepdim=True)  # (N, K, 1)
        residuals = self.clusters * weighted_mass - weighted_feat  # (N, K, C)

        descriptor = F.normalize(residuals, p=2, dim=2)  # intra-normalization per cluster
        descriptor = descriptor.reshape(n, self.n_clusters * c)
        descriptor = F.normalize(descriptor, p=2, dim=1)
        return descriptor


class HFNet(nn.Module):
    """Faithful port of ethz-asl/hfnet's `HfNet._model` graph: shared MobileNetV2 encoder,
    local head (SuperPoint-style detector + descriptor) tapped at layer_7, global head
    (NetVLAD) tapped at layer_18."""

    def __init__(self, depth_multiplier=1.0, descriptor_dim=256, detector_grid=8, n_clusters=64):
        super().__init__()
        self.encoder = MobileNetV2Encoder(in_ch=1, depth_multiplier=depth_multiplier)
        self.local_head = LocalHead(self.encoder.local_channels, descriptor_dim, detector_grid)
        self.global_head = NetVLAD(self.encoder.global_channels, n_clusters)

    def forward(self, image):
        image = (image - 128.0) / 128.0  # image_normalization (layers.py)
        local_feat, global_feat = self.encoder(image)
        local_descriptor_map, scores_dense = self.local_head(local_feat)
        global_descriptor = self.global_head(global_feat)
        return local_descriptor_map, scores_dense, global_descriptor


MENAGERIE_ZOO = "ported-pytorch"


def build_hfnet():
    # Tiny size: depth_multiplier=1.0 kept faithful (channel counts are HF-Net's own spec,
    # already small at the low end), input resolution and descriptor_dim/n_clusters shrunk
    # for a fast trace while keeping every layer/mechanism from the original.
    return HFNet(depth_multiplier=1.0, descriptor_dim=32, detector_grid=8, n_clusters=8)


def example_input_hfnet():
    # H, W must be multiples of 8 (encoder has 3 stride-2 downsamples after the stem's own
    # stride-2, i.e. total stride 32 to global_feat; detector_grid=8 needs H,W divisible by 8
    # at the local_feat tap). 64x64 keeps the trace fast.
    return torch.rand(1, 1, 64, 64) * 255.0


MENAGERIE_ENTRIES = [
    ("HF-Net", build_hfnet, example_input_hfnet, 2019, MENAGERIE_ZOO),
]
