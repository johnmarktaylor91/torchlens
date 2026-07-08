# FAITHFUL PORT of tensorflow/models/research/keypointnet @ archive (original framework: TensorFlow 1.x + tf.contrib.slim)
# https://raw.githubusercontent.com/tensorflow/models/archive/research/keypointnet/main.py
#
# Suwajanakorn, Snavely, Tompson, Norouzi, 2018 (NeurIPS) "Discovery of Latent
# 3D Keypoints via End-to-end Geometric Reasoning" (arXiv:1807.03146). This is
# the official "KeypointNet" network (distinct from the qq456cvb/KeypointNet
# *dataset*-only repo, which ships no model code of its own -- see queue note).
# The official implementation (`tensorflow/models/research/keypointnet`) is
# TF1.x + `tf.contrib.slim`, both long-deprecated/removed from modern
# TensorFlow and not installable in this base env, so the forward-pass network
# architecture is transcribed FAITHFULLY from the real `main.py` source (every
# layer/mechanism, not a paper-level gist):
#   - `dilated_cnn`: a 12-layer dilated-conv trunk (rates
#     [1,1,2,4,8,16,1,2,4,8,16,1], each 3x3 conv + BatchNorm + LeakyReLU(0.1),
#     "SAME" spatial padding) shared by both sub-networks below.
#   - `orientation_network`: dilated_cnn(images, num_filters/2) -> 2-channel
#     1x1-rate conv -> per-pixel softmax over the two "modules" -> spatial
#     expectation against an [-1,1] meshgrid, giving two 2D "orientation"
#     points (real code's `out_xy`).
#   - `keypoint_network`: builds a binary left/right flag from the orientation
#     estimate (or ground truth, annealed), tiles+concats it onto the RGB
#     image, feeds the 4-channel image through `dilated_cnn`, then regresses
#     per-keypoint probability maps (`conv_xy`) and per-keypoint depth maps
#     (`conv_z`, with the real code's fixed `-30` camera-distance bias); the
#     final `uv`/`z` are the probability-weighted (softmax) spatial expectation
#     of each map, exactly as in `keypoint_network`'s real body. The silhouette
#     loss (`sill`) and variance loss (`variance`) are part of the same forward
#     computation graph in the real function and are kept as additional
#     outputs to stay faithful to the real `keypoint_network` return signature.
#
# Not ported (training/data-pipeline-only, not part of the network's forward
# computation graph): `Transformer`/`estimate_rotation`/`relative_pose_loss`/
# `separation_loss`/`consistency_loss` (multi-view geometric losses over a
# *pair* of already-computed `uvz` outputs + camera matrices), the TFRecord
# `input_fn`, `model_fn` (Estimator wiring), and `predict`/`main` (checkpoint
# I/O + argparse). Those consume the network's outputs; they are not part of
# the "KeypointNet" architecture itself.

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "ported-pytorch"


def _same_pad_dilated_conv3x3(in_ch, out_ch, rate):
    """3x3 conv with slim's 'SAME' padding at the given dilation rate."""
    padding = rate  # (kernel_size - 1) // 2 * dilation, kernel_size=3 => rate
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=rate)


class DilatedCNN(nn.Module):
    """Port of `dilated_cnn(images, num_filters, is_training)`.

    Real code: 12 slim.conv2d layers (3x3, BatchNorm, LeakyReLU(alpha=0.1))
    with dilation rates [1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1].
    """

    RATES = [1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]

    def __init__(self, in_channels, num_filters):
        super().__init__()
        layers = []
        ch = in_channels
        for rate in self.RATES:
            layers.append(_same_pad_dilated_conv3x3(ch, num_filters, rate))
            layers.append(nn.BatchNorm2d(num_filters))
            ch = num_filters
        self.convs = nn.ModuleList(layers[0::2])
        self.bns = nn.ModuleList(layers[1::2])

    def forward(self, images):
        net = images
        for conv, bn in zip(self.convs, self.bns):
            net = F.leaky_relu(bn(conv(net)), negative_slope=0.1)
        return net


def _meshgrid(h, w, device, dtype):
    """Port of `meshgrid(h)`: a [-1, 1]-ranging grid (real code assumes square
    vh == vw; generalized here to h, w for arbitrary trace input sizes)."""
    rx = torch.arange(0.5, w, 1, device=device, dtype=dtype) / (w / 2) - 1
    ry = torch.arange(0.5, h, 1, device=device, dtype=dtype) / (h / 2) - 1
    ranx, rany = torch.meshgrid(rx, -ry, indexing="xy")
    return ranx, rany


class OrientationNetwork(nn.Module):
    """Port of `orientation_network(images, num_filters, is_training)`."""

    def __init__(self, num_filters):
        super().__init__()
        half_filters = int(num_filters * 0.5)
        self.trunk = DilatedCNN(in_channels=3, num_filters=half_filters)
        self.prob_conv = nn.Conv2d(half_filters, 2, kernel_size=3, padding=1, dilation=1)

    def forward(self, images):
        net = self.trunk(images)
        prob = self.prob_conv(
            net
        )  # [batch, 2, h, w] (NCHW; real code's NHWC->transpose is a no-op here)

        b, modules, h, w = prob.shape
        prob = prob.reshape(b, modules, h * w)
        prob = F.softmax(prob, dim=2)
        prob = prob.reshape(b, modules, h, w)

        ranx, rany = _meshgrid(h, w, prob.device, prob.dtype)

        sx = (prob * ranx).sum(dim=(2, 3))
        sy = (prob * rany).sum(dim=(2, 3))
        out_xy = torch.stack([sx, sy], dim=-1).reshape(b, modules, 2)
        return out_xy


class KeypointNetwork(nn.Module):
    """Port of `keypoint_network(rgba, num_filters, num_kp, is_training,
    lr_gt=None, anneal=1)`.

    Returns (uv, z, orient, sill, variance) matching the real function's first
    five return values (`prob_viz`/`prob_vizs` are visualization-only tensors
    dropped here, matching this module's forward-computation-graph scope).
    """

    def __init__(self, num_filters=64, num_kp=10):
        super().__init__()
        self.num_filters = num_filters
        self.num_kp = num_kp
        self.orientation_net = OrientationNetwork(num_filters)
        self.trunk = DilatedCNN(in_channels=4, num_filters=num_filters)
        self.conv_xy = nn.Conv2d(num_filters, num_kp, kernel_size=3, padding=1, dilation=1)
        self.conv_z = nn.Conv2d(num_filters, num_kp, kernel_size=3, padding=1, dilation=1)

    def forward(self, rgba, lr_gt=None, anneal=1.0):
        images = rgba[:, :3, :, :]

        orient = self.orientation_net(images)  # [batch, 2, 2]

        lr_estimated = torch.clamp(
            torch.sign(orient[:, 0, :1] - orient[:, 1, :1]), min=0.0
        )  # [batch, 1]

        if lr_gt is None:
            lr = lr_estimated
        else:
            lr_gt_sign = torch.clamp(torch.sign(lr_gt[:, :1]), min=0.0)
            lr = torch.round(lr_gt_sign * anneal + lr_estimated * (1 - anneal))

        b, _, h, w = images.shape
        lrtiled = lr.reshape(b, 1, 1, 1).expand(b, 1, h, w)
        images_lr = torch.cat([images, lrtiled], dim=1)  # [batch, 4, h, w]

        mask = rgba[:, 3, :, :]
        mask = (mask > 0).to(images.dtype)

        net = self.trunk(images_lr)

        prob = self.conv_xy(net)  # [batch, num_kp, h, w]
        z = -30 + self.conv_z(net)  # [batch, num_kp, h, w]

        prob_flat = prob.reshape(b, self.num_kp, h * w)
        prob_flat = F.softmax(prob_flat, dim=2)
        prob = prob_flat.reshape(b, self.num_kp, h, w)

        ranx, rany = _meshgrid(h, w, prob.device, prob.dtype)

        sx = (prob * ranx).sum(dim=(2, 3))
        sy = (prob * rany).sum(dim=(2, 3))

        sill = (prob * mask.unsqueeze(1)).sum(dim=(2, 3))
        sill = torch.mean(-torch.log(sill + 1e-12))

        z_expected = (prob * z).sum(dim=(2, 3))
        uv = torch.stack([sx, sy], dim=-1).reshape(b, self.num_kp, 2)

        variance = self._variance_loss(prob, ranx, rany, uv)

        return uv, z_expected, orient, sill, variance

    @staticmethod
    def _variance_loss(probmap, ranx, rany, uv):
        """Port of `variance_loss(probmap, ranx, rany, uv)`."""
        ran = torch.stack([ranx, rany], dim=-1)  # [h, w, 2]
        h, w = ran.shape[0], ran.shape[1]
        ran = ran.reshape(1, 1, h, w, 2)

        b, num_kp = uv.shape[0], uv.shape[1]
        uv_r = uv.reshape(b, num_kp, 1, 1, 2)

        diff = ((uv_r - ran) ** 2).sum(dim=4)  # [batch, num_kp, h, w]
        diff = diff * probmap

        return torch.mean(diff.sum(dim=(2, 3)))


# ============================================================================
# build_/example_input_ harness
# ============================================================================


def build_keypointnet():
    # Real repo defaults (main.py::_default_hparams): num_filters=64, num_kp=10.
    # A shrunk num_filters keeps the trace fast; num_kp kept at the real default.
    model = KeypointNetwork(num_filters=16, num_kp=10)
    model.eval()
    return model


def example_input_keypointnet():
    torch.manual_seed(0)
    batch = 1
    # Real repo's fixed input size is 128x128 (main.py: `vw = vh = 128`); shrunk
    # here for a fast trace since KeypointNetwork is fully convolutional.
    h = w = 32
    rgba = torch.rand(batch, 4, h, w)
    return rgba


MENAGERIE_ENTRIES = [
    ("KeypointNet", build_keypointnet, example_input_keypointnet, 2018, "ported-pytorch"),
]
