# SOURCE: vendored from PopicLab/cue @ master
# https://raw.githubusercontent.com/PopicLab/cue/master/models/cue_net.py
# https://raw.githubusercontent.com/PopicLab/cue/master/models/modules.py
# https://raw.githubusercontent.com/PopicLab/cue/master/img/heatmap.py (heatmaps2predictions, used
#   unconditionally by MultiSVHG.forward for postprocessing predicted heatmaps into keypoints)
#
# Popic et al, "Cue: a deep-learning framework for structural variant discovery and genotyping
# using breakpoint-image-like feature representations" -- Cue is a stacked-hourglass CNN
# (`MultiSVHG`) that reads multi-channel "signal images" encoding short/long-read alignment
# evidence around candidate SV breakpoints and predicts per-class keypoint heatmaps, following
# the stacked-hourglass architecture from Newell et al human-pose-estimation (ECCV 2016 /
# NeurIPS 2017 PoseNet). `Conv`/`Residual`/`Hourglass`/`HourglassBackbone` (models/modules.py)
# and `MultiSVHG` (models/cue_net.py) are copied verbatim from the real repo; only the enclosing
# `CueModelConfig` factory wrapper is dropped (not architecture) since we construct `MultiSVHG`
# directly with a tiny config object. `SVKeypointHeatmapUtility` (img/heatmap.py) is vendored
# too since `MultiSVHG.forward` calls `heatmap_generator.heatmaps2predictions` unconditionally
# (not gated on `self.training`) to convert predicted heatmaps into keypoint predictions; its
# `keypoints2heatmaps` (target-heatmap generation, training-only) and the genomics-specific
# `img.utils`/`seq.*` modules it would otherwise import are dropped since they are not needed
# for the forward-pass architecture (image-space postprocessing math + torchvision + numpy/cv2/
# scipy only, exactly as in the real file). No architectural changes.

from enum import Enum
from collections import namedtuple, defaultdict

import numpy as np
import cv2
from scipy.ndimage import maximum_filter, gaussian_filter
import torch
import torch.nn as nn
import torch.nn.modules.utils as torch_utils
import torchvision.transforms as transforms


# ---- img/constants.py (TargetType + label constants only, verbatim) ----


class TargetType(str, Enum):
    boxes = "boxes"
    keypoints = "keypoints"
    labels = "labels"
    classes = "classes"
    image_id = "image_id"
    area = "area"
    heatmaps = "heatmaps"
    weight = "weight"
    scores = "scores"
    gloc = "gloc"
    dataset_id = "dataset_id"


LABEL_BACKGROUND = 0
KP_FILTERED = -1


# ---- models/modules.py (verbatim) ----

ConvLayerConfig = namedtuple("LayerConfig", "in_channels out_channels kernel_size padding pool")


class Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        pool=True,
        relu=True,
        bn=False,
    ):
        super(Conv, self).__init__()
        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, depth, nc, expansion):
        super(Hourglass, self).__init__()
        self.depth = depth
        nc_expanded = nc + expansion
        self.up1 = Residual(nc, nc)
        self.pool = nn.MaxPool2d(2, 2)
        self.low1 = Residual(nc, nc_expanded)
        if self.depth > 1:
            self.low2 = Hourglass(self.depth - 1, nc_expanded, expansion)
        else:
            self.low2 = Residual(nc_expanded, nc_expanded)
        self.low3 = Residual(nc_expanded, nc)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        pool = self.pool(x)
        low1 = self.low1(pool)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up1 = self.up1(x)
        up2 = self.up2(low3)
        return up1 + up2


class HourglassBackbone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HourglassBackbone, self).__init__()
        self.layers = nn.Sequential(
            Conv(
                in_channels, 64, kernel_size=7, stride=2, pool=False, padding=3, relu=True, bn=True
            ),
            Residual(64, 128),
            nn.MaxPool2d(2, 2),
            Residual(128, 128),
            Residual(128, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


# ---- img/heatmap.py (SVKeypointHeatmapUtility, verbatim minus genomics-specific
#      keypoints2heatmaps target-construction path which forward() never calls) ----


class SVKeypointHeatmapUtility:
    def __init__(
        self, image_dim, num_kps_per_sv=1, num_sv_labels=3, sigma=20, stride=4, peak_threshold=0.4
    ):
        self.heatmap_stride = stride
        self.sigma = sigma
        self.peak_threshold = peak_threshold
        self.num_kps_per_sv = num_kps_per_sv
        self.heatmap_dim = int(image_dim / self.heatmap_stride)
        self.num_heatmap_channels = num_sv_labels * num_kps_per_sv
        self.refine = True

    def heatmaps2predictions(self, target):
        kp_list_per_sv_type = []
        num_kps = 0
        heatmaps = target[TargetType.heatmaps].permute(1, 2, 0).detach().cpu().numpy()
        labels = []
        keypoints_out = []
        scores = []
        for idx in range(self.num_heatmap_channels):
            sv_label = idx + 1
            heatmap = heatmaps[:, :, idx]
            peaks = self.find_peaks(heatmap, self.peak_threshold)
            keypoints = np.zeros((len(peaks), 4))  # x, y, score, id
            for i, peak in enumerate(peaks):
                if self.refine:
                    pw = 4
                    x_min, y_min = np.maximum(0, peak - pw)
                    x_max, y_max = np.minimum(np.array(heatmap.T.shape) - 1, peak + pw)
                    patch = heatmap[y_min : y_max + 1, x_min : x_max + 1]
                    location_of_patch_center = upscale_keypoints(
                        peak[::-1] - [y_min, x_min], self.heatmap_stride
                    )
                    patch_upscaled = cv2.resize(
                        patch,
                        None,
                        fx=self.heatmap_stride,
                        fy=self.heatmap_stride,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    patch_upscaled = gaussian_filter(patch_upscaled, sigma=5)
                    location_of_max = np.unravel_index(
                        patch_upscaled.argmax(), patch_upscaled.shape
                    )
                    peak_score = patch_upscaled[location_of_max]
                    refined_center = location_of_max - location_of_patch_center
                else:
                    refined_center = [0, 0]
                    peak_score = heatmap[tuple(peak[::-1])]
                refined_kp = upscale_keypoints(peak, self.heatmap_stride) + refined_center[::-1]
                keypoints[i, :] = tuple(x for x in refined_kp) + (peak_score, num_kps)
                num_kps += 1
                labels.append(sv_label)
                keypoints_out.append([[refined_kp[0], refined_kp[1], 1]])
                scores.append(min(1, peak_score))
            kp_list_per_sv_type.append(keypoints)
        target[TargetType.labels] = torch.as_tensor(labels, dtype=torch.int64)
        target[TargetType.keypoints] = torch.as_tensor(keypoints_out, dtype=torch.float32)
        target[TargetType.scores] = torch.as_tensor(scores, dtype=torch.float32)
        return kp_list_per_sv_type

    @staticmethod
    def add_gaussian_at_point(point, heatmap, sigma, grid_y, grid_x, stride):
        x, y = np.meshgrid([i for i in range(int(grid_y))], [i for i in range(int(grid_x))])
        offset = stride / 2.0 - 0.5
        x = x * stride + offset
        y = y * stride + offset
        d = (x - point[0]) ** 2 + (y - point[1]) ** 2
        exponent = d / 2.0 / sigma / sigma
        mask = exponent <= 4.6052
        gauss_peak = np.multiply(mask, np.exp(-exponent))
        heatmap += gauss_peak
        heatmap[heatmap > 1.0] = 1.0
        return heatmap

    @staticmethod
    def find_peaks(heatmap, threshold):
        keypoints_binary = (maximum_filter(heatmap, size=8, mode="constant") == heatmap) * (
            heatmap > threshold
        )
        return np.array(np.nonzero(keypoints_binary)[::-1]).T


def upscale_keypoints(keypoints, ratio):
    return keypoints * ratio


def batch_images(images):
    batch_shape = [len(images)] + list(images[0].shape)
    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    return batched_imgs


# ---- models/cue_net.py (MultiSVHG, verbatim) ----


class MultiSVHG(nn.Module):
    # Stacked hourglass network for SV breakpoint prediction
    # Implementation based on Newell et al human pose estimation models (ECCV 2016, NeurIPS 2017)
    # PoseNet (https://github.com/princeton-vl/pose-ae-train)

    def __init__(self, config):
        super(MultiSVHG, self).__init__()
        self.config = config
        self.heatmap_generator = SVKeypointHeatmapUtility(
            config.image_dim,
            num_kps_per_sv=config.num_keypoints,
            num_sv_labels=config.num_classes - 1,
            sigma=config.sigma,
            stride=config.stride,
            peak_threshold=config.heatmap_peak_threshold,
        )
        self.hg_in_dim = 256
        self.hg_out_dim = self.heatmap_generator.num_heatmap_channels
        self.hg_expansion = 128
        self.hg_depth = 4
        self.hg_stack_size = 4
        self.backbone = HourglassBackbone(self.config.n_signals, self.hg_in_dim)
        self.hg_stack = nn.ModuleList(
            [
                nn.Sequential(Hourglass(self.hg_depth, self.hg_in_dim, self.hg_expansion))
                for _ in range(self.hg_stack_size)
            ]
        )
        self.features = nn.ModuleList(
            [
                nn.Sequential(
                    Residual(self.hg_in_dim, self.hg_in_dim),
                    Conv(
                        self.hg_in_dim,
                        self.hg_in_dim,
                        kernel_size=1,
                        pool=False,
                        bn=True,
                        relu=True,
                    ),
                )
                for _ in range(self.hg_stack_size)
            ]
        )
        self.outs = nn.ModuleList(
            [
                Conv(self.hg_in_dim, self.hg_out_dim, 1, pool=False, relu=False, bn=False)
                for _ in range(self.hg_stack_size)
            ]
        )
        self.merge_features = nn.ModuleList(
            [
                Conv(
                    self.hg_in_dim, self.hg_in_dim, kernel_size=1, pool=False, relu=False, bn=False
                )
                for _ in range(self.hg_stack_size)
            ]
        )
        self.merge_preds = nn.ModuleList(
            [
                Conv(
                    self.hg_out_dim, self.hg_in_dim, kernel_size=1, pool=False, relu=False, bn=False
                )
                for _ in range(self.hg_stack_size)
            ]
        )

    def forward(self, images, targets=None):
        images = batch_images(images)
        x = self.backbone(images)
        stage_outputs = []
        for i in range(self.hg_stack_size):
            hg = self.hg_stack[i](x)
            feature = self.features[i](hg)
            stack_output = self.outs[i](feature)
            stage_outputs.append(stack_output)
            if i < self.hg_stack_size - 1:
                x = x + self.merge_preds[i](stack_output) + self.merge_features[i](feature)

        outputs = [{TargetType.heatmaps: heatmaps} for heatmaps in stage_outputs[-1]]
        for output in outputs:
            self.heatmap_generator.heatmaps2predictions(output)
        if self.training:
            losses = {"loss_heatmaps": self.loss(stage_outputs, targets)}
            return losses, outputs
        return outputs

    def loss(self, stage_outputs, targets):
        for target in targets:
            self.heatmap_generator.keypoints2heatmaps(target)
        heatmaps_gt = torch.stack(
            [t[TargetType.heatmaps].to(self.config.device) for t in targets], dim=0
        )
        stage_outputs = torch.stack(stage_outputs, dim=0)
        stage_weights = [1] * stage_outputs.shape[0]
        loss = self.focal_loss(stage_outputs, heatmaps_gt, stage_weights=stage_weights)
        return loss

    def focal_loss(
        self, outputs, targets, gamma=1, stage_weights=None, alpha=0.1, beta=0.02, theta=0.01
    ):
        # Focal L2 loss adapted from SimplePose (Li et al, AAAI 2020)
        dkt = torch.where(torch.ge(targets, theta), outputs - alpha, 1 - outputs - beta)
        factor = torch.abs(1.0 - dkt) ** gamma
        lkt = (outputs - targets) ** 2 * factor
        fl = lkt.sum(dim=(1, 2, 3, 4))
        weight_loss = [fl[i] * stage_weights[i] for i in range(len(stage_weights))]
        loss = sum(weight_loss) / sum(stage_weights)
        return loss


class _CueConfig:
    """Tiny stand-in for the real repo's engine/config_utils.py DataConfig, holding only the
    fields MultiSVHG.__init__ reads (defaults match the real repo's default_params.yaml:
    num_keypoints=1, model_architecture='HG', image_dim=256, sigma=10, stride=4,
    heatmap_peak_threshold=0.4; num_classes/n_signals mirror a BASIC4 (4-class) / SHORT3
    (3-signal) real configuration)."""

    def __init__(
        self,
        image_dim=256,
        num_keypoints=1,
        num_classes=4,
        n_signals=3,
        sigma=10,
        stride=4,
        heatmap_peak_threshold=0.4,
    ):
        self.image_dim = image_dim
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.n_signals = n_signals
        self.sigma = sigma
        self.stride = stride
        self.heatmap_peak_threshold = heatmap_peak_threshold
        self.device = torch.device("cpu")


def build_cue():
    config = _CueConfig(
        image_dim=256,
        num_keypoints=1,
        num_classes=4,
        n_signals=3,
        sigma=10,
        stride=4,
        heatmap_peak_threshold=0.4,
    )
    model = MultiSVHG(config)
    model.eval()
    return model


def example_input_cue():
    # forward() expects a list of per-sample CHW signal-image tensors (n_signals channels),
    # batched internally via batch_images(); real images are square, image_dim x image_dim.
    return ([torch.randn(3, 256, 256)],)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("Cue", "build_cue", "example_input_cue", 2022, "vendored"),
]
