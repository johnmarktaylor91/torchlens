"""Caffe-era dense prediction and detection landmarks, 2013-2017.

Paper: Caffe Model Zoo and landmark Caffe repositories for HED, ENet, DenseBox,
MTCNN, SqueezeDet, R-FCN, DeViSE, and Hypercolumns.

This module collects compact random-init PyTorch reimplementations of notable
architectures that were prominent in the BVLC Caffe ecosystem or official Caffe
repos but were not already present in the TorchLens catalog under their own
names.  The implementations keep the distinctive architectural primitives while
using small channel counts and inputs for fast trace/draw verification.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _ConvRelu(nn.Module):
    """Convolution followed by ReLU."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ) -> None:
        """Initialize the convolution block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        kernel_size:
            Spatial convolution kernel size.
        stride:
            Convolution stride.
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolution and ReLU.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Activated output feature map.
        """
        return F.relu(self.conv(x))


class HEDNet(nn.Module):
    """Holistically-Nested Edge Detection with deep side-output supervision."""

    def __init__(self, channels: tuple[int, ...] = (8, 16, 24, 32, 32)) -> None:
        """Initialize compact VGG-style HED blocks and fusion head.

        Parameters
        ----------
        channels:
            Output channels for the five VGG-like stages.
        """
        super().__init__()
        in_channels = 3
        self.blocks = nn.ModuleList()
        self.side_heads = nn.ModuleList()
        for out_channels in channels:
            block = nn.Sequential(
                _ConvRelu(in_channels, out_channels),
                _ConvRelu(out_channels, out_channels),
            )
            self.blocks.append(block)
            self.side_heads.append(nn.Conv2d(out_channels, 1, 1))
            in_channels = out_channels
        self.fuse = nn.Conv2d(len(channels), 1, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        """Predict fused edge probability and five side outputs.

        Parameters
        ----------
        x:
            RGB image tensor ``(B, 3, H, W)``.

        Returns
        -------
        tuple[Tensor, tuple[Tensor, ...]]
            Fused edge map and side edge maps.
        """
        size = x.shape[2:]
        side_outputs = []
        h = x
        for index, block in enumerate(self.blocks):
            h = block(h)
            side = self.side_heads[index](h)
            side_outputs.append(
                F.interpolate(side, size=size, mode="bilinear", align_corners=False)
            )
            if index != len(self.blocks) - 1:
                h = F.max_pool2d(h, 2)
        fused = torch.sigmoid(self.fuse(torch.cat(side_outputs, dim=1)))
        return fused, tuple(torch.sigmoid(side) for side in side_outputs)


class ENetBottleneck(nn.Module):
    """ENet bottleneck with projection, asymmetric/dilated conv, and residual path."""

    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        asymmetric: bool = False,
        downsample: bool = False,
    ) -> None:
        """Initialize an ENet bottleneck.

        Parameters
        ----------
        channels:
            Input and output channel count.
        dilation:
            Dilation for the central convolution.
        asymmetric:
            Whether to use factorized ``5x1`` then ``1x5`` convolutions.
        downsample:
            Whether to reduce spatial resolution by two.
        """
        super().__init__()
        mid_channels = max(4, channels // 4)
        stride = 2 if downsample else 1
        self.proj = nn.Conv2d(channels, mid_channels, 1, stride=stride, bias=False)
        if asymmetric:
            self.conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, (5, 1), padding=(2, 0), bias=False),
                nn.ReLU(),
                nn.Conv2d(mid_channels, mid_channels, (1, 5), padding=(0, 2), bias=False),
            )
        else:
            self.conv = nn.Conv2d(
                mid_channels,
                mid_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
        self.expand = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.skip = nn.AvgPool2d(2, 2) if downsample else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the bottleneck and residual sum.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Output feature map.
        """
        y = F.relu(self.proj(x))
        y = F.relu(self.conv(y))
        y = self.expand(y)
        return F.relu(y + self.skip(x))


class ENetSmall(nn.Module):
    """Compact ENet encoder-decoder for real-time semantic segmentation."""

    def __init__(self, num_classes: int = 12, channels: int = 16) -> None:
        """Initialize ENet's early-downsample, bottleneck, and decoder pattern.

        Parameters
        ----------
        num_classes:
            Number of segmentation classes.
        channels:
            Internal feature width.
        """
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, channels - 3, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(
            ENetBottleneck(channels),
            ENetBottleneck(channels, dilation=2),
            ENetBottleneck(channels, asymmetric=True),
        )
        self.stage2 = nn.Sequential(
            ENetBottleneck(channels, downsample=True),
            ENetBottleneck(channels, dilation=4),
            ENetBottleneck(channels, asymmetric=True),
            ENetBottleneck(channels, dilation=8),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, channels, 2, stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Predict dense segmentation logits.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Segmentation logits at the input resolution.
        """
        pooled = F.max_pool2d(x, 2)
        h = torch.cat([self.initial(x), pooled], dim=1)
        h = self.stage2(self.stage1(h))
        h = self.up(h)
        return self.classifier(h)


class DenseBoxNet(nn.Module):
    """DenseBox FCN with dense objectness, box-offset, and landmark heads."""

    def __init__(self, channels: int = 16, landmarks: int = 5) -> None:
        """Initialize the dense detector.

        Parameters
        ----------
        channels:
            Base feature width.
        landmarks:
            Number of landmark points predicted per spatial location.
        """
        super().__init__()
        self.features = nn.Sequential(
            _ConvRelu(3, channels),
            _ConvRelu(channels, channels),
            nn.MaxPool2d(2),
            _ConvRelu(channels, channels * 2),
            _ConvRelu(channels * 2, channels * 2),
            nn.MaxPool2d(2),
            _ConvRelu(channels * 2, channels * 4),
            _ConvRelu(channels * 4, channels * 4),
        )
        self.score = nn.Conv2d(channels * 4, 1, 1)
        self.box = nn.Conv2d(channels * 4, 4, 1)
        self.landmarks = nn.Conv2d(channels * 4, landmarks * 2, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict dense confidence, box distances, and landmarks.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Objectness map, box-offset map, and landmark-offset map.
        """
        h = self.features(x)
        return torch.sigmoid(self.score(h)), self.box(h), self.landmarks(h)


class _MTCNNStage(nn.Module):
    """One MTCNN cascade stage with class, box, and landmark heads."""

    def __init__(self, in_size: int, channels: tuple[int, ...], pooled: bool) -> None:
        """Initialize a stage network.

        Parameters
        ----------
        in_size:
            Spatial input size after optional adaptive pooling.
        channels:
            Convolutional channel widths.
        pooled:
            Whether to pool to ``in_size`` and use linear heads.
        """
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channels:
            layers.append(_ConvRelu(in_channels, out_channels))
            in_channels = out_channels
            if pooled:
                layers.append(nn.MaxPool2d(2, ceil_mode=True))
        self.features = nn.Sequential(*layers)
        self.pooled = pooled
        if pooled:
            self.pool = nn.AdaptiveAvgPool2d((in_size, in_size))
            hidden = in_channels * in_size * in_size
            self.cls = nn.Linear(hidden, 2)
            self.box = nn.Linear(hidden, 4)
            self.landmark = nn.Linear(hidden, 10)
        else:
            self.cls_map = nn.Conv2d(in_channels, 2, 1)
            self.box_map = nn.Conv2d(in_channels, 4, 1)
            self.landmark_map = nn.Conv2d(in_channels, 10, 1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run one cascade stage.

        Parameters
        ----------
        x:
            Stage input image/crop tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Classification, box regression, and landmark predictions.
        """
        h = self.features(x)
        if not self.pooled:
            return self.cls_map(h), self.box_map(h), self.landmark_map(h)
        flat = self.pool(h).flatten(1)
        return self.cls(flat), self.box(flat), self.landmark(flat)


class MTCNNCascade(nn.Module):
    """MTCNN P-Net/R-Net/O-Net multi-task face detector cascade."""

    def __init__(self) -> None:
        """Initialize the three cascade subnetworks."""
        super().__init__()
        self.pnet = _MTCNNStage(1, (10, 16, 32), pooled=False)
        self.rnet = _MTCNNStage(2, (16, 32, 48), pooled=True)
        self.onet = _MTCNNStage(2, (16, 32, 64, 64), pooled=True)

    def forward(self, image: Tensor, crop24: Tensor, crop48: Tensor) -> tuple[Tensor, ...]:
        """Evaluate all three cascade stages.

        Parameters
        ----------
        image:
            Image pyramid crop for P-Net.
        crop24:
            Candidate face crop for R-Net.
        crop48:
            Candidate face crop for O-Net.

        Returns
        -------
        tuple[Tensor, ...]
            P-Net, R-Net, and O-Net multi-task outputs.
        """
        return (*self.pnet(image), *self.rnet(crop24), *self.onet(crop48))


class FireModule(nn.Module):
    """SqueezeNet fire module used by SqueezeDet."""

    def __init__(self, in_channels: int, squeeze: int, expand: int) -> None:
        """Initialize squeeze and expand branches.

        Parameters
        ----------
        in_channels:
            Input channel count.
        squeeze:
            Squeeze ``1x1`` channel count.
        expand:
            Output channels for each expand branch.
        """
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze, 1)
        self.expand1 = nn.Conv2d(squeeze, expand, 1)
        self.expand3 = nn.Conv2d(squeeze, expand, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Apply squeeze and concatenated expand branches.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Concatenated fire-module output.
        """
        h = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.expand1(h)), F.relu(self.expand3(h))], dim=1)


class SqueezeDetSmall(nn.Module):
    """SqueezeDet with Fire modules and a ConvDet anchor prediction head."""

    def __init__(self, num_classes: int = 3, anchors: int = 4) -> None:
        """Initialize the compact SqueezeDet detector.

        Parameters
        ----------
        num_classes:
            Number of object classes.
        anchors:
            Number of anchors per spatial location.
        """
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.features = nn.Sequential(
            _ConvRelu(3, 16, 3, 2),
            nn.MaxPool2d(2),
            FireModule(16, 8, 16),
            FireModule(32, 8, 16),
            nn.MaxPool2d(2),
            FireModule(32, 12, 24),
            FireModule(48, 12, 24),
        )
        self.convdet = nn.Conv2d(48, anchors * (num_classes + 1 + 4), 3, padding=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict class, confidence, and box tensors per anchor.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Class logits, confidence logits, and box deltas.
        """
        h = self.features(x)
        pred = self.convdet(h)
        b, _, height, width = pred.shape
        pred = pred.view(b, self.anchors, self.num_classes + 5, height, width)
        cls = pred[:, :, : self.num_classes]
        conf = pred[:, :, self.num_classes : self.num_classes + 1]
        box = pred[:, :, self.num_classes + 1 :]
        return cls, conf, box


class RFCNSmall(nn.Module):
    """R-FCN with position-sensitive score maps and PSRoI pooling surrogate."""

    def __init__(self, num_classes: int = 4, bins: int = 3, channels: int = 16) -> None:
        """Initialize R-FCN score-map heads.

        Parameters
        ----------
        num_classes:
            Number of detection classes including background.
        bins:
            Position-sensitive pooling grid size.
        channels:
            Backbone channel width.
        """
        super().__init__()
        self.num_classes = num_classes
        self.bins = bins
        self.backbone = nn.Sequential(
            _ConvRelu(3, channels),
            nn.MaxPool2d(2),
            _ConvRelu(channels, channels * 2),
            _ConvRelu(channels * 2, channels * 2),
            nn.MaxPool2d(2),
            _ConvRelu(channels * 2, channels * 4),
        )
        self.cls_maps = nn.Conv2d(channels * 4, num_classes * bins * bins, 1)
        self.box_maps = nn.Conv2d(channels * 4, 4 * bins * bins, 1)

    def _ps_pool(self, maps: Tensor, channels: int) -> Tensor:
        """Average each position-sensitive bin over its matching grid cell.

        Parameters
        ----------
        maps:
            Position-sensitive maps ``(B, channels*bins*bins, H, W)``.
        channels:
            Number of semantic channels per bin.

        Returns
        -------
        Tensor
            Pooled bin responses ``(B, channels, bins, bins)``.
        """
        pooled = F.adaptive_avg_pool2d(maps, (self.bins, self.bins))
        b = pooled.shape[0]
        pooled = pooled.view(b, channels, self.bins, self.bins, self.bins, self.bins)
        diagonal = []
        for row in range(self.bins):
            for col in range(self.bins):
                diagonal.append(pooled[:, :, row, col, row, col])
        return torch.stack(diagonal, dim=-1).view(b, channels, self.bins, self.bins)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Predict R-FCN class scores and box deltas.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        tuple[Tensor, Tensor]
            Region class logits and box deltas.
        """
        h = self.backbone(x)
        cls_bins = self._ps_pool(self.cls_maps(h), self.num_classes)
        box_bins = self._ps_pool(self.box_maps(h), 4)
        return cls_bins.mean(dim=(2, 3)), box_bins.mean(dim=(2, 3))


class DeViSESmall(nn.Module):
    """DeViSE visual-semantic embedding model."""

    def __init__(self, vocab_size: int = 20, embedding_dim: int = 32) -> None:
        """Initialize visual and label embedding branches.

        Parameters
        ----------
        vocab_size:
            Number of candidate semantic labels.
        embedding_dim:
            Shared visual/text embedding dimension.
        """
        super().__init__()
        self.visual = nn.Sequential(
            _ConvRelu(3, 16, 3, 2),
            _ConvRelu(16, 32, 3, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
        )
        self.label_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, image: Tensor, label_ids: Tensor) -> Tensor:
        """Score image compatibility with semantic label embeddings.

        Parameters
        ----------
        image:
            RGB image tensor.
        label_ids:
            Candidate label token ids.

        Returns
        -------
        Tensor
            Cosine-like compatibility scores for each label.
        """
        visual = F.normalize(self.visual(image), dim=-1)
        labels = F.normalize(self.label_embeddings(label_ids), dim=-1)
        return torch.matmul(visual, labels.transpose(0, 1))


class HypercolumnNet(nn.Module):
    """Hypercolumn segmentation using concatenated multi-layer pixel descriptors."""

    def __init__(self, num_classes: int = 6, channels: int = 12) -> None:
        """Initialize the multi-scale feature extractor and pixel MLP.

        Parameters
        ----------
        num_classes:
            Number of dense prediction classes.
        channels:
            Base channel width.
        """
        super().__init__()
        self.conv1 = _ConvRelu(3, channels)
        self.conv2 = _ConvRelu(channels, channels * 2, stride=2)
        self.conv3 = _ConvRelu(channels * 2, channels * 4, stride=2)
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(channels * 7, channels * 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, num_classes, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Classify pixels from concatenated layer-wise hypercolumns.

        Parameters
        ----------
        x:
            RGB image tensor.

        Returns
        -------
        Tensor
            Dense logits at the input resolution.
        """
        size = x.shape[2:]
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c2_up = F.interpolate(c2, size=size, mode="bilinear", align_corners=False)
        c3_up = F.interpolate(c3, size=size, mode="bilinear", align_corners=False)
        return self.pixel_mlp(torch.cat([c1, c2_up, c3_up], dim=1))


def build_hed() -> nn.Module:
    """Build compact Holistically-Nested Edge Detection.

    Returns
    -------
    nn.Module
        Random-initialized HED model.
    """
    return HEDNet().eval()


def build_enet() -> nn.Module:
    """Build compact ENet segmentation model.

    Returns
    -------
    nn.Module
        Random-initialized ENet model.
    """
    return ENetSmall().eval()


def build_densebox() -> nn.Module:
    """Build compact DenseBox detector.

    Returns
    -------
    nn.Module
        Random-initialized DenseBox model.
    """
    return DenseBoxNet().eval()


def build_mtcnn() -> nn.Module:
    """Build compact MTCNN cascade.

    Returns
    -------
    nn.Module
        Random-initialized MTCNN model.
    """
    return MTCNNCascade().eval()


def build_squeezedet() -> nn.Module:
    """Build compact SqueezeDet detector.

    Returns
    -------
    nn.Module
        Random-initialized SqueezeDet model.
    """
    return SqueezeDetSmall().eval()


def build_rfcn() -> nn.Module:
    """Build compact R-FCN detector.

    Returns
    -------
    nn.Module
        Random-initialized R-FCN model.
    """
    return RFCNSmall().eval()


def build_devise() -> nn.Module:
    """Build compact DeViSE image-label embedding model.

    Returns
    -------
    nn.Module
        Random-initialized DeViSE model.
    """
    return DeViSESmall().eval()


def build_hypercolumns() -> nn.Module:
    """Build compact Hypercolumns dense predictor.

    Returns
    -------
    nn.Module
        Random-initialized Hypercolumns model.
    """
    return HypercolumnNet().eval()


def example_image() -> Tensor:
    """Return a small RGB image input.

    Returns
    -------
    Tensor
        Image tensor ``(1, 3, 64, 64)``.
    """
    return torch.randn(1, 3, 64, 64)


def example_mtcnn() -> tuple[Tensor, Tensor, Tensor]:
    """Return P-Net image plus R-Net/O-Net candidate crops.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        P-Net, R-Net, and O-Net inputs.
    """
    return torch.randn(1, 3, 32, 32), torch.randn(1, 3, 24, 24), torch.randn(1, 3, 48, 48)


def example_devise() -> tuple[Tensor, Tensor]:
    """Return image and candidate semantic label ids.

    Returns
    -------
    tuple[Tensor, Tensor]
        Image tensor and label-id tensor.
    """
    return torch.randn(1, 3, 64, 64), torch.arange(8)


MENAGERIE_ENTRIES = [
    (
        "Holistically-Nested Edge Detection (HED)",
        "build_hed",
        "example_image",
        "2015",
        "E5",
    ),
    (
        "ENet real-time semantic segmentation",
        "build_enet",
        "example_image",
        "2016",
        "E5",
    ),
    (
        "DenseBox fully convolutional detector",
        "build_densebox",
        "example_image",
        "2015",
        "E5",
    ),
    (
        "MTCNN face detection and alignment cascade",
        "build_mtcnn",
        "example_mtcnn",
        "2016",
        "E5",
    ),
    (
        "SqueezeDet ConvDet object detector",
        "build_squeezedet",
        "example_image",
        "2017",
        "E5",
    ),
    (
        "R-FCN position-sensitive object detector",
        "build_rfcn",
        "example_image",
        "2016",
        "E5",
    ),
    (
        "DeViSE visual-semantic embedding model",
        "build_devise",
        "example_devise",
        "2013",
        "E5",
    ),
    (
        "Hypercolumns pixel descriptor network",
        "build_hypercolumns",
        "example_image",
        "2015",
        "E5",
    ),
]
