"""MMOCR text-detection architectures reimplemented in base PyTorch.

Sources: MMOCR model configs and papers for FCENet/FCE, PANet, PSENet, and
Mask R-CNN text detection.  The modules are compact random-initialized
reconstructions for TorchLens tracing: they keep the defining architecture
primitives while using small widths and inputs.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Convolution, batch normalization, and ReLU activation."""

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, groups: int = 1
    ) -> None:
        """Initialize a convolution block.

        Parameters
        ----------
        in_channels:
            Number of input channels.
        out_channels:
            Number of output channels.
        stride:
            Spatial stride.
        groups:
            Convolution groups.
        """

        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=stride,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply convolutional feature extraction.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Activated feature map.
        """

        return F.relu(self.bn(self.conv(x)), inplace=False)


class DeformableConvLite(nn.Module):
    """Small deformable-convolution analogue using learned offsets and sampling."""

    def __init__(self, channels: int) -> None:
        """Initialize offset prediction and sampled convolution.

        Parameters
        ----------
        channels:
            Number of input and output channels.
        """

        super().__init__()
        self.offset = nn.Conv2d(channels, 2, 3, padding=1)
        self.conv = ConvBNAct(channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        """Sample features with learned offsets before convolution.

        Parameters
        ----------
        x:
            Feature map.

        Returns
        -------
        Tensor
            Deformably sampled feature map.
        """

        batch, _, height, width = x.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype),
            torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        offset = torch.tanh(self.offset(x)).permute(0, 2, 3, 1) * 0.08
        sampled = F.grid_sample(x, base + offset, align_corners=False)
        return self.conv(sampled)


class ResidualStage(nn.Module):
    """Compact ResNet stage with optional deformable refinement."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, dcn: bool = False) -> None:
        """Initialize residual stage.

        Parameters
        ----------
        in_channels:
            Input channel count.
        out_channels:
            Output channel count.
        stride:
            First block stride.
        dcn:
            Whether to append a deformable-convolution refinement.
        """

        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        self.conv1 = ConvBNAct(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBNAct(out_channels, out_channels)
        self.dcn = DeformableConvLite(out_channels) if dcn else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply residual feature extraction.

        Parameters
        ----------
        x:
            Input feature map.

        Returns
        -------
        Tensor
            Stage output.
        """

        residual = self.proj(x)
        y = self.conv2(self.conv1(x))
        return self.dcn(F.relu(y + residual, inplace=False))


class ResNetFPNBackbone(nn.Module):
    """Compact ResNet backbone with lateral feature pyramid outputs."""

    def __init__(self, base: int = 16, dcn: bool = False) -> None:
        """Initialize the backbone.

        Parameters
        ----------
        base:
            Base feature width.
        dcn:
            Whether to use deformable refinement in the deepest stage.
        """

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.c2 = ResidualStage(base, base * 2, stride=1)
        self.c3 = ResidualStage(base * 2, base * 4, stride=2)
        self.c4 = ResidualStage(base * 4, base * 8, stride=2, dcn=dcn)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ResNet feature maps.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            C2, C3, and C4 feature maps.
        """

        c1 = self.stem(image)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        return c2, c3, c4


class FPNF(nn.Module):
    """MMOCR FPNF neck that upsamples and concatenates feature levels."""

    def __init__(self, channels: tuple[int, int, int], width: int = 32) -> None:
        """Initialize lateral projections and fusion.

        Parameters
        ----------
        channels:
            Input channels for C2/C3/C4.
        width:
            Shared pyramid width.
        """

        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(ch, width, 1) for ch in channels])
        self.fuse = nn.Sequential(ConvBNAct(width * 3, width), ConvBNAct(width, width))

    def forward(self, feats: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Fuse multi-scale features.

        Parameters
        ----------
        feats:
            Backbone feature maps.

        Returns
        -------
        Tensor
            Fused feature map at the finest pyramid resolution.
        """

        c2, c3, c4 = feats
        p4 = self.laterals[2](c4)
        p3 = self.laterals[1](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.laterals[0](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        fused = torch.cat(
            [
                p2,
                F.interpolate(p3, size=p2.shape[-2:], mode="nearest"),
                F.interpolate(p4, size=p2.shape[-2:], mode="nearest"),
            ],
            dim=1,
        )
        return self.fuse(fused)


class FPEMFFM(nn.Module):
    """PANet feature-pyramid enhancement and feature-fusion module."""

    def __init__(self, channels: tuple[int, int, int], width: int = 32) -> None:
        """Initialize PANet neck.

        Parameters
        ----------
        channels:
            Input feature channels.
        width:
            Shared neck width.
        """

        super().__init__()
        self.laterals = nn.ModuleList([nn.Conv2d(ch, width, 1) for ch in channels])
        self.up = nn.ModuleList([ConvBNAct(width, width, groups=width) for _ in range(2)])
        self.down = nn.ModuleList(
            [ConvBNAct(width, width, stride=2, groups=width) for _ in range(2)]
        )
        self.fuse = ConvBNAct(width * 3, width)

    def forward(self, feats: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """Apply bidirectional enhancement and final fusion.

        Parameters
        ----------
        feats:
            Backbone feature maps.

        Returns
        -------
        Tensor
            PANet fused feature map.
        """

        p2, p3, p4 = [lat(feat) for lat, feat in zip(self.laterals, feats, strict=True)]
        p3 = p3 + F.interpolate(self.up[0](p4), size=p3.shape[-2:], mode="nearest")
        p2 = p2 + F.interpolate(self.up[1](p3), size=p2.shape[-2:], mode="nearest")
        p3 = p3 + F.interpolate(self.down[0](p2), size=p3.shape[-2:], mode="nearest")
        p4 = p4 + F.interpolate(self.down[1](p3), size=p4.shape[-2:], mode="nearest")
        return self.fuse(
            torch.cat(
                [
                    p2,
                    F.interpolate(p3, size=p2.shape[-2:], mode="nearest"),
                    F.interpolate(p4, size=p2.shape[-2:], mode="nearest"),
                ],
                dim=1,
            )
        )


class FCENet(nn.Module):
    """FCENet text detector with Fourier contour embedding head."""

    def __init__(self, dcn: bool = False) -> None:
        """Initialize FCENet.

        Parameters
        ----------
        dcn:
            Whether to use deformable convolution in the ResNet stage.
        """

        super().__init__()
        self.backbone = ResNetFPNBackbone(dcn=dcn)
        self.neck = FPNF((32, 64, 128))
        self.tr = nn.Conv2d(32, 1, 1)
        self.tcl = nn.Conv2d(32, 1, 1)
        self.fourier = nn.Conv2d(32, 22, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict text regions, center lines, and Fourier contour coefficients.

        Parameters
        ----------
        image:
            RGB image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Text-region logits, center-line logits, and Fourier coefficients.
        """

        feat = self.neck(self.backbone(image))
        return self.tr(feat), self.tcl(feat), self.fourier(feat)


class PANetTextDetector(nn.Module):
    """PANet text detector with FPEM-FFM neck and pixel aggregation embeddings."""

    def __init__(self, base: int = 16) -> None:
        """Initialize PANet.

        Parameters
        ----------
        base:
            Base ResNet width.
        """

        super().__init__()
        self.backbone = ResNetFPNBackbone(base=base)
        channels = (base * 2, base * 4, base * 8)
        self.neck = FPEMFFM(channels)
        self.kernel = nn.Conv2d(32, 1, 1)
        self.text = nn.Conv2d(32, 1, 1)
        self.embedding = nn.Conv2d(32, 4, 1)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Predict text masks, kernels, and aggregation embeddings.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Text mask logits, kernel logits, and pixel embeddings.
        """

        feat = self.neck(self.backbone(image))
        return self.text(feat), self.kernel(feat), self.embedding(feat)


class PSENetTextDetector(nn.Module):
    """PSENet detector with progressive scale-expansion kernel masks."""

    def __init__(self) -> None:
        """Initialize PSENet."""

        super().__init__()
        self.backbone = ResNetFPNBackbone()
        self.neck = FPNF((32, 64, 128))
        self.kernels = nn.Conv2d(32, 7, 1)

    def forward(self, image: Tensor) -> Tensor:
        """Predict progressively smaller text kernels.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        Tensor
            Kernel mask logits from full text region to inner kernels.
        """

        return self.kernels(self.neck(self.backbone(image)))


class TextMaskRCNN(nn.Module):
    """Mask R-CNN text detector with RPN, ROI transform, and mask head."""

    def __init__(self, proposals: int = 8) -> None:
        """Initialize the two-stage text detector.

        Parameters
        ----------
        proposals:
            Number of compact proposal summaries.
        """

        super().__init__()
        self.proposals = proposals
        self.backbone = ResNetFPNBackbone()
        self.neck = FPNF((32, 64, 128))
        self.rpn_obj = nn.Conv2d(32, 3, 1)
        self.rpn_box = nn.Conv2d(32, 12, 1)
        self.roi_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.box_fc = nn.Sequential(nn.Linear(32 * 3 * 3, 64), nn.ReLU(), nn.Linear(64, 5))
        self.mask_head = nn.Sequential(
            ConvBNAct(32, 32),
            nn.ConvTranspose2d(32, 16, 2, stride=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run RPN proposal scoring and text mask prediction.

        Parameters
        ----------
        image:
            RGB text image.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            RPN boxes, compact ROI box predictions, and text masks.
        """

        feat = self.neck(self.backbone(image))
        obj = self.rpn_obj(feat).flatten(2).max(dim=1).values
        idx = torch.topk(obj, self.proposals, dim=1).indices
        flat = feat.flatten(2).transpose(1, 2)
        gather = idx.unsqueeze(-1).expand(-1, -1, flat.shape[-1])
        roi_tokens = torch.gather(flat, 1, gather)
        roi_map = roi_tokens.transpose(1, 2).reshape(image.shape[0], 32, 2, self.proposals // 2)
        pooled = self.roi_pool(roi_map)
        return self.rpn_box(feat), self.box_fc(pooled.flatten(1)), self.mask_head(pooled)


def build_fcenet_resnet50() -> nn.Module:
    """Build compact FCENet with ResNet/FPNF and Fourier head.

    Returns
    -------
    nn.Module
        Evaluation-mode FCENet.
    """

    return FCENet(dcn=False).eval()


def build_fce_resnet50_dcn() -> nn.Module:
    """Build compact FCE/FCENet with deformable ResNet refinement.

    Returns
    -------
    nn.Module
        Evaluation-mode FCE detector.
    """

    return FCENet(dcn=True).eval()


def build_panet_resnet18() -> nn.Module:
    """Build compact PANet with ResNet18-style width.

    Returns
    -------
    nn.Module
        Evaluation-mode PANet.
    """

    return PANetTextDetector(base=12).eval()


def build_panet_resnet50() -> nn.Module:
    """Build compact PANet with ResNet50-style width.

    Returns
    -------
    nn.Module
        Evaluation-mode PANet.
    """

    return PANetTextDetector(base=16).eval()


def build_psenet_resnet50() -> nn.Module:
    """Build compact PSENet with progressive scale expansion head.

    Returns
    -------
    nn.Module
        Evaluation-mode PSENet.
    """

    return PSENetTextDetector().eval()


def build_maskrcnn_ocr_text() -> nn.Module:
    """Build compact Mask R-CNN text detector.

    Returns
    -------
    nn.Module
        Evaluation-mode Mask R-CNN text detector.
    """

    return TextMaskRCNN().eval()


def example_input() -> Tensor:
    """Return a compact RGB document/text image.

    Returns
    -------
    Tensor
        Image tensor with shape ``(1, 3, 64, 64)``.
    """

    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("FCE-ResNet50-DCN", "build_fce_resnet50_dcn", "example_input", "2019", "OCR"),
    ("FCENet-ResNet50", "build_fcenet_resnet50", "example_input", "2021", "OCR"),
    ("MaskRCNN-OCR-Text", "build_maskrcnn_ocr_text", "example_input", "2017", "OCR"),
    ("PANet-ResNet18", "build_panet_resnet18", "example_input", "2019", "OCR"),
    ("PANet-ResNet50", "build_panet_resnet50", "example_input", "2019", "OCR"),
    ("PSENet-ResNet50", "build_psenet_resnet50", "example_input", "2019", "OCR"),
]
