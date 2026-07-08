# SOURCE: vendored from ShiqiYu/libfacedetection.train @ dca340aa082c71081a68d17db8e58b33a58a914b
# https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/yunet_train/models/layers.py
# https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/yunet_train/models/init.py
# https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/yunet_train/models/config.py
# https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/yunet_train/models/backbone.py
# https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/yunet_train/models/neck.py
# https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/yunet_train/tasks/face/head.py
# https://raw.githubusercontent.com/ShiqiYu/libfacedetection.train/master/yunet_train/tasks/face/model.py
#
# YuNet (Wu, Zhang, Xu, Yu, "libfacedetection", 2018-2023; officially released in the
# OpenCV Zoo as `face_detection_yunet`): an ultra-light (~75K params) single-shot face
# detector with joint bounding-box + 5-point landmark regression, designed for <1ms
# inference on mobile/embedded CPUs. The OpenCV Zoo entry itself ships only the exported
# ONNX weights (`face_detection_yunet_2023mar.onnx`) plus a thin `cv2.FaceDetectorYN`
# inference wrapper -- no PyTorch source. The actual PyTorch training/architecture code
# lives in the author's own companion repo, `ShiqiYu/libfacedetection.train`, which is
# the official upstream source these ONNX exports are produced from (see the repo's own
# `yunet_train/engine/onnx_export.py` export pipeline and the shipped `weights/yunet_n.pth`
# / `weights/yunet_s.pth` checkpoints).
#
# All classes below (`ConvDPUnit`, `Conv_head`, `Conv4layerBlock`, `init_yunet_weights`,
# `YuNetBackbone`, `TFPN`, `YuNetHead`, `YuNet`) and the `YUNET_N`/`YUNET_S` config
# dataclass instances are copied verbatim from the real repo files listed above (only
# `from __future__ import annotations` headers are dropped as redundant under this repo's
# Python floor, and module-relative imports are flattened into this single file with no
# behavioral change). `torch.onnx.is_in_onnx_export()` in `YuNetHead.forward` is always
# False for a plain eager forward pass, so the real (non-ONNX) branch executes, exactly as
# it would for the official repo's own eager training/inference forward.
"""YuNet (libfacedetection, ShiqiYu et al.): ultra-light single-shot face detector with
joint bbox + 5-point landmark regression, ConvDPUnit-based backbone/neck/head, vendored
from the official PyTorch training repo `ShiqiYu/libfacedetection.train`."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------------------
# yunet_train/models/layers.py (verbatim)
# ---------------------------------------------------------------------------------------


class ConvDPUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, withBNRelu: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True, groups=1)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            1,
            1,
            bias=True,
            groups=out_channels,
        )
        self.withBNRelu = withBNRelu
        if withBNRelu:
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        if self.withBNRelu:
            x = self.bn(x)
            x = self.relu(x)
        return x


class Conv_head(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 2, 1, bias=True, groups=1)
        self.conv2 = ConvDPUnit(mid_channels, out_channels, True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


class Conv4layerBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, withBNRelu: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = ConvDPUnit(in_channels, in_channels, True)
        self.conv2 = ConvDPUnit(in_channels, out_channels, withBNRelu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# ---------------------------------------------------------------------------------------
# yunet_train/models/init.py (verbatim)
# ---------------------------------------------------------------------------------------


def init_yunet_weights(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            if layer.bias is not None:
                nn.init.xavier_normal_(layer.weight.data)
                layer.bias.data.fill_(0.02)
            else:
                layer.weight.data.normal_(0, 0.01)
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()


# ---------------------------------------------------------------------------------------
# yunet_train/models/config.py (verbatim)
# ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class YuNetModelConfig:
    variant: str
    stage_channels: tuple[tuple[int, ...], ...]
    downsample_idx: tuple[int, ...]
    out_idx: tuple[int, ...]
    neck_in_channels: tuple[int, ...]
    neck_out_idx: tuple[int, ...]
    num_classes: int
    in_channels: int
    feat_channels: int
    shared_stacked_convs: int
    stacked_convs: int
    strides: tuple[int, ...]
    use_kps: bool
    kps_num: int


YUNET_N = YuNetModelConfig(
    variant="yunet_n",
    stage_channels=(
        (3, 16, 16),
        (16, 64),
        (64, 64),
        (64, 64),
        (64, 64),
        (64, 64),
    ),
    downsample_idx=(0, 2, 3, 4),
    out_idx=(3, 4, 5),
    neck_in_channels=(64, 64, 64),
    neck_out_idx=(0, 1, 2),
    num_classes=1,
    in_channels=64,
    feat_channels=64,
    shared_stacked_convs=1,
    stacked_convs=0,
    strides=(8, 16, 32),
    use_kps=True,
    kps_num=5,
)

YUNET_S = YuNetModelConfig(
    variant="yunet_s",
    stage_channels=(
        (3, 16, 16),
        (16, 32),
        (32, 64),
        (64, 64),
        (64, 64),
        (64, 64),
    ),
    downsample_idx=(0, 2, 3, 4),
    out_idx=(3, 4, 5),
    neck_in_channels=(64, 64, 64),
    neck_out_idx=(0, 1, 2),
    num_classes=1,
    in_channels=64,
    feat_channels=64,
    shared_stacked_convs=0,
    stacked_convs=0,
    strides=(8, 16, 32),
    use_kps=True,
    kps_num=5,
)

MODEL_CONFIGS = {
    YUNET_N.variant: YUNET_N,
    YUNET_S.variant: YUNET_S,
}


def get_model_config(variant: str) -> YuNetModelConfig:
    try:
        return MODEL_CONFIGS[variant]
    except KeyError as exc:
        names = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"Unknown YuNet variant {variant!r}. Expected one of: {names}") from exc


# ---------------------------------------------------------------------------------------
# yunet_train/models/backbone.py (verbatim)
# ---------------------------------------------------------------------------------------


class YuNetBackbone(nn.Module):
    def __init__(
        self,
        stage_channels: tuple[tuple[int, ...], ...],
        downsample_idx: tuple[int, ...],
        out_idx: tuple[int, ...],
    ):
        super().__init__()
        self.layer_num = len(stage_channels)
        self.downsample_idx = downsample_idx
        self.out_idx = out_idx
        self.model0 = Conv_head(*stage_channels[0])
        for i in range(1, self.layer_num):
            self.add_module(f"model{i}", Conv4layerBlock(*stage_channels[i]))
        self.init_weights()

    def init_weights(self) -> None:
        init_yunet_weights(self)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out = []
        for i in range(self.layer_num):
            x = getattr(self, f"model{i}")(x)
            if i in self.out_idx:
                out.append(x)
            if i in self.downsample_idx:
                x = F.max_pool2d(x, 2)
        return out


# ---------------------------------------------------------------------------------------
# yunet_train/models/neck.py (verbatim)
# ---------------------------------------------------------------------------------------


class TFPN(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], out_idx: tuple[int, ...]):
        super().__init__()
        self.num_layers = len(in_channels)
        self.out_idx = out_idx
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.lateral_convs.append(ConvDPUnit(in_channels[i], in_channels[i], True))
        self.init_weights()

    def init_weights(self) -> None:
        init_yunet_weights(self)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        feats = list(feats)
        num_feats = len(feats)

        for i in range(num_feats - 1, 0, -1):
            feats[i] = self.lateral_convs[i](feats[i])
            feats[i - 1] = feats[i - 1] + F.interpolate(
                feats[i],
                scale_factor=2.0,
                mode="nearest",
            )

        feats[0] = self.lateral_convs[0](feats[0])
        return [feats[i] for i in self.out_idx]


# ---------------------------------------------------------------------------------------
# yunet_train/tasks/face/head.py (verbatim)
# ---------------------------------------------------------------------------------------


class YuNetHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        shared_stacked_convs: int,
        stacked_convs: int,
        strides: tuple[int, ...],
        use_kps: bool = True,
        kps_num: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.NK = kps_num
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_sigmoid_cls = True
        self.use_kps = use_kps
        self.shared_stack_convs = shared_stacked_convs
        self.strides = tuple((stride, stride) for stride in strides)
        self.strides_num = len(self.strides)

        self._init_layers()
        self.init_weights()

    def _init_layers(self) -> None:
        if self.shared_stack_convs > 0:
            self.multi_level_share_convs = nn.ModuleList()
        if self.stacked_convs > 0:
            self.multi_level_cls_convs = nn.ModuleList()
            self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_cls = nn.ModuleList()
        self.multi_level_bbox = nn.ModuleList()
        self.multi_level_obj = nn.ModuleList()
        if self.use_kps:
            self.multi_level_kps = nn.ModuleList()

        for _ in self.strides:
            if self.shared_stack_convs > 0:
                single_level_share_convs = []
                for i in range(self.shared_stack_convs):
                    chn = self.in_channels if i == 0 else self.feat_channels
                    single_level_share_convs.append(ConvDPUnit(chn, self.feat_channels))
                self.multi_level_share_convs.append(nn.Sequential(*single_level_share_convs))

            if self.stacked_convs > 0:
                single_level_cls_convs = []
                single_level_reg_convs = []
                for i in range(self.stacked_convs):
                    chn = (
                        self.in_channels
                        if i == 0 and self.shared_stack_convs == 0
                        else self.feat_channels
                    )
                    single_level_cls_convs.append(ConvDPUnit(chn, self.feat_channels))
                    single_level_reg_convs.append(ConvDPUnit(chn, self.feat_channels))
                self.multi_level_reg_convs.append(nn.Sequential(*single_level_reg_convs))
                self.multi_level_cls_convs.append(nn.Sequential(*single_level_cls_convs))

            chn = (
                self.in_channels
                if self.stacked_convs == 0 and self.shared_stack_convs == 0
                else self.feat_channels
            )
            self.multi_level_cls.append(ConvDPUnit(chn, self.num_classes, False))
            self.multi_level_bbox.append(ConvDPUnit(chn, 4, False))
            if self.use_kps:
                self.multi_level_kps.append(ConvDPUnit(chn, self.NK * 2, False))
            self.multi_level_obj.append(ConvDPUnit(chn, 1, False))

    def init_weights(self) -> None:
        init_yunet_weights(self)

    def forward(
        self,
        feats: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        if self.shared_stack_convs > 0:
            feats = [convs(feat) for feat, convs in zip(feats, self.multi_level_share_convs)]

        if self.stacked_convs > 0:
            feats_cls, feats_reg = [], []
            for i in range(self.strides_num):
                feats_cls.append(self.multi_level_cls_convs[i](feats[i]))
                feats_reg.append(self.multi_level_reg_convs[i](feats[i]))
            cls_preds = [convs(feat) for feat, convs in zip(feats_cls, self.multi_level_cls)]
            bbox_preds = [convs(feat) for feat, convs in zip(feats_reg, self.multi_level_bbox)]
            obj_preds = [convs(feat) for feat, convs in zip(feats_reg, self.multi_level_obj)]
            kps_preds = (
                [convs(feat) for feat, convs in zip(feats_reg, self.multi_level_kps)]
                if self.use_kps
                else []
            )
        else:
            cls_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_cls)]
            bbox_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_bbox)]
            obj_preds = [convs(feat) for feat, convs in zip(feats, self.multi_level_obj)]
            kps_preds = (
                [convs(feat) for feat, convs in zip(feats, self.multi_level_kps)]
                if self.use_kps
                else []
            )

        if torch.onnx.is_in_onnx_export():
            cls = [
                f.permute(0, 2, 3, 1).view(f.shape[0], -1, self.num_classes).sigmoid()
                for f in cls_preds
            ]
            obj = [f.permute(0, 2, 3, 1).view(f.shape[0], -1, 1).sigmoid() for f in obj_preds]
            bbox = [f.permute(0, 2, 3, 1).view(f.shape[0], -1, 4) for f in bbox_preds]
            kps = [f.permute(0, 2, 3, 1).view(f.shape[0], -1, self.NK * 2) for f in kps_preds]
            return cls, obj, bbox, kps

        return cls_preds, bbox_preds, obj_preds, kps_preds


# ---------------------------------------------------------------------------------------
# yunet_train/tasks/face/model.py (verbatim)
# ---------------------------------------------------------------------------------------


class YuNet(nn.Module):
    def __init__(self, config: YuNetModelConfig):
        super().__init__()
        self.config = config
        self.backbone = YuNetBackbone(
            stage_channels=config.stage_channels,
            downsample_idx=config.downsample_idx,
            out_idx=config.out_idx,
        )
        self.neck = TFPN(
            in_channels=config.neck_in_channels,
            out_idx=config.neck_out_idx,
        )
        self.bbox_head = YuNetHead(
            num_classes=config.num_classes,
            in_channels=config.in_channels,
            feat_channels=config.feat_channels,
            shared_stacked_convs=config.shared_stacked_convs,
            stacked_convs=config.stacked_convs,
            strides=config.strides,
            use_kps=config.use_kps,
            kps_num=config.kps_num,
        )

    def extract_feat(self, img: torch.Tensor) -> list[torch.Tensor]:
        feats = self.backbone(img)
        return self.neck(feats)

    def forward(
        self,
        img: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        feats = self.extract_feat(img)
        return self.bbox_head(feats)


def build_yunet_variant(variant: str = "yunet_n") -> YuNet:
    return YuNet(get_model_config(variant))


def build_yunet():
    torch.manual_seed(0)
    model = build_yunet_variant("yunet_n")
    model.eval()
    return model


def example_input_yunet():
    # 128x128 (multiple of the network's max downsample factor 16, 4 pooling stages) --
    # official OpenCV Zoo deployment uses dynamic input size; this is a light faithful size.
    torch.manual_seed(0)
    return (torch.randn(1, 3, 128, 128),)


MENAGERIE_ENTRIES = [
    ("YuNet", "build_yunet", "example_input_yunet", 2023, "vendored"),
]
