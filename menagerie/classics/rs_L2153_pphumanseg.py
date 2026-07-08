# SOURCE: vendored from https://github.com/cavalleria/humanseg.pytorch @ 77663504b32d777ae22345e660f8bc2132c1c817
# (models/fast_scnn.py, base/base_model.py) -- PP-HumanSeg's official implementation lives
# in PaddleSeg (PaddlePaddle/PaddleSeg); this is a real, independent PyTorch reimplementation
# of a portrait/human segmentation stack (Fast-SCNN, arXiv:1902.04502) built specifically for
# the human-segmentation task, vendored verbatim. Only the relative import
# `from base.base_model import BaseModel` was inlined (the BaseModel class body below is an
# exact copy of base/base_model.py); FastSCNN's own module/forward structure is unchanged.
"""Fast-SCNN human/portrait segmentation network, vendored from cavalleria/humanseg.pytorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# ============================== base/base_model.py ==============================


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def load_pretrained_model(self, pretrained):
        if isinstance(pretrained, str):
            pretrain_dict = torch.load(pretrained, map_location="cpu")
            if "state_dict" in pretrain_dict:
                pretrain_dict = pretrain_dict["state_dict"]
        elif isinstance(pretrained, dict):
            pretrain_dict = pretrained

        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                if state_dict[k].shape == v.shape:
                    model_dict[k] = v
            state_dict.update(model_dict)
        self.load_state_dict(state_dict)


# ============================== models/fast_scnn.py ==============================


class FastSCNN(BaseModel):
    def __init__(self, num_classes=2, pretrained_backbone=None, aux=True, **kwargs):
        super(FastSCNN, self).__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifer(128, num_classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1),
            )
        self.init_weights()
        if pretrained_backbone is not None:
            self.load_pretrained_model(pretrained_backbone)

    def forward(self, x):
        size = x.size()[2:]
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        outputs = []
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(auxout, size, mode="bilinear", align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels * t, 1),
            _DWConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1, **kwargs)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode="bilinear", align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x


class LearningToDownsample(nn.Module):
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64, **kwargs):
        super(LearningToDownsample, self).__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels=64,
        block_channels=(64, 96, 128),
        out_channels=128,
        t=6,
        num_blocks=(3, 3, 3),
        **kwargs,
    ):
        super(GlobalFeatureExtractor, self).__init__()
        self.bottleneck1 = self._make_layer(
            LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2
        )
        self.bottleneck2 = self._make_layer(
            LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2
        )
        self.bottleneck3 = self._make_layer(
            LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1
        )
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = []
        layers.append(block(inplanes, planes, t, stride))
        for i in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class FeatureFusionModule(nn.Module):
    def __init__(
        self, highter_in_channels, lower_in_channels, out_channels, scale_factor=4, **kwargs
    ):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(
            lower_res_feature, scale_factor=4, mode="bilinear", align_corners=True
        )
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(Classifer, self).__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(nn.Dropout(0.1), nn.Conv2d(dw_channels, num_classes, 1))

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


def build_pphumanseg():
    # eval() mode: PyramidPooling's global-average-pool branch collapses to a
    # 1x1 spatial map, which BatchNorm2d rejects in training mode for batch
    # size 1 (needs >1 value per channel); eval mode uses running stats and
    # is also the real deployment mode for this segmentation network.
    model = FastSCNN(num_classes=2)
    model.eval()
    return model


def example_input_pphumanseg():
    return torch.randn(1, 3, 64, 64)


MENAGERIE_ENTRIES = [
    ("PP-HumanSeg", "build_pphumanseg", "example_input_pphumanseg", 2021, "SOURCE_AVAILABLE"),
]
