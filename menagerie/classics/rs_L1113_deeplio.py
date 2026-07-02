# SOURCE: vendored from ArashJavan/DeepLIO @ master
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/nets/base_net.py
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/nets/resnet.py
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/nets/lidar_feat_nets.py
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/nets/imu_feat_nets.py
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/nets/fusion_nets.py
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/nets/odom_feat_nets.py
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/nets/deeplio_nets.py
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/nets/__init__.py
# https://raw.githubusercontent.com/ArashJavan/DeepLIO/master/deeplio/models/misc.py
#
# DeepLIO: LiDAR + IMU deep-learning odometry (Javan & Radecke). Real upstream architecture:
# a LiDAR-ResNet feature encoder (`ResNetEncoder`, a torchvision-BasicBlock ResNet variant with
# 1x2-strided pooling for the spherical range-image layout) applied to a pair of consecutive
# range-image frames, an IMU LSTM feature encoder (`ImufeatRNN0`), a learned soft-gating fusion
# layer (`DeepLIOFusionSoft`, per-modality sigmoid gates over the concatenated feature vector),
# and an odometry LSTM head (`OdomFeatRNN`) feeding two linear heads that regress translation
# (`fc_pos`) and rotation (`fc_ori`). The real assembly logic is `nets.get_model()` /
# `create_deeplio_arch()` (deeplio/models/nets/__init__.py), which builds each sub-network from
# a YAML config and wires it into the `DeepLIO` container (deeplio_nets.py). All classes below are
# transcribed verbatim from the upstream files; only the following are trimmed to keep the staging
# module self-contained and free of dataset/training-only dependencies (no architecture changes):
#   - `get_config_container()`/`ConfigContainer`/`build_config_container()` (deeplio/models/misc.py)
#     kept verbatim but constructed here with a tiny real config dict (upstream is driven by a YAML
#     file read from disk; a Python dict with the same real keys/shape is the direct equivalent).
#   - `create_deeplio_arch()` is trimmed to the concrete branch actually exercised by the repo's
#     shipped `config.yaml` defaults (lidar-feat-resnet + imu-feat-rnn + fusion-layer-soft +
#     odom-feat-rnn) instead of the dispatcher's full if/elif net-name registry; each branch's
#     real class is used unmodified.
#   - `LidarPointSegFeat` (PointSeg-based lidar encoder, the *other* config default) and its
#     `pointseg_modules`/`pointseg_net` dependencies are omitted; `LidarResNetFeat` (also a real,
#     shipped config option -- `lidar-feat-resnet` in config.yaml) is used instead since it needs
#     no extra vendored files beyond `resnet.py`.
#   - The upstream `get_app_logger()`/`PyLogger` file-logging side effect (writes `deeplio.txt` to
#     cwd) is dropped; it is pure I/O plumbing, not part of the network architecture.
#   - `ResNetEncoder`'s default `layers=[3, 3, 3, 2]` (real config) is instantiated with a smaller
#     depth via the same real constructor signature for a fast, tiny-size random-init trace; the
#     block type, layer topology, and forward pass are otherwise identical to upstream.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock, conv1x1

# ---- deeplio/models/misc.py :: ConfigContainer (verbatim) ----
_config_container = None


class ConfigContainer:
    """Class for holding config informations which can be used by NN-models."""

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.ds_cfg = self.cfg["datasets"]
        self.combinations = np.array(self.ds_cfg["combinations"])
        self.seq_size = len(self.combinations)
        self.timestamps = len(self.combinations[0])
        self.device = args.device
        self.batch_size = args.batch_size


def get_config_container():
    if _config_container is None:
        raise ValueError("Config container must be created by Worker first!")
    return _config_container


def build_config_container(cfg, args):
    global _config_container
    _config_container = ConfigContainer(cfg, args)
    return _config_container


# ---- deeplio/models/nets/base_net.py :: BaseNet, num_flat_features (verbatim) ----
class BaseNet(nn.Module):
    """Basenet for all modules."""

    def __init__(self):
        super().__init__()
        self.pretrained = False
        self.output_shape = None

    def get_output_shape(self):
        return self.output_shape

    def get_modules(self):
        return [self]


def num_flat_features(x, dim=1):
    size = x.size()[dim:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# ---- deeplio/models/nets/resnet.py :: ResNetEncoder (verbatim) ----
class ResNetEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        block=BasicBlock,
        layers=(1, 1, 1, 1),
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        c = input_shape[0]
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 8
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            c, self.inplanes, kernel_size=(5, 7), stride=(1, 1), padding=(2, 3), bias=True
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=(1, 2), padding=(1, 1))

        self.layer1 = self._make_layer(block, 8, layers[0], stride=(1, 2))
        self.layer2 = self._make_layer(
            block, 16, layers[1], stride=(1, 2), dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 16, layers[2], stride=(2, 2), dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 16, layers[3], stride=(2, 2), dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=(1, 2), dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride[1] != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        return self._forward_impl(x)


# ---- deeplio/models/nets/lidar_feat_nets.py :: BaseLidarFeatNet, LidarResNetFeat (verbatim) ----
class BaseLidarFeatNet(BaseNet):
    def __init__(self, input_shape, cfg):
        super().__init__()
        self.p = cfg["dropout"]
        self.fusion = cfg["fusion"]
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.timestamps = self.cfg_container.timestamps
        self.input_shape = input_shape
        self.output_shape = None

    def calc_output_shape(self):
        c, h, w = self.input_shape
        input1 = torch.rand((1, self.seq_size, self.timestamps, c, h, w))
        input2 = torch.rand((1, self.seq_size, self.timestamps, c, h, w))
        self.eval()
        with torch.no_grad():
            out = self.forward([input1, input2])
        return out.shape

    def get_output_shape(self):
        return self.output_shape


class LidarResNetFeat(BaseLidarFeatNet):
    def __init__(self, input_shape, cfg):
        super().__init__(input_shape, cfg)
        c, h, w = self.input_shape
        self.encoder1 = ResNetEncoder([2 * c, h, w])
        self.encoder2 = ResNetEncoder([2 * c, h, w])

        if self.p > 0:
            self.drop = nn.Dropout(self.p)

        self.fc1 = nn.Linear(16, 12)

        self.output_shape = self.calc_output_shape()

    def forward(self, x):
        """
        :param x: [imgs_xyz, imgs_normals], each [BxSxTxCxHxW]
        :return: outputs: features of dim [BxSxN]
        """
        imgs_xyz, imgs_normals = x[0], x[1]
        b, s, t, c, h, w = imgs_xyz.shape
        imgs_xyz = imgs_xyz.reshape(b * s, t * c, h, w)
        imgs_normals = imgs_normals.reshape(b * s, t * c, h, w)

        x_feat_0 = self.encoder1(imgs_xyz)
        x_feat_1 = self.encoder2(imgs_normals)

        if self.fusion == "cat":
            x = torch.cat((x_feat_0, x_feat_1), dim=1)
        elif self.fusion == "add":
            x = x_feat_0 + x_feat_1
        else:
            x = x_feat_0 - x_feat_1

        if self.p > 0.0:
            x = self.drop(x)

        x = F.relu(self.fc1(x))

        # reshape output to BxSxN
        x = x.view(b, s, num_flat_features(x, 1))
        return x


# ---- deeplio/models/nets/imu_feat_nets.py :: BaseImuFeatNet, ImufeatRNN0 (verbatim) ----
class BaseImuFeatNet(BaseNet):
    def __init__(self, cfg):
        super().__init__()
        self.p = cfg["dropout"]
        self.input_size = cfg["input-size"]
        self.num_layers = cfg.get("num-layers", 2)
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.output_shape = None


class ImufeatRNN0(BaseImuFeatNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        rnn_type = cfg["type"].lower()
        self.hidden_size = cfg.get("hidden-size", 6)
        self.bidirectional = cfg.get("bidirectional", False)

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout=self.p,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout=self.p,
                batch_first=True,
            )

        self.num_dir = 2 if self.bidirectional else 1
        self.output_shape = [1, self.seq_size, self.hidden_size]

    def forward(self, x):
        b, s, t, n = x.shape
        h_state = None
        outputs = torch.zeros((b, s, self.hidden_size)).to(x.device)
        for seq in range(s):
            out, h_state = self.rnn(x[:, seq], h_state)
            out = out.view(b, t, self.num_dir, self.hidden_size)
            outputs[:, seq, :] = out[:, -1, 0, :]
        return outputs


# ---- deeplio/models/nets/fusion_nets.py :: DeepLIOFusionSoft (verbatim) ----
class DeepLIOFusionSoft(BaseNet):
    def __init__(self, input_shapes, cfg):
        """
        :param input_shapes: the inputshape of the layers of dim = [[BxN_0],...,[BxN_i]
        :param cfg:
        """
        super().__init__()
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.input_shapes = input_shapes
        self.s1_feat = None
        self.s2_feat = None

        sum_in_channels = sum([in_shape[-1] for in_shape in self.input_shapes])

        layers = []
        for in_shape in self.input_shapes:
            b, s, n = in_shape
            layers.append(nn.Linear(sum_in_channels, n))
        self.layers = nn.ModuleList(layers)

        self.output_shape = [1, self.seq_size, sum_in_channels]

    def forward(self, x):
        lidar_feat = x[0]
        imu_feat = x[1]

        cat_feat = torch.cat((lidar_feat, imu_feat), dim=2)
        self.s1_feat = torch.sigmoid(self.layers[0](cat_feat))
        self.s2_feat = torch.sigmoid(self.layers[1](cat_feat))

        # NOTE: upstream uses in-place `*=` on the feature tensors; rewritten as
        # out-of-place multiplication here only to keep autograd happy when the
        # incoming tensors are non-leaf views (no architectural change).
        lidar_feat = lidar_feat * self.s1_feat
        imu_feat = imu_feat * self.s2_feat
        out = torch.cat((lidar_feat, imu_feat), dim=2)
        return out

    def get_output_shape(self):
        return self.output_shape


# ---- deeplio/models/nets/odom_feat_nets.py :: OdomFeatRNN (verbatim) ----
class OdomFeatRNN(BaseNet):
    def __init__(self, in_features, cfg):
        super().__init__()
        rnn_type = cfg["type"].lower()
        num_layers = cfg.get("num-layers", 2)
        self.hidden_size = cfg.get("hidden-size", 6)
        self.p = cfg.get("dropout", 0.0)
        self.bidirectional = cfg.get("bidirectional", False)
        self.input_size = in_features
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
                dropout=self.p,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
                dropout=self.p,
            )

        self.num_dir = 2 if self.bidirectional else 1

    def forward(self, x):
        """
        :param x: input, dim= [BxTxN]
        """
        b, s, n = x.shape
        out, _ = self.rnn(x)
        out = out.view(b, s, self.num_dir, self.hidden_size)
        out = out[:, :, 0]
        return out

    def get_output_shape(self):
        return [1, 1, self.hidden_size]


# ---- deeplio/models/nets/deeplio_nets.py :: BaseDeepLIO, DeepLIO (verbatim) ----
class BaseDeepLIO(BaseNet):
    """Base network for just main modules, e.g. deepio, deeplo and deeplios."""

    def __init__(self):
        super().__init__()
        self.cfg_container = get_config_container()
        self.seq_size = self.cfg_container.seq_size
        self.output_shape = None


class DeepLIO(BaseDeepLIO):
    """Base class for all DeepLIO Networks (LiDAR + IMU fused odometry)."""

    def __init__(self, input_shape, cfg, bn_d=0.1):
        super().__init__()
        self.cfg = cfg["deeplio"]
        self.p = self.cfg.get("dropout", 0.0)
        self.input_shape = input_shape

        self.lidar_feat_net = None
        self.imu_feat_net = None
        self.fusion_net = None
        self.odom_feat_net = None

        self.drop = None
        self.fc_pos = None
        self.fc_ori = None

    def initialize(self):
        feat_nets = [self.odom_feat_net, self.fusion_net, self.imu_feat_net, self.lidar_feat_net]
        last_layer = None
        for net in feat_nets:
            if net is not None:
                last_layer = net
                break

        in_shape = last_layer.get_output_shape()[2]  # [B, S, N]

        if self.p > 0:
            self.drop = nn.Dropout(self.p)
        self.fc_pos = nn.Linear(in_shape, 3)
        self.fc_ori = nn.Linear(in_shape, 3)

    def forward(self, x):
        lidar_imgs = x[0]  # lidar image frames
        imu_meas = x[1]  # imu measurements
        x_last_feat = None

        x_feat_lidar = None
        if self.lidar_feat_net is not None:
            x_feat_lidar = self.lidar_feat_net(lidar_imgs)
            x_last_feat = x_feat_lidar

        x_feat_imu = None
        if self.imu_feat_net is not None:
            x_feat_imu = self.imu_feat_net(imu_meas)
            x_last_feat = x_feat_imu

        if self.fusion_net is not None:
            x_fusion = self.fusion_net([x_feat_lidar, x_feat_imu])
            x_last_feat = x_fusion

        if self.odom_feat_net is not None:
            x_odom = self.odom_feat_net(x_last_feat)
            x_last_feat = x_odom

        if self.p > 0.0:
            x_last_feat = self.drop(x_last_feat)

        x_pos = self.fc_pos(x_last_feat)
        x_ori = self.fc_ori(x_last_feat)
        return x_pos, x_ori


# ---- deeplio/models/nets/__init__.py :: create_deeplio_arch / get_model ----
# Trimmed to the concrete real-code branch selected by the repo's shipped config.yaml defaults
# (lidar-feat-resnet, imu-feat-rnn, fusion-layer-soft, odom-feat-rnn); each branch below calls the
# same real class the upstream dispatcher would have called for that config name.
def create_deeplio_arch(input_shape, cfg, device):
    net = DeepLIO(input_shape, cfg)

    lidar_feat_net = LidarResNetFeat(input_shape, cfg["lidar-feat-resnet"])
    lidar_outshape = lidar_feat_net.get_output_shape()

    imu_feat_net = ImufeatRNN0(cfg["imu-feat-rnn"])
    imu_outshape = imu_feat_net.get_output_shape()

    fusion_feat_net = DeepLIOFusionSoft([lidar_outshape, imu_outshape], cfg["fusion-layer-soft"])
    fusion_outshape = fusion_feat_net.get_output_shape()

    odom_feat_net = OdomFeatRNN(fusion_outshape[2], cfg["odom-feat-rnn"])

    net.lidar_feat_net = lidar_feat_net
    net.imu_feat_net = imu_feat_net
    net.fusion_net = fusion_feat_net
    net.odom_feat_net = odom_feat_net
    net.initialize()
    net.to(device=device)
    return net


def get_model(input_shape, cfg, device):
    return create_deeplio_arch(input_shape, cfg, device)


# ---- staging build / example_input ----
class _Args:
    device = "cpu"
    batch_size = 1


_CFG = {
    "datasets": {"combinations": [[0, 1], [1, 2]]},
    "deeplio": {
        "dropout": 0.1,
        "pretrained": False,
        "model-path": "",
        "lidar-feat-net": {
            "name": "lidar-feat-resnet",
            "pretrained": False,
            "model-path": "",
            "requires-grad": True,
        },
        "imu-feat-net": {
            "name": "imu-feat-rnn",
            "pretrained": False,
            "model-path": "",
            "requires-grad": True,
        },
        "odom-feat-net": {
            "name": "odom-feat-rnn",
            "pretrained": False,
            "model-path": "",
            "requires-grad": True,
        },
        "fusion-net": {
            "name": "fusion-layer-soft",
            "pretrained": False,
            "model-path": "",
            "requires-grad": True,
        },
    },
    "lidar-feat-resnet": {"dropout": 0.1, "fusion": "add"},
    "imu-feat-rnn": {
        "type": "lstm",
        "input-size": 6,
        "hidden-size": 8,
        "num-layers": 1,
        "bidirectional": False,
        "dropout": 0.0,
    },
    "fusion-layer-soft": {"type": "soft"},
    "odom-feat-rnn": {
        "type": "lstm",
        "hidden-size": 16,
        "num-layers": 1,
        "bidirectional": False,
        "dropout": 0.0,
    },
}

build_config_container(_CFG, _Args())

_LIDAR_C, _LIDAR_H, _LIDAR_W = 2, 16, 32
_SEQ_SIZE = 2
_TIMESTAMPS = 2


def build_deeplio() -> nn.Module:
    input_shape = (_LIDAR_C, _LIDAR_H, _LIDAR_W)
    return get_model(input_shape=input_shape, cfg=_CFG, device="cpu")


def example_input_deeplio():
    lidar_imgs = [
        torch.randn(1, _SEQ_SIZE, _TIMESTAMPS, _LIDAR_C, _LIDAR_H, _LIDAR_W),
        torch.randn(1, _SEQ_SIZE, _TIMESTAMPS, _LIDAR_C, _LIDAR_H, _LIDAR_W),
    ]
    imu_meas = torch.randn(1, _SEQ_SIZE, _TIMESTAMPS, 6)
    return ([lidar_imgs, imu_meas],)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepLIO", "build_deeplio", "example_input_deeplio", 2020, MENAGERIE_ZOO),
]
