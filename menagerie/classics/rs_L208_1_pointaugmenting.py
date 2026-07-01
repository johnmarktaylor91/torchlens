# SOURCE: vendored from VISION-SJTU/PointAugmenting @ main
# (det3d/models/readers/pillar_encoder.py, det3d/models/backbones/{__init__,scn}.py's
#  PointPillarsScatter export path, det3d/models/img_backbones/resnet.py,
#  det3d/models/necks/rpn.py, det3d/models/bbox_heads/center_head.py,
#  det3d/models/detectors/{single_stage,pp_fusion}.py, det3d/models/utils/misc.py)
"""PointAugmenting (CVPR 2021): cross-modal LiDAR+camera 3D object detection that
"paints" PointPillars voxel features with CNN image features via camera-projected
grid-sampling, then runs a CenterPoint-style anchor-free detection head.

The official repo (VISION-SJTU/PointAugmenting) is a large ``det3d`` framework with
config-driven module construction (registries), NuScenes/Waymo data loaders, and a
custom-CUDA-op image backbone path (``DLASeg`` uses ``DCNv2``'s ``dcn_v2.DCN``, a
non-base-lib compiled extension, and hardcodes loading local pretrained-weight files
in ``__init__``). The repo's own config comments list ``ResNet18`` as the alternative,
DCN-free ``img_backbone`` (``configs/nusc/pp/nusc_pp_img.py``: ``type="DLASeg", #
ResNet18, DLASeg``) -- this vendors that variant so the fusion architecture traces in
a base-torch environment with random init.

Classes below are copied verbatim (module logic unmodified; only glue removed --
registry decorators, mmcv-style ``builder``/``registry`` indirection, logger
plumbing, and the two hardcoded local-checkpoint ``torch.load(...)`` calls in
``ResNet18.__init__``/``DLASeg.__init__``, replaced by an explicit ``pretrained``
flag defaulting to ``False`` for from-scratch construction):

- ``PFNLayer`` / ``PillarFeatureNet`` (``readers/pillar_encoder.py``): pillar feature
  encoder from PointPillars (Lang & Beijbom 2018), producing per-pillar features from
  raw point clusters.
- ``PointPillarsScatter`` (``readers/pillar_encoder.py``): scatters pillar features
  back into a dense BEV pseudo-image canvas.
- ``BasicBlock`` / ``Bottleneck`` / ``ResNet`` / ``Conv2d`` / ``ResNet18``
  (``img_backbones/resnet.py``): the image branch -- a ResNet-18 trunk (frozen in the
  original training recipe; kept trainable here since we only trace forward) that
  yields the ``layer3`` feature map used for LiDAR-point painting via grid-sample.
- ``RPN`` (``necks/rpn.py``): the multi-scale down/up-sample BEV backbone-neck from
  CenterPoint, producing the fused BEV feature map for detection.
- ``SepHead`` / ``CenterHead`` (``bbox_heads/center_head.py``, non-DCN path only --
  ``DCNSepHead``/``FeatureAdaption`` require the DCN op and are omitted; the config's
  default ``dcn_head=False`` already selects ``SepHead``): CenterPoint-style anchor-free
  multi-task detection head (heatmap + box regression per class group).
- ``PPFusion`` (``detectors/pp_fusion.py`` + ``detectors/single_stage.py``): the top-level
  fusion detector -- projects LiDAR points into the image plane, grid-samples image
  features at those (u, v) locations, concatenates them onto the raw point features
  before pillar encoding, then runs PointPillars-Scatter -> RPN -> CenterHead.

``Sequential``/``Empty``/``get_paddings_indicator``/``build_norm_layer`` are the
repo's small pure-torch utility helpers (``models/utils/misc.py``,
``models/utils/norm.py``), copied verbatim.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# models/utils/misc.py (verbatim, glue-free)
# ---------------------------------------------------------------------------
class Sequential(nn.Module):
    """A sequential container supporting .add(module) like the original det3d util."""

    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class Empty(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args


def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor."""
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    paddings_indicator = actual_num.int() > max_num
    return paddings_indicator


# models/utils/norm.py (verbatim, minus SyncBatchNorm distributed bits which are
# never exercised by the norm_cfg the config below selects -- "BN")
_norm_cfg = {
    "BN": ("bn", nn.BatchNorm2d),
    "BN1d": ("bn1d", nn.BatchNorm1d),
    "GN": ("gn", nn.GroupNorm),
}


def build_norm_layer(cfg, num_features, postfix=""):
    assert isinstance(cfg, dict) and "type" in cfg
    cfg_ = cfg.copy()
    layer_type = cfg_.pop("type")
    abbr, norm_layer = _norm_cfg[layer_type]
    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)
    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    if layer_type != "GN":
        layer = norm_layer(num_features, **cfg_)
    else:
        assert "num_groups" in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return name, layer


# ---------------------------------------------------------------------------
# det3d/models/readers/pillar_encoder.py (verbatim)
# ---------------------------------------------------------------------------
class PFNLayer(nn.Module):
    """Pillar Feature Net Layer (PointPillars, Lang & Beijbom 2018)."""

    def __init__(self, in_channels, out_channels, norm_cfg=None, last_layer=False):
        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs):
        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarFeatureNet(nn.Module):
    """Pillar Feature Net: prepares pillar features, forwards through PFNLayers."""

    def __init__(
        self,
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        pc_range=(0, -40, -3, 70.4, 40, 1),
        norm_cfg=None,
    ):
        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(num_filters) > 0

        self.num_input = num_input_features
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            last_layer = i >= len(num_filters) - 2
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer)
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        dtype = features.dtype

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset
        )
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset
        )

        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


class PointPillarsScatter(nn.Module):
    """Converts learned pillar features from a sparse list to a dense BEV canvas."""

    def __init__(self, num_input_features=64, norm_cfg=None, name="PointPillarsScatter", **kwargs):
        super().__init__()
        self.name = "PointPillarsScatter"
        self.nchannels = num_input_features

    def forward(self, voxel_features, coords, batch_size, input_shape):
        self.nx = input_shape[0]
        self.ny = input_shape[1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            canvas = torch.zeros(
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            canvas[:, indices] = voxels
            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)
        return batch_canvas


# ---------------------------------------------------------------------------
# det3d/models/img_backbones/resnet.py (verbatim; DCN-free ``ResNet18`` branch)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Conv2d(nn.Module):
    """2D convolution optionally with batch-norm and relu (det3d util, verbatim)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        relu=True,
        bn=True,
        bn_momentum=0.1,
        **kwargs,
    ):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, bias=(not bn), **kwargs
        )
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class ResNet18(nn.Module):
    """Image branch used for camera->LiDAR painting (config's DCN-free alternative to
    ``DLASeg``; frozen up to layer3 in the original training recipe)."""

    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()
        self.pretrained = pretrained
        resnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
        # NOTE: original loads a local checkpoint here when pretrained=True;
        # omitted -- this module is constructed with pretrained=False (random init).
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        self.smooth1 = nn.Conv2d(64, 64, 3, 1, 1)
        self.smooth2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.latlayer1 = nn.Conv2d(256, 64, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(128, 64, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(64, 64, 1, 1, 0)
        self.reduce = nn.Sequential(Conv2d(64, 16, 3, 1, padding=1))

    def forward(self, input):
        x = self.conv1(input)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        return x3


# ---------------------------------------------------------------------------
# det3d/models/necks/rpn.py (verbatim)
# ---------------------------------------------------------------------------
class RPN(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        **kwargs,
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = self._upsample_strides[i - self._upsample_start_idx]
                if stride > 1:
                    deblock = Sequential(
                        nn.ConvTranspose2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                else:
                    inv_stride = round(1 / stride)
                    deblock = Sequential(
                        nn.Conv2d(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            inv_stride,
                            stride=inv_stride,
                            bias=False,
                        ),
                        build_norm_layer(
                            self._norm_cfg,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                        )[1],
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        block = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self._norm_cfg, planes)[1])
            block.add(nn.ReLU())

        return block, planes

    def forward(self, x):
        ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)
        return x


# ---------------------------------------------------------------------------
# det3d/models/bbox_heads/center_head.py (verbatim; non-DCN ``SepHead`` path only --
# ``DCNSepHead``/``FeatureAdaption`` omitted, the config's default ``dcn_head=False``
# never constructs them)
# ---------------------------------------------------------------------------
class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(
                    nn.Conv2d(
                        in_channels,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                    )
                )
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )

            if "hm" in head:
                fc[-1].bias.data.fill_(init_bias)

            self.__setattr__(head, fc)

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)
        return ret_dict


class CenterHead(nn.Module):
    def __init__(
        self,
        in_channels=[128],
        tasks=[],
        dataset="nuscenes",
        weight=0.25,
        code_weights=[],
        common_heads=dict(),
        init_bias=-2.19,
        share_conv_channel=64,
        num_hm_conv=2,
        dcn_head=False,
    ):
        super(CenterHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]
        self.code_weights = code_weights
        self.weight = weight
        self.dataset = dataset

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.box_n_dim = 9 if "vel" in common_heads else 7
        self.use_direction_classifier = False

        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, share_conv_channel, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True),
        )

        self.tasks = nn.ModuleList()
        assert not dcn_head, "DCN head path requires the DCNv2 custom op; not vendored."

        for num_cls in num_classes:
            heads = dict(common_heads)
            heads.update(dict(hm=(num_cls, num_hm_conv)))
            self.tasks.append(
                SepHead(share_conv_channel, heads, bn=True, init_bias=init_bias, final_kernel=3)
            )

    def forward(self, x, *kwargs):
        ret_dicts = []
        x = self.shared_conv(x)
        for task in self.tasks:
            ret_dicts.append(task(x))
        return ret_dicts


# ---------------------------------------------------------------------------
# det3d/models/detectors/{single_stage,pp_fusion}.py (verbatim ``forward`` logic;
# registry-based ``builder.build_*`` calls replaced with direct construction of the
# modules above, per the ``configs/nusc/pp/nusc_pp_img.py`` config)
# ---------------------------------------------------------------------------
class PPFusion(nn.Module):
    """PointAugmenting's camera-LiDAR fusion detector: paints per-voxel-point
    features with grid-sampled image-backbone features, then runs
    PillarFeatureNet -> PointPillarsScatter -> RPN -> CenterHead."""

    def __init__(self, reader, backbone, img_backbone, neck, bbox_head, img_feat_num=64):
        super(PPFusion, self).__init__()
        self.reader = reader
        self.backbone = backbone
        self.img_backbone = img_backbone
        self.neck = neck
        self.bbox_head = bbox_head
        self.with_neck = neck is not None

        for name, p in self.img_backbone.named_parameters():
            p.requires_grad = False
        self.img_backbone.eval()

        # NOTE: original hardcodes 64 (DLASeg's fixed output-channel count). Made a
        # constructor arg here since we swap in the config's DCN-free ResNet18
        # alternative image backbone, whose layer3 output has a different channel
        # count (256, from BasicBlock's expansion=1 at planes=256) -- this is purely
        # a wiring constant for the grid-sampled-feature reshape, not an
        # architectural choice.
        self.img_feat_num = img_feat_num
        self.max_points_in_voxel = 20

    def get_img_feat(self, img, pts_uv, voxels_valid):
        # pts_uv: (B, N, max_points_in_voxel, 2) normalized grid-sample coords
        batch_size = img.shape[0]
        with torch.no_grad():
            img = img.view(-1, 3, img.shape[3], img.shape[4])
            img_feat = self.img_backbone(img)
            img_feat = img_feat.view(
                batch_size, 6, -1, img_feat.shape[2], img_feat.shape[3]
            ).transpose(2, 1)

            voxel_img_feat = F.grid_sample(img_feat, pts_uv, mode="bilinear", padding_mode="zeros")

            voxel_img_feat = voxel_img_feat.transpose(1, 4).contiguous()
            voxel_img_feat = voxel_img_feat.view(
                -1, self.max_points_in_voxel, self.img_feat_num
            ).contiguous()
            voxel_img_feat = voxel_img_feat[voxels_valid]
        return voxel_img_feat

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"], data["coors"])
        x = self.backbone(input_features, data["coors"], data["batch_size"], data["input_shape"])
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        voxels_uv = example["voxels_uv"]
        voxels_valid = example["voxel_valid"]

        batch_size = len(num_voxels)

        with torch.no_grad():
            voxels_feat = self.get_img_feat(example["img"], voxels_uv, voxels_valid)
            voxels_feat = voxels_feat * (voxels[:, :, -1].view(-1, self.max_points_in_voxel, 1))
            voxels_feat = torch.cat([voxels[:, :, :-4], voxels_feat], dim=2)

        data = dict(
            features=voxels_feat,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        preds = self.bbox_head(x)
        return preds


# ---------------------------------------------------------------------------
# Staging build/example helpers (matching configs/nusc/pp/nusc_pp_img.py, scaled
# down to a tiny voxel grid / class-task set for a fast, faithful-architecture trace)
# ---------------------------------------------------------------------------
_TASKS = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
]
_VOXEL_SIZE = (0.8, 0.8, 8)
_PC_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
_NX, _NY = 8, 8  # tiny BEV grid (real config uses 128x128)
_MAX_POINTS_IN_VOXEL = 20
_N_VOXELS = 12
_IMG_FEAT_NUM = 256  # ResNet18(BasicBlock).layer3 output channels
_NUM_INPUT_FEATURES = 5 + _IMG_FEAT_NUM


def build_pointaugmenting():
    reader = PillarFeatureNet(
        num_filters=[64, 64],
        num_input_features=_NUM_INPUT_FEATURES,
        with_distance=False,
        voxel_size=_VOXEL_SIZE,
        pc_range=_PC_RANGE,
    )
    img_backbone = ResNet18(pretrained=False)
    backbone = PointPillarsScatter(num_input_features=64)
    neck = RPN(
        layer_nums=[1, 1, 1],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[32, 32, 32],
        num_input_features=64,
    )
    bbox_head = CenterHead(
        in_channels=sum([32, 32, 32]),
        tasks=_TASKS,
        dataset="nuscenes",
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={
            "reg": (2, 2),
            "height": (1, 2),
            "dim": (3, 2),
            "rot": (2, 2),
            "vel": (2, 2),
        },
    )
    model = PPFusion(reader, backbone, img_backbone, neck, bbox_head, img_feat_num=256)
    # The original recipe only ever runs this fusion detector at inference time with
    # BatchNorm in eval mode (the img_backbone is explicitly frozen/eval'd in
    # __init__, and training uses much larger batches/BEV grids than this tiny
    # architecture-preserving trace input); .eval() avoids a spurious
    # batch-size-1-spatial-1x1 BatchNorm training-mode error that is an artifact of
    # the deliberately tiny synthetic BEV grid, not of the architecture itself.
    model.eval()
    return model


def example_input_pointaugmenting():
    n_voxels = _N_VOXELS
    # 5 raw geometric/intensity features + 4 trailing columns; forward() keeps only
    # the first (9-4)=5 raw channels (voxels[:, :, :-4]) and treats the last column
    # as a point-validity multiplicative mask, exactly as det3d's PPFusion.forward.
    voxels = torch.randn(n_voxels, _MAX_POINTS_IN_VOXEL, 9)
    voxels[:, :, -1] = 1.0
    coordinates = torch.zeros(n_voxels, 4, dtype=torch.long)
    coordinates[:, 0] = 0  # batch index
    coordinates[:, 2] = torch.randint(0, _NY, (n_voxels,))
    coordinates[:, 3] = torch.randint(0, _NX, (n_voxels,))
    num_points = torch.full((n_voxels,), _MAX_POINTS_IN_VOXEL, dtype=torch.long)
    num_voxels = torch.tensor([n_voxels], dtype=torch.long)
    img = torch.randn(1, 6, 3, 64, 64)  # (B, 6 cams, 3, H, W)
    # get_img_feat 5D-grid-samples img_feat shaped (B, C, n_cams, H, W) with a grid of
    # shape (B, D_out=1, H_out=n_voxels, W_out=max_points_in_voxel, 3) -- the trailing
    # 3 coords are (u, v, cam_id) normalized to [-1, 1] per torch's 5D grid_sample.
    voxels_uv = torch.rand(1, 1, n_voxels, _MAX_POINTS_IN_VOXEL, 3) * 2 - 1
    voxel_valid = torch.ones(n_voxels, dtype=torch.bool)

    example = dict(
        voxels=voxels,
        coordinates=coordinates,
        num_points=num_points,
        num_voxels=num_voxels,
        voxels_uv=voxels_uv,
        voxel_valid=voxel_valid,
        img=img,
        shape=[(_NX, _NY, 1)],
    )
    return (example,)


MENAGERIE_ENTRIES = [
    (
        "PointAugmenting",
        "build_pointaugmenting",
        "example_input_pointaugmenting",
        "2021",
        "CV",
    )
]

MENAGERIE_ZOO = "vendored-pytorch"
