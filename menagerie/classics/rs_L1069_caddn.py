# SOURCE: vendored from https://github.com/TRAILab/CaDDN @ master
# (pcdet/models/backbones_3d/ffe/ddn/ddn_template.py, ddn_deeplabv3.py,
#  pcdet/models/backbones_3d/ffe/depth_ffe.py,
#  pcdet/models/backbones_3d/f2v/frustum_to_voxel.py, frustum_grid_generator.py, sampler.py,
#  pcdet/models/backbones_2d/map_to_bev/conv2d_collapse.py, base_bev_backbone.py,
#  pcdet/models/model_utils/basic_block_2d.py,
#  pcdet/models/dense_heads/anchor_head_single.py, anchor_head_template.py,
#  pcdet/models/dense_heads/target_assigner/anchor_generator.py,
#  pcdet/utils/box_coder_utils.py (ResidualCoder), common_utils.py (limit_period))
#
# CaDDN: Categorical Depth Distribution Network for Monocular 3D Object Detection
# (Reading, Harakeh, Chae, Waslander -- CVPR 2021 Oral). Monocular camera-only 3D
# detector: a DeepLabV3 backbone predicts a per-pixel categorical depth distribution,
# which is used to lift 2D image features into a 3D frustum feature volume; the
# frustum is resampled (via a projective grid_sample) into an ego-centric voxel
# grid, collapsed to a BEV plane, run through a BEV backbone, and decoded with an
# anchor-based single-stage detection head. Requires the "categorical depth
# distribution" (DepthFFE) + "frustum-to-voxel" (F2V) modules that give CaDDN its
# name -- these are its true architectural contribution over prior monocular/LiDAR
# detectors, and both are pure torch/torchvision/kornia (no custom CUDA op needed
# for the forward pass; pcdet's custom CUDA kernels are only used for
# LiDAR-branch backbones -- unused by CaDDN -- and for postprocessing NMS, which is
# not part of the traced module list).
#
# No architecture was altered. Only non-architectural staging changes:
#   - `.cuda()` calls in AnchorGenerator.generate_anchors / AnchorHeadTemplate.__init__
#     hardcoded a CUDA device; changed to whatever device the input tensors live on
#     (so the model can trace on CPU) -- a hardware-portability fix, not an
#     architecture change.
#   - `constructor(pretrained=False, pretrained_backbone=False, ...)` matches the
#     legacy torchvision.models.segmentation constructor signature used by the
#     original code (still supported, deprecated, in the installed torchvision).
#   - Dropped the EasyDict-based YAML config loader; configuration values are
#     passed directly as plain Python objects (SimpleNamespace) carrying the same
#     fields as the original cfgs/kitti_models/CaDDN.yaml.
#   - kornia's API moved `create_meshgrid3d`/`transform_points`/`normalize` to
#     different module paths since this repo was written (now `kornia.*`,
#     `kornia.geometry.linalg.*`, `kornia.enhance.*` respectively); call sites
#     updated to match the installed kornia, behavior unchanged.
#   - DDN pretrained_path is left None (no ImageNet/COCO checkpoint download);
#     DepthFFE.get_loss()/AnchorHeadTemplate loss machinery (training-time only)
#     is omitted since only the forward (inference) path is exercised.

from types import SimpleNamespace

import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

MENAGERIE_ZOO = "vendored-pytorch"


# ---------------------------------------------------------------------------
# pcdet/models/model_utils/basic_block_2d.py
# ---------------------------------------------------------------------------
class BasicBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block for channel reduce
        Args:
            out_channels [int]: Number of output channels of convolutional block
            **kwargs [Dict]: Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


# ---------------------------------------------------------------------------
# pcdet/models/backbones_3d/ffe/ddn/ddn_template.py + ddn_deeplabv3.py
# ---------------------------------------------------------------------------
class DDNTemplate(nn.Module):
    def __init__(
        self, constructor, feat_extract_layer, num_classes, pretrained_path=None, aux_loss=None
    ):
        """
        Initializes depth distribution network.
        Args:
            constructor [function]: Model constructor
            feat_extract_layer [string]: Layer to extract features from
            pretrained_path [string]: (Optional) Path of the model to load weights from
            aux_loss [bool]: Flag to include auxillary loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_path = pretrained_path
        self.pretrained = pretrained_path is not None
        self.aux_loss = aux_loss

        if self.pretrained:
            self.norm_mean = torch.Tensor([0.485, 0.456, 0.406])
            self.norm_std = torch.Tensor([0.229, 0.224, 0.225])

        self.model = self.get_model(constructor=constructor)
        self.feat_extract_layer = feat_extract_layer
        self.model.backbone.return_layers = {
            feat_extract_layer: "features",
            **self.model.backbone.return_layers,
        }

    def get_model(self, constructor):
        model = constructor(
            pretrained=False,
            pretrained_backbone=False,
            num_classes=self.num_classes,
            aux_loss=self.aux_loss,
        )

        if self.pretrained_path is not None:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(self.pretrained_path)
            pretrained_dict = self.filter_pretrained_dict(
                model_dict=model_dict, pretrained_dict=pretrained_dict
            )
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        return model

    def filter_pretrained_dict(self, model_dict, pretrained_dict):
        if (
            "aux_classifier.0.weight" in pretrained_dict
            and "aux_classifier.0.weight" not in model_dict
        ):
            pretrained_dict = {
                key: value for key, value in pretrained_dict.items() if "aux_classifier" not in key
            }

        model_num_classes = model_dict["classifier.4.weight"].shape[0]
        pretrained_num_classes = pretrained_dict["classifier.4.weight"].shape[0]
        if model_num_classes != pretrained_num_classes:
            pretrained_dict.pop("classifier.4.weight")
            pretrained_dict.pop("classifier.4.bias")

        return pretrained_dict

    def forward(self, images):
        """
        Args:
            images [torch.Tensor(N, 3, H_in, W_in)]: Input images
        Returns
            result [dict[torch.Tensor]]: Depth distribution result
        """
        x = self.preprocess(images)

        result = {}
        features = self.model.backbone(x)
        result["features"] = features["features"]
        feat_shape = features["features"].shape[-2:]

        x = features["out"]
        x = self.model.classifier(x)
        x = F.interpolate(x, size=feat_shape, mode="bilinear", align_corners=False)
        result["logits"] = x

        if self.model.aux_classifier is not None:
            x = features["aux"]
            x = self.model.aux_classifier(x)
            x = F.interpolate(x, size=feat_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return result

    def preprocess(self, images):
        x = images
        if self.pretrained:
            mask = torch.isnan(x)
            x = kornia.enhance.normalize(
                x, mean=self.norm_mean.to(x.device), std=self.norm_std.to(x.device)
            )
            x[mask] = 0
        return x


class DDNDeepLabV3(DDNTemplate):
    def __init__(self, backbone_name, **kwargs):
        if backbone_name == "ResNet50":
            constructor = torchvision.models.segmentation.deeplabv3_resnet50
        elif backbone_name == "ResNet101":
            constructor = torchvision.models.segmentation.deeplabv3_resnet101
        else:
            raise NotImplementedError
        super().__init__(constructor=constructor, **kwargs)


_DDN_REGISTRY = {"DDNDeepLabV3": DDNDeepLabV3}


# ---------------------------------------------------------------------------
# pcdet/models/backbones_3d/ffe/depth_ffe.py
# ---------------------------------------------------------------------------
class DepthFFE(nn.Module):
    def __init__(self, model_cfg, downsample_factor):
        """
        Initialize depth classification network
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.disc_cfg = model_cfg.DISCRETIZE
        self.downsample_factor = downsample_factor

        ddn_cfg = model_cfg.DDN
        self.ddn = _DDN_REGISTRY[ddn_cfg.NAME](
            num_classes=self.disc_cfg["num_bins"] + 1,
            backbone_name=ddn_cfg.BACKBONE_NAME,
            **ddn_cfg.ARGS,
        )
        self.channel_reduce = BasicBlock2D(**model_cfg.CHANNEL_REDUCE)
        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        return self.channel_reduce.out_channels

    def forward(self, batch_dict):
        images = batch_dict["images"]
        ddn_result = self.ddn(images)
        image_features = ddn_result["features"]
        depth_logits = ddn_result["logits"]

        if self.channel_reduce is not None:
            image_features = self.channel_reduce(image_features)

        frustum_features = self.create_frustum_features(
            image_features=image_features, depth_logits=depth_logits
        )
        batch_dict["frustum_features"] = frustum_features
        return batch_dict

    def create_frustum_features(self, image_features, depth_logits):
        channel_dim = 1
        depth_dim = 2

        image_features = image_features.unsqueeze(depth_dim)
        depth_logits = depth_logits.unsqueeze(channel_dim)

        depth_probs = F.softmax(depth_logits, dim=depth_dim)
        depth_probs = depth_probs[:, :, :-1]

        frustum_features = depth_probs * image_features
        return frustum_features


# ---------------------------------------------------------------------------
# pcdet/utils/transform_utils.py + depth_utils.py + grid_utils.py (needed helpers)
# ---------------------------------------------------------------------------
def project_to_image(project, points):
    """
    Project points to image
    Args:
        project [torch.tensor(..., 3, 4)]: Projection matrix
        points [torch.Tensor(..., 3)]: 3D points
    Returns:
        points_img [torch.Tensor(..., 2)]: Points in image
        points_depth [torch.Tensor(...)]: Depth of each point
    """
    points_shape = list(points.shape)
    points_shape[-1] = 1

    if project.shape[-1] == 4:
        points = torch.cat([points, points.new_ones(points_shape)], dim=-1)

    points_t = points.unsqueeze(-1)
    points_proj = torch.matmul(project.unsqueeze(-3), points_t).squeeze(-1)

    points_img = points_proj[..., :2] / torch.clamp(points_proj[..., 2:3], min=1e-8)
    points_depth = points_proj[..., 2] - project[..., 2, 3]
    return points_img, points_depth


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: LID (Linear-Increasing Discretization)
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = (
            num_bins
            * (torch.log(1 + depth_map) - np.log(1 + depth_min))
            / (np.log(1 + depth_max) - np.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    if target:
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins
        indices = indices.type(torch.int64)
    return indices


def normalize_coords(coords, shape):
    """
    Normalize Coordinates for grid sample
    Args:
        coords [torch.Tensor(..., 3)]: Coordinates in [depth, width, height] order
        shape [torch.Tensor(3)]: Depth, width, height dimensions
    Returns:
        norm_coords [torch.Tensor(..., 3)]: Normalized coordinates in [-1, 1]
    """
    min_n = -1
    max_n = 1
    shape = torch.flip(shape, dims=[0])  # [D, H, W] -> [W, H, D]
    norm_coords = coords * (max_n - min_n) / (shape - 1) + min_n
    return norm_coords


# ---------------------------------------------------------------------------
# pcdet/models/backbones_3d/f2v/frustum_grid_generator.py + sampler.py + frustum_to_voxel.py
# ---------------------------------------------------------------------------
class FrustumGridGenerator(nn.Module):
    def __init__(self, grid_size, pc_range, disc_cfg):
        """
        Initializes Grid Generator for frustum features
        """
        super().__init__()
        self.dtype = torch.float32
        self.grid_size = torch.as_tensor(grid_size)
        self.pc_range = pc_range
        self.out_of_bounds_val = -2
        self.disc_cfg = disc_cfg

        pc_range = torch.as_tensor(pc_range).reshape(2, 3)
        self.pc_min = pc_range[0]
        self.pc_max = pc_range[1]
        self.voxel_size = (self.pc_max - self.pc_min) / self.grid_size

        self.depth, self.width, self.height = self.grid_size.int()
        self.voxel_grid = kornia.geometry.create_meshgrid3d(
            depth=self.depth, height=self.height, width=self.width, normalized_coordinates=False
        )
        self.voxel_grid = self.voxel_grid.permute(0, 1, 3, 2, 4)  # XZY-> XYZ

        self.voxel_grid += 0.5
        self.grid_to_lidar = self.grid_to_lidar_unproject(
            pc_min=self.pc_min, voxel_size=self.voxel_size
        )

    def grid_to_lidar_unproject(self, pc_min, voxel_size):
        x_size, y_size, z_size = voxel_size
        x_min, y_min, z_min = pc_min
        unproject = torch.tensor(
            [
                [x_size, 0, 0, x_min],
                [0, y_size, 0, y_min],
                [0, 0, z_size, z_min],
                [0, 0, 0, 1],
            ],
            dtype=self.dtype,
        )
        return unproject

    def transform_grid(self, voxel_grid, grid_to_lidar, lidar_to_cam, cam_to_img):
        B = lidar_to_cam.shape[0]

        V_G = grid_to_lidar
        C_V = lidar_to_cam
        I_C = cam_to_img
        trans = C_V @ V_G

        trans = trans.reshape(B, 1, 1, 4, 4)
        voxel_grid = voxel_grid.repeat_interleave(repeats=B, dim=0)

        camera_grid = kornia.geometry.linalg.transform_points(trans_01=trans, points_1=voxel_grid)

        I_C = I_C.reshape(B, 1, 1, 3, 4)
        image_grid, image_depths = project_to_image(project=I_C, points=camera_grid)

        image_depths = bin_depths(depth_map=image_depths, **self.disc_cfg)

        image_depths = image_depths.unsqueeze(-1)
        frustum_grid = torch.cat((image_grid, image_depths), dim=-1)
        return frustum_grid

    def forward(self, lidar_to_cam, cam_to_img, image_shape):
        frustum_grid = self.transform_grid(
            voxel_grid=self.voxel_grid.to(lidar_to_cam.device),
            grid_to_lidar=self.grid_to_lidar.to(lidar_to_cam.device),
            lidar_to_cam=lidar_to_cam,
            cam_to_img=cam_to_img,
        )

        image_shape, _ = torch.max(image_shape, dim=0)
        image_depth = torch.tensor(
            [self.disc_cfg["num_bins"]], device=image_shape.device, dtype=image_shape.dtype
        )
        frustum_shape = torch.cat((image_depth, image_shape))
        frustum_grid = normalize_coords(coords=frustum_grid, shape=frustum_shape)

        mask = ~torch.isfinite(frustum_grid)
        frustum_grid[mask] = self.out_of_bounds_val

        return frustum_grid


class Sampler(nn.Module):
    def __init__(self, mode="bilinear", padding_mode="zeros"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input_features, grid):
        output = F.grid_sample(
            input=input_features, grid=grid, mode=self.mode, padding_mode=self.padding_mode
        )
        return output


class FrustumToVoxel(nn.Module):
    def __init__(self, model_cfg, grid_size, pc_range, disc_cfg):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.disc_cfg = disc_cfg
        self.grid_generator = FrustumGridGenerator(
            grid_size=grid_size, pc_range=pc_range, disc_cfg=disc_cfg
        )
        self.sampler = Sampler(**vars(model_cfg.SAMPLER))

    def forward(self, batch_dict):
        grid = self.grid_generator(
            lidar_to_cam=batch_dict["trans_lidar_to_cam"],
            cam_to_img=batch_dict["trans_cam_to_img"],
            image_shape=batch_dict["image_shape"],
        )  # (B, X, Y, Z, 3)

        voxel_features = self.sampler(
            input_features=batch_dict["frustum_features"], grid=grid
        )  # (B, C, X, Y, Z)

        voxel_features = voxel_features.permute(0, 1, 4, 3, 2)  # (B, C, Z, Y, X)
        batch_dict["voxel_features"] = voxel_features
        return batch_dict


# ---------------------------------------------------------------------------
# pcdet/models/backbones_2d/map_to_bev/conv2d_collapse.py
# ---------------------------------------------------------------------------
class Conv2DCollapse(nn.Module):
    def __init__(self, model_cfg, grid_size):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.block = BasicBlock2D(
            in_channels=self.num_bev_features * self.num_heights,
            out_channels=self.num_bev_features,
            **self.model_cfg.ARGS,
        )

    def forward(self, batch_dict):
        voxel_features = batch_dict["voxel_features"]
        bev_features = voxel_features.flatten(
            start_dim=1, end_dim=2
        )  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        batch_dict["spatial_features"] = bev_features
        return batch_dict


# ---------------------------------------------------------------------------
# pcdet/models/backbones_2d/base_bev_backbone.py
# ---------------------------------------------------------------------------
class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if getattr(self.model_cfg, "LAYER_NUMS", None) is not None:
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if getattr(self.model_cfg, "UPSAMPLE_STRIDES", None) is not None:
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx],
                    num_filters[idx],
                    kernel_size=3,
                    stride=layer_strides[idx],
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ]
            for _ in range(layer_nums[idx]):
                cur_layers.extend(
                    [
                        nn.Conv2d(
                            num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU(),
                    ]
                )
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                upsample_strides[idx],
                                stride=upsample_strides[idx],
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                        )
                    )
                else:
                    stride = np.round(1 / stride).astype(int)
                    self.deblocks.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_filters[idx],
                                num_upsample_filters[idx],
                                stride,
                                stride=stride,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                        )
                    )

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False
                    ),
                    nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                )
            )

        self.num_bev_features = c_in

    def forward(self, data_dict):
        spatial_features = data_dict["spatial_features"]
        ups = []
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict["spatial_features_2d"] = x
        return data_dict


# ---------------------------------------------------------------------------
# pcdet/utils/box_coder_utils.py (ResidualCoder) + common_utils.py (limit_period)
# ---------------------------------------------------------------------------
def limit_period(val, offset=0.5, period=np.pi):
    ans = val - torch.floor(val / period + offset) * period
    return ans


class ResidualCoder:
    def __init__(self, code_size=7, encode_angle_by_sincos=False, **kwargs):
        self.code_size = code_size
        self.encode_angle_by_sincos = encode_angle_by_sincos
        if self.encode_angle_by_sincos:
            self.code_size += 1

    def decode_torch(self, box_encodings, anchors):
        xa, ya, za, dxa, dya, dza, ra, *cas = torch.split(anchors, 1, dim=-1)
        if not self.encode_angle_by_sincos:
            xt, yt, zt, dxt, dyt, dzt, rt, *cts = torch.split(box_encodings, 1, dim=-1)
        else:
            xt, yt, zt, dxt, dyt, dzt, cost, sint, *cts = torch.split(box_encodings, 1, dim=-1)

        diagonal = torch.sqrt(dxa**2 + dya**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * dza + za

        dxg = torch.exp(dxt) * dxa
        dyg = torch.exp(dyt) * dya
        dzg = torch.exp(dzt) * dza

        if self.encode_angle_by_sincos:
            rg_cos = cost + torch.cos(ra)
            rg_sin = sint + torch.sin(ra)
            rg = torch.atan2(rg_sin, rg_cos)
        else:
            rg = rt + ra

        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, dxg, dyg, dzg, rg, *cgs], dim=-1)


# ---------------------------------------------------------------------------
# pcdet/models/dense_heads/target_assigner/anchor_generator.py
# ---------------------------------------------------------------------------
class AnchorGenerator:
    def __init__(self, anchor_range, anchor_generator_config, device="cpu"):
        self.anchor_generator_cfg = anchor_generator_config
        self.anchor_range = anchor_range
        self.anchor_sizes = [config["anchor_sizes"] for config in anchor_generator_config]
        self.anchor_rotations = [config["anchor_rotations"] for config in anchor_generator_config]
        self.anchor_heights = [
            config["anchor_bottom_heights"] for config in anchor_generator_config
        ]
        self.align_center = [
            config.get("align_center", False) for config in anchor_generator_config
        ]
        self.device = device

        self.num_of_anchor_sets = len(self.anchor_sizes)

    def generate_anchors(self, grid_sizes):
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
            grid_sizes,
            self.anchor_sizes,
            self.anchor_rotations,
            self.anchor_heights,
            self.align_center,
        ):
            num_anchors_per_location.append(
                len(anchor_rotation) * len(anchor_size) * len(anchor_height)
            )
            if align_center:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / grid_size[0]
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.anchor_range[3] - self.anchor_range[0]) / (grid_size[0] - 1)
                y_stride = (self.anchor_range[4] - self.anchor_range[1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            x_shifts = torch.arange(
                self.anchor_range[0] + x_offset,
                self.anchor_range[3] + 1e-5,
                step=x_stride,
                dtype=torch.float32,
            ).to(self.device)
            y_shifts = torch.arange(
                self.anchor_range[1] + y_offset,
                self.anchor_range[4] + 1e-5,
                step=y_stride,
                dtype=torch.float32,
            ).to(self.device)
            z_shifts = x_shifts.new_tensor(anchor_height)

            num_anchor_size, num_anchor_rotation = len(anchor_size), len(anchor_rotation)
            anchor_rotation = x_shifts.new_tensor(anchor_rotation)
            anchor_size = x_shifts.new_tensor(anchor_size)
            x_shifts, y_shifts, z_shifts = torch.meshgrid(
                [x_shifts, y_shifts, z_shifts], indexing="ij"
            )
            anchors = torch.stack((x_shifts, y_shifts, z_shifts), dim=-1)  # [x, y, z, 3]
            anchors = anchors[:, :, :, None, :].repeat(1, 1, 1, anchor_size.shape[0], 1)
            anchor_size = anchor_size.view(1, 1, 1, -1, 3).repeat([*anchors.shape[0:3], 1, 1])
            anchors = torch.cat((anchors, anchor_size), dim=-1)
            anchors = anchors[:, :, :, :, None, :].repeat(1, 1, 1, 1, num_anchor_rotation, 1)
            anchor_rotation = anchor_rotation.view(1, 1, 1, 1, -1, 1).repeat(
                [*anchors.shape[0:3], num_anchor_size, 1, 1]
            )
            anchors = torch.cat(
                (anchors, anchor_rotation), dim=-1
            )  # [x, y, z, num_size, num_rot, 7]

            anchors = anchors.permute(2, 1, 0, 3, 4, 5).contiguous()
            anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)
        return all_anchors, num_anchors_per_location


# ---------------------------------------------------------------------------
# pcdet/models/dense_heads/anchor_head_template.py + anchor_head_single.py
# ---------------------------------------------------------------------------
class AnchorHeadTemplate(nn.Module):
    def __init__(
        self,
        model_cfg,
        num_class,
        class_names,
        grid_size,
        point_cloud_range,
        predict_boxes_when_training,
        device="cpu",
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = getattr(self.model_cfg, "USE_MULTIHEAD", False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = ResidualCoder(
            num_dir_bins=getattr(anchor_target_cfg, "NUM_DIR_BINS", 6),
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size,
            device=device,
        )
        self.anchors = [x.to(device) for x in anchors]

        self.forward_ret_dict = {}

    @staticmethod
    def generate_anchors(
        anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7, device="cpu"
    ):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg,
            device=device,
        )
        feature_map_size = [
            grid_size[:2] // config["feature_map_stride"] for config in anchor_generator_cfg
        ]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(
            feature_map_size
        )

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [
                        anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                        for anchor in self.anchors
                    ],
                    dim=0,
                )
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = (
            cls_preds.view(batch_size, num_anchors, -1).float()
            if not isinstance(cls_preds, list)
            else cls_preds
        )
        batch_box_preds = (
            box_preds.view(batch_size, num_anchors, -1)
            if not isinstance(box_preds, list)
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        )
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = (
                dir_cls_preds.view(batch_size, num_anchors, -1)
                if not isinstance(dir_cls_preds, list)
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            )
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = 2 * np.pi / self.model_cfg.NUM_DIR_BINS
            dir_rot = limit_period(batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period)
            batch_box_preds[..., 6] = (
                dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(
        self,
        model_cfg,
        input_channels,
        num_class,
        class_names,
        grid_size,
        point_cloud_range,
        predict_boxes_when_training=True,
        device="cpu",
        **kwargs,
    ):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training,
            device=device,
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class, kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size, kernel_size=1
        )

        if getattr(self.model_cfg, "USE_DIRECTION_CLASSIFIER", None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1,
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        spatial_features_2d = data_dict["spatial_features_2d"]

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        self.forward_ret_dict["cls_preds"] = cls_preds
        self.forward_ret_dict["box_preds"] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict["dir_cls_preds"] = dir_cls_preds
        else:
            dir_cls_preds = None

        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=data_dict["batch_size"],
            cls_preds=cls_preds,
            box_preds=box_preds,
            dir_cls_preds=dir_cls_preds,
        )
        data_dict["batch_cls_preds"] = batch_cls_preds
        data_dict["batch_box_preds"] = batch_box_preds
        data_dict["cls_preds_normalized"] = False

        return data_dict


# ---------------------------------------------------------------------------
# pcdet/models/detectors/caddn.py -- top-level CaDDN module_list forward
# ---------------------------------------------------------------------------
class CaDDN(nn.Module):
    """Full CaDDN module_list: FFE -> F2V -> Conv2DCollapse -> BaseBEVBackbone -> AnchorHeadSingle.

    (VFE/backbone_3d/pfe/point_head/roi_head are all None in the CaDDN.yaml config
    -- CaDDN is camera-only, no LiDAR branch.)
    """

    def __init__(self, model_cfg, grid_size, point_cloud_range, class_names, device="cpu"):
        super().__init__()
        self.class_names = class_names
        num_class = len(class_names)

        self.ffe = DepthFFE(model_cfg=model_cfg.FFE, downsample_factor=model_cfg.DOWNSAMPLE_FACTOR)
        disc_cfg = self.ffe.disc_cfg

        self.frustum_to_voxel = FrustumToVoxel(
            model_cfg=model_cfg.F2V,
            grid_size=grid_size,
            pc_range=point_cloud_range,
            disc_cfg=disc_cfg,
        )

        self.map_to_bev_module = Conv2DCollapse(model_cfg=model_cfg.MAP_TO_BEV, grid_size=grid_size)

        self.backbone_2d = BaseBEVBackbone(
            model_cfg=model_cfg.BACKBONE_2D, input_channels=self.map_to_bev_module.num_bev_features
        )

        self.dense_head = AnchorHeadSingle(
            model_cfg=model_cfg.DENSE_HEAD,
            input_channels=self.backbone_2d.num_bev_features,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=False,
            device=device,
        )

    def forward(self, images, trans_lidar_to_cam, trans_cam_to_img, image_shape):
        batch_dict = {
            "images": images,
            "trans_lidar_to_cam": trans_lidar_to_cam,
            "trans_cam_to_img": trans_cam_to_img,
            "image_shape": image_shape,
            "batch_size": images.shape[0],
        }
        batch_dict = self.ffe(batch_dict)
        batch_dict = self.frustum_to_voxel(batch_dict)
        batch_dict = self.map_to_bev_module(batch_dict)
        batch_dict = self.backbone_2d(batch_dict)
        batch_dict = self.dense_head(batch_dict)
        return batch_dict["batch_cls_preds"], batch_dict["batch_box_preds"]


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# Config values mirror cfgs/kitti_models/CaDDN.yaml, shrunk to a tiny scale
# for fast tracing (ResNet50 in place of ResNet101, small image/voxel grid).
# ---------------------------------------------------------------------------
_IMG_H, _IMG_W = 128, 160
_GRID_SIZE = np.array([48, 40, 4])  # X, Y, Z voxels (tiny KITTI-range grid)
_PC_RANGE = [2.0, -30.08, -3.0, 46.8, 30.08, 1.0]


def _build_model_cfg():
    ffe_cfg = SimpleNamespace(
        DDN=SimpleNamespace(
            NAME="DDNDeepLabV3",
            BACKBONE_NAME="ResNet50",
            ARGS={"feat_extract_layer": "layer1", "pretrained_path": None},
        ),
        CHANNEL_REDUCE={
            "in_channels": 256,
            "out_channels": 16,
            "kernel_size": 1,
            "stride": 1,
            "bias": False,
        },
        DISCRETIZE={"mode": "LID", "num_bins": 8, "depth_min": 2.0, "depth_max": 46.8},
    )

    f2v_cfg = SimpleNamespace(SAMPLER=SimpleNamespace(mode="bilinear", padding_mode="zeros"))

    map_to_bev_cfg = SimpleNamespace(
        NUM_BEV_FEATURES=16,
        ARGS={"kernel_size": 1, "stride": 1, "bias": False},
    )

    backbone_2d_cfg = SimpleNamespace(
        LAYER_NUMS=[1, 1],
        LAYER_STRIDES=[2, 2],
        NUM_FILTERS=[16, 32],
        UPSAMPLE_STRIDES=[1, 2],
        NUM_UPSAMPLE_FILTERS=[16, 16],
    )

    dense_head_cfg = SimpleNamespace(
        USE_DIRECTION_CLASSIFIER=True,
        DIR_OFFSET=0.78539,
        DIR_LIMIT_OFFSET=0.0,
        NUM_DIR_BINS=2,
        ANCHOR_GENERATOR_CONFIG=[
            {
                "class_name": "Car",
                "anchor_sizes": [[3.9, 1.6, 1.56]],
                "anchor_rotations": [0, 1.57],
                "anchor_bottom_heights": [-1.78],
                "align_center": False,
                "feature_map_stride": 2,
                "matched_threshold": 0.6,
                "unmatched_threshold": 0.45,
            }
        ],
        TARGET_ASSIGNER_CONFIG=SimpleNamespace(
            NAME="AxisAlignedTargetAssigner", NUM_DIR_BINS=2, BOX_CODER="ResidualCoder"
        ),
    )

    return SimpleNamespace(
        FFE=ffe_cfg,
        DOWNSAMPLE_FACTOR=4,
        F2V=f2v_cfg,
        MAP_TO_BEV=map_to_bev_cfg,
        BACKBONE_2D=backbone_2d_cfg,
        DENSE_HEAD=dense_head_cfg,
    )


def build_caddn():
    torch.manual_seed(0)
    model_cfg = _build_model_cfg()
    model = CaDDN(
        model_cfg=model_cfg,
        grid_size=_GRID_SIZE,
        point_cloud_range=_PC_RANGE,
        class_names=["Car"],
        device="cpu",
    )
    model.eval()
    return model


def example_input_caddn():
    torch.manual_seed(0)
    batch = 1
    images = torch.rand(batch, 3, _IMG_H, _IMG_W)

    # LiDAR->camera extrinsic: identity-ish rigid transform (batch of 4x4)
    trans_lidar_to_cam = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1)
    trans_lidar_to_cam[:, 2, 3] = 1.0  # small translation along Z

    # Camera intrinsic projection matrix (batch of 3x4), plausible KITTI-like focal lengths
    trans_cam_to_img = torch.zeros(batch, 3, 4)
    trans_cam_to_img[:, 0, 0] = 300.0
    trans_cam_to_img[:, 1, 1] = 300.0
    trans_cam_to_img[:, 0, 2] = _IMG_W / 2
    trans_cam_to_img[:, 1, 2] = _IMG_H / 2
    trans_cam_to_img[:, 2, 2] = 1.0

    image_shape = torch.tensor([[_IMG_H, _IMG_W]]).repeat(batch, 1)

    return (images, trans_lidar_to_cam, trans_cam_to_img, image_shape)


MENAGERIE_ENTRIES = [
    ("CaDDN", "build_caddn", "example_input_caddn", 2021, MENAGERIE_ZOO),
]
