# SOURCE: vendored from devendrachaplot/Neural-SLAM @ master
# https://github.com/devendrachaplot/Neural-SLAM/blob/master/model.py
# https://github.com/devendrachaplot/Neural-SLAM/blob/master/utils/model.py
# "Learning to Explore using Active Neural SLAM" (Chaplot, Gandhi, Gupta,
# Gupta, Salakhutdinov, ICLR 2020). `Neural_SLAM_Module` is the real
# `nn.Module` from `model.py`: a ResNet-18 visual encoder + deconvolutional
# egocentric-map decoder + pose estimator + spatial-transformer map
# registration. Copied verbatim from the real repo; only relative imports
# (`utils.model` -> module-local `get_grid`/`ChannelPool`/`Flatten`/`NNBase`)
# and the `args` object (replaced with a plain `SimpleNamespace` populated
# from `arguments.py`'s real defaults) were adjusted so it runs standalone.
"""SOURCE: vendored Neural-SLAM `Neural_SLAM_Module` (Chaplot et al., ICLR 2020)."""

from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# utils/model.py (verbatim, only the AddBias helper -- unused by
# Neural_SLAM_Module -- is dropped to keep the staging module lean).
# ---------------------------------------------------------------------------
def get_grid(pose: torch.Tensor, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    t = t * np.pi / 180.0
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t, torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t, torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack(
        [
            torch.ones(x.shape).to(device),
            -torch.zeros(x.shape).to(device),
            x,
        ],
        1,
    )
    theta22 = torch.stack(
        [
            torch.zeros(x.shape).to(device),
            torch.ones(x.shape).to(device),
            y,
        ],
        1,
    )
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = x.contiguous()
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def rec_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks[:, None])
        else:
            N = hxs.size(0)
            T = int(x.size(0) / N)
            x = x.view(T, N, x.size(1))
            masks = masks.view(T, N, 1)
            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)
            x = torch.stack(outputs, dim=0)
            x = x.view(T * N, -1)
        return x, hxs


# ---------------------------------------------------------------------------
# model.py::Neural_SLAM_Module (verbatim architecture from the real repo)
# ---------------------------------------------------------------------------
class Neural_SLAM_Module(nn.Module):
    """"""

    def __init__(self, args):
        super(Neural_SLAM_Module, self).__init__()

        self.device = args.device
        self.screen_h = args.frame_height
        self.screen_w = args.frame_width
        self.resolution = args.map_resolution
        self.map_size_cm = args.map_size_cm // args.global_downscaling
        self.n_channels = 3
        self.vision_range = args.vision_range
        self.dropout = 0.5
        self.use_pe = args.use_pose_estimation

        # Visual Encoding
        resnet = models.resnet18(pretrained=args.pretrained_resnet)
        self.resnet_l5 = nn.Sequential(*list(resnet.children())[0:8])
        self.conv = nn.Sequential(
            *filter(
                bool,
                [
                    nn.Conv2d(512, 64, (1, 1), stride=(1, 1)),
                    nn.ReLU(),
                ],
            )
        )

        # convolution output size
        input_test = torch.randn(1, self.n_channels, self.screen_h, self.screen_w)
        conv_output = self.conv(self.resnet_l5(input_test))

        self.pool = ChannelPool(1)
        self.conv_output_size = conv_output.view(-1).size(0)

        # projection layer
        self.proj1 = nn.Linear(self.conv_output_size, 1024)
        self.proj2 = nn.Linear(1024, 4096)

        if self.dropout > 0:
            self.dropout1 = nn.Dropout(self.dropout)
            self.dropout2 = nn.Dropout(self.dropout)

        # Deconv layers to predict map
        self.deconv = nn.Sequential(
            *filter(
                bool,
                [
                    nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
                    nn.ReLU(),
                    nn.ConvTranspose2d(16, 2, (4, 4), stride=(2, 2), padding=(1, 1)),
                ],
            )
        )

        # Pose Estimator
        self.pose_conv = nn.Sequential(
            *filter(
                bool,
                [
                    nn.Conv2d(4, 64, (4, 4), stride=(2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(64, 32, (4, 4), stride=(2, 2)),
                    nn.ReLU(),
                    nn.Conv2d(32, 16, (3, 3), stride=(1, 1)),
                    nn.ReLU(),
                ],
            )
        )

        pose_conv_output = self.pose_conv(torch.randn(1, 4, self.vision_range, self.vision_range))
        self.pose_conv_output_size = pose_conv_output.view(-1).size(0)

        # projection layer
        self.pose_proj1 = nn.Linear(self.pose_conv_output_size, 1024)
        self.pose_proj2_x = nn.Linear(1024, 128)
        self.pose_proj2_y = nn.Linear(1024, 128)
        self.pose_proj2_o = nn.Linear(1024, 128)
        self.pose_proj3_x = nn.Linear(128, 1)
        self.pose_proj3_y = nn.Linear(128, 1)
        self.pose_proj3_o = nn.Linear(128, 1)

        if self.dropout > 0:
            self.pose_dropout1 = nn.Dropout(self.dropout)

        self.st_poses_eval = torch.zeros(args.num_processes, 3).to(self.device)
        self.st_poses_train = torch.zeros(args.slam_batch_size, 3).to(self.device)

        grid_size = self.vision_range * 2
        self.grid_map_eval = (
            torch.zeros(args.num_processes, 2, grid_size, grid_size).float().to(self.device)
        )
        self.grid_map_train = (
            torch.zeros(args.slam_batch_size, 2, grid_size, grid_size).float().to(self.device)
        )

        self.agent_view = (
            torch.zeros(
                args.num_processes,
                2,
                self.map_size_cm // self.resolution,
                self.map_size_cm // self.resolution,
            )
            .float()
            .to(self.device)
        )

    def forward(self, obs_last, obs, poses, maps, explored, current_poses, build_maps=True):
        # Get egocentric map prediction for the current obs
        bs, c, h, w = obs.size()
        resnet_output = self.resnet_l5(obs[:, :3, :, :])
        conv_output = self.conv(resnet_output)

        proj1 = nn.ReLU()(self.proj1(conv_output.view(-1, self.conv_output_size)))
        if self.dropout > 0:
            proj1 = self.dropout1(proj1)
        proj3 = nn.ReLU()(self.proj2(proj1))

        deconv_input = proj3.view(bs, 64, 8, 8)
        deconv_output = self.deconv(deconv_input)
        pred = torch.sigmoid(deconv_output)

        proj_pred = pred[:, :1, :, :]
        fp_exp_pred = pred[:, 1:, :, :]

        with torch.no_grad():
            # Get egocentric map prediction for the last obs
            bs, c, h, w = obs_last.size()
            resnet_output = self.resnet_l5(obs_last[:, :3, :, :])
            conv_output = self.conv(resnet_output)

            proj1 = nn.ReLU()(self.proj1(conv_output.view(-1, self.conv_output_size)))
            if self.dropout > 0:
                proj1 = self.dropout1(proj1)
            proj3 = nn.ReLU()(self.proj2(proj1))

            deconv_input = proj3.view(bs, 64, 8, 8)
            deconv_output = self.deconv(deconv_input)
            pred_last = torch.sigmoid(deconv_output)

            # ST of proj
            vr = self.vision_range
            grid_size = vr * 2

            if build_maps:
                st_poses = self.st_poses_eval.detach_()
                grid_map = self.grid_map_eval.detach_()
            else:
                st_poses = self.st_poses_train.detach_()
                grid_map = self.grid_map_train.detach_()

            st_poses.fill_(0.0)
            st_poses[:, 0] = poses[:, 1] * 200.0 / self.resolution / grid_size
            st_poses[:, 1] = poses[:, 0] * 200.0 / self.resolution / grid_size
            st_poses[:, 2] = poses[:, 2] * 57.29577951308232
            rot_mat, trans_mat = get_grid(st_poses, (bs, 2, grid_size, grid_size), self.device)

            grid_map.fill_(0.0)
            grid_map[:, :, vr:, int(vr / 2) : int(vr / 2 + vr)] = pred_last
            translated = F.grid_sample(grid_map, trans_mat)
            rotated = F.grid_sample(translated, rot_mat)
            rotated = rotated[:, :, vr:, int(vr / 2) : int(vr / 2 + vr)]

            pred_last_st = rotated

        # Pose estimator
        pose_est_input = torch.cat((pred.detach(), pred_last_st.detach()), dim=1)
        pose_conv_output = self.pose_conv(pose_est_input)
        pose_conv_output = pose_conv_output.view(-1, self.pose_conv_output_size)

        proj1 = nn.ReLU()(self.pose_proj1(pose_conv_output))

        if self.dropout > 0:
            proj1 = self.pose_dropout1(proj1)

        proj2_x = nn.ReLU()(self.pose_proj2_x(proj1))
        pred_dx = self.pose_proj3_x(proj2_x)

        proj2_y = nn.ReLU()(self.pose_proj2_y(proj1))
        pred_dy = self.pose_proj3_y(proj2_y)

        proj2_o = nn.ReLU()(self.pose_proj2_o(proj1))
        pred_do = self.pose_proj3_o(proj2_o)

        pose_pred = torch.cat((pred_dx, pred_dy, pred_do), dim=1)
        if self.use_pe == 0:
            pose_pred = pose_pred * self.use_pe

        if build_maps:
            # Aggregate egocentric map prediction in the geocentric map
            # using the predicted pose
            with torch.no_grad():
                agent_view = self.agent_view.detach_()
                agent_view.fill_(0.0)

                x1 = self.map_size_cm // (self.resolution * 2) - self.vision_range // 2
                x2 = x1 + self.vision_range
                y1 = self.map_size_cm // (self.resolution * 2)
                y2 = y1 + self.vision_range
                agent_view[:, :, y1:y2, x1:x2] = pred

                corrected_pose = poses + pose_pred

                def get_new_pose_batch(pose, rel_pose_change):
                    pose[:, 1] += rel_pose_change[:, 0] * torch.sin(
                        pose[:, 2] / 57.29577951308232
                    ) + rel_pose_change[:, 1] * torch.cos(pose[:, 2] / 57.29577951308232)
                    pose[:, 0] += rel_pose_change[:, 0] * torch.cos(
                        pose[:, 2] / 57.29577951308232
                    ) - rel_pose_change[:, 1] * torch.sin(pose[:, 2] / 57.29577951308232)
                    pose[:, 2] += rel_pose_change[:, 2] * 57.29577951308232

                    pose[:, 2] = torch.fmod(pose[:, 2] - 180.0, 360.0) + 180.0
                    pose[:, 2] = torch.fmod(pose[:, 2] + 180.0, 360.0) - 180.0

                    return pose

                current_poses = get_new_pose_batch(current_poses, corrected_pose)
                st_pose = current_poses.clone().detach()

                st_pose[:, :2] = -(
                    st_pose[:, :2] * 100.0 / self.resolution
                    - self.map_size_cm // (self.resolution * 2)
                ) / (self.map_size_cm // (self.resolution * 2))
                st_pose[:, 2] = 90.0 - (st_pose[:, 2])

                rot_mat, trans_mat = get_grid(st_pose, agent_view.size(), self.device)

                rotated = F.grid_sample(agent_view, rot_mat)
                translated = F.grid_sample(rotated, trans_mat)

                maps2 = torch.cat((maps.unsqueeze(1), translated[:, :1, :, :]), 1)
                explored2 = torch.cat((explored.unsqueeze(1), translated[:, 1:, :, :]), 1)

                map_pred = self.pool(maps2).squeeze(1)
                exp_pred = self.pool(explored2).squeeze(1)

        else:
            map_pred = None
            exp_pred = None
            current_poses = None

        return (
            proj_pred,
            fp_exp_pred,
            map_pred,
            exp_pred,
            pose_pred,
            current_poses,
        )


class _NeuralSLAMTraceWrapper(nn.Module):
    """Tuple-in wrapper so TorchLens sees a single positional call."""

    def __init__(self, module: Neural_SLAM_Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, inputs):
        obs_last, obs, poses, maps, explored, current_poses = inputs
        return self.module(obs_last, obs, poses, maps, explored, current_poses, build_maps=True)


# Real defaults from arguments.py (`-fh`/`-fw`/`--map_resolution`/`--global_downscaling`
# /`--map_size_cm`/`--vision_range`/`-pe`/`-pt`), unchanged. NOTE: the decoder path
# fixes `deconv_input.view(bs, 64, 8, 8)` then applies 3 stride-2 ConvTranspose2d
# layers regardless of frame_height/width, so `pred`/`pred_last` are always spatially
# 8*2*2*2 = 64 on each side; `vision_range` must be 64 to match
# (`grid_map[:, :, vr:, vr/2:vr/2+vr]` must fit `pred_last`), which is exactly the
# real repo's default `-vr/--vision_range` value.
_FRAME_H = 128
_FRAME_W = 128
_MAP_RESOLUTION = 5
_GLOBAL_DOWNSCALING = 2
_MAP_SIZE_CM = 2400
_VISION_RANGE = 64
_USE_POSE_ESTIMATION = 2
_PRETRAINED_RESNET = False
_NUM_PROCESSES = 1
_SLAM_BATCH_SIZE = 1


def _build_args() -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        frame_height=_FRAME_H,
        frame_width=_FRAME_W,
        map_resolution=_MAP_RESOLUTION,
        map_size_cm=_MAP_SIZE_CM,
        global_downscaling=_GLOBAL_DOWNSCALING,
        vision_range=_VISION_RANGE,
        use_pose_estimation=_USE_POSE_ESTIMATION,
        pretrained_resnet=_PRETRAINED_RESNET,
        num_processes=_NUM_PROCESSES,
        slam_batch_size=_SLAM_BATCH_SIZE,
    )


def build_neural_slam() -> nn.Module:
    module = Neural_SLAM_Module(_build_args())
    module.eval()
    return _NeuralSLAMTraceWrapper(module)


def example_input_neural_slam() -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    bs = 1
    obs_last = torch.rand(bs, 3, _FRAME_H, _FRAME_W)
    obs = torch.rand(bs, 3, _FRAME_H, _FRAME_W)
    poses = torch.zeros(bs, 3)
    map_grid = _MAP_SIZE_CM // _GLOBAL_DOWNSCALING // _MAP_RESOLUTION
    maps = torch.zeros(bs, map_grid, map_grid)
    explored = torch.zeros(bs, map_grid, map_grid)
    current_poses = torch.zeros(bs, 3)
    return (obs_last, obs, poses, maps, explored, current_poses)


MENAGERIE_ZOO = "vendored-pytorch"

MENAGERIE_ENTRIES = [
    (
        "Neural SLAM",
        build_neural_slam,
        example_input_neural_slam,
        2020,
        MENAGERIE_ZOO,
    ),
]
