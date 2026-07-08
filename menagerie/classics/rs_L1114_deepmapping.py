# SOURCE: vendored from ai4ce/DeepMapping @ master
# https://raw.githubusercontent.com/ai4ce/DeepMapping/master/models/deepmapping.py
# https://raw.githubusercontent.com/ai4ce/DeepMapping/master/models/networks.py
# https://raw.githubusercontent.com/ai4ce/DeepMapping/master/utils/geometry_utils.py
#
# DeepMapping (Ding & Feng, CVPR 2019 Oral) -- unsupervised point-cloud registration and occupancy
# mapping. The 2D variant (`DeepMapping2D`) is a self-supervised "L-Net + M-Net" architecture: an
# L-Net (`LocNetReg2D`) is a 1D-CNN point-cloud feature extractor (`ObsFeat2D`, dilated Conv1d
# stack + global max-pool) feeding a 3-layer MLP that regresses a per-frame SE(2) pose estimate
# (x, y, theta); the estimated pose is used to transform the local point cloud into the global
# frame (`transform_to_global_2D`, a batched rotation+translation via `torch.bmm`); an M-Net
# (`occup_net`, a plain pointwise MLP `MLP`/`PointwiseMLP`) then scores occupancy probability for
# both the registered points and points sampled as unoccupied along the sensor ray
# (`sample_unoccupied_point`). All classes/functions below are transcribed verbatim from the
# upstream `models/deepmapping.py`, `models/networks.py`, and the one needed helper from
# `utils/geometry_utils.py` (`transform_to_global_2D`); only the training-loss branch (BCE/
# BCE+Chamfer loss selection, which needs the repo's separate `loss/` module and only runs when a
# `loss_fn` callable is supplied) is omitted from `forward()` since it is orthogonal to the network
# architecture -- this staging module's `forward()` stops after producing `occp_prob`, matching the
# upstream computation up to (but not including) the loss reduction.

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- utils/geometry_utils.py :: transform_to_global_2D (verbatim) ----
def transform_to_global_2D(pose, obs_local):
    """
    transform local point cloud to global frame
    row-based matrix product
    pose: <Bx3> each row represents <x,y,theta>
    obs_local: <BxLx2>
    """
    L = obs_local.shape[1]
    # c0 is the loc of sensor in global coord. frame c0: <Bx2>
    c0, theta0 = pose[:, 0:2], pose[:, 2]
    c0 = c0.unsqueeze(1).expand(-1, L, -1)  # <BxLx2>

    cos = torch.cos(theta0).unsqueeze(-1).unsqueeze(-1)
    sin = torch.sin(theta0).unsqueeze(-1).unsqueeze(-1)
    R_transpose = torch.cat((cos, sin, -sin, cos), dim=1).reshape(-1, 2, 2)

    obs_global = torch.bmm(obs_local, R_transpose) + c0
    return obs_global


# ---- models/networks.py (verbatim) ----
def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(li.weight.data, gain=nn.init.calculate_gain("relu"))
    li.bias.data.fill_(0.0)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super().__init__(*layers)


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, x):
        return self.mlp.forward(x)


class ObsFeat2D(nn.Module):
    """Feature extractor for 1D organized point clouds"""

    def __init__(self, n_points, n_out=1024):
        super().__init__()
        self.n_out = n_out
        k = 3
        p = int(np.floor(k / 2)) + 2
        self.conv1 = nn.Conv1d(2, 64, kernel_size=k, padding=p, dilation=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=k, padding=p, dilation=3)
        self.conv3 = nn.Conv1d(128, self.n_out, kernel_size=k, padding=p, dilation=3)
        self.mp = nn.MaxPool1d(n_points)

    def forward(self, x):
        assert x.shape[1] == 2, "the input size must be <Bx2xL> "

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.mp(x)
        x = x.view(-1, self.n_out)  # <Bx1024>
        return x


class LocNetReg2D(nn.Module):
    def __init__(self, n_points, out_dims):
        super().__init__()
        self.obs_feat_extractor = ObsFeat2D(n_points)
        n_in = self.obs_feat_extractor.n_out
        self.fc = MLP([n_in, 512, 256, out_dims])

    def forward(self, obs):
        obs = obs.transpose(1, 2)
        obs_feat = self.obs_feat_extractor(obs)
        obs = obs.transpose(1, 2)

        x = self.fc(obs_feat)
        return x


# ---- models/deepmapping.py (verbatim, minus the loss_fn-dependent branch) ----
def get_M_net_inputs_labels(occupied_points, unoccupited_points):
    """
    get global coord (occupied and unoccupied) and corresponding labels
    """
    n_pos = occupied_points.shape[1]
    inputs = torch.cat((occupied_points, unoccupited_points), 1)
    bs, N, _ = inputs.shape

    gt = torch.zeros([bs, N, 1], device=occupied_points.device)
    gt.requires_grad_(False)
    gt[:, :n_pos, :] = 1
    return inputs, gt


def sample_unoccupied_point(local_point_cloud, n_samples, center):
    """
    sample unoccupied points along rays in local point cloud
    local_point_cloud: <BxLxk>
    n_samples: number of samples on each ray
    center: location of sensor <Bx1xk>
    """
    bs, L, k = local_point_cloud.shape
    center = center.expand(-1, L, -1)  # <BxLxk>
    unoccupied = torch.zeros(bs, L * n_samples, k, device=local_point_cloud.device)
    for idx in range(1, n_samples + 1):
        fac = torch.rand(1).item()
        unoccupied[:, (idx - 1) * L : idx * L, :] = center + (local_point_cloud - center) * fac
    return unoccupied


class DeepMapping2D(nn.Module):
    def __init__(self, n_obs=256, n_samples=19, dim=(2, 64, 512, 512, 256, 128, 1)):
        super().__init__()
        self.n_obs = n_obs
        self.n_samples = n_samples
        self.loc_net = LocNetReg2D(n_points=n_obs, out_dims=3)
        self.occup_net = MLP(list(dim))

    def forward(self, obs_local, valid_points, sensor_pose):
        # obs_local: <BxLx2>
        # sensor_pose: init pose <Bx1x3>
        self.obs_local = obs_local
        self.valid_points = valid_points

        self.pose_est = self.loc_net(self.obs_local)

        self.obs_global_est = transform_to_global_2D(self.pose_est, self.obs_local)

        sensor_center = sensor_pose[:, :, :2]
        self.unoccupied_local = sample_unoccupied_point(
            self.obs_local, self.n_samples, sensor_center
        )
        self.unoccupied_global = transform_to_global_2D(self.pose_est, self.unoccupied_local)

        inputs, self.gt = get_M_net_inputs_labels(self.obs_global_est, self.unoccupied_global)
        self.occp_prob = self.occup_net(inputs)
        return self.occp_prob


# ---- staging build / example_input ----
def build_deepmapping2d() -> nn.Module:
    return DeepMapping2D(n_obs=8, n_samples=3, dim=(2, 16, 16, 8, 1))


def example_input_deepmapping2d():
    bs = 2
    n_obs = 8
    obs_local = torch.randn(bs, n_obs, 2)
    valid_points = torch.ones(bs, n_obs)
    sensor_pose = torch.zeros(bs, 1, 3)
    return obs_local, valid_points, sensor_pose


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("DeepMapping2D", "build_deepmapping2d", "example_input_deepmapping2d", 2019, MENAGERIE_ZOO),
]
