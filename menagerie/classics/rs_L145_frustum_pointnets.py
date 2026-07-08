# SOURCE: vendored from simon3dv/frustum_pointnets_pytorch @ dff1ed937be649172c22f1c705c0830a5833608a
# https://raw.githubusercontent.com/simon3dv/frustum_pointnets_pytorch/dff1ed937be649172c22f1c705c0830a5833608a/models/frustum_pointnets_v1.py
# https://raw.githubusercontent.com/simon3dv/frustum_pointnets_pytorch/dff1ed937be649172c22f1c705c0830a5833608a/models/model_util.py
#
# Qi, Liu, Wu, Su, Guibas 2018 (CVPR) "Frustum PointNets for 3D Object Detection from
# RGB-D Data" -- the seminal frustum-based 3D detection pipeline: a 2D detector proposes
# a frustum, then three cascaded PointNet-style sub-networks operate on the raw point
# cloud inside that frustum: (1) `PointNetInstanceSeg` does binary point-wise instance
# segmentation (foreground/background) via a PointNet encoder-decoder over the raw
# points, (2) a lightweight T-Net (`STNxyz`, a spatial-transformer-style regression
# PointNet) predicts a coarse centroid shift from the masked foreground points, (3)
# `PointNetEstimation` (Amodal 3D Box Estimation PointNet) regresses the final box
# center-residual, heading-bin classification+residual, and size-cluster
# classification+residual from the re-centered points. `FrustumPointNetv1` wires these
# three sub-networks together exactly as in the official repo's forward pass.
#
# `PointNetInstanceSeg`, `STNxyz`, `PointNetEstimation` are copied verbatim from the
# real `models/frustum_pointnets_v1.py`. `point_cloud_masking`, `gather_object_pts`,
# and `parse_output_to_tensors` are copied verbatim from the real `models/model_util.py`
# (the exact helper functions `frustum_pointnets_v1.py` imports and calls). The
# NUM_HEADING_BIN/NUM_OBJECT_POINT constants are identical across both source files;
# NUM_SIZE_CLUSTER=3 is taken from `model_util.py` (the file `frustum_pointnets_v1.py`
# actually imports NUM_SIZE_CLUSTER from), which is the live train-time KITTI
# Car/Pedestrian/Cyclist 3-cluster config -- the `NUM_SIZE_CLUSTER=8` in
# `frustum_pointnets_v1.py`'s own module scope is dead/shadowed (never read; the file
# imports the name from model_util instead of using its own local redefinition).
#
# `FrustumPointNetv1.forward` in the real repo is a *training/eval* forward: it takes a
# dict of ground-truth labels, runs the three-stage network, then immediately computes
# `FrustumPointNetLoss` (needs GT labels) and `compute_box3d_iou` (calls
# `.cpu().numpy()` and a KITTI-specific numpy IoU routine from `provider.py`, a
# non-torch, non-traceable data-loading module). `FrustumPointNetInference` below
# reproduces exactly the real network path shared by both train and inference in the
# original file -- `InsSeg` -> `point_cloud_masking` -> `STN` -> center-recentering ->
# `est` -> `parse_output_to_tensors` -> `box3d_center = center_boxnet + stage1_center`,
# i.e. every real layer/tensor op up to raw box parameters -- and stops there instead of
# also computing the training loss (needs unavailable GT box labels) and the numpy IoU
# metrics (not a traceable torch computation), neither of which is part of the network
# architecture. All `.cuda()` calls in the copied helpers were changed to run on
# whatever device the input tensors are already on (`.to(pts.device)` /
# `.to(logits.device)`) so the module runs on CPU; this is a mechanical device-portability
# fix only, not an architectural change (the upstream repo hardcodes CUDA throughout).

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# -----------------
# Global Constants (from model_util.py, the file frustum_pointnets_v1.py imports from)
# -----------------
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3  # Car/Pedestrian/Cyclist
NUM_OBJECT_POINT = 512

g_type2class = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type_mean_size = {
    "Car": np.array([3.88311640418, 1.62856739989, 1.52563191462]),
    "Pedestrian": np.array([0.84422524, 0.66068622, 1.76255119]),
    "Cyclist": np.array([1.76282397, 0.59706367, 1.73698127]),
}
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


class PointNetInstanceSeg(nn.Module):
    def __init__(self, n_classes=3, n_channel=3):
        """v1 3D Instance Segmentation PointNet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        """
        super(PointNetInstanceSeg, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.n_classes = n_classes
        self.dconv1 = nn.Conv1d(1088 + n_classes, 512, 1)
        self.dconv2 = nn.Conv1d(512, 256, 1)
        self.dconv3 = nn.Conv1d(256, 128, 1)
        self.dconv4 = nn.Conv1d(128, 128, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dconv5 = nn.Conv1d(128, 2, 1)
        self.dbn1 = nn.BatchNorm1d(512)
        self.dbn2 = nn.BatchNorm1d(256)
        self.dbn3 = nn.BatchNorm1d(128)
        self.dbn4 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec):  # bs,4,n
        """
        :param pts: [bs,4,n]: x,y,z,intensity
        :return: logits: [bs,n,2],scores for bkg/clutter and object
        """
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = F.relu(self.bn1(self.conv1(pts)))  # bs,64,n
        out2 = F.relu(self.bn2(self.conv2(out1)))  # bs,64,n
        out3 = F.relu(self.bn3(self.conv3(out2)))  # bs,64,n
        out4 = F.relu(self.bn4(self.conv4(out3)))  # bs,128,n
        out5 = F.relu(self.bn5(self.conv5(out4)))  # bs,1024,n
        global_feat = torch.max(out5, 2, keepdim=True)[0]  # bs,1024,1

        expand_one_hot_vec = one_hot_vec.view(bs, -1, 1)  # bs,3,1
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec], 1)  # bs,1027,1
        expand_global_feat_repeat = expand_global_feat.view(bs, -1, 1).repeat(
            1, 1, n_pts
        )  # bs,1027,n
        concat_feat = torch.cat([out2, expand_global_feat_repeat], 1)
        # bs, (641024+3)=1091, n

        x = F.relu(self.dbn1(self.dconv1(concat_feat)))  # bs,512,n
        x = F.relu(self.dbn2(self.dconv2(x)))  # bs,256,n
        x = F.relu(self.dbn3(self.dconv3(x)))  # bs,128,n
        x = F.relu(self.dbn4(self.dconv4(x)))  # bs,128,n
        x = self.dropout(x)
        x = self.dconv5(x)  # bs, 2, n

        seg_pred = x.transpose(2, 1).contiguous()  # bs, n, 2
        return seg_pred


class PointNetEstimation(nn.Module):
    def __init__(self, n_classes=2):
        """v1 Amodal 3D Box Estimation Pointnet
        :param n_classes:3
        :param one_hot_vec:[bs,n_classes]
        """
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        self.n_classes = n_classes

        self.fc1 = nn.Linear(512 + 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4)
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)

    def forward(self, pts, one_hot_vec):  # bs,3,m
        """
        :param pts: [bs,3,m]: x,y,z after InstanceSeg
        :return: box_pred: [bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4]
            including box centers, heading bin class scores and residual,
            and size cluster scores and residual
        """
        bs = pts.size()[0]

        out1 = F.relu(self.bn1(self.conv1(pts)))  # bs,128,n
        out2 = F.relu(self.bn2(self.conv2(out1)))  # bs,128,n
        out3 = F.relu(self.bn3(self.conv3(out2)))  # bs,256,n
        out4 = F.relu(self.bn4(self.conv4(out3)))  # bs,512,n
        global_feat = torch.max(out4, 2, keepdim=False)[0]  # bs,512

        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        expand_global_feat = torch.cat([global_feat, expand_one_hot_vec], 1)  # bs,515

        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))  # bs,512
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,256
        box_pred = self.fc3(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred


class STNxyz(nn.Module):
    def __init__(self, n_classes=3):
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256 + n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)

    def forward(self, pts, one_hot_vec):
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))  # bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))  # bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))  # bs,256,n
        x = torch.max(x, 2)[0]  # bs,256
        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        x = torch.cat([x, expand_one_hot_vec], 1)  # bs,259
        x = F.relu(self.fcbn1(self.fc1(x)))  # bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,128
        x = self.fc3(x)  # bs,3
        return x


def point_cloud_masking(pts, logits, xyz_only=True):
    """
    :param pts: bs,c,n in frustum
    :param logits: bs,n,2
    :param xyz_only: bool
    :return:
    """
    bs = pts.shape[0]
    n_pts = pts.shape[2]
    # Binary Classification for each point
    mask = logits[:, :, 0] < logits[:, :, 1]  # (bs, n)
    mask = mask.unsqueeze(1).float()  # (bs, 1, n)
    mask_count = mask.sum(2, keepdim=True).repeat(1, 3, 1)  # (bs, 3, 1)
    pts_xyz = pts[:, :3, :]  # (bs,3,n)
    mask_xyz_mean = (mask.repeat(1, 3, 1) * pts_xyz).sum(2, keepdim=True)  # (bs, 3, 1)
    mask_xyz_mean = mask_xyz_mean / torch.clamp(mask_count, min=1)  # (bs, 3, 1)
    mask = mask.squeeze()  # (bs,n)
    pts_xyz_stage1 = pts_xyz - mask_xyz_mean.repeat(1, 1, n_pts)

    if xyz_only:
        pts_stage1 = pts_xyz_stage1
    else:
        pts_features = pts[:, 3:, :]
        pts_stage1 = torch.cat([pts_xyz_stage1, pts_features], dim=-1)
    object_pts, _ = gather_object_pts(pts_stage1, mask, NUM_OBJECT_POINT)
    object_pts = object_pts.reshape(bs, NUM_OBJECT_POINT, -1)
    object_pts = object_pts.float().view(bs, 3, -1)
    return object_pts, mask_xyz_mean.squeeze(), mask


def gather_object_pts(pts, mask, n_pts=NUM_OBJECT_POINT):
    """
    :param pts: (bs,c,1024)
    :param mask: (bs,1024)
    :param n_pts: max number of points of an object
    :return:
        object_pts:(bs,c,n_pts)
        indices:(bs,n_pts)
    """
    bs = pts.shape[0]
    indices = torch.zeros((bs, n_pts), dtype=torch.int64)  # (bs, 512)
    object_pts = torch.zeros((bs, pts.shape[1], n_pts))

    for i in range(bs):
        pos_indices = torch.where(mask[i, :] > 0.5)[0]
        if len(pos_indices) > 0:
            if len(pos_indices) > n_pts:
                choice = np.random.choice(len(pos_indices), n_pts, replace=False)
            else:
                choice = np.random.choice(len(pos_indices), n_pts - len(pos_indices), replace=True)
                choice = np.concatenate((np.arange(len(pos_indices)), choice))
            np.random.shuffle(choice)
            indices[i, :] = pos_indices[choice]
            object_pts[i, :, :] = pts[i, :, indices[i, :]]
    return object_pts, indices


def parse_output_to_tensors(box_pred, logits, mask, stage1_center):
    """
    :param box_pred: (bs,59)
    :param logits: (bs,1024,2)
    :param mask: (bs,1024)
    :param stage1_center: (bs,3)
    :return:
        center_boxnet:(bs,3)
        heading_scores:(bs,12)
        heading_residual_normalized:(bs,12),-1 to 1
        heading_residual:(bs,12)
        size_scores:(bs,8)
        size_residual_normalized:(bs,8)
        size_residual:(bs,8)
    """
    bs = box_pred.shape[0]
    # center
    center_boxnet = box_pred[:, :3]  # 0:3
    c = 3

    # heading
    heading_scores = box_pred[:, c : c + NUM_HEADING_BIN]
    c += NUM_HEADING_BIN
    heading_residual_normalized = box_pred[:, c : c + NUM_HEADING_BIN]
    heading_residual = heading_residual_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # size
    size_scores = box_pred[:, c : c + NUM_SIZE_CLUSTER]
    c += NUM_SIZE_CLUSTER
    size_residual_normalized = box_pred[:, c : c + 3 * NUM_SIZE_CLUSTER].contiguous()
    size_residual_normalized = size_residual_normalized.view(bs, NUM_SIZE_CLUSTER, 3)
    size_residual = size_residual_normalized * torch.from_numpy(g_mean_size_arr).unsqueeze(
        0
    ).repeat(bs, 1, 1).to(box_pred.device)
    return (
        center_boxnet,
        heading_scores,
        heading_residual_normalized,
        heading_residual,
        size_scores,
        size_residual_normalized,
        size_residual,
    )


class FrustumPointNetInference(nn.Module):
    """Inference-only FrustumPointNetv1: the real three-stage network path from
    `FrustumPointNetv1.forward`, stopping at raw box parameters (no GT-label-dependent
    loss, no numpy IoU metrics -- those are training/eval bookkeeping, not architecture).
    """

    def __init__(self, n_classes=3, n_channel=3):
        super(FrustumPointNetInference, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.InsSeg = PointNetInstanceSeg(n_classes=3, n_channel=n_channel)
        self.STN = STNxyz(n_classes=3)
        self.est = PointNetEstimation(n_classes=3)

    def forward(self, point_cloud, one_hot):
        # point_cloud: (bs, n_channel, n) x,y,z[,intensity]; one_hot: (bs, 3)
        point_cloud = point_cloud[:, : self.n_channel, :]

        # 3D Instance Segmentation PointNet
        logits = self.InsSeg(point_cloud, one_hot)  # bs,n,2

        # Mask Point Centroid
        object_pts_xyz, mask_xyz_mean, mask = point_cloud_masking(point_cloud, logits)

        # T-Net
        center_delta = self.STN(object_pts_xyz, one_hot)  # (bs,3)
        stage1_center = center_delta + mask_xyz_mean  # (bs,3)

        object_pts_xyz_new = object_pts_xyz - center_delta.view(
            center_delta.shape[0], -1, 1
        ).repeat(1, 1, object_pts_xyz.shape[-1])

        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new, one_hot)  # (bs, 3+2*NH+4*NS)

        (
            center_boxnet,
            heading_scores,
            heading_residual_normalized,
            heading_residual,
            size_scores,
            size_residual_normalized,
            size_residual,
        ) = parse_output_to_tensors(box_pred, logits, mask, stage1_center)

        box3d_center = center_boxnet + stage1_center  # bs,3

        return logits, box3d_center, heading_scores, heading_residual, size_scores, size_residual


def build_frustum_pointnets():
    return FrustumPointNetInference(n_classes=3, n_channel=4)


def example_input_frustum_pointnets():
    point_cloud = torch.randn(2, 4, 256)
    one_hot = F.one_hot(torch.randint(0, 3, (2,)), num_classes=3).float()
    return (point_cloud, one_hot)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    (
        "Frustum PointNets",
        "build_frustum_pointnets",
        "example_input_frustum_pointnets",
        2018,
        "vendored",
    ),
]
