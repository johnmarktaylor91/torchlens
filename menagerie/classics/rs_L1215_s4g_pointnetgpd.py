# SOURCE: vendored from yzqin/s4g-release @ master
# https://github.com/yzqin/s4g-release/blob/master/inference/grasp_proposal/network_models/models/PointNetGPD.py
# S4G (arXiv:1910.14218, CoRL 2019) ships this PointNetGPD network as one of its own
# registered "baseline" architectures in network_models/models/build_model.py
# (`build_pointnetgpd`, dispatched for cfg.MODEL.TYPE == "PointNetGPD"), alongside the
# paper's own EdgePointNet2Down(Up) SA/FP networks. The S4G-native EdgePointNet2* models
# require a custom-compiled CUDA extension (pointnet2_utils/setup.py -> `pn2_ext`, not
# installed here), so this build uses the PointNetGPD baseline instead: it is real S4G
# repo code, pure PyTorch/torch.nn only, no custom ops. Architecture (STN3d input-
# transform T-Net + shared 1D-conv PointNet feature extractor + 3-layer MLP grasp-score
# classification head) is transcribed VERBATIM from PointNetGPD.py; only change is
# dropping the `if __name__ == "__main__":` demo block.
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class STN3d(nn.Module):
    def __init__(self, input_chann=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, 2)
        x = x.view(batchsize, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(-1, 3, 3)

        I = torch.eye(3, dtype=x.dtype, device=x.device)  # noqa: E741 (verbatim source var name)
        x = x.add(I)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, input_chann=3, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(input_chann=input_chann)
        self.conv1 = torch.nn.Conv1d(input_chann, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(batchsize, 1024)
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1), trans


class PointNetClassifier(nn.Module):
    def __init__(self, input_chann, score_classes):
        super(PointNetClassifier, self).__init__()
        self.out_channels = score_classes
        self.feat = PointNetfeat(input_chann=input_chann, global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, score_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, data_batch):
        close_region_points = data_batch["close_region_points"]
        if len(close_region_points.shape) == 3:
            batch_size, _, num_points = close_region_points.shape
        else:
            assert len(close_region_points.shape) == 4
            batch_size, num_grasp, _, num_points = close_region_points.shape
            close_region_points = close_region_points.view(batch_size * num_grasp, -1, num_points)

        x, trans = self.feat(close_region_points)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return {
            "grasp_logits": x,
        }


def build_s4g_pointnetgpd():
    # Real build_pointnetgpd(cfg) wiring: input_chann=3 (xyz), score_classes shrunk
    # from the released S4G config's typical class count to 4 for menagerie scale
    # (matches the __main__ smoke-test call `PointNetClassifier(3, 4)` in the source).
    return PointNetClassifier(input_chann=3, score_classes=4)


def example_input_s4g_pointnetgpd():
    # Mirrors the source file's own __main__ smoke test:
    # preds = pointnet({"close_region_points": torch.rand(2, 10, 3, 128)})
    # shrunk num_grasp/num_points for menagerie scale.
    torch.manual_seed(0)
    return ({"close_region_points": torch.rand(2, 4, 3, 32)},)


MENAGERIE_ENTRIES = [
    (
        "S4G-PointNetGPD",
        "build_s4g_pointnetgpd",
        "example_input_s4g_pointnetgpd",
        2019,
        "vendored-pytorch",
    ),
]
