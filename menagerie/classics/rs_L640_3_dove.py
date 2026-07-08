# FAITHFUL PORT of kiharalab/DOVE @ master (original framework: Keras/TensorFlow)
#
# DOVE (Wang & Kihara, Bioinformatics 2020) is "A Deep-learning based dOcking decoy
# eValuation mEthod": a 3D CNN that scores protein-protein docking decoys from
# voxelized atom-interaction-based feature grids (GOAP + ITScore potential channels
# rasterized onto a 20x20x20 grid around the interface). The real repo
# (Prediction/Build_Model.py `makecnn`) is Keras/TF1.x (`from keras.models import
# Sequential`), which is not in the base torch env and is not reasonably installable
# alongside torch/torchvision/timm here, so the architecture is faithfully transcribed
# layer-for-layer into torch: 3x[Conv3D(valid) -> BatchNorm -> LeakyReLU(0.2)] with a
# MaxPool3D after the 2nd and 3rd conv blocks, Flatten, then
# 2x[Dense -> BatchNorm -> LeakyReLU(0.2) -> Dropout(0.3)] and a final
# Dense(1) -> Sigmoid classifier head. Filter counts (100/200/400), kernel size
# (3x3x3, `padding='valid'`, i.e. no torch padding), pool size (2x2x2), dense widths
# (1000/100), LeakyReLU slope (0.2), and dropout rate (0.3) are copied from
# `makecnn(learningrate, regular, decay, channel_number)`; the L1/L2 activity
# regularizers (`activity_regularizer=l2(regular)`) and the Nadam optimizer/loss/metric
# wiring are training-time-only concerns with no forward-pass effect and are dropped.
import torch
import torch.nn as nn

MENAGERIE_ZOO = "ported-pytorch"


class DOVE(nn.Module):
    def __init__(self, channel_number, grid_size=20):
        super().__init__()
        self.channel_number = channel_number

        # Conv3D(100, k=3, valid, channels_last input=(20,20,20,channel_number))
        self.conv1 = nn.Conv3d(channel_number, 100, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(100)
        self.act1 = nn.LeakyReLU(0.2)

        # Conv3D(200, k=3, valid)
        self.conv2 = nn.Conv3d(100, 200, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm3d(200)
        self.act2 = nn.LeakyReLU(0.2)

        # MaxPooling3D(2,2,2) then BatchNorm3d(axis=1 in Keras channels_last == the
        # channel axis, matching torch's default BatchNorm3d(200) channel-axis behavior)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        self.bn_pool1 = nn.BatchNorm3d(200)

        # Conv3D(400, k=3, valid)
        self.conv3 = nn.Conv3d(200, 400, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm3d(400)
        self.act3 = nn.LeakyReLU(0.2)

        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.dropout0 = nn.Dropout(p=0.3)

        # Dense(1000, input_shape=(32000,)) -- 32000 = 400 * 4*4*4 for the real
        # grid_size=20 input; computed dynamically below from grid_size so the module
        # stays correct for a shrunk tiny trace too.
        flat_dim = 400 * self._spatial_after_convs(grid_size) ** 3
        self.fc1 = nn.Linear(flat_dim, 1000)
        self.bn_fc1 = nn.BatchNorm1d(1000)
        self.act_fc1 = nn.LeakyReLU(0.2)
        self.dropout1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(1000, 100)
        self.bn_fc2 = nn.BatchNorm1d(100)
        self.act_fc2 = nn.LeakyReLU(0.2)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    @staticmethod
    def _spatial_after_convs(grid_size):
        # 3x (kernel=3, valid) shrinks by 2 each time; 2x (pool, k=2) halves.
        s = grid_size - 2  # conv1
        s = s - 2  # conv2
        s = s // 2  # pool1
        s = s - 2  # conv3
        s = s // 2  # pool2
        return s

    def forward(self, x):
        # x: [N, channel_number, D, H, W] (torch channels-first; the real Keras model
        # uses channels_last (N,D,H,W,C) over the same (20,20,20,channel_number) grid).
        x = self.act1(self.bn1(self.conv1(x)))

        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.bn_pool1(x)

        x = self.act3(self.bn3(self.conv3(x)))
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.dropout0(x)

        x = self.act_fc1(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)

        x = self.act_fc2(self.bn_fc2(self.fc2(x)))
        x = self.dropout2(x)

        x = self.out_act(self.fc3(x))
        return x


def build_dove():
    # Real caller (Prediction/Build_Model.py makecnn) uses channel_number set by the
    # number of stacked potential-feature channels (GOAP/ITScore atom-pair grids, see
    # ops/Extract_Indicate.py Indicate_to_channel) and the real 20x20x20 voxel grid.
    # channel_number kept small (a real, if minimal, feature-channel count); grid_size
    # kept at the real 20 since the spatial-downsample arithmetic needs >= ~14 to reach
    # a positive 4x4x4 bottleneck (matches the real repo's fixed 20x20x20 grid exactly).
    return DOVE(channel_number=2, grid_size=20)


def example_input_dove():
    torch.manual_seed(0)
    # [batch, channel_number, 20, 20, 20] voxelized docking-interface potential grid.
    return (torch.randn(2, 2, 20, 20, 20),)


MENAGERIE_ENTRIES = [
    ("DOVE", "build_dove", "example_input_dove", 2020, MENAGERIE_ZOO),
]
