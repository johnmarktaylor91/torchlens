# SOURCE: vendored from CathIAS/TLIO @ master
# https://github.com/CathIAS/TLIO/blob/master/src/network/model_resnet.py
# https://github.com/CathIAS/TLIO/blob/master/src/network/model_factory.py
# "TLIO: Tight Learned Inertial Odometry" (Liu et al., IEEE RA-L 2020).
# `conv3x1`/`conv1x1`, `BasicBlock1D`, `FcBlock`, and `ResNet1D` (the
# "resnet" arch selected by `get_model()` in `model_factory.py`, the
# network's default/primary architecture) are transcribed VERBATIM from the
# real repo's `model_resnet.py`. The network consumes a windowed 6-channel
# IMU signal (3-axis gyro + 3-axis accel, shape (B, 6, window_size)) and
# regresses a 3D displacement mean and log-std (fed downstream into the
# EKF measurement update; only the network itself is captured here). The
# unused `Bottleneck` class (for a Bottleneck1D variant not exercised by
# `get_model()`) is dropped for brevity.
import torch.nn as nn


def conv3x1(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """1D convolution with kernel size 3"""
    return nn.Conv1d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1D convolution with kernel size 1"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock1D(nn.Module):
    """Supports: groups=1, dilation=1"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes * self.expansion)
        self.bn2 = nn.BatchNorm1d(planes * self.expansion)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FcBlock(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim):
        super(FcBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.prep_channel = 128
        self.fc_dim = 512
        self.in_dim = in_dim

        # prep layer2
        self.prep1 = nn.Conv1d(self.in_channel, self.prep_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.prep_channel)
        # fc layers
        self.fc1 = nn.Linear(self.prep_channel * self.in_dim, self.fc_dim)
        self.fc2 = nn.Linear(self.fc_dim, self.fc_dim)
        self.fc3 = nn.Linear(self.fc_dim, self.out_channel)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.prep1(x)
        x = self.bn1(x)
        x = self.fc1(x.view(x.size(0), -1))
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ResNet1D(nn.Module):
    """
    ResNet 1D
    in_dim: input channel (for IMU data, in_dim=6)
    out_dim: output dimension (3)
    len(group_sizes) = 4
    """

    def __init__(
        self,
        block_type,
        in_dim,
        out_dim,
        group_sizes,
        inter_dim,
        zero_init_residual=False,
    ):
        super(ResNet1D, self).__init__()
        self.base_plane = 64
        self.inplanes = self.base_plane

        # Input module
        self.input_block = nn.Sequential(
            nn.Conv1d(in_dim, self.base_plane, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.base_plane),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual groups
        self.residual_groups = nn.Sequential(
            self._make_residual_group1d(block_type, 64, group_sizes[0], stride=1),
            self._make_residual_group1d(block_type, 128, group_sizes[1], stride=2),
            self._make_residual_group1d(block_type, 256, group_sizes[2], stride=2),
            self._make_residual_group1d(block_type, 512, group_sizes[3], stride=2),
        )

        # Output module
        self.output_block1 = FcBlock(512 * block_type.expansion, out_dim, inter_dim)
        self.output_block2 = FcBlock(512 * block_type.expansion, out_dim, inter_dim)

        self._initialize(zero_init_residual)

    def _make_residual_group1d(self, block, planes, group_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, group_size):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        mean = self.output_block1(x)  # mean
        logstd = self.output_block2(x)  # covariance sigma = exp(2 * logstd)
        return mean, logstd


MENAGERIE_ZOO = "vendored-pytorch"


def build_tlio():
    import torch

    torch.manual_seed(0)
    # window_size=200 IMU samples (6-channel gyro+accel) downsamples to
    # spatial length 7 through the 4 stride-2 stages -> in_dim=7 for FcBlock.
    model = ResNet1D(BasicBlock1D, in_dim=6, out_dim=3, group_sizes=[2, 2, 2, 2], inter_dim=7)
    model.eval()
    return model


def example_input_tlio():
    import torch

    torch.manual_seed(0)
    return (torch.randn(2, 6, 200),)


MENAGERIE_ENTRIES = [
    ("TLIO", "build_tlio", "example_input_tlio", 2020, MENAGERIE_ZOO),
]
