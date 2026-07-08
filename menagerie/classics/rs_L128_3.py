# SOURCE: vendored from hzwer/ICCV2019-LearningToPaint @ 615c0707ce265706af41fd63a78d0c7824639f75
# https://raw.githubusercontent.com/hzwer/ICCV2019-LearningToPaint/615c0707ce265706af41fd63a78d0c7824639f75/baseline/DRL/actor.py
# https://raw.githubusercontent.com/hzwer/ICCV2019-LearningToPaint/615c0707ce265706af41fd63a78d0c7824639f75/baseline/DRL/critic.py
# https://raw.githubusercontent.com/hzwer/ICCV2019-LearningToPaint/615c0707ce265706af41fd63a78d0c7824639f75/baseline/Renderer/model.py
#
# Huang et al. 2019 (ICCV) "Learning to Paint With Model-based Deep Reinforcement
# Learning" -- a model-based DDPG stroke-based painting agent. `DRL/actor.py` defines
# the policy network as a plain-BatchNorm `ResNet` (18/34/50/101/152 configurable,
# `sigmoid`-squashed continuous stroke-parameter output). `DRL/critic.py` defines the
# Q-value network as a *weight-normalized*, BatchNorm-free `ResNet_wobn` using a
# learnable-threshold `TReLU` activation (`F.relu(x - alpha) + alpha`, alpha a
# per-layer scalar parameter) in place of plain ReLU throughout every residual block --
# this weight-norm + TReLU combination (used specifically to stabilize the
# actor-critic value estimate) is the architectural delta from a stock ResNet.
# `Renderer/model.py` defines `FCN`, the differentiable neural stroke renderer: an
# MLP (10 -> 512 -> 1024 -> 2048 -> 4096) reshaped to a 16x16x16 feature volume, then
# three conv + `PixelShuffle(2)` upsampling stages to a 128x128 single-channel stroke
# mask (`1 - sigmoid(...)`), trained to imitate a real (non-differentiable) painting
# simulator so strokes can be back-propagated through end-to-end. Together the Actor
# (policy) + Critic (value) + FCN (differentiable renderer) form the three real
# `nn.Module`s that make up the paper's model-based RL agent.
#
# `ResNet` (actor.py), `ResNet_wobn`/`TReLU` (critic.py), and `FCN` (Renderer/model.py)
# are the models exactly as defined upstream (unchanged). No architectural changes
# were made; only mechanical fixes for self-containment:
#   - Both `actor.py` and `critic.py` independently define `conv3x3`/`BasicBlock`/
#     `Bottleneck`/`cfg` with the same names but different bodies (actor.py's `conv3x3`
#     has no weight_norm/bias and its blocks use BatchNorm2d; critic.py's `conv3x3` is
#     weight-normalized with bias and its blocks use `TReLU`) -- both variants are kept,
#     suffixed `_actor`/`_critic` on the private helper classes to avoid a name
#     collision in one module, with the public `ResNet`/`ResNet_wobn` class names
#     preserved exactly as upstream (these are the names used by `DRL/ddpg.py`:
#     `self.actor = ResNet(9, 18, 65)`, `self.critic = ResNet_wobn(3 + 9, 18, 1)`).
#   - `Renderer/model.py`'s real forward path
#     (`Decoder(x[:, :10])` with a 10-dim stroke-parameter input, per `DRL/ddpg.py`'s
#     `decode()`) is used unmodified.
#   - The unused `torch.autograd.Variable` import in the originals is dropped (no
#     behavior change; `Variable` is a no-op wrapper in modern torch).

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

# ---------------------------------------------------------------------------
# DRL/actor.py -- policy network (plain BatchNorm ResNet, sigmoid output)
# ---------------------------------------------------------------------------


def conv3x3_actor(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert depth in depth_lst, "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        "18": (BasicBlock, [2, 2, 2, 2]),
        "34": (BasicBlock, [3, 4, 6, 3]),
        "50": (Bottleneck, [3, 4, 6, 3]),
        "101": (Bottleneck, [3, 4, 23, 3]),
        "152": (Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_actor(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3_actor(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                (
                    nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                    )
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                (
                    nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                    )
                ),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_inputs, depth, num_outputs):
        super(ResNet, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3_actor(num_inputs, 64, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_outputs)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# DRL/critic.py -- Q-value network (weight-normalized ResNet, TReLU activation)
# ---------------------------------------------------------------------------


def conv3x3_critic(in_planes, out_planes, stride=1):
    return weightNorm(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
    )


class TReLU(nn.Module):
    def __init__(self):
        super(TReLU, self).__init__()
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0)

    def forward(self, x):
        x = F.relu(x - self.alpha) + self.alpha
        return x


class BasicBlock_wobn(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_wobn, self).__init__()
        self.conv1 = conv3x3_critic(in_planes, planes, stride)
        self.conv2 = conv3x3_critic(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weightNorm(
                    nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True
                    )
                ),
            )
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.relu_2(out)

        return out


class Bottleneck_wobn(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_wobn, self).__init__()
        self.conv1 = weightNorm(nn.Conv2d(in_planes, planes, kernel_size=1, bias=True))
        self.conv2 = weightNorm(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        )
        self.conv3 = weightNorm(
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True)
        )
        self.relu_1 = TReLU()
        self.relu_2 = TReLU()
        self.relu_3 = TReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                weightNorm(
                    nn.Conv2d(
                        in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True
                    )
                ),
            )

    def forward(self, x):
        out = self.relu_1(self.conv1(x))
        out = self.relu_2(self.conv2(out))
        out = self.conv3(out)
        out += self.shortcut(x)
        out = self.relu_3(out)

        return out


def cfg_wobn(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert depth in depth_lst, "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        "18": (BasicBlock_wobn, [2, 2, 2, 2]),
        "34": (BasicBlock_wobn, [3, 4, 6, 3]),
        "50": (Bottleneck_wobn, [3, 4, 6, 3]),
        "101": (Bottleneck_wobn, [3, 4, 23, 3]),
        "152": (Bottleneck_wobn, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]


class ResNet_wobn(nn.Module):
    def __init__(self, num_inputs, depth, num_outputs):
        super(ResNet_wobn, self).__init__()
        self.in_planes = 64

        block, num_blocks = cfg_wobn(depth)

        self.conv1 = conv3x3_critic(num_inputs, 64, 2)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_outputs)
        self.relu_1 = TReLU()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu_1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Renderer/model.py -- differentiable stroke renderer
# ---------------------------------------------------------------------------


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv5 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1, 16, 16, 16)
        x = F.relu(self.conv1(x))
        x = self.pixel_shuffle(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pixel_shuffle(self.conv6(x))
        x = torch.sigmoid(x)
        return 1 - x.view(-1, 128, 128)


def build_l2p_actor():
    # Real constructor args used by DRL/ddpg.py: ResNet(9, 18, 65)
    # (target(3) + canvas(3) + stepnum(1) + coordconv(2) channels, depth-18 config,
    # 65-dim continuous stroke-parameter output).
    model = ResNet(9, 18, 65)
    model.eval()
    return model


def example_input_l2p_actor():
    # 128x128 matches the real training resolution (Renderer output size / coord grid);
    # depth-18 ResNet needs the input divisible by 32 so avg_pool2d(x, 4) lands exactly.
    return (torch.randn(1, 9, 128, 128),)


def build_l2p_critic():
    # Real constructor args used by DRL/ddpg.py: ResNet_wobn(3 + 9, 18, 1)
    # (extra target-canvas-diff channels(3) + actor input channels(9), depth-18, scalar Q-value).
    model = ResNet_wobn(12, 18, 1)
    model.eval()
    return model


def example_input_l2p_critic():
    return (torch.randn(1, 12, 128, 128),)


def build_l2p_renderer():
    model = FCN()
    model.eval()
    return model


def example_input_l2p_renderer():
    # 10 stroke parameters per DRL/ddpg.py's decode(): Decoder(x[:, :10]).
    return (torch.randn(4, 10),)


MENAGERIE_ZOO = "vendored-pytorch"
MENAGERIE_ENTRIES = [
    ("LearningToPaint-Actor", "build_l2p_actor", "example_input_l2p_actor", 2019, "vendored"),
    ("LearningToPaint-Critic", "build_l2p_critic", "example_input_l2p_critic", 2019, "vendored"),
    (
        "LearningToPaint-Renderer",
        "build_l2p_renderer",
        "example_input_l2p_renderer",
        2019,
        "vendored",
    ),
]
