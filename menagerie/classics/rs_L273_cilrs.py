# SOURCE: vendored from felipecode/coiltraine @ master
# Files combined:
#   network/models/coil_icra.py (CoILICRA -- CILRS backbone, "CoIL-ICRA" model type)
#   network/models/building_blocks/conv.py (Conv)
#   network/models/building_blocks/fc.py (FC)
#   network/models/building_blocks/join.py (Join)
#   network/models/building_blocks/branching.py (Branching)
#   network/models/building_blocks/resnet.py (ResNet / resnet34 -- repo-local resnet variant
#       with a hardcoded FC head sized for the 88x200 CARLA input, used by CILRS's
#       perception='res' branch per the official nocrash/resnet34imnet10S1.yaml config)
#
# CILRS (Codevilla et al., "Exploring the Limitations of Behavior Cloning for Autonomous
# Driving", ICCV 2019) is exactly the CoILICRA network trained with an added speed-prediction
# auxiliary branch; the architecture module itself (perception module + measurement FC +
# join + command-conditioned branches + speed branch) is unmodified from CoILICRA, so this
# vendors the real coiltraine model code, not a paper reimplementation.
#
# The staged build_cilrs() uses CoILICRA's 'conv' perception path (the repo's own
# configs/sample/coil_icra.yaml -- named exactly "coil_icra" -- and configs/eccv/*.yaml all
# select this branch: channels [32,32,64,64,128,128,256,256]). The repo's alternate 'res'
# perception path (used by configs/nocrash/resnet34imnet*.yaml) is also vendored below
# (ResNet/BasicBlock/resnet34) for fidelity, but its `self.avgpool = nn.AvgPool2d(2,
# stride=0)` call raises "stride should not be zero" on all currently-supported PyTorch
# (2.1+; verified back to the oldest torch on PyPI, 1.13) -- a real upstream bug in that
# code path, not something introduced here -- so it is not the traced entry.
#
# Import-only fixes applied (no architectural change):
#   - g_conf / coil_logger / coilutils.general.command_number_to_index (training-harness
#     globals) removed; CoILICRA.__init__ now takes the same `params` dict directly plus the
#     small number of scalar values that CoILICRA.__init__ previously pulled off the global
#     `g_conf` singleton (sensor channel count, sensor spatial shape, number of speed inputs,
#     number of regression targets). forward()/forward_branch() logic is untouched apart from
#     removing the CUDA-only branch-selection helper (unused when calling forward() directly,
#     which is how build_/example_input_ exercise the model).
#
# MENAGERIE_ZOO = "vendored-pytorch"

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# building_blocks/conv.py
# ---------------------------------------------------------------------------
class Conv(nn.Module):
    def __init__(self, params=None, module_name="Default"):
        super(Conv, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if "channels" not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if "kernels" not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if "strides" not in params:
            raise ValueError(" Missing the strides parameter ")
        if "dropouts" not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if "end_layer" not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params["dropouts"]) != len(params["channels"]) - 1:
            raise ValueError("Dropouts should be from the len of channel_sizes minus 1")

        self.layers = []

        for i in range(0, len(params["channels"]) - 1):
            conv = nn.Conv2d(
                in_channels=params["channels"][i],
                out_channels=params["channels"][i + 1],
                kernel_size=params["kernels"][i],
                stride=params["strides"][i],
            )

            dropout = nn.Dropout2d(p=params["dropouts"][i])
            relu = nn.ReLU(inplace=True)
            bn = nn.BatchNorm2d(params["channels"][i + 1])

            layer = nn.Sequential(*[conv, bn, dropout, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name

    def forward(self, x):
        """Each conv is: conv + batch normalization + dropout + relu"""
        x = self.layers(x)
        x = x.view(-1, self.num_flat_features(x))
        return x, self.layers

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_conv_output(self, shape):
        bs = 1
        inp = torch.rand(bs, *shape)
        output_feat, _ = self.forward(inp)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size


# ---------------------------------------------------------------------------
# building_blocks/fc.py
# ---------------------------------------------------------------------------
class FC(nn.Module):
    def __init__(self, params=None, module_name="Default"):
        super(FC, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if "neurons" not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if "dropouts" not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if "end_layer" not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params["dropouts"]) != len(params["neurons"]) - 1:
            raise ValueError("Dropouts should be from the len of kernels minus 1")

        self.layers = []

        for i in range(0, len(params["neurons"]) - 1):
            fc = nn.Linear(params["neurons"][i], params["neurons"][i + 1])
            dropout = nn.Dropout2d(p=params["dropouts"][i])
            relu = nn.ReLU(inplace=True)

            if i == len(params["neurons"]) - 2 and params["end_layer"]:
                self.layers.append(nn.Sequential(*[fc, dropout]))
            else:
                self.layers.append(nn.Sequential(*[fc, dropout, relu]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        if type(x) is tuple:
            return self.layers(x[0]), x[1]
        else:
            return self.layers(x)


# ---------------------------------------------------------------------------
# building_blocks/join.py
# ---------------------------------------------------------------------------
class Join(nn.Module):
    def __init__(self, params=None, module_name="Default"):
        super(Join, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if "mode" not in params:
            raise ValueError(" Missing the mode parameter ")
        if "after_process" not in params:
            raise ValueError(" Missing the after_process parameter ")

        self.after_process = params["after_process"]
        self.mode = params["mode"]

    def forward(self, x, m):
        if self.mode == "cat":
            j = torch.cat((x, m), 1)
        else:
            raise ValueError("Mode to join networks not found")

        return self.after_process(j)


# ---------------------------------------------------------------------------
# building_blocks/branching.py
# ---------------------------------------------------------------------------
class Branching(nn.Module):
    def __init__(self, branched_modules=None):
        super(Branching, self).__init__()

        if branched_modules is None:
            raise ValueError("No model provided after branching")

        self.branched_modules = nn.ModuleList(branched_modules)

    def forward(self, x):
        branches_outputs = []
        for branch in self.branched_modules:
            branches_outputs.append(branch(x))

        return branches_outputs


# ---------------------------------------------------------------------------
# building_blocks/resnet.py (repo-local resnet variant, hardcoded FC head sized
# for the CARLA 88x200 input -- used by CoILICRA's perception='res' path)
# ---------------------------------------------------------------------------
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(2, stride=0)

        if block.__name__ == "Bottleneck":
            self.fc = nn.Linear(6144, num_classes)
        else:
            self.fc = nn.Linear(1536, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, [x0, x1, x2, x3, x4]


def resnet34(pretrained=False, **kwargs):
    """Constructs the repo-local ResNet-34 model (CoIL-ICRA perception backbone)."""
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


# ---------------------------------------------------------------------------
# network/models/coil_icra.py (CoILICRA)
# ---------------------------------------------------------------------------
class CoILICRA(nn.Module):
    """CILRS / CoIL-ICRA conditional imitation-learning driving policy.

    `params` mirrors the repo's MODEL_CONFIGURATION dict. The training-harness globals
    (g_conf.SENSORS / g_conf.NUMBER_FRAMES_FUSION / g_conf.INPUTS / g_conf.TARGETS) are
    replaced by explicit scalar constructor args carrying the same information; the network
    topology built below is byte-for-byte the same as the original CoILICRA.__init__/forward.
    """

    def __init__(
        self,
        params,
        sensor_channels=3,
        sensor_hw=(88, 200),
        num_frames_fusion=1,
        num_speed_inputs=1,
        num_targets=3,
    ):
        super(CoILICRA, self).__init__()
        self.params = params

        number_first_layer_channels = sensor_channels * num_frames_fusion
        sensor_input_shape = [number_first_layer_channels, sensor_hw[0], sensor_hw[1]]

        if "conv" in params["perception"]:
            perception_convs = Conv(
                params={
                    "channels": [number_first_layer_channels]
                    + params["perception"]["conv"]["channels"],
                    "kernels": params["perception"]["conv"]["kernels"],
                    "strides": params["perception"]["conv"]["strides"],
                    "dropouts": params["perception"]["conv"]["dropouts"],
                    "end_layer": True,
                }
            )

            perception_fc = FC(
                params={
                    "neurons": [perception_convs.get_conv_output(sensor_input_shape)]
                    + params["perception"]["fc"]["neurons"],
                    "dropouts": params["perception"]["fc"]["dropouts"],
                    "end_layer": False,
                }
            )

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = params["perception"]["fc"]["neurons"][-1]

        elif "res" in params["perception"]:  # pre defined residual networks
            resnet_module = resnet34
            self.perception = resnet_module(
                pretrained=False, num_classes=params["perception"]["res"]["num_classes"]
            )

            number_output_neurons = params["perception"]["res"]["num_classes"]

        else:
            raise ValueError("invalid convolution layer type")

        self.measurements = FC(
            params={
                "neurons": [num_speed_inputs] + params["measurements"]["fc"]["neurons"],
                "dropouts": params["measurements"]["fc"]["dropouts"],
                "end_layer": False,
            }
        )

        self.join = Join(
            params={
                "after_process": FC(
                    params={
                        "neurons": [
                            params["measurements"]["fc"]["neurons"][-1] + number_output_neurons
                        ]
                        + params["join"]["fc"]["neurons"],
                        "dropouts": params["join"]["fc"]["dropouts"],
                        "end_layer": False,
                    }
                ),
                "mode": "cat",
            }
        )

        self.speed_branch = FC(
            params={
                "neurons": [params["join"]["fc"]["neurons"][-1]]
                + params["speed_branch"]["fc"]["neurons"]
                + [1],
                "dropouts": params["speed_branch"]["fc"]["dropouts"] + [0.0],
                "end_layer": True,
            }
        )

        branch_fc_vector = []
        for _ in range(params["branches"]["number_of_branches"]):
            branch_fc_vector.append(
                FC(
                    params={
                        "neurons": [params["join"]["fc"]["neurons"][-1]]
                        + params["branches"]["fc"]["neurons"]
                        + [num_targets],
                        "dropouts": params["branches"]["fc"]["dropouts"] + [0.0],
                        "end_layer": True,
                    }
                )
            )

        self.branches = Branching(branch_fc_vector)

        if "conv" in params["perception"]:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x, a):
        """###### APPLY THE PERCEPTION MODULE"""
        x, inter = self.perception(x)

        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)
        """ Join measurements and perception"""
        j = self.join(x, m)

        branch_outputs = self.branches(j)

        speed_branch_output = self.speed_branch(x)

        return branch_outputs + [speed_branch_output]


# ---------------------------------------------------------------------------
# menagerie staging entry points
# ---------------------------------------------------------------------------
MENAGERIE_ZOO = "vendored-pytorch"


def _cilrs_params():
    # Mirrors configs/sample/coil_icra.yaml / configs/eccv/experiment_1.yaml
    # MODEL_CONFIGURATION (the repo's own 'conv' perception config), with the FC widths
    # shrunk for fast tracing -- conv channel/kernel/stride topology is untouched.
    return {
        "perception": {
            "conv": {
                "channels": [32, 32, 64, 64, 128, 128, 256, 256],
                "kernels": [5, 3, 3, 3, 3, 3, 3, 3],
                "strides": [2, 1, 2, 1, 2, 1, 1, 1],
                "dropouts": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "fc": {"neurons": [32], "dropouts": [0.0]},
        },
        "measurements": {"fc": {"neurons": [16, 16], "dropouts": [0.0, 0.0]}},
        "join": {"fc": {"neurons": [32], "dropouts": [0.0]}},
        "speed_branch": {"fc": {"neurons": [16, 16], "dropouts": [0.0, 0.5]}},
        "branches": {"number_of_branches": 4, "fc": {"neurons": [16, 16], "dropouts": [0.0, 0.5]}},
    }


def build_cilrs():
    return CoILICRA(
        _cilrs_params(),
        sensor_channels=3,
        sensor_hw=(88, 200),
        num_frames_fusion=1,
        num_speed_inputs=1,
        num_targets=3,
    )


def example_input_cilrs():
    image = torch.rand(1, 3, 88, 200)
    speed = torch.rand(1, 1)
    return (image, speed)


MENAGERIE_ENTRIES = [
    ("CILRS", "build_cilrs", "example_input_cilrs", 2019, MENAGERIE_ZOO),
]
