# SOURCE: vendored from dog-qiuqiu/FastestDet @ main
# https://raw.githubusercontent.com/dog-qiuqiu/FastestDet/main/module/detector.py
# https://raw.githubusercontent.com/dog-qiuqiu/FastestDet/main/module/shufflenetv2.py
# https://raw.githubusercontent.com/dog-qiuqiu/FastestDet/main/module/custom_layers.py
#
# dog-qiuqiu 2022 "FastestDet: Ultra lightweight anchor-free real-time object detection
# algorithm" -- a ShuffleNetV2 backbone (module/shufflenetv2.py, copied verbatim) feeding an
# SPP neck + single anchor-free DetectHead (module/custom_layers.py, copied verbatim),
# combined in module/detector.py::Detector (copied verbatim). 250K-parameter anchor-free
# detector designed for ARM/NCNN deployment. `load_param=False` (an existing constructor
# argument) is used to skip the repo's local `./module/shufflenetv2.pth` checkpoint load in
# favor of random init; the relative `from .shufflenetv2 import ...` imports are inlined
# since this is a single-file staging module.
"""FastestDet: ShuffleNetV2 backbone + SPP neck + anchor-free detect head."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from module/shufflenetv2.py ---
class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(
                mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


class ShuffleNetV2(nn.Module):
    def __init__(self, stage_repeats, stage_out_channels, load_param):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(
                        ShuffleV2Block(
                            input_channel,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=2,
                        )
                    )
                else:
                    stageSeq.append(
                        ShuffleV2Block(
                            input_channel // 2,
                            output_channel,
                            mid_channels=output_channel // 2,
                            ksize=3,
                            stride=1,
                        )
                    )
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))

        if not load_param:
            self._initialize_weights()
        else:
            print("load param...")

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        P1 = self.stage2(x)
        P2 = self.stage3(P1)
        P3 = self.stage4(P2)

        return P1, P2, P3

    def _initialize_weights(self):
        # NOTE: original repo loads a local ./module/shufflenetv2.pth checkpoint here;
        # that checkpoint file is not vendored, so this staging module always uses the
        # nn.Module default random init instead (no behavior change to the architecture).
        pass


# --- vendored from module/custom_layers.py ---
class Conv1x1(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv1x1(x)


class Head(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Head, self).__init__()
        self.conv5x5 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 5, 1, 2, groups=input_channels, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        return self.conv5x5(x)


class SPP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPP, self).__init__()
        self.Conv1x1 = Conv1x1(input_channels, output_channels)

        self.S1 = nn.Sequential(
            nn.Conv2d(
                output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.S2 = nn.Sequential(
            nn.Conv2d(
                output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.S3 = nn.Sequential(
            nn.Conv2d(
                output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.output = nn.Sequential(
            nn.Conv2d(output_channels * 3, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv1x1(x)

        y1 = self.S1(x)
        y2 = self.S2(x)
        y3 = self.S3(x)

        y = torch.cat((y1, y2, y3), dim=1)
        y = self.relu(x + self.output(y))

        return y


class DetectHead(nn.Module):
    def __init__(self, input_channels, category_num):
        super(DetectHead, self).__init__()
        self.conv1x1 = Conv1x1(input_channels, input_channels)

        self.obj_layers = Head(input_channels, 1)
        self.reg_layers = Head(input_channels, 4)
        self.cls_layers = Head(input_channels, category_num)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1x1(x)

        obj = self.sigmoid(self.obj_layers(x))
        reg = self.reg_layers(x)
        cls = self.softmax(self.cls_layers(x))

        return torch.cat((obj, reg, cls), dim=1)


# --- vendored from module/detector.py ---
class Detector(nn.Module):
    def __init__(self, category_num, load_param):
        super(Detector, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.backbone = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.SPP = SPP(sum(self.stage_out_channels[-3:]), self.stage_out_channels[-2])

        self.detect_head = DetectHead(self.stage_out_channels[-2], category_num)

    def forward(self, x):
        P1, P2, P3 = self.backbone(x)
        P3 = self.upsample(P3)
        P1 = self.avg_pool(P1)
        P = torch.cat((P1, P2, P3), dim=1)

        y = self.SPP(P)

        return self.detect_head(y)


def build_fastestdet():
    return Detector(category_num=80, load_param=False)


def example_input_fastestdet():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 352, 352),)


MENAGERIE_ENTRIES = [
    ("FastestDet", "build_fastestdet", "example_input_fastestdet", 2022, "vendored"),
]
