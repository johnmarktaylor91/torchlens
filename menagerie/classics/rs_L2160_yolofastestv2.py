# SOURCE: vendored from dog-qiuqiu/Yolo-FastestV2 @ main
# https://raw.githubusercontent.com/dog-qiuqiu/Yolo-FastestV2/main/model/detector.py
# https://raw.githubusercontent.com/dog-qiuqiu/Yolo-FastestV2/main/model/fpn.py
# https://raw.githubusercontent.com/dog-qiuqiu/Yolo-FastestV2/main/model/backbone/shufflenetv2.py
#
# `queue.tsv` groups this candidate under YOLO-Fastest (dog-qiuqiu/Yolo-Fastest, the V1
# family). That repo is a fork of AlexeyAB/darknet: the actual network is a Darknet
# `.cfg` text config compiled by C/CUDA darknet code, with no PyTorch model definition
# anywhere in the repo. The same author's successor repo, dog-qiuqiu/Yolo-FastestV2 (an
# even more ultra-lightweight ~250K-parameter single-stage detector, same design lineage
# "Based on Yolo's low-power, ultra-lightweight universal target detection algorithm"),
# ships the real PyTorch reimplementation used for training/export
# (`model/detector.py`: `Detector`; `model/fpn.py`: `DWConvblock`, `LightFPN`;
# `model/backbone/shufflenetv2.py`: `ShuffleV2Block`, `ShuffleNetV2`), so that is the
# real source vendored here, copied verbatim: a ShuffleNetV2 backbone feeding a
# depthwise-separable "LightFPN" two-scale (P2/P3) neck with per-scale
# cls/obj/reg heads -- the actual anchor-based one-stage detector architecture.
#
# The only non-architectural changes: (1) `from torchsummary import summary` is dropped
# from `shufflenetv2.py` -- it is an unused debug-print import, never called anywhere in
# the class bodies; (2) `ShuffleNetV2.__init__` is constructed with `load_param=True` so
# it takes the branch that skips `self._initialize_weights()` (which the source has
# unconditionally read a `backbone.pth` checkpoint file off disk via
# `self.load_state_dict(torch.load(...))`) -- this is the source's own
# not-loading-a-pretrained-checkpoint code path, leaving the layers at PyTorch's default
# random init, exactly the same probe convention used elsewhere in this ladder; no
# architecture is touched. `Detector.forward` is constructed with `export_onnx=False`
# (also the source's default), which is its plain-tensor-output path -- the
# `export_onnx=True` branch in the original additionally references a bare `F.softmax`
# without importing `torch.nn.functional as F` in `detector.py`, a genuine unrelated bug
# in the source's export path, not exercised here.
"""Yolo-FastestV2: ~250K-parameter anchor-based one-stage detector -- ShuffleNetV2
backbone + a depthwise-separable two-scale FPN neck with per-scale cls/obj/reg heads
(dog-qiuqiu, ultra-lightweight-detector lineage)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


# --- vendored from model/backbone/shufflenetv2.py ---
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


# --- vendored from model/backbone/shufflenetv2.py ---
class ShuffleNetV2(nn.Module):
    def __init__(self, stage_out_channels, load_param):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = [4, 8, 4]
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
        C1 = self.stage2(x)
        C2 = self.stage3(C1)
        C3 = self.stage4(C2)

        return C2, C3

    def _initialize_weights(self):
        print("initialize_weights...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_state_dict(
            torch.load("./model/backbone/backbone.pth", map_location=device), strict=True
        )


# --- vendored from model/fpn.py ---
class DWConvblock(nn.Module):
    def __init__(self, input_channels, output_channels, size):
        super(DWConvblock, self).__init__()
        self.size = size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.block = nn.Sequential(
            nn.Conv2d(
                output_channels, output_channels, size, 1, 2, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(
                output_channels, output_channels, size, 1, 2, groups=output_channels, bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
        )

    def forward(self, x):
        x = self.block(x)
        return x


# --- vendored from model/fpn.py ---
class LightFPN(nn.Module):
    def __init__(self, input2_depth, input3_depth, out_depth):
        super(LightFPN, self).__init__()

        self.conv1x1_2 = nn.Sequential(
            nn.Conv2d(input2_depth, out_depth, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_depth),
            nn.ReLU(inplace=True),
        )

        self.conv1x1_3 = nn.Sequential(
            nn.Conv2d(input3_depth, out_depth, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_depth),
            nn.ReLU(inplace=True),
        )

        self.cls_head_2 = DWConvblock(input2_depth, out_depth, 5)
        self.reg_head_2 = DWConvblock(input2_depth, out_depth, 5)

        self.reg_head_3 = DWConvblock(input3_depth, out_depth, 5)
        self.cls_head_3 = DWConvblock(input3_depth, out_depth, 5)

    def forward(self, C2, C3):
        S3 = self.conv1x1_3(C3)
        cls_3 = self.cls_head_3(S3)
        obj_3 = cls_3
        reg_3 = self.reg_head_3(S3)

        P2 = F.interpolate(C3, scale_factor=2)
        P2 = torch.cat((P2, C2), 1)
        S2 = self.conv1x1_2(P2)
        cls_2 = self.cls_head_2(S2)
        obj_2 = cls_2
        reg_2 = self.reg_head_2(S2)

        return cls_2, obj_2, reg_2, cls_3, obj_3, reg_3


# --- vendored from model/detector.py ---
class Detector(nn.Module):
    def __init__(self, classes, anchor_num, load_param, export_onnx=False):
        super(Detector, self).__init__()
        out_depth = 72
        stage_out_channels = [-1, 24, 48, 96, 192]

        self.export_onnx = export_onnx
        self.backbone = ShuffleNetV2(stage_out_channels, load_param)
        self.fpn = LightFPN(
            stage_out_channels[-2] + stage_out_channels[-1], stage_out_channels[-1], out_depth
        )

        self.output_reg_layers = nn.Conv2d(out_depth, 4 * anchor_num, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(out_depth, anchor_num, 1, 1, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(out_depth, classes, 1, 1, 0, bias=True)

    def forward(self, x):
        C2, C3 = self.backbone(x)
        cls_2, obj_2, reg_2, cls_3, obj_3, reg_3 = self.fpn(C2, C3)

        out_reg_2 = self.output_reg_layers(reg_2)
        out_obj_2 = self.output_obj_layers(obj_2)
        out_cls_2 = self.output_cls_layers(cls_2)

        out_reg_3 = self.output_reg_layers(reg_3)
        out_obj_3 = self.output_obj_layers(obj_3)
        out_cls_3 = self.output_cls_layers(cls_3)

        if self.export_onnx:
            out_reg_2 = out_reg_2.sigmoid()
            out_obj_2 = out_obj_2.sigmoid()
            out_cls_2 = F.softmax(out_cls_2, dim=1)

            out_reg_3 = out_reg_3.sigmoid()
            out_obj_3 = out_obj_3.sigmoid()
            out_cls_3 = F.softmax(out_cls_3, dim=1)

            print("export onnx ...")
            return torch.cat((out_reg_2, out_obj_2, out_cls_2), 1).permute(0, 2, 3, 1), torch.cat(
                (out_reg_3, out_obj_3, out_cls_3), 1
            ).permute(0, 2, 3, 1)

        else:
            return out_reg_2, out_obj_2, out_cls_2, out_reg_3, out_obj_3, out_cls_3


def build_yolofastestv2():
    model = Detector(classes=80, anchor_num=3, load_param=True, export_onnx=False)
    model.eval()
    return model


def example_input_yolofastestv2():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 352, 352),)


MENAGERIE_ENTRIES = [
    ("Yolo-FastestV2", "build_yolofastestv2", "example_input_yolofastestv2", 2021, "vendored"),
]
