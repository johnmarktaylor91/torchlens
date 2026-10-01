# SOURCE: vendored from clovaai/ext_portrait_segmentation @ master
# https://raw.githubusercontent.com/clovaai/ext_portrait_segmentation/master/models/SINet.py
#
# Park, Hong, Roh, Ha, Han 2020 "SINet: Extreme Lightweight Portrait Segmentation
# Networks with Spatial Squeeze Module and Information Blocking Decoder" (WACV 2020).
# The S2 (spatial-squeeze) module, SEseparableCBR (squeeze-excite depthwise-separable
# conv), and the information-blocking decoder (`SINet.forward`'s stage1/stage2 confidence
# gating) are copied verbatim from models/SINet.py, including the paper's default
# `Enc_SINet`/`Dnc_SINet` config table. Only the CLI `__main__` FLOP-profiling block
# (which depends on the repo's `etc.flops_counter` helper) is omitted.
"""SINet: extreme-lightweight portrait segmentation with spatial-squeeze S2 blocks and an
information-blocking decoder."""

import torch
import torch.nn as nn

MENAGERIE_ZOO = "vendored-pytorch"

BN_moment = 0.1


# --- vendored from models/SINet.py ---
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class SqueezeBlock(nn.Module):
    def __init__(self, exp_size, divide=4.0):
        super(SqueezeBlock, self).__init__()

        if divide > 1:
            self.dense = nn.Sequential(
                nn.Linear(exp_size, int(exp_size / divide)),
                nn.PReLU(int(exp_size / divide)),
                nn.Linear(int(exp_size / divide), exp_size),
                nn.PReLU(exp_size),
            )
        else:
            self.dense = nn.Sequential(nn.Linear(exp_size, exp_size), nn.PReLU(exp_size))

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = torch.nn.functional.avg_pool2d(x, kernel_size=[height, width]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        # out = hard_sigmoid(out)

        return out * x


class SEseparableCBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, divide=2.0):
        super().__init__()
        padding = int((kSize - 1) / 2)

        self.conv = nn.Sequential(
            nn.Conv2d(
                nIn,
                nIn,
                (kSize, kSize),
                stride=stride,
                padding=(padding, padding),
                groups=nIn,
                bias=False,
            ),
            SqueezeBlock(nIn, divide=divide),
            nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False),
        )

        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
    This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, group=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
            groups=group,
        )

    def forward(self, input):
        output = self.conv(input)
        return output


class S2block(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, config):
        super().__init__()
        kSize = config[0]
        avgsize = config[1]

        self.resolution_down = False
        if avgsize > 1:
            self.resolution_down = True
            self.down_res = nn.AvgPool2d(avgsize, avgsize)
            self.up_res = nn.UpsamplingBilinear2d(scale_factor=avgsize)
            self.avgsize = avgsize

        padding = int((kSize - 1) / 2)
        self.conv = nn.Sequential(
            nn.Conv2d(
                nIn,
                nIn,
                kernel_size=(kSize, kSize),
                stride=1,
                padding=(padding, padding),
                groups=nIn,
                bias=False,
            ),
            nn.BatchNorm2d(nIn, eps=1e-03, momentum=BN_moment),
        )

        self.act_conv1x1 = nn.Sequential(
            nn.PReLU(nIn),
            nn.Conv2d(nIn, nOut, kernel_size=1, stride=1, bias=False),
        )

        self.bn = nn.BatchNorm2d(nOut, eps=1e-03, momentum=BN_moment)

    def forward(self, input):
        if self.resolution_down:
            input = self.down_res(input)
        output = self.conv(input)

        output = self.act_conv1x1(output)
        if self.resolution_down:
            output = self.up_res(output)
        return self.bn(output)


class S2module(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, nOut, add=True, config=[[3, 1], [5, 1]]):
        super().__init__()

        group_n = len(config)
        n = int(nOut / group_n)
        n1 = nOut - group_n * n

        self.c1 = C(nIn, n, 1, 1, group=group_n)

        for i in range(group_n):
            var_name = "d{}".format(i + 1)
            if i == 0:
                self.__dict__["_modules"][var_name] = S2block(n, n + n1, config[i])
            else:
                self.__dict__["_modules"][var_name] = S2block(n, n, config[i])

        self.BR = BR(nOut)
        self.add = add
        self.group_n = group_n

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = channel_shuffle(output1, self.group_n)

        for i in range(self.group_n):
            var_name = "d{}".format(i + 1)
            result_d = self.__dict__["_modules"][var_name](output1)
            if i == 0:
                combine = result_d
            else:
                combine = torch.cat([combine, result_d], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.BR(combine)
        return output


class SINet_Encoder(nn.Module):
    def __init__(self, config, classes=20, p=5, q=3, chnn=1.0):
        super().__init__()
        dim1 = 16
        dim2 = 48 + 4 * (chnn - 1)
        dim3 = 96 + 4 * (chnn - 1)

        self.level1 = CBR(3, 12, 3, 2)

        self.level2_0 = SEseparableCBR(12, dim1, 3, 2, divide=1)

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            if i == 0:
                self.level2.append(S2module(dim1, dim2, config=config[i], add=False))
            else:
                self.level2.append(S2module(dim2, dim2, config=config[i]))
        self.BR2 = BR(dim2 + dim1)

        self.level3_0 = SEseparableCBR(dim2 + dim1, dim2, 3, 2, divide=2)
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            if i == 0:
                self.level3.append(S2module(dim2, dim3, config=config[2 + i], add=False))
            else:
                self.level3.append(S2module(dim3, dim3, config=config[2 + i]))
        self.BR3 = BR(dim3 + dim2)

        self.classifier = C(dim3 + dim2, classes, 1, 1)

    def forward(self, input):
        output1 = self.level1(input)  # 8h 8w

        output2_0 = self.level2_0(output1)  # 4h 4w

        for i, layer in enumerate(self.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)  # 2h 2w

        output3_0 = self.level3_0(self.BR2(torch.cat([output2_0, output2], 1)))  # h w

        for i, layer in enumerate(self.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.BR3(torch.cat([output3_0, output3], 1))

        classifier = self.classifier(output3_cat)

        return classifier


class SINet(nn.Module):
    def __init__(self, config, classes=20, p=2, q=3, chnn=1.0, encoderFile=None):
        super().__init__()
        dim1 = 16  # noqa: F841 (kept as in upstream source; unused in the decoder __init__)
        dim2 = 48 + 4 * (chnn - 1)
        dim3 = 96 + 4 * (chnn - 1)  # noqa: F841 (kept as in upstream source)

        self.encoder = SINet_Encoder(config, classes, p, q, chnn)
        # # load the encoder modules
        if encoderFile is not None:
            if torch.cuda.device_count() == 0:
                self.encoder.load_state_dict(torch.load(encoderFile, map_location="cpu"))
            else:
                self.encoder.load_state_dict(torch.load(encoderFile))

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)  # (scale_factor=2, mode='bilinear')
        self.bn_3 = nn.BatchNorm2d(classes, eps=1e-03)

        self.level2_C = CBR(dim2, classes, 1, 1)

        self.bn_2 = nn.BatchNorm2d(classes, eps=1e-03)

        self.classifier = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(classes, classes, 3, 1, 1, bias=False),
        )

    def forward(self, input):
        output1 = self.encoder.level1(input)  # 8h 8w
        output2_0 = self.encoder.level2_0(output1)  # 4h 4w

        for i, layer in enumerate(self.encoder.level2):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)  # 2h 2w

        output3_0 = self.encoder.level3_0(
            self.encoder.BR2(torch.cat([output2_0, output2], 1))
        )  # h w

        for i, layer in enumerate(self.encoder.level3):
            if i == 0:
                output3 = layer(output3_0)
            else:
                output3 = layer(output3)

        output3_cat = self.encoder.BR3(torch.cat([output3_0, output3], 1))
        Enc_final = self.encoder.classifier(output3_cat)  # 1/8

        Dnc_stage1 = self.bn_3(self.up(Enc_final))  # 1/4
        stage1_confidence = torch.max(nn.Softmax2d()(Dnc_stage1), dim=1)[0]
        b, c, h, w = Dnc_stage1.size()

        stage1_gate = (1 - stage1_confidence).unsqueeze(1).expand(b, c, h, w)

        Dnc_stage2_0 = self.level2_C(output2)  # 2h 2w
        Dnc_stage2 = self.bn_2(self.up(Dnc_stage2_0 * stage1_gate + (Dnc_stage1)))  # 4h 4w

        classifier = self.classifier(Dnc_stage2)

        return classifier


def Dnc_SINet(classes, p, q, chnn, encoderFile=None):
    # real config table from models/SINet.py's Dnc_SINet()
    config = [
        [[3, 1], [5, 1]],
        [[3, 1], [3, 1]],
        [[3, 1], [5, 1]],
        [[3, 1], [3, 1]],
        [[5, 1], [3, 2]],
        [[5, 2], [3, 4]],
        [[3, 1], [3, 1]],
        [[5, 1], [5, 1]],
        [[3, 2], [3, 4]],
        [[3, 1], [5, 2]],
    ]

    model = SINet(config, classes=classes, p=p, q=q, chnn=chnn, encoderFile=encoderFile)
    return model


def build_sinet_portrait():
    # real config from the repo's __main__ demo: classes=2 (portrait/background), chnn=1;
    # p/q kept at the paper's small "SINet" head sizes (p=2, q=3, encoder default) for a
    # fast trace.
    model = Dnc_SINet(classes=2, p=2, q=3, chnn=1)
    model.eval()
    return model


def example_input_sinet_portrait():
    torch.manual_seed(0)
    return (torch.randn(1, 3, 224, 224),)


MENAGERIE_ENTRIES = [
    ("SINet Portrait", "build_sinet_portrait", "example_input_sinet_portrait", 2020, "vendored"),
]
