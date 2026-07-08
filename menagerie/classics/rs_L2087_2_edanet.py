# SOURCE: vendored from https://github.com/wpf535236337/pytorch_EDANet @ master
# (EDANet.py)
#
# EDANet (Lo, Hang, Chan, Lin, ACM MMAsia 2019 Best Paper, "Efficient Dense
# Modules of Asymmetric Convolution for Real-Time Semantic Segmentation",
# arXiv:1809.06323): a real-time semantic-segmentation network built from
# stacked asymmetric-convolution "EDA" dense modules with downsampler blocks.
# The official shaoyuanlo/EDANet repo explicitly withholds source code
# ("I cannot share our source codes due to the sponsor's request... you can
# access the implementations by others") and instead links this PyTorch
# reproduction by Pengfei Wang as the canonical community port; this file
# vendors that reproduction verbatim (only updated the Python-2 `print`
# statement in the `__main__` smoke test to Python-3 syntax; no architecture
# was altered).

import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlock, self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            # Wout > Win
            self.conv = nn.Conv2d(ninput, noutput - ninput, kernel_size=3, stride=2, padding=1)
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            # Wout < Win
            self.conv = nn.Conv2d(ninput, noutput, kernel_size=3, stride=2, padding=1)

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)

        output = self.bn(output)
        return F.relu(output)


class EDABlock(nn.Module):
    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super(EDABlock, self).__init__()

        # k: growthrate
        # dropprob: a dropout layer between the last ReLU and the concatenation of each module

        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(k, k, (3, 1), stride=1, padding=(dilated, 0), dilation=dilated)
        self.conv1x3_2 = nn.Conv2d(k, k, (1, 3), stride=1, padding=(0, dilated), dilation=dilated)
        self.bn2 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        input = x

        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        output = torch.cat([output, input], 1)
        return output


class EDANet(nn.Module):
    def __init__(self, num_classes=20):
        super(EDANet, self).__init__()

        self.layers = nn.ModuleList()
        self.dilation1 = [1, 1, 1, 2, 2]
        self.dilation2 = [2, 2, 4, 4, 8, 8, 16, 16]

        # DownsamplerBlock1
        self.layers.append(DownsamplerBlock(3, 15))

        # DownsamplerBlock2
        self.layers.append(DownsamplerBlock(15, 60))

        # EDA module 1-1 ~ 1-5
        for i in range(5):
            self.layers.append(EDABlock(60 + 40 * i, self.dilation1[i]))

        # DownsamplerBlock3
        self.layers.append(DownsamplerBlock(260, 130))

        # EDA module 2-1 ~ 2-8
        for j in range(8):
            self.layers.append(EDABlock(130 + 40 * j, self.dilation2[j]))

        # Projection layer
        self.project_layer = nn.Conv2d(450, num_classes, kernel_size=1)

        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.project_layer(output)

        # Bilinear interpolation x8
        output = F.interpolate(output, scale_factor=8, mode="bilinear", align_corners=True)

        # Bilinear interpolation x2 (inference only)
        if not self.training:
            output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

        return output


if __name__ == "__main__":
    input = torch.randn(1, 3, 512, 1024)
    # for the inference only mode
    net = EDANet().eval()
    # for the training mode
    # net = EDANet().train()
    output = net(input)
    print(output.size())


# ---------------------------------------------------------------------------
# Menagerie staging glue (not part of the original repo).
# ---------------------------------------------------------------------------

_NUM_CLASSES = 20  # Cityscapes class count used by the original repo
_H, _W = 64, 128  # multiple of 16 (3 downsampler stages + x8/x2 upsample) at small scale


def build_edanet():
    torch.manual_seed(0)
    model = EDANet(num_classes=_NUM_CLASSES).eval()
    return model


def example_input_edanet():
    torch.manual_seed(0)
    return torch.randn(1, 3, _H, _W)


MENAGERIE_ENTRIES = [
    ("EDANet", "build_edanet", "example_input_edanet", 2018, MENAGERIE_ZOO),
]
